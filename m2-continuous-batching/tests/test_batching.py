"""Check that batching does not change the output: greedy tokens from the
batched ModelRunner must match HF `generate` run on each prompt alone.

    uv run pytest m2-continuous-batching/tests/test_batching.py -m model --model Qwen/Qwen3-0.6B

Requests have different prompt lengths and join the batch at different steps,
so this covers right padding in prefill, left padding in decode, and merging
a fresh prefill into a running batch.

On CPU (float32) every token should match. On GPU (bfloat16) a padded batch
does slightly different arithmetic than a single request, so a near-tie
between two tokens can flip and the texts diverge after that point. That is
not a bug in the batching, and the "first mismatch" position shows how far
they agreed. (SGLang has a whole mode for this, --enable-deterministic-inference.)
"""

from typing import List

import pytest
import torch
from transformers import AutoTokenizer

from model_runner import FakeReq, ModelRunner

pytestmark = pytest.mark.model

PROMPTS = [
    "The capital of France is",
    "Write a haiku about the ocean.",
    "List three prime numbers greater than 100, and explain how you checked each one.",
    "def fibonacci(n):",
    "Translate to German: The weather is nice today.",
]
# Step at which each prompt joins the batch. Several share a step, the rest
# arrive while the others are already decoding.
JOIN_AT_STEP = [0, 0, 3, 7, 7]


def run_batched(runner: ModelRunner, prompts_ids: List[List[int]], max_new_tokens: int):
    reqs = [FakeReq(ids) for ids in prompts_ids]
    running: List[FakeReq] = []
    step = 0
    while step <= max(JOIN_AT_STEP) or running:
        # Same order as the scheduler: prefill new arrivals first, else decode.
        new = [r for r, s in zip(reqs, JOIN_AT_STEP) if s == step]
        if new:
            for r, tok in zip(new, runner.forward(new, is_extend=True)):
                r.output_ids.append(tok)
            running += new
        elif running:
            for r, tok in zip(running, runner.forward(running, is_extend=False)):
                r.output_ids.append(tok)
            running = [r for r in running if len(r.output_ids) < max_new_tokens]
        step += 1
    return [r.output_ids[:max_new_tokens] for r in reqs]


@torch.inference_mode()
def run_reference(runner: ModelRunner, prompt_ids: List[int], max_new_tokens: int):
    out = runner.model.generate(
        torch.tensor([prompt_ids], device=runner.device),
        attention_mask=torch.ones(1, len(prompt_ids), dtype=torch.long, device=runner.device),
        max_new_tokens=max_new_tokens,
        min_new_tokens=max_new_tokens,  # Don't stop at EOS: the batched loop doesn't.
        do_sample=False,
    )
    return out[0, len(prompt_ids) :].tolist()


def test_matches_hf(model: str, max_new_tokens: int):
    """Compare batched greedy decoding with HF generate, prompt by prompt."""
    runner = ModelRunner(model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    prompts_ids = [tokenizer(p).input_ids for p in PROMPTS]
    print(f"device={runner.device}, prompt lengths={[len(p) for p in prompts_ids]}")

    batched = run_batched(runner, prompts_ids, max_new_tokens)
    num_exact = 0
    for prompt, ids, got in zip(PROMPTS, prompts_ids, batched):
        want = run_reference(runner, ids, max_new_tokens)
        mismatch = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
        if mismatch is None:
            num_exact += 1
            print(f"MATCH     {prompt!r}")
        else:
            print(f"DIVERGE   {prompt!r} at token {mismatch}/{max_new_tokens}")
            print(f"  batched: {tokenizer.decode(got)!r}")
            print(f"  HF:      {tokenizer.decode(want)!r}")
    print(f"\n{num_exact}/{len(PROMPTS)} prompts match token-for-token.")
    if runner.device == "cpu":
        assert num_exact == len(PROMPTS), "on CPU (float32) all prompts should match"
