"""Check that the KV pool and the prefix cache do not change the output: greedy
tokens from the batched ModelRunner must match HF `generate` on each prompt alone.

    uv run pytest m4-radix-cache/tests/test_batching.py -m model --model Qwen/Qwen3-0.6B

Requests have different prompt lengths, join the batch at different steps and
finish at different steps, so a request's slots end up scattered through the
pool. The pool is only slightly bigger than one pass needs, so the second pass
runs almost entirely on recycled slots. After each pass every slot must be free or
idle in the tree, and GPU memory must not have grown.

Pass 1 meets the cache as it fills: prompts share a system prompt, and most of them
find it cached by whoever ran first. Pass 2 finds every prompt and most outputs there,
as far as eviction left them. With the cache disabled, this is M3's test.

One request is retracted partway: its slots are freed and it prefills
its prompt and output again, which must not change a single token. Other runs turn
on the options that change how a batch runs, and the output must not change.
The test runs both ways.

On CPU (float32) every token should match. On GPU (bfloat16) a request's result
can depend slightly on the other requests in its batch, so a near-tie between
two tokens can flip and the texts diverge after that point. That is not a bug
in the pool, and the "first mismatch" position shows how far they agreed.
(SGLang has a whole mode for this, --enable-deterministic-inference.)
"""

from typing import List

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import EngineOptions
from model_runner import FakeReq, ModelRunner

pytestmark = pytest.mark.model

SYSTEM = "You are a careful assistant. Answer briefly, and check your work twice. "
PROMPTS = [
    SYSTEM + "The capital of France is",
    SYSTEM + "Write a haiku about the ocean.",
    "List three prime numbers greater than 100, and explain how you checked each one.",
    SYSTEM + "def fibonacci(n):",
    SYSTEM + "Translate to German: The weather is nice today.",
]
# Step at which each prompt joins the batch. Several share a step, the rest
# arrive while the others are already decoding.
JOIN_AT_STEP = [0, 0, 3, 7, 7]
# Each prompt generates max_new_tokens / this, so they leave at different steps.
LENGTH_DIVISOR = [1, 2, 1, 4, 1]
# After this step, the newest running request is retracted and joins again.
RETRACT_AT_STEP = 12
NUM_PASSES = 2


def run_batched(runner: ModelRunner, prompts_ids: List[List[int]], max_new_tokens: int):
    reqs = [
        FakeReq(ids, max_new_tokens // div)
        for ids, div in zip(prompts_ids, LENGTH_DIVISOR)
    ]
    running: List[FakeReq] = []
    retracted: List[FakeReq] = []
    step = 0
    total_cached = 0
    while step <= max(JOIN_AT_STEP) or running or retracted:
        # Same order as the scheduler: prefill new arrivals first, else decode.
        new = retracted + [r for r, s in zip(reqs, JOIN_AT_STEP) if s == step]
        retracted = []
        if new:
            for r in new:
                runner.tree_cache.match_req(r)
                runner.tree_cache.lock_req(r)
                total_cached += r.prefix_len
            for r, tok in zip(new, runner.forward(new, is_extend=True)):
                r.output_ids.append(tok)
                runner.tree_cache.checkpoint(r)
            running += new
        elif running:
            # As the scheduler does. The pool has room, if not for idle cache as well.
            assert runner.ensure_free(runner.decode_slots_needed(running))
            for r, tok in zip(running, runner.forward(running, is_extend=False)):
                r.output_ids.append(tok)
        for r in running:
            if len(r.output_ids) >= r.max_new_tokens:
                runner.free_req(r)
        running = [r for r in running if r.req_pool_idx is not None]
        if step == RETRACT_AT_STEP and running:
            retracted = [running.pop()]
            runner.free_req(retracted[0], cache=False)
        step += 1
    return [r.output_ids for r in reqs], total_cached


@torch.inference_mode()
def run_reference(reference, prompt_ids: List[int], max_new_tokens: int):
    device = reference.device
    out = reference.generate(
        torch.tensor([prompt_ids], device=device),
        attention_mask=torch.ones(1, len(prompt_ids), dtype=torch.long, device=device),
        max_new_tokens=max_new_tokens,
        min_new_tokens=max_new_tokens,  # Don't stop at EOS: the batched loop doesn't.
        do_sample=False,
    )
    return out[0, len(prompt_ids) :].tolist()


def compare(tokenizer, got_all, want_all) -> int:
    num_exact = 0
    for prompt, got, want in zip(PROMPTS, got_all, want_all):
        mismatch = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
        if mismatch is None:
            num_exact += 1
            print(f"MATCH     {prompt!r}")
        else:
            print(f"DIVERGE   {prompt!r} at token {mismatch}/{len(got)}")
            print(f"  batched: {tokenizer.decode(got)!r}")
            print(f"  HF:      {tokenizer.decode(want[: len(got)])!r}")
    return num_exact


@pytest.mark.parametrize(
    "batched_prefill, page_size, lazy_free, disable_radix_cache",
    [(False, 1, False, False), (True, 64, True, False), (False, 1, False, True)],
    ids=["default", "options", "no-cache"],
)
def test_matches_hf(
    model: str,
    max_new_tokens: int,
    batched_prefill: bool,
    page_size: int,
    lazy_free: bool,
    disable_radix_cache: bool,
):
    """Compare greedy decoding over the KV pool with HF generate, prompt by prompt."""
    tokenizer = AutoTokenizer.from_pretrained(model)
    prompts_ids = [tokenizer(p).input_ids for p in PROMPTS]

    # Enough for one pass and little more, so that pass 2 reuses pass 1's slots.
    pool_size = sum(len(p) + max_new_tokens + page_size for p in prompts_ids) + 16
    runner = ModelRunner(
        model,
        max_running_requests=len(PROMPTS),
        max_total_tokens=pool_size,
        options=EngineOptions(
            batched_prefill=batched_prefill,
            page_size=page_size,
            lazy_free=lazy_free,
            check_double_free=True,
            disable_radix_cache=disable_radix_cache,
        ),
    )
    pool_size = runner.max_total_tokens  # Rounded down to whole pages.
    print(
        f"device={runner.device}, prompt lengths={[len(p) for p in prompts_ids]}, "
        f"pool={pool_size} tokens"
    )

    # A separate copy: runner.model has PoolAttention, which can't run generate.
    reference = (
        AutoModelForCausalLM.from_pretrained(model, dtype=runner.model.dtype)
        .to(runner.device)
        .eval()
    )
    want_all = [run_reference(reference, ids, max_new_tokens) for ids in prompts_ids]
    del reference
    if runner.device == "cuda":
        torch.cuda.empty_cache()

    memory_mb = []
    num_cached_by_pass: List[int] = []
    total_exact = 0
    for pass_id in range(NUM_PASSES):
        print(f"\n--- pass {pass_id + 1}/{NUM_PASSES}")
        got_all, total_cached = run_batched(runner, prompts_ids, max_new_tokens)
        total_exact += compare(tokenizer, got_all, want_all)
        memory_mb.append(runner.gpu_memory_mb())

        tree = runner.tree_cache
        accounted = runner.allocator.available_size() + tree.total_size()
        assert accounted == pool_size, f"leaked slots, {accounted}/{pool_size} accounted for"
        assert tree.protected_size == 0, "a finished request still holds a lock"
        print(
            f"all {pool_size} slots free or idle ({tree.total_size()} cached), "
            f"{total_cached} prompt tokens reused, GPU memory {memory_mb[-1]:.1f} MB"
        )
        num_cached_by_pass.append(total_cached)

    num_expected = len(PROMPTS) * NUM_PASSES
    print(f"\n{total_exact}/{num_expected} prompts match token-for-token.")
    assert memory_mb[-1] <= memory_mb[0], f"GPU memory grew: {memory_mb} MB"
    if runner.device == "cpu":
        assert total_exact == num_expected, "on CPU (float32) all prompts should match"
    if not disable_radix_cache:
        assert num_cached_by_pass[-1] > 0, "the cache never hit, so this checked nothing"
