"""A cached prefix must give the same tokens as computing it. No download, no GPU.

    uv run pytest m4-radix-cache/tests/test_prefix_output.py -m model

A tiny randomly initialized Qwen3 is enough: nothing here depends on the weights being
any good, only on K/V being the same whichever way it got into the pool. On a CPU the
outputs must be exactly equal to HF's, so one wrong slot or mask shows up as a wrong
token. test_batching.py does the same with a real model and real text.
"""

from typing import List

import pytest
import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from common import EngineOptions
from model_runner import FakeReq, ModelRunner

pytestmark = pytest.mark.model

VOCAB = 256
MAX_NEW_TOKENS = 12
SHARED = list(range(40, 80))
PROMPTS = [
    SHARED + [1, 2, 3],
    SHARED + [4, 5],
    SHARED[:23] + [6],  # Splits the node in the middle of the shared part.
    [90, 91, 92, 93, 94],  # Shares nothing.
]


def tiny_model(path: str):
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        sliding_window=None,
        tie_word_embeddings=False,
    )
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    # Random weights are nearly flat, which makes ties likely. Sharpen the output.
    with torch.no_grad():
        model.get_output_embeddings().weight *= 30
    model.save_pretrained(path)
    return model.eval()


@torch.inference_mode()
def reference_tokens(model, ids: List[int]) -> List[int]:
    out = model.generate(
        torch.tensor([ids]),
        attention_mask=torch.ones(1, len(ids), dtype=torch.long),
        max_new_tokens=MAX_NEW_TOKENS,
        min_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        eos_token_id=None,
        pad_token_id=0,
    )
    return out[0, len(ids) :].tolist()


def generate(runner: ModelRunner, batch_ids: List[List[int]]):
    """One prefill for the whole batch, then decode to the end."""
    tree = runner.tree_cache
    reqs = [FakeReq(list(ids), MAX_NEW_TOKENS) for ids in batch_ids]
    for r in reqs:
        tree.match_req(r)
        tree.lock_req(r)
    for r, tok in zip(reqs, runner.forward(reqs, is_extend=True)):
        r.output_ids.append(tok)
        tree.checkpoint(r)
    prefix_lens = [r.cache_protected_len for r in reqs]
    for _ in range(MAX_NEW_TOKENS - 1):
        assert runner.ensure_free(runner.decode_slots_needed(reqs))  # As the scheduler does.
        for r, tok in zip(reqs, runner.forward(reqs, is_extend=False)):
            r.output_ids.append(tok)
    for r in reqs:
        runner.free_req(r)
    return [r.output_ids for r in reqs], prefix_lens


def make_runner(path: str, pool: int = 1024, **options) -> ModelRunner:
    return ModelRunner(
        path,
        max_running_requests=len(PROMPTS),
        max_total_tokens=pool,
        context_length=128,
        device="cpu",
        options=EngineOptions(check_double_free=True, **options),
    )


def expect(got: List[List[int]], want: List[List[int]], name: str):
    for i, (a, b) in enumerate(zip(got, want)):
        assert a == b, f"{name}: prompt {i} differs:\n  got  {a}\n  want {b}"


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """The path of a saved tiny model, and what HF generates for each prompt."""
    path = str(tmp_path_factory.mktemp("tiny-qwen3"))
    reference = tiny_model(path)
    want = [reference_tokens(reference, ids) for ids in PROMPTS]
    assert len({tuple(w) for w in want}) > 1, "all outputs equal: this proves little"
    return path, want


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"batched_prefill": True},
        {"page_size": 4},
        {"page_size": 4, "batched_prefill": True},
        {"lazy_free": True, "fast_match": True},
    ],
    ids=str,
)
def test_matches_hf(tiny, options):
    path, want = tiny

    # One at a time: each one sees what the previous ones left behind.
    runner = make_runner(path, **options)
    got = [generate(runner, [ids])[0][0] for ids in PROMPTS]
    expect(got, want, "sequential")
    tree = runner.tree_cache
    assert tree.total_size() > 0

    # The same again: everything is cached, up to the last token of the prompt.
    again = [generate(runner, [ids]) for ids in PROMPTS]
    expect([g[0][0] for g in again], want, "cached")
    assert runner.allocator.available_size() + tree.total_size() == runner.max_total_tokens
    assert tree.protected_size == 0

    # All at once, into an empty tree: they race, then share.
    runner = make_runner(path, **options)
    got, _ = generate(runner, PROMPTS)
    expect(got, want, "racing")

    # All at once, into a warm tree, in another order.
    got, _ = generate(runner, PROMPTS[::-1])
    expect(got, want[::-1], "warm")


def test_a_pool_too_small_to_keep_everything(tiny):
    path, want = tiny
    runner = make_runner(path, pool=96)
    for ids in PROMPTS * 4:
        got, _ = generate(runner, [ids])
        expect(got, [want[PROMPTS.index(ids)]], "evicting")
    assert runner.tree_cache.total_size() <= 96


def test_the_tree_is_really_used(tiny):
    """Or the checks above prove nothing."""
    path, _ = tiny
    runner = make_runner(path)
    generate(runner, [PROMPTS[0]])
    req = FakeReq(PROMPTS[1], 1)
    runner.tree_cache.match_req(req)
    assert req.prefix_len == len(SHARED), req.prefix_len
