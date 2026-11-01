"""Exercise 2: the padded prefill must compute what the per-request loop computes.

    uv run m3-kv-cache/test/test_attention.py

No weights to download and no GPU needed: a tiny randomly initialized attention
layer over random K/V is enough, because the two paths only differ in how they
line tokens up. test_batching.py --batched-prefill checks a real model.
"""

import sys
from pathlib import Path

import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

# Run as a script from test/, so make the chapter's modules importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attention import ForwardBatch, PoolAttention

NUM_HEADS = 4
NUM_KV_HEADS = 2  # Fewer than the query heads, as in Qwen3.
HEAD_DIM = 8
POOL_SIZE = 64
MAX_LEN = 16


def make_attention(lens):
    """Random K/V, and each request's slots shuffled through the pool."""
    torch.manual_seed(0)
    config = Qwen3Config(
        hidden_size=NUM_HEADS * HEAD_DIM,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
    )
    k_buffer = torch.randn(POOL_SIZE + 1, NUM_KV_HEADS, HEAD_DIM)
    v_buffer = torch.randn(POOL_SIZE + 1, NUM_KV_HEADS, HEAD_DIM)
    table = torch.zeros(len(lens), MAX_LEN, dtype=torch.int64)

    attention = PoolAttention(Qwen3Attention(config, layer_idx=0), k_buffer, v_buffer, table)

    num_tokens = sum(lens)
    out_cache_loc = torch.randperm(POOL_SIZE)[:num_tokens] + 1
    for row, row_slots in enumerate(out_cache_loc.split(lens)):
        table[row, : len(row_slots)] = row_slots

    lens_t = torch.tensor(lens)
    forward_batch = ForwardBatch(
        is_extend=True,
        input_ids=torch.zeros(num_tokens, dtype=torch.int64),
        positions=torch.cat([torch.arange(n) for n in lens]),
        out_cache_loc=out_cache_loc,
        req_pool_indices=torch.arange(len(lens)),
        seq_lens=lens_t,
        extend_lens=lens_t,
    )
    q = torch.randn(num_tokens, NUM_HEADS, HEAD_DIM)
    return attention, q, forward_batch


def test_batched_matches_the_loop():
    for lens in ([5], [4, 4], [1, 16], [7, 2, 11, 3]):
        attention, q, fb = make_attention(lens)
        want = attention._run_extend(q, fb)
        got = attention._run_extend_batched(q, fb)
        assert got.shape == (sum(lens), NUM_HEADS, HEAD_DIM)
        assert torch.allclose(got, want, atol=1e-4), lens


def test_stale_table_entries_are_masked():
    attention, q, fb = make_attention([2, 8])
    want = attention._run_extend_batched(q, fb)
    # Past its length the first row holds zeros, so it points at slot 0. Fill that
    # slot with K/V that would change the result if anything read it.
    attention.k_buffer[0] = 100.0
    attention.v_buffer[0] = 100.0
    got = attention._run_extend_batched(q, fb)
    assert torch.allclose(got[:2], want[:2], atol=1e-4)


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    main()
