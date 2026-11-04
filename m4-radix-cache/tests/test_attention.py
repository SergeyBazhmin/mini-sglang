"""Both prefill paths, and both with and without a cached prefix.

The padded prefill must compute what the per-request loop computes, and running
only the tail of a sequence must compute what running all of it would have.

    uv run pytest m4-radix-cache/tests/test_attention.py -m model

No weights to download and no GPU needed: a tiny randomly initialized attention
layer over random K/V is enough, because the two paths only differ in how they
line tokens up. test_batching.py --batched-prefill checks a real model.
"""


import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

from attention import ForwardBatch, PoolAttention

pytestmark = pytest.mark.model

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
        prefix_lens=torch.zeros(len(lens), dtype=torch.int64),
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


def with_prefix(fb: ForwardBatch, q: torch.Tensor, prefix_lens):
    """The same requests, as if their first prefix_lens tokens were cached."""
    keep = fb.positions >= torch.tensor(prefix_lens).repeat_interleave(fb.seq_lens)
    prefix = torch.tensor(prefix_lens)
    return q[keep], ForwardBatch(
        is_extend=True,
        input_ids=fb.input_ids[keep],
        positions=fb.positions[keep],
        out_cache_loc=fb.out_cache_loc[keep],
        req_pool_indices=fb.req_pool_indices,
        seq_lens=fb.seq_lens,
        extend_lens=fb.seq_lens - prefix,
        prefix_lens=prefix,
    )


def test_a_cached_prefix_does_not_change_the_rest():
    for lens, prefixes in (
        ([5], [3]),
        ([5], [4]),
        ([4, 4], [2, 3]),
        ([1, 16], [0, 15]),
        ([7, 2, 11, 3], [6, 1, 0, 2]),
    ):
        attention, q, fb = make_attention(lens)
        full = attention._run_extend(q, fb)
        q_tail, fb_tail = with_prefix(fb, q, prefixes)
        want = full[fb.positions >= torch.tensor(prefixes).repeat_interleave(fb.seq_lens)]
        for run in (attention._run_extend, attention._run_extend_batched):
            got = run(q_tail, fb_tail)
            assert got.shape == want.shape
            assert torch.allclose(got, want, atol=1e-4), (lens, prefixes, run.__name__)


def test_a_cached_prefix_reads_the_pool():
    attention, q, fb = make_attention([6])
    q_tail, fb_tail = with_prefix(fb, q, [3])
    want = attention._run_extend(q_tail, fb_tail)
    # Change the K/V of the first cached token.
    attention.k_buffer[fb.out_cache_loc[0]] += 1.0
    attention.v_buffer[fb.out_cache_loc[0]] += 1.0
    for run in (attention._run_extend, attention._run_extend_batched):
        assert not torch.allclose(run(q_tail, fb_tail), want, atol=1e-4)
