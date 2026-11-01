"""Attention that keeps K/V in the token pool, plus the batch description it reads.

SGLang equivalents:
  - ForwardBatch       model_executor/forward_batch_info.py
  - PoolAttention      layers/attention/torch_native_backend.py:
                       _run_sdpa_forward_extend (~line 59),
                       _run_sdpa_forward_decode (~line 174)

Batches are FLAT, with no padding: all tokens of all requests are one sequence of
[num_tokens]. Everything except attention treats tokens independently (embedding,
MLP, norms, and rope, which only needs each token's position), so only attention
has to know which token belongs to which request. It learns that from
ForwardBatch, and finds each request's K/V through req_to_token.

This module is Qwen3-only for now because it reuses HF's Qwen3 projections and
q_norm/k_norm. M5 replaces the HF modeling code with our own, behind a backend
interface.
"""

import contextlib
import dataclasses
import time
from typing import Iterator, List, Optional, Tuple

import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    apply_rotary_pos_emb,
)


@dataclasses.dataclass
class ForwardBatch:
    """GPU-side view of one step."""

    is_extend: bool
    input_ids: torch.Tensor  # [num_tokens]
    positions: torch.Tensor  # [num_tokens], position inside its own request
    out_cache_loc: torch.Tensor  # [num_tokens], the slot where each token's K/V goes
    req_pool_indices: torch.Tensor  # [batch], row in req_to_token
    seq_lens: torch.Tensor  # [batch], tokens of each request in the pool, this step included
    extend_lens: torch.Tensor  # [batch], tokens of each request in input_ids


class GatherTimer:
    """Exercise 3: time spent in _run_decode's gathers, added up over all layers."""

    def __init__(self, device: str):
        self.device = device
        self.seconds = 0.0

    def _now(self) -> float:
        # GPU work is asynchronous. Without this, the gather would be charged for
        # earlier work, or finish after the clock stopped.
        if self.device == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    @contextlib.contextmanager
    def measure(self) -> Iterator[None]:
        start = self._now()
        yield
        self.seconds += self._now() - start


class PoolAttention(nn.Module):
    """Replaces Qwen3Attention. It takes over the weights of the module it replaces."""

    def __init__(
        self,
        attn: Qwen3Attention,
        k_buffer: torch.Tensor,
        v_buffer: torch.Tensor,
        req_to_token: torch.Tensor,
        batched_prefill: bool = False,
        gather_timer: Optional[GatherTimer] = None,
    ):
        super().__init__()
        self.batched_prefill = batched_prefill
        self.gather_timer = gather_timer
        self.head_dim = attn.head_dim
        self.q_proj = attn.q_proj
        self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj
        self.o_proj = attn.o_proj
        self.q_norm = attn.q_norm
        self.k_norm = attn.k_norm
        self.k_buffer = k_buffer  # [pool_size + 1, num_kv_heads, head_dim]
        self.v_buffer = v_buffer
        self.req_to_token = req_to_token  # [num_rows, max_context_len]

    def forward(
        self,
        hidden_states: torch.Tensor,  # [1, num_tokens, hidden_size]
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        forward_batch: ForwardBatch,
        **kwargs,  # HF's decoder layer also passes its mask and cache: both unused.
    ) -> Tuple[torch.Tensor, None]:
        num_tokens = hidden_states.shape[1]
        shape = (1, num_tokens, -1, self.head_dim)
        # [1, heads, num_tokens, head_dim]; K and V have fewer heads than Q (GQA).
        q = self.q_norm(self.q_proj(hidden_states).view(shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)

        # [num_tokens, heads, head_dim], the same shape as one row of the pool.
        q, k, v = (x[0].transpose(0, 1) for x in (q, k, v))

        # Store first: each token attends to itself too. SGLang: set_kv_buffer.
        self.k_buffer[forward_batch.out_cache_loc] = k
        self.v_buffer[forward_batch.out_cache_loc] = v

        if forward_batch.is_extend and self.batched_prefill:
            output = self._run_extend_batched(q, forward_batch)
        elif forward_batch.is_extend:
            output = self._run_extend(q, forward_batch)
        else:
            output = self._run_decode(q, forward_batch)
        return self.o_proj(output.reshape(1, num_tokens, -1)), None

    def _run_extend(self, q: torch.Tensor, fb: ForwardBatch) -> torch.Tensor:
        """One causal attention per request, because prompts differ in length.

        The same per-request loop as SGLang's torch-native backend. With many
        new requests this is the slow part of prefill.
        """
        outputs: List[torch.Tensor] = []
        rows = fb.req_pool_indices.tolist()
        lens = fb.extend_lens.tolist()
        for query, row, length in zip(q.split(lens), rows, lens):
            slots = self.req_to_token[row, :length]
            out = scaled_dot_product_attention(
                query.transpose(0, 1)[None],  # [1, heads, length, head_dim]
                self.k_buffer[slots].transpose(0, 1)[None],
                self.v_buffer[slots].transpose(0, 1)[None],
                is_causal=True,
                enable_gqa=True,
            )
            outputs.append(out[0].transpose(0, 1))
        return torch.cat(outputs)

    def _run_extend_batched(self, q: torch.Tensor, fb: ForwardBatch) -> torch.Tensor:
        """Exercise 2: one attention call for all new requests, the way _run_decode works.

        Padding brings M2's rectangle back, inside attention only. The cost is
        O(batch * max_len^2), so one long prompt makes every short one pay for
        its length.
        """
        max_len = int(fb.extend_lens.max())
        lengths = torch.arange(max_len, device=q.device)
        # [batch, max_len], True on real tokens. Boolean indexing walks it row by
        # row, which is the order of the flat batch.
        keep = lengths[None, :] < fb.extend_lens[:, None]

        padded_q = q.new_zeros((len(keep), max_len, *q.shape[1:]))
        padded_q[keep] = q
        slots = self.req_to_token[fb.req_pool_indices, :max_len]

        # A padded query can still see its request's real tokens, so no row of
        # the mask is empty, which would give NaNs. Its output is dropped anyway.
        causal = lengths[None, :] <= lengths[:, None]  # [max_len, max_len]
        mask = causal[None, :, :] & keep[:, None, :]  # [batch, max_len, max_len]

        out = scaled_dot_product_attention(
            padded_q.transpose(1, 2),  # [batch, heads, max_len, head_dim]
            self.k_buffer[slots].transpose(1, 2),
            self.v_buffer[slots].transpose(1, 2),
            attn_mask=mask[:, None, :, :],
            enable_gqa=True,
        )
        return out.transpose(1, 2)[keep]

    def _gather(self, slots: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.gather_timer is None:
            return self.k_buffer[slots], self.v_buffer[slots]
        with self.gather_timer.measure():
            return self.k_buffer[slots], self.v_buffer[slots]

    def _run_decode(self, q: torch.Tensor, fb: ForwardBatch) -> torch.Tensor:
        """One query per request, over that request's whole history.

        This gather reads every request's K/V into a rectangular tensor, once per
        layer, per step. It is a copy, but a temporary, read-only one. Kernels
        like Triton (M5) read the buffers through the index without copying.
        """
        max_len = int(fb.seq_lens.max())
        # [batch, max_len]. Past a request's length these are stale slots, which
        # are valid indices, and the mask below hides them.
        slots = self.req_to_token[fb.req_pool_indices, :max_len]
        keep = torch.arange(max_len, device=q.device)[None, :] < fb.seq_lens[:, None]
        k, v = self._gather(slots)

        out = scaled_dot_product_attention(
            q[:, :, None, :],  # [batch, heads, 1, head_dim]
            k.transpose(1, 2),  # [batch, kv_heads, max_len, head_dim]
            v.transpose(1, 2),
            attn_mask=keep[:, None, None, :],
            enable_gqa=True,
        )
        return out[:, :, 0, :]
