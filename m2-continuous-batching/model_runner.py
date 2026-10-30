"""ModelRunner: owns the model weights and runs one batched forward pass.

SGLang equivalents:
  - ModelRunner.forward          model_executor/model_runner.py
    (reached via TpModelWorker.forward_batch_generation in managers/tp_worker.py)
  - ForwardBatch                 model_executor/forward_batch_info.py, the GPU-side
                                 view of a batch: input_ids, positions, seq_lens, ...
  - Sampler                      layers/sampler.py
  - LogitsProcessor              layers/logits_processor.py

The scheduler decides WHICH requests run; this class only knows HOW to run them.
Keeping the two apart is what lets test_batching.py check the batched forward
against HF directly, without any processes or sockets.

The KV cache still lives inside each request, as one tensor. To run a batch we
pad all requests' caches to the same length, and after the step we copy the
new K/V back out. That copying grows with the total context length and is
exactly the waste the token pool in M3 removes.
"""

from typing import List, Optional, Protocol, Set

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, DynamicCache


class ReqLike(Protocol):
    """What ModelRunner needs from a request (scheduler.Req has it)."""

    input_ids: List[int]
    output_ids: List[int]
    temperature: float
    top_p: float
    # [num_tokens, num_layers, 2 (k/v), num_kv_heads, head_dim], or None before prefill.
    kv: torch.Tensor


def sample(
    logits: torch.Tensor, temperatures: torch.Tensor, top_ps: torch.Tensor
) -> List[int]:
    """Sample one token per row of [batch, vocab] logits, each row with its own params.

    SGLang: layers/sampler.py. It does the same thing with fused kernels
    (flashinfer top-k/top-p sampling) instead of a full sort.
    """
    logits = logits.float()
    greedy = logits.argmax(dim=-1)

    # temperature 0 means greedy. Clamp so the division below stays finite;
    # those rows take the greedy result anyway.
    probs = torch.softmax(logits / temperatures.clamp(min=1e-5)[:, None], dim=-1)
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    # Drop a token if the tokens before it already cover top_p.
    sorted_probs[sorted_probs.cumsum(-1) - sorted_probs > top_ps[:, None]] = 0
    choice = torch.multinomial(sorted_probs, num_samples=1)
    sampled = sorted_idx.gather(-1, choice).squeeze(-1)

    return torch.where(temperatures <= 0, greedy, sampled).tolist()


class ModelRunner:
    def __init__(self, model_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
            .to(self.device)
            .eval()
        )
        self.num_layers = self.model.config.num_hidden_layers
        self.eos_ids = self._get_eos_ids()

    def _get_eos_ids(self) -> Set[int]:
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.model.config.eos_token_id
        if isinstance(eos, int):
            eos = [eos]
        return set(eos or [])

    @torch.inference_mode()
    def forward(self, reqs: List[ReqLike], is_extend: bool) -> List[int]:
        """One forward pass for the whole batch. Returns one sampled token per request."""
        if is_extend:
            logits = self.forward_extend(reqs)
        else:
            logits = self.forward_decode(reqs)

        temperatures = torch.tensor([r.temperature for r in reqs], device=self.device)
        top_ps = torch.tensor([r.top_p for r in reqs], device=self.device)
        return sample(logits, temperatures, top_ps)

    def forward_extend(self, reqs: List[ReqLike]) -> torch.Tensor:
        """Prefill prompts of different lengths together, padded on the right.

        Right padding keeps every real token at the same position it would
        have alone, and the causal mask already stops real tokens from seeing
        the padding after them. The price: each row's last real token is at a
        different index, so we gather those hidden states before the LM head.
        (SGLang does not pad at all: it concatenates all prompts into one flat
        sequence and tells the attention kernel where each one starts.)
        """
        lens = torch.tensor([len(r.input_ids) for r in reqs], device=self.device)
        input_ids = pad_sequence(
            [torch.tensor(r.input_ids) for r in reqs], batch_first=True
        ).to(self.device)
        positions = torch.arange(input_ids.shape[1], device=self.device)
        attention_mask = (positions[None, :] < lens[:, None]).long()

        cache = DynamicCache()
        hidden = self.model.get_decoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=positions[None, :].expand_as(input_ids),
            past_key_values=cache,
            use_cache=True,
        ).last_hidden_state

        # Logits only for the last real token of each request, not for the
        # whole [batch, seq, vocab]. SGLang: layers/logits_processor.py.
        last_hidden = hidden[torch.arange(len(reqs), device=self.device), lens - 1]
        logits = self.model.get_output_embeddings()(last_hidden)

        # Move each request's K/V (without the padding) into the request.
        kv = self._stack_cache(cache)  # [batch, seq, layers, 2, heads, dim]
        for i, req in enumerate(reqs):
            req.kv = kv[i, : len(req.input_ids)].clone()
        return logits

    def forward_decode(self, reqs: List[ReqLike]) -> torch.Tensor:
        """One new token for every running request, KV caches padded on the left.

        Left padding lines up the newest token at the end of every row, so the
        new K/V of every request lands at the same index, -1.
        """
        lens = torch.tensor([r.kv.shape[0] for r in reqs], device=self.device)
        # [batch, max_len, layers, 2, heads, dim], zeros where padded.
        kv = pad_sequence([r.kv for r in reqs], batch_first=True, padding_side="left")
        max_len = kv.shape[1]
        cache = DynamicCache(
            ddp_cache_data=[
                (
                    kv[:, :, layer, 0].transpose(1, 2).contiguous(),  # [batch, heads, len, dim]
                    kv[:, :, layer, 1].transpose(1, 2).contiguous(),
                )
                for layer in range(self.num_layers)
            ]
        )
        del kv

        input_ids = torch.tensor([[r.output_ids[-1]] for r in reqs], device=self.device)
        # Mask covers the cache plus the new token: 0 on the padding, 1 elsewhere.
        cols = torch.arange(max_len + 1, device=self.device)
        attention_mask = (cols[None, :] >= (max_len - lens)[:, None]).long()

        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # The new token's real position is its request's KV length, not max_len.
            position_ids=lens[:, None],
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        ).logits[:, -1]

        # Append the new token's K/V to each request. torch.cat copies the
        # whole cache of every request, every step. M3 fixes this.
        new_kv = self._stack_cache(cache, last_only=True)  # [batch, 1, layers, 2, heads, dim]
        for i, req in enumerate(reqs):
            req.kv = torch.cat([req.kv, new_kv[i]])
        return logits

    def _stack_cache(self, cache: DynamicCache, last_only: bool = False) -> torch.Tensor:
        """HF cache -> one [batch, seq, layers, 2, heads, dim] tensor."""
        seq = slice(-1, None) if last_only else slice(None)
        return torch.stack(
            [
                torch.stack([layer.keys[:, :, seq], layer.values[:, :, seq]], dim=1)
                for layer in cache.layers
            ],
            dim=1,
        ).permute(0, 4, 1, 2, 3, 5)  # [batch, layers, 2, heads, seq, dim] -> token-major
