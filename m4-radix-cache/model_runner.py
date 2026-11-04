"""ModelRunner: owns the weights and the KV pools, and runs one batched forward pass.

SGLang equivalents:
  - ModelRunner.forward          model_executor/model_runner.py
    (reached via TpModelWorker.forward_batch_generation in managers/tp_worker.py)
  - _prepare_extend/_prepare_decode
                                 ScheduleBatch.prepare_for_extend (~line 2859) and
                                 prepare_for_decode (~line 3743) in managers/schedule_batch.py
  - free_req                     mem_cache/common.py: release_kv_cache (~line 419)
  - ensure_free                  mem_cache/common.py: evict_from_tree_cache (~line 284)
  - pool sizing                  model_executor/pool_configurator.py (~line 915)
  - Sampler                      layers/sampler.py
  - LogitsProcessor              layers/logits_processor.py

The scheduler decides WHICH requests run; this class only knows HOW to run them.
Keeping the two apart is what lets tests/test_batching.py check the batched forward
against HF directly, without any processes or sockets.

In M2 HF's model ran the batch, as padded rectangles. Here we keep HF's layers
but call them ourselves on one flat sequence, with PoolAttention in place of
Qwen3Attention. That's why self.model can no longer run `generate`.
"""

import dataclasses
from typing import List, Optional, Protocol, Set

import torch
from transformers import AutoModelForCausalLM

from attention import ForwardBatch, GatherTimer, PoolAttention
from common import EngineOptions
from memory_pool import ReqToTokenPool, TokenAllocator, TokenToKVPool
from radix_cache import RadixCache, TreeNode

# Arbitrary. There's no free-memory query on CPU, so pick something small.
CPU_DEFAULT_TOTAL_TOKENS = 8192


class ReqLike(Protocol):
    """What ModelRunner needs from a request (scheduler.Req has it)."""

    input_ids: List[int]
    output_ids: List[int]
    temperature: float
    top_p: float
    req_pool_idx: Optional[int]  # Row in req_to_token, None while it holds no KV.
    kv_allocated_len: int  # How many tokens of this request have a slot.
    prefix_len: int  # See radix_cache.CacheReq for these.
    prefix_slots: Optional[torch.Tensor]
    last_node: Optional[TreeNode]
    cache_protected_len: int
    cache_inserted_len: int


@dataclasses.dataclass
class FakeReq:
    """A ReqLike with no scheduler behind it, for scripts that drive ModelRunner directly."""

    input_ids: List[int]
    max_new_tokens: int
    output_ids: List[int] = dataclasses.field(default_factory=list)
    temperature: float = 0.0
    top_p: float = 1.0
    req_pool_idx: Optional[int] = None
    kv_allocated_len: int = 0
    prefix_len: int = 0
    prefix_slots: Optional[torch.Tensor] = None
    last_node: Optional[TreeNode] = None
    cache_protected_len: int = 0
    cache_inserted_len: int = 0


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
    def __init__(
        self,
        model_path: str,
        max_running_requests: int = 64,
        max_total_tokens: Optional[int] = None,
        mem_fraction_static: float = 0.6,
        context_length: int = 8192,
        device: Optional[str] = None,
        options: Optional[EngineOptions] = None,
    ):
        options = options or EngineOptions()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
            .to(self.device)
            .eval()
        )
        config = self.model.config
        if config.model_type != "qwen3" or config.sliding_window is not None:
            raise ValueError("PoolAttention only supports Qwen3 without sliding window.")
        self.eos_ids = self._get_eos_ids()

        layers = self.model.get_decoder().layers
        num_kv_heads = config.num_key_value_heads
        head_dim = layers[0].self_attn.head_dim
        self.cell_size = TokenToKVPool.cell_size(
            len(layers), num_kv_heads, head_dim, dtype
        )
        self.max_total_tokens = max_total_tokens or self._size_from_free_memory(
            mem_fraction_static
        )
        # Whole pages only. Otherwise the longest request could need one page
        # more than exist, and it would wait in the queue forever.
        self.max_total_tokens -= self.max_total_tokens % options.page_size
        self.max_context_len = min(
            context_length, config.max_position_embeddings, self.max_total_tokens
        )

        self.allocator = TokenAllocator(
            self.max_total_tokens,
            self.device,
            page_size=options.page_size,
            lazy_free=options.lazy_free,
            check_double_free=options.check_double_free,
        )
        self.req_to_token_pool = ReqToTokenPool(
            max_running_requests, self.max_context_len, self.device
        )
        self.kv_pool = TokenToKVPool(
            self.max_total_tokens, len(layers), num_kv_heads, head_dim, dtype, self.device
        )
        self.tree_cache = self._build_tree_cache(options)
        self.gather_timer = GatherTimer(self.device) if options.profile_gather else None
        for i, layer in enumerate(layers):
            layer.self_attn = PoolAttention(
                layer.self_attn,
                self.kv_pool.k_buffer[i],
                self.kv_pool.v_buffer[i],
                self.req_to_token_pool.req_to_token,
                batched_prefill=options.batched_prefill,
                gather_timer=self.gather_timer,
            )

    def _build_tree_cache(self, options: EngineOptions) -> RadixCache:
        return RadixCache(
            self.allocator,
            self.req_to_token_pool.req_to_token,
            page_size=options.page_size,
            disable=options.disable_radix_cache,
            eviction_policy=options.eviction_policy,
            fast_match=options.fast_match,
            trim_leaves=options.cache_retracted,
        )

    def _get_eos_ids(self) -> Set[int]:
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.model.config.eos_token_id
        if isinstance(eos, int):
            eos = [eos]
        return set(eos or [])

    def _size_from_free_memory(self, mem_fraction_static: float) -> int:
        """How many tokens fit. Called after loading, so the weights are already out."""
        if self.device == "cpu":
            return CPU_DEFAULT_TOTAL_TOKENS
        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        return int(free_bytes * mem_fraction_static) // self.cell_size

    def gpu_memory_mb(self) -> float:
        if self.device == "cpu":
            return 0.0
        return torch.cuda.memory_allocated(self.device) / 2**20

    # ------------------------------------------------------------------ KV slots

    def ensure_free(self, num_slots: int) -> bool:
        """Evict until the allocator has num_slots. False if even that isn't enough."""
        missing = num_slots - self.allocator.available_size()
        if missing > 0:
            self.tree_cache.evict(missing)
        return self.allocator.available_size() >= num_slots

    def _prepare_extend(self, reqs: List[ReqLike]) -> ForwardBatch:
        """Point each request's row at its cached prefix, and take new slots for the rest.

        The cached prefix needs no slots and no forward pass, which is the whole
        point of M4. Its length is in whole pages, so the new tokens start on a
        fresh page.

        A retracted request prefills its output too, the newest
        token included, so its state after this step is what a decode step
        would have left.
        """
        all_ids = [r.input_ids + r.output_ids for r in reqs]
        lens = [len(ids) for ids in all_ids]
        prefix_lens = [r.prefix_len for r in reqs]
        extend_lens = [n - p for n, p in zip(lens, prefix_lens)]
        held = [self.allocator.aligned(n) for n in extend_lens]
        # Admission counted free and evictable slots, so these can't run out.
        assert self.ensure_free(sum(held))
        allocated = self.allocator.alloc(sum(held))
        assert allocated is not None

        table = self.req_to_token_pool.req_to_token
        # With pages, a request may not use all of its last page.
        used = [s[:n] for s, n in zip(allocated.split(held), extend_lens)]
        for req, slots, prefix_len, seq_len in zip(reqs, used, prefix_lens, lens):
            req.req_pool_idx = self.req_to_token_pool.alloc()
            assert req.req_pool_idx is not None
            row = table[req.req_pool_idx]
            if prefix_len > 0:
                row[:prefix_len] = req.prefix_slots
            row[prefix_len:seq_len] = slots
            req.kv_allocated_len = seq_len
            req.cache_protected_len = prefix_len

        return ForwardBatch(
            is_extend=True,
            input_ids=torch.tensor(
                [t for ids, p in zip(all_ids, prefix_lens) for t in ids[p:]],
                device=self.device,
            ),
            positions=torch.cat(
                [torch.arange(p, n, device=self.device) for p, n in zip(prefix_lens, lens)]
            ),
            out_cache_loc=torch.cat(used),
            req_pool_indices=torch.tensor(
                [r.req_pool_idx for r in reqs], device=self.device
            ),
            seq_lens=torch.tensor(lens, device=self.device),
            extend_lens=torch.tensor(extend_lens, device=self.device),
            prefix_lens=torch.tensor(prefix_lens, device=self.device),
        )

    def _prepare_decode(self, reqs: List[ReqLike]) -> ForwardBatch:
        """Take one slot per request and append it to the request's table row.

        This is all a decode step does to memory: O(batch), however long the
        requests are. M2's torch.cat here was O(everything cached).

        With pages, a request only touches the allocator when it crosses into a
        new page. Until then its next slot is the one after its last slot.
        """
        table = self.req_to_token_pool.req_to_token
        rows = torch.tensor([r.req_pool_idx for r in reqs], device=self.device)
        # The newest output token hasn't been through the model yet.
        seq_lens = torch.tensor(
            [len(r.input_ids) + len(r.output_ids) for r in reqs], device=self.device
        )
        positions = seq_lens - 1

        page_size = self.allocator.page_size
        out_cache_loc = table[rows, positions - 1] + 1
        new_page = positions % page_size == 0
        num_new_pages = int(new_page.sum())
        if num_new_pages:
            allocated = self.allocator.alloc(num_new_pages * page_size)
            # update_running_batch makes room before every decode step.
            assert allocated is not None
            out_cache_loc[new_page] = allocated[::page_size]

        table[rows, positions] = out_cache_loc
        for req, seq_len in zip(reqs, seq_lens.tolist()):
            req.kv_allocated_len = seq_len

        return ForwardBatch(
            is_extend=False,
            input_ids=torch.tensor([r.output_ids[-1] for r in reqs], device=self.device),
            positions=positions,
            out_cache_loc=out_cache_loc,
            req_pool_indices=rows,
            seq_lens=seq_lens,
            extend_lens=torch.ones(len(reqs), dtype=torch.int64, device=self.device),
            prefix_lens=positions,
        )

    def decode_slots_needed(self, reqs: List[ReqLike]) -> int:
        """Slots the next decode step takes from the allocator. SGLang:
        new_tokens_required_next_decode in schedule_batch.py (~line 3415)."""
        page_size = self.allocator.page_size
        return page_size * sum(r.kv_allocated_len % page_size == 0 for r in reqs)

    def page_waste(self, reqs: List[ReqLike]) -> float:
        """Share of the slots held by these requests that no token uses."""
        used = sum(r.kv_allocated_len for r in reqs)
        held = sum(self.allocator.aligned(r.kv_allocated_len) for r in reqs)
        return 1 - used / held if held else 0.0

    def free_req(self, req: ReqLike, cache: bool = True):
        """Release the request. No K/V is touched.

        With cache=True the tree keeps what it can, and only the tail past the last
        whole page goes back to the allocator. With cache=False the tree keeps only
        what it already had, and the prefix that this request reused stays in it.
        """
        row = req.req_pool_idx
        if row is None:
            # See the README (Double free) for what this stops.
            raise RuntimeError("free_req on a request that holds no KV: double free?")
        if cache:
            self.tree_cache.checkpoint(req)
        table = self.req_to_token_pool.req_to_token
        self.allocator.free(table[row, req.cache_protected_len : req.kv_allocated_len])
        self.tree_cache.unlock_req(req)
        self.req_to_token_pool.free(row)
        req.req_pool_idx = None
        req.kv_allocated_len = 0
        req.prefix_len = 0
        req.prefix_slots = None
        req.cache_protected_len = 0

    # ------------------------------------------------------------------- forward

    @torch.inference_mode()
    def forward(self, reqs: List[ReqLike], is_extend: bool) -> List[int]:
        """One forward pass for the whole batch. Returns one sampled token per request."""
        if is_extend:
            forward_batch = self._prepare_extend(reqs)
        else:
            forward_batch = self._prepare_decode(reqs)

        hidden = self._run_layers(forward_batch)

        # Logits only for the last token of each request. SGLang: layers/logits_processor.py.
        last_tokens = forward_batch.extend_lens.cumsum(0) - 1
        logits = self.model.get_output_embeddings()(hidden[last_tokens])

        temperatures = torch.tensor([r.temperature for r in reqs], device=self.device)
        top_ps = torch.tensor([r.top_p for r in reqs], device=self.device)
        return sample(logits, temperatures, top_ps)

    def _run_layers(self, fb: ForwardBatch) -> torch.Tensor:
        """Qwen3Model.forward without its masks and cache. Returns [num_tokens, hidden]."""
        decoder = self.model.get_decoder()
        hidden = decoder.embed_tokens(fb.input_ids[None])
        position_embeddings = decoder.rotary_emb(hidden, fb.positions[None])
        for layer in decoder.layers:
            hidden = layer(
                hidden, position_embeddings=position_embeddings, forward_batch=fb
            )
        return decoder.norm(hidden)[0]
