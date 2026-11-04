"""ModelRunner's slot logic without a model, for tests and cache_sim.py.

Slot bookkeeping is the easy thing to get subtly wrong here: a request that reads
another one's slot gives wrong text, and nothing crashes. So this fake keeps a
stand-in for K/V: each slot holds a hash of the tokens up to and including its own,
which is exactly what real K/V depends on. A forward pass writes it into the new
slots and checks it on every slot it only reads.
"""

from typing import Dict, List

from common import EngineOptions
from memory_pool import ReqToTokenPool, TokenAllocator
from model_runner import ModelRunner

TOKEN = 1  # What every "model" step outputs. Not an EOS id.


class FakeRunner(ModelRunner):
    verify_kv = True  # cache_sim.py turns this off to go faster.

    def __init__(
        self,
        model_path: str,
        max_running_requests: int,
        max_total_tokens: int,
        mem_fraction_static: float,
        context_length: int,
        options: EngineOptions,
    ):
        self.device = "cpu"
        self.eos_ids = {0}
        self.cell_size = 1
        self.max_total_tokens = max_total_tokens
        self.max_context_len = min(context_length, max_total_tokens)
        self.allocator = TokenAllocator(
            max_total_tokens,
            "cpu",
            page_size=options.page_size,
            lazy_free=options.lazy_free,
            check_double_free=True,
        )
        self.req_to_token_pool = ReqToTokenPool(
            max_running_requests, self.max_context_len, "cpu"
        )
        self.tree_cache = self._build_tree_cache(options)
        self.kv: Dict[int, int] = {}  # Slot -> hash of the tokens up to it.

    def forward(self, reqs, is_extend: bool) -> List[int]:
        fb = self._prepare_extend(reqs) if is_extend else self._prepare_decode(reqs)
        self.last_batch = fb
        if self.verify_kv:
            table = self.req_to_token_pool.req_to_token
            for req, first_new in zip(reqs, fb.prefix_lens.tolist()):
                ids = req.input_ids + req.output_ids
                slots = table[req.req_pool_idx, : len(ids)].tolist()
                state = 0
                for pos, (token, slot) in enumerate(zip(ids, slots)):
                    state = hash((state, token))
                    if pos >= first_new:
                        self.kv[slot] = state
                    elif self.kv.get(slot) != state:
                        raise AssertionError(f"wrong K/V at position {pos}, slot {slot}")
        return [TOKEN] * len(reqs)
