"""Scheduler process: owns the model and the GPU, and runs the generation loop.

SGLang equivalent: python/sglang/srt/managers/scheduler.py
  - Scheduler.event_loop_normal       (~line 2036)
  - Scheduler.get_next_batch_to_run   (~line 3970)
  - Scheduler.get_new_batch_prefill   (~line 4128), with PrefillAdder in schedule_policy.py
  - Scheduler.update_running_batch    (~line 4550)
  - Scheduler.run_batch               (~line 4703)
  - Scheduler.process_batch_result    (~line 5170), then
    scheduler_components/batch_result_processor.py (process_batch_result_prefill/_decode)
  - ScheduleBatch, Req                managers/schedule_batch.py
  - ForwardMode                       model_executor/forward_batch_info.py (~line 199)

The forward pass itself lives in model_runner.py.

Each loop iteration runs ONE forward pass, either:
  - EXTEND: prefill a batch of newly arrived requests, or
  - DECODE: one new token for every running request.
New requests join the running batch right after their prefill, and finished
requests leave it right after the step that finished them. Nobody waits for
the whole batch to drain, which is the difference from "static" batching.

v2 adds a TOKEN BUDGET. The KV pool has a fixed number of slots, so admission
checks that a request's slots will exist before starting it:
  - rem_tokens = free slots - what running requests may still need
  - a request runs only if prompt + max_new_tokens fits in rem_tokens
Reserving the full max_new_tokens means a decode step can never run out of
slots. It also wastes capacity, because most requests stop at EOS long before
that. SGLang's PrefillAdder (schedule_policy.py ~line 605) overcommits instead
and handles the overflow.

So does this scheduler with --new-token-ratio below 1: it reserves
only that share of each request's new tokens. If a decode step then doesn't fit,
update_running_batch retracts requests: it frees their KV and puts them back at
the head of the queue. Their output so far is kept, and they prefill it again.

v3 adds the PREFIX CACHE, which touches three places:
  - admission matches a request against the tree, and locks what it found. It only
    pays for what it has to compute, so the budget above gets three changes:
      rem_tokens = free + EVICTABLE - what running requests may still need
      a request costs (uncached tokens + max_new_tokens) + what locking takes
      out of evictable. SGLang: PrefillAdder.add_one_req (schedule_policy.py ~1358).
  - allocation evicts when the free list runs short (ModelRunner.ensure_free).
  - results: after prefill the request's K/V goes to the tree, so that the next
    arrival hits it at once. A finished request gives the tree the rest.
"""

import collections
import enum
import time
import traceback
from typing import Deque, List, Optional, Set

import torch
import zmq
from loguru import logger

from common import (
    AbortReq,
    BatchTokenIDOut,
    EngineOptions,
    FlushCacheReq,
    PortArgs,
    TokenizedGenerateReq,
    get_socket,
    setup_logging,
)
from model_runner import ModelRunner
from radix_cache import RadixCache, TreeNode

# How many prompt tokens to send to the detokenizer as decoding context.
# SGLang: INIT_INCREMENTAL_DETOKENIZATION_OFFSET in schedule_batch.py.
PROMPT_TAIL_LEN = 5

# Log decode stats every N decode steps. SGLang: --decode-log-interval (40).
DECODE_LOG_INTERVAL = 40

# --schedule-policy lpm: sorting matches the whole queue, so skip it when the queue is long.
# SGLang: WAITING_QUEUE_PREFIX_MATCH_MAX in schedule_policy.py.
WAITING_QUEUE_PREFIX_MATCH_MAX = 128

# --in-batch-dedup. SGLang: IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD and
# ..._DEPRIORITIZE_THRESHOLD in schedule_policy.py. Short prefixes such as "The"
# are shared by chance, and aren't worth delaying anybody for.
IN_BATCH_PREFIX_THRESHOLD = 32


class ForwardMode(enum.Enum):
    # SGLang calls prefill "extend": with a prefix cache (M4) a request only
    # computes the part of the prompt that is not cached yet, i.e. it extends it.
    EXTEND = enum.auto()
    DECODE = enum.auto()


class Req:
    """Per-request state on the scheduler side. SGLang: schedule_batch.Req."""

    def __init__(self, r: TokenizedGenerateReq):
        self.rid = r.rid
        self.input_ids = r.input_ids
        self.max_new_tokens = r.max_new_tokens
        self.temperature = r.temperature
        self.top_p = r.top_p

        self.output_ids: List[int] = []
        self.num_sent = 0  # How many output_ids were already streamed out.
        self.finish_reason: Optional[str] = None

        # The K/V itself lives in the pool. The request only holds its table row.
        self.req_pool_idx: Optional[int] = None
        # Covers input_ids + output_ids[:-1]: the last sampled token has not
        # been through the model yet, it is the input of the next decode step.
        self.kv_allocated_len = 0

        # What the tree offers. See radix_cache.CacheReq.
        self.prefix_len = 0
        self.prefix_slots: Optional[torch.Tensor] = None
        self.last_node: Optional[TreeNode] = None
        self.cache_protected_len = 0
        self.cache_inserted_len = 0

    def remaining_kv_tokens(self, ratio: float = 1.0, page_size: int = 1) -> int:
        """How many KV slots to reserve for this request.

        With ratio=1 this is the most it can still take. With a lower ratio it's
        a bet that it stops early. Either way, slots come in whole pages.
        """

        def aligned(num_tokens: int) -> int:
            return (num_tokens + page_size - 1) // page_size * page_size

        new_tokens = self.max_new_tokens - len(self.output_ids)
        if self.req_pool_idx is None:
            # Prefill covers input_ids + output_ids. The last output token never
            # needs a slot, see kv_allocated_len. The cached prefix has its slots,
            # and they stay in the tree. It's in whole pages, so this stays aligned.
            num_prefill = len(self.input_ids) + len(self.output_ids)
            return aligned(num_prefill + int((new_tokens - 1) * ratio)) - self.prefix_len
        held = self.kv_allocated_len
        return aligned(held + int(new_tokens * ratio)) - aligned(held)

    def check_finished(self, eos_ids: Set[int]):
        if self.output_ids[-1] in eos_ids:
            self.finish_reason = "stop"
        elif len(self.output_ids) >= self.max_new_tokens:
            self.finish_reason = "length"


class ScheduleBatch:
    """The requests that run in one forward pass. SGLang: schedule_batch.ScheduleBatch.

    SGLang's version also holds the GPU tensors for the step (input_ids,
    positions, KV slot indices, sampling info). Ours are in ForwardBatch, which
    ModelRunner builds.
    """

    def __init__(self, reqs: List[Req], forward_mode: ForwardMode):
        self.reqs = reqs
        self.forward_mode = forward_mode

    def __len__(self):
        return len(self.reqs)


class Scheduler:
    def __init__(
        self,
        model_path: str,
        port_args: PortArgs,
        max_running_requests: int,
        max_prefill_tokens: int,
        max_total_tokens: Optional[int],
        mem_fraction_static: float,
        context_length: int,
        options: Optional[EngineOptions] = None,
    ):
        self.options = options or EngineOptions()
        context = zmq.Context(2)
        self.recv_from_tokenizer = get_socket(
            context, zmq.PULL, port_args.scheduler_input, bind=True
        )
        self.send_to_detokenizer = get_socket(
            context, zmq.PUSH, port_args.detokenizer_input, bind=False
        )

        self.model_runner = ModelRunner(
            model_path,
            max_running_requests=max_running_requests,
            max_total_tokens=max_total_tokens,
            mem_fraction_static=mem_fraction_static,
            context_length=context_length,
            options=self.options,
        )
        self.eos_ids = self.model_runner.eos_ids
        self.tree_cache = self.model_runner.tree_cache
        pool_gb = self.model_runner.max_total_tokens * self.model_runner.cell_size / 2**30
        logger.info(
            f"Loaded {model_path} on {self.model_runner.device}, eos_ids={self.eos_ids}, "
            f"KV pool: {self.model_runner.max_total_tokens} tokens, "
            f"{self.model_runner.cell_size} bytes each, {pool_gb:.2f} GB"
        )

        self.max_running_requests = max_running_requests
        self.max_prefill_tokens = max_prefill_tokens

        self.waiting_queue: Deque[Req] = collections.deque()
        # Requests that finished prefill and are now decoding.
        self.running_batch: List[Req] = []

        # Stats for the log lines.
        self.num_decode_steps = 0
        self.num_decode_tokens = 0
        self.last_decode_log_time = time.perf_counter()
        self.num_retracted = 0
        self.num_prefill_tokens = 0  # Tokens actually computed by prefill.
        self.num_cached_tokens = 0  # Tokens that prefill skipped.

    # ----------------------------------------------------------------- loop

    def event_loop(self):
        """SGLang: event_loop_normal. Ingest -> pick a batch -> run it -> process the result."""
        while True:
            # When there is nothing to do, block on the socket instead of spinning.
            idle = not self.running_batch and not self.waiting_queue
            self.ingest_requests(block=idle)

            batch = self.get_next_batch_to_run()
            if batch is None:
                continue
            next_token_ids = self.run_batch(batch)
            self.process_batch_result(batch, next_token_ids)

    def ingest_requests(self, block: bool):
        """Drain everything waiting on the socket without blocking the GPU loop."""
        recv_reqs = []
        if block:
            recv_reqs.append(self.recv_from_tokenizer.recv_pyobj())
        while True:
            try:
                recv_reqs.append(self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK))
            except zmq.ZMQError:
                break

        for recv_req in recv_reqs:
            if isinstance(recv_req, TokenizedGenerateReq):
                self.handle_generate_request(Req(recv_req))
            elif isinstance(recv_req, AbortReq):
                self.abort_request(recv_req.rid)
            elif isinstance(recv_req, FlushCacheReq):
                self.flush_cache()
            else:
                raise ValueError(f"Unknown request type: {type(recv_req)}")

    def handle_generate_request(self, req: Req):
        """Make sure a request can ever fit. SGLang: Scheduler ~line 3307, and
        validate_input_length in managers/utils.py.

        Without this, a request bigger than the pool would sit at the head of
        the queue forever and block everyone behind it.
        """
        limit = self.model_runner.max_context_len
        if len(req.input_ids) >= limit:
            logger.warning(f"Rejected a {len(req.input_ids)}-token prompt, limit is {limit}")
            req.finish_reason = "abort"
            self.stream_output([req])
            return
        req.max_new_tokens = min(req.max_new_tokens, limit - len(req.input_ids))
        self.waiting_queue.append(req)

    def abort_request(self, rid: str):
        for req in self.waiting_queue:
            if req.rid == rid:
                self.waiting_queue.remove(req)
                # Usually it never started, so the detokenizer has no state for
                # it. A retracted request did start, and needs a final message.
                if req.num_sent > 0:
                    req.finish_reason = "abort"
                    self.stream_output([req])
                return
        for req in self.running_batch:
            if req.rid == rid:
                # It still runs in the next decode step, but process_batch_result
                # drops its token and sends a final message, so the detokenizer
                # frees its state and the request leaves the batch.
                req.finish_reason = "abort"
                return

    def flush_cache(self) -> bool:
        """Forget all cached K/V. SGLang: Scheduler.flush_cache (~line 5628).

        Only while idle. In SGLang that's because it resets the pools outright.
        Ours only evicts, so it would be safe mid-flight: locked paths would just
        stay. We keep the rule, because callers want "the cache is empty" to be true
        when this returns, and that's only possible with nobody running.
        """
        if self.running_batch or self.waiting_queue:
            logger.warning(
                f"Cache not flushed, #running-req: {len(self.running_batch)}, "
                f"#queue-req: {len(self.waiting_queue)}"
            )
            return False
        num_evicted = self.tree_cache.evict(self.tree_cache.evictable_size)
        self.num_prefill_tokens = self.num_cached_tokens = 0
        logger.info(f"Cache flushed, {num_evicted} tokens freed")
        return True

    # ------------------------------------------------------------ scheduling

    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
        """Prefill first: if new requests fit, run them now, and decode next step.

        Prioritizing prefill gives new requests a low time to first token, at the
        cost of a small hiccup for everyone already decoding. This is SGLang's
        default too.
        """
        new_batch = self.get_new_batch_prefill()
        if new_batch is not None:
            return new_batch
        self.update_running_batch()
        if self.running_batch:
            return ScheduleBatch(self.running_batch, ForwardMode.DECODE)
        return None

    def update_running_batch(self):
        """Retract requests until the next decode step fits. SGLang:
        Scheduler.update_running_batch (~line 4550), retract_decode (~line 3463).

        This can't fire with new_token_ratio=1, because admission reserved every
        slot. It can't empty the batch either: a request alone in the pool always
        fits, because handle_generate_request capped it at max_context_len.

        Eviction comes first, so a running request is never pushed out to keep idle
        cache. Only when nothing is left to evict does somebody lose real work.

        Like SGLang, a retracted request gives its K/V back instead of caching it
        (the TODO at schedule_batch.py ~2400). --cache-retracted changes that.
        """
        runner = self.model_runner
        while not runner.ensure_free(runner.decode_slots_needed(self.running_batch)):
            req = max(self.running_batch, key=lambda r: r.max_new_tokens - len(r.output_ids))
            self.running_batch.remove(req)
            aborted = req.finish_reason is not None
            runner.free_req(req, cache=self.options.cache_retracted and not aborted)
            if aborted:
                self.stream_output([req])  # Aborted: this was its final step anyway.
                continue
            self.waiting_queue.appendleft(req)
            self.num_retracted += 1
            logger.warning(
                f"KV pool is full. Retracted a request with {len(req.input_ids)} prompt and "
                f"{len(req.output_ids)} output tokens, #retracted: {self.num_retracted}, "
                f"token usage: {self._token_usage():.2f}"
            )

    def _reserved_tokens(self, req: Req) -> int:
        return req.remaining_kv_tokens(
            self.options.new_token_ratio, self.model_runner.allocator.page_size
        )

    def _get_rem_tokens(self) -> int:
        """Slots that no running request has reserved. Evictable cache counts as free:
        admission may spend it, and ModelRunner.ensure_free turns it into slots."""
        available = self.model_runner.allocator.available_size()
        return (
            available
            + self.tree_cache.evictable_size
            - sum(self._reserved_tokens(r) for r in self.running_batch)
        )

    def _sort_waiting_queue(self):
        """Longest prefix match first. SGLang: calc_priority, and
        _sort_by_longest_prefix in schedule_policy.py (~line 439).

        The sort is stable, so ties stay first come, first served.
        """
        if self.options.schedule_policy != "lpm":
            return
        if not 1 < len(self.waiting_queue) <= WAITING_QUEUE_PREFIX_MATCH_MAX:
            return
        for req in self.waiting_queue:
            self.tree_cache.match_req(req)
        self.waiting_queue = collections.deque(
            sorted(self.waiting_queue, key=lambda r: -r.prefix_len)
        )

    def _shares_with_batch(self, req: Req, batch_tree: RadixCache) -> bool:
        """In-batch dedup: does this batch already compute req's prefix?

        batch_tree holds the prompts of the batch so far. If req has little in the
        cache but a long match in there, then running it next step instead makes it
        a cache hit, so it's better to wait. SGLang: _compute_prefix_matches in
        schedule_policy.py (~line 344), which sorts such requests last instead.
        """
        ids = req.input_ids + req.output_ids
        if req.prefix_len > IN_BATCH_PREFIX_THRESHOLD:
            return False
        if len(batch_tree.match_prefix(ids)[0]) >= IN_BATCH_PREFIX_THRESHOLD:
            return True
        batch_tree.insert(ids, torch.zeros(len(ids), dtype=torch.int64))
        return False

    def get_new_batch_prefill(self) -> Optional[ScheduleBatch]:
        """Take requests from the waiting queue while they fit. SGLang: PrefillAdder."""
        tree = self.tree_cache
        self._sort_waiting_queue()
        can_run: List[Req] = []
        num_tokens = 0
        num_cached = 0
        rem_tokens = self._get_rem_tokens()
        batch_tree = None
        if self.options.in_batch_dedup:
            batch_tree = RadixCache(None, None, page_size=tree.page_size)
        while self.waiting_queue:
            if len(self.running_batch) + len(can_run) >= self.max_running_requests:
                break
            req = self.waiting_queue[0]
            # The tree changes while a request waits, so match at the last moment.
            tree.match_req(req)
            if batch_tree is not None and self._shares_with_batch(req, batch_tree):
                break
            # Always admit at least one request, even if its prompt alone is
            # over the budget, or it would wait forever.
            num_prefill = len(req.input_ids) + len(req.output_ids) - req.prefix_len
            if can_run and num_tokens + num_prefill > self.max_prefill_tokens:
                break
            # Lock before anything can evict. What this lock protects can no longer
            # pay for anybody's slots, so it's charged here too. Requests in this
            # batch can share a prefix, and the second one is only charged what the
            # first one left evictable.
            locked = tree.lock_req(req)
            reserved = self._reserved_tokens(req)
            # FIFO: a request that doesn't fit blocks the ones behind it, even
            # if they are smaller, so that big requests can't starve.
            if reserved + locked > rem_tokens:
                tree.unlock_req(req)
                break
            can_run.append(self.waiting_queue.popleft())
            num_tokens += num_prefill
            num_cached += req.prefix_len
            rem_tokens -= reserved + locked

        if not can_run:
            return None
        self.num_prefill_tokens += num_tokens
        self.num_cached_tokens += num_cached
        logger.info(
            f"Prefill batch. #new-seq: {len(can_run)}, #new-token: {num_tokens}, "
            f"#cached-token: {num_cached}, cache hit rate: {self._cache_hit_rate():.2f}, "
            f"#running-req: {len(self.running_batch)}, #queue-req: {len(self.waiting_queue)}, "
            f"token usage: {self._token_usage():.2f}"
        )
        return ScheduleBatch(can_run, ForwardMode.EXTEND)

    def _cache_hit_rate(self) -> float:
        """Since the start, not per batch. SGLang logs the rate of each batch."""
        total = self.num_prefill_tokens + self.num_cached_tokens
        return self.num_cached_tokens / total if total else 0.0

    def _token_usage(self) -> float:
        """Share of the pool that can't be used right now. A pool full of idle cache
        is at 0: it's as good as free. _cache_usage is the other half of the story."""
        runner = self.model_runner
        available = runner.allocator.available_size() + self.tree_cache.evictable_size
        return 1 - available / runner.max_total_tokens

    def _cache_usage(self) -> float:
        return self.tree_cache.total_size() / self.model_runner.max_total_tokens

    # -------------------------------------------------------------- forward

    def run_batch(self, batch: ScheduleBatch) -> List[int]:
        """SGLang: Scheduler.run_batch -> TpModelWorker -> ModelRunner.forward."""
        return self.model_runner.forward(
            batch.reqs, is_extend=batch.forward_mode == ForwardMode.EXTEND
        )

    # --------------------------------------------------------------- output

    def process_batch_result(self, batch: ScheduleBatch, next_token_ids: List[int]):
        """Append the new tokens, stream them, and update who is still running."""
        for req, token_id in zip(batch.reqs, next_token_ids):
            if req.finish_reason is not None:
                continue  # Aborted while in this batch: drop the token.
            req.output_ids.append(token_id)
            req.check_finished(self.eos_ids)

        self.stream_output(batch.reqs)

        for req in batch.reqs:
            if req.finish_reason is not None:
                # Free the KV slots right away. An aborted request's K/V is as good
                # as any other, but its client is gone, so SGLang doesn't keep it.
                self.model_runner.free_req(req, cache=req.finish_reason != "abort")
            elif batch.forward_mode == ForwardMode.EXTEND:
                # Now, not at the end: long answers would keep their prompt out of
                # the cache for as long as they run. SGLang: checkpoint_kv_cache.
                self.tree_cache.checkpoint(req)

        # Finished requests leave the batch; freshly prefilled ones join it.
        # SGLang: ScheduleBatch.filter_batch / merge_batch.
        still_running = [r for r in batch.reqs if r.finish_reason is None]
        if batch.forward_mode == ForwardMode.EXTEND:
            self.running_batch.extend(still_running)
        else:
            self.running_batch = still_running
            self._log_decode_stats(len(batch))

    def stream_output(self, reqs: List[Req]):
        """Send the new token ids of all these requests in ONE message.

        SGLang: OutputStreamer.stream_output. It can also send only every
        `stream_interval` tokens.
        """
        out = BatchTokenIDOut(
            rids=[], new_token_ids=[], prompt_tails=[], finish_reasons=[], completion_tokens=[]
        )
        for req in reqs:
            new_ids = req.output_ids[req.num_sent :]
            if not new_ids and req.finish_reason is None:
                continue
            is_first = req.num_sent == 0
            req.num_sent = len(req.output_ids)

            out.rids.append(req.rid)
            out.new_token_ids.append(new_ids)
            out.prompt_tails.append(req.input_ids[-PROMPT_TAIL_LEN:] if is_first else None)
            out.finish_reasons.append(req.finish_reason)
            out.completion_tokens.append(len(req.output_ids))

        if out.rids:
            self.send_to_detokenizer.send_pyobj(out)

    def _log_decode_stats(self, batch_size: int):
        self.num_decode_steps += 1
        self.num_decode_tokens += batch_size
        if self.num_decode_steps % DECODE_LOG_INTERVAL != 0:
            return
        now = time.perf_counter()
        throughput = self.num_decode_tokens / (now - self.last_decode_log_time)
        logger.info(
            f"Decode batch. #running-req: {len(self.running_batch)}, "
            f"#queue-req: {len(self.waiting_queue)}, token usage: {self._token_usage():.2f}, "
            f"cache usage: {self._cache_usage():.2f}, "
            f"page waste: {self.model_runner.page_waste(self.running_batch):.2f}, "
            f"#retracted: {self.num_retracted}, "
            f"gpu mem: {self.model_runner.gpu_memory_mb():.0f} MB, "
            f"gen throughput (token/s): {throughput:.1f}"
        )
        self.num_decode_tokens = 0
        self.last_decode_log_time = now


def run_scheduler_process(
    model_path: str,
    port_args: PortArgs,
    max_running_requests: int,
    max_prefill_tokens: int,
    max_total_tokens: Optional[int],
    mem_fraction_static: float,
    context_length: int,
    options: EngineOptions,
    ready_pipe,
):
    setup_logging("scheduler")
    try:
        scheduler = Scheduler(
            model_path,
            port_args,
            max_running_requests,
            max_prefill_tokens,
            max_total_tokens,
            mem_fraction_static,
            context_length,
            options,
        )
    except Exception:
        ready_pipe.send(traceback.format_exc())
        raise
    ready_pipe.send("ready")
    scheduler.event_loop()
