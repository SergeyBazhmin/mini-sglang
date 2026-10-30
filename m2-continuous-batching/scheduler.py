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

v1 does CONTINUOUS BATCHING. Each loop iteration runs ONE forward pass, either:
  - EXTEND: prefill a batch of newly arrived requests, or
  - DECODE: one new token for every running request.
New requests join the running batch right after their prefill, and finished
requests leave it right after the step that finished them. Nobody waits for
the whole batch to drain, which is the difference from "static" batching.
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
    PortArgs,
    TokenizedGenerateReq,
    get_socket,
    setup_logging,
)
from model_runner import ModelRunner

# How many prompt tokens to send to the detokenizer as decoding context.
# SGLang: INIT_INCREMENTAL_DETOKENIZATION_OFFSET in schedule_batch.py.
PROMPT_TAIL_LEN = 5

# Log decode stats every N decode steps. SGLang: --decode-log-interval (40).
DECODE_LOG_INTERVAL = 40


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

        # The KV cache of this request: [num_tokens, num_layers, 2 (k/v), num_kv_heads, head_dim].
        # Token-major so that pad_sequence can batch many requests in one call.
        # It covers input_ids + output_ids[:-1]: the last sampled token has not
        # been through the model yet, it is the input of the next decode step.
        # M3 replaces this with indices into a preallocated token pool.
        self.kv: Optional[torch.Tensor] = None

    def check_finished(self, eos_ids: Set[int]):
        if self.output_ids[-1] in eos_ids:
            self.finish_reason = "stop"
        elif len(self.output_ids) >= self.max_new_tokens:
            self.finish_reason = "length"


class ScheduleBatch:
    """The requests that run in one forward pass. SGLang: schedule_batch.ScheduleBatch.

    SGLang's version also holds the GPU tensors for the step (input_ids,
    positions, KV slot indices, sampling info). Our ModelRunner builds those
    itself, because without a token pool they have to be rebuilt each step.
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
    ):
        context = zmq.Context(2)
        self.recv_from_tokenizer = get_socket(
            context, zmq.PULL, port_args.scheduler_input, bind=True
        )
        self.send_to_detokenizer = get_socket(
            context, zmq.PUSH, port_args.detokenizer_input, bind=False
        )

        self.model_runner = ModelRunner(model_path)
        self.eos_ids = self.model_runner.eos_ids
        logger.info(
            f"Loaded {model_path} on {self.model_runner.device}, eos_ids={self.eos_ids}"
        )

        # Scheduling limits. Without a memory pool these are the only thing
        # that keeps us from running out of memory. M3 adds a real token budget.
        self.max_running_requests = max_running_requests
        self.max_prefill_tokens = max_prefill_tokens

        self.waiting_queue: Deque[Req] = collections.deque()
        # Requests that finished prefill and are now decoding.
        self.running_batch: List[Req] = []

        # Stats for the log lines.
        self.num_decode_steps = 0
        self.num_decode_tokens = 0
        self.last_decode_log_time = time.perf_counter()

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
                self.waiting_queue.append(Req(recv_req))
            elif isinstance(recv_req, AbortReq):
                self.abort_request(recv_req.rid)
            else:
                raise ValueError(f"Unknown request type: {type(recv_req)}")

    def abort_request(self, rid: str):
        for req in self.waiting_queue:
            if req.rid == rid:
                self.waiting_queue.remove(req)
                # Never started, so the detokenizer has no state for it.
                return
        for req in self.running_batch:
            if req.rid == rid:
                # It still runs in the next decode step, but process_batch_result
                # drops its token and sends a final message, so the detokenizer
                # frees its state and the request leaves the batch.
                req.finish_reason = "abort"
                return

    # ------------------------------------------------------------ scheduling

    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
        """Prefill first: if new requests fit, run them now, and decode next step.

        Prioritizing prefill gives new requests a low time to first token, at the
        cost of a small hiccup for everyone already decoding. This is SGLang's
        default too. (Chunked prefill, --chunked-prefill-size, splits a long
        prompt over several steps so that the hiccup stays small.)
        """
        new_batch = self.get_new_batch_prefill()
        if new_batch is not None:
            return new_batch
        if self.running_batch:
            return ScheduleBatch(self.running_batch, ForwardMode.DECODE)
        return None

    def get_new_batch_prefill(self) -> Optional[ScheduleBatch]:
        """Take requests from the waiting queue while they fit. SGLang: PrefillAdder."""
        can_run: List[Req] = []
        num_tokens = 0
        while self.waiting_queue:
            if len(self.running_batch) + len(can_run) >= self.max_running_requests:
                break
            req = self.waiting_queue[0]
            # Always admit at least one request, even if its prompt alone is
            # over the budget, or it would wait forever.
            if can_run and num_tokens + len(req.input_ids) > self.max_prefill_tokens:
                break
            can_run.append(self.waiting_queue.popleft())
            num_tokens += len(req.input_ids)

        if not can_run:
            return None
        logger.info(
            f"Prefill batch. #new-seq: {len(can_run)}, #new-token: {num_tokens}, "
            f"#running-req: {len(self.running_batch)}, #queue-req: {len(self.waiting_queue)}"
        )
        return ScheduleBatch(can_run, ForwardMode.EXTEND)

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
                req.kv = None  # Free the KV cache right away.

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
            f"#queue-req: {len(self.waiting_queue)}, gen throughput (token/s): {throughput:.1f}"
        )
        self.num_decode_tokens = 0
        self.last_decode_log_time = now


def run_scheduler_process(
    model_path: str,
    port_args: PortArgs,
    max_running_requests: int,
    max_prefill_tokens: int,
    ready_pipe,
):
    setup_logging("scheduler")
    try:
        scheduler = Scheduler(
            model_path, port_args, max_running_requests, max_prefill_tokens
        )
    except Exception:
        ready_pipe.send(traceback.format_exc())
        raise
    ready_pipe.send("ready")
    scheduler.event_loop()
