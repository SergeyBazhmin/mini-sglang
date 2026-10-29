"""Scheduler process: owns the model and the GPU, and runs the generation loop.

SGLang equivalent: python/sglang/srt/managers/scheduler.py
  - Scheduler.event_loop_normal     (~line 2036)
  - Scheduler.ingest_requests       (~line 2194)
  - Scheduler.handle_generate_request (~line 3019)
  - OutputStreamer.stream_output    (scheduler_components/output_streamer.py)

v0 runs ONE request at a time. The loop is already shaped like SGLang's
(ingest -> pick work -> run one step -> stream output), so M2 (continuous
batching) mostly changes "pick work" and "run one step".
"""

import collections
import traceback
from typing import Deque, List, Optional, Set

import torch
import zmq
from loguru import logger
from transformers import AutoModelForCausalLM

from common import (
    AbortReq,
    BatchTokenIDOut,
    PortArgs,
    TokenizedGenerateReq,
    get_socket,
    setup_logging,
)

# How many prompt tokens to send to the detokenizer as decoding context.
# SGLang: INIT_INCREMENTAL_DETOKENIZATION_OFFSET in schedule_batch.py.
PROMPT_TAIL_LEN = 5


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

        # The KV cache. v0 uses HF's own cache object, one per request.
        # M3 replaces this with our own preallocated token pool.
        self.past_key_values = None

    def check_finished(self, eos_ids: Set[int]):
        if self.output_ids[-1] in eos_ids:
            self.finish_reason = "stop"
        elif len(self.output_ids) >= self.max_new_tokens:
            self.finish_reason = "length"


def sample(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    """Sample one token id from a 1-D logits vector."""
    if temperature <= 0:
        return int(torch.argmax(logits))
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = probs.sort(descending=True)
        # Drop a token if the tokens before it already cover top_p.
        sorted_probs[sorted_probs.cumsum(-1) - sorted_probs > top_p] = 0
        probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)
    return int(torch.multinomial(probs, num_samples=1))


class Scheduler:
    def __init__(self, model_path: str, port_args: PortArgs):
        context = zmq.Context(2)
        self.recv_from_tokenizer = get_socket(
            context, zmq.PULL, port_args.scheduler_input, bind=True
        )
        self.send_to_detokenizer = get_socket(
            context, zmq.PUSH, port_args.detokenizer_input, bind=False
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
            .to(self.device)
            .eval()
        )
        self.eos_ids = self._get_eos_ids()
        logger.info(f"Loaded {model_path} on {self.device}, eos_ids={self.eos_ids}")

        self.waiting_queue: Deque[Req] = collections.deque()
        self.running_req: Optional[Req] = None

    def _get_eos_ids(self) -> Set[int]:
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.model.config.eos_token_id
        if isinstance(eos, int):
            eos = [eos]
        return set(eos or [])

    # ----------------------------------------------------------------- loop

    def event_loop(self):
        while True:
            # When there is nothing to do, block on the socket instead of spinning.
            idle = self.running_req is None and not self.waiting_queue
            self.ingest_requests(block=idle)

            # Pick work. v0: finish the current request before starting the next.
            # M2: this becomes "build a batch from running + waiting requests".
            if self.running_req is None and self.waiting_queue:
                self.running_req = self.waiting_queue.popleft()

            req = self.running_req
            if req is None:
                continue
            if req.finish_reason is None:
                self.run_step(req)
            self.stream_output(req)
            if req.finish_reason is not None:
                self.running_req = None

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
        if self.running_req is not None and self.running_req.rid == rid:
            # Stop it on the next loop iteration. stream_output still sends a
            # final message so the detokenizer frees its state.
            self.running_req.finish_reason = "abort"

    # -------------------------------------------------------------- forward

    @torch.inference_mode()
    def run_step(self, req: Req):
        """One forward pass. The first one is prefill, the rest are decode."""
        if req.past_key_values is None:
            # Prefill: run the whole prompt and fill the KV cache.
            input_ids = torch.tensor([req.input_ids], device=self.device)
        else:
            # Decode: run only the last generated token and reuse the KV cache.
            input_ids = torch.tensor([[req.output_ids[-1]]], device=self.device)

        out = self.model(
            input_ids=input_ids, past_key_values=req.past_key_values, use_cache=True
        )
        req.past_key_values = out.past_key_values

        next_id = sample(out.logits[0, -1], req.temperature, req.top_p)
        req.output_ids.append(next_id)
        req.check_finished(self.eos_ids)

        if req.finish_reason is not None:
            req.past_key_values = None  # Free the KV cache right away.

    # --------------------------------------------------------------- output

    def stream_output(self, req: Req):
        """Send new token ids to the detokenizer. Sent after every token in v0.

        SGLang batches outputs for all running requests into one message and
        can send only every `stream_interval` tokens.
        """
        new_ids = req.output_ids[req.num_sent :]
        if not new_ids and req.finish_reason is None:
            return
        is_first = req.num_sent == 0
        req.num_sent = len(req.output_ids)

        self.send_to_detokenizer.send_pyobj(
            BatchTokenIDOut(
                rids=[req.rid],
                new_token_ids=[new_ids],
                prompt_tails=[req.input_ids[-PROMPT_TAIL_LEN:] if is_first else None],
                finish_reasons=[req.finish_reason],
                completion_tokens=[len(req.output_ids)],
            )
        )


def run_scheduler_process(model_path: str, port_args: PortArgs, ready_pipe):
    setup_logging("scheduler")
    try:
        scheduler = Scheduler(model_path, port_args)
    except Exception:
        ready_pipe.send(traceback.format_exc())
        raise
    ready_pipe.send("ready")
    scheduler.event_loop()
