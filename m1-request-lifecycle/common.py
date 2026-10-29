"""Messages and IPC helpers shared by the three processes.

SGLang equivalents:
  - message types: python/sglang/srt/managers/io_struct.py
    (TokenizedGenerateReqInput, AbortReq, BatchTokenIDOutput, BatchStrOutput)
  - socket setup:  sglang.srt.utils.get_zmq_socket, PortArgs in server_args.py
"""

import dataclasses
import sys
import tempfile
from typing import List, Optional

import zmq
from loguru import logger


def setup_logging(proc_name: str):
    """Call once at the start of each process. Tags every line with the process name."""
    logger.remove()
    logger.configure(extra={"proc": proc_name})
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <7}</level> | "
            "<cyan>{extra[proc]: <11}</cyan> | <level>{message}</level>"
        ),
    )

# ---------------------------------------------------------------------------
# IPC addresses
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PortArgs:
    """One ZMQ address per *receiving* process. The receiver binds, senders connect."""

    scheduler_input: str  # TokenizerManager -> Scheduler
    detokenizer_input: str  # Scheduler -> Detokenizer
    tokenizer_input: str  # Detokenizer -> TokenizerManager

    @staticmethod
    def new() -> "PortArgs":
        d = tempfile.mkdtemp(prefix="nano-serve-")
        return PortArgs(
            scheduler_input=f"ipc://{d}/scheduler",
            detokenizer_input=f"ipc://{d}/detokenizer",
            tokenizer_input=f"ipc://{d}/tokenizer",
        )


def get_socket(context: zmq.Context, socket_type: int, addr: str, bind: bool):
    sock = context.socket(socket_type)
    # Never drop messages because a queue is "full".
    sock.setsockopt(zmq.SNDHWM, 0)
    sock.setsockopt(zmq.RCVHWM, 0)
    if bind:
        sock.bind(addr)
    else:
        sock.connect(addr)
    return sock


# ---------------------------------------------------------------------------
# Messages. They are pickled with send_pyobj / recv_pyobj.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TokenizedGenerateReq:
    """TokenizerManager -> Scheduler: a new request."""

    rid: str
    input_ids: List[int]
    max_new_tokens: int
    temperature: float
    top_p: float


@dataclasses.dataclass
class AbortReq:
    """TokenizerManager -> Scheduler: the client went away, stop generating."""

    rid: str


@dataclasses.dataclass
class BatchTokenIDOut:
    """Scheduler -> Detokenizer: new token ids for a batch of requests.

    Batch-shaped even though v0 only ever sends one request at a time,
    so the format does not change when we add continuous batching in M2.
    """

    rids: List[str]
    new_token_ids: List[List[int]]
    # Last few prompt tokens, sent only with a request's first output. They
    # give the detokenizer context to decode the first token correctly.
    prompt_tails: List[Optional[List[int]]]
    finish_reasons: List[Optional[str]]  # None while running, else "stop" | "length" | "abort"
    completion_tokens: List[int]


@dataclasses.dataclass
class BatchStrOut:
    """Detokenizer -> TokenizerManager: text deltas for a batch of requests."""

    rids: List[str]
    text_deltas: List[str]
    finish_reasons: List[Optional[str]]
    completion_tokens: List[int]
