"""Check decode-first and stream_interval. No model and no GPU needed.

    uv run pytest m2-continuous-batching/tests/test_scheduler.py

A fake ModelRunner returns a fixed token, so these tests only pin down which
batch runs at each step and which tokens get streamed.
"""

from typing import List, Optional

import scheduler as scheduler_module
from common import SchedulerOptions, PortArgs, TokenizedGenerateReq
from scheduler import ForwardMode, Req, Scheduler

TOKEN = 1  # Not an EOS id, so requests run until max_new_tokens.


class FakeRunner:
    device = "cpu"
    eos_ids = {0}

    def __init__(self, model_path: str):
        pass

    def forward(self, reqs, is_extend: bool) -> List[int]:
        return [TOKEN] * len(reqs)


class Sink:
    """Stands in for the socket to the detokenizer."""

    def __init__(self):
        self.messages = []

    def send_pyobj(self, msg):
        self.messages.append(msg)


def make_scheduler(**options) -> Scheduler:
    scheduler_module.ModelRunner = FakeRunner
    s = Scheduler("fake", PortArgs.new(), 64, 8192, SchedulerOptions(**options))
    s.send_to_detokenizer = Sink()
    return s


def add(s: Scheduler, rid: str, prompt_len: int, max_new_tokens: int = 100):
    s.waiting_queue.append(
        Req(TokenizedGenerateReq(rid, [5] * prompt_len, max_new_tokens, 0.0, 1.0))
    )


def step(s: Scheduler) -> Optional[ForwardMode]:
    batch = s.get_next_batch_to_run()
    if batch is None:
        return None
    s.process_batch_result(batch, s.run_batch(batch))
    return batch.forward_mode


def sent_to(s: Scheduler, rid: str) -> List[List[int]]:
    """The token lists streamed for one request, message by message."""
    return [
        m.new_token_ids[m.rids.index(rid)]
        for m in s.send_to_detokenizer.messages
        if rid in m.rids
    ]


def test_prefill_first_by_default():
    s = make_scheduler()
    add(s, "a", 4)
    assert step(s) == ForwardMode.EXTEND
    add(s, "b", 4)
    assert step(s) == ForwardMode.EXTEND
    assert len(s.running_batch) == 2


def test_decode_first_waits_for_the_batch_to_drain():
    s = make_scheduler(decode_first=True)
    add(s, "a", 4, max_new_tokens=3)
    assert step(s) == ForwardMode.EXTEND
    add(s, "b", 4)
    assert [step(s), step(s)] == [ForwardMode.DECODE, ForwardMode.DECODE]
    assert not s.running_batch
    assert step(s) == ForwardMode.EXTEND


def test_stream_interval():
    s = make_scheduler(stream_interval=4)
    add(s, "a", 4, max_new_tokens=10)
    while step(s) is not None:
        pass
    # First token at once, then every 4, and the finish flushes the rest.
    assert [len(ids) for ids in sent_to(s, "a")] == [1, 4, 4, 1]
