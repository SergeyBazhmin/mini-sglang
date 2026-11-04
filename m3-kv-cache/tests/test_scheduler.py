"""Check retraction, page accounting and double frees. No model and no GPU needed.

    uv run pytest m3-kv-cache/tests/test_scheduler.py

A fake ModelRunner returns a fixed token but keeps the real slot bookkeeping
(_prepare_extend, _prepare_decode, free_req), so these tests pin down which
requests run and where their slots go. test_batching.py checks that a retracted
request still produces the right tokens.
"""

from typing import List, Optional

import pytest
import scheduler as scheduler_module
from common import EngineOptions, PortArgs, TokenizedGenerateReq
from memory_pool import ReqToTokenPool, TokenAllocator
from model_runner import ModelRunner
from scheduler import ForwardMode, Req, Scheduler

TOKEN = 1  # Not an EOS id, so requests run until max_new_tokens.
PROMPT_TOKEN = 5


class FakeRunner(ModelRunner):
    """The slot logic of ModelRunner, without a model or KV buffers."""

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

    def forward(self, reqs, is_extend: bool) -> List[int]:
        self.last_batch = self._prepare_extend(reqs) if is_extend else self._prepare_decode(reqs)
        return [TOKEN] * len(reqs)


class Sink:
    """Stands in for the socket to the detokenizer."""

    def __init__(self):
        self.messages = []

    def send_pyobj(self, msg):
        self.messages.append(msg)


def make_scheduler(pool: int, context_length: int = 8192, **options) -> Scheduler:
    scheduler_module.ModelRunner = FakeRunner
    s = Scheduler(
        "fake", PortArgs.new(), 8, 8192, pool, 0.6, context_length, EngineOptions(**options)
    )
    s.send_to_detokenizer = Sink()
    return s


def add(s: Scheduler, rid: str, prompt_len: int, max_new_tokens: int) -> Req:
    req = Req(
        TokenizedGenerateReq(rid, [PROMPT_TOKEN] * prompt_len, max_new_tokens, 0.0, 1.0)
    )
    s.handle_generate_request(req)
    return req


def step(s: Scheduler) -> Optional[ForwardMode]:
    batch = s.get_next_batch_to_run()
    if batch is None:
        return None
    s.process_batch_result(batch, s.run_batch(batch))
    return batch.forward_mode


def drain(s: Scheduler, max_steps: int = 10000) -> List[ForwardMode]:
    modes = []
    for _ in range(max_steps):
        mode = step(s)
        if mode is None:
            return modes
        modes.append(mode)
    raise AssertionError("the scheduler did not finish: livelock?")


def sent_to(s: Scheduler, rid: str) -> List[int]:
    """All tokens streamed for one request, in order."""
    tokens: List[int] = []
    for m in s.send_to_detokenizer.messages:
        if rid in m.rids:
            tokens += m.new_token_ids[m.rids.index(rid)]
    return tokens


def final_reason(s: Scheduler, rid: str) -> Optional[str]:
    reasons = [
        m.finish_reasons[m.rids.index(rid)]
        for m in s.send_to_detokenizer.messages
        if rid in m.rids
    ]
    return reasons[-1] if reasons else None


def assert_pool_is_empty(s: Scheduler):
    runner = s.model_runner
    assert runner.allocator.available_size() == runner.max_total_tokens
    assert len(runner.req_to_token_pool.free_slots) == 8


# ---------------------------------------------------------------- reservation


def test_ratio_1_reserves_like_the_chapter():
    s = make_scheduler(pool=64)
    a = add(s, "a", 10, 40)
    assert a.remaining_kv_tokens() == 49
    step(s)
    assert a.remaining_kv_tokens() == 39  # 10 slots held, 49 at the end.


def test_ratio_1_queues_and_never_retracts():
    s = make_scheduler(pool=64)
    add(s, "a", 10, 40)
    add(s, "b", 10, 40)
    step(s)
    assert len(s.running_batch) == 1 and len(s.waiting_queue) == 1
    drain(s)
    assert s.num_retracted == 0
    assert_pool_is_empty(s)


def test_a_lower_ratio_admits_more():
    s = make_scheduler(pool=64, new_token_ratio=0.5)
    add(s, "a", 10, 40)
    add(s, "b", 10, 40)
    step(s)
    assert len(s.running_batch) == 2 and not s.waiting_queue


# ----------------------------------------------------------------- retraction


def test_retraction_when_the_bet_fails():
    s = make_scheduler(pool=64, new_token_ratio=0.5)
    add(s, "a", 10, 40)
    add(s, "b", 10, 40)
    modes = drain(s)
    assert s.num_retracted >= 1
    # One prefill for both requests, then one more for every retraction.
    assert modes.count(ForwardMode.EXTEND) == 1 + s.num_retracted
    for rid in ("a", "b"):
        assert sent_to(s, rid) == [TOKEN] * 40, "no token lost or sent twice"
        assert final_reason(s, rid) == "length"
    assert_pool_is_empty(s)


def test_retraction_picks_the_most_remaining_work():
    s = make_scheduler(pool=40, new_token_ratio=0.1)
    short, long = add(s, "short", 10, 14), add(s, "long", 10, 30)
    while not s.num_retracted:
        step(s)
    assert list(s.waiting_queue) == [long]
    assert s.running_batch == [short]
    assert long.req_pool_idx is None and long.kv_allocated_len == 0
    drain(s)
    assert_pool_is_empty(s)


def test_a_retracted_request_prefills_its_output():
    s = make_scheduler(pool=40, new_token_ratio=0.1)
    add(s, "short", 10, 14)
    long = add(s, "long", 10, 30)
    while not s.num_retracted:
        step(s)
    num_output = len(long.output_ids)
    assert num_output > 0
    while s.running_batch:
        step(s)
    assert step(s) == ForwardMode.EXTEND
    fb = s.model_runner.last_batch
    assert fb.input_ids.tolist() == [PROMPT_TOKEN] * 10 + [TOKEN] * num_output
    assert fb.positions.tolist() == list(range(10 + num_output))
    # Same invariant as after a decode step: the newest token has no slot yet.
    assert long.kv_allocated_len == 10 + len(long.output_ids) - 1


def test_every_mix_finishes_without_leaking():
    for ratio in (0.1, 0.3, 0.5, 0.9):
        for page_size in (1, 4):
            s = make_scheduler(pool=64, new_token_ratio=ratio, page_size=page_size)
            for i in range(6):
                add(s, f"r{i}", 3 + 7 * i % 13, 5 + 11 * i % 30)
            drain(s)
            for i in range(6):
                assert sent_to(s, f"r{i}") == [TOKEN] * (5 + 11 * i % 30)
            assert_pool_is_empty(s)


def test_abort_of_a_retracted_request_tells_the_detokenizer():
    s = make_scheduler(pool=40, new_token_ratio=0.1)
    add(s, "short", 10, 14)
    add(s, "long", 10, 30)
    while not s.num_retracted:
        step(s)
    s.abort_request("long")
    assert not s.waiting_queue
    assert final_reason(s, "long") == "abort"
    drain(s)
    assert_pool_is_empty(s)


def test_an_aborted_request_is_not_requeued_by_retraction():
    s = make_scheduler(pool=40, new_token_ratio=0.1)
    add(s, "short", 10, 14)
    add(s, "long", 10, 30)
    step(s)
    s.abort_request("long")  # Marked only: it stays in the batch until its next step.
    s.model_runner.allocator.alloc(s.model_runner.allocator.available_size())  # Pool full.
    s.update_running_batch()
    assert [r.rid for r in s.running_batch] == ["short"]
    assert not s.waiting_queue
    assert s.num_retracted == 0
    assert final_reason(s, "long") == "abort"


# ---------------------------------------------------------------------- pages


def test_reservations_are_whole_pages():
    s = make_scheduler(pool=16, page_size=4)
    a = add(s, "a", 5, 8)
    assert s._reserved_tokens(a) == 12  # 5 + 7 tokens fill 3 pages exactly.
    step(s)
    assert s.model_runner.allocator.available_size() == 8  # 5 tokens, 2 pages.
    assert s._reserved_tokens(a) == 4


def test_decode_takes_a_page_only_at_a_boundary():
    s = make_scheduler(pool=16, page_size=4)
    a = add(s, "a", 2, 8)
    runner = s.model_runner
    step(s)
    available = [runner.allocator.available_size()]
    needed = [runner.decode_slots_needed(s.running_batch)]
    while s.running_batch:
        step(s)
        available.append(runner.allocator.available_size())
        needed.append(runner.decode_slots_needed(s.running_batch))
    # Positions 2 and 3 fill page 0. Position 4 takes a page, then 5, 6, 7 fill it,
    # and position 8 takes the last one.
    assert available == [12, 12, 12, 8, 8, 8, 8, 16], available
    assert needed[:-1] == [0, 0, 4, 0, 0, 0, 4], needed
    assert a.finish_reason == "length"


def test_a_request_gets_consecutive_slots_on_each_page():
    s = make_scheduler(pool=16, page_size=4)
    a = add(s, "a", 2, 12)
    while len(a.output_ids) < 11:
        step(s)
    row = s.model_runner.req_to_token_pool.req_to_token[a.req_pool_idx].tolist()
    slots = row[: a.kv_allocated_len]
    assert a.kv_allocated_len == 12
    for pos in range(12):
        if pos % 4 == 0:
            assert (slots[pos] - 1) % 4 == 0, slots  # A page's first slot.
        else:
            assert slots[pos] == slots[pos - 1] + 1, slots


# ---------------------------------------------------------------- double free


def test_free_req_twice_raises():
    s = make_scheduler(pool=16)
    a = add(s, "a", 4, 8)
    step(s)
    available = s.model_runner.allocator.available_size()
    s.model_runner.free_req(a)
    with pytest.raises(RuntimeError, match="double free"):
        s.model_runner.free_req(a)
    assert s.model_runner.allocator.available_size() == available + 4
