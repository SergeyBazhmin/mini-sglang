"""Check retraction, pages, double frees and the prefix cache. No model and no GPU.

    uv run pytest m4-radix-cache/tests/test_scheduler.py

A fake ModelRunner returns a fixed token but keeps the real slot bookkeeping
(_prepare_extend, _prepare_decode, free_req), so these tests pin down which
requests run and where their slots go. It also checks every K/V read, see
fake_runner.py. test_batching.py checks that real tokens stay the same.

The M3 tests run with the cache off, because they count free slots exactly. The
ones that aren't about counting run both ways.
"""

import random
from typing import List, Optional, Sequence, Union

import pytest
import zmq

import scheduler as scheduler_module
from common import EngineOptions, FlushCacheReq, PortArgs, TokenizedGenerateReq
from fake_runner import TOKEN, FakeRunner
from scheduler import ForwardMode, Req, Scheduler

PROMPT_TOKEN = 5


class Sink:
    """Stands in for the socket to the detokenizer."""

    def __init__(self):
        self.messages = []

    def send_pyobj(self, msg):
        self.messages.append(msg)


def make_scheduler(pool: int, context_length: int = 8192, **options) -> Scheduler:
    """The cache is off unless a test asks for it: pass disable_radix_cache=False."""
    options.setdefault("disable_radix_cache", True)
    scheduler_module.ModelRunner = FakeRunner
    s = Scheduler(
        "fake", PortArgs.new(), 8, 8192, pool, 0.6, context_length, EngineOptions(**options)
    )
    s.send_to_detokenizer = Sink()
    return s


def add(s: Scheduler, rid: str, prompt: Union[int, Sequence[int]], max_new_tokens: int) -> Req:
    """prompt is a length, or the tokens themselves."""
    ids = [PROMPT_TOKEN] * prompt if isinstance(prompt, int) else list(prompt)
    req = Req(TokenizedGenerateReq(rid, ids, max_new_tokens, 0.0, 1.0))
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
    """Nothing is running, so every slot is free or idle in the tree."""
    runner = s.model_runner
    tree = s.tree_cache
    assert tree.protected_size == 0, "a finished request still holds a lock"
    assert runner.allocator.available_size() + tree.total_size() == runner.max_total_tokens
    assert len(runner.req_to_token_pool.free_slots) == 8
    tree.evict(runner.max_total_tokens)
    assert tree.total_size() == 0
    assert runner.allocator.available_size() == runner.max_total_tokens


def check_slot_owners(s: Scheduler):
    """Every slot has exactly one owner, see radix_cache.py.

    With pages, whole pages are counted, because that's what the allocator owns.
    """
    runner = s.model_runner
    tree = s.tree_cache
    page_size = runner.allocator.page_size
    table = runner.req_to_token_pool.req_to_token

    private = []
    for r in s.running_batch:
        row = table[r.req_pool_idx, : r.kv_allocated_len].tolist()
        private += row[r.cache_protected_len :]
        # What the request reads through the tree is what the tree says.
        assert row[: r.cache_protected_len] == tree.path_slots(r.last_node).tolist()[
            : r.cache_protected_len
        ]
    in_tree = []
    stack = list(tree.root.children.values())
    while stack:
        node = stack.pop()
        in_tree += node.value.tolist()
        stack.extend(node.children.values())

    def pages(slots):
        return {(slot - 1) // page_size for slot in slots}

    assert len(private) == len(set(private)), "a slot is held by two requests"
    assert len(in_tree) == len(set(in_tree)), "a slot is in two nodes"
    assert not pages(private) & pages(in_tree), "a slot is held and cached"
    held = len(pages(private) | pages(in_tree)) * page_size
    assert held + runner.allocator.available_size() == runner.max_total_tokens


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
            for disable in (True, False):
                s = make_scheduler(
                    pool=64,
                    new_token_ratio=ratio,
                    page_size=page_size,
                    disable_radix_cache=disable,
                )
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


# --------------------------------------------------------------- prefix cache


def cached(pool: int = 256, **options) -> Scheduler:
    return make_scheduler(pool, disable_radix_cache=False, **options)


def prompt(start: int, n: int) -> List[int]:
    return list(range(start, start + n))


SHARED = prompt(100, 20)


def test_a_finished_request_is_cached():
    s = cached()
    add(s, "a", SHARED, 3)
    drain(s)
    # Prompt + 2 outputs went through the model. The third was never an input.
    assert s.tree_cache.total_size() == 22
    assert_pool_is_empty(s)


def test_the_same_prompt_skips_all_but_one_token():
    s = cached()
    add(s, "a", SHARED, 3)
    drain(s)
    b = add(s, "b", SHARED, 3)
    assert step(s) == ForwardMode.EXTEND
    assert b.prefix_len == 19, "the last token must run, to give logits"
    fb = s.model_runner.last_batch
    assert fb.input_ids.tolist() == [SHARED[-1]]
    assert fb.positions.tolist() == [19]
    assert fb.seq_lens.tolist() == [20] and fb.extend_lens.tolist() == [1]
    drain(s)
    assert sent_to(s, "b") == [TOKEN] * 3
    assert_pool_is_empty(s)


def test_a_shared_prefix_is_computed_once():
    s = cached()
    add(s, "a", SHARED + prompt(200, 5), 2)
    drain(s)
    b = add(s, "b", SHARED + prompt(300, 5), 2)
    step(s)
    assert b.prefix_len == 20
    assert s.model_runner.last_batch.input_ids.tolist() == prompt(300, 5)
    drain(s)
    assert_pool_is_empty(s)


def test_a_prompt_is_cached_while_its_request_runs():
    s = cached()
    add(s, "a", SHARED + [1], 100)
    step(s)
    b = add(s, "b", SHARED + [2], 2)
    step(s)
    assert b.prefix_len == 20
    check_slot_owners(s)
    drain(s)
    assert_pool_is_empty(s)


def test_the_answer_is_cached_too():
    s = cached()
    add(s, "a", SHARED, 5)
    drain(s)
    b = add(s, "b", SHARED + [TOKEN] * 4 + [9], 2)
    step(s)
    assert b.prefix_len == 24
    drain(s)
    assert_pool_is_empty(s)


def test_requests_that_race_on_a_prefix_end_up_sharing_its_slots():
    s = cached()
    a = add(s, "a", SHARED + [1, 2], 10)
    b = add(s, "b", SHARED + [3, 4], 10)
    step(s)  # One batch: nothing is cached yet, so both compute the prefix.
    assert a.prefix_len == b.prefix_len == 0
    table = s.model_runner.req_to_token_pool.req_to_token
    assert table[a.req_pool_idx, :20].tolist() == table[b.req_pool_idx, :20].tolist()
    # Prefix once, then each tail, and b's copy of the prefix went back.
    assert s.model_runner.allocator.available_size() == 256 - 24
    check_slot_owners(s)
    drain(s)
    assert_pool_is_empty(s)


def test_a_new_request_does_not_steal_the_tail_of_a_page():
    s = cached(page_size=4)
    add(s, "a", prompt(100, 10), 20)
    step(s)
    check_slot_owners(s)
    assert s.tree_cache.total_size() == 8, "the last partial page stays with the request"
    b = add(s, "b", prompt(100, 10), 2)
    step(s)
    assert b.prefix_len == 8
    check_slot_owners(s)
    drain(s)
    assert_pool_is_empty(s)


def test_a_locked_prefix_survives_eviction():
    s = cached(pool=40)
    add(s, "a", prompt(100, 10), 5)
    drain(s)
    assert s.tree_cache.total_size() == 14  # Prompt + 4 outputs.
    b = add(s, "b", prompt(100, 10), 20)
    step(s)
    assert b.prefix_len == 9
    assert s.tree_cache.evict(100) == 4, "only a's output: b locks the prompt"
    assert s.tree_cache.total_size() == 10
    check_slot_owners(s)
    drain(s)
    assert sent_to(s, "b") == [TOKEN] * 20
    assert_pool_is_empty(s)


def test_a_full_pool_of_idle_cache_blocks_nobody():
    s = cached(pool=32)
    for i in range(10):
        add(s, f"r{i}", prompt(1000 * (i + 1), 10), 8)
        drain(s)
        assert sent_to(s, f"r{i}") == [TOKEN] * 8
        assert s.tree_cache.total_size() <= 32
    assert s.num_retracted == 0
    assert_pool_is_empty(s)


def test_idle_cache_goes_before_running_requests():
    s = cached(pool=32, new_token_ratio=0.1)
    add(s, "old", prompt(5000, 16), 2)
    drain(s)
    assert s.tree_cache.total_size() == 17
    add(s, "a", prompt(100, 10), 12)
    add(s, "b", prompt(200, 10), 12)
    drain(s)
    assert len(s.tree_cache.match_prefix(prompt(5000, 16))[0]) == 0, "it had to go"
    assert_pool_is_empty(s)


def test_tokens_usage_ignores_idle_cache():
    s = cached(pool=32)
    add(s, "a", prompt(100, 16), 2)
    drain(s)
    assert s._token_usage() == 0.0
    assert s._cache_usage() > 0.5


def test_ratio_1_never_retracts_with_the_cache_on():
    for page_size in (1, 4):
        s = cached(pool=64, page_size=page_size)
        for i in range(12):
            add(s, f"r{i}", SHARED + prompt(1000 * i, 6), 20)
        drain(s)
        assert s.num_retracted == 0
        assert_pool_is_empty(s)


def test_a_lock_is_charged_at_admission():
    # All 16 slots are cache, and the next prompt reuses 15 of them. If the lock
    # didn't count, "b" would see 16 free slots, reserve its 14 new ones from the
    # same 16, and hit the assertion in _prepare_extend.
    s = cached(pool=16)
    add(s, "a", prompt(100, 15), 2)
    drain(s)
    add(s, "b", prompt(100, 15) + [1], 2)
    add(s, "c", prompt(500, 8), 8)
    drain(s)
    assert_pool_is_empty(s)


@pytest.mark.parametrize("seed", range(60))
def test_stress_with_overlapping_prompts(seed):
    """Requests share prefixes and race, over a small pool. Every K/V read is checked."""
    rng = random.Random(seed)
    page_size = rng.choice((1, 1, 2, 4))
    s = cached(
        pool=128,
        page_size=page_size,
        new_token_ratio=rng.choice((1.0, 0.5, 0.2)),
        eviction_policy=rng.choice(("lru", "lfu")),
        schedule_policy=rng.choice(("fcfs", "lpm")),
        in_batch_dedup=rng.random() < 0.5,
        cache_retracted=rng.random() < 0.5,
        lazy_free=rng.random() < 0.5,
        fast_match=rng.random() < 0.5,
    )
    lengths = {}
    for i in range(40):
        if rng.random() < 0.5 or not s.waiting_queue:
            base = rng.choice((1000, 2000, 3000))
            ids = prompt(base, rng.randint(1, 40)) + prompt(rng.randint(1, 3), rng.randint(1, 8))
            rid = f"r{i}"
            lengths[rid] = rng.randint(1, 24)
            add(s, rid, ids, lengths[rid])
        if rng.random() < 0.1 and s.running_batch:
            victim = rng.choice(s.running_batch)
            s.abort_request(victim.rid)
            lengths.pop(victim.rid, None)
        for _ in range(rng.randint(0, 6)):
            step(s)
            check_slot_owners(s)
    drain(s)
    for rid, n in lengths.items():
        assert sent_to(s, rid) == [TOKEN] * n, (seed, rid)
        assert final_reason(s, rid) == "length"
    assert_pool_is_empty(s)


# ------------------------------------------------------------------ cache policies


def test_lpm_runs_the_best_match_first():
    s = cached(schedule_policy="lpm")
    add(s, "seed", SHARED, 2)
    drain(s)
    s.max_running_requests = 1
    add(s, "miss", prompt(5000, 10), 2)
    hit = add(s, "hit", SHARED + [1], 2)
    step(s)
    assert s.running_batch == [hit], "the later arrival goes first"
    s.max_running_requests = 8
    drain(s)
    assert_pool_is_empty(s)


def test_lpm_keeps_arrival_order_on_ties():
    s = cached(schedule_policy="lpm")
    s.max_running_requests = 1
    first = add(s, "first", prompt(1000, 10), 2)
    add(s, "second", prompt(2000, 10), 2)
    step(s)
    assert s.running_batch == [first]
    s.max_running_requests = 8
    drain(s)


def test_in_batch_dedup_runs_the_prefix_once():
    prompts = [SHARED * 2 + [i] for i in range(4)]
    computed = {}
    for dedup in (False, True):
        s = cached(in_batch_dedup=dedup)
        for i, ids in enumerate(prompts):
            add(s, f"r{i}", ids, 2)
        drain(s)
        computed[dedup] = s.num_prefill_tokens
        assert_pool_is_empty(s)
    assert computed[False] == 41 * 4
    assert computed[True] == 41 + 3, computed


def test_in_batch_dedup_ignores_short_prefixes():
    s = cached(in_batch_dedup=True)
    a = add(s, "a", prompt(100, 8) + [1], 2)
    b = add(s, "b", prompt(100, 8) + [2], 2)
    assert step(s) == ForwardMode.EXTEND
    assert not s.waiting_queue and a in s.running_batch and b in s.running_batch


def test_retraction_can_keep_the_work():
    computed = {}
    for keep in (False, True):
        s = cached(pool=40, new_token_ratio=0.1, cache_retracted=keep)
        add(s, "short", prompt(100, 10), 14)
        add(s, "long", prompt(200, 10), 30)
        drain(s)
        assert s.num_retracted >= 1
        assert sent_to(s, "long") == [TOKEN] * 30
        computed[keep] = s.num_prefill_tokens
        assert_pool_is_empty(s)
    assert computed[True] < computed[False], computed


def test_flush_cache_when_idle():
    s = cached()
    add(s, "a", SHARED, 2)
    drain(s)
    assert s.flush_cache()
    assert s.tree_cache.total_size() == 0
    assert s.model_runner.allocator.available_size() == 256
    assert s.num_cached_tokens == 0


def test_flush_cache_refuses_while_busy():
    s = cached()
    add(s, "a", SHARED, 20)
    step(s)
    assert not s.flush_cache()
    assert s.tree_cache.total_size() > 0
    drain(s)
    assert_pool_is_empty(s)


def test_flush_message_reaches_the_scheduler():
    s = cached()
    add(s, "a", SHARED, 2)
    drain(s)

    class Socket:
        def __init__(self):
            self.messages = [FlushCacheReq()]

        def recv_pyobj(self, *args):
            if not self.messages:
                raise zmq.ZMQError()
            return self.messages.pop()

    s.recv_from_tokenizer = Socket()
    s.ingest_requests(block=False)
    assert s.tree_cache.total_size() == 0
