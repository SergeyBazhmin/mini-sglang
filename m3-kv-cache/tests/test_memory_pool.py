"""Unit checks for the allocator and the table. No model, no GPU, takes a second.

    uv run pytest m3-kv-cache/tests/test_memory_pool.py
"""


import pytest
import torch

from memory_pool import ReqToTokenPool, TokenAllocator, TokenToKVPool


def test_alloc_and_free():
    allocator = TokenAllocator(size=8, device="cpu")
    first = allocator.alloc(5)
    assert first.tolist() == [1, 2, 3, 4, 5]
    assert allocator.available_size() == 3

    allocator.free(first[[1, 3]])  # Slots 2 and 4: holes in the middle.
    assert allocator.available_size() == 5


def test_lazy_free_keeps_the_slots_after_their_row_is_reused():
    allocator = TokenAllocator(size=8, device="cpu", lazy_free=True)
    table = torch.zeros(1, 8, dtype=torch.int64)
    table[0, :4] = allocator.alloc(4)
    allocator.free(table[0, :4])  # A view of the table, as in free_req.
    table[0, :4] = allocator.alloc(4)
    allocator.merge_released()
    assert sorted(allocator.free_pages.tolist()) == [1, 2, 3, 4]


def test_alloc_failure_takes_nothing():
    allocator = TokenAllocator(size=4, device="cpu")
    assert allocator.alloc(5) is None
    assert allocator.available_size() == 4
    assert len(allocator.alloc(4)) == 4
    assert allocator.alloc(1) is None


def test_slot_zero_is_never_handed_out():
    allocator = TokenAllocator(size=4, device="cpu")
    for _ in range(10):  # Enough rounds to go through every recycled slot.
        slots = allocator.alloc(4)
        assert 0 not in slots.tolist()
        allocator.free(slots)


def test_slots_are_unique_after_recycling():
    allocator = TokenAllocator(size=6, device="cpu")
    a = allocator.alloc(3)
    b = allocator.alloc(3)
    allocator.free(a)
    c = allocator.alloc(3)
    assert sorted(b.tolist() + c.tolist()) == [1, 2, 3, 4, 5, 6]


def test_many_requests_keep_the_free_list_whole():
    allocator = TokenAllocator(size=100, device="cpu")
    for round_id in range(1000):
        held = [allocator.alloc(n) for n in (7, 1 + round_id % 30, 13)]
        for slots in reversed(held):
            allocator.free(slots)
    assert allocator.available_size() == 100
    assert sorted(allocator.free_pages.tolist()) == list(range(1, 101))


def test_table_rows():
    pool = ReqToTokenPool(size=2, max_context_len=4, device="cpu")
    a, b = pool.alloc(), pool.alloc()
    assert {a, b} == {0, 1}
    assert pool.alloc() is None
    pool.free(a)
    assert pool.alloc() == a


def test_kv_buffers_have_a_row_for_every_slot():
    pool = TokenToKVPool(
        size=8, num_layers=3, num_kv_heads=2, head_dim=4, dtype=torch.float16, device="cpu"
    )
    assert len(pool.k_buffer) == len(pool.v_buffer) == 3
    assert pool.k_buffer[0].shape == (9, 2, 4)  # 8 slots + slot 0.
    # 3 layers, K + V, 2 heads, 4 dims, 2 bytes.
    assert TokenToKVPool.cell_size(3, 2, 4, torch.float16) == 3 * 2 * 2 * 4 * 2


def test_pages_are_consecutive_slots():
    allocator = TokenAllocator(size=12, device="cpu", page_size=4)
    assert allocator.available_size() == 12
    assert allocator.alloc(5).tolist() == [1, 2, 3, 4, 5, 6, 7, 8]  # Rounded up.
    assert allocator.available_size() == 4
    assert allocator.alloc(5) is None
    assert allocator.alloc(4).tolist() == [9, 10, 11, 12]


def test_freeing_a_table_row_returns_whole_pages():
    allocator = TokenAllocator(size=12, device="cpu", page_size=4)
    slots = allocator.alloc(8)
    allocator.free(slots[:5])  # 5 tokens used, on two pages.
    assert allocator.available_size() == 12
    assert sorted(allocator.free_pages.tolist()) == [1, 5, 9]


def test_pool_size_must_be_whole_pages():
    with pytest.raises(ValueError):
        TokenAllocator(size=10, device="cpu", page_size=4)


def test_lazy_free_counts_released_slots_and_merges_when_short():
    allocator = TokenAllocator(size=4, device="cpu", lazy_free=True)
    first = allocator.alloc(3)
    allocator.free(first)
    assert allocator.available_size() == 4
    assert len(allocator.free_pages) == 1  # Not merged yet.
    assert len(allocator.alloc(1)) == 1  # Still enough without merging.
    assert len(allocator.free_pages) == 0
    assert sorted(allocator.alloc(3).tolist()) == first.tolist()  # Merged.
    assert allocator.available_size() == 0


def test_lazy_free_keeps_the_free_list_whole():
    allocator = TokenAllocator(size=100, device="cpu", lazy_free=True)
    for round_id in range(1000):
        held = [allocator.alloc(n) for n in (7, 1 + round_id % 30, 13)]
        for slots in reversed(held):
            allocator.free(slots)
    assert allocator.available_size() == 100
    allocator.merge_released()
    assert sorted(allocator.free_pages.tolist()) == list(range(1, 101))


def expect_double_free(allocator: TokenAllocator, slots: torch.Tensor):
    with pytest.raises(RuntimeError, match="Double free"):
        allocator.free(slots)


def test_double_free_is_caught_at_the_first_repeat():
    for lazy_free in (False, True):
        allocator = TokenAllocator(
            size=8, device="cpu", lazy_free=lazy_free, check_double_free=True
        )
        slots = allocator.alloc(3)
        allocator.free(slots)
        expect_double_free(allocator, slots)


def test_double_free_within_one_call_and_of_slot_zero():
    allocator = TokenAllocator(size=8, device="cpu", check_double_free=True)
    slots = allocator.alloc(3)
    expect_double_free(allocator, slots[[0, 0]])
    expect_double_free(allocator, torch.tensor([0]))


def test_reallocated_slots_can_be_freed_again():
    allocator = TokenAllocator(size=2, device="cpu", check_double_free=True)
    for _ in range(4):  # The check must clear on alloc, or this would raise.
        allocator.free(allocator.alloc(2))


def test_double_free_with_pages():
    allocator = TokenAllocator(
        size=8, device="cpu", page_size=4, check_double_free=True
    )
    slots = allocator.alloc(8)
    allocator.free(slots)
    expect_double_free(allocator, slots)


def main():
    tests =[v for k, v in globals().items() if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)} tests passed.")
