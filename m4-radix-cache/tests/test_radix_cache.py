"""The tree on its own: match, insert, split, evict and locks. No model, no GPU.

    uv run pytest m4-radix-cache/tests/test_radix_cache.py

SGLang's equivalent: test/registered/unit/mem_cache/test_radix_cache_unit.py.

A slot here is just an integer that the test picks: the tree never looks inside.
Each sequence gets slots starting from a new base, so a test can tell whose they are.
"""

import random
from array import array
from typing import List

import pytest
import torch

from memory_pool import TokenAllocator
from radix_cache import RadixCache, match_len_gallop, match_len_loop


def slots_from(start: int, n: int) -> torch.Tensor:
    return torch.arange(start, start + n)


def make_tree(pool: int = 1000, page_size: int = 1, **kwargs) -> RadixCache:
    allocator = TokenAllocator(pool, "cpu", page_size=page_size, check_double_free=True)
    return RadixCache(allocator, None, page_size=page_size, **kwargs)


def insert(tree: RadixCache, ids: List[int]):
    """Insert with slots from the allocator, and free what the tree didn't take."""
    slots = tree.allocator.alloc(len(ids))
    assert slots is not None
    matched, node = tree.insert(ids, slots)
    tree.allocator.free(slots[:matched])
    return matched, node


def check_accounting(tree: RadixCache):
    """Walk the tree and compare with its running totals."""
    total = protected = 0
    leaves = set()
    stack = list(tree.root.children.values())
    while stack:
        node = stack.pop()
        assert len(node.key) == len(node.value), "a key and its slots differ in length"
        assert len(node.key) > 0
        assert node.parent.children[tree._child_key(node.key, 0)] is node
        assert node.lock_ref <= node.parent.lock_ref or node.parent is tree.root
        total += len(node.key)
        protected += len(node.key) if node.lock_ref else 0
        if not node.children and node.lock_ref == 0:
            leaves.add(node)
        stack.extend(node.children.values())
    assert tree.total_size() == total
    assert tree.protected_size == protected
    assert tree.evictable_leaves == leaves


# -------------------------------------------------------------------- match


def test_empty_tree_matches_nothing():
    slots, node = make_tree().match_prefix([1, 2, 3])
    assert len(slots) == 0


def test_exact_match():
    tree = make_tree()
    tree.insert([1, 2, 3], slots_from(10, 3))
    slots, _ = tree.match_prefix([1, 2, 3])
    assert slots.tolist() == [10, 11, 12]


def test_a_longer_key_matches_the_whole_node():
    tree = make_tree()
    tree.insert([1, 2, 3], slots_from(10, 3))
    slots, _ = tree.match_prefix([1, 2, 3, 4, 5])
    assert slots.tolist() == [10, 11, 12]


def test_no_shared_first_token():
    tree = make_tree()
    tree.insert([1, 2, 3], slots_from(10, 3))
    assert len(tree.match_prefix([2, 2, 3])[0]) == 0


def test_a_match_crosses_nodes():
    tree = make_tree()
    tree.insert([1, 2, 3], slots_from(10, 3))
    tree.insert([1, 2, 3, 4, 5], slots_from(20, 5))
    assert len(tree.root.children) == 1
    slots, _ = tree.match_prefix([1, 2, 3, 4, 5, 6])
    assert slots.tolist() == [10, 11, 12, 23, 24]


# -------------------------------------------------------------------- insert


def test_insert_returns_what_was_already_there():
    tree = make_tree()
    assert tree.insert([1, 2, 3, 4], slots_from(10, 4))[0] == 0
    assert tree.insert([1, 2, 3, 4], slots_from(20, 4))[0] == 4
    assert tree.insert([1, 2, 9], slots_from(30, 3))[0] == 2
    assert tree.total_size() == 5
    check_accounting(tree)


def test_a_second_insert_keeps_the_first_slots():
    tree = make_tree()
    tree.insert([1, 2, 3], slots_from(10, 3))
    tree.insert([1, 2, 3], slots_from(20, 3))
    assert tree.match_prefix([1, 2, 3])[0].tolist() == [10, 11, 12]


def test_tokens_that_differ_share_only_their_prefix():
    tree = make_tree()
    tree.insert([1, 2, 3, 4], slots_from(10, 4))
    tree.insert([1, 2, 7, 8], slots_from(20, 4))
    assert tree.match_prefix([1, 2, 3, 4])[0].tolist() == [10, 11, 12, 13]
    assert tree.match_prefix([1, 2, 7, 8])[0].tolist() == [10, 11, 22, 23]
    assert tree.total_size() == 6


# --------------------------------------------------------------------- split


def test_a_match_inside_a_node_splits_it():
    tree = make_tree()
    tree.insert([1, 2, 3, 4, 5], slots_from(10, 5))
    slots, node = tree.match_prefix([1, 2, 9])
    assert slots.tolist() == [10, 11]
    assert list(node.key) == [1, 2]
    assert [list(c.key) for c in node.children.values()] == [[3, 4, 5]]
    assert tree.total_size() == 5, "a split moves no slots"
    # The long path still works.
    assert tree.match_prefix([1, 2, 3, 4, 5])[0].tolist() == [10, 11, 12, 13, 14]
    check_accounting(tree)


def test_a_split_keeps_locks():
    tree = make_tree()
    tree.insert([1, 2, 3, 4], slots_from(10, 4))
    _, leaf = tree.match_prefix([1, 2, 3, 4])
    tree.inc_lock_ref(leaf)
    _, upper = tree.match_prefix([1, 2])
    assert upper is not leaf
    assert upper.lock_ref == leaf.lock_ref == 1
    assert tree.protected_size == 4 and tree.evictable_size == 0
    check_accounting(tree)
    tree.dec_lock_ref(leaf)
    assert upper.lock_ref == 0 and tree.evictable_size == 4
    check_accounting(tree)


def test_a_key_that_ends_inside_a_node():
    tree = make_tree()
    tree.insert([1, 2, 3, 4], slots_from(10, 4))
    assert tree.match_prefix([1, 2])[0].tolist() == [10, 11]
    check_accounting(tree)


# --------------------------------------------------------------------- pages


def test_matches_round_down_to_pages():
    tree = make_tree(page_size=4)
    tree.insert(list(range(12)), slots_from(1, 12))
    assert len(tree.match_prefix(list(range(11)))[0]) == 8
    assert len(tree.match_prefix(list(range(6)) + [99, 99, 99])[0]) == 4
    check_accounting(tree)


def test_a_partial_page_is_not_stored():
    tree = make_tree(page_size=4)
    tree.insert(list(range(7)), slots_from(1, 7))
    assert tree.total_size() == 4


def test_a_sequence_shorter_than_a_page_is_not_stored():
    tree = make_tree(page_size=4)
    assert tree.insert([1, 2, 3], slots_from(1, 3))[0] == 0
    assert tree.total_size() == 0
    assert len(tree.match_prefix([1, 2, 3])[0]) == 0


def test_a_shared_first_token_with_a_different_first_page_is_a_sibling():
    tree = make_tree(page_size=4)
    tree.insert([1, 2, 3, 4], slots_from(1, 4))
    assert tree.insert([1, 2, 3, 9], slots_from(5, 4))[0] == 0
    assert len(tree.root.children) == 2
    assert tree.match_prefix([1, 2, 3, 9])[0].tolist() == [5, 6, 7, 8]
    assert tree.match_prefix([1, 2, 3, 4])[0].tolist() == [1, 2, 3, 4]
    check_accounting(tree)


# --------------------------------------------------------------------- evict


def test_lru_evicts_the_oldest_leaf():
    tree = make_tree()
    insert(tree, [1, 2, 3])
    insert(tree, [4, 5, 6])
    insert(tree, [7, 8, 9])
    tree.match_prefix([1, 2, 3])  # Now [4, 5, 6] is the oldest.
    assert tree.evict(1) == 3
    assert len(tree.match_prefix([4, 5, 6])[0]) == 0
    assert len(tree.match_prefix([1, 2, 3])[0]) == 3
    assert len(tree.match_prefix([7, 8, 9])[0]) == 3
    check_accounting(tree)


def test_evicted_slots_go_back_to_the_allocator():
    tree = make_tree(pool=8)
    insert(tree, [1, 2, 3, 4, 5, 6])
    assert tree.allocator.available_size() == 2
    assert tree.evict(100) == 6
    assert tree.allocator.available_size() == 8
    assert tree.evictable_size == 0


def test_a_locked_node_stays():
    tree = make_tree()
    _, node = insert(tree, [1, 2, 3])
    insert(tree, [4, 5, 6])
    tree.inc_lock_ref(node)
    assert tree.evict(100) == 3
    assert len(tree.match_prefix([1, 2, 3])[0]) == 3
    tree.dec_lock_ref(node)
    assert tree.evict(100) == 3


def test_a_parent_goes_only_after_its_children():
    tree = make_tree()
    insert(tree, [1, 2, 3])
    insert(tree, [1, 2, 3, 4, 5])
    insert(tree, [1, 2, 3, 6, 7])
    tree.match_prefix([1, 2, 3, 6, 7])
    # Each call is one step: oldest leaf, newest leaf, then the parent they exposed.
    assert tree.evict(1) == 2
    assert len(tree.match_prefix([1, 2, 3, 4, 5])[0]) == 3
    assert tree.match_prefix([1, 2, 3, 6, 7])[0].numel() == 5
    assert tree.evict(1) == 2
    assert tree.evict(1) == 3
    assert tree.total_size() == 0


def test_eviction_continues_into_exposed_parents_in_one_call():
    tree = make_tree()
    insert(tree, [1, 2, 3])
    insert(tree, [1, 2, 3, 4, 5])
    assert tree.evict(5) == 5
    assert tree.total_size() == 0


def test_eviction_can_free_more_than_asked():
    tree = make_tree()
    insert(tree, list(range(10)))
    assert tree.evict(1) == 10  # Nodes go whole.


def test_trimming_frees_only_the_end_of_a_leaf():
    tree = make_tree(trim_leaves=True)
    insert(tree, [1, 2, 3, 4, 5, 6])
    assert tree.evict(2) == 2
    assert len(tree.match_prefix([1, 2, 3, 4, 5, 6])[0]) == 4
    assert tree.allocator.available_size() == 996
    check_accounting(tree)


def test_trimming_keeps_whole_pages():
    tree = make_tree(page_size=4, trim_leaves=True)
    insert(tree, list(range(12)))
    assert tree.evict(5) == 8
    assert len(tree.match_prefix(list(range(12)))[0]) == 4
    check_accounting(tree)


def test_trimming_a_whole_leaf_continues_into_its_parent():
    tree = make_tree(trim_leaves=True)
    insert(tree, [1, 2, 3])
    insert(tree, [1, 2, 3, 4, 5])
    assert tree.evict(4) == 4
    assert len(tree.match_prefix([1, 2, 3, 4, 5])[0]) == 1
    check_accounting(tree)


def test_lfu_keeps_what_is_used_often():
    tree = make_tree(eviction_policy="lfu")
    insert(tree, [1, 2, 3])
    insert(tree, [4, 5, 6])
    for _ in range(3):
        insert(tree, [1, 2, 3])  # Counts as a hit each time.
    insert(tree, [7, 8, 9])  # The newest, and used least along with [4, 5, 6].
    tree.evict(1)
    assert len(tree.match_prefix([4, 5, 6])[0]) == 0
    tree.evict(1)
    assert len(tree.match_prefix([7, 8, 9])[0]) == 0
    assert len(tree.match_prefix([1, 2, 3])[0]) == 3


def test_a_request_inserting_twice_counts_once():
    tree = make_tree(eviction_policy="lfu")
    slots = tree.allocator.alloc(3)
    _, node = tree.insert([1, 2, 3], slots, inserted_len=0)
    assert node.hit_count == 1
    tree.insert([1, 2, 3], slots, inserted_len=3)
    assert node.hit_count == 1
    tree.insert([1, 2, 3], slots, inserted_len=0)
    assert node.hit_count == 2


# --------------------------------------------------------------------- locks


def test_lock_sizes():
    tree = make_tree()
    tree.insert([1, 2, 3, 4], slots_from(10, 4))
    _, a = tree.match_prefix([1, 2, 3, 4])
    assert tree.inc_lock_ref(a) == 4
    assert tree.inc_lock_ref(a) == 0, "already protected: this lock costs nothing"
    tree.dec_lock_ref(a)
    assert tree.protected_size == 4
    tree.dec_lock_ref(a)
    assert tree.protected_size == 0 and tree.evictable_size == 4


def test_locking_a_longer_path_only_charges_the_new_part():
    tree = make_tree()
    tree.insert([1, 2, 3, 4, 5, 6], slots_from(10, 6))
    _, short = tree.match_prefix([1, 2, 3])
    assert tree.inc_lock_ref(short) == 3
    _, long = tree.match_prefix([1, 2, 3, 4, 5, 6])
    assert tree.inc_lock_ref(long) == 3
    assert short.lock_ref == 2 and long.lock_ref == 1
    check_accounting(tree)


def test_an_unbalanced_unlock_raises():
    tree = make_tree()
    tree.insert([1, 2, 3], slots_from(10, 3))
    _, node = tree.match_prefix([1, 2, 3])
    with pytest.raises(AssertionError, match="nobody holds"):
        tree.dec_lock_ref(node)


def test_disabled_tree_does_nothing():
    tree = make_tree(disable=True)
    assert tree.insert([1, 2, 3], slots_from(10, 3))[0] == 0
    assert len(tree.match_prefix([1, 2, 3])[0]) == 0
    assert tree.total_size() == 0
    assert tree.evict(10) == 0


# ------------------------------------------------------- against brute force


def longest_common_prefix(ids: List[int], stored: List[List[int]], page_size: int) -> int:
    best = 0
    for other in stored:
        n = 0
        for a, b in zip(ids, other):
            if a != b:
                break
            n += 1
        best = max(best, n // page_size * page_size)
    return best


def stored_sequences(tree: RadixCache) -> List[List[int]]:
    """Everything the tree spells out, one list per leaf."""
    result = []
    stack = [(tree.root, [])]
    while stack:
        node, ids = stack.pop()
        ids = ids + list(node.key)
        if not node.children and ids:
            result.append(ids)
        stack.extend((c, ids) for c in node.children.values())
    return result


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("fast_match", [False, True])
@pytest.mark.parametrize("page_size", [1, 2, 4])
def test_random_operations_match_a_brute_force_model(page_size, fast_match, seed):
    rng = random.Random(seed)
    tree = make_tree(pool=64, page_size=page_size, fast_match=fast_match)
    locks = []

    for _ in range(300):
        op = rng.random()
        # A tiny alphabet, so that sequences keep sharing prefixes.
        ids = [rng.randint(0, 1) for _ in range(rng.randint(1, 12) * page_size)]
        if op < 0.4:
            tree.evict(len(ids) - tree.allocator.available_size())
            slots = tree.allocator.alloc(len(ids))
            if slots is None:
                continue
            matched, _ = tree.insert(ids, slots)
            tree.allocator.free(slots[:matched])
        elif op < 0.7:
            want = longest_common_prefix(ids, stored_sequences(tree), page_size)
            slots, node = tree.match_prefix(ids)
            assert len(slots) == want, (seed, ids, want, len(slots))
            assert tree.path_slots(node).tolist() == slots.tolist()
        elif op < 0.8:
            _, node = tree.match_prefix(ids)
            tree.inc_lock_ref(node)
            locks.append(node)
        elif op < 0.9 and locks:
            tree.dec_lock_ref(locks.pop(rng.randrange(len(locks))))
        else:
            tree.evict(rng.randint(1, 16))
        check_accounting(tree)
        # Every slot is either in the tree or on the free list.
        assert tree.total_size() + tree.allocator.available_size() == 64


def test_gallop_agrees_with_the_loop():
    rng = random.Random(0)
    for _ in range(2000):
        n = rng.randint(0, 40)
        a = array("q", [rng.randint(0, 1) for _ in range(n)])
        b = array("q", a[: rng.randint(0, n)] + array("q", [rng.randint(0, 1) for _ in range(5)]))
        offset = rng.randint(0, len(b))
        assert match_len_loop(a, b, offset) == match_len_gallop(a, b, offset)
        assert match_len_loop(b, b, 0) == match_len_gallop(b, b, 0) == len(b)
