"""Radix tree over token ids: which K/V in the pool can be reused by a new request.

SGLang equivalents, all in mem_cache/radix_cache.py unless noted:
  - TreeNode                      TreeNode (~line 250)
  - RadixCache.match_prefix       match_prefix (~line 364), _match_prefix_helper (~line 651)
  - RadixCache.insert             insert (~line 428), _insert_helper (~line 705)
  - RadixCache._split             _split_node (~line 677)
  - RadixCache.checkpoint         checkpoint (~line 510), _insert_cache (~line 450)
  - RadixCache.evict              evict (~line 552)
  - inc_lock_ref, dec_lock_ref    inc_lock_ref (~line 582), dec_lock_ref (~line 597)
  - match_len_gallop              RadixKey.match_at (~line 182)
  - EVICTION_POLICIES             mem_cache/evict_policy.py

Why it works: the K/V of the token at position i depends only on tokens 0..i, because
attention is causal. So two requests that start with the same tokens have the same
K/V there, whatever they do afterwards, and the second can read the first one's.

A node is one edge of the tree: `key` is a run of tokens, and `value` is the slot
of each token's K/V, so the two always have the same length. A path from the root
spells out a cached sequence.

    root
     └── [You are a helpful bot.]         value=[ 3,  4,  5,  6,  7,  8]
          ├── [ What is 2+2?]             value=[ 9, 10, 11, 12, 13, 14]
          └── [ Tell me a joke.]          value=[21, 22, 23, 24, 25, 26]

Every slot has exactly one owner, and this is what the whole module maintains:
  - the free list (the allocator),
  - the tree: a slot in a node's value,
  - a request: slots in its table row at positions >= cache_protected_len.
Positions below cache_protected_len are read from the tree, and belong to it.

A tree slot is either EVICTABLE or PROTECTED. A running request reads its prefix on
every step, so it locks the path it uses: lock_ref is how many requests pass through
a node. Only a leaf with lock_ref 0 can go. A parent's K/V can't be evicted before
its children's, since a child without its parent can never be reached by a match.
"""

import heapq
from array import array
from typing import Callable, Dict, Hashable, List, Optional, Protocol, Sequence, Set, Tuple

import torch

from memory_pool import TokenAllocator


class CacheReq(Protocol):
    """What the tree needs from a request (scheduler.Req and FakeReq have it)."""

    input_ids: List[int]
    output_ids: List[int]
    req_pool_idx: Optional[int]
    kv_allocated_len: int
    prefix_len: int  # Tokens found in the tree by the last match.
    prefix_slots: Optional[torch.Tensor]  # Their slots.
    last_node: Optional["TreeNode"]  # End of the path this request locks or matched.
    cache_protected_len: int  # Row positions below this belong to the tree.
    cache_inserted_len: int  # Tokens this request already offered to the tree.


class TreeNode:
    next_id = 0

    def __init__(self):
        self.children: Dict[Hashable, "TreeNode"] = {}
        self.parent: Optional["TreeNode"] = None
        self.key = array("q")  # 8-byte ints, not Python ints: slices compare in C.
        self.value = torch.empty(0, dtype=torch.int64)
        self.lock_ref = 0
        self.last_access = 0
        self.hit_count = 0
        self.id = TreeNode.next_id  # Breaks ties, so eviction is deterministic.
        TreeNode.next_id += 1


# Lower evicts first. A logical clock instead of SGLang's monotonic time
# makes the order the same on every run.
EVICTION_POLICIES: Dict[str, Callable[[TreeNode], Tuple[int, ...]]] = {
    "lru": lambda node: (node.last_access,),
    "lfu": lambda node: (node.hit_count, node.last_access),
}


def match_len_loop(a: array, b: array, offset: int) -> int:
    """How many leading tokens of a equal those of b[offset:]. One Python step per token."""
    n = min(len(a), len(b) - offset)
    for i in range(n):
        if a[i] != b[offset + i]:
            return i
    return n


def match_len_gallop(a: array, b: array, offset: int) -> int:
    """Same answer, faster on long matches. Compare doubling windows with one C-level
    slice compare each, then binary search the window that differs."""
    n = min(len(a), len(b) - offset)
    lo, step = 0, 1
    while lo < n:
        hi = min(lo + step, n)
        if a[lo:hi] != b[offset + lo : offset + hi]:
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if a[lo:mid] == b[offset + lo : offset + mid]:
                    lo = mid
                else:
                    hi = mid
            return lo
        lo = hi
        step *= 2
    return n


class RadixCache:
    def __init__(
        self,
        allocator: Optional[TokenAllocator],
        req_to_token: Optional[torch.Tensor],
        page_size: int = 1,
        disable: bool = False,
        eviction_policy: str = "lru",
        fast_match: bool = False,
        trim_leaves: bool = False,
    ):
        """allocator and req_to_token can be None for a tree that only answers
        matches, which is how SGLang builds waiting_queue_radix_tree."""
        self.allocator = allocator
        self.req_to_token = req_to_token
        self.page_size = page_size
        self.disable = disable
        self.priority = EVICTION_POLICIES[eviction_policy]
        self.match_len = match_len_gallop if fast_match else match_len_loop
        self.trim_leaves = trim_leaves  # See evict.

        self.root = TreeNode()
        self.root.lock_ref = 1  # Never a candidate for eviction.
        self.evictable_leaves: Set[TreeNode] = set()
        self.evictable_size = 0
        self.protected_size = 0
        self.clock = 0

    # ------------------------------------------------------------- the tree

    def _child_key(self, key: array, offset: int) -> Hashable:
        """A node's children differ in their first token, or first page with pages.

        A child that shares only part of its first page with a key couldn't be
        split, because a node holds whole pages. So such a child doesn't count as
        the same branch at all, and it gets a sibling.
        """
        if self.page_size == 1:
            return key[offset]
        return tuple(key[offset : offset + self.page_size])

    def _common_len(self, child: TreeNode, key: array, offset: int) -> int:
        """Tokens shared by child.key and key[offset:], in whole pages."""
        n = self.match_len(child.key, key, offset)
        return n // self.page_size * self.page_size

    def _key(self, ids: Sequence[int]) -> array:
        return array("q", ids[: len(ids) // self.page_size * self.page_size])

    def _tick(self) -> int:
        self.clock += 1
        return self.clock

    def match_prefix(self, ids: Sequence[int]) -> Tuple[torch.Tensor, TreeNode]:
        """Longest cached prefix of ids, as slots, and the node where it ends.

        If it ends inside a node, the node is split. A lock must cover exactly the
        slots that are used, and a lock covers whole nodes.
        """
        key = self._key(ids)
        if self.disable or not key:
            return self.root.value, self.root

        now = self._tick()
        values: List[torch.Tensor] = []
        node = self.root
        offset = 0
        while offset < len(key):
            child = node.children.get(self._child_key(key, offset))
            if child is None:
                break
            child.last_access = now
            matched = self._common_len(child, key, offset)
            if matched < len(child.key):
                child = self._split(child, matched)
            values.append(child.value)
            node = child
            offset += matched
        slots = torch.cat(values) if values else self.root.value
        return slots, node

    def insert(
        self, ids: Sequence[int], slots: torch.Tensor, inserted_len: int = 0
    ) -> Tuple[int, TreeNode]:
        """Store whole pages of ids. Returns how many tokens were already in the tree,
        and the node where the sequence ends.

        The tree takes ownership of slots[matched:]. slots[:matched] are not stored,
        so the caller still owns them, and must see that they end up freed.

        inserted_len says how much of ids this request offered before, so that a
        request counts as one hit on a node, not one per insert (LFU).
        """
        key = self._key(ids)
        if self.disable or not key:
            return 0, self.root

        now = self._tick()
        node = self.root
        offset = 0
        while offset < len(key):
            child = node.children.get(self._child_key(key, offset))
            if child is None:
                break
            child.last_access = now
            matched = self._common_len(child, key, offset)
            if matched < len(child.key):
                child = self._split(child, matched)
            node = child
            offset += matched
            if offset > inserted_len:
                node.hit_count += 1

        if offset < len(key):
            new = TreeNode()
            new.parent = node
            new.key = key[offset:]
            new.value = slots[offset : len(key)].clone()
            new.last_access = now
            new.hit_count = 1
            node.children[self._child_key(new.key, 0)] = new
            self.evictable_size += len(new.key)
            self._update_leaf(node)
            self._update_leaf(new)
            node = new
        return offset, node

    def _split(self, child: TreeNode, split_len: int) -> TreeNode:
        """parent -> child becomes parent -> new -> child. new takes child's first
        split_len tokens. No K/V moves, and no slot changes owner.

        new.lock_ref = child.lock_ref is right because every lock that passes through
        child also passes through new. Locks that ended at child's parent don't
        reach it, and none can end inside it.
        """
        parent = child.parent
        new = TreeNode()
        new.parent = parent
        new.key = child.key[:split_len]
        new.value = child.value[:split_len].clone()
        new.lock_ref = child.lock_ref
        new.last_access = child.last_access
        new.hit_count = child.hit_count

        child.key = child.key[split_len:]
        child.value = child.value[split_len:].clone()
        child.parent = new
        new.children[self._child_key(child.key, 0)] = child
        parent.children[self._child_key(new.key, 0)] = new
        return new

    def path_slots(self, node: TreeNode) -> torch.Tensor:
        """Slots of the whole sequence that ends at node."""
        values: List[torch.Tensor] = []
        while node is not self.root:
            values.append(node.value)
            node = node.parent
        return torch.cat(values[::-1]) if values else self.root.value

    def total_size(self) -> int:
        return self.evictable_size + self.protected_size

    # ------------------------------------------------------------- eviction

    def _update_leaf(self, node: TreeNode):
        if node is self.root:
            return
        if node.lock_ref == 0 and not node.children:
            self.evictable_leaves.add(node)
        else:
            self.evictable_leaves.discard(node)

    def evict(self, num_tokens: int) -> int:
        """Free at least num_tokens if that many are evictable. Returns the slots freed.

        Whole nodes go, so this can free more than asked. The heap is rebuilt on each
        call, which is O(leaves). It's fine because a call is rare: after the pool
        fills up, about one per decode step at worst.

        With trim_leaves, only the end of the last leaf goes. The start of a sequence
        stays valid without its end, so this keeps what's worth keeping. SGLang
        evicts whole nodes.
        """
        if self.disable:
            return 0
        heap = [(self.priority(n), n.id, n) for n in self.evictable_leaves]
        heapq.heapify(heap)
        evicted = 0
        while evicted < num_tokens and heap:
            _, _, node = heapq.heappop(heap)
            if self.trim_leaves:
                keep = len(node.key) - (num_tokens - evicted)
                keep = keep // self.page_size * self.page_size
                if keep > 0:
                    self._split(node, keep)
            self.allocator.free(node.value)
            evicted += len(node.key)

            parent = node.parent
            del parent.children[self._child_key(node.key, 0)]
            self.evictable_leaves.discard(node)
            self.evictable_size -= len(node.key)
            self._update_leaf(parent)
            # Its parent may have just become a leaf. It competes with the others.
            if parent in self.evictable_leaves:
                heapq.heappush(heap, (self.priority(parent), parent.id, parent))
        return evicted

    def inc_lock_ref(self, node: TreeNode) -> int:
        """Lock the path from node to the root. Returns how many tokens stopped being
        evictable, which the scheduler must count against its budget."""
        delta = 0
        while node is not self.root:
            if node.lock_ref == 0:
                self.evictable_size -= len(node.key)
                self.protected_size += len(node.key)
                delta += len(node.key)
            node.lock_ref += 1
            self._update_leaf(node)
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode):
        while node is not self.root:
            assert node.lock_ref > 0, "unlocked a node nobody holds"
            if node.lock_ref == 1:
                self.evictable_size += len(node.key)
                self.protected_size -= len(node.key)
            node.lock_ref -= 1
            self._update_leaf(node)
            node = node.parent

    # -------------------------------------------------------------- requests

    def match_req(self, req: CacheReq):
        """Find what req can skip. SGLang: match_kv_cache in mem_cache/common.py (~222),
        Req._compute_max_prefix_len in schedule_batch.py (~1778).

        The last token is left out of the key. Prefill must run at least one token
        through the model, because the logits for the next token come from there.
        """
        ids = req.input_ids + req.output_ids
        slots, node = self.match_prefix(ids[:-1])
        req.prefix_len = len(slots)
        req.prefix_slots = slots
        req.last_node = node

    def lock_req(self, req: CacheReq) -> int:
        return self.inc_lock_ref(req.last_node) if req.last_node is not None else 0

    def unlock_req(self, req: CacheReq):
        if req.last_node is not None:
            self.dec_lock_ref(req.last_node)
            req.last_node = None

    def checkpoint(self, req: CacheReq):
        """Hand the request's K/V to the tree, while it keeps running.

        The scheduler calls this after prefill, so that requests arriving from now
        on hit this prompt, and at the end of a request, so that they also hit its
        output (the next turn of a chat starts with it).

        Two requests can compute the same tokens, if they ran in one prefill batch.
        Then the second finds its K/V already in the tree. It frees its copy and
        points its table row at the tree's slots: the values are the same, and a
        slot can have only one owner.

        Afterwards the request locks the whole inserted path, so what it reads can't
        be evicted, and the tail past the last whole page stays its own.
        """
        num_tokens = req.kv_allocated_len // self.page_size * self.page_size
        protected = req.cache_protected_len
        if self.disable or num_tokens <= protected:
            return

        row = req.req_pool_idx
        slots = self.req_to_token[row, :num_tokens].clone()
        ids = (req.input_ids + req.output_ids)[:num_tokens]
        matched, last_node = self.insert(ids, slots, req.cache_inserted_len)
        # The protected prefix is locked, so it can't have left the tree.
        assert matched >= protected, f"{matched=} {protected=}"

        if matched > protected:
            self.allocator.free(slots[protected:matched])
            self.req_to_token[row, protected:matched] = self.path_slots(last_node)[
                protected:matched
            ]

        self.unlock_req(req)
        req.last_node = last_node
        self.lock_req(req)
        req.cache_protected_len = num_tokens
        req.cache_inserted_len = max(req.cache_inserted_len, num_tokens)
