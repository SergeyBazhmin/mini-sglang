"""KV cache memory: fixed buffers allocated once at startup, and the indices into them.

SGLang equivalents:
  - TokenToKVPool       mem_cache/memory_pool.py: MHATokenToKVPool (~line 2152)
  - TokenAllocator      mem_cache/allocator/token.py: TokenToKVPoolAllocator (~line 28)
  - ReqToTokenPool      mem_cache/memory_pool.py: ReqToTokenPool (~line 379)

Two levels of indirection replace M2's "one growing tensor per request":

    req_pool_idx        position in the request
         |                   |
         v                   v
    req_to_token[row,        pos] = slot            ReqToTokenPool
                                     |
                                     v
                       k_buffer[layer][slot] = K    TokenToKVPool
                       v_buffer[layer][slot] = V

A request's slots can be anywhere in the buffers, so nothing ever has to be
moved or grown, and the free list can hand out any slot. Only the table says
which slots belong together.

SGLang's real allocator is usually the paged one (allocator/paged.py): it hands
out pages of 64 slots, so a request's tokens are mostly contiguous, which the
GPU attention kernels need. By default ours is page_size=1, the case of SGLang's
TokenToKVPoolAllocator. --page-size turns on the paged behaviour (exercise 4).
"""

from typing import List, Optional

import torch


class TokenAllocator:
    """Free list of pages in the KV buffers. It only moves integers, never K/V.

    A page is page_size consecutive slots, named by its first slot, so with
    page_size=1 free_pages is simply a list of free slots. Sizes in this API are
    always in slots: available_size() is a multiple of page_size, and alloc()
    rounds up to whole pages.
    """

    def __init__(
        self,
        size: int,
        device: str,
        page_size: int = 1,
        lazy_free: bool = False,
        check_double_free: bool = False,
    ):
        if size <= 0 or size % page_size != 0:
            raise ValueError(f"Pool size {size} must be a positive multiple of {page_size}.")
        self.size = size
        self.page_size = page_size
        self.lazy_free = lazy_free
        # Slot 0 is never handed out. The table starts as zeros, so a read of an
        # unused table entry lands on a harmless slot. SGLang reserves it too.
        self.free_pages = torch.arange(
            1, size + 1, page_size, dtype=torch.int64, device=device
        )
        self.page_offsets = torch.arange(page_size, device=device)

        # Exercise 5: frees wait here, and join free_pages when alloc runs short.
        self.released: List[torch.Tensor] = []
        self.num_released = 0

        # Exercise 6: which pages are free. Slot 0 counts as free, so that
        # freeing it is caught as well.
        self.is_free: Optional[torch.Tensor] = None
        if check_double_free:
            self.is_free = torch.zeros(size + 1, dtype=torch.bool, device=device)
            self.is_free[self.free_pages] = True
            self.is_free[0] = True

    def aligned(self, num_tokens: int) -> int:
        """Slots taken up by num_tokens tokens: rounded up to whole pages."""
        return (num_tokens + self.page_size - 1) // self.page_size * self.page_size

    def available_size(self) -> int:
        return (len(self.free_pages) + self.num_released) * self.page_size

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        """Take the whole pages covering need_size slots, or return None and take
        nothing. Returns every slot of those pages, in order.
        """
        num_pages = self.aligned(need_size) // self.page_size
        if num_pages > len(self.free_pages):
            self.merge_released()
        if num_pages > len(self.free_pages):
            return None
        pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]
        if self.is_free is not None:
            self.is_free[pages] = False
        if self.page_size == 1:
            return pages
        return (pages[:, None] + self.page_offsets).flatten()

    def free(self, free_index: torch.Tensor):
        """Give back slots. The slots of a request must include each page's first
        slot, which holds for a table row: the rest of a page needs no bookkeeping.
        """
        pages = free_index
        if self.page_size > 1:
            pages = free_index[(free_index - 1) % self.page_size == 0]
        if self.is_free is not None:
            self._mark_free(pages)
        if self.lazy_free:
            self.released.append(pages)
            self.num_released += len(pages)
        else:
            self.free_pages = torch.cat((self.free_pages, pages))

    def merge_released(self):
        """SGLang: merge_and_sort_free. We don't sort, see the README."""
        if self.released:
            self.free_pages = torch.cat((self.free_pages, *self.released))
            self.released = []
            self.num_released = 0

    def _mark_free(self, pages: torch.Tensor):
        if self.is_free[pages].any() or len(pages.unique()) != len(pages):
            raise RuntimeError(f"Double free among slots {pages.tolist()}.")
        self.is_free[pages] = True


class ReqToTokenPool:
    """One row per running request. req_to_token[row, pos] is the slot of token `pos`.

    Entries past a request's length are stale slots from the row's previous
    owner. Readers must stop at (or mask past) the request's length.
    """

    def __init__(self, size: int, max_context_len: int, device: str):
        self.max_context_len = max_context_len
        # int64 so it can index the KV buffers without a cast. SGLang uses int32
        # because its table is read by kernels, and because it's half the size.
        self.req_to_token = torch.zeros(
            (size, max_context_len), dtype=torch.int64, device=device
        )
        self.free_slots: List[int] = list(range(size))

    def alloc(self) -> Optional[int]:
        return self.free_slots.pop() if self.free_slots else None

    def free(self, row: int):
        self.free_slots.append(row)


class TokenToKVPool:
    """The K and V of every token of every request, one pair of buffers per layer."""

    def __init__(
        self,
        size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
    ):
        # +1 for slot 0, see TokenAllocator. Slots are token-major, so one
        # token's K is one row: [size + 1, num_kv_heads, head_dim].
        shape = (size + 1, num_kv_heads, head_dim)
        self.k_buffer = [
            torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)
        ]
        self.v_buffer = [
            torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)
        ]

    @staticmethod
    def cell_size(
        num_layers: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype
    ) -> int:
        """Bytes for one token: K + V in every layer."""
        return num_layers * 2 * num_kv_heads * head_dim * (torch.finfo(dtype).bits // 8)
