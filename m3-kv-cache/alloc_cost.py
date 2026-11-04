"""What do pages and lazy frees do to the allocator?

    uv run m3-kv-cache/alloc_cost.py

No model: this replays what decoding does to the allocator. Every step, each of
--batch-size requests takes a slot, or a page when it crosses a page boundary.
When a request reaches its length, it frees everything and a new one takes its
place. Lengths differ, so requests finish at different steps. Only the alloc and
free calls are timed.

Prints microseconds per step, and the share of held slots that no token uses.
With pages, a request wastes half a page on average, so the share should be near
(page_size / 2) / (mean_length + page_size / 2).
"""

import time
from typing import Annotated, List

import torch
import typer

from memory_pool import TokenAllocator

MIN_LENGTH = 128
LENGTH_RANGE = 896


def simulate(
    pool_size: int, device: str, page_size: int, lazy_free: bool, batch_size: int, steps: int
):
    pool_size -= pool_size % page_size
    allocator = TokenAllocator(pool_size, device, page_size=page_size, lazy_free=lazy_free)
    lengths = [MIN_LENGTH + i * 37 % LENGTH_RANGE for i in range(batch_size)]
    positions = [i % length for i, length in enumerate(lengths)]  # Out of phase.
    held: List[List[torch.Tensor]] = [
        [allocator.alloc(allocator.aligned(p))] if p else [] for p in positions
    ]

    seconds = 0.0
    waste_sum = 0.0
    for _ in range(steps):
        for i in range(batch_size):
            start = time.perf_counter()
            if positions[i] % page_size == 0:
                slots = allocator.alloc(page_size)
                seconds += time.perf_counter() - start
                held[i].append(slots)
            positions[i] += 1
            if positions[i] == lengths[i]:
                slots = torch.cat(held[i])
                start = time.perf_counter()
                allocator.free(slots)
                seconds += time.perf_counter() - start
                held[i] = []
                positions[i] = 0
        used = sum(positions)
        waste_sum += 1 - used / max(1, sum(allocator.aligned(p) for p in positions))
    if device == "cuda":
        torch.cuda.synchronize()
    return seconds / steps * 1e6, waste_sum / steps


def main(
    pool_size: Annotated[int, typer.Option(help="Pool size in slots. It sets the free list size.")] = 1_000_000,
    batch_size: Annotated[int, typer.Option(help="Requests decoding at once.")] = 64,
    # Enough allocs to use up the pool, or a lazy free list would never merge.
    steps: Annotated[int, typer.Option(help="Decode steps to simulate.")] = 20000,
    device: Annotated[str, typer.Option(help="Where the free list lives.")] = "cpu",
):
    """Compare page sizes and eager against lazy frees."""
    print(f"device={device} pool={pool_size} batch_size={batch_size} steps={steps}")
    print(f"{'page size':>10} {'free':>6} {'us per step':>12} {'unused slots':>13}")
    for page_size in (1, 16, 64, 256):
        for lazy_free in (False, True):
            step_us, waste = simulate(pool_size, device, page_size, lazy_free, batch_size, steps)
            name = "lazy" if lazy_free else "eager"
            print(f"{page_size:>10} {name:>6} {step_us:>12.1f} {waste:>12.1%}")


if __name__ == "__main__":
    typer.run(main)
