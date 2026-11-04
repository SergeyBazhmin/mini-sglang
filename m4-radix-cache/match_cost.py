"""What a key comparison costs, token by token against sliced compares.

    uv run m4-radix-cache/match_cost.py

No model. A chat's whole history is matched on every turn, and the scheduler matches
the head of the queue on every pass while the pool is full, so this runs on the
scheduler's thread, and it holds up the GPU for as long as it takes.
"""

import time
from array import array
from typing import Annotated, Callable

import typer

from radix_cache import match_len_gallop, match_len_loop


def time_it(match_len: Callable, a: array, b: array, rounds: int) -> float:
    start = time.perf_counter()
    for _ in range(rounds):
        match_len(a, b, 0)
    return (time.perf_counter() - start) / rounds * 1e6


def main(rounds: Annotated[int, typer.Option(help="Matches to average over.")] = 200):
    """Time both comparisons, on a full match and on an early mismatch."""
    print(f"{'tokens':>8} {'case':>10} {'loop us':>10} {'gallop us':>10} {'speedup':>8}")
    for length in (64, 1024, 8192, 65536):
        a = array("q", range(length))
        for case, position in (("full", length), ("early", 8)):
            b = array("q", a)
            if position < length:
                b[position] = -1
            loop = time_it(match_len_loop, a, b, rounds)
            gallop = time_it(match_len_gallop, a, b, rounds)
            print(f"{length:>8} {case:>10} {loop:>10.2f} {gallop:>10.2f} {loop / gallop:>7.1f}x")


if __name__ == "__main__":
    typer.run(main)
