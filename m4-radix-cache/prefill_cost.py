"""A burst of new requests, per-request attention against one padded call.

    uv run m4-radix-cache/prefill_cost.py --model Qwen/Qwen3-0.6B

Runs ModelRunner directly, with no server. "Even" prompts are all about the same
length, which is the best case for padding. In "skewed" prompts one is four times
as long as the others, so every short prompt pays for padding up to it.
"""

import random
import statistics
import time
from typing import Annotated, List

import typer

from common import EngineOptions
from model_runner import FakeReq, ModelRunner

MIN_TOKEN_ID, MAX_TOKEN_ID = 1000, 20000  # Ordinary tokens. Their meaning doesn't matter.
NUM_WARMUP_ROUNDS = 1


def prompt_lengths(burst_size: int, prompt_len: int, skewed: bool) -> List[int]:
    lengths = [prompt_len - i % 8 for i in range(burst_size)]
    if skewed:
        lengths[0] = prompt_len * 4
    return lengths


def run_burst(runner: ModelRunner, lengths: List[int]) -> float:
    """Time one prefill step. forward returns plain ints, so the GPU is done by then."""
    reqs = [
        FakeReq([random.randint(MIN_TOKEN_ID, MAX_TOKEN_ID) for _ in range(n)], 1)
        for n in lengths
    ]
    start = time.perf_counter()
    runner.forward(reqs, is_extend=True)
    elapsed = time.perf_counter() - start
    for req in reqs:
        runner.free_req(req)
    return elapsed


def main(
    model: Annotated[str, typer.Option(help="HF model id or local path.")] = "Qwen/Qwen3-0.6B",
    burst_size: Annotated[int, typer.Option(help="New requests in the prefill.")] = 32,
    prompt_len: Annotated[int, typer.Option(help="Typical prompt length.")] = 256,
    rounds: Annotated[int, typer.Option(help="Bursts to average over.")] = 10,
):
    """Compare the two prefill paths on the same model."""
    longest = max(prompt_lengths(burst_size, prompt_len, skewed=True))
    runner = ModelRunner(
        model,
        max_running_requests=burst_size,
        max_total_tokens=(burst_size + 3) * longest,
        context_length=longest + 1,
        options=EngineOptions(disable_radix_cache=True),
    )
    layers = runner.model.get_decoder().layers

    print(f"device={runner.device} burst_size={burst_size} prompt_len={prompt_len}")
    print(f"{'prompts':>8} {'loop ms':>9} {'batched ms':>11} {'speedup':>8}")
    for skewed in (False, True):
        lengths = prompt_lengths(burst_size, prompt_len, skewed)
        times = {}
        for batched in (False, True):
            for layer in layers:
                layer.self_attn.batched_prefill = batched
            samples = [run_burst(runner, lengths) for _ in range(NUM_WARMUP_ROUNDS + rounds)]
            times[batched] = statistics.mean(samples[NUM_WARMUP_ROUNDS:]) * 1000
        name = "skewed" if skewed else "even"
        print(
            f"{name:>8} {times[False]:>9.1f} {times[True]:>11.1f} "
            f"{times[False] / times[True]:>7.2f}x"
        )


if __name__ == "__main__":
    typer.run(main)
