"""How much of a decode step is copying K/V out of the pool?

    uv run m3-kv-cache/read_cost.py --model Qwen/Qwen3-0.6B

Runs ModelRunner directly, with no server, so that the context length is exact:
every request has a random prompt of that length, and nothing stops at EOS.

  step     one whole ModelRunner.forward
  gather   the two reads in PoolAttention._run_decode, all layers added up

This is the number M5 removes. The timer calls torch.cuda.synchronize() around
each gather, only in this script, so a step is a bit slower than in a real run.
"""

import random
import time
from typing import Annotated, List

import typer

from common import EngineOptions
from model_runner import FakeReq, ModelRunner

MIN_TOKEN_ID, MAX_TOKEN_ID = 1000, 20000  # Ordinary tokens. Their meaning doesn't matter.
CONTEXTS = [128, 1024, 4096]


def main(
    model: Annotated[str, typer.Option(help="HF model id or local path.")] = "Qwen/Qwen3-0.6B",
    batch_size: Annotated[int, typer.Option(help="Requests decoding together.")] = 16,
    steps: Annotated[int, typer.Option(help="Decode steps to average over.")] = 32,
):
    """Time the K/V gathers against the whole decode step, at several context lengths."""
    longest = max(CONTEXTS) + steps + 1
    runner = ModelRunner(
        model,
        max_running_requests=batch_size,
        max_total_tokens=batch_size * longest,
        context_length=longest,
        options=EngineOptions(profile_gather=True),
    )

    print(f"device={runner.device} batch_size={batch_size}")
    print(f"{'context':>8} {'step ms':>9} {'gather ms':>10} {'gather share':>13} {'MB read':>9}")
    for context in CONTEXTS:
        reqs: List[FakeReq] = []
        # One at a time: this is setup, and no burst is being measured.
        for _ in range(batch_size):
            ids = [random.randint(MIN_TOKEN_ID, MAX_TOKEN_ID) for _ in range(context)]
            req = FakeReq(ids, steps + 1)
            req.output_ids.append(runner.forward([req], is_extend=True)[0])
            reqs.append(req)

        runner.gather_timer.seconds = 0.0
        step_seconds = 0.0
        for _ in range(steps):
            start = time.perf_counter()
            tokens = runner.forward(reqs, is_extend=False)
            step_seconds += time.perf_counter() - start
            for req, token in zip(reqs, tokens):
                req.output_ids.append(token)

        for req in reqs:
            runner.free_req(req)

        step_ms = step_seconds / steps * 1000
        gather_ms = runner.gather_timer.seconds / steps * 1000
        read_mb = batch_size * context * runner.cell_size / 2**20
        print(
            f"{context:>8} {step_ms:>9.2f} {gather_ms:>10.2f} "
            f"{gather_ms / step_ms:>12.0%} {read_mb:>9.0f}"
        )


if __name__ == "__main__":
    typer.run(main)
