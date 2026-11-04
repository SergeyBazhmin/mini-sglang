"""Cache policies without a model: how many tokens does each cache setup compute?

    uv run m4-radix-cache/cache_sim.py

This runs the real Scheduler and the real tree on fake_runner.py, so the numbers come
from the code under test, with no GPU. The result is how much prefill work each setup
does on the same requests, and that's a good proxy here: prefill's share of the time
depends on the machine, but what the cache skips doesn't.

"worst wait" is the most scheduler steps any request spent in the queue. It's the
price of reordering it: a request nobody else shares a prefix with can be passed
over again and again.

The workload is a closed loop: `concurrency` requests in the system at all times.
Each picks one of `num_prefixes` system prompts, with Zipf popularity, so a few are
hot and most are rare, and adds a private question. Try --pool up and down: with a
pool that holds everything, the policies can't differ.
"""

import random
from typing import Annotated, Dict, List, Tuple

import typer
from loguru import logger

import scheduler as scheduler_module
from common import EngineOptions, PortArgs, TokenizedGenerateReq
from fake_runner import FakeRunner
from scheduler import ForwardMode, Req, Scheduler

SETUPS: Dict[str, Dict] = {
    "no cache": {"disable_radix_cache": True},
    "lru": {},
    "lfu": {"eviction_policy": "lfu"},
    "lru + lpm": {"schedule_policy": "lpm"},
    "lru + dedup": {"in_batch_dedup": True},
    "lru + lpm + dedup": {"schedule_policy": "lpm", "in_batch_dedup": True},
}


class NullSocket:
    def send_pyobj(self, msg):
        pass


def make_workload(
    seed: int,
    num_requests: int,
    num_prefixes: int,
    prefix_len: int,
    question_len: int,
    max_new_tokens: int,
) -> List[TokenizedGenerateReq]:
    rng = random.Random(seed)
    weights = [1 / (rank + 1) for rank in range(num_prefixes)]
    prefixes = [
        [100_000 * (p + 1) + i for i in range(prefix_len)] for p in range(num_prefixes)
    ]
    reqs = []
    for rid in range(num_requests):
        prefix = rng.choices(prefixes, weights)[0]
        question = [rng.randint(1000, 90_000) for _ in range(question_len)]
        reqs.append(
            TokenizedGenerateReq(
                str(rid),
                prefix + question,
                rng.randint(max_new_tokens // 2, max_new_tokens),
                0.0,
                1.0,
            )
        )
    return reqs


def simulate(
    workload: List[TokenizedGenerateReq], pool: int, concurrency: int, options: EngineOptions
) -> Tuple[Scheduler, int]:
    scheduler_module.ModelRunner = FakeRunner
    FakeRunner.verify_kv = False
    s = Scheduler("fake", PortArgs.new(), concurrency, 8192, pool, 0.6, 8192, options)
    s.send_to_detokenizer = NullSocket()
    pending = list(reversed(workload))
    arrived_at: Dict[str, int] = {}
    worst_wait = 0
    steps = 0
    while pending or s.waiting_queue or s.running_batch:
        while pending and len(s.waiting_queue) + len(s.running_batch) < concurrency:
            req = Req(pending.pop())
            arrived_at[req.rid] = steps
            s.handle_generate_request(req)
        batch = s.get_next_batch_to_run()
        if batch is not None:
            if batch.forward_mode == ForwardMode.EXTEND:
                for req in batch.reqs:
                    worst_wait = max(worst_wait, steps - arrived_at[req.rid])
            s.process_batch_result(batch, s.run_batch(batch))
        steps += 1
    return s, worst_wait


def main(
    pool: Annotated[int, typer.Option(help="KV pool size in tokens.")] = 8192,
    num_requests: Annotated[int, typer.Option(help="Requests to run.")] = 1000,
    concurrency: Annotated[int, typer.Option(help="Requests in the system.")] = 32,
    num_prefixes: Annotated[int, typer.Option(help="Distinct system prompts.")] = 50,
    prefix_len: Annotated[int, typer.Option(help="Tokens in a system prompt.")] = 1000,
    question_len: Annotated[int, typer.Option(help="Tokens in a question.")] = 32,
    max_new_tokens: Annotated[int, typer.Option(help="Longest answer.")] = 32,
    seed: Annotated[int, typer.Option(help="Workload seed.")] = 0,
):
    """Compare cache setups on one workload."""
    logger.remove()
    workload = make_workload(
        seed, num_requests, num_prefixes, prefix_len, question_len, max_new_tokens
    )
    print(
        f"pool={pool} requests={num_requests} concurrency={concurrency} "
        f"prefixes={num_prefixes} prefix_len={prefix_len}"
    )
    print(f"{'setup':>20} {'computed':>10} {'skipped':>10} {'hit rate':>9} {'worst wait':>11}")
    for name, setup in SETUPS.items():
        s, worst_wait = simulate(workload, pool, concurrency, EngineOptions(**setup))
        print(
            f"{name:>20} {s.num_prefill_tokens:>10} {s.num_cached_tokens:>10} "
            f"{s._cache_hit_rate():>9.3f} {worst_wait:>11}"
        )


if __name__ == "__main__":
    typer.run(main)
