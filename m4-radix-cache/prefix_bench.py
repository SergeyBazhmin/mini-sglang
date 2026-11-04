"""Time to first token with and without a cached prefix.

    uv run m4-radix-cache/prefix_bench.py --port 8000

All requests share a long system prompt and differ in their question, which is what
a chatbot or an agent framework sends. Three measurements, each from an empty cache:

  cold    the first request, which has to compute the system prompt
  warm    the requests after it, one at a time, which should skip it
  burst   many at once. They run in one prefill, so nothing is cached yet for any of
          them. --in-batch-dedup is about this row.

Start the server with --disable-radix-cache to compare. Prefill's share of a request
is what the cache saves, so the longer the system prompt, the larger the difference.

The scheduler log has the other half of the story: #cached-token and cache hit rate.
"""

import asyncio
import statistics
import time
import urllib.request
from typing import Annotated, List

import typer
from openai import AsyncOpenAI

EXTRA = {"chat_template_kwargs": {"enable_thinking": False}}
SENTENCE = "You answer questions about the company handbook, politely and with care. "


async def ttft(client, model: str, system: str, question: str, max_tokens: int) -> float:
    start = time.perf_counter()
    first = None
    stream = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        max_tokens=max_tokens,
        temperature=0,
        stream=True,
        extra_body=EXTRA,
    )
    async for chunk in stream:
        if first is None and chunk.choices and chunk.choices[0].delta.content:
            first = time.perf_counter() - start
    return first if first is not None else time.perf_counter() - start


def report(name: str, times: List[float]):
    ms = [t * 1000 for t in times]
    print(
        f"{name:>6} {len(ms):>4} {statistics.mean(ms):>10.1f} "
        f"{statistics.median(ms):>10.1f} {max(ms):>10.1f}"
    )


async def run(port: int, system_repeat: int, num_warm: int, burst: int, max_tokens: int):
    base = f"http://127.0.0.1:{port}"
    client = AsyncOpenAI(base_url=f"{base}/v1", api_key="EMPTY", timeout=3600)
    model = (await client.models.list()).data[0].id
    system = SENTENCE * system_repeat

    def flush():
        urllib.request.urlopen(urllib.request.Request(f"{base}/flush_cache", method="POST"))

    flush()
    print(f"model={model} system prompt: {system_repeat} sentences, max_tokens={max_tokens}")
    print(f"{'':>6} {'n':>4} {'mean ms':>10} {'p50 ms':>10} {'max ms':>10}")

    cold = await ttft(client, model, system, "Question 0: how many days off?", max_tokens)
    report("cold", [cold])
    warm = [
        await ttft(client, model, system, f"Question {i}: how many days off?", max_tokens)
        for i in range(1, num_warm + 1)
    ]
    report("warm", warm)

    flush()
    times = await asyncio.gather(
        *(
            ttft(client, model, system, f"Question {i}: who do I ask?", max_tokens)
            for i in range(burst)
        )
    )
    report("burst", list(times))


def main(
    port: Annotated[int, typer.Option(help="Port nano-serve is listening on.")] = 8000,
    system_repeat: Annotated[int, typer.Option(help="Sentences in the system prompt.")] = 100,
    num_warm: Annotated[int, typer.Option(help="Requests after the cold one.")] = 16,
    burst: Annotated[int, typer.Option(help="Requests in the burst.")] = 16,
    max_tokens: Annotated[int, typer.Option(help="max_tokens of every request.")] = 4,
):
    """Measure the TTFT of requests that share a system prompt."""
    asyncio.run(run(port, system_repeat, num_warm, burst, max_tokens))


if __name__ == "__main__":
    typer.run(main)
