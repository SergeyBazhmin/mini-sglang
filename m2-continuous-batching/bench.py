"""Throughput benchmark: send many chat requests, at most C at a time.

    uv run m2-continuous-batching/bench.py --port 8000 --num-requests 32 --concurrency 32

Run it against M1 and M2 with the same settings and compare. M1 serves one
request at a time, so its output tokens/s stays flat as concurrency grows.
M2 should scale up until the GPU is busy.

Every request asks for a long answer at temperature 0, so most of them run
until max_tokens and the two servers do about the same amount of work.

With --min-tokens, each request instead draws its own max_tokens from
[min_tokens, max_tokens]. Comparing --decode-first needs this: if every request
has the same length they all finish on the same step, and decode-first looks just as good
as continuous batching.

Token counts come from the server's usage chunk (stream_options.include_usage).
M1 doesn't send one, so for M1 the output text is re-tokenized instead; that
can be off by a token or so per request.

SGLang's real tool for this: python -m sglang.bench_serving.
"""

import asyncio
import random
import statistics
import time
from typing import Annotated, Dict, List, Optional

import typer
from openai import AsyncOpenAI
from transformers import AutoTokenizer

EXTRA = {"chat_template_kwargs": {"enable_thinking": False}}

TOPICS = [
    "the history of the printing press",
    "how a CPU cache works",
    "the water cycle",
    "the rules of chess",
    "how vaccines train the immune system",
    "the life of a star",
    "how TCP handles packet loss",
    "the causes of the French Revolution",
]


async def one_request(client, model, prompt, max_tokens):
    start = time.perf_counter()
    ttft: Optional[float] = None
    text: List[str] = []
    completion_tokens: Optional[int] = None
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
        extra_body=EXTRA,
    )
    async for chunk in stream:
        if chunk.usage is not None:
            completion_tokens = chunk.usage.completion_tokens
        if chunk.choices and chunk.choices[0].delta.content:
            if ttft is None:
                ttft = time.perf_counter() - start
            text.append(chunk.choices[0].delta.content)
    return "".join(text), completion_tokens, ttft, time.perf_counter() - start


async def run_bench(
    port: int,
    num_requests: int,
    concurrency: int,
    max_tokens: int,
    min_tokens: Optional[int] = None,
) -> Dict[str, float]:
    client = AsyncOpenAI(
        base_url=f"http://127.0.0.1:{port}/v1", api_key="EMPTY", timeout=3600
    )
    model = (await client.models.list()).data[0].id
    prompts = [
        f"Write a detailed essay about {TOPICS[i % len(TOPICS)]}. (Request {i}.)"
        for i in range(num_requests)
    ]

    # Seeded, so that two servers get the same lengths in the same order.
    rng = random.Random(0)
    lengths = [
        max_tokens if min_tokens is None else rng.randint(min_tokens, max_tokens)
        for _ in prompts
    ]

    sem = asyncio.Semaphore(concurrency)

    async def limited(prompt, length):
        async with sem:
            return await one_request(client, model, prompt, length)

    print(
        f"model={model} requests={num_requests} concurrency={concurrency} "
        f"max_tokens={max_tokens} min_tokens={min_tokens}"
    )
    start = time.perf_counter()
    results = await asyncio.gather(*(limited(p, n) for p, n in zip(prompts, lengths)))
    duration = time.perf_counter() - start

    if any(n is None for _, n, _, _ in results):
        print("No usage chunk from the server (M1?), re-tokenizing the outputs.")
        tokenizer = AutoTokenizer.from_pretrained(model)
        counts = [len(tokenizer(t, add_special_tokens=False).input_ids) for t, _, _, _ in results]
    else:
        counts = [n for _, n, _, _ in results]
    ttfts = sorted(t for _, _, t, _ in results if t is not None)
    latencies = [lat for _, _, _, lat in results]

    print(f"duration:            {duration:8.2f} s")
    print(f"output tokens:       {sum(counts):8d}")
    print(f"output throughput:   {sum(counts) / duration:8.1f} tok/s")
    print(f"request throughput:  {num_requests / duration:8.2f} req/s")
    print(f"TTFT mean / p50 / max: {statistics.mean(ttfts):.2f} / {statistics.median(ttfts):.2f} / {ttfts[-1]:.2f} s")
    print(f"latency mean:        {statistics.mean(latencies):8.2f} s")
    return {
        "concurrency": concurrency,
        "throughput": sum(counts) / duration,
        "ttft_mean": statistics.mean(ttfts),
        "latency_mean": statistics.mean(latencies),
    }


def main(
    port: Annotated[int, typer.Option(help="Port nano-serve is listening on.")] = 8000,
    num_requests: Annotated[int, typer.Option(help="Total requests to send.")] = 32,
    concurrency: Annotated[int, typer.Option(help="Most requests in flight at once.")] = 32,
    max_tokens: Annotated[int, typer.Option(help="max_tokens of every request.")] = 256,
    min_tokens: Annotated[
        Optional[int], typer.Option(help="Draw each request's max_tokens from [min_tokens, max_tokens].")
    ] = None,
):
    """Measure output throughput and TTFT of a running nano-serve server."""
    asyncio.run(run_bench(port, num_requests, concurrency, max_tokens, min_tokens))


if __name__ == "__main__":
    typer.run(main)
