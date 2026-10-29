"""Checks for M1, using the official OpenAI client.

    uv run m1-request-lifecycle/client.py --port 8000

1. Streaming works.
2. Non-ASCII output streams without the "�" character.
3. Two concurrent requests both finish. In v0 the second one waits for the
   first; look at the timings. M2 (continuous batching) fixes this.
"""

import argparse
import asyncio
import time

from openai import AsyncOpenAI

# Qwen3 thinks by default. Turn it off so the outputs are short and easy to read.
EXTRA = {"chat_template_kwargs": {"enable_thinking": False}}


async def stream_one(client, model, prompt, max_tokens=200, echo=True):
    start = time.perf_counter()
    first_token_at = None
    text = []
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.8,
        stream=True,
        extra_body=EXTRA,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            if first_token_at is None:
                first_token_at = time.perf_counter() - start
            text.append(delta)
            if echo:
                print(delta, end="", flush=True)
    if echo:
        print()
    return "".join(text), first_token_at, time.perf_counter() - start


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    client = AsyncOpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="EMPTY")
    model = (await client.models.list()).data[0].id

    print("=== 1. streaming ===")
    await stream_one(client, model, "Explain what a KV cache is in two sentences.")

    print("\n=== 2. non-ASCII ===")
    text, _, _ = await stream_one(
        client, model, "Write one sentence in Russian and one in Japanese."
    )
    print("PASS: no U+FFFD" if "�" not in text else "FAIL: found U+FFFD")

    print("\n=== 3. two concurrent requests ===")
    prompts = ["Count from 1 to 30.", "Name 10 colors."]
    results = await asyncio.gather(
        *(stream_one(client, model, p, echo=False) for p in prompts)
    )
    for p, (_, ttft, total) in zip(prompts, results):
        print(f"{p!r:28} TTFT={ttft:.2f}s total={total:.2f}s")
    print("In v0, one request's TTFT is about the other's total time. M2 fixes that.")


if __name__ == "__main__":
    asyncio.run(main())
