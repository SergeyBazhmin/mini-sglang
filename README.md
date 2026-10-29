# nano-serve

A toy LLM serving engine built step by step to learn how SGLang works.
Each chapter rebuilds one SGLang idea in a small form. The docstrings point to the real SGLang code.

Each chapter is a self-contained folder that starts as a copy of the previous one,
so you can diff two chapters to see exactly what changed.

| chapter | topic |
|---|---|
| [`m1-request-lifecycle`](m1-request-lifecycle/) | processes, ZMQ ring, streaming, incremental detokenization |

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/) and shared by all chapters.

```bash
uv sync                       # creates .venv from uv.lock
uv run m1-request-lifecycle/server.py --port 8000
```
