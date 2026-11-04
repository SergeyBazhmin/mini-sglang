# nano-serve

A toy LLM serving engine built step by step to learn how SGLang works.
Each chapter rebuilds one SGLang idea in a small form. The docstrings point to the real SGLang code.

Each chapter is a self-contained folder that starts as a copy of the previous one,
so you can diff two chapters to see exactly what changed.

| chapter | topic |
|---|---|
| [`m1-request-lifecycle`](m1-request-lifecycle/) | processes, ZMQ ring, streaming, incremental detokenization, abort, stream interval, per-request timing |
| [`m2-continuous-batching`](m2-continuous-batching/) | prefill/decode batches, requests joining and leaving every step, batched sampling, decode-first baseline, stream interval |
| [`m3-kv-cache`](m3-kv-cache/) | preallocated token pool, free-list allocator, req-to-token table, attention over the pool, token budget, retraction, padded prefill, pages, lazy free |
| [`m4-radix-cache`](m4-radix-cache/) | radix tree over token ids, match/insert/split/evict, lock refs, slot ownership, prefix-aware budget and prefill, LFU, longest prefix first, in-batch dedup, keeping retracted work, faster matching, flush |

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/) and shared by all chapters.

```bash
uv sync                       # creates .venv from uv.lock
uv run m1-request-lifecycle/server.py --port 8000
```

Tests use pytest, one chapter at a time, because every chapter has its own `scheduler.py`
and so on. Tests that build or load a model only run with `-m model`.

```bash
uv run pytest m4-radix-cache                                    # no model needed
uv run pytest m4-radix-cache -m model --model Qwen/Qwen3-0.6B   # only those with a model
```
