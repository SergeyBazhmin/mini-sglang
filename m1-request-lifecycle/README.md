# M1: the request lifecycle (nano-serve v0)

Three processes, connected in a ring by ZMQ, the same way SGLang does it:

```
 main process                         Scheduler process        Detokenizer process
 FastAPI + TokenizerManager --PUSH--> model + gen loop --PUSH--> token ids -> text
          ^                                                             |
          +---------------------------------PULL------------------------+
```

| file | role | SGLang counterpart |
|---|---|---|
| `server.py` | HTTP, chat template, tokenize, one `ReqState` + `asyncio.Event` per request, a single `handle_loop`, per-request timing | `entrypoints/http_server.py`, `entrypoints/openai/serving_chat.py`, `managers/tokenizer_manager.py` |
| `scheduler.py` | loop: ingest → pick → forward → stream; prefill then decode; sampling; abort; `stream_interval` | `managers/scheduler.py`, `managers/schedule_batch.py` (`Req`), `scheduler_components/output_streamer.py` |
| `detokenizer.py` | incremental decoding with `surr_offset`/`read_offset` windows, printable prefix | `managers/detokenizer_manager.py` |
| `common.py` | message dataclasses, socket helper | `managers/io_struct.py`, `utils.get_zmq_socket` |

What v0 does **not** do yet:
- **No batching.** The scheduler finishes one request before starting the next. That comes in M2.
- **The KV cache is HF's `DynamicCache`, one per request.** Our own memory pool comes in M3.

## Run (GPU machine)

From the repo root (`uv sync` once, see the root README):

```bash
uv run m1-request-lifecycle/server.py --model Qwen/Qwen3-0.6B --port 8000
# another shell:
uv run m1-request-lifecycle/client.py --port 8000
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":50,"chat_template_kwargs":{"enable_thinking":false}}'
```

It also runs on CPU, slowly (float32).

## Done when
1. `client.py` streams a coherent answer.
2. The non-ASCII test prints `PASS`.
3. Both concurrent requests finish, and the timings show the second one waiting for the first.

## Details worth a closer look

1. **The detokenizer window matters.** Start the server with `--naive-detokenizer`,
   which decodes only the new ids each time, and run `client.py` again. Check 2 now
   fails: characters split across tokens come out as `�`, and spacing can break.
2. **Abort.** Start a long streaming request, then press Ctrl-C in the client. The
   main process notices first: Starlette cancels the response, so
   `TokenizerManager.generate_request` logs `Client disconnected` and sends `AbortReq`.
   Then the scheduler logs `Abort ...: stopped after N tokens` from `abort_request`.
3. **`stream_interval`.** `--stream-interval N` sends a request's output only every N
   tokens. The first token still goes out at once so that TTFT doesn't suffer, and a
   finished request always flushes. The scheduler logs `N tokens in M messages` for
   every finished request, so compare M at 1, 4 and 16. The stream gets choppier as
   N grows, since text arrives in bursts of N tokens.
4. **Printable prefix.** When the window's new text ends in `�`, the clean part before
   it still goes out, cut back to the last space, newline or CJK character so that it
   can't change later (SGLang: `find_printable_text`, `detokenizer_manager.py`
   ~395–420). The rest waits for the next token.
5. **Timing.** Every finished request logs its tokenize time, time to first token and
   decode tokens/s. Keep these numbers: they're the baseline for M2.

```bash
uv run m1-request-lifecycle/server.py --model Qwen/Qwen3-0.6B --port 8000 --stream-interval 4
uv run m1-request-lifecycle/server.py --model Qwen/Qwen3-0.6B --port 8000 --naive-detokenizer
```
