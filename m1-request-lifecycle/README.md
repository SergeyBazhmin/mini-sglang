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
| `server.py` | HTTP, chat template, tokenize, one `ReqState` + `asyncio.Event` per request, a single `handle_loop` | `entrypoints/http_server.py`, `entrypoints/openai/serving_chat.py`, `managers/tokenizer_manager.py` |
| `scheduler.py` | loop: ingest → pick → forward → stream; prefill then decode; sampling; abort | `managers/scheduler.py`, `managers/schedule_batch.py` (`Req`), `scheduler_components/output_streamer.py` |
| `detokenizer.py` | incremental decoding with `surr_offset`/`read_offset` windows | `managers/detokenizer_manager.py` |
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

## Exercises (modify the code)
1. **Prove the detokenizer matters.** In `detokenizer.py`, replace the window logic with `self._decode(new_ids)`. Run test 2 again and watch for `�` and broken spacing.
2. **Abort.** Start a long streaming request, then press Ctrl-C in the client. Add a log line in `Scheduler.abort_request` to confirm it's called. Which process noticed the disconnect first?
3. **`stream_interval`.** Make the scheduler send output only every N tokens, or when the request finishes. Measure the effect on the number of ZMQ messages, and on how smooth the stream looks.
4. **Printable prefix.** When `new_text` ends in `�`, SGLang still emits the clean part before it. Implement this; read `detokenizer_manager.py` lines ~395–420 first.
5. **Timing.** Log tokenize time, time to the first token, and decode tokens/s per request. Keep these numbers: they're the baseline for M2.
