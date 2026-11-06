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

## How the code works

### Startup (`server.py: main`)

1. `PortArgs.new()` makes a temp dir and three `ipc://` addresses, one per
   *receiving* process. Receivers bind, senders connect (`get_socket`). High-water
   marks are 0, so ZMQ never drops a message.
2. The scheduler and detokenizer start as `spawn` processes (a forked child can't
   initialize CUDA). `main` blocks on a pipe until `run_scheduler_process` sends
   `"ready"`, i.e. until the model is loaded, or a traceback if loading failed.
3. `TokenizerManager` is built, then uvicorn serves `build_app(...)`. The app's
   `lifespan` starts `handle_loop` as a background task.

### One request

**Main process, in.** `chat_completions` calls `tokenize_chat`, which renders the
chat template to text and then tokenizes it, so the engine only ever sees token
ids. `generate_request` is an async generator:

- it makes a `rid` (the only key all three processes share) and a `ReqState`
  with an `asyncio.Event` and an `out_list`,
- sends a `TokenizedGenerateReq` to the scheduler,
- then loops: wait on the event, take `out_list`, yield each output, return once
  `state.finished`.

For `stream: true` the generator is wrapped by `_stream_chunks`, which turns
each output into an OpenAI SSE chunk. Otherwise the endpoint drains it into one
JSON response.

**Scheduler process.** `event_loop` is the shape every later chapter keeps:

```
ingest_requests  ->  pick work  ->  run_step  ->  stream_output
```

- `ingest_requests(block=idle)` blocks on the socket only when there's nothing
  to run, then drains everything pending with `NOBLOCK`. A `TokenizedGenerateReq`
  becomes a `Req` in `waiting_queue`. An `AbortReq` goes to `abort_request`.
- Pick work: if nothing is running, pop the queue head into `running_req`.
  One request at a time.
- `run_step` is one forward pass. If `req.past_key_values` is `None` it's
  **prefill**: the whole prompt goes in. Otherwise it's **decode**: only
  `output_ids[-1]` goes in, and HF's cache supplies the rest. `sample` picks a
  token (greedy at temperature 0, otherwise temperature + top-p), it's appended
  to `output_ids`, and `check_finished` sets `finish_reason` to `stop` (EOS) or
  `length`. A finished request drops its cache at once.
- `stream_output` sends `output_ids[num_sent:]` as a `BatchTokenIDOut` (batch-shaped
  already, with one entry). With `--stream-interval N` it skips sending until N
  tokens are pending, except for the first token and the final message. The
  first message also carries the last 5 prompt tokens (`prompt_tails`) as
  decoding context.

**Detokenizer process.** `handle_batch_token_id_out` keeps a `DecodeStatus` per
`rid`, created on the first message and seeded with the prompt tail:

- `decode_ids[surr_offset:read_offset]` is text already sent, kept only as
  context. It decodes that window and the window plus the new ids, and the
  difference is the new text.
- Clean new text is sent and the window slides forward.
- New text ending in `�` means a UTF-8 character is split across tokens. The
  offsets stay put so the next token retries, and only `printable_prefix` (up to
  the last space, newline or CJK character) goes out. `sent_len` remembers how
  much of it was sent.
- On the final message it flushes everything and deletes the state.

**Main process, out.** `handle_loop` is one coroutine for all requests. For each
entry of a `BatchStrOut` it appends to that rid's `out_list`, marks it finished
if there's a `finish_reason`, and sets the event, which wakes `generate_request`.
On the way out `generate_request` calls `log_timing` (tokenize time, TTFT, decode
tok/s).

### Abort

If the client disconnects, Starlette cancels the response. `_stream_chunks`'s
`finally` closes the inner generator at once, and `generate_request`'s `finally`
sees `not state.finished` and sends `AbortReq`. In the scheduler,
`abort_request` removes a waiting request outright (the detokenizer never saw
it), or sets `finish_reason = "abort"` on the running one. The loop then skips
`run_step` for it, and `stream_output` sends the final message so the
detokenizer frees its state.

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
