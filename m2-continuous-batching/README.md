# M2: continuous batching (nano-serve v1)

M1 ran one request at a time, so a GPU that could decode dozens of sequences
per step decoded one. M2 runs every active request in each forward pass, and
lets requests join and leave the batch at every step.

```
 loop:  ingest ──> get_next_batch_to_run ──> run_batch ──> process_batch_result
                    │                          │             │
                    │ new requests fit?        │ ModelRunner │ append tokens, stream ONE
                    │   yes: EXTEND batch      │ .forward    │ message for the whole batch,
                    │   no:  DECODE running    │             │ drop finished, merge prefilled
```

| file | what changed since M1 | SGLang counterpart |
|---|---|---|
| `scheduler.py` | `ScheduleBatch`, `ForwardMode`, `get_next_batch_to_run`, `get_new_batch_prefill`, `process_batch_result` | `managers/scheduler.py`, `managers/schedule_batch.py`, `managers/schedule_policy.py` (`PrefillAdder`) |
| `model_runner.py` | **new.** Padded batched prefill and decode over the HF model, batched sampling | `model_executor/model_runner.py`, `layers/sampler.py`, `layers/logits_processor.py` |
| `server.py` | `--max-running-requests`, `--max-prefill-tokens`, `stream_options.include_usage` | `server_args.py` |
| `bench.py` | **new.** Async load generator | `python -m sglang.bench_serving` |
| `test_batching.py` | **new.** Batched greedy output vs HF `generate` | `test/registered/...` accuracy tests |
| `client.py`, `common.py`, `detokenizer.py` | unchanged (docstrings only) | |

`diff -r m1-request-lifecycle m2-continuous-batching` shows all of it.

## Key ideas

- **Two kinds of step.** EXTEND (prefill) processes whole prompts and is
  compute-bound. DECODE processes one token per request and is memory-bound:
  reading the weights costs the same for 1 request or 64, which is why batching
  decode is nearly free throughput.
- **Prefill first.** If new requests fit, the scheduler prefills them before the
  next decode step. New requests get a low TTFT; running requests see a short
  pause. SGLang does the same by default, and chunked prefill keeps the pause short.
- **Admission limits.** `max_running_requests` caps the decode batch, and
  `max_prefill_tokens` caps the prompt tokens in one prefill. There is no memory
  pool yet, so too many long requests can still run out of memory. M3 fixes that.
- **Padding is the toy part.** HF's model wants rectangular tensors, so prefill
  pads prompts on the right and decode pads KV caches on the left, then copies
  every request's K/V in and out each step. SGLang never pads: it flattens all
  tokens into one sequence and gives the attention kernel per-request offsets
  and KV slot indices. That needs our own KV pool (M3) and attention (M5).

## Run (GPU machine)

From the repo root:

```bash
uv run m2-continuous-batching/test_batching.py --model Qwen/Qwen3-0.6B
uv run m2-continuous-batching/server.py --model Qwen/Qwen3-0.6B --port 8000
# another shell:
uv run m2-continuous-batching/client.py --port 8000
uv run m2-continuous-batching/bench.py --port 8000 --num-requests 32 --concurrency 32
```

For the baseline, start the M1 server and run the same `bench.py` against it.
On M1, expect it to take about 32 times one request's latency.

## Done when
1. `test_batching.py` matches on every prompt (exactly on CPU; on GPU bf16,
   divergence late in a sequence is expected, see the test's docstring).
2. `client.py` passes as in M1, and both concurrent requests in check 3 have a short TTFT.
3. `bench.py` with 32 concurrent requests shows output tokens/s many times
   higher than M1. Write both numbers down.
4. The scheduler log shows `Prefill batch` lines with `#new-seq` > 1, and
   `Decode batch` lines with `#running-req` going up and down.

## Exercises
1. **Scaling curve.** Run `bench.py` with concurrency 1, 2, 4, …, 64. Plot
   tokens/s and mean latency. Where does throughput stop growing, and why?
2. **Decode first.** Change `get_next_batch_to_run` to decode whenever anything
   is running and prefill only when the batch is empty. Compare TTFT and
   throughput. This is "static batching" in all but name.
3. **The cost of copying.** Time `forward_decode` and split it into the model
   call vs `pad_sequence` + `torch.cat`. Grow `max_tokens` to 1024 and watch
   the copy share rise. This is the motivation for M3.
4. **Chunked prefill.** Send one request with a 4000-token prompt while 16
   others are decoding. Watch their streams stall. Then split long prompts
   into chunks of `--chunked-prefill-size` tokens over several EXTEND steps.
   (You need to carry the partial KV across steps; `forward_extend` must accept
   requests that already have some KV.) Read `Scheduler.chunked_req` in SGLang first.
5. **`stream_interval`.** With 64 running requests, the scheduler now sends a
   message every step. Send only every N tokens and measure the effect on
   throughput (SGLang: `--stream-interval`).
