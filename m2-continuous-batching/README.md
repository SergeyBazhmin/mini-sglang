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
| `model_runner.py` | **new.** Padded batched prefill and decode over the HF model, batched sampling, `DecodeTimer` | `model_executor/model_runner.py`, `layers/sampler.py`, `layers/logits_processor.py` |
| `server.py` | `--max-running-requests`, `--max-prefill-tokens`, `--decode-first`, `--stream-interval`, `stream_options.include_usage` | `server_args.py` |
| `bench.py` | **new.** Async load generator, `--min-tokens` for mixed lengths | `python -m sglang.bench_serving` |
| `tests/test_batching.py` | **new.** Batched greedy output vs HF `generate` | `test/registered/...` accuracy tests |
| `tests/test_scheduler.py` | **new.** Decode-first and `stream_interval`, no model needed | |
| `sweep.py`, `copy_cost.py` | **new.** Measurements, see [Experiments](#experiments) | |
| `common.py` | `SchedulerOptions` | |
| `client.py`, `detokenizer.py` | unchanged (docstrings only) | |

`diff -r m1-request-lifecycle m2-continuous-batching` shows all of it. With no flags,
the server does plain continuous batching, prefill first.

## Key ideas

- **Two kinds of step.** EXTEND (prefill) processes whole prompts and is
  compute-bound. DECODE processes one token per request and is memory-bound:
  reading the weights costs the same for 1 request or 64, which is why batching
  decode is nearly free throughput.
- **Prefill first.** If new requests fit, the scheduler prefills them before the
  next decode step. New requests get a low TTFT; running requests see a short
  pause. SGLang does the same by default.
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
uv run pytest m2-continuous-batching                                    # no model needed
uv run pytest m2-continuous-batching -m model --model Qwen/Qwen3-0.6B
uv run m2-continuous-batching/server.py --model Qwen/Qwen3-0.6B --port 8000
# another shell:
uv run m2-continuous-batching/client.py --port 8000
uv run m2-continuous-batching/bench.py --port 8000 --num-requests 32 --concurrency 32
```

For the baseline, start the M1 server and run the same `bench.py` against it.
On M1, expect it to take about 32 times one request's latency.

## Done when
1. `tests/test_scheduler.py` passes, and `tests/test_batching.py` matches on every prompt
   (exactly on CPU; on GPU bf16, divergence late in a sequence is expected, see
   the test's docstring).
2. `client.py` passes as in M1, and both concurrent requests in check 3 have a short TTFT.
3. `bench.py` with 32 concurrent requests shows output tokens/s many times
   higher than M1. Write both numbers down.
4. The scheduler log shows `Prefill batch` lines with `#new-seq` > 1, and
   `Decode batch` lines with `#running-req` going up and down.

## Experiments

Every server below is started as above, plus the flags named in each section:

```bash
uv run m2-continuous-batching/server.py --model Qwen/Qwen3-0.6B --port 8000 [flags]
```

| experiment | switch | code | measured with |
|---|---|---|---|
| Scaling curve | none | `bench.py` returns its numbers | `sweep.py` |
| Decode first | `--decode-first` | `Scheduler.get_next_batch_to_run` | `bench.py --min-tokens` |
| Cost of copying | none (standalone) | `DecodeTimer` in `model_runner.py` | `copy_cost.py` |
| `stream_interval` | `--stream-interval N` | `Scheduler.stream_output` | `bench.py` |

### Scaling curve

Run `bench.py` with concurrency 1, 2, 4, …, 64 and plot tokens/s and mean latency.

```bash
uv run --with matplotlib m2-continuous-batching/sweep.py --port 8000
```

Writes `sweep.csv` and `sweep.png`. Throughput should rise almost linearly while
a decode step costs about the same at any batch size. It bends once the step time
grows as fast as the batch does. The likely causes, in the order to check them:
compute starts to dominate memory, the copying measured below grows with
`batch × context`, or the Python processes around the GPU saturate. Check the
last one with `top` while the top of the curve runs.

### Decode first

`--decode-first` decodes whenever anything is running and prefills only when the
batch is empty. This is "static batching" in all but name, kept as a baseline.

```bash
# Run against a default server, then a --decode-first server.
uv run m2-continuous-batching/bench.py --port 8000 \
    --num-requests 96 --concurrency 32 --min-tokens 32 --max-tokens 512
```

**The lengths must differ.** With one fixed `max_tokens`, all 32 requests finish
on the same step, the batch empties, and decode-first looks just as good. That is
why `bench.py` has `--min-tokens`. Requests must also outnumber the concurrency,
so that new ones arrive while others run.

With decode-first, each finished request leaves an empty seat until the longest
request in the batch ends. Expect lower throughput and a much higher TTFT.

### The cost of copying

`DecodeTimer` splits `forward_decode` into the model call and the copying around
it (`pad_sequence` + `torch.cat`). This is the motivation for M3.

```bash
uv run m2-continuous-batching/copy_cost.py --model Qwen/Qwen3-0.6B
```

Prints time per step split into `pad`, `model` and `cat`, as the context grows.
It calls `ModelRunner` directly so that EOS can't end a run early. `pad` and `cat`
should grow with the context, and so should the copy share.

The share is a lower bound: HF's `DynamicLayer.update` runs another `torch.cat` on
every layer inside the model call. The timer calls `torch.cuda.synchronize()`,
which is only on in this script, so the numbers are slightly slower than a real run.

### `stream_interval`

With 64 running requests, the scheduler sends a message every step.
`--stream-interval N` sends a request's output only every N tokens (SGLang:
`--stream-interval`).

```bash
# Run against servers with --stream-interval 1, 4 and 16.
uv run m2-continuous-batching/bench.py --port 8000 --num-requests 64 --concurrency 64
```

The rule matches SGLang's `OutputStreamer`: the first token goes out at once so
that TTFT doesn't suffer, then a request is sent once N tokens are unsent, and a
finished request always flushes.

Don't expect a big win on a small model. The scheduler only pickles a message and
hands it to ZMQ, and the slow work (detokenizing, HTTP, the client) runs in other
processes. It matters when those processes can't keep up: watch `top`, and compare
how much text is still arriving after the scheduler log goes quiet. Also note that
requests are out of phase, so the scheduler may still send a message most steps,
each one just carries fewer requests.

## Not run yet

The experiments above were only checked by compiling them, by the model-free tests
and by reading the installed `transformers` sources. The expected results are
predictions, not measurements. Write your numbers here.
