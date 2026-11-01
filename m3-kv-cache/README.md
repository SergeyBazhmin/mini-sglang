# M3: KV cache memory management (nano-serve v2)

In M2 each request owned one growing tensor, so every step allocated, padded and
copied, and nothing stopped a burst of long requests from running out of memory.
M3 allocates all KV memory once at startup. After that, a decode step only moves
integers.

```
 req_pool_idx        position in the request
      |                   |
      v                   v
 req_to_token[row,        pos] = slot            ReqToTokenPool  (a table of ints)
                                  |
                                  v
                    k_buffer[layer][slot] = K    TokenToKVPool   (fixed buffers)
                    v_buffer[layer][slot] = V

 free_pages = [slots nobody owns]                TokenAllocator  (a free list)
```

| file | what changed since M2 | SGLang counterpart |
|---|---|---|
| `memory_pool.py` | **new.** `TokenAllocator`, `ReqToTokenPool`, `TokenToKVPool` | `mem_cache/memory_pool.py`, `mem_cache/allocator/token.py` |
| `attention.py` | **new.** `PoolAttention` (replaces `Qwen3Attention`), `ForwardBatch` | `layers/attention/torch_native_backend.py`, `model_executor/forward_batch_info.py` |
| `model_runner.py` | Flat batches with no padding, slot allocation, `free_req`, pool sizing | `managers/schedule_batch.py` (`prepare_for_extend`/`prepare_for_decode`), `mem_cache/common.py` (`release_kv_cache`), `model_executor/pool_configurator.py` |
| `scheduler.py` | Token budget at admission, oversized requests rejected or clamped, token usage in the logs | `managers/schedule_policy.py` (`PrefillAdder`), `managers/scheduler.py` |
| `server.py` | `--max-total-tokens`, `--mem-fraction-static`, `--context-length` | `server_args.py` |
| `test/test_memory_pool.py` | **new.** Allocator and table unit checks, no model needed | `test/registered/unit/mem_cache/` |
| `test/test_batching.py` | Compares with a separate HF model, requests leave at different steps, two passes over recycled slots, one retraction | `test/registered/...` accuracy tests |
| `test/test_scheduler.py`, `test/test_attention.py` | **new.** Retraction, page accounting, double frees and padded prefill, no download needed | `test/registered/unit/` |
| `prefill_cost.py`, `read_cost.py`, `alloc_cost.py` | **new.** The measurements for exercises 2 to 5 | |
| `common.py` | `ExerciseOptions` | |
| `client.py`, `bench.py`, `detokenizer.py` | unchanged (docstrings only) | |

`diff -r -x exercises m2-continuous-batching m3-kv-cache` shows all of it. This
chapter has its exercises built in, behind the switches listed under
[Exercises](#exercises). With no flags, the server reserves and allocates as
described below.

## Read in SGLang first

Paths are under `python/sglang/srt/`.

1. `mem_cache/allocator/token.py`, the whole file. It's short, and ours is a copy.
2. `mem_cache/memory_pool.py`: `ReqToTokenPool` (~379), `MHATokenToKVPool` (~2152).
   In `_create_buffers_normal` (~2546), skip the quantized cases.
3. `layers/attention/torch_native_backend.py`: `_run_sdpa_forward_extend` (~59),
   `_run_sdpa_forward_decode` (~174). `per_req_tokens` is the line that matters.
4. `managers/schedule_batch.py`: `prepare_for_decode` (~3743), then
   `check_decode_mem` (~3444) and `retract_decode` (~3463).
5. `mem_cache/allocator/paged.py`: `alloc_extend_naive` only.

Check yourself before reading on:

- `_run_sdpa_forward_decode` still slices K/V per request. Then why does SGLang's
  Triton backend need `req_to_token` at all, and what does it do differently?
- Why is slot 0 never handed out, and why does `ReqToTokenPool` have a padding
  row? (Ours has the slot but not the row. M6 explains the difference.)
- Why does the decode step allocate a slot for the token that was sampled in the
  *previous* step, and not for the one it's about to sample?

## Key ideas

- **Two indirections.** The table maps a request's positions to slots, and slots
  can be anywhere. A request's K/V doesn't have to be contiguous, so nothing
  moves, and any free slot serves any request.
- **Freeing is cheap.** `free_req` gives integers back. It never touches K/V, and
  a slot is reused by overwriting it. M4 relies on this: slots can outlive the
  request that wrote them, because a cached prefix is just slots nobody freed.
- **Flat batches.** Embeddings, MLPs, norms and rope don't care which request a
  token belongs to, so a batch is one sequence. Only attention needs to know, and
  `ForwardBatch` tells it. This removes the left and right padding of M2.
- **Reservation at admission.** The pool is fixed, so running out is now a
  scheduling problem, not a crash. A request starts only if
  `prompt + max_new_tokens - 1` fits in `free - (what running requests may still need)`.
  So `_prepare_decode`'s allocation can't fail, and a request waits in the queue
  instead of taking the server down.

## What did NOT get better

The M2 notes say M3 removes the copying. That's only half true:

| per decode step | M2 | M3 |
|---|---|---|
| write new K/V | re-copy each request's whole cache (`torch.cat`) | one row per request |
| read K/V | pad everything, then HF's `DynamicCache` copies it again | **gather everything, once per layer** |

`self.k_buffer[slots]` in `_run_decode` is still a read of the whole history into
a temporary tensor. Reads stay O(context) per step, as they must, because attention
looks at every cached token. But the *copy* is avoidable: Triton and friends read
the pool in place through the index. That's M5. Expect M3 to beat M2 clearly on
memory behaviour, and by a smaller margin on throughput.

M3 can also be **slower** than M2 at prefill when many requests arrive together,
because `_run_extend` runs one attention per request per layer (`--batched-prefill`
is the fix, see exercise 2).

## Run (GPU machine)

From the repo root:

```bash
uv run m3-kv-cache/test/test_memory_pool.py
uv run m3-kv-cache/test/test_scheduler.py
uv run m3-kv-cache/test/test_attention.py
uv run m3-kv-cache/test/test_batching.py --model Qwen/Qwen3-0.6B
uv run m3-kv-cache/test/test_batching.py --model Qwen/Qwen3-0.6B --batched-prefill --page-size 64 --lazy-free
uv run m3-kv-cache/server.py --model Qwen/Qwen3-0.6B --port 8000
# another shell:
uv run m3-kv-cache/client.py --port 8000
uv run m3-kv-cache/bench.py --port 8000 --num-requests 200 --concurrency 32

# Backpressure: a pool too small for all of them at once. Requests must queue, not crash.
uv run m3-kv-cache/server.py --model Qwen/Qwen3-0.6B --port 8000 --max-total-tokens 2048
uv run m3-kv-cache/bench.py --port 8000 --num-requests 32 --concurrency 32
```

Qwen3 only, for now: `PoolAttention` reuses HF's Qwen3 projections and norms.

## Done when
1. `test/test_memory_pool.py`, `test/test_scheduler.py` and `test/test_attention.py` pass.
2. `test/test_batching.py`, with and without the exercise flags, matches on every
   prompt in both passes (exactly on CPU; on
   GPU bf16, divergence late in a sequence is expected, see its docstring),
   reports all slots free, and memory did not grow.
3. `client.py` passes as in M1 and M2.
4. In the scheduler log, `gpu mem` stays flat through `bench.py` with 200 requests,
   and `token usage` returns to 0.00 once they finish.
5. The small-pool run finishes all 32 requests, with `#queue-req` above 0 in the log.
6. M2 vs M3 tokens/s, at `--max-tokens 256` and `--max-tokens 1024`. Write down
   all four numbers.

## Exercises

All six are done and built into this chapter. Each section gives the task, then
the solution and how to measure it.

| exercise | switch | changed | measured with |
|---|---|---|---|
| 1. Overcommit and retract | `--new-token-ratio R` | `Req.remaining_kv_tokens`, `Scheduler.update_running_batch`, `ModelRunner._prepare_extend` | `bench.py` and the scheduler log |
| 2. Faster prefill | `--batched-prefill` | `PoolAttention._run_extend_batched` | `prefill_cost.py` |
| 3. Measure the read copy | none (standalone) | `GatherTimer` in `attention.py` | `read_cost.py` |
| 4. Pages | `--page-size N` | `TokenAllocator`, `ModelRunner._prepare_decode`, `decode_slots_needed` | `alloc_cost.py` and `page waste` in the log |
| 5. Allocator cost | `--lazy-free` | `TokenAllocator.free`, `merge_released` | `alloc_cost.py` |
| 6. Double free | `--check-double-free` | `ModelRunner.free_req`, `TokenAllocator._mark_free` | `test/test_memory_pool.py`, `test/test_scheduler.py` |

### 1. Overcommit and retract

*Task: our reservation is safe but wasteful, because most requests hit EOS early.
Admit with `max_new_tokens * ratio`, for `ratio` around 0.5. Before each decode,
check that the step fits. If not, push the request with the most remaining work
back to the queue and free its slots. It must re-prefill `input_ids + output_ids`.
Compare token usage and throughput. Read `update_running_batch` (scheduler.py
~4550) and `new_token_ratio_tracker.py` first.*

```bash
# Run with --new-token-ratio 1, then 0.5, then 0.2.
uv run m3-kv-cache/server.py --model Qwen/Qwen3-0.6B --port 8000 \
    --max-total-tokens 16384 --new-token-ratio 0.5
uv run m3-kv-cache/bench.py --port 8000 --num-requests 64 --concurrency 32 --max-tokens 4096
```

`--max-tokens` must be well above the length of a real answer, or there's no waste
to win back. At ratio 1 each request reserves about 4100 slots, so about 4 run at
once. Expect more `#running-req`, higher `token usage` and higher throughput at
0.5. At 0.2, expect `#retracted` to climb and throughput to fall again, because
each retraction repeats a whole prefill.

How it works: a retracted request keeps `output_ids` and `num_sent`, so the client
never notices and no token is streamed twice. `_prepare_extend` prefills
`input_ids + output_ids`, newest token included, so the token it samples is the
one the next decode step would have sampled, and `kv_allocated_len` ends up with
the usual meaning. A retracted request goes to the head of the queue.

Retraction can't livelock: output is never thrown away, and every forward adds a
token to at least one request. It can thrash, though, which is what the ratio
tracker is for.

Two cases this adds to `abort_request` and `update_running_batch`: a retracted
request in the queue has detokenizer state, so an abort must send a final message,
and an aborted request that gets picked for retraction finishes instead of going
back to the queue.

Differences from SGLang:

- **The ratio is fixed here.** SGLang's `NewTokenRatioTracker` starts at
  `init` (0.7 by default), decays on every step without retraction, and jumps to
  an estimate from the retracted requests when retraction happens. Good next step.
- **Who gets retracted differs.** We follow the exercise and pick the most
  remaining work. Read `_get_decode_retraction_order` in `schedule_batch.py`, which
  sorts by output and prompt length, and has a priority mode.
- **SGLang doesn't retract when a finishing request is about to free KV.** That
  only matters with the overlap loop (M6), where results arrive a step late.

### 2. Faster prefill

*Task: vectorize `_run_extend` the way `_run_decode` works, with a padded gather
and a mask combining "causal" and "within this request". Measure a burst of 32 new
requests, before and after.*

```bash
uv run m3-kv-cache/prefill_cost.py --model Qwen/Qwen3-0.6B
```

Expect a clear win on even prompts. In the skewed row, one long prompt makes the
rectangle `batch * max_len` and the mask `batch * max_len^2`, so most of the win
should go. Try `--prompt-len 64` and `--prompt-len 1024` too. Real kernels get the
same gain without padding, which is M5.

Results may differ from the loop by floating-point noise, because a different
batch shape can take a different code path. `test/test_attention.py` allows for that.

### 3. Measure the read copy

*Task: time the two gathers in `_run_decode` against the whole step, at context
lengths 128, 1024 and 4096.*

```bash
uv run m3-kv-cache/read_cost.py --model Qwen/Qwen3-0.6B
```

Gather time should grow linearly with context, since `MB read` does. Its share of
the step tells you how much M5 can win. It may grow more slowly than you expect,
because attention itself also reads every cached token: that part stays after M5.

### 4. Pages

*Task: make the allocator hand out 64 slots at a time, so a request only needs a
new page every 64 tokens. What share of slots goes unused on average? What gets
faster? Read `alloc_extend_naive` first.*

```bash
uv run m3-kv-cache/alloc_cost.py
# In a server run, watch `page waste` in the Decode batch lines:
uv run m3-kv-cache/server.py --model Qwen/Qwen3-0.6B --port 8000 --page-size 64
```

**Unused slots:** half a page per request on average, so a share of
`32 / (length + 32)` at page size 64: about 3% at 1000 tokens and 11% at 256.

**What gets faster:** the allocator. A request calls it once per page instead of
once per token, and the free list is 64 times shorter, so `torch.cat` in `free`
copies 64 times less. Attention does **not** get faster here, because our gather
reads through the table either way. The real payoff is that a request's K/V is in
contiguous runs, which is what SGLang's kernels depend on. That comes with M5.

How it works: a page is named by its first slot. Inside a page the next slot is
the last slot plus one, so a decode step only reaches the allocator at a boundary.
`free` keeps only each page's first slot from the table row. Three things had to
change because of pages: budgets are in whole pages (`remaining_kv_tokens`), the
pool size is rounded down to whole pages, and the check before decode asks how
many requests cross a boundary (`decode_slots_needed`).

SGLang's `alloc_extend_naive` also handles a request that starts mid-page, which
matters once a cached prefix ends there (M4). Ours always starts a fresh row at
position 0.

### 5. Allocator cost

*Task: `free` copies the whole free list. Replace it with SGLang's `release_pages`
idea: buffer frees, and merge only when `alloc` runs short.*

```bash
uv run m3-kv-cache/alloc_cost.py
```

Compare `eager` and `lazy` at page size 1. Eager frees copy the whole free list,
so their cost grows with `--pool-size`. Lazy frees append to a Python list, and a
merge copies everything once in a while. Try `--pool-size 100000` as well, then
`--device cuda`.

Don't expect this to move `bench.py`. It's small next to a model step, unless the
pool is huge and requests are short.

Differences from SGLang: its `release_pages` is the sort path. It exists so that
`merge_and_sort_free` can **sort** the free list and hand out contiguous slots
again, not to save a copy, and its base `free` still runs `torch.cat` on the
buffer. The paged allocator's `staged_pages` is closer to ours. We don't sort, so
over time a request's slots scatter. That costs us nothing with page size 1, but
try adding the sort and measuring it.

### 6. Double free

*Task: call `free_req` twice on the same request in a test. What happens, and what
breaks later? Add a check that catches it at the first call.*

```bash
uv run m3-kv-cache/test/test_memory_pool.py
uv run m3-kv-cache/test/test_scheduler.py
```

**What happened before the check.** `free_req` clears `req_pool_idx`, so a second
call indexed the table with `None` and should have crashed on `torch.cat` with a
shape error far from the cause. That's the lucky case. The dangerous one is a
double free of the same slots by any other route, such as a stale copy of a
request. The slots enter the free list twice, so two later requests get the same
slot. Each one overwrites the other's K/V, nothing crashes, and the only symptom
is wrong text. The same goes for a table row freed twice.

**The check, in two parts.** `free_req` raises at once if the request holds no KV.
That's cheap, so it's always on. `TokenAllocator` can also keep a free/used bit per
page and raise on any repeat, including a repeat inside one call and a free of
slot 0. That's a lookup on every free, so it's `--check-double-free`, and always on
in the tests.

## Not run yet

None of this has been executed, not the chapter and not the tests. It was only
checked by compiling it and by reading the installed `torch` and `transformers`
sources. All expected results above are predictions, not measurements. Write your
numbers here.
