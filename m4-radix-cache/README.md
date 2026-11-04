# M4: Radix prefix cache (nano-serve v3)

In M3 a request's K/V was thrown away when it finished, and the next request with
the same system prompt computed it all again. Attention is causal, so the K/V of a
token depends only on the tokens up to it. Two requests that start the same way have
the same K/V there, so the second one can read the first one's.

M3 already did the hard part: a request's K/V is just slots, and freeing is just
giving integers back. M4 keeps some of those integers in a tree instead.

```
 root
  └── [You are a helpful bot.]        value=[ 3,  4,  5,  6,  7,  8]  lock_ref=2
       ├── [ What is 2+2?]            value=[ 9, 10, 11, 12, 13, 14]  lock_ref=1
       └── [ Tell me a joke.]         value=[21, 22, 23, 24, 25, 26]  lock_ref=1

 req_to_token[row of "What is 2+2?"] = 3 4 5 6 7 8 | 9 10 11 12 13 14 | 40 41 ...
                                       shared        from the tree     its own
                                                                       ^ cache_protected_len
```

| file | what changed since M3 | SGLang counterpart |
|---|---|---|
| `radix_cache.py` | **new.** `TreeNode`, `RadixCache`: match, insert, split, evict, locks, `checkpoint` | `mem_cache/radix_cache.py`, `mem_cache/evict_policy.py` |
| `attention.py` | Extend runs only the uncached tail, over the whole sequence: `prefix_lens` | `layers/attention/torch_native_backend.py` |
| `model_runner.py` | Prefix in `_prepare_extend`, `ensure_free`, `free_req` hands K/V to the tree | `mem_cache/common.py` (`release_kv_cache`, `evict_from_tree_cache`), `mem_cache/allocation.py` |
| `scheduler.py` | Match and lock at admission, evictable slots in the budget, checkpoint after prefill, `flush_cache`, hit rate in the logs | `managers/schedule_policy.py` (`PrefillAdder`), `managers/scheduler.py` |
| `server.py` | `--disable-radix-cache`, cache policy flags, `POST /flush_cache` | `server_args.py`, `entrypoints/http_server.py` |
| `fake_runner.py` | **new.** Slot logic without a model, which checks every K/V read. Moved out of `test_scheduler.py` | |
| `tests/test_radix_cache.py` | **new.** Tree unit tests, and random operations against a brute-force model | `test/registered/unit/mem_cache/test_radix_cache_unit.py` |
| `tests/test_scheduler.py` | Cache tests, a stress test over all the options | |
| `tests/test_attention.py` | Running the tail must give what running everything gives | |
| `tests/test_prefix_output.py` | **new.** Tiny random Qwen3 against HF, token for token, no download | |
| `tests/test_batching.py` | Prompts share a system prompt, and pass 2 runs mostly from cache | |
| `prefix_bench.py`, `cache_sim.py`, `match_cost.py` | **new.** The measurements | |
| `common.py` | `EngineOptions`, `FlushCacheReq` | |

`diff -r m3-kv-cache m4-radix-cache` shows all of it. The cache policies are
switches, listed under [Cache policies](#cache-policies). With no flags, the server
has the cache on, with LRU eviction and first-come scheduling.

## Read in SGLang first

Paths are under `python/sglang/srt/`.

1. `mem_cache/radix_cache.py`: `TreeNode` (~250), `_match_prefix_helper` (~651),
   `_split_node` (~677), `_insert_helper` (~705), `evict` (~552),
   `inc_lock_ref` and `dec_lock_ref` (~582). Skip the bigram and `kv_events` parts.
2. The same file: `_insert_cache` (~450) and `checkpoint` (~510). This is where
   ownership moves, and it's the part to read twice.
3. `mem_cache/common.py`: `match_kv_cache` (~222), `checkpoint_kv_cache` (~272),
   `release_kv_cache` (~419).
4. `managers/schedule_batch.py`: `_compute_max_prefix_len` (~1778).
5. `managers/schedule_policy.py`: `add_one_req` (~1358) up to the first `_lock_node`,
   and `_lock_node` itself (~1188).
6. `layers/attention/torch_native_backend.py`: `_run_sdpa_forward_extend` (~59),
   especially `per_req_query_redudant`.

Check yourself before reading on:

- Why does `match_prefix` split a node, when it only reads? What would break if it
  didn't?
- A prompt is entirely in the cache. Why must one token still run?
- In `_split_node`, why can the new node copy `child.lock_ref` as it is?
- Two requests in one prefill batch computed the same prefix. Which K/V survives,
  and what must the loser do to its table row?
- Why does `add_one_req` lock the prefix *before* it checks the budget?

## Key ideas

- **Every slot has one owner.** The free list, the tree, or a request, and a request
  only owns positions at or above `cache_protected_len`. Nearly every bug in this area
  is a slot with two owners or none. `check_slot_owners` in `test_scheduler.py` is
  this sentence as code.
- **Locks protect what's being read.** A running request reads its prefix on every
  step, so it locks its path, and only unlocked leaves can be evicted. A parent can't
  go before its children, because a child without its parent can't be reached.
- **Idle cache counts as free.** The budget becomes `free + evictable - reserved`, so
  a pool full of cache blocks nobody. `token usage` no longer counts idle cache, so
  it still returns to 0.00 once requests finish. The new `cache usage` doesn't.
- **A lock costs budget.** Locking a cached prefix turns evictable slots into
  protected ones, so they can't pay for anybody's allocation. Admission charges
  `reserved + locked`. Without that, the same slots get spent twice, which ends in the
  assertion in `_prepare_extend`. `test_a_lock_is_charged_at_admission` shows it.
- **Checkpoint after prefill, not only at the end.** Otherwise a long answer keeps its
  prompt out of the cache for as long as it runs.
- **Eviction before retraction.** Idle cache goes first. A running request loses its
  work only when nothing else can.

## What did NOT get better

- **A burst gets nothing.** Requests in one prefill batch all find an empty cache, so
  they all compute the shared prefix, and all but one throw it away, unless
  `--in-batch-dedup` is on.
- **Decode is unchanged.** A cached prefix is read like any other K/V, and each step
  still gathers it all. Only prefill got shorter.
- **Reads got more expensive for each hit.** `_run_extend` still gathers the whole
  sequence, once per layer, so a 1-token extend over a 2000-token prefix reads 2000
  rows. It computes far less than a full prefill, but it isn't free. Avoiding that
  copy is M5's job.
- **Unique prompts pay a small price.** Their K/V stays until it's evicted, and
  eviction takes place on the scheduler thread, in the middle of a step.
- **A float caveat.** K/V computed in a different batch can differ by rounding on
  GPU bf16 (see `test_batching.py`), so a cached prefix can flip a near-tie that a
  fresh run wouldn't. SGLang has the same issue.

## A bug from M3

The stress test here found a bug in M3, now fixed in both chapters. With `--lazy-free` at page size 1,
`TokenAllocator.free` stored the tensor it was given, and `free_req` passes a slice of
the table, which is a view. When the row was reused, the pending free list changed
with it: the new owner's slots went on the free list, and the old slots were lost.
Nothing crashes, and two requests end up sharing K/V. M3's test missed it because it
only uses `--lazy-free` with `--page-size 64`, where boolean indexing makes a copy.

The fix is a clone in `free`, with a regression test in each chapter.

## Run

From the repo root. With no GPU or model, the first block is everything you can run.

```bash
# No model needed.
uv run pytest m4-radix-cache
uv run m4-radix-cache/cache_sim.py
uv run m4-radix-cache/match_cost.py

# A tiny random model, built on the fly. No download.
uv run pytest m4-radix-cache/tests/test_attention.py m4-radix-cache/tests/test_prefix_output.py -m model

# A real model.
uv run pytest m4-radix-cache/tests/test_batching.py -m model --model Qwen/Qwen3-0.6B
uv run m4-radix-cache/server.py --model Qwen/Qwen3-0.6B --port 8000
# another shell:
uv run m4-radix-cache/client.py --port 8000
uv run m4-radix-cache/prefix_bench.py --port 8000

# The same measurement without the cache:
uv run m4-radix-cache/server.py --model Qwen/Qwen3-0.6B --port 8000 --disable-radix-cache
uv run m4-radix-cache/prefix_bench.py --port 8000
```

Qwen3 only, for now, as in M3.

## Done when

1. All the tests pass.
2. `test_batching.py` matches on every prompt, on CPU exactly, and reports more prompt
   tokens reused in pass 2 than in pass 1.
3. `client.py` passes as in M1 to M3.
4. With the cache on, `prefix_bench.py` shows `warm` well below `cold`. Without it, the
   two are about the same. Write down all the numbers.
5. In the scheduler log, `cache hit rate` climbs through the warm requests, `gpu mem`
   stays flat, and `token usage` returns to 0.00 while `cache usage` stays up.

## Cache policies

Each section gives the problem, the solution and how to measure it. The
`cache_sim.py` numbers below were measured: it runs the real scheduler and tree on
the fake runner, so they don't depend on a GPU.

| feature | switch | code | measured with |
|---|---|---|---|
| LFU | `--eviction-policy lfu` | `EVICTION_POLICIES`, `hit_count`, `cache_inserted_len` | `cache_sim.py` |
| Longest prefix first | `--schedule-policy lpm` | `Scheduler._sort_waiting_queue` | `cache_sim.py` |
| In-batch dedup | `--in-batch-dedup` | `Scheduler._shares_with_batch` | `cache_sim.py`, `prefix_bench.py` |
| Keep retracted work | `--cache-retracted` | `update_running_batch`, `trim_leaves` in `RadixCache.evict` | `tests/test_scheduler.py` |
| Faster matching | `--fast-match` | `match_len_gallop` | `match_cost.py` |
| Flush | `POST /flush_cache` | `Scheduler.flush_cache`, `FlushCacheReq` | `tests/test_scheduler.py` |

The numbers for LFU, LPM and dedup (1000 requests, 32 at a time, 50 system prompts of
1000 tokens with Zipf popularity, 32-token questions):

| setup | computed, pool 8192 | hit rate | computed, pool 40000 | hit rate |
|---|---:|---:|---:|---:|
| no cache | 1,032,000 | 0.000 | 1,032,000 | 0.000 |
| lru | 615,000 | 0.404 | 182,000 | 0.824 |
| lfu | 615,000 | 0.404 | 140,000 | 0.864 |
| lru + lpm | 363,000 | 0.648 | 182,000 | 0.824 |
| lru + dedup | 608,000 | 0.411 | 177,000 | 0.828 |
| lru + lpm + dedup | 361,000 | 0.650 | 177,000 | 0.828 |

### LFU

`--eviction-policy lfu`, after SGLang's `evict_policy.py`. The question is what counts
as a hit, if a request can be matched many times while it waits, and insert twice.

A hit is counted on insert, not on match, and `cache_inserted_len` stops one request
from counting twice. That's SGLang's rule: matches repeat as long as a request waits.

Look at the table: **LFU wins only at the large pool** (140k against 182k computed).
At 8192 the two tie exactly. Running requests hold most of a small pool, so eviction
has little to choose from. A policy can't help when there's nothing to choose between.

### Longest prefix first

`--schedule-policy lpm` sorts the queue by match length before admission, and skips
the sort when the queue is long.

```bash
uv run m4-radix-cache/cache_sim.py --pool 8192
uv run m4-radix-cache/cache_sim.py --pool 8192 --num-prefixes 5 --prefix-len 3000 --num-requests 400
```

This is the biggest win here, on the small pool: 41% fewer tokens computed, because
requests run while their prefix is still in memory. With 5 prompts of 3000 tokens, the
hit rate goes from 0.445 to 0.950. At the large pool it changes nothing, since nothing
is evicted between a prompt's uses.

It's paid for in fairness. `worst wait` goes from 99 to 116 steps in the first run. It
isn't a disaster here because the workload is a closed loop: only 32 requests exist at
a time, so a passed-over request soon reaches the front. With open arrivals, a rare
prefix can wait as long as popular ones keep coming. Sorting also matches the whole
queue on every pass, and may split nodes for requests that won't run yet.

### In-batch dedup

With 32 requests arriving together, they all compute the same system prompt.
`--in-batch-dedup` makes only one do it. SGLang: `_compute_prefix_matches`.

If a request has little in the cache but a long match with a request already in this
batch, it stops admission, and runs next step as a cache hit. In the unit test, 4
requests sharing 40 tokens compute 44 tokens instead of 164. Since first-come order
must hold, it's a `break`. SGLang sorts such requests last, under LPM.

The simulator shows little (608k against 615k), because in a closed loop requests
trickle in and rarely arrive together. It's for bursts, so compare the `burst` row of
`prefix_bench.py`. The price is that the held-back requests start one step later.

### Keep retracted work

`schedule_batch.py` ~2400 has a TODO, "for preempted requests, we may want to insert
into the tree". `--cache-retracted` does it. Does it help?

**On its own, no: the first attempt saved exactly zero tokens** (31 and 31 in the test).
Retraction only happens when `ensure_free` has already evicted everything evictable.
So the K/V just handed to the tree is the only thing left to evict, and the next
`ensure_free` throws away the whole node to free one slot.

The fix is to evict only the end of a leaf. The start of a sequence stays valid without
its end, so `trim_leaves` splits the leaf and frees just what's needed.
`--cache-retracted` turns on both. `test_retraction_can_keep_the_work` checks that fewer
tokens are computed. SGLang's `evict` takes whole nodes, so the TODO would need the same
change. This might make a real contribution for M7, but measure it there first.

### Faster matching

Comparing token by token is a Python loop on the scheduler thread. `--fast-match`
replaces it with the idea of SGLang's `RadixKey.match_at`.

```bash
uv run m4-radix-cache/match_cost.py
```

Measured on this machine:

| tokens | case | loop | gallop | speedup |
|---:|---|---:|---:|---:|
| 1024 | full match | 128 us | 7.6 us | 17x |
| 8192 | full match | 1054 us | 16 us | 66x |
| 65536 | full match | 8603 us | 64 us | 135x |
| 65536 | mismatch at 8 | 0.95 us | 3.1 us | 0.3x |

It loses on an early mismatch, by 2 us, and it wins by milliseconds on a long match.
That's a good trade, because a millisecond here is a millisecond the GPU sits idle.

### Flush

`POST /flush_cache` empties the cache, after SGLang's `Scheduler.flush_cache`. Why only
when idle?

`prefix_bench.py` needs this, to measure a cold start without restarting the server.
The server sends `FlushCacheReq` and answers at once. A ZMQ socket keeps its order, so
a request sent after the response finds the cache flushed. SGLang waits for a reply
instead, which tells the caller if it worked. Ours only logs a warning.

SGLang flushes only when idle because it resets the pools outright, and running
requests hold slots in them. Ours only evicts, so it would be safe mid-flight, and
locked paths would stay. The rule is kept so that "the cache is empty" is true after
a flush.

## How far this was tested

Run on a machine with 4 CPUs and no GPU:

- `test_radix_cache.py`, `test_memory_pool.py`, `test_scheduler.py` and
  `test_attention.py` pass. The stress test runs 60 random setups over all the options.
- `cache_sim.py` and `match_cost.py` gave the numbers above.
- 14 bugs were injected on purpose, one at a time, such as a split that forgets locks,
  an uncharged lock, and a release that frees the shared prefix. The model-free tests
  caught all 14.
- `test_prefix_output.py` and `test_batching.py` (10 of 10 prompts token-exact,
  with and without flags) passed on a tiny **random** Qwen3 on CPU, built locally.

**Not done: anything with a real model or a GPU.** `Qwen/Qwen3-0.6B` hasn't been run,
`prefix_bench.py` has no recorded result, and `client.py` wasn't run. So there are
no TTFT numbers yet, and all expected TTFT results above are predictions. `/flush_cache`
and rising `cache hit rate` were seen once, on the tiny model, with `bench.py`. Write
your numbers here.
