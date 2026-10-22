# Design: one cold compile at a time on the direct Neural Engine lane

Status: **approved and implemented**, with option A of section 3, the orphan
expiry rule of section 1, and the changes recorded under "Changes during
implementation" at the end. The hardware check on the M4 mini is pending.

## Why

Apple's Neural Engine compiler runs one shared step for the whole machine, and
that step serves one layer at a time. A small shape compiled beside a wide one
therefore waits for one wide layer per layer of its own:

| Measurement (M4 mini, macOS 27.0) | Result |
|---|---|
| GTE@128 alone / started together with Qwen3@4096 | 5.2 s / 66.7 s (12.8×); Qwen3@4096 itself 1.0× |
| Two compiles started together, makespan / sum of standalone times | 0.75–1.04 |
| Four compiles started together, makespan / sum of standalone times | 1.03–1.91 |

Running compiles side by side gains no throughput, and it makes small shapes
pay for wide ones. Wide shapes are also expensive on their own: Qwen3@4096
takes about 90 s on the M4 and 114 s on the M5 Max, and Qwen3@8192 is
estimated at 230 s on the M5. Production bounds an admission wait at 600 s
(`ANE_ADMISSION_WAIT_TIMEOUT`), and AFT's search embeds through the Qwen3 lane,
so after an OS update or restart a queue of cold compiles can fail its requests.

## 1. The compile queue

**Rule:** at most one direct-ANE shape compile (`ANE_ADMIT_SHAPE` or
`ANE_ADMIT_PACKED`) is in flight in the module at a time, across all direct-ANE
workers. Queued compiles are started cheapest first, with aging (section 2).

**Where it lives:** in `AneResidencySupervisor`
(`crates/synapse-module/src/worker_host/mod.rs`, module `ane_residency`). The
supervisor already sees every admission of every direct-ANE worker, and the
direct-ANE lane lock (`~/Library/Caches/ck-synapse/ane-direct.lock`) lets only
one module process run direct-ANE workers. So "one per module" is "one per
machine" for Synapse. Compiles by other processes (Core ML, other apps) are
not controlled; they only make predictions late (see "Cost estimate").

The supervisor already has a queue: requests whose shape has no slot wait in a
FIFO (`State::waiters`, served by `budget_head`), and the head reserves budget,
evicts if needed, and dispatches the compile at once. Every request in that
queue needs a compile, because a request whose shape already has a slot
(resident, admitting or evicting) waits outside it. The change is therefore
small:

1. **Order.** `budget_head` returns the queued compile with the smallest
   *start key* (section 2) instead of the oldest ticket.
2. **One in flight.** `next_step` does not reserve a slot (`Step::Admit`) while
   another slot is `Admitting`; the head waits for that compile to finish.
   The admission's single resource-refusal retry
   (`retry_resource_admission`) stays inside the same turn.
3. **Single flight extends to the queue.** A second request for a shape that is
   already queued joins that queue entry, as it joins an `Admitting` slot today.
4. **Queue entries outlive their requests.** Today a request that times out or
   is abandoned while queued drops its ticket, so its compile never runs and
   every retry starts again at the back. A queued entry instead stays until it
   is compiled or its worker retires; requests only attach to it. Single flight
   bounds the number of entries to the distinct (worker, model, shape, program)
   keys, about 30 for four models.
5. **Orphaned entries expire.** An entry is kept for its requests' sake only
   when the last of them was answered `shape_compiling` (refused early, or its
   deadline expired while it waited on the compile). A caller that simply went
   away (cancelled or disconnected) gets no promise, and its entry is dropped
   with it. A kept entry no request is attached to expires once twice the
   `retry_after_ms` it last promised has passed, measured from that promise,
   and at once when its worker retires. Otherwise demand from a caller that
   never retries would keep compiling and evicting shapes other requests use.

### Cost estimate

`cost(model, shape) = layers × per_layer_seconds(model, shape)`

- `layers` comes from `AneShapeWorker::executable_count`, already used for the
  budget (22 for GTE-ModernBERT, 28 for Qwen3-0.6B; 28 for a packed program).
- `per_layer_seconds` starts from a built-in table measured on the M4 (median
  per-layer compile and load, seconds):

  | Width | 128 | 512 | 2048 | 4096 |
  |---|---:|---:|---:|---:|
  | GTE (hidden 768) | 0.21 | 0.37 | 0.86 | – |
  | Qwen3 (hidden 1024) | 0.38 | 0.76 | 1.49 | 3.02 |

  Missing cells are interpolated linearly in width, per hidden size.
- **Learning.** Every finished compile records its measured duration per
  (model_ref, shape) and a machine factor, `measured / table`, as a moving
  average across all compiles. Unseen shapes use `table × machine factor`, so
  the first compile on a slower machine (the M5 is about 1.3× the M4) corrects
  the rest. This state is in memory only; after a restart the table is used.
- **While a compile runs**, its remaining time is
  `(layers − layers_done) × measured seconds per layer so far`. The worker
  already sends one setup-progress frame per compiled layer; the module counts
  them in `liveness::exchange` and needs to pass that count to the supervisor.
  This is module plumbing, not a protocol change.

Only the order depends on the cost being roughly right. The refusal in section 3
depends on its absolute value, which is why it is learned and corrected by progress.

### Interaction with existing pieces

- **100-executable budget** (`ANE_RESIDENT_EXECUTABLES_TOTAL`): accounting is
  unchanged; a slot is charged from reservation until its `EVICTED` ack. Since
  only one slot can be `Admitting`, compiles that are queued but not running hold
  no budget, so queuing never reduces resident capacity. The head may already
  run its evictions while the current compile runs, so its compile starts at
  once when the turn comes. Eviction RPCs are not compiles and are not queued.
  The 10-05 serialized-admission stress failure came from hardware resource
  exhaustion during eviction churn, not from serialization, and this design does
  not change eviction churn.
- **250 ms waiter grace** (`ADMISSION_HANDOFF`): unchanged. When a compile
  finishes, the requests that joined it get their closed handoff cohort exactly as
  today. The next compile may start immediately; it can wait for that grace only
  if the shape just compiled is the only eviction victim it could use.
- **Eviction preference** (idle stream, then no queued demand, then
  non-smallest rung, then LRU): unchanged. "Queued demand" now also counts the
  shapes of queued compile entries. The starved-head rule (`starved_head`) still
  applies to the head chosen by the new order.
- **`shape_compiling` and the request deadline:** a request waiting for its
  queued compile's turn is classified as waiting on a compile
  (`WaitPhase::Compile(shape)`), so its deadline answers `shape_compiling`, not
  `deadline_exceeded`. Today it waits as `WaitPhase::Other`. The detached
  compile continues either way, as the contract already promises.
- **Worker streams:** a worker's stream is no longer held by a compile while that
  compile waits for its turn, so resident shapes on that worker keep serving.
- **Packed programs:** `prepare_packed_lease` already runs only when no request
  is queued. It stays lowest priority: it starts only when the queue is empty.

## 2. Starvation: aging

Each queue entry gets a fixed start key when it is created:

`start_key = enqueued_at + cost`

and the entry with the smallest key starts next. This is cheapest-first with
linear aging, expressed as one static key: an entry of cost *C* can be passed
only by entries that arrive less than *C* after it and are cheaper than the time
it has left. So a Qwen3@4096 compile (about 90 s) queued at *t* starts no later
than the work that arrives before *t + 90 s* with an earlier key. That work is
finite, because single flight allows at most one entry per shape.

One more rule keeps earlier promises: a newcomer is not placed ahead of an
accepted entry if that would push the entry's predicted completion past the
bound it was accepted under (section 3). It goes behind it instead.

Once started, a compile runs to completion. The worker compiles all layers of a
shape in one RPC, so pausing a wide compile between layers would need a worker
protocol change; that is out of scope.

## 3. The 600 s bound and the request deadline

For each request, the supervisor predicts when its shape will be resident:

`predicted_done = now + remaining(current compile) + Σ cost(entries ahead) + cost(own entry)`

and compares it with `bound = min(request deadline, queue entry time + 600 s)`.

### Options

- **A. Refuse early, honestly.** If `predicted_done > bound` when the request
  arrives, or later when the queue changes, answer at once with
  `shape_compiling` and `retry_after_ms = predicted_done − now` (at least 1000).
  The queue entry stays (section 1, item 4), so a retry at that time finds the
  shape resident or almost so. A request that is predicted to fit waits as
  today. If the prediction was too optimistic, the request's deadline still
  ends with the existing `shape_compiling` path.
- **B. Keep wide shapes off the request path.** Compile 4096 and 8192 in
  advance or when idle, so requests only meet small cold compiles.

### Recommendation: A

- It answers in milliseconds instead of hanging for 600 s, and tells the caller
  when to come back. Today, a request without a deadline that waits longer than
  600 s in the queue gets `ane_resources_exhausted` with `retry_after_ms: 250`,
  which is wrong on both counts.
- B does not fit the budget. One wide shape costs 22–28 of the 100 executables,
  whatever its width, so the 4096 and 8192 shapes of four models cannot stay
  resident next to the small shapes that queries use. Precompiled wide shapes
  would evict the small shapes, or be evicted and need a cold compile again.
  B also still leaves a cold compile on the request path whenever a long input
  arrives before the idle compile.
- **Trade-off of A:** a long input that arrives with a cold queue fails fast
  instead of eventually succeeding; the caller has to retry. Callers that ignore
  `retry_after_ms` and retry at once get refused again cheaply, without extra
  compiler work, because the entry is shared.
- **Wire-contract text (decision needed):** A uses the existing code
  `shape_compiling`, the existing fields and the existing `details`
  (`{lane_id, shape}`); nothing in the schema changes. But
  `docs/wire-contract-v1.md` §2 currently says `retry_after_ms: 1000` and
  that the code is returned "only when that deadline expires". A changes the
  value and the timing, so those two sentences would be amended. The
  alternative that leaves the text untouched is to keep 1000 and answer only at
  the deadline, keeping the persistent queue entry; that is less honest and
  makes callers wait out their deadline.

## 4. What does not change

- Shape and model fingerprints, certified identities and placement inventories.
- The bucket ladder (`ANE_SHAPE_LADDER`, 128 to 8192) and how sequences are
  assigned to rungs (`run_by_published_rung_until`).
- The wire contract's codes, fields and schema (but see the text question in
  section 3), and the worker protocol: `ANE_ADMIT_SHAPE`, `ANE_EVICT_SHAPE`,
  progress frames.
- The Core ML lane, the direct-ANE worker and its vendored compiler crate.
- The 100-executable budget, the per-model and total shape limits, the 250 ms
  handoff, eviction preference and quarantine.
- The 600 s setup-idle liveness window (`SETUP_IDLE`).

## 5. Tests that will prove it

### Test double: a shared per-layer compiler

A new fake `AneShapeWorker`, used only in tests, models the measured behaviour.
Each fake worker compiles a shape as `layers` steps. Each step spends a private
time `p` on its own and then holds one compiler shared by all fake workers for
`per_layer(model, shape)`. The shared compiler is a FIFO lock (`tokio::sync::Mutex`
is fair), so two compiles alternate layer by layer. The tests run on Tokio's
paused clock (`start_paused = true`), so durations are exact virtual seconds and
the tests are deterministic and fast. With the M4 numbers it reproduces today's
behaviour: GTE@128 (22 × 0.21 s) beside Qwen3@4096 (28 × 3.02 s) takes about
22 × 3.2 = 70 s instead of 4.6 s.

### Portable tests (each fails on current code)

| Test | Asserts | Why current code fails |
|---|---|---|
| `small_compile_beside_wide_finishes_in_its_standalone_time` | Qwen3@4096 requested, then GTE@128 1 ms later on another worker: GTE@128 is resident within 1.1× its standalone time, and Qwen3@4096 within its standalone time plus GTE@128's | both compile together; GTE@128 took 71.1 s |
| `at_most_one_cold_compile_runs_across_workers` | the fake records the peak number of overlapping compiles across four workers and three shapes each: exactly 1 | four ran at once |
| `queued_compiles_start_cheapest_first` | while Qwen3@2048 runs, queue Qwen3@4096, GTE@512, Qwen3@128; start order is GTE@512, Qwen3@128, Qwen3@4096 | they started in arrival order, at once |
| `wide_compile_starts_after_its_aging_window` | under a stream of small compiles, every small one whose start key is earlier than the wide one's starts before it, and every later one after it | the wide compile started at once, beside the small ones |
| `queued_compile_holds_no_executable_budget` | with three compiles queued behind Qwen3@2048, 28 executables are reserved | 100 were reserved |
| `predicted_overrun_refuses_early_with_honest_retry` | a GTE@128 request with a 30 s deadline queued behind Qwen3@4096 is answered at once with `shape_compiling` and `retry_after_ms` within 10% of the prediction; its compile still runs, and a retry then leases without a second compile | it waited 70.1 s |
| `queue_beyond_600s_refuses_early_without_a_deadline` | of eight Qwen3@4096 requests without a deadline, the eighth (predicted at 676 s) is refused at once with `shape_compiling`; the other seven succeed | the eighth succeeded at 676 s |
| `deadline_while_queued_for_compile_answers_shape_compiling` | the fake runs twice as slowly as predicted; the queued request is answered `shape_compiling` exactly at its deadline, and a retry within the promised delay is served by one compile that starts after the running one | answered after 73.8 s |
| `machine_factor_corrects_unseen_shape_predictions` | after one compile 1.3× slower than the table, a request is refused at once with a prediction scaled by 1.3 | not refused |
| `orphaned_entry_expires_after_twice_its_promise` | a refused request's compile, which nobody retries, is dropped before its turn when twice its promise passes, and never compiles | not refused; waited 212 s |

All ten were run against the base commit, `2c1774c3`, with only the test file
added, and all ten failed; the right-hand column quotes those failures.

**An existing test changes meaning.** It was
`eviction_prefers_an_idle_stream_and_admits_before_the_long_compile_ends`. It
asserted that a second worker's compile is dispatched while another compile runs,
which this design forbids. Its eviction claim stays: the idle stream is evicted
before the long compile ends. Its admission claim is reversed: the admission
starts after the long compile. The test is now
`eviction_prefers_an_idle_stream_and_evicts_before_the_long_compile_ends`, and the
mutation catalogue row that names it is updated. The FIFO statements in
`docs/ane-serving.md` ("Admission fairness and eviction") and in the
`ane_residency` module comment are updated to match. The handoff and busy-stream
eviction tests pass unchanged.

### Hardware check on the M4 mini

The operator runs the kit from their own shell; it never runs from an agent
session. It follows the October latency kit's conventions. It refuses to run
while another Neural Engine user is present, unless that user is allowed by
name. It holds the direct-ANE lane lock, unsets `TMPDIR`, and builds in release
with full Xcode's `DEVELOPER_DIR` in a throwaway clone. It runs three repeats
and keeps per-run JSON with host load and OS build. Only the change is built:
the October kit already measured the base behaviour on this mini, in 108 runs.

It drives the module supervisor itself, not a bare compile probe: the ignored
case `real_direct_ane_compile_scheduling` beside `real_direct_ane_residency_safety`,
run through `crates/synapse-worker-ane-direct/tests/run_stress.py --case
compile-scheduling`. Each arm uses fresh worker processes and a fresh
supervisor, so every shape compiles cold. Packages: `gte-modernbert-base` and
`qwen3-embedding-0.6b` (the ones on the mini). Arms:

| Arm | Pass on the change | Before (October kit) |
|---|---|---|
| standalone: GTE@128/512/2048, Qwen3@128/512/2048/4096, one at a time | all succeed; these times are the reference | |
| Qwen3@4096 asked, GTE@128 1 ms later | GTE@128 ≤ 1.5× its standalone time; Qwen3@4096 ≤ 1.1× (its standalone + GTE@128's) | 66.7 s (12.8×) |
| GTE@128 asked 10 s after Qwen3@4096 | GTE@128 ≤ the rest of the Qwen3 compile + 1.1× its standalone + 1 s | one wide layer per layer of its own |
| GTE and Qwen3 at 128, 512 and 2048, all asked at once | makespan ≤ 1.05× the sum of standalone times | 1.03–1.91× |
| Qwen3@4096 asked, GTE@512 with a 30 s deadline 1 s later | answered `shape_compiling` within 1 s; `retry_after_ms` within 25% of when GTE@512 actually finished compiling | answered at 30 s with 1000 ms |

Every arm also requires that no two shape compiles overlap, no
`ane_resources_exhausted`, and a drained queue. The kit reports pass or fail per
arm and keeps the raw timelines. A hardware run needs approval and exclusive use
of the mini's Neural Engine.

## Changes during implementation

- **Decisions:** option A of section 3. The two sentences in
  `docs/wire-contract-v1.md` §2 are amended. The renamed test above is
  approved. The orphan-expiry rule (section 1, item 5) was added at approval.
- **Gathering window.** A compile that arrives while nothing is compiling
  would otherwise start at once, so a wide compile requested 1 ms before a small
  one would hold the small one up for its full length. A compile with a known
  cost therefore waits 50 ms after its request before it may start, so compiles
  requested together start cheapest first. This adds 50 ms to a compile of 4 s
  or more. Packed programs skip it; they are prepared only while nothing else
  is queued.
- **Remaining time of the running compile** is `predicted cost − elapsed`, but
  at least one layer's share. The per-layer progress frames are not wired into
  the supervisor yet. The machine factor corrects predictions after the first
  measured compile, not during it.
- **Accepted requests keep their bound.** A request predicted to be served in
  time records its bound. A new compile is ordered behind that entry if going
  first would push the entry past the bound.
- **Unknown cost.** A worker whose seed is zero has an unknown cost: it
  queues in arrival order, never gathers, and is never refused early. Test
  channels backed by instant mock workers use that; production seeds always
  come from the pinned manifest.
