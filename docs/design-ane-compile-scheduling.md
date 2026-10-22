# Design: one cold compile at a time on the direct Neural Engine lane

Status: **approved and implemented**, with option A of section 3, the orphan
expiry rule of section 1, and the changes recorded under "Changes during
implementation" at the end. The hardware check on the M4 mini (macOS 27.0
26A428, 3 repeats at `7f096b96`) passed 13 of 15 scheduling arm runs. No two
compiles overlapped, and there was no resource refusal. Every compile started
in start-key order, within 0.003 s of the previous one ending, after the 50 ms
gathering window. The two misses were single compiles that ran slower than
their standalone time in the same repeat, with nothing else compiling:

- In one `together` run, GTE@128 started first, 0.07 s after its request, and
  compiled in 9.8 s against 5.8 s standalone.
- In one `batch` run, the makespan of 139.6 s equals the sum of that arm's six
  serial compiles. Qwen3@128 took 20.1 s against 12.2 s standalone.

The standalone times themselves varied between repeats: Qwen3@128 took
12.2–24.0 s.

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

### Changes after the October audit of `5631cba9`

Each change below was made after a deterministic test failed against
`5631cba9`. The tests are in
`crates/synapse-module/src/worker_host/compile_queue_fault_tests.rs`.

- **A failed compile holds the gate until exit is confirmed.** The gate counts
  both `Admitting` and `Failed` slots. A failed slot leaves only when recovery or
  retirement confirms that the old worker process has exited, so a compile that
  may still be running cannot overlap another one. A panicked admission task is
  marked failed, and its worker is restarted on a task of its own, including on
  the orphan path. Once the old process's exit is confirmed, the failed slot
  releases the gate and its budget even if the restart then fails, so one
  broken worker cannot stall every other worker's cold compiles.
- **One driver per orphaned entry.** The driver's queue ticket is created and
  recorded on the entry under the same lock as the decision to keep it, and that
  ticket never counts as a waiting request. A request is registered for its
  shape when it enqueues, so a retry that has not decided yet counts as attached.
  A driver does not evict while another compile runs. Once it has evicted, it is
  committed to its compile and no longer expires.
- **Promises are what the caller was told.** An entry is kept only for a caller
  answered `shape_compiling`, for twice the `retry_after_ms` that caller
  received. When the caller's own timer answers first, the module asks the
  queue for a fresh prediction instead of reusing the one from the start of the
  wait. An admission cancelled before it reached the worker goes back into the
  queue in its old place when its caller was told to retry or other requests
  joined it.
- **Refusal before dispatch or eviction.** The queue head's prediction is
  checked against its bound before it reserves, dispatches or evicts anything. A
  wait on the compile queue that reaches the 600 s backstop is answered
  `shape_compiling`, with a fresh prediction and a kept promise. Only a wait for
  budget keeps `ane_resources_exhausted`.
- **Learning.** The clock starts when the admission is written to the worker,
  after any wait for its stream. Compiles retried after a resource refusal, and
  packed programs, are not learned. Queued entries keep their start key, but
  their predicted duration is recomputed from the current estimate on every read.
- **Wakeups and recovery.** A waiting request arms its wakeup timer from the
  same instant as its decision, so the end of a gathering window is never
  skipped. Entries whose worker is recovering or retired are not elected, so
  they cannot block other workers. A failed restart clears `recovering`, so a
  later recovery can try again. The exception is a restart that could not
  confirm the old process's exit: that worker stays closed, as before, and its
  queued compiles are dropped.

## 6. Rebuild: one scheduler task (proposed after the second audit of `c85405b3`)

Status: **implemented**, with the two changes approved with it (6.4, exit
never confirmed; 6.9, a stopped scheduler) and the choices recorded in 6.10.
Most findings of the second audit come
from request tasks and orphan drivers driving the queue at the same time, each
under its own lock acquisitions, instants and cancellation. The rebuild moves
every queue decision into one task.

### 6.1 One task owns the scheduling state

`AneResidencySupervisor` starts a scheduler task that alone owns:

- queue entries and their start keys;
- waiters and the promises given to them;
- slot reservations and the 100-executable budget;
- the one-compile gate;
- measured costs and the machine factor;
- recovery state per worker.

The scheduler runs one loop:

1. Wait for the earliest of: a message, its next timer, or a finished operation.
2. Take one instant, `now`.
3. Decide everything that changed, at `now`.
4. Spawn the resulting compile, evict and recover operations into a `JoinSet`
   the scheduler awaits itself.
5. Compute the next wake-up from the same `now`: gather windows, handoff ends,
   promise expiries and waiter bounds.

Operations report back by finishing, and a compile also sends a "dispatched at"
message. A panic or `JoinError` therefore always reaches the scheduler, whoever
asked for the work.

Resident leases are the main state shared with requests (6.3). They sit in a
small residency table behind a short mutex. The scheduler is the only writer of
slot states. Requests add and drop leases and write a few ranking and cohort
fields under the same mutex; 6.10 lists them and why that is safe.

### 6.2 Requests never drive the queue

A request that cannot lease a resident shape sends `Want { shape key, worker
generation, executables, bound }` with a reply channel, and awaits one answer:

- **Ready:** the shape is resident and this waiter may lease it (6.3).
- **`shape_compiling { retry_after_ms }`:** the scheduler's prediction at the
  instant it decided, which is also the promise it records.

The scheduler answers at the bound itself: the waiter's bound is one of its
timers. So a direct `lease_until` caller and a serving request get the same
fresh prediction.

When the serving caller's own timer fires during a compile wait, it sends
`Expired` and awaits the scheduler's answer. The scheduler records one decision
per waiter, so the caller path and the inner path cannot give different delays.
Stream-wait classification asks the scheduler for the running compile's remaining
time, instead of using the 1000 ms floor.

Cancelling a request just drops its receiver; the scheduler notices the closed
channel on its next pass. Whether a compile stays queued is decided only by the
scheduler, from what it promised:

- an entry lives while a waiter is attached, or until twice its last promise;
- promises belong to the compile, not to a request: they move from the entry to
  the slot at reservation, and back if the compile is withdrawn.

There are no orphan driver tasks.

Compile and evict operations are never tied to a request's cancellation. While
an operation still waits for the worker stream, the scheduler can withdraw it
with its own token. It does so only when the entry has no waiter and no live
promise. That keeps "a caller that went away leaves nothing on the wire" true,
which the existing cancellation test checks.

### 6.3 Resident leases stay on the fast path

Inference leases each rung with `try_lease`: one lock of the residency mutex. It
checks that the slot is `Resident` and of the current generation, and that no
scheduler barrier is set on it, then increments `leases`. There is no message
and no await. Dropping the lease decrements under the same lock, and notifies the
scheduler only when it has asked to hear about that slot: it is waiting for a
victim, or a handoff cohort is open.

The scheduler sets a barrier in two cases:

- while a starved head has a turn, as `behind_starved_head` does today;
- for non-cohort requests during a 250 ms handoff.

A request that finds a barrier, or no resident slot, falls back to `Want`.
Eviction marks a slot `Evicting` under the same mutex, so a lease and an
eviction cannot race. The uncontended mutex costs tens of nanoseconds on the hot
path. The scheduler round trip (mpsc plus oneshot, a few microseconds) is paid
only on a cold or contended rung.

### 6.4 Failures

- **Definite worker error** (the worker answered `ERR`, for example
  `compile_failed`): nothing can still be compiling, so the gate is released at
  once.
  - The slot's executables stay charged until the worker rolls back or the
    worker is recovered, as today. A failed slot no longer counts as running.
  - Recovery, where it runs, no longer blocks other workers' compiles.
- **Transport fault, timeout or panic** (the outcome is unknown): the slot
  holds the gate only until the old process's exit is confirmed.
  - The worker trait's restart is split into two steps: stop and confirm exit,
    then start a replacement and restore its models.
  - The scheduler gets exit confirmation as its own event, so the gate is
    released before the replacement starts, not after it.
  - While exit is unconfirmed, predictions count the failed slot as one layer
    of remaining time per pass, not zero.
- **After a confirmed exit,** every slot of the old generation is removed and
  refunded, whatever its state: `Resident`, `Evicting`, `Failed` or
  `Admitting`. Late results from that generation are ignored by slot id. This
  removes the ghost slots left by a failed replacement.
- **Exit never confirmed:** the worker stays closed, its entries are dropped and
  its slots stay charged, as today. Other workers keep compiling, but only once
  no slot of that worker can still be compiling.

### 6.5 Recovery

Starting recovery is a scheduler event, so the next pass re-elects heads and
re-checks bounds at once; there is no missed notify.

A recovering worker's entries are not elected and not counted ahead of anyone.
When the worker rejoins, its entries keep their start keys, but each is placed
under the accepted-bound rule, the same rule a newcomer follows. An entry that
would push an accepted waiter past its bound goes behind that waiter.

### 6.6 Learning

The compile operation reports its dispatch instant, the moment the admission
reaches the worker:

- the running prediction counts from dispatch, not from reservation;
- the sample is `finish − dispatch`;
- retried compiles and packed programs are not learned.

Queued predictions are re-estimated on every pass from the current estimate.
Start keys stay fixed.

### 6.7 What stays the same

- The wire contract as amended in this change: `shape_compiling` with a
  predicted `retry_after_ms` of at least 1000, possibly before the deadline.
- Fingerprints, the bucket ladder and the 100-executable budget, with the
  per-model and total limits.
- Eviction preference.
- Cheapest first with aging, the 50 ms gathering window, and the
  accepted-bound rule.
- The orphan expiry factor of 2.
- The 250 ms handoff, which becomes a scheduler timer. When a compile finishes,
  its waiters get Ready and a 250 ms cohort barrier. The barrier ends when every
  cohort member has leased or left, or when the timer fires, whichever comes
  first; that is today's rule.

### 6.8 Tests

All existing compile-queue and fault tests keep their assertions wherever the
behaviour stays.

Tests that now call `next_step`, `QueueTicket` or `inner.lock()` directly are
rewritten to drive `Want` messages. They read state through a test-only
`snapshot()` query that returns entries, slots, gate and promises. The
assertions are kept; only how the state is reached changes.

The 600 s backstop test asserts the predicted `retry_after_ms` (about 7.64 s),
not just the code.

Each finding of the second audit becomes a test or is ruled out by the
structure:

| Finding | Test, or why it cannot happen |
|---|---|
| Ghost old-generation slots after confirmed exit and failed replacement (P1) | Test: a resident shape on W, then a confirmed exit, then a failed replacement. Assert no W slot is left, no `Evicting` charge survives, and no lease is granted. |
| Promise lost when a joiner was told and the owner was cancelled before the wire; promise given while `Admitting` | Test: B is told `shape_compiling` while A is `Admitting` behind inference, then A's caller is cancelled. Assert one compile, the same start key and a kept promise. The structure helps too: no request can cancel an operation. |
| Exhaustion-retry `Abandoned` arms forget the compile | Test: cancel the caller during the retry's eviction. Assert the compile still runs. |
| Direct caller at the backstop gets 1000 ms | The backstop test asserts the about 7.64 s prediction. |
| Failed slot holds the gate through replacement restore | Test: hold the replacement start after exit is confirmed. Assert a healthy worker compiles meanwhile. |
| Definite `ERR` holds the gate through drain and restart; prediction ignores failed slots | Test: `compile_failed` while that worker has a lease in flight. Assert another worker compiles at once. |
| Committed orphan evicts again, or starves the queue after a failed eviction | No drivers. Test: a failed eviction plus an expired promise do not block a smaller request that fits. |
| `gate_held` check-then-act race | One owner: there is no second lock acquisition. |
| Handoff lost wakeup | One instant per pass, and the handoff is a timer. Test: a handoff boundary crossed between decision and timer. |
| Panic cleanup depends on the awaiting future | The scheduler awaits the `JoinSet`. Test: abort a direct `lease_until` after reservation, then panic the admission. Assert the cleanup runs and another worker compiles. |
| No wakeup when recovery begins | Recovery is an event. Test: start recovery with no other event and assert the next head is elected. |
| Recovering entries rejoin ahead of accepted bounds | Test: an accepted waiter keeps its bound when X's entry rejoins. |
| Requeue/adopt race | No requeue or adopt: a promise moves with its compile inside one task. |
| Running prediction counts from reservation | Test: hold an admission behind inference and check the prediction right after dispatch. |
| Inner and caller expiry give different delays | One decision per waiter. Test: both timers fire together and the answer equals the recorded promise. |

The first audit's tests stay. Some first-audit findings can no longer happen,
such as two orphan drivers or the gather wakeup race. Their tests stay as
regression checks on behaviour that can still be observed, such as an expired
promise leaving no entry and a head waking at the end of its gathering window.

The hardware kit is unchanged; whether to re-run it is decided after review.

### 6.9 A stopped scheduler closes the lane

If the scheduler's loop ends for any reason (a panic, or its task being
aborted), its pending requests' reply channels close and every later request is
refused at once: requests get `SchedulerStopped`, which the wire maps to
`engine_crashed` with the lane id. The supervisor does not restart the
scheduler. A restarted scheduler would have to rebuild in-flight compiles and
evictions it no longer owns, and a scheduler stopped by a bug would likely stop
again. The lane stays closed until the module restarts, and the module logs one
error.

### 6.10 Choices made while implementing

- **Where the state lives.** Queue entries, slots and learning state stay in the
  supervisor's shared `State` behind its mutex, so the lease fast path,
  statistics and tests can read them. Only the scheduler task changes queue
  entries, waiters, promises and slot lifecycles (states, reservations,
  removal). Requests write a few other fields, always under the same mutex and
  never across an await:
  - their own leases, and the slot's LRU tick (`last_used`) when they lease;
  - `joined`: a cohort member removes itself when it leases through the
    handoff, or when it leaves without leasing;
  - `smallest_rungs`: the serving path records each model's smallest rung,
    which eviction preference reads;
  - `packed_qualified`: packed preparation marks a packed program qualified
    after its output matched.

  These are safe because none of them changes what the scheduler owns: a
  request can only shrink a cohort (and wakes the scheduler when it empties),
  and the LRU tick, rung table and qualification flag only rank or allow a
  later lease of a slot that is already `Resident`. Waiters' reply channels
  live in the scheduler task, so they close if it stops.
- **Where the scheduler runs.** The module's supervisor (`with_lane_lock`) runs
  its scheduler on a thread with its own runtime, because direct-ANE engines each
  have their own runtime and any of them may be unloaded. A supervisor built with
  `new` (tests) runs it on the first caller's runtime, whose paused clock it then
  shares.
- **Exit confirmation.** The worker trait gains `restart_with_exit_signal`,
  which reports the old process's confirmed exit before the replacement starts.
  That is an internal change only. The scheduler gives up on confirmation for
  the gate after `EXIT_CONFIRMATION_GIVE_UP` (35 s). The worker channel waits
  2 s, sends SIGKILL, and allows 30 s in total, and the extra 5 s lets the
  channel's own timeout report first. After giving up, the slot stays charged
  as `Abandoned`, the worker stays closed, and one WARN names the worker and its
  pid.
- **A definite worker error** leaves the slot `Rejected`: charged, but not
  holding the gate, until the worker's recovery confirms its exit.
- **Whose wait the compile bounds.** The request whose turn started a compile
  waits for that compile to finish, as it always has. Its serving task keeps the
  lane's serving gate until the compile's reply has drained, and its caller is
  answered at the deadline through `Expired`. Requests that only joined the
  compile are answered at their own deadline. The 600 s backstop bounds queue
  waits, not a running compile.
- **Compiles kept only for a promise** (nobody waits for them now) are elected
  like any other. While another compile runs they do not evict ahead of their
  turn, so a compile that may expire before it can start never evicts.
- **Requests register when the scheduler receives them.** The window the audit
  found, a retry enqueued but not yet decided, no longer exists. The test for it
  now holds a registered request unpolled across the expiry.
- **The handoff** uses `Ready` answers plus a cohort barrier on the slot, as
  designed. A cohort member leases through the fast path. The barrier ends when
  the last member leases, or at the scheduler's 250 ms timer.

## 7. Cache-aware predictions and prewarm (proposed)

Status: **approved and implemented**, with the changes in 7.7. aned now keeps compiled programs across
restarts (see "Compiled programs survive a worker restart" in
`docs/ane-serving.md`), so a shape it holds loads in about 1–2 s instead of
compiling for 12–230 s. The scheduler does not know that yet. It predicts a
full compile, for example about 114 s for Qwen3@4096, and refuses early with a
long `retry_after_ms` a request that would be served in a second. It also
learns every admission as a compile. After a restart on a cached machine, the
first loads (about 1.5 s against a 10.6 s seed) pull the machine factor down
to about 0.15, and a shape that really is cold is then under-predicted about
seven times.

### 7.1 What the scheduler knows about the cache

**The check.** A shape is cached when aned holds its layers' programs. To ask
aned (`compiledModelExists`), the worker needs each layer's identifier, which
means building the layer graph and hashing its MIL text and weights. Measured
on the M5 Max (release build, Qwen3-sized weights, no Neural Engine use), graph
build plus identifier takes per layer:

| Width | 128–2048 | 4096 | 8192 |
|---|---:|---:|---:|
| ms per layer | 56–67 | 97 | 191 |
| every layer of a Qwen3 shape (28) | 1.6–1.9 s | 2.7 s | 5.4 s |
| first and last layer only | 0.11–0.14 s | 0.19 s | 0.38 s |

Checking every layer of the whole ladder costs about 17 s of worker CPU, close
to what loading the shapes would cost. A cached admission already spends most
of its 1.3–1.8 s on this same work. **Proposal: check the first and the last
layer.** A shape's layers are compiled in order within one admission, so a
cached last layer means the whole shape was compiled once. A gap in the
middle (aned removing single programs) is caught at admission (7.3). The
whole ladder then costs about 1.2 s. The `compiledModelExists` call itself
could not be timed here: production holds the Neural Engine. The hardware kit
times it.

**When it runs.** Once, in the worker's `LOAD`, for every ladder rung, after
the weights are read. Afterwards, knowledge comes from admissions: a shape
admitted successfully since is cached. The scheduler keeps a table
`cached[(model_ref, shape)]` with three values: cached, not cached, unknown.

- Each `LOADED` overwrites the model's row.
- Each successful admission sets the shape to cached, because a compile leaves
  the program in aned's cache.
- Nothing else invalidates the table. aned empties its cache on an OS update,
  which needs a reboot, so the module restarts and loads again. aned can also
  drop programs after 14 days without use or under storage pressure while the
  module runs. Admission handles that case (7.3).

**How it is used.** `State::estimate` returns the **load cost** for a shape
marked cached, and the compile cost otherwise (the compile path is unchanged).
Load costs are learned separately: `measured_loads[(model_ref, shape)]` from
admissions that loaded every layer from the cache. Load samples never update
the machine factor. Until a shape's load has been measured, the load seed is
`layers ×` 0.10 s up to width 2048, 0.15 s at 4096 and 0.30 s at 8192. These
values are the CPU cost above plus aned's 3.5–45 ms load, rounded up: about
2.8 s for Qwen3@512 and 8.4 s at 8192. Start keys use the same estimate, so
cached shapes start first. A cached load still goes through the one-compile
gate, because it may turn into a compile.

### 7.2 Protocol: two optional reply fields, no version bump

- `LOADED` gains `ane_cached_shapes: [usize]`, the ladder rungs whose first
  and last layer aned holds. Omitted when absent.
- `ADMITTED` gains `cache_hit_layers: u32` and `compiled_layers: u32`, the
  counts the worker already logs. Omitted when absent.

Nothing else changes: no new request, no HELLO field, no
`WORKER_PROTOCOL_VERSION` bump, no `EngineIdentity` or fingerprint change.

- **An older worker** sends neither field. Every shape stays unknown and is
  predicted as a compile, and every admission is learned as a compile. That is
  today's behaviour.
- **An older module** reading a newer worker ignores the fields, because
  `WorkerResponse` does not deny unknown fields. A protocol test pins that, so
  adding `deny_unknown_fields` later turns a test red.

The worker needs one function from the vendored binding: whether aned holds a
graph's program. The binding already has `compiled_model_exists` on the
in-memory model, so the new function is a small wrapper next to
`program_identifier`. `VENDORED.md` and the harness pin are updated to match.

### 7.3 A cached load that compiles after all

If aned dropped the program after the check, or the cached copy fails to load,
the worker compiles that layer (`load_program`, unchanged). The prediction
stays correct:

- A slot admitted as cached also records the shape's compile estimate. Once
  its admission has run longer than twice its load estimate plus 1 s, its
  remaining time becomes `compile estimate − elapsed`, but at least one layer.
  Requests queued behind it then get honest predictions within seconds of the
  miss, rather than a load's worth of time each pass.
- On completion, `compiled_layers == layers` is learned as a compile and
  `cache_hit_layers == layers` as a load. A mix is not learned, because it
  measures neither cost. The shape is cached afterwards in either case.
- A request that was accepted on the load prediction and misses its deadline
  is answered `shape_compiling` with a fresh prediction: the existing path.

### 7.4 Prewarm

When a direct-ANE catalog lane becomes ready, that is after its self-check
passed, the module sends the scheduler `Prewarm { worker, model_ref, shapes:
[128, 256, 512] ∩ published ladder }`. The scheduler keeps a per-worker list
and owns it like everything else:

- **Lowest priority.** In a pass, a prewarm shape starts only if no request is
  waiting or queued, no compile entry exists, and the compile gate is free.
  Then it is reserved and dispatched like a compile with no owner and no
  waiter. A request that arrives later for the same shape joins it.
- **No eviction, within the budget.** A prewarm shape starts only if the free
  budget already holds it (`reserved + executables ≤ 100`) and the per-model
  (4) and total (8) shape limits allow it. It never evicts, so it never
  touches a leased shape or one with waiters. Shapes are taken in order, and
  the first one that does not fit ends that lane's prewarm, with one INFO
  line. Qwen3 alone takes 84 executables. With a GTE shape resident (22),
  78 are free, so 128 and 256 prewarm and 512 does not.
- **Delay to requests.** Prewarm never starts ahead of a queued request, so it
  cannot break an accepted bound. A request that arrives while a prewarm
  admission runs waits for that one admission; a running admission cannot be
  stopped without a protocol change. Its prediction counts that admission,
  as for any running compile. On a cached machine that is at most about 2 s.
  On a cold M4 it is at most one small compile: by the seed table about 11 s
  at 128, 14 s at 256 and 21 s at 512, and measured runs varied up to twice
  that. Going smallest first keeps each step short.
- **Cancellation.** Lane unload, pause and retirement all retire the worker
  (`AneServing::retire`). Retirement drops that worker's prewarm list. A
  prewarm admission already running finishes and is then released by the
  retirement, as any running admission is today.
- **Pause and self-check.** `lanes.pause` unloads every Neural Engine lane, so
  prewarm stops with it. After resume, a lane prewarms again when it next
  becomes ready. Prewarm is queued only after the self-check, so the self-check
  never waits behind it, and a lane whose self-check fails is never prewarmed.

On a cached machine, the three Qwen3 shapes load in about 5 s. On a cold
machine they compile in about 46 s by the M4 seed table, before traffic needs
them.

### 7.5 Out of scope

No fingerprint, numeric profile or `REVISION` change. The bucket ladder, the
100-executable budget and the shape limits stay as they are. Packed programs
are not checked against the cache and are predicted as today.

### 7.6 Tests (each fails on the base commit first)

| Test | Asserts |
|---|---|
| `cached_shape_is_predicted_at_load_cost_and_not_refused_early` | a Qwen3@4096 request with a 30 s deadline, on a worker whose `LOADED` reported 4096 cached, waits and is served, not refused with about 114 s |
| `cached_load_that_compiles_is_repriced_as_a_compile` | the fake reports cached but compiles: after the miss grace, a queued request's prediction includes the compile, and only a full compile is learned as one |
| `cached_loads_do_not_change_the_machine_factor` | after three cached admissions, an uncached shape is predicted at its seed |
| `prewarm_never_delays_a_queued_request_or_evicts` | with a request queued, prewarm does not start; with a resident leased shape and a short budget, prewarm skips instead of evicting |
| `prewarm_stops_when_the_lane_retires_or_pauses` | retirement during prewarm leaves no further prewarm admissions |
| `older_worker_without_cache_fields_keeps_full_compile_predictions` | `LOADED` and `ADMITTED` without the fields: predictions, refusals and learning are today's |
| `admitted_and_loaded_decode_with_and_without_cache_fields` (synapse-core) | round trip and unknown-field tolerance |
| worker: `loaded_reports_the_cached_ladder_rungs` | with a fake binding, only rungs whose first and last layer are cached are listed |

Each guard also gets a mutation row. The hardware kit (M6 mini) restarts a
worker on a cached machine and checks that `LOADED` lists the cached rungs, that
a Qwen3@4096 request with a 30 s deadline is served, and how long prewarm
takes. It also times `compiledModelExists`.

### 7.7 Changes during implementation

- **Packed preparation outranks prewarm.** The tight-packing lane tests showed
  packed preparation being turned away behind a prewarm admission and never
  retried, because a prewarm is not request activity. Preparation now treats
  a running prewarm admission as no obstacle: it waits for it rather than
  giving up, and prewarm starts nothing while preparation waits.
- **A prewarm admission is never withdrawn.** A request that joins a prewarm
  admission and is cancelled, or whose `shape_compiling` promise lapses, does
  not withdraw it, because nobody asked for it in the first place. A resource
  refusal ends the prewarm without evicting anything to retry.
- **What the scheduler keys cache knowledge by.** `(model_ref, shape)`, not the
  worker: aned files programs under the worker binary's signing identifier,
  so every worker of a model shares them. Admissions of packed programs are
  neither checked nor learned as loads.
- **Logs.** The worker logs one line per cache check; the module logs one INFO
  line when a lane's prewarm ends (see `docs/ane-serving.md`).
- **Tests.** The table in 7.6 maps to
  `crates/synapse-module/src/worker_host/compile_cache_prewarm_tests.rs`, with
  `prewarm_never_delays_a_queued_request`,
  `prewarm_waits_for_a_request_in_its_gathering_window`,
  `prewarm_skips_what_the_free_budget_cannot_hold_and_evicts_nothing`,
  `prewarm_refused_for_resources_evicts_nothing`,
  `prewarm_stops_when_the_lane_retires`, the two withdrawal tests and
  `prewarm_loads_cached_shapes_without_compiling`. The protocol tests are
  `cache_fields_are_optional_in_both_directions` and
  `worker_side_decoding_ignores_unknown_fields` in `synapse-core`. The worker
  test is `loaded_reports_the_cached_ladder_rungs`. Mutation rows are in
  `mutations.d/ane-direct-cache-prewarm.toml`.
- **Hardware check on the M6 mini** (macOS 27.0, two repeats, run
  20261022T193842Z, at `4f8d623d` with the kit's test-only patch).
  `compiledModelExists` took 0.08–0.46 ms per layer, and the whole LOAD-time
  check 352–364 ms for Qwen3's seven rungs.
  - Cold case: prewarm compiled 128 in 37.6 s. A Qwen3@1024 request asked
    during it started right after that one admission (its compile took
    131.7 s). Then 256 was prewarmed, and 512 was skipped for budget.
  - Cached case: prewarm loaded 128, 256 and 512 from aned in 3.5 s in total.
    A Qwen3@4096 request with a 30 s deadline was served in 1.9 s, not
    refused.
  - Every `LOADED` report matched what the admissions then found in the cache.
  - Two runs failed on the kit's own expectations, not on the product, and
    the kit has since been corrected. The first cached run expected 512 cached
    after the cold run had skipped it. In the second cold run, the kit's
    purge reported every layer gone, yet a fresh worker found and loaded all
    of them from the cache. Why the purge did not take there is unresolved.
