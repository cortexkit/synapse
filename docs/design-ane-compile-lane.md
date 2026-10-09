# Direct Neural Engine lane: compile waits, lane lifecycle and status

This note covers how a catalog lane serves requests while a first-time Neural
Engine shape compiles, which lock protects what, and what a client sees. It
applies to catalog lanes in `crates/synapse-module` (`lib.rs`) and to the
direct-ANE residency code in `worker_host/mod.rs` (`ane_residency`).

## Facts the design starts from

- **The worker is serial.** `request_loop`
  (`crates/synapse-worker-ane-direct/src/worker.rs`) reads one frame at a time
  and runs `Model::admit` synchronously, and a first-time compile builds every
  layer program (11-27 s per Qwen3 shape on an M4). The host's
  `AneWorkerChannel::exchange_inner` holds the one stream mutex from writing
  the request until the reply has been read, and shape admission uses that same
  exchange. **While a shape compiles, the worker cannot answer any inference,
  not even one for a resident shape.** Running compile and inference
  concurrently would need an asynchronous admit protocol, plus a hardware test
  of compiling while executing. That is out of scope here.
- **Neural Engine work stays serialised.** There is one worker per direct-ANE
  lane and one exchange at a time on its stream. The module-wide execution
  permits (`max_concurrent_workers`) limit module concurrency; they are not
  hardware streams.
- **The residency supervisor already single-flights.** A slot in `Admitting`
  makes every later request for that shape wait, and they lease it once it is
  `Resident`. Queue waits are bounded by the request deadline and by the 600 s
  admission timeout. Admit and evict exchanges run on their own tasks, so their
  accounting completes even if the requester goes away.
- **A stream wait can restart the worker today.** `exchange_sequences` wraps the
  stream-lock wait and the I/O in one 30 s timeout. A timeout there is a
  `Channel` error, and that error triggers worker recovery, which restarts the
  worker and drops every compiled shape. Today the lane mutex keeps a second
  request from ever waiting on the stream, so this never fires. Once requests
  can queue on the stream, it would.

## What each lock protects after the change

| Lock | Protects | Held by |
| --- | --- | --- |
| Catalog lane mutex (`catalog_lane_lock`) | Lane lifecycle: verify and load, the numerical self-check, unload, failed-load acknowledgement | Resolver readiness, self-check, unload. **Never by serving inference.** |
| Serving gate (new, per lane, `tokio::sync::RwLock<()>`) | In-flight lifetime: no unload, reload or self-check while an inference is running or draining | Serving holds a READ guard until the worker reply drains. Self-check holds the WRITE guard. Unload uses `try_write`. |
| Lane execution mutex (new, per lane, non-ANE engines only) | Same-lane serialisation for engines without internal concurrency (owned Metal, supervised worker engines, the test engine) | Serving on those engines, from before the permit until the engine call returns |
| Execution permit | Module-wide concurrency (`max_concurrent_workers`) | Non-ANE engines: around the engine call, as today. Direct-ANE: around each inference exchange only (see below). |
| Supervisor queue and lease | Residency budget, single-flight admission, no eviction of a leased shape | Direct-ANE requests, one rung at a time |
| Worker stream mutex | One exchange at a time with the worker process | Each admit, evict, inference, restart and shutdown exchange |
| Per-user direct-ANE advisory lock | Machine-wide ownership of the Neural Engine | Unchanged |

Invariant: **no path acquires the lane mutex while it holds a serving-gate
guard.** Serving takes its gate guard inside `execute_*`, after resolution and
certification have finished and released the lane mutex.

## Lock and await order, per path

Waits marked *(deadline)* are bounded by the request's absolute deadline.

### Serving, catalog lane, direct-ANE

Before:
1. Resolver: lane mutex, by `try_lock`, else a wait of at most min(budget, 5 s).
   If that wait expires, the answer is `model_loading`, or `deadline_exceeded`
   for budgets of 5 s or less. Readiness is checked under the mutex, which is
   then released.
2. Profile certification (`ensure_profile_preload_ready`): lane mutex (no time
   limit), check status, release.
3. `execute_embedding`/`execute_rerank`: lane mutex (no time limit), then the
   permit *(deadline)*. A detached task takes ownership of both.
4. In the detached task, for each rung:
   1. Supervisor queue ticket, strict FIFO.
   2. If an eviction is needed: the evict exchange (stream lock).
   3. If the shape is not resident: the admit exchange (stream lock, compile).
   4. Lease, then the inference exchange: stream lock plus I/O, together inside
      the 30 s timeout.
   5. Release the lease.
5. When the task ends: release the permit and the lane mutex.

After:
1. Resolver: unchanged. The lane mutex is now contended only by real lifecycle
   work, so `model_loading` means "loading or self-checking".
2. Profile certification: lane mutex, waited for only until the request
   deadline (answered `model_loading` when it expires, as the resolver does).
   It is held only briefly unless the self-check itself runs.
3. Serving-gate READ guard *(deadline; `deadline_exceeded` when it expires)*.
   No lane mutex. No permit yet.
4. Detached task (owns the read guard), for each rung:
   1. Supervisor step *(deadline)*. A resident shape is leased at once, or the
      request joins the in-flight admission of its shape (single-flight), or it
      queues FIFO for budget and then admits or evicts.
   2. If admitting: the admit exchange (stream lock, compile), with no permit
      held.
   3. Lease.
   4. Stream lock *(deadline)*. Giving up here is safe: nothing has been
      written, so there is no fault and no restart.
   5. Execution permit *(deadline)*.
   6. Inference I/O, inside the 30 s timeout, which now starts only after the
      stream lock is held.
   7. Release the permit, the stream and the lease.
5. When the task ends (after the reply drains, even if the caller has gone):
   release the read guard.

### Serving, catalog lane, other engines (owned Metal, supervised worker engines, test)

Before: lane mutex, then permit *(deadline)*, then the engine call, then
release both.
After: gate READ guard *(deadline)*, then the lane execution mutex *(deadline)*,
then the permit *(deadline)*, then the engine call, then release all three.
Permit count and holding time are the same as today.

### Self-check (profile numerical check; non-profile catalog check)

Before: lane mutex (no time limit). The reference cases then execute through
the serving path, reusing that mutex guard: permit, supervisor lease or admit,
stream, then release.
After:
1. Lane mutex.
2. Gate WRITE guard. This waits for in-flight readers to drain. Tokio's
   `RwLock` is fair, so new readers queue behind it.
3. The reference cases execute through the serving path, passing the held
   write guard in place of a read guard: supervisor, stream, permit, I/O.
4. Release the write guard, then the lane mutex.

The non-profile 2000 ms check keeps both guards in `CatalogInvocation` until its
blocking task ends.

### Unload

Before:
1. Lane mutex `try_lock`. Failure returns `model_in_use`.
2. `strong_count > 2` returns `model_in_use`.
3. Slot set to `Unloaded`.
4. Engine unload under the guard: the supervisor `retire` waits for leases to
   drain, then channel shutdown takes the stream lock.

After:
1. Lane mutex `try_lock`, then gate `try_write`. Either failure returns the
   same `model_in_use` error as today.
2. `strong_count > 2` returns `model_in_use` (unchanged).
3. Slot set to `Unloaded`.
4. Engine unload with both guards held: `retire`, then the stream lock and
   shutdown.

### Reload (load after an unload or a failed load)

Before and after: the resolver holds the lane mutex, verifies the files, loads
the engine (control-load permit), and marks the slot `Ready`. Reload takes no
gate guard. A lane with no loaded model has no readers, because the unload that
removed the model only proceeded after `try_write` succeeded. Every in-flight
task, including those of cancelled callers, holds its read guard until its
reply drains.

## Why the direct-ANE permit after the stream lock cannot deadlock

Question: can a request hold a worker's stream lock while it waits for a permit
that another request holds while it waits for that stream?

No. After the change, a permit holder is one of two things:
- A non-ANE engine call. It never touches a direct-ANE stream, the supervisor
  or the serving gate. It finishes and releases the permit.
- A direct-ANE inference exchange. It takes the permit only after it already
  holds its own worker's stream lock, then waits only on that worker's I/O.

No path holds a permit while it waits for a stream lock. `lib.rs` no longer
takes a permit before calling `infer_guarded` for direct-ANE. Admit, evict,
restart and shutdown exchanges hold the stream but take no permit. So every
wait-for edge points from stream to permit and none from permit to stream, and
there is no cycle.

The other edges are acyclic too:
- Supervisor steps run before the stream lock and hold no permit.
- An admission that evicts on another worker waits for that worker's stream
  while holding neither a stream nor a permit.
- Recovery and `retire` wait for this worker's leases. A lease holder is either
  in an exchange, which finishes, or between rungs, holding nothing it waits
  for.
- Self-check and unload hold the lane mutex and the gate; serving never waits
  for the lane mutex while holding the gate.

Full order: lane mutex → serving gate → (lane execution mutex | supervisor
queue/lease → worker stream) → execution permit.

Waits are short in practice:
- Holding a stream while waiting for a permit lasts only as long as some other
  permit holder runs, and it is bounded by the request deadline.
- A request waiting behind a 27 s compile holds no permit, so other engines
  keep both permits for the whole compile. Today a second same-lane request
  waits on the lane mutex without a permit; this keeps that property.

Visible effect on `in_flight` statistics: a compile no longer counts as an
in-flight execution. The cancelled cold-admission test
(`direct_ane_cold_admission_after_absolute_deadline_is_drained_without_inference`)
asserts `in_flight == 1` during the compile. It changes to assert that the
serving-gate guard, not a permit, is retained until the admission drains.

## Module permits for other engines

They are unchanged: the same count, taken at the same point (just before the
engine call), held for the same span. The only difference is that the per-lane
serialisation those engines need comes from the lane execution mutex instead
of the lifecycle mutex. Non-catalog models never took a lane lock and still
don't.

## The `shape_compiling` error

Wire shape. It uses the same module vocabulary as `ane_resources_exhausted`;
there is no core `StableErrorCode` change.

```json
{"code": "shape_compiling", "class": "transient", "retry_after_ms": 1000,
 "safe_to_retry_same_request": true,
 "message": "lane '<lane>' is compiling Neural Engine shape <n>",
 "details": {"lane_id": "<lane>", "shape": <n>}}
```

`shape` is the padded shape being compiled.

**When a request waits and when it gets the error.** A request never fails
early because a compile is running. It waits as long as its own deadline
allows, and it is answered `shape_compiling` only if that deadline expires
while it is blocked on a compile. Blocked on a compile means one of:
1. It is the request whose miss started the compile.
2. It is waiting for another request's in-flight compile of the same shape
   (single-flight).
3. It is waiting for the worker stream while an admit exchange holds it.

A deadline that expires on any other wait (permit, ordinary inference ahead of
it, eviction) keeps `deadline_exceeded`. On expiry, the compile continues in its
own task: the shape becomes resident, the budget is accounted, and no inference
is dispatched for the expired request. A retry finds the shape resident, or
joins the same compile. It never starts a second compile.

**What AFT sees**, for an AFT request on a loaded direct-ANE lane while another
request compiles shape S:
- **AFT's request uses a resident shape R.** It leases R at once, so R cannot be
  evicted while it waits, then it waits on the worker stream.
  - If S finishes inside AFT's deadline: success, after the rest of the compile
    plus its own inference.
  - Otherwise, at AFT's deadline: `shape_compiling`, `details.shape = S`.
- **AFT's request needs S itself.** It joins the in-flight compile.
  - If S finishes inside its deadline: success.
  - Otherwise, at its deadline: `shape_compiling`, `details.shape = S`.
- **In neither case** does AFT see `model_loading`, and neither request restarts
  the worker. `model_loading` stays reserved for a lane that is actually loading
  or self-checking (the resolver's 5 s lifecycle wait, unchanged).

Today AFT instead sees `model_loading` after 5 s if its budget is above 5 s,
and `deadline_exceeded` otherwise.

A consumer that retries any transient error after `retry_after_ms` needs no
change. A consumer that retries only on `model_loading` must add
`shape_compiling`.

`docs/wire-contract-v1.md` changes:
- The `model_loading` bullet gains: "Never returned because another request is
  executing or a Neural Engine shape is compiling."
- A new bullet documents `shape_compiling`, with the shape above and this
  consumer disposition: the model is loaded; retry after `retry_after_ms`; a
  deadline that covers a first-time compile (about 30 s) avoids it.

## Eviction: soft-pin of the smallest rung (option A)

Rule: when the supervisor needs a victim, it skips each model's smallest
published rung as long as another unleased resident shape would free enough
budget, and evicts that rung, as today, only when it is the sole evictable
shape. This applies to both the budget eviction and the worker-reported
resource-exhaustion eviction.

Budget arithmetic (100 executables; per-model limit 4; total limit 8):
- **Qwen3, 28 executables per shape.** floor(100/28) = 3 resident shapes (84
  executables). With the pin, 128 stays, and 256, 512 and 1024 or larger rotate
  through 2 slots. Admitting 1024 now evicts the LRU of 256 and 512, so the next
  mid-size chunk batch pays an 11-15 s recompile instead of the next short query
  paying 10.7 s for 128.
- **gte, 22 executables per shape.** floor(100/22) = 4 (88), which is also the
  per-model limit. 128 stays, and the other rungs rotate through 3 slots.
- **Two Qwen3-sized lanes sharing the budget.** 2 pinned plus 1 rotating, still
  84.

If the pinned shape is the only evictable one, for example Qwen3 with 256 and
512 both leased while 1024 is admitted, it is evicted exactly as today.
Liveness and the budget are unchanged.

A shape another request is about to use is protected by that request's lease.
A request for a resident shape leases it immediately, before it waits for the
stream. The supervisor queue therefore applies FIFO only to requests that need
budget (admit or evict). Resident-shape leases and single-flight waiters do not
hold the queue head and do not wait behind it. Starvation guard: while the head
is waiting for a victim and none is free, new resident-shape leases fall back
to FIFO.

## Test (d): unload, reload and self-check against in-flight inference

What it asserts, on a mock direct-ANE lane with the worker reply held open by
the mock's inference gate and the caller cancelled (only the detached task
holds guards):
1. `model.unload` returns `model_in_use` while the reply has not drained. The
   engine's unload is not called.
2. A self-check started at the same time queues for the serving gate. A
   request that passed certification earlier then waits at the gate, is
   answered `deadline_exceeded` at its deadline, and never reaches the worker
   (no exchange is issued while the check holds the lane).
3. Once the reply drains, the self-check runs and unload succeeds.

Reload needs no separate assertion: it only runs on a lane whose model was
unloaded or failed to load, and assertion 1 shows unload cannot proceed while
any inference is in flight.

A serving request that waits at the gate gets `deadline_exceeded`, not
`model_loading`, because admitted batch jobs must never fail with
`model_loading`. Inline requests meet a running self-check earlier, in
certification, which answers `model_loading` at their deadline.

**On the base commit this test fails at assertion 2** (the request waits on
the lane mutex past its deadline). Assertion 1 already holds there, because
today's lane mutex serialises unload against inference. Its non-vacuity on
the new code is shown by mutation. Each mutation is staged, applied, run and
restored per the repository's mutation procedure:
- **Unload skips the serving-gate `try_write`.** Expected red: assertion 1.
- **The self-check skips the write guard.** Expected red: assertion 2.
