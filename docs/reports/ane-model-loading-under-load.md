# A Ready ANE lane refused as `model_loading` under load

## Scope and citation key

This is a source-and-retained-evidence investigation, **not a fix or a packing
rollout approval**. Source was read before the measurement documentation. No
hardware experiment, SSH command on the M4 Mac mini, production-store access,
or module/worker change was made.

Source citations below are pinned to `0c5ee1435d7888309ece679c89c05da98be96147`:

- **S** = `crates/synapse-module/src/lib.rs`.
- **W** = `crates/synapse-module/src/worker_host/mod.rs`.
- **C** = `crates/synapse-core/src/error_contract.rs`.

The first measurement used `b755101caef11f00dff9c3dd916fad93c01560fc`. Comparing
that revision with the source baseline shows only the six-line `manifest_json`
addition in S; W and C are unchanged. The cited serving behavior therefore
matches the first measurement, although late S line numbers differ by six.

Evidence citations use the branch containing the retained measurement report
and compact summaries of the first and instrumented comparisons,
`alfonso/task/bg_dispatch_5be382bda09a8fa0db76f9fc-m4-measurement-report`, pinned
at `2b79a25fd6afb5511a6ded06b4e48888ac1959b5`:

- **E** = `docs/evidence/ane-tight-packing/README.md` on that revision.
- **F** = `docs/evidence/ane-tight-packing/module-serving-first.json` there.
- **I** = `docs/evidence/ane-tight-packing/module-serving-instrumented.json` there.
- **R/** = the retained local archive `/tmp/m4-packing-evidence-5be382bd/`.
  Raw citations identify the file and line; these logs are outside Git. Only
  operation names, counts, identities and timings are reproduced here, not
  input text, token IDs or vector payloads.

## Conclusions, ranked by confidence

1. **Very high: the refusal came from the catalog resolver's five-second
   lifecycle/readiness timeout family, not a 30-second worker I/O timeout.**
   The exact terminal message, `catalog lane 'qwen3-embedding-0.6b-ane' is
   loading`, is constructed by `catalog_loading_response` for budgets above
   5,000 ms. Both the lifecycle-lock wait and the later detached readiness-task
   wait call it. Neither checks that the lane is actually loading when choosing
   that message (S:29577-29665; R/first-run.log:30). The private inline budget
   was 600 seconds, not five (E:716-721).
2. **High, with a correlation limitation: the recorded sequence points to
   the detached readiness-task timeout, rather than waiting five seconds for
   the lifecycle mutex.** The final resolver record says the mutex was acquired
   in **0.004083 ms**. It is followed about **6.544 s** later by the last
   `embed.batch` handler completion, without a `batch_resolve` record for that
   new request. The source emits `resolver_lane_wait` only after acquiring the
   mutex; a mutex-wait timeout exits before that record. See the timeline and
   its request-ID limitation below (S:29609-29629, 8905-8906;
   R/first-failed-arm/worker-profile/module-56298.jsonl:465-472).
3. **Very high: ordinary slow inference does not hold the catalog lifecycle
   mutex; idle packed preparation does not acquire it or a write serving
   gate.** A request waiting behind a slow exchange is not, solely for that
   reason, waiting on the five-second resolver lock. Serving holds a shared
   gate and serializes direct-ANE work on the worker stream. Packed preparation
   also takes only a nonblocking shared gate. At the execution deadline the
   stream wait reports `deadline_exceeded`, or `shape_compiling` if admission
   overlapped the wait, not `model_loading` (S:1521-1537, 12815-12843,
   12974-13007, 27487-27499; W:5625-5700).
4. **High: a late numerical self-check or packed re-preparation is not a
   supported explanation of this incident.** The retained check is passed at
   generation one, and serving's passed-check path starts no check. Successful
   packed dispatch preceded the refusal; the trace has one packed admission,
   no packed re-admission and no `ERR`. Preparation stops retrying once it has
   qualified the current model. This does not amount to a complete historical
   crash ledger (S:1501-1511, 1539-1547, 29343-29350, 29426-29463;
   R/first-failed-arm/events.json:9-29; E:688-696).
5. **High: packing off does not eliminate the resolver timeout hazard.** All
   inline catalog requests traverse the same five-second lock/readiness wrapper
   before the packed-versus-standalone execution branch. Production's default
   budget is 30,000 ms, so this wrapper can return `model_loading` at five
   seconds even for an already loaded lane. **But a slow standalone ANE
   exchange alone is not a demonstrated causal chain to that wrapper's
   timeout**: readiness/scheduling must also be delayed, or genuine lifecycle
   work must own the mutex (S:205, 8853-8865, 29577-29665, 12980-12999).
6. **Medium: system paging is a plausible contributor to delayed work, not
   a proven root cause of the first refusal.** The roughly **109.156 GiB** of
   swap-in growth was measured in the *second*, instrumented comparison. The
   first comparison had no memory samples. The second termination was an
   NDJSON-reader EOF, not another recorded `model_loading` refusal
   (E:725-732, 773-783, 809-837, 856-874).

## 1. Request flow, lock ownership and wait budgets

For the measured inline catalog `embed.batch`, the relevant order is:

```text
preflight / alias lookup
  -> resolve_serving_model
       -> catalog_resolution_guard (lifecycle mutex; min(budget, 5 s))
       -> spawned ensure_catalog_lane_ready_owned (remaining five-second budget)
  -> token composition / ensure_profile_request_certified
       -> ensure_profile_preload_ready (remaining absolute request deadline)
  -> admission / engine quanta
       -> shared lane_serving_gate
       -> direct-ANE worker-stream wait / permit / inference
```

Resolution precedes certification and inference (S:8802-8824, 8849-8884,
8905-8937, 8990-9016). This ordering matters: the resolver can mislabel a timeout
before the request ever joins the worker-stream queue.

### Readiness paths that can answer `model_loading`

| Path | Budget and returned error | Relation to an already Ready lane |
| --- | --- | --- |
| `resolve_serving_model` -> `catalog_resolution_guard` | `try_lock_owned`, then at most `min(budget, 5000)` ms for the lifecycle mutex. Expiry calls `catalog_loading_response`: `model_loading`, retry-after 250 ms, when the **original budget > 5000**; otherwise `deadline_exceeded`. Zero budget takes the latter immediately. | No Ready fast path before this lock for a reserved catalog lane. A self-check, unload or another resolver can hold the mutex despite the lane having been loaded. S:29577-29616, 29633-29665. |
| `resolve_serving_model` -> detached `ensure_catalog_lane_ready_owned` | After lock acquisition, `remaining = min(budget, 5000) - elapsed`, saturating at zero. `timeout(remaining, handle)` returns the same loading response on expiry; a task join failure instead returns `engine_crashed`. This is one combined lock/readiness budget, **not five seconds for each**. | Even a loaded, profile-backed lane is sent through a new spawned task. That task computes the self-check key and projection before retrieving the loaded model and returning it. Async scheduling and synchronous profile/store work are within this wait; no worker exchange is required. S:29614-29630, 29820-29836, 29924-29933; projection uses a store connection at S:28899-28925. |
| `ensure_profile_request_certified` -> `ensure_profile_preload_ready` | Caller waits on a shared watch outcome until its own absolute deadline. Both timeout and a completion observed at/after that deadline return `model_loading`, retry-after 250 ms, with **`is running its numerical self-check`**, not the observed `is loading` message. Without a deadline it waits without an inner cap. | A slot can be Ready while numerical certification is still pending. A passed check returns immediately; no lifecycle lock is taken on that fast path. S:13697-13716, 29343-29390, 29392-29405. |
| `join_lane_check` -> `run_lane_check` | The single shared task waits without an inner timer for the lifecycle mutex. It returns `model_loading`, retry-after 250 ms, with **`model '…' is not loaded`** if the slot is no longer Ready/loaded when it gets the lock. | A lane that was Ready when the caller began may have been unloaded meanwhile. This is not an error for a slot still Ready at that check. S:29426-29489. |

A timeout of the resolver's spawned `JoinHandle` stops the caller's wait, not the
underlying task: the task owns the lifecycle guard (S:29618-29623,
29812-29816). A delayed Ready-path task can therefore keep that guard and make
other resolvers wait without any LOAD or self-check inference. This is a
possible chain, not a measured owner trace.

### Other locks and preparation: not five-second `model_loading` waits

| Work | Ownership and budget | Error / consequence |
| --- | --- | --- |
| Profile numerical self-check | `run_lane_check` owns the lifecycle mutex. `check_profile_model` rechecks the passed status, then takes a **write** serving gate and runs all reference cases through normal execution, reusing that guard. The task has no overall caller deadline; reference executions pass `None`. S:29466-29487, 29513-29558, 29239-29267. | A pending check can wait for slow in-flight readers, and then prevent new readers. Callers waiting for certification may get the self-check `model_loading` above; callers resolving behind its lifecycle guard may get the generic five-second message. Once passed, it does not take the write gate again merely because inference is slow. |
| Non-profile owned numerical check | `ensure_catalog_lane_ready_owned` returns immediately for a passed projection; otherwise it takes a write gate and holds lifecycle + gate in `CatalogInvocation`. The blocking check's result wait is **2000 ms**, after acquiring the write gate. S:29931-29957. | Its own timeout returns `engine_crashed` (`catalog self-check exceeded 2000 ms`), not `model_loading`; detached blocking work can retain the guards. An outer resolver or a resolver waiting behind it can still hit the five-second wrapper. S:29988-29999. This is not Qwen's profile-backed check. |
| Serving gate / execution mutex | Shared gate, `timeout_at(request/job deadline)` or no inner cap if `None`. Non-direct-ANE catalog engines also take the execution mutex under that deadline; direct ANE skips it. S:12815-12843, 27532-27573. | Expiry is `deadline_exceeded`, never a fresh `model_loading`. Serving does **not** own the lifecycle mutex. |
| `model_unload` | Takes lifecycle `try_lock_owned` and serving **`try_write_owned`**, without waiting. Guards are retained through blocking unload. S:6102-6124, 6166-6171. | If busy, returns `model_in_use`; does not queue a writer behind active inference. Loading-state checks return `model_loading` only for a slot in a loading state. |
| `ActivityGuard::drop` -> `prepare_when_idle` | When activity reaches zero, calls the retry scheduler; scheduler spawns work, confirms the same Ready model, and uses **`try_read_owned`** on the serving gate. No lifecycle lock and no module-level request deadline/five-second timer. S:1483-1549, 1577-1596. | Can admit/qualify a packed shape on the worker stream, not write-lock the lane. Failures/defer log a warning and leave standalone serving available. `finished=true` suppresses subsequent idle retries for that model. |
| Packed worker preparation | Has no caller deadline: standalone fixture inference and packed qualification pass no deadline. Shape admission's stream queue is uncapped internally; once holding the stream, shape RPCs have a **600-second** timeout. Qualification inference I/O uses the worker inference timeout (normally **30 seconds**). W:5449, 5526-5559, 5625-5700, 6494-6590. | A concurrent request is still bounded by *its* deadline; if compile/admission overlapped its wait, the result is `shape_compiling`, not `model_loading`. |
| Packed scheduling after a load | A separate task briefly acquires and **drops** the lifecycle mutex before checking persisted certification and starting preparation. It has no inner timer. S:6458-6473. | This is a lifecycle-lock user while the slot is already Ready, but the guard is not held across packed compilation. It is distinct from the idle-transition retry and requires a load completion. |

The lifecycle mutex has no read/write modes. Only the serving gate is an RW
lock. Its production writers in S are profile self-check, non-profile numerical
self-check, and unload; packed preparation is a reader. The lock-order comment
at S:27487-27499 describes these roles, but the actual acquisition sites above
are the basis of this conclusion.

Large catalog batches diverted before inline resolution use a different route:
`items.len() > inline.max_items` -> `submit_catalog_embed_job`
(S:8825-8847). `ensure_catalog_lane_ready` waits for the lifecycle mutex under
the **job's absolute deadline**, checks expiry again after acquiring it, and
reports `deadline_exceeded`; profile certification has no inner timeout there
because the outer job timer bounds readiness (S:29701-29749, 30090-30145).
This distinction should survive a deadline-handling correction: these jobs are
already admitted, so their readiness wait consumes their execution budget rather
than returning a pre-admission retry signal. Their timeout should remain
`deadline_exceeded`, not a fresh readiness `model_loading` response.

### Completeness: remaining `model_loading` constructors in S

The table above covers the catalog resolver and profile-check constructors.
The remaining constructors are not evidence of an inference-induced Ready
lane loading transition. Retry delays below are **not wait budgets**.

- S:5810-5814: poisoned runtime-catalog mutex during registration ->
  `model catalog state is unavailable`, retry-after 100 ms; no timed wait.
- S:5903-5909: failed model-load job status with no stored typed error ->
  fallback `model_loading`, retry-after 100 ms; no wait and not an embed error.
- S:6086-6100 and 6131-6144: unload rejects resolving/downloading/validating/
  loading states, retry-after 250 ms; neither branch matches a Ready snapshot.
- S:6280-6300: legacy/nonreserved resolver returns a Ready loaded model
  immediately. Loading states or a newly started background load return
  `model_loading`, retry-after 250 ms, without waiting.
- S:6341-6398: nonreserved profile/control load path uses the specified budget,
  or `MODEL_LOAD_CONTROL_TIMEOUT_MS` = **900,000 ms** by default
  (S:208, 6338-6339). Ready returns immediately. Slot-notification wait expiry
  produces `model_loading` with the timeout as its retry delay. Failed slots
  forward the stored error, which can itself be `model_loading`.
- S:6404-6415: a closed control-load semaphore returns `model_loading`,
  retry-after 100 ms. Acquisition has no own timeout; caller wrappers bound
  their wait. This path starts a load, not warm inference.
- S:7644-7678: `transient_model_load_error` returns `model_loading`, retry-after
  1000 ms, for cache/download/JSON/lease/I/O failures. It is also used for CUDA
  evidence task failure (S:7620), model-load completion failures
  (S:7922, 7947), and poisoned owned-engine mutex in the non-profile check
  (S:28978-28980). None constructs the observed `catalog lane '…' is loading`
  message. Native check errors are normalized to `engine_crashed` by its
  result arm (S:29988-29999); catalog engine-load errors are likewise
  normalized at S:6477-6485.

Stored errors can be forwarded by load/status paths, including
`catalog_failed_load` (S:29667-29685). That requires a Failed slot, not the
Ready-serving premise. Direct-ANE worker errors are another forwarding route:
S:6964-6978 preserves an arbitrary `WorkerErr.code` (W:3670-3677). If a worker
actually sent `model_loading`, a Ready lane could return that code, but with the
worker-error message and permanent/no-retry classification, not the resolver's
loading response. There were no `ERR` responses here. Tests containing
`model_loading` assertions are not additional production constructors.

## 2. Which path fired in the first comparison?

### Correlated timeline

All times below are UTC on 2026-10-11. The first arm's module PID is 56298;
the retained worker profile is `worker-56326.jsonl`.

| Time | Observation | Citation |
| --- | --- | --- |
| 14:26:12.516 | Private module starts the third packed arm. | R/first-failed-arm/synapse.2026-10-11.log:1. |
| 14:26:49.403 | Persisted numerical check is passed, generation 1; next millisecond the warm-up is admitted. | R/first-failed-arm/events.json:19-21; R/first-failed-arm/synapse.2026-10-11.log:2. |
| 14:27:49.813 | First packed admission completes; later packed exchanges prove qualification and dispatch occurred. | R/first-failed-arm/worker-profile/worker-56326.jsonl:161; R/first-failed-arm/wire.jsonl:95-103. |
| 14:27:51.281 | `inline-1-1` completes in 844 ms. | R/first-failed-arm/synapse.2026-10-11.log:5. |
| 14:28:14.659 / 14:28:32.325 | Jobs complete in 18,452 / 13,900 ms: long job durations precede the later loading refusal. | R/first-failed-arm/synapse.2026-10-11.log:18,22. |
| 14:28:32.466592 | The next `embed.batch` is decoded, after the previous handler's completion at 14:28:32.329221. | R/first-failed-arm/worker-profile/module-56298.jsonl:464-465. |
| 14:28:36.989349 | Still-running `inline-1-8` records a 4,661.794292 ms direct-ANE roundtrip. An earlier roundtrip was already 5,859.078542 ms **without terminating that call**. | R/first-failed-arm/worker-profile/module-56298.jsonl:439,466-467. |
| 14:28:36.992402 | New resolver's lifecycle-lock record is only 0.004083 ms. | R/first-failed-arm/worker-profile/module-56298.jsonl:468. |
| 14:28:43.536383 | Last `embed.batch` handler interval is 11,069.915166 ms; the archive ends with the loading refusal. No subsequent `batch_resolve`/admission for that new call is recorded. | R/first-failed-arm/worker-profile/module-56298.jsonl:469-472; R/first-run.log:30. |
| 14:28:45.474414 | Last recorded worker exchange, `embed-101`, successfully returns vectors, worker wall 1,872.109209 ms (relay wall 1,872.184875 ms). | R/first-failed-arm/worker-profile/worker-56326.jsonl:241; R/first-failed-arm/wire.jsonl:103. |

The profile-to-profile gap after acquiring the resolver mutex is
`1791728923536383 - 1791728916992402 = 6,543,981` microseconds. The decode-to-lock
interval is another 4,525,810 microseconds. The handler thus includes time before
entering the five-second resolver budget as well as delayed observation of its
timer; **11.07 seconds is not a second configured readiness timeout**. Timers
bound async waits when polled, not preemptively every synchronous operation or
an unscheduled process. The request's preflight and alias lookup precede this
resolver timer (S:8802-8824, 8849-8861, 29608-29623).

**Best-supported attribution:** `resolve_serving_model`'s
`timeout(remaining, handle)` arm at S:29623-29629. A pure lifecycle-lock timeout
would return at S:29645-29647 *before* logging `resolver_lane_wait`. The observed
record instead locates the stalled interval after lock acquisition. The two
in-flight calls and the ongoing `inline-1-8` records make it reasonable to pair
the new request with that resolver record (F:143,177; E:551-557).

**Not certain at sub-branch level:** resolution and handler records have
`job_id:null`, no wire request ID, and no refusal code. A missing/interleaved
record cannot be ruled out. The exact message proves the common helper, not
which of its two timeout callers executed. The retained profiles improve on the
measurement report's explicitly unresolved lock-versus-task attribution
(E:725-728), but do not justify treating the pairing as a definitive trace.

The observed worker-side late durations are 829.582750, 1100.528000,
1582.755792 and 1872.109209 ms (R/first-failed-arm/worker-profile/worker-56326.jsonl:234-236,241).
These and the 4.66-second module roundtrip are different measurements: the
module includes waits and scheduling around exchange, not just worker compute.
Neither duration, by itself, identifies a held lifecycle mutex.

### Checking the proposed culprits

- **Idle retry:** after qualification succeeds, `state.finished` prevents
  further idle retries. Before it succeeds, an idle drop can schedule/retry
  preparation, but it takes a shared gate and uses the worker stream. It can
  cause shape-admission contention, not hold a lifecycle/write gate through
  preparation. There is no recorded packed re-admission in this arm
  (S:1483-1549; E:688-696). The separate post-LOAD scheduling task briefly uses
  the lifecycle mutex; there is no second LOAD to support a late instance
  (S:6458-6473).
- **Self-check:** a Ready slot is not sufficient to assert certification, but
  the retained passed check and generation-one run sequence support the
  passed-check fast path here. `join_lane_check` rechecks that status under its
  map lock, preventing a stale pending observation from starting a second
  check. An actual pending check *could* own lifecycle + write gate while
  waiting for an inference to drain; the record does not show one here
  (S:29392-29405, 29426-29463, 29532-29548;
  R/first-failed-arm/events.json:19-29).
- **Another Ready resolver:** this can hold the lifecycle mutex while its
  detached readiness task is delayed, even though it needs no new LOAD/check.
  The source permits it; profiles do not name the owner or readiness-task
  stages (S:29618-29623, 29820-29836, 29924-29933).
- **Unload/recovery/crash:** unload does not queue a writer behind active
  readers. One PID/LOAD, no `ERR`, no packed eviction/re-admission, a successful
  last exchange and empty retained stderr argue against a worker crash as the
  initiating event. They do not exclude every recovery event. The relay's
  broken pipe can be teardown after the driver stops the other in-flight call,
  not the cause of the module refusal (S:6102-6124; E:688-696, 721-723).

## 3. Can production with packing off do this?

**Yes, the same resolver wrapper remains reachable, but not as a direct
five-second timeout on ordinary inference.** Packing is selected only later
at S:12980-12999; the resolver has no tight-packing condition. Its long-budget
loading response therefore applies equally to an unpacked Ready lane if the
lifecycle mutex or detached readiness task is delayed. The normal 30-second
inline deadline also exceeds the five-second threshold (S:205, 29604-29629).

There are two materially different slow-load scenarios:

1. **Only the in-flight exchange is slow:** the next resolved/certified request
   can obtain a shared gate and queue on the worker stream. A wait longer than
   five seconds does **not** automatically fail; W:5625-5700 uses its absolute
   deadline, not the resolver's five-second cap. Its expiry is
   `deadline_exceeded` unless a compile overlapped the wait. The first trace
   itself contains a successful module roundtrip longer than five seconds.
2. **Readiness/lifecycle work is also delayed:** a pending self-check can hold
   lifecycle while waiting for the write gate behind slow readers; another
   resolver may be unscheduled or delayed in profile/store work while holding
   lifecycle; or this request's own spawned readiness task may miss the
   five-second window. These paths can yield the generic `model_loading`
   before execution. Memory pressure could affect such work, but a slow ANE
   reply is not proof that it did (S:29543-29558, 29618-29629,
   29820-29836, 28899-28925).

Thus the incident identifies a **packing-independent classification/latency
hazard**, not proof that every request queued behind a slow ANE exchange has
that error. The lock separation was explicitly designed to avoid the latter
(S:27487-27499).

## 4. Correct behavior and error meanings (proposal only)

- A lane that is loaded and numerically certified should not be described as
  loading because a resolver task was slow or an earlier inference is using
  the worker stream. Reuse validated readiness without a load/check attempt,
  while preserving unload safety, identity/generation checks and reply-drain
  lifetime guards. Alternatively, if readiness must still be consulted, bound
  it by the **remaining absolute request deadline**, not a separate five-second
  loading heuristic. Do not simply remove the lifecycle/serving guards.
- Use one deadline across preflight, resolution, certification, queueing and
  execution. The existing batch code already subtracts resolution elapsed time
  for later phases (S:8866-8884); a future correction should not grant a fresh
  budget at each phase or grant five more seconds after acquiring the mutex.
- For a loaded/certified lane blocked by ordinary work, wait until that deadline
  and then return **`deadline_exceeded`**. This is already the execution gate
  and ordinary stream-wait meaning (S:27532-27573; W:5574-5590, 5647-5685).
  The stable contract is transient, with no retry-after and
  `safe_to_retry_same_request=false`; preserve that declared behavior rather
  than silently inheriting loading's safe-to-retry flag (C:144-160).
- Use **`shape_compiling`** only when an actual shape admission/compile overlaps
  the request's wait. It is not a generic memory-pressure or busy-worker error.
  Existing worker tracking classifies that wait; the module reports transient,
  retry-after **1000 ms**, safe-to-retry, with shape and lane details
  (W:3456, 5574-5590, 5647-5673; S:6954-6962, 6985-7001).
- Reserve **`model_loading`** for genuinely unresolved/loading lifecycle work
  (and the existing explicitly identified numerical-self-check wait, if that
  contract is retained). A genuine not-yet-certified check is different from
  slow service on a passed lane; changing its deadline contract would require
  an explicit decision and updates to the tests that currently require
  `model_loading` on self-check deadline expiry (S:19426-19494,
  19585-19637). A failed numerical check remains `self_check_failed`, not
  `shape_compiling` or a busy-lane refusal.
- Keep worker transport silence separate. Normal inference's **30-second I/O
  timer starts after acquiring the stream**; stream queueing is bounded by the
  request deadline. Its transport-fault/recovery path is not the catalog
  readiness error and should not be relabeled as `model_loading`
  (W:115, 6177-6178, 6243-6249, 5625-5631, 5693-5700).

## Unresolved findings and how to discriminate them

1. **The likely detached-task stall has no attributed phase.** The logs do not
   record readiness spawn/start/finish, profile-key construction, store-lock
   wait, lifecycle ownership or request-correlated timeout branch. The passed
   loaded path still computes a key/projection, including synchronous store
   access, before returning. Scheduling delay, page faults and store/mutex
   contention are separate candidates, not interchangeable diagnoses.
   Add request IDs and explicit branch/owner/stage timings in a future change;
   that would distinguish the two resolver arms and locate a repeated stall.
2. **First-run memory-pressure causality is not established.** The second
   comparison's 1,147 samples and 109.156 GiB swap-in delta establish global
   paging there, not its attribution to this model or its presence at the first
   refusal. Neither rising per-process RSS nor a module/worker leak was shown
   (E:811-837). The second EOF is independently explained by a trailing-NDJSON
   reader race, not a repeated ANE readiness failure (E:849-865).
3. **Historical recovery coverage is incomplete.** No general crash/recovery
   ledger is retained; one worker profile and no errors are bounded observations,
   not proof that every internal generation/recovery transition was absent
   (E:688-696). A generation/owner event trace would resolve that separately
   from adding inference timings.
4. **A CPU-only regression can distinguish the code hazards without hardware.**
   A mock Ready/passed lane should survive an ordinary exchange held beyond
   five seconds when its request deadline is longer, and expire as
   `deadline_exceeded` at its own deadline. Separately delay the resolver's
   spawned Ready-path task beyond five seconds and capture today's generic
   `model_loading`; hold lifecycle and compare that branch. Existing tests
   exercise shape compilation, passed-check reuse and self-check expiry, but
   do not by their names establish this incident's delayed warm-resolver task
   (S:19181-19226, 19426-19494, 19853-19979). No test or behavior change is
   included in this report.
5. **A future hardware trace could identify a real stall, not retrospectively
   prove this one.** If CPU-only tracing does not explain it, an explicitly
   owner-approved private M4 comparison could collect the above correlated
   readiness stages together with system memory, process scheduling and page
   fault samples. It should preserve the workload/deadlines, run packing-off
   controls, and not manufacture pressure or interfere with serving clients.
   That would require asking the owner before any invocation; **no hardware
   run is proposed for execution or performed by this investigation**.

The first comparison and the second, memory-instrumented comparison remain
incomplete; the packing rollout gate remains
**NOT PASSED** (F:146-148,163; I:146-148; E:878-882). Diagnosing a resolver
classification does not turn either run into an accepted throughput result.
