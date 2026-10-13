# Worker progress and request liveness

Status: approved for implementation by the project maintainer. Decisions below
justify fixed 120-second inference and 600-second setup windows and unchanged
opaque batching; certification means checking complete pinned numerical fixtures
and token-bound refusals on the declared hardware.

## Motivation and scope

A working GPU is not a crashed worker. The worker host currently treats a
request exceeding its wall-clock timeout as `engine_crashed`, adds a crash
strike, and can quarantine the lane. On the Windows Vulkan ROG Ally X,
Qwen3's completed results have cosine similarity 0.9999998 and reranker score
error 5.9e-6, but a single 8,192-token embedding row exceeds 30 seconds and a
64-candidate rerank takes 30.2 seconds. Killing those requests is a liveness
policy error, not evidence of incorrect computation. See
[evidence and the four certification cells](evidence/catalog-backends-windows/README.md).

Replace the worker request's fixed total-time cap with a watchdog measuring
time since **completed work**, independently of the caller's deadline. A
request may take arbitrarily long in total while making progress. A genuinely
stalled worker still receives a crash strike and remains subject to quarantine.

This note concerns catalog workers hosted by `synapse-module`. It does not
change lane fingerprints, certification identities, the in-process Metal
path, or owned-decode streaming. The 8,192-token ceiling and
`sequence_too_long` refusal remain unchanged. Existing public wire error codes
keep their meanings; progress is private worker-protocol traffic, not a public
response event.

## Existing constraints

- `WorkerHost::send_request` currently times the whole write/read exchange:
  30 seconds ordinarily, 180 seconds for LOAD. HELLO has a separate five-second
  bootstrap timeout. Standard requests have a `req_id` but no worker-wire
  deadline (`crates/synapse-module/src/worker_host/mod.rs:108-129,1072-1122`;
  `crates/synapse-core/src/worker_protocol.rs:364-443`).
- Workers process requests serially. A vector/score response is a JSON header
  followed by a raw float frame. A concurrent PING cannot demonstrate that the
  active GPU operation is progressing: the worker cannot read it until the
  operation finishes.
- Direct ANE already detaches **host-side** inference. Its task keeps the stream,
  execution permit, residency lease and lane/accounting guards until the entire
  reply drains; the worker itself is synchronous. Shape admission/eviction also
  runs independently of a waiting caller. See
  [direct ANE serving](direct-ane-serving.md#requests) and
  [compile-lane design](design-ane-compile-lane.md).
- Direct ANE's inference exchange still has a 30-second I/O timer; its shape
  exchange has a 600-second timer, starting only after acquiring the stream
  (`crates/synapse-module/src/worker_host/mod.rs:5526-5561,5625-5701`). Progress
  handling must cover both the generic host and this specialized channel.

## Decision 1: completed-work frames, not timer heartbeats

Use a mandatory, private worker response frame:

```json
{"type":"PROGRESS","req_id":"embed-17","completed":3,"phase":"inference"}
```

`completed` is a `u64`, starts at one and strictly increases within one
request, including across phase changes. It measures completed work units, not
elapsed time or an estimated percentage; units may differ by engine. `phase`
is required and is one of `setup`, `inference`, or `control`, constrained by the
request type. Emit a frame after preprocessing or another real setup milestone,
then at the boundaries below. There is no
periodic background heartbeat and no increment merely for enqueueing a GPU
kernel. Such a heartbeat would keep a dead GPU alive indefinitely.

The synchronous operation's progress callback writes and flushes frames on the
same stream as its final reply. Do not introduce another unsynchronized writer.
A progress write failure terminates the exchange; it must not be silently
ignored. Once a terminal JSON header is sent, its raw result frame follows
immediately: progress cannot appear inside that header/raw pair. PING, UNLOAD,
refusals and other short operations can finish directly, with no intermediate
frame. SHUTDOWN keeps its existing no-reply semantics.

The host consumes progress until one terminal response arrives. Validate every
progress frame against the dispatched request ID and require a strictly larger
counter. A duplicate, decreasing or wrong-ID frame is a protocol fault, never
an idle-timer renewal. Validate the terminal ID and drain its full raw payload
before releasing the stream. A worker generation's counter cannot renew
another generation's watchdog.

### Engine boundaries

| Engine/operation | Progress boundary | Reason |
| --- | --- | --- |
| Vulkan inference | After each completed transformer layer and each completed output row; also after input preparation and output readback. | The runtime already waits for `queue_wait_idle` at every shader dispatch. The layer callback therefore describes completed GPU work without adding a new synchronization policy (`crates/synapse-worker-vulkan/src/runtime.rs:430-453,1019-1098,1114-1260`). Reporting only completed rows would leave the entire slow 8,192-token row silent; layer completions show that work is advancing inside that row. |
| CUDA inference | In the Rust worker, after validating/assembling inputs immediately before forward, then after each successful batch forward returns. | Preserve the master kernels and C ABI byte-for-byte. Each current engine batch invokes one opaque forward, including native upload and plan/graph setup; do not split batching or add synchronization to obtain progress. The rental's bounded 8,192-token calls, including first shape use, are far inside 120 seconds. The first frame records completed Rust input preparation and switches to inference; it does not claim GPU upload or a layer has completed. |
| Direct ANE inference | After each synchronous layer evaluation and output row; packed execution reports layers and completed passes. | Cached executable evaluation already blocks through execution. The callback belongs after that call, not before it (`crates/synapse-worker-ane-direct/src/backend.rs:396-424`; `crates/synapse-worker-ane-direct/src/worker.rs:433-514`). |
| Direct ANE shape/packed admission | After each layer's graph compilation/load, and after final residency publication. | A whole cold shape can take minutes. One native compiler invocation is opaque, but successive layer completions are observable (`crates/synapse-worker-ane-direct/src/backend.rs:257-320`). |
| Swift Core ML inference | Keep the existing whole-batch `predictions(fromBatch:)` call and report only its terminal reply. | Product Core ML runs on Apple Silicon with fixed buckets of at most 512 tokens; the approved 120-second window exceeds the slowest served batch by more than 70 times. Do not change batching, throughput or numerics merely to obtain progress. The worker implementation has no mid-call completion callback (`crates/synapse-worker-ane/swift/ane_worker.swift:267-331,413-437`). |
| llama embeddings/rerank | Preserve current batching and native calls; return only the terminal reply. | Catalog embedding/rerank uses the owned Metal, Vulkan, CUDA or ANE backends; these llama operations remain for legacy/free-form API users, not catalog serving. Splitting their calls would change compatibility behavior without improving catalog liveness (`crates/synapse-worker-llama/src/runner.rs:642-681,888-955`). |
| llama generation | Keep existing prompt prefill exactly as it is; report each completed generated-token decode. | Generation is the product llama path. Token decode boundaries provide progress at negligible cost, but the prompt native call remains opaque. Do not split prefill or use loader/abort polling as evidence of completed forward work (`crates/synapse-worker-llama/src/runner.rs:704-762`). |
| LOAD/setup, all workers | After each completed artifact verification/upload/model-bucket load or other observable setup stage; final response if the native loader is wholly opaque. | A native Core ML model load or llama model load can remain silent within its setup idle allowance. Do not send timer heartbeats while a loader or graph capture is blocked. |

Completed work is reported at **every** listed boundary, including on requests
shorter than the idle window. Workers must not deliberately coalesce multiple
layers into a gap longer than their window. A single native call cannot promise
a clock-based reporting frequency: its silence allowance is specified below.

## Decision 2: idle watchdogs, separate from queue and caller clocks

Use these production defaults, selected locally by the host from the engine and
operation. A worker reports only completed units and an allowed phase; it
cannot advertise an `idle_window_ms` value or request a longer watchdog:

| Operation | Maximum silence after dispatch or last valid progress |
| --- | ---: |
| All engines' inference, including opaque Core ML/llama calls and CUDA from its Rust pre-forward transition | 120 seconds |
| LOAD, first-time ANE shape/packed compilation, eviction and CUDA input preparation before forward | 600 seconds |
| PING, UNLOAD and other short control exchanges | 120 seconds |

These are **idle** windows, not total budgets. Replace the 30/180-second generic
exchange timers and direct ANE's 30-second inference/600-second shape total
exchange timers with this policy. Repeated genuine layer completions can keep
an inference or compile alive beyond ten minutes. Keep the separate HELLO
bootstrap and confirmed-exit/retirement lifecycle watchdogs: those are not
inference-request total-time limits.

Start the watchdog only once the exchange owns the stream and begins writing
its request. Waiting for the lane mutex, residency, a stream held by a compile,
or an execution permit consumes the caller's deadline, not a crash watchdog.
This avoids blaming a queued request for the previous request's work. Keep one
absolute monotonic `idle_until` across frame reads; after each validated
progress frame set it to `now + window`. Do not restart it for a partial frame,
raw response header, malformed traffic, duplicate counters, or retries of the
same read. The request write and terminal raw-frame transfer must also finish
within a silence window; a peer stuck mid-frame is not healthy progress.

At idle expiry, fault the connection, kill/reap only the owned worker, and use
the existing crash-accounting authority. Generic worker lanes use consecutive
faults, not the former 60-second rolling window: a crash, protocol fault or
no-progress hang adds one strike; two consecutive strikes quarantine with the
existing quarantine state and recovery policy. Only fully drained successful
serving work (embed, rerank/score or generation) on the owning generation
clears strikes. LOAD/reload, PING, admission, eviction and setup do not clear.
Caller expiry neither strikes nor clears; an abandoned serving request's
eventual drained success does clear. The
independent owned-decode supervisor's strike policy is unchanged. The
catalog disposition is transient `engine_crashed` with the remaining fixed
ten-minute cool-down and `{lane_id, quarantined: true}`, never `probe_required`.
After expiry, the next serving request starts a fresh worker with zero strikes;
explicit unload/reload or module restart clears it immediately. Legacy lanes
keep `probe_required`. Direct ANE uses the same rule per channel/generation;
confirmed exit precedes residency refunds and no replacement is spawned during
its cool-down. The existing health field name `crash_count_window` is retained
for consumers, but now counts consecutive faults rather than timed history. The
public error remains `engine_crashed`; diagnostics say which operation and
worker generation stalled, its last counter, its idle duration and window,
rather than “request exceeded 30000 ms”. Real exits, broken pipes and malformed
protocol still follow their existing fault paths. Do not replay a dispatched
request: it may already have executed.

### Why these windows

Generic `WorkerHost` supervises catalog and legacy/free-form workers; direct
ANE uses its specialized channel. Both use consecutive faults. Catalog versus
legacy identifies consumer recovery policy, not a different fault counter. A
catalog lane's ten-minute cool-down starts at its second consecutive fault.
The independently persisted owned-decode supervisor remains outside this policy.

The Ally data measures **requests**, not gaps between layers. Qwen3 takes
2.6 seconds at 512 tokens; treating all that work as quadratic at 8,192 tokens
would estimate roughly 666 seconds for an entire row, not a single layer. A
28-layer model divides the long row into many completion points. Allowing
120 seconds of silence per layer is intentionally conservative; it does not
assume that the whole row finishes in 120 seconds. Vulkan's existing
per-dispatch queue waits mean a layer completion proves actual GPU retirement.
No per-layer maximum has yet been measured on the Ally; Phase 2 must record
the maximum observed progress gap as well as total 8,192-token time.

The NVIDIA rental at commit `db3b0176` rejected the original per-layer CUDA
instrumentation: all four cells at `b13add21` failed the first-load self-check
with "the operation cannot be performed in the present state", while master
`5f6d7419` passed all four on the same RTX 5090 (driver 595.84, API 13020).
Per-layer completion events inside captured-graph work are removed, rather than
changing graph execution or adding another native callback ABI.

That rental measured master 8,192-token warm RPCs at 0.19 s (gte embed), 0.23 s
(gte rerank), 0.31 s (Qwen3 embed) and 0.33 s (Qwen3 rerank). First use of each
new shape, including plan and graph setup, took 0.81-0.91 s across those models.
A compute-7.5 floor card is expected to be slower, but not the more than 130-fold
cold or 360-fold warm slowdown needed to reach 120 seconds for these bounded
calls. That is conservative performance reasoning, not a measurement or
certification on a compute-7.5 card.

A CUDA exchange still starts with the 600-second setup allowance. Its completed
Rust input validation/assembly emits `phase: "inference"` immediately before the
opaque forward. The maintainer approved this boundary because the unchanged C
ABI exposes no post-upload/plan-ready hook: the 120-second window also covers
native upload and plan/graph setup, whether cold or warm. Forward already waits
for output, so its successful return provides the next completed-work point
without any new GPU event, callback or host synchronization. Preserve current
batch geometry; there are no artificial sub-batches or timer heartbeats. The host allows only
a one-way setup-to-inference transition for CUDA and does not accept setup
frames during Vulkan/direct ANE inference. LOAD/admission/eviction use `setup`,
other inference uses `inference`, and short controls use `control`. A worker
cannot switch back to setup to gain extra silence.

Direct ANE has synchronous layer evaluations and the same bounded ladder
through 8,192. A cold shape has been observed taking most of two to three
minutes **in total**, while individual compiler calls remain opaque
([serving observations](direct-ane-serving.md#hardware-certification)). The
existing 600-second admission watchdog is a useful conservative starting
allowance for an individual compile gap. Preserve it as an idle allowance;
compilation can take longer overall if layers keep completing. A request
waiting behind that compile has no inference idle timer running.

All inference uses the same 120-second window, including opaque Core ML/llama
calls. Core ML's fixed product buckets are at most 512 tokens on Apple Silicon;
the slowest served batch is more than 70 times shorter than this window. llama
product serving is generation only and emits progress after generated tokens.
Keep existing batching and prompt prefill unchanged. A shape bound is not a
mathematical wall-clock guarantee, but a real hang must not block a lane for ten
minutes merely because its native API is opaque.

Use fixed constants in code, with the reasoning beside them, not configuration
overrides or caller/worker-supplied extensions. Supporting hardware that needs a
larger window requires a code change justified by measurements. Record the
maximum progress gap per engine and operation. If any measured gap exceeds a
quarter of its window (30 seconds for inference, 150 seconds for setup), stop
and report it to the maintainer before certifying; do not manufacture timer
heartbeats to hide it.

## Decision 3: caller deadline, not a request lifetime limit

Resolve an optional `deadline_ms` once, using the existing public deadline
semantics, into an absolute monotonic deadline. Do not reset it per layer,
row, retry, stream wait or worker restart. It bounds admission, queue/permit
waits and how long the caller waits for its result. If absent, there is no
synthetic worker total-time limit: background/bulk work can run for hours with
real progress. Existing caller/job execution budgets still apply when supplied;
this policy does not silently remove a job's budget or numerical self-check
budget.

Deadline expiry during live inference answers `deadline_exceeded` promptly and
adds **no crash strike**. It does not kill/restart/quarantine the worker. If the
caller disappears, treat it like cancellation of its wait, not evidence of a
worker crash. If a detached operation later genuinely stalls or exits, that is
a separate fault and can strike even though no caller is awaiting its result.

Keep direct ANE's existing compile-wait classification: a deadline expiring
while blocked by shape compilation still answers `shape_compiling`; other
waits and active live inference answer `deadline_exceeded`. This is the
already-documented narrow exception in
[the wire contract](wire-contract-v1.md#errors-stable-codes-typed-recovery), not a
new error code or a reason to restart a compiling worker. Changing it would
break the existing consumer retry contract. Time on the compile continues
under its own progress watchdog independently of that caller.

When caller and idle clocks expire together, an already-expired caller wait
gets its deadline disposition, while the transport owner independently
investigates idle expiry. “Deadline exceeded” alone never increments a crash
counter; observed no-progress expiry still can. A queued request that reaches
its deadline before dispatch writes nothing and leaves the connection healthy.

## Decision 4: drain already-dispatched work on every engine

Choose **detached drain**, not device cancellation, for Vulkan, CUDA, direct
ANE, Swift Core ML and llama. Do not add a CANCEL worker message in this
cutover. A completed-progress point gives a safe reporting boundary, but it
does not by itself provide uniform, tested native cancellation and cleanup.
Core ML and native llama calls are not generally interruptible, and cancelling
CUDA/Vulkan work after submission risks releasing live buffers. Keeping one
completion/drain path is preferable to adding five cancellation implementations.

The host-side exchange owner, not the caller future, owns the stream read and
all execution/lifecycle guards after dispatch. The caller selects between its
deadline/cancellation and a completion notification; giving up drops only that
notification receiver, never aborts the owner. The owner continues consuming
progress and the final header/raw pair under the idle watchdog, discards a
result with no receiver, then releases the stream, lane execution mutex,
execution permit, in-flight accounting and, for direct ANE, the residency lease.
Use the ordinary catalog's existing `spawn_blocking` ownership where applicable;
it already moves serial guards into the blocking closure
(`crates/synapse-module/src/lib.rs:12833-12861,13031-13080`). Audit the caller's
deadline wait and the low-level host API rather than assuming that dropping any
async read is safe. Direct ANE's detached task demonstrates the required
behavior: caller expiry drops the result wait, while the task retains the
exchange and guards until the worker's reply drains
(`crates/synapse-module/src/worker_host/mod.rs:6618-6680`).

Only one owner may use a worker stream at a time. A later request waits for the
earlier owner's complete drain, subject to its own deadline, before it can be
written. Use monotonically unique host request IDs and generation-fenced
completion delivery. Enforce ID checks on progress **and** every terminal
response, including errors when associated with a request. Keep the raw-frame
association with that terminal header. Never let a subsequent caller consume
the abandoned request's response, even if both operations return the same type.
Do not reuse a stream after a partial/malformed reply or an aborted owner;
recovery requires faulting it and confirmed worker exit. The public low-level
`WorkerHost` async API enforces an interrupted-exchange guard too: dropping its
future after dispatch marks the connection unusable, and the next call performs
lifecycle teardown/reload on a fresh generation without charging cancellation
as a crash. Module caller deadlines do not drop that reader; their detached
owner remains the preferred drain path.

Drain the already-dispatched RPC in full, even if it contains several rows.
Do not start further host-side RPCs/buckets for an abandoned request. The
worker is serial, so this can delay a subsequent request, but it keeps the
connection usable without restarting and does not leak concurrent GPU work.
Keep retirement's existing bounded drain/confirmed-exit policy, documented in
[direct ANE lifecycle](direct-ane-serving.md#unload-and-shutdown): stop new
leases, wait for active work within the supervisor's drain allowance, close or
kill the owned worker, and retain reservations until process exit is confirmed.
Retirement may shut
down an owned worker as lifecycle management; it is not a caller-deadline
crash strike.

## Decision 5: protocol v3, one clean deployment

Bump `WORKER_PROTOCOL_VERSION` from 2 to **3**. Progress support is mandatory
for all catalog workers, not an optional HELLO capability. Ship the module and
all worker binaries together; exact-version HELLO matching rejects v2 before
LOAD. Operators must upgrade the whole packaged runtime, not replace just the
module. Existing packed capability negotiation remains about packed execution
only; it must not select an old 30-second/no-progress transport fallback.

A v3 worker need not send progress for a short request: its terminal reply is
sufficient. A v3 worker that sends none during long work exhausts the locally
chosen silence window and is treated as hung. This includes a worker that
claims v3 but has missed an inference instrumentation path. Opaque calls get
the explicit larger window above, not an endless liveness exemption.

Retain length framing, little-endian `i32` token and `f32` result payloads,
manifest/kernel HELLO bindings, fingerprint derivation and packed capability
contents. Add only the progress response variant and private phase field;
standard request IDs and public request/response schemas do not change. The
separate owned-decode envelope's `protocol_version`, progress/streaming and
supervisor remain separate. Any owned-decode binary using the shared HELLO
constant must advertise v3 for the transport handshake, without changing its
independent envelope version. The owned-decode supervisor remains the sole
owner of its crash counter and quarantine decisions; the generic host must not
add a second strike for the same fault.

Cutover covers Rust CUDA/Vulkan/direct-ANE/llama workers, the Swift Core ML
worker and its Rust launcher/relay, mock/timeout workers, catalog certification
adapters, candidate/packaged worker HELLO tests, protocol fixtures and worker
protocol documentation. Update hardcoded v2 assertions/fixtures; replace the
legacy-v2 packed interoperability test with v3-without-packed plus explicit
v2-handshake rejection. Preserving optional **packed** support is not preserving
optional **progress** support. The prefill sidecar has a different protocol and
is not part of this change; neither is the in-process Metal engine.

No new public error code is necessary. Update `docs/wire-contract-v1.md` to
explain that `engine_crashed` includes no-progress hang detection and that a
live inference caller's deadline does not strike. Preserve
`deadline_exceeded`, `shape_compiling`, `sequence_too_long`, and other existing
codes and retry dispositions. Update stale fixed-30-second references in the
worker/direct-ANE serving docs when implementing, not historical evidence.

## Approval boundary and implementation proof

The project maintainer approved implementation with fixed windows, unchanged
Core ML/llama batching and mandatory progress-gap measurements. The 120/600-second
windows are operational policy defaults, not new hardware measurements.

After maintainer approval of this note, implement and commit at each verified
green step:

1. Add tests against the existing behavior first. Use a framed fake worker,
   exercising the real host transport/response drain and crash authority, not
   just a mocked clock or a helper detached from the request loop:
   - `slow_progress_worker_exceeds_30_seconds_without_crash_strike`: perform at
     least 45 seconds of real monotonic work/progress, return distinctive
     vectors, assert the result and zero crash strikes. Include a separate
     paused-clock unit test for repeated watchdog renewal, but do not use that
     as the only evidence that a real request can exceed 30 seconds.
   - `stalled_progress_worker_strikes_within_idle_window`: complete an initial
     work unit, then stop progress and remain connected. Exercise the fixed
     inference idle window, assert detection within that window plus a small scheduler
     allowance, exactly one strike, and existing quarantine on repeated hangs.
   - `live_worker_deadline_drains_before_next_correct_response`: expire a short
     caller deadline on live work that continues beyond 30 seconds, assert
     prompt `deadline_exceeded` and no strike, then run a second request with a
     different result and assert its ID, payload and unchanged worker
     generation. Cover the wait-before-dispatch case as well.
2. Implement v3 progress producers and shared host watchdog semantics, including
   direct ANE shape/inference channels and CUDA cold setup. Prove that duplicates,
   wrong IDs, invalid phases, partial frames, header/raw misassociation and
   queued waits cannot keep a hung exchange alive or steal a result. Preserve
   compile-wait `shape_compiling` tests and genuine crash/quarantine tests.
   Inventory existing host/soak/probe-required quarantine tests before changing
   crash accounting. Add two sequential hangs quarantine and hang-success-hang
   does not quarantine, and show both red against rolling-window accounting.
   Report any changed timing expectation by test name and reason.
3. Show each named acceptance test green and red with the relevant production
   behavior reverted. Keep v3 handshake/test scaffolding intact so a red run
   cannot be explained by version rejection before reaching inference. Use the
   prescribed stage/mutate/diff-stat/restore sequence, mark mutants
   `NON-VACUITY BREAK`, and capture exact red test names, other results and both
   runs outside regenerable build directories. Restore and rerun impacted tests
   before committing; never commit a mutant.
4. Run the module's lib, it and e2e tests, clippy and fmt on Linux with
   `runon: "linux,8c"`; run Windows-target clippy using the train preflight's
   `scripts/lib/zig-cc-windows.sh` and `scripts/lib/zig-ar.sh` wrappers; run
   `ckdev-mutate check`. Mac-only ANE/Core ML compilation and tests stay on the
   Mac. Record tool versions, counts and any platform gaps. Test protocol-v3
   HELLO/refusal/PING/SHUTDOWN on every packaged worker, not only Vulkan.
5. From a clean green commit, rebuild on `ssh ally` (Windows/Git Bash) using the
   evidence README's setup and release runner. Report all four Vulkan cells:
   gte-modernbert-base embedding and reranking, and Qwen3-0.6B embedding and
   reranking. Include each cell's outcome, both Qwen3 certification outcomes,
   exact 8,192-token time, maximum observed progress gap, and the unchanged
   8,193-token refusal/no-worker-dispatch check. Identify CPU/GPU, build SHA and
   binary provenance. Stop only processes started by this work.

6. Measure direct ANE gaps on the M4 Mac mini (`ssh macmini`, macOS 27, 16 GiB)
   using only a private daemon. The maintainer approved this conservative
   substitute because the M5's machine-wide lane lock is held by production.
   Measure Qwen3 inference at 128, 512 and 8,192 tokens plus first-time shape
   compilation. Check `pgrep -fl "ane|synapse"` before each run and never stop
   unrelated processes. The 2026-10-13 allocation ends at 07:00Z. Do not run ANE
   work on the M5 during its 01:50Z–02:50Z reservation. Record engine,
   operation, model, sequence length, maximum gap and binary provenance. Stop
   and report any gap above a quarter of its window before certifying.

If a legitimate measured single work unit exceeds a quarter of its window,
stop and report it before claiming certification. Timer heartbeats would hide a
stalled device; splitting noncausal attention changes outputs, widening token
limits changes the refusal contract, and relaxed tolerances hide numerical
errors. None is a valid workaround for a slow work unit.
