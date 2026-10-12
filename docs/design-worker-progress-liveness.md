# Worker progress and request liveness

Status: proposed; implementation requires approval of this note.

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
| CUDA inference | After each retired layer and completed output sub-batch, plus completed weight-upload/shape-setup milestones. | Rust currently invokes one opaque batch forward C ABI call. Extend that ABI with a progress callback. Use CUDA completion events on the actual execution stream (including captured-graph layers), waited on by the host before reporting; never call socket code during graph capture or mistake kernel submission for retirement (`crates/synapse-engine-cuda/src/cuda.rs:391-431,648-665`; `crates/synapse-engine-cuda/src/port/cuda_modernbert.cu:351-466,485-517`). |
| Direct ANE inference | After each synchronous layer evaluation and output row; packed execution reports layers and completed passes. | Cached executable evaluation already blocks through execution. The callback belongs after that call, not before it (`crates/synapse-worker-ane-direct/src/backend.rs:396-424`; `crates/synapse-worker-ane-direct/src/worker.rs:433-514`). |
| Direct ANE shape/packed admission | After each layer's graph compilation/load, and after final residency publication. | A whole cold shape can take minutes. One native compiler invocation is opaque, but successive layer completions are observable (`crates/synapse-worker-ane-direct/src/backend.rs:257-320`). |
| Swift Core ML inference | One row per native prediction call, then report its completed output. Keep the bucket chosen for the original request, existing padding and row order. | Today `predictions(fromBatch:)` submits the whole batch with no progress inside it, and the worker itself has no numeric batch-count bound. Splitting independent rows bounds each opaque call to one fixed model input shape, rather than an arbitrary batch. It does not split a row or change attention semantics (`crates/synapse-worker-ane/swift/ane_worker.swift:267-331,413-437`). |
| llama embeddings/rerank | One independent sequence/candidate per complete native encode/decode call; report after output extraction. Preserve context sizing, positions, pooling and sequence independence. | Today embedding encodes the whole batch at once; rerank already calls encode per candidate. Never split an encoder's attention sequence into token chunks, which would change the result (`crates/synapse-worker-llama/src/runner.rs:642-681,888-955`). |
| llama generation | Decode causal prompt prefill in at most 512-token chunks, preserving KV state, then report each completed generated-token decode. | The existing worker has no separate prompt-length ceiling. Causal chunking makes native work bounded without reducing that ceiling or generated-token limit. A loader progress callback or an abort/poll callback is not evidence that a forward pass completed (`crates/synapse-worker-llama/src/runner.rs:704-762`). |
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
| Vulkan, CUDA (after setup) and direct ANE inference | 120 seconds |
| Core ML or llama inference with an opaque native forward call | 600 seconds |
| LOAD, first-time shape/packed compilation, eviction and CUDA cold setup | 600 seconds |
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
the existing crash-accounting authority and recovery/quarantine policy. The
public error remains `engine_crashed`; diagnostics say which operation and
worker generation stalled, its last counter, its idle duration and window,
rather than “request exceeded 30000 ms”. Real exits, broken pipes and malformed
protocol still follow their existing fault paths. Do not replay a dispatched
request: it may already have executed.

### Why these windows

The Ally data measures **requests**, not gaps between layers. Qwen3 takes
2.6 seconds at 512 tokens; treating all that work as quadratic at 8,192 tokens
would estimate roughly 666 seconds for an entire row, not a single layer. A
28-layer model divides the long row into many completion points. Allowing
120 seconds of silence per layer is intentionally conservative; it does not
assume that the whole row finishes in 120 seconds. Vulkan's existing
per-dispatch queue waits mean a layer completion proves actual GPU retirement.
No per-layer maximum has yet been measured on the Ally; Phase 2 must record
the maximum observed progress gap as well as total 8,192-token time. The same
120-second window for CUDA is an estimate based on synchronized layer-sized
work, not on an observed slowest CUDA gap; cold graph setup uses the setup
window until that setup has completed. A CUDA inference exchange starts with
a 600-second setup allowance; its first completed input/plan-preparation frame
with `phase: "inference"` switches to 120 seconds. If setup is already cached,
emit that frame as soon as input preparation completes. The host allows only
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

The 600-second opaque-forward allowance intentionally grants more silence than
layer-instrumented inference. A token/shape bound is not a mathematical bound
on native execution time, and neither this repository nor the Windows report
establishes a worst-case Core ML/llama call duration. Use bounded work per native
call, report at each completed call, and treat ten minutes as a conservative
operational estimate, not a measured guarantee. Expose validated, positive
host configuration overrides for these silence windows so slower supported
hardware can increase them without reinstating a total request cap. Do not
allow an individual caller or a progress frame to disable the watchdog.
Record calibration gaps honestly; an independent timer thread cannot fill
them with evidence of GPU liveness.

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
recovery requires faulting it and confirmed worker exit.

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

This commit is a design note only. No timeout, protocol, worker, public contract
or hardware certification record changes before approval. The estimates above
are proposed policy defaults, not new hardware measurements.

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
     work unit, then stop progress and remain connected. Use a short injectable
     idle window, assert detection within that window plus a small scheduler
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

If a legitimate measured single work unit approaches its silence window,
increase or refine the relevant engine window before claiming certification;
do not add fabricated heartbeats, split noncausal attention, widen token limits,
or weaken numerical tolerances to make the result green.
