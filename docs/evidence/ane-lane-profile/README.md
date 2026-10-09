# Direct Neural Engine lane: profiling harness and lock analysis

## Status and decision

**Not measured on this host: the global ANE lock is held by production.**
No worker was started and no layer dispatch was measured in either attempted
profiling arm. Do not interpret the refusal latencies below as embedding latency.
The cold-shape live reproduction is **not run**: it never reached shape admission.

The task reviewer explicitly instructed us to preserve the global lock, deliver
Part C and the code-level Part B analysis, and leave a runnable Part A harness for
an otherwise idle second Mac. The lock must not be bypassed by changing `HOME`,
using another lock path, or stopping a production worker. A second compiler could
exhaust the Neural Engine's approximately 100 loaded-executable capacity and break
production embedding. Production traffic would also contaminate timings.

The supplied prior measurements remain background, **not results of this run**:
6,341 chunks / 121 s / 52 texts/s / call p50 2.06 s on direct ANE; Metal 30.5 s;
llama.cpp Q8 79 s. The supplied one-row measurements were 100 vs 23 ms at 512
tokens and 24.6 vs 16.8 ms at 128 tokens. These numbers have no paired load
samples here and are not used to estimate dispatch overhead.

## Method and runnable commands

Use the existing `aft_embed_headtohead` example. Its new opt-in catalog-profile
mode uses the same in-process daemon, private store/configuration, explicit
connection file and `ckdev-*` child aliases as the original driver. There is no
production daemon discovery. It keeps the real Qwen ANE profile, checkpoints,
conversion and numerical self-check. A schema-valid test catalog selects only
the ANE backend, and an in-process loopback HTTP server serves the pinned original
checkpoint to the private `models.download` path. The rerank default entry has
no backends or self-check; no rerank model is downloaded.

Prerequisites: an Apple Silicon Mac with a usable direct Neural Engine runtime,
Xcode, the pinned original checkpoint, and **no running Synapse process holding
the global direct-ANE lock**. Do not stop an unrelated process to meet this
prerequisite. The private installer duplicates checkpoint bytes in scratch space;
allow several GiB of free space. Run from the task checkout root.

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export SYNAPSE_QWEN_WEIGHTS="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
export SYNAPSE_COMPARE_ASSETS="$PWD/target/release"
export SYNAPSE_HEADTOHEAD_INPUT="$HOME/.local/share/cortexkit/synapse/aft-headtohead/engram.jsonl"

# Build the worker and private module, including the test-only catalog override.
env -u TMPDIR CARGO_BUILD_JOBS=1 cargo build --release --locked \
  -p synapse-module -p synapse-worker-ane-direct --bins \
  --example aft_embed_headtohead --features synapse-module/test-support
ln -f target/release/examples/aft_embed_headtohead target/release/ckdev-aft-embed-headtohead

# Choose a NEW output directory for each measurement window; do not append runs.
export PROFILE_OUT="$PWD/docs/evidence/ane-lane-profile/second-mac-run"
mkdir -p "$PROFILE_OUT/normal" "$PROFILE_OUT/hardware"

# Back-to-back arms. No quiet-load gate: load is evidence on a shared machine.
env -u TMPDIR -u SYNAPSE_ANE_PROFILE_HW \
  SYNAPSE_ANE_PROFILE_CATALOG=1 \
  SYNAPSE_ANE_PROFILE_DIR="$PROFILE_OUT/normal" \
  SYNAPSE_HEADTOHEAD_OUT="$PROFILE_OUT/normal/calls.json" \
  target/release/ckdev-aft-embed-headtohead && \
env -u TMPDIR SYNAPSE_ANE_PROFILE_CATALOG=1 SYNAPSE_ANE_PROFILE_HW=1 \
  SYNAPSE_ANE_PROFILE_DIR="$PROFILE_OUT/hardware" \
  SYNAPSE_HEADTOHEAD_OUT="$PROFILE_OUT/hardware/calls.json" \
  target/release/ckdev-aft-embed-headtohead

python3 docs/evidence/ane-lane-profile/summarize.py \
  "$PROFILE_OUT/normal" "$PROFILE_OUT/hardware"
```

`engram.jsonl.meta.json` beside the input is required. The driver verifies 6,341
sequential chunks and a complete 127-batch plan, with at most 64 rows per batch.
It checks all checkpoint file hashes against `bench/parity/models.json`. Outputs
contain input and metadata SHA-256 hashes, not the input texts. Do not override
`HOME`, the global ANE lock, or the production store. `TMPDIR` is deliberately
unset for compilation and execution. No Bionic/LM Studio arm is invoked.

### Inputs and experiment order

1. Load/install and self-check the private real catalog lane
   `qwen3-embedding-0.6b-ane` (`ane-direct-worker`). Initial readiness refusals
   receive `initial-*` labels. The final harness saves these immediately and
   stops with nonzero exit on a non-loading error, including `ane_lane_busy`.
2. Prefix the concatenation of exported chunks 0–31 (newline separated) at UTF-8
   boundaries to obtain exactly 32, 128, 256 and 512 composed tokens. The original
   export's longest chunk has only 658 characters; concatenation is needed for
   the longer probes. These are real code texts, but **not the original AFT batch
   plan**. Token counts include the catalog's one terminal token. Each length has
   one first-use call and five warm repeats. A `cold-32` label is not proof of
   compilation: the self-check may already have admitted shape 128.
3. Start a 1,024-token prefix, then a concurrent 32-token request one second later.
   Shapes 128/256/512 were exercised; 1,024 has not been deliberately warmed.
   Confirm `kind=admit, shape=1024, cached=false` in worker evidence rather than
   assuming it was cold. This experiment precedes the 64-row batch to prevent
   that batch from warming the incident shape.
4. Replay the **first actual 64-row batch** from the export plan, twice serially,
   then twice concurrently. This is the AFT-shaped call, with real lengths/order.
   It may span several bounded engine calls; it is not assumed to be one worker
   RPC. Bulk scheduling sorts/group rows and restores the response order.

The default supervisor budget is 100 executables (`worker_host/mod.rs:3416–3424`).
Three Qwen shapes reserve 84; the fourth needs an eviction before admission
(`:4184–4188,4208–4232`), with budget freed only after the eviction acknowledgement
(`:4304–4318`). The oldest unleased warmed shape will normally be 128. Thus the
short request after the incident may recompile its evicted shape; it is a
post-incident liveness check, not necessarily a warm-latency sample.

Every call records one-minute load before and after it, wall time, returned token
counts, error classification and a completion timestamp. Successful replies must
have the requested row count, 1,024-dimensional finite vectors and unit norm.
An expected `model_loading` refusal for the incident's short waiter is retained;
other measured-request failures cause a nonzero exit after saving the report.

### Expected outputs and interpretation

| Output | Meaning |
| --- | --- |
| `calls.json` | Driver timings/load, counts, portable refusal classification, workload hashes |
| `module-<pid>.jsonl` | Resolver lane wait, batch tokenization/admission, bulk scheduler dispatch, execution lane/permit wait, direct-ANE roundtrip |
| `worker-<pid>.jsonl` | Shape admission/eviction totals; per-row gather/packing/readback/tail; 28 per-layer timings; sequence decoding/inter-row gaps; request total |
| `summary.json` | Generated timestamp-window aggregation; regenerate with `summarize.py` |

`SYNAPSE_ANE_PROFILE_DIR` enables the instrumentation. With it unset, no profile
files are written and the ordinary execution calls are unchanged.
`SYNAPSE_ANE_PROFILE_HW` selects the **separate measurement arm**, only when the
profile directory is enabled. Normal profiling uses `run_cached_profiled`:
request preparation and synchronous submission/wait. Hardware profiling uses
`run_cached_with_stats`: enclosing layer wall time and runtime-reported hardware
nanoseconds. It does **not** evaluate a layer twice. Both paths cache requests,
although the stats path has a separate request cache and first-use setup.
The host timer alone is not device compute (vendor `ane/src/executable.rs:81–105`).
Hardware stats are documented to exclude XPC overhead and may return zero if
unsupported (`:107–148`). Zero stats are **unavailable**, not zero device work.

The worker writes evidence after timing ends so stderr forwarding cannot drop
rows. The sequence envelope additionally includes that file I/O; subtract row
wall from row-envelope wall to quantify perturbation. Host profile file I/O,
clock/JSON collection, scheduler work and client encoding still contribute some
residual. Do not subtract overlapping timers twice. The module roundtrip minus
worker request total is an **IPC/supervisor/scheduling/codec residual**, not a
pure socket-transit timer: host transport lives outside the permitted edit fence.

`summary.json` groups records by driver completion intervals. Concurrent calls'
intervals overlap, so their per-call worker aggregates may include the other
call's rows. For overlap experiments use the **union** of the two intervals
(deduplicate worker records), and inspect module `job_id` waits and worker row
intervals separately. Never add the two overlapping summaries. Sub-millisecond
boundary attribution may be affected by clock/read/encoding overhead.

### Analysis to run on the second Mac

Use warm samples, excluding first-use request creation and shape compilation.
For each padded shape, sum 28 layer times per row, then report median/range for
five repeats and the associated load range. The ladder is 128, 256, 512, 1,024,
2,048, 4,096, 8,192 (`backend.rs:14`); 32 and 128 both compute shape 128.

For runtimes that provide nonzero hardware execution-time counters:

- Per-dispatch non-hardware cost = enclosing layer wall minus hardware time.
  Its median across layers/repeats estimates fixed dispatch/host overhead.
- Fit `row_time = intercept + slope * padded_tokens` over warm shapes
  128/256/512, **separately** for hardware and non-hardware time. Divide a row
  intercept/slope by 28 for per-layer estimates. Report the residuals and do not
  assume attention's quadratic term or shared-machine contention is linear.
- A 64-row call has 1,792 layer dispatches if all rows finish. Sum hardware time,
  non-hardware layer time, gather/pack/readback/tail, row gaps, compile admission
  and module waits. Report the measured accounting residual, not an invented
  percentage. Compare normal versus stats arms before trusting stats overhead.
- Compare the pair's union occupancy with serial 64-row calls. The code proves
  serialization points below, but actual throughput and device idle gaps still
  need measurement. Two configured permits alone cannot provide ANE overlap.

## Part A tables — measurement pending

All cells below are **not measured on this host: the global ANE lock is held by
production**. There is no defensible fixed dispatch cost, token slope or 64-row
overhead/compute split yet.

| Probe | Padded shape | Module tokenize/admit/queue/locks | IPC residual | Gather/pack/readback/tail | ANE per-layer/row | Inter-row gaps | Load |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| 32 composed tokens | 128 | not measured | not measured | not measured | not measured | n/a | not measured |
| 128 composed tokens | 128 | not measured | not measured | not measured | not measured | n/a | not measured |
| 256 composed tokens | 256 | not measured | not measured | not measured | not measured | n/a | not measured |
| 512 composed tokens | 512 | not measured | not measured | not measured | not measured | n/a | not measured |
| Actual AFT 64-row batch | per-row ladder | not measured | not measured | not measured | not measured | not measured | not measured |
| Two AFT calls in flight | per-row ladder | not measured | not measured | not measured | not measured | not measured | not measured |

### Raw attempted runs, NOT performance measurements

The native private catalog installer and original-checkpoint conversion completed,
but every serving attempt failed before worker startup. The earlier harness
revision retried readiness 100 times and then continued labeling further failed
calls as warm/cold. The final harness fixes that behavior: it fails fast on the
first non-loading readiness error. Preserve the old attempts as refusal evidence,
not successful measurements. Free-form messages/details were removed from the
raw files to eliminate machine paths; `cause=ane_lane_busy` preserves the reason.

| Attempted arm | Calls | `model_loading` | `engine_crashed` with `ane_lane_busy` | Executed rows | Load range (1 min) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Normal cached timing | 131 | 2 | 129 | 0 | 10.22–14.74 |
| Hardware stats (next arm) | 131 | 7 | 124 | 0 | 12.42–22.90 |

Selected raw refusals (wall ms, **not inference ms**):

| Label | Normal wall ms | Normal load | Hardware wall ms | Hardware load |
| --- | ---: | ---: | ---: | ---: |
| initial-0 (`model_loading`) | 5008.160 | 14.49 | 5003.854 | 15.29 |
| initial-1 (`ane_lane_busy`) | 1307.982 | 14.45 | 1388.220 | 14.47 |
| warm-32-0 | 1.162 | 14.36 | 10.892 | 21.30 |
| warm-128-0 | 1.062 | 14.14 | 12.556 | 21.37 |
| warm-256-0 | 1.242 | 13.76 | 9.278 | 19.61 |
| warm-512-0 | 1.206 | 13.69 | 3336.474 | 21.09 |
| incident-cold-1024 | 4301.513 | 13.50 | 10.985 | 20.71 |
| incident-concurrent-short | 3297.580 | 13.50 | 4311.309 | 20.71 |
| aft64-warm | 1.553 | 13.38 | 1.212 | 20.41 |

The initial five-second `model_loading` lines concern **load attempts whose
background task continues after the requesting caller stops waiting**,
not the requested cold-shape incident reproduction. No worker JSONL exists in
these occupied-host runs. `normal/` and `hardware/` contain their raw driver and
module records only; generated summaries are not committed.

## Part B — incident explained from code

### What the code establishes (high confidence)

**Yes: a serving engine inference holds the catalog lane mutex, including a
first-time shape compile.** The catalog mutex protects serving execution, not
only model loading.

| Serialization point | Exact source and lifetime |
| --- | --- |
| Catalog serving lane | `crates/synapse-module/src/lib.rs:12291–12303`: acquire the lane's owned mutex before execution; `:12308–12310`: acquire the execution permit afterwards; `:12387–12392`: pass `(permit, catalog_guard, _activity)` to direct-ANE `infer_guarded` |
| Detached direct-ANE inference | `crates/synapse-module/src/worker_host/mod.rs:5331–5357`: spawn the inference task and retain the supplied guards; `:5363–5399`: inference exchanges are awaited within that task. A cancelled caller does not release the guard before the worker reply drains |
| Residency before inference | `worker_host/mod.rs:4090–4148`, especially `:4123–4132`: obtain a shape lease before the inference callback and release it afterwards; `:3967–4049`: the lease path can wait/admit/evict; `:4394–4436`: the driver performs shape admission |
| Worker stream | `worker_host/mod.rs:4644–4659`: one Tokio mutex protects the stream session; `:4775–4819`: `exchange_inner` holds it across writing request/raw frames and reading response/raw frames. Shape admission uses that same exchange (`:4867–4905`) |
| Worker loop | `crates/synapse-worker-ane-direct/src/worker.rs:183–205,245–261`: one request at a time; admission calls `Model::admit` synchronously before replying. `:350–389`: sequences run rows one after another |
| First-time compilation | `crates/synapse-worker-ane-direct/src/backend.rs:224–234`: `Model::admit` invokes graph compilation; `:251–254`: a resident shape returns immediately; `:259–307`: otherwise build/compile every layer before publishing the resident entry. Qwen has 28 layers |
| Per-row layer work | `backend.rs:389–421`: execute the layer programs in order; no per-layer host copy between their IOSurfaces |
| Machine-global owner fence | `worker_host/mod.rs:3429,3809–3829`: acquire the per-user direct-ANE advisory lock; `:5112` puts its inherited descriptor in worker configuration and `:971–972` supplies it as worker stdin (`:5031–5035` documents the duplicated descriptor lifetime). `lib.rs:6663–6666` uses the default lock; this is separate from the per-model catalog mutex |

The residency supervisor controls admission, leases, budget and eviction; it is
not by itself an inference-overlap guarantee. Its waiters do not hold the socket
while waiting for admission, but the **compile RPC does hold the socket**. Even
if the catalog mutex were shortened, the single stream and worker loop would
still serialize calls. The configured two execution permits are a module-wide
limit, not two independent hardware streams.

**Important scope nuance:** an `embed.batch` logical call is not necessarily one
continuous lock hold across all 64 rows. `lib.rs:11489–11596` schedules bounded
engine batches; `:11557–11566` awaits each `execute_embedding`, and `:11581–11585`
yields between them. Each engine batch acquires/retains the catalog guard. A cold
compile blocks its entire engine inference, and resolver waiters for the same
lane, even when the encompassing AFT call has multiple quanta.

### Why a loaded lane can say “loading”

`resolve_serving_model`, `lib.rs:25121–25127`, tries the **same catalog lane
mutex** and waits at most `min(request_budget, 5000)` ms. A failure to acquire it
calls `catalog_loading_response`. `lib.rs:25154–25164` returns
`deadline_exceeded` for budgets at most five seconds, otherwise `model_loading`
with `catalog lane '<lane>' is loading` and retry-after 250 ms. It does not check
whether the owner is loading, compiling a shape, or doing ordinary inference.
The lock owner's readiness task is also detached (`:25141–25150`), with a
separate five-second readiness wait. The resolver guard is not transferred
straight to serving: execution acquires the same lane lock again later.

Therefore the supplied 09:02:56Z admission and the 09:03:31Z `model_loading`
response are **consistent with** a cold-shape compile holding that lane mutex.
A shape compile does not require a worker restart or loss of model residency.
The request at 09:03:31Z can be rejected before its own admission, so no admitted
closing line is owed for that refusal. Requests that already passed resolution
can queue later on the execution lane lock as well.

This is a code-supported mechanism, **not proof of the historical cause**.
The missing closing event for `inline-154-456` does not identify its error, shape,
or compile duration. No production logs/store were read for this task. The
09:03:39Z recovery could also follow long inference, scheduling contention, or
another failure/drain path. Part C closes returned-error paths going forward;
it cannot retroactively classify that request.

### Live reproduction status and expected observation

**Not run: no worker or cold compile could start while production owned the
global ANE lock.** The occupied-host attempts in the raw files are not a
substitute reproduction. On the second Mac, inspect the `incident-*` records:

- `incident-cold-1024` should cause an uncached 1,024-shape admission after the
  lane is already loaded; worker admission evidence must establish that fact.
- A short call starts one second later. If compilation holds the mutex longer
  than another five seconds, it should return `model_loading` after about five
  seconds, despite a loaded model and no restart.
- The long call should eventually complete (600 s request budget), followed by
  a successful `incident-short-after`. Record actual compile time and load.
- If compilation finishes in less than six seconds, the short request may instead
  succeed: that does not disprove the locking mechanism; it does not reproduce
  the incident's long hold. Report that outcome explicitly.

### Recommended fix shape — NOT implemented

1. Separate lane **state/lifetime protection** from **execution serialization**.
   Hold the catalog mutex only while inspecting/publishing load and self-check
   state and selecting a live model generation; represent in-flight lifetime
   separately so unload/reload cannot invalidate an inference. Do not merely
   delete the guard: cancellation/drain and numerical self-check guarantees
   must survive.
2. Make first-time shape preparation single-flight in residency management,
   with explicit `shape_compiling`/queue state and absolute-deadline accounting.
   Do not hold the catalog metadata mutex through synchronous compilation or
   worker inference. Preserve the machine-global ownership/capacity fence.
3. Report loading only for actual loading. Report ordinary execution/residency
   contention with an appropriate busy/queue/deadline classification rather
   than the misleading model-loading message. Preserve distinct caller deadline
   and detached worker-drain behavior.
4. Add private-daemon tests for warm-lane contention, cold compile plus short
   waiter, cancellation/drain, concurrent self-check, unload and generation
   replacement before changing these lock scopes.

This fixes head-of-line blocking/misreporting, not necessarily throughput: the
socket and worker loop remain serial. Decide row batching, layer fusion or other
compute changes only after obtaining the missing hardware/dispatch split.

## Part C — admitted-error closing events

Implemented `job failed` at info level in the same tracing log category (`perf`)
as `job done`,
with `model_id`, `job_id`, `lane`, stable wire `code` and `wall_ms`. Success logs,
wire errors, inference behavior and scheduling remain unchanged. Coverage is:

- `embed.query`: post-admission tokenization, composition and profile-check errors;
- inline query/batch shared execution: engine/scheduler/permit/worker failures and
  vector-count mismatch;
- `rerank.score`: execution failures and score-count mismatch;
- inline remote embed query/batch: gateway errors after admission;
- durable local/remote embed jobs: failure helper, claim errors, failed continuity
  check and remote reauthentication pause (closing the failed current attempt).

Failures before admission are not logged as failed admitted jobs. Model-load,
probe and generation lifecycles are outside this logging change. Durable failure
wall time uses `JobRecord.created_ms`, the persisted time of job admission; inline wall time uses
the existing monotonic start. If the durable record is unreadable, the closing
line still appears with unknown model/lane metadata and wall time zero. A process
crash, abort or dropped request future without a returned error cannot promise a
closing line; no panic/cancellation redesign was attempted.

The regression `tests::failing_admitted_embed_and_rerank_log_closing_perf_lines`
uses real handler dispatch for query, batch and rerank, an already-certified
fixture and a closed execution semaphore. It confirms actual admission, returned
`queue_full`, exactly one `perf: job failed`, matching job ID, lane, code, wall
field, and no success event. It does not just invoke the logging helper.

Mutation proof: staged the live implementation, confirmed empty unstaged diff,
removed the central failure log call with a temporary `NON-VACUITY BREAK`, and
ran the **whole** module lib suite on Linux. Exactly the named regression failed
(`missing or duplicate closing line`, left 0, right 1); 621 other tests passed,
5 were ignored. Restored with `git checkout -- crates/synapse-module/src/lib.rs`
and `touch`, and confirmed empty unstaged diff again. The mutation's diff was
one file, 2 insertions / 1 deletion. Restored suites pass on Linux and macOS.
See `verification.json` for the path-free failure excerpt and examples of tests
that still passed while the closing event was removed.

## Verification

- Cargo 1.99.0 / rustc 1.99.0; Clippy 0.1.99.
- `cargo clippy --locked -p synapse-module --all-targets -- -D warnings` with
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`: passed on Linux,
  all module targets (not the macOS-only worker backend).
- Module lib tests with that `DEVELOPER_DIR`: Linux 622 passed / 5 ignored;
  native macOS `cargo test --release --locked -p synapse-module --lib`:
  626 passed / 10 ignored.
- `aft_embed_headtohead` example tests: 7 passed on each platform, including the
  added portable-error and composed-prefix tests.
- Native release module/direct-worker/profile-example build: passed. Initial
  native builds timed out after 30 and 120 minutes while waiting behind shared
  compile slots; the resumed build succeeded. A dependency (`block` 0.1.6)
  emits an existing future-incompatibility warning.
- AFT health inspection was partial while rust-analyzer indexed and the checkout
  callgraph was unavailable; cargo gates above are authoritative.
- Comment review completed; unclear comments were rewritten.

## Conclusions ranked by confidence

1. **High, source- and failure-evidence-backed:** the global advisory file lock
   prevented the private worker from starting on this host. No device performance claim is valid.
2. **High, source-backed:** serving retains the catalog lane mutex and execution
   permit through direct-ANE shape admission/compile and inference/drain. Bulk
   logical calls release/reacquire that guard between engine quanta.
3. **High, source-backed:** a resolver lock timeout misclassifies ordinary serving
   or cold-shape contention as `model_loading` for budgets greater than five
   seconds. A worker restart is not necessary for this symptom.
4. **High, source-backed:** requests/rows/layers are serial at the channel/worker;
   two module permits do not imply overlapping Neural Engine execution.
5. **High, tested and mutation-defended:** returned admitted embed/rerank errors
   now emit a closing perf event; success/error response contracts are preserved.
6. **Unresolved, not ranked as a performance result:** relative dispatch versus
   compute cost, token slope, CPU gather/readback/tail share and idle gaps.

## Unresolved

- The exact cause/shape/error of the historical 09:02:56Z admitted request remains
  unknown. The code establishes a plausible failure mechanism, not attribution.
- All requested timing tables and the live cold-compile reproduction await the
  second Mac. Raw occupied-host refusals cannot be substituted for these results.
- Runtime hardware stats may be zero, stale or perturb timing on another macOS
  version. Compare the back-to-back arms and keep first-use samples separate.
- The normal host layer timer combines submission and waiting; only valid hardware
  counters permit a non-hardware/device split. The IPC residual also combines
  host supervision, socket queueing, codecs and scheduling.
- The actual AFT 64-row call can cause residency churn or multiple RPCs; capture
  shape admissions/evictions and do not assume 64 identical padded rows.
- Concurrent-window attribution in the summary is deliberately conservative;
  analyze union occupancy and job-specific waits before claiming overlap.
- The final fail-fast harness is compiler/test verified and its private catalog
  installation path was exercised, but successful inference on an unoccupied Mac
  was not available to validate the full measurement sequence end to end.
