# Worker progress liveness: audit remediation

The independent adversarial review of worker-progress framing, cancellation,
validation and fault accounting at commit `77124019981fe278a50e149d2a777b8e2082d7f3`
was read from its supplied Git object without changing the audit branch.
Findings 1–5 are addressed as separate fixes; native CUDA execution/performance
proof (finding 6) remains assigned to the maintainer's rental check.

Commands below use rustc/cargo 1.99.0 on Linux with 8 vCPUs unless stated
otherwise. Test additions were run against the preceding production code before
its corresponding fix. Red runs are observable regressions, not failed builds
or handshake mismatches.

## 1. Validate serving output before clearing ANE strikes

The real serving fixture disconnects once, then returns a matching request ID,
valid terminal header/raw frame and either NaN or a model-dimension mismatch.
Both rejected results must preserve the prior strike, charge the second fault
and quarantine without another replacement worker.

```text
cargo test --locked -p synapse-module --lib ane_disconnect_then_ -- --nocapture
BEFORE: ane_disconnect_then_nan_result_quarantines ... FAILED
BEFORE: ane_disconnect_then_dimension_mismatch_quarantines ... FAILED
assertion: correctly framed but rejected result must strike, never clear history
left: 0, right: 2
0 passed; 2 failed; 738 filtered out
AFTER: both exact tests ... ok
2 passed; 0 failed; 738 filtered out
```

Both names are under `worker_host::ane_residency::tests::`. Existing two-hang and
hang-success-hang channel tests also passed (2), and the stale-generation
recovery test passed (1). Strict module all-targets clippy passed.

Serving validation now remains inside the channel's stream lock, after full
reply draining and before strike reset: declared byte shape, expected model
dimensions and finite decoded values must pass. Validation errors fault/charge
the owning generation exactly once, then use existing confirmed-exit recovery.
An expired caller can still have a later validated detached success clear
strikes; expiry itself never clears or charges them.

## 2. Defend renewal through the real generic host transport

The 45-second process test is retained and its documentation now limits its
claim to removing the old 30-second total cap. It is shorter than the production
120-second idle window, so it cannot prove renewal alone.

`worker_host::progress_tests::real_transport_progress_renews_multiple_idle_windows`
uses the public host over a real framed Unix stream, a 200 ms unit-test-only
window and thirty completed 50 ms work units (about 1.53 s, over seven windows).
It checks distinctive output, no strike and the original generation.

The runtime already renews correctly. To demonstrate the test's new protection,
only `idle_until = last + window(phase)` was replaced by a marked no-op; the
initial allowance, protocol and producer stayed unchanged. The implementation
was staged before mutation and local diff-stat was empty. Applied diff:
`worker_host/liveness.rs | 2 +-` (1 insertion, 1 deletion).

```text
BASELINE/GREEN: real_transport_progress_renews_multiple_idle_windows ... ok
1 passed; finished in 1.53s
RENEWAL-ONLY CONTROL: the same exact test ... FAILED
EngineCrashed: no-progress hang ... completed=3 ... idle_window_ms=200
0 passed; 1 failed; 740 filtered out
```

`git checkout -- <path> && touch <path>` restored renewal with empty local
`git diff --stat`; the corresponding green rerun passed. The renewal test's
short 200 ms idle window is compiled only in unit tests; production windows
remain fixed constants.

## 3. Fence a dropped public low-level host exchange

`worker_liveness::cancelled_low_level_host_starts_fresh_worker_without_strike`
uses the spawned framed fake worker, cancels `host.embed_batch([7])` directly
after it was written, then requests `[9]` on the same public host. The fake's
owned PID log independently verifies a new process/generation; the next value
must be 9. Recorded compute faults and quarantined model counts must stay zero:
the worker was live, and abandoning a caller's wait is lifecycle management,
not evidence that its compute path crashed.

```text
BEFORE: cancelled_low_level_host_starts_fresh_worker_without_strike ... FAILED
EngineCrashed: expected embed-2, got embed-1
0 passed; 1 failed; 25 filtered out
AFTER: same exact test ... ok
1 passed; 0 failed; 25 filtered out; finished in 3.55s
```

The generic host now has an exchange-drop guard. An interrupted write/read marks
the connection unusable immediately. Before reusing a model reference or issuing
the next request, the host closes/reaps only its owned child, forgets its
worker-local references, and starts/reloads a fresh worker. That invalidation
is lifecycle and adds no fault strike. The normal module request path still
leaves its detached owner draining after caller expiry; it retains the original
usable generation rather than invoking this low-level cancellation fallback.

The real-transport renewal test still passed (1) and strict module all-targets
clippy passed after adding the cancellation guard.

## 4. Report CUDA inference phase after input completion, before layer zero

The native instrumentation described below was subsequently rejected by the
RTX 5090 rental. The current worker instead reports completed Rust input
preparation immediately before the unchanged opaque forward, then its successful
completion. Native upload and plan/graph setup now share the 120-second inference
window. The design note records the measured timing margin and the reason for
preserving the master C ABI; the historical ordering proof below is not a claim
that the removed native callbacks worked on a GPU.

All three native ports now report the inference transition in `ShapePlan::run`
after the plan is ready and the actual H2D input/mask uploads complete, on both
cached and cold paths, before submitting/waiting on layer zero. The old report
outside `run` did not establish input-upload completion. With a callback present,
this adds one input-stage host stream synchronization; its performance cost is
unmeasured and is included in the README's CUDA cost discussion. Arithmetic and
numerical kernel revision remain unchanged; production source digests were
updated as instrumentation provenance.

Two real-host framed tests use distinct unit-test-only 200 ms inference and
2-second setup windows. One sends the inference transition then goes silent;
it must strike around 250 ms, not retain setup. The other delays that transition
400 ms inside cold setup and must succeed.

```text
cuda_inference_transition_uses_short_window_before_first_layer ... ok (0.25s)
cuda_cold_setup_keeps_distinct_initial_silence_allowance ... ok (0.40s)
TRANSITION-ONLY CONTROL: remove phase = next_phase, keep all initial windows
cuda_inference_transition_uses_short_window_before_first_layer ... FAILED
inference transition must not retain the two-second setup allowance: Elapsed(())
0 passed; 1 failed; 742 filtered out
```

The production phase assignment was staged before mutation, local diff-stat
was empty, and the marked control changed `liveness.rs | 2 +-` (1 insertion,
1 deletion). Checkout/touch restored it with empty local diff. The host already
handled a delivered transition correctly; this negative control defends that
policy, while native input-stage ordering is separately compiled. No fake host
test is claimed as a CUDA GPU proof. Finding 6 remains the rental check.

## 5. Real-host fragmented and malformed frame fences

Five public-host/Unix-stream adversarial tests were added:

- `fragmented_progress_crossing_idle_deadline_faults_real_host`: a valid progress
  JSON frame sent in two-byte writes straddles the original 200 ms deadline.
  Partial traffic must not count as a completed frame or extend the window.
- `terminal_short_raw_then_eof_faults_real_host`: a matching header declares four
  raw bytes, but only two arrive before EOF.
- `raw_payload_longer_than_terminal_header_faults_real_host`: the raw frame has
  eight bytes while the valid header declares one float.
- `progress_between_terminal_header_and_raw_faults_real_host`: a valid progress
  JSON frame is inserted where the terminal header requires its raw float frame.
- `terminal_header_does_not_renew_raw_deadline_real_host`: a header arrives near
  idle expiry, but the raw body arrives after the original absolute deadline.

All names are under `worker_host::framing_tests::`. Every test requires an error,
no delivered vectors, no reusable connection and one recorded worker fault,
which counts toward the two-consecutive-fault quarantine threshold.
A split frame is not intrinsically invalid: the tested fragmentation deliberately
outlives the idle deadline. Existing production guards already rejected these
cases; this fix supplies executable adversarial coverage, not an invented parser
bug or a claim that normal transport segmentation must fail.

```text
BASELINE/RESTORED: cargo test --locked -p synapse-module --lib worker_host::framing_tests
5 passed; 0 failed; 743 filtered out; finished in 0.30s
```

Each requested case was also shown red in a separate one-test run after
neutralizing the production exchange's owned-child close and fault-accounting
behavior: it returned the error directly instead of closing and classifying the
exchange as a worker fault. The fragmented,
short-EOF, oversized and inserted-progress tests each failed their transport-
fault assertion, rather than a build or HELLO check. The safe staged-source
mutation changed `worker_host/mod.rs | 8 ++------` (2 insertions, 6 deletions);
checkout/touch restored it and local diff-stat became empty.

Targeted additional controls defend the actual byte/deadline boundaries:

```text
SKIP ONLY TERMINAL/RAW SIZE FENCE:
raw_payload_longer_than_terminal_header_faults_real_host ... FAILED
late local decoder saw 2 floats, expected 1 (transport was incorrectly uncharged)
progress_between_terminal_header_and_raw_faults_real_host ... FAILED
late local decoder saw 18 floats, expected 1
OMIT ONLY JSON FRAME DEADLINE:
fragmented_progress_crossing_idle_deadline_faults_real_host ... FAILED
partial bytes must not renew original idle deadline (finished 1.18s)
RESTART RAW DEADLINE AFTER HEADER:
terminal_header_does_not_renew_raw_deadline_real_host ... FAILED
unwrap_err on Ok([[7.0]]) (finished 0.30s)
```

Those were separate named tests with no other test failure in each run. Applied
local diff-stats were respectively `liveness.rs | 2 +-`, `5 ++---` and `3 ++-`;
each contained `NON-VACUITY BREAK`, the marker identifying the intentional
production defect described above, and had staged live source before mutation.
The exact named failures above demonstrate detection. Each restored to empty
local diff after checkout/touch. Production code delivered
with this coverage keeps its existing absolute frame deadlines and raw fences.

## Final verification of the completed host fixes

The complete source at `046145fc4d90572361fce3aa972d6b53582d3d6c` was checked
again after all five host fixes. The preservation-only commit `a3988da8` has
the same source tree; no interrupted edits were discarded. Linux checks used
8 vCPUs, rustc/cargo 1.99.0, clippy 0.1.99 and rustfmt 1.10.0-stable:

| Command | Result |
| --- | --- |
| `cargo test --locked -p synapse-module --lib` | 734 passed, 14 ignored; 748 total, 68.85 s |
| `cargo test --locked -p synapse-module --test it` | 26 passed, 245.12 s |
| `cargo test --locked -p synapse-module --test skeleton_e2e` | 57 passed, 1 profiling test ignored; 126.41 s |
| `cargo test --locked -p synapse-core --lib worker_protocol` | 16 passed |
| `cargo test --locked -p synapse-module --test catalog_fingerprints` | 5 passed, 1 pinned-artifact test ignored |
| `cargo clippy --locked -p synapse-module --all-targets -- -D warnings` | Passed, one package, all targets |
| `cargo clippy --locked -p synapse-core -p synapse-engine-cuda -p synapse-worker-cuda -p synapse-worker-vulkan --all-targets -- -D warnings` | Passed, four portable/feature-off packages, all targets |
| `cargo fmt --all -- --check` | Passed (silent on success) |

The default `candidate_workers` test invocation ran zero tests and ignored its
one packaged-binary matrix. It is not counted as a passing deployment check;
the earlier executed Linux/macOS matrices remain recorded in the hardware
README and worker-liveness test record.

Windows-target clippy was repeated for `synapse-module --all-targets` using
the repository's cross-compilation wrappers `scripts/lib/zig-cc-windows.sh`
and `scripts/lib/zig-ar.sh`, target
`x86_64-pc-windows-gnu`, `CARGO_BUILD_JOBS=4` and `-D warnings`. It passed with
the installed Mac Zig 0.16.0 and clippy 0.1.99. This uses the Mac's installed
cross-toolchain because Zig was unavailable in the Linux build image; it is
not a substitute for the executed native Windows Vulkan certification.

`ckdev-mutate check` passed with ckdev-mutate 0.12.0, validating the catalogue's
66 control anchors and exact test names, without replaying mutations. Its
Mac prebuild warned that Metal developer tools were unavailable; no Metal
inference is claimed. The archived evidence check also passed with Python
3.9.6:

```text
python3 docs/evidence/worker-progress-liveness/summarize_raw.py --raw-root "$HOME/.local/share/cortexkit/synapse/evidence-raw/worker-progress-liveness" --check
PASS: 5 archived hashes, 4 fixture pins, 80 Ally and 17 M4 requests
```

These final host checks do not add native CUDA GPU execution or performance
evidence. The maintainer still needs to validate native execution and compare
latency/throughput with progress reporting disabled and enabled on rented CUDA
hardware. The hardware README describes the unmeasured synchronization, event
and reporting costs.
