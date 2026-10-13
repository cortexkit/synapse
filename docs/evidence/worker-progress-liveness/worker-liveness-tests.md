# Worker progress liveness: real transport tests

The independent catalog quarantine correction and its three red controls are
recorded in [quarantine-contract.md](quarantine-contract.md). This record covers
completed-work liveness, deadlines and consecutive-fault accounting.

Tools: rustc 1.99.0 (b940084d7 2026-09-28), cargo 1.99.0
(5f94df478 2026-08-27), clippy 0.1.99 (b940084d7e 2026-09-28).
All acceptance/red controls ran on Linux with 8 vCPUs. The slow worker uses the
spawned timeout mock over the real host transport, with 45 seconds of work and
one progress frame per completed one-second unit. That 45-second process test
proves the former 30-second total cap is gone; because 45 seconds is less than
the 120-second idle window, it does not by itself prove renewal. End-to-end
renewal is defended separately by `real_transport_progress_renews_multiple_idle_windows`
with a short unit-test-only host window and by removing only the renewal line.
Hung cases use the actual
120-second inference window, not a shortened production setting. The deadline
case uses the production caller-wait helper and real framed Unix stream with a
45-second exchange owner; its next request must return 9, not the abandoned
request's 7, with the same worker generation and zero strikes. Direct ANE's
framed channel tests use a unit-test-only short idle allowance and confirmed
peer exit; no production override exists.

## Green runs

```text
cargo test --locked -p synapse-module --test it worker_liveness
running 4 tests
worker_liveness::slow_progress_worker_exceeds_30_seconds_without_crash_strike ... ok
worker_liveness::stalled_progress_worker_strikes_within_idle_window ... ok
worker_liveness::two_sequential_hangs_quarantine ... ok
worker_liveness::hang_success_hang_does_not_quarantine ... ok
test result: ok. 4 passed; 0 failed; 21 filtered out; finished in 243.24s
```

```text
cargo test --locked -p synapse-module --lib live_worker_deadline_drains_before_next_correct_response
worker_host::progress_tests::live_worker_deadline_drains_before_next_correct_response ... ok
test result: ok. 1 passed; 0 failed; finished in 45.06s
```

```text
cargo test --locked -p synapse-module --lib ane_residency::quarantine_tests
worker_host::ane_residency::quarantine_tests::two_consecutive_ane_hangs_use_catalog_quarantine ... ok
worker_host::ane_residency::quarantine_tests::ane_hang_success_hang_does_not_quarantine ... ok
test result: ok. 2 passed; 0 failed; 734 filtered out; finished in 0.05s
```

The final module library run also included these tests:
`724 passed; 0 failed; 14 ignored` (738 total). The integration target passed
25 tests, and `skeleton_e2e` passed 57 with 1 profiling test ignored. Frozen
catalog fingerprint checks passed 5 with 1 artifact-dependent test ignored.

## Red production controls and restoration

For each control, the implementation files were staged and the local
`git diff --stat` was empty. The marked `NON-VACUITY BREAK` then produced the
non-empty diff specified below. After the test, `git checkout -- <path> &&
touch <path>` restored the staged implementation and local diff-stat was empty.
Private v3 handshake/scaffolding remained intact: these failures reached the
request or deadline behavior, not a rejected HELLO. Each named command below
ran one test; no additional test failed in that run.

### Slow but alive: restore the 30-second total cap

Production change: renew `idle_until` from request start + 30 seconds instead
of last completed work + idle window. Path:
`crates/synapse-module/src/worker_host/liveness.rs`.
Applied diff: `3 ++-` (2 insertions, 1 deletion); restored diff: empty.

```text
cargo test --locked -p synapse-module --test it slow_progress_worker_exceeds_30_seconds_without_crash_strike -- --nocapture
worker_liveness::slow_progress_worker_exceeds_30_seconds_without_crash_strike ... FAILED
unwrap on Err: EngineCrashed { stage: "embed_batch", detail: "... no-progress hang: EMBED_BATCH req_id=embed-1 completed=30 phase=Inference idle_window_ms=120000", ... }
test result: FAILED. 0 passed; 1 failed; 24 filtered out; finished in 31.24s
```

### Hang rule: a total cap is not an idle window

The same restored total-cap behavior must not kill a request after only 30
seconds of silence when the chosen idle window is 120 seconds. This test checks
both the lower bound and detection by 125 seconds, plus one strike.
Path and applied/restored diff are the same as the previous control.

```text
cargo test --locked -p synapse-module --test it stalled_progress_worker_strikes_within_idle_window -- --nocapture
worker_liveness::stalled_progress_worker_strikes_within_idle_window ... FAILED
assertion: a progress frame is not a terminal reply [elapsed lower bound failed]
test result: FAILED. 0 passed; 1 failed; 24 filtered out; finished in 31.32s
```

### Caller deadline: restore waiting for native completion

Production change: await the owner instead of selecting the caller deadline.
Path: `crates/synapse-module/src/lib.rs`.
Applied diff: `5 ++---` (2 insertions, 3 deletions); restored diff: empty.

```text
cargo test --locked -p synapse-module --lib live_worker_deadline_drains_before_next_correct_response -- --nocapture
worker_host::progress_tests::live_worker_deadline_drains_before_next_correct_response ... FAILED
assertion: deadline must answer before native work finishes
test result: FAILED. 0 passed; 1 failed; 735 filtered out; finished in 45.05s
```

### Two sequential hangs: restore rolling-window retention

Production change: expire strikes when the previous fault is older than 60
seconds, and do not clear strikes on a healthy serving result. This restores
the old policy's behavior for these two cases. Path:
`crates/synapse-module/src/worker_host/mod.rs`.
Applied diff: `9 ++++++++-` (8 insertions, 1 deletion); restored diff: empty.

```text
cargo test --locked -p synapse-module --test it two_sequential_hangs_quarantine -- --nocapture
worker_liveness::two_sequential_hangs_quarantine ... FAILED
assertion left == right failed: left: 1, right: 2
test result: FAILED. 0 passed; 1 failed; 24 filtered out; finished in 241.42s
```

### Hang, success, hang: old history is not consecutive faults

Same production control, in a separate one-test run:

```text
cargo test --locked -p synapse-module --test it hang_success_hang_does_not_quarantine -- --nocapture
worker_liveness::hang_success_hang_does_not_quarantine ... FAILED
assertion: drained serving success must clear strikes; left: 1, right: 0
test result: FAILED. 0 passed; 1 failed; 24 filtered out; finished in 121.42s
```

The old healthy-completion and short-timeout tests changed deliberately:
`crash_window_health_degrades_then_recovers` became
`fault_health_does_not_recover_with_elapsed_time`, because time alone no longer
clears faults; `load_uses_long_timeout_but_embed_keeps_short_timeout` became
`catalog_inference_uses_fixed_idle_window_not_owned_decode_timeout`, because
catalog inference no longer has a configurable total cap. The existing legacy
soak observable remains two aborted generations then `probe_required` on the
third request: internal reload never clears serving strikes.

### Direct ANE: restore its original no-strike recovery path

Production change: neutralize `AneWorkerChannel::charge_fault`. Path:
`crates/synapse-module/src/worker_host/mod.rs`.
Applied diff: `13 +++----------` (3 insertions, 10 deletions); restored diff:
empty. These are separate one-test runs:

```text
cargo test --locked -p synapse-module --lib two_consecutive_ane_hangs_use_catalog_quarantine -- --nocapture
worker_host::ane_residency::quarantine_tests::two_consecutive_ane_hangs_use_catalog_quarantine ... FAILED
assertion: quarantine confirms exit without spawning again; left: 3, right: 2
test result: FAILED. 0 passed; 1 failed; 735 filtered out; finished in 0.04s
```

```text
cargo test --locked -p synapse-module --lib ane_hang_success_hang_does_not_quarantine -- --nocapture
worker_host::ane_residency::quarantine_tests::ane_hang_success_hang_does_not_quarantine ... FAILED
assertion left == right failed: left: 0, right: 1
test result: FAILED. 0 passed; 1 failed; 735 filtered out; finished in 0.04s
```

The second test checks that its final hang still charges one strike. Merely
asserting “not quarantined” would be vacuous on the old path, which had no
strike book at all.

## Tooling and platform checks

- Private protocol v3 codec: 16 passing core protocol tests. The Linux packaged
  HELLO/PING/refusal/SHUTDOWN matrix passed for llama, CUDA and Vulkan, and the
  dedicated llama v3 integration test passed. No GPU inference was claimed by
  these feature-off checks.
- Module and portable worker clippy with `-D warnings` passed. Mac-only direct
  ANE and Swift Core ML producer clippy passed with 4 build jobs; no ANE
  inference was run by that compilation step.
- The Linux build image initially lacked nvcc and CMake. CUDA compilation used
  the six checksum-pinned NVIDIA redistributables from the repository's release
  setup, downloaded locally and supplied offline to the Linux job. Native
  C++ compilation passed with nvcc 13.2.78. Native-feature clippy exposed four
  pre-existing findings (three old high-arity forward APIs and the existing
  non-Send/Sync Arc model storage); the narrower native check passed with only
  those two lint classes excluded. Portable and module clippy stayed strict.
- llama compilation used official checksum-verified CMake 3.31.6
  (`5a1133ff103c71eb5120e2cc3de922733e7d8a26a98ae716397e8676adb367bf`),
  supplied offline. Temporary SDK archives are not committed.
- The Vulkan/ANE numerical arithmetic, kernel revision and frozen lane
  fingerprints remain unchanged. CUDA source digests classify event reporting
  as instrumentation in `PORT-PROVENANCE.md`, not a numerical revision.

Hardware outcomes and measured progress gaps are recorded separately; these
framed fake-worker tests do not certify an accelerator.

## Swift version invariant after the clean cutover

`ckdev-mutate check` initially found the old Swift control's v2 anchor missing.
The manifest now mutates the actual v3 declaration to v4 while retaining the
same guard/test claim. A safe staged-source control on the Mac produced:

```text
tests::swift_protocol_version_matches_the_worker_protocol_version ... FAILED
assertion left == right failed: left: 4, right: 3
test result: FAILED. 0 passed; 1 failed; 1 filtered out; finished in 0.00s
```

Applied source diff: `crates/synapse-worker-ane/swift/ane_worker.swift | 3 ++-`
(2 insertions, 1 deletion). After `git checkout -- <path> && touch <path>` the
local diff-stat was empty and the same test passed (1 passed, 1 filtered out).
This source-scan check ran no Neural Engine prediction.
