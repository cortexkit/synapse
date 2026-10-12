# Catalog worker quarantine contract correction

This correction is independent of replacing fixed worker request timeouts with
completed-work liveness. The existing
Windows record at
`../catalog-backends-windows/runs/6fbf7209/probe.qwen3-reranker-0.6b.jsonl:41`
returned permanent `probe_required` for a catalog Vulkan lane after two worker
faults. That contradicted the catalog/legacy separation in `ARCHITECTURE.md`.

Catalog ownership now comes from the release registry's reserved-ID lookup,
travels outside the stored numerical profile, and is passed explicitly to the
worker host. A quarantined catalog worker answers transient `engine_crashed`,
with the remaining ten-minute cool-down and `{lane_id, quarantined: true}`.
The next request after expiry clears strikes and reloads a fresh worker;
unload/reload clears quarantine immediately. Legacy workers still require their
existing probe/recovery procedure. Fingerprints and worker protocol v2 are
unchanged in this correction; progress and consecutive-fault accounting follow
in a separate change.

## Boundary tests and green run

Tests use the host's actual framed Unix stream and crash accounting. The fake
peer completes LOAD then disconnects after reading inference. Its placeholder
child is owned by the test and its launch nonce is stripped. A host-local clock
injected only in unit tests advances the cool-down without a ten-minute sleep.
The recovery test supplies a new peer generation, verifies that LOAD precedes
serving, checks the distinct vector result and zero strikes. It does not claim
physical GPU certification.

Command:

```text
cargo test --locked -p synapse-module --lib worker_host::quarantine_tests
```

Tools: rustc 1.99.0 (b940084d7 2026-09-28), cargo 1.99.0
(5f94df478 2026-08-27). Runs were on Linux with 8 vCPUs except the two narrow
controls noted below, which used 4 after an 8-vCPU job was refused for capacity.

Final green run:

```text
running 4 tests
quarantined_catalog_lane_returns_transient_engine_crashed_with_remaining_cooldown ... ok
catalog_unload_clears_quarantine_without_waiting_for_cooldown ... ok
quarantined_legacy_lane_keeps_probe_required_without_auto_recovery ... ok
catalog_quarantine_cooldown_reopens_serving_with_zero_strikes ... ok
test result: ok. 4 passed; 0 failed; 0 ignored; 729 filtered out; finished in 0.02s
```

All four names above are under `worker_host::quarantine_tests::`.
The first mapper-only run failed the catalog test on missing `details`
(`left: None`, expected `{lane_id, quarantined: true}`), with the other two
then-existing tests passing. The controls below demonstrate the original
incorrect disposition, permanent quarantine, and the catalog-only boundary.

## Red controls, restored before committing

Each control was applied only after staging the catalog cool-down implementation and
confirming an empty local `git diff --stat`. Each was marked
`NON-VACUITY BREAK`, followed by a non-empty local diff, the same four-test
command, then `git checkout -- <path> && touch <path>` and an empty local diff.
Remote jobs use their own Git index, so mutation diff-stat evidence was captured
**locally**, not inferred from a remote snapshot's index.

### Original catalog `probe_required` disposition

Changed `worker_error_to_wire` back to `StableError::probe_required`, which asks
a caller to perform legacy probe/approval recovery instead of retrying after a
cool-down. Applying it to catalog work restores the original contract violation.
Applied diff:
`crates/synapse-module/src/lib.rs | 3 ++-` (2 insertions, 1 deletion).
After restore: empty.

```text
quarantined_catalog_lane_returns_transient_engine_crashed_with_remaining_cooldown ... FAILED
catalog_quarantine_cooldown_reopens_serving_with_zero_strikes ... ok
catalog_unload_clears_quarantine_without_waiting_for_cooldown ... ok
quarantined_legacy_lane_keeps_probe_required_without_auto_recovery ... ok
assertion left == right failed: left: "probe_required", right: "engine_crashed"
test result: FAILED. 3 passed; 1 failed; 729 filtered out; finished in 0.02s
```

### Original permanent quarantine behavior

Disabled the expiry branch in `WorkerHost::quarantine_error`, preventing the
catalog lane from automatically reopening when ten minutes have elapsed.
Applied diff:
`crates/synapse-module/src/worker_host/mod.rs | 2 +-` (1 insertion, 1 deletion).
After restore: empty. This control ran with 4 vCPUs after the Linux scheduler
refused 8 because 60/64 permits were occupied; no other process was stopped.

```text
catalog_quarantine_cooldown_reopens_serving_with_zero_strikes ... FAILED
quarantined_catalog_lane_returns_transient_engine_crashed_with_remaining_cooldown ... ok
quarantined_legacy_lane_keeps_probe_required_without_auto_recovery ... ok
catalog_unload_clears_quarantine_without_waiting_for_cooldown ... ok
unwrap on Err: Quarantined { key: "{\"artifact_path\":\"/tmp/quarantine-fixture\"}", retry_after_ms: Some(1) }
test result: FAILED. 3 passed; 1 failed; 729 filtered out; finished in 0.03s
```

### Legacy isolation guard

Removed the catalog-only discriminator so transient `engine_crashed` plus
cool-down recovery also applied to legacy workers instead of `probe_required`.
Applied diff:
`crates/synapse-module/src/lib.rs | 3 ++-` (2 insertions, 1 deletion).
After restore: empty. This narrow control used 4 vCPUs.

```text
quarantined_legacy_lane_keeps_probe_required_without_auto_recovery ... FAILED
catalog_unload_clears_quarantine_without_waiting_for_cooldown ... ok
quarantined_catalog_lane_returns_transient_engine_crashed_with_remaining_cooldown ... ok
catalog_quarantine_cooldown_reopens_serving_with_zero_strikes ... ok
assertion left == right failed: left: "engine_crashed", right: "probe_required"
test result: FAILED. 3 passed; 1 failed; 729 filtered out; finished in 0.04s
```

The legacy assertion is already true on the old implementation, so restoring
that implementation cannot make it red. This control proves the guard instead
of falsely claiming an unchanged property failed on its correct baseline.

## Gates and setup corrections

- Module lib suite: 718 passed, 14 ignored initially; the one failed case was
  the feature-off GPU worker fixture because its two binaries had not been
  built. After `cargo build --locked -p synapse-worker-cuda
  -p synapse-worker-vulkan --bins`, rerunning the failed case passed (1 test).
- `cargo test --locked -p synapse-module --test it`: 20 passed initially; the
  nonce-stripping source guard found the new test child constructor. The child
  now uses `without_launch_nonce` before conversion to Tokio, and rerunning
  that guard passed (1 test). No guard expectation was weakened.
- `cargo test --locked -p synapse-module --test skeleton_e2e`: 57 passed,
  1 profiling test ignored. The load/reload case was rerun after the load-context
  refactor and passed (1 test).
- `cargo test --locked -p synapse-module --test catalog_fingerprints`:
  5 passed, 1 artifact-dependent test ignored; frozen fingerprints unchanged.
- `cargo clippy --locked -p synapse-module --all-targets -- -D warnings`:
  passed with clippy 0.1.99 (b940084d7e 2026-09-28). Clippy initially caught the
  eighth loader argument; catalog ownership and the stored spec now travel in
  one explicit context object instead of suppressing the lint.
- `cargo fmt --all -- --check`: passed with rustfmt 1.10.0-stable
  (b940084d7e 2026-09-28).

The end-to-end catalog tests are a submodule of `skeleton_e2e`, not a separate
`catalog_e2e` Cargo target. An initial command naming that nonexistent target
was corrected before any suite ran. All package gates above used Linux with
8 vCPUs; the permanent-quarantine and legacy-isolation controls used 4 vCPUs
after the stated scheduler capacity refusal.
No package manifest or lockfile changed. Windows-target clippy also passed:
`CC_x86_64_pc_windows_gnu="$PWD/scripts/lib/zig-cc-windows.sh"
AR_x86_64_pc_windows_gnu="$PWD/scripts/lib/zig-ar.sh" CARGO_BUILD_JOBS=4
cargo clippy --locked --target x86_64-pc-windows-gnu -p synapse-module
--all-targets -- -D warnings`, using the installed Mac Zig 0.16.0 and the same
Rust/clippy versions. The Linux attempt could not run because Zig was absent;
no local test-suite fallback was used. `ckdev-mutate check` requires clean
tracked target files and is run after committing this correction. Accelerator
certification and progress-gap measurements remain part of the subsequent
worker-liveness step.
