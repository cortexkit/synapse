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
