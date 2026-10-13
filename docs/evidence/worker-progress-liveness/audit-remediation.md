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
