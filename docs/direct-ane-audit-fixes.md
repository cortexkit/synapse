# Direct ANE serving audit regressions

All commands use `env -u TMPDIR DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer cargo test --locked -p synapse-module --lib <filter>` with Cargo 1.99.0. Failures below are captured from deterministic regression runs, not hardware evidence.

## Generation-qualified recovery

`serving_admission_channel_fault_restarts_exactly_once` failed against the original serving code:

```text
assertion `left == right` failed
  left: 2
 right: 1
test result: FAILED. 0 passed; 1 failed
```

The injected admission timeout had already recovered the owner before serving handled its returned error. Recovery now carries the fault generation and ignores already replaced owners. Admission, eviction, resource retry, and inference paths retain the generation that produced the fault.

`stale_serving_fault_after_recovery_does_not_restart_restored_owner` reproduced the original unqualified dispatch in a staged/restored control:

```text
assertion `left == right` failed
  left: 2
 right: 1
test result: FAILED. 0 passed; 1 failed
```

The control replaced only the qualified serving recovery wrapper with the former unqualified call. Its diff was one file, two insertions and one deletion; restoration returned an empty unstaged diff before verification. Both named tests pass after restoration. The complete mock residency suite passed: 38 tests, no failures.

## Detached task scheduling ownership

`cancelled_direct_ane_keeps_module_permit_and_inflight_until_reply_drains` failed at the module execution boundary:

```text
assertion `left == right` failed: cancelled caller must not release an executing ANE task's permit
  left: 1
 right: 0
test result: FAILED. 0 passed; 1 failed
```

Both embedding and composed rerank now transfer the execution permit, catalog lane guard, and activity guard into the detached inference task. The same module-boundary test passes for both operations: while the reply is gated, the sole execution permit remains held and in-flight accounting stays at one; after draining, both return to idle.
