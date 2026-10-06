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

## Absolute request deadlines

Both `direct_ane_warm_reply_after_absolute_deadline_is_discarded` and `direct_ane_cold_admission_after_absolute_deadline_is_drained_without_inference` failed on the original deadline-blind module path:

```text
called `Result::unwrap_err()` on an `Ok` value: [[128.0]]
test result: FAILED. 0 passed; 2 failed
```

The absolute deadline now reaches the serving task and the shared rung runner. Queue waits use the remaining deadline; expired requests dispatch no additional rungs. The caller returns the existing `deadline_exceeded` wire error on time, while already active admission/inference RPCs finish their response read and accounting with their scheduling guards still owned. Late successful results are discarded without a restart. Both deadline tests pass, and the cancellation/guard regression still passes.

## Owner-qualified residency keys

`identical_models_in_distinct_workers_each_admit_their_own_shape` failed when two channels loaded identical profile/package inputs through one shared supervisor:

```text
called `Result::unwrap()` on an `Err` value: WorkerErr { code: "shape_not_admitted", msg: "ane-direct:gte-modernbert-base.ane-direct-worker:sha256:test rungs [128] are not resident" }
test result: FAILED. 0 passed; 1 failed
```

Residency keys now include the owning worker ID in addition to model reference and shape. Budget totals and per-model caps remain shared. The regression passes: two admissions reserve 44 executables, retiring A refunds only its 22, and B continues serving. All 39 mock residency tests pass; existing assertions were retained.

## Atomic retired-owner admission refusal

`retirement_between_precheck_and_grant_leaves_no_phantom_reservation` pauses the real lease path after its retired precheck, completes retirement, then resumes the grant decision. The original decision created a failed slot and retained its charge:

```text
assertion `left == right` failed: retired owner must not acquire an Admitting reservation
  left: 4
 right: 0
test result: FAILED. 0 passed; 1 failed
```

`next_step` now checks retirement under the same state lock that grants leases and reserves admission slots, returning a terminal refusal instead of a grant. The barrier regression and existing queued-retirement test both pass; no slot or worker admission remains.
