# Direct Neural Engine serving

Catalog direct-ANE models run in `ck-synapse-worker-ane-direct`, using the private
operating-system Neural Engine framework. This is a product dependency, not a
promise of availability on every Apple platform. Generic worker lanes and the
direct-ANE residency path have different ownership requirements.

## 1. Own admission before inference

The first direct-ANE load creates a shared cross-process lane lock and residency
supervisor. Each loaded model owns its worker generation and admitted shapes.
Workers inherit the hardware lock, so it remains held while any owned worker can
still use the device. Reservations are charged by owning worker and shape, not
merely by package identity.

Check: `identical_models_in_distinct_workers_each_admit_their_own_shape` in module
worker-host residency tests.

## 2. Lease one shape through its complete reply

Composed embed or rerank sequences use the smallest worker-reported bucket that
fits. Each group admits and leases its shape, validates its reply, then releases
the lease before requesting the next bucket. Responses are restored to input
order. Empty or over-ceiling composed sequences are refused before inference;
there is no silent truncation in this path.

Checks: the module's residency/serving tests, including
`packed_cancelled_mixed_call_drains_with_one_lease_at_a_time`; catalog refusal
coverage includes `catalog_semantic_refusals_match_golden_and_never_fetch`.

## 3. Keep ownership when the caller leaves

A detached task owns the lease, execution permit, catalog guard and in-flight
count. Caller cancellation does not restart a healthy worker: the task drains its
reply before releasing ownership. The absolute deadline bounds both admission
and exchanges, and no later bucket starts after expiry.

Checks: `serving_caller_cancellation_drains_reply_without_restart`,
`cancelled_direct_ane_keeps_module_permit_and_inflight_until_reply_drains`,
`direct_ane_warm_reply_after_absolute_deadline_is_discarded`, and
`direct_ane_cold_admission_after_absolute_deadline_is_drained_without_inference`.

## 4. Fence recovery by worker generation

I/O errors, malformed replies and bounded exchange failures go through confirmed
exit recovery after the lease is released. One fault restarts its generation once;
a stale fault does not restart a recovered owner. An exchange that may already
have executed returns an error rather than being silently replayed.

Checks: `serving_admission_channel_fault_restarts_exactly_once` and
`stale_serving_fault_after_recovery_does_not_restart_restored_owner`.

## 5. Refund only after release or confirmed exit

Retirement stops new leases and replacement spawning, drains in-flight work,
closes the channel, and confirms exit before refunding reservations. An
unconfirmed exit keeps resources charged. Retirement is checked in the locked
grant decision, and recovery rechecks it before connecting a replacement.

Checks: `retirement_between_precheck_and_grant_leaves_no_phantom_reservation`,
`retirement_during_exit_confirmation_never_connects_replacement`, and
`serving_io_fault_recovers_and_retire_refunds_only_after_exit`.

## 6. Certification observes independent inventories

With `certify_observation` enabled, the supervisor records worker inventories and
an independently maintained admission count. Missing inventories must not look
like a complete empty observation. Ordinary serving does not record this stream.

Checks: `serialized_inventory_omission_fails_placement_gate` and
`validator_refuses_placement_inventory_count_mismatch` in `synapse-certify`.

## Verify and reverse a change

Portable mock-transport proof, from the repository root:

```sh
cargo build --locked --no-default-features -p synapse-worker-cuda -p synapse-worker-vulkan
cargo test --locked -p synapse-module --features test-support --lib --test it --test skeleton_e2e
```

For a real candidate, follow [backend certification](backend-certification.md).
Use an exclusive hardware slot and the pinned checkpoint; never bypass a live
lane's lock or stop unrelated processes to obtain a slot. Full Xcode is required
for Apple shader builds, and the native ANE compiler requires its normal per-user
temporary directory (`env -u TMPDIR` for hardware commands).

To reverse a serving change, restore the compatible module/worker pair and its
matching certified identity. Do not reuse approvals or vector-space identity
from a different binary, model, conversion policy or OS build.
