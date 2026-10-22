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

## Admission fairness and eviction

Apple's Neural Engine compiler serves one layer at a time for the whole
machine, so shape compiles run side by side gain no throughput and make a
small shape wait one wide layer per layer of its own. The supervisor therefore
runs **at most one cold shape compile at a time** across all direct-ANE
workers. Queued compiles start in order of a fixed start key, their arrival
plus their predicted cost: cheapest first, and a wide compile is passed only by
compiles that arrive less than its own cost after it. A compile with a known
cost waits **50 ms** after arrival before it may start, so compiles requested
together start cheapest first. The predicted cost is the worker's seed estimate
(layers times a measured per-layer time for the model's width), replaced by the
measured time once a shape has compiled and scaled for unmeasured shapes by
how much slower this machine compiled so far. A queued compile reserves no
executable budget until it starts; the queue's head may evict while the
running compile finishes, so its compile starts at once.

A request whose shape is predicted to be ready only after its deadline, or after
its 600 s admission bound, is answered at once with `shape_compiling` and the
predicted delay as `retry_after_ms`. Its compile stays queued, so the retry
finds the shape ready. A queued compile no request waits for any more is
dropped after **twice** the delay it last promised, and at once when its worker
retires, so a caller that never retries does not keep compiling and evicting
shapes others use. A new compile is not ordered ahead of a queued one if that
would make an accepted request miss its bound.

The supervisor reserves at most **100 layer executables**, including pending
admissions and evictions. A leased shape is never an eviction candidate. Eviction
refunds its charge only after the worker acknowledges release; a busy stream is
not permission to refund early or to unlock a request before its reply.

When a single-flight admission completes, the tickets that actually joined it
during compilation receive a closed handoff cohort. They may take their first
leases ahead of a starved budget head. The shape is protected until every cohort
member leases or leaves the queue, or **250 ms after admission completion**,
whichever comes first. A suspended joiner cannot pin residency forever: expiry
wakes the budget queue even without another worker event. Requests arriving after
completion cannot join or extend the cohort. Outside that bounded exception,
resident traffic still yields to the starved FIFO budget head.

Among unleased, non-handoff resident candidates, eviction prefers an **idle worker
stream**, then a shape with **no queued demand**, then a non-smallest rung, then
LRU. A model at its shape limit must evict one of its own shapes. Idle candidates
are reclaimed first, repeating if more executable budget is needed; a busy-stream
victim remains available when idle candidates cannot free the required budget.
Stream idleness is a scheduling hint, not an ownership guarantee: the actual RPC
still acquires the stream mutex and holds it for the entire request/reply exchange.

Portable checks include `joined_admission_hands_off_before_a_starved_head_and_excludes_newcomers`,
`suspended_joiner_handoff_expires_without_an_external_wakeup`,
`eviction_prefers_an_idle_stream_and_evicts_before_the_long_compile_ends`, and
`eviction_uses_a_busy_stream_when_it_is_the_only_option`. The compile queue is
checked against a stand-in for the shared per-layer compiler, on a paused clock,
by the tests in `crates/synapse-module/src/worker_host/compile_queue_tests.rs`.

### Development hardware stress cases

`crates/synapse-worker-ane-direct/tests/run_stress.py --case residency-safety`
runs the ignored 37-request concurrent residency-safety workload, including its
numerical, leased-eviction, budget and hardware-resource-refusal checks.
`--case cold-throughput` runs the same cold workload as characterization: its
report records completions, errors and wall time without a throughput assertion.
`--case compile-scheduling` times cold compiles of `gte-modernbert-base` and
`qwen3-embedding-0.6b` alone, a small shape asked beside a wide one, six shapes
asked at once, and a request the queue refuses early. It asserts that no two
compiles overlap and no resource refusal occurs; latency is judged by the
operator's runner against the standalone times in the same report.
Run either only in an exclusive hardware window; neither is release certification.

Both use a **test-only 2100 s admission-wait backstop**, not a production deadline
change. The measured `qwen3-embedding-0.6b` admission path ended at 528.815 s
with 492.198 s spent in five compile/load RPCs. `gte-modernbert-base` then needed
another 118.356 s, already beyond 600 s with ten model/shape keys still unmeasured.
Three times that 647.171 s path, rounded up, budgets safety coverage rather than
predicting a full cold run. Production admission waiting remains bounded at 600 s.

Schema **2** adds the case, wall time, test admission bound and a payload-free
admission timeline. Events record monotonic milliseconds since test start,
request id, model reference, shape and ticket when available: queue entry,
reservation, stream grant, RPC start/end, eviction dispatch/acknowledgement and
lease grant. Detached exchanges retain the initiating request's identity; eviction
events identify the victim's model and shape. Stream grant is distinct from queue
entry, so a compile's RPC duration does not hide stream contention. Schema 1
reports remain historical artifacts, not schema 2 inputs.
