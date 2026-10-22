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
`eviction_prefers_an_idle_stream_and_admits_before_the_long_compile_ends`, and
`eviction_uses_a_busy_stream_when_it_is_the_only_option`.

### Development hardware stress cases

`crates/synapse-worker-ane-direct/tests/run_stress.py --case residency-safety`
runs the ignored 37-request concurrent residency-safety workload, including its
numerical, leased-eviction, budget and hardware-resource-refusal checks.
`--case cold-throughput` runs the same cold workload as characterization: its
report records completions, errors and wall time without a throughput assertion.
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

## Compiled programs survive a worker restart

macOS's `aned` keeps compiled Neural Engine programs on disk, keyed by the
program's content identifier (hashes of its MIL text, the Model Intermediate
Language program the binding emits, and of its weights and options) and
the worker's code-signing identifier. The vendored binding emits constants in a
fixed order, so the same layer at the same shape always has the same identifier.
When aned already holds that program, the worker loads it without writing
sources or compiling: on an M6 mini a cached layer loaded in 3.5–45 ms against
0.37 s (width 128) to 15.8 s (width 4096) to compile it, with byte-identical
output. Otherwise, or if that load fails, the layer is compiled as before.

Only a compile writes to the per-user temp folder, so `ane_temp_space_low` is
checked before each layer that must be compiled, never before a cached load.
Each shape admission writes one stderr line, which the module logs at INFO:

```text
ane-direct shape admission: shape=128 layers_loaded=22 cache_hit=22 compiled=0 total_ms=95.3 outcome=admitted
```

With `SYNAPSE_ANE_PROFILE_DIR` set, the `admit` profile record also carries
`cache_hit`, `compiled` and per-layer `layer_loads` rows (`origin` and `ms`).

The cache is aned's: an OS update, 14 days without use, storage pressure or a
changed signing identifier empties it, and the next admission compiles once.
Neither the cache nor its disk use is visible to Synapse.

aned files each cached program under the worker binary's code-signing
identifier, not under its code hash. Production signs the worker with
`codesign --identifier ck-synapse-worker-ane-direct`, so a rebuilt or upgraded
worker that keeps that identifier still finds the programs its predecessor
compiled; a worker signed under another identifier (an unsigned `cargo build`
is linker-signed as `ck_synapse_worker_ane_direct-<hash>`) has its own, empty
cache. Keep the identifier stable across deploys.

A cached load that the Neural Engine refuses for lack of resources is returned
at once as `ane_resources_exhausted`, without compiling; the residency
supervisor evicts and retries. Only an unusable cached copy falls back to a
compile.

Checks: `graph::compile::tests::identical_graphs_emit_identical_sources` and
`client::tests::a_cached_program_is_loaded_without_writing_or_compiling` in the
vendored binding (`cargo test --locked -p ane --lib`, macOS), and
`backend::temp_space_tests::a_layer_aned_holds_is_not_refused_on_a_full_disk`.
