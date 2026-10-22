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

The queue head's prediction is checked against its bound before anything is
dispatched or evicted for it. A wait on the compile queue that reaches the 600 s
backstop is answered `shape_compiling` with a fresh prediction, not
`ane_resources_exhausted`. A compile that failed or whose admission task
panicked keeps the compile gate, and its budget, until the old worker process's
exit is confirmed. They are released then even if the replacement fails to start,
so one broken worker does not stall the others; while the exit is unconfirmed
they stay held. A recovering worker's queued compiles are not
elected, so they cannot hold up other workers. Compile times are learned from the
moment the admission reaches the worker, and never from retried compiles or
packed programs.

One scheduler task makes every compile-queue decision for all direct-ANE
workers: it elects the next compile, reserves budget, evicts, answers waiting
requests at their bounds, keeps promised compiles, observes every compile,
eviction and recovery it starts (a panic included) and drives worker recovery.
Requests only lease resident shapes on a lock-only fast path or wait for its
answer. A compile that failed with an unknown outcome holds the compile gate
until the old process's exit is confirmed, or for at most 35 s after its restart
began; a worker answering an admission with an error frees the gate at once.
After a confirmed exit every slot of the old process is released. If the
scheduler itself stops, every request is answered `engine_crashed` with the lane
id until the module restarts.

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

### Predictions use the cache

While loading a model, the worker asks aned whether it holds the first and the
last layer of each ladder rung. Layers compile in order within one admission,
so a cached last layer means the whole rung was compiled once. Checking every
layer would cost about as much as loading the rung. Both layers of all seven
rungs take about 1.2 s of worker CPU for Qwen3. The worker logs one line:

```text
ane-direct compile cache check: model_ref=... cached=[128, 256, 512] checked=7 total_ms=1180.4
```

`LOADED` carries the cached rungs in the optional `ane_cached_shapes` field.
`ADMITTED` carries the optional `cache_hit_layers` and `compiled_layers`
counts. Both are additions to worker protocol v3 that need no version change:
an older worker omits them, and an older module ignores them.

The compile queue (`docs/design-ane-compile-scheduling.md` §7) predicts a rung
reported cached at its load cost rather than its compile cost. Load times are
learned separately from compile times and never change the machine factor. An
admission predicted as a load that runs past twice that prediction plus 1 s is
repriced as a compile, because aned is evidently compiling it. Knowledge of a
cached rung comes from `LOADED` and from every admission the worker reports.
With a worker that sends neither field, every rung is predicted and learned as
a compile, as before.

### Prewarm

Once a direct-ANE catalog lane is ready (its self-check passed), the module
loads rungs 128, 256 and 512 of it in the background:

- at the lowest priority, one admission at a time, only while no request waits;
- only into free executable budget and free shape slots, never evicting;
- the first rung that does not fit ends the prewarm.

Packed preparation outranks prewarm: it waits for a running prewarm admission
instead of giving up. A request that arrives while a prewarm admission runs
waits for that one admission, which is about 2 s on a cached machine and one
small compile on a cold one. Unloading the lane, pausing the Neural Engine
(`lanes.pause` unloads every lane) and retiring the worker cancel the rest.
A running admission finishes first. Each worker is prewarmed once. The end is
logged once at INFO, `direct-ANE prewarm finished` (or `cancelled`), with the
fields `lane`, `loaded` (the rungs admitted), `loaded_from_cache` and
`compiled` (how many of those rungs), `already_resident`,
`skipped_for_budget`, `failed` and `total_ms`.

Checks: `graph::compile::tests::identical_graphs_emit_identical_sources` and
`client::tests::a_cached_program_is_loaded_without_writing_or_compiling` in the
vendored binding (`cargo test --locked -p ane --lib`, macOS), and
`backend::temp_space_tests::a_layer_aned_holds_is_not_refused_on_a_full_disk`.
