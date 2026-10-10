# Direct ANE serving

How the module serves catalog models on the direct Apple Neural Engine worker
(`ck-synapse-worker-ane-direct`), which compiles each (model, sequence length)
shape on demand through the private `AppleNeuralEngine.framework` API.

## Ownership

Other worker lanes go through the generic `WorkerEngine`, which sends each
request straight to its worker. The direct-ANE worker can't take a request for
a shape it hasn't compiled and admitted, so direct-ANE loads use their own
backend, one that admits shapes before it sends anything. The first direct-ANE
load in a module process creates two things that every later direct-ANE load
shares:

- **The direct-ANE lane lock.** It's a cross-process lock, so only one module
  process at a time drives the direct-ANE hardware. Each worker inherits the
  lock's file descriptor, so the lock stays held until every worker the module
  started has exited, even if the module itself dies first.
- **One residency supervisor.** The Neural Engine limits how many compiled
  layer executables one process can hold (about 115 on an M5 Max). The
  supervisor charges each shape its layer count against a budget of 100,
  admits, leases and evicts shapes, and runs confirmed-exit recovery when a
  worker stays out of resources. One supervisor serves every worker, because
  the limit is hardware's, not a worker's.

Each loaded model gets its own worker process and channel, with a worker ID
unique to that load. Residency is keyed by (worker ID, model reference, shape),
so two models loaded from the same package in two processes each admit their
own shapes.

Workers are spawned through `WorkerHost`. It forwards their logs, removes the
daemon launch nonce from the child's environment, and checks their HELLO. The
channel caches each successful LOAD and replays it after a restart. LOAD and
PONG replies carry the worker's bucket ladder (its sequence-length shapes,
ending at 8192), and serving uses that ladder rather than one of its own.

## Requests

Embedding sequences and fully composed rerank sequences are grouped by the
smallest bucket that fits them, and run bucket by bucket. Each group leases
its shape, which admits it first if it isn't resident, and holds the lease
until its whole reply is validated. It then releases the lease before
requesting the next bucket, so a request holds at most one lease at a time.
Results come back in input order. A sequence longer than 8192 tokens, or an
empty one, is refused before any shape is admitted or any request reaches the
worker. Nothing is truncated.

Inference runs on a detached task that owns the request's lease, its module
execution permit, its catalog-lane guard and its in-flight accounting. When
the caller goes away, the task still reads the worker's reply, so the stream
stays in sync and nothing restarts. Only then does it release those guards
and discard the result. The request's absolute deadline bounds admission waits
and each bucket's exchange. On expiry, the caller gets `deadline_exceeded` at
once, and no further buckets are started.

Real faults (I/O errors, malformed replies, an exchange that runs past its
bound) go through the supervisor's confirmed-exit recovery after the lease is
released. Recovery is tied to the worker generation that faulted, so one fault
restarts a worker once, and a stale error that arrives after recovery restarts
nothing. A request whose exchange broke may already have run on the worker, so
it returns an error rather than being replayed.

## Unload and shutdown

Retirement happens in this order:
1. It stops new leases and replacement spawning.
2. It drains in-flight work within the supervisor's bounded wait.
3. It closes the channel and waits for the worker process to exit, killing it
   if it has to.
4. Only after the exit is confirmed does it refund the worker's executable
   reservations, because a live process can still hold Neural Engine
   resources. An unconfirmed exit keeps the reservations charged.

Retirement is checked inside the same locked decision that grants leases and
reserves admissions, so a lease that raced it can't leave a reservation behind.
A restart that was already under way re-checks retirement after the old
process exits, and doesn't connect a replacement. Dropping the backend runs
this teardown on its own thread and bounds the caller's wait, as the generic
worker engine does.

## Certification observation

`ckdev-synapse-certify run` sets `certify_observation` in the config it generates
for the candidate. The supervisor then records every placement inventory the
workers report: executable IDs, which layers they cover, and which stages run
on the CPU. It keeps those records alongside its own count of admissions,
which it maintains separately. The `certify.observations` query returns both,
plus each worker channel's request count. The placement check refuses when the
inventories and the admission count disagree, so a dropped inventory can't
pass. With observation off, which is normal serving, nothing is recorded.

## Hardware certification

`ckdev-synapse-certify run --row ane-m5` drives a candidate build through this serving path
against the real Neural Engine. It needs a quiet machine (1-minute load under
16) and the original checkpoint at the revision and digests pinned in
`bench/parity/models.json`. Build from a clean, committed tree so the build
declares its commit, then run:

```sh
env -u TMPDIR DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  cargo build --release --locked -p synapse-module -p synapse-worker-ane-direct -p synapse-certify-runner

env -u TMPDIR DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  "$PWD/target/release/ckdev-synapse-certify" run \
  --row ane-m5 --model gte-modernbert-base \
  --assets "$PWD/target/release" --checkout "$PWD" \
  --weights "${GTE_MODERNBERT_WEIGHTS:?set to the original pinned checkpoint directory}"
```

For this development example, `--assets` is the `target/release` directory of the same clean build: the
record attests to the `ck-synapse` and `ck-synapse-worker-ane-direct` binaries
it finds there. Execution uses `ckdev-*` hard links in the run's scratch tree;
the record retains the original file names and hashes. `TMPDIR` must be unset, because the private ANE compiler only
accepts the per-user temporary directory.

For release certification, use extracted candidate assets without rebuilding them.
Build only the runner from the candidate's clean source commit; its source probe
must match. Keep assets and scratch on one filesystem so hard links succeed.
The runner logs its own path, commit and SHA-256 separately from the record.

Cold shape compiles dominate a first run: the 8192-token case spends most of
its 2-3 minutes compiling its shape, while warm requests take milliseconds. A
passing development run on an M5 Max (macOS 27.0.1) gave gte-modernbert-base a
minimum cosine of 0.99973 against the fp32 reference, across all 17 fixtures.

## Tests that guard these rules

All of these are in `crates/synapse-module/src/worker_host/mod.rs` and
`crates/synapse-module/src/lib.rs`, and run on mock transports:

| Rule | Test |
| --- | --- |
| A cancelled caller doesn't restart the worker | `serving_caller_cancellation_drains_reply_without_restart` |
| The detached task keeps the module permit and in-flight count | `cancelled_direct_ane_keeps_module_permit_and_inflight_until_reply_drains` |
| One fault restarts a worker once | `serving_admission_channel_fault_restarts_exactly_once` |
| A stale fault restarts nothing | `stale_serving_fault_after_recovery_does_not_restart_restored_owner` |
| Deadlines bound warm and cold requests | `direct_ane_warm_reply_after_absolute_deadline_is_discarded`, `direct_ane_cold_admission_after_absolute_deadline_is_drained_without_inference` |
| Residency is per owning worker | `identical_models_in_distinct_workers_each_admit_their_own_shape` |
| Retirement leaves no reservation behind | `retirement_between_precheck_and_grant_leaves_no_phantom_reservation` |
| Retirement stops a restart from replacing the worker | `retirement_during_exit_confirmation_never_connects_replacement` |
