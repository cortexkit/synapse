# Direct ANE serving ownership

Direct ANE catalog loads use a dedicated backend rather than `WorkerEngine` inference. The module lazily acquires one direct-ANE lane lock, excluding other module processes from this hardware lane, and one shared residency supervisor, budgeting the compiled layer executables across workers. Every loaded direct-ANE model has its own worker channel and I/O runtime, but all those workers share the supervisor's hardware budget.

The production connector uses `WorkerHost` for spawn, log forwarding, removal of the module's daemon-authentication nonce from the child environment, and HELLO validation. Workers inherit the lane lock. Successful LOAD requests are cached by `AneWorkerChannel`, which restores them after confirmed-exit recovery. LOAD and PONG advertise the worker's bucket ladder; serving consumes LOAD metadata rather than choosing its own ladder.

Embedding and fully composed rerank sequences run in ascending buckets. Each inference task holds its shape lease until the entire response has been validated, then releases it before requesting another bucket. Results are restored to input order. An oversized sequence is rejected before admission or inference transport; there is no truncation or fallback.

Caller cancellation abandons the result, not the detached inference task. The task drains the response and releases its lease without restarting the worker. Actual I/O faults, malformed responses, and bounded inference timeouts invoke the supervisor's existing confirmed-exit recovery after releasing the lease. A broken exchange may already have executed inference even though its response is missing. Such requests return an error rather than being automatically replayed.

Unload prevents replacement spawning and new leases, drains existing work with the supervisor's bounded wait, closes the channel, and waits for the owned child to exit (terminating it if necessary). Budget reservations for compiled layer executables are refunded only after confirmed exit, because a still-live child can retain ANE resources. Unconfirmed exit retains reservations. Drop performs cleanup on a dedicated thread and bounds the caller's wait, as the generic worker engine does. Worker IDs are unique per load so a retired owner's identity is not reused.

With `certify_observation` enabled before admission, `certify.observations` combines direct-channel request counters with the shared supervisor's retained inventories (executable IDs, layer coverage, and CPU stages) and its admission-event count, maintained separately from both the inventory list and transport counters. Normal serving retains no inventories.

## Hardware certification remains gated

The existing `ane-m5` live-certification refusal has intentionally **not** been removed. No hardware evidence is claimed for this integration: observed one-minute system load averages from `uptime` (a dimensionless measure of runnable work) during verification were 30.62, 21.88, and 29.94, all above the permitted threshold of 16. Mock transport tests prove admission, response ordering, cancellation draining, actual-fault recovery, and confirmed-exit retirement, not hardware parity or placement.

The hardware follow-up must first enable the `ane-m5` live row for the trial and commit that source change, then build from a clean tree. Remove the refusal permanently only after that row reaches its actual pass/fail gates. Once the one-minute load is below 16 and a GTE checkpoint directory matching the revision and file digests in `bench/parity/models.json` is available, use the following exact acceptance commands. Set `GTE_MODERNBERT_WEIGHTS` to that directory, containing the original checkpoint files rather than a converted worker package:


```sh
env -u TMPDIR DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  cargo build --release --locked -p synapse-module -p synapse-worker-ane-direct

env -u TMPDIR DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  target/release/ck-synapse certify run \
  --row ane-m5 --model gte-modernbert-base \
  --assets "$PWD/target/release" --checkout "$PWD" \
  --weights "${GTE_MODERNBERT_WEIGHTS:?set to the original pinned checkpoint directory}"
```

Use the explicit candidate asset directory; do not substitute another checkout's binary. Recheck `uptime` before starting the hardware run. Compilation of ANE models requires `TMPDIR` unset and the Xcode developer directory shown above.
