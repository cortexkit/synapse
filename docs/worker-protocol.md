# Worker protocol and supervision

This contract describes implemented local worker transport and host supervision.
Message types live in `crates/synapse-core/src/worker_protocol.rs`; the module host
lives in `crates/synapse-module/src/worker_host/`. Owned decode also has its own
supervisor under `crates/synapse-engine-owned/owned-decode-worker/`.

## 1. Authenticate the spawned worker

The host creates a fresh worker nonce and verifies HELLO before accepting work.
The protocol version and expected engine identity must also agree. The daemon's
launch credentials are removed from child environments: authenticating a worker
must not delegate the module's launch authority.

Checks: `worker_host::tests::rejects_stale_worker_nonce` in the module library;
`wrong_engine_identity_and_old_protocol_are_rejected` in core Unix transport tests;
`stripped_child_environment_contains_neither_launch_nonce_variable` and
`stripped_tokio_child_environment_contains_neither_launch_nonce_variable` in
`crates/synapse-core/tests/child_process.rs`.

## 2. Preserve frame and request boundaries

Messages are length-prefixed JSON. Operations carrying token IDs or vectors also
carry the declared raw frame, using little-endian i32 or f32 values. Request IDs
correlate replies. A short raw frame, excess payload, or progress inserted between
a terminal header and its raw payload is a protocol fault, not a partial result.

Checks: `raw_i32_frames_round_trip`, `raw_f32_frames_round_trip` in core protocol
tests; `terminal_short_raw_then_eof_faults_real_host`,
`raw_payload_longer_than_terminal_header_faults_real_host`, and
`progress_between_terminal_header_and_raw_faults_real_host` in module framing tests.

## 3. Progress must represent completed work

Only validated, monotonically advancing completed-work messages renew an idle
window. A heartbeat or repeated counter cannot keep stalled inference alive.
Setup and inference have different silence budgets; neither is the request's
absolute total deadline. Progress phases and IDs must belong to the active
exchange. A fragmented frame crossing its idle bound is still a fault.

Checks: `validated_progress_renews_silence_window_not_total_budget`,
`stale_duplicate_zero_or_invalid_phase_progress_is_a_protocol_fault`,
`real_transport_progress_renews_multiple_idle_windows`, and
`fragmented_progress_crossing_idle_deadline_faults_real_host` in module host tests.

## 4. Drain a timed-out or abandoned exchange

A caller deadline can return an error before a live worker finishes. The detached
exchange keeps reading its reply before the stream can serve another request.
Ownership of execution permits and hardware leases lasts through that drain.
A genuine transport fault instead uses bounded recovery with generation fencing;
a stale fault must not restart a replacement worker.

Checks: `live_worker_deadline_drains_before_next_correct_response`,
`serving_caller_cancellation_drains_reply_without_restart`, and
`stale_serving_fault_after_recovery_does_not_restart_restored_owner` in the module.
The direct-ANE lifecycle is specified separately in [ANE serving](ane-serving.md).

## 5. Use the platform transport, not an alternate wire contract

Unix hosts use local sockets; Windows hosts use named pipes with the deterministic
`\\.\pipe\synapse-<digest>` name. The same framing and handshake semantics apply.
`pipe_name_matches_contract` proves naming; the Windows-only
`pipe_framing_round_trip` proves a named-pipe JSON/raw-frame exchange.

## Verify a protocol change

Run from the repository root:

```sh
cargo test --locked -p synapse-core --lib --test child_process
cargo build --locked --no-default-features -p synapse-worker-cuda -p synapse-worker-vulkan
cargo test --locked -p synapse-module --features test-support --lib --test it --test skeleton_e2e
```

Run the Windows transport tests on Windows as well; a Linux pass cannot exercise
named pipes. To reverse a protocol change, restore both peers together and rerun
their transport tests. Do not mix message revisions while an exchange is in flight.
