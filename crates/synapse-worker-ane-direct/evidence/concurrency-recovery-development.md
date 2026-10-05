# Residency concurrency regressions (development)

Snapshot under audit: `0293b3ba10e995f761e5cfb9225eb374e84a9d78`. Deterministic mock-transport tests, not hardware certification. Toolchain Rust/cargo1.99.0.

## Failed eviction during resource retry

Before the fix:

```text
failed_resource_victim_eviction_recovers_without_waiting_on_its_own_admission ... FAILED
recovery must not wait for the triggering admission's own slot: Elapsed(())
0 passed; 1 failed; finished in0.21s
```

The test uses a three-second recovery deadline and requires a refusal within200ms, a confirmed-exit restart, empty reservations/recovery state, and a successful later lease. On a failed retry eviction, mark the already-finished triggering admission as Failed without refunding it before running recovery. Failed eviction reservations remain charged until confirmed child exit. Ordinary eviction still owns its detached recovery path. Cross-worker eviction does not incorrectly mark the requester's reservation Failed.

After fix: the self-deadlock test passes;23 ordinary residency/schema/timing tests passed (three hardware tests ignored).

## Drain deadline escalation

Before the fix, reaching the drain deadline left the worker closed with no recovery path:

```text
drain_timeout_forces_confirmed_exit_and_reopens_worker ... FAILED
bounded drain timeout must escalate to confirmed owner exit, not permanently close a live worker
0 passed;1 failed; finished in0.03s
```

After fix:25 ordinary tests pass (three hardware ignored), including the drain-timeout regression and `unconfirmed_exit_keeps_worker_closed_and_budget_charged`. The leader escalates at its drain deadline; channel restart faults/notifies active I/O before taking the stream lock, closes the connection and awaits the owning-session exit confirmation. The real process exit callback already escalates to killing its owned child after a ten-second graceful wait and confirms the exit status. Only unconfirmed ownership or failed replacement/restoration remains permanently closed, with reservations charged; a mere drain deadline no longer does. Existing lease-draining and exit-order tests remain green.

## Old-owner eviction queued behind restart

Before owning-process generation checks, queued work could affect a restarted server:

```text
queued_eviction_from_old_owner_is_refused_after_restart ... FAILED
an eviction queued for the retired owner must not reach its replacement
0 passed;1 failed; finished in0.01s
```

The test deterministically queues restart first, eviction second behind the same connection mutex, then releases the mutex. After restart, the old eviction must be refused and no eviction command may arrive in the new server. An additional assertion retains a newly admitted sequence-length128 shape while holding its lease and refuses an explicitly old-generation eviction without removing it.

A monotonic owning-process generation is captured in each reservation and supplied to admit/retry/evict operations. The channel checks it while holding the stream mutex before writing. Restart advances it under that same mutex. Detached work whose old reservation has already retired cannot evict or admit on the replacement; stale refusals do not restart the new owner. Direct `exchange` callers (LOAD metadata and inference, as well as shape commands) also capture/check their generation across connection-lock waits. No worker wire protocol changed.

## Cancellation after request write

Before the guard, dropping a written request left its unread response reusable by the next request:

```text
cancelled_exchange_faults_channel_before_a_later_request_reads_stale_reply ... FAILED
cancelled exchange must fault the stream instead of returning the abandoned eviction response to Ping
0 passed;1 failed; finished in0.12s
```

The mock acknowledges receiving the eviction request, withholds its response, and the client task is aborted. Releasing the mock response must not let a subsequent Ping read the abandoned eviction response. An RAII guard faults any exchange dropped before both JSON and optional raw response frames complete, before releasing the connection mutex. Confirmed-exit restart permits a fresh Ping afterward.

Caller audit of snapshot `0293b3ba`: production `AneWorkerChannel::exchange` calls are the channel's `admit_shape` and `evict_shape` trait methods. The direct embed/rerank exchange in `infer` belongs to the macOS-only hardware-test module; `run_by_rung` groups requests by admitted sequence-length bucket and accepts caller-supplied inference futures, which can be dropped, and the same guard covers those direct exchanges. Regular `WorkerHost` embed/rerank uses separate `send_request`, not this channel, with request-timeout wrappers. Frame `read_json`/`read_raw` in `synapse-core` Unix/Windows transports and framing `read_exact` have no intrinsic timeout; only handshake and ordinary host request wrappers supply one.

## Unacknowledged shape commands

Before adding shape exchange deadlines, the admission and eviction no-reply tests failed while the worker held its socket open and read further requests without acknowledging the shape command:

```text
unacknowledged_admission_times_out_without_refunding_before_exit ... FAILED
unacknowledged_eviction_times_out_without_refunding_before_exit ... FAILED
unacknowledged shape RPC must time out and request owner exit confirmation: Elapsed(())
0 passed;2 failed; finished in0.41s
```

After fix: 30 ordinary residency/schema/timing tests pass; three hardware tests remain ignored. The tests use 20 ms exchange deadlines and hold the mock exit-confirmation future on a notification until explicitly released: four executables remain charged, the channel is faulted, and no replacement connection exists before confirmation. After confirmed exit, failed admission refuses while recovered eviction can transparently resume the waiting admission. Later requests succeed in both cases. A separate test bounds an exit-confirmation future that never completes and proves reservations remain charged and no replacement is spawned.

Production admit/evict exchanges have a 600-second deadline covering connection-lock wait, writes, JSON response, and any raw response. The largest measured debug shape compile was 115.495 s (`capacity-development-scope-large.json`); the 100-program budget allows at most four simultaneous shape reservations, each at least 22 programs. Four such measured compiles total about 462 s; 600 s adds about 138 s of headroom while remaining finite under a silent worker. Expiry faults/interrupts the owner and follows confirmed-exit recovery; it never refunds a reservation by itself. Exit confirmation is bounded at 30 s (the owned real child already gets a 10 s graceful wait before kill/exit confirmation). Replacement connection and pinned-model replay are bounded too; failed confirmation or restoration stays closed. Interrupted compile timing is recorded on drop; a refused request that never acquires the stream records zero dispatched compile time.

The mock-transport regression tests verify deadlock recovery, drain escalation, stale-operation rejection, cancellation, and no-reply timeouts; they do not certify hardware behavior. No additional hardware measurement was run. No process not started by this worker was stopped.

Final corrected regression output (same test names as the failing runs):

```text
failed_resource_victim_eviction_recovers_without_waiting_on_its_own_admission ... ok
drain_timeout_forces_confirmed_exit_and_reopens_worker ... ok
queued_eviction_from_old_owner_is_refused_after_restart ... ok
cancelled_exchange_faults_channel_before_a_later_request_reads_stale_reply ... ok
unacknowledged_admission_times_out_without_refunding_before_exit ... ok
unacknowledged_eviction_times_out_without_refunding_before_exit ... ok
test result: ok. 30 passed;0 failed;3 ignored
```

## Final gates

- `cargo clippy --locked -p synapse-module -p synapse-worker-ane-direct --all-targets -- -D warnings`: passed, clippy 0.1.99. This checks both packages and their test targets.
- `cargo fmt --all -- --check`: passed. Fixed three vendored import-order differences and refreshed only their file digests.
- `cargo test --locked -p synapse-module --lib -- --test-threads=1`: 577 passed, 9 ignored, zero failures.
- `cargo test --locked -p synapse-worker-ane-direct`: 11 unit tests passed, 12 hardware tests ignored; vendored integrity test passed.
- The test-only mutex guard now ends in a lexical scope before the later lease await. The compile-concurrency probe iterates the same shape values without indexing a fixed length array by a range variable.

Commands ran with `TMPDIR` unset and `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`. Dependency `block` reports a future-incompatibility notice, not a current warnings-as-errors failure. Scoped AFT diagnostics were unavailable because language servers disconnected during initialize; compiler-backed clippy and the full suites are the verification authority.
