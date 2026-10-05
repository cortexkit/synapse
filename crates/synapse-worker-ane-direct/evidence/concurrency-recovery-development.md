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
