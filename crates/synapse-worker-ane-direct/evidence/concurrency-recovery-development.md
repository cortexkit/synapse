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
