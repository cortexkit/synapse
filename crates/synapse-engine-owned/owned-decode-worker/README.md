# owned-decode-worker

Module-side supervisor for the `owned-metal-decode-worker-v1` protocol — the
worker-supervision layer of the production decode port (see the checked-in
production decode port specification).

## What this is

A pure-Rust state machine that supervises the owned Metal decode worker:

- **Start validation** with dedicated, non-overlapping mismatch mappings
  (resolution r2 #7): protocol/frame-structure → `owned_decode_protocol_mismatch`,
  loaded-model / decode-fingerprint / runtime-manifest identity →
  `owned_decode_runtime_config_mismatch`, constraint identity →
  `owned_decode_constraint_version_mismatch`, non-greedy sampling →
  `owned_decode_sampling_unsupported`.
- **One-generation residency** and **progress/continuation framing** with
  sequence and session validation. Frames from closed or superseded sessions are
  rejected; repeated or skipped sequences are protocol-fatal.
- **Terminal-control boundary** precedence: a natural terminal completion
  (`stop_token`, `max_tokens`, `grammar_complete`) wins outright; otherwise
  cancellation is evaluated before the deadline, so cancellation wins when
  both are pending at the same boundary (resolution r2 #4 reconciled with
  `error_contract`).
- **Crash-budget persistence and quarantine.** Crashes, protocol-fatal responses,
  startup failures, timeouts, and failed cancellations each charge one unit;
  acknowledged cancellation and deadline cleanup before timeout charge nothing.
  A terminal exhausting charge quarantines the key. State persists through a
  `CrashBudgetStore` (in-memory for fixtures, JSON-file for production).
- **The single permitted worker-crash redispatch**: only a crash classification
  redispatches, exactly once, from the original prompt and initial constraint
  state on a fresh worker generation and session (token-zero restart,
  attempt-local sequence reset, `generation_id` preserved). Timeout,
  protocol-fatal, and startup failures are terminal after one charge.
- **Process supervision and fault injection** via the `DecodeWorker` /
  `WorkerFactory` traits and a `ScriptedWorkerFactory` test double.
- **`decode-ownership-manifest-v1`** fault sites plus an `OwnershipLedger` the
  fixtures use to prove no double free, invalid free, use-after-free, or leak at
  every fault site.

It does **not** link Metal and does **not** depend on `synapse-engine-owned`, so
its fixtures run on any host. The real Metal worker satisfies the `DecodeWorker`
transport trait defined here.

## Why a standalone workspace

This crate is nested inside the synapse workspace tree but is intentionally its
own workspace root: it is developed and tested independently and is not
registered as a member of the parent workspace. The empty `[workspace]` table in
`Cargo.toml` opts this package out of the parent workspace and makes it its own
workspace root. The parent workspace ignores it entirely (verified: `cargo
metadata` at the repo root does not list it, and root `cargo fmt --all --check`
is unaffected).

## Build and test

```sh
cd crates/synapse-engine-owned/owned-decode-worker
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test
```

All gates pass: 43 library unit tests, 30 protocol-behavior fixtures, and 12
ownership-safety fixtures (85 total).

## Fixture coverage

`tests/protocol_fixtures.rs` proves, without a GPU:

- dedicated mismatch mappings (runtime, every constraint field, protocol, sampling);
- timeout classification and crash-budget consequences (terminal, no redispatch);
- terminal-control precedence (completion > cancellation > deadline);
- literal cleanup-timeout and cancellation wire errors (`deadline_exceeded`,
  `cancelled`) — never the symbolic placeholders;
- repeated/skipped sequences, stale-session rejection, malformed frames;
- every successful finish reason and stop-control omission (asserted against
  the modeled greedy union selection, not pass-through);
- failed cancellation at the deadline and cancellation boundaries: kill
  escalation, one `failed_cancellation` strike, and budget exhaustion;
- multi-quantum progress/continuation with an exact sequence trace and
  remaining-budget truncation;
- the single crash redispatch: token-zero restart, sequence reset, delayed
  prior-session rejection, `generation_id` preservation, at most one redispatch,
  and redispatch barred by deadline / cancellation / exhausted budget.

`tests/ownership_fixtures.rs` proves the four ownership-safety properties at
every `decode-ownership-manifest-v1` fault site (allocation, ownership transfer,
partial initialization, generation, cancellation, timeout cleanup, unload,
shutdown, LFM2 conv-cache FFI). The mandatory `macos-metal` lane runs the real
worker under AddressSanitizer against the same fault-site IDs.

## Wire-error bindings

The literal deadline and cancellation IDs mirror the module-owned
`owned-decode-wire-error-bindings-v1` manifest (which lives under
`crates/synapse-module/`, outside this slice's fence). `deadline_exceeded` is the
existing stable wire literal; `cancelled` matches the external
`finish_reason=cancelled` normalization. `wire_error_bindings::assert_no_symbolic_placeholders`
guards that no symbolic placeholder ever appears as an emitted ID.
