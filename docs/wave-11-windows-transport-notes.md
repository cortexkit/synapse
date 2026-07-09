# Lane 1 wave 11 — Windows transport notes

## Windows-only tests (run when CI has a Windows lane)

| Location | Test | What it covers |
|----------|------|----------------|
| `crates/synapse-core/src/worker_transport/mod.rs` | `pipe_name_matches_contract` | `\\.\pipe\synapse-<digest>` naming |
| `crates/synapse-core/src/worker_transport/windows.rs` | `pipe_framing_round_trip` | Named-pipe HELLO + length-prefixed JSON + raw frame loopback |

## macOS / portable tests (no regression)

- Full workspace `cargo nextest run` on macOS.
- `crates/synapse-worker-llama/tests/worker_host.rs` — embed/rerank/generate round-trips and `host_classifies_crashes_and_quarantines_after_budget` (crash budget / quarantine; uses `Child::kill` + `wait`, portable to Windows).

## Ally (ROG Ally) validation after merge

1. Module boot with ORT: set `ORT_DYLIB_PATH` to official `onnxruntime.dll` (>= 1.23) from [onnxruntime releases](https://github.com/microsoft/onnxruntime/releases).
2. `synapse-worker-llama` spawn via module worker host (`--pipe`, nonce handshake).
3. MiniLM (or cached GGUF) embed round-trip through worker host.
4. Crash/quarantine: worker with `--test-abort-on-request`, confirm quarantine after crash budget.

## Cross-compile check

From macOS (if `rustup target add x86_64-pc-windows-msvc`):

`cargo check --target x86_64-pc-windows-msvc -p synapse-module -p synapse-worker-llama -p synapse-engine-ort`

MLX/ANE worker crates are macOS-only; they are excluded from non-mac Windows workspace builds via `cfg(target_os = "macos")` entry points.