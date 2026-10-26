# Worker build and test guidance

Work from the repository root and use `Cargo.lock` with `--locked`. Keep changes
package-scoped while iterating; run only one heavy native build at a time.
Rust 1.89 or newer is required. Apple Metal crates need the full Xcode toolchain;
Command Line Tools alone do not supply `metal`/`metallib`. Hardware tests need their
actual platform, pinned model files and any explicit opt-in described by the test.
A portable test pass or a skipped hardware case is not hardware certification.

## What to run for a change

Implement the change, then run only the filtered tests for the code you touched,
plus `cargo fmt --all --check` and Clippy for the touched crates, on the build
server (`runon: "linux,4c"`). For a bug fix, add one ordinary test that fails
without the fix, and show that once: revert only the fix, run that test, see it
fail. Do not write mutation rows, break proofs or deliberate-break controls
(edits that break production code on purpose to prove some test notices), and
do not run full suites or repeat runs to prove stability. The head runs the full
suite once per integrated batch, and CI runs it before master.

## Landing execution location

Run portable Cargo checks, Clippy, tests and Nextest on the Linux build server,
not on the Mac. With the agent shell use `runon: "linux,4c"`. From a shell use
`ck motor run --vcpus 4 --cwd "$PWD" -- COMMAND...`; it snapshots the current
checkout, including dirty source. Only release builds being placed here and
checks requiring Apple frameworks, Metal, Swift or actual macOS behavior belong
on the Mac.

`scripts/land-gates.sh` routes its portable plan to MOTOR; `--full` adds the
fixtures, module check/tests and Clippy. When invoked on macOS, narrow
tests for Apple-only packages and the module's active macOS-gated tests run locally;
on Linux those native implementations are not verified. On macOS,
`scripts/train-push.local.sh` joins remote Windows GNU
cross-Clippy, formatting and repository checks with native
Clippy for `synapse-module`, `synapse-engine-owned`, `synapse-worker-ane`, plus
the engine-owned library tests. Native MSVC remains a CI gate. Both halves print
source-content fingerprints and refuse a mismatched snapshot or a tree changed
during verification. Each step and the combined wall time are printed. Only
MOTOR exit 200 (refused before start) retries, every 15 seconds for up to 3 minutes;
connection loss, timeouts and check failures are never resubmitted.

Preflight orchestration tests (Python with PyYAML installed):

```sh
python3 -m unittest scripts.test_ci_parallel scripts.test_remote_gate scripts.test_land_gates -v
```

## Portable module gates

These are the module's gate commands; pick the ones that cover what you touched
and filter the tests to it.

```sh
cargo --version
rustc --version
cargo build --locked --no-default-features -p synapse-worker-cuda -p synapse-worker-vulkan
cargo check --locked -p synapse-module --features test-support
cargo test --locked -p synapse-module --features test-support,e2e --lib --test it --test skeleton_e2e --test wire_contract
cargo fmt --all --check
bash scripts/public-banlist-check.sh
bash scripts/check-cortexkit-layout.sh
```

Feature-off GPU worker builds are required refusal fixtures for the module
library tests; they do not exercise GPU hardware.
`cargo fmt --all --check` is silent on success; record its version and exit status.

## Documentation, release and helper gates

```sh
cargo test --locked -p synapse-certify --lib --test records
cargo test --locked -p synapse-core --test dev_binary_guard --test child_process
python3 -m unittest discover -s crates/synapse-release-checks -v
python3 scripts/check-release-candidate.py --self-test
python3 -m unittest discover -s bench/harness -p test_release_matrix.py -v
bash bench/campaign/ane-direct-embed-harness.sh --self-test .
bash scripts/train-push.test.sh
```

Doc changes can affect include_str! and release inputs.
`bench/campaign/ane-direct-embed-pack.jsonc` selects document excerpts by literal
path and line range; update those ranges when the referenced text moves.
Keep every live reader's required files tracked. Public docs are
flat topic guides/contracts; canonical certification JSON remains at the paths
consumed by validators. Internal originals and operational records are ignored.

## Native and evidence boundaries

On macOS, verify `xcrun --find metal` before a Metal build. For direct-ANE hardware
work, arrange exclusive Neural Engine access with the hardware owner, preserve
the cross-process lane lock and unset a custom `TMPDIR`. Do not stop unrelated
processes to obtain hardware. Release
certification executes extracted candidate assets through same-filesystem hard
links under development names; never rebuild or copy the attested binaries.

Before committing, inspect the diff and run the filtered tests and formatting for
what you touched. Use
[development verification](../docs/development.md) and
[backend certification](../docs/backend-certification.md) for exact procedures.
