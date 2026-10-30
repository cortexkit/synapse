# Worker build and test guidance

Work from the repository root and use `Cargo.lock` with `--locked`. Keep changes
package-scoped while iterating; run only one heavy native build at a time.
Rust 1.89 or newer is required. Apple Metal crates need the full Xcode toolchain;
Command Line Tools alone do not supply `metal`/`metallib`. Hardware tests need their
actual platform, pinned model files and any explicit opt-in described by the test.
A portable test pass or a skipped hardware case is not hardware certification.

## What to run for a change

While iterating, run only the filtered tests for the code you touched,
plus `cargo fmt --all --check` and Clippy for the touched crates, on the build
server (`runon: "linux,4c"`). For a bug fix, add one ordinary test that fails
without the fix, and show that once: revert only the fix, run that test, see it
fail. Do not write mutation rows, break proofs or deliberate-break controls
(edits that break production code on purpose to prove some test notices), and
do not repeat runs to prove stability. Before delivering, run `scripts/gate.sh`:
it is the single push-CI parity gate, with fast guards first and full Linux and
native Windows suites on the build servers in parallel. CI confirms this result;
do not wait for CI to discover failures. A filtered iteration check or
`scripts/gate.sh --fast` is not a full delivery gate.

## Standing pre-approvals

Routine verification repairs do not need a question or permission pause:

- Refresh path-dependent standalone lockfiles whenever their dependencies move.
  `scripts/check-path-deps.py` discovers standalone workspaces as well as the
  root; use its diagnostics to identify the affected lockfiles, regenerate only
  those files on a build server, and retain `--locked` in verification.
- Regenerate a pinned golden when the output differs only by the stated cause.
  Explain that cause and inspect the diff; unrelated output changes still need
  investigation rather than a blanket re-pin.
- Feature-list corrections and test-only seam fixes are pre-approved. Keep them
  scoped and verify the affected feature/test configuration.
- Submit build/test jobs to the build servers without asking. Never compile on
  the Mac except macOS-only Metal/ANE type-checks, with
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`.

## Landing execution location

Run portable Cargo checks, Clippy, tests and Nextest on the Linux build server,
not on the Mac. With the agent shell use `runon: "linux,4c"`. From a shell use
`ck motor run --vcpus 4 --cwd "$PWD" -- COMMAND...`; it snapshots the current
checkout, including dirty source. Native Windows checks use the Windows build
server, not Linux GNU cross-compilation. Only macOS-only Metal/ANE type-checks
belong on the Mac, with the full Xcode `DEVELOPER_DIR` above. Hardware execution
still needs the hardware owner's arrangement and explicit opt-ins below.

`scripts/gate.sh` reads executable push steps from the CI workflows. Phase 1
runs the guard scripts, workflow/release/doc-contract self-tests and formatting
on a small Linux job. Phase 2 runs full Linux Clippy/Nextest, native MSVC
Clippy/Nextest/llama, and Linux llama/parity jobs concurrently. The Linux suite
gets 12 cores and llama/parity gets 4, staying within the 16-core concurrent Linux
budget; the 4-core quick phase completes before either starts. The suite covers
the union of CI's four nextest shards. Windows jobs omit `--net` and run Cargo
locked/offline against the runner's prefetched lockfile dependencies. Every job prints its
failing step and duration; the gate prints a phase timing table and preserves
logs/build evidence under `.cortexkit/internal-docs/investigations/gate/`.
Independent detached Git worktrees live under
`~/Work/Projects/CortexKit-scratch/synapse-gate/<run-id>/<job>/`, outside the
checkout. The macOS MOTOR build runner requires workspace paths under `/Users`; system
temporary directories are not admissible. The gate refuses invalid or nested
destinations before creating worktrees or submitting jobs. Each worktree contains
the submitted tracked and non-ignored untracked source inputs. Its source-content
fingerprint must match the submitting tree and remain unchanged during checks.
Worktrees are removed after all jobs join, including failure and interruption,
followed by `git worktree prune`.
A refused, cancelled, disconnected or timed-out job is never a pass or silently
resubmitted. Runner prerequisites are stable Rust with Clippy/rustfmt, Nextest,
and native MSVC/LLVM on Windows; Phase 1 bootstraps hash-pinned pure-Python
PyYAML under the ignored target directory without modifying system Python.
The Linux suite jobs provision hash-pinned CMake 3.31.6 under their temporary
directory because the shared MOTOR image does not supply CMake. Runner Rust
toolchains are immutable: the gate probes their versions/components instead of
trying to update shared rustup state. Keep the provisioned release synchronized
with CI's moving stable toolchain.
If locked metadata cannot resolve from the immutable offline Cargo cache, the
job switches to a writable temporary `CARGO_HOME` with network fetches enabled;
warm cached jobs keep their existing paths and outputs.

`scripts/train-push.local.sh` invokes this gate before every train without
changing the shared train driver. For an explicitly chosen small doc-only
landing, `TRAIN_GATE_FAST=1` uses the previous preflight (`gate.sh --fast`), not
full parity. The older `land-gates.sh` remains a changed-target iteration helper;
its `--full` means module coverage, not push-CI parity. Legacy helpers retain
their Apple-only coverage but are not permission to compile portable code on
the Mac. Their MOTOR exit-200 retry budget remains 3 minutes; the full gate
fails immediately on refusal so no unexecuted suite is presented as verified.

Preflight orchestration tests (Python with PyYAML installed):

```sh
python3 -m unittest scripts.test_ci_parallel scripts.test_remote_gate scripts.test_land_gates scripts.test_gate -v
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
what you touched, followed by `scripts/gate.sh` before delivery. Use
[development verification](../docs/development.md) and
[backend certification](../docs/backend-certification.md) for exact procedures.
