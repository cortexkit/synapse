# Worker build and test guidance

Work from the repository root and use `Cargo.lock` with `--locked`. Keep changes
package-scoped while iterating; run only one heavy native build at a time.
Rust 1.89 or newer is required. Apple Metal crates need the full Xcode toolchain;
Command Line Tools alone do not supply `metal`/`metallib`. Hardware tests need their
actual platform, pinned model files and any explicit opt-in described by the test.
A portable test pass or a skipped hardware case is not hardware certification.

## Portable module gates

```sh
cargo --version
rustc --version
cargo build --locked --no-default-features -p synapse-worker-cuda -p synapse-worker-vulkan
cargo check --locked -p synapse-module --features test-support
cargo test --locked -p synapse-module --features test-support --lib --test it --test skeleton_e2e --test wire_contract
cargo fmt --all --check
bash scripts/public-banlist-check.sh
bash scripts/check-cortexkit-layout.sh
ckdev-mutate check
```

Feature-off GPU worker builds are required refusal fixtures for the module
library tests; they do not exercise GPU hardware. Use the `ckdev-mutate` revision pinned in `.github/workflows/tests.yml`, with
cargo-nextest and the workflow's Python dependencies. The catalogue is the
`mutations.d/` fragment directory; do not add a legacy `mutations.toml` shim.
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

Doc changes can affect include_str!, mutation anchors and release inputs.
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

Before committing, inspect the diff, run the relevant package tests and formatting,
and confirm that deliberate mutation controls have been restored. Use
[development verification](../docs/development.md) and
[backend certification](../docs/backend-certification.md) for exact procedures.
