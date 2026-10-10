# Owned Metal real-weight parity

This standalone test crate keeps evaluator-only dependencies out of the
production engine and the root workspace lockfile. Run from the repository root:

```sh
python3 crates/synapse-engine-owned/tests/hardware/prepare_assets.py
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export CARGO_TARGET_DIR="$PWD/target"
cargo test --manifest-path crates/synapse-engine-owned/tests/hardware/Cargo.toml \
  catalog_real_weights_pass_committed_evaluator -- --ignored --nocapture
```

The preparation script copies the manifest-pinned snapshots from the local
Hugging Face cache with symlinks dereferenced and aborts on any file digest
mismatch. The test checks all four snapshots again before loading any weights.
It loads each model with `<slug>.owned-metal` and the manifest operation, executes
all committed reference sequences (including the 8192-token case and both
reranker pools), and uses `synapse_parity::evaluator::evaluate` for every gate.
Set `METAL_PARITY_MODEL` to one slug to rerun a single model. JSON evaluation
reports are printed verbatim and saved under the ignored `results/` directory.
The report fingerprint is labeled `hardware-check:<manifest profile digest>`;
this tests engine math, not module fingerprint construction or certification.

To compare production preload logits, run the ignored
`preload_gte_raw_logits_match_baseline` test in a release build:

```sh
cargo test --release --manifest-path crates/synapse-engine-owned/tests/hardware/Cargo.toml \
  preload_gte_raw_logits_match_baseline -- --ignored --nocapture
```

It scores all 125 committed reference pairs and compares their little-endian
f32 bytes with the committed `evidence/preload-baseline.f32le`, not rounded JSON
numbers. The current result is saved under `results/` and its common SHA-256 is
printed. The test refuses to run in a debug build: production binaries are
optimized, the comparison is exact, and optimized and unoptimized builds
legitimately produce different logits.

To recapture the baseline, run the same release command with
`METAL_PRELOAD_LABEL=baseline`. This writes `results/preload-baseline.f32le`
without overwriting the committed baseline; copy it over
`evidence/preload-baseline.f32le` deliberately and record its provenance in
`evidence/README.md`. Both runs must use the same checked weights, toolchain,
fixture and cache setup. See `evidence/README.md` for the recorded machine,
source revision, build profile and verbatim evaluator reports.
