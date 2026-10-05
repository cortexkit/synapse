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
`preload_gte_raw_logits_match_baseline` test against the pre-slice source with
`METAL_PRELOAD_LABEL=baseline`, then against the current source without that
variable. The test scores all 125 committed reference pairs and compares their
little-endian f32 bytes, not rounded JSON numbers. Baseline and current result
files remain under `results/`; the current run prints the common SHA-256.
Both runs must use the same checked weights, toolchain, fixture and cache setup.
