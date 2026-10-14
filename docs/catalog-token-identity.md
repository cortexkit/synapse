# Catalog token identity

## 1. Compare against independently captured input IDs

The literal expectations in
`crates/synapse-module/tests/fixtures/catalog-master-token-ids.json` were captured
from source `027e1d30b11feac82dd4ac4a343c2de9e023bad3`, not calculated by the candidate
under test. The harness observes engine-bound IDs after native token-policy and
bucket planning, or the actual raw inference frame at the mock worker boundary.
Synthetic outputs avoid any model-weight or accelerator prerequisite.

Check: `catalog_token_identity_tests::catalog_boundary_token_ids_match_master`
in the module library (macOS, explicitly ignored because it needs upstream
checksum-pinned tokenizer/config files).

## 2. Fetch only pinned prerequisites and verify the candidate

From the repository root:

```sh
python3 scripts/fetch-token-identity-assets.py
SYNAPSE_TOKEN_IDENTITY_ASSETS="$PWD/target/token-identity-assets" \
  cargo test --locked -p synapse-module --lib \
  catalog_token_identity_tests::catalog_boundary_token_ids_match_master \
  -- --ignored --exact --nocapture
```

The fetcher verifies tokenizer/config digests and does not download model weights.
Run on the test's supported platform with the appropriate developer toolchain;
a platform skip is not a token-identity proof.

## 3. Audit the original capture without changing expectations

Only use the reference source for recapture, never an unqualified candidate:

```sh
mkdir -p target/token-identity-master
git archive 027e1d30b11feac82dd4ac4a343c2de9e023bad3 \
  | tar -x -C target/token-identity-master
git apply --directory=target/token-identity-master \
  results/catalog-token-identity-instrumentation.patch
cp crates/synapse-module/src/catalog_token_identity_tests.rs \
  target/token-identity-master/crates/synapse-module/src/
cp crates/synapse-engine-owned/src/token_capture_test_model.rs \
  target/token-identity-master/crates/synapse-engine-owned/src/
cp crates/synapse-module/tests/fixtures/catalog-master-token-ids.json \
  target/token-identity-master/crates/synapse-module/tests/fixtures/
SYNAPSE_TOKEN_IDENTITY_ASSETS="$PWD/target/token-identity-assets" \
SYNAPSE_TOKEN_IDENTITY_CAPTURE="$PWD/target/master-token-ids-audit.json" \
CARGO_TARGET_DIR="$PWD/target/token-identity-master-build" \
  cargo test --manifest-path target/token-identity-master/Cargo.toml \
  --locked -p synapse-module --lib \
  catalog_token_identity_tests::capture_master_boundary_token_ids \
  -- --ignored --exact --nocapture
```

The retained instrumentation patch changes test declarations and boundary
observation, not either source's tokenizer or engine batching policy. The capture
model is enabled only by test support on macOS. Verify exact ordered integer
arrays, not just token counts or expected values recomputed by the candidate.

To reverse this audit, remove only the scratch checkout/output created for it;
leave the tracked literal fixture and capture patch unchanged. A deliberate token
policy change needs reviewed independent expectations and identity handling.
