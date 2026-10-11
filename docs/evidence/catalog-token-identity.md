# Catalog token identity against master

Reference: `027e1d30b11feac82dd4ac4a343c2de9e023bad3`.

`crates/synapse-module/tests/fixtures/catalog-master-token-ids.json` contains
literal token IDs captured from that reference, not expectations calculated by
the candidate's preflight. The same module harness was added to both source
snapshots. It dispatches public `embed.query`, `embed.batch`, and `rerank.score`
requests with the full checksum-pinned upstream tokenizers.

The capture uses native `EmbedBackend::Owned` for Metal and
`EmbedBackend::DirectAne` for ANE. Metal IDs are observed by a test model family
after the real engine's terminal-token validation and bucket planning. ANE IDs
are decoded from the actual inference IPC raw frames received by the mock
worker. Both return synthetic outputs; no model weights, neural-network
kernels, Neural Engine programs, or production store are needed.

All six catalog lanes served by this macOS build are explicit in the fixture:

| Lane | Short / longer composed token counts | Operations |
| --- | --- | --- |
| `gte-modernbert-base-metal` | 12 / 502 | query, batch |
| `gte-modernbert-base-ane` | 12 / 502 | query, batch |
| `gte-reranker-modernbert-base-metal` | 19 / 509 | rerank |
| `qwen3-embedding-0.6b-metal` | 11 / 501 | query, batch |
| `qwen3-embedding-0.6b-ane` | 11 / 501 | query, batch |
| `qwen3-reranker-0.6b-ane` | 87 / 577 | rerank |

The candidate matched all 20 sequences, including every individual integer.
CUDA and Vulkan catalog lanes are platform-refused on this macOS build; the
catalog does not declare a Metal Qwen reranker.

## Re-run the candidate comparison

```sh
python3 scripts/fetch-token-identity-assets.py
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
SYNAPSE_TOKEN_IDENTITY_ASSETS="$PWD/target/token-identity-assets" \
  cargo test --locked -p synapse-module --lib \
  catalog_token_identity_tests::catalog_boundary_token_ids_match_master \
  -- --ignored --exact --nocapture
```

The test is explicitly ignored by the ordinary suite because it requires the
full upstream tokenizer/config files. Fetching verifies seven pinned inputs
(about 30 MB) and never downloads weights. The committed literal expectations
and capture patch are outside regenerable build directories.

## Reproduce the reference capture

Run this only to audit the reference; never regenerate expectations from a
candidate whose token policy is under review.

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
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
SYNAPSE_TOKEN_IDENTITY_ASSETS="$PWD/target/token-identity-assets" \
SYNAPSE_TOKEN_IDENTITY_CAPTURE="$PWD/target/master-token-ids-audit.json" \
CARGO_TARGET_DIR="$PWD/target/token-identity-master-build" \
  cargo test --manifest-path target/token-identity-master/Cargo.toml \
  --locked -p synapse-module --lib \
  catalog_token_identity_tests::capture_master_boundary_token_ids \
  -- --ignored --exact --nocapture
```

The retained instrumentation patch changes only test-module declarations and
mock IPC observation. The native capture model is enabled only by
`test-support` on macOS. It does not replace either snapshot's token preparation
or engine bucket dispatch.

## Why Qwen Metal did not gain an EOS token

On the reference, native Metal has no `profile` build flag, so
`compose_catalog_embed` is a no-op. `apply_owned_tokenizer_policy` nevertheless
appends the engine-declared terminal token, `151643`, for Qwen embedding.
`OwnedMetalEmbedEngine::run_bucketed` does **not** append it: it refuses an input
whose last ID is not the required terminal token. The candidate's preflight
adds the same token earlier; the existing policy removes and restores the
already-present terminal rather than duplicating it. Both Qwen embedding lanes
therefore end in exactly one `151643` in the recorded controls.

Native Metal rerank on the reference already used module-composed pairs through
`owned_rerank_pairs`, because its backend is `Owned`, even without a profile.
The candidate uses the extracted tokenizer routine for those same pairs; the
GTE Metal rerank controls pin the actual CLS/SEP framing as well as the text IDs.
