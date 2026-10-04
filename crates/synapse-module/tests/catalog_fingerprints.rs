use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use synapse_core::fingerprint::NumericProfile;
use synapse_engine_owned::{engine_identity, ModelFamily, OwnedDType};

#[test]
fn every_catalog_backend_fingerprint_matches_its_declared_numeric_profile() {
    let catalog: Value = serde_json::from_str(include_str!("../src/catalog/models.json")).unwrap();
    // These are canonical sanitizer outputs for the pinned release tokenizers, not expected profile IDs.
    // Keeping just the digests makes the identity guard independent of external model snapshots.
    let tokenizers: BTreeMap<String, String> =
        serde_json::from_str(include_str!("fixtures/catalog-tokenizer-digests.json")).unwrap();
    let mut checked = 0;
    for entry in catalog["models"].as_array().unwrap() {
        for backend in entry["backends"].as_array().unwrap() {
            let roles: BTreeMap<String, String> = entry["files"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|file| {
                    file["backends"]
                        .as_array()
                        .unwrap()
                        .contains(&backend["backend"])
                })
                .map(|file| {
                    (
                        file["role"].as_str().unwrap().to_string(),
                        format!("sha256:{}", file["sha256"].as_str().unwrap()),
                    )
                })
                .collect();
            let roles = roles.into_iter().collect::<Vec<_>>();
            let artifact = format!(
                "sha256:{}",
                hex::encode(Sha256::digest(serde_json::to_vec(&roles).unwrap()))
            );
            let pooling = backend["pooling"].as_str().unwrap_or("cls");
            let profile: NumericProfile = serde_json::from_value(json!({
                "model_digest":artifact, "quant":backend["dtype"],
                "engine":engine_identity(ModelFamily::parse(backend["family"].as_str().unwrap()).unwrap(), OwnedDType::parse(backend["dtype"].as_str().unwrap()).unwrap()),
                "sanitized_tokenizer_digest":tokenizers[entry["id"].as_str().unwrap()],
                "pooling":if pooling == "last" { "last_token" } else { pooling },
                "normalization":if backend["normalize"] == true {"l2"} else {"none"},
                "dtype":backend["dtype"], "flash_attention":"disabled",
                "certified_shape":{"max_context_tokens":backend["max_tokens"],"max_batch_tokens":8192,"max_micro_batch_tokens":3072,"max_sequences":64},
                "prompt_template":if entry["task"] == "rerank" {Some("synapse-rerank-bos-query-sep-doc-eos-v1")} else {None}, "thread_policy":"balanced"
            })).unwrap();
            let computed = profile.fingerprint().0;
            assert_eq!(
                backend["fingerprint"], computed,
                "catalog backend {} {}",
                entry["id"], backend["backend"]
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 3);
}
