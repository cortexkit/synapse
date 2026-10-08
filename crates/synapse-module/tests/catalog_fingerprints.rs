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
            // Keep the existing Metal fingerprint calculation unchanged so adding
            // ANE entries does not change consumers' Metal model identities. The
            // test below recomputes each ANE catalog fingerprint from the numeric
            // settings in bench/parity/models.json and checks its recorded value.
            if backend["backend"] != "metal" {
                continue;
            }
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

#[test]
fn ane_catalog_fingerprints_bind_the_pinned_manifest_profiles() {
    let catalog: Value = serde_json::from_str(include_str!("../src/catalog/models.json")).unwrap();
    let manifest = synapse_parity::manifest::Manifest::from_slice(include_bytes!(
        "../../../bench/parity/models.json"
    ))
    .unwrap();
    let tokenizers: BTreeMap<String, String> =
        serde_json::from_str(include_str!("fixtures/catalog-tokenizer-digests.json")).unwrap();
    let mut computed_pins = Vec::new();
    for entry in catalog["models"].as_array().unwrap() {
        for backend in entry["backends"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|backend| backend["backend"] == "ane")
        {
            let id = backend["profile"].as_str().unwrap();
            let declared = &manifest.profiles[id];
            let model = manifest.model(&declared.model).unwrap();
            let profile: NumericProfile = serde_json::from_value(json!({
                "model_digest": model.checkpoint_digest,
                "quant": declared.storage_dtype,
                "engine": {"engine": "ane-direct-worker", "version": "protocol-v2", "build_flags": {
                    "profile": id, "compute_dtype": declared.compute_dtype, "storage_dtype": declared.storage_dtype,
                    "risk_class": "abort_capable"
                }},
                "sanitized_tokenizer_digest": tokenizers[&declared.model],
                "pooling": if backend["pooling"] == "last" || entry["task"] == "rerank" { "last_token" } else { "cls" },
                "normalization": model.output.normalization,
                "dtype": declared.storage_dtype, "flash_attention": "disabled",
                "certified_shape": {"max_context_tokens": 8192, "max_batch_tokens": 8192, "max_micro_batch_tokens": 3072, "max_sequences": 64},
                "thread_policy": "balanced", "operation": model.operation,
                "input_grammar": format!("synapse-input-grammar-v1:{}", manifest.grammar_digest(&declared.model).unwrap()),
                "kernel_revision": synapse_core::ANE_DIRECT_KERNEL_REVISION,
                "rotation": declared.rotation,
                "converted_package_digest": declared.converted_package_digest,
                "manifest_profile_digest": synapse_parity::canonical::sha256_hex(&synapse_parity::canonical::canonical_bytes(&manifest.profile_entry(id).unwrap()))
            })).unwrap();
            computed_pins.push((
                entry["id"].clone(),
                backend["fingerprint"].clone(),
                profile.fingerprint().0,
            ));
        }
    }
    assert_eq!(computed_pins.len(), 3);
    for (id, pinned, computed) in computed_pins {
        assert_eq!(pinned, computed, "ANE catalog {id}");
    }
}

#[test]
#[ignore = "requires original pinned Hugging Face snapshots; no accelerator inference"]
fn catalog_tokenizer_pins_match_original_snapshots() {
    let hub = std::path::PathBuf::from(
        std::env::var_os("SYNAPSE_PINNED_HF_CACHE").expect("SYNAPSE_PINNED_HF_CACHE"),
    );
    let catalog: Value = serde_json::from_str(include_str!("../src/catalog/models.json")).unwrap();
    let tokenizers: BTreeMap<String, String> =
        serde_json::from_str(include_str!("fixtures/catalog-tokenizer-digests.json")).unwrap();
    for entry in catalog["models"].as_array().unwrap() {
        let repository = entry["upstream"]["hf_repo"]
            .as_str()
            .unwrap()
            .replace('/', "--");
        let path = hub.join(format!(
            "models--{repository}/snapshots/{}/tokenizer.json",
            entry["upstream"]["revision"].as_str().unwrap()
        ));
        let tokenizer = synapse_core::SanitizedTokenizer::from_file(
            path,
            synapse_core::TokenizerConfig {
                max_tokens: usize::MAX,
            },
        )
        .unwrap();
        assert_eq!(
            format!("sha256:{}", tokenizer.sanitized_sha256()),
            tokenizers[entry["id"].as_str().unwrap()],
            "{}",
            entry["id"]
        );
    }
}
