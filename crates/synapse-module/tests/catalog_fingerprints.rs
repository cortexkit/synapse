use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use synapse_core::fingerprint::NumericProfile;
use synapse_engine_owned::{
    engine_identity, engine_identity_with_projections, ModelFamily, OwnedDType, ProjectionDtype,
};

/// Fingerprint of a catalog Metal lane, computed from its catalog parameters and
/// the given engine identity the same way the module computes it at load.
fn metal_lane_fingerprint(
    entry: &Value,
    backend: &Value,
    tokenizers: &BTreeMap<String, String>,
    engine: &synapse_core::EngineIdentity,
) -> String {
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
        "engine":engine,
        "sanitized_tokenizer_digest":tokenizers[entry["id"].as_str().unwrap()],
        "pooling":if pooling == "last" { "last_token" } else { pooling },
        "normalization":if backend["normalize"] == true {"l2"} else {"none"},
        "dtype":backend["dtype"], "flash_attention":"disabled",
        "certified_shape":{"max_context_tokens":backend["max_tokens"],"max_batch_tokens":8192,"max_micro_batch_tokens":3072,"max_sequences":64},
        "prompt_template":if entry["task"] == "rerank" {Some("synapse-rerank-bos-query-sep-doc-eos-v1")} else {None}, "thread_policy":"balanced"
    })).unwrap();
    profile.fingerprint().0
}

/// Engine identity of a catalog Metal lane: catalog lanes opt in to f16 weight
/// projections exactly when they are f16 gte-modernbert, as the module's
/// `catalog_projection_dtype` decides.
fn catalog_engine_identity(backend: &Value) -> synapse_core::EngineIdentity {
    let family = ModelFamily::parse(backend["family"].as_str().unwrap()).unwrap();
    let dtype = OwnedDType::parse(backend["dtype"].as_str().unwrap()).unwrap();
    let projections = if family == ModelFamily::GteModernBert && dtype == OwnedDType::F16 {
        ProjectionDtype::F16
    } else {
        ProjectionDtype::F32
    };
    engine_identity_with_projections(family, dtype, projections)
}

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
            let computed = metal_lane_fingerprint(
                entry,
                backend,
                &tokenizers,
                &catalog_engine_identity(backend),
            );
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
fn the_gte_metal_catalog_lane_moved_only_by_its_projection_opt_in() {
    // The gte-modernbert-base catalog Metal lane opts in to f16 weight
    // projections. Without the opt-in (graph revision 4, no projection_dtype
    // flag) the lane's fingerprint is b904dd7b...; recomputing it that way must
    // give exactly that value, which shows the opt-in is the only input that
    // differs between the two fingerprints.
    let catalog: Value = serde_json::from_str(include_str!("../src/catalog/models.json")).unwrap();
    let tokenizers: BTreeMap<String, String> =
        serde_json::from_str(include_str!("fixtures/catalog-tokenizer-digests.json")).unwrap();
    let entry = catalog["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["id"] == "gte-modernbert-base")
        .unwrap();
    let backend = entry["backends"]
        .as_array()
        .unwrap()
        .iter()
        .find(|backend| backend["backend"] == "metal")
        .unwrap();
    let opted_in = catalog_engine_identity(backend);
    assert_eq!(opted_in.build_flags["graph_revision"], "5");
    assert_eq!(opted_in.build_flags["projection_dtype"], "f16");
    let without = engine_identity(ModelFamily::GteModernBert, OwnedDType::F16);
    assert_eq!(
        metal_lane_fingerprint(entry, backend, &tokenizers, &without),
        "b904dd7b9b8b1ca713f489127bde3566467aedfaffe8c3286031a65e1f1027f3"
    );
    assert_eq!(
        metal_lane_fingerprint(entry, backend, &tokenizers, &opted_in),
        backend["fingerprint"].as_str().unwrap()
    );
    assert_ne!(
        backend["fingerprint"],
        "b904dd7b9b8b1ca713f489127bde3566467aedfaffe8c3286031a65e1f1027f3"
    );
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
