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
fn catalog_engine_identity(entry: &Value, backend: &Value) -> synapse_core::EngineIdentity {
    let family = ModelFamily::parse(backend["family"].as_str().unwrap()).unwrap();
    let dtype = OwnedDType::parse(backend["dtype"].as_str().unwrap()).unwrap();
    let projections = if family == ModelFamily::GteModernBert && dtype == OwnedDType::F16 {
        ProjectionDtype::F16
    } else {
        ProjectionDtype::F32
    };
    let mut identity = engine_identity_with_projections(family, dtype, projections);
    if entry["task"] == "rerank" {
        identity
            .build_flags
            .insert("rerank_readout".into(), "sigmoid_classifier_logit".into());
    }
    identity
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
                &catalog_engine_identity(entry, backend),
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
    let opted_in = catalog_engine_identity(entry, backend);
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
fn the_reranker_metal_pin_moved_only_by_probability_readout() {
    let catalog: Value = serde_json::from_str(include_str!("../src/catalog/models.json")).unwrap();
    let tokenizers: BTreeMap<String, String> =
        serde_json::from_str(include_str!("fixtures/catalog-tokenizer-digests.json")).unwrap();
    let entry = catalog["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["id"] == "gte-reranker-modernbert-base")
        .unwrap();
    let backend = entry["backends"]
        .as_array()
        .unwrap()
        .iter()
        .find(|backend| backend["backend"] == "metal")
        .unwrap();
    let identity = catalog_engine_identity(entry, backend);
    assert_eq!(
        metal_lane_fingerprint(entry, backend, &tokenizers, &identity),
        backend["fingerprint"]
    );
    let mut legacy_readout = identity;
    assert_eq!(
        legacy_readout
            .build_flags
            .remove("rerank_readout")
            .as_deref(),
        Some("sigmoid_classifier_logit")
    );
    assert_eq!(
        metal_lane_fingerprint(entry, backend, &tokenizers, &legacy_readout),
        "6da1fa7f1dfa0c7c883dd34ca9c3e5d06e50ceaaabbb13d49fd7b7b291b0b9b9"
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

pub(crate) mod qwen3_ane_golden {
    // Captured at 0075afb7bb47f460b65a800c4ee6bebdc12d3f45 from both the
    // hand-built profile and verified catalog installation. Keep complete bytes
    // rather than only a hash so changes to any identity field are visible.
    pub(crate) const STABLE_BYTES: &[u8] = br#"{"model_digest":"sha256:0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd","quant":"f16","engine":{"engine":"ane-direct-worker","version":"protocol-v2","build_flags":{"compute_dtype":"f16","profile":"qwen3-embedding-0.6b.ane-direct-worker","risk_class":"abort_capable","storage_dtype":"f16"}},"sanitized_tokenizer_digest":"sha256:71241c4586582e04d5d51a420bad74d3ef5353bfdac5a53a7ba7440dbf33e350","pooling":"last_token","normalization":"l2","dtype":"f16","flash_attention":"disabled","certified_shape":{"max_context_tokens":8192,"max_batch_tokens":8192,"max_micro_batch_tokens":3072,"max_sequences":64},"thread_policy":"balanced","operation":"embed","input_grammar":"synapse-input-grammar-v1:0786dec33dd85e7bf6da2bfb5ebfd16736c33dfc87fd8680575853f30facbd21","kernel_revision":"ane-direct-graph-v1","rotation":"none","converted_package_digest":"sha256:1af6d091d7d5f21a998af0f438333d9af051c82af8b27bd2439c60efa09bd5db","manifest_profile_digest":"05e8ac1af8e3e78e2cbde230ddb16d6fb08379df42b984f2a366e7da2b72a511"}"#;
    pub(crate) const FINGERPRINT: &str =
        "5f7ac7cdbf63e71f04405b2130b2bde8eb7b307e5d929419bcd747b129806a6b";
    pub(crate) const ENGINE_IDENTITY: &[u8] = br#"{"engine":"ane-direct-worker","version":"protocol-v2","build_flags":{"compute_dtype":"f16","profile":"qwen3-embedding-0.6b.ane-direct-worker","risk_class":"abort_capable","storage_dtype":"f16"}}"#;
}

#[test]
fn qwen3_ane_numeric_profile_and_engine_identity_match_golden() {
    let catalog: Value = serde_json::from_str(include_str!("../src/catalog/models.json")).unwrap();
    let manifest = synapse_parity::manifest::Manifest::from_slice(include_bytes!(
        "../../../bench/parity/models.json"
    ))
    .unwrap();
    let tokenizers: BTreeMap<String, String> =
        serde_json::from_str(include_str!("fixtures/catalog-tokenizer-digests.json")).unwrap();
    let entry = catalog["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["id"] == "qwen3-embedding-0.6b")
        .unwrap();
    let backend = entry["backends"]
        .as_array()
        .unwrap()
        .iter()
        .find(|backend| backend["backend"] == "ane")
        .unwrap();
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
        "pooling": if backend["pooling"] == "last" { "last_token" } else { "cls" },
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
    assert_eq!(profile.stable_bytes(), qwen3_ane_golden::STABLE_BYTES);
    assert_eq!(backend["fingerprint"], qwen3_ane_golden::FINGERPRINT);
    assert_eq!(profile.fingerprint().0, qwen3_ane_golden::FINGERPRINT);
    let expected_engine: synapse_core::EngineIdentity =
        serde_json::from_slice(qwen3_ane_golden::ENGINE_IDENTITY).unwrap();
    assert_eq!(profile.engine, expected_engine);
    assert_eq!(
        serde_json::to_vec(&profile.engine).unwrap(),
        qwen3_ane_golden::ENGINE_IDENTITY
    );

    // A catalog fingerprint hashes [checkpoint, quant, SHA-256(stable bytes)],
    // not the stable bytes directly. Hash the committed literal independently
    // of the current NumericProfile serializer to bind it to the catalog pin.
    let literal: NumericProfile = serde_json::from_slice(qwen3_ane_golden::STABLE_BYTES).unwrap();
    let numeric_id = hex::encode(Sha256::digest(qwen3_ane_golden::STABLE_BYTES));
    let payload =
        serde_json::to_vec(&json!([literal.model_digest, literal.quant, numeric_id])).unwrap();
    assert_eq!(
        hex::encode(Sha256::digest(payload)),
        qwen3_ane_golden::FINGERPRINT
    );
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
