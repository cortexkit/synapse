use serde_json::{json, Value};
use std::collections::BTreeMap;
use synapse_core::{fingerprint::NumericProfile, EngineIdentity};
use synapse_parity::{canonical, manifest::Manifest};

fn manifest() -> Manifest {
    Manifest::from_slice(include_bytes!("../../../bench/parity/models.json")).unwrap()
}

fn worker_identity(lane: &str, version: &str, flags: &[(&str, &str)]) -> EngineIdentity {
    let mut build_flags = BTreeMap::from([("risk_class".into(), "abort_capable".into())]);
    build_flags.extend(flags.iter().map(|(k, v)| (k.to_string(), v.to_string())));
    EngineIdentity {
        engine: lane.into(),
        version: version.into(),
        build_flags,
    }
}

fn transcribe_identity(lane: &str, dtype: &str) -> EngineIdentity {
    use synapse_core::*;
    let revision = match lane {
        "owned-metal" => METAL_TRANSCRIBE_KERNEL_REVISION,
        "owned-cuda" => CUDA_TRANSCRIBE_KERNEL_REVISION,
        "owned-vulkan" => VULKAN_TRANSCRIBE_KERNEL_REVISION,
        "ane-direct-worker" => ANE_DIRECT_TRANSCRIBE_KERNEL_REVISION,
        _ => panic!("unexpected lane"),
    };
    match lane {
        "owned-metal" => EngineIdentity {
            engine: lane.into(),
            version: synapse_engine_owned::ENGINE_VERSION.into(),
            build_flags: BTreeMap::from([
                ("backend".into(), "metal-mpsgraph".into()),
                ("family".into(), "parakeet-tdt".into()),
                ("dtype".into(), dtype.into()),
                ("kernel_revision".into(), revision.into()),
                ("risk_class".into(), "abort_safe".into()),
            ]),
        },
        "owned-cuda" => owned_cuda_engine_identity("parakeet-tdt", dtype, revision),
        _ => worker_identity(
            lane,
            revision,
            &[("family", "parakeet-tdt"), ("kernel_revision", revision)],
        ),
    }
}

#[test]
fn transcribe_catalog_identities_and_fingerprints_bind_only_speech_revisions() {
    let manifest = manifest();
    let catalog: Value = serde_json::from_str(include_str!("../src/catalog/models.json")).unwrap();
    let entry = catalog["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["task"] == "transcribe")
        .unwrap();
    let mut pins = Vec::new();
    for backend in entry["backends"].as_array().unwrap() {
        let id = backend["profile"].as_str().unwrap();
        let profile = &manifest.profiles[id];
        let model = manifest.model(&profile.model).unwrap();
        let dtype = backend["dtype"].as_str().unwrap();
        let mut engine = transcribe_identity(profile.lane.as_str(), dtype);
        engine.build_flags.extend([
            ("profile".into(), id.into()),
            (
                "compute_dtype".into(),
                serde_json::to_value(profile.compute_dtype)
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .into(),
            ),
            ("storage_dtype".into(), dtype.into()),
        ]);
        let identity = serde_json::to_string(&engine).unwrap();
        for forbidden in [
            synapse_core::CUDA_KERNEL_REVISION,
            synapse_core::VULKAN_KERNEL_REVISION,
            synapse_core::ANE_DIRECT_KERNEL_REVISION,
            "graph_revision",
            "GRAPH_REVISION",
            "protocol-v2",
        ] {
            assert!(!identity.contains(forbidden), "{id} contains {forbidden}");
        }
        let frames =
            synapse_speech::valid_encoder_frames(backend["max_duration_ms"].as_u64().unwrap() * 16);
        assert_eq!(frames, 7500);
        let numeric: NumericProfile = serde_json::from_value(json!({
            "model_digest": model.checkpoint_digest, "quant": profile.storage_dtype, "engine": engine,
            "sanitized_tokenizer_digest": model.tokenizer_digest,
            "pooling": {"custom": "none"}, "normalization": "none", "dtype": profile.storage_dtype,
            "flash_attention": "disabled", "certified_shape": {"max_context_tokens": frames,
                "max_batch_tokens": frames, "max_micro_batch_tokens": frames, "max_sequences": 1},
            "thread_policy": "balanced", "operation": "transcribe",
            "kernel_revision": engine.build_flags["kernel_revision"], "rotation": profile.rotation,
            "converted_package_digest": profile.converted_package_digest,
            "manifest_profile_digest": manifest.profile_digest(id).unwrap(),
            "speech_frontend": synapse_speech::SPEECH_FRONTEND_REVISION,
            "decode_rule": synapse_speech::DECODE_RULE_REVISION,
        })).unwrap();
        assert!(numeric.input_grammar.is_none());
        assert!(numeric.prompt_template.is_none());
        pins.push((
            id.to_string(),
            backend["fingerprint"].as_str().unwrap().to_string(),
            numeric.fingerprint().0,
        ));
    }
    assert_eq!(pins.len(), 4);
    assert!(
        pins.iter().all(|(_, pin, actual)| pin == actual),
        "{pins:#?}"
    );
}

#[test]
fn existing_manifest_profiles_and_catalog_profile_fingerprints_are_unchanged() {
    let manifest = manifest();
    let catalog: Value = serde_json::from_str(include_str!("../src/catalog/models.json")).unwrap();
    let tokenizers: BTreeMap<String, String> =
        serde_json::from_str(include_str!("fixtures/catalog-tokenizer-digests.json")).unwrap();
    let mut profiles = 0;
    for (id, profile) in &manifest.profiles {
        if profile.model == "parakeet-tdt-0.6b-v3" {
            continue;
        }
        assert_eq!(
            manifest.profile_digest(id).unwrap(),
            manifest.digests.profiles[id],
            "{id}"
        );
        profiles += 1;
    }
    assert_eq!(profiles, 16);
    let mut lanes = 0;
    for entry in catalog["models"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| matches!(e["task"].as_str(), Some("embed" | "rerank")))
    {
        for backend in entry["backends"].as_array().unwrap() {
            let Some(id) = backend["profile"].as_str() else {
                continue;
            };
            let declared = &manifest.profiles[id];
            let model = manifest.model(&declared.model).unwrap();
            let mut engine = match declared.lane.as_str() {
                "owned-cuda" => synapse_core::owned_cuda_engine_identity(
                    backend["family"].as_str().unwrap(),
                    backend["dtype"].as_str().unwrap(),
                    synapse_core::CUDA_KERNEL_REVISION,
                ),
                lane => worker_identity(lane, "protocol-v2", &[]),
            };
            engine.build_flags.extend([
                ("profile".into(), id.into()),
                (
                    "compute_dtype".into(),
                    serde_json::to_value(declared.compute_dtype)
                        .unwrap()
                        .as_str()
                        .unwrap()
                        .into(),
                ),
                (
                    "storage_dtype".into(),
                    backend["dtype"].as_str().unwrap().into(),
                ),
            ]);
            let kernel = match declared.lane.as_str() {
                "owned-cuda" => synapse_core::CUDA_KERNEL_REVISION,
                "owned-vulkan" => synapse_core::VULKAN_KERNEL_REVISION,
                "ane-direct-worker" => synapse_core::ANE_DIRECT_KERNEL_REVISION,
                _ => panic!("unexpected existing profile lane"),
            };
            let numeric: NumericProfile = serde_json::from_value(json!({
                "model_digest": model.checkpoint_digest, "quant": declared.storage_dtype, "engine": engine,
                "sanitized_tokenizer_digest": tokenizers[&declared.model],
                "pooling": match serde_json::to_value(model.grammar().unwrap().pooling).unwrap().as_str().unwrap() { "cls" => "cls", "masked_mean" => "mean", _ => "last_token" },
                "normalization": model.output().unwrap().normalization, "dtype": declared.storage_dtype,
                "flash_attention": "disabled", "certified_shape": {"max_context_tokens": 8192,
                    "max_batch_tokens": 8192, "max_micro_batch_tokens": 3072, "max_sequences": 64},
                "thread_policy": "balanced", "operation": model.operation,
                "input_grammar": format!("synapse-input-grammar-v1:{}", manifest.grammar_digest(&declared.model).unwrap()),
                "prompt_template": if entry["id"] == "gte-reranker-modernbert-base" {
                    Some(format!("synapse-input-grammar-v1:{}", manifest.grammar_digest(&declared.model).unwrap()))
                } else { None },
                "kernel_revision": kernel, "rotation": declared.rotation,
                "converted_package_digest": declared.converted_package_digest,
                "manifest_profile_digest": canonical::sha256_hex(&canonical::canonical_bytes(&manifest.profile_entry(id).unwrap())),
            })).unwrap();
            assert_eq!(numeric.fingerprint().0, backend["fingerprint"], "{id}");
            lanes += 1;
        }
    }
    assert_eq!(lanes, 11);
}
