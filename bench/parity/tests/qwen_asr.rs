use serde_json::json;
use synapse_parity::arch;
use synapse_parity::canonical::sha256_hex;
use synapse_parity::manifest::{Digests, Family, Manifest, Operation};
use synapse_parity::{parity_dir, qwen_asr, validate};

fn manifest() -> Manifest {
    Manifest::from_slice(include_bytes!("../models.json")).unwrap()
}

#[test]
fn qwen_asr_model_is_pinned_without_admitting_a_serving_lane() {
    let manifest = manifest();
    validate::validate_manifest(&manifest, &parity_dir()).unwrap();
    let model = &manifest.models["qwen3-asr-1.7b"];
    assert_eq!(model.architecture.family, Family::Qwen3Asr);
    assert_eq!(model.operation, Operation::Transcribe);
    assert_eq!(
        model.hf_revision,
        "7278e1e70fe206f11671096ffdd38061171dd6e5"
    );
    assert_eq!(model.files.len(), 10);
    assert!(!manifest
        .profiles
        .values()
        .any(|p| p.model == "qwen3-asr-1.7b"));
}

#[test]
fn qwen_asr_addition_preserves_every_existing_profile_and_grammar_digest() {
    let expected: Digests =
        serde_json::from_str(include_str!("qwen3-asr-legacy-digests.json")).unwrap();
    let computed = manifest().computed_digests().unwrap();
    assert_eq!(computed.profiles.len(), 20);
    assert_eq!(computed.grammar.len(), 4);
    assert_eq!(computed, expected);
}

#[test]
fn qwen_asr_independent_tensor_inventory_matches_both_shards() {
    let manifest = manifest();
    let model = &manifest.models["qwen3-asr-1.7b"];
    let bytes = include_bytes!("../checkpoints/qwen3-asr-1.7b/tensor-index.json");
    assert_eq!(sha256_hex(bytes), model.tensor_index_sha256);
    let header = synapse_parity::safetensors::parse_header_json(bytes).unwrap();
    let actual = header
        .tensors
        .into_iter()
        .map(|(key, value)| (key, value.shape))
        .collect();
    assert_eq!(arch::expected_tensors(model).unwrap().len(), 708);
    validate::check_tensor_index(model, &actual).unwrap();
    let mut wrong = model.clone();
    wrong
        .architecture
        .params
        .insert("audio_output_dim".into(), json!(1024));
    assert!(validate::check_tensor_index(&wrong, &actual).is_err());
}

#[test]
fn qwen_asr_aggregate_digests_bind_shard_names_and_tokenizer_components() {
    let manifest = manifest();
    let model = &manifest.models["qwen3-asr-1.7b"];
    assert_eq!(
        qwen_asr::file_map_digest(model, &qwen_asr::SHARDS).unwrap(),
        model.checkpoint_digest
    );
    assert_eq!(
        qwen_asr::file_map_digest(model, &qwen_asr::TOKENIZER_FILES).unwrap(),
        model.tokenizer_digest
    );
    let mut missing = model.clone();
    missing.files.remove(qwen_asr::SHARDS[1]);
    assert!(qwen_asr::file_map_digest(&missing, &qwen_asr::SHARDS).is_err());
    let mut changed = model.clone();
    changed.files.insert("merges.txt".into(), "0".repeat(64));
    assert_ne!(
        qwen_asr::file_map_digest(&changed, &qwen_asr::TOKENIZER_FILES).unwrap(),
        model.tokenizer_digest
    );
}

#[test]
fn qwen_asr_nested_config_parameters_are_checked_not_assumed() {
    let manifest = manifest();
    let model = &manifest.models["qwen3-asr-1.7b"];
    let mut config =
        serde_json::from_str(include_str!("../checkpoints/qwen3-asr-1.7b/config.json")).unwrap();
    arch::check_against_config("qwen3-asr-1.7b", &model.architecture, &config, None).unwrap();
    config["thinker_config"]["audio_config"]["d_model"] = json!(512);
    assert!(
        arch::check_against_config("qwen3-asr-1.7b", &model.architecture, &config, None).is_err()
    );
}

#[test]
fn qwen_asr_header_merge_checks_shard_membership_and_duplicate_names() {
    let dir = parity_dir()
        .join("target/scratch")
        .join(format!("qwen-header-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let write_header = |shard: &str, name: &str| {
        let header = synapse_parity::canonical::canonical_bytes(&json!({
            "__metadata__": {"format": "pt"},
            (name): {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}
        }));
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&[0; 4]);
        std::fs::write(dir.join(shard), bytes).unwrap();
    };
    write_header(qwen_asr::SHARDS[0], "a");
    write_header(qwen_asr::SHARDS[1], "b");
    let write_map = |value| {
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&value).unwrap(),
        )
        .unwrap()
    };
    write_map(json!({"weight_map": {"a": qwen_asr::SHARDS[0], "b": qwen_asr::SHARDS[1]}}));
    let merged = qwen_asr::merged_header(&dir).unwrap();
    let parsed = synapse_parity::safetensors::parse_header_json(&merged).unwrap();
    assert_eq!(parsed.tensors.len(), 2);
    write_map(json!({"weight_map": {"a": qwen_asr::SHARDS[0], "b": qwen_asr::SHARDS[0]}}));
    assert!(qwen_asr::merged_header(&dir)
        .unwrap_err()
        .to_string()
        .contains("weight_map disagrees"));
    write_header(qwen_asr::SHARDS[1], "a");
    assert!(qwen_asr::merged_header(&dir)
        .unwrap_err()
        .to_string()
        .contains("duplicate shard tensor"));
    std::fs::remove_dir_all(dir).unwrap();
}
