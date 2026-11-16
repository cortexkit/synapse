use super::*;
use half::f16;
use safetensors::{
    tensor::{serialize_to_file, Dtype, TensorView},
    SafeTensors,
};
use serde_json::json;
use std::{
    collections::HashMap,
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

fn checkpoint(tied: bool) -> PathBuf {
    let root = PathBuf::from("target/scratch").join(format!(
        "asr-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("config.json"), serde_json::to_vec(&json!({"thinker_config":{"text_config":{"num_hidden_layers":1,"tie_word_embeddings":tied}}})).unwrap()).unwrap();
    let mut names = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]
    .into_iter()
    .map(|s| format!("thinker.model.layers.0.{s}"))
    .collect::<Vec<_>>();
    names.extend(
        [
            "thinker.model.embed_tokens.weight",
            "thinker.model.norm.weight",
            "thinker.lm_head.weight",
            "thinker.audio_tower.proj2.weight",
        ]
        .map(str::to_string),
    );
    let mut weight_map = serde_json::Map::new();
    let values = [1.0_f32 + 1.0 / 2048.0, 1.0 + 3.0 / 2048.0]
        .into_iter()
        .cycle()
        .take(32)
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let head_values = [3.0_f32; 32]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    for shard in 0..2 {
        let filename = format!("model-0000{}-of-00002.safetensors", shard + 1);
        let mut tensors = HashMap::new();
        for (index, name) in names.iter().enumerate().filter(|(i, _)| i % 2 == shard) {
            let data = if name == "thinker.lm_head.weight" && !tied {
                &head_values
            } else {
                &values
            };
            let shape = if index == 8 { vec![32] } else { vec![1, 32] };
            tensors.insert(
                name.clone(),
                TensorView::new(Dtype::F32, shape, data).unwrap(),
            );
            weight_map.insert(name.clone(), json!(filename));
        }
        serialize_to_file(&tensors, None, &root.join(filename)).unwrap();
    }
    fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({"weight_map":weight_map})).unwrap(),
    )
    .unwrap();
    root
}

fn derive(root: &Path) -> (CachedQ8Artifact, PathBuf, AsrArtifactPins) {
    let q8 = q8artifact::derive_q8(
        root,
        &root.join("cache"),
        QuantizerFamily::Qwen3Asr,
        "source",
        QUANTIZER_REVISION,
    )
    .unwrap();
    let f16 = derive_f16(root, &q8).unwrap();
    let pins = AsrArtifactPins {
        quantizer_revision: QUANTIZER_REVISION.into(),
        q8_0_digest: q8artifact::sha256_file(&q8.object_path).unwrap(),
        f16_digest: q8artifact::sha256_file(&f16).unwrap(),
    };
    (q8, f16, pins)
}

#[test]
fn sharded_asr_has_seven_linears_and_head_and_canonical_f16_remainder() {
    let mut tied_digest = None;
    for tied in [true, false] {
        let root = checkpoint(tied);
        let (q8, f16_path, pins) = derive(&root);
        if tied {
            tied_digest = Some(q8.derived_digest.clone());
        } else {
            assert_ne!(tied_digest.as_ref(), Some(&q8.derived_digest));
        }
        let lineage: serde_json::Value =
            serde_json::from_slice(&fs::read(q8.metadata_path).unwrap()).unwrap();
        assert_eq!(lineage["tensors"].as_array().unwrap().len(), 8);
        assert_eq!(lineage["tensors"][7]["name"], "thinker.lm_head.weight");
        let bytes = fs::read(&f16_path).unwrap();
        let tensors = SafeTensors::deserialize(&bytes).unwrap();
        assert_eq!(tensors.len(), 3);
        assert!(tensors.tensor("thinker.lm_head.weight").is_err());
        let mut canonical = std::collections::BTreeMap::new();
        for name in [
            "thinker.audio_tower.proj2.weight",
            "thinker.model.embed_tokens.weight",
            "thinker.model.norm.weight",
        ] {
            let tensor = tensors.tensor(name).unwrap();
            assert_eq!(tensor.dtype(), Dtype::F16);
            let bits = tensor
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u16::from_le_bytes(*b))
                .collect::<Vec<_>>();
            assert_eq!(
                bits,
                [0x3c00, 0x3c02]
                    .into_iter()
                    .cycle()
                    .take(32)
                    .collect::<Vec<_>>()
            );
            assert_eq!(f16::from_bits(bits[0]).to_f32(), 1.0);
            canonical.insert(
                name.to_string(),
                synapse_parity::safetensors::PackageTensor {
                    dtype: synapse_parity::safetensors::StDType::F16,
                    shape: tensor.shape().iter().map(|d| *d as u64).collect(),
                    bytes: bits.iter().flat_map(|b| b.to_le_bytes()).collect(),
                },
            );
        }
        assert_eq!(
            bytes,
            synapse_parity::safetensors::write_package(F16_PROFILE, &canonical)
        );
        let mut registry = Q8IngestRegistry::new();
        load_with_pins(&root, &root.join("cache"), &mut registry, "source", &pins).unwrap();
        assert!(
            load_with_pins(&root, &root.join("cache"), &mut registry, "source", &pins)
                .unwrap()
                .q8
                .reused
        );
        fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn trusted_cached_asr_tampering_fails_closed_for_each_derived_file() {
    for tamper_q8 in [true, false] {
        let root = checkpoint(true);
        let (q8, f16, pins) = derive(&root);
        let mut registry = Q8IngestRegistry::new();
        load_with_pins(&root, &root.join("cache"), &mut registry, "source", &pins).unwrap();
        assert_eq!(
            registry
                .entry("source", QUANTIZER_REVISION)
                .unwrap()
                .trust_state,
            super::super::q8ingest::TrustState::Trusted
        );
        let path = if tamper_q8 { q8.object_path } else { f16 };
        let mut bytes = fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        fs::write(path, bytes).unwrap();
        let error =
            load_with_pins(&root, &root.join("cache"), &mut registry, "source", &pins).unwrap_err();
        assert_eq!(
            error.downcast_ref::<OwnedDecodeError>(),
            Some(&OwnedDecodeError::ArtifactPoisoned)
        );
        assert_eq!(
            registry
                .entry("source", QUANTIZER_REVISION)
                .unwrap()
                .trust_state,
            super::super::q8ingest::TrustState::Poisoned
        );
        fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn cold_asr_load_rejects_wrong_pin_and_missing_or_corrupt_shard() {
    let root = checkpoint(true);
    let (_, _, mut pins) = derive(&root);
    fs::remove_dir_all(root.join("cache")).unwrap();
    pins.f16_digest = "0".repeat(64);
    let error = load_with_pins(
        &root,
        &root.join("cache"),
        &mut Q8IngestRegistry::new(),
        "source",
        &pins,
    )
    .unwrap_err();
    assert_eq!(
        error.downcast_ref::<OwnedDecodeError>(),
        Some(&OwnedDecodeError::ArtifactPoisoned)
    );
    fs::remove_dir_all(root.join("cache")).unwrap();
    let shard = root.join("model-00002-of-00002.safetensors");
    fs::write(&shard, b"corrupt").unwrap();
    assert!(derive_q8_error(&root));
    fs::remove_file(shard).unwrap();
    assert!(derive_q8_error(&root));
    fs::remove_dir_all(root).unwrap();
}

fn derive_q8_error(root: &Path) -> bool {
    q8artifact::derive_q8(
        root,
        &root.join("cache"),
        QuantizerFamily::Qwen3Asr,
        "source",
        QUANTIZER_REVISION,
    )
    .is_err()
}

#[test]
fn f16_non_finite_and_overflow_are_refused_without_publishing_a_package() {
    for value in [f32::NAN, 100_000.0] {
        let root = checkpoint(true);
        let index: serde_json::Value =
            serde_json::from_slice(&fs::read(root.join("model.safetensors.index.json")).unwrap())
                .unwrap();
        let name = "thinker.audio_tower.proj2.weight";
        let shard = root.join(index["weight_map"][name].as_str().unwrap());
        let mut bytes = fs::read(&shard).unwrap();
        let length = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + length]).unwrap();
        let offset = 8 + length + header[name]["data_offsets"][0].as_u64().unwrap() as usize;
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        fs::write(shard, bytes).unwrap();
        let q8 = q8artifact::derive_q8(
            &root,
            &root.join("cache"),
            QuantizerFamily::Qwen3Asr,
            "source",
            QUANTIZER_REVISION,
        )
        .unwrap();
        assert!(derive_f16(&root, &q8)
            .unwrap_err()
            .to_string()
            .contains("non-finite f16"));
        assert!(!q8
            .object_path
            .with_file_name("remaining.f16.safetensors")
            .exists());
        fs::remove_dir_all(root).unwrap();
    }
}
