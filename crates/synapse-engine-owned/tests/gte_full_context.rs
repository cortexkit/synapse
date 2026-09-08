#![cfg(target_os = "macos")]

use std::fs;
use std::path::PathBuf;

use synapse_core::{EmbedEngine, RuntimeConfig, TokenBatch, ValidatedArtifact};
use synapse_engine_owned::{ModelFamily, OwnedDType, OwnedMetalEmbedEngine};

#[test]
fn gte_public_load_and_inference_cover_8192_context() {
    let Some(snapshot) = gte_snapshot() else {
        eprintln!("skipping owned-metal GTE full-context test: local snapshot is missing");
        return;
    };
    for required in ["model.safetensors", "config.json"] {
        if !snapshot.join(required).is_file() {
            eprintln!(
                "skipping owned-metal GTE full-context test: {required} is missing from {}",
                snapshot.display()
            );
            return;
        }
    }

    let cache = unique_temp_dir("owned-metal-gte-full-context");
    let mut runtime = RuntimeConfig::default();
    runtime.values.insert(
        "model_path".to_string(),
        snapshot.join("model.safetensors").display().to_string(),
    );
    runtime.values.insert(
        "package_cache_root".to_string(),
        cache.display().to_string(),
    );
    runtime
        .values
        .insert("execution".to_string(), "explicit".to_string());
    runtime
        .values
        .insert("max_tokens".to_string(), "8192".to_string());
    runtime
        .values
        .insert("attention_units".to_string(), "67108864".to_string());

    let mut engine = OwnedMetalEmbedEngine::new(ModelFamily::GteModernBert, OwnedDType::F16);
    let loaded = engine
        .load(
            &ValidatedArtifact {
                digest: "sha256:full-context-fixture".to_string(),
                format: "safetensors-package".to_string(),
            },
            &runtime,
        )
        .expect("public owned-metal load accepts the certified 8192-token budget");
    assert_eq!(
        package_count(&cache),
        10,
        "full-context load keeps the established short-shape eager set"
    );
    let batch = TokenBatch {
        items: vec![vec![1_u32; 8192]],
    };
    let vectors = engine
        .embed_batch(&loaded, batch.clone())
        .expect("public owned-metal inference covers 8192 tokens");
    assert_eq!(package_count(&cache), 11);
    let repeated = engine
        .embed_batch(&loaded, batch)
        .expect("full-context graph is reused");
    assert_eq!(package_count(&cache), 11);
    assert_eq!(vectors, repeated);
    assert_eq!(vectors.len(), 1);
    assert_eq!(vectors[0].len(), 768);
    let _ = fs::remove_dir_all(cache);
}

fn gte_snapshot() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("SYNAPSE_GTE_MODERNBERT_SAFETENSORS_SNAPSHOT") {
        return Some(PathBuf::from(path));
    }
    let snapshots = PathBuf::from(std::env::var("HOME").ok()?)
        .join(".cache/huggingface/hub/models--Alibaba-NLP--gte-modernbert-base/snapshots");
    fs::read_dir(snapshots)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| path.join("model.safetensors").is_file())
}

fn package_count(root: &std::path::Path) -> usize {
    fs::read_dir(root)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .flat_map(|entry| fs::read_dir(entry.path()).into_iter().flatten())
        .filter_map(Result::ok)
        .filter(|entry| {
            entry.path().extension().and_then(|value| value.to_str()) == Some("mpsgraphpackage")
        })
        .count()
}

fn unique_temp_dir(label: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("synapse-{label}-{}-{nonce}", std::process::id()))
}
