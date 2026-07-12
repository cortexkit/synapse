#![cfg(target_os = "macos")]

use std::{
    fs,
    path::{Path, PathBuf},
};

use synapse_core::{EmbedEngine, RuntimeConfig, TokenBatch, ValidatedArtifact};
use synapse_engine_owned::{OwnedDType, OwnedMetalEmbedEngine};
use tokenizers::{Tokenizer, TruncationParams};

#[test]
fn minilm_reuses_precompiled_packages_across_calls() {
    let Some(snapshot) = minilm_snapshot() else {
        eprintln!(
            "skipping owned-metal MiniLM multi-call test: local safetensors snapshot is missing"
        );
        return;
    };
    let model_path = snapshot.join("model.safetensors");
    let tokenizer_path = snapshot.join("tokenizer.json");
    if !model_path.is_file() || !tokenizer_path.is_file() {
        eprintln!(
            "skipping owned-metal MiniLM multi-call test: missing {} or {}",
            model_path.display(),
            tokenizer_path.display()
        );
        return;
    }

    let cache = unique_temp_dir("owned-metal-packages");
    let mut runtime = RuntimeConfig::default();
    runtime.values.insert(
        "model_path".to_string(),
        model_path.to_string_lossy().to_string(),
    );
    runtime.values.insert(
        "package_cache_root".to_string(),
        cache.to_string_lossy().to_string(),
    );
    runtime
        .values
        .insert("execution".to_string(), "explicit".to_string());
    runtime
        .values
        .insert("max_tokens".to_string(), "512".to_string());
    runtime
        .values
        .insert("attention_units".to_string(), "4000000".to_string());

    let mut engine =
        OwnedMetalEmbedEngine::new(synapse_engine_owned::ModelFamily::MiniLm, OwnedDType::F16);
    let loaded = engine
        .load(
            &ValidatedArtifact {
                digest: "sha256:fixture".to_string(),
                format: "safetensors-package".to_string(),
            },
            &runtime,
        )
        .expect("load owned-metal MiniLM");
    let compiled_count = package_count(&cache);
    assert_eq!(
        compiled_count, 10,
        "bucket policy v1 compiles ten sequence shapes"
    );

    let mut tokenizer = Tokenizer::from_file(&tokenizer_path).expect("load tokenizer");
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: 512,
            ..Default::default()
        }))
        .expect("configure tokenizer truncation");
    let first_batch = token_batch(&tokenizer, &["hello world", "the quick brown fox"]);
    let first = engine
        .embed_batch(&loaded, first_batch.clone())
        .expect("first embedding call");
    let second = engine
        .embed_batch(&loaded, first_batch)
        .expect("second embedding call");
    let third = engine
        .embed_batch(
            &loaded,
            token_batch(&tokenizer, &["a third independent call"]),
        )
        .expect("third embedding call");

    assert_eq!(first.len(), 2);
    assert_eq!(second.len(), 2);
    assert_eq!(third.len(), 1);
    for (left, right) in first.iter().zip(second.iter()) {
        assert_eq!(left.len(), 384);
        assert_eq!(left, right, "identical calls must remain deterministic");
    }
    assert_eq!(third[0].len(), 384);
    assert_eq!(
        package_count(&cache),
        compiled_count,
        "serving calls must not compile new graph packages"
    );
    let _ = fs::remove_dir_all(cache);
}

fn token_batch(tokenizer: &Tokenizer, texts: &[&str]) -> TokenBatch {
    let encodings = tokenizer
        .encode_batch(texts.to_vec(), true)
        .expect("tokenize MiniLM inputs");
    TokenBatch {
        items: encodings
            .iter()
            .map(|encoding| encoding.get_ids().to_vec())
            .collect(),
    }
}

fn package_count(root: &Path) -> usize {
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

fn minilm_snapshot() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("SYNAPSE_MINILM_SAFETENSORS_SNAPSHOT") {
        return Some(PathBuf::from(path));
    }
    let snapshots = PathBuf::from(std::env::var("HOME").ok()?)
        .join(".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots");
    fs::read_dir(snapshots)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| path.join("model.safetensors").is_file())
}

fn unique_temp_dir(label: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("synapse-{label}-{}-{nonce}", std::process::id()))
}
