use std::path::PathBuf;

use serde::Deserialize;
use synapse_core::{EmbedEngine, RuntimeConfig, TokenBatch, ValidatedArtifact, WorkerPooling};
use synapse_engine_ort::OrtEmbedEngine;
use tokenizers::{Tokenizer, TruncationParams};

#[derive(Debug, Deserialize)]
struct Fixture {
    items: Vec<FixtureItem>,
}

#[derive(Debug, Deserialize)]
struct FixtureItem {
    text: String,
    vector: Vec<f32>,
}

#[test]
fn embeds_fixed_minilm_strings_against_committed_goldens() {
    // Golden vectors were generated from the bench reference lane with:
    // cargo run -p lane-ort-embed -- --model "$HOME/.cache/huggingface/hub/models--Qdrant--all-MiniLM-L6-v2-onnx/snapshots/manual/model.onnx" --tokenizer "$HOME/.cache/huggingface/hub/models--Qdrant--all-MiniLM-L6-v2-onnx/snapshots/manual/tokenizer.json" --corpus /tmp/synapse-minilm-golden-corpus.jsonl --out /tmp/synapse-minilm-golden-result.json --vectors-out /tmp/synapse-minilm-golden-vectors.jsonl --pooling mean --max-length 512 --model-label all-MiniLM-L6-v2@ort-cpu-fp32
    let Some(snapshot) = minilm_snapshot() else {
        eprintln!(
            "skipping MiniLM ORT golden test: Qdrant ONNX snapshot is not in the local HF cache"
        );
        return;
    };
    let model_path = snapshot.join("model.onnx");
    let tokenizer_path = snapshot.join("tokenizer.json");
    if !model_path.exists() || !tokenizer_path.exists() {
        eprintln!(
            "skipping MiniLM ORT golden test: missing {} or {}",
            model_path.display(),
            tokenizer_path.display()
        );
        return;
    }

    let fixture: Fixture = serde_json::from_str(include_str!("fixtures/minilm_golden.json"))
        .expect("golden fixture should decode");
    let mut tokenizer = Tokenizer::from_file(&tokenizer_path).expect("load MiniLM tokenizer");
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: 512,
            ..Default::default()
        }))
        .expect("configure tokenizer truncation");

    let encodings = tokenizer
        .encode_batch(
            fixture
                .items
                .iter()
                .map(|item| item.text.as_str())
                .collect::<Vec<_>>(),
            true,
        )
        .expect("tokenize fixture strings");
    let batch = TokenBatch {
        items: encodings
            .iter()
            .map(|encoding| encoding.get_ids().to_vec())
            .collect(),
    };

    let mut cfg = RuntimeConfig::default();
    cfg.values.insert(
        "model_path".to_string(),
        model_path.to_string_lossy().to_string(),
    );
    cfg.values.insert(
        "pooling".to_string(),
        WorkerPooling::Mean.as_str().to_string(),
    );
    let mut engine = OrtEmbedEngine::new();
    let loaded = engine
        .load(
            &ValidatedArtifact {
                digest: String::new(),
                format: "onnx".to_string(),
            },
            &cfg,
        )
        .expect("load ORT MiniLM model");
    let vectors = engine
        .embed_batch(&loaded, batch)
        .expect("embed fixed strings");
    assert_eq!(vectors.len(), fixture.items.len());

    for (index, (actual, expected)) in vectors
        .iter()
        .zip(fixture.items.iter().map(|item| &item.vector))
        .enumerate()
    {
        assert_eq!(
            actual.len(),
            expected.len(),
            "vector dim mismatch at item {index}"
        );
        let cosine = cosine(actual, expected);
        assert!(
            cosine >= 0.999_999,
            "golden cosine for item {index} was {cosine:.9}"
        );
    }
}

fn minilm_snapshot() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("SYNAPSE_MINILM_ONNX_SNAPSHOT") {
        return Some(PathBuf::from(path));
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Qdrant--all-MiniLM-L6-v2-onnx/snapshots");
    let manual = snapshots.join("manual");
    if manual.exists() {
        return Some(manual);
    }
    std::fs::read_dir(snapshots)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| path.join("model.onnx").exists())
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    let dot = left
        .iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum::<f32>();
    let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();
    dot / (left_norm * right_norm + 1e-12)
}
