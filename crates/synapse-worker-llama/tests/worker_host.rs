use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Deserialize;
use synapse_core::{RuntimeConfig, TokenBatch, ValidatedArtifact, WorkerPooling};
use synapse_module::worker_host::{CrashBudget, WorkerHost, WorkerHostConfig, WorkerHostError};
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

#[tokio::test]
async fn worker_loads_minilm_and_embeds_batch_with_ort_parity() {
    let Some(paths) = model_paths() else {
        eprintln!("skipping worker MiniLM round-trip: local HF cache is missing ONNX or GGUF MiniLM snapshots");
        return;
    };

    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../synapse-engine-ort/tests/fixtures/minilm_golden.json"
    ))
    .expect("golden fixture should decode");
    let batch = tokenize_fixture(&paths.tokenizer, &fixture);

    let mut host = WorkerHost::new(worker_config("minilm-roundtrip"));
    let model = host
        .load_model(&gguf_artifact(), &worker_runtime_config(&paths.gguf))
        .await
        .expect("worker LOAD should succeed");
    let vectors = host
        .embed_batch(&model, batch)
        .await
        .expect("worker EMBED_BATCH should succeed");
    assert_eq!(vectors.len(), fixture.items.len());
    for (index, (actual, expected)) in vectors
        .iter()
        .zip(fixture.items.iter().map(|item| &item.vector))
        .enumerate()
    {
        let cosine = cosine(actual, expected);
        assert!(
            cosine >= 0.9999,
            "worker/ORT golden cosine for item {index} was {cosine:.9}"
        );
    }
}

#[tokio::test]
async fn host_classifies_crashes_and_quarantines_after_budget() {
    let mut config = worker_config("abort-budget");
    config
        .extra_args
        .push("--test-abort-on-request".to_string());
    config.request_timeout = Duration::from_secs(2);
    config.crash_budget = CrashBudget {
        max_crashes: 2,
        window: Duration::from_secs(30),
    };
    let mut host = WorkerHost::new(config);
    let mut runtime_config = RuntimeConfig::default();
    runtime_config.values.insert(
        "artifact_path".to_string(),
        "/tmp/abort-before-load.gguf".to_string(),
    );

    let first = host.load_model(&gguf_artifact(), &runtime_config).await;
    assert!(matches!(first, Err(WorkerHostError::EngineCrashed { .. })));
    let second = host.load_model(&gguf_artifact(), &runtime_config).await;
    assert!(matches!(second, Err(WorkerHostError::EngineCrashed { .. })));
    let third = host.load_model(&gguf_artifact(), &runtime_config).await;
    assert!(matches!(third, Err(WorkerHostError::Quarantined { .. })));
}

fn worker_config(worker_id: &str) -> WorkerHostConfig {
    let mut config = WorkerHostConfig::new(
        env!("CARGO_BIN_EXE_synapse-worker-llama"),
        PathBuf::from(format!("/tmp/synw-{}", std::process::id())),
    );
    config.worker_id = format!("{worker_id}-{}", short_suffix());
    config.pooling = WorkerPooling::Mean;
    config.normalize = true;
    config.handshake_timeout = Duration::from_secs(5);
    config.request_timeout = Duration::from_secs(30);
    config
}

fn worker_runtime_config(gguf: &Path) -> RuntimeConfig {
    let mut values = BTreeMap::new();
    values.insert(
        "artifact_path".to_string(),
        gguf.to_string_lossy().to_string(),
    );
    values.insert("pooling_implementation".to_string(), "manual".to_string());
    values.insert("forward_pass".to_string(), "encode".to_string());
    values.insert("ctx_size".to_string(), "512".to_string());
    values.insert("batch_size".to_string(), "4096".to_string());
    values.insert("ubatch_size".to_string(), "1024".to_string());
    RuntimeConfig { values }
}

fn gguf_artifact() -> ValidatedArtifact {
    ValidatedArtifact {
        digest: String::new(),
        format: "gguf".to_string(),
    }
}

fn tokenize_fixture(tokenizer_path: &Path, fixture: &Fixture) -> TokenBatch {
    let mut tokenizer = Tokenizer::from_file(tokenizer_path).expect("load MiniLM tokenizer");
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
    TokenBatch {
        items: encodings
            .iter()
            .map(|encoding| encoding.get_ids().to_vec())
            .collect(),
    }
}

struct ModelPaths {
    tokenizer: PathBuf,
    gguf: PathBuf,
}

fn model_paths() -> Option<ModelPaths> {
    let onnx_snapshot = minilm_onnx_snapshot()?;
    let gguf_snapshot = minilm_gguf_snapshot()?;
    let tokenizer = onnx_snapshot.join("tokenizer.json");
    let gguf = gguf_snapshot.join("all-MiniLM-L6-v2-ggml-model-f16.gguf");
    if tokenizer.exists() && gguf.exists() {
        Some(ModelPaths { tokenizer, gguf })
    } else {
        None
    }
}

fn minilm_onnx_snapshot() -> Option<PathBuf> {
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
    first_snapshot_with(&snapshots, "tokenizer.json")
}

fn minilm_gguf_snapshot() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("SYNAPSE_MINILM_GGUF_SNAPSHOT") {
        return Some(PathBuf::from(path));
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots = PathBuf::from(home).join(
        ".cache/huggingface/hub/models--second-state--All-MiniLM-L6-v2-Embedding-GGUF/snapshots",
    );
    first_snapshot_with(&snapshots, "all-MiniLM-L6-v2-ggml-model-f16.gguf")
}

fn first_snapshot_with(snapshots: &Path, file_name: &str) -> Option<PathBuf> {
    std::fs::read_dir(snapshots)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| path.join(file_name).exists())
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

fn short_suffix() -> String {
    format!(
        "{:x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos()
    )
}
