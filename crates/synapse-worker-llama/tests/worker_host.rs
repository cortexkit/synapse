use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Deserialize;
use synapse_core::{
    GenerateRequest, RerankRequest, RuntimeConfig, TokenBatch, ValidatedArtifact, WorkerPooling,
};
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
async fn worker_reranks_gte_modernbert_when_cached() {
    let Some(paths) = gte_reranker_paths() else {
        eprintln!(
            "skipping worker RERANK round-trip: local gte-reranker GGUF or tokenizer is missing"
        );
        return;
    };
    let mut config = worker_config("gte-rerank");
    config.request_timeout = Duration::from_secs(120);
    let mut host = WorkerHost::new(config);
    let model = host
        .load_model(&gguf_artifact(), &worker_runtime_config(&paths.gguf))
        .await
        .expect("worker LOAD should succeed");
    let query = "Which document is about a cat sitting on a mat?";
    let candidates = [
        "A small cat sits on a woven mat near the window.",
        "The spacecraft entered orbit after a six month flight.",
        "A dog chased a tennis ball across the park.",
    ];
    let request = tokenize_rerank_request(&paths.tokenizer, query, &candidates);
    let scores = host
        .rerank(&model, request)
        .await
        .expect("worker RERANK should succeed")
        .scores;
    eprintln!("gte-reranker cat fixture scores: {scores:?}");
    assert_eq!(scores.len(), candidates.len());
    assert!(
        scores[0] > scores[1] && scores[0] > scores[2],
        "relevant cat document should score highest: {scores:?}"
    );

    let weather_scores = host
        .rerank(
            &model,
            tokenize_rerank_request(
                &paths.tokenizer,
                "Which passage discusses rainfall and umbrellas?",
                &[
                    "Heavy rain fell downtown, so commuters opened their umbrellas.",
                    "The recipe calls for flour, butter, and sliced apples.",
                    "A pianist practiced scales before the evening concert.",
                ],
            ),
        )
        .await
        .expect("worker RERANK weather fixture should succeed")
        .scores;
    eprintln!("gte-reranker weather fixture scores: {weather_scores:?}");
    assert!(
        weather_scores[0] > weather_scores[1] && weather_scores[0] > weather_scores[2],
        "relevant weather document should score highest: {weather_scores:?}"
    );
}

#[tokio::test]
async fn worker_generates_qwen3_when_cached() {
    let Some(paths) = qwen3_generate_paths() else {
        eprintln!("skipping worker GENERATE round-trip: local Qwen3 GGUF or tokenizer is missing");
        return;
    };
    let mut host = WorkerHost::new(worker_config("qwen-generate"));
    let model = host
        .load_model(&gguf_artifact(), &generate_runtime_config(&paths.gguf))
        .await
        .expect("worker LOAD should succeed");
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../synapse-module/src/fixtures/probe_generate_qwen3_0_6b_v1.json"
    ))
    .expect("generate fixture should decode");
    let mut matches = 0_usize;
    for item in fixture["items"].as_array().expect("fixture items") {
        let prompt = item["prompt"].as_str().expect("fixture prompt");
        let expected = item["expected_label"].as_str().expect("fixture label");
        let max_tokens = item["max_tokens"].as_u64().unwrap_or(4) as u32;
        let output = host
            .generate(
                &model,
                GenerateRequest {
                    prompt: tokenize_prompt(&paths.tokenizer, prompt),
                    max_tokens,
                    grammar: None,
                },
            )
            .await
            .expect("worker GENERATE should succeed");
        eprintln!("qwen3 fixture {} output: {:?}", item["id"], output);
        assert!(output.n_prompt > 0);
        assert!(output.n_gen <= max_tokens as usize);
        if first_label(&output.text) == first_label(expected) {
            matches += 1;
        }
    }
    assert!(
        matches >= 7,
        "expected at least 7 fixture label matches, got {matches}"
    );
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
        env!("CARGO_BIN_EXE_ck-synapse-worker-llama"),
        PathBuf::from(format!("/tmp/synw-{}", std::process::id())),
    );
    config.worker_id = format!("{worker_id}-{}", short_suffix());
    config.pooling = WorkerPooling::Mean;
    config.normalize = true;
    config.handshake_timeout = Duration::from_secs(5);
    config.request_timeout = Duration::from_secs(120);
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

fn generate_runtime_config(gguf: &Path) -> RuntimeConfig {
    let mut values = BTreeMap::new();
    values.insert(
        "artifact_path".to_string(),
        gguf.to_string_lossy().to_string(),
    );
    values.insert("ctx_size".to_string(), "1024".to_string());
    values.insert("batch_size".to_string(), "1024".to_string());
    values.insert("ubatch_size".to_string(), "512".to_string());
    RuntimeConfig { values }
}

fn gguf_artifact() -> ValidatedArtifact {
    ValidatedArtifact {
        digest: String::new(),
        format: "gguf".to_string(),
    }
}

fn tokenize_rerank_request(
    tokenizer_path: &Path,
    query: &str,
    candidates: &[&str],
) -> RerankRequest {
    let mut tokenizer = Tokenizer::from_file(tokenizer_path).expect("load rerank tokenizer");
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: 512,
            ..Default::default()
        }))
        .expect("configure tokenizer truncation");
    let query_ids = tokenizer
        .encode(query, false)
        .expect("tokenize rerank query")
        .get_ids()
        .to_vec();
    let candidate_ids = candidates
        .iter()
        .map(|candidate| {
            tokenizer
                .encode(*candidate, false)
                .expect("tokenize rerank candidate")
                .get_ids()
                .to_vec()
        })
        .collect();
    RerankRequest {
        query: query_ids,
        candidates: candidate_ids,
    }
}

fn tokenize_prompt(tokenizer_path: &Path, prompt: &str) -> Vec<u32> {
    let mut tokenizer = Tokenizer::from_file(tokenizer_path).expect("load generate tokenizer");
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: 1024,
            ..Default::default()
        }))
        .expect("configure tokenizer truncation");
    tokenizer
        .encode(prompt, true)
        .expect("tokenize generate prompt")
        .get_ids()
        .to_vec()
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

fn gte_reranker_paths() -> Option<ModelPaths> {
    let tokenizer_snapshot = gte_reranker_tokenizer_snapshot()?;
    let gguf_snapshot = gte_reranker_gguf_snapshot()?;
    let tokenizer = tokenizer_snapshot.join("tokenizer.json");
    let gguf = gguf_snapshot.join("gte-reranker-modernbert-base-f16.gguf");
    if tokenizer.exists() && gguf.exists() {
        return Some(ModelPaths { tokenizer, gguf });
    }
    ms_marco_reranker_paths()
}

fn ms_marco_reranker_paths() -> Option<ModelPaths> {
    let tokenizer_snapshot = ms_marco_reranker_tokenizer_snapshot()?;
    let gguf_snapshot = ms_marco_reranker_gguf_snapshot()?;
    let tokenizer = tokenizer_snapshot.join("tokenizer.json");
    let gguf = gguf_snapshot.join("ms-marco-MiniLM-L6-v2-F16.gguf");
    if tokenizer.exists() && gguf.exists() {
        return Some(ModelPaths { tokenizer, gguf });
    }
    bge_reranker_paths()
}

fn bge_reranker_paths() -> Option<ModelPaths> {
    let tokenizer_snapshot = bge_reranker_tokenizer_snapshot()?;
    let gguf_snapshot = bge_reranker_gguf_snapshot()?;
    let tokenizer = tokenizer_snapshot.join("tokenizer.json");
    let gguf = gguf_snapshot.join("bge-reranker-v2-m3-Q8_0.gguf");
    if tokenizer.exists() && gguf.exists() {
        Some(ModelPaths { tokenizer, gguf })
    } else {
        None
    }
}

fn qwen3_generate_paths() -> Option<ModelPaths> {
    let tokenizer_snapshot = qwen3_tokenizer_snapshot()?;
    let gguf_snapshot = qwen3_gguf_snapshot()?;
    let tokenizer = tokenizer_snapshot.join("tokenizer.json");
    let gguf = gguf_snapshot.join("Qwen3-0.6B-Q8_0.gguf");
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

fn gte_reranker_tokenizer_snapshot() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("SYNAPSE_GTE_RERANKER_TOKENIZER_SNAPSHOT") {
        return Some(PathBuf::from(path));
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Alibaba-NLP--gte-reranker-modernbert-base/snapshots");
    first_snapshot_with(&snapshots, "tokenizer.json")
}

fn gte_reranker_gguf_snapshot() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("SYNAPSE_GTE_RERANKER_GGUF_SNAPSHOT") {
        return Some(PathBuf::from(path));
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots = PathBuf::from(home).join(
        ".cache/huggingface/hub/models--jolleyboy--gte-reranker-modernbert-base-GGUF/snapshots",
    );
    first_snapshot_with(&snapshots, "gte-reranker-modernbert-base-f16.gguf")
}

fn ms_marco_reranker_tokenizer_snapshot() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("SYNAPSE_MSMARCO_RERANKER_TOKENIZER_SNAPSHOT") {
        return Some(PathBuf::from(path));
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L6-v2/snapshots");
    first_snapshot_with(&snapshots, "tokenizer.json")
}

fn ms_marco_reranker_gguf_snapshot() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("SYNAPSE_MSMARCO_RERANKER_GGUF_SNAPSHOT") {
        return Some(PathBuf::from(path));
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--sinjab--ms-marco-MiniLM-L6-v2-F16-GGUF/snapshots");
    first_snapshot_with(&snapshots, "ms-marco-MiniLM-L6-v2-F16.gguf")
}

fn bge_reranker_tokenizer_snapshot() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("SYNAPSE_BGE_RERANKER_TOKENIZER_SNAPSHOT") {
        return Some(PathBuf::from(path));
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots");
    first_snapshot_with(&snapshots, "tokenizer.json")
}

fn bge_reranker_gguf_snapshot() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("SYNAPSE_BGE_RERANKER_GGUF_SNAPSHOT") {
        return Some(PathBuf::from(path));
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--gpustack--bge-reranker-v2-m3-GGUF/snapshots");
    first_snapshot_with(&snapshots, "bge-reranker-v2-m3-Q8_0.gguf")
}

fn qwen3_tokenizer_snapshot() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("SYNAPSE_QWEN3_TOKENIZER_SNAPSHOT") {
        return Some(PathBuf::from(path));
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots =
        PathBuf::from(home).join(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots");
    first_snapshot_with(&snapshots, "tokenizer.json")
}

fn qwen3_gguf_snapshot() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("SYNAPSE_QWEN3_GGUF_SNAPSHOT") {
        return Some(PathBuf::from(path));
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots =
        PathBuf::from(home).join(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B-GGUF/snapshots");
    first_snapshot_with(&snapshots, "Qwen3-0.6B-Q8_0.gguf")
}

fn first_snapshot_with(snapshots: &Path, file_name: &str) -> Option<PathBuf> {
    std::fs::read_dir(snapshots)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| path.join(file_name).exists())
}

fn first_label(output: &str) -> String {
    output
        .split(|ch: char| ch.is_ascii_punctuation() || ch.is_whitespace())
        .find(|part| !part.is_empty())
        .unwrap_or("")
        .to_ascii_lowercase()
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
