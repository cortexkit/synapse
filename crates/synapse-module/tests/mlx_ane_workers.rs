#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Deserialize;
use synapse_core::{EmbedEngine, RuntimeConfig, TokenBatch, ValidatedArtifact, WorkerPooling};
use synapse_module::worker_host::{CrashBudget, WorkerEngine, WorkerHostConfig};
use tokenizers::Tokenizer;

#[derive(Deserialize)]
struct GoldenFile {
    items: Vec<GoldenItem>,
}

#[derive(Deserialize)]
struct GoldenItem {
    text: String,
    vector: Vec<f32>,
}

#[test]
fn mlx_worker_minilm_matches_ort_golden_when_artifacts_exist() {
    let Some(worker_bin) = env_path("SYNAPSE_MLX_WORKER_BIN") else {
        skip("SYNAPSE_MLX_WORKER_BIN is not set");
        return;
    };
    let Some(model_path) = env_path("SYNAPSE_MLX_MINILM_SAFETENSORS") else {
        skip("SYNAPSE_MLX_MINILM_SAFETENSORS is not set");
        return;
    };
    let Some(tokenizer_path) = env_path("SYNAPSE_MINILM_TOKENIZER_JSON") else {
        skip("SYNAPSE_MINILM_TOKENIZER_JSON is not set");
        return;
    };

    let actual = run_embedding_worker(
        "mlx",
        worker_bin,
        model_path,
        "safetensors",
        tokenizer_path,
        &[("architecture", "bert")],
    );
    let expected = golden_items();
    let mean = mean_pairwise_cosine(&actual, &expected);
    assert!(mean >= 0.99, "MLX MiniLM mean cosine {mean:.6} < 0.99");
}

#[test]
fn ane_worker_minilm_matches_ort_golden_when_artifacts_exist() {
    let Some(worker_bin) = env_path("SYNAPSE_ANE_WORKER_BIN") else {
        skip("SYNAPSE_ANE_WORKER_BIN is not set");
        return;
    };
    let Some(model_path) = env_path("SYNAPSE_ANE_MINILM_MLMODELC").or_else(default_ane_fixture)
    else {
        skip("SYNAPSE_ANE_MINILM_MLMODELC is not set and the bench-tools fixture is absent");
        return;
    };
    let Some(tokenizer_path) = env_path("SYNAPSE_MINILM_TOKENIZER_JSON") else {
        skip("SYNAPSE_MINILM_TOKENIZER_JSON is not set");
        return;
    };

    let actual = run_embedding_worker(
        "ane",
        worker_bin,
        model_path,
        "mlmodelc",
        tokenizer_path,
        &[],
    );
    let expected = golden_items();
    let mean = mean_pairwise_cosine(&actual, &expected);
    assert!(mean >= 0.99, "ANE MiniLM mean cosine {mean:.6} < 0.99");
}

#[test]
fn mlx_and_ane_workers_share_crash_quarantine_path_when_available() {
    for (label, env_key) in [
        ("mlx", "SYNAPSE_MLX_WORKER_BIN"),
        ("ane", "SYNAPSE_ANE_WORKER_BIN"),
    ] {
        let Some(worker_bin) = env_path(env_key) else {
            skip(&format!("{env_key} is not set"));
            continue;
        };
        assert_worker_crash_is_quarantined(label, worker_bin);
    }
}

fn run_embedding_worker(
    label: &str,
    worker_bin: PathBuf,
    model_path: PathBuf,
    format: &str,
    tokenizer_path: PathBuf,
    runtime_overrides: &[(&str, &str)],
) -> Vec<Vec<f32>> {
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .unwrap_or_else(|error| panic!("load tokenizer {}: {error}", tokenizer_path.display()));
    let golden = golden_file();
    let items = golden
        .items
        .iter()
        .map(|item| {
            tokenizer
                .encode(item.text.clone(), true)
                .unwrap_or_else(|error| panic!("tokenize '{}': {error}", item.text))
                .get_ids()
                .to_vec()
        })
        .collect::<Vec<_>>();

    let mut config = WorkerHostConfig::new(worker_bin, temp_runtime_dir(label));
    config.worker_id = format!("{label}-e2e-{}", std::process::id());
    config.pooling = WorkerPooling::Mean;
    config.normalize = true;
    config.request_timeout = Duration::from_secs(60);
    let mut engine = WorkerEngine::new(config).expect("worker host should initialize");
    let mut runtime_config = RuntimeConfig::default();
    runtime_config.values.insert(
        "artifact_path".to_string(),
        model_path.display().to_string(),
    );
    for (key, value) in runtime_overrides {
        runtime_config
            .values
            .insert((*key).to_string(), (*value).to_string());
    }
    let loaded = engine
        .load(
            &ValidatedArtifact {
                digest: String::new(),
                format: format.to_string(),
            },
            &runtime_config,
        )
        .unwrap_or_else(|error| panic!("load {label} worker model: {error:?}"));
    engine
        .embed_batch(&loaded, TokenBatch { items })
        .unwrap_or_else(|error| panic!("embed with {label} worker: {error:?}"))
}

fn assert_worker_crash_is_quarantined(label: &str, worker_bin: PathBuf) {
    let mut config = WorkerHostConfig::new(worker_bin, temp_runtime_dir(label));
    config.worker_id = format!("{label}-crash-{}", std::process::id());
    config
        .extra_args
        .push("--test-abort-on-request".to_string());
    config.crash_budget = CrashBudget {
        max_crashes: 0,
        window: Duration::from_secs(60),
    };
    let mut engine = WorkerEngine::new(config).expect("worker host should initialize");
    let mut runtime_config = RuntimeConfig::default();
    runtime_config.values.insert(
        "artifact_path".to_string(),
        "/tmp/synapse-crash-probe".to_string(),
    );
    let first = engine.load(
        &ValidatedArtifact {
            digest: String::new(),
            format: "safetensors".to_string(),
        },
        &runtime_config,
    );
    eprintln!("{label} first load error: {:?}", first.as_ref().err());
    assert!(
        first.is_err(),
        "{label} worker load should fail after forced abort"
    );
    let second = engine.load(
        &ValidatedArtifact {
            digest: String::new(),
            format: "safetensors".to_string(),
        },
        &runtime_config,
    );
    assert!(
        second
            .err()
            .map(|error| error.message.contains("quarantined"))
            .unwrap_or(false),
        "{label} worker should quarantine the crashed model key"
    );
}

fn golden_file() -> GoldenFile {
    serde_json::from_str(include_str!(
        "../../synapse-engine-ort/tests/fixtures/minilm_golden.json"
    ))
    .expect("golden fixture should parse")
}

fn golden_items() -> Vec<Vec<f32>> {
    golden_file()
        .items
        .into_iter()
        .map(|item| item.vector)
        .collect()
}

fn mean_pairwise_cosine(actual: &[Vec<f32>], expected: &[Vec<f32>]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(left, right)| cosine(left, right))
        .sum::<f32>()
        / actual.len() as f32
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());
    let (dot, left_norm, right_norm) = left.iter().zip(right).fold(
        (0.0_f32, 0.0_f32, 0.0_f32),
        |(dot, left_norm, right_norm), (left, right)| {
            (
                dot + left * right,
                left_norm + left * left,
                right_norm + right * right,
            )
        },
    );
    dot / (left_norm.sqrt() * right_norm.sqrt()).max(f32::EPSILON)
}

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var_os(key)
        .map(PathBuf::from)
        .filter(|path| path.exists())
}

fn default_ane_fixture() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let path =
        Path::new(&home).join("bench-tools/ane-spike/models/all-MiniLM-L6-v2-seq256.mlmodelc");
    path.exists().then_some(path)
}

fn temp_runtime_dir(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "synapse-{label}-worker-test-{}",
        std::process::id()
    ))
}

fn skip(reason: &str) {
    eprintln!("skip: {reason}");
}
