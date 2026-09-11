//! Measures one pre-tokenized GTE ModernBERT row through the owned Metal f16 engine.
//!
//! The input is shared with the Core ML long-row probe. Timing begins immediately
//! before `embed_batch` and ends after normalized vectors return to the caller.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use synapse_core::{EmbedEngine, RuntimeConfig, TokenBatch, ValidatedArtifact};
use synapse_engine_owned::{
    engine_identity, ModelFamily, OwnedDType, OwnedMetalEmbedEngine, BUCKET_POLICY_VERSION,
};

const MAX_TOKENS: usize = 8192;
const ATTENTION_UNITS: usize = 67_108_864;

#[derive(Deserialize)]
struct InputCase {
    id: String,
    real_tokens: usize,
    input_ids: Vec<u32>,
}

#[derive(Serialize)]
struct ProbeRun {
    case_id: String,
    lane: &'static str,
    family: &'static str,
    dtype: &'static str,
    bucket_policy_version: u32,
    engine_identity: synapse_core::EngineIdentity,
    real_tokens: usize,
    cold_load_s: f64,
    first_use_engine_wall_s: f64,
    warmup_engine_wall_s: f64,
    measured_engine_wall_s: Vec<f64>,
    measurement_window_unix_s: [f64; 2],
    vector_sha256: String,
    repeated_determinism_max_abs: f32,
    vector: Vec<f32>,
}

fn unix_time_s() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before Unix epoch")
        .as_secs_f64()
}

fn vector_sha256(vector: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(vector.len() * 4);
    for value in vector {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    format!("{:x}", Sha256::digest(bytes))
}

fn max_abs(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len(), "vector dimensions changed");
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max)
}

fn parse_positive<T>(text: &str, label: &str) -> T
where
    T: std::str::FromStr + PartialOrd + Default,
    T::Err: std::fmt::Debug,
{
    let value = text
        .parse::<T>()
        .unwrap_or_else(|error| panic!("{label} is invalid: {error:?}"));
    assert!(value > T::default(), "{label} must be positive");
    value
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    assert_eq!(
        args.len(),
        8,
        "usage: long_row_probe MODEL_DIR CASE.json OUT.json CACHE_DIR MIN_REPETITIONS MIN_DURATION_S MAX_REPETITIONS"
    );
    let model_dir = PathBuf::from(&args[1]);
    let case_path = PathBuf::from(&args[2]);
    let output_path = PathBuf::from(&args[3]);
    let cache_dir = PathBuf::from(&args[4]);
    let minimum_repetitions = parse_positive::<usize>(&args[5], "MIN_REPETITIONS");
    let minimum_duration_s = parse_positive::<f64>(&args[6], "MIN_DURATION_S");
    let maximum_repetitions = parse_positive::<usize>(&args[7], "MAX_REPETITIONS");
    assert!(
        maximum_repetitions >= minimum_repetitions,
        "MAX_REPETITIONS must cover MIN_REPETITIONS"
    );

    let case: InputCase = serde_json::from_slice(&fs::read(&case_path).expect("read input case"))
        .expect("parse case");
    assert_eq!(
        case.input_ids.len(),
        case.real_tokens,
        "case real-token count does not match input IDs"
    );
    assert!(!case.input_ids.is_empty(), "case input IDs are empty");
    assert!(
        case.real_tokens <= MAX_TOKENS,
        "case exceeds the full-context limit"
    );

    let family = ModelFamily::GteModernBert;
    let dtype = OwnedDType::F16;
    let weights_path = model_dir.join("model.safetensors");
    let mut config = RuntimeConfig::default();
    config
        .values
        .insert("model_path".to_string(), weights_path.display().to_string());
    config.values.insert(
        "package_cache_root".to_string(),
        cache_dir.display().to_string(),
    );
    config
        .values
        .insert("execution".to_string(), "explicit".to_string());
    config
        .values
        .insert("max_tokens".to_string(), MAX_TOKENS.to_string());
    config
        .values
        .insert("attention_units".to_string(), ATTENTION_UNITS.to_string());

    let mut engine = OwnedMetalEmbedEngine::new(family, dtype);
    let load_started = Instant::now();
    let loaded = engine
        .load(
            &ValidatedArtifact {
                digest: "sha256:local-ane-vs-metal-long-row-probe".to_string(),
                format: "safetensors-package".to_string(),
            },
            &config,
        )
        .expect("load GTE ModernBERT");
    let cold_load_s = load_started.elapsed().as_secs_f64();
    let batch = TokenBatch {
        items: vec![case.input_ids],
    };

    let first_batch = batch.clone();
    let first_started = Instant::now();
    let first = engine
        .embed_batch(&loaded, first_batch)
        .expect("first-use embedding")
        .into_iter()
        .next()
        .expect("first-use vector");
    let first_use_engine_wall_s = first_started.elapsed().as_secs_f64();

    let warmup_batch = batch.clone();
    let warmup_started = Instant::now();
    let warmup = engine
        .embed_batch(&loaded, warmup_batch)
        .expect("warmup embedding")
        .into_iter()
        .next()
        .expect("warmup vector");
    let warmup_engine_wall_s = warmup_started.elapsed().as_secs_f64();
    assert_eq!(first, warmup, "first-use and warmup vectors differ");

    let measurement_start_unix_s = unix_time_s();
    let measurement_started = Instant::now();
    let mut measured_engine_wall_s = Vec::new();
    let mut determinism_max_abs = 0.0_f32;
    while measured_engine_wall_s.len() < maximum_repetitions {
        let measured_batch = batch.clone();
        let started = Instant::now();
        let vector = engine
            .embed_batch(&loaded, measured_batch)
            .expect("measured embedding")
            .into_iter()
            .next()
            .expect("measured vector");
        measured_engine_wall_s.push(started.elapsed().as_secs_f64());
        determinism_max_abs = determinism_max_abs.max(max_abs(&first, &vector));
        if measured_engine_wall_s.len() >= minimum_repetitions
            && measurement_started.elapsed() >= Duration::from_secs_f64(minimum_duration_s)
        {
            break;
        }
    }
    let measurement_end_unix_s = unix_time_s();
    assert!(
        measurement_started.elapsed() >= Duration::from_secs_f64(minimum_duration_s),
        "MAX_REPETITIONS was reached before MIN_DURATION_S"
    );
    assert_eq!(
        determinism_max_abs, 0.0,
        "measured vectors were not byte-identical"
    );

    let report = ProbeRun {
        case_id: case.id,
        lane: "production-owned-metal",
        family: family.as_str(),
        dtype: dtype.as_str(),
        bucket_policy_version: BUCKET_POLICY_VERSION,
        engine_identity: engine_identity(family, dtype),
        real_tokens: batch.items[0].len(),
        cold_load_s,
        first_use_engine_wall_s,
        warmup_engine_wall_s,
        measured_engine_wall_s,
        measurement_window_unix_s: [measurement_start_unix_s, measurement_end_unix_s],
        vector_sha256: vector_sha256(&first),
        repeated_determinism_max_abs: determinism_max_abs,
        vector: first,
    };
    if let Some(parent) = Path::new(&output_path).parent() {
        fs::create_dir_all(parent).expect("create output directory");
    }
    fs::write(output_path, serde_json::to_vec_pretty(&report).unwrap()).expect("write probe JSON");
}
