//! Serial, bounded owned-Metal bucket-policy probe.
//!
//! Build this example at the baseline and candidate revisions, preserve each binary,
//! then run each with `SYNAPSE_EMBED_PROFILE=1` and separate empty package caches:
//! `embed_bucket_probe MODEL_DIR OUT.json CACHE_DIR [REPEATS] [CASE_CSV] 2> profile.stderr`.
//! The JSON records exact inputs, configuration, cold load, warm timings, and vectors;
//! stderr records the shapes selected by the engine itself.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Serialize;
use sha2::{Digest, Sha256};
use synapse_core::{EmbedEngine, RuntimeConfig, TokenBatch, ValidatedArtifact};
use synapse_engine_owned::{
    engine_identity, ModelFamily, OwnedDType, OwnedMetalEmbedEngine, BUCKET_POLICY_VERSION,
};

const MAX_TOKENS: usize = 8192;
const ATTENTION_UNITS: usize = 67_108_864;
const INPUT_TOKEN_ID: u32 = 1;

#[derive(Serialize)]
struct ProbeRun {
    metadata: ProbeMetadata,
    cold_load_s: f64,
    cases: Vec<CaseRun>,
}

#[derive(Serialize)]
struct ProbeMetadata {
    family: &'static str,
    dtype: &'static str,
    model_config_sha256: String,
    max_tokens: usize,
    attention_units: usize,
    execution: &'static str,
    bucket_policy_version: u32,
    engine_identity: synapse_core::EngineIdentity,
    input_token_id: u32,
    warmup_runs_per_case: usize,
    measured_repeats_per_case: usize,
    case_order: Vec<&'static str>,
}

#[derive(Serialize)]
struct CaseRun {
    name: &'static str,
    lengths: Vec<usize>,
    real_tokens: usize,
    warm_engine_wall_s: Vec<f64>,
    vectors: Vec<Vec<f32>>,
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    assert!(
        (4..=6).contains(&args.len()),
        "usage: embed_bucket_probe MODEL_DIR OUT.json CACHE_DIR [REPEATS] [CASE_CSV]"
    );
    let model_dir = PathBuf::from(&args[1]);
    let output_path = PathBuf::from(&args[2]);
    let cache_dir = PathBuf::from(&args[3]);
    let repeats = args
        .get(4)
        .map(|value| value.parse::<usize>().expect("REPEATS must be an integer"))
        .unwrap_or(2);
    assert!(repeats > 0, "REPEATS must be positive");

    let family = ModelFamily::GteModernBert;
    let dtype = OwnedDType::F16;
    let mut config = RuntimeConfig::default();
    config.values.insert(
        "model_path".to_string(),
        model_dir.join("model.safetensors").display().to_string(),
    );
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

    let requested_cases = args
        .get(5)
        .map(|value| value.split(',').collect::<Vec<_>>());
    let cases = probe_cases()
        .into_iter()
        .filter(|(name, _)| {
            requested_cases
                .as_ref()
                .is_none_or(|requested| requested.contains(name))
        })
        .collect::<Vec<_>>();
    assert!(!cases.is_empty(), "CASE_CSV selected no known cases");
    let mut engine = OwnedMetalEmbedEngine::new(family, dtype);
    let cold_started = Instant::now();
    let loaded = engine
        .load(
            &ValidatedArtifact {
                digest: "sha256:local-bounded-bucket-probe".to_string(),
                format: "safetensors-package".to_string(),
            },
            &config,
        )
        .expect("load GTE ModernBERT");
    let cold_load_s = cold_started.elapsed().as_secs_f64();

    let mut results = Vec::with_capacity(cases.len());
    for (name, lengths) in cases {
        let batch = token_batch(&lengths);
        engine
            .embed_batch(&loaded, batch.clone())
            .unwrap_or_else(|error| panic!("warm {name}: {error:?}"));
        let mut walls = Vec::with_capacity(repeats);
        let mut vectors = Vec::new();
        for pass in 0..repeats {
            let started = Instant::now();
            let output = engine
                .embed_batch(&loaded, batch.clone())
                .unwrap_or_else(|error| panic!("measure {name} pass {pass}: {error:?}"));
            walls.push(started.elapsed().as_secs_f64());
            if pass == 0 {
                vectors = output;
            }
        }
        results.push(CaseRun {
            name,
            real_tokens: lengths.iter().sum(),
            lengths,
            warm_engine_wall_s: walls,
            vectors,
        });
    }

    let case_order = results.iter().map(|case| case.name).collect();
    let output = ProbeRun {
        metadata: ProbeMetadata {
            family: family.as_str(),
            dtype: dtype.as_str(),
            model_config_sha256: sha256(&model_dir.join("config.json")),
            max_tokens: MAX_TOKENS,
            attention_units: ATTENTION_UNITS,
            execution: "explicit",
            bucket_policy_version: BUCKET_POLICY_VERSION,
            engine_identity: engine_identity(family, dtype),
            input_token_id: INPUT_TOKEN_ID,
            warmup_runs_per_case: 1,
            measured_repeats_per_case: repeats,
            case_order,
        },
        cold_load_s,
        cases: results,
    };
    fs::write(output_path, serde_json::to_vec_pretty(&output).unwrap()).expect("write probe JSON");
}

fn probe_cases() -> Vec<(&'static str, Vec<usize>)> {
    vec![
        ("singleton-511", vec![511]),
        ("singleton-513", vec![513]),
        ("singleton-603", vec![603]),
        ("singleton-1203", vec![1203]),
        ("singleton-2600", vec![2600]),
        ("singleton-4096", vec![4096]),
        ("filled-131", vec![131; 8]),
        ("filled-320", vec![320; 8]),
        ("filled-448", vec![448; 8]),
        ("filled-603", vec![603; 8]),
        ("filled-1203", vec![1203; 8]),
        ("filled-2600", vec![2600; 7]),
        ("filled-4096", vec![4096; 4]),
        ("mixed", vec![511, 513, 603, 1203, 2600, 4096]),
    ]
}

fn token_batch(lengths: &[usize]) -> TokenBatch {
    TokenBatch {
        items: lengths
            .iter()
            .map(|&length| vec![INPUT_TOKEN_ID; length])
            .collect(),
    }
}

fn sha256(path: &Path) -> String {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    format!("{:x}", Sha256::digest(bytes))
}
