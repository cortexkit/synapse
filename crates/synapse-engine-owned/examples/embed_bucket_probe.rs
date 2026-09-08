//! Serial, bounded owned-Metal bucket-policy probe with representative tokenized rows.
//!
//! Build this example at the baseline and candidate revisions, preserve each binary,
//! then run each with `SYNAPSE_EMBED_PROFILE=1` and separate empty package caches:
//! `embed_bucket_probe MODEL_DIR OUT.json CACHE_DIR [REPEATS] [CASE_CSV] 2> profile.stderr`.
//! Output is compact JSON containing exact token IDs, model/tokenizer digests, first-use
//! and repeated warm timings, vectors, and a digest for every repeated result.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Serialize;
use sha2::{Digest, Sha256};
use synapse_core::{EmbedEngine, RuntimeConfig, TokenBatch, ValidatedArtifact};
use synapse_engine_owned::{
    engine_identity, ModelFamily, OwnedDType, OwnedMetalEmbedEngine, BUCKET_POLICY_VERSION,
};
use tokenizers::Tokenizer;

const MAX_TOKENS: usize = 8192;
const ATTENTION_UNITS: usize = 67_108_864;

#[derive(Serialize)]
struct ProbeRun {
    metadata: ProbeMetadata,
    cold_load_s: f64,
    cases: Vec<CaseRun>,
}

#[derive(Serialize)]
struct ProbeMetadata {
    probe_version: &'static str,
    family: &'static str,
    dtype: &'static str,
    model_config_sha256: String,
    model_weights_sha256: String,
    tokenizer_sha256: String,
    max_tokens: usize,
    attention_units: usize,
    execution: &'static str,
    bucket_policy_version: u32,
    engine_identity: synapse_core::EngineIdentity,
    first_use_runs_per_case: usize,
    measured_repeats_per_case: usize,
    case_order: Vec<&'static str>,
}

#[derive(Clone, Serialize)]
struct InputRow {
    id: &'static str,
    seed_text: &'static str,
    target_tokens: usize,
    input_ids: Vec<u32>,
}

#[derive(Serialize)]
struct CaseRun {
    name: &'static str,
    rows: Vec<InputRow>,
    real_tokens: usize,
    first_use_engine_wall_s: f64,
    warm_engine_wall_s: Vec<f64>,
    repeat_vector_sha256: Vec<Vec<String>>,
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
        .unwrap_or(3);
    assert!(repeats > 0, "REPEATS must be positive");

    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path).expect("load tokenizer.json");
    let requested_cases = args
        .get(5)
        .map(|value| value.split(',').collect::<Vec<_>>());
    let cases = probe_cases(&tokenizer)
        .into_iter()
        .filter(|(name, _)| {
            requested_cases
                .as_ref()
                .is_none_or(|requested| requested.contains(name))
        })
        .collect::<Vec<_>>();
    assert!(!cases.is_empty(), "CASE_CSV selected no known cases");

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
    for (name, rows) in cases {
        let batch = token_batch(&rows);
        let first_started = Instant::now();
        let first_vectors = engine
            .embed_batch(&loaded, batch.clone())
            .unwrap_or_else(|error| panic!("first use {name}: {error:?}"));
        let first_use_engine_wall_s = first_started.elapsed().as_secs_f64();
        let mut walls = Vec::with_capacity(repeats);
        let mut repeat_vector_sha256 = Vec::with_capacity(repeats);
        for pass in 0..repeats {
            let started = Instant::now();
            let output = engine
                .embed_batch(&loaded, batch.clone())
                .unwrap_or_else(|error| panic!("measure {name} pass {pass}: {error:?}"));
            walls.push(started.elapsed().as_secs_f64());
            assert_eq!(
                output, first_vectors,
                "{name} pass {pass} changed vectors or output order"
            );
            repeat_vector_sha256.push(output.iter().map(|row| vector_sha256(row)).collect());
        }
        results.push(CaseRun {
            name,
            real_tokens: rows.iter().map(|row| row.input_ids.len()).sum(),
            rows,
            first_use_engine_wall_s,
            warm_engine_wall_s: walls,
            repeat_vector_sha256,
            vectors: first_vectors,
        });
    }

    let case_order = results.iter().map(|case| case.name).collect();
    let output = ProbeRun {
        metadata: ProbeMetadata {
            probe_version: "representative-text-v1",
            family: family.as_str(),
            dtype: dtype.as_str(),
            model_config_sha256: file_sha256(&model_dir.join("config.json")),
            model_weights_sha256: file_sha256(&weights_path),
            tokenizer_sha256: file_sha256(&tokenizer_path),
            max_tokens: MAX_TOKENS,
            attention_units: ATTENTION_UNITS,
            execution: "explicit",
            bucket_policy_version: BUCKET_POLICY_VERSION,
            engine_identity: engine_identity(family, dtype),
            first_use_runs_per_case: 1,
            measured_repeats_per_case: repeats,
            case_order,
        },
        cold_load_s,
        cases: results,
    };
    fs::write(output_path, serde_json::to_vec(&output).unwrap()).expect("write probe JSON");
}

fn probe_cases(tokenizer: &Tokenizer) -> Vec<(&'static str, Vec<InputRow>)> {
    let seeds = [
        ("climate", "Coastal climate adaptation combines wetlands, resilient transit, and neighborhood planning."),
        ("database", "A database transaction remains atomic when concurrent writers update related account records."),
        ("biology", "Protein folding depends on amino acid interactions, solvent conditions, and cellular machinery."),
        ("history", "Archive letters describe trade routes, local elections, and daily life across several decades."),
        ("music", "The chamber ensemble balances a lyrical violin melody against quiet rhythmic variations."),
        ("astronomy", "Astronomers compare repeated spectra to estimate a distant planet's atmosphere and orbit."),
        ("cooking", "Slow roasting vegetables develops sweetness while herbs and citrus preserve a bright finish."),
        ("software", "The release pipeline validates schemas, runs deterministic tests, and signs immutable artifacts."),
        ("education", "Students revise explanations after comparing evidence from several carefully controlled experiments."),
        ("transport", "A regional rail timetable coordinates transfers while leaving recovery time for disruptions."),
    ];
    let single_targets = [511, 513, 603, 1203];
    let mut cases = single_targets
        .iter()
        .enumerate()
        .map(|(index, &target)| {
            let (id, text) = seeds[index];
            (
                match target {
                    511 => "text-singleton-511",
                    513 => "text-singleton-513",
                    603 => "text-singleton-603",
                    1203 => "text-singleton-1203",
                    _ => unreachable!(),
                },
                vec![tokenized_row(tokenizer, id, text, target)],
            )
        })
        .collect::<Vec<_>>();
    let mixed_targets = [127, 255, 383, 511, 603, 1193];
    let mixed = mixed_targets
        .iter()
        .enumerate()
        .map(|(index, &target)| {
            let (id, text) = seeds[index + 4];
            tokenized_row(tokenizer, id, text, target)
        })
        .collect();
    cases.push(("text-mixed-budget-3072", mixed));
    cases
}

fn tokenized_row(
    tokenizer: &Tokenizer,
    id: &'static str,
    seed_text: &'static str,
    target_tokens: usize,
) -> InputRow {
    let mut repetitions = 1;
    loop {
        let text = std::iter::repeat_n(seed_text, repetitions)
            .collect::<Vec<_>>()
            .join(" ");
        let encoding = tokenizer.encode(text, true).expect("tokenize seed text");
        if encoding.len() >= target_tokens {
            let mut input_ids = encoding.get_ids().to_vec();
            input_ids.truncate(target_tokens);
            return InputRow {
                id,
                seed_text,
                target_tokens,
                input_ids,
            };
        }
        repetitions *= 2;
    }
}

fn token_batch(rows: &[InputRow]) -> TokenBatch {
    TokenBatch {
        items: rows.iter().map(|row| row.input_ids.clone()).collect(),
    }
}

fn file_sha256(path: &Path) -> String {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    format!("{:x}", Sha256::digest(bytes))
}

fn vector_sha256(vector: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in vector {
        digest.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}
