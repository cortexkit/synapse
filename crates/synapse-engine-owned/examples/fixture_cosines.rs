//! Per-case cosine of owned-Metal embeddings against a committed fp32 parity fixture.
//!
//! Usage:
//! `fixture_cosines FAMILY MODEL_DIR FIXTURE_JSON CACHE_DIR DTYPE EXECUTION [CASE_ID]`
//!
//! - `FAMILY` is `gte-modernbert` or `qwen3`; `DTYPE` is `f16` or `f32`;
//!   `EXECUTION` is `explicit` or `lazy`.
//! - `FIXTURE_JSON` is a `bench/parity/fixtures/...ref-v1...json` embedding fixture:
//!   each case carries its exact `input_ids` and the fp32 reference `output`.
//! - Without `CASE_ID`, every case runs in fixture order in batches of eight, the
//!   way the real-weight hardware parity test feeds them, and one JSON line per
//!   case reports its token count and cosine against the reference.
//! - With `CASE_ID`, only that case runs, alone. Pair it with `EXECUTION=lazy`
//!   and `SYNAPSE_MODERNBERT_DUMP_DIR` to dump ModernBERT's per-stage tensors.
//!
//! The engine is loaded with the settings of the owned-metal catalog profiles:
//! `max_tokens` 8,192, `attention_units` 8,192 squared.

use std::path::PathBuf;

use serde_json::{json, Value};
use synapse_core::{EmbedEngine, RuntimeConfig, TokenBatch, ValidatedArtifact};
use synapse_engine_owned::{ModelFamily, OwnedDType, OwnedMetalEmbedEngine};

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    assert!(
        (7..=8).contains(&args.len()),
        "usage: fixture_cosines FAMILY MODEL_DIR FIXTURE_JSON CACHE_DIR DTYPE EXECUTION [CASE_ID]"
    );
    let family = ModelFamily::parse(&args[1]).expect("FAMILY");
    let model_dir = PathBuf::from(&args[2]);
    let fixture: Value =
        serde_json::from_slice(&std::fs::read(&args[3]).expect("read FIXTURE_JSON"))
            .expect("parse FIXTURE_JSON");
    let cache_dir = PathBuf::from(&args[4]);
    let dtype = OwnedDType::parse(&args[5]).expect("DTYPE");
    let execution = args[6].clone();
    let only = args.get(7).cloned();

    let mut config = RuntimeConfig::default();
    for (key, value) in [
        (
            "model_path",
            model_dir.join("model.safetensors").display().to_string(),
        ),
        ("package_cache_root", cache_dir.display().to_string()),
        ("execution", execution),
        ("max_tokens", "8192".to_string()),
        ("attention_units", (8192 * 8192).to_string()),
    ] {
        config.values.insert(key.to_string(), value);
    }
    // Load through the catalog profile, as the hardware parity test and catalog
    // lanes do: architecture parameters come from bench/parity/models.json. A
    // profile pins its dtype, so FIXTURE_COSINES_NO_PROFILE=1 loads from the
    // checkpoint's config.json instead, for a dtype the profile does not declare.
    let slug = fixture["model"].as_str().expect("fixture model");
    if std::env::var_os("FIXTURE_COSINES_NO_PROFILE").is_none() {
        config
            .values
            .insert("profile".to_string(), format!("{slug}.owned-metal"));
        config.values.insert(
            "operation".to_string(),
            fixture["operation"]
                .as_str()
                .expect("fixture operation")
                .to_string(),
        );
    }
    let mut engine = OwnedMetalEmbedEngine::new(family, dtype);
    let loaded = engine
        .load(
            &ValidatedArtifact {
                digest: "sha256:local-fixture-cosines".to_string(),
                format: "safetensors-package".to_string(),
            },
            &config,
        )
        .expect("load owned-metal model");

    let cases = fixture["cases"]
        .as_array()
        .expect("fixture cases")
        .iter()
        .filter(|case| only.as_deref().is_none_or(|id| case["id"] == id))
        .collect::<Vec<_>>();
    assert!(!cases.is_empty(), "no fixture case selected");
    for chunk in cases.chunks(if only.is_some() { 1 } else { 8 }) {
        let items = chunk
            .iter()
            .map(|case| {
                case["input_ids"]
                    .as_array()
                    .expect("input_ids")
                    .iter()
                    .map(|id| u32::try_from(id.as_u64().expect("token id")).expect("u32 id"))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let vectors = engine
            .embed_batch(&loaded, TokenBatch { items })
            .expect("embed fixture cases");
        for (case, vector) in chunk.iter().zip(vectors) {
            let reference = case["output"]
                .as_array()
                .expect("reference output")
                .iter()
                .map(|value| value.as_f64().expect("reference value"))
                .collect::<Vec<_>>();
            assert_eq!(reference.len(), vector.len(), "dimension mismatch");
            let dot = vector
                .iter()
                .zip(&reference)
                .map(|(&a, &b)| f64::from(a) * b)
                .sum::<f64>();
            let norm = |values: &mut dyn Iterator<Item = f64>| {
                values.map(|value| value * value).sum::<f64>().sqrt()
            };
            let cosine = dot
                / (norm(&mut vector.iter().map(|&value| f64::from(value)))
                    * norm(&mut reference.iter().copied()));
            println!(
                "{}",
                json!({
                    "id": case["id"],
                    "category": case["category"],
                    "tokens": case["input_ids"].as_array().map_or(0, Vec::len),
                    "cosine": cosine,
                })
            );
        }
    }
}
