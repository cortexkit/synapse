//! Embeds a module probe corpus through the owned-Metal engine, the way the
//! module's `probe.start` does for a preloaded owned-metal embedding lane, and
//! writes the vectors so their neighbour ranks can be compared offline.
//!
//! Usage:
//! `probe_corpus_vectors FAMILY MODEL_DIR CORPUS_JSON CACHE_DIR DTYPE MAX_TOKENS OUT_JSON`
//!
//! - `CORPUS_JSON` is a built-in probe corpus such as
//!   `crates/synapse-module/src/fixtures/probe_corpus_gte_modernbert_ort_fp32.json`.
//! - Every item's text is tokenized with the module's `SanitizedTokenizer` capped
//!   at `MAX_TOKENS`, and all items go to the engine in one `embed_batch` call, as
//!   the probe's `execute_embedding` sends them. The engine keeps its default
//!   attention budget, as a `model.load` that does not set one does.
//! - `OUT_JSON` holds `{"ids", "vectors", "tokens"}`, each in corpus order.

use std::path::PathBuf;

use serde_json::{json, Value};
use synapse_core::tokenizer::{SanitizedTokenizer, TokenizerConfig};
use synapse_core::{EmbedEngine, RuntimeConfig, ValidatedArtifact};
use synapse_engine_owned::{ModelFamily, OwnedDType, OwnedMetalEmbedEngine};

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    assert_eq!(
        args.len(),
        8,
        "usage: probe_corpus_vectors FAMILY MODEL_DIR CORPUS_JSON CACHE_DIR DTYPE MAX_TOKENS OUT_JSON"
    );
    let family = ModelFamily::parse(&args[1]).expect("FAMILY");
    let model_dir = PathBuf::from(&args[2]);
    let corpus: Value = serde_json::from_slice(&std::fs::read(&args[3]).expect("read CORPUS_JSON"))
        .expect("parse CORPUS_JSON");
    let cache_dir = PathBuf::from(&args[4]);
    let dtype = OwnedDType::parse(&args[5]).expect("DTYPE");
    let max_tokens = args[6].parse::<usize>().expect("MAX_TOKENS");

    let tokenizer = SanitizedTokenizer::from_file(
        model_dir.join("tokenizer.json"),
        TokenizerConfig { max_tokens },
    )
    .expect("load tokenizer.json");
    let mut config = RuntimeConfig::default();
    for (key, value) in [
        (
            "model_path",
            model_dir.join("model.safetensors").display().to_string(),
        ),
        ("package_cache_root", cache_dir.display().to_string()),
        ("execution", "explicit".to_string()),
        ("max_tokens", max_tokens.to_string()),
    ] {
        config.values.insert(key.to_string(), value);
    }
    let mut engine = OwnedMetalEmbedEngine::new(family, dtype);
    let loaded = engine
        .load(
            &ValidatedArtifact {
                digest: "sha256:local-probe-corpus-vectors".to_string(),
                format: "safetensors-package".to_string(),
            },
            &config,
        )
        .expect("load owned-metal model");

    let items = corpus["items"].as_array().expect("corpus items");
    let tokenized = tokenizer
        .tokenize_batch(
            items
                .iter()
                .map(|item| item["text"].as_str().expect("text")),
        )
        .expect("tokenize corpus")
        .batch;
    let vectors = engine
        .embed_batch(&loaded, tokenized.clone())
        .expect("embed corpus");
    std::fs::write(
        &args[7],
        serde_json::to_vec(&json!({
            "ids": items.iter().map(|item| item["id"].clone()).collect::<Vec<_>>(),
            "vectors": vectors,
            "tokens": tokenized.items,
        }))
        .expect("serialize vectors"),
    )
    .expect("write OUT_JSON");
}
