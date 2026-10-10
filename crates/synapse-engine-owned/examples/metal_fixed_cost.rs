//! Replays the engram code chunks exported by AFT (the agent file-tools client,
//! which embeds code chunks through Synapse, 64 rows per call and two calls in
//! flight) through the owned Metal engine, in the module's own engine-call
//! shape, to attribute where serving time goes.
//!
//! Usage:
//! `metal_fixed_cost FAMILY MODEL_DIR INPUT_JSONL OUT_JSON CACHE_DIR [REPLAYS] [DTYPE]`
//!
//! - `FAMILY` is `gte-modernbert` or `qwen3`. `DTYPE` defaults to `f16`, the
//!   dtype of both families' owned-metal embedding catalog profiles; `f32` runs
//!   the full-precision graph, for example as a reference for f16 vectors.
//! - `INPUT_JSONL` is the engram export; `INPUT_JSONL.meta.json` beside it holds
//!   the 127-call batch plan AFT used.
//! - Each 64-row call is tokenized with the module's `SanitizedTokenizer`, then
//!   split exactly as the module's bulk path splits it: rows sorted by length,
//!   at most 8 rows and 3,072 tokens per engine call. Every engine call goes to
//!   `OwnedMetalEmbedEngine::embed_batch`, serially, like the module's engine
//!   mutex makes them.
//! - One unrecorded warm-up replay compiles every shape first; then `REPLAYS`
//!   (default 1) measured replays follow. Measured replays must reproduce the
//!   warm-up vectors bit for bit.
//!
//! Run with `SYNAPSE_EMBED_PROFILE=1` to get the engine's per-pass stage lines on
//! stderr. Add `SYNAPSE_EMBED_PROFILE_GPU=1` to split each pass's synchronous run
//! into CPU encode time and GPU time. Before each engine call this probe prints a
//! `[metal-fixed-cost] call ...` marker on stderr, so the profile lines that
//! follow belong to that call.
//!
//! Writes `OUT_JSON` (timings, per-row vector digests) and `OUT_JSON.vectors.f32`
//! (every measured vector, little-endian f32, in input order).

use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use synapse_core::tokenizer::{SanitizedTokenizer, TokenizerConfig};
use synapse_core::{EmbedEngine, RuntimeConfig, TokenBatch, ValidatedArtifact};
use synapse_engine_owned::{
    engine_identity, ModelFamily, OwnedDType, OwnedMetalEmbedEngine, BUCKET_POLICY_VERSION,
};

/// The owned-metal catalog profiles serve up to 8,192 tokens with an attention
/// budget of 8,192 squared (see the module's `owned_config`).
const MAX_TOKENS: usize = 8192;
const ATTENTION_UNITS: usize = 8192 * 8192;
/// The module's bulk path limits (`MAX_ENGINE_BATCH_ITEMS` and
/// `DEFAULT_ENGINE_BATCH_TOKEN_BUDGET` in `synapse-module`).
const ENGINE_CALL_ITEMS: usize = 8;
const ENGINE_CALL_TOKENS: usize = 3072;

#[derive(Deserialize)]
struct Chunk {
    seq: usize,
    text: String,
}

#[derive(Clone, Deserialize)]
struct Batch {
    seq: usize,
    first_chunk_seq: usize,
    chunk_count: usize,
}

#[derive(Deserialize)]
struct Metadata {
    batches: Vec<Batch>,
}

#[derive(Serialize)]
struct Output {
    family: &'static str,
    dtype: &'static str,
    engine_identity: synapse_core::EngineIdentity,
    bucket_policy_version: u32,
    max_tokens: usize,
    attention_units: usize,
    input_sha256: String,
    meta_sha256: String,
    rows: usize,
    calls: usize,
    cold_load_s: f64,
    warmup_wall_s: f64,
    replays: Vec<Replay>,
    vectors_sha256: String,
    row_sha256: Vec<String>,
}

#[derive(Serialize)]
struct Replay {
    wall_s: f64,
    tokenize_ms: f64,
    engine_ms: f64,
    calls: Vec<CallRecord>,
}

#[derive(Serialize)]
struct CallRecord {
    batch_seq: usize,
    rows: usize,
    tokens: usize,
    tokenize_ms: f64,
    engine_calls: Vec<EngineCallRecord>,
}

#[derive(Serialize)]
struct EngineCallRecord {
    rows: usize,
    tokens: usize,
    max_len: usize,
    ms: f64,
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    assert!(
        (6..=8).contains(&args.len()),
        "usage: metal_fixed_cost FAMILY MODEL_DIR INPUT_JSONL OUT_JSON CACHE_DIR [REPLAYS] [DTYPE]"
    );
    let family = ModelFamily::parse(&args[1]).expect("FAMILY must be gte-modernbert or qwen3");
    assert!(
        matches!(family, ModelFamily::GteModernBert | ModelFamily::Qwen3),
        "FAMILY must be gte-modernbert or qwen3"
    );
    let model_dir = PathBuf::from(&args[2]);
    let input_path = PathBuf::from(&args[3]);
    let output_path = PathBuf::from(&args[4]);
    let cache_dir = PathBuf::from(&args[5]);
    let replays = args
        .get(6)
        .map(|value| value.parse::<usize>().expect("REPLAYS must be an integer"))
        .unwrap_or(1);
    assert!(replays > 0, "REPLAYS must be positive");

    let mut meta_path = input_path.clone().into_os_string();
    meta_path.push(".meta.json");
    let meta_path = PathBuf::from(meta_path);
    let chunks = fs::read_to_string(&input_path)
        .expect("read INPUT_JSONL")
        .lines()
        .enumerate()
        .map(|(index, line)| {
            let chunk: Chunk = serde_json::from_str(line).expect("parse input row");
            assert_eq!(chunk.seq, index, "input seq must be 0..N-1 in file order");
            chunk.text
        })
        .collect::<Vec<_>>();
    let meta: Metadata =
        serde_json::from_slice(&fs::read(&meta_path).expect("read INPUT_JSONL.meta.json"))
            .expect("parse batch metadata");
    let mut covered = BTreeSet::new();
    for batch in &meta.batches {
        assert!(
            (1..=64).contains(&batch.chunk_count),
            "batch outside 1..=64"
        );
        for row in batch.first_chunk_seq..batch.first_chunk_seq + batch.chunk_count {
            assert!(row < chunks.len() && covered.insert(row), "bad batch plan");
        }
    }
    assert_eq!(
        covered.len(),
        chunks.len(),
        "batch plan must cover every row"
    );

    let dtype = args.get(7).map_or(OwnedDType::F16, |value| {
        OwnedDType::parse(value).expect("DTYPE")
    });
    let tokenizer = SanitizedTokenizer::from_file(
        model_dir.join("tokenizer.json"),
        TokenizerConfig {
            max_tokens: MAX_TOKENS,
        },
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
        ("max_tokens", MAX_TOKENS.to_string()),
        ("attention_units", ATTENTION_UNITS.to_string()),
    ] {
        config.values.insert(key.to_string(), value);
    }
    let mut engine = OwnedMetalEmbedEngine::new(family, dtype);
    let cold_started = Instant::now();
    let loaded = engine
        .load(
            &ValidatedArtifact {
                digest: "sha256:local-metal-fixed-cost".to_string(),
                format: "safetensors-package".to_string(),
            },
            &config,
        )
        .expect("load owned-metal model");
    let cold_load_s = cold_started.elapsed().as_secs_f64();
    let terminal = engine
        .tokenizer_policy(&loaded)
        .expect("tokenizer policy")
        .terminal_token_id;

    let run_replay = |label: &str, vectors: &mut Vec<Vec<f32>>| -> Replay {
        let started = Instant::now();
        let mut tokenize_ms = 0.0;
        let mut engine_ms = 0.0;
        let mut calls = Vec::with_capacity(meta.batches.len());
        for (call_index, batch) in meta.batches.iter().enumerate() {
            let rows = &chunks[batch.first_chunk_seq..batch.first_chunk_seq + batch.chunk_count];
            let tokenize_started = Instant::now();
            let mut tokenized = tokenizer
                .tokenize_batch(rows.iter().map(String::as_str))
                .expect("tokenize call")
                .batch;
            // The catalog profile's token composition ends every Qwen3 row with
            // exactly one terminal token; the engine refuses rows without it.
            if let Some(terminal) = terminal {
                for ids in &mut tokenized.items {
                    if ids.last() == Some(&terminal) {
                        ids.pop();
                    }
                    ids.truncate(MAX_TOKENS - 1);
                    ids.push(terminal);
                }
            }
            let call_tokenize_ms = tokenize_started.elapsed().as_secs_f64() * 1_000.0;
            tokenize_ms += call_tokenize_ms;
            let mut engine_calls = Vec::new();
            for (engine_index, indices) in plan_engine_calls(&tokenized).into_iter().enumerate() {
                let items = indices
                    .iter()
                    .map(|&index| tokenized.items[index].clone())
                    .collect::<Vec<_>>();
                let tokens = items.iter().map(Vec::len).sum::<usize>();
                let max_len = items.iter().map(Vec::len).max().unwrap_or(0);
                eprintln!(
                    "[metal-fixed-cost] call replay={label} call={call_index} engine_call={engine_index} rows={} tokens={tokens} max_len={max_len}",
                    items.len()
                );
                let engine_started = Instant::now();
                let produced = engine
                    .embed_batch(&loaded, TokenBatch { items })
                    .unwrap_or_else(|error| panic!("embed call {call_index}: {error:?}"));
                let ms = engine_started.elapsed().as_secs_f64() * 1_000.0;
                engine_ms += ms;
                for (&index, vector) in indices.iter().zip(produced) {
                    vectors[batch.first_chunk_seq + index] = vector;
                }
                engine_calls.push(EngineCallRecord {
                    rows: indices.len(),
                    tokens,
                    max_len,
                    ms,
                });
            }
            calls.push(CallRecord {
                batch_seq: batch.seq,
                rows: rows.len(),
                tokens: tokenized.items.iter().map(Vec::len).sum(),
                tokenize_ms: call_tokenize_ms,
                engine_calls,
            });
        }
        Replay {
            wall_s: started.elapsed().as_secs_f64(),
            tokenize_ms,
            engine_ms,
            calls,
        }
    };

    let mut warm_vectors = vec![Vec::new(); chunks.len()];
    let warmup_wall_s = run_replay("warmup", &mut warm_vectors).wall_s;
    let mut measured = Vec::with_capacity(replays);
    for replay in 0..replays {
        let mut vectors = vec![Vec::new(); chunks.len()];
        measured.push(run_replay(&replay.to_string(), &mut vectors));
        assert!(
            vectors == warm_vectors,
            "replay {replay} changed vectors relative to warm-up"
        );
    }

    let row_sha256 = warm_vectors
        .iter()
        .map(|vector| hex_sha256(&f32_bytes(vector)))
        .collect::<Vec<_>>();
    let all_bytes = warm_vectors
        .iter()
        .flat_map(|vector| f32_bytes(vector))
        .collect::<Vec<_>>();
    let mut vectors_path = output_path.clone().into_os_string();
    vectors_path.push(".vectors.f32");
    fs::File::create(PathBuf::from(vectors_path))
        .and_then(|mut file| file.write_all(&all_bytes))
        .expect("write vectors");
    let output = Output {
        family: family.as_str(),
        dtype: dtype.as_str(),
        engine_identity: engine_identity(family, dtype),
        bucket_policy_version: BUCKET_POLICY_VERSION,
        max_tokens: MAX_TOKENS,
        attention_units: ATTENTION_UNITS,
        input_sha256: file_sha256(&input_path),
        meta_sha256: file_sha256(&meta_path),
        rows: chunks.len(),
        calls: meta.batches.len(),
        cold_load_s,
        warmup_wall_s,
        replays: measured,
        vectors_sha256: hex_sha256(&all_bytes),
        row_sha256,
    };
    fs::write(&output_path, serde_json::to_vec_pretty(&output).unwrap()).expect("write output");
    for replay in &output.replays {
        println!(
            "{}: wall {:.3}s ({:.1} rows/s), tokenize {:.1} ms, engine {:.1} ms, cold load {:.1}s, warm-up {:.1}s",
            output.family,
            replay.wall_s,
            output.rows as f64 / replay.wall_s,
            replay.tokenize_ms,
            replay.engine_ms,
            cold_load_s,
            warmup_wall_s
        );
    }
    println!("vectors_sha256 {}", output.vectors_sha256);
}

/// The module's bulk-path split: rows sorted by token length (stable), packed
/// into calls of at most eight rows and 3,072 tokens.
fn plan_engine_calls(batch: &TokenBatch) -> Vec<Vec<usize>> {
    let mut order = (0..batch.items.len()).collect::<Vec<_>>();
    order.sort_by_key(|&index| batch.items[index].len());
    let mut calls = Vec::new();
    let mut start = 0;
    while start < order.len() {
        let mut end = start;
        let mut tokens = 0;
        while end < order.len() {
            let item_tokens = batch.items[order[end]].len().max(1);
            if end > start
                && (end - start >= ENGINE_CALL_ITEMS || tokens + item_tokens > ENGINE_CALL_TOKENS)
            {
                break;
            }
            tokens += item_tokens;
            end += 1;
        }
        calls.push(order[start..end].to_vec());
        start = end;
    }
    calls
}

fn f32_bytes(vector: &[f32]) -> Vec<u8> {
    vector
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect()
}

fn hex_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn file_sha256(path: &Path) -> String {
    hex_sha256(&fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display())))
}
