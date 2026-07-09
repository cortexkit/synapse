//! Bench lane: wrapped local inference servers (LMStudio, Ollama) via their
//! OpenAI-compatible /v1/embeddings endpoint.
//!
//! This measures the "wrap an existing app" option of the runtime decision:
//! the server is EXTERNAL (user-launched GUI app or daemon we don't control),
//! so cold_load_s only covers first-request latency against an already-running
//! server, and RSS is sampled from the named server processes, not our child.
//!
//! Batching mirrors the scar tissue: small sub-batches with bounded retry on
//! transient failures (LMStudio 400s under concurrent load).

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use synapse_bench::{
    parity::{cosine, load_corpus, load_reference, Chunk},
    results::LaneResult,
};
use tokenizers::{Tokenizer, TruncationParams};

#[derive(Parser)]
struct Args {
    /// Base URL, e.g. http://127.0.0.1:1234 (LMStudio) or http://127.0.0.1:11434 (Ollama)
    #[arg(long)]
    base_url: String,
    /// Model name as the server knows it
    #[arg(long)]
    model: String,
    /// Lane label, e.g. "wrap-lmstudio"
    #[arg(long)]
    lane: String,
    /// Corpus JSONL
    #[arg(long)]
    corpus: PathBuf,
    /// Output LaneResult JSON
    #[arg(long)]
    out: PathBuf,
    /// Optional parity reference vectors JSONL ({id, vec})
    #[arg(long)]
    reference: Option<PathBuf>,
    /// tokenizer.json for input-token accounting (same as other lanes)
    #[arg(long)]
    tokenizer: PathBuf,
    /// Items per request batch
    #[arg(long, default_value_t = 32)]
    batch: usize,
    /// Process names to RSS-sample (comma separated)
    #[arg(long, default_value = "")]
    rss_process_names: String,
    /// Max chunks (smoke)
    #[arg(long)]
    limit: Option<usize>,
    #[arg(long)]
    model_label: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut tokenizer =
        Tokenizer::from_file(&args.tokenizer).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: 512,
            ..Default::default()
        }))
        .map_err(|e| anyhow::anyhow!("truncation: {e}"))?;

    let mut chunks: Vec<Chunk> = load_corpus(&args.corpus, None)?;
    if let Some(limit) = args.limit {
        chunks.truncate(limit);
    }
    anyhow::ensure!(!chunks.is_empty(), "empty corpus");

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;
    let url = format!("{}/v1/embeddings", args.base_url.trim_end_matches('/'));

    // Warmup/cold-load probe: one tiny request against the running server.
    let started = Instant::now();
    let _ = embed_batch(&client, &url, &args.model, &["warmup".to_string()])?;
    let cold_load_s = started.elapsed().as_secs_f64();

    let reference = args.reference.as_deref().map(load_reference).transpose()?;

    let infer_started = Instant::now();
    let mut input_tokens = 0u64;
    let mut peak_rss = 0u64;
    let mut cos_sum = 0f64;
    let mut cos_n = 0usize;

    for batch in chunks.chunks(args.batch) {
        let texts: Vec<String> = batch
            .iter()
            .map(|c| {
                // Pre-truncate to 512 tokens so the server sees the same inputs
                // as other lanes (server-side truncation behavior varies).
                let enc = tokenizer
                    .encode(c.text.as_str(), true)
                    .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
                input_tokens += enc.get_ids().len() as u64;
                Ok(if enc.get_ids().len() >= 512 {
                    let dec = tokenizer
                        .decode(&enc.get_ids()[..512], true)
                        .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
                    dec
                } else {
                    c.text.clone()
                })
            })
            .collect::<Result<_>>()?;

        // Bounded retry with backoff on transient failures (the scar).
        let mut attempt = 0;
        let vectors = loop {
            match embed_batch(&client, &url, &args.model, &texts) {
                Ok(v) => break v,
                Err(e) if attempt < 3 => {
                    attempt += 1;
                    eprintln!("transient failure (attempt {attempt}): {e}; backing off");
                    std::thread::sleep(Duration::from_millis(500 * (1 << attempt)));
                }
                Err(e) => return Err(e).context("embed batch failed after retries"),
            }
        };
        anyhow::ensure!(vectors.len() == batch.len(), "vector count mismatch");

        if let Some(reference) = &reference {
            for (chunk, vec) in batch.iter().zip(&vectors) {
                if let Some(ref_vec) = reference.get(&chunk.id) {
                    cos_sum += cosine(vec, ref_vec);
                    cos_n += 1;
                }
            }
        }

        if !args.rss_process_names.is_empty() {
            for name in args.rss_process_names.split(',') {
                if let Some(rss) = rss_of_named(name) {
                    peak_rss = peak_rss.max(rss);
                }
            }
        }
    }
    let infer_wall_s = infer_started.elapsed().as_secs_f64();

    let result = LaneResult {
        lane: args.lane.clone(),
        workload: "embed-corpus-v1".into(),
        model: args.model_label,
        cold_load_s,
        infer_wall_s,
        input_tokens,
        tok_per_s: input_tokens as f64 / infer_wall_s,
        items: chunks.len() as u64,
        parity_mean_cosine: (cos_n > 0).then(|| cos_sum / cos_n as f64),
        self_peak_rss_bytes: (peak_rss > 0).then_some(peak_rss),
        notes: format!(
            "external server (cold_load = first-request latency only, server pre-running), batch={}, parity_ids={cos_n}",
            args.batch
        ),
    };
    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.out, serde_json::to_string_pretty(&result)?)?;
    eprintln!(
        "{}: {} items, {:.1} tok/s, parity {:?}",
        result.lane, result.items, result.tok_per_s, result.parity_mean_cosine
    );
    Ok(())
}

fn embed_batch(
    client: &reqwest::blocking::Client,
    url: &str,
    model: &str,
    texts: &[String],
) -> Result<Vec<Vec<f32>>> {
    let resp = client
        .post(url)
        .json(&serde_json::json!({ "model": model, "input": texts }))
        .send()?;
    anyhow::ensure!(
        resp.status().is_success(),
        "server returned {}",
        resp.status()
    );
    let body: serde_json::Value = resp.json()?;
    let data = body["data"].as_array().context("missing data[]")?;
    let mut out = Vec::with_capacity(data.len());
    // OpenAI contract: entries carry an index; sort by it rather than trusting order.
    let mut indexed: Vec<(usize, Vec<f32>)> = data
        .iter()
        .map(|d| {
            let idx = d["index"].as_u64().unwrap_or(0) as usize;
            let vec = d["embedding"]
                .as_array()
                .context("missing embedding")?
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();
            Ok((idx, vec))
        })
        .collect::<Result<_>>()?;
    indexed.sort_by_key(|(i, _)| *i);
    for (_, v) in indexed {
        out.push(v);
    }
    Ok(out)
}

fn rss_of_named(name: &str) -> Option<u64> {
    let out = std::process::Command::new("pgrep")
        .args(["-f", name])
        .output()
        .ok()?;
    let pids = String::from_utf8_lossy(&out.stdout);
    let mut total = 0u64;
    for pid in pids.lines() {
        let out = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", pid.trim()])
            .output()
            .ok()?;
        if let Ok(kb) = String::from_utf8_lossy(&out.stdout).trim().parse::<u64>() {
            total += kb * 1024;
        }
    }
    (total > 0).then_some(total)
}
