use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use clap::Parser;
use serde_json::json;
use stt_common::{
    artifact::artifacts,
    input::sha256,
    load_corpus,
    measurement::{percentile, WorkerResult},
    score,
};

#[derive(Parser)]
#[command(about = "Verify and score a full-corpus local STT benchmark")]
struct Args {
    #[arg(long)]
    wavs: PathBuf,
    #[arg(long, default_value = "bench/parity")]
    parity: PathBuf,
    #[arg(long)]
    worker: Option<PathBuf>,
    #[arg(long)]
    transcripts: Option<PathBuf>,
    #[arg(long)]
    prepare_only: bool,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    model_revision: String,
    #[arg(long)]
    backend: String,
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        if args.prepare_only {
            args.worker.is_none() && args.transcripts.is_none()
        } else {
            args.worker.is_some() != args.transcripts.is_some()
        },
        "use --prepare-only or exactly one of --worker and --transcripts"
    );
    ensure!(
        !args.output.exists(),
        "refusing to overwrite an existing result"
    );
    let corpus = load_corpus(&args.parity, &args.wavs)?;
    if args.prepare_only {
        if let Some(parent) = args.output.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        fs::write(&args.output, serde_json::to_vec_pretty(&corpus.request)?)?;
        println!(
            "verified {} utterances: {}",
            corpus.request.utterances.len(),
            args.output.display()
        );
        return Ok(());
    }
    let model_artifacts = if args.backend == "apple-speech" {
        Vec::new()
    } else {
        artifacts(&args.model)?
    };
    let pins: serde_json::Value = serde_json::from_str(include_str!("../models.json"))?;
    if matches!(args.backend.as_str(), "metal" | "cuda" | "vulkan") {
        let pin = [
            "transcribe_ultra_q8",
            "transcribe_v3_q8",
            "transcribe_v3_f16",
        ]
        .iter()
        .map(|key| &pins[key])
        .find(|pin| pin["file"].as_str() == args.model.file_name().and_then(|s| s.to_str()))
        .context("unknown GGUF filename; use a pinned model from models.json")?;
        ensure!(
            args.model_revision == pin["revision"].as_str().unwrap(),
            "GGUF revision differs from pin"
        );
        ensure!(
            model_artifacts.len() == 1
                && model_artifacts[0].sha256 == pin["sha256"].as_str().unwrap(),
            "GGUF checksum differs from pin"
        );
    } else if matches!(args.backend.as_str(), "ane-pal6" | "ane-int8") {
        ensure!(
            args.model_revision == pins["fluidaudio_v3"]["revision"].as_str().unwrap(),
            "CoreML revision differs from pin"
        );
    } else {
        ensure!(args.backend == "apple-speech", "unknown backend");
    }
    let parent = args
        .output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(std::path::Path::new("."));
    fs::create_dir_all(parent)?;
    let request_path = args.output.with_extension("request.json");
    let worker_path = args.output.with_extension("worker.json");
    ensure!(
        !request_path.exists() && !worker_path.exists(),
        "request or worker result already exists"
    );
    fs::write(&request_path, serde_json::to_vec_pretty(&corpus.request)?)?;
    let worker_artifacts;
    let wall;
    let raw;
    if let Some(worker) = &args.worker {
        worker_artifacts = artifacts(worker)?;
        let start = Instant::now();
        let status = Command::new(worker)
            .args(["--request"])
            .arg(&request_path)
            .args(["--output"])
            .arg(&worker_path)
            .args(["--model"])
            .arg(&args.model)
            .args(["--backend"])
            .arg(&args.backend)
            .status()
            .with_context(|| format!("launch {}", worker.display()))?;
        wall = Some(start.elapsed().as_secs_f64());
        ensure!(status.success(), "worker exited with {status}");
        raw = fs::read(&worker_path)?;
    } else {
        worker_artifacts = Vec::new();
        wall = None;
        raw = fs::read(args.transcripts.as_ref().unwrap())?;
    }
    let result: WorkerResult = serde_json::from_slice(&raw)?;
    ensure!(
        result.backend == args.backend,
        "worker backend does not match request"
    );
    let splits = score(&corpus, &result)?;
    let times: Vec<_> = result
        .transcripts
        .iter()
        .filter_map(|t| t.seconds)
        .collect();
    let audio: f64 = splits.values().map(|s| s.audio_seconds).sum();
    let warm_seconds: f64 = times.iter().sum();
    let lane_result = synapse_bench::results::LaneResult {
        lane: format!("{}-{}", result.engine, result.backend), workload: "parakeet-corpus-v1".into(),
        model: args.model.display().to_string(), cold_load_s: result.cold_start_seconds,
        infer_wall_s: result.warm_wall_seconds, input_tokens: 0, tok_per_s: 0.0,
        items: times.len() as u64, parity_mean_cosine: None,
        self_peak_rss_bytes: Some(result.peak_rss_bytes),
        notes: "Audio workload: token metrics are not applicable (zero). See STT fields for WER, latency, coverage and real-time factor. RSS excludes out-of-process OS services.".into(),
    };
    let machine = Command::new("hostname")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_default();
    let os = if cfg!(windows) {
        Command::new("cmd").args(["/c", "ver"]).output()
    } else {
        Command::new("uname").arg("-a").output()
    }?;
    ensure!(os.status.success(), "cannot identify operating system");
    let output = json!({
        "schema": "synapse-stt-benchmark-v1", "engine": result.engine,
        "lane_result": lane_result,
        "version": result.version, "commit": result.commit, "backend": result.backend,
        "machine": machine, "os": String::from_utf8_lossy(&os.stdout).trim(),
        "model": args.model, "model_revision": args.model_revision,
        "model_artifacts": model_artifacts, "worker_artifacts": worker_artifacts,
        "corpus_manifest_sha256": sha256(&fs::read(args.parity.join("reference/parakeet-corpus.json"))?),
        "fixture_index_sha256": sha256(&fs::read(args.parity.join("fixtures/index.json"))?),
        "total_wall_seconds": wall, "cold_start_seconds": result.cold_start_seconds,
        "warm_wall_seconds": result.warm_wall_seconds, "warm_inference_seconds": warm_seconds,
        "real_time_factor": if audio > 0.0 { Some(warm_seconds / audio) } else { None },
        "latency_p50_seconds": percentile(&times, 0.5), "latency_p95_seconds": percentile(&times, 0.95),
        "peak_rss_bytes": result.peak_rss_bytes, "gpu_peak_bytes": result.gpu_peak_bytes,
        "energy_joules": result.energy_joules, "requested": corpus.request.utterances.len(),
        "completed": times.len(), "splits": splits, "transcripts": result.transcripts,
    });
    fs::write(&args.output, serde_json::to_vec_pretty(&output)?)?;
    println!("{}", args.output.display());
    Ok(())
}
