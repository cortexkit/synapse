//! Bench lane: supervise a local `llama-server` child process (Metal) and talk to it
//! over HTTP, matching the module architecture we would actually ship.
//!
//! Subcommands:
//! - `embed`: workload A, batched embeddings over a JSONL corpus.
//! - `microllm`: workload B, single-turn intent classification prompts.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};
use clap::{Args, Parser, Subcommand};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use synapse_bench::{
    parity::{load_corpus, load_jsonl, load_reference, mean_parity, Chunk, Prompt},
    results::LaneResult,
};
use tokenizers::{Tokenizer, TruncationParams};

const DEFAULT_SERVER_BINARY: &str = "/opt/zerobrew/bin/llama-server";
const HEALTH_TIMEOUT: Duration = Duration::from_secs(120);
const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(250);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(10);
const RSS_SAMPLE_INTERVAL: Duration = Duration::from_millis(200);
const EMBED_WARMUP_TEXT: &str = "warmup";
const MICROLLM_WARMUP_PROMPT: &str = "Reply with exactly one word: warmup";
const VALID_LABELS: &[&str] = &["config", "test", "logic", "io", "types", "docs"];

#[derive(Parser)]
#[command(name = "lane-llama")]
struct Cli {
    #[command(subcommand)]
    command: LaneCommand,
}

#[derive(Subcommand)]
enum LaneCommand {
    /// Workload A: embed a JSONL corpus through llama-server's HTTP API.
    Embed(EmbedArgs),
    /// Workload B: run one-shot micro-LLM prompts through chat completions.
    Microllm(MicrollmArgs),
}

#[derive(Args)]
struct EmbedArgs {
    /// Path to the embedding GGUF model.
    #[arg(long)]
    model: PathBuf,
    /// Path to tokenizer.json used for input token accounting.
    #[arg(long)]
    tokenizer: PathBuf,
    /// Corpus JSONL ({id, path, text, tokens} per line).
    #[arg(long)]
    corpus: PathBuf,
    /// Output LaneResult JSON path.
    #[arg(long)]
    out: PathBuf,
    /// Optional: write produced vectors (JSONL: {id, vec}) for parity reference.
    #[arg(long)]
    vectors_out: Option<PathBuf>,
    /// Optional: compare produced vectors against reference JSONL ({id, vec}).
    #[arg(long)]
    reference: Option<PathBuf>,
    /// Model label for the result.
    #[arg(long)]
    model_label: String,
    /// Path to the llama-server binary.
    #[arg(long, default_value = DEFAULT_SERVER_BINARY)]
    server_binary: PathBuf,
    /// Tokenizer truncation max length.
    #[arg(long, default_value_t = 512)]
    max_length: usize,
    /// Pooling applied by llama-server for embeddings.
    #[arg(long, default_value = "last")]
    pooling: String,
    /// Embedding normalization mode passed to llama-server.
    #[arg(long, default_value_t = 2)]
    embd_normalize: i32,
    /// Context window passed to llama-server.
    #[arg(long, default_value_t = 1024)]
    ctx_size: usize,
    /// Logical batch-size token budget for llama-server.
    #[arg(long, default_value_t = 4096)]
    batch_size: usize,
    /// Physical batch-size token budget for llama-server.
    #[arg(long, default_value_t = 1024)]
    ubatch_size: usize,
    /// Number of layers to place on the GPU.
    #[arg(long, default_value_t = 99)]
    gpu_layers: usize,
    /// Number of server slots.
    #[arg(long, default_value_t = 1)]
    parallel: usize,
}

#[derive(Args)]
struct MicrollmArgs {
    /// Path to the chat GGUF model.
    #[arg(long)]
    model: PathBuf,
    /// Prompt JSONL ({id, prompt} per line).
    #[arg(long)]
    prompts: PathBuf,
    /// Output LaneResult JSON path.
    #[arg(long)]
    out: PathBuf,
    /// Model label for the result.
    #[arg(long)]
    model_label: String,
    /// Optional limit for smoke runs.
    #[arg(long)]
    limit: Option<usize>,
    /// Path to the llama-server binary.
    #[arg(long, default_value = DEFAULT_SERVER_BINARY)]
    server_binary: PathBuf,
    /// Maximum completion tokens per prompt.
    #[arg(long, default_value_t = 16)]
    max_tokens: u32,
    /// Context window passed to llama-server.
    #[arg(long, default_value_t = 2048)]
    ctx_size: usize,
    /// Logical batch-size token budget for llama-server.
    #[arg(long, default_value_t = 1024)]
    batch_size: usize,
    /// Physical batch-size token budget for llama-server.
    #[arg(long, default_value_t = 512)]
    ubatch_size: usize,
    /// Number of layers to place on the GPU.
    #[arg(long, default_value_t = 99)]
    gpu_layers: usize,
    /// Number of server slots.
    #[arg(long, default_value_t = 1)]
    parallel: usize,
}

#[derive(Debug)]
struct ProducedVector {
    id: String,
    vec: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct EmbeddingsRequest<'a> {
    model: &'a str,
    input: &'a [&'a str],
}

#[derive(Debug, Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingDatum>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingDatum {
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest<'a> {
    model: &'a str,
    messages: [ChatMessage<'a>; 1],
    temperature: f64,
    max_tokens: u32,
    chat_template_kwargs: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
    usage: Option<Usage>,
    timings: Option<Timings>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
}

#[derive(Debug, Deserialize)]
struct ChatMessageResponse {
    content: Option<String>,
    reasoning_content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Usage {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct Timings {
    prompt_n: Option<u64>,
    prompt_ms: Option<f64>,
    prompt_per_second: Option<f64>,
    predicted_n: Option<u64>,
    predicted_ms: Option<f64>,
    predicted_per_second: Option<f64>,
}

#[derive(Debug, Default)]
struct TimingAccumulator {
    prompt_tokens: u64,
    prompt_ms: f64,
    predicted_tokens: u64,
    predicted_ms: f64,
    prompt_per_second_samples: Vec<f64>,
    predicted_per_second_samples: Vec<f64>,
    missing: u64,
}

#[derive(Debug)]
struct LlamaServer {
    child: Child,
    base_url: String,
    stop_sampler: Arc<AtomicBool>,
    peak_rss_bytes: Arc<AtomicU64>,
    rss_sampler: Option<JoinHandle<()>>,
    cleaned_up: bool,
}

#[derive(Clone, Copy)]
enum WarmupKind {
    Embed,
    Chat,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        LaneCommand::Embed(args) => run_embed(args),
        LaneCommand::Microllm(args) => run_microllm(args),
    }
}

fn run_embed(args: EmbedArgs) -> Result<()> {
    let tokenizer = load_tokenizer(&args.tokenizer, args.max_length)?;
    let chunks: Vec<Chunk> = load_corpus(&args.corpus, None)?;

    let texts: Vec<&str> = chunks.iter().map(|chunk| chunk.text.as_str()).collect();
    let encodings = tokenizer
        .encode_batch(texts, true)
        .map_err(|err| anyhow::anyhow!("encode_batch: {err}"))?;
    let token_counts: Vec<usize> = encodings
        .iter()
        .map(|encoding| encoding.get_ids().len())
        .collect();
    let max_seen_tokens = token_counts.iter().copied().max().unwrap_or(0);
    ensure!(
        max_seen_tokens <= args.ctx_size,
        "tokenized input length {max_seen_tokens} exceeds ctx-size {}",
        args.ctx_size
    );

    let server_args = vec![
        "--embedding".to_string(),
        "--pooling".to_string(),
        args.pooling.clone(),
        "--embd-normalize".to_string(),
        args.embd_normalize.to_string(),
        "--ctx-size".to_string(),
        args.ctx_size.to_string(),
        "--batch-size".to_string(),
        args.batch_size.to_string(),
        "--ubatch-size".to_string(),
        args.ubatch_size.to_string(),
        "-ngl".to_string(),
        args.gpu_layers.to_string(),
        "--parallel".to_string(),
        args.parallel.to_string(),
    ];

    let client = build_http_client()?;
    let started = Instant::now();
    let mut server = LlamaServer::spawn(&args.server_binary, &args.model, &server_args)?;
    wait_for_health_and_warmup(&client, &mut server, WarmupKind::Embed)?;
    let cold_load_s = started.elapsed().as_secs_f64();

    let infer_started = Instant::now();
    let mut input_tokens = 0u64;
    let mut produced = Vec::with_capacity(chunks.len());

    let mut batch_start = 0usize;
    let mut batch_tokens = 0usize;
    for index in 0..=chunks.len() {
        let should_flush = if index == chunks.len() {
            index > batch_start
        } else {
            let next = token_counts[index].max(1);
            index > batch_start && batch_tokens + next > args.batch_size
        };

        if should_flush {
            let batch_chunks = &chunks[batch_start..index];
            let batch_inputs: Vec<&str> = batch_chunks
                .iter()
                .map(|chunk| chunk.text.as_str())
                .collect();
            let embeddings = request_embeddings(&client, &server.base_url, &batch_inputs)?;
            ensure!(
                embeddings.len() == batch_chunks.len(),
                "embedding count mismatch: got {}, expected {}",
                embeddings.len(),
                batch_chunks.len()
            );
            for (offset, embedding) in embeddings.into_iter().enumerate() {
                let chunk_index = batch_start + offset;
                input_tokens += token_counts[chunk_index] as u64;
                produced.push(ProducedVector {
                    id: chunks[chunk_index].id.clone(),
                    vec: embedding,
                });
            }
            batch_start = index;
            batch_tokens = 0;
            if index == chunks.len() {
                break;
            }
        }

        if index < chunks.len() {
            batch_tokens += token_counts[index].max(1);
        }
    }
    let infer_wall_s = infer_started.elapsed().as_secs_f64();

    if let Some(path) = &args.vectors_out {
        write_vectors(path, &produced)?;
    }

    let (parity_mean_cosine, parity_matches) = match &args.reference {
        Some(reference) => {
            let reference_vectors = load_reference(reference)?;
            let (mean_cosine, matches) = mean_parity(
                produced.iter().map(|vector| (vector.id.clone(), vector.vec.clone())),
                &reference_vectors,
            );
            ensure!(
                matches > 0,
                "no overlapping ids with reference vectors"
            );
            let mean_cosine = mean_cosine.expect("matched count implies a parity mean");
            ensure!(
                mean_cosine >= 0.98,
                "parity {:.6} is below 0.98; check pooling and normalization",
                mean_cosine
            );
            (Some(mean_cosine), matches)
        }
        None => (None, 0),
    };

    server.shutdown()?;
    let peak_rss = server.peak_rss_bytes();

    let notes = format!(
        "endpoint=/v1/embeddings; request_batching=greedy_sum_tokens<=batch_size; cold_load=health+warmup_request; pooling={}; embd_normalize={}; ctx_size={}; batch_size={}; ubatch_size={}; ngl={}; parallel={}; reference_matches={}",
        args.pooling,
        args.embd_normalize,
        args.ctx_size,
        args.batch_size,
        args.ubatch_size,
        args.gpu_layers,
        args.parallel,
        parity_matches
    );

    let result = LaneResult {
        lane: "llama-metal-embed".into(),
        workload: "embed-corpus-v1".into(),
        model: args.model_label,
        cold_load_s,
        infer_wall_s,
        input_tokens,
        tok_per_s: rate(input_tokens as f64, infer_wall_s),
        items: produced.len() as u64,
        parity_mean_cosine,
        self_peak_rss_bytes: Some(peak_rss),
        notes,
    };
    write_result(&args.out, &result)?;
    eprintln!(
        "llama-metal-embed: {} items, {} tokens, {:.1} tok/s, cold_load {:.1}s, infer {:.1}s",
        result.items,
        result.input_tokens,
        result.tok_per_s,
        result.cold_load_s,
        result.infer_wall_s
    );
    Ok(())
}

fn run_microllm(args: MicrollmArgs) -> Result<()> {
    let mut prompts: Vec<Prompt> = load_jsonl(&args.prompts)?;
    if let Some(limit) = args.limit {
        prompts.truncate(limit);
    }
    ensure!(
        !prompts.is_empty(),
        "empty prompt set: {}",
        args.prompts.display()
    );

    let server_args = vec![
        "--ctx-size".to_string(),
        args.ctx_size.to_string(),
        "--batch-size".to_string(),
        args.batch_size.to_string(),
        "--ubatch-size".to_string(),
        args.ubatch_size.to_string(),
        "-ngl".to_string(),
        args.gpu_layers.to_string(),
        "--parallel".to_string(),
        args.parallel.to_string(),
    ];

    let client = build_http_client()?;
    let started = Instant::now();
    let mut server = LlamaServer::spawn(&args.server_binary, &args.model, &server_args)?;
    wait_for_health_and_warmup(&client, &mut server, WarmupKind::Chat)?;
    let cold_load_s = started.elapsed().as_secs_f64();

    let infer_started = Instant::now();
    let mut input_tokens = 0u64;
    let mut generated_tokens = 0u64;
    let mut valid_labels = 0u64;
    let mut thinking_responses = 0u64;
    let mut invalid_examples = Vec::new();
    let mut timing_accumulator = TimingAccumulator::default();

    for prompt in &prompts {
        let response =
            request_chat_completion(&client, &server.base_url, &prompt.prompt, args.max_tokens)?;
        let choice = response
            .choices
            .first()
            .context("chat completion returned no choices")?;
        let content = choice.message.content.clone().unwrap_or_default();
        if choice
            .message
            .reasoning_content
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty())
        {
            thinking_responses += 1;
        }
        let normalized = normalize_label(&content);
        if normalized.is_some() {
            valid_labels += 1;
        } else if invalid_examples.len() < 3 {
            invalid_examples.push(format!("{}={:?}", prompt.id, content));
        }

        let usage = response.usage.as_ref();
        input_tokens += usage
            .and_then(|usage| usage.prompt_tokens)
            .or_else(|| {
                response
                    .timings
                    .as_ref()
                    .and_then(|timings| timings.prompt_n)
            })
            .unwrap_or(0);
        generated_tokens += usage
            .and_then(|usage| usage.completion_tokens)
            .or_else(|| {
                response
                    .timings
                    .as_ref()
                    .and_then(|timings| timings.predicted_n)
            })
            .unwrap_or(0);
        timing_accumulator.record(response.timings.as_ref());
    }
    let infer_wall_s = infer_started.elapsed().as_secs_f64();

    server.shutdown()?;
    let peak_rss = server.peak_rss_bytes();

    let valid_fraction = valid_labels as f64 / prompts.len() as f64;
    let prompt_tok_s = timing_accumulator.prompt_rate();
    let decode_tok_s = timing_accumulator.predicted_rate();
    let invalid_note = if invalid_examples.is_empty() {
        "none".to_string()
    } else {
        invalid_examples.join(" | ")
    };
    let notes = format!(
        "endpoint=/v1/chat/completions; cold_load=health+warmup_request; thinking_control=chat_template_kwargs.enable_thinking=false; server_timings=weighted_from_prompt_ms_and_predicted_ms; max_tokens={}; generated_tokens={}; valid_labels={}/{} ({:.3}); invalid_examples={}; thinking_responses={}; server_prompt_tok_per_s={:.1}; server_decode_tok_per_s={:.1}; prompt_per_second_samples={}; predicted_per_second_samples={}; timings_missing={}; ctx_size={}; batch_size={}; ubatch_size={}; ngl={}; parallel={}",
        args.max_tokens,
        generated_tokens,
        valid_labels,
        prompts.len(),
        valid_fraction,
        invalid_note,
        thinking_responses,
        prompt_tok_s,
        decode_tok_s,
        timing_accumulator.prompt_per_second_samples.len(),
        timing_accumulator.predicted_per_second_samples.len(),
        timing_accumulator.missing,
        args.ctx_size,
        args.batch_size,
        args.ubatch_size,
        args.gpu_layers,
        args.parallel
    );

    let result = LaneResult {
        lane: "llama-metal-microllm".into(),
        workload: "microllm-oneshot-v1".into(),
        model: args.model_label,
        cold_load_s,
        infer_wall_s,
        input_tokens,
        tok_per_s: rate(input_tokens as f64, infer_wall_s),
        items: prompts.len() as u64,
        parity_mean_cosine: None,
        self_peak_rss_bytes: Some(peak_rss),
        notes,
    };
    write_result(&args.out, &result)?;
    eprintln!(
        "llama-metal-microllm: {} items, {} prompt tokens, {:.1} tok/s, cold_load {:.1}s, infer {:.1}s",
        result.items, result.input_tokens, result.tok_per_s, result.cold_load_s, result.infer_wall_s
    );
    Ok(())
}

fn load_tokenizer(path: &Path, max_length: usize) -> Result<Tokenizer> {
    let mut tokenizer =
        Tokenizer::from_file(path).map_err(|err| anyhow::anyhow!("tokenizer: {err}"))?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }))
        .map_err(|err| anyhow::anyhow!("truncation: {err}"))?;
    Ok(tokenizer)
}

fn build_http_client() -> Result<Client> {
    Client::builder()
        .connect_timeout(Duration::from_secs(1))
        .timeout(REQUEST_TIMEOUT)
        .build()
        .context("build reqwest client")
}

fn request_embeddings(client: &Client, base_url: &str, inputs: &[&str]) -> Result<Vec<Vec<f32>>> {
    let response: EmbeddingsResponse = client
        .post(format!("{base_url}/v1/embeddings"))
        .json(&EmbeddingsRequest {
            model: "llama-server",
            input: inputs,
        })
        .send()
        .context("POST /v1/embeddings")?
        .error_for_status()
        .context("/v1/embeddings returned error status")?
        .json()
        .context("decode /v1/embeddings response")?;

    let mut data = response.data;
    data.sort_by_key(|datum| datum.index);
    Ok(data.into_iter().map(|datum| datum.embedding).collect())
}

fn request_chat_completion(
    client: &Client,
    base_url: &str,
    prompt: &str,
    max_tokens: u32,
) -> Result<ChatCompletionResponse> {
    // Prefer template kwargs over a prompt prefix so the benchmark keeps the workload text
    // unchanged while still disabling Qwen3 thinking mode on this llama-server build.
    client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&ChatCompletionRequest {
            model: "llama-server",
            messages: [ChatMessage {
                role: "user",
                content: prompt,
            }],
            temperature: 0.0,
            max_tokens,
            chat_template_kwargs: json!({ "enable_thinking": false }),
        })
        .send()
        .context("POST /v1/chat/completions")?
        .error_for_status()
        .context("/v1/chat/completions returned error status")?
        .json()
        .context("decode /v1/chat/completions response")
}

fn wait_for_health_and_warmup(
    client: &Client,
    server: &mut LlamaServer,
    kind: WarmupKind,
) -> Result<()> {
    let deadline = Instant::now() + HEALTH_TIMEOUT;
    let health_url = format!("{}/health", server.base_url);

    loop {
        if Instant::now() > deadline {
            bail!("timed out waiting for llama-server health and warmup");
        }
        server.ensure_running()?;

        match client.get(&health_url).send() {
            Ok(response) if response.status().is_success() => {
                let warmup = match kind {
                    WarmupKind::Embed => {
                        request_embeddings(client, &server.base_url, &[EMBED_WARMUP_TEXT])
                            .map(|_| ())
                    }
                    WarmupKind::Chat => {
                        request_chat_completion(client, &server.base_url, MICROLLM_WARMUP_PROMPT, 1)
                            .map(|_| ())
                    }
                };
                if warmup.is_ok() {
                    return Ok(());
                }
            }
            Ok(_) | Err(_) => {}
        }

        thread::sleep(HEALTH_POLL_INTERVAL);
    }
}

fn write_result(path: &Path, result: &LaneResult) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create result parent {}", parent.display()))?;
    }
    std::fs::write(path, serde_json::to_string_pretty(result)?)
        .with_context(|| format!("write {}", path.display()))
}

fn write_vectors(path: &Path, vectors: &[ProducedVector]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create vector parent {}", parent.display()))?;
    }
    let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for vector in vectors {
        serde_json::to_writer(
            &mut writer,
            &json!({ "id": &vector.id, "vec": &vector.vec }),
        )?;
        writer.write_all(b"\n")?;
    }
    writer.flush().context("flush vectors writer")
}

fn normalize_label(output: &str) -> Option<&'static str> {
    let cleaned = output
        .trim()
        .trim_matches(|ch: char| ch.is_ascii_punctuation() || ch.is_whitespace())
        .to_ascii_lowercase();
    VALID_LABELS.iter().copied().find(|label| *label == cleaned)
}

fn rate(tokens: f64, seconds: f64) -> f64 {
    if seconds > 0.0 {
        tokens / seconds
    } else {
        0.0
    }
}

impl TimingAccumulator {
    fn record(&mut self, timings: Option<&Timings>) {
        let Some(timings) = timings else {
            self.missing += 1;
            return;
        };

        if let (Some(tokens), Some(ms)) = (timings.prompt_n, timings.prompt_ms) {
            self.prompt_tokens += tokens;
            self.prompt_ms += ms;
        }
        if let (Some(tokens), Some(ms)) = (timings.predicted_n, timings.predicted_ms) {
            self.predicted_tokens += tokens;
            self.predicted_ms += ms;
        }
        if let Some(rate) = timings.prompt_per_second {
            self.prompt_per_second_samples.push(rate);
        }
        if let Some(rate) = timings.predicted_per_second {
            self.predicted_per_second_samples.push(rate);
        }
    }

    fn prompt_rate(&self) -> f64 {
        rate(self.prompt_tokens as f64, self.prompt_ms / 1000.0)
    }

    fn predicted_rate(&self) -> f64 {
        rate(self.predicted_tokens as f64, self.predicted_ms / 1000.0)
    }
}

impl LlamaServer {
    fn spawn(server_binary: &Path, model: &Path, extra_args: &[String]) -> Result<Self> {
        let port = reserve_free_port()?;
        let mut command = Command::new(server_binary);
        command
            .arg("-m")
            .arg(model)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .args(extra_args)
            .stdout(Stdio::null())
            .stderr(Stdio::null());

        let child = command.spawn().with_context(|| {
            format!(
                "spawn llama-server {} with model {}",
                server_binary.display(),
                model.display()
            )
        })?;
        let pid = child.id();
        let (stop_sampler, peak_rss_bytes, rss_sampler) = spawn_rss_sampler(pid);

        Ok(Self {
            child,
            base_url: format!("http://127.0.0.1:{port}"),
            stop_sampler,
            peak_rss_bytes,
            rss_sampler: Some(rss_sampler),
            cleaned_up: false,
        })
    }

    fn ensure_running(&mut self) -> Result<()> {
        if let Some(status) = self.child.try_wait().context("poll llama-server child")? {
            bail!("llama-server exited early with status {status}");
        }
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        self.cleanup()
    }

    fn peak_rss_bytes(&self) -> u64 {
        self.peak_rss_bytes.load(Ordering::Relaxed)
    }

    fn cleanup(&mut self) -> Result<()> {
        if self.cleaned_up {
            return Ok(());
        }
        self.cleaned_up = true;

        let pid = self.child.id() as i32;
        let _ = send_signal(pid, libc::SIGTERM);

        let deadline = Instant::now() + SHUTDOWN_TIMEOUT;
        loop {
            match self
                .child
                .try_wait()
                .context("poll llama-server during shutdown")?
            {
                Some(_) => break,
                None if Instant::now() < deadline => thread::sleep(Duration::from_millis(100)),
                None => {
                    self.child.kill().context("force kill llama-server")?;
                    let _ = self.child.wait();
                    break;
                }
            }
        }

        self.stop_sampler.store(true, Ordering::Relaxed);
        if let Some(handle) = self.rss_sampler.take() {
            let _ = handle.join();
        }
        Ok(())
    }
}

impl Drop for LlamaServer {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

fn reserve_free_port() -> Result<u16> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).context("bind free localhost port")?;
    let port = listener
        .local_addr()
        .context("read bound localhost port")?
        .port();
    drop(listener);
    Ok(port)
}

fn spawn_rss_sampler(pid: u32) -> (Arc<AtomicBool>, Arc<AtomicU64>, JoinHandle<()>) {
    let stop = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicU64::new(0));
    let stop_clone = Arc::clone(&stop);
    let peak_clone = Arc::clone(&peak);
    let handle = thread::spawn(move || {
        while !stop_clone.load(Ordering::Relaxed) {
            if let Some(bytes) = sample_rss_bytes(pid) {
                update_peak(&peak_clone, bytes);
            }
            thread::sleep(RSS_SAMPLE_INTERVAL);
        }
        if let Some(bytes) = sample_rss_bytes(pid) {
            update_peak(&peak_clone, bytes);
        }
    });
    (stop, peak, handle)
}

fn update_peak(peak: &AtomicU64, sample: u64) {
    let mut current = peak.load(Ordering::Relaxed);
    while sample > current {
        match peak.compare_exchange(current, sample, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
}

fn sample_rss_bytes(pid: u32) -> Option<u64> {
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let rss_kib: u64 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .ok()?;
    Some(rss_kib.saturating_mul(1024))
}

fn send_signal(pid: i32, signal: i32) -> Result<()> {
    let rc = unsafe { libc::kill(pid, signal) };
    if rc == 0 {
        return Ok(());
    }
    let err = std::io::Error::last_os_error();
    if err.raw_os_error() == Some(libc::ESRCH) {
        return Ok(());
    }
    Err(err).with_context(|| format!("send signal {signal} to pid {pid}"))
}
