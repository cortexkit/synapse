#![forbid(unsafe_code)]

use std::{
    env,
    path::PathBuf,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use subc_client_rs::{CallOptions, ConsumerOptions, SubcConsumer};
use subc_protocol::{BindIdentity, RouteTarget};
use tokio::time::{interval, sleep, MissedTickBehavior};

const DEFAULT_SUBC: &str = "/Users/[owner]/.local/share/cortexkit/run/subc-connection.json";
const DEFAULT_MODEL: &str = "gte-modernbert-base-f16";
const DEFAULT_BATCHES: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 256];
const DEFAULT_CLASSES: &[TextClass] = &[TextClass::Memory, TextClass::Chunk];
const DEFAULT_REPETITIONS: usize = 3;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);
const ADMISSION_SAMPLE_INTERVAL: Duration = Duration::from_millis(100);

#[derive(Clone, Copy, Debug)]
enum TextClass {
    Memory,
    Chunk,
}

impl TextClass {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "memory" => Ok(Self::Memory),
            "chunk" => Ok(Self::Chunk),
            other => bail!("unknown text class {other}; expected memory or chunk"),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Memory => "MEMORY",
            Self::Chunk => "CHUNK",
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse(env::args().skip(1))?;
    let consumer = SubcConsumer::connect(&args.subc, ConsumerOptions::default())
        .await
        .with_context(|| format!("connect to subc through {}", args.subc.display()))?;
    let identity = BindIdentity {
        project_root: env::current_dir().context("find exerciser project root")?,
        harness: "inline-embed-throughput".to_string(),
        session: format!("inline-embed-throughput-{}", std::process::id()),
    };
    let nonce = format!(
        "{}-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("read invocation clock")?
            .as_nanos(),
        std::process::id()
    );

    let warmup_class = args.classes.first().copied().unwrap_or(TextClass::Memory);
    let warmup = embedding_batch(
        &consumer,
        &identity,
        &args.model,
        1,
        warmup_class,
        &format!("{nonce}-warmup"),
    )
    .await?;
    ensure_response_vectors(&warmup, 1)?;
    ensure_response_ids(&warmup, 1)?;

    let mut cells = Vec::with_capacity(args.classes.len() * args.batches.len());
    for &class in &args.classes {
        for &batch_size in &args.batches {
            let mut repetitions = Vec::with_capacity(args.repetitions);
            for repetition in 0..args.repetitions {
                let request_nonce = format!(
                    "{nonce}-{}-{batch_size}-{repetition}",
                    class.as_str().to_ascii_lowercase()
                );
                let started_at_ms = epoch_ms()?;
                let started = Instant::now();
                let (response, admission_samples) = timed_embedding_batch(
                    &consumer,
                    &identity,
                    &args.model,
                    batch_size,
                    class,
                    &request_nonce,
                )
                .await?;
                let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
                let finished_at_ms = epoch_ms()?;
                let token_count = response_token_count(&response);
                let submitted_token_count = response_submitted_token_count(&response);
                ensure_response_vectors(&response, batch_size)?;
                ensure_response_ids(&response, batch_size)?;
                repetitions.push(json!({
                    "repetition": repetition + 1,
                    "started_at_ms": started_at_ms,
                    "finished_at_ms": finished_at_ms,
                    "batch": batch_size,
                    "submitted_tokens": submitted_token_count,
                    "effective_tokens": token_count,
                    "elapsed_ms": elapsed_ms,
                    "items_per_second": batch_size as f64 / (elapsed_ms / 1_000.0),
                    "tokens_per_second": token_count as f64 / (elapsed_ms / 1_000.0),
                    "admission": admission_summary(&admission_samples),
                    "admission_samples": admission_samples,
                }));
            }
            cells.push(aggregate_cell(class, batch_size, repetitions)?);
        }
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "model": args.model,
            "input_type": "document",
            "batches": args.batches,
            "classes": args.classes.iter().map(|class| class.as_str()).collect::<Vec<_>>(),
            "repetitions": args.repetitions,
            "warmup": "one throwaway embed.batch outside timing",
            "cells": cells,
        }))?
    );
    consumer.close().await;
    Ok(())
}

struct Args {
    subc: PathBuf,
    model: String,
    batches: Vec<usize>,
    classes: Vec<TextClass>,
    repetitions: usize,
}

impl Args {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Self> {
        let mut subc = env::var_os("SUBC_CONNECTION_FILE")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_SUBC));
        let mut model = DEFAULT_MODEL.to_string();
        let mut batches = DEFAULT_BATCHES.to_vec();
        let mut classes = DEFAULT_CLASSES.to_vec();
        let mut repetitions = DEFAULT_REPETITIONS;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--subc" => {
                    subc = PathBuf::from(args.next().context("--subc requires a path")?);
                }
                "--model" => {
                    model = args.next().context("--model requires a model id")?;
                }
                "--batches" => {
                    batches = args
                        .next()
                        .context("--batches requires comma-separated sizes")?
                        .split(',')
                        .map(|value| value.parse::<usize>().context("invalid batch size"))
                        .collect::<Result<Vec<_>>>()?;
                }
                "--classes" => {
                    classes = args
                        .next()
                        .context("--classes requires comma-separated classes")?
                        .split(',')
                        .map(TextClass::parse)
                        .collect::<Result<Vec<_>>>()?;
                }
                "--repetitions" => {
                    repetitions = args
                        .next()
                        .context("--repetitions requires a positive integer")?
                        .parse()
                        .context("invalid repetitions")?;
                }
                "-h" | "--help" => {
                    println!(
                        "Usage: inline_embed_throughput [--subc PATH] [--model ID] [--batches 1,2,4,8,16,32,64,128,256] [--classes memory,chunk] [--repetitions 3]"
                    );
                    std::process::exit(0);
                }
                other => bail!("unknown argument {other}"),
            }
        }
        ensure!(!batches.is_empty() && batches.iter().all(|&size| size > 0));
        ensure!(!classes.is_empty());
        ensure!(repetitions > 0);
        Ok(Self {
            subc,
            model,
            batches,
            classes,
            repetitions,
        })
    }
}

async fn start_embedding_batch(
    consumer: &SubcConsumer,
    identity: &BindIdentity,
    model: &str,
    batch_size: usize,
    class: TextClass,
    nonce: &str,
) -> Result<Value> {
    let items = (0..batch_size)
        .map(|index| {
            json!({
                "id": format!("item-{index}"),
                "text": fixture_text(index, class, nonce),
            })
        })
        .collect::<Vec<_>>();
    call(
        consumer,
        identity,
        "embed.batch",
        json!({
            "model": model,
            "input_type": "document",
            "items": items,
            "accept_declared": true,
            "request_key": format!("inline-throughput-{nonce}-{batch_size}"),
        }),
    )
    .await
}

async fn embedding_batch(
    consumer: &SubcConsumer,
    identity: &BindIdentity,
    model: &str,
    batch_size: usize,
    class: TextClass,
    nonce: &str,
) -> Result<Value> {
    let response =
        start_embedding_batch(consumer, identity, model, batch_size, class, nonce).await?;
    if let Some(job_id) = response["result"]["job_id"].as_str() {
        poll_job(consumer, identity, job_id).await
    } else {
        Ok(response)
    }
}

async fn timed_embedding_batch(
    consumer: &SubcConsumer,
    identity: &BindIdentity,
    model: &str,
    batch_size: usize,
    class: TextClass,
    nonce: &str,
) -> Result<(Value, Vec<Value>)> {
    let mut admission_samples = Vec::new();
    let mut sampler = interval(ADMISSION_SAMPLE_INTERVAL);
    sampler.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let request = embedding_batch(consumer, identity, model, batch_size, class, nonce);
    tokio::pin!(request);
    let response = loop {
        tokio::select! {
            response = &mut request => break response?,
            _ = sampler.tick() => {
                admission_samples.push(call(consumer, identity, "admission.status", json!({})).await?);
            }
        }
    };
    Ok((response, admission_samples))
}

async fn poll_job(consumer: &SubcConsumer, identity: &BindIdentity, job_id: &str) -> Result<Value> {
    let deadline = Instant::now() + REQUEST_TIMEOUT;
    let first_page = loop {
        if Instant::now() >= deadline {
            bail!("embedding job {job_id} did not finish before {REQUEST_TIMEOUT:?}");
        }
        let page = call(
            consumer,
            identity,
            "embed.result",
            json!({ "job_id": job_id }),
        )
        .await?;
        match page["result"]["state"].as_str() {
            Some("done") => break page,
            Some("failed_transient" | "failed_permanent") => {
                bail!("embedding job {job_id} failed: {page}")
            }
            Some("queued" | "running") => sleep(Duration::from_millis(25)).await,
            other => bail!("unexpected embedding job state {other:?}: {page}"),
        }
    };

    let page_count = first_page["result"]["page_count"]
        .as_u64()
        .context("completed job omitted page_count")?;
    let mut merged = first_page;
    for page_no in 1..page_count {
        let page = call(
            consumer,
            identity,
            "embed.result",
            json!({ "job_id": job_id, "page": page_no }),
        )
        .await?;
        for field in ["vectors", "real_token_counts", "truncation_disclosures"] {
            let additions = page["result"][field]
                .as_array()
                .with_context(|| format!("job page {page_no} omitted {field}"))?
                .clone();
            merged["result"][field]
                .as_array_mut()
                .with_context(|| format!("job page 0 omitted {field}"))?
                .extend(additions);
        }
    }
    Ok(merged)
}

async fn call(
    consumer: &SubcConsumer,
    identity: &BindIdentity,
    method: &str,
    params: Value,
) -> Result<Value> {
    let request = serde_json::to_vec(&json!({ "method": method, "params": params }))?;
    let response = consumer
        .call(
            RouteTarget::ManagementSurface {
                module_id: "synapse".to_string(),
            },
            identity.clone(),
            request,
            CallOptions {
                timeout: REQUEST_TIMEOUT,
                ..CallOptions::default()
            },
        )
        .await
        .with_context(|| format!("call {method}"))?;
    let value: Value = serde_json::from_slice(&response).context("decode Synapse response")?;
    if value["result"]["error"].is_object() {
        bail!("{method} returned an error: {value}");
    }
    Ok(value)
}

fn aggregate_cell(class: TextClass, batch: usize, repetitions: Vec<Value>) -> Result<Value> {
    let elapsed = repetitions
        .iter()
        .map(|row| {
            row["elapsed_ms"]
                .as_f64()
                .context("repetition omitted elapsed_ms")
        })
        .collect::<Result<Vec<_>>>()?;
    let items_per_second = repetitions
        .iter()
        .map(|row| {
            row["items_per_second"]
                .as_f64()
                .context("repetition omitted items_per_second")
        })
        .collect::<Result<Vec<_>>>()?;
    let tokens_per_second = repetitions
        .iter()
        .map(|row| {
            row["tokens_per_second"]
                .as_f64()
                .context("repetition omitted tokens_per_second")
        })
        .collect::<Result<Vec<_>>>()?;
    let effective_tokens = repetitions
        .iter()
        .map(|row| {
            row["effective_tokens"]
                .as_u64()
                .context("repetition omitted tokens")
        })
        .collect::<Result<Vec<_>>>()?;
    let submitted_tokens = repetitions
        .iter()
        .map(|row| {
            row["submitted_tokens"]
                .as_u64()
                .context("repetition omitted submitted_tokens")
        })
        .collect::<Result<Vec<_>>>()?;
    let all_samples = repetitions
        .iter()
        .flat_map(|row| {
            row["admission_samples"]
                .as_array()
                .into_iter()
                .flatten()
                .cloned()
        })
        .collect::<Vec<_>>();
    let latency_p50_ms = percentile(&elapsed, 0.50);
    let latency_p95_ms = percentile(&elapsed, 0.95);
    Ok(json!({
        "class": class.as_str(),
        "batch": batch,
        "effective_tokens_median": median_u64(&effective_tokens),
        "submitted_tokens_median": median_u64(&submitted_tokens),
        "elapsed_median_ms": median(&elapsed),
        "items_per_second_median": median(&items_per_second),
        "tokens_per_second_median": median(&tokens_per_second),
        "single_item_latency_p50_ms": if batch == 1 { Some(latency_p50_ms) } else { None::<f64> },
        "single_item_latency_p95_ms": if batch == 1 { Some(latency_p95_ms) } else { None::<f64> },
        "admission": admission_summary(&all_samples),
        "repetitions": repetitions,
    }))
}

fn admission_summary(samples: &[Value]) -> Value {
    let waiters = samples
        .iter()
        .filter_map(|sample| sample["result"]["execution_waiters"].as_u64())
        .collect::<Vec<_>>();
    let in_flight = samples
        .iter()
        .filter_map(|sample| sample["result"]["inline_in_flight_executions"].as_u64())
        .collect::<Vec<_>>();
    let acquire_p50 = samples
        .iter()
        .filter_map(|sample| sample["result"]["execution_wait_p50_ms"].as_f64())
        .collect::<Vec<_>>();
    let acquire_p95 = samples
        .iter()
        .filter_map(|sample| sample["result"]["execution_wait_p95_ms"].as_f64())
        .collect::<Vec<_>>();
    json!({
        "samples": samples.len(),
        "execution_waiters_max": waiters.iter().copied().max().unwrap_or(0),
        "inline_in_flight_executions_max": in_flight.iter().copied().max().unwrap_or(0),
        "acquire_wait_p50_ms_p95": percentile(&acquire_p50, 0.95),
        "acquire_wait_p95_ms_p95": percentile(&acquire_p95, 0.95),
        "acquire_wait_p50_ms_max": acquire_p50.iter().copied().fold(0.0, f64::max),
        "acquire_wait_p95_ms_max": acquire_p95.iter().copied().fold(0.0, f64::max),
    })
}

fn percentile(samples: &[f64], quantile: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() as f64 * quantile).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[index]
}

fn median(samples: &[f64]) -> f64 {
    percentile(samples, 0.50)
}

fn median_u64(samples: &[u64]) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    sorted[(sorted.len() - 1) / 2]
}

fn fixture_text(index: usize, class: TextClass, nonce: &str) -> String {
    const PROSE: &[&str] = &[
        "The archive team records each decision beside the evidence that justified it, so later readers can distinguish a stable rule from a temporary workaround.",
        "A useful memory names the actors, the constraint, and the consequence without pretending that an uncertain observation is a measured fact.",
        "During review, engineers compare neighboring reports and preserve the smallest reproducible example because concise context makes future retrieval reliable.",
        "The service accepts ordinary prose, follows the request through admission, and returns an envelope whose fingerprint identifies the exact serving lane.",
        "When a queue is quiet, a caller can attribute latency to the selected engine rather than to another consumer competing for the same execution permit.",
        "Operators keep the original wording of an incident, then add a short synthesis that explains what changed and which assumptions remain open.",
        "A durable record links measurements to the machine profile, model artifact, and software generation so that comparisons do not mix incompatible runs.",
        "Small passages describe decisions, observations, and next actions in a deliberately plain style that resembles the notes a production team would actually store.",
    ];
    let repetitions = match class {
        TextClass::Memory => 8,
        TextClass::Chunk => 120,
    };
    let mut text = (0..repetitions)
        .map(|part| {
            format!(
                "{} Passage {} belongs to item {}.",
                PROSE[part % PROSE.len()],
                part + 1,
                index
            )
        })
        .collect::<Vec<_>>()
        .join(" ");
    text.push_str(&format!(" Run nonce {nonce}."));
    text
}

fn epoch_ms() -> Result<u128> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("read wall clock")?
        .as_millis())
}

fn response_token_count(response: &Value) -> u64 {
    response["result"]["real_token_counts"]
        .as_array()
        .map(|counts| counts.iter().filter_map(Value::as_u64).sum())
        .unwrap_or(0)
}

fn response_submitted_token_count(response: &Value) -> u64 {
    response["result"]["truncation_disclosures"]
        .as_array()
        .map(|disclosures| {
            disclosures
                .iter()
                .filter_map(|disclosure| disclosure["submitted_tokens"].as_u64())
                .sum()
        })
        .unwrap_or_else(|| response_token_count(response))
}

fn ensure_response_vectors(response: &Value, expected: usize) -> Result<()> {
    let vectors = response["result"]["vectors"]
        .as_array()
        .with_context(|| format!("response has no vectors: {response}"))?;
    ensure!(
        vectors.len() == expected,
        "expected {expected} vectors, got {}",
        vectors.len()
    );
    Ok(())
}

fn ensure_response_ids(response: &Value, expected: usize) -> Result<()> {
    let vectors = response["result"]["vectors"]
        .as_array()
        .context("response has no vectors while checking ids")?;
    let mut ids = vectors
        .iter()
        .map(|vector| {
            vector["id"]
                .as_str()
                .map(str::to_string)
                .context("vector omitted id")
        })
        .collect::<Result<Vec<_>>>()?;
    ids.sort();
    let mut expected_ids = (0..expected)
        .map(|index| format!("item-{index}"))
        .collect::<Vec<_>>();
    expected_ids.sort();
    ensure!(ids == expected_ids, "response ids do not cover the batch");
    Ok(())
}
