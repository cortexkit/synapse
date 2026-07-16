#![forbid(unsafe_code)]

use std::{
    env,
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use subc_client_rs::{CallOptions, ConsumerOptions, SubcConsumer};
use subc_protocol::{BindIdentity, RouteTarget};
use tokio::time::sleep;

const DEFAULT_SUBC: &str = "/Users/[owner]/.local/share/cortexkit/run/subc-connection.json";
const DEFAULT_MODEL: &str = "gte-modernbert-base-f16";
const DEFAULT_BATCHES: &[usize] = &[8, 16, 32, 64, 128, 256];
const QUERY_SAMPLES: usize = 10;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

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

    let warmup = embedding_batch(&consumer, &identity, &args.model, 1).await?;
    ensure_response_vectors(&warmup, 1)?;

    let mut rows = Vec::with_capacity(args.batches.len());
    for &batch_size in &args.batches {
        let started = Instant::now();
        let response = embedding_batch(&consumer, &identity, &args.model, batch_size).await?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let token_count = response_token_count(&response).max(batch_size as u64);
        ensure_response_vectors(&response, batch_size)?;
        rows.push(json!({
            "batch": batch_size,
            "tokens": token_count,
            "elapsed_ms": elapsed_ms,
            "ms_per_item": elapsed_ms / batch_size as f64,
            "tokens_per_second": token_count as f64 / (elapsed_ms / 1_000.0),
        }));
    }

    let query_warmup = query_embedding(&consumer, &identity, &args.model).await?;
    ensure_response_vectors(&query_warmup, 1)?;
    let mut query_samples_ms = Vec::with_capacity(QUERY_SAMPLES);
    for _ in 0..QUERY_SAMPLES {
        let started = Instant::now();
        let response = query_embedding(&consumer, &identity, &args.model).await?;
        ensure_response_vectors(&response, 1)?;
        query_samples_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
    }
    query_samples_ms.sort_by(f64::total_cmp);
    let query_p50_ms = query_samples_ms[query_samples_ms.len() / 2];

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "model": args.model,
            "warmup": "one embed.batch outside timing",
            "rows": rows,
            "embed_query_p50_ms": query_p50_ms,
            "query_samples": query_samples_ms,
        }))?
    );
    consumer.close().await;
    Ok(())
}

struct Args {
    subc: PathBuf,
    model: String,
    batches: Vec<usize>,
}

impl Args {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Self> {
        let mut subc = env::var_os("SUBC_CONNECTION_FILE")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_SUBC));
        let mut model = DEFAULT_MODEL.to_string();
        let mut batches = DEFAULT_BATCHES.to_vec();
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
                "-h" | "--help" => {
                    println!(
                        "Usage: inline_embed_throughput [--subc PATH] [--model ID] [--batches 8,16,32,64,128,256]"
                    );
                    std::process::exit(0);
                }
                other => bail!("unknown argument {other}"),
            }
        }
        ensure!(!batches.is_empty() && batches.iter().all(|&size| size > 0));
        Ok(Self {
            subc,
            model,
            batches,
        })
    }
}

async fn embedding_batch(
    consumer: &SubcConsumer,
    identity: &BindIdentity,
    model: &str,
    batch_size: usize,
) -> Result<Value> {
    let items = (0..batch_size)
        .map(|index| json!({ "id": format!("item-{index}"), "text": fixture_text(index) }))
        .collect::<Vec<_>>();
    let response = call(
        consumer,
        identity,
        "embed.batch",
        json!({
            "model": model,
            "items": items,
            "accept_declared": true,
            "request_key": format!("inline-throughput-{}-{batch_size}", std::process::id()),
        }),
    )
    .await?;
    if let Some(job_id) = response["result"]["job_id"].as_str() {
        poll_job(consumer, identity, job_id).await
    } else {
        Ok(response)
    }
}

async fn query_embedding(
    consumer: &SubcConsumer,
    identity: &BindIdentity,
    model: &str,
) -> Result<Value> {
    call(
        consumer,
        identity,
        "embed.query",
        json!({
            "model": model,
            "id": "query",
            "text": fixture_text(0),
            "accept_declared": true,
        }),
    )
    .await
}

async fn poll_job(consumer: &SubcConsumer, identity: &BindIdentity, job_id: &str) -> Result<Value> {
    let deadline = Instant::now() + REQUEST_TIMEOUT;
    while Instant::now() < deadline {
        let page = call(
            consumer,
            identity,
            "embed.result",
            json!({ "job_id": job_id }),
        )
        .await?;
        match page["result"]["state"].as_str() {
            Some("done") => return Ok(page),
            Some("failed_transient" | "failed_permanent") => {
                bail!("embedding job {job_id} failed: {page}")
            }
            Some("queued" | "running") => sleep(Duration::from_millis(25)).await,
            other => bail!("unexpected embedding job state {other:?}: {page}"),
        }
    }
    bail!("embedding job {job_id} did not finish before {REQUEST_TIMEOUT:?}")
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
        bail!("{method} returned an error: {value}")
    }
    Ok(value)
}

fn fixture_text(index: usize) -> String {
    (0..45)
        .map(|word| format!("retrieval fixture item {index} token {word}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn response_token_count(response: &Value) -> u64 {
    response["result"]["real_token_counts"]
        .as_array()
        .map(|counts| counts.iter().filter_map(Value::as_u64).sum())
        .unwrap_or(0)
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
