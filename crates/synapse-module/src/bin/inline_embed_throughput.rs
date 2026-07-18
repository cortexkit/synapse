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
use tokio::time::sleep;

const DEFAULT_SUBC: &str = "/Users/[owner]/.local/share/cortexkit/run/subc-connection.json";
const DEFAULT_MODEL: &str = "gte-modernbert-base-f16";
const DEFAULT_BATCHES: &[usize] = &[8, 16, 32, 64, 128, 256];
const QUERY_SAMPLES: usize = 50;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);

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
    // Add a timestamp-plus-process nonce so each invocation measures actual
    // inference instead of receiving a cached idempotent response for the same
    // request key and digest.
    let nonce = format!(
        "{}-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("read invocation clock")?
            .as_nanos(),
        std::process::id()
    );

    let warmup = embedding_batch(&consumer, &identity, &args.model, 1, &nonce).await?;
    ensure_response_vectors(&warmup, 1)?;
    ensure_response_ids(&warmup, 1)?;

    let mut rows = Vec::with_capacity(args.batches.len());
    for &batch_size in &args.batches {
        let started = Instant::now();
        let response =
            embedding_batch(&consumer, &identity, &args.model, batch_size, &nonce).await?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let token_count = response_token_count(&response).max(batch_size as u64);
        ensure_response_vectors(&response, batch_size)?;
        ensure_response_ids(&response, batch_size)?;
        rows.push(json!({
            "batch": batch_size,
            "tokens": token_count,
            "elapsed_ms": elapsed_ms,
            "ms_per_item": elapsed_ms / batch_size as f64,
            "tokens_per_second": token_count as f64 / (elapsed_ms / 1_000.0),
        }));
    }

    let query_warmup = query_embedding(&consumer, &identity, &args.model, &nonce).await?;
    ensure_response_vectors(&query_warmup, 1)?;
    let idle_query_samples_ms = query_samples(&consumer, &identity, &args.model, &nonce).await?;
    let idle_query_p50_ms = percentile(&idle_query_samples_ms, 0.50);
    let idle_query_p95_ms = percentile(&idle_query_samples_ms, 0.95);

    let concurrent_query = if args.concurrent {
        Some(
            run_concurrent_batch_and_queries(
                &consumer,
                &identity,
                &args.model,
                &format!("{nonce}-concurrent"),
            )
            .await?,
        )
    } else {
        None
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "model": args.model,
            "concurrent": args.concurrent,
            "warmup": "one embed.batch outside timing",
            "rows": rows,
            "embed_query_idle_p50_ms": idle_query_p50_ms,
            "embed_query_idle_p95_ms": idle_query_p95_ms,
            "query_samples": idle_query_samples_ms,
            "concurrent": concurrent_query,
        }))?
    );
    consumer.close().await;
    Ok(())
}

struct Args {
    subc: PathBuf,
    model: String,
    batches: Vec<usize>,
    concurrent: bool,
}

impl Args {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Self> {
        let mut subc = env::var_os("SUBC_CONNECTION_FILE")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_SUBC));
        let mut model = DEFAULT_MODEL.to_string();
        let mut batches = DEFAULT_BATCHES.to_vec();
        let mut concurrent = false;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--subc" => {
                    subc = PathBuf::from(args.next().context("--subc requires a path")?);
                }
                "--model" => {
                    model = args.next().context("--model requires a model id")?;
                }
                "--concurrent" => concurrent = true,
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
                        "Usage: inline_embed_throughput [--subc PATH] [--model ID] [--batches 8,16,32,64,128,256] [--concurrent]"
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
            concurrent,
        })
    }
}

async fn start_embedding_batch(
    consumer: &SubcConsumer,
    identity: &BindIdentity,
    model: &str,
    batch_size: usize,
    nonce: &str,
) -> Result<Value> {
    let items = (0..batch_size)
        .map(|index| json!({ "id": format!("item-{index}"), "text": fixture_text(index, nonce) }))
        .collect::<Vec<_>>();
    call(
        consumer,
        identity,
        "embed.batch",
        json!({
            "model": model,
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
    nonce: &str,
) -> Result<Value> {
    let response = start_embedding_batch(consumer, identity, model, batch_size, nonce).await?;
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
    nonce: &str,
) -> Result<Value> {
    call(
        consumer,
        identity,
        "embed.query",
        json!({
            "model": model,
            "id": "query",
            "text": fixture_text(0, nonce),
            "accept_declared": true,
        }),
    )
    .await
}

async fn query_samples(
    consumer: &SubcConsumer,
    identity: &BindIdentity,
    model: &str,
    nonce: &str,
) -> Result<Vec<f64>> {
    let mut samples = Vec::with_capacity(QUERY_SAMPLES);
    for _ in 0..QUERY_SAMPLES {
        let started = Instant::now();
        let response = query_embedding(consumer, identity, model, nonce).await?;
        ensure_response_vectors(&response, 1)?;
        samples.push(started.elapsed().as_secs_f64() * 1_000.0);
    }
    samples.sort_by(f64::total_cmp);
    Ok(samples)
}

fn percentile(samples: &[f64], quantile: f64) -> f64 {
    let index = ((samples.len() as f64 * quantile).ceil() as usize)
        .saturating_sub(1)
        .min(samples.len().saturating_sub(1));
    samples[index]
}

async fn run_concurrent_batch_and_queries(
    consumer: &SubcConsumer,
    identity: &BindIdentity,
    model: &str,
    nonce: &str,
) -> Result<Value> {
    let accepted = start_embedding_batch(consumer, identity, model, 256, nonce).await?;
    let job_id = accepted["result"]["job_id"]
        .as_str()
        .context("--concurrent requires batch=256 to be job-shaped")?
        .to_string();
    let (batch, query) = tokio::join!(
        poll_job(consumer, identity, &job_id),
        query_samples(consumer, identity, model, nonce),
    );
    let batch = batch?;
    let query = query?;
    ensure_response_vectors(&batch, 256)?;
    ensure_response_ids(&batch, 256)?;
    Ok(json!({
        "batch": 256,
        "query_p50_ms": percentile(&query, 0.50),
        "query_p95_ms": percentile(&query, 0.95),
        "query_samples": query,
        "vectors": batch["result"]["vectors"].as_array().map(Vec::len),
    }))
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
        bail!("{method} returned an error: {value}")
    }
    Ok(value)
}

fn fixture_text(index: usize, nonce: &str) -> String {
    (0..45)
        .map(|word| format!("retrieval fixture item {index} token {word} run {nonce}"))
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
