//! Quiet-window comparison using an isolated daemon and production module.
//! No request is sent to a discovered or already-running Synapse instance.
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    env,
    path::PathBuf,
    process::Command,
    sync::Arc,
    time::{Duration, Instant},
};
use subc_client_rs::{CallOptions, ConsumerOptions, SubcConsumer};
use subc_daemon::{
    daemon_config::StorageConfig, serve_listener, ControlHandler, Registry, Router, ServerAuth,
};
use subc_protocol::{BindIdentity, RouteTarget};
use subc_transport::{
    generate_daemon_id, generate_key, write_atomic, ConnectionInfo, Endpoint, SCHEMA_VERSION,
};
use synapse_core::{dev_binary::ckdev_binary_hard_link, SanitizedTokenizer, TokenizerConfig};
use synapse_parity::{canonical::sha256_file, manifest::Manifest};
use tokio::net::TcpListener;

const MODEL: &str = "qwen3-embedding-0.6b";
const ARMS: [&str; 2] = ["ane", "metal"];

fn quiet() -> Result<Vec<f64>> {
    let output = Command::new("sysctl").args(["-n", "vm.loadavg"]).output()?;
    ensure!(output.status.success(), "sysctl vm.loadavg failed");
    let values = String::from_utf8(output.stdout)?
        .split_whitespace()
        .filter_map(|part| part.parse::<f64>().ok())
        .collect::<Vec<_>>();
    ensure!(
        quiet_load(&values),
        "quiet-window load gate refused: {values:?}"
    );
    Ok(values)
}

fn quiet_load(values: &[f64]) -> bool {
    values.len() == 3
        && values.iter().all(|value| value.is_finite())
        && (0.0..16.0).contains(&values[0])
}

async fn call(
    consumer: &SubcConsumer,
    identity: &BindIdentity,
    method: &str,
    params: Value,
) -> Result<Value> {
    let bytes = consumer
        .call(
            RouteTarget::ManagementSurface {
                module_id: "synapse".into(),
            },
            identity.clone(),
            serde_json::to_vec(&json!({"method":method,"params":params}))?,
            CallOptions {
                timeout: Duration::from_secs(600),
                ..CallOptions::default()
            },
        )
        .await?;
    let response: Value = serde_json::from_slice(&bytes)?;
    ensure!(
        !response["error"].is_object() && !response["result"]["error"].is_object(),
        "{method} refused: {response}"
    );
    Ok(response["result"].clone())
}

fn validate_vectors(response: &Value, rows: usize, tokens: Option<usize>) -> Result<()> {
    let vectors = response["vectors"]
        .as_array()
        .context("inline vectors missing (job diversion is not a measured call)")?;
    ensure!(vectors.len() == rows, "wrong vector count");
    for vector in vectors {
        let values = vector["vector"].as_array().context("vector missing")?;
        ensure!(
            values.len() == 1024
                && values
                    .iter()
                    .all(|value| value.as_f64().is_some_and(f64::is_finite)),
            "invalid Qwen vector"
        );
    }
    let counts = response["real_token_counts"]
        .as_array()
        .context("token counts missing")?;
    ensure!(counts.len() == rows, "wrong token count cardinality");
    for count in counts {
        let count = count.as_u64().context("invalid token count")? as usize;
        ensure!(
            tokens.map_or((100..=150).contains(&count), |expected| expected == count),
            "unexpected composed tokens: {count}"
        );
    }
    Ok(())
}

async fn timed(
    consumer: &SubcConsumer,
    identity: &BindIdentity,
    method: &str,
    params: Value,
    rows: usize,
    tokens: Option<usize>,
) -> Result<Value> {
    let before = quiet()?;
    let started = Instant::now();
    let response = call(consumer, identity, method, params).await?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    validate_vectors(&response, rows, tokens)?;
    let after = quiet()?;
    Ok(
        json!({"elapsed_ms":elapsed_ms,"load_before":before,"load_after":after,"real_token_counts":response["real_token_counts"],"fingerprint":response["fingerprint"]}),
    )
}

fn percentile(samples: &[f64], fraction: f64) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[((fraction * sorted.len() as f64).ceil() as usize).saturating_sub(1)]
}

fn median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

fn batch(arm: &str, chunks: &[String], key: &str) -> Value {
    json!({"model":format!("compare-{arm}"),"input_type":"document","accept_declared":true,"request_key":key,
        "items":chunks.iter().enumerate().map(|(id,text)| json!({"id":format!("row-{id}"),"text":text})).collect::<Vec<_>>()})
}

fn code_chunks(tokenizer: &SanitizedTokenizer, terminal: u32) -> Result<Vec<String>> {
    let mut chunks = Vec::new();
    for row in 0..64 {
        let mut text = format!("fn chunk_{row}(input: &[u8]) -> usize {{\n");
        loop {
            text.push_str(
                "    let value = input.iter().map(|byte| usize::from(*byte)).sum::<usize>();\n",
            );
            let ids = tokenizer
                .tokenizer()
                .encode(text.as_str(), true)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?;
            // Count the end-of-sequence token specified in bench/parity/models.json,
            // since the model receives it even when it is not part of the text.
            // Limit each row to 128 tokens so all 64 rows fit the 8192-token limit
            // for a direct response instead of being diverted to a background job.
            let composed = ids.len() + usize::from(ids.get_ids().last() != Some(&terminal));
            if composed >= 100 {
                ensure!(
                    composed <= 128,
                    "synthetic chunk overshot the inline budget: {composed}"
                );
                chunks.push(text);
                break;
            }
        }
    }
    Ok(chunks)
}

#[tokio::main]
async fn main() -> Result<()> {
    ensure!(cfg!(target_os = "macos"), "comparison requires this Mac");
    ensure!(env::var_os("TMPDIR").is_none(), "TMPDIR must be unset");
    ensure!(
        env::var("DEVELOPER_DIR").as_deref() == Ok("/Applications/Xcode.app/Contents/Developer"),
        "Xcode DEVELOPER_DIR required"
    );
    let load_start = quiet()?;
    let checkout = env::current_dir()?;
    let assets =
        PathBuf::from(env::var_os("SYNAPSE_COMPARE_ASSETS").context("SYNAPSE_COMPARE_ASSETS")?)
            .canonicalize()?;
    let weights =
        PathBuf::from(env::var_os("SYNAPSE_QWEN_WEIGHTS").context("SYNAPSE_QWEN_WEIGHTS")?)
            .canonicalize()?;
    let out = PathBuf::from(env::var_os("SYNAPSE_COMPARE_OUT").context("SYNAPSE_COMPARE_OUT")?);
    let manifest = Manifest::from_slice(include_bytes!("../../../bench/parity/models.json"))?;
    let pinned = manifest.model(MODEL)?;
    for (file, digest) in &pinned.files {
        ensure!(
            sha256_file(&weights.join(file))? == *digest,
            "original checkpoint digest mismatch: {file}"
        );
    }
    let fixture_id = synapse_parity::evaluator::fixture_set_id(&manifest, MODEL);
    let index: Value =
        serde_json::from_slice(include_bytes!("../../../bench/parity/fixtures/index.json"))?;
    let fixture_path = checkout.join("bench/parity").join(
        index[&fixture_id]["path"]
            .as_str()
            .context("fixture path")?,
    );
    ensure!(
        sha256_file(&fixture_path)?
            == index[&fixture_id]["sha256"]
                .as_str()
                .context("fixture seal")?,
        "fixture digest mismatch"
    );
    let fixtures: Value = serde_json::from_slice(&std::fs::read(fixture_path)?)?;
    let tokenizer = SanitizedTokenizer::from_file(
        weights.join("tokenizer.json"),
        TokenizerConfig {
            max_tokens: usize::MAX,
        },
    )?;
    let chunks = code_chunks(&tokenizer, pinned.grammar.terminal_tokens[0].id)?;
    let root = checkout
        .join("target")
        .join(format!("qwen-compare-{}", std::process::id()));
    std::fs::create_dir_all(root.join("data"))?;
    let package = root.join("qwen-ane.safetensors");
    std::fs::write(
        &package,
        synapse_parity::convert::convert_profile_file(
            &manifest,
            &format!("{MODEL}.ane-direct-worker"),
            &weights.join("model.safetensors"),
        )?,
    )?;
    let module = ckdev_binary_hard_link(assets.join("ck-synapse"), &root)?;
    let worker = ckdev_binary_hard_link(assets.join("ck-synapse-worker-ane-direct"), &root)?;
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let conn = ConnectionInfo {
        schema: SCHEMA_VERSION,
        endpoints: vec![Endpoint {
            host: "127.0.0.1".into(),
            port: listener.local_addr()?.port(),
        }],
        key: generate_key()?,
        daemon_id: generate_daemon_id()?,
        pid: std::process::id(),
        daemon_ver: "qwen-compare".into(),
        wire_version: Some(subc_protocol::PROTOCOL_VERSION),
    };
    let conn_path = root.join("connection.json");
    write_atomic(&conn_path, &conn)?;
    let router = Arc::new(Router::with_control_handler(Arc::new(
        ControlHandler::new(Arc::new(Registry::default())).with_storage_config(Some(
            StorageConfig::Sqlite {
                data_home: root.join("data"),
            },
        )),
    )));
    let daemon = tokio::spawn(serve_listener(
        listener,
        router,
        ServerAuth::new(conn.key.clone(), conn.daemon_id, conn.daemon_ver.clone()),
    ));
    let config = root.join("config.json");
    std::fs::write(
        &config,
        serde_json::to_vec(&json!({"preload_models":[
        {"model_id":"compare-ane","engine":"ane-direct-worker","profile":format!("{MODEL}.ane-direct-worker"),"task":"embed","model_path":package,"tokenizer_path":weights.join("tokenizer.json"),"pooling":"last","normalize":true,"worker_bin":worker,"execution":"explicit","attention_units":8192*8192},
        {"model_id":"compare-metal","engine":"owned-metal","profile":format!("{MODEL}.owned-metal"),"task":"embed","model_path":weights.join("model.safetensors"),"tokenizer_path":weights.join("tokenizer.json"),"pooling":"last","normalize":true,"execution":"explicit","attention_units":8192*8192}],
        "inline":{"max_items":64,"max_tokens":8192,"deadline_ms":600000,"max_queue_ms":600000,"max_concurrent_workers":2}}))?,
    )?;
    let mut child = synapse_core::without_launch_nonce_tokio(tokio::process::Command::new(module))
        .arg("--subc")
        .arg(&conn_path)
        .env("SUBC_MODULE_ID", "synapse")
        .env("SYNAPSE_CONFIG_PATH", config)
        .env("XDG_DATA_HOME", root.join("data"))
        .env("CORTEXKIT_LEASE_ROOT", root.join("leases"))
        .env("CORTEXKIT_STORE_ROOT", root.join("store"))
        .kill_on_drop(true)
        .spawn()?;
    let consumer = SubcConsumer::connect(&conn_path, ConsumerOptions::default()).await?;
    let identity = BindIdentity::new(
        checkout.clone(),
        "qwen-compare",
        format!("qwen-compare-{}", std::process::id()),
    );
    // Registration is asynchronous. This only polls our own candidate, never
    // daemon discovery, and the timeout bounds a candidate that cannot load.
    tokio::time::timeout(Duration::from_secs(600), async {
        loop {
            if call(&consumer, &identity, "models.list", json!({}))
                .await
                .is_ok()
            {
                break;
            }
            ensure!(
                child.try_wait()?.is_none(),
                "candidate exited before registering"
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        Ok::<_, anyhow::Error>(())
    })
    .await??;
    let mut latency = Vec::new();
    for tokens in [128, 512] {
        let case = fixtures["cases"]
            .as_array()
            .context("fixture cases")?
            .iter()
            .find(|case| {
                case["input_ids"]
                    .as_array()
                    .is_some_and(|ids| ids.len() == tokens)
            })
            .context("exact-length fixture")?;
        for arm in ARMS {
            timed(&consumer, &identity, "embed.query", json!({"model":format!("compare-{arm}"),"text":case["text"],"accept_declared":true}), 1, Some(tokens)).await?;
        }
        let mut samples = [Vec::new(), Vec::new()];
        for repetition in 0..9 {
            for index in if repetition % 2 == 0 { [0, 1] } else { [1, 0] } {
                let arm = ARMS[index];
                samples[index].push(timed(&consumer, &identity, "embed.query", json!({"model":format!("compare-{arm}"),"text":case["text"],"accept_declared":true}), 1, Some(tokens)).await?);
            }
        }
        let medians = samples
            .iter()
            .map(|samples| {
                median(
                    &samples
                        .iter()
                        .map(|sample| sample["elapsed_ms"].as_f64().unwrap())
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        latency.push(json!({"tokens":tokens,"samples":{"ane":samples[0],"metal":samples[1]},"median_ms":{"ane":medians[0],"metal":medians[1]},"ane_over_metal":medians[0]/medians[1]}));
    }
    let mut throughput = Vec::new();
    for arm in ARMS {
        timed(
            &consumer,
            &identity,
            "embed.batch",
            batch(arm, &chunks, &format!("warm-{}-{arm}", std::process::id())),
            64,
            None,
        )
        .await?;
        let mut samples = Vec::new();
        let mut total_seconds = 0.0;
        for pair in 0..10 {
            let started = Instant::now();
            let (a, b) = tokio::join!(
                timed(
                    &consumer,
                    &identity,
                    "embed.batch",
                    batch(
                        arm,
                        &chunks,
                        &format!("{}-{arm}-{pair}-a", std::process::id())
                    ),
                    64,
                    None
                ),
                timed(
                    &consumer,
                    &identity,
                    "embed.batch",
                    batch(
                        arm,
                        &chunks,
                        &format!("{}-{arm}-{pair}-b", std::process::id())
                    ),
                    64,
                    None
                )
            );
            total_seconds += started.elapsed().as_secs_f64();
            samples.extend([a?, b?]);
        }
        let times = samples
            .iter()
            .map(|sample| sample["elapsed_ms"].as_f64().unwrap())
            .collect::<Vec<_>>();
        throughput.push(json!({"arm":arm,"rows":1280,"calls":20,"calls_in_flight":2,"rows_per_minute":1280.0*60.0/total_seconds,"call_p50_ms":percentile(&times,0.5),"call_p90_ms":percentile(&times,0.9),"samples":samples}));
    }
    let report = json!({"load_start":load_start,"load_end":quiet()?,"source_commit":String::from_utf8(Command::new("git").args(["rev-parse","HEAD"]).output()?.stdout)?.trim(),"candidate_sha256":sha256_file(&assets.join("ck-synapse"))?,"worker_sha256":sha256_file(&assets.join("ck-synapse-worker-ane-direct"))?,"latency":latency,"passes_3x_at_512":latency[1]["ane_over_metal"].as_f64().unwrap() <= 3.0,"throughput":throughput,"ane_dispatch":"one row at a time; one executable evaluation per layer per row (worker.rs embed loop)"});
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&out, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    consumer.close().await;
    child.start_kill()?;
    child.wait().await?;
    daemon.abort();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quiet_window_excludes_sixteen_and_missing_load() {
        assert!(quiet_load(&[15.99, 50.0, 60.0]));
        assert!(!quiet_load(&[16.0, 1.0, 1.0]));
        assert!(!quiet_load(&[36.0, 1.0, 1.0]));
        assert!(!quiet_load(&[]));
        assert!(!quiet_load(&[f64::NAN, 1.0, 1.0]));
    }

    #[test]
    fn call_percentiles_are_nearest_rank_not_mean() {
        assert_eq!(percentile(&[10.0, 1.0, 9.0, 2.0], 0.5), 2.0);
        assert_eq!(percentile(&[10.0, 1.0, 9.0, 2.0], 0.9), 10.0);
    }

    #[test]
    fn latency_medians_handle_odd_and_even_sample_counts() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 2.0, 8.0]), 3.0);
    }

    #[test]
    #[ignore = "requires original Qwen tokenizer; no accelerator inference"]
    fn synthetic_code_chunks_fit_the_aft_shape_and_inline_budget() {
        let weights =
            PathBuf::from(env::var_os("SYNAPSE_QWEN_WEIGHTS").expect("SYNAPSE_QWEN_WEIGHTS"));
        let tokenizer = SanitizedTokenizer::from_file(
            weights.join("tokenizer.json"),
            TokenizerConfig {
                max_tokens: usize::MAX,
            },
        )
        .unwrap();
        let manifest =
            Manifest::from_slice(include_bytes!("../../../bench/parity/models.json")).unwrap();
        let terminal = manifest.model(MODEL).unwrap().grammar.terminal_tokens[0].id;
        let chunks = code_chunks(&tokenizer, terminal).unwrap();
        assert_eq!(chunks.len(), 64);
        let mut total = 0;
        for text in &chunks {
            let ids = tokenizer.tokenizer().encode(text.as_str(), true).unwrap();
            let count = ids.len() + usize::from(ids.get_ids().last() != Some(&terminal));
            assert!((100..=150).contains(&count));
            total += count;
        }
        assert!(total <= 8192);
        println!("verified 64 code-like rows, {total} composed tokens, <=8192 inline budget");
    }
}
