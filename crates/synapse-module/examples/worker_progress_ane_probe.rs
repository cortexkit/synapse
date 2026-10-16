//! Runs Qwen3 at 128, 512 and 8192 tokens through a private daemon started here.
//! The module log's worker_liveness gap_ms measures time between completed-work
//! frames; SYNAPSE_PROGRESS_PROBE_OUT records request times, lengths and norms.
//! Use a Mac reserved for this measurement with no competing ANE work, which
//! would skew the gaps. No production discovery or global-lock bypass is used.
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};
#[path = "support/qwen_compare.rs"]
mod qwen_compare;

async fn mirror(weights: &Path) -> Result<(String, tokio::task::JoinHandle<()>)> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let endpoint = format!("http://127.0.0.1:{}", listener.local_addr()?.port());
    let root = weights.to_path_buf();
    let server = tokio::spawn(async move {
        while let Ok((mut stream, _)) = listener.accept().await {
            let root = root.clone();
            tokio::spawn(async move {
                let mut request = Vec::new();
                let mut byte = [0u8; 1];
                while request.len() < 8192 && !request.ends_with(b"\r\n\r\n") {
                    if stream.read_exact(&mut byte).await.is_err() {
                        return;
                    }
                    request.push(byte[0]);
                }
                let request = String::from_utf8_lossy(&request);
                let name = request
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("")
                    .rsplit('/')
                    .next()
                    .unwrap_or("");
                if !["model.safetensors", "tokenizer.json", "config.json"].contains(&name) {
                    return;
                }
                if let Ok(mut file) = std::fs::File::open(root.join(name)) {
                    let Ok(metadata) = file.metadata() else {
                        return;
                    };
                    let header = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        metadata.len()
                    );
                    if stream.write_all(header.as_bytes()).await.is_err() {
                        return;
                    }
                    let mut buffer = vec![0; 1024 * 1024];
                    loop {
                        let n = std::io::Read::read(&mut file, &mut buffer).unwrap_or(0);
                        if n == 0 || stream.write_all(&buffer[..n]).await.is_err() {
                            break;
                        }
                    }
                }
            });
        }
    });
    Ok((endpoint, server))
}

fn path(name: &str) -> Result<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .with_context(|| format!("set {name}"))
}

#[tokio::main]
async fn main() -> Result<()> {
    ensure!(cfg!(target_os = "macos"), "Mac-only probe");
    ensure!(std::env::var_os("TMPDIR").is_none(), "leave TMPDIR unset");
    ensure!(
        std::env::var("DEVELOPER_DIR").as_deref()
            == Ok("/Applications/Xcode.app/Contents/Developer"),
        "select Xcode"
    );
    let weights = path("SYNAPSE_QWEN_WEIGHTS")?.canonicalize()?;
    let assets = path("SYNAPSE_COMPARE_ASSETS")?.canonicalize()?;
    let out = path("SYNAPSE_PROGRESS_PROBE_OUT")?;
    let (endpoint, server) = mirror(&weights).await?;
    let mut catalog: Value = serde_json::from_str(include_str!("../src/catalog/models.json"))?;
    let entries = catalog["models"].as_array_mut().context("catalog models")?;
    entries.retain(|entry| {
        entry["id"] == "qwen3-embedding-0.6b"
            || entry["default_for_task"] == true && entry["task"] == "rerank"
    });
    for entry in entries {
        if entry["task"] == "embed" {
            entry["default_for_task"] = json!(true);
            entry["backends"]
                .as_array_mut()
                .unwrap()
                .retain(|backend| backend["backend"] == "ane");
            entry["files"].as_array_mut().unwrap().retain(|file| {
                file["backends"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|backend| backend == "ane")
            });
            for file in entry["files"].as_array_mut().unwrap() {
                file["backends"] = json!(["ane"]);
            }
        } else {
            entry["backends"] = json!([]);
            entry["files"] = json!([]);
            entry.as_object_mut().unwrap().remove("self_check");
        }
    }
    let root = std::env::current_dir()?
        .join("target")
        .join(format!("progress-probe-{}", std::process::id()));
    std::fs::create_dir_all(&root)?;
    let catalog_path = root.join("catalog.json");
    std::fs::write(&catalog_path, serde_json::to_vec(&catalog)?)?;
    std::env::set_var("SYNAPSE_TEST_CATALOG", catalog_path);
    std::env::set_var("SYNAPSE_TEST_RUNNABLE_BACKENDS", "ane");
    std::env::set_var("SYNAPSE_ANE_PROFILE_CATALOG", "1");
    std::env::set_var("SYNAPSE_ANE_PROFILE_ENDPOINT", endpoint);
    std::env::set_var("CORTEXKIT_MODEL_CACHE", root.join("model-cache"));
    // Per-frame progress lines are trace-level, so the candidate module (which
    // inherits this environment) must raise worker_liveness to record the gaps.
    std::env::set_var("CK_LOG", "info,worker_liveness=trace");
    let candidate = qwen_compare::Candidate::start(
        &std::env::current_dir()?,
        &assets,
        &weights,
        "worker-progress-private",
        ["unused-ane", "unused-metal"],
        8192,
    )
    .await?;
    let downloaded = qwen_compare::call(
        &candidate.consumer,
        &candidate.identity,
        "models.download",
        json!({"catalog_id":"qwen3-embedding-0.6b","request_key":"progress-install"}),
    )
    .await?;
    let job = downloaded["job_id"].as_str().context("download job")?;
    tokio::time::timeout(Duration::from_secs(600), async {
        loop {
            let status = qwen_compare::call(
                &candidate.consumer,
                &candidate.identity,
                "model.status",
                json!({"job_id":job}),
            )
            .await?;
            if status["state"] == "committed" {
                return Ok::<_, anyhow::Error>(());
            }
            ensure!(
                status["state"] != "failed" && status["state"] != "cancelled",
                "private install failed: {status}"
            );
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    })
    .await??;
    let tokenizer = tokenizers::Tokenizer::from_file(weights.join("tokenizer.json"))
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let mut records = Vec::new();
    for length in [128, 512, 8192] {
        let text = vec!["hello"; length - 1].join(" ");
        let tokens = tokenizer
            .encode(text.as_str(), false)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?
            .len()
            + 1;
        ensure!(
            tokens == length,
            "synthetic probe composition is {tokens}, expected {length}"
        );
        for repeat in 0..2 {
            let started = Instant::now();
            let mut body;
            loop {
                body = qwen_compare::raw_call(&candidate.consumer, &candidate.identity, "embed.batch",
                    json!({"model":"qwen3-embedding-0.6b-ane","texts":[text],"accept_declared":true,"deadline_ms":600000})).await?;
                if body["result"]["error"]["code"] != "model_loading" {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
            ensure!(
                body["error"].is_null() && body["result"]["error"].is_null(),
                "probe refused: {body}"
            );
            ensure!(
                body["result"]["real_token_counts"] == json!([length]),
                "worker input length differs: {body}"
            );
            let vectors = body["result"]["vectors"].as_array().context("vectors")?;
            ensure!(vectors.len() == 1, "one probe row");
            let vector = vectors[0]["vector"].as_array().context("vector")?;
            ensure!(vector.len() == 1024, "Qwen dimension");
            let norm = vector
                .iter()
                .map(|value| value.as_f64().unwrap_or(f64::NAN).powi(2))
                .sum::<f64>()
                .sqrt();
            ensure!(
                norm.is_finite() && (0.99..1.01).contains(&norm),
                "invalid output norm"
            );
            records.push(json!({"tokens":length,"repeat":repeat,"wall_ms":started.elapsed().as_secs_f64()*1000.0,"norm":norm}));
            std::fs::write(
                &out,
                serde_json::to_vec_pretty(&json!({"private_daemon":true,"records":records}))?,
            )?;
            println!("progress probe tokens={length} repeat={repeat} completed");
        }
    }
    candidate.shutdown().await?;
    server.abort();
    Ok(())
}
