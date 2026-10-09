//! Measure token preparation, inference and first-use compilation on an isolated Synapse daemon.
use super::{load_one_minute, qwen_compare, required_path, Workload};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    path::Path,
    time::{Duration, Instant},
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};

const LANE: &str = "qwen3-embedding-0.6b-ane";

async fn measured(
    candidate: &qwen_compare::Candidate,
    label: &str,
    texts: &[String],
) -> Result<Value> {
    let load = load_one_minute()?;
    let start = Instant::now();
    let body = qwen_compare::raw_call(
        &candidate.consumer,
        &candidate.identity,
        "embed.batch",
        json!({"model":LANE,"texts":texts,"accept_declared":true,"deadline_ms":600000}),
    )
    .await?;
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    let unix_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_micros();
    let error = sanitized_error(&body["result"]["error"]);
    if error.is_null() {
        let vectors = body["result"]["vectors"]
            .as_array()
            .context("embedding vectors")?;
        ensure!(vectors.len() == texts.len(), "embedding row count mismatch");
        for vector in vectors {
            let values = vector["vector"].as_array().context("embedding vector")?;
            ensure!(values.len() == 1024, "Qwen embedding dimension mismatch");
            let norm = values
                .iter()
                .map(|v| v.as_f64().unwrap_or(f64::NAN).powi(2))
                .sum::<f64>()
                .sqrt();
            ensure!(
                norm.is_finite() && (0.99..1.01).contains(&norm),
                "invalid embedding norm"
            );
        }
    }
    Ok(
        json!({"label":label,"load_1m":load,"load_end_1m":load_one_minute()?,"wall_ms":ms,
        "rows":texts.len(),"counts":body["result"]["real_token_counts"],
        "error":error,"unix_us":unix_us}),
    )
}

fn sanitized_error(error: &Value) -> Value {
    let mut error = error.clone();
    if let Some(fields) = error.as_object_mut() {
        if fields
            .get("message")
            .and_then(Value::as_str)
            .is_some_and(|s| s.contains("ane_lane_busy"))
        {
            fields.insert("cause".into(), json!("ane_lane_busy"));
        }
        // Free-form errors can contain user paths. Keep only wire classification
        // and the known lock-refusal cause in portable measurement evidence.
        fields.remove("message");
        fields.remove("details");
    }
    error
}

fn save(out: &Path, workload: &Workload, records: &[Value]) -> Result<()> {
    std::fs::write(
        out,
        serde_json::to_vec_pretty(&json!({"schema":1,
        "input_sha256":workload.input_sha256,"meta_sha256":workload.meta_sha256,"records":records}))?,
    )?;
    Ok(())
}

fn text_at_length(tokenizer: &tokenizers::Tokenizer, text: &str, target: usize) -> Result<String> {
    // Prefix exported AFT code chunks by UTF-8 boundaries. Include the one
    // end-of-sequence token appended by catalog serving, rather than guessing
    // token counts from bytes.
    let mut last = String::new();
    for (end, _) in text.char_indices().skip(1) {
        let prefix = &text[..end];
        let n = tokenizer
            .encode(prefix, false)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .len()
            + 1;
        if n == target {
            return Ok(prefix.into());
        }
        if n > target {
            break;
        }
        last = prefix.into();
    }
    anyhow::bail!(
        "real prefix did not reach target {target}, last bytes {}",
        last.len()
    )
}

async fn local_checkpoint(weights: &Path) -> Result<(String, tokio::task::JoinHandle<()>)> {
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
                let text = String::from_utf8_lossy(&request);
                let file = text
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("")
                    .rsplit('/')
                    .next()
                    .unwrap_or("");
                if !["model.safetensors", "tokenizer.json"].contains(&file) {
                    return;
                }
                if let Ok(mut file) = std::fs::File::open(root.join(file)) {
                    let size = file.metadata().unwrap().len();
                    let head = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {size}\r\nConnection: close\r\n\r\n"
                    );
                    if stream.write_all(head.as_bytes()).await.is_ok() {
                        let mut buffer = vec![0u8; 1024 * 1024];
                        loop {
                            let n = std::io::Read::read(&mut file, &mut buffer).unwrap_or(0);
                            if n == 0 || stream.write_all(&buffer[..n]).await.is_err() {
                                break;
                            }
                        }
                    }
                }
            });
        }
    });
    Ok((endpoint, server))
}

pub(super) async fn run(workload: &Workload, out: &Path) -> Result<()> {
    ensure!(std::env::var_os("TMPDIR").is_none(), "unset TMPDIR");
    ensure!(
        std::env::var("DEVELOPER_DIR").as_deref()
            == Ok("/Applications/Xcode.app/Contents/Developer"),
        "select Xcode"
    );
    let dir = out.parent().context("output directory")?;
    std::fs::create_dir_all(dir)?;
    let weights = required_path("SYNAPSE_QWEN_WEIGHTS")?.canonicalize()?;
    let assets = required_path("SYNAPSE_COMPARE_ASSETS")?.canonicalize()?;
    let (endpoint, server) = local_checkpoint(&weights).await?;
    let mut catalog: Value = serde_json::from_str(include_str!("../../src/catalog/models.json"))?;
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
                .retain(|b| b["backend"] == "ane");
            entry["files"]
                .as_array_mut()
                .unwrap()
                .retain(|f| f["backends"].as_array().unwrap().iter().any(|b| b == "ane"));
            for file in entry["files"].as_array_mut().unwrap() {
                file["backends"] = json!(["ane"]);
            }
        } else {
            entry["backends"] = json!([]);
            entry["files"] = json!([]);
            entry.as_object_mut().unwrap().remove("self_check");
        }
    }
    let catalog_path = std::env::current_dir()?
        .join("target")
        .join(format!("ane-profile-catalog-{}.json", std::process::id()));
    std::fs::create_dir_all(catalog_path.parent().unwrap())?;
    std::fs::write(&catalog_path, serde_json::to_vec(&catalog)?)?;
    std::env::set_var("SYNAPSE_TEST_CATALOG", &catalog_path);
    std::env::set_var("SYNAPSE_TEST_RUNNABLE_BACKENDS", "ane");
    std::env::set_var("SYNAPSE_ANE_PROFILE_ENDPOINT", endpoint);
    let candidate = qwen_compare::Candidate::start(
        &std::env::current_dir()?,
        &assets,
        &weights,
        "ane-lane-profile",
        ["profile-ane", "profile-metal"],
        64 * 8192,
    )
    .await?;
    let download = qwen_compare::call(
        &candidate.consumer,
        &candidate.identity,
        "models.download",
        json!({"catalog_id":"qwen3-embedding-0.6b","request_key":"profile-install"}),
    )
    .await?;
    let job = download["job_id"].as_str().context("download job")?;
    tokio::time::timeout(Duration::from_secs(600), async {
        loop {
            let status = qwen_compare::call(
                &candidate.consumer,
                &candidate.identity,
                "model.status",
                json!({"job_id":job}),
            )
            .await?;
            match status["state"].as_str() {
                Some("committed") => break,
                Some("failed" | "cancelled") => anyhow::bail!("private install failed: {status}"),
                _ => tokio::time::sleep(Duration::from_millis(250)).await,
            }
        }
        Ok::<_, anyhow::Error>(())
    })
    .await??;
    let tokenizer = tokenizers::Tokenizer::from_file(weights.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let corpus = workload
        .chunks
        .iter()
        .take(32)
        .map(|c| c.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let mut inputs = Vec::new();
    for length in [32, 128, 256, 512, 1024] {
        inputs.push(text_at_length(&tokenizer, &corpus, length)?);
    }
    let mut records = Vec::new();
    let mut ready = false;
    // Before the isolated model finishes loading and checking numerical accuracy,
    // resolution may return model_loading. Keep those attempts in the raw records
    // with initial-* labels so they cannot be mistaken for warm serving samples.
    for attempt in 0..100 {
        let record = measured(&candidate, &format!("initial-{attempt}"), &inputs[..1]).await?;
        ready = record["error"].is_null();
        let loading = record["error"]["code"] == "model_loading";
        records.push(record);
        save(out, workload, &records)?;
        if ready {
            break;
        }
        if !loading {
            candidate.shutdown().await?;
            server.abort();
            anyhow::bail!(
                "private lane could not load; see portable error classification in calls.json"
            );
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    if !ready {
        candidate.shutdown().await?;
        server.abort();
        anyhow::bail!("private lane did not become ready within 100 resolution attempts");
    }
    for (index, length) in [32, 128, 256, 512].into_iter().enumerate() {
        records.push(
            measured(
                &candidate,
                &format!("cold-{length}"),
                &inputs[index..index + 1],
            )
            .await?,
        );
        for repeat in 0..5 {
            records.push(
                measured(
                    &candidate,
                    &format!("warm-{length}-{repeat}"),
                    &inputs[index..index + 1],
                )
                .await?,
            );
        }
    }
    let long = measured(&candidate, "incident-cold-1024", &inputs[4..5]);
    let short = async {
        tokio::time::sleep(Duration::from_secs(1)).await;
        measured(&candidate, "incident-concurrent-short", &inputs[..1]).await
    };
    let (a, b) = tokio::join!(long, short);
    records.push(a?);
    records.push(b?);
    records.push(measured(&candidate, "incident-short-after", &inputs[..1]).await?);
    let batch = workload
        .batches
        .iter()
        .find(|b| b.chunk_count == 64)
        .context("64-row AFT batch")?;
    let texts = workload.chunks[batch.first_chunk_seq..batch.first_chunk_seq + 64]
        .iter()
        .map(|c| c.text.clone())
        .collect::<Vec<_>>();
    records.push(measured(&candidate, "aft64-cold", &texts).await?);
    records.push(measured(&candidate, "aft64-warm", &texts).await?);
    let (a, b) = tokio::join!(
        measured(&candidate, "aft64-overlap-a", &texts),
        measured(&candidate, "aft64-overlap-b", &texts)
    );
    records.push(a?);
    records.push(b?);

    save(out, workload, &records)?;
    let failed = records.iter().any(|r| {
        !r["error"].is_null()
            && !r["label"].as_str().unwrap_or("").starts_with("initial-")
            && !(r["label"] == "incident-concurrent-short" && r["error"]["code"] == "model_loading")
    });
    candidate.shutdown().await?;
    server.abort();
    ensure!(
        !failed,
        "one or more measured requests failed; see calls.json"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portable_errors_keep_lock_cause_without_user_paths() {
        let error = sanitized_error(
            &json!({"code":"engine_crashed","message":"ane_lane_busy: holds /private/user/cache/lock", "details":{"path":"/private/user/cache"}}),
        );
        assert_eq!(
            error,
            json!({"code":"engine_crashed","cause":"ane_lane_busy"})
        );
        assert!(sanitized_error(&Value::Null).is_null());
    }

    #[test]
    fn real_text_prefix_counts_the_catalog_terminal_token() {
        let tokenizer: tokenizers::Tokenizer = serde_json::from_str(r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"WhitespaceSplit"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"a":1,"b":2,"c":3},"unk_token":"[UNK]"}}"#).unwrap();
        let text = text_at_length(&tokenizer, "a b c", 3).unwrap();
        assert_eq!(tokenizer.encode(text, false).unwrap().len() + 1, 3);
    }
}
