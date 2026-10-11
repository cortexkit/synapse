//! Matched full-export serving measurements against an isolated catalog ANE lane.
//! See docs/evidence/ane-tight-packing/README.md for approval and signing requirements.
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    env,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use subc_client_rs::{ConsumerOptions, SubcConsumer};
use subc_daemon::{
    daemon_config::StorageConfig, serve_listener, ControlHandler, Registry, Router, ServerAuth,
};
use subc_protocol::BindIdentity;
use subc_transport::{
    generate_daemon_id, generate_key, write_atomic, ConnectionInfo, Endpoint, SCHEMA_VERSION,
};
use synapse_parity::canonical::sha256_file;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    task::JoinSet,
};

#[allow(dead_code)]
#[path = "support/qwen_compare.rs"]
mod qwen_compare;

const ROWS: usize = 6341;
const LANE: &str = "qwen3-embedding-0.6b-ane";
const PIN: &str = "5f7ac7cdbf63e71f04405b2130b2bde8eb7b307e5d929419bcd747b129806a6b";
const DIM: usize = 1024;
const IN_FLIGHT: usize = 2;

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
struct Workload {
    chunks: Vec<Chunk>,
    batches: Vec<Batch>,
    input_sha256: String,
    meta_sha256: String,
}

fn validate_plan(batches: &[Batch], rows: usize) -> Result<()> {
    let mut covered = vec![false; rows];
    let mut ids = BTreeSet::new();
    ensure!(rows > 0 && !batches.is_empty(), "empty export");
    for batch in batches {
        ensure!(ids.insert(batch.seq), "duplicate batch id");
        ensure!(
            (1..=64).contains(&batch.chunk_count),
            "batch exceeds AFT's 64-row call shape"
        );
        let end = batch
            .first_chunk_seq
            .checked_add(batch.chunk_count)
            .context("batch overflow")?;
        ensure!(end <= rows, "batch out of bounds");
        for row in &mut covered[batch.first_chunk_seq..end] {
            ensure!(!*row, "overlapping export batches");
            *row = true;
        }
    }
    ensure!(covered.iter().all(|v| *v), "export batch gap");
    Ok(())
}
impl Workload {
    fn load(path: &Path) -> Result<Self> {
        let mut meta = path.as_os_str().to_os_string();
        meta.push(".meta.json");
        let meta = PathBuf::from(meta);
        let input = std::fs::read_to_string(path)?;
        let chunks = input
            .lines()
            .enumerate()
            .map(|(seq, line)| {
                let row: Chunk =
                    serde_json::from_str(line).with_context(|| format!("invalid row {seq}"))?;
                ensure!(row.seq == seq, "export row sequence mismatch");
                Ok(row)
            })
            .collect::<Result<Vec<_>>>()?;
        let metadata: Metadata = serde_json::from_slice(&std::fs::read(&meta)?)?;
        ensure!(
            chunks.len() == ROWS && metadata.batches.len() == 127,
            "expected full 6,341-row / 127-call export"
        );
        validate_plan(&metadata.batches, ROWS)?;
        Ok(Self {
            chunks,
            batches: metadata.batches,
            input_sha256: sha256_file(path)?,
            meta_sha256: sha256_file(&meta)?,
        })
    }
}

fn percentile(values: &[f64], p: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[((p * sorted.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1)]
}
fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}
#[derive(Serialize)]
struct Latencies {
    count: usize,
    p50_ms: f64,
    p99_ms: f64,
}
fn latencies(values: &[f64]) -> Latencies {
    Latencies {
        count: values.len(),
        p50_ms: percentile(values, 0.50),
        p99_ms: percentile(values, 0.99),
    }
}
fn signed_copy(source: &Path, root: &Path, name: &str) -> Result<PathBuf> {
    let path = root.join(name);
    std::fs::copy(source, &path)?;
    ensure!(
        Command::new("codesign")
            .args(["--force", "--sign", "-"])
            .arg(&path)
            .status()?
            .success(),
        "ad-hoc signing failed"
    );
    ensure!(
        Command::new("codesign")
            .args(["--verify", "--strict"])
            .arg(&path)
            .status()?
            .success(),
        "signature verification failed"
    );
    Ok(path)
}
fn hardware_preflight() -> Result<()> {
    let output = Command::new("ps").args(["-axo", "pid=,comm="]).output()?;
    ensure!(output.status.success(), "cannot inspect ANE process owners");
    for line in String::from_utf8(output.stdout)?.lines() {
        let mut fields = line.split_whitespace();
        let pid = fields.next().unwrap_or("");
        let name = fields.collect::<Vec<_>>().join(" ").to_lowercase();
        if pid == std::process::id().to_string() {
            continue;
        }
        ensure!(
            !["ck-synapse", "ck_synapse", "ckdev", "synapse-worker"]
                .iter()
                .any(|needle| name.contains(needle)),
            "ANE window busy: pid {pid}; ask the owner, never stop or bypass it"
        );
    }
    Ok(())
}

async fn checkpoint_server(weights: &Path) -> Result<(String, tokio::task::JoinHandle<()>)> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let endpoint = format!("http://127.0.0.1:{}", listener.local_addr()?.port());
    let weights = weights.to_path_buf();
    let task = tokio::spawn(async move {
        while let Ok((mut stream, _)) = listener.accept().await {
            let weights = weights.clone();
            tokio::spawn(async move {
                let mut request = Vec::new();
                let mut byte = [0; 1];
                while request.len() < 8192 && !request.ends_with(b"\r\n\r\n") {
                    if stream.read_exact(&mut byte).await.is_err() {
                        return;
                    }
                    request.push(byte[0]);
                }
                let text = String::from_utf8_lossy(&request);
                let filename = text
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("")
                    .rsplit('/')
                    .next()
                    .unwrap_or("");
                if !["model.safetensors", "tokenizer.json"].contains(&filename) {
                    return;
                }
                if let Ok(mut file) = std::fs::File::open(weights.join(filename)) {
                    if let Ok(meta) = file.metadata() {
                        let header = format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            meta.len()
                        );
                        if stream.write_all(header.as_bytes()).await.is_ok() {
                            let mut buffer = vec![0u8; 1024 * 1024];
                            loop {
                                let n = std::io::Read::read(&mut file, &mut buffer).unwrap_or(0);
                                if n == 0 || stream.write_all(&buffer[..n]).await.is_err() {
                                    break;
                                }
                            }
                        }
                    }
                }
            });
        }
    });
    Ok((endpoint, task))
}

struct Candidate {
    consumer: SubcConsumer,
    identity: BindIdentity,
    child: tokio::process::Child,
    daemon: tokio::task::JoinHandle<()>,
    server: tokio::task::JoinHandle<()>,
    profile_dir: PathBuf,
}
impl Drop for Candidate {
    fn drop(&mut self) {
        self.daemon.abort();
        self.server.abort();
    }
}
impl Candidate {
    async fn call(&self, method: &str, params: Value) -> Result<Value> {
        // Keep the refusal code and diagnostic message, not the request or response payload.
        let response =
            qwen_compare::raw_call(&self.consumer, &self.identity, method, params).await?;
        for error in [&response["error"], &response["result"]["error"]] {
            ensure!(
                !error.is_object(),
                "{method}: {}: {}",
                error["code"].as_str().unwrap_or("unclassified_refusal"),
                error["message"].as_str().unwrap_or("no error message")
            );
        }
        Ok(response["result"].clone())
    }
    async fn start(root: &Path, assets: &Path, weights: &Path, packing: bool) -> Result<Self> {
        hardware_preflight()?;
        ensure!(!root.exists(), "private run directory already exists");
        std::fs::create_dir_all(root.join("data"))?;
        let profile_dir = root.join("worker-profile");
        std::fs::create_dir_all(&profile_dir)?;
        let module = signed_copy(&assets.join("ck-synapse"), root, "ckdev-ane-packing-module")?;
        let worker = signed_copy(
            &assets.join("ck-synapse-worker-ane-direct"),
            root,
            "ckdev-ane-packing-worker",
        )?;
        let proxy = signed_copy(&env::current_exe()?, root, "ckdev-ane-packing-worker-proxy")?;
        let (endpoint, server) = checkpoint_server(weights).await?;
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
        let catalog_path = root.join("catalog.json");
        std::fs::write(&catalog_path, serde_json::to_vec(&catalog)?)?;
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
            daemon_ver: "ane-packing-serving".into(),
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
        let config = root.join("config.json");
        // Only the packing switch differs between arms; the serving quantum remains the default eight rows.
        std::fs::write(
            &config,
            serde_json::to_vec(
                &json!({"hf_endpoint":endpoint,"ane_direct":{"tight_packing":packing},"certify_observation":true,"inline":{"max_items":64,"max_tokens":64*8192,"deadline_ms":600000,"max_queue_ms":600000,"max_concurrent_workers":2}}),
            )?,
        )?;
        let child = synapse_core::without_launch_nonce_tokio(tokio::process::Command::new(module))
            .arg("--subc")
            .arg(&conn_path)
            .env("SUBC_MODULE_ID", "synapse")
            .env("SYNAPSE_CONFIG_PATH", config)
            .env("XDG_DATA_HOME", root.join("data"))
            .env("CORTEXKIT_LEASE_ROOT", root.join("leases"))
            .env("CORTEXKIT_STORE_ROOT", root.join("store"))
            .env("SYNAPSE_TEST_CATALOG", catalog_path)
            .env("SYNAPSE_TEST_RUNNABLE_BACKENDS", "ane")
            .env("SYNAPSE_ANE_PROFILE_DIR", &profile_dir)
            .env("SYNAPSE_PACKING_REAL_WORKER", worker)
            .env("SYNAPSE_PACKING_WIRE_LOG", root.join("wire.jsonl"))
            .env(
                synapse_core::worker_binary_env_var("ane-direct-worker"),
                proxy,
            )
            .stdout(Stdio::null())
            .stderr(std::fs::File::create(root.join("module.stderr"))?)
            .kill_on_drop(true)
            .spawn()?;
        let daemon = tokio::spawn(async move {
            let _ = serve_listener(
                listener,
                router,
                ServerAuth::new(conn.key, conn.daemon_id, conn.daemon_ver),
            )
            .await;
        });
        let consumer = SubcConsumer::connect(&conn_path, ConsumerOptions::default()).await?;
        let identity = BindIdentity::new(
            env::current_dir()?,
            "ane-packing-serving",
            format!("ane-packing-{}", std::process::id()),
        );
        let mut candidate = Self {
            consumer,
            identity,
            child,
            daemon,
            server,
            profile_dir,
        };
        tokio::time::timeout(Duration::from_secs(600), async {
            loop {
                if candidate.call("models.list", json!({})).await.is_ok() {
                    break;
                }
                ensure!(
                    candidate.child.try_wait()?.is_none(),
                    "private module exited before registering"
                );
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            Ok::<_, anyhow::Error>(())
        })
        .await??;
        let download = candidate
            .call(
                "models.download",
                json!({"catalog_id":"qwen3-embedding-0.6b","request_key":"packing-install"}),
            )
            .await?;
        let job = download["job_id"].as_str().context("download job")?;
        tokio::time::timeout(Duration::from_secs(600), async {
            loop {
                let status = candidate
                    .call("model.status", json!({"job_id":job}))
                    .await?;
                match status["state"].as_str() {
                    Some("committed") => break,
                    Some("failed" | "cancelled") => anyhow::bail!("private catalog install failed"),
                    _ => tokio::time::sleep(Duration::from_millis(100)).await,
                }
            }
            Ok::<_, anyhow::Error>(())
        })
        .await??;
        Ok(candidate)
    }
    async fn observations(&self) -> Result<Value> {
        let mut observation = self.call("certify.observations", json!({})).await?;
        observation["wire"] =
            wire_snapshot(&self.profile_dir.parent().unwrap().join("wire.jsonl"))?;
        Ok(observation)
    }
    async fn shutdown(mut self) -> Result<()> {
        self.call("model.unload", json!({"model_id":LANE})).await?;
        self.consumer.close().await;
        if self.child.try_wait()?.is_none() {
            self.child.start_kill()?;
        }
        self.child.wait().await?;
        Ok(())
    }
}

fn vectors(response: &Value, seqs: &[usize], query: bool) -> Result<Vec<Vec<u32>>> {
    ensure!(
        response["fingerprint"] == PIN,
        "catalog fingerprint changed"
    );
    let entries = if query {
        vec![json!({"id":format!("row-{}",seqs[0]),"vector":response["vector"]})]
    } else {
        response["vectors"]
            .as_array()
            .context("inline vectors missing")?
            .clone()
    };
    ensure!(entries.len() == seqs.len(), "vector count mismatch");
    let mut ordered = vec![None; seqs.len()];
    for entry in entries {
        let id = entry["id"].as_str().context("vector id")?;
        let index = seqs
            .iter()
            .position(|seq| id == format!("row-{seq}"))
            .context("unknown vector id")?;
        ensure!(ordered[index].is_none(), "duplicate vector id");
        let bits = entry["vector"]
            .as_array()
            .context("vector array")?
            .iter()
            .map(|v| {
                let f = v.as_f64().context("vector number")? as f32;
                ensure!(f.is_finite(), "nonfinite vector");
                Ok(f.to_bits())
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(bits.len() == DIM, "vector dimension mismatch");
        ordered[index] = Some(bits);
    }
    ordered
        .into_iter()
        .map(|v| v.context("missing vector id"))
        .collect()
}
async fn embed(
    candidate: &Candidate,
    work: &Workload,
    batch: &Batch,
    key: &str,
) -> Result<Vec<Vec<u32>>> {
    let chunks = &work.chunks[batch.first_chunk_seq..batch.first_chunk_seq + batch.chunk_count];
    let params = json!({"model":LANE,"input_type":"document","accept_declared":true,"request_key":key,
        "items":chunks.iter().map(|c|json!({"id":format!("row-{}",c.seq),"text":c.text})).collect::<Vec<_>>()});
    let response = candidate.call("embed.batch", params).await?;
    vectors(
        &response,
        &chunks.iter().map(|c| c.seq).collect::<Vec<_>>(),
        false,
    )
}
async fn query(candidate: &Candidate, chunk: &Chunk) -> Result<Vec<u32>> {
    let response = candidate
        .call(
            "embed.query",
            json!({"model":LANE,"text":chunk.text,"accept_declared":true}),
        )
        .await?;
    Ok(vectors(&response, &[chunk.seq], true)?.remove(0))
}
fn packed_requests(observation: &Value) -> u64 {
    observation["wire"]["EMBED_PACKED"].as_u64().unwrap_or(0)
}
fn compile_records(dir: &Path) -> Result<Vec<f64>> {
    let mut records = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().and_then(|v| v.to_str()) != Some("jsonl") {
            continue;
        }
        for line in std::fs::read_to_string(path)?.lines() {
            // A concurrent append may expose an incomplete final line; it is retried on the next poll.
            if let Ok(record) = serde_json::from_str::<Value>(line) {
                if record["req_id"]
                    .as_str()
                    .is_some_and(|id| id.contains("admit-packed"))
                {
                    records.push(record["wall_ms"].as_f64().context("compile wall time")?);
                }
            }
        }
    }
    Ok(records)
}
async fn warm(candidate: &Candidate, work: &Workload, packing: bool) -> Result<Value> {
    let batch = work
        .batches
        .iter()
        .find(|batch| batch.chunk_count == 64)
        .context("full AFT warm-up batch")?;
    tokio::time::timeout(Duration::from_secs(600), async {
        loop {
            match embed(candidate, work, batch, "warm-ready").await {
                Ok(_) => break,
                Err(error) if error.to_string().contains("model_loading") => {
                    tokio::time::sleep(Duration::from_millis(250)).await
                }
                Err(error) => return Err(error),
            }
        }
        if packing {
            // Compiling the packed graph is not enough: the module first compares
            // packed fixture vectors with independent standalone vectors. Wait for
            // those checks, then require an ordinary module call to send EMBED_PACKED.
            loop {
                let observation = candidate.observations().await?;
                if !compile_records(&candidate.profile_dir)?.is_empty()
                    && packed_requests(&observation)
                        >= synapse_core::packed_qualification_fixtures_v1().len() as u64
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
            let before = packed_requests(&candidate.observations().await?);
            embed(candidate, work, batch, "warm-qualified-dispatch").await?;
            ensure!(
                packed_requests(&candidate.observations().await?) > before,
                "packing did not qualify or dispatch"
            );
        }
        candidate.observations().await
    })
    .await?
}

#[derive(Serialize)]
struct Arm {
    repetition: usize,
    packing: bool,
    export_seconds: f64,
    calls_ms: Vec<f64>,
    call_latency: Latencies,
    identical_rows: usize,
    mismatched_rows: Vec<usize>,
    vectors_sha256: String,
    query_ms: Vec<f64>,
    query_latency: Latencies,
    identical_queries: usize,
    background_calls: usize,
    background_identical_rows: usize,
    background_mismatched_rows: usize,
    packed_compile_request_ms: Vec<f64>,
    observations_before: Value,
    observations_after_export: Value,
    observations_after_queries: Value,
}
fn vector_digest(rows: &[Vec<u32>]) -> String {
    let mut hash = Sha256::new();
    for row in rows {
        for value in row {
            hash.update(value.to_le_bytes());
        }
    }
    hex::encode(hash.finalize())
}
async fn export(
    candidate: Arc<Candidate>,
    work: Arc<Workload>,
    label: &str,
) -> Result<(f64, Vec<f64>, Vec<Vec<u32>>)> {
    let wall = Instant::now();
    let mut active = JoinSet::new();
    let mut next = 0;
    let mut calls = vec![0.0; work.batches.len()];
    let mut rows = vec![Vec::new(); work.chunks.len()];
    loop {
        while next < work.batches.len() && active.len() < IN_FLIGHT {
            let index = next;
            let candidate = candidate.clone();
            let work = work.clone();
            let label = label.to_string();
            active.spawn(async move {
                let started = Instant::now();
                let values = embed(
                    &candidate,
                    &work,
                    &work.batches[index],
                    &format!("{label}-export-{index}"),
                )
                .await?;
                Ok::<_, anyhow::Error>((index, started.elapsed().as_secs_f64() * 1000.0, values))
            });
            next += 1;
        }
        let Some(result) = active.join_next().await else {
            break;
        };
        let (index, ms, values) = result??;
        calls[index] = ms;
        let batch = &work.batches[index];
        for (offset, value) in values.into_iter().enumerate() {
            rows[batch.first_chunk_seq + offset] = value;
        }
    }
    ensure!(rows.iter().all(|r| r.len() == DIM), "incomplete export");
    Ok((wall.elapsed().as_secs_f64(), calls, rows))
}
async fn query_stratum(
    candidate: Arc<Candidate>,
    work: Arc<Workload>,
    reference: Arc<Vec<Vec<u32>>>,
    query_reference: &[u32],
    samples: usize,
    label: &str,
) -> Result<(Vec<f64>, usize, usize, usize, usize)> {
    for _ in 0..5 {
        query(&candidate, &work.chunks[0]).await?;
    }
    let stop = Arc::new(AtomicBool::new(false));
    let barrier = Arc::new(tokio::sync::Barrier::new(IN_FLIGHT + 1));
    let mut background = JoinSet::new();
    for slot in 0..IN_FLIGHT {
        let candidate = candidate.clone();
        let work = work.clone();
        let reference = reference.clone();
        let stop = stop.clone();
        let barrier = barrier.clone();
        let label = label.to_string();
        background.spawn(async move {
            let mut calls = 0;
            let mut identical = 0;
            let mut mismatched = 0;
            barrier.wait().await;
            while !stop.load(Ordering::Acquire) {
                let batch = &work.batches[(slot + calls * IN_FLIGHT) % work.batches.len()];
                let values = embed(
                    &candidate,
                    &work,
                    batch,
                    &format!("{label}-background-{slot}-{calls}"),
                )
                .await?;
                for (offset, value) in values.iter().enumerate() {
                    if *value == reference[batch.first_chunk_seq + offset] {
                        identical += 1;
                    } else {
                        mismatched += 1;
                    }
                }
                calls += 1;
            }
            Ok::<_, anyhow::Error>((calls, identical, mismatched))
        });
    }
    barrier.wait().await;
    let mut timings = Vec::with_capacity(samples);
    let mut identical_queries = 0;
    let result = async {
        for _ in 0..samples {
            let started = Instant::now();
            let value = query(&candidate, &work.chunks[0]).await?;
            timings.push(started.elapsed().as_secs_f64() * 1000.0);
            if value == query_reference {
                identical_queries += 1;
            }
        }
        Ok::<_, anyhow::Error>(())
    }
    .await;
    stop.store(true, Ordering::Release);
    let mut background_calls = 0;
    let mut identical = 0;
    let mut mismatched = 0;
    // Drain both bulk requests before the arm releases its module and ANE lock.
    while let Some(done) = background.join_next().await {
        let (calls, good, bad) = done??;
        background_calls += calls;
        identical += good;
        mismatched += bad;
    }
    result?;
    ensure!(
        background_calls >= IN_FLIGHT,
        "bulk stratum did not execute"
    );
    Ok((
        timings,
        identical_queries,
        background_calls,
        identical,
        mismatched,
    ))
}

#[derive(Serialize)]
struct Report {
    schema: u32,
    source_commit: String,
    chip: String,
    input_sha256: String,
    meta_sha256: String,
    checkpoint_sha256: String,
    tokenizer_sha256: String,
    module_sha256: String,
    worker_sha256: String,
    driver_sha256: String,
    rows: usize,
    outer_calls: usize,
    calls_in_flight: usize,
    engine_quantum_rows: usize,
    fingerprint: String,
    query_row: usize,
    arms: Vec<Arm>,
    gates: Value,
}
fn assess(arms: &[Arm]) -> Value {
    let serial = arms.iter().filter(|a| !a.packing).collect::<Vec<_>>();
    let packed = arms.iter().filter(|a| a.packing).collect::<Vec<_>>();
    if serial.is_empty() || packed.is_empty() {
        return Value::Null;
    }
    let serial_seconds = median(&serial.iter().map(|a| a.export_seconds).collect::<Vec<_>>());
    let packed_seconds = median(&packed.iter().map(|a| a.export_seconds).collect::<Vec<_>>());
    let serial_queries = serial
        .iter()
        .flat_map(|a| a.query_ms.iter().copied())
        .collect::<Vec<_>>();
    let packed_queries = packed
        .iter()
        .flat_map(|a| a.query_ms.iter().copied())
        .collect::<Vec<_>>();
    let baseline = latencies(&serial_queries);
    let concurrent = latencies(&packed_queries);
    let p50_limit = baseline.p50_ms + (baseline.p50_ms * 0.10).max(5.0);
    let p99_limit = baseline.p99_ms + (baseline.p99_ms * 0.10).max(5.0);
    let identity = arms.iter().all(|a| {
        a.identical_rows == ROWS
            && a.identical_queries == a.query_ms.len()
            && a.background_mismatched_rows == 0
    });
    let dispatch = arms.iter().all(|a| {
        let before = packed_requests(&a.observations_before);
        let after = packed_requests(&a.observations_after_export);
        let max_rows = a.observations_after_export["wire"]["max_embed_rows"]
            .as_u64()
            .unwrap_or(0);
        if !(1..=8).contains(&max_rows) {
            return false;
        }
        if a.packing {
            after > before && !a.packed_compile_request_ms.is_empty()
        } else {
            after == 0 && a.packed_compile_request_ms.is_empty()
        }
    });
    let time = packed_seconds <= serial_seconds * 0.80;
    let queries = concurrent.p50_ms <= p50_limit && concurrent.p99_ms <= p99_limit;
    json!({"serial_median_seconds":serial_seconds,"packed_median_seconds":packed_seconds,"packed_over_serial":packed_seconds/serial_seconds,"serial_queries":baseline,"packed_queries":concurrent,"query_p50_limit_ms":p50_limit,"query_p99_limit_ms":p99_limit,"all_vectors_bit_identical":identity,"packed_dispatch_and_off_switch":dispatch,"export_time_pass":time,"query_latency_pass":queries,"passed":time && queries && identity && dispatch && serial.len()>=3 && packed.len()>=3})
}
fn save(path: &Path, report: &Report) -> Result<()> {
    let temporary = path.with_extension("json.tmp");
    std::fs::write(&temporary, serde_json::to_vec_pretty(report)?)?;
    std::fs::rename(temporary, path)?;
    Ok(())
}
fn required_path(name: &str) -> Result<PathBuf> {
    env::var_os(name)
        .map(PathBuf::from)
        .with_context(|| format!("set {name}"))
}

#[tokio::main]
async fn main() -> Result<()> {
    if env::current_exe()?
        .file_name()
        .unwrap()
        .to_string_lossy()
        .ends_with("worker-proxy")
    {
        return worker_proxy();
    }
    ensure!(cfg!(target_os = "macos"), "ANE hardware requires macOS");
    ensure!(
        env::var("SYNAPSE_PACKING_APPROVED").as_deref() == Ok("1"),
        "obtain owner approval before hardware runs, then set SYNAPSE_PACKING_APPROVED=1"
    );
    ensure!(env::var_os("TMPDIR").is_none(), "unset TMPDIR");
    ensure!(
        env::var("DEVELOPER_DIR").as_deref() == Ok("/Applications/Xcode.app/Contents/Developer"),
        "select Xcode"
    );
    let executable = env::current_exe()?;
    ensure!(
        executable
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("ckdev-"),
        "invoke a signed ckdev- alias"
    );
    ensure!(
        Command::new("codesign")
            .args(["--verify", "--strict"])
            .arg(&executable)
            .status()?
            .success(),
        "driver is not signed"
    );
    hardware_preflight()?;
    let out = required_path("SYNAPSE_PACKING_OUT")?;
    let parent = out.parent().context("output parent")?;
    std::fs::create_dir_all(parent)?;
    let parent = parent.canonicalize()?;
    ensure!(
        !parent.starts_with(env::current_dir()?.canonicalize()?),
        "raw output must be outside the repository"
    );
    ensure!(!out.exists(), "never overwrite an existing run");
    let work = Arc::new(Workload::load(&required_path("SYNAPSE_PACKING_INPUT")?)?);
    let weights = required_path("SYNAPSE_QWEN_WEIGHTS")?.canonicalize()?;
    let assets = required_path("SYNAPSE_COMPARE_ASSETS")?.canonicalize()?;
    ensure!(
        qwen_compare::model() == "qwen3-embedding-0.6b",
        "only the pinned Qwen lane is supported"
    );
    qwen_compare::verify_checkpoint(&weights)?;
    let repeats: usize = env::var("SYNAPSE_PACKING_REPEATS")
        .unwrap_or_else(|_| "3".into())
        .parse()?;
    let samples: usize = env::var("SYNAPSE_PACKING_QUERY_SAMPLES")
        .unwrap_or_else(|_| "1000".into())
        .parse()?;
    ensure!(
        repeats >= 3 && samples >= 1000,
        "at least 3 repetitions and 1,000 queries per arm required"
    );
    let git = Command::new("git").args(["rev-parse", "HEAD"]).output()?;
    ensure!(git.status.success(), "source commit unavailable");
    let chip = Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()?;
    ensure!(chip.status.success(), "chip identity unavailable");
    let mut report = Report {
        schema: 1,
        source_commit: String::from_utf8(git.stdout)?.trim().into(),
        chip: String::from_utf8(chip.stdout)?.trim().into(),
        input_sha256: work.input_sha256.clone(),
        meta_sha256: work.meta_sha256.clone(),
        checkpoint_sha256: sha256_file(&weights.join("model.safetensors"))?,
        tokenizer_sha256: sha256_file(&weights.join("tokenizer.json"))?,
        module_sha256: sha256_file(&assets.join("ck-synapse"))?,
        worker_sha256: sha256_file(&assets.join("ck-synapse-worker-ane-direct"))?,
        driver_sha256: sha256_file(&executable)?,
        rows: ROWS,
        outer_calls: work.batches.len(),
        calls_in_flight: IN_FLIGHT,
        engine_quantum_rows: 8,
        fingerprint: PIN.into(),
        query_row: 0,
        arms: Vec::new(),
        gates: Value::Null,
    };
    let mut reference = None;
    let mut query_reference = None;
    save(&out, &report)?;
    for repetition in 0..repeats {
        for packing in [false, true] {
            let label = format!("{}-{repetition}", if packing { "packed" } else { "serial" });
            println!("starting {label}");
            let candidate = Arc::new(
                Candidate::start(
                    &parent.join(format!("private-{}-{label}", std::process::id())),
                    &assets,
                    &weights,
                    packing,
                )
                .await?,
            );
            let observations_before = warm(&candidate, &work, packing).await?;
            let (export_seconds, calls_ms, rows) =
                export(candidate.clone(), work.clone(), &label).await?;
            let observations_after_export = candidate.observations().await?;
            let reference = reference
                .get_or_insert_with(|| Arc::new(rows.clone()))
                .clone();
            let mismatched_rows = rows
                .iter()
                .zip(reference.iter())
                .enumerate()
                .filter_map(|(i, (a, b))| (a != b).then_some(i))
                .collect::<Vec<_>>();
            let query_value = query(&candidate, &work.chunks[0]).await?;
            let query_reference = query_reference.get_or_insert(query_value);
            let (
                query_ms,
                identical_queries,
                background_calls,
                background_identical_rows,
                background_mismatched_rows,
            ) = query_stratum(
                candidate.clone(),
                work.clone(),
                reference,
                query_reference,
                samples,
                &label,
            )
            .await?;
            let observations_after_queries = candidate.observations().await?;
            let packed_compile_request_ms = compile_records(&candidate.profile_dir)?;
            let arm = Arm {
                repetition,
                packing,
                export_seconds,
                call_latency: latencies(&calls_ms),
                calls_ms,
                identical_rows: ROWS - mismatched_rows.len(),
                mismatched_rows,
                vectors_sha256: vector_digest(&rows),
                query_latency: latencies(&query_ms),
                query_ms,
                identical_queries,
                background_calls,
                background_identical_rows,
                background_mismatched_rows,
                packed_compile_request_ms,
                observations_before,
                observations_after_export,
                observations_after_queries,
            };
            println!(
                "{label}: {:.3}s, query p50 {:.3} / p99 {:.3}ms",
                arm.export_seconds, arm.query_latency.p50_ms, arm.query_latency.p99_ms
            );
            report.arms.push(arm);
            report.gates = assess(&report.arms);
            save(&out, &report)?;
            Arc::try_unwrap(candidate)
                .map_err(|_| anyhow::anyhow!("private candidate still referenced"))?
                .shutdown()
                .await?;
        }
    }
    println!("{}", report.gates);
    ensure!(
        report.gates["passed"] == true,
        "measurement gate failed; retain the raw report, do not tune inputs"
    );
    Ok(())
}

// The worker timing log records request IDs but not operation names. This relay
// forwards frames between the module and worker sockets in both arms and records
// operation names without persisting token IDs, input text, or output tensors.
#[cfg(unix)]
fn worker_proxy() -> Result<()> {
    use std::os::unix::net::{UnixListener, UnixStream};
    let worker = required_path("SYNAPSE_PACKING_REAL_WORKER")?;
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    if args
        .iter()
        .any(|arg| arg == "--version" || arg == "--probe-floor")
    {
        ensure!(
            Command::new(worker)
                .args(args)
                .stdin(Stdio::inherit())
                .status()?
                .success(),
            "worker probe failed"
        );
        return Ok(());
    }
    let index = args
        .iter()
        .position(|arg| arg == "--socket")
        .context("worker socket argument")?
        + 1;
    let host_socket = PathBuf::from(args.get(index).context("worker socket path")?);
    let socket = host_socket
        .parent()
        .context("socket directory")?
        .join(format!("relay-{}.sock", std::process::id()));
    let listener = UnixListener::bind(&socket)?;
    listener.set_nonblocking(true)?;
    args[index] = socket.to_string_lossy().into_owned();
    // stdin is the global lane-lock descriptor, not a protocol stream. The real
    // worker must keep its duplicate until it exits, even if the relay is killed.
    let mut child = Command::new(worker)
        .args(args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(std::fs::File::create(
            required_path("SYNAPSE_PACKING_WIRE_LOG")?.with_extension("worker.stderr"),
        )?)
        .spawn()?;
    let result: Result<()> = (|| {
        let deadline = Instant::now() + Duration::from_secs(20);
        let worker_stream = loop {
            match listener.accept() {
                Ok((stream, _)) => break stream,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    ensure!(
                        Instant::now() < deadline && child.try_wait()?.is_none(),
                        "worker did not connect to relay"
                    );
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(e) => return Err(e.into()),
            }
        };
        let mut worker_stream = blocking_worker_stream(worker_stream)?;
        let mut host = UnixStream::connect(host_socket)?;
        let mut log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(required_path("SYNAPSE_PACKING_WIRE_LOG")?)?;
        relay_frames(&mut host, &mut worker_stream, &mut log)
    })();
    if let Err(error) = &result {
        let _ = child.kill();
        let _ = std::fs::write(
            required_path("SYNAPSE_PACKING_WIRE_LOG")?.with_extension("relay.error"),
            error.to_string(),
        );
    }
    let status = child.wait()?;
    let _ = std::fs::remove_file(socket);
    result?;
    ensure!(status.success(), "worker relay child failed");
    Ok(())
}
#[cfg(not(unix))]
fn worker_proxy() -> Result<()> {
    anyhow::bail!("worker relay requires Unix sockets")
}

fn wire_snapshot(path: &Path) -> Result<Value> {
    let mut counts = serde_json::Map::new();
    if !path.exists() {
        return Ok(Value::Object(counts));
    }
    for line in std::fs::read_to_string(path)?.lines() {
        let record: Value = serde_json::from_str(line)?;
        ensure!(
            record["response"] != "ERR",
            "worker exchange refused: {}",
            record["operation"]
        );
        let op = record["operation"].as_str().context("wire operation")?;
        let next = counts.get(op).and_then(Value::as_u64).unwrap_or(0) + 1;
        counts.insert(op.into(), json!(next));
        let rows = record["rows"].as_u64().unwrap_or(0);
        let maximum = counts
            .get("max_embed_rows")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            .max(rows);
        counts.insert("max_embed_rows".into(), json!(maximum));
    }
    Ok(Value::Object(counts))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn export_plan_covers_every_row_exactly_once() {
        let good = vec![
            Batch {
                seq: 0,
                first_chunk_seq: 0,
                chunk_count: 2,
            },
            Batch {
                seq: 1,
                first_chunk_seq: 2,
                chunk_count: 1,
            },
        ];
        assert!(validate_plan(&good, 3).is_ok());
        let mut overlap = good.clone();
        overlap[1].first_chunk_seq = 1;
        assert!(validate_plan(&overlap, 3).is_err());
        assert!(validate_plan(&good[..1], 3).is_err());
        let mut oversized = good;
        oversized[0].chunk_count = 65;
        assert!(validate_plan(&oversized, 66).is_err());
    }
    #[test]
    fn percentiles_use_nearest_rank_and_even_medians() {
        assert_eq!(percentile(&[3.0, 1.0, 2.0, 4.0], 0.50), 2.0);
        assert_eq!(
            percentile(&(1..=100).map(f64::from).collect::<Vec<_>>(), 0.99),
            99.0
        );
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
    }
    #[test]
    fn wire_vectors_restore_ids_and_compare_f32_bits() {
        let a = vec![json!(0.5); DIM];
        let b = vec![json!(-0.0); DIM];
        let response = json!({"fingerprint":PIN,"vectors":[{"id":"row-2","vector":b},{"id":"row-1","vector":a}]});
        let rows = vectors(&response, &[1, 2], false).unwrap();
        assert_eq!(rows[0], vec![0.5_f32.to_bits(); DIM]);
        assert_eq!(rows[1], vec![(-0.0_f32).to_bits(); DIM]);
        assert_ne!(rows[1], vec![0.0_f32.to_bits(); DIM]);
        let mut duplicate = response.clone();
        duplicate["vectors"][0]["id"] = json!("row-1");
        assert!(vectors(&duplicate, &[1, 2], false).is_err());
        let mut wrong = response;
        wrong["fingerprint"] = json!("different");
        assert!(vectors(&wrong, &[1, 2], false).is_err());
    }
    #[test]
    fn vector_digest_hashes_bits_not_display_values() {
        assert_ne!(
            vector_digest(&[vec![0.0_f32.to_bits()]]),
            vector_digest(&[vec![(-0.0_f32).to_bits()]])
        );
    }
}

#[cfg(test)]
mod gate_tests {
    use super::*;
    fn arm(packing: bool) -> Arm {
        let before = if packing {
            json!({"wire":{"EMBED_PACKED":3}})
        } else {
            json!({"wire":{}})
        };
        let after = if packing {
            json!({"wire":{"EMBED_PACKED":103,"max_embed_rows":8}})
        } else {
            json!({"wire":{"EMBED_BATCH":100,"max_embed_rows":8}})
        };
        Arm {
            repetition: 0,
            packing,
            export_seconds: if packing { 80.0 } else { 100.0 },
            calls_ms: vec![1.0],
            call_latency: latencies(&[1.0]),
            identical_rows: ROWS,
            mismatched_rows: vec![],
            vectors_sha256: "hash".into(),
            query_ms: vec![10.0; 1000],
            query_latency: latencies(&[10.0; 1000]),
            identical_queries: 1000,
            background_calls: 2,
            background_identical_rows: 128,
            background_mismatched_rows: 0,
            packed_compile_request_ms: if packing { vec![500.0] } else { vec![] },
            observations_before: before,
            observations_after_export: after.clone(),
            observations_after_queries: after,
        }
    }
    fn passing() -> Vec<Arm> {
        (0..3).flat_map(|_| [arm(false), arm(true)]).collect()
    }
    #[test]
    fn acceptance_uses_export_medians_and_baseline_plus_max_tolerance() {
        let mut arms = passing();
        assert_eq!(assess(&arms)["passed"], true);
        arms[1].query_ms = vec![15.0; 1000];
        arms[3].query_ms = vec![15.0; 1000];
        arms[5].query_ms = vec![15.0; 1000];
        assert_eq!(assess(&arms)["query_latency_pass"], true);
        arms[1].query_ms = vec![15.01; 1000];
        arms[3].query_ms = vec![15.01; 1000];
        arms[5].query_ms = vec![15.01; 1000];
        assert_eq!(assess(&arms)["query_latency_pass"], false);
        arms[3].export_seconds = 80.01;
        arms[5].export_seconds = 80.01;
        assert_eq!(assess(&arms)["export_time_pass"], false);
    }
    #[test]
    fn acceptance_requires_real_packed_dispatch_and_serial_off_switch() {
        let mut arms = passing();
        arms[1].observations_after_export = arms[1].observations_before.clone();
        assert_eq!(assess(&arms)["packed_dispatch_and_off_switch"], false);
        let mut arms = passing();
        arms[0].observations_after_export = json!({"wire":{"EMBED_PACKED":1}});
        assert_eq!(assess(&arms)["packed_dispatch_and_off_switch"], false);
        let mut arms = passing();
        arms[1].packed_compile_request_ms.clear();
        assert_eq!(assess(&arms)["packed_dispatch_and_off_switch"], false);
    }
    #[test]
    fn acceptance_rejects_any_document_query_or_background_bit_mismatch() {
        let mut arms = passing();
        arms[1].identical_rows -= 1;
        assert_eq!(assess(&arms)["all_vectors_bit_identical"], false);
        let mut arms = passing();
        arms[1].identical_queries -= 1;
        assert_eq!(assess(&arms)["all_vectors_bit_identical"], false);
        let mut arms = passing();
        arms[1].background_mismatched_rows = 1;
        assert_eq!(assess(&arms)["all_vectors_bit_identical"], false);
        assert_eq!(assess(&passing()[..4])["passed"], false);
    }
}

fn relay_frames(
    host: &mut (impl std::io::Read + std::io::Write),
    worker: &mut (impl std::io::Read + std::io::Write),
    log: &mut impl std::io::Write,
) -> Result<()> {
    use synapse_core::{
        worker_framing_sync::{read_frame, write_frame},
        WorkerRequest, DEFAULT_MAX_FRAME_BYTES,
    };
    let hello = read_frame(worker, DEFAULT_MAX_FRAME_BYTES)?;
    write_frame(host, &hello, DEFAULT_MAX_FRAME_BYTES)?;
    let ack = read_frame(host, DEFAULT_MAX_FRAME_BYTES)?;
    let parsed: Value = serde_json::from_slice(&ack)?;
    let max = u32::try_from(
        parsed["max_frame"]
            .as_u64()
            .context("HELLO_ACK max frame")?,
    )?;
    ensure!(
        max > 0 && max <= DEFAULT_MAX_FRAME_BYTES && parsed["accept"] == true,
        "relay handshake refused"
    );
    write_frame(worker, &ack, DEFAULT_MAX_FRAME_BYTES)?;
    loop {
        let frame = match read_frame(host, max) {
            Ok(frame) => frame,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        };
        let request: WorkerRequest = serde_json::from_slice(&frame)?;
        let start = Instant::now();
        write_frame(worker, &frame, max)?;
        if request.carries_raw_frame() {
            let raw = read_frame(host, max)?;
            write_frame(worker, &raw, max)?;
        }
        if matches!(request, WorkerRequest::Shutdown {}) {
            break;
        }
        let response = read_frame(worker, max)?;
        let parsed: Value = serde_json::from_slice(&response)?;
        let raw = if parsed["type"] == "VECTORS" || parsed["type"] == "SCORES" {
            Some(read_frame(worker, max)?)
        } else {
            None
        };
        let rows = match &request {
            WorkerRequest::EmbedBatch { items, .. } | WorkerRequest::EmbedPacked { items, .. } => {
                items.len()
            }
            _ => 0,
        };
        let record = json!({"req_id":request.req_id(),"operation":request.wire_type(),"rows":rows,"response":parsed["type"],"wall_ms":start.elapsed().as_secs_f64()*1000.0});
        writeln!(log, "{record}")?;
        write_frame(host, &response, max)?;
        if let Some(raw) = raw {
            write_frame(host, &raw, max)?;
        }
    }
    Ok(())
}

#[cfg(all(test, unix))]
mod relay_tests {
    use super::*;
    use std::os::unix::net::UnixStream;
    use synapse_core::worker_framing_sync::{read_frame, write_frame};
    #[test]
    fn socket_relay_preserves_handshake_json_and_raw_frames_without_recording_payloads() {
        let (mut host, mut host_relay) = UnixStream::pair().unwrap();
        let (mut worker, mut worker_relay) = UnixStream::pair().unwrap();
        for stream in [&host, &host_relay, &worker, &worker_relay] {
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
        }
        let relay = std::thread::spawn(move || {
            let mut log = Vec::new();
            relay_frames(&mut host_relay, &mut worker_relay, &mut log).unwrap();
            log
        });
        let hello = br#"{"v":2,"nonce":"test-secret","pid":7}"#;
        write_frame(&mut worker, hello, 1024).unwrap();
        assert_eq!(read_frame(&mut host, 1024).unwrap(), hello);
        let ack = br#"{"v":2,"accept":true,"max_frame":1024}"#;
        write_frame(&mut host, ack, 1024).unwrap();
        assert_eq!(read_frame(&mut worker, 1024).unwrap(), ack);
        let request=br#"{"type":"EMBED_PACKED","req_id":"probe","model_ref":"m","program_revision":1,"items":[{"id":"0","n_tokens":1}],"passes":[[0]]}"#;
        let raw = b"secret-token-frame";
        write_frame(&mut host, request, 1024).unwrap();
        write_frame(&mut host, raw, 1024).unwrap();
        assert_eq!(read_frame(&mut worker, 1024).unwrap(), request);
        assert_eq!(read_frame(&mut worker, 1024).unwrap(), raw);
        let response = br#"{"type":"VECTORS","req_id":"probe","n":1,"dims":1}"#;
        let vectors = b"secret-vector-frame";
        write_frame(&mut worker, response, 1024).unwrap();
        write_frame(&mut worker, vectors, 1024).unwrap();
        assert_eq!(read_frame(&mut host, 1024).unwrap(), response);
        assert_eq!(read_frame(&mut host, 1024).unwrap(), vectors);
        let shutdown = br#"{"type":"SHUTDOWN"}"#;
        write_frame(&mut host, shutdown, 1024).unwrap();
        assert_eq!(read_frame(&mut worker, 1024).unwrap(), shutdown);
        let log = String::from_utf8(relay.join().unwrap()).unwrap();
        assert!(log.contains("EMBED_PACKED"));
        assert!(!log.contains("secret"));
        assert_eq!(log.lines().count(), 1);
    }
}

#[cfg(unix)]
fn blocking_worker_stream(
    stream: std::os::unix::net::UnixStream,
) -> Result<std::os::unix::net::UnixStream> {
    // BSD can inherit the listener's nonblocking flag on accepted sockets;
    // read_exact must wait for a complete worker frame rather than return EAGAIN.
    stream.set_nonblocking(false)?;
    Ok(stream)
}
#[cfg(all(test, unix))]
mod socket_mode_tests {
    use super::*;
    use std::{
        io::{Read, Write},
        os::unix::net::UnixStream,
    };
    #[test]
    fn worker_stream_waits_for_frames_after_inheriting_nonblocking_mode() {
        let (reader, mut writer) = UnixStream::pair().unwrap();
        reader.set_nonblocking(true).unwrap();
        reader
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut reader = blocking_worker_stream(reader).unwrap();
        let peer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            writer.write_all(b"x").unwrap();
        });
        let mut byte = [0];
        reader.read_exact(&mut byte).unwrap();
        assert_eq!(byte, *b"x");
        peer.join().unwrap();
    }
}
