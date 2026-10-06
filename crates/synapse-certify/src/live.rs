//! Executes candidate artifacts against a private in-process daemon.
use crate::{
    refuse, Admission, ArtifactFile, Inventory, Outcome, Parity, Result, RunEvidence, Runner,
};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};
use subc_daemon::{
    daemon_config::StorageConfig, serve_listener, ControlHandler, Registry, Router, ServerAuth,
};
use subc_protocol::{BindIdentity, Flags, Frame, FrameType, Priority, RouteTarget};
use subc_transport::{
    authenticate_client, generate_daemon_id, generate_key, read_frame, write_atomic, write_frame,
    ConnectionInfo, Endpoint, SCHEMA_VERSION,
};
use synapse_parity::{
    evaluator::{evaluate, FixtureSet, ObservedCase, Output},
    manifest::Manifest,
};
use tokio::{
    net::{TcpListener, TcpStream},
    process::Command,
    time::timeout,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Options {
    pub assets: PathBuf,
    pub checkout: PathBuf,
    pub weights: PathBuf,
}

pub struct LiveRunner {
    options: Options,
    source: String,
    runtime: tokio::runtime::Runtime,
    floor: Option<Value>,
}
fn err(error: impl std::fmt::Display) -> crate::Error {
    refuse(error.to_string())
}
fn binary(root: &Path, role: &str) -> PathBuf {
    root.join(format!("{role}{}", std::env::consts::EXE_SUFFIX))
}
fn lane(row: &str) -> &'static str {
    if row == "metal-m5" {
        "owned-metal"
    } else if row == "ane-m5" {
        "ane-direct-worker"
    } else if row.starts_with("cuda-") {
        "owned-cuda"
    } else {
        "owned-vulkan"
    }
}
fn worker(row: &str) -> Option<&'static str> {
    match lane(row) {
        "owned-cuda" => Some("ck-synapse-worker-cuda"),
        "owned-vulkan" => Some("ck-synapse-worker-vulkan"),
        "ane-direct-worker" => Some("ck-synapse-worker-ane-direct"),
        _ => None,
    }
}
impl LiveRunner {
    pub fn new(options: Options, source: &str) -> Result<Self> {
        Ok(Self {
            options,
            source: source.into(),
            runtime: tokio::runtime::Runtime::new().map_err(err)?,
            floor: None,
        })
    }
}
impl Runner for LiveRunner {
    fn probe_floor(&mut self, row: &str, _model: &str) -> Result<String> {
        let role = worker(row).ok_or_else(|| refuse("metal-m5 has no floor probe"))?;
        let output = self.runtime.block_on(async {
            timeout(
                Duration::from_secs(60),
                synapse_core::without_launch_nonce_tokio(Command::new(binary(
                    &self.options.assets,
                    role,
                )))
                .arg("--probe-floor")
                .kill_on_drop(true)
                .output(),
            )
            .await
            .map_err(err)?
            .map_err(err)
        })?;
        let floor: Value = serde_json::from_slice(&output.stdout).map_err(err)?;
        let status = floor["status"].as_str().unwrap_or("refused").to_string();
        self.floor = Some(floor);
        Ok(if output.status.success() {
            status
        } else {
            "refused".into()
        })
    }
    fn observe(&mut self, row: &str, model: &str) -> Result<RunEvidence> {
        self.runtime.block_on(observe(
            &self.options,
            &self.source,
            row,
            model,
            self.floor.as_ref(),
        ))
    }
}

struct DaemonTask(tokio::task::JoinHandle<Result<()>>);
impl Drop for DaemonTask {
    fn drop(&mut self) {
        self.0.abort();
    }
}

struct Session {
    child: tokio::process::Child,
    daemon: DaemonTask,
    stream: TcpStream,
    channel: u16,
    epoch: u32,
    corr: u64,
}
impl Drop for Session {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
        self.daemon.0.abort();
    }
}
async fn rpc(
    stream: &mut TcpStream,
    channel: u16,
    epoch: u32,
    corr: u64,
    body: Value,
) -> Result<Value> {
    let frame = Frame::build(
        FrameType::Request,
        Flags::new(false, Priority::Passive, false),
        channel,
        epoch,
        corr,
        serde_json::to_vec(&body).map_err(err)?,
    )
    .map_err(err)?;
    write_frame(stream, &frame).await.map_err(err)?;
    timeout(Duration::from_secs(3600), async {
        loop {
            let frame = read_frame(stream)
                .await
                .map_err(err)?
                .ok_or_else(|| refuse("candidate connection closed"))?;
            if frame.header.channel == channel
                && frame.header.corr == corr
                && matches!(frame.header.ty, FrameType::Response | FrameType::Error)
            {
                let value: Value = serde_json::from_slice(&frame.body).map_err(err)?;
                if frame.header.ty == FrameType::Error {
                    return Ok(json!({"error": value}));
                }
                return Ok(value);
            }
        }
    })
    .await
    .map_err(err)?
}
impl Session {
    async fn shutdown(&mut self) -> Result<()> {
        self.child.start_kill().map_err(err)?;
        self.child.wait().await.map_err(err)?;
        self.daemon.0.abort();
        Ok(())
    }
    async fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        self.corr += 1;
        let value = rpc(
            &mut self.stream,
            self.channel,
            self.epoch,
            self.corr,
            json!({"method": method, "params": params}),
        )
        .await?;
        Ok(value.get("result").cloned().unwrap_or(value))
    }
    async fn observations(&mut self) -> Result<Value> {
        let value = self.request("certify.observations", json!({})).await?;
        if value["available"] != true {
            return Err(refuse(format!("certify.observations unavailable: {value}")));
        }
        Ok(value)
    }
}

async fn start(options: &Options, row: &str, model: &str, manifest: &Value) -> Result<Session> {
    let root = options
        .checkout
        .join("crates/synapse-certify/.live")
        .join(format!("{}-{}", std::process::id(), row));
    std::fs::create_dir_all(root.join("data")).map_err(err)?;
    let listener = TcpListener::bind("127.0.0.1:0").await.map_err(err)?;
    let conn = ConnectionInfo {
        schema: SCHEMA_VERSION,
        endpoints: vec![Endpoint {
            host: "127.0.0.1".into(),
            port: listener.local_addr().map_err(err)?.port(),
        }],
        key: generate_key().map_err(err)?,
        daemon_id: generate_daemon_id().map_err(err)?,
        pid: std::process::id(),
        daemon_ver: "synapse-certify".into(),
        wire_version: Some(subc_protocol::PROTOCOL_VERSION),
    };
    let conn_path = root.join("connection.json");
    write_atomic(&conn_path, &conn).map_err(err)?;
    let registry = Arc::new(Registry::default());
    let control = ControlHandler::new(registry).with_storage_config(Some(StorageConfig::Sqlite {
        data_home: root.join("data"),
    }));
    let router = Arc::new(Router::with_control_handler(Arc::new(control)));
    let auth = ServerAuth::new(conn.key.clone(), conn.daemon_id, conn.daemon_ver.clone());
    let daemon = DaemonTask(tokio::spawn(async move {
        serve_listener(listener, router, auth).await.map_err(err)
    }));
    let pinned = &manifest["models"][model];
    let model_path = if worker(row).is_some() {
        let manifest =
            Manifest::from_slice(&serde_json::to_vec(manifest).map_err(err)?).map_err(err)?;
        let package = synapse_parity::convert::convert_profile_file(
            &manifest,
            &format!("{model}.{}", lane(row)),
            &options.weights.join("model.safetensors"),
        )
        .map_err(err)?;
        let path = root.join("profile.safetensors");
        std::fs::write(&path, package).map_err(err)?;
        path
    } else {
        options.weights.join("model.safetensors")
    };
    let operation = pinned["operation"]
        .as_str()
        .ok_or_else(|| refuse("manifest operation missing"))?;
    let mut preload = json!({
        "model_id": "certify-candidate", "engine": lane(row), "profile": format!("{model}.{}", lane(row)),
        "task": operation, "model_path": model_path,
        "tokenizer_path": options.weights.join("tokenizer.json"),
        "pooling": match pinned["grammar"]["pooling"].as_str() { Some("cls") => "cls", Some("masked_mean") => "mean", _ => "last" }, "normalize": pinned["output"]["normalization"] == "l2",
        "execution": "explicit", "attention_units": 8192 * 8192,
    });
    if let Some(role) = worker(row) {
        preload["worker_bin"] = json!(binary(&options.assets, role));
        preload["worker_runtime_dir"] = json!(options.assets);
    }
    let config = root.join("config.json");
    std::fs::write(&config, serde_json::to_vec(&json!({"certify_observation": true, "preload_models": [preload], "inline": {"deadline_ms": 3600000, "max_queue_ms": 3600000, "max_items": INLINE_ITEMS, "max_tokens": INLINE_TOKENS}})).map_err(err)?).map_err(err)?;
    let child = synapse_core::without_launch_nonce_tokio(Command::new(binary(
        &options.assets,
        "ck-synapse",
    )))
    .arg("--subc")
    .arg(&conn_path)
    .env("SUBC_MODULE_ID", "synapse")
    .env("SYNAPSE_CONFIG_PATH", &config)
    .env("XDG_DATA_HOME", root.join("data"))
    .env("CORTEXKIT_LEASE_ROOT", root.join("leases"))
    .env("CORTEXKIT_STORE_ROOT", root.join("store"))
    .stderr(std::process::Stdio::inherit())
    .kill_on_drop(true)
    .spawn()
    .map_err(err)?;
    let mut stream = TcpStream::connect(("127.0.0.1", conn.endpoints[0].port))
        .await
        .map_err(err)?;
    authenticate_client(&mut stream, &conn, Duration::from_secs(10))
        .await
        .map_err(err)?;
    let mut session = Session {
        child,
        daemon,
        stream,
        channel: 0,
        epoch: 0,
        corr: 1,
    };
    for attempt in 0..600 {
        session.corr += 1;
        let response = rpc(&mut session.stream, 0, 0, session.corr, json!({"op": "route.open", "target": RouteTarget::ManagementSurface { module_id: "synapse".into() }, "identity": BindIdentity::new(options.checkout.clone(), "certify".to_string(), "certify".to_string())})).await?;
        if let (Some(channel), Some(epoch)) = (
            response["route_channel"].as_u64(),
            response["route_epoch"].as_u64(),
        ) {
            session.channel = channel as u16;
            session.epoch = epoch as u32;
            return Ok(session);
        }
        if session.child.try_wait().map_err(err)?.is_some() {
            return Err(refuse(format!(
                "candidate exited before registration: {response}"
            )));
        }
        if attempt == 599 {
            return Err(refuse(format!(
                "candidate registration timed out: {response}"
            )));
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    unreachable!()
}
fn request_params(case: &Value, operation: &str) -> Value {
    if operation == "embed" {
        json!({"model": "certify-candidate", "text": case["text"], "accept_declared": true})
    } else {
        json!({"model": "certify-candidate", "query": case["query"], "candidates": [case["document"]], "accept_declared": true})
    }
}
fn observed_output(response: &Value, index: usize, operation: &str) -> serde_json::Result<Output> {
    // ResponseEnvelope flattens the inference payload alongside its metadata.
    if operation == "embed" {
        serde_json::from_value::<Vec<f64>>(response["vectors"][index]["vector"].clone())
            .map(Output::Embedding)
    } else {
        serde_json::from_value::<f64>(response["scores"][index].clone()).map(Output::Score)
    }
}

const INLINE_ITEMS: usize = 64;
const INLINE_TOKENS: usize = 8192;

// Preserve candidate order and test padding with real multi-item calls. Count
// every engine-input token, including query/template and special tokens in each
// pair; their sum bounds the module's inline token limit without understating
// the tokenizer's request size.
fn fixture_requests<'a>(cases: &'a [Value], operation: &str) -> Result<Vec<Vec<&'a Value>>> {
    let mut groups: Vec<Vec<&Value>> = Vec::new();
    let mut grouped = BTreeMap::new();
    for case in cases {
        let count = case["input_ids"]
            .as_array()
            .ok_or_else(|| refuse("fixture input IDs missing"))?
            .len();
        let key = if count == INLINE_TOKENS {
            None
        } else if case["category"] == "batched" {
            Some("batched".to_string())
        } else if operation == "rerank" {
            case["pool"].as_str().map(str::to_string)
        } else {
            None
        };
        if let Some(key) = key {
            let index = *grouped.entry(key).or_insert_with(|| {
                groups.push(Vec::new());
                groups.len() - 1
            });
            groups[index].push(case);
        } else {
            groups.push(vec![case]);
        }
    }
    let mut requests = Vec::new();
    for group in groups {
        let counts = group
            .iter()
            .map(|case| case["input_ids"].as_array().expect("validated IDs").len())
            .collect::<Vec<_>>();
        let ranges = synapse_parity::evaluator::split_pool(&counts, INLINE_ITEMS, INLINE_TOKENS)
            .map_err(err)?;
        for range in ranges {
            requests.push(group[range].to_vec());
        }
    }
    Ok(requests)
}

fn group_params(group: &[&Value], operation: &str) -> Result<(&'static str, Value)> {
    if group.is_empty() {
        return Err(refuse("empty fixture request"));
    }
    if operation == "embed" {
        if group.len() == 1 {
            return Ok(("embed.query", request_params(group[0], operation)));
        }
        let items = group
            .iter()
            .map(|case| json!({"id": case["id"], "text": case["text"]}))
            .collect::<Vec<_>>();
        Ok((
            "embed.batch",
            json!({"model": "certify-candidate", "items": items, "accept_declared": true}),
        ))
    } else {
        let query = &group[0]["query"];
        if group.iter().any(|case| &case["query"] != query) {
            return Err(refuse("fixture batch queries differ"));
        }
        let candidates = group
            .iter()
            .map(|case| &case["document"])
            .collect::<Vec<_>>();
        Ok((
            "rerank.score",
            json!({"model": "certify-candidate", "query": query, "candidates": candidates, "accept_declared": true}),
        ))
    }
}

fn sent(snapshot: &Value) -> Result<usize> {
    snapshot["worker_requests"]
        .as_object()
        .ok_or_else(|| refuse("missing worker request counters"))?
        .values()
        .try_fold(0usize, |sum, value| {
            let count = value
                .as_u64()
                .ok_or_else(|| refuse("invalid worker counter"))?;
            sum.checked_add(count as usize)
                .ok_or_else(|| refuse("worker counter overflow"))
        })
}
async fn observe(
    options: &Options,
    source: &str,
    row: &str,
    model: &str,
    floor: Option<&Value>,
) -> Result<RunEvidence> {
    let parity_root = options.checkout.join("bench/parity");
    let manifest = Manifest::load(&parity_root.join("models.json")).map_err(err)?;
    synapse_parity::checkpoint::check_checkpoint(&manifest, model, &options.weights, &parity_root)
        .map_err(err)?;
    let raw: Value =
        serde_json::from_slice(&std::fs::read(parity_root.join("models.json")).map_err(err)?)
            .map_err(err)?;
    let fixtures = FixtureSet::load(&parity_root, &manifest, model).map_err(err)?;
    let index: Value = serde_json::from_slice(
        &std::fs::read(parity_root.join("fixtures/index.json")).map_err(err)?,
    )
    .map_err(err)?;
    let fixture_id = synapse_parity::evaluator::fixture_set_id(&manifest, model);
    let document: Value = serde_json::from_slice(
        &std::fs::read(
            parity_root.join(
                index[&fixture_id]["path"]
                    .as_str()
                    .ok_or_else(|| refuse("fixture path missing"))?,
            ),
        )
        .map_err(err)?,
    )
    .map_err(err)?;
    let cases = document["cases"]
        .as_array()
        .ok_or_else(|| refuse("fixture cases missing"))?;
    let operation = raw["models"][model]["operation"]
        .as_str()
        .ok_or_else(|| refuse("operation missing"))?;
    let method = if operation == "embed" {
        "embed.query"
    } else {
        "rerank.score"
    };
    let profile_id = format!("{model}.{}", lane(row));
    let mut session = start(options, row, model, &raw).await?;
    let mut outputs = BTreeMap::new();
    let listed = session.request("models.list", json!({})).await?;
    let mut fingerprint = listed["models"]
        .as_array()
        .and_then(|models| {
            models
                .iter()
                .find(|model| model["model_id"] == "certify-candidate")
        })
        .and_then(|model| model["fingerprint"].as_str())
        .map(str::to_string);
    let mut long_response = None;
    let mut long_request_count = 0;
    for group in fixture_requests(cases, operation)? {
        let case = group[0];
        let is_long = case["input_ids"]
            .as_array()
            .is_some_and(|ids| ids.len() == 8192);
        let before_long = if is_long {
            Some(sent(&session.observations().await?)?)
        } else {
            None
        };
        let (request_method, params) = group_params(&group, operation)?;
        let response = session.request(request_method, params).await?;
        if outputs.is_empty() && response.get("error").is_some() {
            eprintln!("certification input {} refused: {response}", case["id"]);
        }
        if let Some(value) = response["fingerprint"].as_str() {
            if fingerprint.as_deref().is_some_and(|old| old != value) {
                return Err(refuse("fingerprint changed during run"));
            }
            fingerprint = Some(value.to_string());
        }
        let ids: Vec<Vec<u32>> =
            match serde_json::from_value(response["observation"]["input_ids"].clone()) {
                Ok(ids) => ids,
                Err(_) => continue,
            };
        if ids.len() != group.len() {
            continue;
        }
        let readout = response["observation"]["readout_ids"]
            .as_array()
            .and_then(|ids| Some((ids.first()?.as_u64()? as u32, ids.get(1)?.as_u64()? as u32)));
        if is_long && ids[0].len() == 8192 {
            long_response = Some(response.clone());
            long_request_count = sent(&session.observations().await?)?
                .checked_sub(before_long.expect("long snapshot captured"))
                .ok_or_else(|| refuse("worker counter regressed on 8192 input"))?;
        }
        for (index, (case, input_ids)) in group.into_iter().zip(ids).enumerate() {
            let output = observed_output(&response, index, operation);
            let Ok(output) = output else {
                continue;
            };
            outputs.insert(
                case["id"]
                    .as_str()
                    .ok_or_else(|| refuse("fixture id missing"))?
                    .to_string(),
                ObservedCase {
                    output,
                    input_ids,
                    readout,
                },
            );
        }
    }
    let fingerprint = fingerprint.ok_or_else(|| refuse("candidate produced no fingerprint"))?;
    let evaluation =
        evaluate(&manifest, &profile_id, &fingerprint, &fixtures, &outputs).map_err(err)?;
    let parity: Parity =
        serde_json::from_value(serde_json::to_value(evaluation).map_err(err)?).map_err(err)?;
    let long_case = cases
        .iter()
        .find(|case| {
            case["input_ids"]
                .as_array()
                .is_some_and(|ids| ids.len() == 8192)
        })
        .ok_or_else(|| refuse("8192 fixture missing"))?;
    let mut oversized = long_case.clone();
    let field = if operation == "embed" {
        "text"
    } else {
        "document"
    };
    oversized[field] = json!(format!(
        "{} a",
        long_case[field]
            .as_str()
            .ok_or_else(|| refuse("long fixture text missing"))?
    ));
    let before = session.observations().await?;
    let rejected = session
        .request(method, request_params(&oversized, operation))
        .await?;
    let after = session.observations().await?;
    if rejected["error"]["code"] == "sequence_too_long"
        && rejected["error"]["details"]["tokens"] != 8193
    {
        return Err(refuse(format!(
            "8193 admission input composed to a different length: {rejected}"
        )));
    }
    let delta = sent(&after)?
        .checked_sub(sent(&before)?)
        .ok_or_else(|| refuse("worker counter regressed"))?;
    let response = long_response.unwrap_or(Value::Null);
    let admission = Admission {
        tokens_8192: Outcome {
            outcome: if response.is_null() {
                "not_processed"
            } else {
                "processed"
            }
            .into(),
            truncated: response["payload"]["truncation_disclosures"]
                .as_array()
                .is_some_and(|items| items.iter().any(|item| item["truncated"] == true)),
            diverted: !response["job_id"].is_null(),
            worker_requests: long_request_count,
        },
        tokens_8193: Outcome {
            outcome: rejected["error"]["code"]
                .as_str()
                .unwrap_or("missing_error")
                .into(),
            truncated: false,
            diverted: !rejected["job_id"].is_null(),
            worker_requests: delta,
        },
    };
    let raw_series = if row == "ane-m5" && model.starts_with("qwen3-") && parity.passed() {
        let warm = cases
            .iter()
            .find(|case| {
                case["input_ids"]
                    .as_array()
                    .is_some_and(|ids| ids.len() == 512)
            })
            .ok_or_else(|| refuse("512 latency fixture missing"))?;
        let ane = latency_series(&mut session, method, request_params(warm, operation)).await?;
        let mut metal_session = start(options, "metal-m5", model, &raw).await?;
        let metal =
            latency_series(&mut metal_session, method, request_params(warm, operation)).await?;
        metal_session.shutdown().await?;
        Some(crate::RawSeries {
            session_id: format!("{}-{}", source, std::process::id()),
            ane,
            metal,
        })
    } else {
        None
    };
    let final_snapshot = session.observations().await?;
    let inventories: Vec<Inventory> =
        serde_json::from_value(final_snapshot["inventories"].clone()).map_err(err)?;
    let mut artifacts = vec![ArtifactFile {
        role: "ck-synapse".into(),
        file: binary(Path::new(""), "ck-synapse")
            .to_string_lossy()
            .into_owned(),
    }];
    if let Some(role) = worker(row) {
        artifacts.push(ArtifactFile {
            role: role.into(),
            file: binary(Path::new(""), role).to_string_lossy().into_owned(),
        });
    }
    if row == "cuda-windows-nvidia" {
        let package_manifest: Value = serde_json::from_slice(
            &std::fs::read(options.assets.join("manifest.json")).map_err(err)?,
        )
        .map_err(err)?;
        for runtime in package_manifest["runtime_files"]
            .as_array()
            .ok_or_else(|| refuse("CUDA runtime_files missing"))?
        {
            let file = runtime["file"]
                .as_str()
                .ok_or_else(|| refuse("CUDA runtime file name missing"))?;
            artifacts.push(ArtifactFile {
                role: format!("cuda-runtime:{file}"),
                file: file.into(),
            });
        }
    }
    let machine = machine(row, floor)?;
    session.shutdown().await?;
    Ok(RunEvidence {
        source_commit: source.into(),
        machine,
        artifacts,
        operation: operation.into(),
        profile_id,
        fingerprint,
        fixture_set_id: fixture_id,
        expected_fixtures: fixtures.cases().len(),
        tolerance_class: parity.tolerance_class.clone(),
        parity: (!outputs.is_empty()).then_some(parity),
        admission,
        layer_count: raw["models"][model]["architecture"]["params"]["num_hidden_layers"]
            .as_u64()
            .ok_or_else(|| refuse("layer count missing"))? as usize,
        inventories,
        admitted_count: final_snapshot["admitted_count"]
            .as_u64()
            .ok_or_else(|| refuse("independent ADMITTED counter missing"))?
            as usize,
        raw_series,
    })
}
async fn latency_series(session: &mut Session, method: &str, params: Value) -> Result<Vec<f64>> {
    let mut samples = Vec::with_capacity(23);
    for _ in 0..23 {
        let started = std::time::Instant::now();
        let response = session.request(method, params.clone()).await?;
        if !response["error"].is_null() || response["payload"].is_null() {
            return Err(refuse(format!("latency inference failed: {response}")));
        }
        samples.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(samples)
}

fn machine(row: &str, floor: Option<&Value>) -> Result<Value> {
    #[cfg(target_os = "macos")]
    {
        let output =
            synapse_core::without_launch_nonce(std::process::Command::new("system_profiler"))
                .args(["SPHardwareDataType", "-json"])
                .output()
                .map_err(err)?;
        let hardware: Value = serde_json::from_slice(&output.stdout).map_err(err)?;
        let hardware = &hardware["SPHardwareDataType"][0];
        if !matches!(row, "metal-m5" | "ane-m5")
            || !hardware["chip_type"]
                .as_str()
                .is_some_and(|chip| chip.contains("M5"))
        {
            return Err(refuse(
                "machine does not match the requested M5 certification row",
            ));
        }
        if hardware["machine_model"].as_str().is_none()
            || hardware["platform_UUID"].as_str().is_none()
        {
            return Err(refuse("Apple machine identifiers missing"));
        }
        let os = synapse_core::without_launch_nonce(std::process::Command::new("sw_vers"))
            .arg("-productVersion")
            .output()
            .map_err(err)?;
        Ok(
            json!({"model_identifier": hardware["machine_model"], "platform_uuid": hardware["platform_UUID"], "os": String::from_utf8_lossy(&os.stdout).trim(), "driver": if row == "metal-m5" { json!(format!("Metal on macOS {}", String::from_utf8_lossy(&os.stdout).trim())) } else { floor.cloned().unwrap_or(Value::Null) }}),
        )
    }
    #[cfg(not(target_os = "macos"))]
    {
        machine_non_apple(row, floor)
    }
}

#[cfg(not(target_os = "macos"))]
fn machine_non_apple(row: &str, floor: Option<&Value>) -> Result<Value> {
    fn command(program: &str, args: &[&str]) -> Result<String> {
        let output = synapse_core::without_launch_nonce(std::process::Command::new(program))
            .args(args)
            .output()
            .map_err(err)?;
        if !output.status.success() {
            return Err(refuse(format!(
                "machine probe {program}: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }
    #[cfg(target_os = "windows")]
    let (product, os) = (
        command("powershell", &["-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystem).Model"] )?,
        command("powershell", &["-NoProfile", "-Command", "(Get-CimInstance Win32_OperatingSystem).Caption + ' ' + (Get-CimInstance Win32_OperatingSystem).Version"] )?,
    );
    #[cfg(not(target_os = "windows"))]
    let (product, os) = (
        std::fs::read_to_string("/sys/class/dmi/id/product_name")
            .map_err(err)?
            .trim()
            .to_string(),
        command("uname", &["-srv"])?,
    );
    let (gpu, driver) = if row.starts_with("cuda-") {
        let output = command(
            "nvidia-smi",
            &[
                "--query-gpu=pci.device_id,name,uuid,driver_version",
                "--format=csv,noheader",
            ],
        )?;
        let lines = output.lines().collect::<Vec<_>>();
        if lines.len() != 1 {
            return Err(refuse(
                "machine identity requires exactly one visible NVIDIA GPU",
            ));
        }
        let fields = lines[0].split(',').map(str::trim).collect::<Vec<_>>();
        if fields.len() != 4 {
            return Err(refuse("invalid nvidia-smi machine identity"));
        }
        let pci = u32::from_str_radix(fields[0].trim_start_matches("0x"), 16).map_err(err)?;
        (
            json!({"vendor_id": pci & 0xffff, "device_id": pci >> 16, "name": fields[1], "uuid": fields[2]}),
            fields[3].to_string(),
        )
    } else {
        let output = command("vulkaninfo", &["--summary"])?;
        let index = floor
            .and_then(|f| f["observed"]["index"].as_u64())
            .ok_or_else(|| refuse("floor probe omitted Vulkan adapter index"))?;
        let section = output
            .split(&format!("GPU{index}:"))
            .nth(1)
            .ok_or_else(|| refuse("vulkaninfo omitted selected adapter"))?;
        let mut properties = BTreeMap::new();
        for line in section
            .lines()
            .take_while(|line| !line.trim_start().starts_with("GPU"))
        {
            if let Some((key, value)) = line.split_once('=') {
                properties.insert(key.trim(), value.trim());
            }
        }
        let get = |key| {
            properties
                .get(key)
                .copied()
                .ok_or_else(|| refuse(format!("Vulkan machine identifier missing: {key}")))
        };
        (
            json!({"vendor_id": get("vendorID")?, "device_id": get("deviceID")?, "name": get("deviceName")?, "uuid": get("deviceUUID")?}),
            format!("{} {}", get("driverName")?, get("driverInfo")?),
        )
    };
    let mut machine =
        json!({"system_product_name": product, "gpu": gpu, "os": os, "driver": driver});
    if row.starts_with("cuda-") {
        machine["driver_api"] = floor.ok_or_else(|| refuse("CUDA floor observation missing"))?
            ["observed"]["driver_api"]
            .clone();
    }
    if let Ok(instance) = std::env::var("SYNAPSE_CERTIFY_INSTANCE_ID") {
        machine["instance_id"] = json!(instance);
    }
    Ok(machine)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_flattened_wire_vectors_and_scores_in_item_order() {
        let embed = json!({"fingerprint": "identity", "vectors": [{"id":"a", "vector":[1.0, 2.0]}, {"id":"b", "vector":[3.0, 4.0]}]});
        let rerank = json!({"fingerprint": "identity", "scores": [0.25, 0.75]});
        assert_eq!(
            observed_output(&embed, 0, "embed").unwrap(),
            Output::Embedding(vec![1.0, 2.0])
        );
        assert_eq!(
            observed_output(&embed, 1, "embed").unwrap(),
            Output::Embedding(vec![3.0, 4.0])
        );
        assert_eq!(
            observed_output(&rerank, 0, "rerank").unwrap(),
            Output::Score(0.25)
        );
        assert_eq!(
            observed_output(&rerank, 1, "rerank").unwrap(),
            Output::Score(0.75)
        );
        assert!(observed_output(&embed, 2, "embed").is_err());
    }

    #[test]
    fn committed_batches_and_pools_use_bounded_multi_item_requests() {
        let parity = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/parity");
        let index: Value =
            serde_json::from_slice(&std::fs::read(parity.join("fixtures/index.json")).unwrap())
                .unwrap();
        for entry in index.as_object().unwrap().values() {
            let bytes = std::fs::read(parity.join(entry["path"].as_str().unwrap())).unwrap();
            assert_eq!(
                synapse_parity::canonical::sha256_hex(&bytes),
                entry["sha256"].as_str().unwrap()
            );
            let document: Value = serde_json::from_slice(&bytes).unwrap();
            let operation = document["operation"].as_str().unwrap();
            let cases = document["cases"].as_array().unwrap();
            let requests = fixture_requests(cases, operation).unwrap();
            assert_eq!(
                requests
                    .iter()
                    .flatten()
                    .map(|case| &case["id"])
                    .collect::<Vec<_>>(),
                cases.iter().map(|case| &case["id"]).collect::<Vec<_>>()
            );
            for request in &requests {
                assert!(request.len() <= 64);
                assert!(
                    request
                        .iter()
                        .map(|case| case["input_ids"].as_array().unwrap().len())
                        .sum::<usize>()
                        <= 8192
                );
                if request[0]["input_ids"].as_array().unwrap().len() == 8192 {
                    assert_eq!(request.len(), 1);
                }
            }
            let batched = requests
                .iter()
                .find(|request| request[0]["category"] == "batched")
                .unwrap();
            let (method, params) = group_params(batched, operation).unwrap();
            if operation == "embed" {
                assert_eq!(method, "embed.batch");
                assert_eq!(params["items"].as_array().unwrap().len(), 6);
                assert_eq!(params["items"][0]["id"], "batch-0");
            } else {
                assert_eq!(method, "rerank.score");
                assert_eq!(params["candidates"].as_array().unwrap().len(), 4);
                let pool = requests
                    .iter()
                    .filter(|request| request[0]["pool"] == "pool-100")
                    .collect::<Vec<_>>();
                assert_eq!(
                    pool.iter().map(|request| request.len()).collect::<Vec<_>>(),
                    [64, 36]
                );
                assert_eq!(pool[0][0]["id"], "pool100-000");
                assert_eq!(pool[1][0]["id"], "pool100-064");
            }
        }
    }
}
