#![cfg(unix)]
#![forbid(unsafe_code)]

mod common;

use std::{
    collections::{hash_map::DefaultHasher, BTreeSet},
    hash::{Hash, Hasher},
    net::Ipv4Addr,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process,
    time::Duration,
};

use common::{
    connect_consumer, raw_route_frame, read_frame_timeout, route_open, route_request,
    unique_temp_dir, wait_for_catalog, TestRoute, MODULE_ID, SETUP_TIMEOUT,
};
use serde_json::Value;
use subc_core::{
    daemon_config::StorageConfig, serve_listener, write_frame, ControlHandler, Frame, Registry,
    Router, ServerAuth,
};
use subc_protocol::{Flags, FrameType, Priority};
use subc_transport::{
    generate_daemon_id, generate_key, write_atomic, ConnectionInfo, Endpoint, SCHEMA_VERSION,
};
use tokio::{
    net::TcpListener,
    process::{Child, Command},
    time::{sleep, timeout, Instant},
};

const SOAK_CONSUMERS: usize = 4;
const SUSTAINED_QUERIES_PER_CONSUMER: usize = 16;
const SOAK_JOB_ITEMS: usize = 5_000;
const BURST_CONNECTIONS: usize = 8;
const BURST_PER_CONNECTION: usize = 8;
const BURST_DEADLINE_MS: u64 = 2_000;

struct TestDaemon {
    registry: std::sync::Arc<Registry>,
    connection_file_path: PathBuf,
    temp_dir: PathBuf,
    task: tokio::task::JoinHandle<Result<(), subc_core::ServerError>>,
}

impl Drop for TestDaemon {
    fn drop(&mut self) {
        self.task.abort();
        let _ = std::fs::remove_dir_all(&self.temp_dir);
    }
}

struct ModuleProcess {
    child: Child,
}

impl Drop for ModuleProcess {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

struct SoakAssets {
    onnx_snapshot: PathBuf,
    gguf_snapshot: PathBuf,
    worker_bin: PathBuf,
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires local MiniLM ONNX, MiniLM GGUF, and llama worker assets"]
async fn mixed_load_burst_and_idempotent_job_invariants_hold() {
    let Some(assets) = soak_assets() else {
        eprintln!("skipping soak: MiniLM ONNX/GGUF snapshots or llama worker binary are missing");
        return;
    };
    let config_temp = unique_temp_dir("synapse-soak-config");
    std::fs::create_dir_all(&config_temp).unwrap();
    let config = soak_config(&assets, &assets.worker_bin, &config_temp);
    let (daemon, _module, mut control, control_route) = open_route_with_config(&config).await;
    certify_models(
        &mut control,
        control_route,
        100,
        &["minilm-ort", "minilm-llama"],
    )
    .await;

    let job_body = serde_json::json!({
        "method": "embed.batch",
        "params": {
            "model": "minilm-ort",
            "request_key": "soak-idempotent-5k",
            "items": soak_items(SOAK_JOB_ITEMS),
        }
    });
    let mut race_handles = Vec::new();
    for index in 0..SOAK_CONSUMERS {
        let (mut stream, route) =
            open_additional_route(&daemon.connection_file_path, 10_000 + (index as u64) * 100)
                .await;
        let body = job_body.clone();
        race_handles.push(tokio::spawn(async move {
            let response = route_request(&mut stream, route, 10_100 + index as u64, body).await;
            ((stream, route), response)
        }));
    }

    let mut consumers = Vec::new();
    let mut admitted_job_ids = BTreeSet::new();
    for handle in race_handles {
        let (consumer, response) = handle.await.unwrap();
        let job_id = response["result"]["job_id"]
            .as_str()
            .expect("job admission returns job_id")
            .to_string();
        admitted_job_ids.insert(job_id);
        consumers.push(consumer);
    }
    assert_eq!(
        admitted_job_ids.len(),
        1,
        "concurrent resubmission with one request_key must admit one job"
    );
    let job_id = admitted_job_ids.into_iter().next().unwrap();

    let query_handles = consumers
        .into_iter()
        .enumerate()
        .map(|(consumer_index, (mut stream, route))| {
            tokio::spawn(async move {
                let mut generations = Vec::new();
                for query_index in 0..SUSTAINED_QUERIES_PER_CONSUMER {
                    let corr = 20_000
                        + (consumer_index as u64) * 1_000
                        + u64::try_from(query_index).unwrap();
                    let body = route_request_with_wall_timeout(
                        &mut stream,
                        route,
                        corr,
                        serde_json::json!({
                            "method": "embed.query",
                            "params": {
                                "model": "minilm-ort",
                                "id": format!("sustain-{consumer_index}-{query_index}"),
                                "text": format!("sustained soak query {consumer_index} {query_index}"),
                                "deadline_ms": 5_000,
                                "max_queue_ms": 750,
                            }
                        }),
                        Duration::from_secs(10),
                    )
                    .await;
                    let body = response_body(body);
                    assert_vector_or_typed_rejection(&body, &["queue_full", "deadline_exceeded"]);
                    generations.push(body["result"]["module_generation"].as_u64());
                }
                generations
            })
        })
        .collect::<Vec<_>>();

    let probe = route_request(
        &mut control,
        control_route,
        30_000,
        serde_json::json!({
            "method": "probe.start",
            "params": { "request_key": "soak-probe-mid", "models": ["minilm-ort"] }
        }),
    )
    .await;
    let probe_id = probe["result"]["job_id"].as_str().unwrap().to_string();
    let probe_done = poll_probe_status(&mut control, control_route, 30_100, &probe_id).await;
    assert_eq!(probe_done["result"]["state"], "done");

    let job_done = poll_embed_job(&mut control, control_route, 40_000, &job_id).await;
    assert_eq!(job_done["result"]["state"], "done");
    let mut returned_items = job_done["result"]["vectors"]
        .as_array()
        .map(Vec::len)
        .unwrap_or(0);
    let page_count = job_done["result"]["page_count"].as_u64().unwrap_or(1);
    for page in 1..page_count {
        let page_body = route_request(
            &mut control,
            control_route,
            41_000 + page,
            serde_json::json!({
                "method": "embed.result",
                "params": { "job_id": &job_id, "page": page }
            }),
        )
        .await;
        returned_items += page_body["result"]["vectors"]
            .as_array()
            .map(Vec::len)
            .unwrap_or(0);
        assert_eq!(page_body["result"]["module_generation"].as_u64(), Some(1));
    }
    assert_eq!(returned_items, SOAK_JOB_ITEMS);

    for handle in query_handles {
        for generation in handle.await.unwrap() {
            assert_eq!(generation, Some(1));
        }
    }

    burst_queries_fast_fail_without_hangs(&daemon.connection_file_path).await;
    wait_for_inline_drain(&mut control, control_route, 50_000).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires local MiniLM ONNX, MiniLM GGUF, and llama worker assets"]
async fn crash_budget_quarantines_llama_without_affecting_ort_lane() {
    let Some(assets) = soak_assets() else {
        eprintln!(
            "skipping crash soak: MiniLM ONNX/GGUF snapshots or llama worker binary are missing"
        );
        return;
    };
    let config_temp = unique_temp_dir("synapse-soak-crash-config");
    std::fs::create_dir_all(&config_temp).unwrap();
    let wrapper = aborting_worker_wrapper(&assets.worker_bin, &config_temp);
    let config = soak_config(&assets, &wrapper, &config_temp);
    let (_daemon, _module, mut control, control_route) = open_route_with_config(&config).await;
    certify_models(&mut control, control_route, 60_000, &["minilm-ort"]).await;

    for attempt in 0..3_u64 {
        let ort = route_request(
            &mut control,
            control_route,
            61_000 + attempt,
            serde_json::json!({
                "method": "embed.query",
                "params": {
                    "model": "minilm-ort",
                    "id": format!("ort-during-crash-{attempt}"),
                    "text": format!("ort lane remains isolated during crash attempt {attempt}")
                }
            }),
        )
        .await;
        assert_eq!(ort["result"]["dims"].as_u64(), Some(384));
        assert_eq!(ort["result"]["module_generation"].as_u64(), Some(1));

        let accepted = route_request(
            &mut control,
            control_route,
            62_000 + attempt,
            serde_json::json!({
                "method": "probe.start",
                "params": {
                    "request_key": format!("soak-crash-llama-{attempt}"),
                    "models": ["minilm-llama"]
                }
            }),
        )
        .await;
        let probe_id = accepted["result"]["job_id"].as_str().unwrap().to_string();
        let done = poll_probe_status(
            &mut control,
            control_route,
            63_000 + attempt * 100,
            &probe_id,
        )
        .await;
        assert_eq!(done["result"]["state"], "done");
        let lane = &done["result"]["lanes"][0];
        assert_eq!(lane["status"], "uncertified");
        let code = lane["error"]["code"].as_str().unwrap_or("");
        if attempt < 2 {
            assert_eq!(
                code, "engine_crashed",
                "probe should expose typed crash: {lane:?}"
            );
        } else {
            assert_eq!(
                code, "probe_required",
                "quarantine should require a fresh probe: {lane:?}"
            );
            assert!(lane["error"]["message"]
                .as_str()
                .unwrap_or("")
                .contains("quarantined"));
        }
    }

    let status = route_request(
        &mut control,
        control_route,
        64_000,
        serde_json::json!({ "method": "model.status", "params": {} }),
    )
    .await;
    let llama_health = status["result"]["health"]["lanes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|lane| lane["model_id"] == "minilm-llama")
        .and_then(|lane| lane.get("worker"))
        .expect("llama lane exposes cached worker health");
    assert_eq!(llama_health["degraded"], true);
    assert!(llama_health["quarantined_models"].as_u64().unwrap_or(0) >= 1);

    let ort_after = route_request(
        &mut control,
        control_route,
        65_000,
        serde_json::json!({
            "method": "embed.query",
            "params": { "model": "minilm-ort", "text": "ort lane still answers after quarantine" }
        }),
    )
    .await;
    assert_eq!(ort_after["result"]["dims"].as_u64(), Some(384));
}

async fn start_daemon() -> TestDaemon {
    let temp_dir = unique_temp_dir("synapse-soak-daemon");
    let data_home = temp_dir.join("data-home");
    std::fs::create_dir_all(&data_home).unwrap();
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let connection_file_path = temp_dir.join("subc-conn.json");
    let conn = ConnectionInfo {
        schema: SCHEMA_VERSION,
        endpoints: vec![Endpoint {
            host: Ipv4Addr::LOCALHOST.to_string(),
            port,
        }],
        key: generate_key().unwrap(),
        daemon_id: generate_daemon_id().unwrap(),
        pid: process::id(),
        daemon_ver: "test-synapse-soak".to_owned(),
        wire_version: Some(subc_protocol::PROTOCOL_VERSION),
    };
    write_atomic(&connection_file_path, &conn).unwrap();

    let registry = std::sync::Arc::new(Registry::default());
    let control = ControlHandler::new(std::sync::Arc::clone(&registry)).with_storage_config(Some(
        StorageConfig::Sqlite {
            data_home: data_home.clone(),
        },
    ));
    let router = std::sync::Arc::new(Router::with_control_handler(std::sync::Arc::new(control)));
    let auth = ServerAuth::new(conn.key, conn.daemon_id, conn.daemon_ver);
    let task = tokio::spawn(serve_listener(listener, router, auth));

    TestDaemon {
        registry,
        connection_file_path,
        temp_dir,
        task,
    }
}

fn spawn_synapse_module_with_config(
    subc_connection_file: &Path,
    config_json: &str,
) -> ModuleProcess {
    let mut command = Command::new(env!("CARGO_BIN_EXE_ck-synapse"));
    command
        .arg("--subc")
        .arg(subc_connection_file)
        .env("SUBC_MODULE_ID", MODULE_ID)
        .env("SYNAPSE_CONFIG_JSON", config_json)
        .stderr(process::Stdio::inherit())
        .kill_on_drop(true);
    let child = command.spawn().expect("spawn synapse-module");
    ModuleProcess { child }
}

async fn open_route_with_config(
    config_json: &str,
) -> (TestDaemon, ModuleProcess, tokio::net::TcpStream, TestRoute) {
    let daemon = start_daemon().await;
    let module = spawn_synapse_module_with_config(&daemon.connection_file_path, config_json);
    wait_for_registration(&daemon.registry, MODULE_ID, SETUP_TIMEOUT).await;
    let (consumer, route) = open_additional_route(&daemon.connection_file_path, 1).await;
    (daemon, module, consumer, route)
}

async fn open_additional_route(
    connection_file_path: &Path,
    corr: u64,
) -> (tokio::net::TcpStream, TestRoute) {
    let project_root = unique_temp_dir("synapse-soak-project");
    std::fs::create_dir_all(&project_root).unwrap();
    let mut consumer = connect_consumer(connection_file_path).await;
    wait_for_catalog(&mut consumer, MODULE_ID, SETUP_TIMEOUT).await;
    let route = route_open(&mut consumer, &project_root, corr).await;
    let _ = std::fs::remove_dir_all(&project_root);
    (consumer, route)
}

async fn wait_for_registration(registry: &Registry, module_id: &str, wait: Duration) {
    let deadline = Instant::now() + wait;
    loop {
        if registry.get_module(module_id).unwrap().is_some() {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "module {module_id} did not register within {wait:?}"
        );
        sleep(Duration::from_millis(20)).await;
    }
}

async fn certify_models(
    consumer: &mut tokio::net::TcpStream,
    route: TestRoute,
    start_corr: u64,
    models: &[&str],
) -> Value {
    let accepted = route_request(
        consumer,
        route,
        start_corr,
        serde_json::json!({
            "method": "probe.start",
            "params": { "request_key": format!("soak-certify-{start_corr}"), "models": models }
        }),
    )
    .await;
    let job_id = accepted["result"]["job_id"].as_str().unwrap().to_string();
    let done = poll_probe_status(consumer, route, start_corr + 1, &job_id).await;
    assert_eq!(done["result"]["state"], "done");
    for lane in done["result"]["lanes"].as_array().unwrap() {
        assert_eq!(lane["status"], "certified", "probe lane failed: {lane:?}");
    }
    done
}

async fn poll_probe_status(
    consumer: &mut tokio::net::TcpStream,
    route: TestRoute,
    start_corr: u64,
    job_id: &str,
) -> Value {
    let deadline = Instant::now() + Duration::from_secs(180);
    let mut corr = start_corr;
    loop {
        let body = route_request(
            consumer,
            route,
            corr,
            serde_json::json!({
                "method": "probe.status",
                "params": { "job_id": job_id }
            }),
        )
        .await;
        match body["result"]["state"].as_str() {
            Some("done") | Some("failed_transient") | Some("failed_permanent") => return body,
            Some("queued" | "running") => {
                assert!(
                    Instant::now() < deadline,
                    "probe did not complete: {body:?}"
                );
                corr += 1;
                sleep(Duration::from_millis(100)).await;
            }
            other => panic!("unexpected probe.status state {other:?}: {body:?}"),
        }
    }
}

async fn poll_embed_job(
    consumer: &mut tokio::net::TcpStream,
    route: TestRoute,
    start_corr: u64,
    job_id: &str,
) -> Value {
    let deadline = Instant::now() + Duration::from_secs(240);
    let mut corr = start_corr;
    loop {
        let body = route_request(
            consumer,
            route,
            corr,
            serde_json::json!({
                "method": "embed.result",
                "params": { "job_id": job_id }
            }),
        )
        .await;
        match body["result"]["state"].as_str() {
            Some("done") | Some("failed_transient") | Some("failed_permanent") => return body,
            Some("queued" | "running") => {
                assert!(
                    Instant::now() < deadline,
                    "embed job did not drain: {body:?}"
                );
                corr += 1;
                sleep(Duration::from_millis(100)).await;
            }
            other => panic!("unexpected embed.result state {other:?}: {body:?}"),
        }
    }
}

async fn route_request_with_wall_timeout(
    stream: &mut tokio::net::TcpStream,
    route: TestRoute,
    corr: u64,
    body: Value,
    wall_timeout: Duration,
) -> Frame {
    timeout(wall_timeout, raw_route_frame(stream, route, corr, body))
        .await
        .expect("route request exceeded wall timeout")
}

fn response_body(frame: Frame) -> Value {
    assert_eq!(frame.header.ty, FrameType::Response);
    serde_json::from_slice(&frame.body).expect("response body is JSON")
}

fn assert_vector_or_typed_rejection(body: &Value, allowed_errors: &[&str]) {
    if let Some(error) = body["result"].get("error") {
        let code = error["code"].as_str().unwrap_or("");
        assert!(
            allowed_errors.contains(&code),
            "unexpected typed rejection {error:?}"
        );
    } else {
        assert_eq!(body["result"]["dims"].as_u64(), Some(384));
        assert_eq!(body["result"]["vectors"].as_array().unwrap().len(), 1);
    }
}

async fn burst_queries_fast_fail_without_hangs(connection_file_path: &Path) {
    let mut handles = Vec::new();
    for connection_index in 0..BURST_CONNECTIONS {
        let (mut stream, route) = open_additional_route(
            connection_file_path,
            70_000 + (connection_index as u64) * 100,
        )
        .await;
        handles.push(tokio::spawn(async move {
            let start_corr = 80_000 + (connection_index as u64) * 1_000;
            let started = Instant::now();
            for offset in 0..BURST_PER_CONNECTION {
                let corr = start_corr + u64::try_from(offset).unwrap();
                let frame = Frame::build(
                    FrameType::Request,
                    Flags::new(false, Priority::Interactive, false),
                    route.channel,
                    route.epoch,
                    corr,
                    serde_json::to_vec(&serde_json::json!({
                        "method": "embed.query",
                        "params": {
                            "model": "minilm-ort",
                            "id": format!("burst-{connection_index}-{offset}"),
                            "text": format!("burst query {connection_index} {offset}"),
                            "deadline_ms": BURST_DEADLINE_MS,
                            "max_queue_ms": 0,
                        }
                    }))
                    .unwrap(),
                )
                .unwrap();
                write_frame(&mut stream, &frame).await.unwrap();
            }

            let mut seen = 0;
            while seen < BURST_PER_CONNECTION {
                let frame = timeout(
                    Duration::from_millis(BURST_DEADLINE_MS * 2),
                    read_frame_timeout(&mut stream),
                )
                .await
                .expect("burst response exceeded 2x deadline");
                if frame.header.corr < start_corr
                    || frame.header.corr
                        >= start_corr + u64::try_from(BURST_PER_CONNECTION).unwrap()
                {
                    continue;
                }
                assert!(
                    started.elapsed() <= Duration::from_millis(BURST_DEADLINE_MS * 2),
                    "burst response exceeded 2x deadline"
                );
                let body = response_body(frame);
                assert_vector_or_typed_rejection(&body, &["queue_full", "deadline_exceeded"]);
                assert_eq!(body["result"]["module_generation"].as_u64(), Some(1));
                seen += 1;
            }
        }));
    }
    for handle in handles {
        handle.await.unwrap();
    }
}

async fn wait_for_inline_drain(
    consumer: &mut tokio::net::TcpStream,
    route: TestRoute,
    start_corr: u64,
) {
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut corr = start_corr;
    loop {
        let body = route_request(
            consumer,
            route,
            corr,
            serde_json::json!({ "method": "admission.status", "params": {} }),
        )
        .await;
        if body["result"]["inline_in_flight_bytes"].as_u64() == Some(0) {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "inline bytes did not drain: {body:?}"
        );
        corr += 1;
        sleep(Duration::from_millis(50)).await;
    }
}

fn soak_items(count: usize) -> Vec<Value> {
    (0..count)
        .map(|index| {
            serde_json::json!({
                "id": format!("soak-item-{index:05}"),
                "text": format!("x {index}")
            })
        })
        .collect()
}

fn soak_config(assets: &SoakAssets, worker_bin: &Path, temp_dir: &Path) -> String {
    let tokenizer_path = assets.onnx_snapshot.join("tokenizer.json");
    let onnx_model = assets.onnx_snapshot.join("model.onnx");
    let gguf_model = assets
        .gguf_snapshot
        .join("all-MiniLM-L6-v2-ggml-model-f16.gguf");
    serde_json::json!({
        "preload_models": [
            {
                "model_id": "minilm-ort",
                "engine": "ort",
                "model_path": onnx_model,
                "tokenizer_path": tokenizer_path,
                "pooling": "mean",
                "normalize": true,
                "max_tokens": 512,
                "quant": "fp32"
            },
            {
                "model_id": "minilm-llama",
                "engine": "llama",
                "model_path": gguf_model,
                "tokenizer_path": assets.onnx_snapshot.join("tokenizer.json"),
                "worker_bin": worker_bin,
                "worker_runtime_dir": short_worker_runtime_dir(temp_dir),
                "pooling": "mean",
                "normalize": true,
                "max_tokens": 512,
                "format": "gguf",
                "quant": "f16"
            }
        ],
        "inline": {
            "max_items": 8,
            "max_tokens": 256,
            "byte_budget": 1_048_576,
            "max_queue_ms": 250,
            "deadline_ms": 5_000,
            "estimated_execution_ms": 50,
            "max_concurrent_workers": 2
        },
        "jobs": {
            "execution_ttl_ms": 300_000,
            "result_page_bytes": 262_144,
            "bulk_quantum_tokens": 65_536
        }
    })
    .to_string()
}

fn short_worker_runtime_dir(temp_dir: &Path) -> PathBuf {
    let mut hasher = DefaultHasher::new();
    temp_dir.hash(&mut hasher);
    PathBuf::from(format!("/tmp/synw-{:016x}", hasher.finish()))
}

fn aborting_worker_wrapper(real_worker: &Path, temp_dir: &Path) -> PathBuf {
    let wrapper = temp_dir.join("ck-synapse-worker-llama-abort.sh");
    let script = format!(
        "#!/bin/sh\nexec '{}' \"$@\" --test-abort\n",
        real_worker.display()
    );
    std::fs::write(&wrapper, script).unwrap();
    let mut perms = std::fs::metadata(&wrapper).unwrap().permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&wrapper, perms).unwrap();
    wrapper
}

fn soak_assets() -> Option<SoakAssets> {
    let onnx_snapshot = minilm_onnx_snapshot()?;
    let gguf_snapshot = minilm_gguf_snapshot()?;
    let worker_bin = llama_worker_bin()?;
    let required = [
        onnx_snapshot.join("model.onnx"),
        onnx_snapshot.join("tokenizer.json"),
        gguf_snapshot.join("all-MiniLM-L6-v2-ggml-model-f16.gguf"),
        worker_bin.clone(),
    ];
    required
        .iter()
        .all(|path| path.exists())
        .then_some(SoakAssets {
            onnx_snapshot,
            gguf_snapshot,
            worker_bin,
        })
}

fn llama_worker_bin() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("SYNAPSE_LLAMA_WORKER_BIN") {
        return Some(PathBuf::from(path));
    }
    let current_exe = std::env::current_exe().ok()?;
    let debug_dir = current_exe.parent()?.parent()?;
    let candidate = debug_dir.join("ck-synapse-worker-llama");
    candidate.exists().then_some(candidate)
}

fn minilm_onnx_snapshot() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("SYNAPSE_MINILM_ONNX_SNAPSHOT") {
        return Some(PathBuf::from(path));
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Qdrant--all-MiniLM-L6-v2-onnx/snapshots");
    let manual = snapshots.join("manual");
    if manual.exists() {
        return Some(manual);
    }
    first_snapshot_with(&snapshots, "model.onnx")
}

fn minilm_gguf_snapshot() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("SYNAPSE_MINILM_GGUF_SNAPSHOT") {
        return Some(PathBuf::from(path));
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots = PathBuf::from(home).join(
        ".cache/huggingface/hub/models--second-state--All-MiniLM-L6-v2-Embedding-GGUF/snapshots",
    );
    first_snapshot_with(&snapshots, "all-MiniLM-L6-v2-ggml-model-f16.gguf")
}

fn first_snapshot_with(snapshots: &Path, file_name: &str) -> Option<PathBuf> {
    std::fs::read_dir(snapshots)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| path.join(file_name).exists())
}
