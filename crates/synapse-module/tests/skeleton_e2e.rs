#![forbid(unsafe_code)]

mod common;

use std::{
    fs::OpenOptions,
    io::ErrorKind,
    net::Ipv4Addr,
    path::{Path, PathBuf},
    process,
    time::{Duration, Instant as StdInstant},
};

use common::{
    connect_consumer, raw_route_frame, read_frame_timeout, route_open, route_request,
    unique_temp_dir, wait_for_catalog, MODULE_ID, SETUP_TIMEOUT,
};
use rusqlite::Connection;
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
    time::{sleep, Instant},
};

struct TestDaemon {
    registry: std::sync::Arc<Registry>,
    connection_file_path: PathBuf,
    temp_dir: PathBuf,
    data_home: PathBuf,
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

async fn start_daemon() -> TestDaemon {
    let temp_dir = unique_temp_dir("synapse-e2e-daemon");
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
        daemon_ver: "test-synapse-e2e".to_owned(),
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
        data_home,
        task,
    }
}

fn spawn_synapse_module(subc_connection_file: &Path) -> ModuleProcess {
    spawn_synapse_module_with_preloads(subc_connection_file, None)
}

fn spawn_synapse_module_with_preloads(
    subc_connection_file: &Path,
    preload_models: Option<&str>,
) -> ModuleProcess {
    let mut command = Command::new(env!("CARGO_BIN_EXE_synapse-module"));
    command
        .arg("--subc")
        .arg(subc_connection_file)
        .env("SUBC_MODULE_ID", MODULE_ID)
        .stderr(process::Stdio::inherit())
        .kill_on_drop(true);
    if let Some(preload_models) = preload_models {
        command.env("SYNAPSE_PRELOAD_MODELS", preload_models);
    }
    let child = command.spawn().expect("spawn synapse-module");
    ModuleProcess { child }
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

async fn open_route() -> (TestDaemon, ModuleProcess, tokio::net::TcpStream, u16) {
    open_route_with_preloads(None).await
}

async fn open_route_with_preloads(
    preload_models: Option<&str>,
) -> (TestDaemon, ModuleProcess, tokio::net::TcpStream, u16) {
    let daemon = start_daemon().await;
    let module = match preload_models {
        Some(preload_models) => {
            spawn_synapse_module_with_preloads(&daemon.connection_file_path, Some(preload_models))
        }
        None => spawn_synapse_module(&daemon.connection_file_path),
    };
    wait_for_registration(&daemon.registry, MODULE_ID, SETUP_TIMEOUT).await;

    let project_root = unique_temp_dir("synapse-e2e-project");
    std::fs::create_dir_all(&project_root).unwrap();

    let mut consumer = connect_consumer(&daemon.connection_file_path).await;
    wait_for_catalog(&mut consumer, MODULE_ID, SETUP_TIMEOUT).await;
    let route_channel = route_open(&mut consumer, &project_root, 1).await;
    let _ = std::fs::remove_dir_all(&project_root);
    (daemon, module, consumer, route_channel)
}

fn expected_store_path(data_home: &Path) -> PathBuf {
    data_home.join("cortexkit").join(MODULE_ID).join("store.db")
}

#[tokio::test]
async fn models_list_round_trips_and_opens_the_daemon_delivered_store() {
    let (daemon, _module, mut consumer, route_channel) = open_route().await;
    let body = route_request(
        &mut consumer,
        route_channel,
        2,
        serde_json::json!({ "method": "models.list", "params": {} }),
    )
    .await;

    assert_eq!(body["result"]["module_generation"].as_u64(), Some(1));
    assert_eq!(body["result"]["table_epoch"].as_u64(), Some(0));
    assert_eq!(body["result"]["models"], Value::Array(Vec::new()));
    assert_eq!(body["result"]["alias_rows"], Value::Array(Vec::new()));

    let store_path = expected_store_path(&daemon.data_home);
    assert!(
        store_path.exists(),
        "synapse should open the HELLO_ACK store at {}",
        store_path.display()
    );

    let conn = Connection::open(&store_path).expect("open migrated synapse store");
    for table in ["module_meta", "jobs", "alias_rows", "cert_rows"] {
        let found: Option<String> = conn
            .query_row(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?1",
                [table],
                |row| row.get(0),
            )
            .ok();
        assert_eq!(found.as_deref(), Some(table), "expected table {table}");
    }
    let module_generation: i64 = conn
        .query_row(
            "SELECT module_generation FROM module_meta WHERE id = 0",
            [],
            |row| row.get(0),
        )
        .expect("read module generation");
    assert_eq!(module_generation, 1);
}

#[tokio::test]
async fn embed_query_returns_typed_probe_required_error() {
    let (_daemon, _module, mut consumer, route_channel) = open_route().await;
    let frame = raw_route_frame(
        &mut consumer,
        route_channel,
        3,
        serde_json::json!({
            "method": "embed.query",
            "params": { "text": "hello world" }
        }),
    )
    .await;

    assert_eq!(frame.header.ty, FrameType::Response);
    let body: Value = serde_json::from_slice(&frame.body).expect("decode response body");
    assert_eq!(body["result"]["module_generation"].as_u64(), Some(1));
    assert_eq!(body["result"]["error"]["code"], "probe_required");
    assert_eq!(body["result"]["error"]["class"], "permanent");
    assert_eq!(
        body["result"]["error"]["safe_to_retry_same_request"],
        Value::Bool(false)
    );
}

#[tokio::test]
async fn embed_query_preloaded_minilm_returns_vectors_and_envelope() {
    let Some(preloads) = minilm_preload_config() else {
        eprintln!("skipping MiniLM embed.query e2e: local HF ONNX snapshot is missing");
        return;
    };
    let _lock = acquire_minilm_e2e_lock();
    let (_daemon, _module, mut consumer, route_channel) =
        open_route_with_preloads(Some(&preloads)).await;

    let first = route_request(
        &mut consumer,
        route_channel,
        4,
        serde_json::json!({
            "method": "embed.query",
            "params": { "id": "q1", "text": "hello world" }
        }),
    )
    .await;
    let second = route_request(
        &mut consumer,
        route_channel,
        5,
        serde_json::json!({
            "method": "embed.query",
            "params": { "id": "q2", "text": "hello world again" }
        }),
    )
    .await;

    let result = &first["result"];
    assert_eq!(result["module_generation"].as_u64(), Some(1));
    assert_eq!(result["table_epoch"].as_u64(), Some(0));
    assert_eq!(result["dims"].as_u64(), Some(384));
    assert_eq!(result["equivalent_to"], Value::Array(Vec::new()));
    assert_eq!(result["vectors"].as_array().unwrap().len(), 1);
    assert_eq!(result["vectors"][0]["id"], "q1");
    assert_eq!(
        result["vectors"][0]["vector"].as_array().unwrap().len(),
        384
    );
    assert!(result["real_token_counts"][0].as_u64().unwrap() > 0);
    assert_eq!(
        result["truncation_disclosures"][0]["truncated"],
        Value::Bool(false)
    );
    assert_eq!(result["provenance"]["engine"]["engine"], "ort");
    assert_eq!(result["fingerprint"], second["result"]["fingerprint"]);
}

#[tokio::test]
async fn embed_batch_preloaded_minilm_preserves_order_and_envelope() {
    let Some(preloads) = minilm_preload_config() else {
        eprintln!("skipping MiniLM embed.batch e2e: local HF ONNX snapshot is missing");
        return;
    };
    let _lock = acquire_minilm_e2e_lock();
    let (_daemon, _module, mut consumer, route_channel) =
        open_route_with_preloads(Some(&preloads)).await;
    let items = (0..16)
        .map(|index| {
            serde_json::json!({
                "id": format!("item-{index:02}"),
                "text": format!("the quick brown fox jumps over item {index}")
            })
        })
        .collect::<Vec<_>>();

    let body = route_request(
        &mut consumer,
        route_channel,
        6,
        serde_json::json!({
            "method": "embed.batch",
            "params": { "items": items }
        }),
    )
    .await;
    let result = &body["result"];
    assert_eq!(result["dims"].as_u64(), Some(384));
    assert_eq!(result["real_token_counts"].as_array().unwrap().len(), 16);
    assert_eq!(
        result["truncation_disclosures"].as_array().unwrap().len(),
        16
    );
    let vectors = result["vectors"].as_array().unwrap();
    assert_eq!(vectors.len(), 16);
    for (index, vector) in vectors.iter().enumerate() {
        assert_eq!(vector["id"], format!("item-{index:02}"));
        assert_eq!(vector["vector"].as_array().unwrap().len(), 384);
    }
}

#[tokio::test]
async fn embed_query_deadline_one_returns_typed_rejection() {
    let Some(preloads) = minilm_preload_config() else {
        eprintln!("skipping MiniLM deadline e2e: local HF ONNX snapshot is missing");
        return;
    };
    let _lock = acquire_minilm_e2e_lock();
    let (_daemon, _module, mut consumer, route_channel) =
        open_route_with_preloads(Some(&preloads)).await;

    let body = route_request(
        &mut consumer,
        route_channel,
        7,
        serde_json::json!({
            "method": "embed.query",
            "params": { "text": "deadline should reject", "deadline_ms": 1 }
        }),
    )
    .await;
    let error = &body["result"]["error"];
    assert!(
        error["code"] == "deadline_exceeded" || error["code"] == "queue_full",
        "unexpected typed rejection: {error:?}"
    );
    assert_eq!(error["class"], "transient");
}

#[tokio::test]
async fn concurrent_embed_burst_finishes_with_vectors_or_typed_rejections() {
    let Some(preloads) = minilm_preload_config() else {
        eprintln!("skipping MiniLM burst e2e: local HF ONNX snapshot is missing");
        return;
    };
    let _lock = acquire_minilm_e2e_lock();
    let (_daemon, _module, mut consumer, route_channel) =
        open_route_with_preloads(Some(&preloads)).await;
    let start_corr = 10_000_u64;
    let count = 32_u64;
    for offset in 0..count {
        let frame = Frame::build(
            FrameType::Request,
            Flags::new(false, Priority::Interactive, false),
            route_channel,
            start_corr + offset,
            serde_json::to_vec(&serde_json::json!({
                "method": "embed.query",
                "params": { "id": format!("burst-{offset}"), "text": format!("burst text {offset}") }
            }))
            .unwrap(),
        )
        .unwrap();
        write_frame(&mut consumer, &frame).await.unwrap();
    }

    let mut seen = 0_u64;
    while seen < count {
        let frame = read_frame_timeout(&mut consumer).await;
        if frame.header.corr < start_corr || frame.header.corr >= start_corr + count {
            continue;
        }
        assert_eq!(frame.header.ty, FrameType::Response);
        let body: Value = serde_json::from_slice(&frame.body).expect("decode burst response");
        if let Some(error) = body["result"].get("error") {
            assert!(
                error["code"] == "queue_full" || error["code"] == "deadline_exceeded",
                "unexpected burst error: {error:?}"
            );
        } else {
            assert_eq!(body["result"]["dims"].as_u64(), Some(384));
            assert_eq!(body["result"]["vectors"].as_array().unwrap().len(), 1);
        }
        seen += 1;
    }
}

fn minilm_preload_config() -> Option<String> {
    let snapshot = minilm_onnx_snapshot()?;
    let model_path = snapshot.join("model.onnx");
    let tokenizer_path = snapshot.join("tokenizer.json");
    if !model_path.exists() || !tokenizer_path.exists() {
        return None;
    }
    Some(
        serde_json::json!([{
            "model_id": "minilm",
            "engine": "ort",
            "model_path": model_path,
            "tokenizer_path": tokenizer_path,
            "pooling": "mean",
            "normalize": true,
            "max_tokens": 512,
            "quant": "fp32"
        }])
        .to_string(),
    )
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

fn first_snapshot_with(snapshots: &Path, file_name: &str) -> Option<PathBuf> {
    std::fs::read_dir(snapshots)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| path.join(file_name).exists())
}

struct MinilmE2eLock {
    path: PathBuf,
}

impl Drop for MinilmE2eLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn acquire_minilm_e2e_lock() -> MinilmE2eLock {
    let path = std::env::temp_dir().join("synapse-minilm-e2e.lock");
    let deadline = StdInstant::now() + Duration::from_secs(120);
    loop {
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(_) => return MinilmE2eLock { path },
            Err(error) if error.kind() == ErrorKind::AlreadyExists => {
                if stale_lock(&path) {
                    let _ = std::fs::remove_file(&path);
                    continue;
                }
                assert!(
                    StdInstant::now() < deadline,
                    "timed out waiting for MiniLM e2e lock at {}",
                    path.display()
                );
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(error) => panic!("create MiniLM e2e lock {}: {error}", path.display()),
        }
    }
}

fn stale_lock(path: &Path) -> bool {
    let Ok(metadata) = std::fs::metadata(path) else {
        return false;
    };
    let Ok(modified) = metadata.modified() else {
        return false;
    };
    modified
        .elapsed()
        .map(|age| age > Duration::from_secs(120))
        .unwrap_or(false)
}
