#![forbid(unsafe_code)]

mod common;

use std::{
    net::Ipv4Addr,
    path::{Path, PathBuf},
    process,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use common::{
    connect_consumer, raw_route_frame, read_frame_timeout, route_open, route_request,
    unique_temp_dir, wait_for_catalog, TestRoute, MODULE_ID, SETUP_TIMEOUT,
};
use rusqlite::{params, Connection};
use serde_json::Value;
use subc_core::{
    daemon_config::StorageConfig, serve_listener, write_frame, ControlHandler, Frame, Registry,
    Router, ServerAuth,
};
use subc_protocol::{Flags, FrameType, Priority};
use subc_transport::{
    generate_daemon_id, generate_key, write_atomic, ConnectionInfo, Endpoint, SCHEMA_VERSION,
};
use synapse_core::worker_engine_names::{DECODE_WORKER_ENGINE, LLAMA_WORKER_ENGINE};
use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace, Tokenizer};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
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

struct RemoteMockProvider {
    port: u16,
    failures_remaining: Arc<AtomicUsize>,
    requests: Arc<AtomicUsize>,
    task: tokio::task::JoinHandle<()>,
}

impl RemoteMockProvider {
    async fn start() -> Self {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let failures_remaining = Arc::new(AtomicUsize::new(0));
        let requests = Arc::new(AtomicUsize::new(0));
        let task_failures = Arc::clone(&failures_remaining);
        let task_requests = Arc::clone(&requests);
        let task = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    return;
                };
                let failures = Arc::clone(&task_failures);
                let requests = Arc::clone(&task_requests);
                tokio::spawn(async move {
                    serve_remote_mock_request(stream, failures, requests).await;
                });
            }
        });
        Self {
            port,
            failures_remaining,
            requests,
            task,
        }
    }

    fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}/v1", self.port)
    }

    fn fail_next(&self, count: usize) {
        self.failures_remaining.store(count, Ordering::SeqCst);
    }

    fn request_count(&self) -> usize {
        self.requests.load(Ordering::SeqCst)
    }
}

impl Drop for RemoteMockProvider {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn serve_remote_mock_request(
    mut stream: TcpStream,
    failures_remaining: Arc<AtomicUsize>,
    requests: Arc<AtomicUsize>,
) {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 4096];
    let header_end = loop {
        let Ok(read) = stream.read(&mut buffer).await else {
            return;
        };
        if read == 0 {
            return;
        }
        request.extend_from_slice(&buffer[..read]);
        if let Some(index) = request.windows(4).position(|window| window == b"\r\n\r\n") {
            break index + 4;
        }
    };
    let headers = String::from_utf8_lossy(&request[..header_end]);
    let content_length = headers
        .lines()
        .find_map(|line| {
            line.split_once(':').and_then(|(name, value)| {
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            })
        })
        .unwrap_or(0);
    while request.len() < header_end + content_length {
        let Ok(read) = stream.read(&mut buffer).await else {
            return;
        };
        if read == 0 {
            return;
        }
        request.extend_from_slice(&buffer[..read]);
    }
    requests.fetch_add(1, Ordering::SeqCst);
    if failures_remaining
        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
            remaining.checked_sub(1)
        })
        .is_ok()
    {
        let _ =
            write_mock_http_response(&mut stream, 500, serde_json::json!({"error":"storm"})).await;
        return;
    }
    let body: Value = serde_json::from_slice(&request[header_end..header_end + content_length])
        .unwrap_or_else(|_| serde_json::json!({"input": []}));
    let inputs = body["input"].as_array().cloned().unwrap_or_default();
    if inputs.iter().any(|input| {
        input
            .as_str()
            .is_some_and(|text| text.split_whitespace().count() > 128)
    }) {
        let _ = write_mock_http_response(
            &mut stream,
            400,
            serde_json::json!({"error":"input exceeds context window"}),
        )
        .await;
        return;
    }
    let data = inputs
        .iter()
        .enumerate()
        .map(|(index, _)| {
            serde_json::json!({
                "index": index,
                "embedding": [index as f64 + 1.0, 1.0, 0.5]
            })
        })
        .collect::<Vec<_>>();
    let _ = write_mock_http_response(
        &mut stream,
        200,
        serde_json::json!({
            "model":"mock-embed",
            "data":data,
            "usage":{"prompt_tokens":inputs.len(),"total_tokens":inputs.len()}
        }),
    )
    .await;
}

async fn write_mock_http_response(
    stream: &mut TcpStream,
    status: u16,
    body: Value,
) -> std::io::Result<()> {
    let body = serde_json::to_vec(&body).unwrap();
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        _ => "Internal Server Error",
    };
    let headers = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nX-Request-Id: mock-{status}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(headers.as_bytes()).await?;
    stream.write_all(&body).await
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
    spawn_synapse_module_with_env(subc_connection_file, preload_models, None, &[])
}

fn spawn_synapse_module_with_config(
    subc_connection_file: &Path,
    config_json: &str,
) -> ModuleProcess {
    spawn_synapse_module_with_env(subc_connection_file, None, Some(config_json), &[])
}

fn spawn_synapse_module_on_shared_lease(
    subc_connection_file: &Path,
    lease_root: &Path,
) -> ModuleProcess {
    spawn_synapse_module_with_env(
        subc_connection_file,
        None,
        None,
        &[(
            "CORTEXKIT_LEASE_ROOT",
            lease_root.to_string_lossy().as_ref(),
        )],
    )
}

fn spawn_synapse_module_with_env(
    subc_connection_file: &Path,
    preload_models: Option<&str>,
    config_json: Option<&str>,
    extra_env: &[(&str, &str)],
) -> ModuleProcess {
    let lease_root = unique_temp_dir("synapse-module-lease");
    std::fs::create_dir_all(&lease_root).unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_ck-synapse"));
    command
        .arg("--subc")
        .arg(subc_connection_file)
        .env("SUBC_MODULE_ID", MODULE_ID)
        .env(
            "CORTEXKIT_LEASE_ROOT",
            lease_root.to_string_lossy().to_string(),
        )
        .stderr(process::Stdio::inherit())
        .kill_on_drop(true);
    // Tests exercise the production config path: write a real file and point
    // SYNAPSE_CONFIG_PATH at it (the only config env var the module honors).
    let config_contents = if let Some(config_json) = config_json {
        Some(config_json.to_string())
    } else {
        preload_models.map(|preload_models| {
            serde_json::json!({ "preload_models": serde_json::from_str::<Value>(preload_models).unwrap_or_else(|_| serde_json::json!([])) }).to_string()
        })
    };
    if let Some(contents) = config_contents {
        let config_path = lease_root.join("synapse.jsonc");
        std::fs::write(&config_path, contents).unwrap();
        command.env(
            "SYNAPSE_CONFIG_PATH",
            config_path.to_string_lossy().to_string(),
        );
    }
    for (key, value) in extra_env {
        command.env(key, value);
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

async fn open_route() -> (TestDaemon, ModuleProcess, tokio::net::TcpStream, TestRoute) {
    open_route_with_preloads(None).await
}

async fn open_route_with_preloads(
    preload_models: Option<&str>,
) -> (TestDaemon, ModuleProcess, tokio::net::TcpStream, TestRoute) {
    let daemon = start_daemon().await;
    let module = match preload_models {
        Some(preload_models) => {
            spawn_synapse_module_with_preloads(&daemon.connection_file_path, Some(preload_models))
        }
        None => spawn_synapse_module(&daemon.connection_file_path),
    };
    open_route_for_started_module(daemon, module).await
}

async fn open_route_with_config(
    config_json: &str,
) -> (TestDaemon, ModuleProcess, tokio::net::TcpStream, TestRoute) {
    let daemon = start_daemon().await;
    let module = spawn_synapse_module_with_config(&daemon.connection_file_path, config_json);
    open_route_for_started_module(daemon, module).await
}

async fn open_route_for_started_module(
    daemon: TestDaemon,
    module: ModuleProcess,
) -> (TestDaemon, ModuleProcess, tokio::net::TcpStream, TestRoute) {
    wait_for_registration(&daemon.registry, MODULE_ID, SETUP_TIMEOUT).await;

    let project_root = unique_temp_dir("synapse-e2e-project");
    std::fs::create_dir_all(&project_root).unwrap();

    let mut consumer = connect_consumer(&daemon.connection_file_path).await;
    wait_for_catalog(&mut consumer, MODULE_ID, SETUP_TIMEOUT).await;
    let route = route_open(&mut consumer, &project_root, 1).await;
    let _ = std::fs::remove_dir_all(&project_root);
    (daemon, module, consumer, route)
}

fn expected_store_path(data_home: &Path) -> PathBuf {
    data_home.join("cortexkit").join(MODULE_ID).join("store.db")
}

#[tokio::test]
async fn models_list_round_trips_and_opens_the_daemon_delivered_store() {
    let (daemon, _module, mut consumer, route) = open_route().await;
    let body = route_request(
        &mut consumer,
        route,
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
    for table in ["module_meta", "jobs", "alias_rows", "cert_rows", "models"] {
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

#[cfg(target_os = "macos")]
#[tokio::test]
async fn probe_worker_death_fails_the_durable_job_with_engine_crashed() {
    let root = unique_temp_dir("synapse-probe-worker-crash");
    let config = timeout_ane_probe_config(&root);
    let daemon = start_daemon().await;
    let module = spawn_synapse_module_with_env(
        &daemon.connection_file_path,
        None,
        Some(&config),
        &[("SYNAPSE_TIMEOUT_WORKER_ABORT_ON_EMBED", "1")],
    );
    let (_daemon, _module, mut consumer, route) =
        open_route_for_started_module(daemon, module).await;

    let accepted = route_request(
        &mut consumer,
        route,
        3_000,
        serde_json::json!({
            "method": "probe.start",
            "params": {
                "request_key": "probe-worker-crash",
                "models": ["minilm-ane-timeout"]
            }
        }),
    )
    .await;
    let job_id = accepted["result"]["job_id"].as_str().unwrap().to_string();
    let started = Instant::now();
    let body = poll_probe_status(&mut consumer, route, 3_001, &job_id).await;

    assert!(
        started.elapsed() < Duration::from_secs(30),
        "worker death must settle within the short request timeout: {body:?}"
    );
    assert_eq!(
        body["result"]["state"], "failed_transient",
        "probe worker crash response: {body:?}"
    );
    assert_eq!(body["result"]["error"]["code"], "engine_crashed");
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn ane_probe_placement_ping_runs_outside_the_module_runtime() {
    let root = unique_temp_dir("synapse-probe-placement-ping");
    let config = timeout_ane_probe_config(&root);
    let daemon = start_daemon().await;
    let module = spawn_synapse_module_with_env(
        &daemon.connection_file_path,
        None,
        Some(&config),
        &[
            ("SYNAPSE_TIMEOUT_WORKER_EMBED_DIMS", "384"),
            ("SYNAPSE_TIMEOUT_WORKER_EMBED_N", "64"),
        ],
    );
    let (_daemon, _module, mut consumer, route) =
        open_route_for_started_module(daemon, module).await;

    let accepted = route_request(
        &mut consumer,
        route,
        3_100,
        serde_json::json!({
            "method": "probe.start",
            "params": {
                "request_key": "probe-placement-ping",
                "models": ["minilm-ane-timeout"]
            }
        }),
    )
    .await;
    let job_id = accepted["result"]["job_id"].as_str().unwrap().to_string();
    let body = poll_probe_status(&mut consumer, route, 3_101, &job_id).await;

    assert_eq!(body["result"]["state"], "done");
    assert_eq!(body["result"]["lanes"][0]["status"], "uncertified");
}

#[cfg(target_os = "macos")]
fn timeout_ane_probe_config(root: &Path) -> String {
    std::fs::create_dir_all(root).unwrap();
    let model_path = root.join("model.mock");
    let tokenizer_path = root.join("tokenizer.json");
    std::fs::write(&model_path, b"timeout-worker-model").unwrap();
    std::fs::write(
        &tokenizer_path,
        r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {"type": "WordLevel", "vocab": {"[UNK]": 0}, "unk_token": "[UNK]"}
        }"#,
    )
    .unwrap();
    let worker_runtime_dir = PathBuf::from(format!(
        "/tmp/synp-{}",
        root.file_name().unwrap().to_string_lossy()
    ));
    serde_json::json!({
        "preload_models": [{
            "model_id": "minilm-ane-timeout",
            "engine": "ane",
            "task": "embed",
            "model_path": model_path,
            "tokenizer_path": tokenizer_path,
            "format": "mock",
            "pooling": "mean",
            "normalize": true,
            "worker_bin": env!("CARGO_BIN_EXE_synapse-worker-timeout-mock"),
            "worker_runtime_dir": worker_runtime_dir
        }]
    })
    .to_string()
}

#[tokio::test]
async fn remote_gateway_declares_calibrates_checkpoints_trips_and_recovers() {
    let provider = RemoteMockProvider::start().await;
    let config = serde_json::json!({
        "inline": {"max_items": 1, "max_tokens": 8192},
        "remote_providers": [{
            "name": "mock",
            "base_url": provider.base_url(),
            "adapter": {"kind": "openai_compatible"},
            "auth": {"kind": "none"},
            "models": [{
                "synapse_model_id": "remote-embed",
                "task": "embed",
                "model": "mock-embed",
                "identity_revision": "r1",
                "dims": 3,
                "input_profile_id": "whitespace-v1",
                "max_input_tokens": 128,
                "sentinel_texts": ["alpha", "beta", "gamma"]
            }],
            "breaker": {"failure_threshold": 3, "cooldown_ms": 100},
            "cold_estimate_ms": {"embed": 10, "rerank": 10, "generate": 10},
            "connect_timeout_ms": 1000,
            "read_timeout_ms": 1000,
            "target_subbatch_ms": 100
        }]
    })
    .to_string();
    let (_daemon, _module, mut consumer, route) = open_route_with_config(&config).await;

    let listed = route_request(
        &mut consumer,
        route,
        20,
        serde_json::json!({"method":"models.list","params":{}}),
    )
    .await;
    let remote = listed["result"]["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["model_id"] == "remote-embed")
        .expect("declared remote model appears in catalog");
    assert_eq!(remote["assurance"], "declared");
    assert_eq!(remote["identity_revision"], "r1");
    assert!(
        remote.get("recommended_batch").is_none(),
        "remote lanes without a measured batch policy omit the advisory"
    );

    let before_probe = route_request(
        &mut consumer,
        route,
        21,
        serde_json::json!({
            "method":"embed.query",
            "params":{"model":"remote-embed","text":"hello","accept_declared":true}
        }),
    )
    .await;
    assert_eq!(before_probe["result"]["error"]["code"], "not_certified");

    for (corr, method, params) in [
        (
            22,
            "rerank.score",
            serde_json::json!({
                "model":"remote-embed","query":"q","candidates":["a"],"accept_declared":true
            }),
        ),
        (
            23,
            "microllm.oneshot",
            serde_json::json!({
                "model":"remote-embed","prompt":"q","max_tokens":1,"accept_declared":true
            }),
        ),
    ] {
        let rejected = route_request(
            &mut consumer,
            route,
            corr,
            serde_json::json!({"method":method,"params":params}),
        )
        .await;
        assert_eq!(
            rejected["result"]["error"]["code"],
            "op_not_supported_for_remote"
        );
    }

    run_probe_job(
        &mut consumer,
        route,
        30,
        serde_json::json!({"models":["remote-embed"],"request_key":"remote-probe"}),
    )
    .await;

    let embedded = route_request(
        &mut consumer,
        route,
        40,
        serde_json::json!({
            "method":"embed.query",
            "params":{"model":"remote-embed","id":"q","text":"hello remote","accept_declared":true}
        }),
    )
    .await;
    assert_eq!(embedded["result"]["assurance"], "declared");
    assert_eq!(embedded["result"]["identity_revision"], "r1");
    assert_eq!(
        embedded["result"]["provenance"]["remote"]["assurance"],
        "declared"
    );
    assert_eq!(embedded["result"]["payload"]["vectors"][0]["id"], "q");
    assert!(embedded["result"]["payload"]["vectors"][0]["content_sha256"].is_string());

    let accepted = route_request(
        &mut consumer,
        route,
        41,
        serde_json::json!({
            "method":"embed.batch",
            "params":{
                "model":"remote-embed",
                "request_key":"remote-batch",
                "accept_declared":true,
                "items":[{"id":"a","text":"alpha"},{"id":"b","text":"beta"}]
            }
        }),
    )
    .await;
    let job_id = accepted["result"]["job_id"].as_str().unwrap();
    let completed = poll_embed_result(&mut consumer, route, 42, job_id).await;
    assert_eq!(completed["result"]["state"], "done");
    assert_eq!(completed["result"]["page_count"], 2);

    provider.fail_next(3);
    let unavailable = route_request(
        &mut consumer,
        route,
        50,
        serde_json::json!({
            "method":"embed.query",
            "params":{"model":"remote-embed","text":"storm","accept_declared":true}
        }),
    )
    .await;
    assert_eq!(
        unavailable["result"]["error"]["code"],
        "provider_unavailable"
    );
    assert!(unavailable["result"]["error"]["retry_after_ms"].is_number());

    sleep(Duration::from_millis(150)).await;
    let recovered = route_request(
        &mut consumer,
        route,
        51,
        serde_json::json!({
            "method":"embed.query",
            "params":{"model":"remote-embed","text":"recovered","accept_declared":true}
        }),
    )
    .await;
    assert_eq!(recovered["result"]["assurance"], "declared");
}

#[tokio::test]
async fn job_resume_respawns_remote_job_and_pages_are_readable() {
    let provider = RemoteMockProvider::start().await;
    let config = serde_json::json!({
        "inline": {"max_items": 1, "max_tokens": 8192},
        "jobs": {
            "execution_ttl_ms": 60_000,
            "result_retention_ttl_ms": 60_000,
            "resume_deadline_ms": 60_000,
            "result_page_bytes": 4_096
        },
        "remote_providers": [{
            "name": "mock",
            "base_url": provider.base_url(),
            "adapter": {"kind": "openai_compatible"},
            "auth": {"kind": "none"},
            "models": [{
                "synapse_model_id": "remote-embed",
                "task": "embed",
                "model": "mock-embed",
                "identity_revision": "r1",
                "dims": 3,
                "input_profile_id": "whitespace-v1",
                "sentinel_texts": ["alpha", "beta", "gamma"]
            }],
            "connect_timeout_ms": 1_000,
            "read_timeout_ms": 1_000
        }]
    })
    .to_string();
    let (daemon, mut module, mut consumer, route) = open_route_with_config(&config).await;
    run_probe_job(
        &mut consumer,
        route,
        10,
        serde_json::json!({"models":["remote-embed"],"request_key":"resume-probe"}),
    )
    .await;
    let accepted = route_request(
        &mut consumer,
        route,
        11,
        serde_json::json!({
            "method": "embed.batch",
            "params": {
                "model": "remote-embed",
                "request_key": "resume-wire",
                "accept_declared": true,
                "items": [
                    {"id": "first", "text": "first resumed item"},
                    {"id": "second", "text": "second resumed item"}
                ]
            }
        }),
    )
    .await;
    let job_id = accepted["result"]["job_id"]
        .as_str()
        .expect("job-shaped remote batch returns a job id")
        .to_string();
    let requests_before_restart = provider.request_count();

    // Stop the original execution owner before representing a vault pause in
    // the durable row. The e2e daemon has no vault fixture, so this preserves
    // the same paused boundary without allowing the first task to race resume.
    drop(consumer);
    module
        .child
        .kill()
        .await
        .expect("original module should stop");
    let _ = module.child.wait().await;
    sleep(Duration::from_millis(100)).await;

    let module = spawn_synapse_module_with_config(&daemon.connection_file_path, &config);
    let (daemon, _module, mut consumer, route) =
        open_route_for_started_module(daemon, module).await;
    let store_path = expected_store_path(&daemon.data_home);
    let connection = Connection::open(store_path).expect("open durable test store");
    let generation: i64 = connection
        .query_row(
            "SELECT module_generation FROM module_meta WHERE id = 0",
            [],
            |row| row.get(0),
        )
        .expect("read current module generation");
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("read test clock")
        .as_millis() as i64;
    let changed = connection
        .execute(
            "UPDATE jobs SET state = 'paused_needs_reauth', module_generation = ?1,
                 updated_ms = ?2, execution_expires_ms = NULL, active_attempt_id = NULL,
                 logical_handle = NULL, paused_at_ms = ?2, resume_deadline_ms = ?3,
                 error_json = NULL, terminal_at_ms = NULL WHERE job_id = ?4",
            params![generation, now, now + 60_000, job_id],
        )
        .expect("represent a durable vault pause");
    assert_eq!(changed, 1, "the submitted job must be paused before resume");

    let resumed = route_request(
        &mut consumer,
        route,
        12,
        serde_json::json!({"method":"job.resume","params":{"job_id":job_id}}),
    )
    .await;
    assert!(
        matches!(
            resumed["result"]["state"].as_str(),
            Some("queued" | "running" | "done")
        ),
        "job.resume should return a live job status: {resumed}"
    );
    let done = poll_embed_result(&mut consumer, route, 13, &job_id).await;
    assert_eq!(done["result"]["state"], "done");
    assert_eq!(done["result"]["page_count"], 2);
    assert!(
        provider.request_count() > requests_before_restart,
        "resume must issue a new provider request rather than only flipping state"
    );
    let second_page = route_request(
        &mut consumer,
        route,
        14,
        serde_json::json!({"method":"embed.result","params":{"job_id":job_id,"page":1}}),
    )
    .await;
    assert_eq!(
        second_page["result"]["payload"]["vectors"][0]["id"], "second",
        "second page should expose the resumed job's second item: {second_page}"
    );
}

#[tokio::test]
async fn embed_query_returns_typed_probe_required_error() {
    let (_daemon, _module, mut consumer, route) = open_route().await;
    let frame = raw_route_frame(
        &mut consumer,
        route,
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
    let (_daemon, _module, mut consumer, route) = open_route_with_preloads(Some(&preloads)).await;
    certify_preloaded_models(&mut consumer, route, 40).await;

    let first = route_request(
        &mut consumer,
        route,
        4,
        serde_json::json!({
            "method": "embed.query",
            "params": { "id": "q1", "text": "hello world" }
        }),
    )
    .await;
    let second = route_request(
        &mut consumer,
        route,
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
async fn probe_refuses_lane_without_matching_reference_fixture() {
    let Some(preloads) = minilm_preload_config() else {
        eprintln!("skipping missing-reference probe e2e: local HF ONNX snapshot is missing");
        return;
    };
    let mut models: Value = serde_json::from_str(&preloads).expect("preload config is json");
    models[0]["model_id"] = Value::String("unreferenced-embed-model".to_string());
    let config = serde_json::json!({ "preload_models": models }).to_string();
    let _lock = acquire_minilm_e2e_lock();
    let (_daemon, _module, mut consumer, route) = open_route_with_config(&config).await;

    let body = poll_probe_job(
        &mut consumer,
        route,
        30,
        serde_json::json!({ "models": ["unreferenced-embed-model"] }),
    )
    .await;
    let lane = body["result"]["lanes"]
        .as_array()
        .and_then(|lanes| lanes.first())
        .expect("probe returns the selected lane");
    assert_eq!(lane["status"], "uncertified");
    assert_eq!(lane["blocking_reason"], "reference_fixture_missing");
    assert_eq!(
        lane["evidence"]["blocking_reason"],
        "reference_fixture_missing"
    );
    assert!(lane["evidence"].get("metrics").is_none());
    assert!(lane["evidence"].get("mean_cosine").is_none());

    let report = route_request(
        &mut consumer,
        route,
        31,
        serde_json::json!({ "method": "probe.report", "params": {} }),
    )
    .await;
    assert_eq!(
        report["result"]["lanes"][0]["blocking_reason"],
        "reference_fixture_missing"
    );
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn embed_query_loaded_owned_metal_carries_distinct_provenance_and_content_hash() {
    let Some(snapshot) = minilm_safetensors_snapshot() else {
        eprintln!("skipping owned-metal MiniLM e2e: local safetensors snapshot is missing");
        return;
    };
    let Some(ort_preloads) = minilm_preload_config() else {
        eprintln!("skipping owned-metal cross-engine e2e: local ONNX snapshot is missing");
        return;
    };
    let _lock = acquire_minilm_e2e_lock();
    let daemon = start_daemon().await;
    let cache = unique_temp_dir("synapse-owned-model-cache");
    let module = spawn_synapse_module_with_env(
        &daemon.connection_file_path,
        Some(&ort_preloads),
        None,
        &[("CORTEXKIT_MODEL_CACHE", cache.to_string_lossy().as_ref())],
    );
    let (_daemon, _module, mut consumer, route) =
        open_route_for_started_module(daemon, module).await;
    let accepted = route_request(
        &mut consumer,
        route,
        46,
        serde_json::json!({
            "method": "model.load",
            "params": {
                "source": "file",
                "path": snapshot,
                "files": {
                    "model": {
                        "url": "model.safetensors",
                        "sha256": test_sha256(snapshot.join("model.safetensors"))
                    },
                    "tokenizer": {
                        "url": "tokenizer.json",
                        "sha256": test_sha256(snapshot.join("tokenizer.json"))
                    },
                    "config": {
                        "url": "config.json",
                        "sha256": test_sha256(snapshot.join("config.json"))
                    }
                },
                "engine": "owned-metal",
                "family": "minilm",
                "dtype": "f16",
                "execution": "explicit",
                "model_id": "minilm-owned",
                "task": "embed",
                "max_tokens": 512,
                "pin": true,
                "request_key": "owned-metal-model-load-e2e"
            }
        }),
    )
    .await;
    let job_id = accepted["result"]["job_id"].as_str().unwrap();
    let ready = poll_model_load_job(&mut consumer, route, 47, job_id).await;
    assert_eq!(
        ready["result"]["state"], "ready",
        "owned load failed: {ready:?}"
    );
    run_probe_job(
        &mut consumer,
        route,
        60,
        serde_json::json!({ "models": ["minilm", "minilm-owned"] }),
    )
    .await;

    let body = route_request(
        &mut consumer,
        route,
        2_000,
        serde_json::json!({
            "method": "embed.query",
            "params": {
                "model": "minilm-owned",
                "id": "owned-q1",
                "text": "hello world"
            }
        }),
    )
    .await;
    let result = &body["result"];
    assert_eq!(result["dims"], 384);
    assert_eq!(result["vectors"][0]["id"], "owned-q1");
    assert_eq!(
        result["vectors"][0]["content_sha256"],
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    );
    assert_eq!(result["provenance"]["engine"]["engine"], "owned-metal");
    assert_eq!(
        result["provenance"]["engine"]["build_flags"]["family"],
        "minilm"
    );
    assert_eq!(
        result["provenance"]["engine"]["build_flags"]["dtype"],
        "f16"
    );
    assert_eq!(
        result["provenance"]["engine"]["build_flags"]["graph_revision"],
        "4"
    );
    assert_eq!(
        result["provenance"]["engine"]["build_flags"]["bucket_policy"],
        "v1"
    );
    assert_eq!(result["fingerprint"].as_str().unwrap().len(), 64);
    let spike_golden: Value =
        serde_json::from_str(include_str!("fixtures/minilm_owned_spike_f16.jsonl"))
            .expect("decode frozen spike handoff vector");
    assert_vector_cosine_at_least(
        &result["vectors"][0]["vector"],
        &spike_golden["vec"],
        0.999_999,
    );

    let ort = route_request(
        &mut consumer,
        route,
        2_001,
        serde_json::json!({
            "method": "embed.query",
            "params": { "model": "minilm", "text": "hello world" }
        }),
    )
    .await;
    let ort_fingerprint = ort["result"]["fingerprint"].as_str().unwrap();
    assert_ne!(ort_fingerprint, result["fingerprint"].as_str().unwrap());
    let rejected = route_request(
        &mut consumer,
        route,
        2_002,
        serde_json::json!({
            "method": "embed.query",
            "params": {
                "model": "minilm-owned",
                "text": "hello world",
                "target_fingerprint": ort_fingerprint
            }
        }),
    )
    .await;
    assert_eq!(rejected["result"]["error"]["code"], "substitution_rejected");
    let _ = std::fs::remove_dir_all(cache);
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn probe_owned_gte_modernbert_certifies_against_family_reference() {
    let Some(snapshot) = gte_safetensors_snapshot() else {
        eprintln!("skipping owned-metal GTE probe e2e: local HF snapshot is missing");
        return;
    };
    for required in ["model.safetensors", "tokenizer.json", "config.json"] {
        if !snapshot.join(required).exists() {
            eprintln!(
                "skipping owned-metal GTE probe e2e: {} is missing from {}",
                required,
                snapshot.display()
            );
            return;
        }
    }
    let _lock = acquire_minilm_e2e_lock();
    let (_daemon, _module, mut consumer, route) = open_route().await;
    let accepted = route_request(
        &mut consumer,
        route,
        80,
        serde_json::json!({
            "method": "model.load",
            "params": {
                "source": "file",
                "path": snapshot,
                "files": {
                    "model": {
                        "url": "model.safetensors",
                        "sha256": test_sha256(snapshot.join("model.safetensors"))
                    },
                    "tokenizer": {
                        "url": "tokenizer.json",
                        "sha256": test_sha256(snapshot.join("tokenizer.json"))
                    },
                    "config": {
                        "url": "config.json",
                        "sha256": test_sha256(snapshot.join("config.json"))
                    }
                },
                "engine": "owned-metal",
                "family": "gte-modernbert",
                "dtype": "f16",
                "execution": "explicit",
                "model_id": "gte-modernbert-base-f16",
                "task": "embed",
                "pooling": "mean",
                "normalize": true,
                "max_tokens": 512,
                "pin": true,
                "request_key": "owned-gte-probe-e2e"
            }
        }),
    )
    .await;
    let job_id = accepted["result"]["job_id"]
        .as_str()
        .expect("owned GTE model.load returns job id");
    let ready = poll_model_load_job(&mut consumer, route, 81, job_id).await;
    assert_eq!(
        ready["result"]["state"], "ready",
        "GTE load failed: {ready:?}"
    );

    let body = run_probe_job(
        &mut consumer,
        route,
        90,
        serde_json::json!({ "models": ["gte-modernbert-base-f16"] }),
    )
    .await;
    let lane = body["result"]["lanes"]
        .as_array()
        .and_then(|lanes| lanes.first())
        .expect("GTE probe returns a lane");
    assert_eq!(lane["status"], "certified", "GTE probe failed: {lane:?}");
    eprintln!("owned GTE probe evidence: {}", lane["evidence"]);
    assert_eq!(lane["evidence"]["items"], 64);
    assert!(
        lane["evidence"]["mean_cosine"]
            .as_f64()
            .expect("mean cosine is numeric")
            >= 0.999
    );
    assert!(
        lane["evidence"]["rank_overlap"]
            .as_f64()
            .expect("rank overlap is numeric")
            >= 0.999
    );

    let listed = route_request(
        &mut consumer,
        route,
        91,
        serde_json::json!({ "method": "models.list", "params": {} }),
    )
    .await;
    let owned = listed["result"]["models"]
        .as_array()
        .and_then(|models| {
            models
                .iter()
                .find(|model| model["model_id"] == "gte-modernbert-base-f16")
        })
        .expect("certified owned-metal model appears in catalog");
    assert_eq!(
        owned["recommended_batch"],
        serde_json::json!({ "rows": 8, "token_budget": 3_072 })
    );

    let rerank_refused = route_request(
        &mut consumer,
        route,
        92,
        serde_json::json!({
            "method": "rerank.score",
            "params": {
                "model": "gte-modernbert-base-f16",
                "query": "query",
                "candidates": ["candidate"]
            }
        }),
    )
    .await;
    assert_eq!(
        rerank_refused["result"]["error"]["code"],
        "artifact_invalid"
    );
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn owned_gte_rerank_loads_certifies_and_serves_deterministically() {
    let Some(snapshot) = gte_reranker_safetensors_snapshot() else {
        eprintln!("skipping owned-metal rerank e2e: local GTE reranker snapshot is missing");
        return;
    };
    for required in ["model.safetensors", "tokenizer.json", "config.json"] {
        assert!(
            snapshot.join(required).is_file(),
            "GTE reranker snapshot is missing {required}"
        );
    }
    let _lock = acquire_minilm_e2e_lock();
    let (_daemon, _module, mut consumer, route) = open_route().await;
    let load_params = serde_json::json!({
        "source": "file",
        "path": snapshot,
        "files": {
            "model": {
                "url": "model.safetensors",
                "sha256": test_sha256(snapshot.join("model.safetensors"))
            },
            "tokenizer": {
                "url": "tokenizer.json",
                "sha256": test_sha256(snapshot.join("tokenizer.json"))
            },
            "config": {
                "url": "config.json",
                "sha256": test_sha256(snapshot.join("config.json"))
            }
        },
        "engine": "owned-metal",
        "family": "gte-modernbert",
        "dtype": "f32",
        "execution": "explicit",
        "model_id": "gte-reranker-modernbert-base-f32",
        "task": "rerank",
        "max_tokens": 8192,
        "attention_units": 67108864,
        "pin": true,
        "request_key": "owned-metal-rerank-e2e"
    });
    let accepted = route_request(
        &mut consumer,
        route,
        72_000,
        serde_json::json!({ "method": "model.load", "params": load_params }),
    )
    .await;
    let job_id = accepted["result"]["job_id"]
        .as_str()
        .expect("owned rerank model.load returns job id")
        .to_string();
    let ready = poll_model_load_job(&mut consumer, route, 72_001, &job_id).await;
    assert_eq!(ready["result"]["state"], "ready", "rerank load: {ready:?}");

    let probe = run_probe_job(
        &mut consumer,
        route,
        72_010,
        serde_json::json!({ "models": ["gte-reranker-modernbert-base-f32"] }),
    )
    .await;
    let lane = probe["result"]["lanes"]
        .as_array()
        .and_then(|lanes| {
            lanes.iter().find(|lane| {
                lane["model_id"] == "gte-reranker-modernbert-base-f32" && lane["task"] == "rerank"
            })
        })
        .expect("rerank probe returns the owned lane");
    assert_eq!(lane["status"], "certified", "rerank probe: {lane:?}");
    assert!(lane["evidence"]["pearson"].as_f64().unwrap() >= 0.999);

    let listed = route_request(
        &mut consumer,
        route,
        72_011,
        serde_json::json!({ "method": "models.list", "params": {} }),
    )
    .await;
    let model = listed["result"]["models"]
        .as_array()
        .and_then(|models| {
            models
                .iter()
                .find(|model| model["model_id"] == "gte-reranker-modernbert-base-f32")
        })
        .expect("owned reranker appears in models.list");
    let fingerprint = model["fingerprints"]
        .as_array()
        .and_then(|fingerprints| fingerprints.first())
        .and_then(Value::as_str)
        .expect("reranker fingerprint")
        .to_string();
    assert_eq!(fingerprint.len(), 64);

    let params = serde_json::json!({
        "method": "rerank.score",
        "params": {
            "model": "gte-reranker-modernbert-base-f32",
            "query": "Which document is about a cat sitting on a mat?",
            "candidates": [
                "A small cat sits on a woven mat near the window.",
                "The spacecraft entered orbit after a six month flight.",
                "A dog chased a tennis ball across the park."
            ]
        }
    });
    let first = route_request(&mut consumer, route, 72_020, params.clone()).await;
    let second = route_request(&mut consumer, route, 72_021, params).await;
    assert_eq!(
        serde_json::to_vec(&first["result"]).unwrap(),
        serde_json::to_vec(&second["result"]).unwrap(),
        "identical rerank requests must produce byte-identical results"
    );
    let result = &first["result"];
    assert_eq!(result["dims"], 1);
    assert_eq!(result["fingerprint"], fingerprint);
    assert_eq!(result["provenance"]["engine"]["engine"], "owned-metal");
    let scores = result["scores"].as_array().expect("rerank envelope scores");
    assert_eq!(scores.len(), 3);
    assert!(scores
        .iter()
        .all(|score| score.as_f64().is_some_and(f64::is_finite)));
    assert!(scores.windows(2).any(|window| window[0] != window[1]));

    let f16_accepted = route_request(
        &mut consumer,
        route,
        72_030,
        serde_json::json!({
            "method": "model.load",
            "params": {
                "source": "file",
                "path": snapshot,
                "files": {
                    "model": {"url": "model.safetensors", "sha256": test_sha256(snapshot.join("model.safetensors"))},
                    "tokenizer": {"url": "tokenizer.json", "sha256": test_sha256(snapshot.join("tokenizer.json"))},
                    "config": {"url": "config.json", "sha256": test_sha256(snapshot.join("config.json"))}
                },
                "engine": "owned-metal",
                "family": "gte-modernbert",
                "dtype": "f16",
                "model_id": "gte-reranker-modernbert-base-f16",
                "task": "rerank",
                "max_tokens": 512,
                "request_key": "owned-metal-rerank-f16-refusal"
            }
        }),
    )
    .await;
    let f16_job = f16_accepted["result"]["job_id"].as_str().unwrap();
    let f16_failed = poll_model_load_job(&mut consumer, route, 72_031, f16_job).await;
    assert_eq!(f16_failed["result"]["state"], "failed");
    assert_eq!(f16_failed["result"]["error"]["code"], "artifact_invalid");
    assert!(f16_failed["result"]["error"]["message"]
        .as_str()
        .unwrap()
        .contains("fp32-only"));

    let Some(minilm_snapshot) = minilm_safetensors_snapshot() else {
        return;
    };
    let minilm_accepted = route_request(
        &mut consumer,
        route,
        72_040,
        serde_json::json!({
            "method": "model.load",
            "params": {
                "source": "file",
                "path": minilm_snapshot,
                "files": {
                    "model": {"url": "model.safetensors", "sha256": test_sha256(minilm_snapshot.join("model.safetensors"))},
                    "tokenizer": {"url": "tokenizer.json", "sha256": test_sha256(minilm_snapshot.join("tokenizer.json"))},
                    "config": {"url": "config.json", "sha256": test_sha256(minilm_snapshot.join("config.json"))}
                },
                "engine": "owned-metal",
                "family": "minilm",
                "dtype": "f32",
                "model_id": "minilm-reranker-no-head",
                "task": "rerank",
                "max_tokens": 512,
                "request_key": "owned-metal-rerank-no-head-refusal"
            }
        }),
    )
    .await;
    let minilm_job = minilm_accepted["result"]["job_id"].as_str().unwrap();
    let minilm_failed = poll_model_load_job(&mut consumer, route, 72_041, minilm_job).await;
    assert_eq!(minilm_failed["result"]["state"], "failed");
    assert_eq!(minilm_failed["result"]["error"]["code"], "artifact_invalid");
    assert!(minilm_failed["result"]["error"]["message"]
        .as_str()
        .unwrap()
        .contains("sequence-classification head"));
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn owned_gte_inline_embed_batch_throughput_sweep() {
    let snapshot = gte_safetensors_snapshot().expect("local GTE ModernBERT snapshot is required");
    for required in ["model.safetensors", "tokenizer.json", "config.json"] {
        assert!(
            snapshot.join(required).exists(),
            "GTE snapshot is missing {}",
            required
        );
    }
    let config = serde_json::json!({
        "inline": { "max_items": 256, "max_tokens": 200_000 },
        "jobs": { "bulk_quantum_tokens": 3_072 }
    })
    .to_string();
    let (_daemon, _module, mut consumer, route) = open_route_with_config(&config).await;
    let accepted = route_request(
        &mut consumer,
        route,
        70_000,
        serde_json::json!({
            "method": "model.load",
            "params": {
                "source": "file",
                "path": snapshot,
                "files": {
                    "model": {
                        "url": "model.safetensors",
                        "sha256": test_sha256(snapshot.join("model.safetensors"))
                    },
                    "tokenizer": {
                        "url": "tokenizer.json",
                        "sha256": test_sha256(snapshot.join("tokenizer.json"))
                    },
                    "config": {
                        "url": "config.json",
                        "sha256": test_sha256(snapshot.join("config.json"))
                    }
                },
                "engine": "owned-metal",
                "family": "gte-modernbert",
                "dtype": "f16",
                "execution": "explicit",
                "model_id": "gte-modernbert-base-f16",
                "task": "embed",
                "pooling": "mean",
                "normalize": true,
                "max_tokens": 512,
                "pin": true,
                "request_key": "owned-gte-throughput-load-v2"
            }
        }),
    )
    .await;
    let load_job = accepted["result"]["job_id"]
        .as_str()
        .expect("throughput model.load returns job id");
    let ready = poll_model_load_job(&mut consumer, route, 70_001, load_job).await;
    assert_eq!(
        ready["result"]["state"], "ready",
        "GTE load failed: {ready:?}"
    );
    run_probe_job(
        &mut consumer,
        route,
        70_002,
        serde_json::json!({ "models": ["gte-modernbert-base-f16"] }),
    )
    .await;

    let fixture_text = |index: usize| {
        (0..45)
            .map(|word| format!("retrieval fixture item {index} token {word}"))
            .collect::<Vec<_>>()
            .join(" ")
    };
    let make_items = |count: usize| {
        (0..count)
            .map(|index| {
                serde_json::json!({
                    "id": format!("item-{index}"),
                    "text": fixture_text(index)
                })
            })
            .collect::<Vec<_>>()
    };
    let warmup = route_request(
        &mut consumer,
        route,
        70_010,
        serde_json::json!({
            "method": "embed.batch",
            "params": {
                "model": "gte-modernbert-base-f16",
                "items": make_items(1),
                "accept_declared": true
            }
        }),
    )
    .await;
    assert_eq!(
        warmup["result"]["vectors"].as_array().map(Vec::len),
        Some(1),
        "warmup response: {warmup:?}"
    );

    let mut rows = Vec::new();
    for (offset, batch_size) in [8_usize, 16, 32, 64, 128, 256].into_iter().enumerate() {
        let started = Instant::now();
        let response = route_request(
            &mut consumer,
            route,
            70_100 + offset as u64,
            serde_json::json!({
                "method": "embed.batch",
                "params": {
                    "model": "gte-modernbert-base-f16",
                    "items": make_items(batch_size),
                    "accept_declared": true
                }
            }),
        )
        .await;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let vectors = response["result"]["vectors"].as_array().unwrap();
        assert_eq!(vectors.len(), batch_size);
        let tokens = response["result"]["real_token_counts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap())
            .sum::<u64>();
        let ms_per_item = elapsed_ms / batch_size as f64;
        let throughput = tokens as f64 / (elapsed_ms / 1_000.0);
        eprintln!(
            "owned GTE throughput row batch={batch_size} tokens={tokens} elapsed_ms={elapsed_ms:.3} ms_per_item={ms_per_item:.3} tok_per_s={throughput:.1}"
        );
        if let Some(previous) = rows.last() {
            assert!(
                ms_per_item <= previous * 1.4,
                "batch={batch_size} has a local throughput cliff: {ms_per_item:.3}ms/item after {previous:.3}ms/item"
            );
        }
        if batch_size >= 64 {
            // Regression floor, not a performance target. This sweep exists to
            // catch the quanta-slicing class where inline batches execute as
            // serial per-quantum engine calls (observed collapse: ~1.5k tok/s,
            // a 7-10x drop). The floor must hold on the slowest CI machine:
            // the M1 Max runner saturates this wire path at ~9.8k tok/s while
            // an M5 Max does 14-15k, so a 10k gate calibrated on the M5 fails
            // healthy M1 runs. 6k passes every healthy machine with margin and
            // still fails the collapse class by 4x.
            assert!(
                throughput >= 6_000.0,
                "batch={batch_size} throughput {throughput:.1} tok/s is below the 6k regression floor"
            );
        }
        rows.push(ms_per_item);
    }

    let mut query_samples = Vec::new();
    for index in 0..5 {
        let started = Instant::now();
        let response = route_request(
            &mut consumer,
            route,
            70_200 + index,
            serde_json::json!({
                "method": "embed.query",
                "params": {
                    "model": "gte-modernbert-base-f16",
                    "id": "query",
                    "text": fixture_text(0),
                    "accept_declared": true
                }
            }),
        )
        .await;
        assert_eq!(response["result"]["vectors"].as_array().unwrap().len(), 1);
        query_samples.push(started.elapsed().as_secs_f64() * 1_000.0);
    }
    query_samples.sort_by(f64::total_cmp);
    eprintln!(
        "owned GTE embed.query p50_ms={:.3} samples={query_samples:?}",
        query_samples[query_samples.len() / 2]
    );
}

#[tokio::test]
async fn embed_batch_preloaded_minilm_preserves_order_and_envelope() {
    let Some(preloads) = minilm_preload_config() else {
        eprintln!("skipping MiniLM embed.batch e2e: local HF ONNX snapshot is missing");
        return;
    };
    let _lock = acquire_minilm_e2e_lock();
    let (_daemon, _module, mut consumer, route) = open_route_with_preloads(Some(&preloads)).await;
    certify_preloaded_models(&mut consumer, route, 60).await;
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
        route,
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
async fn over_budget_embed_batch_returns_job_and_pages_results() {
    let Some(preloads) = minilm_preload_config() else {
        eprintln!("skipping MiniLM job-tier e2e: local HF ONNX snapshot is missing");
        return;
    };
    let preload_models: Value = serde_json::from_str(&preloads).expect("preload config is json");
    let config = serde_json::json!({
        "preload_models": preload_models,
        "inline": { "max_items": 2 },
        "jobs": {
            "execution_ttl_ms": 60_000,
            "result_retention_ttl_ms": 60_000,
            "resume_deadline_ms": 60_000,
            "result_page_bytes": 4_096,
            "bulk_quantum_tokens": 16
        }
    })
    .to_string();
    let _lock = acquire_minilm_e2e_lock();
    let (_daemon, _module, mut consumer, route) = open_route_with_config(&config).await;
    certify_preloaded_models(&mut consumer, route, 80).await;
    let texts = [
        "job tier first text",
        "job tier second text",
        "job tier third text",
    ];

    let mut inline_vectors = Vec::new();
    for (index, text) in texts.iter().enumerate() {
        let body = route_request(
            &mut consumer,
            route,
            100 + index as u64,
            serde_json::json!({
                "method": "embed.query",
                "params": { "id": format!("item-{index}"), "text": text }
            }),
        )
        .await;
        inline_vectors.push(body["result"]["vectors"][0]["vector"].clone());
    }

    let items = texts
        .iter()
        .enumerate()
        .map(|(index, text)| {
            serde_json::json!({
                "id": format!("item-{index}"),
                "text": text,
            })
        })
        .collect::<Vec<_>>();
    let accepted = route_request(
        &mut consumer,
        route,
        200,
        serde_json::json!({
            "method": "embed.batch",
            "params": { "request_key": "job-tier-e2e", "items": items }
        }),
    )
    .await;
    let job_id = accepted["result"]["job_id"]
        .as_str()
        .expect("job response includes job_id")
        .to_string();
    assert!(matches!(
        accepted["result"]["state"].as_str(),
        Some("queued" | "running" | "done")
    ));
    assert!(accepted["result"]["pages_available"].is_number());

    let duplicate = route_request(
        &mut consumer,
        route,
        201,
        serde_json::json!({
            "method": "embed.batch",
            "params": {
                "request_key": "job-tier-e2e",
                "items": texts.iter().enumerate().map(|(index, text)| serde_json::json!({
                    "id": format!("item-{index}"),
                    "text": text,
                })).collect::<Vec<_>>()
            }
        }),
    )
    .await;
    assert_eq!(duplicate["result"]["job_id"], job_id);

    let conflict = route_request(
        &mut consumer,
        route,
        202,
        serde_json::json!({
            "method": "embed.batch",
            "params": {
                "request_key": "job-tier-e2e",
                "items": [
                    { "id": "item-0", "text": "different request content" },
                    { "id": "item-1", "text": texts[1] },
                    { "id": "item-2", "text": texts[2] }
                ]
            }
        }),
    )
    .await;
    assert_eq!(conflict["result"]["error"]["code"], "idempotency_conflict");
    assert_eq!(conflict["result"]["error"]["class"], "permanent");
    assert_eq!(
        conflict["result"]["error"]["safe_to_retry_same_request"],
        Value::Bool(false)
    );

    let done = poll_embed_result(&mut consumer, route, 300, &job_id).await;
    assert_eq!(done["result"]["state"], "done");
    assert_eq!(
        done["result"]["pages_available"],
        done["result"]["page_count"]
    );
    let page_count = done["result"]["page_count"].as_u64().unwrap();
    assert!(
        page_count > 1,
        "regression batch must straddle the configured result page boundary"
    );
    let mut vectors = done["result"]["vectors"].as_array().unwrap().clone();
    for page in 1..page_count {
        let body = route_request(
            &mut consumer,
            route,
            300 + page,
            serde_json::json!({
                "method": "embed.result",
                "params": { "job_id": &job_id, "page": page }
            }),
        )
        .await;
        vectors.extend(
            body["result"]["vectors"]
                .as_array()
                .unwrap()
                .iter()
                .cloned(),
        );
    }
    assert_eq!(vectors.len(), texts.len());
    for (index, vector) in vectors.iter().enumerate() {
        assert_eq!(vector["id"], format!("item-{index}"));
        assert_vectors_close(&vector["vector"], &inline_vectors[index]);
    }
}

#[tokio::test]
async fn admission_status_reports_execution_waiters_during_concurrent_batches() {
    let Some(preloads) = minilm_preload_config() else {
        eprintln!("skipping execution admission e2e: local HF ONNX snapshot is missing");
        return;
    };
    let preload_models: Value = serde_json::from_str(&preloads).expect("preload config is json");
    let config = serde_json::json!({
        "preload_models": preload_models,
        "inline": {
            "max_items": 64,
            "max_tokens": 100_000,
            "max_concurrent_workers": 1
        }
    })
    .to_string();
    let _lock = acquire_minilm_e2e_lock();
    let (daemon, _module, mut first_consumer, first_route) = open_route_with_config(&config).await;
    certify_preloaded_models(&mut first_consumer, first_route, 400).await;

    let mut second_consumer = connect_consumer(&daemon.connection_file_path).await;
    wait_for_catalog(&mut second_consumer, MODULE_ID, SETUP_TIMEOUT).await;
    let second_project = unique_temp_dir("synapse-admission-second");
    std::fs::create_dir_all(&second_project).unwrap();
    let second_route = route_open(&mut second_consumer, &second_project, 2).await;

    let mut status_consumer = connect_consumer(&daemon.connection_file_path).await;
    wait_for_catalog(&mut status_consumer, MODULE_ID, SETUP_TIMEOUT).await;
    let status_project = unique_temp_dir("synapse-admission-status");
    std::fs::create_dir_all(&status_project).unwrap();
    let status_route = route_open(&mut status_consumer, &status_project, 3).await;

    let text = "admission semaphore hold ".repeat(512);
    let make_items = || {
        (0..64)
            .map(|index| {
                serde_json::json!({
                    "id": format!("admission-{index}"),
                    "text": text.clone()
                })
            })
            .collect::<Vec<_>>()
    };
    let first_items = make_items();
    let second_items = make_items();
    let first_task = tokio::spawn(async move {
        route_request(
            &mut first_consumer,
            first_route,
            401,
            serde_json::json!({
                "method": "embed.batch",
                "params": { "items": first_items }
            }),
        )
        .await
    });
    sleep(Duration::from_millis(5)).await;
    let second_task = tokio::spawn(async move {
        route_request(
            &mut second_consumer,
            second_route,
            402,
            serde_json::json!({
                "method": "embed.batch",
                "params": { "items": second_items }
            }),
        )
        .await
    });

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut observed_waiter = false;
    while Instant::now() < deadline {
        let status = route_request(
            &mut status_consumer,
            status_route,
            403,
            serde_json::json!({ "method": "admission.status", "params": {} }),
        )
        .await;
        let result = &status["result"];
        if result["execution_waiters"].as_u64().unwrap_or(0) > 0 {
            assert!(result["inline_in_flight_executions"].as_u64().unwrap_or(0) >= 1);
            observed_waiter = true;
            break;
        }
        sleep(Duration::from_millis(5)).await;
    }
    assert!(
        observed_waiter,
        "admission.status never exposed the semaphore waiter"
    );
    let _ = first_task.await.unwrap();
    let _ = second_task.await.unwrap();
}

#[tokio::test]
async fn alias_surface_certifies_declares_retracts_and_preserves_old_job_pages() {
    let Some(preloads) = minilm_alias_preload_config() else {
        eprintln!("skipping MiniLM alias e2e: local HF ONNX snapshot is missing");
        return;
    };
    let config = serde_json::json!({
        "preload_models": preloads,
        "inline": { "max_items": 1 },
        "jobs": { "execution_ttl_ms": 60_000, "result_page_bytes": 4096, "bulk_quantum_tokens": 2048 },
        "alias_admin_enabled": true
    })
    .to_string();
    let _lock = acquire_minilm_e2e_lock();
    let (_daemon, _module, mut consumer, route) = open_route_with_config(&config).await;
    let probe = certify_preloaded_models(&mut consumer, route, 120).await;
    let lanes = probe["result"]["lanes"].as_array().expect("probe lanes");
    let fingerprint_a = lanes
        .iter()
        .find(|lane| lane["model_id"] == "minilm-a")
        .and_then(|lane| lane["fingerprint"].as_str())
        .expect("minilm-a fingerprint")
        .to_string();
    let fingerprint_b = lanes
        .iter()
        .find(|lane| lane["model_id"] == "minilm-b")
        .and_then(|lane| lane["fingerprint"].as_str())
        .expect("minilm-b fingerprint")
        .to_string();
    assert_ne!(fingerprint_a, fingerprint_b);

    let before = route_request(
        &mut consumer,
        route,
        500,
        serde_json::json!({
            "method": "embed.query",
            "params": { "model": "minilm-a", "text": "alias before retract" }
        }),
    )
    .await;
    assert_eq!(before["result"]["equivalent_to"][0], fingerprint_b);

    let valid = route_request(
        &mut consumer,
        route,
        501,
        serde_json::json!({
            "method": "aliases.check_index",
            "params": {
                "index_fingerprint": fingerprint_a,
                "provenance_set": [fingerprint_a, fingerprint_b]
            }
        }),
    )
    .await;
    assert_eq!(valid["result"]["verdict"]["status"], "valid");

    let accepted = route_request(
        &mut consumer,
        route,
        502,
        serde_json::json!({
            "method": "embed.batch",
            "params": {
                "model": "minilm-a",
                "request_key": "alias-retroactive-e2e",
                "items": [
                    {"id": "a", "text": "alias job page one"},
                    {"id": "b", "text": "alias job page two"}
                ]
            }
        }),
    )
    .await;
    let job_id = accepted["result"]["job_id"].as_str().unwrap().to_string();
    let done = poll_embed_result(&mut consumer, route, 503, &job_id).await;
    assert_eq!(done["result"]["equivalent_to"][0], fingerprint_b);

    let retracted = route_request(
        &mut consumer,
        route,
        600,
        serde_json::json!({
            "method": "alias.retract",
            "params": {
                "fingerprint_a": fingerprint_a,
                "fingerprint_b": fingerprint_b,
                "evidence": {"reason": "e2e"}
            }
        }),
    )
    .await;
    assert_eq!(retracted["result"]["changed"], Value::Bool(true));

    let migration = route_request(
        &mut consumer,
        route,
        601,
        serde_json::json!({
            "method": "aliases.check_index",
            "params": {
                "index_fingerprint": fingerprint_a,
                "provenance_set": [fingerprint_a, fingerprint_b]
            }
        }),
    )
    .await;
    assert_eq!(
        migration["result"]["verdict"]["status"],
        "migration_required"
    );
    let pair_a = migration["result"]["verdict"]["retracted_pair"]["fingerprint_a"]
        .as_str()
        .unwrap();
    let pair_b = migration["result"]["verdict"]["retracted_pair"]["fingerprint_b"]
        .as_str()
        .unwrap();
    assert!(
        (pair_a == fingerprint_a && pair_b == fingerprint_b)
            || (pair_a == fingerprint_b && pair_b == fingerprint_a),
        "unexpected retracted pair: {migration:?}"
    );

    let old_page = route_request(
        &mut consumer,
        route,
        602,
        serde_json::json!({
            "method": "embed.result",
            "params": { "job_id": job_id, "page": 0 }
        }),
    )
    .await;
    assert_eq!(old_page["result"]["equivalent_to"][0], fingerprint_b);

    let after = route_request(
        &mut consumer,
        route,
        603,
        serde_json::json!({
            "method": "embed.query",
            "params": { "model": "minilm-a", "text": "alias after retract" }
        }),
    )
    .await;
    assert_eq!(after["result"]["equivalent_to"], Value::Array(Vec::new()));
}

#[tokio::test]
async fn probe_report_exposes_blocking_reasons_perf_rows_and_default_assignments() {
    let Some(preloads) = minilm_preload_config() else {
        eprintln!("skipping probe.report e2e: local HF ONNX snapshot is missing");
        return;
    };
    let _lock = acquire_minilm_e2e_lock();
    let (_daemon, _module, mut consumer, route) = open_route_with_preloads(Some(&preloads)).await;

    let before = route_request(
        &mut consumer,
        route,
        85,
        serde_json::json!({ "method": "probe.report", "params": {} }),
    )
    .await;
    assert_eq!(before["result"]["current_knob"], "balanced");
    assert_eq!(before["result"]["lanes"].as_array().unwrap().len(), 1);
    assert_eq!(
        before["result"]["lanes"][0]["blocking_reason"],
        "probe_required"
    );
    assert_eq!(before["result"]["lanes"][0]["certification_required"], true);
    assert_eq!(
        before["result"]["lanes"][0]["certification_status"],
        "uncertified"
    );
    assert_eq!(before["result"]["lanes"][0]["performance"], Value::Null);

    let probed = certify_preloaded_models(&mut consumer, route, 86).await;
    assert_eq!(probed["result"]["current_knob"], "balanced");
    assert_eq!(
        probed["result"]["knob_assignments"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    let perf = &probed["result"]["lanes"][0]["performance"];
    assert!(perf["throughput_tok_s"].as_f64().unwrap() > 0.0);
    assert!(perf["cold_load_ms"].as_f64().unwrap() >= 0.0);
    assert!(perf["single_item_latency_p50_ms"].as_f64().unwrap() >= 0.0);

    let report = route_request(
        &mut consumer,
        route,
        87,
        serde_json::json!({ "method": "probe.report", "params": {} }),
    )
    .await;
    let result = &report["result"];
    assert_eq!(result["current_knob"], "balanced");
    assert_eq!(result["knob_assignments"].as_array().unwrap().len(), 3);
    assert_eq!(result["active_assignments"].as_array().unwrap().len(), 1);
    assert_eq!(result["active_assignments"][0]["model_id"], "minilm");
    assert_eq!(result["lanes"][0]["blocking_reason"], Value::Null);
    assert_eq!(result["lanes"][0]["certification_required"], true);
    assert_eq!(result["lanes"][0]["certification_status"], "certified");
    assert_eq!(result["lanes"][0]["certification"]["status"], "certified");
    assert_eq!(result["lanes"][0]["certification_stale"], false);
    assert_eq!(result["lanes"][0]["performance_stale"], false);
    assert_eq!(result["lanes"][0]["performance"]["stale"], false);
}

#[tokio::test]
async fn quiet_knob_restart_uses_persisted_assignment() {
    let Some(preloads) = minilm_alias_preload_config() else {
        eprintln!("skipping knob routing e2e: local HF ONNX snapshot is missing");
        return;
    };
    let config = module_config_with_preloads(preloads.clone(), "balanced");
    let _lock = acquire_minilm_e2e_lock();
    let (daemon, mut module, mut consumer, route) = open_route_with_config(&config).await;
    certify_preloaded_models(&mut consumer, route, 120).await;

    let report = route_request(
        &mut consumer,
        route,
        121,
        serde_json::json!({ "method": "probe.report", "params": {} }),
    )
    .await;
    let lanes = report["result"]["lanes"].as_array().unwrap();
    assert_eq!(
        lanes.len(),
        2,
        "unexpected probe lanes (model_id/engine/fingerprint): {lanes:?}"
    );
    let current_model_id = report["result"]["active_assignments"][0]["model_id"]
        .as_str()
        .unwrap();
    let quiet_target = lanes
        .iter()
        .find(|lane| lane["model_id"].as_str().unwrap() != current_model_id)
        .cloned()
        .expect("alternate lane for quiet assignment");
    let quiet_model_id = quiet_target["model_id"].as_str().unwrap().to_string();
    let quiet_fingerprint = quiet_target["fingerprint"].as_str().unwrap().to_string();
    overwrite_knob_assignment(
        &expected_store_path(&daemon.data_home),
        report["result"]["machine_profile_hash"].as_str().unwrap(),
        "embed",
        "quiet",
        &quiet_target,
    );

    let _ = module.child.start_kill();
    let _ = module.child.wait().await;
    drop(consumer);

    let restarted = spawn_synapse_module_with_config(
        &daemon.connection_file_path,
        &module_config_with_preloads(preloads, "quiet"),
    );
    let (_daemon, _module, mut consumer, route) =
        open_route_for_started_module(daemon, restarted).await;

    let quiet_report = route_request(
        &mut consumer,
        route,
        122,
        serde_json::json!({ "method": "probe.report", "params": {} }),
    )
    .await;
    assert_eq!(quiet_report["result"]["current_knob"], "quiet");
    assert_eq!(
        quiet_report["result"]["active_assignments"][0]["model_id"],
        quiet_model_id
    );

    let first = route_request(
        &mut consumer,
        route,
        123,
        serde_json::json!({
            "method": "embed.query",
            "params": { "id": "quiet-knob", "text": "quiet knob uses the persisted assignment" }
        }),
    )
    .await;
    if first["result"]["error"]["code"] == "model_loading" {
        let ready = poll_model_ready(&mut consumer, route, 124, &quiet_model_id).await;
        assert_eq!(ready["result"]["state"], "ready");
    }
    let routed = route_request(
        &mut consumer,
        route,
        125,
        serde_json::json!({
            "method": "embed.query",
            "params": { "id": "quiet-knob-2", "text": "quiet knob uses the persisted assignment after restart" }
        }),
    )
    .await;
    assert_eq!(routed["result"]["fingerprint"], quiet_fingerprint);
}

#[tokio::test]
async fn os_build_override_marks_probe_rows_stale_in_report_and_status() {
    let Some(preloads) = minilm_preload_config() else {
        eprintln!("skipping stale probe e2e: local HF ONNX snapshot is missing");
        return;
    };
    let _lock = acquire_minilm_e2e_lock();
    let (daemon, mut module, mut consumer, route) = open_route_with_preloads(Some(&preloads)).await;
    certify_preloaded_models(&mut consumer, route, 130).await;

    let _ = module.child.start_kill();
    let _ = module.child.wait().await;
    drop(consumer);

    let restarted = spawn_synapse_module_with_env(
        &daemon.connection_file_path,
        Some(&preloads),
        None,
        &[("SYNAPSE_OS_BUILD_OVERRIDE", "synthetic-stale-build")],
    );
    let (_daemon, _module, mut consumer, route) =
        open_route_for_started_module(daemon, restarted).await;

    let report = route_request(
        &mut consumer,
        route,
        131,
        serde_json::json!({ "method": "probe.report", "params": {} }),
    )
    .await;
    assert_eq!(report["result"]["certification_stale"], true);
    assert_eq!(report["result"]["performance_stale"], true);
    assert_eq!(
        report["result"]["lanes"][0]["blocking_reason"],
        "probe_required"
    );
    assert_eq!(report["result"]["lanes"][0]["certification"]["stale"], true);
    assert_eq!(report["result"]["lanes"][0]["performance"]["stale"], true);
    assert_eq!(
        report["result"]["lanes"][0]["certification"]["stale_os_build"],
        true
    );
    assert_eq!(
        report["result"]["lanes"][0]["performance"]["stale_os_build"],
        true
    );

    ensure_model_loaded_by_query(&mut consumer, route, 132, "minilm").await;
    let status = route_request(
        &mut consumer,
        route,
        135,
        serde_json::json!({ "method": "admission.status", "params": {} }),
    )
    .await;
    assert_eq!(status["result"]["certification_stale"], true);
    assert_eq!(status["result"]["performance_stale"], true);
    assert_eq!(status["result"]["lanes"][0]["certification_stale"], true);
    assert_eq!(status["result"]["lanes"][0]["performance_stale"], true);
}

#[tokio::test]
async fn embed_query_deadline_one_returns_typed_rejection() {
    let Some(preloads) = minilm_preload_config() else {
        eprintln!("skipping MiniLM deadline e2e: local HF ONNX snapshot is missing");
        return;
    };
    let _lock = acquire_minilm_e2e_lock();
    let (_daemon, _module, mut consumer, route) = open_route_with_preloads(Some(&preloads)).await;
    certify_preloaded_models(&mut consumer, route, 90).await;

    let body = route_request(
        &mut consumer,
        route,
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
    let (_daemon, _module, mut consumer, route) = open_route_with_preloads(Some(&preloads)).await;
    certify_preloaded_models(&mut consumer, route, 110).await;
    let start_corr = 10_000_u64;
    let count = 32_u64;
    for offset in 0..count {
        let frame = Frame::build(
            FrameType::Request,
            Flags::new(false, Priority::Interactive, false),
            route.channel,
            route.epoch,
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

#[tokio::test]
async fn model_load_file_source_reaches_ready_and_lazy_reload_after_unload() {
    let Some(source_dir) = copied_minilm_source_dir("synapse-model-load-file") else {
        eprintln!("skipping model.load file e2e: local HF ONNX snapshot is missing");
        return;
    };
    let _lock = acquire_minilm_e2e_lock();
    let (_daemon, _module, mut consumer, route) = open_route().await;

    let accepted = route_request(
        &mut consumer,
        route,
        20_000,
        serde_json::json!({
            "method": "model.load",
            "params": {
                "source": "file",
                "path": source_dir,
                "files": { "model": "model.onnx", "tokenizer": "tokenizer.json" },
                "engine": "ort",
                "pooling": "mean",
                "task": "embed",
                "model_id": "minilm-loaded",
                "pin": true,
                "request_key": "model-load-file-e2e"
            }
        }),
    )
    .await;
    let job_id = accepted["result"]["job_id"].as_str().unwrap().to_string();
    let ready = poll_model_load_job(&mut consumer, route, 20_001, &job_id).await;
    assert_eq!(ready["result"]["state"], "ready");
    let model_id = ready["result"]["model_id"].as_str().unwrap().to_string();

    let uncertified = route_request(
        &mut consumer,
        route,
        20_100,
        serde_json::json!({
            "method": "embed.query",
            "params": { "model": &model_id, "text": "uncertified should fail" }
        }),
    )
    .await;
    assert!(
        uncertified["result"]["error"]["code"] == "probe_required"
            || uncertified["result"]["error"]["code"] == "not_certified"
    );

    run_probe_job(
        &mut consumer,
        route,
        20_200,
        serde_json::json!({ "models": [model_id.clone()] }),
    )
    .await;

    let first = route_request(
        &mut consumer,
        route,
        20_300,
        serde_json::json!({
            "method": "embed.query",
            "params": { "model": &model_id, "id": "first", "text": "hello from model.load" }
        }),
    )
    .await;
    assert_eq!(first["result"]["dims"].as_u64(), Some(384));

    let unloaded = route_request(
        &mut consumer,
        route,
        20_400,
        serde_json::json!({
            "method": "model.unload",
            "params": { "model_id": &model_id }
        }),
    )
    .await;
    assert_eq!(unloaded["result"]["state"], "unloaded");

    let loading = route_request(
        &mut consumer,
        route,
        20_500,
        serde_json::json!({
            "method": "embed.query",
            "params": { "model": &model_id, "text": "reload after unload" }
        }),
    )
    .await;
    assert_eq!(loading["result"]["error"]["code"], "model_loading");

    let ready_again = poll_model_ready(&mut consumer, route, 20_600, &model_id).await;
    assert_eq!(ready_again["result"]["state"], "ready");

    let second = route_request(
        &mut consumer,
        route,
        20_700,
        serde_json::json!({
            "method": "embed.query",
            "params": { "model": &model_id, "id": "second", "text": "reload succeeded" }
        }),
    )
    .await;
    assert_eq!(second["result"]["dims"].as_u64(), Some(384));
}

#[tokio::test]
async fn model_load_digest_mismatch_fails_with_artifact_invalid() {
    let Some(source_dir) = copied_minilm_source_dir("synapse-model-load-digest-mismatch") else {
        eprintln!("skipping model.load digest mismatch e2e: local HF ONNX snapshot is missing");
        return;
    };
    let _lock = acquire_minilm_e2e_lock();
    let (_daemon, _module, mut consumer, route) = open_route().await;

    let accepted = route_request(
        &mut consumer,
        route,
        21_000,
        serde_json::json!({
            "method": "model.load",
            "params": {
                "source": "file",
                "path": source_dir,
                "files": { "model": "model.onnx", "tokenizer": "tokenizer.json" },
                "expected_digest": format!("sha256:{}", "0".repeat(64)),
                "engine": "ort",
                "pooling": "mean",
                "task": "embed",
                "request_key": "model-load-digest-mismatch"
            }
        }),
    )
    .await;
    let job_id = accepted["result"]["job_id"].as_str().unwrap().to_string();
    let failed = poll_model_load_job(&mut consumer, route, 21_001, &job_id).await;
    assert_eq!(failed["result"]["state"], "failed");
    assert_eq!(failed["result"]["error"]["code"], "artifact_invalid");
    assert_eq!(failed["result"]["error"]["class"], "permanent");
}

#[tokio::test]
async fn model_load_restart_mid_download_marks_job_restarted_and_resubmit_succeeds() {
    let Some(source_dir) = copied_minilm_source_dir("synapse-model-load-restart") else {
        eprintln!("skipping model.load restart e2e: local HF ONNX snapshot is missing");
        return;
    };
    let _lock = acquire_minilm_e2e_lock();
    let daemon = start_daemon().await;
    let module = spawn_synapse_module_with_env(
        &daemon.connection_file_path,
        None,
        None,
        &[("SYNAPSE_TEST_MODEL_LOAD_CHUNK_DELAY_MS", "25")],
    );
    let (daemon, mut module, mut consumer, route) =
        open_route_for_started_module(daemon, module).await;

    let request = serde_json::json!({
        "method": "model.load",
        "params": {
            "source": "file",
            "path": source_dir,
            "files": { "model": "model.onnx", "tokenizer": "tokenizer.json" },
            "engine": "ort",
            "pooling": "mean",
            "task": "embed",
            "request_key": "model-load-restart"
        }
    });
    let accepted = route_request(&mut consumer, route, 22_000, request.clone()).await;
    let job_id = accepted["result"]["job_id"].as_str().unwrap().to_string();

    let deadline = Instant::now() + Duration::from_secs(30);
    let mut corr = 22_001;
    loop {
        let body = route_request(
            &mut consumer,
            route,
            corr,
            serde_json::json!({
                "method": "model.status",
                "params": { "job_id": &job_id }
            }),
        )
        .await;
        match body["result"]["state"].as_str() {
            Some("downloading") => break,
            Some("resolving" | "validating" | "loading") => {
                assert!(
                    Instant::now() < deadline,
                    "model.load never reached downloading: {body:?}"
                );
                corr += 1;
                sleep(Duration::from_millis(50)).await;
            }
            other => panic!("unexpected model.load state before restart {other:?}: {body:?}"),
        }
    }

    let _ = module.child.start_kill();
    let _ = module.child.wait().await;
    drop(consumer);

    let restarted = spawn_synapse_module_with_env(&daemon.connection_file_path, None, None, &[]);
    let (_daemon, _module, mut consumer, route) =
        open_route_for_started_module(daemon, restarted).await;

    let restarted_status = route_request(
        &mut consumer,
        route,
        22_100,
        serde_json::json!({
            "method": "model.status",
            "params": { "job_id": &job_id }
        }),
    )
    .await;
    assert_eq!(restarted_status["result"]["state"], "failed");
    assert_eq!(
        restarted_status["result"]["error"]["code"],
        "module_restarted"
    );

    let retried = route_request(&mut consumer, route, 22_200, request).await;
    let retried_job_id = retried["result"]["job_id"].as_str().unwrap().to_string();
    assert_ne!(retried_job_id, job_id);
    let ready = poll_model_load_job(&mut consumer, route, 22_201, &retried_job_id).await;
    assert_eq!(ready["result"]["state"], "ready");
}

async fn certify_preloaded_models(
    consumer: &mut tokio::net::TcpStream,
    route: TestRoute,
    start_corr: u64,
) -> Value {
    run_probe_job(consumer, route, start_corr, serde_json::json!({})).await
}

#[cfg(target_os = "macos")]
async fn poll_probe_status(
    consumer: &mut tokio::net::TcpStream,
    route: TestRoute,
    start_corr: u64,
    job_id: &str,
) -> Value {
    let deadline = Instant::now() + Duration::from_secs(1_800);
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
            Some("done" | "failed_transient" | "failed_permanent") => return body,
            Some("queued" | "running") => {
                assert!(
                    Instant::now() < deadline,
                    "probe did not complete before timeout: {body:?}"
                );
                corr += 1;
                sleep(Duration::from_millis(100)).await;
            }
            other => panic!("unexpected probe.status state {other:?}: {body:?}"),
        }
    }
}

async fn poll_probe_job(
    consumer: &mut tokio::net::TcpStream,
    route: TestRoute,
    start_corr: u64,
    params: Value,
) -> Value {
    let accepted = route_request(
        consumer,
        route,
        start_corr,
        serde_json::json!({ "method": "probe.start", "params": params }),
    )
    .await;
    let job_id = accepted["result"]["job_id"]
        .as_str()
        .expect("probe.start returns job_id")
        .to_string();
    let deadline = Instant::now() + Duration::from_secs(1_800);
    let mut corr = start_corr + 1;
    loop {
        let body = route_request(
            consumer,
            route,
            corr,
            serde_json::json!({
                "method": "probe.status",
                "params": { "job_id": &job_id }
            }),
        )
        .await;
        match body["result"]["state"].as_str() {
            Some("done") => return body,
            Some("failed_transient" | "failed_permanent") => panic!("probe failed: {body:?}"),
            Some("queued" | "running") => {
                assert!(
                    Instant::now() < deadline,
                    "probe did not complete before timeout: {body:?}"
                );
                corr += 1;
                sleep(Duration::from_millis(100)).await;
            }
            other => panic!("unexpected probe.status state {other:?}: {body:?}"),
        }
    }
}

async fn run_probe_job(
    consumer: &mut tokio::net::TcpStream,
    route: TestRoute,
    start_corr: u64,
    params: Value,
) -> Value {
    let body = poll_probe_job(consumer, route, start_corr, params).await;
    let lanes = body["result"]["lanes"].as_array().expect("probe lanes");
    assert!(!lanes.is_empty(), "probe should certify at least one lane");
    for lane in lanes {
        assert_eq!(lane["status"], "certified", "probe lane failed: {lane:?}");
    }
    body
}

async fn poll_model_load_job(
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
                "method": "model.status",
                "params": { "job_id": job_id }
            }),
        )
        .await;
        match body["result"]["state"].as_str() {
            Some("ready" | "failed") => return body,
            Some("resolving" | "downloading" | "validating" | "loading") => {
                assert!(
                    Instant::now() < deadline,
                    "model.load did not reach a terminal state before timeout: {body:?}"
                );
                corr += 1;
                sleep(Duration::from_millis(100)).await;
            }
            other => panic!("unexpected model.status state {other:?}: {body:?}"),
        }
    }
}

async fn poll_model_ready(
    consumer: &mut tokio::net::TcpStream,
    route: TestRoute,
    start_corr: u64,
    model_id: &str,
) -> Value {
    let deadline = Instant::now() + Duration::from_secs(180);
    let mut corr = start_corr;
    loop {
        let body = route_request(
            consumer,
            route,
            corr,
            serde_json::json!({
                "method": "model.status",
                "params": { "model_id": model_id }
            }),
        )
        .await;
        match body["result"]["state"].as_str() {
            Some("ready" | "failed") => return body,
            Some("unloaded" | "loading") => {
                assert!(
                    Instant::now() < deadline,
                    "model.status(model_id) did not become ready before timeout: {body:?}"
                );
                corr += 1;
                sleep(Duration::from_millis(100)).await;
            }
            other => panic!("unexpected model.status(model_id) state {other:?}: {body:?}"),
        }
    }
}

async fn poll_embed_result(
    consumer: &mut tokio::net::TcpStream,
    route: TestRoute,
    start_corr: u64,
    job_id: &str,
) -> Value {
    let deadline = Instant::now() + Duration::from_secs(30);
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
                    "embed.result did not reach a terminal state before timeout: {body:?}"
                );
                corr += 1;
                sleep(Duration::from_millis(100)).await;
            }
            other => panic!("unexpected embed.result state {other:?}: {body:?}"),
        }
    }
}

async fn ensure_model_loaded_by_query(
    consumer: &mut tokio::net::TcpStream,
    route: TestRoute,
    start_corr: u64,
    model_id: &str,
) {
    let body = route_request(
        consumer,
        route,
        start_corr,
        serde_json::json!({
            "method": "embed.query",
            "params": {
                "model": model_id,
                "id": "stale-load",
                "text": "load the model so admission.status can report stale probe rows"
            }
        }),
    )
    .await;
    match body["result"]["error"]["code"].as_str() {
        Some("model_loading") => {
            let ready = poll_model_ready(consumer, route, start_corr + 1, model_id).await;
            assert_eq!(ready["result"]["state"], "ready");
        }
        Some("not_certified" | "probe_required") | None => {}
        other => panic!("unexpected model load kick response {other:?}: {body:?}"),
    }
}

fn module_config_with_preloads(preloads: Value, knob: &str) -> String {
    serde_json::json!({
        "preload_models": preloads,
        "knob": knob,
    })
    .to_string()
}

fn overwrite_knob_assignment(
    store_path: &Path,
    machine_profile_hash: &str,
    workload: &str,
    knob: &str,
    lane: &Value,
) {
    let performance = lane["performance"]
        .as_object()
        .expect("lane performance row");
    let conn = Connection::open(store_path).expect("open synapse store");
    let changed = conn
        .execute(
            "UPDATE knob_assignments
             SET model_id = ?1,
                 numeric_profile_id = ?2,
                 fingerprint = ?3,
                 engine = ?4,
                 measured_at_ms = ?5,
                 os_build = ?6,
                 module_generation = ?7,
                 throughput_tok_s = ?8,
                 single_item_latency_p50_ms = ?9
             WHERE machine_profile_hash = ?10 AND workload = ?11 AND knob = ?12",
            params![
                lane["model_id"].as_str().unwrap(),
                lane["numeric_profile_id"].as_str().unwrap(),
                lane["fingerprint"].as_str().unwrap(),
                performance["engine"].as_str().unwrap(),
                performance["measured_at_ms"].as_u64().unwrap() as i64,
                performance["os_build"].as_str().unwrap(),
                performance["module_generation"].as_u64().unwrap() as i64,
                performance["throughput_tok_s"].as_f64().unwrap(),
                performance["single_item_latency_p50_ms"].as_f64().unwrap(),
                machine_profile_hash,
                workload,
                knob,
            ],
        )
        .expect("rewrite knob assignment");
    assert_eq!(changed, 1, "expected one knob assignment row to update");
}

#[cfg(target_os = "macos")]
fn assert_vector_cosine_at_least(actual: &Value, expected: &Value, minimum: f64) {
    let actual = actual.as_array().expect("actual vector is an array");
    let expected = expected.as_array().expect("expected vector is an array");
    assert_eq!(actual.len(), expected.len());
    let dot = actual
        .iter()
        .zip(expected)
        .map(|(left, right)| left.as_f64().unwrap() * right.as_f64().unwrap())
        .sum::<f64>();
    let actual_norm = actual
        .iter()
        .map(|value| value.as_f64().unwrap().powi(2))
        .sum::<f64>()
        .sqrt();
    let expected_norm = expected
        .iter()
        .map(|value| value.as_f64().unwrap().powi(2))
        .sum::<f64>()
        .sqrt();
    let cosine = dot / (actual_norm * expected_norm + 1e-12);
    assert!(
        cosine >= minimum,
        "vector cosine {cosine:.9} is below {minimum:.9}"
    );
}

fn assert_vectors_close(actual: &Value, expected: &Value) {
    let actual = actual.as_array().expect("actual vector is an array");
    let expected = expected.as_array().expect("expected vector is an array");
    assert_eq!(actual.len(), expected.len());
    for (left, right) in actual.iter().zip(expected) {
        let left = left.as_f64().unwrap();
        let right = right.as_f64().unwrap();
        assert!(
            (left - right).abs() < 1e-5,
            "vector components differ: {left} vs {right}"
        );
    }
}

fn copied_minilm_source_dir(label: &str) -> Option<PathBuf> {
    let snapshot = minilm_onnx_snapshot()?;
    let model_path = snapshot.join("model.onnx");
    let tokenizer_path = snapshot.join("tokenizer.json");
    if !model_path.exists() || !tokenizer_path.exists() {
        return None;
    }
    let source_dir = unique_temp_dir(label);
    std::fs::create_dir_all(&source_dir).ok()?;
    std::fs::copy(&model_path, source_dir.join("model.onnx")).ok()?;
    std::fs::copy(&tokenizer_path, source_dir.join("tokenizer.json")).ok()?;
    Some(source_dir)
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

#[cfg(target_os = "macos")]
fn test_sha256(path: PathBuf) -> String {
    use sha2::{Digest, Sha256};

    format!(
        "sha256:{}",
        hex::encode(Sha256::digest(std::fs::read(path).unwrap()))
    )
}

#[cfg(target_os = "macos")]
fn minilm_safetensors_snapshot() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("SYNAPSE_MINILM_SAFETENSORS_SNAPSHOT") {
        return Some(PathBuf::from(path));
    }
    let snapshots = PathBuf::from(std::env::var("HOME").ok()?)
        .join(".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots");
    first_snapshot_with(&snapshots, "model.safetensors")
}

#[cfg(target_os = "macos")]
fn gte_safetensors_snapshot() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("SYNAPSE_GTE_MODERNBERT_SAFETENSORS_SNAPSHOT") {
        return Some(PathBuf::from(path));
    }
    let snapshots = PathBuf::from(std::env::var("HOME").ok()?)
        .join(".cache/huggingface/hub/models--Alibaba-NLP--gte-modernbert-base/snapshots");
    first_snapshot_with(&snapshots, "model.safetensors")
}

#[cfg(target_os = "macos")]
fn gte_reranker_safetensors_snapshot() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("SYNAPSE_GTE_RERANKER_MODERNBERT_SAFETENSORS_SNAPSHOT") {
        return Some(PathBuf::from(path));
    }
    let snapshots = PathBuf::from(std::env::var("HOME").ok()?)
        .join(".cache/huggingface/hub/models--Alibaba-NLP--gte-reranker-modernbert-base/snapshots");
    first_snapshot_with(&snapshots, "model.safetensors")
}

fn minilm_alias_preload_config() -> Option<Value> {
    let snapshot = minilm_onnx_snapshot()?;
    let model_path = snapshot.join("model.onnx");
    let tokenizer_path = snapshot.join("tokenizer.json");
    if !model_path.exists() || !tokenizer_path.exists() {
        return None;
    }
    Some(serde_json::json!([
        {
            "model_id": "minilm-a",
            "engine": "ort",
            "model_path": model_path,
            "tokenizer_path": tokenizer_path,
            "pooling": "mean",
            "normalize": true,
            "max_tokens": 512,
            "quant": "fp32"
        },
        {
            "model_id": "minilm-b",
            "engine": "ort",
            "model_path": model_path,
            "tokenizer_path": tokenizer_path,
            "pooling": "mean",
            "normalize": true,
            "max_tokens": 512,
            "quant": "fp32-alias"
        }
    ]))
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
    #[allow(dead_code)]
    file: std::fs::File,
}

impl Drop for MinilmE2eLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[tokio::test]
async fn singleton_lease_blocks_second_module_on_same_daemon() {
    let shared_lease_root = unique_temp_dir("synapse-singleton-lease");
    std::fs::create_dir_all(&shared_lease_root).unwrap();
    let daemon = start_daemon().await;
    let first =
        spawn_synapse_module_on_shared_lease(&daemon.connection_file_path, &shared_lease_root);
    wait_for_registration(&daemon.registry, MODULE_ID, SETUP_TIMEOUT).await;
    let mut second =
        spawn_synapse_module_on_shared_lease(&daemon.connection_file_path, &shared_lease_root);
    let status = second
        .child
        .wait()
        .await
        .expect("second module should exit");
    assert!(
        !status.success(),
        "second synapse module should refuse singleton lease"
    );
    drop(second);
    drop(first);
}

#[tokio::test]
async fn microllm_grammar_disabled_when_module_gate_is_off() {
    let config = serde_json::json!({ "grammar_enabled": false }).to_string();
    let (_daemon, _module, mut consumer, route) = open_route_with_config(&config).await;
    let frame = raw_route_frame(
        &mut consumer,
        route,
        90_001,
        serde_json::json!({
            "method": "microllm.oneshot",
            "params": {
                "model": "missing",
                "prompt": "yes or no",
                "max_tokens": 4,
                "grammar": "root ::= \"yes\" | \"no\""
            }
        }),
    )
    .await;
    assert_eq!(frame.header.ty, FrameType::Error);
    let body: Value = serde_json::from_slice(&frame.body).unwrap();
    assert_eq!(body["code"], "grammar_disabled");
    assert!(body["message"]
        .as_str()
        .unwrap_or_default()
        .contains("disabled in module config"));
}

#[tokio::test]
async fn microllm_grammar_disabled_without_certified_owned_lane() {
    let config = serde_json::json!({ "grammar_enabled": true }).to_string();
    let (_daemon, _module, mut consumer, route) = open_route_with_config(&config).await;
    let frame = raw_route_frame(
        &mut consumer,
        route,
        90_002,
        serde_json::json!({
            "method": "microllm.oneshot",
            "params": {
                "model": "any",
                "prompt": "yes or no",
                "max_tokens": 4,
                "grammar": "root ::= \"yes\" | \"no\""
            }
        }),
    )
    .await;
    assert_eq!(frame.header.ty, FrameType::Error);
    let body: Value = serde_json::from_slice(&frame.body).unwrap();
    assert_eq!(body["code"], "grammar_disabled");
    assert!(body["message"]
        .as_str()
        .unwrap_or_default()
        .contains("certified and enabled owned-decode"));
}

#[tokio::test]
async fn uncertified_owned_target_refuses_and_maps_grammar_over_the_wire() {
    let fixture_dir = unique_temp_dir("owned-decode-preflight");
    std::fs::create_dir_all(&fixture_dir).unwrap();
    let model_path = fixture_dir.join("model.safetensors");
    std::fs::write(&model_path, b"owned decode preflight fixture").unwrap();
    let tokenizer_path = fixture_dir.join("tokenizer.json");
    let vocabulary = [("[UNK]".to_string(), 0), ("hello".to_string(), 1)]
        .into_iter()
        .collect();
    let word_level = WordLevel::builder()
        .vocab(vocabulary)
        .unk_token("[UNK]".to_string())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(word_level);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer.save(&tokenizer_path, false).unwrap();
    let config = serde_json::json!({
        "grammar_enabled": true,
        "preload_models": [{
            "model_id": "owned-qwen",
            "engine": DECODE_WORKER_ENGINE,
            "task": "generate",
            "model_path": model_path,
            "tokenizer_path": tokenizer_path,
            "format": "owned-safetensors",
            "max_tokens": 512,
            "quant": "f16",
            "family": "qwen3-0.6b",
            "dtype": "f16",
            "execution": "supervised"
        }]
    })
    .to_string();
    let (_daemon, _module, mut consumer, route) = open_route_with_config(&config).await;

    let refusal = raw_route_frame(
        &mut consumer,
        route,
        90_003,
        serde_json::json!({
            "method": "microllm.oneshot",
            "params": {
                "model": "owned-qwen",
                "prompt": "hello",
                "max_tokens": 4
            }
        }),
    )
    .await;
    assert_eq!(refusal.header.ty, FrameType::Error);
    let body: Value = serde_json::from_slice(&refusal.body).unwrap();
    // On macOS the owned lane exists but is uncertified; elsewhere the owned
    // engine is platform-gated out entirely, so resolution attaches the
    // truthful platform refusal before certification is ever consulted.
    let expected_refusal = if cfg!(target_os = "macos") {
        "owned_decode_not_certified"
    } else {
        "owned_decode_unsupported"
    };
    assert_eq!(body["code"], expected_refusal);

    let grammar_refusal = raw_route_frame(
        &mut consumer,
        route,
        90_004,
        serde_json::json!({
            "method": "microllm.oneshot",
            "params": {
                "model": "owned-qwen",
                "prompt": "hello",
                "max_tokens": 4,
                "grammar": r#"{"type":"string"}"#
            }
        }),
    )
    .await;
    assert_eq!(grammar_refusal.header.ty, FrameType::Error);
    let body: Value = serde_json::from_slice(&grammar_refusal.body).unwrap();
    assert_eq!(body["code"], "grammar_disabled");
    let _ = std::fs::remove_dir_all(fixture_dir);
}

#[tokio::test]
async fn substitutable_owned_refusal_falls_back_to_llama_with_lane_provenance() {
    let fixture_dir = unique_temp_dir("owned-decode-fallback");
    std::fs::create_dir_all(&fixture_dir).unwrap();
    let owned_model = fixture_dir.join("owned.safetensors");
    let llama_model = fixture_dir.join("fallback.gguf");
    std::fs::write(&owned_model, b"owned decode fallback fixture").unwrap();
    std::fs::write(&llama_model, b"llama fallback fixture").unwrap();
    let tokenizer_path = fixture_dir.join("tokenizer.json");
    let vocabulary = [
        ("[UNK]".to_string(), 0),
        ("hello".to_string(), 1),
        ("fallback".to_string(), 2),
    ]
    .into_iter()
    .collect();
    let word_level = WordLevel::builder()
        .vocab(vocabulary)
        .unk_token("[UNK]".to_string())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(word_level);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer.save(&tokenizer_path, false).unwrap();
    let llama_runtime = std::env::temp_dir().join(format!("syn-llama-{}", process::id()));
    let config = serde_json::json!({
        "dev": { "owned_decode_cutover_for_test": true },
        "preload_models": [
            {
                "model_id": "owned-qwen",
                "engine": DECODE_WORKER_ENGINE,
                "task": "generate",
                "model_path": owned_model,
                "tokenizer_path": tokenizer_path,
                "format": "owned-safetensors",
                "max_tokens": 512,
                "quant": "f16",
                "family": "qwen3-0.6b",
                "dtype": "f16",
                "execution": "supervised"
            },
            {
                "model_id": "llama-fallback",
                "engine": LLAMA_WORKER_ENGINE,
                "task": "generate",
                "model_path": llama_model,
                "tokenizer_path": tokenizer_path,
                "format": "gguf",
                "max_tokens": 512,
                "worker_bin": env!("CARGO_BIN_EXE_synapse-worker-timeout-mock"),
                "worker_runtime_dir": llama_runtime
            }
        ]
    })
    .to_string();
    let (_daemon, _module, mut consumer, route) = open_route_with_config(&config).await;

    let fallback = route_request(
        &mut consumer,
        route,
        90_005,
        serde_json::json!({
            "method": "microllm.oneshot",
            "params": {
                "model": "owned-qwen",
                "prompt": "hello",
                "max_tokens": 4
            }
        }),
    )
    .await;
    let result = &fallback["result"];
    assert_eq!(result["text"], "fallback");
    assert_eq!(result["provenance"]["lane"], "llama");
    assert_eq!(
        result["provenance"]["fallback_reason"],
        if !cfg!(target_os = "macos") {
            // The platform refusal fires at resolution, before certification
            // or cutover are consulted, regardless of profile.
            "owned_decode_unsupported"
        } else if cfg!(debug_assertions) {
            "owned_decode_not_certified"
        } else {
            "cutover_disabled"
        }
    );
    assert!(result["provenance"]["decode_fingerprint"].is_string());
    assert!(result["provenance"]["processing_fingerprint"].is_string());
    assert_ne!(
        result["provenance"]["decode_fingerprint"],
        Value::String(String::new())
    );

    let llama_only = route_request(
        &mut consumer,
        route,
        90_006,
        serde_json::json!({
            "method": "microllm.oneshot",
            "params": {
                "model": "llama-fallback",
                "prompt": "hello",
                "max_tokens": 4
            }
        }),
    )
    .await;
    let llama_provenance = &llama_only["result"]["provenance"];
    assert!(llama_provenance.get("fallback_reason").is_none());
    assert!(llama_provenance.get("decode_fingerprint").is_none());
    assert!(llama_provenance.get("processing_fingerprint").is_none());
    let _ = std::fs::remove_dir_all(fixture_dir);
}

#[cfg(target_os = "macos")]
async fn certified_owned_checkpoint_lane(
    checkpoint_env: &str,
    model_id: &str,
    family: &str,
    quant: &str,
    derived_digest: Option<&str>,
    constrained_serve: bool,
) {
    let Some(snapshot) = std::env::var_os(checkpoint_env).map(PathBuf::from) else {
        eprintln!("skipping certified owned wire e2e: set {checkpoint_env}");
        return;
    };
    let Some(worker_bin) = std::env::var_os("SYNAPSE_OWNED_DECODE_WORKER_BIN").map(PathBuf::from)
    else {
        eprintln!("skipping certified owned wire e2e: set SYNAPSE_OWNED_DECODE_WORKER_BIN");
        return;
    };
    let model_path = snapshot.join("model.safetensors");
    let tokenizer_path = snapshot.join("tokenizer.json");
    if !model_path.is_file() || !tokenizer_path.is_file() || !worker_bin.is_file() {
        eprintln!("skipping certified owned wire e2e: checkpoint or worker binary is missing");
        return;
    }
    let worker_runtime = std::env::temp_dir().join(format!(
        "sod-{}-{}-{}",
        &family[..1],
        &quant[..1],
        process::id()
    ));
    let artifact_digest = match family {
        "qwen3-0.6b" => "sha256:0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
        "lfm2-1.2b" => "sha256:60fef6ef4481c533ce7427793bed50200b55b3c68d0d00c52bc56f207a9acecd",
        _ => panic!("unsupported checkpoint family {family}"),
    };
    let mut preload = serde_json::json!({
        "model_id": model_id,
        "engine": DECODE_WORKER_ENGINE,
        "task": "generate",
        "model_path": model_path,
        "tokenizer_path": tokenizer_path,
        "format": "owned-safetensors",
        "artifact_digest": artifact_digest,
        "max_tokens": 512,
        "quant": quant,
        "family": family,
        "dtype": "f16",
        "execution": "supervised",
        "worker_bin": worker_bin,
        "worker_runtime_dir": worker_runtime
    });
    if let Some(derived_digest) = derived_digest {
        preload["quantizer_revision"] = Value::String("q8-ingest-v1".to_string());
        preload["derived_digest"] = Value::String(derived_digest.to_string());
    }
    let config = serde_json::json!({
        "grammar_enabled": true,
        "dev": { "owned_decode_cutover_for_test": true },
        "preload_models": [preload]
    })
    .to_string();
    let (_daemon, _module, mut consumer, route) = open_route_with_config(&config).await;
    let probe = certify_preloaded_models(&mut consumer, route, 91_000).await;
    let lane = &probe["result"]["lanes"][0];
    let exact = lane["evidence"]["metrics"]["token_exact_matches"]
        .as_u64()
        .expect("probe exact-match count");
    let accepted = lane["evidence"]["metrics"]["accepted_structural_forks"]
        .as_u64()
        .expect("probe structural-fork count");
    assert_eq!(exact + accepted, 20, "checkpoint prompt battery failed");

    let owned = route_request(
        &mut consumer,
        route,
        91_001,
        serde_json::json!({
            "method": "microllm.oneshot",
            "params": {
                "model": model_id,
                "prompt": "The capital of France is",
                "max_tokens": 64
            }
        }),
    )
    .await;
    let result = &owned["result"];
    assert_eq!(
        result["provenance"]["engine"]["engine"],
        DECODE_WORKER_ENGINE
    );
    assert_eq!(result["provenance"]["lane"], "decode");
    assert_eq!(result["provenance"]["worker"], "supervised");
    assert_eq!(
        result["provenance"]["decode_fingerprint"], lane["fingerprint"],
        "served envelope must carry the certified lane fingerprint"
    );
    assert!(result["provenance"]["processing_fingerprint"].is_string());
    assert!(result["provenance"]["worker_generation"].as_u64().is_some());
    assert_eq!(result["n_gen"], 64);

    if !constrained_serve {
        return;
    }
    let constrained = route_request(
        &mut consumer,
        route,
        91_002,
        serde_json::json!({
            "method": "microllm.oneshot",
            "params": {
                "model": model_id,
                "prompt": "Respond with exactly the JSON literal null and nothing else:\n",
                "max_tokens": 64,
                "grammar": r#"{"type":"null"}"#
            }
        }),
    )
    .await;
    let constrained_result = &constrained["result"];
    let value: Value = serde_json::from_str(
        constrained_result["text"]
            .as_str()
            .expect("constrained response text"),
    )
    .expect("constrained response is JSON");
    assert!(value.is_null());
    assert!(constrained_result["provenance"]["constraint_runtime_identity"].is_string());
    assert!(constrained_result["provenance"]["constraint_fingerprint"].is_string());
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore]
async fn certified_owned_oneshot_and_constraint_serve_after_real_worker_probe() {
    certified_owned_checkpoint_lane(
        "SYNAPSE_OWNED_DECODE_QWEN3_0_6B",
        "qwen3-0.6b-decode-f16",
        "qwen3-0.6b",
        "f16",
        None,
        true,
    )
    .await;
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore]
async fn checkpoint_lfm2_f16_certified_owned_lane_serves_exact_tokens() {
    certified_owned_checkpoint_lane(
        "SYNAPSE_OWNED_DECODE_LFM2_1_2B",
        "lfm2-1.2b-decode-f16",
        "lfm2-1.2b",
        "f16",
        None,
        false,
    )
    .await;
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore]
async fn checkpoint_qwen3_q8_0_certified_owned_lane_serves_exact_tokens() {
    certified_owned_checkpoint_lane(
        "SYNAPSE_OWNED_DECODE_QWEN3_0_6B",
        "qwen3-0.6b-decode-q8_0",
        "qwen3-0.6b",
        "q8_0",
        Some("17d2fbfeff90269190287f324ed93bab3bb1b4fa4aad98c3fbba1868c01cb0f2"),
        false,
    )
    .await;
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore]
async fn checkpoint_lfm2_q8_0_certified_owned_lane_serves_exact_tokens() {
    certified_owned_checkpoint_lane(
        "SYNAPSE_OWNED_DECODE_LFM2_1_2B",
        "lfm2-1.2b-decode-q8_0",
        "lfm2-1.2b",
        "q8_0",
        Some("5874faabdce2567dcc0e7339e9547d79421ba312c71e3442c9cc3c4ed3cb47d0"),
        false,
    )
    .await;
}

#[test]
fn production_binary_carries_owned_decode_errors_and_retires_legacy_grammar_literal() {
    let binary = std::fs::read(env!("CARGO_BIN_EXE_ck-synapse")).unwrap();
    let contains = |literal: &[u8]| {
        binary
            .windows(literal.len())
            .any(|window| window == literal)
    };
    assert!(contains(b"grammar_disabled"));
    assert!(contains(b"owned_decode_not_certified"));
    // The legacy literal may appear ONLY as retirement-changelog data: the
    // baked wire-error-bindings manifest records the retirement (retired_id +
    // notes). The binary must carry exactly the manifest's occurrences and
    // not one more — an extra occurrence means an emittable legacy error
    // string came back.
    let count = |haystack: &[u8], literal: &[u8]| {
        haystack
            .windows(literal.len())
            .filter(|window| *window == literal)
            .count()
    };
    let legacy = b"grammar_unavailable_in_build";
    let manifest =
        include_bytes!("../owned-decode-manifests/owned-decode-wire-error-bindings-v1.json");
    let sanctioned = count(manifest, legacy);
    assert!(sanctioned > 0, "manifest no longer records the retirement");
    let in_binary = count(&binary, legacy);
    assert!(
        in_binary == sanctioned,
        "legacy grammar literal appears outside the retirement changelog: {in_binary} in binary vs {sanctioned} sanctioned by the bindings manifest"
    );
}

fn acquire_minilm_e2e_lock() -> MinilmE2eLock {
    let path = std::env::temp_dir().join("synapse-minilm-e2e.lock");
    loop {
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(file) => return MinilmE2eLock { path, file },
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(error) => panic!("failed to acquire MiniLM e2e lock: {error}"),
        }
    }
}
