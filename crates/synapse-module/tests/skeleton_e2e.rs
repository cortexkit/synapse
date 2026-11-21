#![forbid(unsafe_code)]

#[cfg(not(target_os = "macos"))]
mod backend_check;
mod catalog_e2e;
mod common;
mod embed_batch_module_cost;

use std::{
    net::Ipv4Addr,
    path::{Path, PathBuf},
    process,
    sync::Arc,
    time::Duration,
};

use common::read_frame_timeout;
use common::{
    configure_test_module_command, connect_consumer, describe_exit_status, install_test_tracing,
    raw_route_frame, route_open, route_request, unique_temp_dir, wait_for_catalog, TestRoute,
    MODULE_ID, SETUP_TIMEOUT,
};
use rusqlite::Connection;
use serde_json::Value;
use subc_daemon::{
    daemon_config::StorageConfig, serve_listener, ControlHandler, Registry, Router, ServerAuth,
};
use subc_protocol::Frame;
use subc_protocol::FrameType;
use subc_protocol::{Flags, Priority};
use subc_transport::write_frame;
use subc_transport::{
    generate_daemon_id, generate_key, write_atomic, ConnectionInfo, Endpoint, SCHEMA_VERSION,
};
use synapse_core::worker_engine_names::LLAMA_WORKER_ENGINE;
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
    data_home: PathBuf,
    task: tokio::task::JoinHandle<Result<(), subc_daemon::ServerError>>,
}

impl Drop for TestDaemon {
    fn drop(&mut self) {
        self.task.abort();
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
    install_test_tracing();
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
    let mut command =
        synapse_module_command(subc_connection_file, preload_models, config_json, extra_env);
    let child = command.spawn().expect("spawn synapse-module");
    ModuleProcess { child }
}

fn synapse_module_command(
    subc_connection_file: &Path,
    preload_models: Option<&str>,
    config_json: Option<&str>,
    extra_env: &[(&str, &str)],
) -> Command {
    let config_contents = if let Some(config_json) = config_json {
        Some(config_json.to_string())
    } else {
        preload_models.map(|preload_models| {
            serde_json::json!({ "preload_models": serde_json::from_str::<Value>(preload_models).unwrap_or_else(|_| serde_json::json!([])) }).to_string()
        })
    };
    let mut command = Command::new(
        synapse_core::dev_binary::ckdev_binary(
            env!("CARGO_BIN_EXE_ck-synapse"),
            subc_connection_file.parent().unwrap(),
        )
        .unwrap(),
    );
    command
        .arg("--subc")
        .arg(subc_connection_file)
        .env("SUBC_MODULE_ID", MODULE_ID);
    configure_test_module_command(&mut command, config_contents.as_deref());
    command.stderr(process::Stdio::inherit()).kill_on_drop(true);
    for (key, value) in extra_env {
        command.env(key, value);
    }
    command
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

/// A daemon cut closes the daemon side of the module's connection. The module
/// must then exit on its own with a success status and give up its singleton
/// lease, so a restarted module can take it. The stop is asserted on that
/// path, not on a kill.
#[tokio::test]
async fn module_exits_cleanly_and_releases_its_lease_when_the_daemon_connection_closes() {
    use cortexkit_lease::{FileLeaseStore, LeaseKey, LeaseStore};

    // The daemon runs each accepted connection as a detached task, so aborting
    // its accept loop leaves the module's connection open and the module never
    // sees it close. Running the daemon on a runtime of its own lets the test
    // shut that runtime down, which cancels every connection task and closes
    // the socket, as a real daemon exit does.
    let (daemon_runtime, daemon) = std::thread::spawn(|| {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("daemon runtime");
        let daemon = runtime.block_on(start_daemon());
        (runtime, daemon)
    })
    .join()
    .expect("start the daemon on its own runtime");
    let lease_root = unique_temp_dir("synapse-lifecycle-lease");
    std::fs::create_dir_all(&lease_root).unwrap();
    let lease_root_arg = lease_root.to_string_lossy().into_owned();
    let module = spawn_synapse_module_with_env(
        &daemon.connection_file_path,
        None,
        None,
        &[("CORTEXKIT_LEASE_ROOT", lease_root_arg.as_str())],
    );
    // Registration alone is not enough before tearing the daemon down: the
    // daemon records the module as registered before its HELLO_ACK reaches the
    // module, so a shutdown in that window ends the module with "connection
    // closed before HELLO_ACK" instead of a served-then-closed connection.
    // A route the module has answered proves it finished the handshake.
    let (daemon, mut module, consumer, _route) =
        open_route_for_started_module(daemon, module).await;
    let singleton = LeaseKey::new(MODULE_ID, "file", "singleton");
    assert!(
        FileLeaseStore::new(&lease_root)
            .acquire(&singleton)
            .is_err(),
        "a serving module must hold its singleton lease"
    );

    drop(consumer);
    drop(daemon);
    daemon_runtime.shutdown_background();
    let status = tokio::time::timeout(Duration::from_secs(30), module.child.wait())
        .await
        .expect("the module must exit once the daemon side of its connection is gone")
        .expect("wait for module after daemon connection close");
    assert!(
        status.success(),
        "module exited unexpectedly: {}",
        describe_exit_status(&status)
    );
    FileLeaseStore::new(&lease_root)
        .acquire(&singleton)
        .expect("a stopped module must have released its singleton lease");
    let _ = std::fs::remove_dir_all(lease_root);
}

/// The daemon hands each module its launch nonce on descriptor 3 (named by
/// `SUBC_LAUNCH_NONCE_FD`) and will stop setting the `SUBC_LAUNCH_NONCE`
/// environment copy. This launches the real module with the nonce ONLY on the
/// descriptor, the way the daemon does, and checks that it still boots,
/// registers, reports where it read the nonce, and serves over a route.
///
/// The in-process test daemon has no supervisor, so it does not check the
/// nonce against one it minted; what this proves is the module side: the
/// descriptor is read, HELLO carries the nonce it held, and serving works.
#[cfg(unix)]
#[tokio::test]
async fn module_launched_with_the_nonce_only_on_fd_3_registers_and_serves() {
    use subc_os::launch_nonce::{LaunchNonceHandoff, LAUNCH_NONCE_ENV, LAUNCH_NONCE_FD_ENV};
    use subc_protocol::manifest::LaunchNonceSource;

    let daemon = start_daemon().await;
    let handoff = LaunchNonceHandoff::new("fd-only-launch-nonce").expect("make the nonce pipe");
    let fd_env = handoff.fd_env_value();
    let mut command = synapse_module_command(
        &daemon.connection_file_path,
        None,
        None,
        &[(LAUNCH_NONCE_FD_ENV, fd_env.as_str())],
    );
    command.env_remove(LAUNCH_NONCE_ENV);
    // Last, after every other pre-exec step, as the handoff requires.
    handoff.install_last(command.as_std_mut());
    let module = ModuleProcess {
        child: command.spawn().expect("spawn synapse-module"),
    };

    let (daemon, _module, mut consumer, route) =
        open_route_for_started_module(daemon, module).await;

    let registration = daemon
        .registry
        .get_module(MODULE_ID)
        .unwrap()
        .expect("the module registered");
    let source = registration
        .manifest
        .provenance
        .as_ref()
        .and_then(|provenance| provenance.launch_nonce_source.clone());
    assert_eq!(
        source,
        Some(LaunchNonceSource::Fd),
        "HELLO must report that the nonce came from the descriptor"
    );

    let body = route_request(
        &mut consumer,
        route,
        2,
        serde_json::json!({ "method": "models.list", "params": {} }),
    )
    .await;
    assert_eq!(
        body["result"]["models"],
        Value::Array(Vec::new()),
        "body: {body}"
    );
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

    let mut consumer = connect_consumer(&daemon.connection_file_path).await;
    wait_for_catalog(&mut consumer, MODULE_ID, SETUP_TIMEOUT).await;
    let route = route_open(&mut consumer, 1).await;
    (daemon, module, consumer, route)
}

fn expected_store_path(data_home: &Path) -> PathBuf {
    data_home.join("cortexkit").join(MODULE_ID).join("store.db")
}

async fn wait_for_module_log(logs_dir: &Path, needle: &str) -> String {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        // Read every segment so a UTC date rollover cannot hide the line.
        let contents = std::fs::read_dir(logs_dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "log"))
            .filter_map(|entry| std::fs::read_to_string(entry.path()).ok())
            .collect::<String>();
        if contents.contains(needle) {
            return contents;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for '{needle}' in {}:\n{contents}",
            logs_dir.display()
        );
        sleep(Duration::from_millis(25)).await;
    }
}

#[tokio::test]
async fn event_publisher_attempts_delivery_after_daemon_boot() {
    let daemon = start_daemon().await;
    let config = serde_json::json!({
        "preload_models": serde_json::from_str::<Value>(&deterministic_preload_config()).unwrap(),
        "inline": { "max_items": 1 }
    })
    .to_string();
    let mut command =
        synapse_module_command(&daemon.connection_file_path, None, Some(&config), &[]);
    command
        .env("XDG_DATA_HOME", &daemon.data_home)
        .env("LOCALAPPDATA", &daemon.data_home)
        .env("CORTEXKIT_MODEL_CACHE", daemon.data_home.join("models"));
    let module = ModuleProcess {
        child: command.spawn().expect("spawn synapse-module"),
    };
    let (daemon, _module, mut consumer, route) =
        open_route_for_started_module(daemon, module).await;

    // Exceed the inline limit to commit a durable result page and its outbox event.
    let accepted = route_request(
        &mut consumer,
        route,
        200,
        serde_json::json!({
            "method": "embed.batch",
            "params": {
                "model": "test-minilm",
                "request_key": "publisher-start-e2e",
                "items": [
                    { "id": "first", "text": "hello world" },
                    { "id": "second", "text": "hello again" }
                ]
            }
        }),
    )
    .await;
    let job_id = accepted["result"]["job_id"].as_str().expect("durable job");
    let done = poll_embed_result(&mut consumer, route, 300, job_id).await;
    assert_eq!(done["result"]["state"], "done", "{done}");
    assert!(done["result"]["page_count"].as_u64().unwrap() > 0);

    // The fixture daemon has no bus configured. A retained row alone cannot
    // distinguish an outage from a publisher that never started; its lifecycle
    // warning proves the publisher reached the delivery loop.
    let logs_dir = daemon
        .data_home
        .join("cortexkit")
        .join(MODULE_ID)
        .join("logs");
    let contents = wait_for_module_log(
        &logs_dir,
        "event publisher unavailable; committed events remain in the outbox",
    )
    .await;
    let warning = contents
        .lines()
        .find(|line| {
            line.contains("event publisher unavailable; committed events remain in the outbox")
        })
        .expect("publisher outage warning");
    assert!(warning.contains("WARN"), "{warning}");
    assert!(warning.contains("lifecycle:"), "{warning}");
    assert!(warning.contains("error="), "{warning}");

    let conn =
        Connection::open(expected_store_path(&daemon.data_home)).expect("open synapse store");
    let (events, unpublished): (u64, u64) = conn
        .query_row(
            "SELECT count(*), count(*) FILTER (WHERE published_at IS NULL) FROM event_outbox \
             WHERE subject = 'event.synapse.job_page.v1' AND json_extract(body, '$.job_id') = ?1",
            [job_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .expect("read committed job-page events");
    assert!(events > 0, "durable batch must enqueue a job-page event");
    assert_eq!(unpublished, events, "an outage must retain pending events");
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
    assert_eq!(body["result"]["table_epoch"].as_u64(), Some(2));
    assert_eq!(body["result"]["models"], Value::Array(Vec::new()));
    assert_eq!(body["result"]["alias_rows"], Value::Array(Vec::new()));

    let store_path = expected_store_path(&daemon.data_home);
    assert!(
        store_path.exists(),
        "synapse should open the HELLO_ACK store at {}",
        store_path.display()
    );

    let conn = Connection::open(&store_path).expect("open migrated synapse store");
    for table in ["module_meta", "jobs", "catalog_self_checks", "models"] {
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
async fn spawned_module_ignores_operator_config() {
    let operator_home = unique_temp_dir("synapse-operator-config");
    let operator_config_dir = operator_home.join(".config").join("cortexkit");
    std::fs::create_dir_all(&operator_config_dir).unwrap();
    std::fs::write(
        operator_config_dir.join("synapse.jsonc"),
        r#"{"knob":"quiet"}"#,
    )
    .unwrap();
    let operator_home = operator_home.to_string_lossy().into_owned();

    let daemon = start_daemon().await;
    let module = spawn_synapse_module_with_env(
        &daemon.connection_file_path,
        None,
        None,
        &[("HOME", &operator_home)],
    );
    let (_daemon, _module, mut consumer, route) =
        open_route_for_started_module(daemon, module).await;
    let report = route_request(
        &mut consumer,
        route,
        2,
        serde_json::json!({ "method": "admission.status", "params": {} }),
    )
    .await;

    assert_eq!(report["result"]["current_knob"], "balanced");
    assert_eq!(report["result"]["lanes"], Value::Array(Vec::new()));
}

#[tokio::test]
async fn embed_query_default_catalog_returns_typed_model_not_installed_error() {
    let daemon = start_daemon().await;
    let module = spawn_synapse_module_with_env(
        &daemon.connection_file_path,
        None,
        None,
        &[("SYNAPSE_TEST_RUNNABLE_BACKENDS", "metal")],
    );
    let (_daemon, _module, mut consumer, route) =
        open_route_for_started_module(daemon, module).await;
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
    assert_eq!(body["result"]["error"]["code"], "model_not_installed");
    assert_eq!(
        body["result"]["error"]["details"]["catalog_id"],
        "gte-modernbert-base"
    );
    assert_eq!(
        body["result"]["error"]["details"]["download_op"],
        "models.download"
    );
    assert_eq!(body["result"]["error"]["class"], "permanent");
    assert_eq!(
        body["result"]["error"]["safe_to_retry_same_request"],
        Value::Bool(false)
    );
}

#[tokio::test]
async fn embed_query_preloaded_minilm_returns_vectors_and_envelope() {
    let preloads = deterministic_preload_config();
    let (_daemon, _module, mut consumer, route) = open_route_with_preloads(Some(&preloads)).await;

    let first = route_request(
        &mut consumer,
        route,
        4,
        serde_json::json!({
            "method": "embed.query",
            "params": { "model": "test-minilm", "id": "q1", "text": "hello world", "allow_equivalent": true, "required_epoch": 2 }
        }),
    )
    .await;
    let second = route_request(
        &mut consumer,
        route,
        5,
        serde_json::json!({
            "method": "embed.query",
            "params": { "model": "test-minilm", "id": "q2", "text": "hello world again" }
        }),
    )
    .await;

    let result = &first["result"];
    assert_eq!(result["module_generation"].as_u64(), Some(1));
    assert_eq!(result["table_epoch"].as_u64(), Some(2));
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
    assert_eq!(
        result["provenance"]["engine"]["engine"],
        "test-deterministic"
    );
    let mut expected = vec![0.0_f32; 384];
    // Golden nonzero bins for hello/world (token IDs 1 and 2) and the seed.
    for bin in [0, 259, 200] {
        expected[bin] = 1.0 / 3.0_f32.sqrt();
    }
    assert_vectors_close(
        &result["vectors"][0]["vector"],
        &serde_json::json!(expected),
    );
    assert_ne!(
        result["vectors"][0]["vector"],
        second["result"]["vectors"][0]["vector"]
    );
    assert_eq!(result["fingerprint"], second["result"]["fingerprint"]);

    let future_epoch = route_request(&mut consumer, route, 6, serde_json::json!({
        "method": "embed.query", "params": {"model":"test-minilm", "text":"hello world", "required_epoch":3}
    })).await;
    assert_eq!(
        future_epoch["result"]["error"]["code"],
        "migration_required"
    );
}

#[tokio::test]
async fn embed_batch_preloaded_minilm_preserves_order_and_envelope() {
    let preloads = deterministic_preload_config();
    let (_daemon, _module, mut consumer, route) = open_route_with_preloads(Some(&preloads)).await;
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
            "params": { "model": "test-minilm", "items": items }
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
    for pair in vectors.windows(2) {
        assert_ne!(pair[0]["vector"], pair[1]["vector"]);
    }
    for (index, vector) in vectors.iter().enumerate() {
        assert_eq!(vector["id"], format!("item-{index:02}"));
        assert_eq!(vector["vector"].as_array().unwrap().len(), 384);
    }
}

#[tokio::test]
async fn over_budget_embed_batch_returns_job_and_pages_results() {
    let preloads = deterministic_preload_config();
    let preload_models: Value = serde_json::from_str(&preloads).expect("preload config is json");
    let config = serde_json::json!({
        "preload_models": preload_models,
        "inline": { "max_items": 2 },
        "jobs": {
            "execution_ttl_ms": 60_000,
            "result_retention_ttl_ms": 60_000,
            "result_page_bytes": 4_096,
            "bulk_quantum_tokens": 16
        }
    })
    .to_string();
    let (_daemon, _module, mut consumer, route) = open_route_with_config(&config).await;
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
                "params": { "model": "test-minilm", "id": format!("item-{index}"), "text": text }
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
            "params": { "model": "test-minilm", "request_key": "job-tier-e2e", "items": items }
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
                "model": "test-minilm",
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
                "model": "test-minilm",
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

/// Every row reports the hash of the text it was sent and the hash of the
/// text it embedded: equal under the token ceiling, different (and disclosed
/// as truncated) over it, for single queries and for each row of a batch.
#[tokio::test]
async fn embed_rows_report_submitted_and_content_hashes_around_the_ceiling() {
    let preloads = deterministic_preload_config();
    let (_daemon, _module, mut consumer, route) = open_route_with_preloads(Some(&preloads)).await;

    let over_cap_text = "the quick brown fox jumps over the lazy dog. ".repeat(60);
    for (correlation, id, text, truncated) in [
        (10, "over-cap-query", over_cap_text.as_str(), true),
        (11, "under-cap-query", "hello world", false),
    ] {
        let body = route_request(
            &mut consumer,
            route,
            correlation,
            serde_json::json!({
                "method": "embed.query",
                "params": { "model": "test-minilm", "id": id, "text": text }
            }),
        )
        .await;
        let result = &body["result"];
        let vector = &result["vectors"][0];
        let submitted_sha = vector["submitted_sha256"]
            .as_str()
            .expect("submitted_sha256 present");
        let content_sha = vector["content_sha256"]
            .as_str()
            .expect("content_sha256 present");
        assert_eq!(submitted_sha, sha256_hex(text.as_bytes()), "{id}");
        assert_eq!(content_sha != submitted_sha, truncated, "{id}");
        assert_eq!(
            result["truncation_disclosures"][0]["truncated"].as_bool(),
            Some(truncated),
            "{id}"
        );
    }

    let short_text_1 = "short alpha";
    let short_text_2 = "short beta";

    let items = vec![
        serde_json::json!({ "id": "w100", "text": &over_cap_text }),
        serde_json::json!({ "id": "w101", "text": short_text_1 }),
        serde_json::json!({ "id": "w102", "text": short_text_2 }),
    ];

    let body = route_request(
        &mut consumer,
        route,
        12,
        serde_json::json!({
            "method": "embed.batch",
            "params": { "model": "test-minilm", "items": items }
        }),
    )
    .await;

    let result = &body["result"];
    let vectors = result["vectors"].as_array().expect("vectors array");
    assert_eq!(
        vectors.len(),
        3,
        "must return vectors for ALL rows in the batch"
    );

    let v0 = &vectors[0];
    assert_eq!(v0["id"], "w100");
    let v0_submitted_sha = v0["submitted_sha256"].as_str().unwrap();
    let v0_content_sha = v0["content_sha256"].as_str().unwrap();
    assert_eq!(v0_submitted_sha, sha256_hex(over_cap_text.as_bytes()));
    assert_ne!(v0_content_sha, v0_submitted_sha);
    assert_eq!(result["truncation_disclosures"][0]["truncated"], true);

    let v1 = &vectors[1];
    assert_eq!(v1["id"], "w101");
    let v1_submitted_sha = v1["submitted_sha256"].as_str().unwrap();
    let v1_content_sha = v1["content_sha256"].as_str().unwrap();
    assert_eq!(v1_submitted_sha, sha256_hex(short_text_1.as_bytes()));
    assert_eq!(v1_content_sha, v1_submitted_sha);
    assert_eq!(result["truncation_disclosures"][1]["truncated"], false);

    let v2 = &vectors[2];
    assert_eq!(v2["id"], "w102");
    let v2_submitted_sha = v2["submitted_sha256"].as_str().unwrap();
    let v2_content_sha = v2["content_sha256"].as_str().unwrap();
    assert_eq!(v2_submitted_sha, sha256_hex(short_text_2.as_bytes()));
    assert_eq!(v2_content_sha, v2_submitted_sha);
    assert_eq!(result["truncation_disclosures"][2]["truncated"], false);
}

#[tokio::test]
async fn job_tier_paged_results_replays_truncated_row_from_storage() {
    let preloads = deterministic_preload_config();
    let preload_models: Value = serde_json::from_str(&preloads).expect("preload config is json");
    let config = serde_json::json!({
        "preload_models": preload_models,
        "inline": { "max_items": 1 },
        "jobs": {
            "execution_ttl_ms": 60_000,
            "result_retention_ttl_ms": 60_000,
            "result_page_bytes": 4_096,
            "bulk_quantum_tokens": 16
        }
    })
    .to_string();
    let (_daemon, _module, mut consumer, route) = open_route_with_config(&config).await;

    let over_cap_text = "the quick brown fox jumps over the lazy dog. ".repeat(60);
    let short_text = "job tier short";

    let items = vec![
        serde_json::json!({ "id": "job-item-0", "text": &over_cap_text }),
        serde_json::json!({ "id": "job-item-1", "text": short_text }),
    ];

    let accepted = route_request(
        &mut consumer,
        route,
        500,
        serde_json::json!({
            "method": "embed.batch",
            "params": { "model": "test-minilm", "request_key": "job-truncation-e2e", "items": items }
        }),
    )
    .await;
    let job_id = accepted["result"]["job_id"]
        .as_str()
        .expect("job response includes job_id")
        .to_string();

    let done = poll_embed_result(&mut consumer, route, 600, &job_id).await;
    assert_eq!(done["result"]["state"], "done");
    let page_count = done["result"]["page_count"].as_u64().unwrap();
    let mut vectors = done["result"]["vectors"].as_array().unwrap().clone();
    let mut disclosures = done["result"]["truncation_disclosures"]
        .as_array()
        .unwrap()
        .clone();
    for page in 1..page_count {
        let body = route_request(
            &mut consumer,
            route,
            600 + page,
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
        disclosures.extend(
            body["result"]["truncation_disclosures"]
                .as_array()
                .unwrap()
                .iter()
                .cloned(),
        );
    }
    assert_eq!(vectors.len(), 2);

    let v0 = &vectors[0];
    assert_eq!(v0["id"], "job-item-0");
    let v0_submitted_sha = v0["submitted_sha256"].as_str().unwrap();
    let v0_content_sha = v0["content_sha256"].as_str().unwrap();
    assert_eq!(v0_submitted_sha, sha256_hex(over_cap_text.as_bytes()));
    assert_ne!(v0_content_sha, v0_submitted_sha);
    assert_eq!(disclosures[0]["truncated"], true);

    let v1 = &vectors[1];
    assert_eq!(v1["id"], "job-item-1");
    let v1_submitted_sha = v1["submitted_sha256"].as_str().unwrap();
    let v1_content_sha = v1["content_sha256"].as_str().unwrap();
    assert_eq!(v1_submitted_sha, sha256_hex(short_text.as_bytes()));
    assert_eq!(v1_content_sha, v1_submitted_sha);
    assert_eq!(disclosures[1]["truncated"], false);
}

#[tokio::test]
async fn admission_status_reports_execution_waiters_during_concurrent_batches() {
    let preloads = deterministic_preload_config();
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
    let gate_listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .await
        .expect("bind deterministic embed gate listener");
    let gate_addr = gate_listener.local_addr().unwrap().to_string();
    let gate_dir = unique_temp_dir("synapse-admission-execution-gate");
    std::fs::create_dir_all(&gate_dir).expect("create deterministic embed gate directory");
    let gate_dir = gate_dir.to_str().expect("gate path is utf-8");
    let daemon = start_daemon().await;
    let module = spawn_synapse_module_with_env(
        &daemon.connection_file_path,
        None,
        Some(&config),
        &[
            ("SYNAPSE_TEST_DETERMINISTIC_EMBED_GATE_DIR", gate_dir),
            ("SYNAPSE_TEST_DETERMINISTIC_EMBED_GATE_ADDR", &gate_addr),
        ],
    );
    let (daemon, _module, mut first_consumer, first_route) =
        open_route_for_started_module(daemon, module).await;
    std::fs::write(Path::new(gate_dir).join("armed"), b"")
        .expect("arm deterministic embed gate after model setup");

    let mut second_consumer = connect_consumer(&daemon.connection_file_path).await;
    wait_for_catalog(&mut second_consumer, MODULE_ID, SETUP_TIMEOUT).await;
    let second_route = route_open(&mut second_consumer, 2).await;

    let mut status_consumer = connect_consumer(&daemon.connection_file_path).await;
    wait_for_catalog(&mut status_consumer, MODULE_ID, SETUP_TIMEOUT).await;
    let status_route = route_open(&mut status_consumer, 3).await;

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
                "params": { "model": "test-minilm", "items": first_items }
            }),
        )
        .await
    });
    let (mut gate_stream, _) = tokio::time::timeout(SETUP_TIMEOUT, gate_listener.accept())
        .await
        .expect("first batch did not enter the execution gate")
        .expect("accept deterministic embed gate connection");
    let mut gate_signal = [0];
    tokio::time::timeout(SETUP_TIMEOUT, gate_stream.read_exact(&mut gate_signal))
        .await
        .expect("first batch did not signal the execution gate")
        .expect("read deterministic embed gate signal");
    assert_eq!(gate_signal, *b"R");

    let second_task = tokio::spawn(async move {
        route_request(
            &mut second_consumer,
            second_route,
            402,
            serde_json::json!({
                "method": "embed.batch",
                "params": { "model": "test-minilm", "items": second_items }
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
            assert!(
                result["inline_in_flight_executions"].as_u64().unwrap_or(0) >= 1,
                "admission.status reported a waiter without an execution in flight"
            );
            observed_waiter = true;
            break;
        }
        tokio::task::yield_now().await;
    }
    assert!(
        observed_waiter,
        "admission.status never exposed the semaphore waiter"
    );
    gate_stream
        .write_all(&[1])
        .await
        .expect("release deterministic embed execution gate");
    let _ = first_task.await.unwrap();
    let _ = second_task.await.unwrap();
}

#[tokio::test]
async fn quiet_knob_restart_keeps_assignment_but_omitted_model_uses_catalog_default() {
    let preloads: Value = serde_json::from_str(&deterministic_preload_config()).unwrap();
    let config = module_config_with_preloads(preloads.clone(), "balanced");
    let (daemon, mut module, mut consumer, route) = open_route_with_config(&config).await;
    let status = route_request(
        &mut consumer,
        route,
        121,
        serde_json::json!({"method":"admission.status"}),
    )
    .await;
    let hash = status["result"]["machine_profile_hash"].as_str().unwrap();
    let path = expected_store_path(&daemon.data_home);
    let conn = Connection::open(&path).unwrap();
    let inserted = conn.execute("INSERT INTO knob_assignments
        (machine_profile_hash, workload, knob, model_id, numeric_profile_id, fingerprint, engine, measured_at_ms, os_build, module_generation, throughput_tok_s, single_item_latency_p50_ms)
        SELECT ?1, 'embed', 'quiet', model_id, 'test-profile', fingerprint, engine, 1, 'test-os', 1, 1, 1 FROM models WHERE model_id = 'test-minilm'", [hash]).unwrap();
    assert_eq!(inserted, 1);
    drop(conn);
    module.child.start_kill().unwrap();
    assert!(!module.child.wait().await.unwrap().success());
    drop(consumer);
    let restarted = spawn_synapse_module_with_env(
        &daemon.connection_file_path,
        None,
        Some(&module_config_with_preloads(preloads, "quiet")),
        &[("SYNAPSE_TEST_RUNNABLE_BACKENDS", "metal")],
    );
    let (_daemon, _module, mut consumer, route) =
        open_route_for_started_module(daemon, restarted).await;
    let status = route_request(
        &mut consumer,
        route,
        122,
        serde_json::json!({"method":"admission.status"}),
    )
    .await;
    assert_eq!(status["result"]["current_knob"], "quiet");
    let stored: String = Connection::open(path)
        .unwrap()
        .query_row(
            "SELECT model_id FROM knob_assignments WHERE workload = 'embed' AND knob = 'quiet'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(stored, "test-minilm");
    let omitted = route_request(
        &mut consumer,
        route,
        123,
        serde_json::json!({"method":"embed.query", "params":{"text":"hello world"}}),
    )
    .await;
    assert_eq!(omitted["result"]["error"]["code"], "model_not_installed");
    assert_eq!(
        omitted["result"]["error"]["details"]["catalog_id"],
        "gte-modernbert-base"
    );
    let explicit = route_request(
        &mut consumer,
        route,
        124,
        serde_json::json!({"method":"embed.query", "params":{"model":stored,"text":"hello world"}}),
    )
    .await;
    assert_eq!(explicit["result"]["dims"], 384);
}

#[tokio::test]
async fn embed_query_deadline_one_returns_typed_rejection() {
    let preloads = deterministic_preload_config();
    let (_daemon, _module, mut consumer, route) = open_route_with_preloads(Some(&preloads)).await;
    // Serve one request first so the preload has finished: while a model is
    // still loading, an expired deadline correctly answers model_loading,
    // which is not the admission path this test pins.
    let warm = route_request(
        &mut consumer,
        route,
        6,
        serde_json::json!({
            "method": "embed.query",
            "params": { "model": "test-minilm", "text": "warm up" }
        }),
    )
    .await;
    assert!(
        warm["result"]["error"].is_null(),
        "warm-up failed: {warm:?}"
    );

    let body = route_request(
        &mut consumer,
        route,
        7,
        serde_json::json!({
            "method": "embed.query",
            "params": { "model": "test-minilm", "text": "deadline should reject", "deadline_ms": 1 }
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
    let preloads = deterministic_preload_config();
    let (_daemon, _module, mut consumer, route) = open_route_with_preloads(Some(&preloads)).await;
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
                "params": { "model": "test-minilm", "id": format!("burst-{offset}"), "text": format!("burst text {offset}") }
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
async fn profile_preload_lazily_reloads_after_unload() {
    let preloads = deterministic_preload_config();
    let (_daemon, _module, mut consumer, route) = open_route_with_preloads(Some(&preloads)).await;
    let call = |method, params| serde_json::json!({"method": method, "params": params});
    let query = call(
        "embed.query",
        serde_json::json!({"model": "test-minilm", "text": "hello world"}),
    );
    let first = route_request(&mut consumer, route, 1, query.clone()).await;
    assert_eq!(first["result"]["dims"], 384);
    let model = serde_json::json!({"model_id": "test-minilm"});
    let unloaded =
        route_request(&mut consumer, route, 2, call("model.unload", model.clone())).await;
    assert_eq!(unloaded["result"]["state"], "unloaded");
    let status = route_request(&mut consumer, route, 3, call("model.status", model.clone())).await;
    assert_eq!(status["result"]["state"], "unloaded");
    let reloaded = route_request(&mut consumer, route, 4, query).await;
    assert_eq!(reloaded["result"]["vectors"], first["result"]["vectors"]);
    let status = route_request(&mut consumer, route, 5, call("model.status", model)).await;
    assert_eq!(status["result"]["state"], "ready");
}

fn deterministic_source_dir(label: &str) -> PathBuf {
    let root = unique_temp_dir(label);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("model.safetensors"), b"SYNTEST1").unwrap();
    let mut vocab: std::collections::HashMap<String, u32> = [
        ("[UNK]".to_string(), 0),
        ("hello".to_string(), 1),
        ("world".to_string(), 2),
        ("again".to_string(), 3),
    ]
    .into_iter()
    .collect();
    for index in 0..64 {
        vocab.insert(index.to_string(), 9 + index);
    }
    for (index, word) in ["first", "second", "third", "alpha", "beta"]
        .into_iter()
        .enumerate()
    {
        vocab.insert(word.into(), 4 + index as u32);
    }
    let model = WordLevel::builder()
        .vocab(vocab.into_iter().collect())
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer.save(root.join("tokenizer.json"), false).unwrap();
    std::fs::write(root.join("config.json"), "{}").unwrap();
    root
}

fn deterministic_preload_config() -> String {
    deterministic_preload_config_with_max_tokens(512)
}

fn deterministic_preload_config_with_max_tokens(max_tokens: usize) -> String {
    let root = deterministic_source_dir("synapse-deterministic-preload");
    serde_json::json!([{
        "model_id": "test-minilm", "engine": "test-deterministic", "task": "embed",
        "profile": "test-deterministic.fixture",
        "model_path": root.join("model.safetensors"), "tokenizer_path": root.join("tokenizer.json"),
        "format": "test-deterministic", "pooling": "mean", "normalize": true,
        "max_tokens": max_tokens, "quant": "fp32"
    }])
    .to_string()
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

fn module_config_with_preloads(preloads: Value, knob: &str) -> String {
    serde_json::json!({
        "preload_models": preloads,
        "knob": knob,
    })
    .to_string()
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

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest;
    hex::encode(sha2::Sha256::digest(bytes))
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
async fn microllm_refuses_nonempty_grammar_before_resolving_the_model() {
    let (_daemon, _module, mut consumer, route) = open_route_with_config("{}").await;
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
    assert_eq!(
        body["message"],
        "microllm.oneshot constrained decoding is disabled in module config"
    );
}
#[tokio::test]
async fn llama_generate_preload_serves_oneshot_without_decode_provenance() {
    let fixture_dir = unique_temp_dir("llama-oneshot");
    std::fs::create_dir_all(&fixture_dir).unwrap();
    let llama_model = fixture_dir.join("fallback.gguf");
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
        "preload_models": [
            {
                "model_id": "llama-fallback",
                "engine": LLAMA_WORKER_ENGINE,
                "task": "generate",
                "model_path": llama_model,
                "tokenizer_path": tokenizer_path,
                "format": "gguf",
                "max_tokens": 512,
                "worker_bin": synapse_core::dev_binary::ckdev_binary(env!("CARGO_BIN_EXE_synapse-worker-timeout-mock"), &llama_runtime).unwrap(),
                "worker_runtime_dir": llama_runtime
            }
        ]
    })
    .to_string();
    let (_daemon, _module, mut consumer, route) = open_route_with_config(&config).await;

    let deadline = Instant::now() + SETUP_TIMEOUT;
    let mut correlation = 90_006;
    let llama_only = loop {
        let reply = route_request(
            &mut consumer,
            route,
            correlation,
            serde_json::json!({
                "method": "microllm.oneshot",
                "params": {"model": "llama-fallback", "prompt": "hello", "max_tokens": 4}
            }),
        )
        .await;
        if reply["result"]["error"]["code"] != "model_loading" {
            break reply;
        }
        assert!(
            Instant::now() < deadline,
            "llama preload did not load: {reply}"
        );
        correlation += 1;
        sleep(Duration::from_millis(20)).await;
    };
    assert_eq!(llama_only["result"]["text"], "fallback", "{llama_only}");
    let llama_provenance = &llama_only["result"]["provenance"];
    assert!(llama_provenance.get("fallback_reason").is_none());
    assert!(llama_provenance.get("decode_fingerprint").is_none());
    assert!(llama_provenance.get("processing_fingerprint").is_none());
    let _ = std::fs::remove_dir_all(fixture_dir);
}

#[tokio::test]
async fn admission_status_tracks_job_counters_over_wire() {
    let preloads: Value = serde_json::from_str(&deterministic_preload_config()).unwrap();
    let config = serde_json::json!({
        "inline": {"max_items": 1, "max_tokens": 8192},
        "preload_models": preloads
    })
    .to_string();
    let (_daemon, _module, mut consumer, route) = open_route_with_config(&config).await;

    // Snapshot before batch job
    let before_status = route_request(
        &mut consumer,
        route,
        10,
        serde_json::json!({"method": "admission.status"}),
    )
    .await;
    let before = &before_status["result"];
    let before_minted = before["jobs_minted"].as_u64().expect("jobs_minted is u64");
    let before_completed = before["jobs_completed"]
        .as_u64()
        .expect("jobs_completed is u64");
    let before_failed = before["jobs_failed"].as_u64().expect("jobs_failed is u64");
    let before_inherited = before["jobs_inherited"]
        .as_u64()
        .expect("jobs_inherited is u64");
    let before_open = before["jobs_open"].as_u64().expect("jobs_open is u64");
    assert_eq!(
        before_open,
        (before_minted + before_inherited)
            .saturating_sub(before_completed)
            .saturating_sub(before_failed)
    );

    // Submit embed.batch job with items exceeding max_items: 1 so it admits a durable job
    let accepted = route_request(
        &mut consumer,
        route,
        11,
        serde_json::json!({
            "method": "embed.batch",
            "params": {
                "model": "test-minilm",
                "request_key": "wire-telemetry-batch-key",
                "items": [
                    {"id": "item-1", "text": "hello wire world"},
                    {"id": "item-2", "text": "second item for durable job"}
                ]
            }
        }),
    )
    .await;
    let job_id = accepted["result"]["job_id"]
        .as_str()
        .expect("job_id returned");

    // Poll until completed
    let completed = poll_embed_result(&mut consumer, route, 12, job_id).await;
    assert_eq!(completed["result"]["state"], "done");

    // Snapshot after job
    let after_status = route_request(
        &mut consumer,
        route,
        13,
        serde_json::json!({"method": "admission.status"}),
    )
    .await;
    let after = &after_status["result"];
    let minted = after["jobs_minted"].as_u64().expect("jobs_minted is u64");
    let completed_count = after["jobs_completed"]
        .as_u64()
        .expect("jobs_completed is u64");
    let failed_count = after["jobs_failed"].as_u64().expect("jobs_failed is u64");
    let inherited_count = after["jobs_inherited"]
        .as_u64()
        .expect("jobs_inherited is u64");
    let open_count = after["jobs_open"].as_u64().expect("jobs_open is u64");

    assert_eq!(minted, before_minted + 1);
    assert_eq!(completed_count, before_completed + 1);
    assert_eq!(failed_count, before_failed);
    assert_eq!(inherited_count, before_inherited);
    // Identity holds
    assert_eq!(
        open_count,
        (minted + inherited_count)
            .saturating_sub(completed_count)
            .saturating_sub(failed_count)
    );
    assert_eq!(open_count, 0);
}
