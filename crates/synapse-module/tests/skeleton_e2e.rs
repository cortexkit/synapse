#![forbid(unsafe_code)]

mod common;

use std::{net::Ipv4Addr, path::Path, path::PathBuf, process, time::Duration};

use common::{
    connect_consumer, raw_route_frame, route_open, route_request, unique_temp_dir, wait_for_catalog,
    MODULE_ID, SETUP_TIMEOUT,
};
use rusqlite::Connection;
use serde_json::Value;
use subc_core::{
    daemon_config::StorageConfig, serve_listener, ControlHandler, Registry, Router, ServerAuth,
};
use subc_protocol::FrameType;
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
    let mut command = Command::new(env!("CARGO_BIN_EXE_synapse-module"));
    command
        .arg("--subc")
        .arg(subc_connection_file)
        .env("SUBC_MODULE_ID", MODULE_ID)
        .stderr(process::Stdio::inherit())
        .kill_on_drop(true);
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
    let daemon = start_daemon().await;
    let module = spawn_synapse_module(&daemon.connection_file_path);
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
    assert!(store_path.exists(), "synapse should open the HELLO_ACK store at {}", store_path.display());

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
