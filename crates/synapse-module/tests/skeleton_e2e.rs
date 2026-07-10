#![forbid(unsafe_code)]

mod common;

use std::{
    net::Ipv4Addr,
    path::{Path, PathBuf},
    process,
    time::Duration,
};

use common::{
    connect_consumer, raw_route_frame, read_frame_timeout, route_open, route_request,
    unique_temp_dir, wait_for_catalog, MODULE_ID, SETUP_TIMEOUT,
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
    open_route_for_started_module(daemon, module).await
}

async fn open_route_with_config(
    config_json: &str,
) -> (TestDaemon, ModuleProcess, tokio::net::TcpStream, u16) {
    let daemon = start_daemon().await;
    let module = spawn_synapse_module_with_config(&daemon.connection_file_path, config_json);
    open_route_for_started_module(daemon, module).await
}

async fn open_route_for_started_module(
    daemon: TestDaemon,
    module: ModuleProcess,
) -> (TestDaemon, ModuleProcess, tokio::net::TcpStream, u16) {
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
    certify_preloaded_models(&mut consumer, route_channel, 40).await;

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
    certify_preloaded_models(&mut consumer, route_channel, 60).await;
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
            "ttl_ms": 60_000,
            "result_page_bytes": 4_096,
            "bulk_quantum_tokens": 16
        }
    })
    .to_string();
    let _lock = acquire_minilm_e2e_lock();
    let (_daemon, _module, mut consumer, route_channel) = open_route_with_config(&config).await;
    certify_preloaded_models(&mut consumer, route_channel, 80).await;
    let texts = [
        "job tier first text",
        "job tier second text",
        "job tier third text",
    ];

    let mut inline_vectors = Vec::new();
    for (index, text) in texts.iter().enumerate() {
        let body = route_request(
            &mut consumer,
            route_channel,
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
        route_channel,
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

    let duplicate = route_request(
        &mut consumer,
        route_channel,
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

    let done = poll_embed_result(&mut consumer, route_channel, 300, &job_id).await;
    assert_eq!(done["result"]["state"], "done");
    let page_count = done["result"]["page_count"].as_u64().unwrap();
    assert!(page_count >= 1);
    let mut vectors = done["result"]["vectors"].as_array().unwrap().clone();
    for page in 1..page_count {
        let body = route_request(
            &mut consumer,
            route_channel,
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
async fn alias_surface_certifies_declares_retracts_and_preserves_old_job_pages() {
    let Some(preloads) = minilm_alias_preload_config() else {
        eprintln!("skipping MiniLM alias e2e: local HF ONNX snapshot is missing");
        return;
    };
    let config = serde_json::json!({
        "preload_models": preloads,
        "inline": { "max_items": 1 },
        "jobs": { "ttl_ms": 60_000, "result_page_bytes": 4096, "bulk_quantum_tokens": 2048 },
        "alias_admin_enabled": true
    })
    .to_string();
    let _lock = acquire_minilm_e2e_lock();
    let (_daemon, _module, mut consumer, route_channel) = open_route_with_config(&config).await;
    let probe = certify_preloaded_models(&mut consumer, route_channel, 120).await;
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
        route_channel,
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
        route_channel,
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
        route_channel,
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
    let done = poll_embed_result(&mut consumer, route_channel, 503, &job_id).await;
    assert_eq!(done["result"]["equivalent_to"][0], fingerprint_b);

    let retracted = route_request(
        &mut consumer,
        route_channel,
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
        route_channel,
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
        route_channel,
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
        route_channel,
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
    let (_daemon, _module, mut consumer, route_channel) =
        open_route_with_preloads(Some(&preloads)).await;

    let before = route_request(
        &mut consumer,
        route_channel,
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
    assert_eq!(before["result"]["lanes"][0]["performance"], Value::Null);

    let probed = certify_preloaded_models(&mut consumer, route_channel, 86).await;
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
        route_channel,
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
    let (daemon, mut module, mut consumer, route_channel) = open_route_with_config(&config).await;
    certify_preloaded_models(&mut consumer, route_channel, 120).await;

    let report = route_request(
        &mut consumer,
        route_channel,
        121,
        serde_json::json!({ "method": "probe.report", "params": {} }),
    )
    .await;
    let lanes = report["result"]["lanes"].as_array().unwrap();
    assert_eq!(lanes.len(), 2);
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
    let (_daemon, _module, mut consumer, route_channel) =
        open_route_for_started_module(daemon, restarted).await;

    let quiet_report = route_request(
        &mut consumer,
        route_channel,
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
        route_channel,
        123,
        serde_json::json!({
            "method": "embed.query",
            "params": { "id": "quiet-knob", "text": "quiet knob uses the persisted assignment" }
        }),
    )
    .await;
    if first["result"]["error"]["code"] == "model_loading" {
        let ready = poll_model_ready(&mut consumer, route_channel, 124, &quiet_model_id).await;
        assert_eq!(ready["result"]["state"], "ready");
    }
    let routed = route_request(
        &mut consumer,
        route_channel,
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
    let (daemon, mut module, mut consumer, route_channel) =
        open_route_with_preloads(Some(&preloads)).await;
    certify_preloaded_models(&mut consumer, route_channel, 130).await;

    let _ = module.child.start_kill();
    let _ = module.child.wait().await;
    drop(consumer);

    let restarted = spawn_synapse_module_with_env(
        &daemon.connection_file_path,
        Some(&preloads),
        None,
        &[("SYNAPSE_OS_BUILD_OVERRIDE", "synthetic-stale-build")],
    );
    let (_daemon, _module, mut consumer, route_channel) =
        open_route_for_started_module(daemon, restarted).await;

    let report = route_request(
        &mut consumer,
        route_channel,
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

    ensure_model_loaded_by_query(&mut consumer, route_channel, 132, "minilm").await;
    let status = route_request(
        &mut consumer,
        route_channel,
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
    let (_daemon, _module, mut consumer, route_channel) =
        open_route_with_preloads(Some(&preloads)).await;
    certify_preloaded_models(&mut consumer, route_channel, 90).await;

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
    certify_preloaded_models(&mut consumer, route_channel, 110).await;
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

#[tokio::test]
async fn model_load_file_source_reaches_ready_and_lazy_reload_after_unload() {
    let Some(source_dir) = copied_minilm_source_dir("synapse-model-load-file") else {
        eprintln!("skipping model.load file e2e: local HF ONNX snapshot is missing");
        return;
    };
    let _lock = acquire_minilm_e2e_lock();
    let (_daemon, _module, mut consumer, route_channel) = open_route().await;

    let accepted = route_request(
        &mut consumer,
        route_channel,
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
                "pin": true,
                "request_key": "model-load-file-e2e"
            }
        }),
    )
    .await;
    let job_id = accepted["result"]["job_id"].as_str().unwrap().to_string();
    let ready = poll_model_load_job(&mut consumer, route_channel, 20_001, &job_id).await;
    assert_eq!(ready["result"]["state"], "ready");
    let model_id = ready["result"]["model_id"].as_str().unwrap().to_string();

    let uncertified = route_request(
        &mut consumer,
        route_channel,
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
        route_channel,
        20_200,
        serde_json::json!({ "models": [model_id.clone()] }),
    )
    .await;

    let first = route_request(
        &mut consumer,
        route_channel,
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
        route_channel,
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
        route_channel,
        20_500,
        serde_json::json!({
            "method": "embed.query",
            "params": { "model": &model_id, "text": "reload after unload" }
        }),
    )
    .await;
    assert_eq!(loading["result"]["error"]["code"], "model_loading");

    let ready_again = poll_model_ready(&mut consumer, route_channel, 20_600, &model_id).await;
    assert_eq!(ready_again["result"]["state"], "ready");

    let second = route_request(
        &mut consumer,
        route_channel,
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
    let (_daemon, _module, mut consumer, route_channel) = open_route().await;

    let accepted = route_request(
        &mut consumer,
        route_channel,
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
    let failed = poll_model_load_job(&mut consumer, route_channel, 21_001, &job_id).await;
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
    let (daemon, mut module, mut consumer, route_channel) =
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
    let accepted = route_request(&mut consumer, route_channel, 22_000, request.clone()).await;
    let job_id = accepted["result"]["job_id"].as_str().unwrap().to_string();

    let deadline = Instant::now() + Duration::from_secs(30);
    let mut corr = 22_001;
    loop {
        let body = route_request(
            &mut consumer,
            route_channel,
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
    let (_daemon, _module, mut consumer, route_channel) =
        open_route_for_started_module(daemon, restarted).await;

    let restarted_status = route_request(
        &mut consumer,
        route_channel,
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

    let retried = route_request(&mut consumer, route_channel, 22_200, request).await;
    let retried_job_id = retried["result"]["job_id"].as_str().unwrap().to_string();
    assert_ne!(retried_job_id, job_id);
    let ready = poll_model_load_job(&mut consumer, route_channel, 22_201, &retried_job_id).await;
    assert_eq!(ready["result"]["state"], "ready");
}

async fn certify_preloaded_models(
    consumer: &mut tokio::net::TcpStream,
    route_channel: u16,
    start_corr: u64,
) -> Value {
    run_probe_job(consumer, route_channel, start_corr, serde_json::json!({})).await
}

async fn run_probe_job(
    consumer: &mut tokio::net::TcpStream,
    route_channel: u16,
    start_corr: u64,
    params: Value,
) -> Value {
    let accepted = route_request(
        consumer,
        route_channel,
        start_corr,
        serde_json::json!({ "method": "probe.start", "params": params }),
    )
    .await;
    let job_id = accepted["result"]["job_id"]
        .as_str()
        .expect("probe.start returns job_id")
        .to_string();
    let deadline = Instant::now() + Duration::from_secs(180);
    let mut corr = start_corr + 1;
    loop {
        let body = route_request(
            consumer,
            route_channel,
            corr,
            serde_json::json!({
                "method": "probe.status",
                "params": { "job_id": &job_id }
            }),
        )
        .await;
        match body["result"]["state"].as_str() {
            Some("done") => {
                let lanes = body["result"]["lanes"].as_array().expect("probe lanes");
                assert!(!lanes.is_empty(), "probe should certify at least one lane");
                for lane in lanes {
                    assert_eq!(lane["status"], "certified", "probe lane failed: {lane:?}");
                }
                return body;
            }
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

async fn poll_model_load_job(
    consumer: &mut tokio::net::TcpStream,
    route_channel: u16,
    start_corr: u64,
    job_id: &str,
) -> Value {
    let deadline = Instant::now() + Duration::from_secs(180);
    let mut corr = start_corr;
    loop {
        let body = route_request(
            consumer,
            route_channel,
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
    route_channel: u16,
    start_corr: u64,
    model_id: &str,
) -> Value {
    let deadline = Instant::now() + Duration::from_secs(180);
    let mut corr = start_corr;
    loop {
        let body = route_request(
            consumer,
            route_channel,
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
    route_channel: u16,
    start_corr: u64,
    job_id: &str,
) -> Value {
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut corr = start_corr;
    loop {
        let body = route_request(
            consumer,
            route_channel,
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
    route_channel: u16,
    start_corr: u64,
    model_id: &str,
) {
    let body = route_request(
        consumer,
        route_channel,
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
            let ready = poll_model_ready(consumer, route_channel, start_corr + 1, model_id).await;
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
async fn microllm_grammar_rejects_when_gate_disabled() {
    let config = serde_json::json!({ "grammar_enabled": false }).to_string();
    let (_daemon, _module, mut consumer, route_channel) = open_route_with_config(&config).await;
    let frame = raw_route_frame(
        &mut consumer,
        route_channel,
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
    assert_eq!(body["code"], "invalid_request");
    assert!(body["message"]
        .as_str()
        .unwrap_or_default()
        .contains("grammar_enabled"));
}

#[tokio::test]
async fn microllm_grammar_unavailable_in_build_when_gate_enabled() {
    let config = serde_json::json!({ "grammar_enabled": true }).to_string();
    let (_daemon, _module, mut consumer, route_channel) = open_route_with_config(&config).await;
    let frame = raw_route_frame(
        &mut consumer,
        route_channel,
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
    assert_eq!(body["code"], "invalid_request");
    assert_eq!(body["message"], "grammar_unavailable_in_build");
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
