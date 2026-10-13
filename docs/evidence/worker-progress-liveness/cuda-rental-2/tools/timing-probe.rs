//! Scratch timing driver (not part of the synapse repo). It starts the release
//! `ck-synapse` the way ckdev-synapse-certify does (private in-process subc
//! daemon, the runner's config.json, isolated roots), downloads the catalog
//! model from a local byte-pinned mirror, loads the lane through a short
//! request, then times single requests on fixture rows of a given length.
//! Requests use the runner's exact parameters: embed.query {text} or
//! rerank.score {query, candidates:[document]}, accept_declared, model = lane
//! id, deadline_ms 3600000.
//! Usage: timing-probe <ck-synapse> <worker> <fixture.json> <catalog_id> <scratch> <hf_endpoint> <repeats> <backend> <len>...
use serde_json::{json, Value};
use std::{path::PathBuf, sync::Arc, time::{Duration, Instant}};
use subc_daemon::{daemon_config::StorageConfig, serve_listener, ControlHandler, Registry, Router, ServerAuth};
use subc_protocol::{BindIdentity, Flags, Frame, FrameType, Priority, RouteTarget};
use subc_transport::{authenticate_client, generate_daemon_id, generate_key, read_frame, write_atomic, write_frame, ConnectionInfo, Endpoint, SCHEMA_VERSION};
use tokio::net::{TcpListener, TcpStream};

struct S { stream: TcpStream, channel: u16, epoch: u32, corr: u64 }

async fn rpc(s: &mut S, channel: u16, epoch: u32, body: Value) -> Value {
    s.corr += 1;
    let corr = s.corr;
    let frame = Frame::build(FrameType::Request, Flags::new(false, Priority::Passive, false), channel, epoch, corr, serde_json::to_vec(&body).unwrap()).unwrap();
    write_frame(&mut s.stream, &frame).await.unwrap();
    loop {
        let frame = read_frame(&mut s.stream).await.unwrap().expect("connection closed");
        if frame.header.channel == channel && frame.header.corr == corr && matches!(frame.header.ty, FrameType::Response | FrameType::Error) {
            let v: Value = serde_json::from_slice(&frame.body).unwrap();
            return if frame.header.ty == FrameType::Error { json!({"error": v}) } else { v };
        }
    }
}
async fn call(s: &mut S, method: &str, params: Value) -> Value {
    let (c, e) = (s.channel, s.epoch);
    let v = rpc(s, c, e, json!({"method": method, "params": params})).await;
    v.get("result").cloned().unwrap_or(v)
}
fn log(step: &str, v: Value) { println!("{}", json!({"step": step, "value": v})); }

fn params(case: &Value, task: &str, lane: &str) -> (&'static str, Value) {
    if task == "embed" {
        ("embed.query", json!({"text": case["text"], "accept_declared": true, "model": lane, "deadline_ms": 3_600_000u64}))
    } else {
        ("rerank.score", json!({"query": case["query"], "candidates": [case["document"]], "accept_declared": true, "model": lane, "deadline_ms": 3_600_000u64}))
    }
}

#[tokio::main]
async fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (candidate, worker, fixture, catalog_id, root, endpoint) = (PathBuf::from(&a[1]), PathBuf::from(&a[2]), PathBuf::from(&a[3]), a[4].clone(), PathBuf::from(&a[5]), a[6].clone());
    let repeats: usize = a[7].parse().unwrap();
    let backend = &a[8];
    assert!(matches!(backend.as_str(), "cuda" | "vulkan"));
    let lengths: Vec<usize> = a[9..].iter().map(|x| x.parse().unwrap()).collect();
    let lane = format!("{catalog_id}-{backend}");
    let worker_prefix = format!("SYNAPSE_OWNED_{}", backend.to_uppercase());
    let doc: Value = serde_json::from_slice(&std::fs::read(&fixture).unwrap()).unwrap();
    let cases = doc["cases"].as_array().unwrap().clone();
    std::fs::create_dir_all(root.join("data")).unwrap();
    std::fs::create_dir_all(root.join("ipc")).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let conn = ConnectionInfo { schema: SCHEMA_VERSION, endpoints: vec![Endpoint { host: "127.0.0.1".into(), port: listener.local_addr().unwrap().port() }], key: generate_key().unwrap(), daemon_id: generate_daemon_id().unwrap(), pid: std::process::id(), daemon_ver: "timing-probe".into(), wire_version: Some(subc_protocol::PROTOCOL_VERSION) };
    let conn_path = root.join("connection.json");
    write_atomic(&conn_path, &conn).unwrap();
    let control = ControlHandler::new(Arc::new(Registry::default())).with_storage_config(Some(StorageConfig::Sqlite { data_home: root.join("data") }));
    let router = Arc::new(Router::with_control_handler(Arc::new(control)));
    let auth = ServerAuth::new(conn.key.clone(), conn.daemon_id, conn.daemon_ver.clone());
    tokio::spawn(async move { let _ = serve_listener(listener, router, auth).await; });
    let config = root.join("config.json");
    // Matches ckdev-synapse-certify's endpoint, observation flag and inline
    // limits in crates/synapse-certify-runner/src/assets.rs::config.
    std::fs::write(&config, serde_json::to_vec(&json!({"hf_endpoint": endpoint, "certify_observation": true,
        "inline": {"deadline_ms": 3600000, "max_queue_ms": 3600000, "max_items": 64, "max_tokens": 8192}})).unwrap()).unwrap();
    let mut child = tokio::process::Command::new(&candidate)
        .current_dir(&root)
        .arg("--subc").arg(&conn_path)
        .env("SUBC_MODULE_ID", "synapse").env("SYNAPSE_CONFIG_PATH", &config)
        .env("XDG_DATA_HOME", root.join("data")).env("CORTEXKIT_LEASE_ROOT", root.join("leases"))
        .env("CORTEXKIT_STORE_ROOT", root.join("store")).env("CORTEXKIT_MODEL_CACHE", root.join("model-cache"))
        .env(format!("{worker_prefix}_WORKER_BIN"), &worker).env(format!("{worker_prefix}_WORKER_RUNTIME_DIR"), "ipc")
        .stderr(std::fs::File::create(root.join("ck-synapse.stderr.log")).unwrap())
        .kill_on_drop(true).spawn().unwrap();
    let mut stream = TcpStream::connect(("127.0.0.1", conn.endpoints[0].port)).await.unwrap();
    authenticate_client(&mut stream, &conn, Duration::from_secs(10)).await.unwrap();
    let mut s = S { stream, channel: 0, epoch: 0, corr: 1 };
    for attempt in 0..600 {
        let r = rpc(&mut s, 0, 0, json!({"op": "route.open", "target": RouteTarget::ManagementSurface { module_id: "synapse".into() }, "identity": BindIdentity::new(root.clone(), "timing-probe".to_string(), "timing-probe".to_string())})).await;
        if let (Some(c), Some(e)) = (r["route_channel"].as_u64(), r["route_epoch"].as_u64()) { s.channel = c as u16; s.epoch = e as u32; break; }
        assert!(child.try_wait().unwrap().is_none(), "ck-synapse exited: {r}");
        assert!(attempt < 599, "registration timed out: {r}");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let accepted = call(&mut s, "models.download", json!({"catalog_id": catalog_id})).await;
    let job = accepted["job_id"].as_str().unwrap_or_else(|| panic!("download refused: {accepted}")).to_string();
    let started = Instant::now();
    loop {
        let st = call(&mut s, "model.status", json!({"job_id": job})).await;
        match st["state"].as_str() {
            Some("committed") => break,
            Some("queued" | "downloading" | "verifying") => {}
            _ => panic!("download: {st}"),
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    log("download", json!({"seconds": started.elapsed().as_secs_f64()}));
    let task = if cases[0].get("text").is_some() { "embed" } else { "rerank" };
    // Load and self-check the lane with the shortest fixture row.
    let shortest = cases.iter().min_by_key(|c| c["input_ids"].as_array().unwrap().len()).unwrap();
    let (method, p) = params(shortest, task, &lane);
    let cold = Instant::now();
    loop {
        let r = call(&mut s, method, p.clone()).await;
        if r["error"]["code"] == "model_loading" { tokio::time::sleep(Duration::from_millis(r["error"]["retry_after_ms"].as_u64().unwrap_or(100).max(1))).await; continue; }
        log("load", json!({"cold_ms_including_load": cold.elapsed().as_secs_f64() * 1e3, "fingerprint": r["fingerprint"], "error": r.get("error")}));
        break;
    }
    for len in lengths {
        let case = cases.iter().find(|c| c["input_ids"].as_array().unwrap().len() == len).unwrap_or_else(|| panic!("no {len}-token row"));
        let (method, p) = params(case, task, &lane);
        for i in 0..repeats {
            let t = Instant::now();
            let r = call(&mut s, method, p.clone()).await;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let tokens = r["usage"].clone();
            log("row", json!({"case": case["id"], "tokens": len, "repeat": i, "elapsed_ms": ms, "ok": r.get("error").is_none(), "error": r.get("error"), "usage": tokens}));
        }
    }
    let _ = child.start_kill();
    let _ = child.wait().await;
}
