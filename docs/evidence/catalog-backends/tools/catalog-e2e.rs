//! Scratch driver (not part of the synapse repo) that exercises the catalog
//! path of a release `ck-synapse` on real hardware: download, load by lane id
//! with self-check, serve one batch, and compare `models.list` with the pin.
//! The daemon/route plumbing mirrors synapse-certify-runner's live session.
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
fn log(step: &str, v: &Value) { println!("{}", json!({"step": step, "value": v})); }

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (assets, checkout, catalog_id, backend, root) = (PathBuf::from(&args[1]), PathBuf::from(&args[2]), args[3].clone(), args[4].clone(), PathBuf::from(&args[5]));
    let lane = format!("{catalog_id}-{backend}");
    let catalog: Value = serde_json::from_slice(&std::fs::read(checkout.join("crates/synapse-module/src/catalog/models.json")).unwrap()).unwrap();
    let entry = catalog["models"].as_array().unwrap().iter().find(|m| m["id"] == catalog_id.as_str()).unwrap().clone();
    let pin = entry["backends"].as_array().unwrap().iter().find(|b| b["backend"] == backend.as_str()).unwrap()["fingerprint"].clone();
    let task = entry["task"].as_str().unwrap().to_string();
    std::fs::create_dir_all(root.join("data")).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let conn = ConnectionInfo { schema: SCHEMA_VERSION, endpoints: vec![Endpoint { host: "127.0.0.1".into(), port: listener.local_addr().unwrap().port() }], key: generate_key().unwrap(), daemon_id: generate_daemon_id().unwrap(), pid: std::process::id(), daemon_ver: "catalog-e2e".into(), wire_version: Some(subc_protocol::PROTOCOL_VERSION) };
    let conn_path = root.join("connection.json");
    write_atomic(&conn_path, &conn).unwrap();
    let control = ControlHandler::new(Arc::new(Registry::default())).with_storage_config(Some(StorageConfig::Sqlite { data_home: root.join("data") }));
    let router = Arc::new(Router::with_control_handler(Arc::new(control)));
    let auth = ServerAuth::new(conn.key.clone(), conn.daemon_id, conn.daemon_ver.clone());
    tokio::spawn(async move { let _ = serve_listener(listener, router, auth).await; });
    let config = root.join("config.json");
    std::fs::write(&config, b"{}").unwrap();
    let mut child = tokio::process::Command::new(assets.join("ck-synapse"))
        .arg("--subc").arg(&conn_path)
        .env("SUBC_MODULE_ID", "synapse").env("SYNAPSE_CONFIG_PATH", &config)
        .env("XDG_DATA_HOME", root.join("data")).env("CORTEXKIT_LEASE_ROOT", root.join("leases"))
        .env("CORTEXKIT_STORE_ROOT", root.join("store")).env("CORTEXKIT_MODEL_CACHE", root.join("model-cache"))
        .stderr(std::fs::File::create(root.join("ck-synapse.stderr.log")).unwrap())
        .kill_on_drop(true).spawn().unwrap();
    let mut stream = TcpStream::connect(("127.0.0.1", conn.endpoints[0].port)).await.unwrap();
    authenticate_client(&mut stream, &conn, Duration::from_secs(10)).await.unwrap();
    let mut s = S { stream, channel: 0, epoch: 0, corr: 1 };
    for attempt in 0..600 {
        let r = rpc(&mut s, 0, 0, json!({"op": "route.open", "target": RouteTarget::ManagementSurface { module_id: "synapse".into() }, "identity": BindIdentity::new(root.clone(), "catalog-e2e".to_string(), "catalog-e2e".to_string())})).await;
        if let (Some(c), Some(e)) = (r["route_channel"].as_u64(), r["route_epoch"].as_u64()) { s.channel = c as u16; s.epoch = e as u32; break; }
        assert!(child.try_wait().unwrap().is_none(), "ck-synapse exited: {r}");
        assert!(attempt < 599, "registration timed out: {r}");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let cat = call(&mut s, "models.catalog", json!({"query": catalog_id})).await;
    log("models.catalog", &cat);
    let accepted = call(&mut s, "models.download", json!({"catalog_id": catalog_id, "request_key": "e2e"})).await;
    log("models.download", &accepted);
    let job = accepted["job_id"].as_str().expect("job id").to_string();
    let started = Instant::now();
    let done = loop {
        let st = call(&mut s, "model.status", json!({"job_id": job})).await;
        if matches!(st["state"].as_str(), Some("committed" | "failed")) { break st; }
        assert!(started.elapsed() < Duration::from_secs(1800), "download stuck: {st}");
        tokio::time::sleep(Duration::from_millis(500)).await;
    };
    log("download.done", &json!({"status": done, "seconds": started.elapsed().as_secs_f64()}));
    log("models.list.before", &call(&mut s, "models.list", json!({})).await);
    let (method, params) = if task == "embed" {
        ("embed.batch", json!({"model": lane, "items": [
            {"id": "a", "text": "The quick brown fox jumps over the lazy dog."},
            {"id": "b", "text": "Rust is a systems programming language focused on safety."},
            {"id": "c", "text": "Vulkan and CUDA are GPU programming interfaces."},
            {"id": "d", "text": "Paris is the capital of France."}]}))
    } else {
        ("rerank.score", json!({"model": lane, "query": "What is the capital of France?", "candidates": [
            "Paris is the capital of France.", "Berlin is the capital of Germany.",
            "The quick brown fox jumps over the lazy dog.", "France is a country in Western Europe."]}))
    };
    let cold = Instant::now();
    let first = loop {
        let r = call(&mut s, method, params.clone()).await;
        if r["error"]["code"] != "model_loading" { break r; }
        assert!(cold.elapsed() < Duration::from_secs(600), "load stuck: {r}");
        tokio::time::sleep(Duration::from_millis(r["error"]["retry_after_ms"].as_u64().unwrap_or(250))).await;
    };
    let cold_ms = cold.elapsed().as_secs_f64() * 1e3;
    let mut short = first.clone();
    if let Some(v) = short.get_mut("vectors").and_then(Value::as_array_mut) { for item in v { if let Some(x) = item.get_mut("vector").and_then(Value::as_array_mut) { x.truncate(4); } } }
    log("serve.first", &json!({"method": method, "cold_ms_including_load": cold_ms, "response_vectors_truncated_to_4": short}));
    let mut warm = Vec::new();
    for _ in 0..5 { let t = Instant::now(); let r = call(&mut s, method, params.clone()).await; assert!(r.get("error").is_none(), "{r}"); warm.push(t.elapsed().as_secs_f64() * 1e3); }
    log("serve.warm_ms", &json!(warm));
    let listed = call(&mut s, "models.list", json!({})).await;
    log("models.list.after", &listed);
    let row = listed["models"].as_array().unwrap().iter().find(|m| m["model_id"] == lane.as_str() || m["lane_id"] == lane.as_str()).cloned().unwrap_or(Value::Null);
    let listed_fp = row["fingerprints"].clone();
    log("pin_check", &json!({"lane": lane, "pinned_fingerprint": pin, "listed_fingerprint": listed_fp, "served_fingerprint": first["fingerprint"], "match": listed_fp == json!([pin.clone()]) && first["fingerprint"] == pin, "self_check": row["self_check"], "certified": row["certified"]}));
    let _ = child.start_kill();
}
