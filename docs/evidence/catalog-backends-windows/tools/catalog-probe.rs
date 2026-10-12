//! Scratch diagnostics (not part of the synapse repo). Repeats
//! ckdev-synapse-certify's catalog session for one Vulkan cell (private subc
//! daemon, the runner's config.json, download from the byte-pinned mirror,
//! requests by catalog lane id with the runner's grouping and parameters), but
//! keeps going after a failed request so every case is reported. Prints each
//! request's wall time, per-case error against the fp32 fixture, the
//! evaluator's output over what completed, and the 8193-token admission check
//! with the worker request counter around it.
//! Usage: catalog-probe <ck-synapse.exe> <worker.exe> <checkout> <catalog_id> <scratch> <hf_endpoint>
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::PathBuf, sync::Arc, time::{Duration, Instant}};
use synapse_parity::evaluator::{evaluate, FixtureSet, ObservedCase, Output};
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


const INLINE_ITEMS: usize = 64;
const INLINE_TOKENS: usize = 8192;
fn fixture_requests<'a>(cases: &'a [Value], operation: &str) -> Vec<Vec<&'a Value>> {
    let mut groups: Vec<Vec<&Value>> = Vec::new();
    let mut grouped = BTreeMap::new();
    for case in cases {
        let count = case["input_ids"].as_array().unwrap().len();
        let key = if count == INLINE_TOKENS { None } else if case["category"] == "batched" { Some("batched".to_string()) } else if operation == "rerank" { case["pool"].as_str().map(str::to_string) } else { None };
        if let Some(key) = key {
            let index = *grouped.entry(key).or_insert_with(|| { groups.push(Vec::new()); groups.len() - 1 });
            groups[index].push(case);
        } else { groups.push(vec![case]); }
    }
    let mut requests = Vec::new();
    for group in groups {
        let counts = group.iter().map(|c| c["input_ids"].as_array().unwrap().len()).collect::<Vec<_>>();
        for range in synapse_parity::evaluator::split_pool(&counts, INLINE_ITEMS, INLINE_TOKENS).unwrap() { requests.push(group[range].to_vec()); }
    }
    requests
}
fn group_params(group: &[&Value], operation: &str, lane: &str) -> (&'static str, Value) {
    let (m, mut p) = if operation == "embed" {
        if group.len() == 1 { ("embed.query", json!({"text": group[0]["text"], "accept_declared": true})) } else {
            let items = group.iter().map(|c| json!({"id": c["id"], "text": c["text"]})).collect::<Vec<_>>();
            ("embed.batch", json!({"items": items, "accept_declared": true}))
        }
    } else {
        let candidates = group.iter().map(|c| &c["document"]).collect::<Vec<_>>();
        ("rerank.score", json!({"query": group[0]["query"], "candidates": candidates, "accept_declared": true}))
    };
    p["model"] = json!(lane);
    p["deadline_ms"] = json!(3_600_000u64);
    (m, p)
}
fn sent(snapshot: &Value) -> u64 {
    snapshot["worker_requests"].as_object().map(|m| m.values().filter_map(Value::as_u64).sum()).unwrap_or(0)
}
fn log(step: &str, v: Value) { println!("{}", json!({"step": step, "value": v})); }

#[tokio::main]
async fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (candidate, worker, checkout, model, root, endpoint) = (PathBuf::from(&a[1]), PathBuf::from(&a[2]), PathBuf::from(&a[3]), a[4].clone(), PathBuf::from(&a[5]), a[6].clone());
    let lane = format!("{model}-vulkan");
    let manifest = synapse_certify::self_check::manifest().unwrap();
    let parity_root = checkout.join("bench/parity");
    let index: Value = serde_json::from_slice(&std::fs::read(parity_root.join("fixtures/index.json")).unwrap()).unwrap();
    let fixture_id = synapse_parity::evaluator::fixture_set_id(&manifest, &model);
    let bytes = std::fs::read(parity_root.join(index[&fixture_id]["path"].as_str().unwrap())).unwrap();
    let fixtures = FixtureSet::from_bytes(&manifest, &model, &bytes, index[&fixture_id]["sha256"].as_str().unwrap()).unwrap();
    let document: Value = serde_json::from_slice(&bytes).unwrap();
    let cases = document["cases"].as_array().unwrap().clone();
    let operation = if cases[0].get("text").is_some() { "embed" } else { "rerank" };
    let profile_id = format!("{model}.owned-vulkan");
    std::fs::create_dir_all(root.join("data")).unwrap();
    std::fs::create_dir_all(root.join("ipc")).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let conn = ConnectionInfo { schema: SCHEMA_VERSION, endpoints: vec![Endpoint { host: "127.0.0.1".into(), port: listener.local_addr().unwrap().port() }], key: generate_key().unwrap(), daemon_id: generate_daemon_id().unwrap(), pid: std::process::id(), daemon_ver: "catalog-probe".into(), wire_version: Some(subc_protocol::PROTOCOL_VERSION) };
    let conn_path = root.join("connection.json");
    write_atomic(&conn_path, &conn).unwrap();
    let control = ControlHandler::new(Arc::new(Registry::default())).with_storage_config(Some(StorageConfig::Sqlite { data_home: root.join("data") }));
    let router = Arc::new(Router::with_control_handler(Arc::new(control)));
    let auth = ServerAuth::new(conn.key.clone(), conn.daemon_id, conn.daemon_ver.clone());
    tokio::spawn(async move { let _ = serve_listener(listener, router, auth).await; });
    let config = root.join("config.json");
    // Identical to the certification runner's config (assets.rs `config`).
    std::fs::write(&config, serde_json::to_vec(&json!({"hf_endpoint": endpoint, "certify_observation": true,
        "inline": {"deadline_ms": 3600000, "max_queue_ms": 3600000, "max_items": 64, "max_tokens": 8192}})).unwrap()).unwrap();
    let mut child = tokio::process::Command::new(&candidate)
        .current_dir(&root)
        .arg("--subc").arg(&conn_path)
        .env("SUBC_MODULE_ID", "synapse").env("SYNAPSE_CONFIG_PATH", &config)
        .env("XDG_DATA_HOME", root.join("data")).env("CORTEXKIT_LEASE_ROOT", root.join("leases"))
        .env("CORTEXKIT_STORE_ROOT", root.join("store")).env("CORTEXKIT_MODEL_CACHE", root.join("model-cache"))
        .env("SYNAPSE_OWNED_VULKAN_WORKER_BIN", &worker).env("SYNAPSE_OWNED_VULKAN_WORKER_RUNTIME_DIR", "ipc")
        .stderr(std::fs::File::create(root.join("ck-synapse.stderr.log")).unwrap())
        .kill_on_drop(true).spawn().unwrap();
    let mut stream = TcpStream::connect(("127.0.0.1", conn.endpoints[0].port)).await.unwrap();
    authenticate_client(&mut stream, &conn, Duration::from_secs(10)).await.unwrap();
    let mut s = S { stream, channel: 0, epoch: 0, corr: 1 };
    for attempt in 0..600 {
        let r = rpc(&mut s, 0, 0, json!({"op": "route.open", "target": RouteTarget::ManagementSurface { module_id: "synapse".into() }, "identity": BindIdentity::new(root.clone(), "catalog-probe".to_string(), "catalog-probe".to_string())})).await;
        if let (Some(c), Some(e)) = (r["route_channel"].as_u64(), r["route_epoch"].as_u64()) { s.channel = c as u16; s.epoch = e as u32; break; }
        assert!(child.try_wait().unwrap().is_none(), "ck-synapse exited: {r}");
        assert!(attempt < 599, "registration timed out: {r}");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let accepted = call(&mut s, "models.download", json!({"catalog_id": model})).await;
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
    let mut fingerprint = String::new();
    let mut outputs = BTreeMap::new();
    let mut loaded = false;
    // CATALOG_PROBE_SKIP_8192=1 leaves out the 8192-token case so later
    // requests are measured on a worker that has not already timed out.
    let skip_long = std::env::var_os("CATALOG_PROBE_SKIP_8192").is_some_and(|v| !v.is_empty());
    for group in fixture_requests(&cases, operation) {
        if skip_long && group.iter().any(|c| c["input_ids"].as_array().unwrap().len() == INLINE_TOKENS) { continue; }
        let (method, params) = group_params(&group, operation, &lane);
        let (response, elapsed_ms) = loop {
            let started = Instant::now();
            let r = call(&mut s, method, params.clone()).await;
            let ms = started.elapsed().as_secs_f64() * 1e3;
            if r["error"]["code"] == "model_loading" { tokio::time::sleep(Duration::from_millis(r["error"]["retry_after_ms"].as_u64().unwrap_or(100).max(1))).await; continue; }
            break (r, ms);
        };
        if !loaded {
            let listed = call(&mut s, "models.list", json!({})).await;
            let row = listed["models"].as_array().and_then(|m| m.iter().find(|r| r["model_id"] == lane.as_str())).cloned().unwrap_or(Value::Null);
            log("models.list", json!({"lane": lane, "fingerprints": row["fingerprints"], "self_check": row["self_check"]["state"], "certified": row["certified"]}));
            loaded = true;
        }
        let tokens: Vec<usize> = group.iter().map(|c| c["input_ids"].as_array().unwrap().len()).collect();
        if response.get("error").is_some() {
            log("request", json!({"first_case": group[0]["id"], "items": group.len(), "tokens": tokens, "elapsed_ms": elapsed_ms, "error": response["error"]}));
            continue;
        }
        if let Some(f) = response["fingerprint"].as_str() { fingerprint = f.to_string(); }
        log("request", json!({"first_case": group[0]["id"], "items": group.len(), "tokens": tokens, "elapsed_ms": elapsed_ms, "fingerprint": response["fingerprint"]}));
        let ids: Vec<Vec<u32>> = serde_json::from_value(response["observation"]["input_ids"].clone()).unwrap_or_default();
        let readout = response["observation"]["readout_ids"].as_array().and_then(|i| Some((i.first()?.as_u64()? as u32, i.get(1)?.as_u64()? as u32)));
        for (index, case) in group.iter().enumerate() {
            let observed_ids = ids.get(index).cloned().unwrap_or_default();
            let fixture_ids: Vec<u32> = serde_json::from_value(case["input_ids"].clone()).unwrap();
            let mut row = json!({"id": case["id"], "tokens": fixture_ids.len(), "input_ids_match": observed_ids == fixture_ids});
            let output = if operation == "embed" {
                let v: Vec<f64> = serde_json::from_value(response["vectors"][index]["vector"].clone()).unwrap();
                let r: Vec<f64> = serde_json::from_value(case["output"].clone()).unwrap();
                let dot: f64 = v.iter().zip(&r).map(|(a, b)| a * b).sum();
                let (nv, nr) = (v.iter().map(|x| x * x).sum::<f64>().sqrt(), r.iter().map(|x| x * x).sum::<f64>().sqrt());
                row["cosine"] = json!(dot / (nv * nr));
                row["max_abs_element_error"] = json!(v.iter().zip(&r).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max));
                Output::Embedding(v)
            } else {
                let sc: f64 = serde_json::from_value(response["scores"][index].clone()).unwrap();
                let r = case["output"].as_f64().unwrap();
                row["abs_error"] = json!((sc - r).abs());
                Output::Score(sc)
            };
            log("case", row);
            outputs.insert(case["id"].as_str().unwrap().to_string(), ObservedCase { output, input_ids: observed_ids, readout });
        }
    }
    let evaluation = evaluate(&manifest, &profile_id, &fingerprint, &fixtures, &outputs).unwrap();
    log("evaluation", json!(evaluation));
    // The runner's 8193 check: the 8192 fixture's text plus " a".
    let long = cases.iter().find(|c| c["input_ids"].as_array().unwrap().len() == 8192).unwrap();
    let mut oversized = long.clone();
    let field = if operation == "embed" { "text" } else { "document" };
    oversized[field] = json!(format!("{} a", long[field].as_str().unwrap()));
    let (method, params) = group_params(&[&oversized], operation, &lane);
    let before = call(&mut s, "certify.observations", json!({})).await;
    let started = Instant::now();
    let rejected = call(&mut s, method, params).await;
    let ms = started.elapsed().as_secs_f64() * 1e3;
    let after = call(&mut s, "certify.observations", json!({})).await;
    log("tokens_8193", json!({"elapsed_ms": ms, "error": rejected["error"], "worker_requests_before": sent(&before), "worker_requests_after": sent(&after)}));
    let _ = child.start_kill();
    let _ = child.wait().await;
}
