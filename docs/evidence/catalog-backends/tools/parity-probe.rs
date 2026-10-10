//! Scratch diagnostics (not part of the synapse repo). Repeats the
//! certification runner's session and request grouping for one row/model,
//! then prints the evaluator's full output plus per-case error figures.
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::PathBuf, sync::Arc, time::Duration};
use subc_daemon::{daemon_config::StorageConfig, serve_listener, ControlHandler, Registry, Router, ServerAuth};
use subc_protocol::{BindIdentity, Flags, Frame, FrameType, Priority, RouteTarget};
use subc_transport::{authenticate_client, generate_daemon_id, generate_key, read_frame, write_atomic, write_frame, ConnectionInfo, Endpoint, SCHEMA_VERSION};
use synapse_parity::{evaluator::{evaluate, FixtureSet, ObservedCase, Output}, manifest::Manifest};
use tokio::net::{TcpListener, TcpStream};

struct S { stream: TcpStream, channel: u16, epoch: u32, corr: u64 }
async fn rpc(s: &mut S, channel: u16, epoch: u32, body: Value) -> Value {
    s.corr += 1;
    let corr = s.corr;
    let frame = Frame::build(FrameType::Request, Flags::new(false, Priority::Passive, false), channel, epoch, corr, serde_json::to_vec(&body).unwrap()).unwrap();
    write_frame(&mut s.stream, &frame).await.unwrap();
    loop {
        let frame = read_frame(&mut s.stream).await.unwrap().expect("closed");
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
fn group_params(group: &[&Value], operation: &str) -> (&'static str, Value) {
    if operation == "embed" {
        if group.len() == 1 { return ("embed.query", json!({"model": "certify-candidate", "text": group[0]["text"], "accept_declared": true})); }
        let items = group.iter().map(|c| json!({"id": c["id"], "text": c["text"]})).collect::<Vec<_>>();
        ("embed.batch", json!({"model": "certify-candidate", "items": items, "accept_declared": true}))
    } else {
        let candidates = group.iter().map(|c| &c["document"]).collect::<Vec<_>>();
        ("rerank.score", json!({"model": "certify-candidate", "query": group[0]["query"], "candidates": candidates, "accept_declared": true}))
    }
}
fn lane(row: &str) -> &'static str { if row.starts_with("cuda-") { "owned-cuda" } else { "owned-vulkan" } }

#[tokio::main]
async fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (row, model, assets, checkout, weights, root) = (a[1].clone(), a[2].clone(), PathBuf::from(&a[3]), PathBuf::from(&a[4]), PathBuf::from(&a[5]), PathBuf::from(&a[6]));
    let worker = if lane(&row) == "owned-cuda" { "ck-synapse-worker-cuda" } else { "ck-synapse-worker-vulkan" };
    let parity_root = checkout.join("bench/parity");
    let manifest = Manifest::load(&parity_root.join("models.json")).unwrap();
    let raw: Value = serde_json::from_slice(&std::fs::read(parity_root.join("models.json")).unwrap()).unwrap();
    let fixtures = FixtureSet::load(&parity_root, &manifest, &model).unwrap();
    let index: Value = serde_json::from_slice(&std::fs::read(parity_root.join("fixtures/index.json")).unwrap()).unwrap();
    let fixture_id = synapse_parity::evaluator::fixture_set_id(&manifest, &model);
    let document: Value = serde_json::from_slice(&std::fs::read(parity_root.join(index[&fixture_id]["path"].as_str().unwrap())).unwrap()).unwrap();
    let cases = document["cases"].as_array().unwrap();
    let operation = raw["models"][&model]["operation"].as_str().unwrap().to_string();
    let profile_id = format!("{model}.{}", lane(&row));
    std::fs::create_dir_all(root.join("data")).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let conn = ConnectionInfo { schema: SCHEMA_VERSION, endpoints: vec![Endpoint { host: "127.0.0.1".into(), port: listener.local_addr().unwrap().port() }], key: generate_key().unwrap(), daemon_id: generate_daemon_id().unwrap(), pid: std::process::id(), daemon_ver: "parity-probe".into(), wire_version: Some(subc_protocol::PROTOCOL_VERSION) };
    let conn_path = root.join("connection.json");
    write_atomic(&conn_path, &conn).unwrap();
    let control = ControlHandler::new(Arc::new(Registry::default())).with_storage_config(Some(StorageConfig::Sqlite { data_home: root.join("data") }));
    let router = Arc::new(Router::with_control_handler(Arc::new(control)));
    let auth = ServerAuth::new(conn.key.clone(), conn.daemon_id, conn.daemon_ver.clone());
    tokio::spawn(async move { let _ = serve_listener(listener, router, auth).await; });
    let pinned = &raw["models"][&model];
    let package = synapse_parity::convert::convert_profile_file(&manifest, &profile_id, &weights.join("model.safetensors")).unwrap();
    let model_path = root.join("profile.safetensors");
    std::fs::write(&model_path, package).unwrap();
    // Same preload the certification runner writes for this row.
    let preload = json!({
        "model_id": "certify-candidate", "engine": lane(&row), "profile": profile_id,
        "task": operation, "model_path": model_path, "tokenizer_path": weights.join("tokenizer.json"),
        "pooling": match pinned["grammar"]["pooling"].as_str() { Some("cls") => "cls", Some("masked_mean") => "mean", _ => "last" }, "normalize": pinned["output"]["normalization"] == "l2",
        "execution": "explicit", "attention_units": 8192 * 8192,
        "worker_bin": assets.join(worker), "worker_runtime_dir": assets,
    });
    let config = root.join("config.json");
    std::fs::write(&config, serde_json::to_vec(&json!({"certify_observation": true, "preload_models": [preload], "inline": {"deadline_ms": 3600000, "max_queue_ms": 3600000, "max_items": INLINE_ITEMS, "max_tokens": INLINE_TOKENS}})).unwrap()).unwrap();
    let mut child = tokio::process::Command::new(assets.join("ck-synapse")).arg("--subc").arg(&conn_path)
        .env("SUBC_MODULE_ID", "synapse").env("SYNAPSE_CONFIG_PATH", &config).env("XDG_DATA_HOME", root.join("data"))
        .env("CORTEXKIT_LEASE_ROOT", root.join("leases")).env("CORTEXKIT_STORE_ROOT", root.join("store"))
        .stderr(std::fs::File::create(root.join("ck-synapse.stderr.log")).unwrap()).kill_on_drop(true).spawn().unwrap();
    let mut stream = TcpStream::connect(("127.0.0.1", conn.endpoints[0].port)).await.unwrap();
    authenticate_client(&mut stream, &conn, Duration::from_secs(10)).await.unwrap();
    let mut s = S { stream, channel: 0, epoch: 0, corr: 1 };
    for attempt in 0..600 {
        let r = rpc(&mut s, 0, 0, json!({"op": "route.open", "target": RouteTarget::ManagementSurface { module_id: "synapse".into() }, "identity": BindIdentity::new(root.clone(), "probe".to_string(), "probe".to_string())})).await;
        if let (Some(c), Some(e)) = (r["route_channel"].as_u64(), r["route_epoch"].as_u64()) { s.channel = c as u16; s.epoch = e as u32; break; }
        assert!(child.try_wait().unwrap().is_none(), "exited: {r}");
        assert!(attempt < 599);
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let mut fingerprint = String::new();
    let mut outputs = BTreeMap::new();
    let mut per_case = Vec::new();
    for group in fixture_requests(cases, &operation) {
        let (method, params) = group_params(&group, &operation);
        let started = std::time::Instant::now();
        let response = call(&mut s, method, params).await;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
        if let Some(f) = response["fingerprint"].as_str() { fingerprint = f.to_string(); }
        if response.get("error").is_some() { println!("{}", json!({"refused": group[0]["id"], "tokens": group[0]["input_ids"].as_array().map(Vec::len), "elapsed_ms": elapsed_ms, "response": response})); continue; }
        let ids: Vec<Vec<u32>> = serde_json::from_value(response["observation"]["input_ids"].clone()).unwrap_or_default();
        let readout = response["observation"]["readout_ids"].as_array().and_then(|i| Some((i.first()?.as_u64()? as u32, i.get(1)?.as_u64()? as u32)));
        for (index, case) in group.iter().enumerate() {
            let observed_ids = ids.get(index).cloned().unwrap_or_default();
            let fixture_ids: Vec<u32> = serde_json::from_value(case["input_ids"].clone()).unwrap();
            let mut row = json!({"id": case["id"], "category": case["category"], "pool": case["pool"], "request": method, "group_size": group.len(), "tokens": fixture_ids.len(), "observed_tokens": observed_ids.len(), "input_ids_match": observed_ids == fixture_ids, "request_elapsed_ms": elapsed_ms});
            let output = if operation == "embed" {
                let v: Vec<f64> = serde_json::from_value(response["vectors"][index]["vector"].clone()).unwrap();
                let r: Vec<f64> = serde_json::from_value(case["output"].clone()).unwrap();
                let dot: f64 = v.iter().zip(&r).map(|(a, b)| a * b).sum();
                let (nv, nr) = (v.iter().map(|x| x * x).sum::<f64>().sqrt(), r.iter().map(|x| x * x).sum::<f64>().sqrt());
                let max_abs = v.iter().zip(&r).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
                row["cosine"] = json!(dot / (nv * nr));
                row["max_abs_element_error"] = json!(max_abs);
                row["norm"] = json!(nv);
                row["finite"] = json!(v.iter().all(|x| x.is_finite()));
                Output::Embedding(v)
            } else {
                let a: f64 = serde_json::from_value(response["scores"][index].clone()).unwrap();
                let r = case["output"].as_f64().unwrap();
                row["reference_score"] = json!(r); row["score"] = json!(a); row["abs_error"] = json!((a - r).abs());
                Output::Score(a)
            };
            per_case.push(row);
            outputs.insert(case["id"].as_str().unwrap().to_string(), ObservedCase { output, input_ids: observed_ids, readout });
        }
    }
    for row in &per_case { println!("{}", json!({"case": row})); }
    let evaluation = evaluate(&manifest, &profile_id, &fingerprint, &fixtures, &outputs).unwrap();
    println!("{}", json!({"evaluation": evaluation}));
    let _ = child.start_kill();
}
