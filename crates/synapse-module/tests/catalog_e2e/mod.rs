use super::*;
use std::sync::Mutex;
use tokio::sync::Notify;

const BODY: &[u8] = b"fixture catalog artifact bytes";

struct CatalogServer {
    endpoint: String,
    requests: Arc<Mutex<Vec<String>>>,
    release: Arc<Notify>,
    task: tokio::task::JoinHandle<()>,
}

impl CatalogServer {
    async fn start(mode: &'static str) -> Self {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let release = Arc::new(Notify::new());
        let seen = requests.clone();
        let gate = release.clone();
        let task = tokio::spawn(async move {
            loop {
                let (mut stream, _) = listener.accept().await.unwrap();
                let seen = seen.clone();
                let gate = gate.clone();
                tokio::spawn(async move {
                    let mut request = Vec::new();
                    let mut buffer = [0; 4096];
                    while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                        let n = stream.read(&mut buffer).await.unwrap();
                        if n == 0 { return; }
                        request.extend_from_slice(&buffer[..n]);
                    }
                    let request = String::from_utf8(request).unwrap();
                    let path = request.lines().next().unwrap().split_whitespace().nth(1).unwrap().to_owned();
                    assert!(request.lines().any(|line| line.to_ascii_lowercase() == format!("host: {address}")));
                    seen.lock().unwrap().push(path.clone());
                    if mode == "redirect" && path.contains("/resolve/") {
                        stream.write_all(format!("HTTP/1.1 302 Found\r\nLocation: http://{address}/redirected\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").as_bytes()).await.unwrap();
                        return;
                    }
                    if mode == "http_error" {
                        stream.write_all(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").await.unwrap();
                        return;
                    }
                    stream.write_all(format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", BODY.len()).as_bytes()).await.unwrap();
                    let _ = stream.write_all(&BODY[..1]).await;
                    if mode == "drop" { return; }
                    if mode == "stall" { gate.notified().await; }
                    let tail = if mode == "bad_hash" { vec![b'x'; BODY.len()-1] } else { BODY[1..].to_vec() };
                    let _ = stream.write_all(&tail).await;
                });
            }
        });
        Self { endpoint: format!("http://{address}"), requests, release, task }
    }

    fn paths(&self) -> Vec<String> { self.requests.lock().unwrap().clone() }

    fn assert_pinned_requests(&self) {
        let paths = self.paths();
        assert!(!paths.is_empty());
        for path in paths.iter().filter(|p| p.contains("/resolve/")) {
            let parts: Vec<_> = path.split('/').collect();
            assert_eq!(parts[3], "resolve", "{path}");
            assert_eq!(parts[4].len(), 40, "{path}");
            assert!(parts[4].bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)));
            assert!(!path.contains("resolve/main"));
        }
    }
}

impl Drop for CatalogServer {
    fn drop(&mut self) { self.release.notify_waiters(); self.task.abort(); }
}

struct CatalogHarness {
    daemon: TestDaemon,
    _module: ModuleProcess,
    consumer: TcpStream,
    route: TestRoute,
    next_id: u64,
    server: CatalogServer,
    fixture_root: PathBuf,
}

impl CatalogHarness {
    async fn start(mode: &'static str, barrier: Option<(&str, &Path)>) -> Self {
        let server = CatalogServer::start(mode).await;
        let root = unique_temp_dir("catalog-fixture");
        std::fs::create_dir_all(&root).unwrap();
        let mut catalog: Value = serde_json::from_str(include_str!("../../src/catalog/models.json")).unwrap();
        // The files share a digest to exercise distinct-blob accounting without downloading release weights.
        for entry in catalog["models"].as_array_mut().unwrap() {
            for file in entry["files"].as_array_mut().unwrap() {
                file["sha256"] = test_sha256(BODY).into();
                file["size_bytes"] = BODY.len().into();
            }
        }
        let catalog_path = root.join("catalog.json");
        std::fs::write(&catalog_path, catalog.to_string()).unwrap();
        let config = serde_json::json!({"hf_endpoint":server.endpoint}).to_string();
        let daemon = start_daemon().await;
        let path = catalog_path.to_str().unwrap();
        let mut overrides = vec![("SYNAPSE_TEST_CATALOG", path), ("SYNAPSE_TEST_RUNNABLE_BACKENDS", "metal")];
        if let Some((name, path)) = barrier { overrides.push((name, path.to_str().unwrap())); }
        let module = spawn_synapse_module_with_env(&daemon.connection_file_path, None, Some(&config), &overrides);
        let (daemon, module, consumer, route) = open_route_for_started_module(daemon, module).await;
        Self { daemon, _module: module, consumer, route, next_id: 100, server, fixture_root: root }
    }

    async fn call(&mut self, method: &str, params: Value) -> Value {
        self.next_id += 1;
        let response = route_request(&mut self.consumer, self.route, self.next_id, serde_json::json!({"method":method,"params":params})).await;
        assert!(response.get("result").is_some(), "{method}: {response}");
        response["result"].clone()
    }

    async fn download(&mut self, id: &str, key: &str) -> Value {
        self.call("models.download", serde_json::json!({"catalog_id":id,"request_key":key})).await
    }

    async fn wait_state(&mut self, job: &str, states: &[&str]) -> Value {
        let until = Instant::now() + Duration::from_secs(15);
        let mut previous = 0;
        loop {
            let status = self.call("model.status", serde_json::json!({"job_id":job})).await;
            assert_eq!(status["kind"], "models.download");
            if let Some(done) = status["bytes_done"].as_u64() { assert!(done >= previous); previous = done; }
            if states.contains(&status["state"].as_str().unwrap()) { return status; }
            assert!(Instant::now() < until, "waiting for {states:?}: {status}");
            sleep(Duration::from_millis(20)).await;
        }
    }

    fn assert_clean(&self) {
        let conn = Connection::open(expected_store_path(&self.daemon.data_home)).unwrap();
        for table in ["catalog_installs", "catalog_install_members", "download_acquisitions"] {
            let count: u64 = conn.query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0)).unwrap();
            assert_eq!(count, 0, "{table}");
        }
    }
}

impl Drop for CatalogHarness {
    fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.fixture_root); }
}

#[tokio::test]
async fn catalog_download_progress_single_flight_cancel_and_remove_holder() {
    let mut h = CatalogHarness::start("stall", None).await;
    let before = h.call("admission.status", serde_json::json!({})).await;
    let first = h.download("gte-modernbert-base", "first").await;
    assert!(["queued", "downloading"].contains(&first["state"].as_str().unwrap()));
    let job = first["job_id"].as_str().unwrap().to_owned();
    let progress = h.wait_state(&job, &["downloading"]).await;
    assert_eq!(progress["bytes_total"], BODY.len());
    assert!(progress["bytes_done"].as_u64().unwrap() < BODY.len() as u64);
    let second = h.download("gte-modernbert-base", "second").await;
    assert_eq!(second["job_id"], job);
    let blank = h.download("gte-modernbert-base", "  ").await;
    assert_eq!(blank["job_id"], job);
    let serving = h.call("embed.query", serde_json::json!({"model":"gte-modernbert-base","text":"hello"})).await;
    assert_eq!(serving["error"]["code"], "model_not_installed");
    assert_eq!(serving["error"]["details"]["download_job_id"], job);
    let remove = h.call("models.remove", serde_json::json!({"catalog_id":"gte-modernbert-base"})).await;
    assert_eq!(remove["error"]["code"], "model_in_use");
    assert!(remove["error"]["details"]["holders"].as_array().unwrap().iter().any(|r| r["job_id"] == job));
    let cancelled = h.call("models.download.cancel", serde_json::json!({"job_id":job})).await;
    assert_eq!(cancelled["state"], "cancelled");
    h.server.release.notify_waiters();
    h.wait_state(&job, &["cancelled"]).await;
    h.assert_clean();
    let after = h.call("admission.status", serde_json::json!({})).await;
    assert_eq!(before["jobs_open"], after["jobs_open"]);
    h.server.assert_pinned_requests();
}

#[tokio::test]
async fn catalog_redirect_commit_reuse_projections_and_removal() {
    let mut h = CatalogHarness::start("redirect", None).await;
    let first = h.download("gte-modernbert-base", "first").await;
    let job = first["job_id"].as_str().unwrap().to_owned();
    let committed = h.wait_state(&job, &["committed", "failed"]).await;
    assert_eq!(committed["state"], "committed", "{committed}");
    assert!(committed.get("bytes_done").is_none());
    assert!(committed.get("error").is_none());
    assert_eq!(h.server.paths().len(), 2);
    assert_eq!(h.server.paths()[1], "/redirected");
    h.server.assert_pinned_requests();
    let listed = h.call("models.list", serde_json::json!({})).await;
    let row = listed["models"].as_array().unwrap().iter().find(|r| r["model_id"] == "gte-modernbert-base-metal").unwrap();
    assert_eq!(row["state"], "unloaded");
    assert_eq!(row["fingerprints"].as_array().unwrap().len(), 1);
    assert_eq!(row["self_check"]["state"], "pending");
    assert_eq!(row["certified"], false);
    assert_eq!(row["serving_admission"], "enabled");
    let again = h.download("gte-modernbert-base", "first").await;
    assert_eq!(again["state"], "committed");
    assert_eq!(again["job_id"], job);
    assert_eq!(h.server.paths().len(), 2);
    let probe = h.call("probe.start", serde_json::json!({"models":["gte-modernbert-base-metal"]})).await;
    assert_eq!(probe["error"]["code"], "invalid_request");
    let removed = h.call("models.remove", serde_json::json!({"catalog_id":"gte-modernbert-base"})).await;
    assert_eq!(removed["freed_bytes"], BODY.len());
    assert_eq!(removed["removed_manifests"].as_array().unwrap().len(), 1);
    h.assert_clean();
    let empty = h.call("models.remove", serde_json::json!({"catalog_id":"gte-modernbert-base"})).await;
    assert_eq!(empty["freed_bytes"], 0);
    assert_eq!(empty["removed_manifests"], serde_json::json!([]));
}

#[tokio::test]
async fn catalog_failed_downloads_cleanup_and_keep_typed_details() {
    for (mode, code) in [("bad_hash", "artifact_invalid"), ("http_error", "download_failed"), ("drop", "download_failed")] {
        let mut h = CatalogHarness::start(mode, None).await;
        let first = h.download("gte-modernbert-base", mode).await;
        let job = first["job_id"].as_str().unwrap();
        let failed = h.wait_state(job, &["failed", "committed"]).await;
        assert_eq!(failed["state"], "failed", "{mode}: {failed}");
        assert_eq!(failed["error"]["code"], code);
        assert!(failed["error"]["message"].is_string());
        assert!(failed["error"]["details"]["file"].is_string());
        if mode == "http_error" { assert_eq!(failed["error"]["details"]["http_status"], 500); }
        if mode == "bad_hash" { assert_eq!(failed["error"]["details"]["expected_sha256"], test_sha256(BODY)); }
        h.assert_clean();
        h.server.assert_pinned_requests();
    }
}

#[tokio::test]
async fn catalog_pre_publish_cancel_cleans_acquisitions() {
    let root = unique_temp_dir("catalog-publish-barrier");
    let mut h = CatalogHarness::start("ok", Some(("SYNAPSE_TEST_DOWNLOAD_PRE_PUBLISH_BARRIER", &root))).await;
    let accepted = h.download("gte-modernbert-base", "cancel").await;
    let job = accepted["job_id"].as_str().unwrap().to_owned();
    let until = Instant::now() + Duration::from_secs(15);
    while !root.join(format!("{job}.ready")).exists() {
        assert!(Instant::now() < until);
        sleep(Duration::from_millis(20)).await;
    }
    let cancelled = h.call("models.download.cancel", serde_json::json!({"job_id":job})).await;
    assert_eq!(cancelled["state"], "cancelled");
    std::fs::write(root.join(format!("{job}.release")), b"release").unwrap();
    h.wait_state(&job, &["cancelled"]).await;
    h.assert_clean();
    h.server.assert_pinned_requests();
    let _ = std::fs::remove_dir_all(root);
}
