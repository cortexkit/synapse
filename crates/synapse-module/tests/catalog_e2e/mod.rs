use super::*;
use std::{collections::BTreeMap, sync::Mutex};
use tokio::sync::Notify;

const BODY: &[u8] = b"fixture catalog artifact bytes";

fn assert_golden_status(status: &Value) {
    let fixtures: Value =
        serde_json::from_str(include_str!("../fixtures/catalog_wire_v1.json")).unwrap();
    let mut actual = status.clone();
    let state = actual["state"].as_str().unwrap().to_owned();
    assert!(actual["job_id"].as_str().is_some_and(|id| !id.is_empty()));
    actual["job_id"] = "<job>".into();
    actual.as_object_mut().unwrap().remove("kind");
    assert_eq!(actual, fixtures[&state], "flat {state} wire shape");
}

fn catalog_sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}

struct CatalogServer {
    endpoint: String,
    requests: Arc<Mutex<Vec<String>>>,
    release: Arc<Notify>,
    task: tokio::task::JoinHandle<()>,
}

impl CatalogServer {
    async fn start(mode: &'static str, files: BTreeMap<String, PathBuf>) -> Self {
        let files = Arc::new(files);
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
                let files = files.clone();
                tokio::spawn(async move {
                    let mut request = Vec::new();
                    let mut buffer = [0; 4096];
                    while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                        let n = stream.read(&mut buffer).await.unwrap();
                        if n == 0 {
                            return;
                        }
                        request.extend_from_slice(&buffer[..n]);
                    }
                    let request = String::from_utf8(request).unwrap();
                    let path = request
                        .lines()
                        .next()
                        .unwrap()
                        .split_whitespace()
                        .nth(1)
                        .unwrap()
                        .to_owned();
                    assert!(request
                        .lines()
                        .any(|line| line.to_ascii_lowercase() == format!("host: {address}")));
                    seen.lock().unwrap().push(path.clone());
                    if mode == "redirect"
                        && path.contains("/resolve/")
                        && !path.starts_with("/redirected")
                    {
                        let target = if files.is_empty() {
                            "/redirected".to_owned()
                        } else {
                            format!("/redirected{path}")
                        };
                        stream.write_all(format!("HTTP/1.1 302 Found\r\nLocation: http://{address}{target}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").as_bytes()).await.unwrap();
                        return;
                    }
                    if mode == "http_error" {
                        stream.write_all(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").await.unwrap();
                        return;
                    }
                    if !files.is_empty() {
                        let file_path = files
                            .get(path.strip_prefix("/redirected").unwrap_or(&path))
                            .expect("requested pinned fixture file");
                        let mut file = std::fs::File::open(file_path).unwrap();
                        let size = file.metadata().unwrap().len();
                        stream.write_all(format!("HTTP/1.1 200 OK\r\nContent-Length: {size}\r\nConnection: close\r\n\r\n").as_bytes()).await.unwrap();
                        let mut buffer = vec![0; 64 * 1024];
                        loop {
                            let count = std::io::Read::read(&mut file, &mut buffer).unwrap();
                            if count == 0 {
                                break;
                            }
                            if stream.write_all(&buffer[..count]).await.is_err() {
                                break;
                            }
                        }
                        return;
                    }
                    stream.write_all(format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", BODY.len()).as_bytes()).await.unwrap();
                    let _ = stream.write_all(&BODY[..1]).await;
                    if mode == "drop" {
                        return;
                    }
                    if mode == "stall" {
                        gate.notified().await;
                    }
                    let tail = if mode == "bad_hash" {
                        vec![b'x'; BODY.len() - 1]
                    } else {
                        BODY[1..].to_vec()
                    };
                    let _ = stream.write_all(&tail).await;
                });
            }
        });
        Self {
            endpoint: format!("http://{address}"),
            requests,
            release,
            task,
        }
    }

    fn paths(&self) -> Vec<String> {
        self.requests.lock().unwrap().clone()
    }

    fn assert_pinned_requests(&self) {
        let paths = self.paths();
        assert!(!paths.is_empty());
        for path in paths
            .iter()
            .filter(|p| p.contains("/resolve/") && !p.starts_with("/redirected"))
        {
            let parts: Vec<_> = path.split('/').collect();
            assert_eq!(parts[3], "resolve", "{path}");
            assert_eq!(parts[4].len(), 40, "{path}");
            assert!(parts[4]
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)));
            assert!(!path.contains("resolve/main"));
        }
    }
}

impl Drop for CatalogServer {
    fn drop(&mut self) {
        self.release.notify_waiters();
        self.task.abort();
    }
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
        let mut catalog: Value =
            serde_json::from_str(include_str!("../../src/catalog/models.json")).unwrap();
        // The files share a digest to exercise distinct-blob accounting without downloading release weights.
        for entry in catalog["models"].as_array_mut().unwrap() {
            for file in entry["files"].as_array_mut().unwrap() {
                file["sha256"] = catalog_sha256(BODY).into();
                file["size_bytes"] = BODY.len().into();
            }
        }
        let mut harness =
            Self::start_catalog(mode, barrier, catalog, BTreeMap::new(), "metal", None).await;
        let catalog = harness
            .call(
                "models.catalog",
                serde_json::json!({"query":"gte-modernbert-base"}),
            )
            .await;
        assert_eq!(
            catalog["models"][0]["download_bytes"],
            3 * BODY.len(),
            "catalog override must be honored"
        );
        assert_eq!(catalog["models"][0]["backends"][0]["runnable"], true);
        harness
    }

    async fn start_catalog(
        mode: &'static str,
        barrier: Option<(&str, &Path)>,
        catalog: Value,
        files: BTreeMap<String, PathBuf>,
        backends: &str,
        fault: Option<&str>,
    ) -> Self {
        Self::start_catalog_env(mode, barrier, catalog, files, backends, fault, &[]).await
    }

    async fn start_catalog_env(
        mode: &'static str,
        barrier: Option<(&str, &Path)>,
        catalog: Value,
        files: BTreeMap<String, PathBuf>,
        backends: &str,
        fault: Option<&str>,
        extra_env: &[(&str, &str)],
    ) -> Self {
        let server = CatalogServer::start(mode, files).await;
        let root = unique_temp_dir("catalog-fixture");
        std::fs::create_dir_all(&root).unwrap();
        let catalog_path = root.join("catalog.json");
        std::fs::write(&catalog_path, catalog.to_string()).unwrap();
        let config = serde_json::json!({"hf_endpoint":server.endpoint}).to_string();
        let daemon = start_daemon().await;
        let path = catalog_path.to_str().unwrap();
        let cache = root.join("cache");
        let timings = root.join("validation-timings.jsonl");
        let mut overrides = vec![
            (
                "SYNAPSE_TEST_CATALOG_VALIDATION_TIMINGS",
                timings.to_str().unwrap(),
            ),
            ("SYNAPSE_TEST_CATALOG", path),
            ("SYNAPSE_TEST_RUNNABLE_BACKENDS", backends),
            ("CORTEXKIT_MODEL_CACHE", cache.to_str().unwrap()),
        ];
        if let Some((name, path)) = barrier {
            overrides.push((name, path.to_str().unwrap()));
        }
        let fault_lane = format!("{}-metal", catalog["models"][0]["id"].as_str().unwrap());
        if let Some(fault) = fault {
            overrides.push(("SYNAPSE_TEST_CATALOG_FAULT_LANE", &fault_lane));
            overrides.push(("SYNAPSE_TEST_CATALOG_FAULT", fault));
        }
        overrides.extend_from_slice(extra_env);
        let module = spawn_synapse_module_with_env(
            &daemon.connection_file_path,
            None,
            Some(&config),
            &overrides,
        );
        let (daemon, module, consumer, route) = open_route_for_started_module(daemon, module).await;
        Self {
            daemon,
            _module: module,
            consumer,
            route,
            next_id: 100,
            server,
            fixture_root: root,
        }
    }

    async fn call(&mut self, method: &str, params: Value) -> Value {
        self.next_id += 1;
        let response = route_request(
            &mut self.consumer,
            self.route,
            self.next_id,
            serde_json::json!({"method":method,"params":params}),
        )
        .await;
        assert!(response.get("result").is_some(), "{method}: {response}");
        response["result"].clone()
    }

    #[cfg(target_os = "macos")]
    async fn serve_when_ready(&mut self, method: &str, params: Value) -> Value {
        let until = Instant::now() + Duration::from_secs(120);
        loop {
            let result = self.call(method, params.clone()).await;
            if result["error"]["code"] != "model_loading" {
                return result;
            }
            assert_eq!(result["error"]["class"], "transient");
            assert_eq!(result["error"]["safe_to_retry_same_request"], true);
            let retry = result["error"]["retry_after_ms"]
                .as_u64()
                .expect("loading retry delay");
            assert_eq!(retry, 250);
            if Instant::now() >= until {
                let lanes = self.call("models.list", serde_json::json!({})).await;
                let status = self
                    .call(
                        "model.status",
                        serde_json::json!({"model_id":lanes["models"][0]["model_id"]}),
                    )
                    .await;
                let timings =
                    std::fs::read_to_string(self.fixture_root.join("validation-timings.jsonl"))
                        .unwrap_or_default();
                panic!("catalog cold load exceeded 120 s: {result}; lanes: {lanes}; loader: {status}; validation timings: {timings}");
            }
            sleep(Duration::from_millis(retry)).await;
        }
    }

    async fn download(&mut self, id: &str, key: &str) -> Value {
        self.call(
            "models.download",
            serde_json::json!({"catalog_id":id,"request_key":key}),
        )
        .await
    }

    async fn wait_state(&mut self, job: &str, states: &[&str]) -> Value {
        let until = Instant::now() + Duration::from_secs(300);
        let mut previous = 0;
        loop {
            let status = self
                .call("model.status", serde_json::json!({"job_id":job}))
                .await;
            assert_eq!(status["kind"], "models.download");
            if let Some(done) = status["bytes_done"].as_u64() {
                assert!(done >= previous);
                previous = done;
            }
            if states.contains(&status["state"].as_str().unwrap()) {
                return status;
            }
            assert!(Instant::now() < until, "waiting for {states:?}: {status}");
            sleep(Duration::from_millis(20)).await;
        }
    }

    fn assert_clean(&self) {
        let conn = Connection::open(expected_store_path(&self.daemon.data_home)).unwrap();
        for table in [
            "catalog_installs",
            "catalog_install_members",
            "download_acquisitions",
        ] {
            let count: u64 = conn
                .query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0))
                .unwrap();
            assert_eq!(count, 0, "{table}");
        }
        for directory in ["blobs", "catalog-staging"] {
            let path = self.fixture_root.join("cache").join(directory);
            if path.exists() {
                assert_eq!(
                    std::fs::read_dir(&path).unwrap().count(),
                    0,
                    "leftover files in {}",
                    path.display()
                );
            }
        }
    }
}

impl Drop for CatalogHarness {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.fixture_root);
    }
}

#[tokio::test]
async fn catalog_download_progress_single_flight_cancel_and_remove_holder() {
    let mut h = CatalogHarness::start("stall", None).await;
    let before = h.call("admission.status", serde_json::json!({})).await;
    let first = h.download("gte-modernbert-base", "first").await;
    assert!(["queued", "downloading"].contains(&first["state"].as_str().unwrap()));
    let job = first["job_id"].as_str().unwrap().to_owned();
    if first["state"] == "queued" {
        assert_golden_status(&first);
    }
    let until = Instant::now() + Duration::from_secs(15);
    let progress = loop {
        let progress = h.wait_state(&job, &["downloading"]).await;
        if progress["bytes_done"] == 1 {
            break progress;
        }
        assert!(
            Instant::now() < until,
            "first body byte was not observed: {progress}"
        );
        sleep(Duration::from_millis(20)).await;
    };
    assert_golden_status(&progress);
    assert_eq!(progress["bytes_total"], BODY.len());
    assert!(progress["bytes_done"].as_u64().unwrap() < BODY.len() as u64);
    let second = h.download("gte-modernbert-base", "second").await;
    assert_eq!(second["job_id"], job);
    let blank = h.download("gte-modernbert-base", "  ").await;
    assert_eq!(blank["job_id"], job);
    let serving = h
        .call(
            "embed.query",
            serde_json::json!({"model":"gte-modernbert-base","text":"hello"}),
        )
        .await;
    assert_eq!(serving["error"]["code"], "model_not_installed");
    assert_eq!(serving["error"]["details"]["download_job_id"], job);
    let remove = h
        .call(
            "models.remove",
            serde_json::json!({"catalog_id":"gte-modernbert-base"}),
        )
        .await;
    assert_eq!(remove["error"]["code"], "model_in_use");
    assert!(remove["error"]["details"]["holders"]
        .as_array()
        .unwrap()
        .iter()
        .any(|r| r["job_id"] == job));
    let cancelled = h
        .call("models.download.cancel", serde_json::json!({"job_id":job}))
        .await;
    assert_eq!(cancelled["state"], "cancelled");
    assert_golden_status(&cancelled);
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
    assert_golden_status(&committed);
    assert!(committed.get("bytes_done").is_none());
    assert!(committed.get("error").is_none());
    assert_eq!(h.server.paths().len(), 2);
    assert_eq!(h.server.paths()[1], "/redirected");
    h.server.assert_pinned_requests();
    let listed = h.call("models.list", serde_json::json!({})).await;
    let row = listed["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["model_id"] == "gte-modernbert-base-metal")
        .unwrap();
    assert_eq!(row["state"], "unloaded");
    assert_eq!(row["fingerprints"].as_array().unwrap().len(), 1);
    assert_eq!(row["self_check"]["state"], "pending");
    assert_eq!(row["certified"], false);
    assert_eq!(row["serving_admission"], "enabled");
    let again = h.download("gte-modernbert-base", "first").await;
    assert_eq!(again["state"], "committed");
    assert_eq!(again["job_id"], job);
    assert_eq!(h.server.paths().len(), 2);
    let probe = h
        .call(
            "probe.start",
            serde_json::json!({"models":["gte-modernbert-base-metal"]}),
        )
        .await;
    assert_eq!(probe["error"]["code"], "invalid_request");
    let removed = h
        .call(
            "models.remove",
            serde_json::json!({"catalog_id":"gte-modernbert-base"}),
        )
        .await;
    assert_eq!(removed["freed_bytes"], BODY.len());
    assert_eq!(removed["removed_manifests"].as_array().unwrap().len(), 1);
    h.assert_clean();
    let empty = h
        .call(
            "models.remove",
            serde_json::json!({"catalog_id":"gte-modernbert-base"}),
        )
        .await;
    assert_eq!(empty["freed_bytes"], 0);
    assert_eq!(empty["removed_manifests"], serde_json::json!([]));
}

#[tokio::test]
async fn catalog_failed_downloads_cleanup_and_keep_typed_details() {
    for (mode, code) in [
        ("bad_hash", "artifact_invalid"),
        ("http_error", "download_failed"),
        ("drop", "download_failed"),
    ] {
        let mut h = CatalogHarness::start(mode, None).await;
        let first = h.download("gte-modernbert-base", mode).await;
        let job = first["job_id"].as_str().unwrap();
        let failed = h.wait_state(job, &["failed", "committed"]).await;
        assert_eq!(failed["state"], "failed", "{mode}: {failed}");
        assert_eq!(failed["error"]["code"], code);
        assert!(failed["error"]["message"].is_string());
        assert!(failed["error"]["details"]["file"].is_string());
        if mode == "http_error" {
            assert_eq!(failed["error"]["details"]["http_status"], 500);
        }
        if mode == "bad_hash" {
            assert_eq!(
                failed["error"]["details"]["expected_sha256"],
                catalog_sha256(BODY)
            );
        }
        h.assert_clean();
        h.server.assert_pinned_requests();
    }
}

#[tokio::test]
async fn catalog_pre_publish_cancel_cleans_acquisitions() {
    let root = unique_temp_dir("catalog-publish-barrier");
    let mut h = CatalogHarness::start(
        "ok",
        Some(("SYNAPSE_TEST_DOWNLOAD_PRE_PUBLISH_BARRIER", &root)),
    )
    .await;
    let accepted = h.download("gte-modernbert-base", "cancel").await;
    let job = accepted["job_id"].as_str().unwrap().to_owned();
    let until = Instant::now() + Duration::from_secs(15);
    while !root.join(format!("{job}.ready")).exists() {
        assert!(Instant::now() < until);
        sleep(Duration::from_millis(20)).await;
    }
    let cancelled = h
        .call("models.download.cancel", serde_json::json!({"job_id":job}))
        .await;
    assert_eq!(cancelled["state"], "cancelled");
    assert_golden_status(&cancelled);
    std::fs::write(root.join(format!("{job}.release")), b"release").unwrap();
    h.wait_state(&job, &["cancelled"]).await;
    h.assert_clean();
    h.server.assert_pinned_requests();
    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn catalog_semantic_refusals_match_golden_and_never_fetch() {
    let mut h = CatalogHarness::start("ok", None).await;
    let golden: Value =
        serde_json::from_str(include_str!("../fixtures/catalog_wire_v1.json")).unwrap();
    let unknown = h
        .call(
            "models.download",
            serde_json::json!({"catalog_id":"no-such-model"}),
        )
        .await;
    assert_eq!(unknown, golden["unknown_download"]);
    for method in ["models.download", "models.remove"] {
        let derived = h
            .call(
                method,
                serde_json::json!({"catalog_id":"gte-modernbert-base-metal"}),
            )
            .await;
        assert_eq!(derived["error"]["code"], "invalid_request");
        assert_eq!(
            derived["error"]["details"],
            serde_json::json!({"catalog_id":"gte-modernbert-base","model_id":"gte-modernbert-base-metal"})
        );
        let unknown = h
            .call(
                method,
                serde_json::json!({"catalog_id":"gte-modernbert-base-f16"}),
            )
            .await;
        assert_eq!(unknown["error"]["code"], "unknown_model");
    }
    let unavailable = h.download("qwen3-reranker-0.6b", "no-backend").await;
    assert_eq!(unavailable["error"]["code"], "backend_unavailable");
    let wrong_task = h
        .call(
            "embed.query",
            serde_json::json!({"model":"gte-reranker-modernbert-base","text":"hello"}),
        )
        .await;
    assert_eq!(wrong_task["error"]["code"], "invalid_request");
    assert_eq!(wrong_task["error"]["details"]["requested_task"], "embed");
    let wrong_task = h.call("rerank.score", serde_json::json!({"model":"gte-modernbert-base","query":"hello","candidates":["hello"]})).await;
    assert_eq!(wrong_task["error"]["code"], "invalid_request");
    let unload = h
        .call(
            "model.unload",
            serde_json::json!({"model_id":"gte-modernbert-base"}),
        )
        .await;
    assert_eq!(unload["error"]["code"], "invalid_request");
    assert_eq!(
        unload["error"]["details"]["lane_ids"],
        serde_json::json!(["gte-modernbert-base-metal"])
    );
    assert!(h.server.paths().is_empty());
    h.assert_clean();
}

#[tokio::test]
async fn catalog_stale_installs_are_visible_but_not_served_or_counted_installed() {
    let mut h = CatalogHarness::start("ok", None).await;
    let conn = Connection::open(expected_store_path(&h.daemon.data_home)).unwrap();
    conn.execute(
        "INSERT INTO catalog_installs VALUES (?1, ?2, 'metal')",
        params!["gte-modernbert-base", "0".repeat(64)],
    )
    .unwrap();
    let digest = catalog_sha256(BODY);
    let blobs = h.fixture_root.join("cache/blobs");
    std::fs::create_dir_all(&blobs).unwrap();
    std::fs::write(blobs.join(&digest), BODY).unwrap();
    conn.execute("INSERT INTO catalog_install_members VALUES ('gte-modernbert-base',?1,'metal','model.safetensors',?2)", params!["0".repeat(64), digest]).unwrap();
    let stale = h
        .call(
            "models.catalog",
            serde_json::json!({"query":"gte-modernbert-base","installed":false}),
        )
        .await;
    assert_eq!(stale["models"][0]["install_state"], "stale");
    let installed = h
        .call(
            "models.catalog",
            serde_json::json!({"query":"gte-modernbert-base","installed":true}),
        )
        .await;
    assert_eq!(installed["models"], serde_json::json!([]));
    let serve = h
        .call(
            "embed.query",
            serde_json::json!({"model":"gte-modernbert-base","text":"hello"}),
        )
        .await;
    assert_eq!(serve["error"]["code"], "model_not_installed");
    assert!(h.server.paths().is_empty());
    let removed = h
        .call(
            "models.remove",
            serde_json::json!({"catalog_id":"gte-modernbert-base"}),
        )
        .await;
    assert_eq!(
        removed["removed_manifests"],
        serde_json::json!(["0".repeat(64)])
    );
    assert_eq!(removed["freed_bytes"], BODY.len());
    h.assert_clean();
}

#[tokio::test]
async fn compiled_catalog_listing_is_frozen_sorted_and_filters_intersect() {
    let daemon = start_daemon().await;
    let module = spawn_synapse_module_with_env(
        &daemon.connection_file_path,
        None,
        None,
        &[("SYNAPSE_TEST_RUNNABLE_BACKENDS", "metal")],
    );
    let (_daemon, _module, mut consumer, route) =
        open_route_for_started_module(daemon, module).await;
    let rows = route_request(
        &mut consumer,
        route,
        100,
        serde_json::json!({"method":"models.catalog","params":{}}),
    )
    .await;
    let rows = rows["result"]["models"].as_array().unwrap();
    let compiled: Value =
        serde_json::from_str(include_str!("../../src/catalog/models.json")).unwrap();
    let ids: Vec<_> = rows.iter().map(|r| r["id"].as_str().unwrap()).collect();
    assert_eq!(
        ids,
        [
            "gte-modernbert-base",
            "gte-reranker-modernbert-base",
            "qwen3-embedding-0.6b",
            "qwen3-reranker-0.6b"
        ]
    );
    for row in rows {
        let entry = compiled["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["id"] == row["id"])
            .unwrap();
        assert_eq!(row["upstream"], entry["upstream"]);
        // This manifest contains ASCII strings and integer sizes, so sorted compact JSON
        // produces the canonical bytes required by the manifest digest contract.
        let manifest = serde_json::json!({"upstream":entry["upstream"],"files":entry["files"]});
        assert_eq!(
            row["manifest_digest"],
            catalog_sha256(manifest.to_string().as_bytes())
        );
        assert_eq!(row["install_state"], "not_installed");
        if row["id"] == "qwen3-reranker-0.6b" {
            assert_eq!(row["backends"], serde_json::json!([]));
            assert_eq!(row["download_bytes"], 0);
            assert_eq!(
                row["upstream"]["revision"],
                "e61197ed45024b0ed8a2d74b80b4d909f1255473"
            );
        } else {
            assert_eq!(row["backends"].as_array().unwrap().len(), 1);
            let backend = &row["backends"][0];
            assert_eq!(backend["backend"], "metal");
            assert_eq!(
                backend["lane_id"],
                format!("{}-metal", row["id"].as_str().unwrap())
            );
            assert_eq!(backend["fingerprint"], entry["backends"][0]["fingerprint"]);
            assert_eq!(backend["runnable"], true);
            assert_eq!(backend["installed"], false);
            assert_eq!(backend["self_check"], Value::Null);
            let total: u64 = entry["files"]
                .as_array()
                .unwrap()
                .iter()
                .map(|f| f["size_bytes"].as_u64().unwrap())
                .sum();
            assert_eq!(row["download_bytes"], total);
        }
    }
    assert_eq!(
        rows[1]["upstream"]["revision"],
        "f7481e6055501a30fb19d090657df9ec1f79ab2c"
    );
    for (i, params, expected) in [
        (
            101,
            serde_json::json!({"query":" RERANK "}),
            vec!["gte-reranker-modernbert-base", "qwen3-reranker-0.6b"],
        ),
        (
            102,
            serde_json::json!({"runnable_here":true}),
            vec![
                "gte-modernbert-base",
                "gte-reranker-modernbert-base",
                "qwen3-embedding-0.6b",
            ],
        ),
        (
            103,
            serde_json::json!({"runnable_here":false}),
            vec!["qwen3-reranker-0.6b"],
        ),
        (104, serde_json::json!({"installed":true}), vec![]),
        (105, serde_json::json!({"installed":false}), ids.clone()),
        (
            106,
            serde_json::json!({"query":"rerank","runnable_here":true,"task":"rerank"}),
            vec!["gte-reranker-modernbert-base"],
        ),
    ] {
        let result = route_request(
            &mut consumer,
            route,
            i,
            serde_json::json!({"method":"models.catalog","params":params}),
        )
        .await;
        let actual: Vec<_> = result["result"]["models"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["id"].as_str().unwrap())
            .collect();
        assert_eq!(actual, expected);
    }
}

#[tokio::test]
async fn catalog_shared_blob_survives_removal_of_one_entry() {
    let mut h = CatalogHarness::start("ok", None).await;
    for id in ["gte-modernbert-base", "qwen3-embedding-0.6b"] {
        let accepted = h.download(id, id).await;
        let job = accepted["job_id"].as_str().unwrap();
        let done = h.wait_state(job, &["committed", "failed"]).await;
        assert_eq!(done["state"], "committed", "{done}");
    }
    assert_eq!(h.server.paths().len(), 1);
    let first = h
        .call(
            "models.remove",
            serde_json::json!({"catalog_id":"gte-modernbert-base"}),
        )
        .await;
    assert_eq!(first["freed_bytes"], 0);
    let installed = h
        .call("models.catalog", serde_json::json!({"installed":true}))
        .await;
    assert_eq!(installed["models"].as_array().unwrap().len(), 1);
    assert_eq!(installed["models"][0]["id"], "qwen3-embedding-0.6b");
    let last = h
        .call(
            "models.remove",
            serde_json::json!({"catalog_id":"qwen3-embedding-0.6b"}),
        )
        .await;
    assert_eq!(last["freed_bytes"], BODY.len());
    h.assert_clean();
}

#[tokio::test]
async fn catalog_conflicting_legacy_rows_do_not_change_catalog_projections() {
    let mut h = CatalogHarness::start("ok", None).await;
    let accepted = h.download("gte-modernbert-base", "first").await;
    let done = h
        .wait_state(
            accepted["job_id"].as_str().unwrap(),
            &["committed", "failed"],
        )
        .await;
    assert_eq!(done["state"], "committed", "{done}");
    let before = h.call("models.list", serde_json::json!({})).await;
    let admission = h.call("admission.status", serde_json::json!({})).await;
    let conn = Connection::open(expected_store_path(&h.daemon.data_home)).unwrap();
    let fingerprint = before["models"][0]["fingerprints"][0].as_str().unwrap();
    let machine = admission["machine_profile_hash"].as_str().unwrap();
    conn.execute("INSERT INTO cert_rows (certification_class,assurance_class,key_hash,machine_profile_hash,numeric_profile_id,fingerprint,certified_at_ms,os_build,module_generation,evidence_json,status) VALUES ('embedding','measured',?2,?2,'conflict',?1,1,'other-os',1,'{}','uncertified')", params![fingerprint, machine]).unwrap();
    conn.execute("INSERT INTO perf_rows VALUES (?2,'gte-modernbert-base-metal','embed','conflict',?1,'ort',1,'other-os',1,1,1,1,'{}')", params![fingerprint, machine]).unwrap();
    conn.execute("INSERT INTO knob_assignments VALUES (?2,'embed','balanced','gte-modernbert-base-metal','conflict',?1,'ort',1,'other-os',1,1,1)", params![fingerprint, machine]).unwrap();
    conn.execute("INSERT INTO approvals (schema_revision,model_id,decode_fingerprint,enabled,grammar_enabled,disabled_reason,updated_at_ms,evidence_requirements_revision,semantic_digest) VALUES ('conflict','gte-modernbert-base-metal',?1,0,0,'conflicting denial',1,'conflict','conflict')", [fingerprint]).unwrap();
    let after = h.call("models.list", serde_json::json!({})).await;
    assert_eq!(before["models"], after["models"]);
    let after_admission = h.call("admission.status", serde_json::json!({})).await;
    for field in [
        "catalog_lanes",
        "certified_lanes",
        "certification_stale",
        "performance_stale",
    ] {
        assert_eq!(admission[field], after_admission[field], "{field}");
    }
    let report = h.call("probe.report", serde_json::json!({})).await;
    let row = report["lanes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["model_id"] == "gte-modernbert-base-metal")
        .unwrap();
    assert_eq!(row["certification_status"], "not_required");
    assert_eq!(row["self_check"]["state"], "pending");
}

#[tokio::test]
async fn catalog_pre_commit_cancel_has_one_terminal_winner() {
    let root = unique_temp_dir("catalog-commit-barrier");
    let mut h = CatalogHarness::start(
        "ok",
        Some(("SYNAPSE_TEST_DOWNLOAD_PRE_COMMIT_BARRIER", &root)),
    )
    .await;
    let accepted = h.download("gte-modernbert-base", "cancel").await;
    let job = accepted["job_id"].as_str().unwrap().to_owned();
    let until = Instant::now() + Duration::from_secs(15);
    while !root.join(format!("{job}.ready")).exists() {
        assert!(Instant::now() < until);
        sleep(Duration::from_millis(20)).await;
    }
    let remove = h
        .call(
            "models.remove",
            serde_json::json!({"catalog_id":"gte-modernbert-base"}),
        )
        .await;
    assert_eq!(remove["error"]["code"], "model_in_use");
    let verifying = h.wait_state(&job, &["verifying"]).await;
    assert_golden_status(&verifying);
    let mut canceller = connect_consumer(&h.daemon.connection_file_path).await;
    let cancel_route = route_open(&mut canceller, &h.fixture_root, 1).await;
    let cancel_job = job.clone();
    let cancel_task = tokio::spawn(async move {
        route_request(
            &mut canceller,
            cancel_route,
            2,
            serde_json::json!({"method":"models.download.cancel","params":{"job_id":cancel_job}}),
        )
        .await
    });
    std::fs::write(root.join(format!("{job}.release")), b"release").unwrap();
    let cancel = cancel_task.await.unwrap();
    let terminal = h.wait_state(&job, &["cancelled", "committed"]).await;
    assert_eq!(cancel["result"]["state"], terminal["state"]);
    let installed = h
        .call("models.catalog", serde_json::json!({"installed":true}))
        .await;
    if terminal["state"] == "committed" {
        assert_eq!(installed["models"].as_array().unwrap().len(), 1);
        let removed = h
            .call(
                "models.remove",
                serde_json::json!({"catalog_id":"gte-modernbert-base"}),
            )
            .await;
        assert_eq!(removed["freed_bytes"], BODY.len());
    } else {
        assert_eq!(installed["models"], serde_json::json!([]));
    }
    h.assert_clean();
    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn catalog_corrupt_blob_is_quarantined_and_explicit_download_repairs_it() {
    let mut h = CatalogHarness::start("ok", None).await;
    let accepted = h.download("gte-modernbert-base", "first").await;
    let done = h
        .wait_state(
            accepted["job_id"].as_str().unwrap(),
            &["committed", "failed"],
        )
        .await;
    assert_eq!(done["state"], "committed", "{done}");
    let digest = catalog_sha256(BODY);
    let blob = h.fixture_root.join("cache/blobs").join(&digest);
    assert!(
        blob.exists(),
        "download must use the isolated cache override"
    );
    let mut corrupt = BODY.to_vec();
    corrupt[0] ^= 1;
    std::fs::write(&blob, corrupt).unwrap();
    let serve = h
        .call(
            "embed.query",
            serde_json::json!({"model":"gte-modernbert-base","text":"hello"}),
        )
        .await;
    assert_eq!(serve["error"]["code"], "artifact_invalid", "{serve}");
    assert_eq!(serve["error"]["details"]["expected_sha256"], digest);
    assert!(serve["error"]["details"]["file"].is_string());
    assert!(!blob.exists());
    let catalog = h
        .call(
            "models.catalog",
            serde_json::json!({"query":"gte-modernbert-base"}),
        )
        .await;
    assert_eq!(catalog["models"][0]["install_state"], "not_installed");
    h.assert_clean();
    assert_eq!(h.server.paths().len(), 1, "serving never fetches");
    let accepted = h.download("gte-modernbert-base", "repair").await;
    let done = h
        .wait_state(
            accepted["job_id"].as_str().unwrap(),
            &["committed", "failed"],
        )
        .await;
    assert_eq!(done["state"], "committed", "{done}");
    assert_eq!(h.server.paths().len(), 2);
    assert_eq!(std::fs::read(blob).unwrap(), BODY);
}

impl CatalogHarness {
    async fn restart(&mut self) {
        self.restart_with_os(None).await;
    }

    async fn restart_with_os(&mut self, os_build: Option<&str>) {
        self._module.child.kill().await.unwrap();
        self._module.child.wait().await.unwrap();
        let until = Instant::now() + SETUP_TIMEOUT;
        while self
            .daemon
            .registry
            .get_module(MODULE_ID)
            .unwrap()
            .is_some()
        {
            assert!(
                Instant::now() < until,
                "module registration did not disappear"
            );
            sleep(Duration::from_millis(20)).await;
        }
        let catalog = self.fixture_root.join("catalog.json");
        let cache = self.fixture_root.join("cache");
        let config = serde_json::json!({"hf_endpoint":self.server.endpoint}).to_string();
        let mut overrides = vec![
            ("SYNAPSE_TEST_CATALOG", catalog.to_str().unwrap()),
            ("SYNAPSE_TEST_RUNNABLE_BACKENDS", "metal"),
            ("CORTEXKIT_MODEL_CACHE", cache.to_str().unwrap()),
        ];
        if let Some(os_build) = os_build {
            overrides.push(("SYNAPSE_OS_BUILD_OVERRIDE", os_build));
        }
        self._module = spawn_synapse_module_with_env(
            &self.daemon.connection_file_path,
            None,
            Some(&config),
            &overrides,
        );
        wait_for_registration(&self.daemon.registry, MODULE_ID, SETUP_TIMEOUT).await;
        self.consumer = connect_consumer(&self.daemon.connection_file_path).await;
        wait_for_catalog(&mut self.consumer, MODULE_ID, SETUP_TIMEOUT).await;
        self.route = route_open(&mut self.consumer, &self.fixture_root, 1).await;
    }
}

#[tokio::test]
async fn catalog_restart_at_commit_barrier_fails_job_and_reclaims_orphans() {
    let root = unique_temp_dir("catalog-restart-barrier");
    let mut h = CatalogHarness::start(
        "ok",
        Some(("SYNAPSE_TEST_DOWNLOAD_PRE_COMMIT_BARRIER", &root)),
    )
    .await;
    let accepted = h.download("gte-modernbert-base", "restart").await;
    let job = accepted["job_id"].as_str().unwrap().to_owned();
    let until = Instant::now() + Duration::from_secs(15);
    while !root.join(format!("{job}.ready")).exists() {
        assert!(Instant::now() < until);
        sleep(Duration::from_millis(20)).await;
    }
    h.restart().await;
    let failed = h.wait_state(&job, &["failed"]).await;
    assert_eq!(failed["error"]["code"], "module_restarted");
    h.assert_clean();
    assert!(!h
        .fixture_root
        .join("cache/blobs")
        .join(catalog_sha256(BODY))
        .exists());
    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn catalog_committed_install_and_job_survive_restart() {
    let mut h = CatalogHarness::start("ok", None).await;
    let accepted = h.download("gte-modernbert-base", "first").await;
    let job = accepted["job_id"].as_str().unwrap().to_owned();
    let done = h.wait_state(&job, &["committed", "failed"]).await;
    assert_eq!(done["state"], "committed", "{done}");
    h.restart().await;
    assert_eq!(
        h.wait_state(&job, &["committed", "failed"]).await["state"],
        "committed"
    );
    let installed = h
        .call("models.catalog", serde_json::json!({"installed":true}))
        .await;
    assert_eq!(installed["models"][0]["id"], "gte-modernbert-base");
    let again = h.download("gte-modernbert-base", "first").await;
    assert_eq!(again["state"], "committed");
    assert_eq!(h.server.paths().len(), 1);
}

#[tokio::test]
async fn catalog_multi_backend_resolver_refuses_substitution_before_load() {
    let mut catalog: Value =
        serde_json::from_str(include_str!("../../src/catalog/models.json")).unwrap();
    let mut entry = catalog["models"][0].clone();
    entry["id"] = "resolver-fixture".into();
    for file in entry["files"].as_array_mut().unwrap() {
        file["sha256"] = catalog_sha256(BODY).into();
        file["size_bytes"] = BODY.len().into();
    }
    let metal = entry["backends"][0]["fingerprint"]
        .as_str()
        .unwrap()
        .to_owned();
    let ane = "a".repeat(64);
    let mut backend = entry["backends"][0].clone();
    backend["backend"] = "ane".into();
    backend["dtype"] = "f32".into();
    backend["fingerprint"] = ane.clone().into();
    entry["backends"].as_array_mut().unwrap().push(backend);
    let mut ane_files = entry["files"].as_array().unwrap().clone();
    for file in &mut ane_files {
        file["path"] = format!("ane/{}", file["path"].as_str().unwrap()).into();
        file["backends"] = serde_json::json!(["ane"]);
    }
    entry["files"].as_array_mut().unwrap().extend(ane_files);
    catalog["models"][0]["default_for_task"] = false.into();
    catalog["models"].as_array_mut().unwrap().push(entry);
    let mut h =
        CatalogHarness::start_catalog("ok", None, catalog, BTreeMap::new(), "metal,ane", None)
            .await;
    let listing = h.call("models.catalog", serde_json::json!({})).await;
    let resolver = listing["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["id"] == "resolver-fixture")
        .unwrap();
    let rows = resolver["backends"].as_array().unwrap();
    assert_eq!(
        rows.len(),
        2,
        "test catalog override must add the second backend"
    );
    assert_eq!(rows[0]["lane_id"], "resolver-fixture-metal");
    assert_eq!(rows[1]["lane_id"], "resolver-fixture-ane");
    assert_eq!(rows[0]["runnable"], true);
    assert_eq!(rows[1]["runnable"], true);
    for (model, required, target) in [
        ("resolver-fixture", "unknown".to_owned(), None),
        ("resolver-fixture-metal", ane.clone(), None),
        ("resolver-fixture", metal.clone(), Some(ane.clone())),
    ] {
        let mut params =
            serde_json::json!({"model":model,"text":"hello","required_fingerprint":required});
        if let Some(target) = target {
            params["target_fingerprint"] = target.into();
        }
        let refused = h.call("embed.query", params).await;
        assert_eq!(
            refused["error"]["code"], "substitution_rejected",
            "{refused}"
        );
        assert!(refused["error"]["message"].is_string());
        assert_eq!(
            refused["error"]["details"]["required_fingerprint"],
            required
        );
    }
    for (pin, selected) in [
        (None, "resolver-fixture-metal"),
        (Some(ane), "resolver-fixture-ane"),
    ] {
        let mut params = serde_json::json!({"model":"resolver-fixture","text":"hello"});
        if let Some(pin) = pin {
            params["required_fingerprint"] = pin.into();
        }
        let refused = h.call("embed.query", params).await;
        assert_eq!(refused["error"]["code"], "model_not_installed");
        assert_eq!(refused["error"]["details"]["lane_id"], selected);
    }
    assert!(h.server.paths().is_empty());
    let models = h.call("models.list", serde_json::json!({})).await;
    assert_eq!(models["models"], serde_json::json!([]));
}

#[cfg(target_os = "macos")]
fn real_catalog_fixture(id: &str) -> Option<(Value, BTreeMap<String, PathBuf>)> {
    let mut catalog: Value =
        serde_json::from_str(include_str!("../../src/catalog/models.json")).unwrap();
    let entry = catalog["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["id"] == id)
        .unwrap()
        .clone();
    let searched = PathBuf::from(std::env::var("HOME").unwrap()).join(format!(
        ".cache/huggingface/hub/models--{}/snapshots",
        entry["upstream"]["hf_repo"]
            .as_str()
            .unwrap()
            .replace('/', "--")
    ));
    let (snapshot, searched) = match id {
        "gte-modernbert-base" => (
            gte_safetensors_snapshot(),
            std::env::var("SYNAPSE_GTE_MODERNBERT_SAFETENSORS_SNAPSHOT")
                .map(PathBuf::from)
                .unwrap_or(searched),
        ),
        "gte-reranker-modernbert-base" => (
            gte_reranker_safetensors_snapshot(),
            std::env::var("SYNAPSE_GTE_RERANKER_MODERNBERT_SAFETENSORS_SNAPSHOT")
                .map(PathBuf::from)
                .unwrap_or(searched),
        ),
        _ => (
            first_snapshot_with(&searched, "model.safetensors"),
            searched,
        ),
    };
    let Some(snapshot) = snapshot else {
        eprintln!(
            "skipping Metal catalog serve: no safetensors snapshot found under {}",
            searched.display()
        );
        return None;
    };
    let mut files = BTreeMap::new();
    for file in entry["files"].as_array().unwrap() {
        let local = snapshot.join(file["path"].as_str().unwrap());
        if !local.is_file() {
            eprintln!(
                "skipping Metal catalog serve: missing {} (snapshot search {})",
                local.display(),
                searched.display()
            );
            return None;
        }
        assert_eq!(
            std::fs::metadata(&local).unwrap().len(),
            file["size_bytes"].as_u64().unwrap(),
            "{} must match the pinned catalog artifact",
            local.display()
        );
        assert_eq!(
            catalog_sha256(&std::fs::read(&local).unwrap()),
            file["sha256"].as_str().unwrap(),
            "{} must match the pinned catalog artifact",
            local.display()
        );
        let path = format!(
            "/{}/resolve/{}/{}",
            entry["upstream"]["hf_repo"].as_str().unwrap(),
            entry["upstream"]["revision"].as_str().unwrap(),
            file["path"].as_str().unwrap()
        );
        files.insert(path, local);
    }
    catalog["catalog_revision"] = 717.into();
    Some((catalog, files))
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn catalog_real_metal_redirect_download_self_checks_and_serves_without_probe() {
    let Some((catalog, files)) = real_catalog_fixture("gte-modernbert-base") else {
        return;
    };
    let expected_fingerprint = catalog["models"][0]["backends"][0]["fingerprint"].clone();
    let mut h =
        CatalogHarness::start_catalog("redirect", None, catalog, files, "metal", None).await;
    assert_eq!(
        h.call("models.catalog", serde_json::json!({})).await["catalog_revision"],
        717
    );
    let accepted = h.download("gte-modernbert-base", "real").await;
    let job = accepted["job_id"].as_str().unwrap();
    let done = h.wait_state(job, &["committed", "failed"]).await;
    assert_eq!(done["state"], "committed", "{done}");
    let before = h.call("models.list", serde_json::json!({})).await;
    assert_eq!(
        before["models"][0]["self_check"]["state"], "pending",
        "{before}"
    );
    assert_eq!(before["models"][0]["certified"], false);
    let requests = h.server.paths().len();
    for model in ["gte-modernbert-base", "gte-modernbert-base-metal"] {
        let served = h.serve_when_ready("embed.query", serde_json::json!({"model":model,"text":"The quick brown fox jumps over the lazy dog.","deadline_ms":30000})).await;
        assert_eq!(served["fingerprint"], expected_fingerprint, "{served}");
        assert_eq!(served["dims"], 768);
    }
    assert_eq!(h.server.paths().len(), requests, "serving must not fetch");
    let listed = h.call("models.list", serde_json::json!({})).await;
    assert_eq!(listed["models"][0]["certified"], true);
    assert_eq!(listed["models"][0]["self_check"]["state"], "passed");
    let status = h.call("admission.status", serde_json::json!({})).await;
    assert_eq!(status["catalog_lanes"], 1);
    assert_eq!(status["lanes"][0]["certification_required"], false);
    assert_eq!(status["lanes"][0]["certification_status"], "not_required");
    h.restart_with_os(Some("catalog-test-changed-os-build"))
        .await;
    let pending = h.call("models.list", serde_json::json!({})).await;
    assert_eq!(
        pending["models"][0]["self_check"]["state"], "pending",
        "{pending}"
    );
    let served_again = h
        .serve_when_ready(
            "embed.query",
            serde_json::json!({"model":"gte-modernbert-base","text":"hello","deadline_ms":30000}),
        )
        .await;
    assert_eq!(
        served_again["fingerprint"], expected_fingerprint,
        "{served_again}"
    );
    assert_eq!(
        h.call("models.list", serde_json::json!({})).await["models"][0]["self_check"]["state"],
        "passed"
    );
    let loaded_remove = h
        .call(
            "models.remove",
            serde_json::json!({"catalog_id":"gte-modernbert-base"}),
        )
        .await;
    assert_eq!(loaded_remove["error"]["code"], "model_in_use");
    assert_eq!(
        h.call(
            "model.unload",
            serde_json::json!({"model_id":"gte-modernbert-base-metal"})
        )
        .await["state"],
        "unloaded"
    );
    let cache = h.fixture_root.join("cache");
    let before_remove = directory_bytes(&cache.join("blobs"))
        + directory_bytes(&cache.join("owned-metal-packages"))
        + directory_bytes(&cache.join("owned-metal-models"));
    let removed = h
        .call(
            "models.remove",
            serde_json::json!({"catalog_id":"gte-modernbert-base"}),
        )
        .await;
    let after_remove = directory_bytes(&cache.join("blobs"))
        + directory_bytes(&cache.join("owned-metal-packages"))
        + directory_bytes(&cache.join("owned-metal-models"));
    assert_eq!(removed["freed_bytes"], before_remove - after_remove);
    assert_eq!(directory_bytes(&cache.join("owned-metal-models")), 0);
    assert_eq!(directory_bytes(&cache.join("blobs")), 0);
    assert_eq!(directory_bytes(&cache.join("owned-metal-packages")), 0);
    assert_eq!(
        h.call("models.list", serde_json::json!({})).await["models"],
        serde_json::json!([])
    );
    h.server.assert_pinned_requests();
}

fn directory_bytes(path: &Path) -> u64 {
    if !path.exists() {
        return 0;
    }
    std::fs::read_dir(path)
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            let metadata = entry.metadata().unwrap();
            if metadata.is_dir() {
                directory_bytes(&entry.path())
            } else {
                metadata.len()
            }
        })
        .sum()
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn catalog_real_metal_stalled_self_check_keeps_single_flight_holder() {
    let Some((catalog, files)) = real_catalog_fixture("gte-modernbert-base") else {
        return;
    };
    let mut h = CatalogHarness::start_catalog(
        "ok",
        None,
        catalog,
        files,
        "metal",
        Some("self_check:stall"),
    )
    .await;
    let accepted = h.download("gte-modernbert-base", "real").await;
    let done = h
        .wait_state(
            accepted["job_id"].as_str().unwrap(),
            &["committed", "failed"],
        )
        .await;
    assert_eq!(done["state"], "committed", "{done}");
    let served = h
        .serve_when_ready(
            "embed.query",
            serde_json::json!({"model":"gte-modernbert-base","text":"hello","deadline_ms":30000}),
        )
        .await;
    assert_eq!(
        served["error"]["code"], "engine_crashed",
        "stall override must time out the numerical check: {served}"
    );
    assert_eq!(served["error"]["retry_after_ms"], 250);
    let listed = h.call("models.list", serde_json::json!({})).await;
    assert_eq!(listed["models"][0]["self_check"]["state"], "pending");
    let unload = h
        .call(
            "model.unload",
            serde_json::json!({"model_id":"gte-modernbert-base-metal"}),
        )
        .await;
    assert_eq!(unload["error"]["code"], "model_in_use");
    assert!(unload["error"]["details"]["holders"]
        .as_array()
        .unwrap()
        .iter()
        .any(|holder| holder["kind"] == "self_check"));
    let started = Instant::now();
    let second = h
        .call(
            "embed.query",
            serde_json::json!({"model":"gte-modernbert-base","text":"hello","deadline_ms":200}),
        )
        .await;
    assert_eq!(second["error"]["code"], "deadline_exceeded");
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "a second numerical check must not start while one remains in flight"
    );
    sleep(Duration::from_secs(11)).await;
    let late = h.call("models.list", serde_json::json!({})).await;
    assert_eq!(
        late["models"][0]["self_check"]["state"], "pending",
        "late check must not overwrite its invalidated generation"
    );
}

#[tokio::test]
async fn catalog_no_body_byte_timeout_fails_and_cleans_download() {
    let mut h = CatalogHarness::start("stall", None).await;
    let accepted = h.download("gte-modernbert-base", "idle-timeout").await;
    let job = accepted["job_id"].as_str().unwrap();
    h.wait_state(job, &["downloading"]).await;
    sleep(Duration::from_secs(61)).await;
    let failed = h.wait_state(job, &["failed"]).await;
    assert_eq!(failed["error"]["code"], "download_failed");
    assert_eq!(failed["error"]["details"]["reason"], "network");
    h.assert_clean();
    h.server.assert_pinned_requests();
}

#[cfg(not(target_os = "macos"))]
#[tokio::test]
async fn catalog_native_non_macos_lists_metal_as_unsupported_without_override() {
    let (_daemon, _module, mut consumer, route) = open_route().await;
    let result = route_request(
        &mut consumer,
        route,
        100,
        serde_json::json!({"method":"models.catalog","params":{}}),
    )
    .await;
    for row in result["result"]["models"].as_array().unwrap() {
        assert_eq!(row["download_bytes"], 0);
        for backend in row["backends"].as_array().unwrap() {
            assert_eq!(backend["runnable"], false);
            assert_eq!(backend["reason"], "not_supported_on_platform");
            assert_eq!(backend["installed"], false);
            assert_eq!(backend["self_check"], Value::Null);
        }
    }
    let runnable = route_request(
        &mut consumer,
        route,
        101,
        serde_json::json!({"method":"models.catalog","params":{"runnable_here":true}}),
    )
    .await;
    assert_eq!(runnable["result"]["models"], serde_json::json!([]));
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn catalog_real_qwen_embedding_serves_catalog_and_lane_ids_then_unloads() {
    let Some((catalog, files)) = real_catalog_fixture("qwen3-embedding-0.6b") else {
        return;
    };
    let fingerprint = catalog["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["id"] == "qwen3-embedding-0.6b")
        .unwrap()["backends"][0]["fingerprint"]
        .clone();
    let mut h = CatalogHarness::start_catalog("ok", None, catalog, files, "metal", None).await;
    let accepted = h.download("qwen3-embedding-0.6b", "real").await;
    let done = h
        .wait_state(
            accepted["job_id"].as_str().unwrap(),
            &["committed", "failed"],
        )
        .await;
    assert_eq!(done["state"], "committed", "{done}");
    let cold_started = Instant::now();
    for model in ["qwen3-embedding-0.6b-metal", "qwen3-embedding-0.6b"] {
        let served = h
            .serve_when_ready(
                "embed.query",
                serde_json::json!({"model":model,"text":"hello","deadline_ms":30000}),
            )
            .await;
        assert_eq!(served["fingerprint"], fingerprint, "{served}");
        assert_eq!(served["dims"], 1024);
    }
    eprintln!(
        "Qwen first cold load wall time: {:?}",
        cold_started.elapsed()
    );
    let unloaded = h
        .call(
            "model.unload",
            serde_json::json!({"model_id":"qwen3-embedding-0.6b-metal"}),
        )
        .await;
    assert_eq!(unloaded["state"], "unloaded");
    let load = std::process::Command::new("uptime").output().unwrap();
    eprintln!(
        "Qwen second cold load machine load: {}",
        String::from_utf8_lossy(&load.stdout)
    );
    let cold_started = Instant::now();
    let served = h.serve_when_ready("embed.query", serde_json::json!({"model":"qwen3-embedding-0.6b-metal","text":"hello","deadline_ms":30000})).await;
    eprintln!(
        "Qwen second cold load wall time: {:?}",
        cold_started.elapsed()
    );
    assert_eq!(served["fingerprint"], fingerprint, "{served}");
    assert_eq!(served["dims"], 1024);
    let unloaded = h
        .call(
            "model.unload",
            serde_json::json!({"model_id":"qwen3-embedding-0.6b-metal"}),
        )
        .await;
    assert_eq!(unloaded["state"], "unloaded");
    h.server.assert_pinned_requests();
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn catalog_real_reranker_loaded_metadata_is_pinned() {
    let Some((catalog, files)) = real_catalog_fixture("gte-reranker-modernbert-base") else {
        return;
    };
    let mut h = CatalogHarness::start_catalog("ok", None, catalog, files, "metal", None).await;
    let accepted = h.download("gte-reranker-modernbert-base", "real").await;
    let done = h
        .wait_state(
            accepted["job_id"].as_str().unwrap(),
            &["committed", "failed"],
        )
        .await;
    assert_eq!(done["state"], "committed", "{done}");
    let served = h.serve_when_ready("rerank.score", serde_json::json!({"model":"gte-reranker-modernbert-base","query":"What is Rust?","candidates":["Rust is a systems programming language."],"deadline_ms":30000})).await;
    assert!(served.get("error").is_none(), "{served}");
    let listed = h.call("models.list", serde_json::json!({})).await;
    let row = &listed["models"][0];
    assert_eq!(row["model_id"], "gte-reranker-modernbert-base-metal");
    assert_eq!(row["fingerprints"][0].as_str().unwrap().len(), 64);
    assert_eq!(row["dims"], 768);
    assert_eq!(row["max_tokens"], 8192);
    assert_eq!(row["max_tokens_source"], "runtime_bucket");
    assert_eq!(row["device_class"], "metal");
    assert_eq!(row["dtype"], "f32");
    h.server.assert_pinned_requests();
}

#[tokio::test]
async fn catalog_endpoint_with_path_refuses_module_start_by_name() {
    let daemon = start_daemon().await;
    let server = CatalogServer::start("ok", BTreeMap::new()).await;
    let config = serde_json::json!({"hf_endpoint":format!("{}/forbidden-path", server.endpoint)})
        .to_string();
    let mut command =
        synapse_module_command(&daemon.connection_file_path, None, Some(&config), &[]);
    command.stderr(process::Stdio::piped());
    let output = tokio::time::timeout(Duration::from_secs(15), command.output())
        .await
        .expect("invalid endpoint must refuse startup")
        .unwrap();
    let exit = super::common::describe_exit_status(&output.status);
    assert!(
        !output.status.success(),
        "module unexpectedly accepted an invalid endpoint ({exit})"
    );
    assert!(
        output.status.code().is_some(),
        "invalid endpoint should be reported as a startup refusal, not {exit}"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("hf_endpoint"),
        "module startup refusal ({exit}); stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(server.paths().is_empty());
}

#[tokio::test]
async fn catalog_free_form_load_refusals_happen_before_network_io() {
    let mut h = CatalogHarness::start("ok", None).await;
    let base = serde_json::json!({"source":"hf","repo":"fixture/model","files":{"model":"model.safetensors","tokenizer":"tokenizer.json"},"engine":"owned-metal","task":"embed","model_id":"free-form","family":"gte-modernbert","dtype":"f16","execution":"explicit","pooling":"cls"});
    let mut requests = vec![base.clone()];
    let mut main = base.clone();
    main["revision"] = "main".into();
    requests.push(main);
    for engine in ["ort", "llama"] {
        let mut request = base.clone();
        request["engine"] = engine.into();
        request["revision"] = "a".repeat(40).into();
        requests.push(request);
    }
    let mut reserved = base.clone();
    reserved["model_id"] = "gte-modernbert-base".into();
    reserved["revision"] = "a".repeat(40).into();
    requests.push(reserved);
    let mut main_url = base.clone();
    main_url["revision"] = "a".repeat(40).into();
    main_url["files"]["model"] = serde_json::json!({"url":format!("{}/fixture/model/resolve/main/model.safetensors", h.server.endpoint), "sha256":catalog_sha256(BODY)});
    requests.push(main_url);
    let mut unpinned_url = base.clone();
    unpinned_url["source"] = "file".into();
    unpinned_url["path"] = h.fixture_root.to_str().unwrap().into();
    unpinned_url["files"]["model"] =
        serde_json::json!({"url":format!("{}/file",h.server.endpoint)});
    requests.push(unpinned_url);
    for params in requests {
        h.next_id += 1;
        let frame = raw_route_frame(
            &mut h.consumer,
            h.route,
            h.next_id,
            serde_json::json!({"method":"model.load","params":params}),
        )
        .await;
        let body: Value = serde_json::from_slice(&frame.body).unwrap();
        let error = if body["result"]["error"].is_object() {
            &body["result"]["error"]
        } else if body["error"].is_object() {
            &body["error"]
        } else {
            &body
        };
        assert_eq!(error["code"], "invalid_request", "{body}");
        assert!(error["message"].is_string());
    }
    assert!(h.server.paths().is_empty());
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn catalog_real_numerical_failure_persists_and_pinned_requests_fail_closed() {
    let Some((mut catalog, files)) = real_catalog_fixture("gte-modernbert-base") else {
        return;
    };
    let fingerprint = catalog["models"][0]["backends"][0]["fingerprint"].clone();
    for vector in catalog["models"][0]["self_check"]["reference"]["vectors"]
        .as_array_mut()
        .unwrap()
    {
        for component in vector.as_array_mut().unwrap() {
            *component = 0.0.into();
        }
    }
    let mut h = CatalogHarness::start_catalog("ok", None, catalog, files, "metal", None).await;
    let accepted = h.download("gte-modernbert-base", "perturbed").await;
    let done = h
        .wait_state(
            accepted["job_id"].as_str().unwrap(),
            &["committed", "failed"],
        )
        .await;
    assert_eq!(done["state"], "committed", "{done}");
    let refused = h
        .serve_when_ready(
            "embed.query",
            serde_json::json!({"model":"gte-modernbert-base","text":"hello","deadline_ms":30000}),
        )
        .await;
    assert_eq!(
        refused["error"]["code"], "self_check_failed",
        "perturbed reference must fail: {refused}"
    );
    assert_eq!(
        refused["error"]["details"]["check_id"]
            .as_str()
            .unwrap()
            .len(),
        64
    );
    let listed = h.call("models.list", serde_json::json!({})).await;
    assert_eq!(listed["models"][0]["self_check"]["state"], "failed");
    assert_eq!(listed["models"][0]["serving_admission"], "disabled");
    assert_eq!(
        listed["models"][0]["serving_admission_reason"],
        "self_check_failed"
    );
    h.restart().await;
    let pinned = h.call("embed.query", serde_json::json!({"model":"gte-modernbert-base-metal","required_fingerprint":fingerprint,"text":"hello","deadline_ms":30000})).await;
    assert_eq!(pinned["error"]["code"], "self_check_failed");
    assert_eq!(
        pinned["error"]["details"]["check_id"],
        refused["error"]["details"]["check_id"]
    );
}

async fn wait_barrier(root: &Path, job: &str) {
    let until = Instant::now() + Duration::from_secs(15);
    while !root.join(format!("{job}.ready")).exists() {
        assert!(
            Instant::now() < until,
            "download did not reach {}",
            root.display()
        );
        sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test]
async fn catalog_reused_acquisition_survives_other_publishers_cancellation() {
    let pre_publish = unique_temp_dir("catalog-shared-publish");
    let pre_commit = unique_temp_dir("catalog-shared-commit");
    let mut catalog: Value =
        serde_json::from_str(include_str!("../../src/catalog/models.json")).unwrap();
    for entry in catalog["models"].as_array_mut().unwrap() {
        for file in entry["files"].as_array_mut().unwrap() {
            file["sha256"] = catalog_sha256(BODY).into();
            file["size_bytes"] = BODY.len().into();
        }
    }
    let mut h = CatalogHarness::start_catalog_env(
        "ok",
        Some(("SYNAPSE_TEST_DOWNLOAD_PRE_COMMIT_BARRIER", &pre_commit)),
        catalog,
        BTreeMap::new(),
        "metal",
        None,
        &[(
            "SYNAPSE_TEST_DOWNLOAD_PRE_PUBLISH_BARRIER",
            pre_publish.to_str().unwrap(),
        )],
    )
    .await;
    let a = h.download("gte-modernbert-base", "a").await;
    let a = a["job_id"].as_str().unwrap().to_owned();
    wait_barrier(&pre_publish, &a).await;
    std::fs::write(pre_publish.join(format!("{a}.release")), b"release").unwrap();
    wait_barrier(&pre_commit, &a).await;
    let b = h.download("qwen3-embedding-0.6b", "b").await;
    let b = b["job_id"].as_str().unwrap().to_owned();
    wait_barrier(&pre_publish, &b).await;
    let conn = Connection::open(expected_store_path(&h.daemon.data_home)).unwrap();
    let acquired: u64 = conn
        .query_row(
            "SELECT count(*) FROM download_acquisitions WHERE job_id=?1 AND newly_published=0",
            [&b],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        acquired, 1,
        "reuse hook must pause after rooting the shared blob"
    );
    let cancel = h
        .call("models.download.cancel", serde_json::json!({"job_id":a}))
        .await;
    assert_eq!(cancel["state"], "cancelled");
    let blob = h
        .fixture_root
        .join("cache/blobs")
        .join(catalog_sha256(BODY));
    assert!(
        blob.exists(),
        "another acquisition must keep the published blob rooted"
    );
    std::fs::write(pre_publish.join(format!("{b}.release")), b"release").unwrap();
    wait_barrier(&pre_commit, &b).await;
    std::fs::write(pre_commit.join(format!("{b}.release")), b"release").unwrap();
    std::fs::write(pre_commit.join(format!("{a}.release")), b"release").unwrap();
    let done = h.wait_state(&b, &["committed", "failed"]).await;
    assert_eq!(done["state"], "committed", "{done}");
    let members: u64 = conn
        .query_row(
            "SELECT count(*) FROM catalog_install_members WHERE catalog_id='qwen3-embedding-0.6b'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(members, 3);
    assert_eq!(
        h.server.paths().len(),
        1,
        "shared digest must not be fetched twice"
    );
    assert_eq!(std::fs::read(blob).unwrap(), BODY);
    let _ = std::fs::remove_dir_all(pre_publish);
    let _ = std::fs::remove_dir_all(pre_commit);
}

#[tokio::test]
async fn catalog_unavailable_backend_is_listed_but_never_downloaded() {
    let catalog: Value =
        serde_json::from_str(include_str!("../../src/catalog/models.json")).unwrap();
    let mut h = CatalogHarness::start_catalog("ok", None, catalog, BTreeMap::new(), "", None).await;
    let listing = h.call("models.catalog", serde_json::json!({})).await;
    assert_eq!(listing["models"].as_array().unwrap().len(), 4);
    for entry in listing["models"].as_array().unwrap() {
        assert_eq!(entry["install_state"], "not_installed");
        for backend in entry["backends"].as_array().unwrap() {
            assert_eq!(backend["installed"], false);
            assert_eq!(backend["runnable"], false);
            assert_eq!(
                backend["reason"],
                if cfg!(target_os = "macos") {
                    "device_missing"
                } else {
                    "not_supported_on_platform"
                }
            );
        }
    }
    let refused = h.download("gte-modernbert-base", "unavailable").await;
    assert_eq!(refused["error"]["code"], "backend_unavailable", "{refused}");
    let refused = h
        .call(
            "embed.query",
            serde_json::json!({"model":"gte-modernbert-base","text":"hello"}),
        )
        .await;
    assert_eq!(refused["error"]["code"], "backend_unavailable", "{refused}");
    assert!(h.server.paths().is_empty());
}

#[cfg(target_os = "macos")]
async fn wait_failed_catalog_load(h: &mut CatalogHarness, lane: &str) -> Value {
    let until = Instant::now() + Duration::from_secs(180);
    loop {
        let status = h
            .call("model.status", serde_json::json!({"model_id":lane}))
            .await;
        if status["state"] == "failed" {
            return status;
        }
        assert!(Instant::now() < until, "load did not fail: {status}");
        sleep(Duration::from_millis(20)).await;
    }
}

// Needs a real model so the load gets far enough to fail in the engine; the
// owned engine and its snapshots exist only on macOS.
#[cfg(target_os = "macos")]
#[tokio::test]
async fn catalog_failed_load_after_inline_timeout_reaches_next_request_and_retry_is_single_flight()
{
    let Some((catalog, files)) = real_catalog_fixture("gte-modernbert-base") else {
        return;
    };
    let root = unique_temp_dir("catalog-load-attempts");
    std::fs::create_dir_all(&root).unwrap();
    let counter = root.join("attempts");
    let mut h = CatalogHarness::start_catalog_env(
        "ok",
        None,
        catalog,
        files,
        "metal",
        Some("load:stall-crash"),
        &[(
            "SYNAPSE_TEST_CATALOG_LOAD_ATTEMPTS",
            counter.to_str().unwrap(),
        )],
    )
    .await;
    let download = h.download("gte-modernbert-base", "load-failure").await;
    let done = h
        .wait_state(
            download["job_id"].as_str().unwrap(),
            &["committed", "failed"],
        )
        .await;
    assert_eq!(done["state"], "committed", "{done}");
    let request =
        serde_json::json!({"model":"gte-modernbert-base", "text":"hello", "deadline_ms":30000});
    let loading = h.call("embed.query", request.clone()).await;
    assert_eq!(loading["error"]["code"], "model_loading", "{loading}");
    let failed = wait_failed_catalog_load(&mut h, "gte-modernbert-base-metal").await;
    assert_eq!(failed["error"]["code"], "engine_crashed");
    assert_eq!(
        std::fs::read_to_string(&counter).unwrap().lines().count(),
        1
    );
    let observed = h.call("embed.query", request.clone()).await;
    assert_eq!(
        observed["error"]["code"], "engine_crashed",
        "a completed failure must not become model_loading: {observed}"
    );
    assert_eq!(observed["error"]["class"], "transient");
    assert_eq!(observed["error"]["retry_after_ms"], 250);
    assert_eq!(observed["error"]["safe_to_retry_same_request"], true);
    assert_eq!(
        observed["error"]["message"],
        "injected catalog engine crash"
    );
    assert_eq!(
        std::fs::read_to_string(&counter).unwrap().lines().count(),
        1,
        "reporting the stored error must not start a new load"
    );
    let mut clients = Vec::new();
    for index in 0..3 {
        let mut client = connect_consumer(&h.daemon.connection_file_path).await;
        wait_for_catalog(&mut client, MODULE_ID, SETUP_TIMEOUT).await;
        let route = route_open(&mut client, &h.fixture_root, index + 1).await;
        clients.push((client, route));
    }
    let mut waiters = Vec::new();
    for (mut client, route) in clients {
        let params = request.clone();
        waiters.push(tokio::spawn(async move {
            route_request(
                &mut client,
                route,
                100,
                serde_json::json!({"method":"embed.query", "params":params}),
            )
            .await["result"]
                .clone()
        }));
    }
    for waiter in waiters {
        let result = waiter.await.unwrap();
        assert_eq!(result["error"]["code"], "model_loading", "{result}");
    }
    wait_failed_catalog_load(&mut h, "gte-modernbert-base-metal").await;
    assert_eq!(
        std::fs::read_to_string(&counter).unwrap().lines().count(),
        2,
        "concurrent retry waiters must share one attempt"
    );
    let observed = h.call("embed.query", request).await;
    assert_eq!(observed["error"]["code"], "engine_crashed", "{observed}");
    assert_eq!(
        std::fs::read_to_string(&counter).unwrap().lines().count(),
        2
    );
    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn catalog_shared_member_keeps_derived_packages_until_last_root_is_removed() {
    let mut h = CatalogHarness::start("ok", None).await;
    let path = h.fixture_root.join("catalog.json");
    let mut catalog: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let mut shared = catalog["models"][0].clone();
    shared["id"] = "shared-package".into();
    shared["default_for_task"] = false.into();
    catalog["models"].as_array_mut().unwrap().push(shared);
    std::fs::write(&path, catalog.to_string()).unwrap();
    h.restart().await;
    for (id, key) in [("gte-modernbert-base", "a"), ("shared-package", "b")] {
        let accepted = h.download(id, key).await;
        let done = h
            .wait_state(
                accepted["job_id"].as_str().unwrap(),
                &["committed", "failed"],
            )
            .await;
        assert_eq!(done["state"], "committed", "{done}");
    }
    let cache = h.fixture_root.join("cache");
    let packages = cache.join("owned-metal-models");
    for key in ["a".repeat(64), "b".repeat(64)] {
        let package = packages.join(key);
        std::fs::create_dir_all(&package).unwrap();
        std::fs::write(package.join("model.safetensors"), BODY).unwrap();
        std::fs::write(package.join("config.json"), b"{}").unwrap();
        std::fs::write(
            package.join("role-digests.json"),
            serde_json::to_vec(&vec![catalog_sha256(BODY)]).unwrap(),
        )
        .unwrap();
        #[cfg(target_os = "macos")]
        {
            let canonical = std::fs::canonicalize(package.join("model.safetensors")).unwrap();
            let mut hash = 1469598103934665603u64;
            for byte in canonical.to_string_lossy().as_bytes() {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x100000001b3);
            }
            let compiled = cache.join("owned-metal-packages").join(format!(
                "gte-modernbert-graph-v1-bucket-policy-v1-{hash:016x}-f16-test"
            ));
            std::fs::create_dir_all(&compiled).unwrap();
            std::fs::write(compiled.join("kernel"), b"compiled").unwrap();
        }
    }
    let derived = directory_bytes(&packages) + directory_bytes(&cache.join("owned-metal-packages"));
    let removed = h
        .call(
            "models.remove",
            serde_json::json!({"catalog_id":"gte-modernbert-base"}),
        )
        .await;
    assert_eq!(removed["freed_bytes"], 0, "{removed}");
    assert_eq!(
        directory_bytes(&packages) + directory_bytes(&cache.join("owned-metal-packages")),
        derived
    );
    let before = derived + directory_bytes(&cache.join("blobs"));
    let removed = h
        .call(
            "models.remove",
            serde_json::json!({"catalog_id":"shared-package"}),
        )
        .await;
    let after = directory_bytes(&packages)
        + directory_bytes(&cache.join("owned-metal-packages"))
        + directory_bytes(&cache.join("blobs"));
    assert_eq!(removed["freed_bytes"], before - after, "{removed}");
    assert_eq!(
        after, 0,
        "last root must reclaim both packages and their compiled entries"
    );
}

#[tokio::test]
async fn catalog_startup_sweeps_renamed_package_removal_orphans() {
    let mut h = CatalogHarness::start("ok", None).await;
    let orphan = h
        .fixture_root
        .join("cache/owned-metal-models")
        .join(format!(".{}.removing-0-1", "a".repeat(64)));
    std::fs::create_dir_all(&orphan).unwrap();
    std::fs::write(orphan.join("model.safetensors"), BODY).unwrap();
    h.restart().await;
    assert!(!orphan.exists());
}
