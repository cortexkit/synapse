use super::*;
use std::sync::Mutex;
use tokio::sync::Notify;

const BODY: &[u8] = b"fixture catalog artifact bytes";

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
        for path in paths.iter().filter(|p| p.contains("/resolve/")) {
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
        let server = CatalogServer::start(mode).await;
        let root = unique_temp_dir("catalog-fixture");
        std::fs::create_dir_all(&root).unwrap();
        let mut catalog: Value =
            serde_json::from_str(include_str!("../../src/catalog/models.json")).unwrap();
        // The files share a digest to exercise distinct-blob accounting without downloading release weights.
        for entry in catalog["models"].as_array_mut().unwrap() {
            for file in entry["files"].as_array_mut().unwrap() {
                file["sha256"] = catalog_sha256(BODY).into();
                file["size_bytes"] = BODY.len().into();
            }
        }
        let catalog_path = root.join("catalog.json");
        std::fs::write(&catalog_path, catalog.to_string()).unwrap();
        let config = serde_json::json!({"hf_endpoint":server.endpoint}).to_string();
        let daemon = start_daemon().await;
        let path = catalog_path.to_str().unwrap();
        let cache = root.join("cache");
        let mut overrides = vec![
            ("SYNAPSE_TEST_CATALOG", path),
            ("SYNAPSE_TEST_RUNNABLE_BACKENDS", "metal"),
            ("CORTEXKIT_MODEL_CACHE", cache.to_str().unwrap()),
        ];
        if let Some((name, path)) = barrier {
            overrides.push((name, path.to_str().unwrap()));
        }
        let module = spawn_synapse_module_with_env(
            &daemon.connection_file_path,
            None,
            Some(&config),
            &overrides,
        );
        let (daemon, module, consumer, route) = open_route_for_started_module(daemon, module).await;
        let mut harness = Self {
            daemon,
            _module: module,
            consumer,
            route,
            next_id: 100,
            server,
            fixture_root: root,
        };
        let catalog = harness
            .call(
                "models.catalog",
                serde_json::json!({"query":"gte-modernbert-base"}),
            )
            .await;
        assert_eq!(
            catalog["models"][0]["download_bytes"],
            BODY.len(),
            "catalog and runnable-backend overrides must be honored"
        );
        assert_eq!(catalog["models"][0]["backends"][0]["runnable"], true);
        harness
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

    async fn download(&mut self, id: &str, key: &str) -> Value {
        self.call(
            "models.download",
            serde_json::json!({"catalog_id":id,"request_key":key}),
        )
        .await
    }

    async fn wait_state(&mut self, job: &str, states: &[&str]) -> Value {
        let until = Instant::now() + Duration::from_secs(15);
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
    let progress = h.wait_state(&job, &["downloading"]).await;
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
    let wrong_task = h.call("rerank.score", serde_json::json!({"model":"gte-modernbert-base","query":"hello","candidates":[{"id":"1","text":"hello"}]})).await;
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
    conn.execute("INSERT INTO cert_rows (assurance_class,key_hash,machine_profile_hash,numeric_profile_id,fingerprint,certified_at_ms,os_build,module_generation,evidence_json,status) VALUES ('measured','conflict','conflict','conflict',?1,1,'other-os',1,'{}','uncertified')", [fingerprint]).unwrap();
    conn.execute("INSERT INTO perf_rows VALUES ('conflict','gte-modernbert-base-metal','embed','conflict',?1,'ort',1,'other-os',1,1,1,1,'{}')", [fingerprint]).unwrap();
    conn.execute("INSERT INTO knob_assignments VALUES ('conflict','embed','interactive','gte-modernbert-base-metal','conflict',?1,'ort',1,'other-os',1,1,1)", [fingerprint]).unwrap();
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
    let cancel = h
        .call("models.download.cancel", serde_json::json!({"job_id":job}))
        .await;
    assert_eq!(cancel["state"], "cancelled");
    std::fs::write(root.join(format!("{job}.release")), b"release").unwrap();
    h.wait_state(&job, &["cancelled"]).await;
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
        self._module = spawn_synapse_module_with_env(
            &self.daemon.connection_file_path,
            None,
            Some(&config),
            &[
                ("SYNAPSE_TEST_CATALOG", catalog.to_str().unwrap()),
                ("SYNAPSE_TEST_RUNNABLE_BACKENDS", "metal"),
                ("CORTEXKIT_MODEL_CACHE", cache.to_str().unwrap()),
            ],
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
