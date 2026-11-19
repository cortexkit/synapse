//! Regression tests for request-path faults in catalog downloads, catalog
//! removal, remote `cache.pin` and in-process Metal deadlines.

use super::*;
use catalog_runtime_tests::isolated_catalog_state;
use std::io::Read;
use tests::{catalog_fixture_config, catalog_test_model, response_result, test_storage_descriptor};

/// Waits for a download job to leave its non-terminal states. The bound only
/// guards against a hang; it matches the other catalog download tests.
async fn wait_for_terminal_download(state: &Arc<ModuleState>, job: &str) -> JobRecord {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    loop {
        let record = state.store.get_job(job).unwrap().unwrap();
        if !store::is_download_non_terminal(&record.state) {
            return record;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "download {job} still {} after 15 s: {record:?}",
            record.state
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// Waits until detached tasks drop their module-state clones, so their effects
/// are complete and Windows can delete the store directory afterwards.
async fn wait_for_sole_reference(state: &Arc<ModuleState>) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    while Arc::strong_count(state) > 1 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "{} module-state references still held",
            Arc::strong_count(state)
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

fn record_install(state: &ModuleState, entry: &catalog::CatalogEntry, members: &[&str]) {
    state
        .store
        .sqlite()
        .with_conn_fenced(|tx| {
            tx.execute(
                "INSERT INTO catalog_installs VALUES (?1,?2,'metal')",
                rusqlite::params![entry.id, entry.manifest_digest()],
            )?;
            for (index, digest) in members.iter().enumerate() {
                tx.execute(
                    "INSERT INTO catalog_install_members VALUES (?1,?2,'metal',?3,?4)",
                    rusqlite::params![
                        entry.id,
                        entry.manifest_digest(),
                        format!("file-{index}"),
                        digest
                    ],
                )?;
            }
            Ok(())
        })
        .unwrap();
}

/// An in-process Metal call whose deadline has passed returns
/// `deadline_exceeded` without waiting for the engine, and the blocking task it
/// started keeps the execution permit until the engine is free again.
// The engine mutex is held across the call on purpose: it stands in for GPU
// work that outlives the caller's deadline.
#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn owned_metal_calls_stop_waiting_at_the_deadline_and_drain_with_their_permit() {
    let (dir, _) = test_storage_descriptor("owned-metal-deadline");
    let mut spec = catalog_fixture_config("gte-reranker-modernbert-base.owned-metal");
    // Not a release-catalog lane id, so the call takes no lane guards.
    spec.model_id = "owned-metal-deadline-test".into();
    let mut model = catalog_test_model(&dir, &spec);
    let engine = Arc::new(Mutex::new(OwnedMetalEmbedEngine::new(
        OwnedFamily::GteModernBert,
        OwnedDType::F32,
    )));
    Arc::get_mut(&mut model).unwrap().backend = EmbedBackend::Owned(Arc::clone(&engine));
    let runtime = RuntimeState::from_catalog(ModuleConfig::default(), vec![]).unwrap();
    let permits = runtime.execution.available_permits();

    for rerank in [false, true] {
        let busy = engine.lock().unwrap();
        let expired = Some(tokio::time::Instant::now());
        let call = async {
            if rerank {
                execute_rerank(
                    &runtime,
                    &model,
                    RerankRequest::default(),
                    Some(vec![vec![1, 2]]),
                    expired,
                    None,
                )
                .await
                .map(|_| ())
            } else {
                execute_embedding_with_catalog_guard(
                    &runtime,
                    &model,
                    TokenBatch {
                        items: vec![vec![1, 2]],
                    },
                    expired,
                    None,
                    None,
                )
                .await
                .map(|_| ())
            }
        };
        // Hang guard only: before the fix the caller waited on the engine
        // forever here, because this test holds the engine until it returns.
        let error = tokio::time::timeout(Duration::from_secs(30), call)
            .await
            .expect("an expired caller must not wait for the engine")
            .expect_err("an expired caller gets no result");
        assert_eq!(
            error.code, "deadline_exceeded",
            "rerank={rerank}: {error:?}"
        );
        assert_eq!(
            runtime.execution.available_permits(),
            permits - 1,
            "the started task keeps its execution permit while it drains"
        );
        drop(busy);
        let drained = tokio::time::Instant::now() + Duration::from_secs(15);
        while runtime.execution.available_permits() != permits {
            assert!(
                tokio::time::Instant::now() < drained,
                "permit never released"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
    drop(model);
    let _ = fs::remove_dir_all(dir);
}

/// `cache.pin` with an `http://` source fetches through a blocking client,
/// which panics when called on a Tokio runtime thread.
#[tokio::test]
async fn cache_pin_fetches_an_http_source_from_the_async_handler() {
    let (root, state) = isolated_catalog_state("cache-pin-http", false);
    let body = b"remote cache.pin fixture bytes".to_vec();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/artifact.bin", listener.local_addr().unwrap());
    let served = body.clone();
    // A plain thread, so the server keeps answering however the handler
    // treats the runtime thread.
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buffer = [0u8; 4096];
        let _ = stream.read(&mut buffer).unwrap();
        stream
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    served.len()
                )
                .as_bytes(),
            )
            .unwrap();
        stream.write_all(&served).unwrap();
    });
    let result = response_result(
        cache_pin(state.clone(), json!({"source_url": url})).await,
        "cache.pin",
    );
    assert!(result.get("error").is_none(), "{result}");
    assert!(
        result["artifact"]["digest"]
            .as_str()
            .unwrap()
            .ends_with(&sha256_hex(&body)),
        "{result}"
    );
    drop(state);
    fs::remove_dir_all(root).unwrap();
}

/// A response that keeps going past the pinned size is refused at the first
/// chunk that would exceed it, before that chunk is written and before the
/// stream ends, so it never reaches the end-of-stream digest check.
#[tokio::test]
async fn catalog_download_refuses_a_body_at_the_chunk_that_passes_its_pinned_size() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let (root, mut state) = isolated_catalog_state("download-oversized", true);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let pinned = b"catalog fixture bytes";
    let runtime = Arc::get_mut(&mut Arc::get_mut(&mut state).unwrap().runtime).unwrap();
    runtime.hf_endpoint = format!("http://{}", listener.local_addr().unwrap());
    let entry = runtime
        .release_catalog
        .models
        .iter_mut()
        .find(|e| e.id == "gte-modernbert-base")
        .unwrap();
    entry.upstream.revision = "a".repeat(40);
    for file in &mut entry.files {
        file.sha256 = sha256_hex(pinned);
        file.size_bytes = pinned.len() as u64;
    }
    // The pinned bytes followed by far more than the pin allows.
    let mut oversized = pinned.to_vec();
    oversized.resize(pinned.len() + (1 << 20), 0);
    let server = tokio::spawn(async move {
        loop {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buffer = [0u8; 4096];
            let _ = stream.read(&mut buffer).await.unwrap();
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                oversized.len()
            );
            if stream.write_all(header.as_bytes()).await.is_ok() {
                let _ = stream.write_all(&oversized).await;
            }
        }
    });
    let accepted = response_result(
        models_download(state.clone(), json!({"catalog_id":"gte-modernbert-base"})).await,
        "models.download",
    );
    let job = accepted["job_id"].as_str().unwrap().to_string();
    let record = wait_for_terminal_download(&state, &job).await;
    server.abort();
    assert_eq!(record.state, "failed", "{record:?}");
    let error = record.error_json.clone().unwrap();
    assert_eq!(error["code"], "artifact_invalid", "{error}");
    let details = &error["details"];
    assert_eq!(details["expected_size"], pinned.len() as u64, "{error}");
    // An end-of-stream refusal would carry the digest of the whole body.
    assert!(details["actual_sha256"].is_null(), "{error}");
    assert!(
        details["actual_size"].as_u64().unwrap() > pinned.len() as u64,
        "{error}"
    );
    wait_for_sole_reference(&state).await;
    drop(state);
    fs::remove_dir_all(root).unwrap();
}

/// A cancel that lands before the detached download task plans its work must
/// not be followed by a staging directory that nothing would ever remove, and
/// the ended job must not leave progress counters or publish locks behind.
#[tokio::test]
async fn early_download_cancel_leaves_no_staging_or_bookkeeping() {
    let (root, state) = isolated_catalog_state("download-early-cancel", true);
    // Neither call awaits anything, so on this single-threaded runtime the
    // spawned download task cannot start until the cancel has committed.
    let accepted = response_result(
        models_download(state.clone(), json!({"catalog_id":"gte-modernbert-base"})).await,
        "models.download",
    );
    let job = accepted["job_id"].as_str().unwrap().to_string();
    let cancelled = response_result(
        models_download_cancel(state.clone(), json!({"job_id": job})).await,
        "models.download.cancel",
    );
    assert_eq!(cancelled["state"], "cancelled", "{cancelled}");
    wait_for_sole_reference(&state).await;
    assert!(
        !catalog_staging(&state.model_cache, &job).exists(),
        "staging recreated after cancellation"
    );
    assert!(!state
        .runtime
        .download_bytes
        .lock()
        .unwrap()
        .contains_key(&job));
    // The next fence lookup prunes every lock nobody holds.
    drop(download_publish_lock(&state.runtime, "unrelated-job"));
    assert!(!state
        .runtime
        .download_locks
        .lock()
        .unwrap()
        .contains_key(&job));
    drop(state);
    fs::remove_dir_all(root).unwrap();
}

/// `models.remove` commits its metadata before unlinking blobs. When one
/// unlink fails, the removal still stands, the blobs already unlinked are not
/// named by any surviving install row, and the failed one is left for cache GC.
#[tokio::test]
async fn models_remove_commits_before_unlinking_and_survives_an_unlink_failure() {
    let (root, state) = isolated_catalog_state("remove-unlink-failure", true);
    let entry = state
        .runtime
        .release_catalog
        .entry("gte-modernbert-base")
        .unwrap()
        .clone();
    let removable = sha256_hex(b"removable blob");
    let stuck = sha256_hex(b"blob that cannot be unlinked");
    record_install(&state, &entry, &[&removable, &stuck]);
    let removable_path = state.model_cache.blob_path(&removable);
    fs::create_dir_all(removable_path.parent().unwrap()).unwrap();
    fs::write(&removable_path, b"removable blob").unwrap();
    // A non-empty directory where the blob file should be makes its unlink fail.
    let stuck_path = state.model_cache.blob_path(&stuck);
    fs::create_dir_all(&stuck_path).unwrap();
    fs::write(stuck_path.join("occupied"), b"x").unwrap();

    let removed = response_result(
        models_remove(state.clone(), json!({"catalog_id": entry.id})).await,
        "models.remove",
    );
    assert!(removed.get("error").is_none(), "{removed}");
    assert_eq!(removed["freed_bytes"], b"removable blob".len() as u64);
    assert!(state.store.catalog_installs(&entry.id).unwrap().is_empty());
    let members: i64 = state
        .store
        .sqlite()
        .with_conn_fenced(|tx| {
            tx.query_row(
                "SELECT COUNT(*) FROM catalog_install_members WHERE catalog_id=?1",
                [&entry.id],
                |row| row.get(0),
            )
        })
        .unwrap();
    assert_eq!(members, 0);
    assert!(!removable_path.exists());
    assert!(
        stuck_path.exists(),
        "the failed unlink is left for cache GC"
    );
    drop(state);
    fs::remove_dir_all(root).unwrap();
}

/// Removing a lane whose self-check failed must wake the findings recompute,
/// or the persisted findings digest keeps warning about a lane that is gone.
#[tokio::test(start_paused = true)]
async fn models_remove_wakes_the_findings_recompute() {
    let (root, state) = isolated_catalog_state("remove-findings", true);
    let entry = state
        .runtime
        .release_catalog
        .entry("gte-modernbert-base")
        .unwrap()
        .clone();
    record_install(&state, &entry, &[]);
    let backend = entry.backend("metal").unwrap().clone();
    let (id, key) = catalog_self_check_key(&state, &entry, &backend).unwrap();
    let generation = catalog_check_generation(&state, &id, &key).unwrap();
    assert!(catalog_complete_check(
        &state,
        &id,
        generation,
        "failed",
        Some("numerical_mismatch")
    )
    .unwrap());
    // Consume the wake-up from the failed check itself.
    tokio::time::timeout(
        Duration::from_millis(1),
        state.runtime.findings_changed.notified(),
    )
    .await
    .unwrap();

    let removed = response_result(
        models_remove(state.clone(), json!({"catalog_id": entry.id})).await,
        "models.remove",
    );
    assert!(removed.get("error").is_none(), "{removed}");
    assert!(
        tokio::time::timeout(
            Duration::from_millis(1),
            state.runtime.findings_changed.notified(),
        )
        .await
        .is_ok(),
        "models.remove did not wake the findings recompute"
    );
    drop(state);
    fs::remove_dir_all(root).unwrap();
}
