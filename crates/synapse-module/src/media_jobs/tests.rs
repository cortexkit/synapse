use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
const MODEL: &str = "parakeet-tdt-0.6b-v3";

fn fixture(label: &str, installed: bool) -> (PathBuf, Arc<ModuleState>) {
    let (root, mut state) = catalog_runtime_tests::isolated_catalog_state(label, false);
    let runtime = Arc::get_mut(&mut Arc::get_mut(&mut state).unwrap().runtime).unwrap();
    #[cfg(target_os = "macos")]
    {
        runtime.runnable_backends = BTreeSet::from(["metal".into(), "ane".into()]);
    }
    #[cfg(not(target_os = "macos"))]
    {
        runtime.runnable_backends = BTreeSet::from(["cuda".into(), "vulkan".into()]);
        runtime.backend_floor_observed.insert(
            "cuda".into(),
            json!({"driver_api":13020,"compute_capability":{"major":9,"minor":0}}),
        );
        runtime.backend_floor_observed.insert(
            "vulkan".into(),
            json!({"device_local_heaps":[u64::MAX],"max_storage_buffer_range":u64::MAX}),
        );
    }
    if installed {
        let entry = state.runtime.release_catalog.entry(MODEL).unwrap();
        state
            .store
            .store
            .with_conn_fenced(|t| {
                for b in &entry.backends {
                    t.execute(
                        "INSERT INTO catalog_installs VALUES(?1,?2,?3)",
                        params![entry.id, entry.manifest_digest(), b.backend],
                    )?;
                }
                Ok(())
            })
            .unwrap();
    }
    *state.runtime.media.runner.lock().unwrap() = Some(Arc::new(|_, _| {
        Ok(json!({"text":"hello","words":[],"no_speech":false}))
    }));
    startup(&state).unwrap();
    (root, state)
}
fn audio() -> Vec<u8> {
    cortexkit_role_media_conformance::fixtures::AudioFixture::Tone.wav()
}
fn media(bytes: &[u8]) -> Value {
    json!({"mime":"audio/wav","bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(bytes)),"data":base64_encode(bytes)})
}
fn request(id: &str) -> Value {
    json!({"client_job_id":id,"kind":"transcribe","model":MODEL,"input":{"audio":media(&audio())}})
}
async fn call(state: &Arc<ModuleState>, owner: &str, method: &str, params: Value) -> Answer<Value> {
    match handle(state.clone(), Some(owner), method, params).await {
        HandlerOutcome::Response(body) => {
            cortexkit_role_media::wire::read_reply(&body).map_err(internal)
        }
        HandlerOutcome::ErrorWithDetail {
            code,
            message,
            detail,
        } => Err(Fault(
            json!({"code":code,"message":message,"detail":detail}),
        )),
        other => panic!("unexpected response {other:?}"),
    }
}
async fn done(state: &Arc<ModuleState>, id: &str) -> Value {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let v = call(state, "alice", "media.status", json!({"job_id":id}))
                .await
                .unwrap();
            if v["status"] != "queued" && v["status"] != "running" {
                return v;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap()
}
fn first(state: &ModuleState) -> String {
    select(state, MODEL).unwrap().1.backend
}

#[tokio::test]
async fn media_declaration_selection_and_five_states() {
    let (root, state) = fixture("media-states", false);
    let described = call(&state, "alice", "role.describe", json!({}))
        .await
        .unwrap();
    let described: cortexkit_role_media::describe::Describe =
        serde_json::from_value(described).unwrap();
    cortexkit_role_media::describe::check_describe(&described).unwrap();
    assert!(described.engines.iter().all(|e| e.local
        && e.provider == "local"
        && e.formats == ["audio/wav"]
        && e.bias_limit.is_none()));
    let selected = first(&state);
    assert_eq!(
        selected,
        if cfg!(target_os = "macos") {
            "metal"
        } else {
            "cuda"
        }
    );
    let err = call(&state, "alice", "media.submit", request("absent"))
        .await
        .unwrap_err();
    assert_eq!(err.0["code"], "model_not_installed");
    assert_eq!(err.0["detail"]["download_op"], "models.download");
    let entry = state.runtime.release_catalog.entry(MODEL).unwrap();
    let backend = entry.backend(&selected).unwrap();
    state
        .store
        .store
        .with_conn_fenced(|t| {
            t.execute(
                "INSERT INTO catalog_installs VALUES(?1,?2,?3)",
                params![entry.id, entry.manifest_digest(), backend.backend],
            )?;
            Ok(())
        })
        .unwrap();
    let (check, key) = catalog_self_check_key(&state, entry, backend).unwrap();
    let generation = catalog_check_generation(&state, &check, &key).unwrap();
    catalog_complete_check(
        &state,
        &check,
        generation,
        "failed",
        Some("numeric mismatch"),
    )
    .unwrap();
    assert_eq!(
        call(&state, "alice", "media.submit", request("failed"))
            .await
            .unwrap_err()
            .0["code"],
        "self_check_failed"
    );
    // Permanent self-check evidence wins over a transient retry cache.
    lane_cache(&state.runtime, &catalog::lane_id(MODEL, &selected))
        .lock()
        .await
        .retry = Some((
        Instant::now() + Duration::from_secs(60),
        fault("engine_crashed", None),
    ));
    assert_eq!(
        call(&state, "alice", "media.submit", request("still-failed"))
            .await
            .unwrap_err()
            .0["code"],
        "self_check_failed"
    );
    state
        .store
        .store
        .with_conn_fenced(|t| {
            t.execute("DELETE FROM catalog_self_checks", [])?;
            Ok(())
        })
        .unwrap();
    assert_eq!(
        call(&state, "alice", "media.submit", request("crashed"))
            .await
            .unwrap_err()
            .0["code"],
        "engine_crashed"
    );
    lane_cache(&state.runtime, &catalog::lane_id(MODEL, &selected))
        .lock()
        .await
        .retry = None;
    let id = call(&state, "alice", "media.submit", request("ready"))
        .await
        .unwrap()["job_id"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(done(&state, &id).await["status"], "done");
    drop(state);
    fs::remove_dir_all(root).unwrap();
    let (root, mut state) = fixture("media-no-backend", false);
    let runtime = Arc::get_mut(&mut Arc::get_mut(&mut state).unwrap().runtime).unwrap();
    runtime.runnable_backends.clear();
    assert!(engines(runtime).is_empty());
    assert_eq!(
        call(&state, "alice", "media.submit", request("unavailable"))
            .await
            .unwrap_err()
            .0["code"],
        "backend_unavailable"
    );
    drop(state);
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn media_replay_is_principal_scoped_and_checks_raw_members_before_validation() {
    let (root, state) = fixture("media-replay", true);
    let calls = Arc::new(AtomicUsize::new(0));
    let count = calls.clone();
    *state.runtime.media.runner.lock().unwrap() = Some(Arc::new(move |_, _| {
        count.fetch_add(1, Ordering::SeqCst);
        Ok(json!({"text":"hello","words":[],"no_speech":false}))
    }));
    let p = request("same");
    let id = call(&state, "alice", "media.submit", p.clone())
        .await
        .unwrap()["job_id"]
        .clone();
    assert_eq!(done(&state, id.as_str().unwrap()).await["status"], "done");
    for field in ["kind", "input", "model", "decision_id", "deadline_ms"] {
        let mut changed = p.clone();
        changed[field] = json!({"invalid":"even malformed reused members must not start work"});
        let e = call(&state, "alice", "media.submit", changed)
            .await
            .unwrap_err();
        assert_eq!(e.0["code"], "client_job_id_reuse");
        assert_eq!(e.0["detail"]["field"], field);
    }
    assert_eq!(
        call(&state, "alice", "media.submit", p.clone())
            .await
            .unwrap()["job_id"],
        id
    );
    for method in ["media.status", "media.result", "media.cancel"] {
        assert_eq!(
            call(&state, "bob", method, json!({"job_id":id}))
                .await
                .unwrap_err()
                .0["code"],
            "unknown_job"
        );
    }
    let other = call(&state, "bob", "media.submit", p).await.unwrap()["job_id"].clone();
    assert_ne!(other, id);
    assert_eq!(
        done_as(&state, "bob", other.as_str().unwrap()).await["status"],
        "done"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    drop(state);
    fs::remove_dir_all(root).unwrap();
}
async fn done_as(state: &Arc<ModuleState>, owner: &str, id: &str) -> Value {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let v = call(state, owner, "media.status", json!({"job_id":id}))
                .await
                .unwrap();
            if !matches!(v["status"].as_str(), Some("queued" | "running")) {
                return v;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap()
}

#[tokio::test]
async fn media_queue_bound_cancel_drains_and_keeps_one_owner_per_lane() {
    let (root, state) = fixture("media-queue", true);
    let started = Arc::new(AtomicUsize::new(0));
    let count = started.clone();
    let release = Arc::new((Mutex::new(false), std::sync::Condvar::new()));
    let held = release.clone();
    *state.runtime.media.runner.lock().unwrap() = Some(Arc::new(move |_, _| {
        count.fetch_add(1, Ordering::SeqCst);
        let (lock, wake) = &*held;
        let mut released = lock.lock().unwrap();
        while !*released {
            released = wake.wait(released).unwrap();
        }
        Ok(json!({"text":"discard me","words":[],"no_speech":false}))
    }));
    let mut ids = Vec::new();
    for n in 0..LANE_MAX {
        let mut p = request(&format!("q{n}"));
        if n % 2 == 1 {
            p["model"] = json!(catalog::lane_id(MODEL, &first(&state)));
        }
        ids.push(call(&state, "alice", "media.submit", p).await.unwrap()["job_id"].clone());
    }
    let before = fs::read_dir(persistence::directory(&state).unwrap())
        .unwrap()
        .count();
    let error = call(&state, "alice", "media.submit", request("overflow"))
        .await
        .unwrap_err();
    assert_eq!(error.0["code"], "queue_full");
    assert!(error.0["detail"]["retry_after_ms"].as_u64().unwrap() > 0);
    assert_eq!(
        fs::read_dir(persistence::directory(&state).unwrap())
            .unwrap()
            .count(),
        before
    );
    tokio::time::timeout(Duration::from_secs(5), async {
        while started.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(started.load(Ordering::SeqCst), 1);
    assert_eq!(
        call(&state, "alice", "media.cancel", json!({"job_id":ids[0]}))
            .await
            .unwrap()["status"],
        "cancelled"
    );
    assert_eq!(
        call(&state, "alice", "media.result", json!({"job_id":ids[0]}))
            .await
            .unwrap_err()
            .0["code"],
        "job_not_done"
    );
    for id in ids.iter().skip(1) {
        call(&state, "alice", "media.cancel", json!({"job_id":id}))
            .await
            .unwrap();
    }
    *release.0.lock().unwrap() = true;
    release.1.notify_all();
    let cache = lane_cache(&state.runtime, &catalog::lane_id(MODEL, &first(&state)));
    let _drained = cache.lock().await;
    assert_eq!(
        call(&state, "alice", "media.status", json!({"job_id":ids[0]}))
            .await
            .unwrap()["status"],
        "cancelled"
    );
    assert_eq!(started.load(Ordering::SeqCst), 1);
    drop(_drained);
    drop(state);
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn media_audio_options_upload_capacity_permissions_retention_and_restart() {
    let (root, state) = fixture("media-objects", true);
    let mut p = request("bias");
    p["input"]["bias"] = json!({"terms":[{"text":"word","weight":1.0}]});
    let e = call(&state, "alice", "media.submit", p).await.unwrap_err();
    assert_eq!(e.0["code"], "invalid_params");
    assert_eq!(e.0["detail"]["field"], "bias");
    let mut bad = request("bad");
    bad["input"]["audio"] = media(b"not a WAV");
    let e = call(&state, "alice", "media.submit", bad)
        .await
        .unwrap_err();
    assert_eq!(e.0["detail"]["field"], "audio");
    let bytes = audio();
    let facts = media(&bytes);
    let begun = call(
        &state,
        "alice",
        "media.upload_begin",
        json!({"mime":"audio/wav","bytes":bytes.len(),"sha256":facts["sha256"]}),
    )
    .await
    .unwrap();
    let object = begun["upload_id"].clone();
    let oversized = call(
        &state,
        "alice",
        "media.upload_append",
        json!({"upload_id":object,"offset":0,"data":base64_encode(&vec![0;2*1024*1024+1])}),
    )
    .await
    .unwrap_err();
    assert_eq!(oversized.0["code"], "upload_chunk_too_large");
    call(
        &state,
        "alice",
        "media.upload_append",
        json!({"upload_id":object,"offset":0,"data":facts["data"]}),
    )
    .await
    .unwrap();
    call(
        &state,
        "alice",
        "media.upload_commit",
        json!({"upload_id":object}),
    )
    .await
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let dir = persistence::directory(&state).unwrap();
        assert_eq!(
            fs::metadata(&dir).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            fs::metadata(dir.join(object.as_str().unwrap()))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
    }
    let mut p = request("object");
    p["input"]["audio"] = facts.clone();
    p["input"]["audio"].as_object_mut().unwrap().remove("data");
    p["input"]["audio"]["object_id"] = object.clone();
    p["input"]["language"] = json!("ignored");
    let id = call(&state, "alice", "media.submit", p.clone())
        .await
        .unwrap()["job_id"]
        .clone();
    assert_eq!(done(&state, id.as_str().unwrap()).await["status"], "done");
    let result = call(&state, "alice", "media.result", json!({"job_id":id}))
        .await
        .unwrap();
    assert!(result.get("language").is_none());
    startup(&state).unwrap();
    assert_eq!(
        call(&state, "alice", "media.submit", p).await.unwrap()["job_id"],
        id
    );
    assert_eq!(
        call(&state, "alice", "media.result", json!({"job_id":id}))
            .await
            .unwrap(),
        result
    );
    assert_eq!(
        call(
            &state,
            "alice",
            "media.read",
            json!({"object_id":object,"offset":0,"length":bytes.len()})
        )
        .await
        .unwrap()["data"],
        facts["data"]
    );
    // Reserve the full budget using four maximum-size open uploads; no payload
    // allocation is needed to test rejection before the fifth file is created.
    state
        .store
        .store
        .with_conn_fenced(|t| {
            t.execute("UPDATE media_objects SET state='gone'", [])?;
            Ok(())
        })
        .unwrap();
    for _ in 0..4 {
        call(
            &state,
            "alice",
            "media.upload_begin",
            json!({"mime":"audio/wav","bytes":UPLOAD_MAX,"sha256":"0".repeat(64)}),
        )
        .await
        .unwrap();
    }
    let files = fs::read_dir(persistence::directory(&state).unwrap())
        .unwrap()
        .count();
    assert_eq!(
        call(
            &state,
            "alice",
            "media.upload_begin",
            json!({"mime":"audio/wav","bytes":1,"sha256":"0".repeat(64)})
        )
        .await
        .unwrap_err()
        .0["code"],
        "queue_full"
    );
    assert_eq!(
        fs::read_dir(persistence::directory(&state).unwrap())
            .unwrap()
            .count(),
        files
    );
    startup(&state).unwrap();
    persistence::sweep(&state, now_ms() + RETENTION_MS + 1).unwrap();
    assert_eq!(
        fs::read_dir(persistence::directory(&state).unwrap())
            .unwrap()
            .count(),
        0
    );
    drop(state);
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn media_observation_pages_reassemble_and_unload_releases_cached_host() {
    let (root, mut state) = fixture("media-observation", true);
    Arc::get_mut(&mut Arc::get_mut(&mut state).unwrap().runtime)
        .unwrap()
        .certify_observation = true;
    let bytes = vec![37u8; PAGE + 31];
    let expected = bytes.clone();
    *state.runtime.media.runner.lock().unwrap() = Some(Arc::new(move |lane, _| {
        Ok(
            json!({"text":"hello","words":[],"no_speech":false,"_observation":{"lane_id":lane,"fingerprint":"pin","valid_encoder_frames":1,"triples":"","encoder_output":base64_encode(&bytes),"decode_trace":""}}),
        )
    }));
    let id = call(&state, "alice", "media.submit", request("observation"))
        .await
        .unwrap()["job_id"]
        .clone();
    assert_eq!(done(&state, id.as_str().unwrap()).await["status"], "done");
    let result = call(&state, "alice", "media.result", json!({"job_id":id}))
        .await
        .unwrap();
    assert_eq!(result["observation"]["encoder_output"]["pages"], 2);
    let mut all = Vec::new();
    for page in 0..2 {
        let data = call(
            &state,
            "alice",
            "certify.transcribe_observation",
            json!({"job_id":id,"part":"encoder_output","page":page}),
        )
        .await
        .unwrap();
        all.extend(STANDARD.decode(data["data"].as_str().unwrap()).unwrap());
    }
    assert_eq!(all, expected);
    assert_eq!(
        result["observation"]["encoder_output"]["sha256"],
        format!("{:x}", Sha256::digest(&all))
    );
    let dropped = Arc::new(AtomicUsize::new(0));
    struct DropMark(Arc<AtomicUsize>);
    impl Drop for DropMark {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let marker = DropMark(dropped.clone());
    let selected = first(&state);
    let lane = catalog::lane_id(MODEL, &selected);
    let profile = transcribe_host::TranscribeProfile::pinned(
        state
            .runtime
            .release_catalog
            .entry(MODEL)
            .unwrap()
            .backend(&selected)
            .unwrap()
            .profile
            .as_deref()
            .unwrap(),
    )
    .unwrap();
    lane_cache(&state.runtime, &lane).lock().await.host = Some(
        transcribe_host::TranscribeHost::in_process_for_test(profile, move |_| {
            let _ = &marker;
            unreachable!()
        }),
    );
    let result = crate::tests::response_result(
        model_unload(state.clone(), json!({"model_id":lane})).await,
        "model.unload",
    );
    assert_eq!(result["state"], "unloaded");
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    drop(state);
    fs::remove_dir_all(root).unwrap();
}

// cortexkit-role-media-conformance 0.2.8's generic job cases configure remote
// engines and demand billing records, so only its local-compatible upload and
// input-bound cases run here.
// The tests above cover local job replay, terminality and cancellation directly.
struct LocalSubject(Arc<ModuleState>);
#[async_trait]
impl cortexkit_role_media_conformance::MediaSubject for LocalSubject {
    async fn configure_engines(
        &self,
        _: Vec<Engine>,
    ) -> cortexkit_role_media_conformance::subject::SubjectResult<()> {
        Ok(())
    }
    async fn configure_peers(
        &self,
        _: cortexkit_role_media_conformance::peers::RecordingRoutingPeer,
        _: cortexkit_role_media_conformance::peers::RecordingVendor,
    ) -> cortexkit_role_media_conformance::subject::SubjectResult<()> {
        Ok(())
    }
    async fn configure_principals(
        &self,
        _: cortexkit_role_media_conformance::subject::Principals,
    ) -> cortexkit_role_media_conformance::subject::SubjectResult<()> {
        Ok(())
    }
    async fn call_as(
        &self,
        principal: &str,
        request: cortexkit_role_media_conformance::subject::Request,
    ) -> cortexkit_role_media_conformance::subject::SubjectResult<Value> {
        use cortexkit_role_media_conformance::subject::SubjectError;
        let mut params = request.params;
        if request.method == "media.submit" && params["model"] == "transcribe" {
            params["model"] = json!(MODEL);
        }
        match handle(self.0.clone(), Some(principal), &request.method, params).await {
            HandlerOutcome::Response(body) => {
                serde_json::from_slice(&body).map_err(|e| SubjectError::new(e.to_string()))
            }
            HandlerOutcome::ErrorWithDetail {
                code,
                message,
                detail,
            } => Err(SubjectError::refused(
                json!({"code":code,"message":message,"detail":detail}),
            )),
            other => Err(SubjectError::new(format!("unexpected {other:?}"))),
        }
    }
    async fn restart(&self) -> cortexkit_role_media_conformance::subject::SubjectResult<()> {
        startup(&self.0).map_err(|e| {
            cortexkit_role_media_conformance::subject::SubjectError::new(e.to_string())
        })
    }
    async fn script_engine(
        &self,
        _: &str,
        script: cortexkit_role_media_conformance::subject::Script,
    ) -> cortexkit_role_media_conformance::subject::SubjectResult<()> {
        let output = script
            .steps
            .into_iter()
            .find_map(|s| match s {
                cortexkit_role_media_conformance::subject::ScriptStep::Output(v) => Some(v),
                _ => None,
            })
            .unwrap();
        *self.0.runtime.media.runner.lock().unwrap() =
            Some(Arc::new(move |_, _| Ok(output.clone())));
        Ok(())
    }
}
#[tokio::test]
async fn media_published_local_compatible_conformance() {
    for name in [
        "reply_envelope",
        "oversize_output_is_object",
        "client_job_id_bounds",
        "upload_resumes_at_received",
        "upload_gone_after_restart",
        "committed_upload_is_object",
    ] {
        let (root, state) = fixture(name, true);
        let outcome =
            cortexkit_role_media_conformance::run_case(&LocalSubject(state.clone()), name).await;
        assert_eq!(
            outcome,
            cortexkit_role_media_conformance::Outcome::Passed,
            "{name}"
        );
        // A job's drain task still holds a state clone, and with it the store
        // connection, for a moment after its result is readable. Windows
        // refuses to delete a directory with an open file, so wait until this
        // test holds the only reference before removing it.
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
        while Arc::strong_count(&state) > 1 {
            assert!(
                tokio::time::Instant::now() < deadline,
                "{name}: {} module-state references still held",
                Arc::strong_count(&state)
            );
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        drop(state);
        fs::remove_dir_all(root).unwrap();
    }
}

#[tokio::test]
async fn media_restart_fails_only_nonterminal_work_and_retains_objects_for_one_hour() {
    let (root, state) = fixture("media-recovery", true);
    let lane = catalog::lane_id(MODEL, &first(&state));
    let bytes = audio();
    let m: objects::MediaValue = serde_json::from_value(media(&bytes)).unwrap();
    let mut ids = Vec::new();
    for name in ["queued", "running", "cancelled", "done"] {
        let p = request(name);
        let typed = serde_json::from_value(p.clone()).unwrap();
        let id = persistence::admit(
            &state,
            "alice",
            &typed,
            &persistence::binding(&p),
            &lane,
            &m,
            &bytes,
        )
        .unwrap();
        if name == "running" {
            assert!(persistence::start(&state, &id).unwrap());
        }
        if name == "cancelled" {
            persistence::job_answer(&state, "alice", "media.cancel", json!({"job_id":id})).unwrap();
        }
        if name == "done" {
            persistence::finish(
                &state,
                &id,
                Ok(json!({"text":"saved","words":[],"no_speech":false})),
            )
            .unwrap();
        }
        ids.push(id);
    }
    fs::write(
        persistence::directory(&state).unwrap().join("orphan"),
        b"private orphan",
    )
    .unwrap();
    startup(&state).unwrap();
    for id in &ids[..2] {
        let v = call(&state, "alice", "media.status", json!({"job_id":id}))
            .await
            .unwrap();
        assert_eq!(v["status"], "failed");
        assert_eq!(v["failure"]["code"], "module_restarted");
        assert_eq!(v["failure"]["class"], "transient");
    }
    assert_eq!(
        call(&state, "alice", "media.status", json!({"job_id":ids[2]}))
            .await
            .unwrap()["status"],
        "cancelled"
    );
    assert_eq!(
        call(&state, "alice", "media.result", json!({"job_id":ids[3]}))
            .await
            .unwrap()["text"],
        "saved"
    );
    persistence::finish(&state, &ids[0], Ok(json!({"text":"late result"}))).unwrap();
    assert_eq!(
        call(&state, "alice", "media.status", json!({"job_id":ids[0]}))
            .await
            .unwrap()["failure"]["code"],
        "module_restarted"
    );
    let earliest: u64 = state
        .store
        .store
        .with_conn(|c| {
            c.query_row(
                "SELECT min(retained_ms) FROM media_objects WHERE state='committed'",
                [],
                |r| r.get(0),
            )
        })
        .unwrap();
    persistence::sweep(&state, earliest - 1).unwrap();
    assert_eq!(
        fs::read_dir(persistence::directory(&state).unwrap())
            .unwrap()
            .count(),
        4
    );
    persistence::sweep(&state, now_ms() + RETENTION_MS + 1).unwrap();
    assert_eq!(
        fs::read_dir(persistence::directory(&state).unwrap())
            .unwrap()
            .count(),
        0
    );
    drop(state);
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn media_large_result_is_a_retained_checksummed_object() {
    let (root, state) = fixture("media-output-object", true);
    let output =
        json!({"text":"x".repeat(cortexkit_role_media::MESSAGE_MAX),"words":[],"no_speech":false});
    let expected = serde_json::to_vec(&output).unwrap();
    *state.runtime.media.runner.lock().unwrap() = Some(Arc::new(move |_, _| Ok(output.clone())));
    let id = call(&state, "alice", "media.submit", request("large"))
        .await
        .unwrap()["job_id"]
        .clone();
    assert_eq!(done(&state, id.as_str().unwrap()).await["status"], "done");
    let output = call(&state, "alice", "media.result", json!({"job_id":id}))
        .await
        .unwrap();
    let object = &output["output_object"];
    assert_eq!(object["mime"], "application/json");
    assert_eq!(object["bytes"], expected.len());
    assert_eq!(object["sha256"], format!("{:x}", Sha256::digest(&expected)));
    let mut all = Vec::new();
    while all.len() < expected.len() {
        let part = call(
            &state,
            "alice",
            "media.read",
            json!({"object_id":object["object_id"],"offset":all.len(),"length":2*1024*1024}),
        )
        .await
        .unwrap();
        all.extend(STANDARD.decode(part["data"].as_str().unwrap()).unwrap());
    }
    assert_eq!(all, expected);
    drop(state);
    fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn media_quarantine_never_falls_back_and_submit_does_not_store() {
    let (root, state) = fixture("media-quarantine", true);
    let backend = first(&state);
    let lane = catalog::lane_id(MODEL, &backend);
    let mut cfg = worker_host::WorkerHostConfig::new("unused", root.join("worker"));
    cfg.catalog_lane = Some(lane.clone());
    let engine = worker_host::WorkerEngine::new(cfg).unwrap();
    let model = engine.insert_loaded_model_for_test("speech".into(), 0, 0, None);
    engine.quarantine_transcribe_for_test(&model);
    let profile = transcribe_host::TranscribeProfile::pinned(
        state
            .runtime
            .release_catalog
            .entry(MODEL)
            .unwrap()
            .backend(&backend)
            .unwrap()
            .profile
            .as_deref()
            .unwrap(),
    )
    .unwrap();
    lane_cache(&state.runtime, &lane).lock().await.host = Some(
        transcribe_host::TranscribeHost::worker_for_test(profile, engine, model),
    );
    for id in [MODEL.to_string(), lane.clone()] {
        let mut p = request(&id);
        p["model"] = json!(id);
        let e = call(&state, "alice", "media.submit", p).await.unwrap_err();
        assert_eq!(e.0["code"], "engine_crashed");
        assert_eq!(e.0["detail"]["lane_id"], lane);
        assert_eq!(e.0["detail"]["quarantined"], true);
        let retry = e.0["detail"]["retry_after_ms"].as_u64().unwrap();
        assert!((599000..=600000).contains(&retry));
    }
    let jobs: u64 = state
        .store
        .store
        .with_conn(|c| c.query_row("SELECT count(*) FROM media_jobs", [], |r| r.get(0)))
        .unwrap();
    assert_eq!(jobs, 0);
    assert_eq!(
        fs::read_dir(persistence::directory(&state).unwrap())
            .unwrap()
            .count(),
        0
    );
    unload(&state.runtime, None, false).await.unwrap();
    drop(state);
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn media_decoded_duration_overflow_fails_accepted_job_without_loading_engine() {
    let (root, state) = fixture("media-duration", true);
    let calls = Arc::new(AtomicUsize::new(0));
    let count = calls.clone();
    *state.runtime.media.runner.lock().unwrap() = Some(Arc::new(move |_, _| {
        count.fetch_add(1, Ordering::SeqCst);
        unreachable!("over-limit audio must not reach a host")
    }));
    let samples = 600_001u32 * 16;
    let size = samples * 2;
    let mut wav = Vec::with_capacity(size as usize + 44);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(size + 36).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&16000u32.to_le_bytes());
    wav.extend_from_slice(&32000u32.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&size.to_le_bytes());
    wav.resize(size as usize + 44, 0);
    let sha = format!("{:x}", Sha256::digest(&wav));
    let id = call(
        &state,
        "alice",
        "media.upload_begin",
        json!({"mime":"audio/wav","bytes":wav.len(),"sha256":sha}),
    )
    .await
    .unwrap()["upload_id"]
        .clone();
    let mut received = 0;
    for chunk in wav.chunks(2 * 1024 * 1024) {
        call(
            &state,
            "alice",
            "media.upload_append",
            json!({"upload_id":id,"offset":received,"data":base64_encode(chunk)}),
        )
        .await
        .unwrap();
        received += chunk.len();
    }
    call(
        &state,
        "alice",
        "media.upload_commit",
        json!({"upload_id":id}),
    )
    .await
    .unwrap();
    let mut p = request("long");
    p["input"]["audio"] = json!({"mime":"audio/wav","bytes":wav.len(),"sha256":sha,"object_id":id});
    let job = call(&state, "alice", "media.submit", p.clone())
        .await
        .unwrap()["job_id"]
        .clone();
    let status = done(&state, job.as_str().unwrap()).await;
    assert_eq!(status["status"], "failed");
    assert_eq!(status["failure"]["code"], "input_too_large");
    assert_eq!(status["failure"]["detail"]["field"], "input.audio");
    assert_eq!(status["failure"]["detail"]["limit"], "max_duration_ms");
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    p["client_job_id"] = json!("stated");
    p["input"]["audio"]["duration_ms"] = json!(600001);
    let e = call(&state, "alice", "media.submit", p).await.unwrap_err();
    assert_eq!(e.0["code"], "input_too_large");
    assert_eq!(e.0["detail"]["limit"], "max_duration_ms");
    assert!(
        lane_cache(&state.runtime, &catalog::lane_id(MODEL, &first(&state)))
            .lock()
            .await
            .host
            .is_none()
    );
    drop(state);
    fs::remove_dir_all(root).unwrap();
}
