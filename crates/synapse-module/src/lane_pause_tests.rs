use super::*;
use worker_host::ane_residency::AneDirectLaneLock;

async fn call(state: &Arc<ModuleState>, method: &str, params: Value) -> Value {
    response_result(
        dispatch_request(
            state.clone(),
            MethodEnvelope {
                method: method.into(),
                params,
            },
            None,
        )
        .await,
        method,
    )
}

async fn start_pause(state: &Arc<ModuleState>) -> Value {
    call(
        state,
        "lanes.pause",
        json!({"scope":"neural_engine", "duration_ms":60_000,
        "reason":"exclusive measurement"}),
    )
    .await
}

async fn wait_phase(state: &Arc<ModuleState>, expected: &str) {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if lane_pause::status(&state.runtime)["phase"] == expected {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap_or_else(|_| {
        panic!(
            "expected {expected}, got {}",
            lane_pause::status(&state.runtime)
        )
    });
}

async fn end_pause(state: &Arc<ModuleState>) {
    call(state, "lanes.resume", json!({})).await;
    wait_phase(state, "running").await;
}

fn fixture_lock_path(fixture: &AneLaneFixture) -> PathBuf {
    let path = runtime_ane_lock_path(&fixture.state.runtime)
        .unwrap()
        .to_path_buf();
    // SQLite may canonicalize macOS's /var -> /private/var temp-directory alias.
    assert_eq!(
        path.parent().unwrap().canonicalize().unwrap(),
        fixture.root.canonicalize().unwrap()
    );
    path
}

fn install_lock(fixture: &AneLaneFixture) -> PathBuf {
    let path = fixture_lock_path(fixture);
    *fixture.state.runtime.ane_supervisor.lock().unwrap() =
        Some(acquire_runtime_ane_supervisor(&fixture.state.runtime).unwrap());
    path
}

async fn query(state: &Arc<ModuleState>) -> Value {
    call(
        state,
        "embed.query",
        json!({"model":ANE_TEST_LANE,"text":"a"}),
    )
    .await
}

async fn pause_fixture() -> AneLaneFixture {
    let fixture = installed_ready_ane_fixture().await;
    let pinned = resolved_catalog_lane(&fixture.state.runtime, ANE_TEST_LANE)
        .unwrap()
        .1
        .fingerprint
        .clone();
    let mut slots = fixture.state.runtime.catalog.lock().unwrap();
    let slot = slots.get_mut(ANE_TEST_LANE).unwrap();
    // The synthetic tokenizer changes the fixture's derived identity. Production
    // loads enforce the catalog pin; use that same served identity in this test.
    slot.spec.fingerprint = Fingerprint(pinned.clone());
    Arc::get_mut(slot.loaded.as_mut().unwrap())
        .unwrap()
        .fingerprint = Fingerprint(pinned);
    drop(slots);
    fixture
}

#[tokio::test]
async fn paused_lane_refuses_before_load_or_self_check_with_the_served_fingerprint() {
    let fixture = pause_fixture().await;
    let state = &fixture.state;
    let before = query(state).await;
    assert!(before.get("error").is_none(), "{before}");
    assert!(before["fingerprint"]
        .as_str()
        .is_some_and(|f| !f.is_empty()));
    let loads = fixture.control.loads();
    let starts = fixture.control.worker_starts();
    let checks = state
        .runtime
        .lane_check_tasks
        .load(std::sync::atomic::Ordering::SeqCst);
    let load_tasks = state
        .runtime
        .model_load_tasks
        .load(std::sync::atomic::Ordering::SeqCst);
    let paused = start_pause(state).await;
    wait_phase(state, "paused").await;
    let refused = query(state).await;
    for _ in 0..50 {
        tokio::task::yield_now().await;
    }
    let error = &refused["error"];
    assert_eq!(error["code"], "lane_paused", "{refused}");
    assert_eq!(error["class"], "transient");
    assert_eq!(error["safe_to_retry_same_request"], true);
    assert!(error["retry_after_ms"]
        .as_u64()
        .is_some_and(|ms| (1000..=60_000).contains(&ms)));
    assert_eq!(error["details"]["lane_id"], ANE_TEST_LANE);
    assert_eq!(error["details"]["fingerprint"], before["fingerprint"]);
    assert_eq!(
        error["details"]["paused_until_ms"],
        paused["paused_until_ms"]
    );
    assert_eq!(error["details"]["reason"], "exclusive measurement");
    assert_eq!(fixture.control.loads(), loads);
    assert_eq!(
        state
            .runtime
            .model_load_tasks
            .load(std::sync::atomic::Ordering::SeqCst),
        load_tasks
    );
    assert_eq!(fixture.control.worker_starts(), starts);
    assert_eq!(
        state
            .runtime
            .lane_check_tasks
            .load(std::sync::atomic::Ordering::SeqCst),
        checks
    );
    assert!(model_slot_snapshot(&state.runtime, ANE_TEST_LANE)
        .unwrap()
        .loaded
        .is_none());
    let list = call(state, "models.list", json!({})).await;
    let row = list["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["model_id"] == ANE_TEST_LANE)
        .unwrap();
    assert_eq!(row["paused_until_ms"], paused["paused_until_ms"]);
    assert_eq!(row["pause_reason"], "exclusive measurement");
    end_pause(state).await;
    let list = call(state, "models.list", json!({})).await;
    let row = list["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["model_id"] == ANE_TEST_LANE)
        .unwrap();
    assert!(row.get("paused_until_ms").is_none());
    assert!(row.get("pause_reason").is_none());
    finish_ane_lane_fixture(fixture).await;
}

fn add_unaffected_metal(fixture: &AneLaneFixture, rerank: bool) -> String {
    let profile = if rerank {
        "qwen3-reranker-0.6b.owned-metal"
    } else {
        "gte-modernbert-base.owned-metal"
    };
    let mut spec = catalog_fixture_config(profile);
    spec.model_id = if rerank {
        "legacy-metal-rerank"
    } else {
        "legacy-metal-embed"
    }
    .into();
    let mut model = catalog_test_model(&fixture.root.join(&spec.model_id), &spec);
    let model_mut = Arc::get_mut(&mut model).unwrap();
    model_mut.engine_identity.build_flags.remove("profile");
    spec.engine_identity.build_flags.remove("profile");
    let EmbedBackend::Worker(engine) = &model_mut.backend else {
        unreachable!()
    };
    engine
        .lock()
        .unwrap()
        .attach_mock_worker_for_test(catalog_boundary_worker);
    fixture
        .state
        .store
        .store_class_scoped_cert_row(&ClassScopedCertificationRow {
            certification_class: if rerank {
                CertificationClass::Rerank
            } else {
                CertificationClass::Embedding
            },
            assurance_class: AssuranceClass::Measured,
            status: CertificationStatus::Certified,
            key_hash: fixture.state.machine_profile_hash.clone(),
            machine_profile_hash: Some(fixture.state.machine_profile_hash.clone()),
            remote_profile_hash: None,
            identity_revision: None,
            numeric_profile_id: Some(spec.numeric_profile_id.clone()),
            fingerprint: spec.fingerprint.clone(),
            certified_at_ms: 11,
            os_build: fixture.state.machine_profile.os_build.clone(),
            module_generation: 1,
            evidence: json!({}),
        })
        .unwrap();
    let id = spec.model_id.clone();
    fixture.state.runtime.catalog.lock().unwrap().insert(
        id.clone(),
        ModelSlot {
            spec,
            loaded: Some(model),
            state: ModelRuntimeState::Ready,
            notify: Arc::new(Notify::new()),
            last_cold_load_ms: None,
        },
    );
    id
}

#[tokio::test(flavor = "multi_thread")]
async fn paused_ane_does_not_block_metal_embeddings_or_rerank() {
    let fixture = installed_ready_ane_fixture().await;
    let embed = tokio::task::block_in_place(|| add_unaffected_metal(&fixture, false));
    let rerank = tokio::task::block_in_place(|| add_unaffected_metal(&fixture, true));
    start_pause(&fixture.state).await;
    wait_phase(&fixture.state, "paused").await;
    let embedded = call(
        &fixture.state,
        "embed.query",
        json!({"model":embed,"text":"a"}),
    )
    .await;
    assert!(embedded.get("error").is_none(), "{embedded}");
    assert_eq!(embedded["vectors"][0]["vector"], json!([1.0]));
    let ranked = call(
        &fixture.state,
        "rerank.score",
        json!({"model":rerank,"query":"a","candidates":["a"]}),
    )
    .await;
    assert!(ranked.get("error").is_none(), "{ranked}");
    assert_eq!(ranked["scores"], json!([0.5]));
    end_pause(&fixture.state).await;
    finish_ane_lane_fixture(fixture).await;
}

#[tokio::test]
async fn request_admitted_before_pause_finishes_normally() {
    let fixture = installed_ready_ane_fixture().await;
    assert!(query(&fixture.state).await.get("error").is_none());
    let (started, release) = ane_gate();
    fixture
        .control
        .hold_next_inference(started.clone(), release.clone());
    let running = tokio::spawn({
        let state = fixture.state.clone();
        async move { query(&state).await }
    });
    started.notified().await;
    start_pause(&fixture.state).await;
    assert_eq!(
        lane_pause::status(&fixture.state.runtime)["phase"],
        "draining"
    );
    release.notify_one();
    let result = running.await.unwrap();
    assert!(result.get("error").is_none(), "{result}");
    assert_eq!(result["vectors"][0]["vector"], json!([128.0]));
    wait_phase(&fixture.state, "paused").await;
    end_pause(&fixture.state).await;
    finish_ane_lane_fixture(fixture).await;
}

#[tokio::test]
async fn pause_unloads_only_after_cancelled_request_reply_drains() {
    let fixture = installed_ready_ane_fixture().await;
    let path = install_lock(&fixture);
    assert!(query(&fixture.state).await.get("error").is_none());
    let (started, release) = ane_gate();
    fixture
        .control
        .hold_next_inference(started.clone(), release.clone());
    let running = tokio::spawn({
        let state = fixture.state.clone();
        async move { query(&state).await }
    });
    started.notified().await;
    running.abort();
    assert!(running.await.unwrap_err().is_cancelled());
    start_pause(&fixture.state).await;
    for _ in 0..50 {
        tokio::task::yield_now().await;
    }
    assert_eq!(
        lane_pause::status(&fixture.state.runtime)["hardware_free"],
        false
    );
    assert_eq!(fixture.control.confirmed_exits(), 0);
    assert!(model_slot_snapshot(&fixture.state.runtime, ANE_TEST_LANE)
        .unwrap()
        .loaded
        .is_some());
    assert!(AneDirectLaneLock::acquire(&path).is_err());
    release.notify_one();
    wait_phase(&fixture.state, "paused").await;
    assert_eq!(fixture.control.confirmed_exits(), 1);
    assert!(model_slot_snapshot(&fixture.state.runtime, ANE_TEST_LANE)
        .unwrap()
        .loaded
        .is_none());
    end_pause(&fixture.state).await;
    finish_ane_lane_fixture(fixture).await;
}

#[tokio::test]
async fn pause_releases_fresh_open_global_lock_and_resume_holds_it_again() {
    let fixture = installed_ready_ane_fixture().await;
    let path = install_lock(&fixture);
    assert!(AneDirectLaneLock::acquire(&path).is_err());
    start_pause(&fixture.state).await;
    wait_phase(&fixture.state, "paused").await;
    let private =
        AneDirectLaneLock::acquire(&path).expect("hardware must be free through a fresh open");
    drop(private);
    end_pause(&fixture.state).await;
    assert!(
        AneDirectLaneLock::acquire(&path).is_err(),
        "resumed module must hold the lock"
    );
    finish_ane_lane_fixture(fixture).await;
}

#[tokio::test]
async fn pause_auto_expires_with_tokio_time_without_resume() {
    let fixture = installed_ready_ane_fixture().await;
    let path = install_lock(&fixture);
    start_pause(&fixture.state).await;
    wait_phase(&fixture.state, "paused").await;
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(61)).await;
    for _ in 0..100 {
        tokio::task::yield_now().await;
    }
    assert_eq!(
        lane_pause::status(&fixture.state.runtime)["phase"],
        "running"
    );
    assert!(AneDirectLaneLock::acquire(&path).is_err());
    assert!(lane_pause::admit(&fixture.state.runtime, ANE_TEST_LANE).is_ok());
    tokio::time::resume();
    finish_ane_lane_fixture(fixture).await;
}

#[tokio::test]
async fn failed_pause_drain_expires_and_serves_again_with_global_lock_held() {
    let fixture = pause_fixture().await;
    let path = install_lock(&fixture);
    assert!(query(&fixture.state).await.get("error").is_none());
    *fixture
        .state
        .runtime
        .lane_pause
        .drain_failure
        .lock()
        .unwrap() = Some((
        ANE_TEST_LANE.into(),
        "injected persistent unload failure".into(),
    ));
    tokio::time::pause();
    start_pause(&fixture.state).await;
    for _ in 0..100 {
        tokio::task::yield_now().await;
    }
    assert!(
        fixture
            .state
            .runtime
            .lane_pause
            .drain_attempts
            .load(std::sync::atomic::Ordering::SeqCst)
            > 0
    );
    let status = lane_pause::status(&fixture.state.runtime);
    assert_eq!(status["hardware_free"], false);
    assert_eq!(status["phase"], "drain_failed");
    assert_eq!(status["drain_error"]["lane_id"], ANE_TEST_LANE);
    assert_eq!(
        status["drain_error"]["message"],
        "injected persistent unload failure"
    );
    tokio::time::advance(Duration::from_secs(61)).await;
    for _ in 0..100 {
        tokio::task::yield_now().await;
    }
    assert_eq!(
        lane_pause::status(&fixture.state.runtime)["phase"],
        "running"
    );
    assert!(AneDirectLaneLock::acquire(&path).is_err());
    tokio::time::resume();
    let reply = query(&fixture.state).await;
    assert!(reply.get("error").is_none(), "{reply}");
    finish_ane_lane_fixture(fixture).await;
}

#[tokio::test]
async fn resume_ends_pause_while_admitted_request_is_still_draining() {
    let fixture = pause_fixture().await;
    let path = install_lock(&fixture);
    let admitted = lane_pause::admit(&fixture.state.runtime, ANE_TEST_LANE).unwrap();
    tokio::time::pause();
    start_pause(&fixture.state).await;
    for _ in 0..100 {
        tokio::task::yield_now().await;
    }
    assert_eq!(
        lane_pause::status(&fixture.state.runtime)["phase"],
        "draining"
    );
    call(&fixture.state, "lanes.resume", json!({})).await;
    for _ in 0..100 {
        tokio::task::yield_now().await;
    }
    assert_eq!(
        lane_pause::status(&fixture.state.runtime)["phase"],
        "running"
    );
    assert_eq!(
        lane_pause::status(&fixture.state.runtime)["hardware_free"],
        false
    );
    assert!(AneDirectLaneLock::acquire(&path).is_err());
    drop(admitted);
    tokio::time::resume();
    let reply = query(&fixture.state).await;
    assert!(reply.get("error").is_none(), "{reply}");
    finish_ane_lane_fixture(fixture).await;
}

#[tokio::test]
async fn resume_detaches_unload_without_publishing_a_retired_model_or_releasing_the_lock() {
    let fixture = pause_fixture().await;
    let path = install_lock(&fixture);
    let (started, release) = ane_gate();
    fixture
        .control
        .hold_next_exit(started.clone(), release.clone());
    start_pause(&fixture.state).await;
    tokio::time::timeout(Duration::from_secs(5), started.notified())
        .await
        .unwrap();
    let selected_before_resume = model_slot_snapshot(&fixture.state.runtime, ANE_TEST_LANE)
        .unwrap()
        .loaded
        .is_some();
    call(&fixture.state, "lanes.resume", json!({})).await;
    wait_phase(&fixture.state, "running").await;
    let selected_after_resume = model_slot_snapshot(&fixture.state.runtime, ANE_TEST_LANE)
        .unwrap()
        .loaded
        .is_some();
    let lifecycle = catalog_lane_lock(&fixture.state.runtime, ANE_TEST_LANE);
    let still_draining = lifecycle.try_lock().is_err();
    let lock_held = AneDirectLaneLock::acquire(&path).is_err();
    release.notify_one();
    tokio::time::timeout(Duration::from_secs(5), async {
        while fixture.control.confirmed_exits() == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(
        !selected_before_resume && !selected_after_resume,
        "a retired model must not be selectable after cancelling drain"
    );
    assert!(
        still_draining,
        "the detached unload must keep its lifecycle guard"
    );
    assert!(lock_held);
    assert!(
        AneDirectLaneLock::acquire(&path).is_err(),
        "a detached drain must not release the resumed module's lock"
    );
    finish_ane_lane_fixture(fixture).await;
}

#[tokio::test]
async fn retained_lock_descriptor_cannot_extend_pause_past_its_deadline() {
    let fixture = pause_fixture().await;
    let path = install_lock(&fixture);
    let retained = fixture
        .state
        .runtime
        .ane_supervisor
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .inherited_lane_lock()
        .unwrap()
        .unwrap();
    start_pause(&fixture.state).await;
    wait_phase(&fixture.state, "drain_failed").await;
    assert_eq!(
        lane_pause::status(&fixture.state.runtime)["drain_error"]["stage"],
        "releasing_global_lock"
    );
    assert_eq!(
        lane_pause::status(&fixture.state.runtime)["hardware_free"],
        false
    );
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(61)).await;
    for _ in 0..100 {
        tokio::task::yield_now().await;
    }
    assert_eq!(
        lane_pause::status(&fixture.state.runtime)["phase"],
        "running"
    );
    assert!(lane_pause::admit(&fixture.state.runtime, ANE_TEST_LANE).is_ok());
    assert!(AneDirectLaneLock::acquire(&path).is_err());
    drop(retained);
    tokio::time::resume();
    finish_ane_lane_fixture(fixture).await;
}

#[tokio::test]
async fn pause_rejects_121_minutes() {
    let fixture = installed_ready_ane_fixture().await;
    let outcome = dispatch_request(
        fixture.state.clone(),
        MethodEnvelope {
            method: "lanes.pause".into(),
            params: json!({"scope":"neural_engine","duration_ms":121*60_000,"reason":"too long"}),
        },
        None,
    )
    .await;
    assert_eq!(
        refusal_code(Some(outcome)).as_deref(),
        Some("invalid_request")
    );
    assert_eq!(
        lane_pause::status(&fixture.state.runtime)["phase"],
        "running"
    );
    finish_ane_lane_fixture(fixture).await;
}

#[tokio::test]
async fn pause_reacquire_contention_keeps_refusing_instead_of_serving() {
    let fixture = installed_ready_ane_fixture().await;
    let path = install_lock(&fixture);
    start_pause(&fixture.state).await;
    wait_phase(&fixture.state, "paused").await;
    let private = AneDirectLaneLock::acquire(&path).unwrap();
    call(&fixture.state, "lanes.resume", json!({})).await;
    wait_phase(&fixture.state, "waiting_for_lock").await;
    let refused = query(&fixture.state).await;
    assert_eq!(refused["error"]["code"], "lane_paused", "{refused}");
    assert_eq!(refused["error"]["retry_after_ms"], 1000);
    assert_eq!(fixture.control.worker_starts(), 1);
    assert!(lane_pause::admit(&fixture.state.runtime, ANE_TEST_LANE).is_err());
    drop(private);
    wait_phase(&fixture.state, "running").await;
    assert!(AneDirectLaneLock::acquire(&path).is_err());
    finish_ane_lane_fixture(fixture).await;
}

#[tokio::test]
async fn pause_covers_cold_coreml_and_omits_unknown_legacy_fingerprint() {
    let fixture = installed_ready_ane_fixture().await;
    assert!(fixture
        .state
        .runtime
        .ane_supervisor
        .lock()
        .unwrap()
        .is_none());
    let path = fixture_lock_path(&fixture);
    let mut spec = model_slot_snapshot(&fixture.state.runtime, ANE_TEST_LANE)
        .unwrap()
        .spec;
    spec.model_id = "legacy-coreml".into();
    spec.engine = "ane".into();
    spec.engine_identity.engine = "ane-coreml-worker".into();
    spec.fingerprint = Fingerprint(String::new());
    fixture.state.runtime.catalog.lock().unwrap().insert(
        spec.model_id.clone(),
        ModelSlot {
            spec,
            loaded: None,
            state: ModelRuntimeState::Unloaded,
            notify: Arc::new(Notify::new()),
            last_cold_load_ms: None,
        },
    );
    let pause = start_pause(&fixture.state).await;
    assert!(pause["lane_ids"]
        .as_array()
        .unwrap()
        .contains(&json!("legacy-coreml")));
    let refused = call(
        &fixture.state,
        "embed.query",
        json!({"model":"legacy-coreml","text":"a"}),
    )
    .await;
    assert_eq!(refused["error"]["code"], "lane_paused");
    assert_eq!(refused["error"]["details"]["lane_id"], "legacy-coreml");
    assert!(refused["error"]["details"].get("fingerprint").is_none());
    assert_eq!(
        fixture
            .state
            .runtime
            .model_load_tasks
            .load(std::sync::atomic::Ordering::SeqCst),
        0
    );
    end_pause(&fixture.state).await;
    assert!(
        AneDirectLaneLock::acquire(&path).is_err(),
        "cold resume must hold the fixture's lane lock"
    );
    finish_ane_lane_fixture(fixture).await;
}

#[tokio::test]
async fn queued_catalog_batch_gets_retryable_pause_without_loading() {
    let fixture = installed_ready_ane_fixture().await;
    let (entry, backend) = ane_lane_entry(&fixture.state);
    let params = json!({"model":ANE_TEST_LANE,"texts":["a"],"request_key":"queued-pause"});
    let digest = "queued-pause-digest";
    let record = fixture
        .state
        .store
        .admit_job(
            "queued-pause",
            digest,
            "embed.batch",
            1,
            None,
            &params,
            now_ms(),
            60_000,
            60_000,
        )
        .unwrap()
        .record()
        .clone();
    start_pause(&fixture.state).await;
    run_catalog_embed_job(
        fixture.state.clone(),
        record.clone(),
        entry,
        backend,
        params,
        digest.into(),
    )
    .await;
    let failed = fixture
        .state
        .store
        .get_job(&record.job_id)
        .unwrap()
        .unwrap();
    assert_eq!(failed.state, store::JOB_STATE_FAILED_TRANSIENT);
    let error = failed.error_json.unwrap();
    assert_eq!(error["code"], "lane_paused");
    assert_eq!(error["safe_to_retry_same_request"], true);
    assert!(error["retry_after_ms"].as_u64().unwrap() >= 1000);
    assert_eq!(
        fixture
            .state
            .runtime
            .model_load_tasks
            .load(std::sync::atomic::Ordering::SeqCst),
        0
    );
    end_pause(&fixture.state).await;
    let replay = fixture
        .state
        .store
        .admit_job(
            "queued-pause",
            digest,
            "embed.batch",
            1,
            None,
            &json!({"model":ANE_TEST_LANE,"texts":["a"],"request_key":"queued-pause"}),
            now_ms(),
            60_000,
            60_000,
        )
        .unwrap();
    assert!(matches!(replay, JobAdmission::Admitted(_)));
    assert_ne!(replay.record().job_id, record.job_id);
    finish_ane_lane_fixture(fixture).await;
}

async fn resident_generation(
    fixture: &AneLaneFixture,
    generation: &str,
) -> Vec<(
    Arc<worker_host::ane_residency::DirectAneEngine>,
    worker_host::ane_residency::ModuleMockControl,
)> {
    let state = fixture.state.clone();
    let root = fixture.root.join(generation);
    let generation = generation.to_owned();
    let supervisor = state
        .runtime
        .ane_supervisor
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .clone();
    tokio::task::spawn_blocking(move || {
        [
            ("gte-modernbert-base.ane-direct-worker", ANE_TEST_LANE),
            (
                "qwen3-embedding-0.6b.ane-direct-worker",
                "qwen3-embedding-0.6b-ane",
            ),
        ]
        .into_iter()
        .map(|(profile, lane)| {
            let mut spec = catalog_fixture_config(profile);
            spec.model_id = lane.into();
            let mut model = catalog_test_model(&root.join(lane), &spec);
            let (engine, control) = worker_host::ane_residency::module_mock_engine_resident(
                supervisor.clone(),
                &format!("{generation}-{lane}"),
                profile,
            );
            Arc::get_mut(&mut model).unwrap().backend = EmbedBackend::DirectAne(engine.clone());
            state.runtime.catalog.lock().unwrap().insert(
                lane.into(),
                ModelSlot {
                    spec,
                    loaded: Some(model),
                    state: ModelRuntimeState::Ready,
                    notify: Arc::new(Notify::new()),
                    last_cold_load_ms: None,
                },
            );
            (engine, control)
        })
        .collect()
    })
    .await
    .unwrap()
}

#[tokio::test]
async fn pause_shutdown_refunds_reserved_executables_before_next_worker_generation() {
    let fixture = ane_lane_fixture("pause-residency-generations", true).await;
    let path = install_lock(&fixture);
    let original = fixture.engine.clone();
    tokio::task::spawn_blocking(move || original.unload())
        .await
        .unwrap()
        .unwrap();
    let supervisor = fixture
        .state
        .runtime
        .ane_supervisor
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .clone();
    let old = resident_generation(&fixture, "generation-N").await;
    for (engine, _) in &old {
        for shape in [128, 256] {
            engine
                .serving
                .infer(vec![vec![1; shape]], false)
                .await
                .unwrap();
        }
    }
    // The pinned gte-modernbert-base and qwen3-embedding-0.6b profiles have 22
    // and 28 layers. Each shape reserves one executable per layer: 2*22+2*28=100.
    // The snapshot reads State::reserved_executables, not historical admission totals.
    assert_eq!(supervisor.reservation_snapshot(), (100, 0));
    let (started, release) = ane_gate();
    old[0].1.hold_next_exit(started.clone(), release.clone());
    start_pause(&fixture.state).await;
    tokio::time::timeout(Duration::from_secs(5), started.notified())
        .await
        .unwrap();
    let during_exit = supervisor.reservation_snapshot();
    assert_eq!(
        lane_pause::status(&fixture.state.runtime)["hardware_free"],
        false
    );
    release.notify_one();
    assert_eq!(
        during_exit,
        (100, 0),
        "reservations must remain charged until exit confirmation"
    );
    wait_phase(&fixture.state, "paused").await;
    assert_eq!(
        supervisor.reservation_snapshot(),
        (0, 0),
        "confirmed pause shutdown must refund reserved_executables"
    );
    assert!(supervisor.resident_shapes().is_empty());
    assert!(old
        .iter()
        .all(|(_, control)| control.confirmed_exits() == 1));
    end_pause(&fixture.state).await;
    assert!(AneDirectLaneLock::acquire(&path).is_err());
    let next = resident_generation(&fixture, "generation-N+1").await;
    for (engine, _) in &next {
        for shape in [128, 256] {
            let result = engine.serving.infer(vec![vec![1; shape]], false).await;
            assert!(
                result.is_ok(),
                "next generation must fit the full 100-executable budget: {result:?}"
            );
        }
    }
    assert_eq!(supervisor.reservation_snapshot(), (100, 0));
    assert_eq!(
        supervisor.stats().evicted,
        0,
        "no eviction against old worker reservations"
    );
    start_pause(&fixture.state).await;
    wait_phase(&fixture.state, "paused").await;
    end_pause(&fixture.state).await;
    drop((old, next));
    finish_ane_lane_fixture(fixture).await;
}
