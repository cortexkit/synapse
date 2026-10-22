use super::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use worker_host::ane_residency::AneDirectLaneLock;

const MODEL: &str = "coreml-load-fixture";

struct CoreMlFixture {
    root: PathBuf,
    state: Arc<ModuleState>,
    loads: Arc<AtomicUsize>,
    unloads: Arc<AtomicUsize>,
    expect_inherited_lock: Arc<AtomicBool>,
}

/// A fixture label no other process can share. The root directory, and with
/// it the direct-ANE lock file these tests take, is named after the label,
/// the process id and a counter; test processes in different PID namespaces
/// (separate build jobs on one machine) can share a temporary directory and
/// a process id, and then their lock files collided (`ane_lane_busy`).
fn unique_label(label: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos());
    format!("{label}-{nanos:x}")
}

/// Who holds or has open the lane lock file at `path`, for a failure message:
/// the kernel's lock table entry for its inode and every process with a
/// descriptor on it (Linux only).
fn lock_owners(path: &Path) -> String {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::MetadataExt;
        let Ok(meta) = fs::metadata(path) else {
            return "the lock file is missing".into();
        };
        let inode = format!(":{} ", meta.ino());
        let locks: Vec<String> = fs::read_to_string("/proc/locks")
            .unwrap_or_default()
            .lines()
            .filter(|line| line.contains(&inode))
            .map(str::to_owned)
            .collect();
        let mut open_by = Vec::new();
        for process in fs::read_dir("/proc").into_iter().flatten().flatten() {
            for fd in fs::read_dir(process.path().join("fd")).into_iter().flatten().flatten() {
                if fs::read_link(fd.path()).is_ok_and(|target| target == path) {
                    let command = fs::read_to_string(process.path().join("cmdline"))
                        .unwrap_or_default()
                        .replace('\0', " ");
                    open_by.push(format!(
                        "pid {} fd {}: {command}",
                        process.file_name().to_string_lossy(),
                        fd.file_name().to_string_lossy()
                    ));
                }
            }
        }
        format!(
            "this test process is pid {}; /proc/locks: {locks:?}; open by: {open_by:?}",
            std::process::id()
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = path;
        "lock owners are inspected only on Linux".into()
    }
}

/// Takes the lane lock, or fails with what is known about its holders
/// (on Linux: the kernel's lock entry and the processes with it open).
fn acquire_or_report(path: &Path, context: &str) -> AneDirectLaneLock {
    AneDirectLaneLock::acquire(path)
        .unwrap_or_else(|error| panic!("{context}: {error}; {}", lock_owners(path)))
}

fn fixture(label: &str) -> CoreMlFixture {
    let (root, descriptor) = test_storage_descriptor(label);
    let store = Arc::new(SynapseStore::open(&descriptor).unwrap());
    let profile = test_machine_profile(label);
    store.activate_profile(&profile, 1, 1000).unwrap();
    let state = test_module_state(store, profile);
    make_test_tokenizer(&root, 8);
    let weights = root.join("weights.fixture");
    fs::write(
        &weights,
        b"synthetic weights interpreted only by the mock worker",
    )
    .unwrap();
    let preload: PreloadModelConfig = serde_json::from_value(json!({
        "model_id":MODEL,"engine":"ane","model_path":weights,
        "tokenizer_path":root.join("tokenizer.json"),"max_tokens":8,
        "format":"fixture-coreml","artifact_digest":format!("sha256:{}", "1".repeat(64)),
        "worker_bin":root.join("must-not-start"),"worker_runtime_dir":root.join("worker-runtime")
    }))
    .unwrap();
    let spec = build_preload_catalog_model(0, preload, &state.runtime.inline, &state.runtime.jobs)
        .unwrap();
    register_runtime_catalog_model(&state.runtime, spec).unwrap();
    let loads = Arc::new(AtomicUsize::new(0));
    let unloads = Arc::new(AtomicUsize::new(0));
    let expect_inherited_lock = Arc::new(AtomicBool::new(false));
    let factory_loads = loads.clone();
    let factory_unloads = unloads.clone();
    let factory_inherited = expect_inherited_lock.clone();
    *state.runtime.coreml_test_worker_factory.lock().unwrap() = Some(Arc::new(move |config| {
        assert_eq!(
            config.engine_identity.as_ref().unwrap().engine,
            "ane-coreml-worker"
        );
        assert_eq!(
            config.inherited_lane_lock.is_some(),
            factory_inherited.load(Ordering::SeqCst)
        );
        let engine = worker_host::WorkerEngine::new(config)?;
        let loads = factory_loads.clone();
        let unloads = factory_unloads.clone();
        // Only replace the OS worker. Normal loading admission, supervisor
        // selection, tokenizer validation and framed LOAD/UNLOAD still execute.
        engine.attach_mock_worker_for_test(move |request, _| {
            use synapse_core::{WorkerRequest, WorkerResponse};
            match request {
                WorkerRequest::Load { req_id, .. } => {
                    loads.fetch_add(1, Ordering::SeqCst);
                    (
                        WorkerResponse::Loaded {
                            req_id: req_id.clone(),
                            model_ref: "coreml:fixture".into(),
                            dims: 1,
                            cold_load_ms: 0,
                            buckets: Some(vec![8]),
                        },
                        None,
                    )
                }
                WorkerRequest::Unload { req_id, .. } => {
                    unloads.fetch_add(1, Ordering::SeqCst);
                    (
                        WorkerResponse::Unloaded {
                            req_id: req_id.clone(),
                        },
                        None,
                    )
                }
                WorkerRequest::EmbedBatch { req_id, items, .. } => (
                    WorkerResponse::Vectors {
                        req_id: req_id.clone(),
                        dims: 1,
                        n: items.len(),
                    },
                    Some(synapse_core::encode_f32_frame(&vec![1.0; items.len()])),
                ),
                other => (WorkerResponse::unsupported_request(other), None),
            }
        });
        Ok(engine)
    }));
    CoreMlFixture {
        root,
        state,
        loads,
        unloads,
        expect_inherited_lock,
    }
}

async fn load(fixture: &CoreMlFixture) -> Arc<EmbeddingModel> {
    let result = load_catalog_model_task(fixture.state.clone(), MODEL.into()).await;
    assert!(
        result.is_ok(),
        "CoreML load must not require the direct-ANE lock: {:?}",
        result.as_ref().err()
    );
    let model = result.unwrap();
    assert_eq!(model.engine_identity.engine, "ane-coreml-worker");
    assert_eq!(
        model_runtime_state_name(
            &model_slot_snapshot(&fixture.state.runtime, MODEL)
                .unwrap()
                .state
        ),
        "ready"
    );
    model
}

async fn finish(fixture: CoreMlFixture) {
    let result = response_result(
        model_unload(fixture.state.clone(), json!({"model_id":MODEL})).await,
        "model.unload",
    );
    assert!(result.get("error").is_none(), "{result}");
    assert!(fixture.unloads.load(Ordering::SeqCst) > 0);
    drop(fixture.state);
    fs::remove_dir_all(fixture.root).unwrap();
}

#[tokio::test]
async fn cold_coreml_load_ignores_another_direct_ane_lock_holder() {
    let fixture = fixture("cold-coreml-lock-holder");
    let path = runtime_ane_lock_path(&fixture.state.runtime)
        .unwrap()
        .to_path_buf();
    let holder = acquire_or_report(&path, "the test could not take the lane lock");
    assert!(fixture
        .state
        .runtime
        .ane_supervisor
        .lock()
        .unwrap()
        .is_none());
    let model = load(&fixture).await;
    assert_eq!(fixture.loads.load(Ordering::SeqCst), 1);
    assert!(fixture
        .state
        .runtime
        .ane_supervisor
        .lock()
        .unwrap()
        .is_none());
    assert!(
        AneDirectLaneLock::acquire(&path).is_err(),
        "the existing lock holder must remain untouched"
    );
    let vectors = execute_embedding(
        &fixture.state.runtime,
        &model,
        TokenBatch {
            items: vec![vec![1]],
        },
        None,
        None,
    )
    .await
    .unwrap();
    assert_eq!(vectors, vec![vec![1.0]]);
    drop(holder);
    let replacement =
        acquire_or_report(&path, "CoreML worker must not have acquired the lock");
    drop((model, replacement));
    finish(fixture).await;
}

#[tokio::test]
async fn coreml_load_does_not_reacquire_a_released_supervisor_lock() {
    let fixture = fixture(&unique_label("coreml-released-supervisor"));
    let path = runtime_ane_lock_path(&fixture.state.runtime)
        .unwrap()
        .to_path_buf();
    let supervisor = acquire_runtime_ane_supervisor(&fixture.state.runtime).unwrap();
    supervisor.release_lane().unwrap();
    *fixture.state.runtime.ane_supervisor.lock().unwrap() = Some(supervisor.clone());
    let holder = acquire_or_report(&path, "the test could not take the released lock");
    let model = load(&fixture).await;
    assert_eq!(fixture.loads.load(Ordering::SeqCst), 1);
    assert!(supervisor.held_lane_lock().is_none());
    assert!(AneDirectLaneLock::acquire(&path).is_err());
    drop(holder);
    let replacement =
        acquire_or_report(&path, "CoreML must not have re-acquired the released lock");
    drop((replacement, model));
    finish(fixture).await;
}

async fn wait_phase(state: &ModuleState, phase: &str) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while lane_pause::status(&state.runtime)["phase"] != phase {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap_or_else(|_| {
        panic!(
            "expected {phase}, got {}",
            lane_pause::status(&state.runtime)
        )
    });
}

async fn pause_and_drain(state: &Arc<ModuleState>) {
    let reply = response_result(
        lane_pause::pause(
            state.clone(),
            json!({
                "scope":"neural_engine","duration_ms":60_000,"reason":"CoreML shutdown test"
            }),
        )
        .await,
        "lanes.pause",
    );
    assert!(reply.get("error").is_none(), "{reply}");
    wait_phase(state, "paused").await;
    assert_eq!(lane_pause::status(&state.runtime)["hardware_free"], true);
}

async fn resume(state: &Arc<ModuleState>) {
    lane_pause::resume(state.clone()).await;
    wait_phase(state, "running").await;
}

#[tokio::test]
async fn coreml_pause_unloads_then_resume_only_inherits_an_already_held_lock() {
    let fixture = fixture(&unique_label("coreml-pause-and-reload"));
    let path = runtime_ane_lock_path(&fixture.state.runtime)
        .unwrap()
        .to_path_buf();
    let first = load(&fixture).await;
    pause_and_drain(&fixture.state).await;
    assert_eq!(fixture.unloads.load(Ordering::SeqCst), 1);
    assert!(model_slot_snapshot(&fixture.state.runtime, MODEL)
        .unwrap()
        .loaded
        .is_none());
    let private = acquire_or_report(&path, "pause confirmed the CoreML worker's exit");
    drop(private);
    resume(&fixture.state).await;
    assert!(AneDirectLaneLock::acquire(&path).is_err());
    fixture.expect_inherited_lock.store(true, Ordering::SeqCst);
    let second = load(&fixture).await;
    assert_eq!(fixture.loads.load(Ordering::SeqCst), 2);
    pause_and_drain(&fixture.state).await;
    assert_eq!(fixture.unloads.load(Ordering::SeqCst), 2);
    let private = acquire_or_report(&path, "pause must also close the inherited descriptor");
    drop(private);
    resume(&fixture.state).await;
    drop((first, second));
    finish(fixture).await;
}
