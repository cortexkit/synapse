use super::*;
use catalog_probe::{cached_probe, run_probe, ProbeKind, PROBE_TIMEOUT};
use std::process::{Command, Stdio};
use std::sync::atomic::AtomicUsize;

static NEXT: AtomicUsize = AtomicUsize::new(0);

struct Stub {
    root: PathBuf,
    worker: PathBuf,
}

fn stub_binary() -> PathBuf {
    if let Some(path) = env::var_os("SYNAPSE_PROBE_STUB_BINARY") {
        return path.into();
    }
    std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join(if cfg!(windows) {
            "ck-synapse-probe-stub.exe"
        } else {
            "ck-synapse-probe-stub"
        })
}

impl Stub {
    fn new(config: Value) -> Self {
        let root = env::temp_dir().join(format!(
            "synapse-detection-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        let worker = synapse_core::dev_binary::ckdev_binary_hard_link(stub_binary(), &root)
            .expect("build ck-synapse-probe-stub before running isolated --lib tests");
        fs::write(worker.with_extension("json"), config.to_string()).unwrap();
        Self { root, worker }
    }

    fn logs(&self) -> Vec<Value> {
        let text = fs::read_to_string(self.worker.with_extension("log")).unwrap_or_default();
        let rows: Vec<Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        for row in &rows {
            assert_eq!(row["nonce"], false, "launch nonce leaked: {row}");
            assert_eq!(row["nonce_fd"], false, "launch nonce FD leaked: {row}");
            assert!(!row["args"]
                .as_array()
                .unwrap()
                .iter()
                .any(|arg| arg == "--model"));
        }
        rows
    }

    fn counts(&self) -> (usize, usize) {
        let rows = self.logs();
        (
            rows.iter()
                .filter(|row| row["args"] == json!(["--version"]))
                .count(),
            rows.iter()
                .filter(|row| row["args"] == json!(["--probe-floor"]))
                .count(),
        )
    }

    fn assert_reaped(&self, pid: u32) {
        let mut child = synapse_core::without_launch_nonce(Command::new(&self.worker))
            .arg("--assert-reaped")
            .stdin(Stdio::piped())
            .spawn()
            .unwrap();
        writeln!(child.stdin.take().unwrap(), "{pid}").unwrap();
        assert!(
            child.wait().unwrap().success(),
            "probe was not killed and reaped"
        );
    }
}

impl Drop for Stub {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn cuda_ok() -> Value {
    json!({"status":"ok", "code":"ok", "required":{"driver_api":13020,"compute_capability":{"major":7,"minor":5}},
        "observed":{"driver_api":13030,"compute_capability":{"major":8,"minor":9}}})
}

fn version(backend: &str) -> Value {
    json!({"stdout":format!("ckdev-worker 0.1 features={backend} manifest_digest=fixture\n")})
}

fn refusing(backend: &str, code: &str) -> Value {
    json!({"version":version(backend), "floor":{"exit":2,"envelope":{"status":"refused","code":code,"required":{},"observed":null}}})
}

fn isolated_test(name: &str, extra_env: &[(&str, &str)]) {
    let executable = std::env::current_exe().unwrap();
    let scratch = env::temp_dir().join(format!(
        "synapse-isolated-{}",
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let executable = synapse_core::dev_binary::ckdev_binary(executable, &scratch).unwrap();
    let mut command = synapse_core::without_launch_nonce(Command::new(executable));
    command
        .args(["--exact", name, "--ignored", "--nocapture"])
        .env("SYNAPSE_PROBE_STUB_BINARY", stub_binary())
        .env(subc_protocol::LAUNCH_NONCE_ENV, "probe-test-nonce")
        .env(subc_protocol::LAUNCH_NONCE_FD_ENV, "123");
    for key in [
        "SYNAPSE_TEST_RUNNABLE_BACKENDS",
        "SYNAPSE_TEST_CATALOG",
        "SYNAPSE_CUDA_DRIVER_API",
        "CUDA_DRIVER_API",
        "SYNAPSE_CUDA_COMPUTE_CAPABILITY",
        "CUDA_COMPUTE_CAPABILITY",
        "SYNAPSE_CUDA_PACKAGING_DRIVER",
    ] {
        command.env_remove(key);
    }
    for (key, value) in extra_env {
        command.env(key, value);
    }
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let _ = fs::remove_dir_all(scratch);
}

#[test]
fn nested_envelopes_reason_table_cache_and_overlap() {
    isolated_test("detection_tests::detection_contract_fixture", &[]);
}

#[test]
#[ignore = "executed in a nonce-bearing isolated process by nested_envelopes_reason_table_cache_and_overlap"]
fn detection_contract_fixture() {
    assert!(cuda_floor_override().is_none());
    let cuda = Stub::new(json!({"version":version("cuda"),"floor":{"envelope":cuda_ok()}}));
    let vulkan = Stub::new(
        json!({"version":version("vulkan"),"floor":{"envelope":{"status":"ok","code":null,"required":{},"observed":{"vendor":4318,"api_major":1,"api_minor":3}}}}),
    );
    let (runnable, reasons) =
        detect_gpu_backends(Some(cuda.worker.clone()), Some(vulkan.worker.clone()));
    assert_eq!(runnable, BTreeSet::from(["cuda".into(), "vulkan".into()]));
    assert!(
        reasons.is_empty(),
        "successful backends have no reason token"
    );
    assert!(owned_cuda_floor_decision(Some(&cuda.worker)).is_supported());
    ensure_owned_cuda_floor(Some(&cuda.worker)).unwrap();
    assert_eq!(cuda.counts(), (1, 1));
    assert_eq!(vulkan.counts(), (1, 1));
    for (backend, code, reason) in [
        ("cuda", "cuda_no_driver", "driver_missing"),
        (
            "cuda",
            "cuda_runtime_missing:libcublas.so.13",
            "runtime_missing",
        ),
        ("cuda", "cuda_driver_too_old", "device_below_floor"),
        (
            "cuda",
            "cuda_compute_capability_too_low",
            "device_below_floor",
        ),
        ("vulkan", "vulkan_no_device", "driver_missing"),
        ("vulkan", "vulkan_software_device", "software_device"),
        ("vulkan", "vulkan_unsupported_vendor", "device_unsupported"),
        ("vulkan", "vulkan_api_too_old", "device_below_floor"),
        (
            "vulkan",
            "vulkan_missing_feature:shaderFloat16",
            "device_below_floor",
        ),
        ("vulkan", "vulkan_insufficient_memory", "device_below_floor"),
        ("vulkan", "model_unsupported", "probe_failed"),
        ("vulkan", "manifest_mismatch", "probe_failed"),
        ("cuda", "unknown_code", "probe_failed"),
    ] {
        let stub = Stub::new(refusing(backend, code));
        assert_eq!(
            detect_worker_backend(backend, Some(&stub.worker)),
            Err(reason),
            "{code}"
        );
        fs::write(
            stub.worker.with_extension("json"),
            json!({"version":version(backend),"floor":{"envelope":cuda_ok()}}).to_string(),
        )
        .unwrap();
        assert_eq!(
            detect_worker_backend(backend, Some(&stub.worker)),
            Err(reason),
            "cached {code}"
        );
        assert_eq!(stub.counts(), (1, 1));
    }
    for (backend, disabled) in [("cuda", "none"), ("vulkan", "none (vulkan disabled)")] {
        let stub = Stub::new(json!({"version":version(disabled),"floor":{"envelope":cuda_ok()}}));
        assert_eq!(
            detect_worker_backend(backend, Some(&stub.worker)),
            Err("backend_not_built")
        );
        assert_eq!(stub.counts(), (1, 0));
    }
    for behavior in [
        json!({"stdout":"malformed"}),
        json!({"exit":7}),
        json!({"stdout":"x".repeat(4097)}),
        json!({"envelope":{"status":"ok","code":null,"required":{},"observed":{"driver_api":13030,"compute_capability":{}}}}),
    ] {
        let stub = Stub::new(json!({"version":version("cuda"),"floor":behavior}));
        assert_eq!(
            detect_worker_backend("cuda", Some(&stub.worker)),
            Err("probe_failed")
        );
        assert_eq!(stub.counts(), (1, 1));
    }
    for version_behavior in [json!({"exit":9}), json!({"stdout":"malformed"})] {
        let stub = Stub::new(json!({"version":version_behavior,"floor":{"envelope":cuda_ok()}}));
        assert_eq!(
            detect_worker_backend("cuda", Some(&stub.worker)),
            Err("probe_failed")
        );
        fs::write(
            stub.worker.with_extension("json"),
            json!({"version":version("cuda"),"floor":{"envelope":cuda_ok()}}).to_string(),
        )
        .unwrap();
        assert_eq!(
            detect_worker_backend("cuda", Some(&stub.worker)),
            Err("probe_failed")
        );
        assert_eq!(stub.counts(), (1, 0));
    }
    assert_eq!(detect_worker_backend("cuda", None), Err("worker_missing"));
    let missing = cuda.root.join("ckdev-absent");
    assert_eq!(
        detect_worker_backend("cuda", Some(&missing)),
        Err("worker_missing")
    );
    let inaccessible = cuda.root.join("ckdev-directory");
    fs::create_dir(&inaccessible).unwrap();
    assert_eq!(
        detect_worker_backend("cuda", Some(&inaccessible)),
        Err("probe_failed")
    );
    let resolved = resolve_catalog_worker(CUDA_WORKER_ENGINE, Some(&cuda.worker)).unwrap();
    assert_eq!(resolved, cuda.worker);
    overlap_is_observed();
}

fn wait_until(mut predicate: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !predicate() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    predicate()
}

fn overlap_is_observed() {
    let cuda = Stub::new(Value::Null);
    let vulkan = Stub::new(Value::Null);
    for (stub, backend) in [(&cuda, "cuda"), (&vulkan, "vulkan")] {
        let config = json!({"version":version(backend),"floor":{"ready":stub.root.join("ready"),"release":stub.root.join("release"),"envelope":cuda_ok()}});
        fs::write(stub.worker.with_extension("json"), config.to_string()).unwrap();
    }
    let paths = (cuda.worker.clone(), vulkan.worker.clone());
    let task = std::thread::spawn(move || detect_gpu_backends(Some(paths.0), Some(paths.1)));
    let overlapped =
        wait_until(|| cuda.root.join("ready").exists() && vulkan.root.join("ready").exists());
    for stub in [&cuda, &vulkan] {
        fs::write(stub.root.join("release"), "release").unwrap();
    }
    let (runnable, reasons) = task.join().unwrap();
    assert!(
        overlapped,
        "both floor children must be in flight before either is released"
    );
    assert_eq!(runnable.len(), 2);
    assert!(reasons.is_empty());
    assert_eq!(cuda.counts(), (1, 1));
    assert_eq!(vulkan.counts(), (1, 1));
}

#[test]
fn each_probe_kind_is_deadlined_killed_reaped_and_stdout_capped() {
    assert_eq!(PROBE_TIMEOUT, Duration::from_secs(10));
    for kind in ["--version", "--probe-floor"] {
        let stub = Stub::new(json!({"version":{"sleep":true},"floor":{"sleep":true}}));
        let mut command = Command::new(&stub.worker);
        command
            .arg(kind)
            .env(subc_protocol::LAUNCH_NONCE_ENV, "nonce")
            .env(subc_protocol::LAUNCH_NONCE_FD_ENV, "123");
        let started = Instant::now();
        assert_eq!(
            run_probe(&mut command, Duration::from_millis(150))
                .unwrap_err()
                .reason,
            "probe_failed"
        );
        assert!(started.elapsed() < Duration::from_secs(3));
        let logs = stub.logs();
        assert_eq!(logs.len(), 1);
        stub.assert_reaped(logs[0]["pid"].as_u64().unwrap() as u32);
        let big = Stub::new(
            json!({"version":{"stdout":"x".repeat(4097)},"floor":{"stdout":"x".repeat(4097)}}),
        );
        assert_eq!(
            run_probe(Command::new(&big.worker).arg(kind), Duration::from_secs(2))
                .unwrap_err()
                .reason,
            "probe_failed"
        );
        assert_eq!(big.logs().len(), 1);
    }
}

#[test]
fn complete_cuda_override_keeps_worker_and_version_admission() {
    isolated_test(
        "detection_tests::complete_cuda_override_fixture",
        &[
            ("SYNAPSE_CUDA_DRIVER_API", "13030"),
            ("SYNAPSE_CUDA_COMPUTE_CAPABILITY", "8.9"),
        ],
    );
}

#[test]
#[ignore = "executed with complete CUDA overrides in an isolated process"]
fn complete_cuda_override_fixture() {
    assert!(cuda_floor_override().is_some());
    assert_eq!(detect_worker_backend("cuda", None), Err("worker_missing"));
    let absent = Stub::new(Value::Null);
    fs::remove_file(&absent.worker).unwrap();
    assert_eq!(
        detect_worker_backend("cuda", Some(&absent.worker)),
        Err("worker_missing")
    );
    assert_eq!(absent.counts(), (0, 0));
    for (version_behavior, expected) in [
        (version("none"), Err("backend_not_built")),
        (json!({"exit":7}), Err("probe_failed")),
        (version("cuda"), Ok(())),
    ] {
        let stub = Stub::new(
            json!({"version":version_behavior,"floor":{"exit":2,"envelope":{"status":"refused","code":"cuda_no_driver"}}}),
        );
        assert_eq!(detect_worker_backend("cuda", Some(&stub.worker)), expected);
        assert_eq!(stub.counts(), (1, 0));
        if expected.is_ok() {
            ensure_owned_cuda_floor(Some(&stub.worker)).unwrap();
            assert_eq!(stub.counts(), (1, 0));
        }
    }
    ensure_owned_cuda_floor(None).unwrap();
    assert_eq!(absent.counts(), (0, 0));
}

#[test]
fn parity_manifest_pins_equal_cuda_floors() {
    let manifest: Value =
        serde_json::from_str(include_str!("../../../bench/parity/models.json")).unwrap();
    let profiles = manifest["profiles"].as_object().unwrap();
    let floor = |profile: &Value| {
        [
            "cuda_min_driver_api",
            "cuda_min_compute_major",
            "cuda_min_compute_minor",
        ]
        .map(|key| profile[key].as_u64().expect("CUDA floor field"))
    };
    let baseline = floor(&profiles["gte-modernbert-base.owned-cuda"]);
    let cuda: Vec<_> = profiles
        .iter()
        .filter(|(id, _)| id.ends_with(".owned-cuda"))
        .collect();
    assert_eq!(cuda.len(), 4);
    assert_eq!(baseline, [13020, 7, 5]);
    for (id, profile) in cuda {
        assert_eq!(floor(profile), baseline, "{id}");
    }
}

#[test]
fn canonical_worker_and_argv_kind_are_distinct_cache_keys() {
    let stub = Stub::new(json!({"version":version("cuda"),"floor":{"envelope":cuda_ok()}}));
    let noncanonical = stub
        .worker
        .parent()
        .unwrap()
        .join(".")
        .join(stub.worker.file_name().unwrap());
    cached_probe(&stub.worker, ProbeKind::Version).unwrap();
    cached_probe(&noncanonical, ProbeKind::Version).unwrap();
    cached_probe(&noncanonical, ProbeKind::Floor).unwrap();
    assert_eq!(stub.counts(), (1, 1));
}

#[test]
fn management_reads_recorded_reasons_without_probing() {
    isolated_test("detection_tests::management_fixture", &[]);
}

#[tokio::test]
#[ignore = "executed in an isolated process with real probe stubs"]
async fn management_fixture() {
    use crate::tests::{
        response_result, test_machine_profile, test_module_state, test_storage_descriptor,
    };
    let cuda = Stub::new(refusing("cuda", "cuda_runtime_missing:libcublas.so.13"));
    let vulkan = Stub::new(refusing("vulkan", "vulkan_missing_feature:shaderFloat16"));
    env::set_var(worker_binary_env_var(CUDA_WORKER_ENGINE), &cuda.worker);
    env::set_var(worker_binary_env_var("owned-vulkan"), &vulkan.worker);
    let (root, descriptor) = test_storage_descriptor("detection-management");
    let store = Arc::new(SynapseStore::open(&descriptor).unwrap());
    let profile = test_machine_profile("detection-os");
    store.observe_profile(&profile, 10, 1).unwrap();
    let mut state = test_module_state(store, profile);
    let state_mut = Arc::get_mut(&mut state).unwrap();
    let runtime = Arc::get_mut(&mut state_mut.runtime).unwrap();
    let (runnable, reasons) =
        detect_gpu_backends(Some(cuda.worker.clone()), Some(vulkan.worker.clone()));
    runtime.runnable_backends = runnable;
    runtime.backend_reasons = reasons;
    // Supply missing CUDA/Vulkan names on the compiled entries for models.catalog
    // and the no-runnable-backend branch of models.download. These branches read
    // backend metadata only; cloned rows are never validated or used to load lanes.
    for entry in &mut runtime.release_catalog.models {
        for backend in ["cuda", "vulkan"] {
            if !entry.backends.iter().any(|row| row.backend == backend) {
                let mut row = entry.backends[0].clone();
                row.backend = backend.into();
                entry.backends.push(row);
            }
        }
    }
    for platform in [
        synapse_core::Platform::Linux,
        synapse_core::Platform::Windows,
    ] {
        assert_eq!(
            catalog_backend_reason(runtime, "cuda", platform),
            Some("runtime_missing")
        );
        assert_eq!(
            catalog_backend_reason(runtime, "vulkan", platform),
            Some("device_below_floor")
        );
        assert_eq!(
            catalog_backend_reason(runtime, "metal", platform),
            Some("not_supported_on_platform")
        );
        assert_eq!(
            catalog_backend_reason(runtime, "ane", platform),
            Some("not_supported_on_platform")
        );
    }
    for backend in ["cuda", "vulkan"] {
        assert_eq!(
            catalog_backend_reason(runtime, backend, synapse_core::Platform::MacOs),
            Some("not_supported_on_platform")
        );
        runtime.runnable_backends.insert(backend.into());
        assert_eq!(
            catalog_backend_reason(runtime, backend, synapse_core::Platform::Linux),
            None
        );
        assert_eq!(
            catalog_backend_reason(runtime, backend, synapse_core::Platform::MacOs),
            Some("not_supported_on_platform")
        );
    }
    runtime.runnable_backends.clear();
    let before = (cuda.counts(), vulkan.counts());
    let rows = response_result(
        models_catalog(state.clone(), json!({})).await,
        "models.catalog",
    );
    assert_eq!(rows["models"].as_array().unwrap().len(), 4);
    for row in rows["models"].as_array().unwrap() {
        let download = response_result(
            models_download(state.clone(), json!({"catalog_id":row["id"]})).await,
            "models.download",
        );
        assert_eq!(download["error"]["code"], "backend_unavailable");
        for backend in ["cuda", "vulkan"] {
            let expected = if cfg!(target_os = "macos") {
                "not_supported_on_platform"
            } else if backend == "cuda" {
                "runtime_missing"
            } else {
                "device_below_floor"
            };
            let reported = row["backends"]
                .as_array()
                .unwrap()
                .iter()
                .find(|b| b["backend"] == backend)
                .unwrap();
            assert_eq!(reported["reason"], expected);
            let error_row = download["error"]["details"]["backends"]
                .as_array()
                .unwrap()
                .iter()
                .find(|b| b["backend"] == backend)
                .unwrap();
            assert_eq!(error_row["reason"], expected);
        }
        let lane_id = format!("{}-cuda", row["id"].as_str().unwrap());
        let refused = response_result(
            models_download(state.clone(), json!({"catalog_id":lane_id})).await,
            "models.download",
        );
        assert_eq!(refused["error"]["code"], "invalid_request");
    }
    assert_eq!((cuda.counts(), vulkan.counts()), before);
    assert_eq!(before, ((1, 1), (1, 1)));
    drop(state);
    fs::remove_dir_all(root).unwrap();
}

fn ack(storage: Option<Value>) -> ModuleHelloAckBody {
    ModuleHelloAckBody {
        negotiated_ver: PROTOCOL_VERSION,
        subc_ops: vec![],
        subc_capabilities: vec![],
        storage,
        machine_id: None,
    }
}

#[tokio::test]
#[should_panic(expected = "synapse boot failed after HELLO_ACK")]
async fn failed_blocking_boot_panics_with_boot_message() {
    let handler = SynapseHandler::new("synapse-test".into(), PathBuf::new());
    handler
        .on_hello_ack(&ack(Some(json!({"malformed":"storage"}))))
        .await;
}

#[cfg(not(target_os = "macos"))]
#[test]
fn joined_hello_ack_callbacks_initialize_once_off_executor() {
    isolated_test("detection_tests::joined_boot_fixture", &[]);
}

#[cfg(not(target_os = "macos"))]
#[tokio::test(flavor = "current_thread")]
#[ignore = "executed with isolated configuration and probe workers"]
async fn joined_boot_fixture() {
    let cuda = Stub::new(Value::Null);
    let vulkan =
        Stub::new(json!({"version":version("none (vulkan disabled)"),"floor":{"sleep":true}}));
    fs::write(cuda.worker.with_extension("json"),json!({"version":version("cuda"),"floor":{"ready":cuda.root.join("ready"),"release":cuda.root.join("release"),"envelope":cuda_ok()}}).to_string()).unwrap();
    let config = cuda.root.join("synapse.jsonc");
    fs::write(&config, "{}").unwrap();
    env::set_var("SYNAPSE_CONFIG_PATH", config);
    env::set_var("CORTEXKIT_MODEL_CACHE", cuda.root.join("cache"));
    env::set_var(worker_binary_env_var(CUDA_WORKER_ENGINE), &cuda.worker);
    env::set_var(worker_binary_env_var("owned-vulkan"), &vulkan.worker);
    let descriptor = StorageDescriptor {
        module_id: "synapse-test".into(),
        storage_namespace: "default".into(),
        isolation: Isolation::Module,
        backend: StorageBackend::Sqlite {
            path: cuda.root.join("store.db").to_string_lossy().into(),
        },
    };
    let ack = ack(Some(serde_json::to_value(descriptor).unwrap()));
    let handler = SynapseHandler::new("synapse-test".into(), PathBuf::new());
    let waiting = async {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !cuda.root.join("ready").exists() && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let in_flight = cuda.root.join("ready").exists();
        let count = handler.inner.initialize_count.load(Ordering::SeqCst);
        let had_state = handler.state().is_some();
        fs::write(cuda.root.join("release"), "release").unwrap();
        assert!(
            in_flight,
            "initialize did not reach floor probe while executor remained responsive"
        );
        assert_eq!(count, 1);
        assert!(!had_state);
    };
    let ((), (), ()) = tokio::join!(
        handler.on_hello_ack(&ack),
        handler.on_hello_ack(&ack),
        waiting
    );
    assert!(handler.state().is_some());
    assert_eq!(handler.inner.initialize_count.load(Ordering::SeqCst), 1);
    handler.on_hello_ack(&ack).await;
    assert_eq!(handler.inner.initialize_count.load(Ordering::SeqCst), 1);
    assert_eq!(cuda.counts(), (1, 1));
    assert_eq!(vulkan.counts(), (1, 0));
}
