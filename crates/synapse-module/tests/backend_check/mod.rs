use super::*;
use serde_json::json;
use std::fs;
use synapse_core::{worker_binary_env_var, worker_binary_file_name};

fn configure_probes(command: &mut process::Command, fixture: &Path) {
    command.env("SYNAPSE_PROBE_STUB_STATE", fixture);
    for name in [
        "SYNAPSE_CUDA_DRIVER_API",
        "CUDA_DRIVER_API",
        "SYNAPSE_CUDA_COMPUTE_CAPABILITY",
        "CUDA_COMPUTE_CAPABILITY",
        "SYNAPSE_TEST_RUNNABLE_BACKENDS",
        "SYNAPSE_TEST_CATALOG",
    ] {
        command.env_remove(name);
    }
}

fn sibling_stub(module: &Path, engine: &str) -> PathBuf {
    let name = worker_binary_file_name(engine).unwrap();
    let worker = module.with_file_name(format!("{name}{}", std::env::consts::EXE_SUFFIX));
    fs::copy(env!("CARGO_BIN_EXE_synapse-probe-stub"), &worker).unwrap();
    worker
}

#[tokio::test]
async fn backend_check_matches_models_catalog_with_sibling_probe_workers() {
    let fixture = unique_temp_dir("backend-check");
    fs::create_dir_all(&fixture).unwrap();
    let binary = synapse_core::dev_binary::ckdev_binary(env!("CARGO_BIN_EXE_ck-synapse"), &fixture)
        .expect("publish module in isolated release layout");
    let cuda = sibling_stub(&binary, "owned-cuda");
    let vulkan = sibling_stub(&binary, "owned-vulkan");
    let stub_config = |worker: &Path| {
        fixture
            .join(worker.file_name().unwrap())
            .with_extension("json")
    };
    fs::write(stub_config(&cuda), json!({
        "version": {"stdout": "stub features=cuda"},
        "floor": {"exit": 2, "envelope": {"status":"refused", "code":"cuda_driver_too_old", "required":{}, "observed":null}}
    }).to_string()).unwrap();
    fs::write(stub_config(&vulkan), json!({
        "version": {"stdout": "stub features=vulkan"},
        "floor": {"envelope": {"status":"ok", "code":null, "required":{}, "observed":{"device_local_heaps":[4294967296u64],"max_storage_buffer_range":134217728}}}
    }).to_string()).unwrap();

    let data = fixture.join("standalone-data");
    let invalid_config = fixture.join("invalid-config.jsonc");
    fs::write(&invalid_config, "not a config").unwrap();
    let check = |options: &[&str]| {
        let mut command = process::Command::new(&binary);
        configure_probes(&mut command, &fixture);
        command
            .arg("--check-backend")
            .args(options)
            .env("SYNAPSE_CONFIG_PATH", &invalid_config)
            .env("XDG_DATA_HOME", &data)
            .env("LOCALAPPDATA", &data)
            .env("CORTEXKIT_MODEL_CACHE", data.join("models"))
            .env(
                subc_protocol::SUBC_LAUNCH_NONCE_ENV,
                "not-for-worker-probes",
            )
            .env("SUBC_LAUNCH_NONCE_FD", "not-for-worker-probes")
            .output()
            .expect("run standalone backend check")
    };
    let output = check(&["--json"]);
    assert!(output.status.success(), "{output:?}");
    assert!(!data.exists(), "standalone check created its data folder");
    let standalone: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(standalone["usable"], true);
    let plain = check(&[]);
    assert!(plain.status.success(), "{plain:?}");
    assert_eq!(
        String::from_utf8(plain.stdout)
            .unwrap()
            .lines()
            .collect::<Vec<_>>(),
        ["vulkan"]
    );

    for worker in [&cuda, &vulkan] {
        let log = fs::read_to_string(stub_config(worker).with_extension("log")).unwrap();
        for line in log.lines() {
            let probe: Value = serde_json::from_str(line).unwrap();
            assert_eq!(probe["nonce"], false);
            assert_eq!(probe["nonce_fd"], false);
        }
    }

    let daemon = start_daemon().await;
    let mut command = synapse_module_command(&daemon.connection_file_path, None, None, &[]);
    configure_probes(command.as_std_mut(), &fixture);
    command
        .env(worker_binary_env_var("owned-cuda"), &cuda)
        .env(worker_binary_env_var("owned-vulkan"), &vulkan)
        .env("CORTEXKIT_MODEL_CACHE", fixture.join("supervised-cache"));
    let module = ModuleProcess {
        child: command.spawn().unwrap(),
    };
    let (_daemon, mut module, mut consumer, route) =
        open_route_for_started_module(daemon, module).await;
    let catalog = route_request(
        &mut consumer,
        route,
        100,
        json!({
            "method":"models.catalog", "params":{"query":"gte-modernbert-base", "task":"embed"}
        }),
    )
    .await;
    let mut expected = catalog["result"]["models"][0]["backends"].as_array().unwrap().iter()
        .map(|row| json!({"backend": row["backend"], "runnable": row["runnable"], "reason": row["reason"]}))
        .collect::<Vec<_>>();
    let mut actual = standalone["backends"].as_array().unwrap().clone();
    for rows in [&mut actual, &mut expected] {
        rows.sort_by(|left, right| left["backend"].as_str().cmp(&right["backend"].as_str()));
    }
    assert_eq!(expected.len(), 4);
    assert_eq!(actual, expected);
    module.child.kill().await.unwrap();

    fs::remove_file(&vulkan).unwrap();
    let unavailable = check(&[]);
    assert_eq!(unavailable.status.code(), Some(2), "{unavailable:?}");
    let plain = String::from_utf8(unavailable.stdout).unwrap();
    assert!(plain
        .lines()
        .any(|line| line == "backend_unavailable: cuda: device_below_floor"));
    assert!(plain
        .lines()
        .any(|line| line == "backend_unavailable: vulkan: worker_missing"));

    fs::write(
        stub_config(&cuda),
        r#"{"version":{"stdout":"invalid version reply"}}"#,
    )
    .unwrap();
    let uncertain = check(&["--json"]);
    assert_eq!(uncertain.status.code(), Some(1), "{uncertain:?}");
    let uncertain: Value = serde_json::from_slice(&uncertain.stdout).unwrap();
    assert_eq!(uncertain["usable"], false);
    assert!(uncertain["backends"]
        .as_array()
        .unwrap()
        .iter()
        .any(|row| row
            == &json!({
                "backend":"cuda", "runnable":false, "reason":"probe_failed"
            })));

    let invalid = check(&["--unknown"]);
    assert_eq!(invalid.status.code(), Some(3), "{invalid:?}");
    fs::remove_dir_all(fixture).unwrap();
}
