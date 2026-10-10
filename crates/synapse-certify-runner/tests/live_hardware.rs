//! Opt-in certification of an extracted release candidate on the M5.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires candidate ck-synapse, M5 and pinned HF-layout mirror"]
fn metal_m5_live_candidate_writes_passed_without_floor_probe() {
    use std::path::PathBuf;
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = crate_root.join(format!(".live/hardware-checkout-{}", std::process::id()));
    std::fs::create_dir_all(root.join("bench")).unwrap();
    let parity = root.join("bench/parity");
    if !parity.exists() {
        std::os::unix::fs::symlink(
            crate_root
                .join("../../bench/parity")
                .canonicalize()
                .unwrap(),
            &parity,
        )
        .unwrap();
    }
    let candidate = PathBuf::from(
        std::env::var_os("SYNAPSE_CERTIFY_CANDIDATE")
            .expect("set SYNAPSE_CERTIFY_CANDIDATE to release ck-synapse"),
    );
    let assets = root.join("assets");
    std::fs::create_dir_all(assets.join("darwin-arm64")).unwrap();
    std::fs::hard_link(candidate, assets.join("darwin-arm64/ck-synapse")).unwrap();
    // The assets directory has no worker binary: any hardware-floor subprocess
    // mistakenly invoked for in-process Metal must fail instead of going unnoticed.
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_ckdev-synapse-certify"))
        .args([
            "run",
            "--row",
            "metal-m5",
            "--model",
            "gte-modernbert-base",
            "--assets",
        ])
        .arg(assets.canonicalize().unwrap())
        .arg("--checkout")
        .arg(root.canonicalize().unwrap())
        .arg("--mirror")
        .arg(
            std::env::var_os("SYNAPSE_CERTIFY_MIRROR")
                .expect("set SYNAPSE_CERTIFY_MIRROR to the HF-layout mirror"),
        )
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let record: synapse_certify::Record = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(record.status, "passed");
    assert_eq!(record.row_id, "metal-m5");
    assert_eq!(record.executed_artifacts.len(), 1);
    assert_eq!(record.schema, 3);
    assert_eq!(record.executed_artifacts[0].file, "darwin-arm64/ck-synapse");
    let bytes = std::fs::read(record.path(&root)).unwrap();
    println!("{}", String::from_utf8(bytes.clone()).unwrap());
    let stored: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(stored, serde_json::to_value(&record).unwrap());
    let catalog = catalog();
    synapse_certify::validate_record(&record, &root, &assets, &catalog).unwrap();
    let alias = synapse_core::dev_binary::ckdev_binary_hard_link(
        assets.join("darwin-arm64/ck-synapse"),
        root.join("validate-bin"),
    )
    .unwrap();
    let validate = || {
        synapse_core::without_launch_nonce(std::process::Command::new(&alias))
            .args(["certify", "validate", "--assets"])
            .arg(&assets)
            .arg("--checkout")
            .arg(&root)
            .output()
            .unwrap()
    };
    let alone = validate();
    assert!(!alone.status.success());
    assert!(
        String::from_utf8_lossy(&alone.stderr).contains("missing_certification_cell:"),
        "{}",
        String::from_utf8_lossy(&alone.stderr)
    );

    // These records test matrix enumeration only; the Metal record above is the sole hardware observation.
    let manifest = synapse_certify::self_check::manifest().unwrap();
    for cell in synapse_certify::derive_cells(&catalog.lanes) {
        if cell.row_id == record.row_id && cell.model == record.model {
            continue;
        }
        let mut synthetic = record.clone();
        synthetic.row_id = cell.row_id;
        synthetic.model = cell.model;
        synthetic.operation = catalog.tasks[&synthetic.model].clone();
        synthetic.fingerprint = cell.fingerprint;
        synthetic.profile_id = format!("{}.{}", synthetic.model, cell.engine);
        let fixtures =
            synapse_parity::evaluator::FixtureSet::load(&parity, &manifest, &synthetic.model)
                .unwrap();
        synthetic.parity.model = synthetic.model.clone();
        synthetic.parity.operation = synthetic.operation.clone();
        synthetic.parity.profile_id = synthetic.profile_id.clone();
        synthetic.parity.fingerprint = synthetic.fingerprint.clone();
        synthetic.parity.fixture_set_id =
            synapse_parity::evaluator::fixture_set_id(&manifest, &synthetic.model);
        synthetic.parity.completed_fixtures = fixtures.cases().len();
        synthetic.parity.expected_fixtures = fixtures.cases().len();
        synthetic.parity.tolerance_class =
            match manifest.profiles[&synthetic.profile_id].rerank_tolerance_class {
                synapse_parity::manifest::ToleranceClass::Fp32 => "fp32",
                synapse_parity::manifest::ToleranceClass::Fp16 => "fp16",
            }
            .into();
        if !matches!(synthetic.row_id.as_str(), "metal-m5" | "ane-m5") {
            synthetic.machine = serde_json::json!({"system_product_name":"synthetic", "gpu":{"uuid_sha256":synapse_certify::gpu_uuid_sha256("GPU-synthetic")}});
        }
        if synthetic.row_id == "ane-m5" {
            let layers = manifest
                .model(&synthetic.model)
                .unwrap()
                .architecture
                .int("num_hidden_layers")
                .unwrap() as usize;
            synthetic.placement = Some(synapse_certify::Placement {
                layer_count: layers,
                admitted_count: 1,
                inventories: vec![synapse_certify::Inventory {
                    executables: vec![synapse_certify::Executable {
                        id: "synthetic".into(),
                        layers: (0..layers).collect(),
                    }],
                    cpu_stages: vec!["token_embedding".into()],
                }],
            });
        }
        synapse_certify::write_record(&synthetic, &root).unwrap();
    }
    let complete = validate();
    assert!(
        complete.status.success(),
        "{}",
        String::from_utf8_lossy(&complete.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&complete.stdout).unwrap();
    assert_eq!(
        value["eligible"].as_array().unwrap().len(),
        synapse_certify::derive_cells(&catalog.lanes).len()
    );
}

#[cfg(target_os = "macos")]
fn catalog() -> synapse_certify::Catalog {
    let raw: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../synapse-module/src/catalog/models.json"
    ))
    .unwrap();
    let mut catalog = synapse_certify::Catalog::default();
    for model in raw["models"].as_array().unwrap() {
        let id = model["id"].as_str().unwrap();
        catalog
            .tasks
            .insert(id.into(), model["task"].as_str().unwrap().into());
        for backend in model["backends"].as_array().unwrap() {
            catalog.lanes.push((
                id.into(),
                backend["backend"].as_str().unwrap().into(),
                backend["engine"].as_str().unwrap().into(),
                backend["fingerprint"].as_str().unwrap().into(),
            ));
        }
    }
    catalog
}
