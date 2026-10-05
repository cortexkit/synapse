//! Opt-in certification of an extracted release candidate on the M5.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires release candidate ck-synapse, M5 and pinned model weights"]
fn metal_m5_live_candidate_writes_passed_without_floor_probe() {
    use std::path::PathBuf;
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = crate_root.join(".live/hardware-checkout");
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
    let assets = crate_root.join(".live/metal-candidate");
    std::fs::create_dir_all(&assets).unwrap();
    std::fs::copy(candidate, assets.join("ck-synapse")).unwrap();
    // The assets directory has no worker binary: any hardware-floor subprocess
    // mistakenly invoked for in-process Metal must fail instead of going unnoticed.
    let output = std::process::Command::new(assets.join("ck-synapse"))
        .args([
            "certify",
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
        .arg("--weights")
        .arg(
            crate_root
                .join(".live/weights/gte-modernbert-base")
                .canonicalize()
                .unwrap(),
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
    let bytes = std::fs::read(record.path(&root)).unwrap();
    println!("{}", String::from_utf8(bytes.clone()).unwrap());
    let stored: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(stored, serde_json::to_value(record).unwrap());
}
