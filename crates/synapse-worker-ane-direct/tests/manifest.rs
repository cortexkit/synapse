use serde_json::Value;
use std::process::Command;

#[test]
fn manifest_cli_reports_provenance_from_the_compiled_tree() {
    let scratch = std::env::temp_dir().join(format!(
        "synapse-ane-direct-manifest-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&scratch).expect("create isolated probe directory");
    let binary = synapse_core::dev_binary::ckdev_binary(
        env!("CARGO_BIN_EXE_ck-synapse-worker-ane-direct"),
        &scratch,
    )
    .expect("publish the worker under a development name");
    let output = Command::new(binary)
        .arg("--manifest")
        .env("XDG_DATA_HOME", scratch.join("data"))
        .env("XDG_RUNTIME_DIR", scratch.join("run"))
        .env("XDG_CONFIG_HOME", scratch.join("config"))
        .output()
        .expect("run the worker manifest probe");
    assert!(
        output.status.success(),
        "--manifest must bypass worker startup: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let manifest: Value = serde_json::from_slice(&output.stdout).expect("manifest is JSON");
    assert_eq!(manifest["module_id"], "synapse-worker-ane-direct");
    let provenance = manifest["provenance"]
        .as_object()
        .expect("SDK worker manifest includes provenance");
    match (
        option_env!("SYNAPSE_BUILD_REV"),
        option_env!("SYNAPSE_BUILD_TREE"),
    ) {
        (Some(revision), Some("clean")) => {
            assert_eq!(provenance["build_git_sha"], revision);
            assert!(provenance.get("build_git_sha_absence_reason").is_none());
        }
        (Some(_), Some("dirty")) => {
            assert!(provenance.get("build_git_sha").is_none());
            assert_eq!(provenance["build_git_sha_absence_reason"], "declined_dirty");
        }
        _ => {
            assert!(provenance.get("build_git_sha").is_none());
            assert_eq!(provenance["build_git_sha_absence_reason"], "no_git_dir");
        }
    }
}
