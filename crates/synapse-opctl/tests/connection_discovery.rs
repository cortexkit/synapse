#![forbid(unsafe_code)]

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

struct TempTree(PathBuf);

impl TempTree {
    fn new(label: &str) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock is after the Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "synapse-opctl-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("create temporary test directory");
        Self(path)
    }
}

impl Drop for TempTree {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn write_connection_file(path: &Path) {
    fs::create_dir_all(path.parent().expect("connection file has a parent"))
        .expect("create production connection directory");
    fs::write(
        path,
        serde_json::to_vec(&serde_json::json!({
            "schema": 1,
            "endpoints": [{"host": "127.0.0.1", "port": 1}],
            "key": vec![7_u8; 32],
            "daemon_id": vec![9_u8; 16],
            "pid": std::process::id(),
            "daemon_ver": "test"
        }))
        .expect("encode connection file"),
    )
    .expect("write production connection file");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))
            .expect("restrict connection file permissions");
    }
}

#[test]
fn set_and_wrong_subc_connection_file_refuses_without_fallback() {
    let root = TempTree::new("exclusive-env");
    let home = root.0.join("home");
    let production = home
        .join(".local")
        .join("share")
        .join("cortexkit")
        .join("run")
        .join("subc-connection.json");
    let missing = root.0.join("operator-selected-missing.json");
    write_connection_file(&production);

    let output = Command::new(env!("CARGO_BIN_EXE_ck-synapse-opctl"))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(["models", "list"])
        .env("HOME", &home)
        .env("SUBC_CONNECTION_FILE", &missing)
        .env_remove("XDG_RUNTIME_DIR")
        .output()
        .expect("run ck-synapse-opctl");

    assert!(!output.status.success(), "a missing named path must fail");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(&missing.display().to_string()),
        "failure must name the exclusively selected path; stderr: {stderr}"
    );
    assert!(
        !stderr.contains(&production.display().to_string()),
        "failure must not fall through to the discoverable production file; stderr: {stderr}"
    );
}
