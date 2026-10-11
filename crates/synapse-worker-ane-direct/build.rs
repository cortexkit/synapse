use sha2::{Digest, Sha256};
use std::path::Path;
use std::process::Command;

fn main() {
    stamp_git_provenance();

    let path = "../../bench/parity/models.json";
    println!("cargo:rerun-if-changed={path}");
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    fn canonical(value: &serde_json::Value, out: &mut Vec<u8>) {
        match value {
            serde_json::Value::Object(map) => {
                let mut keys: Vec<_> = map.keys().collect();
                keys.sort();
                out.push(b'{');
                for (i, key) in keys.into_iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    out.extend(serde_json::to_vec(key).unwrap());
                    out.push(b':');
                    canonical(&map[key], out);
                }
                out.push(b'}');
            }
            serde_json::Value::Array(items) => {
                out.push(b'[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    canonical(item, out);
                }
                out.push(b']');
            }
            scalar => out.extend(serde_json::to_vec(scalar).unwrap()),
        }
    }
    let mut bytes = Vec::new();
    canonical(&manifest, &mut bytes);
    println!(
        "cargo:rustc-env=ANE_MANIFEST_DIGEST={:x}",
        Sha256::digest(&bytes)
    );
    std::fs::write(
        std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("models.json"),
        bytes,
    )
    .unwrap();
}

fn stamp_git_provenance() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let workspace = Path::new(&manifest_dir).join("../..");
    let git_dir = workspace.join(".git");
    if git_dir.exists() {
        let mut names = vec![
            "HEAD".to_string(),
            "index".to_string(),
            "packed-refs".to_string(),
        ];
        if let Some(reference) = git(&manifest_dir, &["symbolic-ref", "-q", "HEAD"]) {
            names.push(reference);
        }
        for name in names {
            if let Some(path) = git(&manifest_dir, &["rev-parse", "--git-path", &name]) {
                let path = Path::new(&manifest_dir).join(path);
                if path.exists() {
                    println!("cargo:rerun-if-changed={}", path.display());
                }
            }
        }
        println!(
            "cargo:rerun-if-changed={}",
            workspace.join("crates").display()
        );
        println!(
            "cargo:rerun-if-changed={}",
            workspace.join("Cargo.toml").display()
        );
        println!(
            "cargo:rerun-if-changed={}",
            workspace.join("Cargo.lock").display()
        );
    }

    let Some(revision) = git(&manifest_dir, &["rev-parse", "HEAD"]) else {
        return;
    };
    if revision.len() != 40 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return;
    }
    let Some(status) = git(&manifest_dir, &["status", "--porcelain"]) else {
        return;
    };
    let tree = if status.is_empty() { "clean" } else { "dirty" };
    println!("cargo:rustc-env=SYNAPSE_BUILD_REV={revision}");
    println!("cargo:rustc-env=SYNAPSE_BUILD_TREE={tree}");
}

fn git(dir: &str, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .env("GIT_OPTIONAL_LOCKS", "0")
        .args(args)
        .current_dir(dir)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8(output.stdout).ok()?.trim().to_string())
}
