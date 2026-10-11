//! Shared Git provenance stamping for Synapse Cargo build scripts.

use std::path::Path;
use std::process::Command;

/// Emit Cargo rerun watches and clean/dirty revision variables for the calling crate.
pub fn emit_git_provenance() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    let workspace = Path::new(&manifest_dir).join("../..");
    let git_dir = workspace.join(".git");
    if git_dir.exists() {
        // A worktree's .git entry is a pointer file, not a directory. Resolve
        // real metadata paths instead of registering always-stale missing files.
        let mut names = vec![
            "HEAD".to_string(),
            "index".to_string(),
            "packed-refs".to_string(),
        ];
        // Other branches moving do not change this checkout's provenance, so
        // watch this branch's loose ref rather than the whole common refs tree.
        // Switching branches or detaching rewrites the separately watched HEAD.
        if let Some(reference) = git(&manifest_dir, &["symbolic-ref", "-q", "HEAD"]) {
            names.push(reference);
        }
        for name in names {
            if let Some(path) = git(&manifest_dir, &["rev-parse", "--git-path", &name]) {
                let path = Path::new(&manifest_dir).join(path);
                // Packed references and loose branch refs are optional. A
                // missing watch would force a rebuild on every Cargo invocation.
                if path.exists() {
                    println!("cargo:rerun-if-changed={}", path.display());
                }
            }
        }
        // An unstaged edit touches none of those, and a cached "clean" stamp
        // over edited sources would attest code HEAD does not describe. So the
        // source trees this binary is built from also rerun the script.
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
    // Dirty source cannot be attested to HEAD, so the runtime reports only why
    // this build omitted its commit rather than carrying a misleading revision.
    let Some(status) = git(&manifest_dir, &["status", "--porcelain"]) else {
        return;
    };
    let tree = if status.is_empty() { "clean" } else { "dirty" };
    println!("cargo:rustc-env=SYNAPSE_BUILD_REV={revision}");
    println!("cargo:rustc-env=SYNAPSE_BUILD_TREE={tree}");
}

fn git(dir: &str, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        // Reading provenance must not refresh the watched index and force a
        // rebuild on every Cargo invocation.
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
