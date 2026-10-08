//! Embeds the git commit this binary was built from, so `ck provenance synapse`
//! can say which source is running.
//!
//! Two compile-time variables, both optional: `SYNAPSE_BUILD_REV` (the
//! 40-character HEAD commit) and `SYNAPSE_BUILD_TREE` (`clean` or `dirty`).
//! Both stay unset when git is unavailable (a source tarball, a vendored
//! checkout), and the module then declares the commit absent with a reason
//! instead of a placeholder that would compare equal between two unidentified
//! builds.

use std::path::Path;
use std::process::Command;

fn main() {
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
    if revision.len() != 40 || !revision.bytes().all(|b| b.is_ascii_hexdigit()) {
        return;
    }
    // A dirty tree runs code HEAD does not describe, so the tree state travels
    // with the commit and the provenance helper declines to attest it.
    let Some(status) = git(&manifest_dir, &["status", "--porcelain"]) else {
        return;
    };
    let tree = if status.is_empty() { "clean" } else { "dirty" };
    println!("cargo:rustc-env=SYNAPSE_BUILD_REV={revision}");
    println!("cargo:rustc-env=SYNAPSE_BUILD_TREE={tree}");
}

fn git(dir: &str, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        // Every git query in this script must not refresh the index as a side
        // effect: the index is a watched file, so reading provenance would
        // otherwise make Cargo rebuild this crate on the next run.
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
