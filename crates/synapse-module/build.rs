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
        // HEAD and refs move on a commit or checkout; the index moves on add.
        println!("cargo:rerun-if-changed={}", git_dir.join("HEAD").display());
        println!("cargo:rerun-if-changed={}", git_dir.join("refs").display());
        println!("cargo:rerun-if-changed={}", git_dir.join("index").display());
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
        .args(args)
        .current_dir(dir)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8(output.stdout).ok()?.trim().to_string())
}
