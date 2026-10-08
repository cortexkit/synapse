use std::process::Command;

fn git_output(args: &[&str]) -> Option<String> {
    Command::new("git")
        .arg("--no-optional-locks")
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if let Some(head) = git_output(&["rev-parse", "--git-path", "HEAD"]) {
        println!("cargo:rerun-if-changed={head}");
    }
    // HEAD is usually symbolic and does not change on a commit. Watch its
    // target as well, but never the index: staging does not change the stamp.
    if let Some(reference) = git_output(&["symbolic-ref", "-q", "HEAD"]) {
        if let Some(path) = git_output(&["rev-parse", "--git-path", &reference]) {
            let path = std::path::Path::new(&path);
            if path.is_file() {
                println!("cargo:rerun-if-changed={}", path.display());
            } else if let Some(parent) = path.ancestors().skip(1).find(|path| path.is_dir()) {
                // A commit recreates a packed branch's loose ref without
                // modifying HEAD or packed-refs. Watch its existing parent
                // until the loose ref appears, not a perpetually missing file.
                println!("cargo:rerun-if-changed={}", parent.display());
            }
        }
    }
    watch_existing_git_path("packed-refs");
    let revision = git_output(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_owned());
    println!("cargo:rustc-env=SYNAPSE_RIG_GIT_REV={revision}");
}

fn watch_existing_git_path(name: &str) {
    if let Some(path) = git_output(&["rev-parse", "--git-path", name]) {
        // Cargo treats a missing watched file as perpetually dirty. Packed
        // refs and loose refs are alternatives and either can be absent.
        if std::path::Path::new(&path).is_file() {
            println!("cargo:rerun-if-changed={path}");
        }
    }
}
