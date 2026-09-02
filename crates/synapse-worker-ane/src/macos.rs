use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::Command;

use anyhow::{bail, Result};

/// File name of the Swift CoreML worker when it ships beside this launcher.
pub const SWIFT_WORKER_SIBLING: &str = "ck-synapse-worker-ane-swift";

/// Locates the Swift worker this launcher execs.
///
/// Resolution order matters for deployment: the build-time path points into
/// cargo's OUT_DIR on the machine that compiled the launcher, which does not
/// exist on any machine the binary is installed to. Installed layouts ship the
/// Swift worker next to the launcher, so that location wins; the OUT_DIR path
/// only serves cargo-driven runs (tests, `cargo run`) on the build machine.
fn resolve_swift_worker() -> Option<PathBuf> {
    if let Some(explicit) = std::env::var_os("SYNAPSE_ANE_SWIFT_WORKER") {
        let explicit = PathBuf::from(explicit);
        if explicit.is_file() {
            return Some(explicit);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let sibling = dir.join(SWIFT_WORKER_SIBLING);
            if sibling.is_file() {
                return Some(sibling);
            }
        }
    }
    let built = env!("SYNAPSE_ANE_SWIFT_WORKER");
    if !built.is_empty() && std::path::Path::new(built).is_file() {
        return Some(PathBuf::from(built));
    }
    None
}

pub fn main() -> Result<()> {
    let Some(worker) = resolve_swift_worker() else {
        bail!(
            "synapse-worker-ane Swift worker not found: expected {SWIFT_WORKER_SIBLING} beside \
             the launcher, SYNAPSE_ANE_SWIFT_WORKER, or the build-time artifact"
        );
    };
    let error = Command::new(worker)
        .args(std::env::args_os().skip(1))
        .exec();
    Err(error.into())
}
