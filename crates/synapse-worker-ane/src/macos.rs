use std::os::unix::process::CommandExt;
use std::process::Command;

use anyhow::{bail, Result};

pub fn main() -> Result<()> {
    let worker = env!("SYNAPSE_ANE_SWIFT_WORKER");
    if worker.is_empty() || !std::path::Path::new(worker).is_file() {
        bail!("synapse-worker-ane Swift worker was not built for this target");
    }
    let error = Command::new(worker)
        .args(std::env::args_os().skip(1))
        .exec();
    Err(error.into())
}
