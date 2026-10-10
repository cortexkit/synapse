//! Bounded, nonce-free worker probes shared by catalog detection and CUDA loading.

use std::{
    collections::HashMap,
    io::Read,
    path::{Path, PathBuf},
    process::{Command, ExitStatus, Stdio},
    sync::{Arc, LazyLock, Mutex, OnceLock},
    time::{Duration, Instant},
};

use serde_json::Value;

pub(crate) const PROBE_TIMEOUT: Duration = Duration::from_secs(10);
const STDOUT_LIMIT: u64 = 4096;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum ProbeKind {
    #[cfg(any(not(target_os = "macos"), test))]
    Version,
    Floor,
}

impl ProbeKind {
    fn argument(self) -> &'static str {
        match self {
            #[cfg(any(not(target_os = "macos"), test))]
            Self::Version => "--version",
            Self::Floor => "--probe-floor",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ProbeOutput {
    pub(crate) status: ExitStatus,
    pub(crate) stdout: Vec<u8>,
    pub(crate) stderr: String,
}

#[derive(Clone, Debug)]
pub(crate) struct ProbeError {
    pub(crate) reason: &'static str,
    pub(crate) detail: String,
}

pub(crate) type ProbeResult = Result<ProbeOutput, ProbeError>;
type ProbeEntry = Arc<OnceLock<ProbeResult>>;
static PROBES: LazyLock<Mutex<HashMap<(PathBuf, ProbeKind), ProbeEntry>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub(crate) fn cached_probe(worker: &Path, kind: ProbeKind) -> ProbeResult {
    let worker = std::fs::canonicalize(worker).unwrap_or_else(|_| worker.to_path_buf());
    let entry = PROBES
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .entry((worker.clone(), kind))
        .or_default()
        .clone();
    // A per-key cell serializes identical probes without blocking other workers.
    // Cache refusals too, so a missing driver cannot repeatedly delay requests
    // with another potentially ten-second probe until the module restarts.
    entry
        .get_or_init(|| {
            let mut command = synapse_core::without_launch_nonce(Command::new(worker));
            command.arg(kind.argument());
            run_probe(&mut command, PROBE_TIMEOUT)
        })
        .clone()
}

/// Drop every cached probe of one worker. Tests run many differently configured
/// probe stubs from one shared executable path, because macOS assesses each new
/// executable path on its first launch; a freshly configured stub must not be
/// answered from an earlier configuration's cache entry.
#[cfg(test)]
pub(crate) fn forget_cached_probes(worker: &Path) {
    let worker = std::fs::canonicalize(worker).unwrap_or_else(|_| worker.to_path_buf());
    PROBES
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .retain(|(path, _), _| path != &worker);
}

pub(crate) fn run_probe(command: &mut Command, timeout: Duration) -> ProbeResult {
    let deadline = Instant::now() + timeout;
    // Take ownership of the caller's command to preserve its argv and environment
    // while stripping the daemon launch nonce and nonce-FD environment variables.
    // Worker probes must not inherit capabilities intended only for module boot.
    let program = command.get_program().to_os_string();
    let mut command =
        synapse_core::without_launch_nonce(std::mem::replace(command, Command::new(program)));
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().map_err(|error| ProbeError {
        reason: if error.kind() == std::io::ErrorKind::NotFound {
            "worker_missing"
        } else {
            "probe_failed"
        },
        detail: format!("spawn worker probe: {error}"),
    })?;
    let stdout = child.stdout.take().expect("piped stdout");
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let result = stdout
            .take(STDOUT_LIMIT + 1)
            .read_to_end(&mut bytes)
            .map(|_| bytes);
        let _ = tx.send(result);
    });
    let mut stderr = child.stderr.take().expect("piped stderr");
    let (stderr_tx, stderr_rx) = std::sync::mpsc::sync_channel(1);
    std::thread::spawn(move || {
        let mut tail = Vec::new();
        let mut chunk = [0_u8; 4096];
        while let Ok(count) = stderr.read(&mut chunk) {
            if count == 0 {
                break;
            }
            let discard = (tail.len() + count).saturating_sub(4096);
            tail.drain(..discard);
            tail.extend_from_slice(&chunk[..count]);
        }
        let _ = stderr_tx.send(String::from_utf8_lossy(&tail).into_owned());
    });
    let result: Result<_, &'static str> = (|| {
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) if Instant::now() < deadline => {
                    std::thread::sleep(
                        Duration::from_millis(10)
                            .min(deadline.saturating_duration_since(Instant::now())),
                    );
                }
                _ => return Err("probe_failed"),
            }
        };
        // A descendant can retain the pipe even after the worker exits. Its EOF
        // must obey the same deadline as the worker, not an unbounded read.
        let stdout = rx
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .map_err(|_| "probe_failed")?
            .map_err(|_| "probe_failed")?;
        if stdout.len() as u64 > STDOUT_LIMIT {
            return Err("probe_failed");
        }
        Ok((status, stdout))
    })();
    if result.is_err() {
        // Only this child is ours; never signal a process group or descendants.
        let _ = child.kill();
        let _ = child.wait();
    }
    let stderr = stderr_rx
        .recv_timeout(deadline.saturating_duration_since(Instant::now()))
        .unwrap_or_default();
    result
        .map(|(status, stdout)| ProbeOutput {
            status,
            stdout,
            stderr: stderr.clone(),
        })
        .map_err(|reason| ProbeError {
            reason,
            detail: format!("worker probe failed or timed out; stderr: {stderr}"),
        })
}

#[cfg(any(not(target_os = "macos"), test))]
pub(crate) fn version_result(output: ProbeOutput, backend: &str) -> Result<(), &'static str> {
    if !output.status.success() {
        return Err("probe_failed");
    }
    let text = std::str::from_utf8(&output.stdout).map_err(|_| "probe_failed")?;
    let features = text.split_once("features=").ok_or("probe_failed")?.1;
    // CUDA follows features with manifest_digest; Vulkan's disabled value
    // contains spaces, so taking just the next whitespace token loses meaning.
    let features = features
        .split(" manifest_digest=")
        .next()
        .unwrap_or(features)
        .trim();
    if features == "none" || features == "none (vulkan disabled)" {
        Err("backend_not_built")
    } else if features
        .split(|c: char| c == ',' || c.is_whitespace())
        .any(|feature| feature == backend)
    {
        Ok(())
    } else {
        Err("probe_failed")
    }
}

pub(crate) fn floor_result(output: ProbeOutput) -> Result<Value, &'static str> {
    let value: Value = serde_json::from_slice(&output.stdout).map_err(|_| "probe_failed")?;
    if !value["required"].is_object()
        || !value
            .get("observed")
            .is_some_and(|observed| observed.is_null() || observed.is_object())
    {
        return Err("probe_failed");
    }
    match (
        output.status.success(),
        output.status.code(),
        value["status"].as_str(),
    ) {
        (true, _, Some("ok"))
            if matches!(value.get("code"), Some(Value::Null))
                || value.get("code").and_then(Value::as_str) == Some("ok") =>
        {
            if value["observed"].is_object() {
                Ok(value)
            } else {
                Err("probe_failed")
            }
        }
        (false, Some(2), Some("refused")) => Err(refusal_reason(
            value["code"].as_str().ok_or("probe_failed")?,
        )),
        _ => Err("probe_failed"),
    }
}

fn refusal_reason(code: &str) -> &'static str {
    match code.split(':').next().unwrap_or(code) {
        "vulkan_no_device" | "cuda_no_driver" => "driver_missing",
        "cuda_runtime_missing" => "runtime_missing",
        "vulkan_software_device" => "software_device",
        "vulkan_unsupported_vendor" => "device_unsupported",
        "vulkan_api_too_old"
        | "vulkan_missing_feature"
        | "vulkan_insufficient_memory"
        | "cuda_driver_too_old"
        | "cuda_compute_capability_too_low" => "device_below_floor",
        _ => "probe_failed",
    }
}
