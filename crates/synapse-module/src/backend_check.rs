//! Backend detection shared by the standalone check and the serving catalog.

use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    ffi::OsString,
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context};
use serde::Serialize;
use synapse_core::{worker_binary_file_name, Platform};

#[cfg(any(not(target_os = "macos"), test))]
use super::{catalog_probe, cuda_floor_override, cuda_reading, owned_cuda_floor_decision};
use super::{resolve_catalog_worker, ModuleConfig};

const HELP: &str = "Usage: ck-synapse --check-backend [--json]

Check for usable Synapse backends without starting the module or loading models.
Print usable backend names, or refusal reasons if none is usable.
Use --json to report all backend rows as JSON.

Exit codes:
  0  At least one backend is runnable.
  1  No backend is runnable and a probe failed; support is uncertain.
  2  No backend is runnable; all refusals are definite.
  3  The check could not run.";

#[derive(Serialize)]
struct BackendStatus {
    backend: &'static str,
    runnable: bool,
    reason: Option<&'static str>,
}

/// Run the standalone support check for the `ck-synapse` command line.
pub fn check_backend(arguments: &[OsString], output: &mut impl Write) -> anyhow::Result<i32> {
    let mut json = false;
    let mut help = false;
    for argument in arguments {
        if argument == "--json" {
            json = true;
        } else if argument == "--help" || argument == "-h" {
            help = true;
        } else if argument != "--check-backend" {
            bail!("unexpected argument {argument:?}\n{HELP}");
        }
    }
    if help {
        writeln!(output, "{HELP}")?;
        output.flush()?;
        return Ok(0);
    }

    let exe = env::current_exe().context("locating ck-synapse")?;
    let directory = exe.parent().context("ck-synapse has no parent directory")?;
    let (runnable, reasons) = detect_backends(|engine| sibling_worker(directory, engine));
    let backends = ["metal", "ane", "cuda", "vulkan"].map(|backend| {
        let reason = backend_reason(&runnable, &reasons, backend, Platform::current());
        BackendStatus {
            backend,
            runnable: reason.is_none(),
            reason,
        }
    });
    let usable = backends.iter().any(|row| row.runnable);
    let exit_code = if usable {
        0
    } else if backends
        .iter()
        .any(|row| row.reason == Some("probe_failed"))
    {
        1
    } else {
        2
    };

    if json {
        writeln!(
            output,
            "{}",
            serde_json::json!({"usable": usable, "backends": backends})
        )?;
    } else if usable {
        for row in backends.iter().filter(|row| row.runnable) {
            writeln!(output, "{}", row.backend)?;
        }
    } else {
        for row in backends {
            if let Some(reason) = row.reason {
                writeln!(output, "backend_unavailable: {}: {reason}", row.backend)?;
            }
        }
    }
    output.flush()?;
    Ok(exit_code)
}

type BackendDetection = (BTreeSet<String>, BTreeMap<String, &'static str>);

pub(super) fn detected_catalog_backends(config: &ModuleConfig) -> BackendDetection {
    #[cfg(feature = "test-support")]
    if let Ok(value) = env::var("SYNAPSE_TEST_RUNNABLE_BACKENDS") {
        return (
            value
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect(),
            BTreeMap::new(),
        );
    }
    detect_backends(|engine| {
        #[cfg(target_os = "macos")]
        let configured = {
            let _ = config;
            None
        };
        #[cfg(not(target_os = "macos"))]
        let configured = config
            .preload_models
            .iter()
            .find(|model| model.engine == engine && model.worker_bin.is_some())
            .and_then(|model| model.worker_bin.as_deref());
        resolve_catalog_worker(engine, configured)
    })
}

fn detect_backends(worker: impl Fn(&str) -> Option<PathBuf>) -> BackendDetection {
    #[cfg(target_os = "macos")]
    {
        let mut backends = BTreeSet::new();
        if metal::Device::system_default().is_some() {
            backends.insert("metal".into());
        }
        if direct_ane_catalog_available(true, worker("ane-direct-worker").as_deref()) {
            backends.insert("ane".into());
        }
        (backends, BTreeMap::new())
    }
    #[cfg(not(target_os = "macos"))]
    {
        detect_gpu_backends(
            worker(synapse_core::CUDA_WORKER_ENGINE),
            worker("owned-vulkan"),
        )
    }
}

/// Resolve an unconfigured worker beside the module executable. Release
/// archives unpack the module and its workers into the same directory.
pub(super) fn resolve_worker_binary_sibling(engine: &str) -> Option<PathBuf> {
    let exe = env::current_exe().ok()?;
    sibling_worker(exe.parent()?, engine)
}

fn sibling_worker(directory: &Path, engine: &str) -> Option<PathBuf> {
    let candidate = directory.join(worker_binary_file_name(engine)?);
    #[cfg(windows)]
    let candidate = candidate.with_extension("exe");
    candidate.is_file().then_some(candidate)
}

#[cfg(any(not(target_os = "macos"), test))]
pub(super) fn detect_gpu_backends(
    cuda: Option<PathBuf>,
    vulkan: Option<PathBuf>,
) -> BackendDetection {
    let outcomes = std::thread::scope(|scope| {
        let cuda = scope.spawn(|| detect_worker_backend("cuda", cuda.as_deref()));
        let vulkan = scope.spawn(|| detect_worker_backend("vulkan", vulkan.as_deref()));
        [
            ("cuda", cuda.join().expect("CUDA detection thread")),
            ("vulkan", vulkan.join().expect("Vulkan detection thread")),
        ]
    });
    let mut runnable = BTreeSet::new();
    let mut reasons = BTreeMap::new();
    for (backend, outcome) in outcomes {
        match outcome {
            Ok(()) => {
                runnable.insert(backend.to_string());
            }
            Err(reason) => {
                reasons.insert(backend.to_string(), reason);
            }
        }
    }
    (runnable, reasons)
}

#[cfg(any(not(target_os = "macos"), test))]
pub(super) fn detect_worker_backend(
    backend: &str,
    worker: Option<&Path>,
) -> Result<(), &'static str> {
    use catalog_probe::{cached_probe, floor_result, version_result, ProbeKind};
    let worker = worker.ok_or("worker_missing")?;
    version_result(
        cached_probe(worker, ProbeKind::Version).map_err(|error| error.reason)?,
        backend,
    )?;
    if backend != "cuda" || cuda_floor_override().is_none() {
        let envelope =
            floor_result(cached_probe(worker, ProbeKind::Floor).map_err(|error| error.reason)?)?;
        if backend == "cuda" && cuda_reading(&envelope).is_none() {
            return Err("probe_failed");
        }
    }
    if backend == "cuda" && !owned_cuda_floor_decision(Some(worker)).is_supported() {
        return Err("device_below_floor");
    }
    Ok(())
}

// The release catalog uses the direct ANE worker, not the worker that accesses
// the Neural Engine through Core ML. Its availability check requires macOS and
// the direct worker binary.
#[cfg(any(target_os = "macos", test))]
pub(super) fn direct_ane_catalog_available(macos: bool, worker: Option<&Path>) -> bool {
    macos && worker.is_some_and(Path::is_file)
}

pub(super) fn backend_reason(
    runnable: &BTreeSet<String>,
    reasons: &BTreeMap<String, &'static str>,
    backend: &str,
    platform: Platform,
) -> Option<&'static str> {
    let macos = platform == Platform::MacOs;
    if macos && matches!(backend, "cuda" | "vulkan") {
        Some("not_supported_on_platform")
    } else if runnable.contains(backend) {
        None
    } else if !macos && matches!(backend, "metal" | "ane") {
        Some("not_supported_on_platform")
    } else if let Some(reason) = reasons.get(backend) {
        Some(*reason)
    } else if backend == "metal" {
        Some("device_missing")
    } else {
        Some("worker_missing")
    }
}
