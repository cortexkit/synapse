//! Parent-side counters cover load, warmup and inference for every worker lane.
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Sender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

pub use crate::gpu::GpuProbe;
use serde::Serialize;

#[derive(Debug, Default, Serialize)]
pub struct Memory {
    pub process_peak_bytes: Option<u64>,
    pub process_peak_method: Option<String>,
    pub gpu_peak_bytes: Option<u64>,
    pub gpu_after_load_idle_bytes: Option<u64>,
    pub gpu_after_load_idle_method: Option<String>,
    pub gpu_device_baseline_bytes: Option<u64>,
    pub gpu_peak_method: Option<String>,
    pub gpu_idle_baseline_bytes: Option<u64>,
    pub gpu_sample_interval_ms: Option<u64>,
    pub gpu_max_sample_gap_ms: Option<u64>,
    pub gpu_samples: u64,
    pub ane_peak_bytes: Option<u64>,
}

pub struct Monitor {
    stop: Sender<()>,
    thread: JoinHandle<Memory>,
}

impl Monitor {
    pub fn start(pid: u32, ready: PathBuf, mut gpu: GpuProbe) -> Self {
        let (stop, receiver) = mpsc::channel();
        let thread = thread::spawn(move || {
            let mut memory = Memory::default();
            let mut previous_gpu = None;
            let mut next_gpu = Instant::now();
            let mut loaded = false;
            loop {
                if !loaded && ready.exists() {
                    gpu.sample(pid, true, &mut memory);
                    // The worker remains idle until this snapshot is complete.
                    // Failure to resume is a fatal worker timeout, never a run
                    // that quietly pretends an idle snapshot was obtained.
                    let _ = std::fs::write(ready.with_extension("memory-resume"), []);
                    loaded = true;
                }
                if let Some(bytes) = footprint(pid) {
                    memory.process_peak_bytes =
                        Some(memory.process_peak_bytes.unwrap_or(0).max(bytes));
                    memory.process_peak_method = Some(
                        "proc_pid_rusage RUSAGE_INFO_V4 lifetime_max_phys_footprint (20ms polls)"
                            .into(),
                    );
                }
                if Instant::now() >= next_gpu {
                    let now = Instant::now();
                    if let Some(previous) = previous_gpu {
                        let gap = now.duration_since(previous).as_millis() as u64;
                        memory.gpu_max_sample_gap_ms =
                            Some(memory.gpu_max_sample_gap_ms.unwrap_or(0).max(gap));
                    }
                    previous_gpu = Some(now);
                    gpu.sample(pid, false, &mut memory);
                    memory.gpu_sample_interval_ms = Some(100);
                    next_gpu = now + Duration::from_millis(100);
                }
                match receiver.recv_timeout(Duration::from_millis(20)) {
                    Err(mpsc::RecvTimeoutError::Timeout) => (),
                    _ => return memory,
                }
            }
        });
        Self { stop, thread }
    }

    pub fn finish(self) -> Memory {
        let _ = self.stop.send(());
        self.thread.join().unwrap_or_default()
    }
}

/// Optional parent probe, after load and before any inference. Return the wait
/// duration so workers can exclude measurement synchronization from cold time.
pub fn loaded_barrier() -> anyhow::Result<Duration> {
    let Some(ready) = std::env::var_os("STT_MEMORY_READY") else {
        return Ok(Duration::ZERO);
    };
    loaded_barrier_at(Path::new(&ready))
}

fn loaded_barrier_at(ready: &Path) -> anyhow::Result<Duration> {
    let started = Instant::now();
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(ready)?;
    while !ready.with_extension("memory-resume").exists() {
        anyhow::ensure!(
            started.elapsed() < Duration::from_secs(30),
            "parent loaded-memory probe timed out"
        );
        thread::sleep(Duration::from_millis(2));
    }
    Ok(started.elapsed())
}

#[test]
fn stt_memory_loaded_barrier_waits_for_parent_resume() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../target/scratch")
        .join(format!("stt-loaded-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let ready = root.join("worker.memory-ready");
    let probe = ready.clone();
    let parent = thread::spawn(move || {
        while !probe.exists() {
            thread::sleep(Duration::from_millis(1));
        }
        std::fs::write(probe.with_extension("memory-resume"), []).unwrap();
    });
    assert!(loaded_barrier_at(&ready).unwrap() > Duration::ZERO);
    parent.join().unwrap();
    std::fs::remove_dir_all(root).unwrap();
}

#[cfg(target_os = "macos")]
fn footprint(pid: u32) -> Option<u64> {
    unsafe extern "C" {
        fn stt_phys_footprint_peak(pid: i32) -> u64;
    }
    let bytes = unsafe { stt_phys_footprint_peak(pid as i32) };
    (bytes > 0).then_some(bytes)
}

#[cfg(not(target_os = "macos"))]
fn footprint(_: u32) -> Option<u64> {
    None
}

#[cfg(target_os = "macos")]
#[test]
fn stt_memory_macos_lifetime_footprint_is_available_for_own_process() {
    assert!(footprint(std::process::id()).is_some_and(|bytes| bytes > 0));
}
