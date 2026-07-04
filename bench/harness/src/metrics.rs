//! Power/RSS measurement wrapper.
//!
//! Runs a child command while sampling macmon (sudo-free power telemetry on
//! Apple Silicon). Emits a measurement JSON: wall time, child exit status,
//! peak RSS of the child process tree (sampled via `ps`), and power series
//! aggregates (avg/peak CPU + GPU + ANE watts over the run window).
//!
//! Every bench lane runs under this wrapper so power numbers are collected
//! identically regardless of runtime.

use std::io::BufRead;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct Measurement {
    cmd: Vec<String>,
    wall_s: f64,
    exit_code: i32,
    peak_rss_bytes: u64,
    power: PowerAgg,
    samples: usize,
    /// Foreign-load check: CPU consumed by processes OTHER than the measured
    /// child tree, sampled during the run. If foreign load exceeded the
    /// threshold the run is marked contaminated and the wrapper exits nonzero
    /// so the matrix runner retries the lane.
    foreign_cpu_avg_pct: f64,
    foreign_cpu_peak_pct: f64,
    contaminated: bool,
}

#[derive(Serialize, Default)]
struct PowerAgg {
    cpu_avg_w: f64,
    cpu_peak_w: f64,
    gpu_avg_w: f64,
    gpu_peak_w: f64,
    ane_avg_w: f64,
    ane_peak_w: f64,
    combined_avg_w: f64,
    /// Energy over the run window: combined average watts * wall seconds.
    /// Joules/task = this divided by the lane's item count.
    energy_j: f64,
}

#[derive(Deserialize)]
struct MacmonSample {
    cpu_power: f64,
    gpu_power: f64,
    ane_power: f64,
    #[serde(default)]
    cpu_usage_pct: f64,
    #[serde(default)]
    gpu_usage: (u64, f64),
}

/// Idle gate: refuse to measure on a busy machine so results are not
/// contaminated by other processes. Samples macmon for a few seconds and
/// requires CPU and GPU utilization below thresholds.
fn assert_idle(max_cpu_pct: f64, max_gpu_pct: f64) -> Result<()> {
    let out = Command::new("macmon")
        .args(["pipe", "-s", "6", "-i", "500"])
        .stderr(Stdio::null())
        .output()
        .context("spawning macmon for idle preflight")?;
    let samples: Vec<MacmonSample> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    anyhow::ensure!(samples.len() >= 3, "idle preflight: too few macmon samples");
    // Drop the first sample (startup transient), average the rest.
    let rest = &samples[1..];
    let cpu = rest.iter().map(|s| s.cpu_usage_pct * 100.0).sum::<f64>() / rest.len() as f64;
    let gpu = rest.iter().map(|s| s.gpu_usage.1 * 100.0).sum::<f64>() / rest.len() as f64;
    anyhow::ensure!(
        cpu <= max_cpu_pct && gpu <= max_gpu_pct,
        "machine not idle: cpu {cpu:.1}% (max {max_cpu_pct}%), gpu {gpu:.1}% (max {max_gpu_pct}%) — close other workloads before measuring"
    );
    eprintln!("idle preflight ok: cpu {cpu:.1}%, gpu {gpu:.1}%");
    Ok(())
}

/// Total CPU percent of all processes except the given pid's tree and our own
/// sampling overhead. Uses `ps` so per-process attribution is possible (macmon
/// only reports machine totals, which cannot distinguish the measured child
/// from a foreign agent burst).
fn foreign_cpu_pct(child_pid: u32) -> Option<f64> {
    let out = Command::new("ps").args(["-axo", "pid=,ppid=,pcpu="]).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let mut rows: Vec<(u32, u32, f64)> = Vec::new();
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let (Some(pid), Some(ppid), Some(pcpu)) = (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        let (Ok(pid), Ok(ppid), Ok(pcpu)) =
            (pid.parse::<u32>(), ppid.parse::<u32>(), pcpu.parse::<f64>())
        else {
            continue;
        };
        rows.push((pid, ppid, pcpu));
    }
    // Build the child's process tree (children of children included).
    let mut tree: std::collections::HashSet<u32> = std::collections::HashSet::new();
    tree.insert(child_pid);
    tree.insert(std::process::id());
    loop {
        let before = tree.len();
        for (pid, ppid, _) in &rows {
            if tree.contains(ppid) {
                tree.insert(*pid);
            }
        }
        if tree.len() == before {
            break;
        }
    }
    // ps pcpu is per-core (100 = one full core); normalize to machine percent
    // so the value is comparable with the preflight gate's macmon threshold.
    let sum: f64 = rows.iter().filter(|(pid, _, _)| !tree.contains(pid)).map(|(_, _, p)| p).sum();
    let ncpu = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) as f64;
    Some(sum / ncpu)
}

pub fn run_wrapped(
    out: &Path,
    interval_ms: u64,
    cmd: &[String],
    skip_idle_check: bool,
    max_cpu_pct: f64,
    max_gpu_pct: f64,
) -> Result<()> {
    anyhow::ensure!(!cmd.is_empty(), "empty command");

    if !skip_idle_check {
        assert_idle(max_cpu_pct, max_gpu_pct)?;
    }

    // Start macmon sampler.
    let mut macmon = Command::new("macmon")
        .args(["pipe", "-i", &interval_ms.to_string()])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("spawning macmon (brew install macmon)")?;
    let macmon_out = macmon.stdout.take().expect("piped stdout");

    let stop = Arc::new(AtomicBool::new(false));
    let sampler_stop = stop.clone();
    let sampler = std::thread::spawn(move || {
        let reader = std::io::BufReader::new(macmon_out);
        let mut samples: Vec<MacmonSample> = Vec::new();
        for line in reader.lines() {
            if sampler_stop.load(Ordering::Relaxed) {
                break;
            }
            let Ok(line) = line else { break };
            if let Ok(s) = serde_json::from_str::<MacmonSample>(&line) {
                samples.push(s);
            }
        }
        samples
    });

    // Run the child.
    let started = Instant::now();
    let mut child = Command::new(&cmd[0])
        .args(&cmd[1..])
        .spawn()
        .with_context(|| format!("spawning {}", cmd[0]))?;
    let child_pid = child.id();

    // Sample child RSS and foreign CPU while it runs. Foreign load is CPU
    // consumed outside the child's process tree; sampled every ~2s (ps is
    // costlier than the RSS probe).
    let mut peak_rss: u64 = 0;
    let mut foreign_samples: Vec<f64> = Vec::new();
    let mut ticks: u32 = 0;
    let exit_code = loop {
        if let Some(status) = child.try_wait()? {
            break status.code().unwrap_or(-1);
        }
        if let Some(rss) = sample_rss(child_pid) {
            peak_rss = peak_rss.max(rss);
        }
        if ticks % 20 == 0 {
            if let Some(pct) = foreign_cpu_pct(child_pid) {
                foreign_samples.push(pct);
            }
        }
        ticks += 1;
        std::thread::sleep(Duration::from_millis(100));
    };
    let wall_s = started.elapsed().as_secs_f64();

    // Stop sampler.
    stop.store(true, Ordering::Relaxed);
    let _ = macmon.kill();
    let _ = macmon.wait();
    let samples = sampler.join().unwrap_or_default();

    let n = samples.len().max(1) as f64;
    let mut agg = PowerAgg::default();
    for s in &samples {
        agg.cpu_avg_w += s.cpu_power;
        agg.gpu_avg_w += s.gpu_power;
        agg.ane_avg_w += s.ane_power;
        agg.cpu_peak_w = agg.cpu_peak_w.max(s.cpu_power);
        agg.gpu_peak_w = agg.gpu_peak_w.max(s.gpu_power);
        agg.ane_peak_w = agg.ane_peak_w.max(s.ane_power);
    }
    agg.cpu_avg_w /= n;
    agg.gpu_avg_w /= n;
    agg.ane_avg_w /= n;
    agg.combined_avg_w = agg.cpu_avg_w + agg.gpu_avg_w + agg.ane_avg_w;
    agg.energy_j = agg.combined_avg_w * wall_s;

    // Contamination verdict: sustained foreign CPU during the run means the
    // numbers are not attributable to the measured child. Threshold mirrors
    // the preflight gate. WindowServer/kernel background sits ~5-10%, so the
    // preflight max_cpu_pct (default 15) is a sane bar here too.
    let n_foreign = foreign_samples.len().max(1) as f64;
    let foreign_avg = foreign_samples.iter().sum::<f64>() / n_foreign;
    let foreign_peak = foreign_samples.iter().cloned().fold(0.0f64, f64::max);
    let contaminated = !skip_idle_check && foreign_avg > max_cpu_pct;

    let m = Measurement {
        cmd: cmd.to_vec(),
        wall_s,
        exit_code,
        peak_rss_bytes: peak_rss,
        power: agg,
        samples: samples.len(),
        foreign_cpu_avg_pct: foreign_avg,
        foreign_cpu_peak_pct: foreign_peak,
        contaminated,
    };
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(out, serde_json::to_string_pretty(&m)?)?;
    eprintln!(
        "measured: wall={:.1}s exit={} peak_rss={:.0}MB cpu_avg={:.1}W gpu_avg={:.1}W ane_avg={:.1}W ({} samples)",
        m.wall_s,
        m.exit_code,
        m.peak_rss_bytes as f64 / 1e6,
        m.power.cpu_avg_w,
        m.power.gpu_avg_w,
        m.power.ane_avg_w,
        m.samples
    );
    anyhow::ensure!(exit_code == 0, "child exited nonzero: {exit_code}");
    anyhow::ensure!(
        !contaminated,
        "run contaminated: foreign CPU avg {foreign_avg:.1}% (max {max_cpu_pct}%) during measurement — discarding; the matrix runner will retry"
    );
    Ok(())
}

fn sample_rss(pid: u32) -> Option<u64> {
    let out = Command::new("ps").args(["-o", "rss=", "-p", &pid.to_string()]).output().ok()?;
    let kb: u64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
    Some(kb * 1024)
}
