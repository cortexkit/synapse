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
}

#[derive(Deserialize)]
struct MacmonSample {
    cpu_power: f64,
    gpu_power: f64,
    ane_power: f64,
}

pub fn run_wrapped(out: &Path, interval_ms: u64, cmd: &[String]) -> Result<()> {
    anyhow::ensure!(!cmd.is_empty(), "empty command");

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

    // Sample child RSS while it runs (process tree via pgrep -P not needed:
    // lanes are single-process; runtimes that spawn children report their own).
    let mut peak_rss: u64 = 0;
    let exit_code = loop {
        if let Some(status) = child.try_wait()? {
            break status.code().unwrap_or(-1);
        }
        if let Some(rss) = sample_rss(child_pid) {
            peak_rss = peak_rss.max(rss);
        }
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

    let m = Measurement {
        cmd: cmd.to_vec(),
        wall_s,
        exit_code,
        peak_rss_bytes: peak_rss,
        power: agg,
        samples: samples.len(),
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
    Ok(())
}

fn sample_rss(pid: u32) -> Option<u64> {
    let out = Command::new("ps").args(["-o", "rss=", "-p", &pid.to_string()]).output().ok()?;
    let kb: u64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
    Some(kb * 1024)
}
