//! Does a resident shape's per-row time depend on what was compiled after it?
//!
//! One process, the Qwen3 embedding profile, no module. The experiment times
//! a batch of identical-length rows on the 128-token shape:
//!
//! - `a`: right after compiling 128 alone;
//! - `d`: again with no compile in between (control for `a`);
//! - `b`: after then compiling 512 and 1024, with 128 kept resident;
//! - `c`: after evicting 128 and compiling it again, 512 and 1024 still
//!   resident;
//! - `c_control`: again with no compile in between (control for `c`).
//!
//! Each phase first runs one untimed row, so request creation on first use is
//! not counted, then times every row of the batch individually. It also
//! hashes every output vector, so a change in results between phases is
//! caught along with a change in speed. Run it several times, each in a fresh
//! process, and compare the medians.
//!
//! Usage (macOS with the private Neural Engine API; the converted package is a
//! `qwen3-embedding-0.6b.safetensors` file, or a directory holding
//! `model.safetensors`, whose digest matches the pinned profile):
//!
//! ```sh
//! cargo run --release -p synapse-worker-ane-direct --example shape_recency -- \
//!     <converted-package> [tokens-per-row, default 100] [rows, default 64]
//! ```
//!
//! It prints one JSON object. Leave `SYNAPSE_ANE_PROFILE_DIR` unset, so the
//! worker's profiling evidence I/O is not part of the timed rows.

#[cfg(target_os = "macos")]
#[allow(dead_code)]
#[path = "../src/backend.rs"]
mod backend;
#[cfg(target_os = "macos")]
#[allow(dead_code)]
#[path = "../src/modernbert.rs"]
mod modernbert;
#[cfg(target_os = "macos")]
#[allow(dead_code)]
#[path = "../src/profile.rs"]
mod profile;
#[cfg(target_os = "macos")]
#[allow(dead_code)]
#[path = "../src/qwen.rs"]
mod qwen;
#[cfg(target_os = "macos")]
#[allow(dead_code)]
#[path = "../src/worker.rs"]
mod worker;

#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    experiment::run()
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("shape_recency needs macOS and the private Neural Engine API");
    std::process::exit(2);
}

#[cfg(target_os = "macos")]
mod experiment {
    use crate::backend::{Model, Profile};
    use anyhow::{ensure, Context, Result};
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::path::PathBuf;
    use std::time::Instant;

    const PROFILE: &str = "qwen3-embedding-0.6b.ane-direct-worker";
    const OS: &str = "shape-recency-experiment";

    fn ms(started: Instant) -> f64 {
        started.elapsed().as_secs_f64() * 1000.0
    }

    fn quantile(sorted: &[f64], q: f64) -> f64 {
        let index = ((sorted.len() - 1) as f64 * q).round() as usize;
        sorted[index]
    }

    fn vector_hash(vector: &[f32]) -> String {
        let mut hasher = Sha256::new();
        for value in vector {
            hasher.update(value.to_bits().to_le_bytes());
        }
        format!("{:x}", hasher.finalize())
    }

    /// One phase: an untimed first row, then `rows` timed rows of `tokens`.
    fn measure(model: &Model, phase: &str, tokens: &[u32], rows: usize) -> Result<Value> {
        let started = Instant::now();
        let first = model.run(tokens)?;
        let first_row_ms = ms(started);
        let mut times = Vec::with_capacity(rows);
        let mut hashes = std::collections::BTreeSet::new();
        hashes.insert(vector_hash(&first));
        for _ in 0..rows {
            let started = Instant::now();
            let vector = model.run(tokens)?;
            times.push(ms(started));
            hashes.insert(vector_hash(&vector));
        }
        let mut sorted = times.clone();
        sorted.sort_by(f64::total_cmp);
        Ok(json!({
            "phase": phase,
            "first_row_ms": first_row_ms,
            "median_ms": quantile(&sorted, 0.5),
            "p10_ms": quantile(&sorted, 0.1),
            "p90_ms": quantile(&sorted, 0.9),
            "min_ms": sorted[0],
            "max_ms": sorted[sorted.len() - 1],
            "total_ms": times.iter().sum::<f64>(),
            "rows_ms": times,
            "vector_sha256": hashes.iter().next().cloned(),
            "distinct_vectors": hashes.len(),
            "resident": model.resident.keys().collect::<Vec<_>>(),
        }))
    }

    fn compile(model: &mut Model, shape: usize, log: &mut Vec<Value>) -> Result<()> {
        let started = Instant::now();
        model.admit(shape, OS)?;
        log.push(json!({"event": "admit", "shape": shape, "ms": ms(started)}));
        Ok(())
    }

    pub fn run() -> Result<()> {
        let mut args = std::env::args().skip(1);
        let package = PathBuf::from(
            args.next()
                .context("usage: shape_recency <package> [tokens] [rows]")?,
        );
        let length: usize = args.next().map(|v| v.parse()).transpose()?.unwrap_or(100);
        let rows: usize = args.next().map(|v| v.parse()).transpose()?.unwrap_or(64);
        ensure!(
            (1..=128).contains(&length),
            "tokens per row must fit the 128 shape"
        );
        ensure!(rows > 0, "rows must be positive");
        ensure!(
            std::env::var_os("SYNAPSE_ANE_PROFILE_DIR").is_none(),
            "unset SYNAPSE_ANE_PROFILE_DIR so evidence I/O is not timed"
        );
        crate::worker::private_api()?;
        let profile = Profile::select(PROFILE, "embed")?;
        let digest = profile.numeric["converted_package_digest"]
            .as_str()
            .context("profile has no converted package digest")?
            .to_owned();
        let mut model = Model::load(profile, &package, &digest)?;
        // Identical rows of ordinary vocabulary ids ending in the profile's
        // terminal token, the same shape of input the module sends a worker
        // after tokenizing (the embedding is read at the terminal token).
        let terminal = model.profile.model["grammar"]["terminal_tokens"][0]["id"]
            .as_u64()
            .context("profile has no terminal token")? as u32;
        let mut tokens: Vec<u32> = (0..length as u32 - 1).map(|i| 1_000 + i).collect();
        tokens.push(terminal);

        let mut events = Vec::new();
        let mut phases = Vec::new();
        compile(&mut model, 128, &mut events)?;
        phases.push(measure(&model, "a", &tokens, rows)?);
        phases.push(measure(&model, "d", &tokens, rows)?);
        compile(&mut model, 512, &mut events)?;
        compile(&mut model, 1024, &mut events)?;
        phases.push(measure(&model, "b", &tokens, rows)?);
        let started = Instant::now();
        model.evict(128);
        events.push(json!({"event": "evict", "shape": 128, "ms": ms(started)}));
        compile(&mut model, 128, &mut events)?;
        phases.push(measure(&model, "c", &tokens, rows)?);
        phases.push(measure(&model, "c_control", &tokens, rows)?);

        let hashes: std::collections::BTreeSet<_> = phases
            .iter()
            .map(|phase| phase["vector_sha256"].clone().to_string())
            .collect();
        let bit_identical = hashes.len() == 1
            && phases
                .iter()
                .all(|phase| phase["distinct_vectors"] == json!(1));
        println!(
            "{}",
            json!({
                "experiment": "shape_recency",
                "pid": std::process::id(),
                "unix_us": std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as u64,
                "tokens_per_row": length,
                "rows": rows,
                "events": events,
                "phases": phases,
                "bit_identical_across_phases": bit_identical,
            })
        );
        Ok(())
    }
}
