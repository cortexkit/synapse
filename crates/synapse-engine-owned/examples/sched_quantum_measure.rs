//! OQ-DEC-SCHED-01 mixed-load decode quantum measurement.
//!
//! Drives the production-owned Metal engines on real hardware to select the
//! production decode quantum N from the candidate set {8, 16, 32}:
//!
//! - Decode arm: Qwen3-0.6B Q8 step decode through the production
//!   `owned-decode-engine` (f16 prefill + Q8 stepping, the parity-certified
//!   Q8 recipe), generating continuously in N-token quanta with yield points
//!   between quanta.
//! - Embed arm: gte-modernbert-base f16 single-query embeds through the
//!   production `OwnedMetalEmbedEngine`, arriving at a steady interactive
//!   rate (one query every 200 ms, i.e. 5/s), measuring per-query latency
//!   from arrival to completion (queue wait included).
//!
//! The arms interleave through a permit model mirroring the scheduler
//! contract: embed queries queue while a decode quantum runs; at each quantum
//! boundary the decode permit is released while embed work is queued
//! (yield-on-contention); a decode operation whose aging anchor is older
//! than the 250 ms DECODE aging window preempts the yield loop, mirroring
//! aged-DECODE precedence in boundary arbitration.
//!
//! Protocol per candidate N in {8, 16, 32}: warmup, then a measured window of
//! at least 60 s and at least 300 completed embed queries (whichever is
//! longer), recording embed p50/p95/p99 (nearest-rank), decode effective
//! tok/s, quantum boundary counts, permit events, queue depth, per-query
//! waiting, and cancellation-latency observations. Cells run serially. A cell
//! is refused while the 1-minute loadavg exceeds 8 at its start (the harness
//! polls until the machine is quiet), and any cell that starts above loadavg
//! 4 is flagged in the report.
//!
//! Generation admission is quantum-bounded (protocol v2): the f16 causal
//! prefill no longer runs as one uninterruptible ~1 s span. It executes in
//! 16-token batched chunks (one bounded command buffer per chunk, mat-mat
//! kernels with weights streamed once per layer — bit-identical to the
//! per-token path, gated below) with yield-on-contention span boundaries
//! between chunks and around the KV handoff spans, exactly like decode
//! quanta. The longest uninterruptible GPU span of the workload is therefore
//! one chunk / handoff span / decode quantum, reported as `max_span_ms`.
//!
//! The run also collects the same-session embed-only baseline before decode
//! admission, the uninterrupted decode throughput baseline, and two
//! bit-exactness spot checks: chunked N=8/16/32 greedy streams must be
//! byte-identical to the uninterrupted stream, and chunked prefill (8/16/32
//! per-token spans and the 16-token batched spans the cells actually run)
//! must produce byte-identical KV state and the same first-token argmax as
//! the uninterrupted single-command-buffer prefill.
//!
//! Selection rule: the LARGEST candidate whose concurrent embed.query p95
//! stays inside the committed SLO (150 ms). If all candidates meet the SLO,
//! N=32 is selected; if none do, the harness records the facts and reports
//! `selected_n: null` for review.
//!
//! Env vars:
//! - `SYNAPSE_OWNED_DECODE_QWEN3_0_6B`: path to the Qwen3-Embedding-0.6B
//!   snapshot (required).
//! - `SYNAPSE_SCHED_MEASURE_GTE_MODERNBERT`: path to the gte-modernbert-base
//!   snapshot (default: auto-detected under `~/.cache/huggingface/hub`).
//! - `SYNAPSE_SCHED_MEASURE_MODE`: `full` (default) or `quick` (short smoke
//!   protocol for validating the harness; never commit quick numbers).
//! - `SYNAPSE_SCHED_MEASURE_MAX_LOAD`: optional stricter 1-minute loadavg
//!   gate for cell starts (default and protocol cap: 8.0; cells above
//!   loadavg 4 are flagged either way).
//! - `SYNAPSE_SCHED_MEASURE_OUT`: optional path that additionally receives the
//!   JSON report (always printed to stdout).
//!
//! Run with:
//! ```text
//! DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
//! SYNAPSE_OWNED_DECODE_QWEN3_0_6B=<qwen3-snapshot> \
//! cargo run -p synapse-engine-owned --release --example sched_quantum_measure
//! ```

#[cfg(target_os = "macos")]
mod imp {
    use std::collections::VecDeque;
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    use serde::Serialize;
    use sha2::{Digest, Sha256};
    use synapse_core::{EmbedEngine, RuntimeConfig, ValidatedArtifact};
    use synapse_engine_owned::owned_decode_engine::{
        DecodeKernel, MetalStepDecoder, MetalStepKvCache, Qwen3DecodeModel, WeightQuantization,
    };
    use synapse_engine_owned::{ModelFamily, OwnedDType, OwnedMetalEmbedEngine, Precision};
    use tokenizers::{Tokenizer, TruncationParams};

    /// Candidate production N values. The measurement commits the largest
    /// candidate meeting the SLO; none is committed when no candidate meets it.
    const CANDIDATES: [u32; 3] = [8, 16, 32];
    /// Committed embed.query p95 SLO in milliseconds (production envelope).
    const SLO_P95_MS: f64 = 150.0;
    /// DECODE aging window: aged decode preempts the embed yield loop.
    const AGING_WINDOW: Duration = Duration::from_millis(250);
    /// Workload record values from `decode-sched-manifest-v1`.
    const PROMPT_LEN: usize = 128;
    const OUTPUT_LEN: u32 = 64;
    /// Interactive embed arrival period (steady 5/s open loop).
    const ARRIVAL_PERIOD: Duration = Duration::from_millis(200);
    /// Prefill chunk size in tokens for the batched prefill path. Each chunk
    /// is ONE bounded command buffer (mat-mat, weights streamed once per
    /// layer, ~40-50 ms on the M5), so prefill is quantum-bounded: the
    /// scheduler can yield between chunks. 16 is the batched-verify sweet
    /// spot for f16 (campaign BATCHED-VERIFY.md: f16 improves monotonically
    /// through K=16) and the largest column count the kernels template.
    const PREFILL_CHUNK_TOKENS: usize = 16;
    /// Cancellation-latency bound from the workload record.
    const CANCELLATION_BOUND_MS: f64 = 400.0;
    /// Refuse to start a cell when the 1-minute loadavg exceeds this.
    const LOADAVG_REFUSE: f64 = 8.0;
    /// Flag cells that start above this 1-minute loadavg.
    const LOADAVG_NOTE: f64 = 4.0;
    /// Protocol identity recorded in the evidence. v2 adds quantum-bounded
    /// chunked prefill (16-token batched spans with yield points) on top of
    /// the v1 mixed-load protocol; the workload, SLO, and selection rule are
    /// unchanged.
    const PROTOCOL_ID: &str = "oq-dec-sched-01-mixed-load-v2";

    /// Completion-style decode prompts (no fixture dependency; token IDs are
    /// cycled to exactly `PROMPT_LEN` tokens for a stable workload shape).
    const DECODE_PROMPTS: &[&str] = &[
        "The capital of France is",
        "Rust ownership prevents data races because",
        "The scheduler releases the decode permit when",
        "Metal command buffers complete in order, so",
        "An embedding query waits at most one quantum because",
    ];

    /// Realistic short interactive embed queries, cycled across arrivals.
    const EMBED_QUERIES: &[&str] = &[
        "how does the scheduler arbitrate decode quanta",
        "what is the capital of France",
        "explain yield-on-contention in one sentence",
        "rust borrow checker rules summary",
        "metal command buffer synchronization semantics",
        "how are embedding percentiles computed",
        "kv cache reservation for decode contexts",
        "difference between p95 and p99 latency",
    ];

    #[derive(Clone, Copy)]
    struct Protocol {
        /// Measured window minimum duration.
        duration: Duration,
        /// Minimum completed embed queries in the measured window.
        min_queries: usize,
        /// Warmup embed queries (discarded).
        warmup_queries: usize,
        /// Warmup decode generations (discarded).
        warmup_generations: usize,
        /// Uninterrupted decode throughput window.
        decode_only: Duration,
        /// Cancellation probes per mixed cell.
        cancel_probes: usize,
    }

    const FULL: Protocol = Protocol {
        duration: Duration::from_secs(60),
        min_queries: 300,
        warmup_queries: 20,
        warmup_generations: 2,
        decode_only: Duration::from_secs(15),
        cancel_probes: 10,
    };

    const QUICK: Protocol = Protocol {
        duration: Duration::from_secs(5),
        min_queries: 25,
        warmup_queries: 3,
        warmup_generations: 1,
        decode_only: Duration::from_secs(3),
        cancel_probes: 2,
    };

    #[derive(Serialize)]
    struct Percentiles {
        p50_ms: f64,
        p95_ms: f64,
        p99_ms: f64,
    }

    #[derive(Serialize)]
    struct CellReport {
        cell: String,
        n: Option<u32>,
        loadavg_before: [f64; 3],
        loadavg_after: [f64; 3],
        ran_above_load4: bool,
        window_ms: u64,
        embed_queries: usize,
        embed_latency: Option<Percentiles>,
        decode_tokens: u64,
        decode_tokens_per_sec: f64,
        generations: u64,
        quantum_boundaries: u64,
        continuations: u64,
        permit_acquired: u64,
        permit_retained: u64,
        permit_released: u64,
        max_quantum_ms: f64,
        /// Longest single uninterruptible GPU span observed in the cell: one
        /// prefill chunk, one KV handoff span, or one decode quantum. This is
        /// the workload's maximum uninterruptible GPU time (protocol v2
        /// quantum-bounds the prefill, so no generation-admission span fuses
        /// them anymore).
        max_span_ms: f64,
        /// Average generation restart cost (prefill + KV handoff), ms.
        restart_cost_ms: f64,
        cancellation_latency_ms: Vec<f64>,
        /// Embed queue depth sampled at each arbitration point.
        queue_depth_samples: Vec<u32>,
        /// Per-query waiting time (arrival to dispatch) in ms.
        per_operation_waiting_ms: Vec<f64>,
    }

    #[derive(Serialize)]
    struct ParitySpotCheck {
        prompt_tokens: usize,
        output_tokens: u32,
        uninterrupted_sha256: String,
        chunked_sha256: Vec<(u32, String)>,
        byte_identical: bool,
    }

    /// One chunked-prefill parity comparison against the uninterrupted
    /// single-command-buffer prefill reference.
    #[derive(Serialize)]
    struct PrefillParityEntry {
        label: String,
        chunk_tokens: usize,
        /// `per-token` (verify path, one command buffer per span) or
        /// `batched` (mat-mat path, the span shape the cells run).
        path: String,
        kv_sha256: String,
        first_token: u32,
        matches_reference: bool,
    }

    /// Chunked-prefill bit-exactness spot check: every chunking of the
    /// workload prompt must leave the KV cache byte-identical and produce the
    /// same first-token argmax as the uninterrupted prefill.
    #[derive(Serialize)]
    struct PrefillParitySpotCheck {
        prompt_tokens: usize,
        reference_kv_sha256: String,
        reference_first_token: u32,
        entries: Vec<PrefillParityEntry>,
        byte_identical: bool,
    }

    #[derive(Serialize)]
    struct Report {
        protocol_id: String,
        measured_at_utc: String,
        mode: &'static str,
        machine_profile_note: String,
        slo_embed_query_p95_ms: f64,
        baseline_embed_only: CellReport,
        decode_only_throughput: CellReport,
        candidates: Vec<CellReport>,
        /// Same-session embed-only cell after decode admission: the regression
        /// calculation compares its p95 against the pre-admission baseline.
        post_decode_embed_only: CellReport,
        embed_regression_pct: f64,
        parity_spot_check: ParitySpotCheck,
        prefill_parity_spot_check: PrefillParitySpotCheck,
        selected_n: Option<u32>,
        selection_note: String,
    }

    pub fn main() {
        if let Err(error) = run() {
            eprintln!("[sched-measure] FAILED: {error:#}");
            std::process::exit(1);
        }
    }

    fn run() -> anyhow::Result<()> {
        let mode = std::env::var("SYNAPSE_SCHED_MEASURE_MODE").unwrap_or_else(|_| "full".into());
        let mode_str: &'static str = if mode == "quick" { "quick" } else { "full" };
        let protocol = match mode.as_str() {
            "quick" => QUICK,
            "full" => FULL,
            other => anyhow::bail!("SYNAPSE_SCHED_MEASURE_MODE must be full or quick, got {other}"),
        };

        let qwen3_path = PathBuf::from(
            std::env::var_os("SYNAPSE_OWNED_DECODE_QWEN3_0_6B").ok_or_else(|| {
                anyhow::anyhow!(
                    "set SYNAPSE_OWNED_DECODE_QWEN3_0_6B to the Qwen3-Embedding-0.6B snapshot"
                )
            })?,
        );
        let gte_path = gte_snapshot()?;
        println!(
            "[sched-measure] mode={} qwen3={} gte={}",
            mode,
            qwen3_path.display(),
            gte_path.display()
        );

        let machine_profile_note = machine_profile_note();
        let measured_at_utc = utc_now();
        println!("[sched-measure] machine: {machine_profile_note}");
        println!("[sched-measure] measured_at: {measured_at_utc}");

        // -- Load engines -------------------------------------------------
        let mut decode = DecodeArm::new(&qwen3_path)?;
        let embed = EmbedArm::new(&gte_path)?;

        // Warmup both arms once (model caches, pipeline compilation, GPU
        // power state). Per-cell warmup still runs inside each cell.
        decode.warmup_generation(16)?;
        embed.warmup()?;

        // -- Parity spot check: chunked vs uninterrupted bit-exactness ---
        let parity = decode.parity_spot_check()?;
        println!(
            "[sched-measure] parity spot check byte_identical={}",
            parity.byte_identical
        );
        if !parity.byte_identical {
            anyhow::bail!("boundary-crossing parity spot check failed: chunked stream diverged from the uninterrupted stream");
        }

        // -- Prefill parity: chunked prefill vs uninterrupted reference --
        let prefill_parity = decode.prefill_parity_spot_check()?;
        println!(
            "[sched-measure] prefill parity spot check byte_identical={} ({} entries)",
            prefill_parity.byte_identical,
            prefill_parity.entries.len()
        );
        if !prefill_parity.byte_identical {
            for entry in prefill_parity
                .entries
                .iter()
                .filter(|e| !e.matches_reference)
            {
                println!(
                    "[sched-measure] prefill parity DIVERGENCE {}: kv_sha256={} first_token={}",
                    entry.label, entry.kv_sha256, entry.first_token
                );
            }
            anyhow::bail!("chunked-prefill parity spot check failed: KV state or first-token argmax diverged from the uninterrupted prefill");
        }

        // -- Baseline: embed-only before decode admission ----------------
        println!(
            "[sched-measure] restart-cost diagnostics will be reported after the decode-only cell"
        );
        let loadavg = wait_for_quiet("baseline-embed-only")?;
        let baseline = embed_only_cell(&embed, protocol, loadavg)?;
        println!(
            "[sched-measure] baseline embed-only p95={:.2} ms ({} queries)",
            baseline
                .embed_latency
                .as_ref()
                .map(|p| p.p95_ms)
                .unwrap_or(f64::NAN),
            baseline.embed_queries
        );

        // -- Uninterrupted decode throughput baseline ---------------------
        let loadavg = wait_for_quiet("decode-only")?;
        let decode_only = decode_only_cell(&mut decode, protocol, loadavg)?;
        println!(
            "[sched-measure] decode-only throughput {:.1} tok/s",
            decode_only.decode_tokens_per_sec
        );
        let timed_generations = decode.generations_timed.max(1);
        println!(
            "[sched-measure] restart-cost breakdown per generation (over {} generations): \
             prefill={:.1} ms, kv-export={:.1} ms, kv-import={:.1} ms, stepping={:.1} ms, \
             max span={:.1} ms",
            timed_generations,
            decode.prefill_ms / timed_generations as f64,
            decode.export_ms / timed_generations as f64,
            decode.import_ms / timed_generations as f64,
            decode.stepping_ms / timed_generations as f64,
            decode.max_span_ms,
        );

        // -- Candidate cells, serially ------------------------------------
        let mut candidates = Vec::new();
        for n in CANDIDATES {
            let loadavg = wait_for_quiet(&format!("n={n}"))?;
            let cell = mixed_cell(&mut decode, &embed, n, protocol, loadavg)?;
            let p95 = cell
                .embed_latency
                .as_ref()
                .map(|p| p.p95_ms)
                .unwrap_or(f64::NAN);
            println!(
                "[sched-measure] n={n}: p50={:.2} p95={:.2} p99={:.2} ms; decode {:.1} tok/s; \
                 boundaries={} continuations={} max_quantum={:.2} ms; meets_slo={}",
                cell.embed_latency
                    .as_ref()
                    .map(|p| p.p50_ms)
                    .unwrap_or(f64::NAN),
                p95,
                cell.embed_latency
                    .as_ref()
                    .map(|p| p.p99_ms)
                    .unwrap_or(f64::NAN),
                cell.decode_tokens_per_sec,
                cell.quantum_boundaries,
                cell.continuations,
                cell.max_quantum_ms,
                p95 <= SLO_P95_MS,
            );
            candidates.push(cell);
        }

        // -- Selection: largest candidate inside the SLO -------------------
        let mut selected_n = None;
        for cell in candidates.iter().rev() {
            let p95 = cell
                .embed_latency
                .as_ref()
                .map(|p| p.p95_ms)
                .unwrap_or(f64::INFINITY);
            if p95 <= SLO_P95_MS {
                selected_n = cell.n;
                break;
            }
        }
        let mut selection_note = match selected_n {
            Some(n) => format!(
                "largest candidate with embed.query p95 <= {SLO_P95_MS} ms SLO: N={n}"
            ),
            None => format!(
                "no candidate met the {SLO_P95_MS} ms SLO; facts recorded, no N committed (review required)"
            ),
        };
        // Cancellation observations are quantum-deferral samples; they must
        // stay inside the workload's cancellation-latency bound.
        let max_cancel_ms = candidates
            .iter()
            .flat_map(|cell| cell.cancellation_latency_ms.iter())
            .copied()
            .fold(0.0f64, f64::max);
        if max_cancel_ms > CANCELLATION_BOUND_MS {
            selection_note.push_str(&format!(
                "; WARNING: max cancellation observation {max_cancel_ms:.2} ms exceeds the {CANCELLATION_BOUND_MS} ms bound"
            ));
        }
        println!("[sched-measure] selection: {selection_note}");

        // -- Same-session embed-only after decode admission (regression) ---
        let loadavg = wait_for_quiet("post-decode-embed-only")?;
        let post_decode = embed_only_cell(&embed, protocol, loadavg)?;
        let baseline_p95 = baseline
            .embed_latency
            .as_ref()
            .map(|p| p.p95_ms)
            .unwrap_or(f64::NAN);
        let post_p95 = post_decode
            .embed_latency
            .as_ref()
            .map(|p| p.p95_ms)
            .unwrap_or(f64::NAN);
        let embed_regression_pct = (post_p95 - baseline_p95) / baseline_p95 * 100.0;
        println!(
            "[sched-measure] post-decode embed-only p95={:.2} ms; same-session regression {:+.2}%",
            post_p95, embed_regression_pct
        );

        let report = Report {
            protocol_id: PROTOCOL_ID.to_string(),
            measured_at_utc,
            mode: mode_str,
            machine_profile_note,
            slo_embed_query_p95_ms: SLO_P95_MS,
            baseline_embed_only: baseline,
            decode_only_throughput: decode_only,
            candidates,
            post_decode_embed_only: post_decode,
            embed_regression_pct,
            parity_spot_check: parity,
            prefill_parity_spot_check: prefill_parity,
            selected_n,
            selection_note,
        };
        let json = serde_json::to_string_pretty(&report)?;
        println!("{json}");
        if let Some(out) = std::env::var_os("SYNAPSE_SCHED_MEASURE_OUT") {
            std::fs::write(&out, &json)?;
            println!(
                "[sched-measure] report written to {}",
                Path::new(&out).display()
            );
        }
        Ok(())
    }

    // -- Environment helpers ----------------------------------------------

    fn gte_snapshot() -> anyhow::Result<PathBuf> {
        if let Some(path) = std::env::var_os("SYNAPSE_SCHED_MEASURE_GTE_MODERNBERT") {
            return Ok(PathBuf::from(path));
        }
        let home =
            PathBuf::from(std::env::var_os("HOME").ok_or_else(|| anyhow::anyhow!("HOME unset"))?);
        let snapshots =
            home.join(".cache/huggingface/hub/models--Alibaba-NLP--gte-modernbert-base/snapshots");
        let mut found = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&snapshots) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.join("model.safetensors").is_file() && path.join("tokenizer.json").is_file()
                {
                    found.push(path);
                }
            }
        }
        found.pop().ok_or_else(|| {
            anyhow::anyhow!(
                "no gte-modernbert-base snapshot under {}",
                snapshots.display()
            )
        })
    }

    fn sysctl_value(key: &str) -> String {
        Command::new("sysctl")
            .args(["-n", key])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
            .unwrap_or_default()
    }

    fn machine_profile_note() -> String {
        let chip = sysctl_value("machdep.cpu.brand_string");
        let model = sysctl_value("hw.model");
        let mem_bytes: u64 = sysctl_value("hw.memsize").parse().unwrap_or(0);
        let cores = sysctl_value("hw.ncpu");
        let macos = Command::new("sw_vers")
            .arg("-productVersion")
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
            .unwrap_or_default();
        format!(
            "{chip} ({model}), {} GiB RAM, {cores} cores, macOS {macos}",
            mem_bytes / (1024 * 1024 * 1024)
        )
    }

    fn utc_now() -> String {
        Command::new("date")
            .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
            .unwrap_or_else(|| {
                let secs = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                format!("epoch+{secs}s")
            })
    }

    fn loadavg() -> [f64; 3] {
        let raw = sysctl_value("vm.loadavg");
        let mut values = [0.0f64; 3];
        let parsed = raw
            .split_whitespace()
            .map(|token| token.trim_matches('{').trim_matches('}'))
            .filter_map(|token| token.parse().ok());
        for (slot, value) in values.iter_mut().zip(parsed) {
            *slot = value;
        }
        values
    }

    /// Refuse to start a cell while the 1-minute loadavg exceeds the refusal
    /// threshold; poll until the shared machine is quiet (bounded wait). The
    /// protocol threshold is 8.0; `SYNAPSE_SCHED_MEASURE_MAX_LOAD` may lower
    /// it to wait for a quieter window on a shared machine.
    fn wait_for_quiet(cell: &str) -> anyhow::Result<[f64; 3]> {
        let limit = std::env::var("SYNAPSE_SCHED_MEASURE_MAX_LOAD")
            .ok()
            .and_then(|value| value.trim().parse::<f64>().ok())
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(LOADAVG_REFUSE)
            .min(LOADAVG_REFUSE);
        let mut waited = Duration::ZERO;
        loop {
            let values = loadavg();
            if values[0] <= limit {
                if values[0] > LOADAVG_NOTE {
                    println!(
                        "[sched-measure] cell {cell} starts above loadavg {:.1} (1-min {:.2}); flagging in evidence",
                        LOADAVG_NOTE, values[0]
                    );
                }
                return Ok(values);
            }
            anyhow::ensure!(
                waited < Duration::from_secs(6 * 3600),
                "cell {cell}: machine stayed above loadavg {limit} for {waited:?}; aborting"
            );
            println!(
                "[sched-measure] cell {cell} refused at 1-min loadavg {:.2} > {limit}; waiting 30 s",
                values[0]
            );
            std::thread::sleep(Duration::from_secs(30));
            waited += Duration::from_secs(30);
        }
    }

    fn percentiles(sorted: &[f64]) -> Percentiles {
        // Nearest-rank percentile: rank = ceil(q/100 * n), 1-based.
        fn rank(sorted: &[f64], q: f64) -> f64 {
            if sorted.is_empty() {
                return f64::NAN;
            }
            let rank = (q / 100.0 * sorted.len() as f64).ceil() as usize;
            sorted[rank.clamp(1, sorted.len()) - 1]
        }
        Percentiles {
            p50_ms: rank(sorted, 50.0),
            p95_ms: rank(sorted, 95.0),
            p99_ms: rank(sorted, 99.0),
        }
    }

    fn engine_error(error: synapse_core::EngineError) -> anyhow::Error {
        anyhow::anyhow!("engine error at {:?}: {}", error.stage, error.message)
    }

    fn tokenizer_error(error: Box<dyn std::error::Error + Send + Sync>) -> anyhow::Error {
        anyhow::anyhow!("tokenizer error: {error}")
    }

    // -- Decode arm --------------------------------------------------------

    struct DecodeArm {
        f16_decoder: MetalStepDecoder,
        q8_decoder: MetalStepDecoder,
        stop_tokens: Vec<u32>,
        prompt_ids: Vec<u32>,
        /// Scratch buffer for the f16->Q8 KV cache handoff between generations.
        cache_bits: Vec<u16>,
        layer_elements: usize,
        layers: usize,
        /// Diagnostic accumulators (ms) for the generation restart cost.
        prefill_ms: f64,
        export_ms: f64,
        import_ms: f64,
        stepping_ms: f64,
        generations_timed: u64,
        /// Longest single uninterruptible GPU span observed: one prefill
        /// chunk, one KV handoff span, or (in the cells) one decode quantum.
        max_span_ms: f64,
    }

    impl DecodeArm {
        /// Reset the diagnostic accumulators so a cell measures only itself.
        fn reset_timing(&mut self) {
            self.prefill_ms = 0.0;
            self.export_ms = 0.0;
            self.import_ms = 0.0;
            self.stepping_ms = 0.0;
            self.generations_timed = 0;
            self.max_span_ms = 0.0;
        }

        fn new(snapshot: &Path) -> anyhow::Result<Self> {
            let tokenizer_path = snapshot.join("tokenizer.json");
            let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(tokenizer_error)?;
            tokenizer.with_padding(None);
            tokenizer.with_truncation(None).map_err(tokenizer_error)?;

            // f16 model for prefill (the parity-certified Q8 recipe prefills
            // with f16 weights and steps with Q8 weights).
            let f16_model = Qwen3DecodeModel::load_with_quant(
                &snapshot.join("model.safetensors"),
                Precision::F16,
                WeightQuantization::None,
            )?;
            let q8_model = Qwen3DecodeModel::load_with_quant(
                &snapshot.join("model.safetensors"),
                Precision::F16,
                WeightQuantization::Q8_0,
            )?;
            let stop_tokens = q8_model.generation_stop_ids().to_vec();
            let layers = q8_model.layers.len();
            let layer_elements =
                2 * q8_model.config.num_key_value_heads * 2048 * q8_model.config.head_dim;
            // Context bucket 2048: the workload record's context_bucket.
            let f16_decoder =
                MetalStepDecoder::new(f16_model, Precision::F16, 2048, WeightQuantization::None)?;
            let q8_decoder =
                MetalStepDecoder::new(q8_model, Precision::F16, 2048, WeightQuantization::Q8_0)?;

            // Build the fixed 128-token workload prompt by cycling the token
            // IDs of the first completion prompt.
            let encoding = tokenizer
                .encode(DECODE_PROMPTS[0], true)
                .map_err(tokenizer_error)?;
            let base = encoding.get_ids().to_vec();
            anyhow::ensure!(!base.is_empty(), "decode prompt produced no tokens");
            let mut prompt_ids = Vec::with_capacity(PROMPT_LEN);
            for index in 0..PROMPT_LEN {
                prompt_ids.push(base[index % base.len()]);
            }

            Ok(Self {
                f16_decoder,
                q8_decoder,
                stop_tokens,
                prompt_ids,
                cache_bits: Vec::with_capacity(layers * layer_elements),
                layer_elements,
                layers,
                prefill_ms: 0.0,
                export_ms: 0.0,
                import_ms: 0.0,
                stepping_ms: 0.0,
                generations_timed: 0,
                max_span_ms: 0.0,
            })
        }

        /// Prefill the workload prompt on the f16 engine and hand the KV
        /// cache to the Q8 stepping engine, quantum-bounded: the prefill runs
        /// in `PREFILL_CHUNK_TOKENS`-token batched chunks (one bounded
        /// command buffer each) with `span_yield` invoked between every span
        /// — between prefill chunks and around the KV export and import
        /// spans — so the scheduler can release the decode permit there
        /// (yield-on-contention). Returns the first generated token and the
        /// Q8-side cache handle.
        fn start_generation_with_yield(
            &mut self,
            mut span_yield: impl FnMut() -> anyhow::Result<()>,
        ) -> anyhow::Result<(u32, MetalStepKvCache)> {
            let mut cache = MetalStepKvCache { position: 0 };
            let mut first = 0u32;
            for (index, chunk) in self.prompt_ids.chunks(PREFILL_CHUNK_TOKENS).enumerate() {
                if index > 0 {
                    span_yield()?;
                }
                let t0 = Instant::now();
                let argmaxes = self.f16_decoder.verify_tokens_batch(&mut cache, chunk)?;
                let chunk_ms = t0.elapsed().as_secs_f64() * 1000.0;
                self.prefill_ms += chunk_ms;
                self.max_span_ms = self.max_span_ms.max(chunk_ms);
                first = *argmaxes.last().expect("non-empty prefill chunk");
            }
            span_yield()?;
            let t0 = Instant::now();
            self.cache_bits.clear();
            for layer in 0..self.layers {
                let bits = self.f16_decoder.inspect_cache_bits(layer)?;
                anyhow::ensure!(
                    bits.len() == self.layer_elements,
                    "layer {layer} cache export has {} elements, expected {}",
                    bits.len(),
                    self.layer_elements
                );
                self.cache_bits.extend_from_slice(&bits);
            }
            let export_ms = t0.elapsed().as_secs_f64() * 1000.0;
            self.export_ms += export_ms;
            self.max_span_ms = self.max_span_ms.max(export_ms);
            span_yield()?;
            let t0 = Instant::now();
            self.q8_decoder.import_caches(&self.cache_bits)?;
            let import_ms = t0.elapsed().as_secs_f64() * 1000.0;
            self.import_ms += import_ms;
            self.max_span_ms = self.max_span_ms.max(import_ms);
            self.generations_timed += 1;
            Ok((
                first,
                MetalStepKvCache {
                    position: cache.position,
                },
            ))
        }

        /// Prefill the workload prompt on the f16 engine and hand the KV
        /// cache to the Q8 stepping engine. Chunked prefill with no yield
        /// points (decode-only cells, warmup, chunked parity streams).
        fn start_generation(&mut self) -> anyhow::Result<(u32, MetalStepKvCache)> {
            self.start_generation_with_yield(|| Ok(()))
        }

        /// Uninterrupted generation start: the v1 path — one single-command-
        /// buffer prefill of the whole prompt, then the KV handoff with no
        /// span boundaries. Kept as the bit-exactness reference for the
        /// chunked-prefill spot check.
        fn start_generation_uninterrupted(&mut self) -> anyhow::Result<(u32, MetalStepKvCache)> {
            let t0 = Instant::now();
            let (f16_cache, first) = self.f16_decoder.prefill(&self.prompt_ids)?;
            let t1 = Instant::now();
            self.cache_bits.clear();
            for layer in 0..self.layers {
                let bits = self.f16_decoder.inspect_cache_bits(layer)?;
                anyhow::ensure!(
                    bits.len() == self.layer_elements,
                    "layer {layer} cache export has {} elements, expected {}",
                    bits.len(),
                    self.layer_elements
                );
                self.cache_bits.extend_from_slice(&bits);
            }
            let t2 = Instant::now();
            self.q8_decoder.import_caches(&self.cache_bits)?;
            let t3 = Instant::now();
            self.prefill_ms += (t1 - t0).as_secs_f64() * 1000.0;
            self.export_ms += (t2 - t1).as_secs_f64() * 1000.0;
            self.import_ms += (t3 - t2).as_secs_f64() * 1000.0;
            self.max_span_ms = self.max_span_ms.max((t3 - t0).as_secs_f64() * 1000.0);
            self.generations_timed += 1;
            Ok((
                first,
                MetalStepKvCache {
                    position: f16_cache.position,
                },
            ))
        }

        /// One decode quantum: at most `budget` chained tokens. Returns the
        /// committed tokens (a stop control inside the batch ends the
        /// generation and is itself not committed).
        fn quantum(
            &mut self,
            cache: &mut MetalStepKvCache,
            seed: u32,
            budget: usize,
        ) -> anyhow::Result<(Vec<u32>, bool)> {
            let t0 = Instant::now();
            let tokens = self.q8_decoder.advance_chain(cache, seed, budget)?;
            self.stepping_ms += t0.elapsed().as_secs_f64() * 1000.0;
            if let Some(stop_at) = tokens
                .iter()
                .position(|token| self.stop_tokens.contains(token))
            {
                let committed = tokens[..stop_at].to_vec();
                return Ok((committed, true));
            }
            Ok((tokens, false))
        }

        fn warmup_generation(&mut self, n: u32) -> anyhow::Result<()> {
            let (first, mut cache) = self.start_generation()?;
            if self.stop_tokens.contains(&first) {
                return Ok(());
            }
            let mut committed = 1u32;
            let mut last = first;
            while committed < OUTPUT_LEN {
                let budget = (OUTPUT_LEN - committed).min(n) as usize;
                let (tokens, stopped) = self.quantum(&mut cache, last, budget)?;
                committed += tokens.len() as u32;
                if stopped || tokens.is_empty() {
                    break;
                }
                last = *tokens.last().unwrap();
            }
            Ok(())
        }

        /// Uninterrupted greedy stream for the parity spot check: the v1
        /// single-command-buffer prefill plus one fused decode span.
        fn uninterrupted_stream(&mut self) -> anyhow::Result<Vec<u32>> {
            let (first, mut cache) = self.start_generation_uninterrupted()?;
            let mut stream = vec![first];
            if self.stop_tokens.contains(&first) {
                return Ok(stream);
            }
            let (tokens, _) = self.quantum(&mut cache, first, (OUTPUT_LEN - 1) as usize)?;
            stream.extend(tokens);
            Ok(stream)
        }

        /// Chunked greedy stream: quanta of exactly `n` tokens (last quantum
        /// truncated to the remainder), as the scheduler runs them.
        fn chunked_stream(&mut self, n: u32) -> anyhow::Result<Vec<u32>> {
            let (first, mut cache) = self.start_generation()?;
            let mut stream = vec![first];
            if self.stop_tokens.contains(&first) {
                return Ok(stream);
            }
            let mut committed = 1u32;
            let mut last = first;
            while committed < OUTPUT_LEN {
                let budget = (OUTPUT_LEN - committed).min(n) as usize;
                let (tokens, stopped) = self.quantum(&mut cache, last, budget)?;
                committed += tokens.len() as u32;
                stream.extend(tokens.iter().copied());
                if stopped || tokens.is_empty() {
                    break;
                }
                last = *tokens.last().unwrap();
            }
            Ok(stream)
        }

        fn parity_spot_check(&mut self) -> anyhow::Result<ParitySpotCheck> {
            let uninterrupted = self.uninterrupted_stream()?;
            let digest = stream_digest(&uninterrupted);
            let mut chunked_sha256 = Vec::new();
            let mut byte_identical = true;
            for n in CANDIDATES {
                let chunked = self.chunked_stream(n)?;
                let chunked_digest = stream_digest(&chunked);
                if chunked != uninterrupted {
                    byte_identical = false;
                }
                chunked_sha256.push((n, chunked_digest));
            }
            Ok(ParitySpotCheck {
                prompt_tokens: PROMPT_LEN,
                output_tokens: OUTPUT_LEN,
                uninterrupted_sha256: digest,
                chunked_sha256,
                byte_identical,
            })
        }

        /// SHA-256 over the f16 engine's full KV cache bits (every layer,
        /// whole bucket) for the chunked-prefill bit-exactness comparison.
        fn f16_kv_digest(&mut self) -> anyhow::Result<String> {
            let mut hasher = Sha256::new();
            for layer in 0..self.layers {
                let bits = self.f16_decoder.inspect_cache_bits(layer)?;
                anyhow::ensure!(
                    bits.len() == self.layer_elements,
                    "layer {layer} cache inspection has {} elements, expected {}",
                    bits.len(),
                    self.layer_elements
                );
                for value in &bits {
                    hasher.update(value.to_le_bytes());
                }
            }
            let digest = hasher.finalize();
            Ok(digest.iter().map(|b| format!("{b:02x}")).collect())
        }

        /// Chunked-prefill bit-exactness gate. The uninterrupted single-
        /// command-buffer prefill is the reference; every chunking of the
        /// workload prompt — per-token spans of 8/16/32 and the 16-token
        /// batched spans the measured cells actually run — must leave the KV
        /// cache byte-identical and produce the same first-token argmax.
        fn prefill_parity_spot_check(&mut self) -> anyhow::Result<PrefillParitySpotCheck> {
            // Reference: one command buffer over the whole prompt.
            let (ref_cache, reference_first_token) = self.f16_decoder.prefill(&self.prompt_ids)?;
            anyhow::ensure!(
                ref_cache.position == PROMPT_LEN,
                "reference prefill advanced to {}, expected {PROMPT_LEN}",
                ref_cache.position
            );
            let reference_kv_sha256 = self.f16_kv_digest()?;

            let mut entries = Vec::new();
            let mut byte_identical = true;

            // Per-token path chunked at the candidate quantum budgets: same
            // arithmetic, host pacing only — each span is one command buffer.
            for chunk_tokens in [8usize, 16, 32] {
                let mut cache = MetalStepKvCache { position: 0 };
                let mut argmaxes = Vec::with_capacity(PROMPT_LEN);
                for chunk in self.prompt_ids.chunks(chunk_tokens) {
                    argmaxes.extend(DecodeKernel::verify_tokens(
                        &mut self.f16_decoder,
                        &mut cache,
                        chunk,
                    )?);
                }
                let first_token = *argmaxes.last().expect("non-empty prompt");
                let kv_sha256 = self.f16_kv_digest()?;
                let matches_reference =
                    kv_sha256 == reference_kv_sha256 && first_token == reference_first_token;
                byte_identical &= matches_reference;
                entries.push(PrefillParityEntry {
                    label: format!("per-token-chunk-{chunk_tokens}"),
                    chunk_tokens,
                    path: "per-token".to_string(),
                    kv_sha256,
                    first_token,
                    matches_reference,
                });
            }

            // Batched path at the prefill chunk size the cells run: mat-mat
            // spans must be arithmetic-identity-preserving vs the per-token
            // reference (accumulator order is campaign-gated).
            let mut cache = MetalStepKvCache { position: 0 };
            let mut argmaxes = Vec::with_capacity(PROMPT_LEN);
            for chunk in self.prompt_ids.chunks(PREFILL_CHUNK_TOKENS) {
                argmaxes.extend(self.f16_decoder.verify_tokens_batch(&mut cache, chunk)?);
            }
            let first_token = *argmaxes.last().expect("non-empty prompt");
            let kv_sha256 = self.f16_kv_digest()?;
            let matches_reference =
                kv_sha256 == reference_kv_sha256 && first_token == reference_first_token;
            byte_identical &= matches_reference;
            entries.push(PrefillParityEntry {
                label: format!("batched-chunk-{PREFILL_CHUNK_TOKENS}"),
                chunk_tokens: PREFILL_CHUNK_TOKENS,
                path: "batched".to_string(),
                kv_sha256,
                first_token,
                matches_reference,
            });

            Ok(PrefillParitySpotCheck {
                prompt_tokens: PROMPT_LEN,
                reference_kv_sha256,
                reference_first_token,
                entries,
                byte_identical,
            })
        }
    }

    fn stream_digest(tokens: &[u32]) -> String {
        let mut bytes = Vec::with_capacity(tokens.len() * 4);
        for token in tokens {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
        let digest = Sha256::digest(&bytes);
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }

    // -- Embed arm ----------------------------------------------------------

    struct EmbedArm {
        engine: OwnedMetalEmbedEngine,
        model: synapse_core::LoadedModel,
        tokenized: Vec<Vec<u32>>,
    }

    impl EmbedArm {
        fn new(snapshot: &Path) -> anyhow::Result<Self> {
            let mut tokenizer =
                Tokenizer::from_file(snapshot.join("tokenizer.json")).map_err(tokenizer_error)?;
            tokenizer.with_padding(None);
            tokenizer
                .with_truncation(Some(TruncationParams {
                    max_length: 512,
                    ..Default::default()
                }))
                .map_err(tokenizer_error)?;
            let mut tokenized = Vec::new();
            for query in EMBED_QUERIES {
                let encoding = tokenizer.encode(*query, true).map_err(tokenizer_error)?;
                let ids = encoding.get_ids().to_vec();
                anyhow::ensure!(!ids.is_empty(), "embed query produced no tokens");
                tokenized.push(ids);
            }

            let cache_root = std::env::temp_dir().join(format!(
                "sched-quantum-measure-packages-{}",
                std::process::id()
            ));
            std::fs::create_dir_all(&cache_root)?;
            let mut runtime = RuntimeConfig::default();
            runtime.values.insert(
                "model_path".to_string(),
                snapshot
                    .join("model.safetensors")
                    .to_string_lossy()
                    .to_string(),
            );
            runtime.values.insert(
                "package_cache_root".to_string(),
                cache_root.to_string_lossy().to_string(),
            );
            runtime
                .values
                .insert("execution".to_string(), "explicit".to_string());
            runtime
                .values
                .insert("max_tokens".to_string(), "512".to_string());
            runtime
                .values
                .insert("attention_units".to_string(), "4000000".to_string());

            let mut engine =
                OwnedMetalEmbedEngine::new(ModelFamily::GteModernBert, OwnedDType::F16);
            let model = engine
                .load(
                    &ValidatedArtifact {
                        digest: "sha256:sched-quantum-measure".to_string(),
                        format: "safetensors-package".to_string(),
                    },
                    &runtime,
                )
                .map_err(engine_error)?;
            Ok(Self {
                engine,
                model,
                tokenized,
            })
        }

        fn embed(&self, query_index: usize) -> anyhow::Result<()> {
            let ids = self.tokenized[query_index % self.tokenized.len()].clone();
            self.engine
                .embed_one(&self.model, ids)
                .map_err(engine_error)?;
            Ok(())
        }

        fn warmup(&self) -> anyhow::Result<()> {
            for index in 0..self.tokenized.len() {
                self.embed(index)?;
            }
            Ok(())
        }
    }

    // -- Cells ---------------------------------------------------------------

    /// Same-session embed-only baseline before decode admission: queries
    /// arrive at the interactive rate and run immediately (no decode load).
    fn embed_only_cell(
        embed: &EmbedArm,
        protocol: Protocol,
        loadavg_before: [f64; 3],
    ) -> anyhow::Result<CellReport> {
        let start = Instant::now();
        let mut latencies = Vec::new();
        let mut next_arrival = 0usize;
        let mut queue: VecDeque<(Instant, usize)> = VecDeque::new();
        loop {
            let now = Instant::now();
            while start + ARRIVAL_PERIOD * next_arrival as u32 <= now {
                queue.push_back((start + ARRIVAL_PERIOD * next_arrival as u32, next_arrival));
                next_arrival += 1;
            }
            while let Some((arrival, index)) = queue.pop_front() {
                embed.embed(index)?;
                latencies.push(arrival.elapsed().as_secs_f64() * 1000.0);
            }
            if start.elapsed() >= protocol.duration && latencies.len() >= protocol.min_queries {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        latencies.sort_by(f64::total_cmp);
        Ok(CellReport {
            cell: "baseline-embed-only".to_string(),
            n: None,
            loadavg_before,
            loadavg_after: loadavg(),
            ran_above_load4: loadavg_before[0] > LOADAVG_NOTE,
            window_ms: start.elapsed().as_millis() as u64,
            embed_queries: latencies.len(),
            embed_latency: Some(percentiles(&latencies)),
            decode_tokens: 0,
            decode_tokens_per_sec: 0.0,
            generations: 0,
            quantum_boundaries: 0,
            continuations: 0,
            permit_acquired: 0,
            permit_retained: 0,
            permit_released: 0,
            max_quantum_ms: 0.0,
            max_span_ms: 0.0,
            restart_cost_ms: 0.0,
            cancellation_latency_ms: Vec::new(),
            queue_depth_samples: Vec::new(),
            per_operation_waiting_ms: Vec::new(),
        })
    }

    /// Uninterrupted decode throughput: continuous generations, no embed load.
    fn decode_only_cell(
        decode: &mut DecodeArm,
        protocol: Protocol,
        loadavg_before: [f64; 3],
    ) -> anyhow::Result<CellReport> {
        decode.reset_timing();
        let start = Instant::now();
        let mut tokens = 0u64;
        let mut generations = 0u64;
        let mut boundaries = 0u64;
        let mut continuations = 0u64;
        let mut max_quantum_ms = 0.0f64;
        while start.elapsed() < protocol.decode_only {
            let (first, mut cache) = decode.start_generation()?;
            generations += 1;
            tokens += 1;
            if decode.stop_tokens.contains(&first) {
                continue;
            }
            let mut committed = 1u32;
            let mut last = first;
            while committed < OUTPUT_LEN {
                let budget = (OUTPUT_LEN - committed).min(16) as usize;
                let t0 = Instant::now();
                let (batch, stopped) = decode.quantum(&mut cache, last, budget)?;
                max_quantum_ms = max_quantum_ms.max(t0.elapsed().as_secs_f64() * 1000.0);
                boundaries += 1;
                committed += batch.len() as u32;
                tokens += batch.len() as u64;
                if stopped || batch.is_empty() {
                    break;
                }
                if committed < OUTPUT_LEN {
                    continuations += 1;
                }
                last = *batch.last().unwrap();
            }
        }
        let window = start.elapsed().as_secs_f64();
        Ok(CellReport {
            cell: "decode-only".to_string(),
            n: Some(16),
            loadavg_before,
            loadavg_after: loadavg(),
            ran_above_load4: loadavg_before[0] > LOADAVG_NOTE,
            window_ms: (window * 1000.0) as u64,
            embed_queries: 0,
            embed_latency: None,
            decode_tokens: tokens,
            decode_tokens_per_sec: tokens as f64 / window,
            generations,
            quantum_boundaries: boundaries,
            continuations,
            permit_acquired: 0,
            permit_retained: boundaries,
            permit_released: 0,
            max_quantum_ms,
            max_span_ms: decode.max_span_ms.max(max_quantum_ms),
            restart_cost_ms: restart_cost(decode),
            cancellation_latency_ms: Vec::new(),
            queue_depth_samples: Vec::new(),
            per_operation_waiting_ms: Vec::new(),
        })
    }

    /// Mixed-load cell for one candidate N: decode generates continuously in
    /// N-token quanta while embed queries arrive at the interactive rate and
    /// are served at span boundaries (yield-on-contention). Generation
    /// admission is quantum-bounded: the chunked prefill and KV handoff spans
    /// yield exactly like decode quanta.
    #[allow(clippy::too_many_lines)]
    fn mixed_cell(
        decode: &mut DecodeArm,
        embed: &EmbedArm,
        n: u32,
        protocol: Protocol,
        loadavg_before: [f64; 3],
    ) -> anyhow::Result<CellReport> {
        // Warmup: a few generations at this N plus warmup embed queries.
        for _ in 0..protocol.warmup_generations {
            decode.warmup_generation(n)?;
        }
        for index in 0..protocol.warmup_queries {
            embed.embed(index)?;
        }
        decode.reset_timing();

        let start = Instant::now();
        let mut queue: VecDeque<(Instant, usize)> = VecDeque::new();
        let mut next_arrival = 0usize;
        let mut latencies = Vec::new();
        let mut waiting_ms = Vec::new();
        let mut depth_samples = Vec::new();
        let mut tokens = 0u64;
        let mut generations = 0u64;
        let mut boundaries = 0u64;
        let mut continuations = 0u64;
        let mut permit_acquired = 0u64;
        let mut permit_retained = 0u64;
        let mut permit_released = 0u64;
        let mut max_quantum_ms = 0.0f64;
        let mut cancellation_latency_ms = Vec::new();

        'window: loop {
            if window_complete(start, protocol, latencies.len()) {
                break 'window;
            }
            // Generation admission boundary: queued embed work is served
            // before the next generation's first span authorizes.
            drain_arrivals(&mut queue, start, &mut next_arrival);
            let mut yielded = false;
            while let Some((arrival, index)) = queue.pop_front() {
                yielded = true;
                permit_released += 1;
                let dispatch = Instant::now();
                waiting_ms.push((dispatch - arrival).as_secs_f64() * 1000.0);
                embed.embed(index)?;
                latencies.push((Instant::now() - arrival).as_secs_f64() * 1000.0);
                drain_arrivals(&mut queue, start, &mut next_arrival);
            }
            if yielded {
                permit_acquired += 1;
            }
            depth_samples.push(queue.len() as u32);

            // Quantum-bounded generation start: chunked prefill plus the KV
            // handoff, each span one bounded command buffer. Between spans
            // the decode permit is released under yield-on-contention, the
            // same arbitration as quantum boundaries.
            let mut anchor = Instant::now();
            let mut span_boundary = || {
                // The previous span just completed, so the decode aging
                // anchor restarts now.
                anchor = Instant::now();
                drain_arrivals(&mut queue, start, &mut next_arrival);
                depth_samples.push(queue.len() as u32);
                let mut yielded_here = false;
                while !queue.is_empty() {
                    if Instant::now().duration_since(anchor) >= AGING_WINDOW {
                        break;
                    }
                    let (arrival, index) = queue.pop_front().unwrap();
                    yielded_here = true;
                    permit_released += 1;
                    let dispatch = Instant::now();
                    waiting_ms.push((dispatch - arrival).as_secs_f64() * 1000.0);
                    embed.embed(index)?;
                    latencies.push((Instant::now() - arrival).as_secs_f64() * 1000.0);
                    drain_arrivals(&mut queue, start, &mut next_arrival);
                }
                if yielded_here {
                    permit_acquired += 1;
                } else {
                    permit_retained += 1;
                }
                Ok(())
            };

            let (first, mut cache) = decode.start_generation_with_yield(&mut span_boundary)?;
            generations += 1;
            tokens += 1;
            if decode.stop_tokens.contains(&first) {
                if window_complete(start, protocol, latencies.len()) {
                    break 'window;
                }
                continue;
            }
            let mut committed = 1u32;
            let mut last = first;
            anchor = Instant::now();

            while committed < OUTPUT_LEN {
                // Quantum boundary: yield-on-contention. Release the decode
                // permit while embed work is queued; an aged decode anchor
                // (250 ms) preempts the yield loop, mirroring aged-DECODE
                // precedence in boundary arbitration.
                drain_arrivals(&mut queue, start, &mut next_arrival);
                depth_samples.push(queue.len() as u32);
                let mut yielded = false;
                while !queue.is_empty() {
                    if Instant::now().duration_since(anchor) >= AGING_WINDOW {
                        break;
                    }
                    let (arrival, index) = queue.pop_front().unwrap();
                    yielded = true;
                    permit_released += 1;
                    let dispatch = Instant::now();
                    waiting_ms.push((dispatch - arrival).as_secs_f64() * 1000.0);
                    embed.embed(index)?;
                    latencies.push((Instant::now() - arrival).as_secs_f64() * 1000.0);
                    drain_arrivals(&mut queue, start, &mut next_arrival);
                }
                if yielded {
                    permit_acquired += 1;
                } else {
                    permit_retained += 1;
                }

                let budget = (OUTPUT_LEN - committed).min(n) as usize;
                let t0 = Instant::now();
                let (batch, stopped) = decode.quantum(&mut cache, last, budget)?;
                let quantum_ms = t0.elapsed().as_secs_f64() * 1000.0;
                max_quantum_ms = max_quantum_ms.max(quantum_ms);
                boundaries += 1;
                // Cancellation probe: a cancellation arriving as this quantum
                // starts is evaluated at its end boundary, deferring exactly
                // the quantum duration. Sample the distribution across the
                // window.
                if cancellation_latency_ms.len() < protocol.cancel_probes && boundaries % 31 == 0 {
                    cancellation_latency_ms.push(quantum_ms);
                }
                committed += batch.len() as u32;
                tokens += batch.len() as u64;
                anchor = Instant::now();
                if stopped || batch.is_empty() {
                    break;
                }
                if committed < OUTPUT_LEN {
                    continuations += 1;
                }
                last = *batch.last().unwrap();

                if window_complete(start, protocol, latencies.len()) {
                    break 'window;
                }
            }
        }

        let window = start.elapsed().as_secs_f64();
        latencies.sort_by(f64::total_cmp);
        Ok(CellReport {
            cell: format!("n={n}"),
            n: Some(n),
            loadavg_before,
            loadavg_after: loadavg(),
            ran_above_load4: loadavg_before[0] > LOADAVG_NOTE,
            window_ms: (window * 1000.0) as u64,
            embed_queries: latencies.len(),
            embed_latency: Some(percentiles(&latencies)),
            decode_tokens: tokens,
            decode_tokens_per_sec: tokens as f64 / window,
            generations,
            quantum_boundaries: boundaries,
            continuations,
            permit_acquired,
            permit_retained,
            permit_released,
            max_quantum_ms,
            max_span_ms: decode.max_span_ms.max(max_quantum_ms),
            restart_cost_ms: restart_cost(decode),
            cancellation_latency_ms,
            queue_depth_samples: depth_samples,
            per_operation_waiting_ms: waiting_ms,
        })
    }

    /// Push every arrival whose scheduled time has passed into the queue.
    fn drain_arrivals(
        queue: &mut VecDeque<(Instant, usize)>,
        start: Instant,
        next_arrival: &mut usize,
    ) {
        let now = Instant::now();
        while start + ARRIVAL_PERIOD * *next_arrival as u32 <= now {
            queue.push_back((start + ARRIVAL_PERIOD * *next_arrival as u32, *next_arrival));
            *next_arrival += 1;
        }
    }

    /// Average generation restart cost: prefill + KV export + KV import.
    fn restart_cost(decode: &DecodeArm) -> f64 {
        let generations = (decode.generations_timed.max(1)) as f64;
        (decode.prefill_ms + decode.export_ms + decode.import_ms) / generations
    }

    fn window_complete(start: Instant, protocol: Protocol, completed_queries: usize) -> bool {
        start.elapsed() >= protocol.duration && completed_queries >= protocol.min_queries
    }
}

#[cfg(target_os = "macos")]
fn main() {
    imp::main();
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("sched_quantum_measure requires macOS with the Metal toolchain");
}
