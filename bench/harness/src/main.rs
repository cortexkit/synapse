//! Synapse decision #1 bench harness.
//!
//! Subcommands:
//! - `corpus`: chunk a local source tree into embedding-workload chunks (JSONL).
//! - `power`: wrap a child command with macmon power sampling and emit a
//!   measurement JSON (used by lane runners so every lane is measured the
//!   same way).
//!
//! Lane binaries (ort / mlx / llama-server / burn) live in sibling crates and
//! read the same corpus + emit the same result schema.

mod corpus;
mod metrics;

use synapse_bench::parity;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "synapse-bench")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Chunk a source tree into a JSONL corpus of code chunks.
    Corpus {
        /// Root of the source tree to chunk.
        #[arg(long)]
        root: std::path::PathBuf,
        /// Output JSONL path.
        #[arg(long)]
        out: std::path::PathBuf,
        /// tokenizer.json used for token counting.
        #[arg(long)]
        tokenizer: std::path::PathBuf,
        /// Target number of chunks (stops early when reached).
        #[arg(long, default_value_t = 4000)]
        target: usize,
        /// Per-chunk token budget (chunks are cut to stay under this).
        #[arg(long, default_value_t = 448)]
        token_budget: usize,
    },
    /// Compare two vector spaces: mean cosine + top-k rank overlap.
    /// Both files are JSONL of {id, vec}; ids are intersected.
    Parity {
        /// Reference vector space (e.g. ort fp32).
        #[arg(long)]
        reference: std::path::PathBuf,
        /// Candidate vector space (e.g. mlx 4-bit DWQ).
        #[arg(long)]
        candidate: std::path::PathBuf,
        /// Top-k for rank-overlap.
        #[arg(long, default_value_t = 10)]
        k: usize,
        /// Use every Nth shared id as a rank query (full corpus x corpus is
        /// quadratic; stride keeps it tractable).
        #[arg(long, default_value_t = 50)]
        stride: usize,
    },
    /// Run a child command under power/RSS sampling; write measurement JSON.
    /// Refuses to run unless the machine is idle (see --max-cpu/--max-gpu).
    Power {
        /// Where to write the measurement JSON.
        #[arg(long)]
        out: std::path::PathBuf,
        /// Sampling interval for macmon, in ms.
        #[arg(long, default_value_t = 250)]
        interval_ms: u64,
        /// Skip the idle preflight (integration smoke only; never for
        /// published numbers).
        #[arg(long)]
        skip_idle_check: bool,
        /// Idle gate: max average CPU utilization percent.
        #[arg(long, default_value_t = 15.0)]
        max_cpu: f64,
        /// Idle gate: max average GPU utilization percent.
        #[arg(long, default_value_t = 5.0)]
        max_gpu: f64,
        /// The command to run (everything after --).
        #[arg(last = true, required = true)]
        cmd: Vec<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Corpus {
            root,
            out,
            tokenizer,
            target,
            token_budget,
        } => corpus::build(&root, &out, &tokenizer, target, token_budget),
        Command::Power {
            out,
            interval_ms,
            skip_idle_check,
            max_cpu,
            max_gpu,
            cmd,
        } => metrics::run_wrapped(&out, interval_ms, &cmd, skip_idle_check, max_cpu, max_gpu),
        Command::Parity {
            reference,
            candidate,
            k,
            stride,
        } => {
            let reference = parity::load_reference(&reference)?;
            let candidate_vecs = parity::load_reference(&candidate)?;
            let (mean, matched) = parity::mean_parity(
                candidate_vecs.iter().map(|(id, v)| (id.clone(), v.clone())),
                &reference,
            );
            let rank = parity::rank_overlap(&reference, &candidate_vecs, k, stride)?;
            let report = serde_json::json!({
                "matched": matched,
                "mean_cosine": mean,
                "rank": rank,
            });
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
    }
}
