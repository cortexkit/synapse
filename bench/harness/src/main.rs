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
    /// Run a child command under power/RSS sampling; write measurement JSON.
    Power {
        /// Where to write the measurement JSON.
        #[arg(long)]
        out: std::path::PathBuf,
        /// Sampling interval for macmon, in ms.
        #[arg(long, default_value_t = 250)]
        interval_ms: u64,
        /// The command to run (everything after --).
        #[arg(last = true, required = true)]
        cmd: Vec<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Corpus { root, out, tokenizer, target, token_budget } => {
            corpus::build(&root, &out, &tokenizer, target, token_budget)
        }
        Command::Power { out, interval_ms, cmd } => metrics::run_wrapped(&out, interval_ms, &cmd),
    }
}
