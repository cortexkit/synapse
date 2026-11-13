//! Aggregate-only bridge to the same corpus validation and WER scorer as stt-bench.
use anyhow::{ensure, Context, Result};
use std::{env, fs, path::PathBuf};
use stt_common::{load_corpus, measurement::WorkerResult, score};

fn main() -> Result<()> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    ensure!(
        args.len() == 3,
        "usage: stt-bias-parity-score WAVS WORKER_JSON OUTPUT"
    );
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(4)
        .context("repository root")?
        .to_path_buf();
    let private = root.join("target/stt-bias").canonicalize()?;
    let input = PathBuf::from(&args[1]).canonicalize()?;
    let output = PathBuf::from(&args[2]);
    ensure!(
        input.starts_with(&private),
        "worker input must stay under target/stt-bias"
    );
    ensure!(!output.exists(), "never overwrite a scored run");
    ensure!(
        output
            .parent()
            .context("output parent")?
            .canonicalize()?
            .starts_with(&private),
        "output must stay under target/stt-bias"
    );
    let corpus = load_corpus(&root.join("bench/parity"), &PathBuf::from(&args[0]))?;
    let result: WorkerResult = serde_json::from_slice(&fs::read(input)?)?;
    ensure!(
        result.transcripts.iter().all(|t| t.text.is_some()),
        "incomplete corpus"
    );
    let splits = score(&corpus, &result)?;
    fs::write(output, serde_json::to_vec_pretty(&splits)?)?;
    println!("Scored {} splits with stt-common/parity WER", splits.len());
    Ok(())
}
