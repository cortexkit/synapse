use std::fs;
use std::path::PathBuf;

use anyhow::{ensure, Result};
use clap::Parser;
use serde_json::Value;

#[derive(Parser)]
struct Args {
    #[arg(required = true)]
    results: Vec<PathBuf>,
}

fn cell(value: &Value) -> String {
    match value {
        Value::Null => "NA".into(),
        Value::String(text) => text.replace(['\t', '\r', '\n'], " "),
        _ => value.to_string(),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("engine\tversion\tcommit\tmodel\tbackend\tsplit\tcompleted\trequested\twer_truth\twer_nvidia\trtf\tp50_s\tp95_s\tcold_s\ttotal_wall_s\twarm_wall_s\tpeak_rss_bytes\tgpu_peak_bytes\tenergy_joules");
    for path in args.results {
        let result: Value = serde_json::from_slice(&fs::read(path)?)?;
        ensure!(
            result["schema"] == "synapse-stt-benchmark-v1",
            "unknown result schema"
        );
        for (split, row) in result["splits"]
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("missing splits"))?
        {
            let mut fields: Vec<_> = ["engine", "version", "commit", "model", "backend"]
                .iter()
                .map(|k| cell(&result[k]))
                .collect();
            fields.push(split.clone());
            fields.extend(
                [
                    "completed",
                    "requested",
                    "ground_truth_wer",
                    "nvidia_fp32_wer",
                    "real_time_factor",
                    "latency_p50_seconds",
                    "latency_p95_seconds",
                ]
                .iter()
                .map(|k| cell(&row[k])),
            );
            fields.extend(
                [
                    "cold_start_seconds",
                    "total_wall_seconds",
                    "warm_wall_seconds",
                    "peak_rss_bytes",
                    "gpu_peak_bytes",
                    "energy_joules",
                ]
                .iter()
                .map(|k| cell(&result[k])),
            );
            println!("{}", fields.join("\t"));
        }
    }
    Ok(())
}

#[test]
fn stt_table_distinguishes_unavailable_from_zero_and_escapes_cells() {
    assert_eq!(cell(&Value::Null), "NA");
    assert_eq!(cell(&serde_json::json!(0)), "0");
    assert_eq!(cell(&serde_json::json!("a\tb\nc")), "a b c");
}
