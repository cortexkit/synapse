use std::fs;
use std::path::PathBuf;

use anyhow::{ensure, Result};
use clap::Parser;
use serde_json::Value;

#[derive(Parser)]
struct Args {
    /// One row per engine/model/hardware result, with WER columns per split.
    #[arg(long)]
    wide: bool,
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
    if args.wide {
        let results = args
            .results
            .iter()
            .map(|p| Ok(serde_json::from_slice(&fs::read(p)?)?))
            .collect::<Result<Vec<Value>>>()?;
        for row in wide_rows(&results)? {
            println!("{}", row.join("\t"));
        }
        return Ok(());
    }
    println!("engine\tversion\tcommit\tmodel\tbackend\tsplit\tcompleted\trequested\twer_truth\twer_nvidia\trtf\tp50_s\tp95_s\tcold_s\ttotal_wall_s\twarm_wall_s\tpeak_rss_bytes\tgpu_peak_bytes\tenergy_joules\tmodel_size_bytes\tprocess_peak_bytes\tprocess_peak_method\tgpu_peak_method\tgpu_idle_baseline_bytes\tgpu_sample_interval_ms\tgpu_max_sample_gap_ms\tane_peak_bytes\tweight_precision\tmachine\tos\tgpu_after_load_idle_bytes");
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
                    "model_size_bytes",
                    "process_peak_bytes",
                    "process_peak_method",
                    "gpu_peak_method",
                    "gpu_idle_baseline_bytes",
                    "gpu_sample_interval_ms",
                    "gpu_max_sample_gap_ms",
                    "ane_peak_bytes",
                    "weight_precision",
                    "machine",
                    "os",
                    "gpu_after_load_idle_bytes",
                ]
                .iter()
                .map(|k| cell(&result[k])),
            );
            println!("{}", fields.join("\t"));
        }
    }
    Ok(())
}

fn wide_rows(results: &[Value]) -> Result<Vec<Vec<String>>> {
    let columns = [
        "engine",
        "version",
        "commit",
        "model",
        "backend",
        "weight_precision",
        "machine",
        "os",
        "model_size_bytes",
        "process_peak_bytes",
        "process_peak_method",
        "peak_rss_bytes",
        "gpu_peak_bytes",
        "gpu_after_load_idle_bytes",
        "gpu_peak_method",
        "gpu_idle_baseline_bytes",
        "gpu_sample_interval_ms",
        "gpu_max_sample_gap_ms",
        "ane_peak_bytes",
        "latency_p50_seconds",
        "latency_p95_seconds",
        "cold_start_seconds",
        "total_wall_seconds",
        "warm_wall_seconds",
        "warm_inference_seconds",
        "real_time_factor",
        "completed",
        "requested",
    ];
    let mut splits = std::collections::BTreeSet::new();
    for result in results {
        ensure!(
            result["schema"] == "synapse-stt-benchmark-v1",
            "unknown result schema"
        );
        splits.extend(
            result["splits"]
                .as_object()
                .ok_or_else(|| anyhow::anyhow!("missing splits"))?
                .keys()
                .cloned(),
        );
    }
    let split_columns = [
        "completed",
        "requested",
        "ground_truth_wer",
        "nvidia_fp32_wer",
    ];
    let mut header: Vec<_> = columns.iter().map(|c| (*c).to_owned()).collect();
    for split in &splits {
        header.extend(split_columns.iter().map(|c| format!("{split}.{c}")));
    }
    let mut rows = vec![header];
    for result in results {
        let mut row: Vec<_> = columns.iter().map(|c| cell(&result[c])).collect();
        for split in &splits {
            row.extend(
                split_columns
                    .iter()
                    .map(|c| cell(&result["splits"][split][c])),
            );
        }
        rows.push(row);
    }
    Ok(rows)
}

#[test]
fn stt_table_distinguishes_unavailable_from_zero_and_escapes_cells() {
    assert_eq!(cell(&Value::Null), "NA");
    assert_eq!(cell(&serde_json::json!(0)), "0");
    assert_eq!(cell(&serde_json::json!("a\tb\nc")), "a b c");
}

#[test]
fn stt_table_wide_keeps_one_row_per_result_and_old_schema_nulls() {
    let old = serde_json::json!({"schema":"synapse-stt-benchmark-v1", "engine":"old", "splits":{"english":{"ground_truth_wer":0.25,"nvidia_fp32_wer":0.5}}});
    let new = serde_json::json!({"schema":"synapse-stt-benchmark-v1", "engine":"new", "model_size_bytes":123, "process_peak_bytes":456, "gpu_peak_bytes":0,"gpu_after_load_idle_bytes":111,"splits":{"other":{"ground_truth_wer":0.75}}});
    let rows = wide_rows(&[old, new]).unwrap();
    assert_eq!(rows.len(), 3);
    assert!(rows.iter().all(|r| r.len() == rows[0].len()));
    let column = |name| rows[0].iter().position(|v| v == name).unwrap();
    assert_eq!(rows[1][column("model_size_bytes")], "NA");
    assert_eq!(rows[2][column("model_size_bytes")], "123");
    assert_eq!(rows[2][column("process_peak_bytes")], "456");
    assert_eq!(rows[2][column("gpu_peak_bytes")], "0");
    assert_eq!(rows[2][column("gpu_after_load_idle_bytes")], "111");
    assert_eq!(rows[1][column("english.ground_truth_wer")], "0.25");
    assert_eq!(rows[1][column("english.nvidia_fp32_wer")], "0.5");
    assert_eq!(rows[2][column("english.ground_truth_wer")], "NA");
    assert_eq!(rows[2][column("other.ground_truth_wer")], "0.75");
}
