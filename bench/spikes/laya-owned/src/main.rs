mod head;
mod sequence;
use anyhow::{anyhow, ensure, Result};
use serde_json::{json, Value};
use std::{path::PathBuf, time::Instant};
use synapse_core::{EmbedEngine, RuntimeConfig, ValidatedArtifact};
use synapse_engine_owned::{ModelFamily, OwnedDType, OwnedMetalEmbedEngine};
use tokenizers::Tokenizer;

const REVISION: &str = "c5d78730f3493e4fe16d61507ef4b78eef7318cf";
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}
fn hf_home() -> PathBuf {
    std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap()).join(".cache/huggingface")
        })
}
fn snapshot() -> PathBuf {
    hf_home()
        .join("hub/models--convaiinnovations--laya/snapshots")
        .join(REVISION)
}
fn battery() -> Result<Value> {
    Ok(serde_json::from_slice(&std::fs::read(
        root().join("battery.json"),
    )?)?)
}
fn check_sequences(b: &Value) -> Result<usize> {
    let tok = Tokenizer::from_file(snapshot().join("tokenizer/tokenizer.json"))
        .map_err(|e| anyhow!("{e}"))?;
    let mut count = 0;
    for case in b["cases"].as_array().unwrap() {
        for (name, q) in case["questions"].as_object().unwrap() {
            let (ids, markers) = sequence::build_sequence(&tok, &case["state"], q, 512, 192)?;
            ensure!(
                json!(ids) == case["rows"][name]["input_ids"],
                "input ids differ: {name}"
            );
            ensure!(
                json!(markers) == case["rows"][name]["markers"],
                "markers differ: {name}"
            );
            count += 1;
        }
    }
    println!("sequence parity: {count}/{count}");
    Ok(count)
}
fn main() -> Result<()> {
    let b = battery()?;
    let count = check_sequences(&b)?;
    if std::env::args().any(|s| s == "--sequences-only") {
        return Ok(());
    }
    let head = head::Head::new(
        &snapshot().join("model.safetensors"),
        serde_json::from_slice(&std::fs::read(snapshot().join("rl_agent_config.json"))?)?,
    )?;
    let dtype = if std::env::args().any(|s| s == "--f32") {
        OwnedDType::F32
    } else {
        OwnedDType::F16
    };
    let mut engine = OwnedMetalEmbedEngine::new(ModelFamily::GteModernBert, dtype);
    let mut cfg = RuntimeConfig::default();
    cfg.values.insert(
        "model_path".into(),
        hf_home()
            .join("laya-owned")
            .join(REVISION)
            .join("model.safetensors")
            .display()
            .to_string(),
    );
    cfg.values.insert(
        "package_cache_root".into(),
        hf_home().join("laya-owned/metal").display().to_string(),
    );
    cfg.values.insert("execution".into(), "lazy".into());
    cfg.values.insert("max_tokens".into(), "512".into());
    cfg.values
        .insert("attention_units".into(), "2097152".into());
    let model = engine
        .load(
            &ValidatedArtifact {
                digest: b["checkpoint_sha256"].as_str().unwrap().into(),
                format: "safetensors-package".into(),
            },
            &cfg,
        )
        .map_err(|e| anyhow!("load: {e:?}"))?;
    println!("encoder loaded: {}", dtype.as_str());
    let mut results = Vec::new();
    for case in b["cases"].as_array().unwrap() {
        for (name, row) in case["rows"].as_object().unwrap() {
            let ids: Vec<u32> = serde_json::from_value(row["input_ids"].clone())?;
            let markers: Vec<usize> = serde_json::from_value(row["markers"].clone())?;
            let hidden = engine
                .encode_hidden(&model, &[ids], None)
                .map_err(|e| anyhow!("encode: {e:?}"))?;
            let (out, head_hidden) = head.forward(
                &hidden.data,
                &hidden.attention_mask,
                &markers,
                row["qtype"].as_u64().unwrap() as usize,
            );
            let mut dumps = std::collections::HashMap::new();
            let encoder_bytes: Vec<u8> = hidden.data.iter().flat_map(|v| v.to_le_bytes()).collect();
            let head_bytes: Vec<u8> = head_hidden.iter().flat_map(|v| v.to_le_bytes()).collect();
            dumps.insert(
                "encoder",
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    vec![1, hidden.seq, hidden.hidden],
                    &encoder_bytes,
                )?,
            );
            dumps.insert(
                "head",
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    vec![1, hidden.seq, hidden.hidden],
                    &head_bytes,
                )?,
            );
            let dir = root().join(format!("dumps/owned-{}", dtype.as_str()));
            std::fs::create_dir_all(&dir)?;
            safetensors::tensor::serialize_to_file(
                dumps,
                None,
                &dir.join(format!("{name}.safetensors")),
            )?;
            results.push(json!({"name":name,"output":out}));
            println!("parity row {name}");
        }
    }
    let output = root().join(format!("owned-{}.json", dtype.as_str()));
    std::fs::write(
        &output,
        serde_json::to_vec_pretty(
            &json!({"dtype":dtype.as_str(),"sequence_matches":count,"rows":results}),
        )?,
    )?;
    if std::env::args().any(|s| s == "--parity-only") {
        return Ok(());
    }
    let tok = Tokenizer::from_file(snapshot().join("tokenizer/tokenizer.json"))
        .map_err(|e| anyhow!("{e}"))?;
    let mut measurement_rows = Vec::new();
    for case in b["cases"].as_array().unwrap().iter().take(8) {
        let q = case["questions"]
            .as_object()
            .unwrap()
            .values()
            .next()
            .unwrap();
        let (ids, markers) = sequence::build_sequence(&tok, &b["cases"][0]["state"], q, 512, 192)?;
        let qt = match q["type"].as_str().unwrap() {
            "choice" => 0,
            "score" => 1,
            _ => 2,
        };
        measurement_rows.push((ids, markers, qt));
    }
    let mut timings = Vec::new();
    let uptime_before = std::process::Command::new("uptime").output()?;
    for (batch, seq) in [(1, 128), (1, 256), (1, 512), (8, 512)] {
        let sequences: Vec<Vec<u32>> = measurement_rows[..batch]
            .iter()
            .map(|r| r.0.clone())
            .collect();
        let mut enc = Vec::new();
        let mut cpu = Vec::new();
        for i in 0..23 {
            let start = Instant::now();
            let hidden = engine
                .encode_hidden(&model, &sequences, Some((batch, seq)))
                .map_err(|e| anyhow!("encode: {e:?}"))?;
            let encoder_ms = start.elapsed().as_secs_f64() * 1000.;
            let start = Instant::now();
            for row in 0..batch {
                std::hint::black_box(head.forward(
                    &hidden.data[row * seq * hidden.hidden..(row + 1) * seq * hidden.hidden],
                    &hidden.attention_mask[row * seq..(row + 1) * seq],
                    &measurement_rows[row].1,
                    measurement_rows[row].2,
                ));
            }
            let head_ms = start.elapsed().as_secs_f64() * 1000.;
            if i >= 3 {
                enc.push(encoder_ms);
                cpu.push(head_ms);
            }
        }
        timings.push(json!({"batch":batch,"seq":seq,"encoder_wall_ms":enc,"head_wall_ms":cpu}));
        println!("measured {batch}x{seq}");
    }
    let uptime_after = std::process::Command::new("uptime").output()?;
    std::fs::write(
        root().join("timings.json"),
        serde_json::to_vec_pretty(
            &json!({"uptime_before":String::from_utf8_lossy(&uptime_before.stdout),"uptime_after":String::from_utf8_lossy(&uptime_after.stdout),"buckets":timings}),
        )?,
    )?;
    Ok(())
}
#[cfg(test)]
mod tests {
    #[test]
    fn battery_ids_and_markers_are_byte_exact() {
        assert_eq!(
            super::check_sequences(&super::battery().unwrap()).unwrap(),
            24
        );
    }
}
