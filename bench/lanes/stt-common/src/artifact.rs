use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Serialize)]
pub struct Artifact {
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
}

pub fn artifacts(path: &Path) -> Result<Vec<Artifact>> {
    let mut result = Vec::new();
    collect(path, &mut result)?;
    result.sort_by(|a, b| a.path.cmp(&b.path));
    ensure!(!result.is_empty(), "empty model directory");
    Ok(result)
}

/// Select only components opened by the chosen lane, not alternate weights or
/// download-cache metadata that happen to share the model directory.
pub fn model_artifacts(path: &Path, backend: &str) -> Result<Vec<Artifact>> {
    if backend == "apple-speech" {
        return Ok(Vec::new());
    }
    if matches!(backend, "ane-pal6" | "ane-int8") {
        let encoder = if backend == "ane-pal6" {
            "Encoder.mlmodelc"
        } else {
            "Encoder_v2.mlmodelc"
        };
        let mut files = Vec::new();
        for name in [
            "Preprocessor.mlmodelc",
            encoder,
            "Decoder.mlmodelc",
            "JointDecisionv3.mlmodelc",
            "parakeet_vocab.json",
        ] {
            files.extend(artifacts(&path.join(name))?);
        }
        return Ok(files);
    }
    if backend.starts_with("synapse-") {
        if backend != "synapse-metal" {
            return artifacts(&if path.is_dir() {
                path.join("model.safetensors")
            } else {
                path.into()
            });
        }
        let mut files = artifacts(&path.join("config.json"))?;
        let single = path.join("model.safetensors");
        if single.is_file() {
            files.extend(artifacts(&single)?);
        } else {
            let index_path = path.join("model.safetensors.index.json");
            let index: serde_json::Value = serde_json::from_slice(&fs::read(&index_path)?)?;
            let map = index["weight_map"]
                .as_object()
                .ok_or_else(|| anyhow::anyhow!("missing weight_map"))?;
            let names: std::collections::BTreeSet<_> = map
                .values()
                .map(|v| {
                    v.as_str()
                        .ok_or_else(|| anyhow::anyhow!("invalid shard name"))
                })
                .collect::<Result<_>>()?;
            files.extend(artifacts(&index_path)?);
            for name in names {
                let shard = Path::new(name);
                ensure!(
                    shard
                        .components()
                        .all(|c| matches!(c, std::path::Component::Normal(_))),
                    "invalid model shard path"
                );
                files.extend(artifacts(&path.join(shard))?);
            }
        }
        return Ok(files);
    }
    artifacts(path)
}

pub fn weight_precision(path: &Path, backend: &str) -> &'static str {
    match backend {
        "synapse-metal" => "f32",
        "synapse-ane" => "mixed-f16-f32",
        "synapse-cuda-f16" | "synapse-vulkan-f16" => "f16",
        "synapse-cuda-f32" | "synapse-vulkan-f32" => "f32-measurement",
        "ane-pal6" => "palettized-6bit",
        "ane-int8" => "int8",
        "apple-speech" => "OS-managed-unknown",
        _ if path
            .file_name()
            .is_some_and(|n| n.to_string_lossy().contains("Q8_0")) =>
        {
            "q8_0"
        }
        _ if path
            .file_name()
            .is_some_and(|n| n.to_string_lossy().contains("F16")) =>
        {
            "f16"
        }
        _ => "unknown",
    }
}

#[test]
fn stt_artifacts_count_only_loaded_components_and_precision() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../target/scratch")
        .join(format!("stt-artifacts-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    for (name, bytes) in [
        ("Preprocessor.mlmodelc/weights", 2),
        ("Encoder.mlmodelc/weights", 3),
        ("Encoder_v2.mlmodelc/weights", 9),
        ("Decoder.mlmodelc/weights", 4),
        ("JointDecisionv3.mlmodelc/weights", 5),
        ("parakeet_vocab.json", 6),
        ("unused", 100),
    ] {
        let path = root.join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, vec![0; bytes]).unwrap();
    }
    let files = model_artifacts(&root, "ane-pal6").unwrap();
    assert_eq!(files.len(), 5);
    assert_eq!(files.iter().map(|f| f.bytes).sum::<u64>(), 20);
    assert_eq!(
        model_artifacts(&root, "ane-int8")
            .unwrap()
            .iter()
            .map(|f| f.bytes)
            .sum::<u64>(),
        26
    );
    fs::write(root.join("model.safetensors"), [0; 7]).unwrap();
    assert_eq!(
        model_artifacts(&root, "synapse-cuda-f16").unwrap()[0].bytes,
        7
    );
    assert_eq!(
        model_artifacts(&root.join("model.safetensors"), "synapse-ane").unwrap()[0].bytes,
        7
    );
    assert!(model_artifacts(&root, "apple-speech").unwrap().is_empty());
    assert_eq!(
        weight_precision(Path::new("parakeet-Q8_0.gguf"), "cuda"),
        "q8_0"
    );
    assert_eq!(weight_precision(&root, "synapse-ane"), "mixed-f16-f32");
    fs::remove_dir_all(root).unwrap();
}

fn collect(path: &Path, result: &mut Vec<Artifact>) -> Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    // Hugging Face snapshots use file symlinks into their content-addressed
    // cache. Follow files but not directory links, which could introduce cycles.
    if path.is_file() {
        let mut file = File::open(path)?;
        let mut digest = Sha256::new();
        let mut count = 0;
        let mut buffer = vec![0; 1024 * 1024];
        loop {
            let length = file.read(&mut buffer)?;
            if length == 0 {
                break;
            }
            digest.update(&buffer[..length]);
            count += length as u64;
        }
        result.push(Artifact {
            path: path.to_path_buf(),
            bytes: count,
            sha256: format!("{:x}", digest.finalize()),
        });
    } else {
        ensure!(
            metadata.is_dir() && !metadata.is_symlink(),
            "not a file or directory: {}",
            path.display()
        );
        for entry in fs::read_dir(path)? {
            collect(&entry?.path(), result)?;
        }
    }
    Ok(())
}
