//! Persistent Q8_0 derived-artifact construction for production decode.
//!
//! The cache object is the exact concatenation of GGUF `block_q8_0` bytes in
//! the model loader's stable tensor order. Its SHA-256 is therefore both the
//! derived digest used by decode identity and a direct integrity check over all
//! blocks. A sidecar records tensor boundaries and lineage; the sidecar is
//! published last so an interrupted ingest never leaves a loadable object.

use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use anyhow::{bail, ensure, Context, Result};
use half::{bf16, f16};
use safetensors::{tensor::Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::owned_decode_routing::family::Family;

const Q8_BLOCK_ELEMENTS: usize = 32;
const Q8_BLOCK_BYTES: usize = 34;
const ARTIFACT_FORMAT: &str = "synapse-owned-decode-q8_0-v1";

/// A verified-on-read or newly published derived Q8 artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CachedQ8Artifact {
    pub derived_digest: String,
    pub object_path: PathBuf,
    pub metadata_path: PathBuf,
    pub reused: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Q8ArtifactMetadata {
    format: String,
    family: String,
    source_manifest_digest: String,
    quantizer_revision: String,
    derived_digest: String,
    object_file: String,
    tensors: Vec<Q8TensorMetadata>,
    reproducible: bool,
    derivable: bool,
    evictable: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Q8TensorMetadata {
    name: String,
    shape: Vec<usize>,
    byte_offset: u64,
    byte_length: u64,
}

/// Derive and atomically cache every quantized matrix block for one decode lane.
///
/// The transaction key includes the quantizer revision. A complete existing
/// object is hashed before reuse; callers compare the returned digest with the
/// registered expected digest through `Q8IngestRegistry`.
pub fn derive_and_cache_q8_blocks(
    source_path: &Path,
    cache_root: &Path,
    family: Family,
    source_manifest_digest: &str,
    quantizer_revision: &str,
) -> Result<CachedQ8Artifact> {
    ensure!(
        !source_manifest_digest.trim().is_empty(),
        "Q8 source manifest digest must not be empty"
    );
    ensure!(
        !quantizer_revision.trim().is_empty(),
        "Q8 quantizer revision must not be empty"
    );

    let source_key = source_manifest_digest
        .strip_prefix("sha256:")
        .unwrap_or(source_manifest_digest);
    let key_digest = hex::encode(Sha256::digest(
        [source_key.as_bytes(), b"\0", quantizer_revision.as_bytes()].concat(),
    ));
    let artifact_dir = cache_root.join("owned-decode-q8").join(key_digest);
    fs::create_dir_all(&artifact_dir)
        .with_context(|| format!("create Q8 cache directory {}", artifact_dir.display()))?;
    let object_path = artifact_dir.join("weights.q8_0");
    let metadata_path = artifact_dir.join("lineage.json");

    if let Some(cached) = load_cached_artifact(
        &object_path,
        &metadata_path,
        family,
        source_manifest_digest,
        quantizer_revision,
    )? {
        return Ok(cached);
    }

    let model_path = resolve_model_path(source_path)?;
    let model_root = model_path.parent().unwrap_or_else(|| Path::new("."));
    let config_path = model_root.join("config.json");
    let config: Value = serde_json::from_slice(
        &fs::read(&config_path)
            .with_context(|| format!("read decode config {}", config_path.display()))?,
    )
    .with_context(|| format!("parse decode config {}", config_path.display()))?;
    let source_bytes = fs::read(&model_path)
        .with_context(|| format!("read safetensors source {}", model_path.display()))?;
    let tensors = SafeTensors::deserialize(&source_bytes)
        .with_context(|| format!("parse safetensors source {}", model_path.display()))?;
    let tensor_names = tensor_order(family, &config, &tensors)?;

    let nonce = format!("{}.{}", std::process::id(), crate::now_ms());
    let temp_object = artifact_dir.join(format!("weights.{nonce}.tmp"));
    let temp_metadata = artifact_dir.join(format!("lineage.{nonce}.tmp"));
    let transaction = (|| -> Result<(String, Vec<Q8TensorMetadata>)> {
        let file = File::create(&temp_object)
            .with_context(|| format!("create private Q8 object {}", temp_object.display()))?;
        let mut writer = BufWriter::new(file);
        let mut digest = Sha256::new();
        let mut offset = 0_u64;
        let mut inventory = Vec::with_capacity(tensor_names.len());
        for name in tensor_names {
            let tensor = lookup_tensor(&tensors, &name)?;
            let shape = tensor.shape().to_vec();
            ensure!(shape.len() == 2, "Q8 tensor {name} must be a matrix");
            let row_width = shape[1];
            ensure!(
                row_width > 0 && row_width % Q8_BLOCK_ELEMENTS == 0,
                "Q8 tensor {name} row width {row_width} is not block aligned"
            );
            let byte_length = write_q8_tensor(&mut writer, &mut digest, name.as_str(), &tensor)?;
            inventory.push(Q8TensorMetadata {
                name,
                shape,
                byte_offset: offset,
                byte_length,
            });
            offset = offset
                .checked_add(byte_length)
                .context("Q8 artifact byte offset overflow")?;
        }
        writer.flush().context("flush private Q8 object")?;
        writer
            .get_ref()
            .sync_all()
            .context("sync private Q8 object")?;
        Ok((hex::encode(digest.finalize()), inventory))
    })();

    let (derived_digest, inventory) = match transaction {
        Ok(outcome) => outcome,
        Err(error) => {
            let _ = fs::remove_file(&temp_object);
            let _ = fs::remove_file(&temp_metadata);
            return Err(error);
        }
    };
    let metadata = Q8ArtifactMetadata {
        format: ARTIFACT_FORMAT.to_string(),
        family: family.as_str().to_string(),
        source_manifest_digest: source_manifest_digest.to_string(),
        quantizer_revision: quantizer_revision.to_string(),
        derived_digest: derived_digest.clone(),
        object_file: "weights.q8_0".to_string(),
        tensors: inventory,
        reproducible: true,
        derivable: true,
        evictable: true,
    };
    let metadata_bytes = serde_json::to_vec_pretty(&metadata).context("serialize Q8 lineage")?;
    {
        let mut file = File::create(&temp_metadata)
            .with_context(|| format!("create private Q8 lineage {}", temp_metadata.display()))?;
        file.write_all(&metadata_bytes)
            .context("write private Q8 lineage")?;
        file.sync_all().context("sync private Q8 lineage")?;
    }

    // The object becomes loadable only when its matching lineage sidecar is
    // published. The sidecar rename is deliberately last.
    let _ = fs::remove_file(&object_path);
    let _ = fs::remove_file(&metadata_path);
    fs::rename(&temp_object, &object_path).with_context(|| {
        format!(
            "publish Q8 object {} -> {}",
            temp_object.display(),
            object_path.display()
        )
    })?;
    if let Err(error) = fs::rename(&temp_metadata, &metadata_path) {
        let _ = fs::remove_file(&object_path);
        let _ = fs::remove_file(&temp_metadata);
        return Err(error).with_context(|| {
            format!(
                "publish Q8 lineage {} -> {}",
                temp_metadata.display(),
                metadata_path.display()
            )
        });
    }

    Ok(CachedQ8Artifact {
        derived_digest,
        object_path,
        metadata_path,
        reused: false,
    })
}

fn load_cached_artifact(
    object_path: &Path,
    metadata_path: &Path,
    family: Family,
    source_manifest_digest: &str,
    quantizer_revision: &str,
) -> Result<Option<CachedQ8Artifact>> {
    if !object_path.is_file() || !metadata_path.is_file() {
        return Ok(None);
    }
    let metadata: Q8ArtifactMetadata = serde_json::from_slice(
        &fs::read(metadata_path)
            .with_context(|| format!("read Q8 lineage {}", metadata_path.display()))?,
    )
    .with_context(|| format!("parse Q8 lineage {}", metadata_path.display()))?;
    if metadata.format != ARTIFACT_FORMAT
        || metadata.family != family.as_str()
        || metadata
            .source_manifest_digest
            .strip_prefix("sha256:")
            .unwrap_or(&metadata.source_manifest_digest)
            != source_manifest_digest
                .strip_prefix("sha256:")
                .unwrap_or(source_manifest_digest)
        || metadata.quantizer_revision != quantizer_revision
        || metadata.object_file != "weights.q8_0"
    {
        return Ok(None);
    }
    let actual_digest = sha256_file(object_path)?;
    Ok(Some(CachedQ8Artifact {
        derived_digest: actual_digest,
        object_path: object_path.to_path_buf(),
        metadata_path: metadata_path.to_path_buf(),
        reused: true,
    }))
}

fn resolve_model_path(source_path: &Path) -> Result<PathBuf> {
    if source_path.is_file() {
        return Ok(source_path.to_path_buf());
    }
    let model_path = source_path.join("model.safetensors");
    ensure!(
        model_path.is_file(),
        "Q8 ingest requires model.safetensors under {}",
        source_path.display()
    );
    Ok(model_path)
}

fn tensor_order(family: Family, config: &Value, tensors: &SafeTensors<'_>) -> Result<Vec<String>> {
    let layers = config
        .get("num_hidden_layers")
        .and_then(Value::as_u64)
        .context("decode config is missing num_hidden_layers")? as usize;
    ensure!(layers > 0, "decode config has no layers");
    let tied = config
        .get("tie_word_embeddings")
        .and_then(Value::as_bool)
        .or_else(|| config.get("tie_embedding").and_then(Value::as_bool))
        .unwrap_or(true);
    let mut names = Vec::new();
    match family {
        Family::Qwen3_0_6b => {
            for index in 0..layers {
                let prefix = format!("layers.{index}");
                for suffix in [
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                    "self_attn.o_proj.weight",
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                    "mlp.down_proj.weight",
                ] {
                    names.push(format!("{prefix}.{suffix}"));
                }
            }
        }
        Family::Lfm2_1_2b => {
            let full_attention = config
                .get("full_attn_idxs")
                .and_then(Value::as_array)
                .map(|indices| {
                    indices
                        .iter()
                        .filter_map(Value::as_u64)
                        .map(|index| index as usize)
                        .collect::<BTreeSet<_>>()
                })
                .unwrap_or_default();
            let layer_types = config.get("layer_types").and_then(Value::as_array);
            for index in 0..layers {
                let prefix = format!("layers.{index}");
                for suffix in [
                    "feed_forward.w1.weight",
                    "feed_forward.w2.weight",
                    "feed_forward.w3.weight",
                ] {
                    names.push(format!("{prefix}.{suffix}"));
                }
                let attention = layer_types
                    .and_then(|types| types.get(index))
                    .and_then(Value::as_str)
                    .map(|kind| matches!(kind, "full_attention" | "attention"))
                    .unwrap_or_else(|| full_attention.contains(&index));
                if attention {
                    for suffix in [
                        "self_attn.q_proj.weight",
                        "self_attn.k_proj.weight",
                        "self_attn.v_proj.weight",
                        "self_attn.out_proj.weight",
                    ] {
                        names.push(format!("{prefix}.{suffix}"));
                    }
                } else {
                    names.push(format!("{prefix}.conv.in_proj.weight"));
                    names.push(format!("{prefix}.conv.out_proj.weight"));
                }
            }
        }
    }

    let head = if tied || lookup_tensor(tensors, "lm_head.weight").is_err() {
        "embed_tokens.weight"
    } else {
        "lm_head.weight"
    };
    names.push(head.to_string());
    Ok(names)
}

fn lookup_tensor<'data>(
    tensors: &SafeTensors<'data>,
    base_name: &str,
) -> Result<safetensors::tensor::TensorView<'data>> {
    let candidates = [
        base_name.to_string(),
        format!("bert.{base_name}"),
        format!("model.{base_name}"),
        format!("model.bert.{base_name}"),
        format!("lfm.{base_name}"),
        format!("model.lfm.{base_name}"),
    ];
    for candidate in &candidates {
        if let Ok(tensor) = tensors.tensor(candidate) {
            return Ok(tensor);
        }
    }
    bail!("missing Q8 tensor; tried {}", candidates.join(", "))
}

fn write_q8_tensor(
    writer: &mut impl Write,
    digest: &mut Sha256,
    name: &str,
    tensor: &safetensors::tensor::TensorView<'_>,
) -> Result<u64> {
    let element_bytes = match tensor.dtype() {
        Dtype::BF16 | Dtype::F16 => 2,
        Dtype::F32 => 4,
        dtype => bail!("Q8 tensor {name} has unsupported dtype {dtype:?}"),
    };
    let elements = tensor
        .shape()
        .iter()
        .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension))
        .context("Q8 tensor element count overflow")?;
    ensure!(
        tensor.data().len() == elements * element_bytes,
        "Q8 tensor {name} byte length does not match its shape"
    );
    ensure!(
        elements % Q8_BLOCK_ELEMENTS == 0,
        "Q8 tensor {name} does not contain complete blocks"
    );

    let mut values = [0.0_f32; Q8_BLOCK_ELEMENTS];
    let mut block_bytes = [0_u8; Q8_BLOCK_BYTES];
    let mut written = 0_u64;
    for block in 0..elements / Q8_BLOCK_ELEMENTS {
        for (element, value) in values.iter_mut().enumerate() {
            let byte_offset = (block * Q8_BLOCK_ELEMENTS + element) * element_bytes;
            *value = match tensor.dtype() {
                Dtype::BF16 => bf16::from_bits(u16::from_le_bytes(
                    tensor.data()[byte_offset..byte_offset + 2]
                        .try_into()
                        .expect("BF16 element width"),
                ))
                .to_f32(),
                Dtype::F16 => f16::from_bits(u16::from_le_bytes(
                    tensor.data()[byte_offset..byte_offset + 2]
                        .try_into()
                        .expect("F16 element width"),
                ))
                .to_f32(),
                Dtype::F32 => f32::from_le_bytes(
                    tensor.data()[byte_offset..byte_offset + 4]
                        .try_into()
                        .expect("F32 element width"),
                ),
                _ => unreachable!("dtype checked above"),
            };
            ensure!(
                value.is_finite(),
                "Q8 tensor {name} contains non-finite weights"
            );
        }
        let maximum = values.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
        let scale = maximum / 127.0;
        block_bytes[..2].copy_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
        let inverse = if scale == 0.0 { 0.0 } else { scale.recip() };
        for (output, value) in block_bytes[2..].iter_mut().zip(values) {
            *output = (value * inverse).round().clamp(-127.0, 127.0) as i8 as u8;
        }
        writer.write_all(&block_bytes).context("write Q8 block")?;
        digest.update(block_bytes);
        written += Q8_BLOCK_BYTES as u64;
    }
    Ok(written)
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)
        .with_context(|| format!("open Q8 object for verification {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("read Q8 object {}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(hex::encode(digest.finalize()))
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, time::SystemTime};

    use safetensors::tensor::{serialize_to_file, TensorView};
    use serde_json::json;

    use super::*;

    fn temp_dir(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "synapse-q8-artifact-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn write_checkpoint(root: &Path, config: Value, names: &[String]) {
        fs::write(
            root.join("config.json"),
            serde_json::to_vec_pretty(&config).unwrap(),
        )
        .unwrap();
        let values = (0..names.len() * Q8_BLOCK_ELEMENTS)
            .map(|index| ((index % Q8_BLOCK_ELEMENTS) as f32 - 16.0) / 8.0)
            .collect::<Vec<_>>();
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let mut tensors = HashMap::new();
        for (index, name) in names.iter().enumerate() {
            let start = index * Q8_BLOCK_ELEMENTS * std::mem::size_of::<f32>();
            let end = start + Q8_BLOCK_ELEMENTS * std::mem::size_of::<f32>();
            tensors.insert(
                format!("model.{name}"),
                TensorView::new(Dtype::F32, vec![1, Q8_BLOCK_ELEMENTS], &bytes[start..end])
                    .unwrap(),
            );
        }
        serialize_to_file(&tensors, None, &root.join("model.safetensors")).unwrap();
    }

    fn qwen_names() -> Vec<String> {
        [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ]
        .into_iter()
        .map(|suffix| format!("layers.0.{suffix}"))
        .chain(["embed_tokens.weight".to_string()])
        .collect()
    }

    fn lfm2_names() -> Vec<String> {
        let mut names = Vec::new();
        for index in 0..2 {
            for suffix in [
                "feed_forward.w1.weight",
                "feed_forward.w2.weight",
                "feed_forward.w3.weight",
            ] {
                names.push(format!("layers.{index}.{suffix}"));
            }
            if index == 1 {
                for suffix in [
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                    "self_attn.out_proj.weight",
                ] {
                    names.push(format!("layers.{index}.{suffix}"));
                }
            } else {
                names.push(format!("layers.{index}.conv.in_proj.weight"));
                names.push(format!("layers.{index}.conv.out_proj.weight"));
            }
        }
        names.push("embed_tokens.weight".to_string());
        names
    }

    #[test]
    fn q8_ingest_derives_and_reuses_complete_blocks_for_both_families() {
        let root = temp_dir("families");
        let cache = root.join("cache");
        let qwen = root.join("qwen");
        let lfm2 = root.join("lfm2");
        fs::create_dir_all(&qwen).unwrap();
        fs::create_dir_all(&lfm2).unwrap();
        let qwen_names = qwen_names();
        let lfm2_names = lfm2_names();
        write_checkpoint(
            &qwen,
            json!({"num_hidden_layers": 1, "tie_word_embeddings": true}),
            &qwen_names,
        );
        write_checkpoint(
            &lfm2,
            json!({
                "num_hidden_layers": 2,
                "full_attn_idxs": [1],
                "tie_word_embeddings": true
            }),
            &lfm2_names,
        );

        for (family, source, source_digest, tensor_count) in [
            (
                Family::Qwen3_0_6b,
                qwen.as_path(),
                "source-qwen",
                qwen_names.len(),
            ),
            (
                Family::Lfm2_1_2b,
                lfm2.as_path(),
                "source-lfm2",
                lfm2_names.len(),
            ),
        ] {
            let first =
                derive_and_cache_q8_blocks(source, &cache, family, source_digest, "q8-ingest-v1")
                    .unwrap();
            assert!(!first.reused);
            assert_eq!(
                fs::metadata(&first.object_path).unwrap().len(),
                (tensor_count * Q8_BLOCK_BYTES) as u64
            );
            assert_eq!(
                sha256_file(&first.object_path).unwrap(),
                first.derived_digest
            );

            let reused =
                derive_and_cache_q8_blocks(source, &cache, family, source_digest, "q8-ingest-v1")
                    .unwrap();
            assert!(reused.reused);
            assert_eq!(reused.derived_digest, first.derived_digest);
            assert_eq!(reused.object_path, first.object_path);

            let rotated =
                derive_and_cache_q8_blocks(source, &cache, family, source_digest, "q8-ingest-v2")
                    .unwrap();
            assert_ne!(rotated.object_path, first.object_path);
        }
        fs::remove_dir_all(root).unwrap();
    }
}

#[cfg(all(test, target_os = "macos"))]
mod checkpoint_tests {
    use serde_json::json;

    use super::*;

    #[test]
    #[ignore = "requires the Qwen3 and LFM2 checkpoints in the Hugging Face cache"]
    fn checkpoint_q8_ingest_derives_both_production_families() {
        let cache_root = std::env::var_os("SYNAPSE_OWNED_DECODE_Q8_CACHE_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("target/owned-decode-q8-checkpoint-test"));
        for (family, env_name, expected_digest) in [
            (
                Family::Qwen3_0_6b,
                "SYNAPSE_UNIFIED_RT_QWEN3_0_6B",
                "17d2fbfeff90269190287f324ed93bab3bb1b4fa4aad98c3fbba1868c01cb0f2",
            ),
            (
                Family::Lfm2_1_2b,
                "SYNAPSE_UNIFIED_RT_LFM2_1_2B",
                "5874faabdce2567dcc0e7339e9547d79421ba312c71e3442c9cc3c4ed3cb47d0",
            ),
        ] {
            let source = PathBuf::from(
                std::env::var_os(env_name)
                    .unwrap_or_else(|| panic!("set {env_name} to a checkpoint snapshot")),
            );
            let source_digest = sha256_file(&source.join("model.safetensors")).unwrap();
            let artifact = derive_and_cache_q8_blocks(
                &source,
                &cache_root,
                family,
                &source_digest,
                "q8-ingest-v1",
            )
            .unwrap();
            assert_eq!(artifact.derived_digest, expected_digest);
            println!(
                "{}",
                serde_json::to_string(&json!({
                    "family": family.as_str(),
                    "source_digest": source_digest,
                    "derived_digest": artifact.derived_digest,
                    "object_path": artifact.object_path,
                    "reused": artifact.reused,
                }))
                .unwrap()
            );
        }
    }
}
