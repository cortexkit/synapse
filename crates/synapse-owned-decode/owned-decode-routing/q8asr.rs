//! Backend-independent ASR artifacts and the mandatory verified load boundary.

use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use anyhow::{ensure, Context, Result};
use half::{bf16, f16};
use safetensors::tensor::Dtype;
use serde::{Deserialize, Serialize};

use super::{
    q8artifact::{self, CachedQ8Artifact, QuantizerFamily},
    q8ingest::Q8IngestRegistry,
    tensor_source::TensorSource,
};
use crate::owned_decode_routing::error::OwnedDecodeError;

pub const QUANTIZER_REVISION: &str = "q8-ingest-v1";
pub const F16_PROFILE: &str = "qwen3-asr-1.7b-q8_0-f16-v1";

/// Pins shared by every ASR backend, stored on the source model in models.json.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AsrArtifactPins {
    pub quantizer_revision: String,
    pub q8_0_digest: String,
    pub f16_digest: String,
}

/// Paths may reach an engine only after both files have passed the pinned hashes.
#[derive(Debug)]
pub struct VerifiedAsrArtifacts {
    pub q8: CachedQ8Artifact,
    pub f16_path: PathBuf,
}

/// Derive missing files and hash both files on *every* load, cold or cached.
///
/// ASR engine and catalog callers must use this boundary so both artifact files
/// are verified before their paths are returned. Unlike `load_or_ingest` alone,
/// registry trust is not treated as evidence about the current file bytes.
/// No path is returned on a mismatch; the typed cause is `ArtifactPoisoned`.
/// Callers must propagate that error rather than treating it as a decode fallback.
pub fn load_verified_asr_artifacts(
    source: &Path,
    cache: &Path,
    registry: &mut Q8IngestRegistry,
) -> Result<VerifiedAsrArtifacts> {
    let manifest: serde_json::Value =
        serde_json::from_str(include_str!("../../../bench/parity/models.json"))?;
    let model = &manifest["models"]["qwen3-asr-1.7b"];
    let pins: AsrArtifactPins = serde_json::from_value(model["derived_artifacts"].clone())?;
    let source_digest = model["checkpoint_digest"]
        .as_str()
        .context("missing ASR checkpoint digest")?;
    load_with_pins(source, cache, registry, source_digest, &pins)
}

fn load_with_pins(
    source: &Path,
    cache: &Path,
    registry: &mut Q8IngestRegistry,
    source_digest: &str,
    pins: &AsrArtifactPins,
) -> Result<VerifiedAsrArtifacts> {
    let q8 = q8artifact::derive_q8(
        source,
        cache,
        QuantizerFamily::Qwen3Asr,
        source_digest,
        &pins.quantizer_revision,
    )?;
    let f16_path = derive_f16(source, &q8)?;
    // Check the bytes, not lineage or the registry's previously settled digest.
    let q8_digest = q8artifact::sha256_file(&q8.object_path)?;
    let f16_digest = q8artifact::sha256_file(&f16_path)?;
    let valid = q8_digest == bare(&pins.q8_0_digest) && f16_digest == bare(&pins.f16_digest);
    if !valid {
        registry.mark_corrupted(source_digest, &pins.quantizer_revision);
        return Err(OwnedDecodeError::ArtifactPoisoned.into());
    }
    registry.register_expected_digest(source_digest, &pins.quantizer_revision, &q8.derived_digest);
    registry.load_or_ingest(source_digest, &pins.quantizer_revision, "q8_0", &[], |_| {
        q8.derived_digest.clone()
    })?;
    Ok(VerifiedAsrArtifacts { q8, f16_path })
}

fn bare(digest: &str) -> &str {
    digest.strip_prefix("sha256:").unwrap_or(digest)
}

#[cfg(test)]
#[path = "q8asr_tests.rs"]
mod tests;

pub(super) fn derive_f16(source: &Path, q8: &CachedQ8Artifact) -> Result<PathBuf> {
    let output = q8.object_path.with_file_name("remaining.f16.safetensors");
    if output.is_file() {
        return Ok(output);
    }
    let tensors = TensorSource::open(source)?;
    let config = serde_json::from_slice(&fs::read(tensors.root.join("config.json"))?)?;
    let names = q8artifact::tensor_order(QuantizerFamily::Qwen3Asr, &config, &tensors)?;
    let quantized = names
        .iter()
        .map(|name| tensors.resolve(name))
        .collect::<Result<BTreeSet<_>>>()?;
    let remaining = tensors
        .tensors
        .iter()
        .filter(|(name, _)| {
            name.as_str() == "thinker.model.embed_tokens.weight"
                || !quantized.contains(name.as_str())
        })
        .collect::<Vec<_>>();
    // Match parity's conversion-rule-v1 package header exactly, without retaining
    // its all-tensors output buffer. Tensor data is appended one tensor at a time.
    let mut header =
        format!("{{\"__metadata__\":{{\"conversion_rule\":\"v1\",\"profile\":\"{F16_PROFILE}\"}}");
    let mut offset = 0_u64;
    for (name, tensor) in &remaining {
        let elements = tensor
            .shape
            .iter()
            .try_fold(1_u64, |n, d| n.checked_mul(*d as u64))
            .context("f16 shape overflow")?;
        let end = offset
            .checked_add(elements.checked_mul(2).context("f16 size overflow")?)
            .context("f16 offset overflow")?;
        header.push_str(&format!(
            ",{}:{{\"dtype\":\"F16\",\"shape\":{},\"data_offsets\":[{offset},{end}]}}",
            serde_json::to_string(name)?,
            serde_json::to_string(&tensor.shape)?
        ));
        offset = end;
    }
    header.push('}');
    while header.len() % 8 != 0 {
        header.push(' ');
    }
    let temp = output.with_extension(format!("{}.{}.tmp", std::process::id(), crate::now_ms()));
    let result = (|| -> Result<()> {
        let mut writer = BufWriter::new(File::create(&temp)?);
        writer.write_all(&(header.len() as u64).to_le_bytes())?;
        writer.write_all(header.as_bytes())?;
        for (name, _) in remaining {
            let (record, mut bytes) = tensors.read(name)?;
            let width = match record.dtype {
                Dtype::BF16 | Dtype::F16 => 2,
                Dtype::F32 => 4,
                dtype => anyhow::bail!("unsupported f16 tensor {name}: {dtype:?}"),
            };
            let elements = bytes.len() / width;
            for index in 0..elements {
                let input = &bytes[index * width..(index + 1) * width];
                let value = match record.dtype {
                    Dtype::BF16 => bf16::from_bits(u16::from_le_bytes(input.try_into()?)).to_f32(),
                    Dtype::F16 => f16::from_bits(u16::from_le_bytes(input.try_into()?)).to_f32(),
                    Dtype::F32 => f32::from_le_bytes(input.try_into()?),
                    _ => unreachable!(),
                };
                let rounded = f16::from_f32(value);
                ensure!(
                    value.is_finite() && rounded.is_finite(),
                    "non-finite f16 weight in {name}"
                );
                bytes[index * 2..index * 2 + 2].copy_from_slice(&rounded.to_bits().to_le_bytes());
            }
            writer.write_all(&bytes[..elements * 2])?;
        }
        writer.flush()?;
        writer.get_ref().sync_all()?;
        fs::rename(&temp, &output)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result?;
    Ok(output)
}
