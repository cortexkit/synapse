#![cfg_attr(
    not(all(feature = "cuda", not(target_os = "macos"))),
    forbid(unsafe_code)
)]

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use synapse_core::{
    EmbedEngine, EngineError, EngineErrorStage, EngineIdentity, EngineRiskClass, LoadedModel,
    RuntimeConfig, TokenBatch, TokenIds, ValidatedArtifact, Vector, Vectors,
};

mod cuda;
mod model;

pub const ENGINE_VERSION: &str = "owned-cuda-v1";
/// The source revision from which the CUDA kernels were ported.
pub const KERNEL_REVISION: &str = "4d0ded67c30286fe2be37cc7413359ad745dd751";
pub const PTX_VIRTUAL_ARCH: &str = "compute_75";
pub const MINIMUM_DEVICE_CC: &str = "7.5";
pub const MINIMUM_CUDA_DRIVER_API: u32 = 12_040;
pub const RISK_CLASS: EngineRiskClass = EngineRiskClass::AbortCapable;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct PortSourceProvenance {
    pub production_path: &'static str,
    pub spike_path: &'static str,
    pub source_revision: &'static str,
    pub source_digest: &'static str,
    pub production_digest: &'static str,
    /// `byte-identical` means the kernel has no production difference.
    pub reviewed_difference: &'static str,
}

#[must_use]
pub const fn port_provenance() -> [PortSourceProvenance; 5] {
    [
        PortSourceProvenance {
            production_path: "src/port/cuda_family_common.cuh",
            spike_path: "bench/spikes/unified-rt/src/cuda_family_common.cuh",
            source_revision: KERNEL_REVISION,
            source_digest: "937a01e329822277fa15eeac8498fcca6890b40665b531adb9c92f67b4eb53c5",
            production_digest: "937a01e329822277fa15eeac8498fcca6890b40665b531adb9c92f67b4eb53c5",
            reviewed_difference: "byte-identical",
        },
        PortSourceProvenance {
            production_path: "src/port/cuda_minilm.h",
            spike_path: "bench/spikes/unified-rt/src/cuda_minilm.h",
            source_revision: KERNEL_REVISION,
            source_digest: "21a0bcee2d21b17fea795807eab2070e848c416df1cb94c7ae7bb9adfa514de5",
            production_digest: "21a0bcee2d21b17fea795807eab2070e848c416df1cb94c7ae7bb9adfa514de5",
            reviewed_difference: "byte-identical",
        },
        PortSourceProvenance {
            production_path: "src/port/cuda_minilm.cu",
            spike_path: "bench/spikes/unified-rt/src/cuda_minilm.cu",
            source_revision: KERNEL_REVISION,
            source_digest: "9848c050015d7f2ecd0ecc6aa628bb4e4d4280778e4a7eb6590475e937f30375",
            production_digest: "9848c050015d7f2ecd0ecc6aa628bb4e4d4280778e4a7eb6590475e937f30375",
            reviewed_difference: "byte-identical",
        },
        PortSourceProvenance {
            production_path: "src/port/cuda_modernbert.cu",
            spike_path: "bench/spikes/unified-rt/src/cuda_modernbert.cu",
            source_revision: KERNEL_REVISION,
            source_digest: "455e4419c0c6996b004153e1709e9d6ef001ca0d865e42bc7af2ad6809b350b9",
            production_digest: "455e4419c0c6996b004153e1709e9d6ef001ca0d865e42bc7af2ad6809b350b9",
            reviewed_difference: "byte-identical",
        },
        PortSourceProvenance {
            production_path: "src/port/cuda_qwen3.cu",
            spike_path: "bench/spikes/unified-rt/src/cuda_qwen3.cu",
            source_revision: KERNEL_REVISION,
            source_digest: "a4c05d7cf119b39ffe52cdfa77f39859ddee8ac3e18139d8094219fbc8a1588b",
            production_digest: "a4c05d7cf119b39ffe52cdfa77f39859ddee8ac3e18139d8094219fbc8a1588b",
            reviewed_difference: "byte-identical",
        },
    ]
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelFamily {
    MiniLm,
    GteModernBert,
    Qwen3,
}

impl ModelFamily {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MiniLm => "minilm",
            Self::GteModernBert => "gte-modernbert",
            Self::Qwen3 => "qwen3-0.6b",
        }
    }

    pub fn parse(value: &str) -> Result<Self, CudaEngineError> {
        match value.trim().to_ascii_lowercase().as_str() {
            "minilm" | "all-minilm-l6-v2" => Ok(Self::MiniLm),
            "gte-modernbert" | "modernbert" => Ok(Self::GteModernBert),
            "qwen3" | "qwen3-0.6b" | "qwen3-embedding-0.6b" => Ok(Self::Qwen3),
            other => Err(CudaEngineError::UnsupportedFamily(other.to_owned())),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StorageDType {
    F16,
    F32,
}

impl StorageDType {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::F32 => "f32",
        }
    }

    pub fn parse(value: &str) -> Result<Self, CudaEngineError> {
        match value.trim().to_ascii_lowercase().as_str() {
            "f16" | "fp16" => Ok(Self::F16),
            "f32" | "fp32" => Ok(Self::F32),
            other => Err(CudaEngineError::UnsupportedDType(other.to_owned())),
        }
    }
}

/// Resolved serving dtypes used by the v1 C1-C3 manifests.
#[must_use]
pub const fn resolved_storage_dtype(family: ModelFamily) -> StorageDType {
    match family {
        ModelFamily::MiniLm | ModelFamily::GteModernBert | ModelFamily::Qwen3 => StorageDType::F16,
    }
}

pub const C1_STORAGE_DTYPE: StorageDType = StorageDType::F16;
pub const C2_STORAGE_DTYPE: StorageDType = StorageDType::F16;
pub const C3_STORAGE_DTYPE: StorageDType = StorageDType::F16;

#[must_use]
pub fn resolved_cell_id(family: ModelFamily) -> String {
    format!(
        "owned-cuda/{}/{}",
        family.as_str(),
        resolved_storage_dtype(family).as_str()
    )
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CudaBuildIdentity {
    pub backend: String,
    pub family: String,
    pub storage_dtype: String,
    pub kernel_revision: String,
    pub ptx_virtual_arch: String,
    pub minimum_device_cc: String,
    pub minimum_cuda_driver_api: u32,
    pub risk_class: EngineRiskClass,
}

#[must_use]
pub fn build_identity(family: ModelFamily, dtype: StorageDType) -> CudaBuildIdentity {
    CudaBuildIdentity {
        backend: "cuda-ptx".to_owned(),
        family: family.as_str().to_owned(),
        storage_dtype: dtype.as_str().to_owned(),
        kernel_revision: KERNEL_REVISION.to_owned(),
        ptx_virtual_arch: PTX_VIRTUAL_ARCH.to_owned(),
        minimum_device_cc: MINIMUM_DEVICE_CC.to_owned(),
        minimum_cuda_driver_api: MINIMUM_CUDA_DRIVER_API,
        risk_class: RISK_CLASS,
    }
}

/// Hardware-floor predicate used by capability probes before worker creation.
#[must_use]
pub fn device_meets_floor(driver_api: u32, compute_major: u32, compute_minor: u32) -> bool {
    driver_api >= MINIMUM_CUDA_DRIVER_API
        && (compute_major > 7 || (compute_major == 7 && compute_minor >= 5))
}

#[derive(Debug, thiserror::Error)]
pub enum CudaEngineError {
    #[error("unsupported owned-cuda family '{0}'")]
    UnsupportedFamily(String),
    #[error("unsupported owned-cuda storage dtype '{0}'")]
    UnsupportedDType(String),
    #[error("owned-cuda is unavailable in this build")]
    Unavailable,
    #[error("owned-cuda model package: {0}")]
    InvalidPackage(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenizerPolicy {
    pub pad_token_id: u32,
    pub terminal_token_id: Option<u32>,
}

enum LoadedFamily {
    MiniLm {
        model: model::MiniLmModel,
        context: cuda::MiniLmContext,
    },
    GteModernBert {
        model: model::ModernBertModel,
        context: cuda::ModernBertContext,
    },
    Qwen3 {
        model: model::Qwen3Model,
        context: cuda::Qwen3Context,
    },
}

pub struct OwnedCudaEmbedEngine {
    family: ModelFamily,
    dtype: StorageDType,
    graphs: bool,
    models: HashMap<String, Arc<Mutex<LoadedFamily>>>,
    next_model: u64,
}

impl OwnedCudaEmbedEngine {
    #[must_use]
    pub fn new(family: ModelFamily, dtype: StorageDType) -> Self {
        Self {
            family,
            dtype,
            graphs: true,
            models: HashMap::new(),
            next_model: 0,
        }
    }

    #[must_use]
    pub fn serving(family: ModelFamily) -> Self {
        Self::new(family, resolved_storage_dtype(family))
    }

    #[must_use]
    pub const fn family(&self) -> ModelFamily {
        self.family
    }

    #[must_use]
    pub const fn storage_dtype(&self) -> StorageDType {
        self.dtype
    }

    pub fn set_graphs(&mut self, enabled: bool) {
        self.graphs = enabled;
    }

    #[must_use]
    pub fn tokenizer_policy(&self, model: &LoadedModel) -> Option<TokenizerPolicy> {
        self.models.get(&model.model_id).and_then(|entry| {
            let entry = entry.lock().ok()?;
            Some(match &*entry {
                LoadedFamily::MiniLm { model, .. } => TokenizerPolicy {
                    pad_token_id: model.pad_token_id(),
                    terminal_token_id: None,
                },
                LoadedFamily::GteModernBert { model, .. } => TokenizerPolicy {
                    pad_token_id: model.pad_token_id,
                    terminal_token_id: None,
                },
                LoadedFamily::Qwen3 { model, .. } => TokenizerPolicy {
                    pad_token_id: 0,
                    terminal_token_id: Some(model.eos_token_id()),
                },
            })
        })
    }

    fn error(stage: EngineErrorStage, message: impl Into<String>) -> EngineError {
        EngineError {
            stage,
            risk_class: RISK_CLASS,
            message: message.into(),
            retry_after_ms: None,
            safe_to_retry_same_request: matches!(stage, EngineErrorStage::Inference),
        }
    }

    fn model_path(cfg: &RuntimeConfig) -> Result<PathBuf, EngineError> {
        cfg.values
            .get("model_path")
            .or_else(|| cfg.values.get("artifact_path"))
            .filter(|path| !path.is_empty())
            .map(PathBuf::from)
            .ok_or_else(|| {
                Self::error(
                    EngineErrorStage::Load,
                    "owned-cuda requires model_path or artifact_path",
                )
            })
    }

    fn parse_graphs(cfg: &RuntimeConfig) -> Result<bool, EngineError> {
        cfg.values
            .get("cuda_graphs")
            .map_or(Ok(true), |value| match value.as_str() {
                "1" | "true" => Ok(true),
                "0" | "false" => Ok(false),
                other => Err(Self::error(
                    EngineErrorStage::Load,
                    format!("invalid cuda_graphs '{other}'"),
                )),
            })
    }

    fn load_cuda(&mut self, path: &Path, cfg: &RuntimeConfig) -> Result<LoadedModel, EngineError> {
        cuda::ensure_available()
            .map_err(|error| Self::error(EngineErrorStage::Load, error.to_string()))?;
        let detected = detect_family(path)
            .map_err(|error| Self::error(EngineErrorStage::Load, error.to_string()))?;
        if detected != self.family {
            return Err(Self::error(
                EngineErrorStage::Load,
                format!(
                    "configured family {} does not match detected family {}",
                    self.family.as_str(),
                    detected.as_str()
                ),
            ));
        }
        self.graphs = Self::parse_graphs(cfg)?;
        let loaded = match self.family {
            ModelFamily::MiniLm => LoadedFamily::MiniLm {
                model: model::MiniLmModel::load(path)
                    .map_err(|error| Self::error(EngineErrorStage::Load, error.to_string()))?,
                context: cuda::MiniLmContext::new(self.graphs)
                    .map_err(|error| Self::error(EngineErrorStage::Load, error.to_string()))?,
            },
            ModelFamily::GteModernBert => LoadedFamily::GteModernBert {
                model: model::ModernBertModel::load(path)
                    .map_err(|error| Self::error(EngineErrorStage::Load, error.to_string()))?,
                context: cuda::ModernBertContext::new(self.graphs, precision(self.dtype))
                    .map_err(|error| Self::error(EngineErrorStage::Load, error.to_string()))?,
            },
            ModelFamily::Qwen3 => {
                if self.dtype != StorageDType::F16 {
                    return Err(Self::error(
                        EngineErrorStage::Load,
                        "Qwen3 owned CUDA requires f16 storage",
                    ));
                }
                LoadedFamily::Qwen3 {
                    model: model::Qwen3Model::load(path)
                        .map_err(|error| Self::error(EngineErrorStage::Load, error.to_string()))?,
                    context: cuda::Qwen3Context::new(self.graphs, precision(self.dtype))
                        .map_err(|error| Self::error(EngineErrorStage::Load, error.to_string()))?,
                }
            }
        };
        let model_id = format!("owned-cuda:{}:{}", self.family.as_str(), self.next_model);
        self.next_model += 1;
        self.models
            .insert(model_id.clone(), Arc::new(Mutex::new(loaded)));
        Ok(LoadedModel { model_id })
    }
}

impl Default for OwnedCudaEmbedEngine {
    fn default() -> Self {
        Self::serving(ModelFamily::MiniLm)
    }
}

impl EmbedEngine for OwnedCudaEmbedEngine {
    fn identity(&self) -> EngineIdentity {
        let build = build_identity(self.family, self.dtype);
        let mut build_flags = BTreeMap::new();
        build_flags.insert("backend".to_owned(), build.backend);
        build_flags.insert("family".to_owned(), build.family);
        build_flags.insert("storage_dtype".to_owned(), build.storage_dtype);
        build_flags.insert("kernel_revision".to_owned(), build.kernel_revision);
        build_flags.insert("ptx_virtual_arch".to_owned(), build.ptx_virtual_arch);
        build_flags.insert("minimum_device_cc".to_owned(), build.minimum_device_cc);
        build_flags.insert(
            "minimum_cuda_driver_api".to_owned(),
            build.minimum_cuda_driver_api.to_string(),
        );
        build_flags.insert("risk_class".to_owned(), "abort_capable".to_owned());
        EngineIdentity {
            engine: "owned-cuda".to_owned(),
            version: ENGINE_VERSION.to_owned(),
            build_flags,
        }
    }

    fn load(
        &mut self,
        artifact: &ValidatedArtifact,
        cfg: &RuntimeConfig,
    ) -> Result<LoadedModel, EngineError> {
        if artifact.format != "safetensors-package" {
            return Err(Self::error(
                EngineErrorStage::Load,
                format!(
                    "owned-cuda requires safetensors-package, got {}",
                    artifact.format
                ),
            ));
        }
        let path = Self::model_path(cfg)?;
        if !artifact.digest.trim().is_empty() && path.is_file() {
            verify_digest(&path, &artifact.digest)?;
        }
        self.load_cuda(&path, cfg)
    }

    fn embed_batch(
        &self,
        model_ref: &LoadedModel,
        batch: TokenBatch,
    ) -> Result<Vectors, EngineError> {
        if batch.items.is_empty() {
            return Ok(Vec::new());
        }
        let entry = self.models.get(&model_ref.model_id).ok_or_else(|| {
            Self::error(
                EngineErrorStage::Inference,
                format!("unknown owned-cuda model ref '{}'", model_ref.model_id),
            )
        })?;
        let mut entry = entry.lock().map_err(|_| {
            Self::error(
                EngineErrorStage::Inference,
                "owned-cuda model mutex was poisoned",
            )
        })?;
        match &mut *entry {
            LoadedFamily::MiniLm { model, context } => model.embed(context, &batch.items),
            LoadedFamily::GteModernBert { model, context } => {
                model.embed(context, &batch.items, precision(self.dtype))
            }
            LoadedFamily::Qwen3 { model, context } => model.embed(context, &batch.items),
        }
        .map_err(|error| Self::error(EngineErrorStage::Inference, error.to_string()))
    }

    fn embed_one(&self, model: &LoadedModel, ids: TokenIds) -> Result<Vector, EngineError> {
        let mut vectors = self.embed_batch(model, TokenBatch { items: vec![ids] })?;
        vectors.pop().ok_or_else(|| {
            Self::error(EngineErrorStage::Inference, "owned-cuda returned no vector")
        })
    }

    fn unload(&mut self, model: &LoadedModel) {
        self.models.remove(&model.model_id);
    }
}

pub(crate) fn encode_f16_bits(values: &[f32]) -> Vec<u16> {
    values
        .iter()
        .map(|&value| half::f16::from_f32(value).to_bits())
        .collect()
}

fn precision(dtype: StorageDType) -> Precision {
    match dtype {
        StorageDType::F16 => Precision::F16,
        StorageDType::F32 => Precision::F32,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Precision {
    F16,
    F32,
}

impl Precision {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::F32 => "f32",
        }
    }
}

pub fn detect_family(model_path: impl AsRef<Path>) -> Result<ModelFamily, CudaEngineError> {
    let path = model_path.as_ref();
    let root = model::resolve_model_root(path)
        .map_err(|error| CudaEngineError::InvalidPackage(error.to_string()))?;
    let bytes = std::fs::read(root.join("config.json"))
        .map_err(|error| CudaEngineError::InvalidPackage(error.to_string()))?;
    let config: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| CudaEngineError::InvalidPackage(error.to_string()))?;
    let model_type = config
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    if model_type == "bert" {
        Ok(ModelFamily::MiniLm)
    } else if model_type == "modernbert" {
        Ok(ModelFamily::GteModernBert)
    } else if model_type.starts_with("qwen3")
        || config
            .get("architectures")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|items| {
                items.iter().any(|item| {
                    item.as_str()
                        .is_some_and(|name| name.to_ascii_lowercase().contains("qwen3"))
                })
            })
    {
        Ok(ModelFamily::Qwen3)
    } else {
        Err(CudaEngineError::UnsupportedFamily(model_type.to_owned()))
    }
}

fn verify_digest(path: &Path, expected: &str) -> Result<(), EngineError> {
    let expected = expected.strip_prefix("sha256:").unwrap_or(expected);
    let bytes = std::fs::read(path)
        .map_err(|error| OwnedCudaEmbedEngine::error(EngineErrorStage::Load, error.to_string()))?;
    let actual = format!("{:x}", Sha256::digest(bytes));
    if actual == expected {
        Ok(())
    } else {
        Err(OwnedCudaEmbedEngine::error(
            EngineErrorStage::Load,
            format!("artifact digest mismatch: expected {expected}, got {actual}"),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_exposes_ptx_and_resolved_dtype() {
        let identity = build_identity(ModelFamily::Qwen3, C3_STORAGE_DTYPE);
        assert_eq!(identity.backend, "cuda-ptx");
        assert_eq!(identity.storage_dtype, "f16");
        assert_eq!(identity.ptx_virtual_arch, "compute_75");
        assert_eq!(identity.minimum_device_cc, "7.5");
        assert_eq!(identity.minimum_cuda_driver_api, 12_040);
    }

    #[test]
    fn floor_includes_driver_and_compute_capability_boundaries() {
        assert!(!device_meets_floor(12_039, 7, 5));
        assert!(!device_meets_floor(12_040, 7, 4));
        assert!(device_meets_floor(12_040, 7, 5));
        assert!(device_meets_floor(12_040, 8, 0));
    }

    #[test]
    fn engine_identity_separates_family_and_dtype() {
        let minilm = OwnedCudaEmbedEngine::serving(ModelFamily::MiniLm).identity();
        let qwen = OwnedCudaEmbedEngine::serving(ModelFamily::Qwen3).identity();
        assert_ne!(
            minilm.build_flags.get("family"),
            qwen.build_flags.get("family")
        );
        assert_eq!(minilm.engine, "owned-cuda");
    }

    #[cfg(not(all(feature = "cuda", not(target_os = "macos"))))]
    #[test]
    fn default_disabled_stub_refuses_load() {
        let mut engine = OwnedCudaEmbedEngine::default();
        let result = engine.load(
            &ValidatedArtifact {
                digest: String::new(),
                format: "safetensors-package".to_owned(),
            },
            &RuntimeConfig::default(),
        );
        assert!(result.is_err());
    }
}
