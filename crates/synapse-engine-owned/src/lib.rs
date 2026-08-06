#![cfg_attr(not(target_os = "macos"), forbid(unsafe_code))]

use std::collections::{BTreeMap, HashMap};
#[cfg(target_os = "macos")]
use std::env;
use std::path::Path;
#[cfg(target_os = "macos")]
use std::path::PathBuf;
#[cfg(target_os = "macos")]
use std::sync::{Arc, Mutex};
#[cfg(target_os = "macos")]
use std::time::Instant;

use serde::{Deserialize, Serialize};
use synapse_core::{
    EmbedEngine, EngineError, EngineErrorStage, EngineIdentity, EngineRiskClass, LoadedModel,
    RerankEngine, RerankRequest, RerankScores, RuntimeConfig, TokenBatch, TokenIds,
    ValidatedArtifact, Vector, Vectors,
};

#[cfg(target_os = "macos")]
mod runtime;

/// Production-owned Metal decode engines for Qwen3-0.6B and LFM2-1.2B.
/// Ported from `bench/spikes/unified-rt/` into production-owned code.
/// See `owned_decode_engine` module docs for scope and byte-identity contract.
#[cfg(target_os = "macos")]
#[path = "../owned-decode-engine/src/lib.rs"]
pub mod owned_decode_engine;

/// Precision enum re-exported for decode engine consumers.
#[cfg(target_os = "macos")]
pub use runtime::Precision;

pub const ENGINE_VERSION: &str = "owned-metal-v1";
// Bump whenever a compiled MPSGraph changes structure (ops added, removed, or reordered).
// The revision is part of both the explicit-executable package cache key and the engine
// identity, so raising it invalidates stale cached executables that still encode the old
// graph and moves the provenance fingerprint to match the new one.
pub const GRAPH_REVISION: u32 = 4;
pub const BUCKET_POLICY_VERSION: u32 = 1;
pub const DEFAULT_ATTENTION_UNITS: usize = 4_000_000;
#[cfg(target_os = "macos")]
const EMBED_PROFILE_ENV: &str = "SYNAPSE_EMBED_PROFILE";

#[cfg(target_os = "macos")]
pub(crate) fn embed_profile_enabled() -> bool {
    env::var_os(EMBED_PROFILE_ENV).is_some_and(|value| value.to_string_lossy() != "0")
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

    pub fn parse(value: &str) -> Result<Self, OwnedEngineError> {
        match value.trim().to_ascii_lowercase().as_str() {
            "minilm" | "all-minilm-l6-v2" => Ok(Self::MiniLm),
            "gte-modernbert" | "modernbert" => Ok(Self::GteModernBert),
            "qwen3" | "qwen3-0.6b" | "qwen3-embedding-0.6b" => Ok(Self::Qwen3),
            other => Err(OwnedEngineError::UnsupportedFamily(other.to_string())),
        }
    }

    #[must_use]
    pub const fn recommended_dtype(self) -> OwnedDType {
        match self {
            Self::MiniLm | Self::Qwen3 => OwnedDType::F16,
            Self::GteModernBert => OwnedDType::F32,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OwnedDType {
    F16,
    F32,
}

impl OwnedDType {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::F32 => "f32",
        }
    }

    pub fn parse(value: &str) -> Result<Self, OwnedEngineError> {
        match value.trim().to_ascii_lowercase().as_str() {
            "f16" | "fp16" => Ok(Self::F16),
            "f32" | "fp32" => Ok(Self::F32),
            other => Err(OwnedEngineError::UnsupportedDType(other.to_string())),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenizerPolicy {
    pub add_special_tokens: bool,
    pub pad_token_id: u32,
    pub terminal_token_id: Option<u32>,
}

#[derive(Debug, thiserror::Error)]
pub enum OwnedEngineError {
    #[error("unsupported owned-metal family '{0}'")]
    UnsupportedFamily(String),
    #[error("unsupported owned-metal dtype '{0}'")]
    UnsupportedDType(String),
    #[error("owned-metal is available only on macOS")]
    UnsupportedPlatform,
    #[error("owned-metal model package: {0}")]
    InvalidPackage(String),
}

pub fn detect_family(model_dir: impl AsRef<Path>) -> Result<ModelFamily, OwnedEngineError> {
    let config = read_model_config(model_dir.as_ref())?;
    let model_type = config
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    if model_type == "bert" {
        Ok(ModelFamily::MiniLm)
    } else if model_type == "modernbert" {
        Ok(ModelFamily::GteModernBert)
    } else if model_type.starts_with("qwen3") {
        Ok(ModelFamily::Qwen3)
    } else {
        Err(OwnedEngineError::UnsupportedFamily(model_type.to_string()))
    }
}

fn read_model_config(model_dir: &Path) -> Result<serde_json::Value, OwnedEngineError> {
    let root = if model_dir.is_dir() {
        model_dir
    } else {
        model_dir.parent().unwrap_or_else(|| Path::new("."))
    };
    let config_path = root.join("config.json");
    let bytes = std::fs::read(&config_path).map_err(|error| {
        OwnedEngineError::InvalidPackage(format!("read {}: {error}", config_path.display()))
    })?;
    serde_json::from_slice(&bytes).map_err(|error| {
        OwnedEngineError::InvalidPackage(format!("parse {}: {error}", config_path.display()))
    })
}

#[must_use]
pub fn engine_identity(family: ModelFamily, dtype: OwnedDType) -> EngineIdentity {
    let mut build_flags = BTreeMap::new();
    build_flags.insert("backend".to_string(), "metal-mpsgraph".to_string());
    build_flags.insert("family".to_string(), family.as_str().to_string());
    build_flags.insert("dtype".to_string(), dtype.as_str().to_string());
    build_flags.insert("graph_revision".to_string(), GRAPH_REVISION.to_string());
    build_flags.insert(
        "bucket_policy".to_string(),
        format!("v{BUCKET_POLICY_VERSION}"),
    );
    build_flags.insert("risk_class".to_string(), "abort_safe".to_string());
    EngineIdentity {
        engine: "owned-metal".to_string(),
        version: ENGINE_VERSION.to_string(),
        build_flags,
    }
}

pub struct OwnedMetalEmbedEngine {
    family: ModelFamily,
    dtype: OwnedDType,
    models: HashMap<String, OwnedModelHandle>,
    // Only the macOS runtime mints model handles; other targets keep the field
    // for struct-shape parity but never read it.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    next_model: u64,
}

#[cfg(target_os = "macos")]
type OwnedModelHandle = Arc<Mutex<OwnedLoadedModel>>;
#[cfg(not(target_os = "macos"))]
type OwnedModelHandle = ();

#[cfg(target_os = "macos")]
struct OwnedLoadedModel {
    family: Box<dyn runtime::ModelFamily>,
    provider: runtime::MetalProvider,
    buckets: Vec<runtime::BatchShape>,
    tokenizer_policy: TokenizerPolicy,
}

impl OwnedMetalEmbedEngine {
    #[must_use]
    pub fn new(family: ModelFamily, dtype: OwnedDType) -> Self {
        Self {
            family,
            dtype,
            models: HashMap::new(),
            next_model: 0,
        }
    }

    #[must_use]
    pub const fn family(&self) -> ModelFamily {
        self.family
    }

    #[must_use]
    pub const fn dtype(&self) -> OwnedDType {
        self.dtype
    }

    fn error(stage: EngineErrorStage, message: impl Into<String>) -> EngineError {
        EngineError {
            stage,
            risk_class: EngineRiskClass::AbortSafe,
            message: message.into(),
            retry_after_ms: None,
            safe_to_retry_same_request: false,
        }
    }

    #[cfg(target_os = "macos")]
    fn load_macos(&mut self, cfg: &RuntimeConfig) -> Result<LoadedModel, EngineError> {
        let model_path = required_path(cfg, "model_path")?;
        let detected = detect_family(&model_path)
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
        let max_length = parse_usize(cfg, "max_tokens", 512)?;
        let attention_units = parse_usize(cfg, "attention_units", DEFAULT_ATTENTION_UNITS)?;
        if max_length == 0 || attention_units < max_length.saturating_mul(max_length) {
            return Err(Self::error(
                EngineErrorStage::Load,
                "bucket attention budget cannot cover max_tokens",
            ));
        }
        let cache_root = required_path(cfg, "package_cache_root")?;
        let execution = match cfg
            .values
            .get("execution")
            .map(String::as_str)
            .unwrap_or("explicit")
        {
            "explicit" => runtime::Execution::Explicit,
            "lazy" => runtime::Execution::Lazy,
            other => {
                return Err(Self::error(
                    EngineErrorStage::Load,
                    format!("unsupported Metal execution mode '{other}'"),
                ))
            }
        };
        let package_root = package_root(&cache_root, &model_path, self.family, self.dtype)
            .map_err(|error| Self::error(EngineErrorStage::Load, error))?;
        let config = runtime::MetalExecutionConfig::new(execution, Some(package_root))
            .map_err(|error| Self::error(EngineErrorStage::Load, error.to_string()))?;
        let family = runtime::load_model_family(&model_path, precision(self.dtype))
            .map_err(|error| Self::error(EngineErrorStage::Load, error.to_string()))?;
        let policy = family.tokenizer_policy();
        let tokenizer_policy = TokenizerPolicy {
            add_special_tokens: true,
            pad_token_id: policy.pad_token_id,
            terminal_token_id: policy.terminal_token_id,
        };
        let buckets = runtime::bucket_shapes(max_length, attention_units);
        if buckets.len() > 12 {
            return Err(Self::error(
                EngineErrorStage::Load,
                format!("bucket policy produced {} shapes", buckets.len()),
            ));
        }
        let mut provider =
            runtime::MetalProvider::new_with_config(precision(self.dtype), config)
                .map_err(|error| Self::error(EngineErrorStage::Load, error.to_string()))?;
        let preload_ids = vec![vec![policy
            .terminal_token_id
            .unwrap_or(policy.pad_token_id)]];
        for &shape in &buckets {
            family
                .embed_batch(&mut provider, &preload_ids, Some(shape))
                .map_err(|error| {
                    Self::error(
                        EngineErrorStage::Load,
                        format!("precompile {}x{}: {error}", shape.batch, shape.seq),
                    )
                })?;
        }
        let model_id = format!("owned-metal:{}:{}", self.family.as_str(), self.next_model);
        self.next_model += 1;
        self.models.insert(
            model_id.clone(),
            Arc::new(Mutex::new(OwnedLoadedModel {
                family,
                provider,
                buckets,
                tokenizer_policy,
            })),
        );
        Ok(LoadedModel { model_id })
    }

    #[cfg(target_os = "macos")]
    pub fn tokenizer_policy(&self, model: &LoadedModel) -> Result<TokenizerPolicy, EngineError> {
        let loaded = self.models.get(&model.model_id).ok_or_else(|| {
            Self::error(
                EngineErrorStage::Inference,
                format!("unknown owned-metal model ref '{}'", model.model_id),
            )
        })?;
        let loaded = loaded.lock().map_err(|_| {
            Self::error(
                EngineErrorStage::Inference,
                "owned-metal model mutex was poisoned",
            )
        })?;
        Ok(loaded.tokenizer_policy)
    }

    #[cfg(not(target_os = "macos"))]
    pub fn tokenizer_policy(&self, _model: &LoadedModel) -> Result<TokenizerPolicy, EngineError> {
        Err(Self::error(
            EngineErrorStage::Load,
            OwnedEngineError::UnsupportedPlatform.to_string(),
        ))
    }

    #[cfg(target_os = "macos")]
    pub fn validate_rerank(&self, model: &LoadedModel) -> Result<(), EngineError> {
        let loaded = self.models.get(&model.model_id).ok_or_else(|| {
            Self::error(
                EngineErrorStage::Load,
                format!("unknown owned-metal model ref '{}'", model.model_id),
            )
        })?;
        let loaded = loaded.lock().map_err(|_| {
            Self::error(
                EngineErrorStage::Load,
                "owned-metal model mutex was poisoned during rerank validation",
            )
        })?;
        if loaded.family.supports_rerank() {
            Ok(())
        } else {
            Err(Self::error(
                EngineErrorStage::Load,
                format!(
                    "owned-metal family '{}' has no sequence-classification head for reranking",
                    loaded.family.family_name()
                ),
            ))
        }
    }

    #[cfg(not(target_os = "macos"))]
    pub fn validate_rerank(&self, _model: &LoadedModel) -> Result<(), EngineError> {
        Err(Self::error(
            EngineErrorStage::Load,
            OwnedEngineError::UnsupportedPlatform.to_string(),
        ))
    }

    #[cfg(target_os = "macos")]
    pub fn rerank_pairs(
        &self,
        model: &LoadedModel,
        pairs: Vec<TokenIds>,
    ) -> Result<RerankScores, EngineError> {
        let loaded = self.models.get(&model.model_id).ok_or_else(|| {
            Self::error(
                EngineErrorStage::Inference,
                format!("unknown owned-metal model ref '{}'", model.model_id),
            )
        })?;
        let mut loaded = loaded.lock().map_err(|_| {
            Self::error(
                EngineErrorStage::Inference,
                "owned-metal model mutex was poisoned during rerank",
            )
        })?;
        run_rerank_bucketed(&mut loaded, pairs)
    }

    #[cfg(not(target_os = "macos"))]
    pub fn rerank_pairs(
        &self,
        _model: &LoadedModel,
        _pairs: Vec<TokenIds>,
    ) -> Result<RerankScores, EngineError> {
        Err(Self::error(
            EngineErrorStage::Inference,
            OwnedEngineError::UnsupportedPlatform.to_string(),
        ))
    }
}

impl EmbedEngine for OwnedMetalEmbedEngine {
    fn identity(&self) -> EngineIdentity {
        engine_identity(self.family, self.dtype)
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
                    "owned-metal requires safetensors-package, got {}",
                    artifact.format
                ),
            ));
        }
        #[cfg(target_os = "macos")]
        {
            self.load_macos(cfg)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = cfg;
            Err(Self::error(
                EngineErrorStage::Load,
                OwnedEngineError::UnsupportedPlatform.to_string(),
            ))
        }
    }

    fn embed_batch(&self, model: &LoadedModel, batch: TokenBatch) -> Result<Vectors, EngineError> {
        #[cfg(target_os = "macos")]
        {
            let loaded = self.models.get(&model.model_id).ok_or_else(|| {
                Self::error(
                    EngineErrorStage::Inference,
                    format!("unknown owned-metal model ref '{}'", model.model_id),
                )
            })?;
            let mut loaded = loaded.lock().map_err(|_| {
                Self::error(
                    EngineErrorStage::Inference,
                    "owned-metal model mutex was poisoned",
                )
            })?;
            run_bucketed(&mut loaded, batch)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (model, batch);
            Err(Self::error(
                EngineErrorStage::Inference,
                OwnedEngineError::UnsupportedPlatform.to_string(),
            ))
        }
    }

    fn embed_one(&self, model: &LoadedModel, ids: TokenIds) -> Result<Vector, EngineError> {
        let mut vectors = self.embed_batch(model, TokenBatch { items: vec![ids] })?;
        vectors.pop().ok_or_else(|| {
            Self::error(
                EngineErrorStage::Inference,
                "owned-metal returned no vector",
            )
        })
    }

    fn unload(&mut self, model: &LoadedModel) {
        self.models.remove(&model.model_id);
    }
}

impl RerankEngine for OwnedMetalEmbedEngine {
    fn identity(&self) -> EngineIdentity {
        <Self as EmbedEngine>::identity(self)
    }

    fn load(
        &mut self,
        artifact: &ValidatedArtifact,
        cfg: &RuntimeConfig,
    ) -> Result<LoadedModel, EngineError> {
        <Self as EmbedEngine>::load(self, artifact, cfg)
    }

    fn rerank(
        &self,
        model: &LoadedModel,
        request: RerankRequest,
    ) -> Result<RerankScores, EngineError> {
        if !request.query.is_empty() {
            return Err(Self::error(
                EngineErrorStage::Inference,
                "owned-metal rerank requires module-framed token-id pairs",
            ));
        }
        self.rerank_pairs(model, request.candidates)
    }

    fn unload(&mut self, model: &LoadedModel) {
        <Self as EmbedEngine>::unload(self, model);
    }
}

#[cfg(target_os = "macos")]
fn run_bucketed(loaded: &mut OwnedLoadedModel, batch: TokenBatch) -> Result<Vectors, EngineError> {
    let profile = embed_profile_enabled();
    let started = Instant::now();
    if batch.items.is_empty() {
        return Ok(Vec::new());
    }
    for (index, ids) in batch.items.iter().enumerate() {
        if ids.is_empty() {
            return Err(OwnedMetalEmbedEngine::error(
                EngineErrorStage::Inference,
                format!("token batch item {index} is empty"),
            ));
        }
        if let Some(terminal) = loaded.tokenizer_policy.terminal_token_id {
            if ids.last() != Some(&terminal) {
                return Err(OwnedMetalEmbedEngine::error(
                    EngineErrorStage::Inference,
                    format!(
                        "token batch item {index} is missing required terminal token {terminal}"
                    ),
                ));
            }
        }
    }
    let mut order = (0..batch.items.len()).collect::<Vec<_>>();
    order.sort_by_key(|&index| batch.items[index].len());
    let mut vectors = vec![Vec::new(); batch.items.len()];
    let mut start = 0;
    let mut bucket_calls = 0_usize;
    while start < order.len() {
        let mut end = start;
        let bucket_started = Instant::now();
        while end < order.len() {
            let length = batch.items[order[end]].len();
            let bucket = runtime::covering_bucket(length, &loaded.buckets).ok_or_else(|| {
                OwnedMetalEmbedEngine::error(
                    EngineErrorStage::Inference,
                    format!("sequence length {length} exceeds certified bucket envelope"),
                )
            })?;
            if end - start + 1 > bucket.batch {
                break;
            }
            end += 1;
        }
        let length = batch.items[order[end - 1]].len();
        let shape =
            runtime::covering_bucket(length, &loaded.buckets).expect("bucket checked above");
        bucket_calls += 1;
        let sequences = order[start..end]
            .iter()
            .map(|&index| batch.items[index].clone())
            .collect::<Vec<_>>();
        if profile {
            eprintln!(
                "[synapse-embed-profile] bucket_select call={} items={} max_tokens={} shape={}x{} select_ms={:.3}",
                bucket_calls,
                sequences.len(),
                length,
                shape.batch,
                shape.seq,
                bucket_started.elapsed().as_secs_f64() * 1_000.0
            );
        }
        let inference_started = Instant::now();
        let produced = loaded
            .family
            .embed_batch(&mut loaded.provider, &sequences, Some(shape))
            .map_err(|error| {
                OwnedMetalEmbedEngine::error(EngineErrorStage::Inference, error.to_string())
            })?;
        if profile {
            eprintln!(
                "[synapse-embed-profile] family_return call={} inference_ms={:.3}",
                bucket_calls,
                inference_started.elapsed().as_secs_f64() * 1_000.0
            );
        }
        for (&original, vector) in order[start..end].iter().zip(produced) {
            vectors[original] = vector;
        }
        start = end;
    }
    if profile {
        eprintln!(
            "[synapse-embed-profile] bucket_total items={} bucket_calls={} total_ms={:.3}",
            batch.items.len(),
            bucket_calls,
            started.elapsed().as_secs_f64() * 1_000.0
        );
    }
    Ok(vectors)
}

#[cfg(target_os = "macos")]
fn run_rerank_bucketed(
    loaded: &mut OwnedLoadedModel,
    pairs: Vec<TokenIds>,
) -> Result<RerankScores, EngineError> {
    if pairs.is_empty() {
        return Ok(RerankScores::default());
    }
    for (index, ids) in pairs.iter().enumerate() {
        if ids.is_empty() {
            return Err(OwnedMetalEmbedEngine::error(
                EngineErrorStage::Inference,
                format!("rerank pair {index} is empty"),
            ));
        }
    }
    let mut order = (0..pairs.len()).collect::<Vec<_>>();
    order.sort_by_key(|&index| pairs[index].len());
    let mut scores = vec![0.0; pairs.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start;
        while end < order.len() {
            let length = pairs[order[end]].len();
            let bucket = runtime::covering_bucket(length, &loaded.buckets).ok_or_else(|| {
                OwnedMetalEmbedEngine::error(
                    EngineErrorStage::Inference,
                    format!("sequence length {length} exceeds certified bucket envelope"),
                )
            })?;
            if end - start + 1 > bucket.batch {
                break;
            }
            end += 1;
        }
        let length = pairs[order[end - 1]].len();
        let shape =
            runtime::covering_bucket(length, &loaded.buckets).expect("rerank bucket checked above");
        let sequences = order[start..end]
            .iter()
            .map(|&index| pairs[index].clone())
            .collect::<Vec<_>>();
        let produced = loaded
            .family
            .rerank_batch(&mut loaded.provider, &sequences, Some(shape))
            .map_err(|error| {
                OwnedMetalEmbedEngine::error(EngineErrorStage::Inference, error.to_string())
            })?;
        if produced.len() != sequences.len() {
            return Err(OwnedMetalEmbedEngine::error(
                EngineErrorStage::Inference,
                format!(
                    "owned-metal rerank returned {} scores for {} pairs",
                    produced.len(),
                    sequences.len()
                ),
            ));
        }
        for (&original, score) in order[start..end].iter().zip(produced) {
            scores[original] = score;
        }
        start = end;
    }
    Ok(RerankScores { scores })
}

#[cfg(target_os = "macos")]
fn precision(dtype: OwnedDType) -> runtime::Precision {
    match dtype {
        OwnedDType::F16 => runtime::Precision::F16,
        OwnedDType::F32 => runtime::Precision::F32,
    }
}

#[cfg(target_os = "macos")]
fn required_path(cfg: &RuntimeConfig, key: &str) -> Result<PathBuf, EngineError> {
    cfg.values
        .get(key)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| {
            OwnedMetalEmbedEngine::error(
                EngineErrorStage::Load,
                format!("runtime config missing {key}"),
            )
        })
}

#[cfg(target_os = "macos")]
fn parse_usize(cfg: &RuntimeConfig, key: &str, default: usize) -> Result<usize, EngineError> {
    cfg.values.get(key).map_or(Ok(default), |value| {
        value.parse::<usize>().map_err(|error| {
            OwnedMetalEmbedEngine::error(
                EngineErrorStage::Load,
                format!("invalid {key} '{value}': {error}"),
            )
        })
    })
}

#[cfg(target_os = "macos")]
fn package_root(
    cache_root: &Path,
    model_path: &Path,
    family: ModelFamily,
    dtype: OwnedDType,
) -> Result<PathBuf, String> {
    let canonical = std::fs::canonicalize(model_path).unwrap_or_else(|_| model_path.to_path_buf());
    let hash = canonical
        .to_string_lossy()
        .bytes()
        .fold(1469598103934665603u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(1099511628211)
        });
    let os_build = std::process::Command::new("sw_vers")
        .arg("-buildVersion")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .filter(|build| !build.is_empty())
        .unwrap_or_else(|| "unknown-os-build".to_string());
    let root = cache_root.join(format!(
        "{}-graph-v{}-bucket-policy-v{}-{hash:016x}-{}-{os_build}",
        family.as_str(),
        GRAPH_REVISION,
        BUCKET_POLICY_VERSION,
        dtype.as_str()
    ));
    std::fs::create_dir_all(&root)
        .map_err(|error| format!("create package root {}: {error}", root.display()))?;
    Ok(root)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_separates_family_dtype_graph_and_policy() {
        let minilm_f16 = engine_identity(ModelFamily::MiniLm, OwnedDType::F16);
        let minilm_f32 = engine_identity(ModelFamily::MiniLm, OwnedDType::F32);
        let qwen_f16 = engine_identity(ModelFamily::Qwen3, OwnedDType::F16);
        assert_ne!(minilm_f16, minilm_f32);
        assert_ne!(minilm_f16, qwen_f16);
        assert_eq!(
            minilm_f16.build_flags["graph_revision"],
            GRAPH_REVISION.to_string()
        );
        assert_eq!(
            minilm_f16.build_flags["bucket_policy"],
            format!("v{BUCKET_POLICY_VERSION}")
        );
    }

    #[test]
    fn recommendations_match_certified_serving_profiles() {
        assert_eq!(ModelFamily::MiniLm.recommended_dtype(), OwnedDType::F16);
        assert_eq!(
            ModelFamily::GteModernBert.recommended_dtype(),
            OwnedDType::F32
        );
        assert_eq!(ModelFamily::Qwen3.recommended_dtype(), OwnedDType::F16);
    }
}
