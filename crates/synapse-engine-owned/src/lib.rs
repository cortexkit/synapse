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
pub const BUCKET_POLICY_VERSION: u32 = 2;
pub const DEFAULT_ATTENTION_UNITS: usize = 4_000_000;
#[cfg(target_os = "macos")]
const MAX_SEQUENCE_BUCKETS: usize = 18;
#[cfg(target_os = "macos")]
const MAX_CACHED_BUCKET_SHAPES: usize = 36;
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnedModelInfo {
    pub dims: usize,
    pub buckets: Vec<usize>,
    pub dtype: OwnedDType,
}

/// Row-major, final-normalized token states, without pooling or a task head.
#[derive(Debug)]
pub struct HiddenStates {
    pub batch: usize,
    pub seq: usize,
    pub hidden: usize,
    pub data: Vec<f32>,
    pub attention_mask: Vec<u8>,
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
    /// Encode tokens without pooling. An explicit `(batch, seq)` must fit a loaded
    /// bucket's capacity; otherwise the smallest covering bucket is selected.
    /// ModernBERT returns states after final_norm and needs no classification head.
    #[cfg(target_os = "macos")]
    pub fn encode_hidden(
        &self,
        model: &LoadedModel,
        sequences: &[Vec<u32>],
        shape: Option<(usize, usize)>,
    ) -> Result<HiddenStates, EngineError> {
        let fail = |message| Self::error(EngineErrorStage::Inference, message);
        let loaded = self
            .models
            .get(&model.model_id)
            .ok_or_else(|| fail("unknown owned-metal model"))?;
        let mut loaded = loaded
            .lock()
            .map_err(|_| fail("owned-metal model mutex was poisoned"))?;
        if sequences.is_empty() || sequences.iter().any(Vec::is_empty) {
            return Err(fail("hidden-state input must be nonempty"));
        }
        let real_seq = sequences.iter().map(Vec::len).max().unwrap();
        let (batch, seq) = shape.unwrap_or_else(|| {
            loaded
                .buckets
                .iter()
                .filter(|s| s.batch >= sequences.len() && s.seq >= real_seq)
                .min_by_key(|s| s.seq)
                .map(|s| (sequences.len(), s.seq))
                .unwrap_or((0, 0))
        });
        if batch < sequences.len()
            || seq < real_seq
            || !loaded
                .buckets
                .iter()
                .any(|s| s.batch >= batch && s.seq >= seq)
        {
            return Err(fail("hidden-state shape exceeds loaded bucket capacity"));
        }
        let OwnedLoadedModel {
            family, provider, ..
        } = &mut *loaded;
        family
            .encode_hidden(provider, sequences, runtime::BatchShape { batch, seq })
            .map_err(|error| Self::error(EngineErrorStage::Inference, error.to_string()))
    }

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

    #[cfg(target_os = "macos")]
    pub fn model_info(&self, model: &LoadedModel) -> Option<OwnedModelInfo> {
        let loaded = self.models.get(&model.model_id)?;
        let loaded = loaded.lock().ok()?;
        let mut buckets = loaded.buckets.iter().map(|b| b.seq).collect::<Vec<_>>();
        buckets.sort_unstable();
        buckets.dedup();
        Some(OwnedModelInfo {
            dims: loaded.family.output_dim(),
            buckets,
            dtype: self.dtype,
        })
    }

    #[cfg(not(target_os = "macos"))]
    pub fn model_info(&self, _model: &LoadedModel) -> Option<OwnedModelInfo> {
        None
    }

    #[cfg(feature = "test-support")]
    pub fn insert_test_model(
        &mut self,
        model_id: String,
        dims: usize,
        bucket_seqs: Vec<usize>,
    ) -> LoadedModel {
        #[cfg(target_os = "macos")]
        {
            struct TestModelFamily(usize);
            impl runtime::ModelFamily for TestModelFamily {
                fn family_name(&self) -> &'static str {
                    "test"
                }
                fn output_dim(&self) -> usize {
                    self.0
                }
                fn tokenizer_policy(&self) -> runtime::FamilyTokenizerPolicy {
                    runtime::FamilyTokenizerPolicy {
                        pad_token_id: 0,
                        terminal_token_id: None,
                    }
                }
                fn embed_batch(
                    &self,
                    _provider: &mut dyn runtime::KernelProvider,
                    _sequences: &[Vec<u32>],
                    _shape: Option<runtime::BatchShape>,
                ) -> anyhow::Result<Vec<Vec<f32>>> {
                    Ok(Vec::new())
                }
            }

            let config = runtime::MetalExecutionConfig::new(runtime::Execution::Explicit, None)
                .expect("test metal config");
            let provider = runtime::MetalProvider::new_with_config(precision(self.dtype), config)
                .expect("test metal provider");
            let buckets = bucket_seqs
                .into_iter()
                .map(|seq| runtime::BatchShape { batch: 1, seq })
                .collect();
            self.models.insert(
                model_id.clone(),
                Arc::new(Mutex::new(OwnedLoadedModel {
                    family: Box::new(TestModelFamily(dims)),
                    provider,
                    buckets,
                    tokenizer_policy: TokenizerPolicy {
                        add_special_tokens: false,
                        pad_token_id: 0,
                        terminal_token_id: None,
                    },
                })),
            );
            LoadedModel { model_id }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (dims, bucket_seqs);
            LoadedModel { model_id }
        }
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
        if buckets.len() > MAX_SEQUENCE_BUCKETS {
            return Err(Self::error(
                EngineErrorStage::Load,
                format!("bucket policy produced {} sequence buckets", buckets.len()),
            ));
        }
        let cache_shapes = runtime::cache_shapes(&buckets);
        if cache_shapes.len() > MAX_CACHED_BUCKET_SHAPES {
            return Err(Self::error(
                EngineErrorStage::Load,
                format!("bucket policy produced {} cache shapes", cache_shapes.len()),
            ));
        }
        let mut provider =
            runtime::MetalProvider::new_with_config(precision(self.dtype), config)
                .map_err(|error| Self::error(EngineErrorStage::Load, error.to_string()))?;
        let preload_ids = vec![vec![policy
            .terminal_token_id
            .unwrap_or(policy.pad_token_id)]];
        // Preserve the established short-shape preload set. Long capacity graphs and
        // singleton graphs compile on first use, avoiding cold-load and resident-plan
        // growth for sequence lengths a process never serves.
        for &shape in runtime::eager_shapes(&buckets) {
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
    let lengths = batch.items.iter().map(Vec::len).collect::<Vec<_>>();
    let plans = runtime::plan_batches(&lengths, &loaded.buckets).map_err(|length| {
        OwnedMetalEmbedEngine::error(
            EngineErrorStage::Inference,
            format!("sequence length {length} exceeds certified bucket envelope"),
        )
    })?;
    let mut vectors = vec![Vec::new(); batch.items.len()];
    let mut bucket_calls = 0_usize;
    for plan in plans {
        let bucket_started = Instant::now();
        bucket_calls += 1;
        let sequences = plan
            .indices
            .iter()
            .map(|&index| batch.items[index].clone())
            .collect::<Vec<_>>();
        if profile {
            eprintln!(
                "[synapse-embed-profile] bucket_select call={} items={} max_tokens={} shape={}x{} select_ms={:.3}",
                bucket_calls,
                sequences.len(),
                plan.max_tokens,
                plan.shape.batch,
                plan.shape.seq,
                bucket_started.elapsed().as_secs_f64() * 1_000.0
            );
        }
        let inference_started = Instant::now();
        let produced = loaded
            .family
            .embed_batch(&mut loaded.provider, &sequences, Some(plan.shape))
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
        for (&original, vector) in plan.indices.iter().zip(produced) {
            vectors[original] = vector;
        }
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
    let lengths = pairs.iter().map(Vec::len).collect::<Vec<_>>();
    let plans = runtime::plan_batches(&lengths, &loaded.buckets).map_err(|length| {
        OwnedMetalEmbedEngine::error(
            EngineErrorStage::Inference,
            format!("sequence length {length} exceeds certified bucket envelope"),
        )
    })?;
    let mut scores = vec![0.0; pairs.len()];
    for plan in plans {
        let sequences = plan
            .indices
            .iter()
            .map(|&index| pairs[index].clone())
            .collect::<Vec<_>>();
        let produced = loaded
            .family
            .rerank_batch(&mut loaded.provider, &sequences, Some(plan.shape))
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
        for (&original, score) in plan.indices.iter().zip(produced) {
            scores[original] = score;
        }
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
        .unwrap_or_else(|| UNKNOWN_OS_BUILD.to_string());
    resolve_package_root(
        cache_root,
        family.as_str(),
        &format!("{hash:016x}"),
        dtype.as_str(),
        &os_build,
    )
}

/// OS build recorded when `sw_vers` cannot be read. Pruning is skipped under it,
/// because without the real build this process cannot tell which keys are stale.
#[cfg(target_os = "macos")]
const UNKNOWN_OS_BUILD: &str = "unknown-os-build";

/// The parts of a package directory name that decide whether it is stale.
#[cfg(target_os = "macos")]
struct PackageKey<'a> {
    graph_revision: u32,
    bucket_policy: u32,
    model_hash: &'a str,
    dtype: &'a str,
    os_build: &'a str,
}

/// Directory name of one compiled-package cache entry. Every component that
/// can make a serialized MPSGraph package unusable is part of the name.
#[cfg(target_os = "macos")]
fn package_key_name(
    family: &str,
    graph_revision: u32,
    bucket_policy: u32,
    model_hash: &str,
    dtype: &str,
    os_build: &str,
) -> String {
    format!("{family}-graph-v{graph_revision}-bucket-policy-v{bucket_policy}-{model_hash}-{dtype}-{os_build}")
}

/// Split a package directory name produced by `package_key_name` for `family`
/// into its parts. Names of other families (including families whose name
/// merely starts with `family`) and names that do not follow the layout return
/// `None`, so they are never treated as a sibling.
#[cfg(target_os = "macos")]
fn parse_package_key<'a>(name: &'a str, family: &str) -> Option<PackageKey<'a>> {
    let rest = name.strip_prefix(family)?.strip_prefix("-graph-v")?;
    let (graph_revision, rest) = rest.split_once("-bucket-policy-v")?;
    let mut parts = rest.splitn(4, '-');
    let bucket_policy = parts.next()?;
    let model_hash = parts.next()?;
    let dtype = parts.next()?;
    let os_build = parts.next()?;
    let is_number = |value: &str| !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit());
    if !(is_number(graph_revision)
        && is_number(bucket_policy)
        && model_hash.len() == 16
        && model_hash.bytes().all(|b| b.is_ascii_hexdigit())
        && !dtype.is_empty()
        && !os_build.is_empty())
    {
        return None;
    }
    Some(PackageKey {
        graph_revision: graph_revision.parse().ok()?,
        bucket_policy: bucket_policy.parse().ok()?,
        model_hash,
        dtype,
        os_build,
    })
}

/// Create the package directory for the current graph revision, bucket policy
/// and OS build, then prune stale keys of the same model and dtype.
#[cfg(target_os = "macos")]
fn resolve_package_root(
    cache_root: &Path,
    family: &str,
    model_hash: &str,
    dtype: &str,
    os_build: &str,
) -> Result<PathBuf, String> {
    let name = package_key_name(
        family,
        GRAPH_REVISION,
        BUCKET_POLICY_VERSION,
        model_hash,
        dtype,
        os_build,
    );
    let root = cache_root.join(&name);
    std::fs::create_dir_all(&root)
        .map_err(|error| format!("create package root {}: {error}", root.display()))?;
    if os_build != UNKNOWN_OS_BUILD {
        prune_stale_package_siblings(cache_root, &name, family, model_hash, dtype, os_build);
    }
    Ok(root)
}

/// Remove package directories of the same family, model and dtype that no
/// binary on this machine can load again: those compiled on a different OS
/// build, and those of an OLDER graph revision or bucket policy.
///
/// Serialized MPSGraph packages are specific to the OS build and to the graph
/// that built them, so without pruning every OS update or revision bump leaves
/// a full stale set per model behind. Keys of a NEWER revision are kept: they
/// belong to a newer binary sharing this cache root (for example a test run
/// from a tree ahead of the deployed module), and pruning in both directions
/// would make the two binaries delete and recompile each other's packages on
/// every load.
///
/// Pruning is best effort: failures are logged and never fail the load.
///
/// Several processes can resolve packages for the same model at once (the
/// module, a certification probe, a bench or a test sharing the cache root),
/// and one of them may be an older binary still compiling into the key this
/// one considers stale. A stale directory is therefore first renamed to a
/// private hidden name and only then deleted. The rename is atomic, so a
/// package at a live key path is never observed half-deleted: an older
/// writer either finds its directory gone (its package load falls back to
/// compiling) or keeps writing into the renamed tree, which is then removed.
/// The current key and keys of other models or dtypes are never touched.
#[cfg(target_os = "macos")]
fn prune_stale_package_siblings(
    cache_root: &Path,
    current: &str,
    family: &str,
    model_hash: &str,
    dtype: &str,
    os_build: &str,
) {
    let entries = match std::fs::read_dir(cache_root) {
        Ok(entries) => entries,
        Err(error) => {
            eprintln!(
                "owned-metal: skip stale package pruning, cannot list {}: {error}",
                cache_root.display()
            );
            return;
        }
    };
    let is_stale_sibling = |name: &str| {
        name != current
            && parse_package_key(name, family).is_some_and(|key| {
                key.model_hash == model_hash
                    && key.dtype == dtype
                    && (key.os_build != os_build
                        || key.graph_revision < GRAPH_REVISION
                        || key.bucket_policy < BUCKET_POLICY_VERSION)
            })
    };
    for entry in entries.flatten() {
        let Ok(name) = entry.file_name().into_string() else {
            continue;
        };
        if !entry.file_type().is_ok_and(|kind| kind.is_dir()) {
            continue;
        }
        // A hidden `.<key>.pruning-*` directory is a stale key whose deletion
        // was interrupted after the rename below; finish removing it.
        let pruning_leftover = name
            .strip_prefix('.')
            .and_then(|hidden| hidden.split_once(".pruning-"))
            .is_some_and(|(key, _)| is_stale_sibling(key));
        if pruning_leftover {
            remove_package_tree(&entry.path());
            continue;
        }
        if !is_stale_sibling(&name) {
            continue;
        }
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos());
        let trash = cache_root.join(format!(".{name}.pruning-{}-{nonce}", std::process::id()));
        match std::fs::rename(entry.path(), &trash) {
            Ok(()) => remove_package_tree(&trash),
            // Another process pruned the same key first.
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => eprintln!(
                "owned-metal: failed to retire stale package {}: {error}",
                entry.path().display()
            ),
        }
    }
}

#[cfg(target_os = "macos")]
fn remove_package_tree(path: &Path) {
    match std::fs::remove_dir_all(path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => eprintln!(
            "owned-metal: failed to remove stale package {}: {error}",
            path.display()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_separates_family_dtype_graph_and_policy() {
        assert_eq!(BUCKET_POLICY_VERSION, runtime::BUCKET_POLICY_VERSION);
        let minilm_f16 = engine_identity(ModelFamily::MiniLm, OwnedDType::F16);
        let minilm_f32 = engine_identity(ModelFamily::MiniLm, OwnedDType::F32);
        let qwen_f16 = engine_identity(ModelFamily::Qwen3, OwnedDType::F16);
        assert_ne!(minilm_f16, minilm_f32);
        assert_ne!(minilm_f16, qwen_f16);
        assert_eq!(
            minilm_f16.build_flags["graph_revision"],
            GRAPH_REVISION.to_string()
        );
        for family in [
            ModelFamily::MiniLm,
            ModelFamily::GteModernBert,
            ModelFamily::Qwen3,
        ] {
            assert_eq!(
                engine_identity(family, family.recommended_dtype()).build_flags["bucket_policy"],
                "v2"
            );
        }
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

    #[cfg(target_os = "macos")]
    #[test]
    fn resolving_a_package_key_prunes_only_stale_keys_of_the_same_model_and_dtype() {
        let cache_root = std::env::temp_dir().join(format!(
            "synapse-owned-package-prune-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&cache_root).unwrap();
        let family = ModelFamily::GteModernBert.as_str();
        let model = "6b82e782f001347b";
        let other_model = "fb2726a2bfc119cb";
        let key = |graph, policy, hash, dtype, os| {
            let name = package_key_name(family, graph, policy, hash, dtype, os);
            let path = cache_root.join(&name);
            std::fs::create_dir_all(path.join("8x128.mpsgraphpackage")).unwrap();
            std::fs::write(path.join("8x128.mpsgraphpackage/data"), b"package").unwrap();
            path
        };
        let current_policy = BUCKET_POLICY_VERSION;
        let current_graph = GRAPH_REVISION;
        let older_os = key(current_graph, current_policy, model, "f16", "25G72");
        let older_policy = key(current_graph, current_policy - 1, model, "f16", "26A428");
        let older_graph = key(current_graph - 1, current_policy, model, "f16", "26A428");
        let interrupted_prune = cache_root.join(format!(
            ".{}.pruning-1-2",
            package_key_name(family, current_graph, current_policy, model, "f16", "25F84")
        ));
        std::fs::create_dir_all(&interrupted_prune).unwrap();
        let current_populated = key(current_graph, current_policy, model, "f16", "26A428");
        let other_dtype = key(current_graph, current_policy, model, "f32", "25G72");
        let other_model_key = key(current_graph, current_policy, other_model, "f16", "25G72");
        let newer_graph = key(current_graph + 1, current_policy, model, "f16", "26A428");
        let newer_policy = key(current_graph, current_policy + 1, model, "f16", "26A428");
        let other_family = cache_root.join(package_key_name(
            "minilm",
            current_graph,
            current_policy,
            model,
            "f16",
            "25G72",
        ));
        std::fs::create_dir_all(&other_family).unwrap();
        let unrelated = cache_root.join("unrelated-directory");
        std::fs::create_dir_all(&unrelated).unwrap();

        let resolved = resolve_package_root(&cache_root, family, model, "f16", "26A428").unwrap();

        assert_eq!(resolved, current_populated);
        assert!(
            resolved.join("8x128.mpsgraphpackage/data").is_file(),
            "the resolved key keeps its compiled packages"
        );
        for stale in [&older_os, &older_policy, &older_graph, &interrupted_prune] {
            assert!(!stale.exists(), "stale key survived: {}", stale.display());
        }
        for kept in [
            &other_dtype,
            &other_model_key,
            &other_family,
            &unrelated,
            &newer_graph,
            &newer_policy,
        ] {
            assert!(
                kept.is_dir(),
                "unrelated key was removed: {}",
                kept.display()
            );
        }
        let leftovers = std::fs::read_dir(&cache_root)
            .unwrap()
            .filter_map(|entry| entry.ok()?.file_name().into_string().ok())
            .filter(|name| name.starts_with('.'))
            .collect::<Vec<_>>();
        assert!(leftovers.is_empty(), "pruning left {leftovers:?}");
        std::fs::remove_dir_all(&cache_root).unwrap();
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn resolving_under_an_unknown_os_build_prunes_nothing() {
        let cache_root = std::env::temp_dir().join(format!(
            "synapse-owned-package-unknown-os-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let family = ModelFamily::GteModernBert.as_str();
        let model = "6b82e782f001347b";
        let real = cache_root.join(package_key_name(
            family,
            GRAPH_REVISION,
            BUCKET_POLICY_VERSION,
            model,
            "f16",
            "26A428",
        ));
        std::fs::create_dir_all(&real).unwrap();

        resolve_package_root(&cache_root, family, model, "f16", UNKNOWN_OS_BUILD).unwrap();

        assert!(
            real.is_dir(),
            "a failed sw_vers read must not prune the real OS build's packages"
        );
        std::fs::remove_dir_all(&cache_root).unwrap();
    }
}
