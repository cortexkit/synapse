#![forbid(unsafe_code)]

use std::{
    collections::BTreeMap,
    env, fs,
    io::Read,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
    time::{SystemTime, UNIX_EPOCH},
};

mod store;
#[cfg(unix)]
pub mod worker_host;

use cortexkit_store_types::{sqlite_store_path, Isolation, StorageBackend, StorageDescriptor};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use store::{
    CatalogSnapshot, JobAdmission, JobRecord, ModelCatalogEntry, SynapseStore, SynapseStoreError,
    JOB_STATE_DONE, JOB_STATE_FAILED_PERMANENT, JOB_STATE_FAILED_TRANSIENT, JOB_STATE_QUEUED,
    JOB_STATE_RUNNING,
};
use subc_client_rs::{
    async_trait, BindDecision, HandlerOutcome, HealthReport, ModuleHandler, RequestCtx,
    RouteBindRequest, SubcModuleError,
};
use subc_protocol::{
    manifest::{
        Bindings, IdentityBinding, IdentityScope, ManagementOperation, ManagementOperationKind,
        ModuleManifest, ProviderRole, StorageBinding, StorageKind, StorageScope, TrustTier,
    },
    ModuleHelloAckBody, PROTOCOL_VERSION, SUBC_MODULE_ID_ENV,
};
use synapse_core::{
    AdmissionDecision, AdmissionRequest, AliasTable, CacheGcOutcome, CertifiedShapeEnvelope, Clock,
    EmbedEngine, EngineError, EngineErrorStage, EngineIdentity, ErrorClass, Fingerprint,
    FlashAttentionSetting, LaneBudgetSnapshot, LaneScheduler, LoadedModel, ModelCache,
    ModelCacheError, ModelCacheIngest, ModelCacheMeta, NormalizationMode, NumericDType,
    NumericProfile, PoolingStrategy, QueueClass, ResponseEnvelope, ResponseProvenance,
    RuntimeConfig, SanitizedTokenizer, SchedulerConfig, StableError, ThreadPolicyClass, TokenBatch,
    TokenizationError, TokenizedBatch, TokenizerConfig, TruncationDisclosure, ValidatedArtifact,
    Vectors, WorkRequest, WorkerPooling,
};
use synapse_engine_ort::OrtEmbedEngine;
use thiserror::Error;
use tokio::sync::Semaphore;

pub const DEFAULT_MODULE_ID: &str = "synapse";

const DEFAULT_INLINE_MAX_ITEMS: usize = 64;
const DEFAULT_INLINE_MAX_TOKENS: u64 = 8_192;
const DEFAULT_INLINE_BYTE_BUDGET: u64 = 64 * 1024 * 1024;
const DEFAULT_MAX_QUEUE_MS: u64 = 5_000;
const DEFAULT_DEADLINE_MS: u64 = 30_000;
const DEFAULT_ESTIMATED_EXECUTION_MS: u64 = 25;
const DEFAULT_MAX_CONCURRENT_WORKERS: usize = 2;
const DEFAULT_JOB_TTL_MS: u64 = 24 * 60 * 60 * 1_000;
const DEFAULT_JOB_RESULT_PAGE_BYTES: usize = 512 * 1024;
const DEFAULT_JOB_BULK_QUANTUM_TOKENS: u64 = 2_048;

pub async fn run_from_env() -> Result<(), ModuleError> {
    let module_id = env::var(SUBC_MODULE_ID_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_MODULE_ID.to_string());
    let handler = SynapseHandler::new(module_id.clone());
    subc_client_rs::serve(manifest(&module_id), handler)
        .await
        .map_err(ModuleError::Serve)
}

#[derive(Debug, Error)]
pub enum ModuleError {
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("storage: {0}")]
    Store(#[from] SynapseStoreError),
    #[error("subc serve: {0}")]
    Serve(#[from] SubcModuleError),
    #[error("tokenization: {0}")]
    Tokenization(#[from] TokenizationError),
    #[error("model cache: {0}")]
    Cache(#[from] ModelCacheError),
    #[error("engine: {0}")]
    Engine(String),
    #[error("config: {0}")]
    Config(String),
}

#[derive(Clone)]
struct SynapseHandler {
    inner: Arc<SynapseHandlerInner>,
}

struct SynapseHandlerInner {
    module_id: String,
    state: OnceLock<Arc<ModuleState>>,
}

struct ModuleState {
    module_id: String,
    store: Arc<SynapseStore>,
    module_generation: u64,
    health: ModuleHealth,
    runtime: Arc<RuntimeState>,
    model_cache: Arc<ModelCache>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct ModuleHealth {
    status: String,
    module_generation: u64,
    loaded_models: usize,
}

#[derive(Debug, Deserialize)]
struct MethodEnvelope {
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct WireOperationError {
    code: String,
    class: ErrorClass,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    retry_after_ms: Option<u64>,
    safe_to_retry_same_request: bool,
    message: String,
}

impl WireOperationError {
    fn from_stable(error: StableError, message: impl Into<String>) -> Self {
        Self {
            code: serde_json::to_value(error.code)
                .expect("stable error code serializes")
                .as_str()
                .expect("stable error code is a string")
                .to_string(),
            class: error.class,
            retry_after_ms: error.retry_after_ms,
            safe_to_retry_same_request: error.safe_to_retry_same_request,
            message: message.into(),
        }
    }

    fn not_implemented(message: impl Into<String>) -> Self {
        Self {
            code: "not_implemented".to_string(),
            class: ErrorClass::Permanent,
            retry_after_ms: None,
            safe_to_retry_same_request: false,
            message: message.into(),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
struct ModuleConfig {
    #[serde(default)]
    preload_models: Vec<PreloadModelConfig>,
    #[serde(default)]
    inline: InlineConfig,
    #[serde(default)]
    jobs: JobConfig,
}

#[derive(Clone, Debug, Deserialize)]
struct PreloadModelConfig {
    #[serde(default)]
    model_id: Option<String>,
    engine: String,
    model_path: PathBuf,
    tokenizer_path: PathBuf,
    #[serde(default)]
    artifact_digest: Option<String>,
    #[serde(default)]
    format: Option<String>,
    #[serde(default)]
    pooling: Option<String>,
    #[serde(default)]
    normalize: Option<bool>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    quant: Option<String>,
    #[serde(default)]
    worker_bin: Option<PathBuf>,
    #[serde(default)]
    worker_runtime_dir: Option<PathBuf>,
}

#[derive(Clone, Debug, Deserialize)]
struct InlineConfig {
    #[serde(default = "default_inline_max_items")]
    max_items: usize,
    #[serde(default = "default_inline_max_tokens")]
    max_tokens: u64,
    #[serde(default = "default_inline_byte_budget")]
    byte_budget: u64,
    #[serde(default = "default_max_queue_ms")]
    max_queue_ms: u64,
    #[serde(default = "default_deadline_ms")]
    deadline_ms: u64,
    #[serde(default = "default_estimated_execution_ms")]
    estimated_execution_ms: u64,
    #[serde(default = "default_max_concurrent_workers")]
    max_concurrent_workers: usize,
}

impl Default for InlineConfig {
    fn default() -> Self {
        Self {
            max_items: default_inline_max_items(),
            max_tokens: default_inline_max_tokens(),
            byte_budget: default_inline_byte_budget(),
            max_queue_ms: default_max_queue_ms(),
            deadline_ms: default_deadline_ms(),
            estimated_execution_ms: default_estimated_execution_ms(),
            max_concurrent_workers: default_max_concurrent_workers(),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
struct JobConfig {
    #[serde(default = "default_job_ttl_ms")]
    ttl_ms: u64,
    #[serde(default = "default_job_result_page_bytes")]
    result_page_bytes: usize,
    #[serde(default = "default_job_bulk_quantum_tokens")]
    bulk_quantum_tokens: u64,
}

impl Default for JobConfig {
    fn default() -> Self {
        Self {
            ttl_ms: default_job_ttl_ms(),
            result_page_bytes: default_job_result_page_bytes(),
            bulk_quantum_tokens: default_job_bulk_quantum_tokens(),
        }
    }
}

fn default_inline_max_items() -> usize {
    DEFAULT_INLINE_MAX_ITEMS
}

fn default_inline_max_tokens() -> u64 {
    DEFAULT_INLINE_MAX_TOKENS
}

fn default_inline_byte_budget() -> u64 {
    DEFAULT_INLINE_BYTE_BUDGET
}

fn default_max_queue_ms() -> u64 {
    DEFAULT_MAX_QUEUE_MS
}

fn default_deadline_ms() -> u64 {
    DEFAULT_DEADLINE_MS
}

fn default_estimated_execution_ms() -> u64 {
    DEFAULT_ESTIMATED_EXECUTION_MS
}

fn default_max_concurrent_workers() -> usize {
    DEFAULT_MAX_CONCURRENT_WORKERS
}

fn default_job_ttl_ms() -> u64 {
    DEFAULT_JOB_TTL_MS
}

fn default_job_result_page_bytes() -> usize {
    DEFAULT_JOB_RESULT_PAGE_BYTES
}

fn default_job_bulk_quantum_tokens() -> u64 {
    DEFAULT_JOB_BULK_QUANTUM_TOKENS
}

struct RuntimeState {
    models: BTreeMap<String, Arc<EmbeddingModel>>,
    default_model_id: Option<String>,
    inline: InlineConfig,
    jobs: JobConfig,
    scheduler: Arc<Mutex<InlineScheduler>>,
    execution: Arc<Semaphore>,
}

struct InlineScheduler {
    in_flight_bytes: u64,
}

struct InlineAdmission {
    scheduler: Arc<Mutex<InlineScheduler>>,
    request_bytes: u64,
}

impl Drop for InlineAdmission {
    fn drop(&mut self) {
        if let Ok(mut scheduler) = self.scheduler.lock() {
            scheduler.in_flight_bytes =
                scheduler.in_flight_bytes.saturating_sub(self.request_bytes);
        }
    }
}

struct EmbeddingModel {
    model_id: String,
    loaded_model: LoadedModel,
    backend: EmbedBackend,
    tokenizer: SanitizedTokenizer,
    fingerprint: Fingerprint,
    engine_identity: EngineIdentity,
}

#[derive(Clone)]
enum EmbedBackend {
    Ort(Arc<Mutex<OrtEmbedEngine>>),
    #[cfg(unix)]
    Llama(Arc<Mutex<worker_host::WorkerEngine>>),
}

#[derive(Clone, Debug, Serialize)]
struct EmbedVector {
    id: String,
    vector: Vec<f32>,
}

#[derive(Clone, Debug, Serialize)]
struct EmbedResponsePayload {
    vectors: Vec<EmbedVector>,
    real_token_counts: Vec<u32>,
    truncation_disclosures: Vec<TruncationDisclosure>,
}

#[derive(Debug, Deserialize)]
struct EmbedQueryParams {
    text: String,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    deadline_ms: Option<u64>,
    #[serde(default)]
    max_queue_ms: Option<u64>,
    #[serde(default)]
    target_fingerprint: Option<String>,
    #[serde(default)]
    required_fingerprint: Option<String>,
    #[serde(default)]
    allow_equivalent: bool,
    #[serde(default)]
    required_epoch: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct EmbedBatchParams {
    #[serde(default)]
    items: Vec<EmbedBatchItemParam>,
    #[serde(default)]
    texts: Vec<String>,
    #[serde(default)]
    request_key: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    deadline_ms: Option<u64>,
    #[serde(default)]
    max_queue_ms: Option<u64>,
    #[serde(default)]
    target_fingerprint: Option<String>,
    #[serde(default)]
    required_fingerprint: Option<String>,
    #[serde(default)]
    allow_equivalent: bool,
    #[serde(default)]
    required_epoch: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EmbedBatchItemParam {
    Object { id: String, text: String },
    Text(String),
}

#[derive(Clone, Debug)]
struct EmbedBatchItem {
    id: String,
    text: String,
}

struct EmbedBatchJobWork {
    model: Arc<EmbeddingModel>,
    ids: Vec<String>,
    tokenized: TokenizedBatch,
    table_epoch: u64,
    request_bytes: u64,
    total_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct EmbedResultParams {
    job_id: String,
    #[serde(default)]
    page: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct CachePinParams {
    #[serde(default)]
    digest: Option<String>,
    #[serde(default)]
    source_url: Option<String>,
    #[serde(default)]
    expected_digest: Option<String>,
    #[serde(default)]
    format: Option<String>,
    #[serde(default)]
    tokenizer_path: Option<PathBuf>,
    #[serde(default)]
    module_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CacheGcParams {
    #[serde(default)]
    digest: Option<String>,
    #[serde(default)]
    grace_ms: Option<u64>,
}

struct SystemClock;

impl Clock for SystemClock {
    fn now_ms(&self) -> u64 {
        now_ms()
    }
}

impl SynapseHandler {
    fn new(module_id: String) -> Self {
        Self {
            inner: Arc::new(SynapseHandlerInner {
                module_id,
                state: OnceLock::new(),
            }),
        }
    }

    fn state(&self) -> Option<Arc<ModuleState>> {
        self.inner.state.get().cloned()
    }

    fn initialize(&self, ack: &ModuleHelloAckBody) -> Result<Arc<ModuleState>, ModuleError> {
        let descriptor = resolve_storage_descriptor(&ack.storage, &self.inner.module_id)?;
        let store = Arc::new(SynapseStore::open(&descriptor)?);
        let module_generation = store.next_module_generation()?;
        let restart_error = WireOperationError::from_stable(
            StableError::module_restarted(),
            "module restarted before the durable job reached a terminal result",
        );
        store.fail_prior_generation_incomplete_jobs(
            module_generation,
            &serde_json::to_value(&restart_error).expect("restart error serializes"),
            now_ms(),
        )?;
        let model_cache = Arc::new(ModelCache::new(ModelCache::default_root()?));
        let runtime = Arc::new(RuntimeState::from_config(
            load_module_config()?,
            Arc::clone(&model_cache),
        )?);
        let health = ModuleHealth {
            status: "ok".to_string(),
            module_generation,
            loaded_models: runtime.models.len(),
        };
        Ok(Arc::new(ModuleState {
            module_id: self.inner.module_id.clone(),
            store,
            module_generation,
            health,
            runtime,
            model_cache,
        }))
    }
}

impl RuntimeState {
    fn from_config(
        config: ModuleConfig,
        model_cache: Arc<ModelCache>,
    ) -> Result<Self, ModuleError> {
        let inline = config.inline;
        let jobs = config.jobs;
        let scheduler = Arc::new(Mutex::new(InlineScheduler { in_flight_bytes: 0 }));
        let execution = Arc::new(Semaphore::new(inline.max_concurrent_workers.max(1)));
        let mut models = BTreeMap::new();
        let ort_engine = Arc::new(Mutex::new(OrtEmbedEngine::new()));

        for (index, preload) in config.preload_models.into_iter().enumerate() {
            let model = preload_embedding_model(
                index,
                preload,
                Arc::clone(&ort_engine),
                &inline,
                &model_cache,
            )?;
            models.insert(model.model_id.clone(), Arc::new(model));
        }
        let default_model_id = models.keys().next().cloned();
        Ok(Self {
            models,
            default_model_id,
            inline,
            jobs,
            scheduler,
            execution,
        })
    }

    fn catalog_entries(&self) -> Vec<ModelCatalogEntry> {
        self.models
            .values()
            .map(|model| ModelCatalogEntry {
                model_id: model.model_id.clone(),
                state: "loaded".to_string(),
                fingerprints: vec![model.fingerprint.clone()],
            })
            .collect()
    }

    fn resolve_model(
        &self,
        requested: Option<&str>,
    ) -> Result<Arc<EmbeddingModel>, WireOperationError> {
        let model_id = requested
            .map(str::to_string)
            .or_else(|| self.default_model_id.clone())
            .ok_or_else(|| {
                WireOperationError::from_stable(
                    StableError::probe_required(),
                    "embed requests require a preloaded v1-dev embedding model",
                )
            })?;
        self.models.get(&model_id).cloned().ok_or_else(|| {
            WireOperationError::from_stable(
                StableError::model_loading(Some(250)),
                format!("embedding model '{model_id}' is not loaded"),
            )
        })
    }

    fn admit_inline(
        &self,
        queue_class: QueueClass,
        request_bytes: u64,
        deadline_ms: Option<u64>,
        max_queue_ms: Option<u64>,
    ) -> Result<InlineAdmission, WireOperationError> {
        let clock = SystemClock;
        let now = clock.now_ms();
        let deadline_at = Some(now.saturating_add(deadline_ms.unwrap_or(self.inline.deadline_ms)));
        let max_queue_ms = max_queue_ms.unwrap_or(self.inline.max_queue_ms);
        let predicted_start_delay_ms = if self.execution.available_permits() == 0 {
            self.inline.estimated_execution_ms
        } else {
            0
        };
        let mut scheduler = self.scheduler.lock().map_err(|_| {
            WireOperationError::from_stable(
                StableError::queue_full(Some(100)),
                "inline scheduler state is unavailable",
            )
        })?;
        let lane = LaneBudgetSnapshot {
            queued_bytes: 0,
            in_flight_bytes: scheduler.in_flight_bytes,
            byte_budget: self.inline.byte_budget,
            predicted_start_delay_ms,
        };
        match synapse_core::decide_admission(
            &clock,
            &AdmissionRequest {
                queue_class,
                deadline_ms: deadline_at,
                max_queue_ms,
                request_bytes,
                estimated_execution_ms: self.inline.estimated_execution_ms,
            },
            &lane,
        ) {
            AdmissionDecision::Accept(_) => {
                scheduler.in_flight_bytes = scheduler.in_flight_bytes.saturating_add(request_bytes);
                Ok(InlineAdmission {
                    scheduler: Arc::clone(&self.scheduler),
                    request_bytes,
                })
            }
            AdmissionDecision::Reject(rejection) => Err(WireOperationError::from_stable(
                rejection.error,
                rejection.reason,
            )),
        }
    }
}

fn preload_embedding_model(
    index: usize,
    preload: PreloadModelConfig,
    ort_engine: Arc<Mutex<OrtEmbedEngine>>,
    inline: &InlineConfig,
    model_cache: &ModelCache,
) -> Result<EmbeddingModel, ModuleError> {
    let model_id = preload
        .model_id
        .clone()
        .unwrap_or_else(|| format!("{}-{index}", preload.engine));
    let engine_name = preload.engine.trim().to_ascii_lowercase();
    let pooling = parse_pooling(preload.pooling.as_deref().unwrap_or("mean"))?;
    let normalize = preload.normalize.unwrap_or(true);
    let max_tokens = preload.max_tokens.unwrap_or(512);
    let artifact_format = preload.format.clone().unwrap_or_else(|| {
        if engine_name == "llama" {
            "gguf".to_string()
        } else {
            "onnx".to_string()
        }
    });
    let model_digest = match preload.artifact_digest.clone() {
        Some(digest) => normalize_digest(&digest),
        None => format!("sha256:{}", sha256_file(&preload.model_path)?),
    };
    let quant = preload.quant.clone().unwrap_or_else(|| {
        if engine_name == "llama" {
            "f16".to_string()
        } else {
            "fp32".to_string()
        }
    });
    let tokenizer =
        SanitizedTokenizer::from_file(&preload.tokenizer_path, TokenizerConfig { max_tokens })?;
    let mut runtime_config = RuntimeConfig::default();
    runtime_config.values.insert(
        "model_path".to_string(),
        preload.model_path.to_string_lossy().to_string(),
    );
    runtime_config.values.insert(
        "artifact_path".to_string(),
        preload.model_path.to_string_lossy().to_string(),
    );
    runtime_config
        .values
        .insert("pooling".to_string(), pooling.as_str().to_string());
    runtime_config.values.insert(
        "normalize".to_string(),
        if normalize { "true" } else { "false" }.to_string(),
    );
    let artifact = ValidatedArtifact {
        digest: model_digest.clone(),
        format: artifact_format,
    };
    let _cache_read_lease = model_cache.acquire_read(&model_digest)?;

    let (backend, engine_identity, loaded_model) = match engine_name.as_str() {
        "ort" | "onnx" => {
            let mut engine = ort_engine.lock().map_err(|_| {
                ModuleError::Engine("ORT engine mutex was poisoned during preload".to_string())
            })?;
            let engine_identity = engine.identity();
            let loaded_model = engine.load(&artifact, &runtime_config).map_err(|error| {
                ModuleError::Engine(format!(
                    "preload model '{model_id}' failed: {}",
                    error.message
                ))
            })?;
            (
                EmbedBackend::Ort(Arc::clone(&ort_engine)),
                engine_identity,
                loaded_model,
            )
        }
        "llama" | "llama.cpp" => {
            preload_llama_backend(&model_id, preload, &artifact, &runtime_config)?
        }
        other => {
            return Err(ModuleError::Config(format!(
                "unsupported preload engine '{other}' for model '{model_id}'"
            )))
        }
    };

    let numeric_profile = NumericProfile {
        model_digest,
        quant,
        engine: engine_identity.clone(),
        sanitized_tokenizer_digest: format!("sha256:{}", tokenizer.sanitized_sha256()),
        pooling: profile_pooling(pooling),
        normalization: if normalize {
            NormalizationMode::L2
        } else {
            NormalizationMode::None
        },
        dtype: if engine_name == "llama" {
            NumericDType::F16
        } else {
            NumericDType::F32
        },
        flash_attention: FlashAttentionSetting::Disabled,
        certified_shape: CertifiedShapeEnvelope {
            max_context_tokens: max_tokens.min(u32::MAX as usize) as u32,
            max_batch_tokens: inline.max_tokens.min(u32::MAX as u64) as u32,
            max_micro_batch_tokens: SchedulerConfig::default()
                .bulk_quantum_tokens
                .min(u32::MAX as u64) as u32,
            max_sequences: inline.max_items.min(u32::MAX as usize) as u32,
        },
        prompt_template: None,
        prefix_template: None,
        thread_policy: ThreadPolicyClass::Balanced,
    };
    let fingerprint = numeric_profile.fingerprint();
    Ok(EmbeddingModel {
        model_id,
        loaded_model,
        backend,
        tokenizer,
        fingerprint,
        engine_identity,
    })
}

#[cfg(unix)]
fn preload_llama_backend(
    model_id: &str,
    preload: PreloadModelConfig,
    artifact: &ValidatedArtifact,
    runtime_config: &RuntimeConfig,
) -> Result<(EmbedBackend, EngineIdentity, LoadedModel), ModuleError> {
    use worker_host::{WorkerEngine, WorkerHostConfig};

    let worker_bin = preload.worker_bin.ok_or_else(|| {
        ModuleError::Config(format!(
            "llama preload for model '{model_id}' requires worker_bin"
        ))
    })?;
    let runtime_dir = preload
        .worker_runtime_dir
        .unwrap_or_else(|| env::temp_dir().join("synapse-workers"));
    let mut config = WorkerHostConfig::new(worker_bin, runtime_dir);
    config.worker_id = format!("synapse-{model_id}");
    config.pooling = parse_pooling(preload.pooling.as_deref().unwrap_or("mean"))?;
    config.normalize = preload.normalize.unwrap_or(true);
    let artifact = artifact.clone();
    let runtime_config = runtime_config.clone();
    let model_id_for_error = model_id.to_string();
    std::thread::spawn(move || {
        let mut engine = WorkerEngine::new(config).map_err(|error| {
            ModuleError::Engine(format!(
                "create llama worker engine for '{model_id_for_error}': {error}"
            ))
        })?;
        let engine_identity = engine.identity();
        let loaded_model = engine.load(&artifact, &runtime_config).map_err(|error| {
            ModuleError::Engine(format!(
                "preload llama model '{model_id_for_error}' failed: {}",
                error.message
            ))
        })?;
        Ok((
            EmbedBackend::Llama(Arc::new(Mutex::new(engine))),
            engine_identity,
            loaded_model,
        ))
    })
    .join()
    .map_err(|_| ModuleError::Engine(format!("llama preload thread for '{model_id}' panicked")))?
}

#[cfg(not(unix))]
fn preload_llama_backend(
    model_id: &str,
    _preload: PreloadModelConfig,
    _artifact: &ValidatedArtifact,
    _runtime_config: &RuntimeConfig,
) -> Result<(EmbedBackend, EngineIdentity, LoadedModel), ModuleError> {
    Err(ModuleError::Config(format!(
        "llama worker preload for model '{model_id}' is only available on unix"
    )))
}

#[async_trait]
impl ModuleHandler for SynapseHandler {
    async fn on_hello_ack(&self, ack: &ModuleHelloAckBody) {
        if self.state().is_some() {
            return;
        }
        let state = self
            .initialize(ack)
            .unwrap_or_else(|error| panic!("synapse boot failed after HELLO_ACK: {error}"));
        let _ = self.inner.state.set(state);
    }

    async fn on_bind(&self, _req: &RouteBindRequest) -> BindDecision {
        if self.state().is_some() {
            BindDecision::accept()
        } else {
            BindDecision::reject(
                "module_not_initialized",
                "synapse has not completed HELLO_ACK initialization",
            )
        }
    }

    async fn health(&self) -> HealthReport {
        let Some(state) = self.state() else {
            return HealthReport::ok();
        };
        HealthReport {
            status: subc_client_rs::HealthStatus::Ok,
            detail: Some("ok".to_string()),
            metrics: Some(
                serde_json::to_value(&state.health).expect("module health should serialize"),
            ),
        }
    }

    async fn handle(&self, _ctx: RequestCtx, body: Vec<u8>) -> HandlerOutcome {
        let Some(state) = self.state() else {
            return channel_error(
                "module_not_initialized",
                "synapse has not completed HELLO_ACK initialization",
            );
        };

        let envelope: MethodEnvelope = match serde_json::from_slice(&body) {
            Ok(envelope) => envelope,
            Err(error) => {
                return channel_error(
                    "invalid_request",
                    format!("route request body is not decodable: {error}"),
                )
            }
        };

        dispatch_request(state, envelope).await
    }
}

async fn dispatch_request(state: Arc<ModuleState>, request: MethodEnvelope) -> HandlerOutcome {
    match request.method.as_str() {
        "models.list" => match state.store.catalog_snapshot() {
            Ok(snapshot) => result_outcome(models_list_payload(&state, snapshot)),
            Err(error) => channel_error("store_failure", error.to_string()),
        },
        "embed.query" => embed_query(state, request.params).await,
        "embed.batch" => embed_batch(state, request.params).await,
        "embed.result" => embed_result(state, request.params).await,
        "rerank.score" | "microllm.oneshot" => result_outcome(error_payload(
            &state,
            WireOperationError::from_stable(
                StableError::probe_required(),
                format!(
                    "{} requires a completed probe before this scaffold can serve requests",
                    request.method
                ),
            ),
        )),
        "model.load" => result_outcome(error_payload(
            &state,
            WireOperationError::from_stable(
                StableError::probe_required(),
                "model.load is disabled for v1-dev; configure startup preloads instead",
            ),
        )),
        "cache.pin" => cache_pin(state, request.params).await,
        "cache.gc" => cache_gc(state, request.params).await,
        "probe.start" => result_outcome(error_payload(
            &state,
            WireOperationError::not_implemented(format!(
                "{} is reserved in the scaffold but not implemented yet",
                request.method
            )),
        )),
        "model.status" | "probe.status" | "aliases.check_index" | "admission.status" => {
            result_outcome(error_payload(
                &state,
                WireOperationError::not_implemented(format!(
                    "{} is not implemented in the wire-alive scaffold",
                    request.method
                )),
            ))
        }
        other => channel_error(
            "unknown_method",
            format!("unknown method '{other}' for synapse management surface"),
        ),
    }
}

async fn embed_query(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let params: EmbedQueryParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid embed.query params: {error}"),
            )
        }
    };
    let table_epoch = table_epoch(&state);
    let model = match state.runtime.resolve_model(params.model.as_deref()) {
        Ok(model) => model,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };
    if let Err(error) = check_fingerprint_constraints(
        &model,
        table_epoch,
        params.target_fingerprint.as_deref(),
        params.required_fingerprint.as_deref(),
        params.allow_equivalent,
        params.required_epoch,
    ) {
        return result_outcome(error_payload(&state, error));
    }

    let request_bytes = request_bytes_for_texts([params.text.as_str()]);
    let _admission = match state.runtime.admit_inline(
        QueueClass::Interactive,
        request_bytes,
        params.deadline_ms,
        params.max_queue_ms,
    ) {
        Ok(admission) => admission,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };
    let tokenized = match model.tokenizer.tokenize_batch([params.text.as_str()]) {
        Ok(tokenized) => tokenized,
        Err(error) => {
            return result_outcome(error_payload(
                &state,
                WireOperationError::from_stable(StableError::artifact_invalid(), error.to_string()),
            ))
        }
    };
    let ids = vec![params.id.unwrap_or_else(|| "query".to_string())];
    embed_tokenized(state, model, ids, tokenized, table_epoch).await
}

async fn embed_batch(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let params: EmbedBatchParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid embed.batch params: {error}"),
            )
        }
    };
    let items = match batch_items(params.items, params.texts) {
        Ok(items) => items,
        Err(message) => return channel_error("invalid_request", message),
    };
    if items.is_empty() {
        return channel_error("invalid_request", "embed.batch requires at least one item");
    }

    let table_epoch = table_epoch(&state);
    let model = match state.runtime.resolve_model(params.model.as_deref()) {
        Ok(model) => model,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };
    if let Err(error) = check_fingerprint_constraints(
        &model,
        table_epoch,
        params.target_fingerprint.as_deref(),
        params.required_fingerprint.as_deref(),
        params.allow_equivalent,
        params.required_epoch,
    ) {
        return result_outcome(error_payload(&state, error));
    }

    let text_refs = items
        .iter()
        .map(|item| item.text.as_str())
        .collect::<Vec<_>>();
    let request_bytes = request_bytes_for_texts(text_refs.iter().copied());
    let tokenized = match model.tokenizer.tokenize_batch(text_refs) {
        Ok(tokenized) => tokenized,
        Err(error) => {
            return result_outcome(error_payload(
                &state,
                WireOperationError::from_stable(StableError::artifact_invalid(), error.to_string()),
            ))
        }
    };
    let total_tokens = tokenized
        .real_token_counts
        .iter()
        .map(|tokens| u64::from(*tokens))
        .sum::<u64>();
    let ids = items.into_iter().map(|item| item.id).collect::<Vec<_>>();

    if ids.len() > state.runtime.inline.max_items || total_tokens > state.runtime.inline.max_tokens
    {
        return submit_embed_batch_job(
            state,
            params.request_key,
            EmbedBatchJobWork {
                model,
                ids,
                tokenized,
                table_epoch,
                request_bytes,
                total_tokens,
            },
        )
        .await;
    }

    let _admission = match state.runtime.admit_inline(
        QueueClass::Bulk,
        request_bytes,
        params.deadline_ms,
        params.max_queue_ms,
    ) {
        Ok(admission) => admission,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };
    embed_tokenized(state, model, ids, tokenized, table_epoch).await
}

async fn submit_embed_batch_job(
    state: Arc<ModuleState>,
    request_key: Option<String>,
    work: EmbedBatchJobWork,
) -> HandlerOutcome {
    let Some(request_key) = request_key.filter(|key| !key.trim().is_empty()) else {
        return channel_error(
            "invalid_request",
            "job-shaped embed.batch requires a non-empty request_key",
        );
    };
    let now = now_ms();
    let admission = match state.store.admit_job(
        &request_key,
        "embed.batch",
        state.module_generation,
        &json!({
            "model": work.model.model_id.clone(),
            "items": work.ids.len(),
            "request_bytes": work.request_bytes,
            "total_tokens": work.total_tokens,
        }),
        now,
        state.runtime.jobs.ttl_ms,
    ) {
        Ok(admission) => admission,
        Err(error) => return channel_error("store_failure", error.to_string()),
    };

    let record = admission.record().clone();
    if matches!(admission, JobAdmission::Admitted(_)) {
        let task_state = Arc::clone(&state);
        let task_job_id = record.job_id.clone();
        tokio::spawn(async move {
            execute_embed_batch_job(task_state, task_job_id, work).await;
        });
    }

    result_outcome(job_status_payload(&state, &record))
}

async fn execute_embed_batch_job(state: Arc<ModuleState>, job_id: String, work: EmbedBatchJobWork) {
    if !matches!(
        state
            .store
            .mark_job_running(&job_id, state.module_generation, now_ms()),
        Ok(true)
    ) {
        return;
    }

    let vectors = match execute_embedding_quanta(
        &state.runtime,
        &work.model,
        work.tokenized.batch.clone(),
        work.total_tokens,
        work.request_bytes,
    )
    .await
    {
        Ok(vectors) => vectors,
        Err(error) => {
            fail_job_with_wire_error(&state, &job_id, true, error);
            return;
        }
    };
    if vectors.len() != work.ids.len() {
        fail_job_with_wire_error(
            &state,
            &job_id,
            true,
            WireOperationError::from_stable(
                StableError::engine_crashed(None),
                format!(
                    "engine returned {} vectors for {} requested job items",
                    vectors.len(),
                    work.ids.len()
                ),
            ),
        );
        return;
    }

    let (summary, pages) = match embed_result_pages(
        &state,
        &work.model,
        work.ids,
        vectors,
        work.tokenized,
        work.table_epoch,
        &job_id,
    ) {
        Ok(pages) => pages,
        Err(error) => {
            fail_job_with_wire_error(&state, &job_id, false, error);
            return;
        }
    };
    if let Err(error) = state
        .store
        .complete_job(&job_id, &summary, &pages, now_ms())
    {
        fail_job_with_wire_error(
            &state,
            &job_id,
            true,
            WireOperationError::from_stable(
                StableError::engine_crashed(Some(100)),
                format!("store completed job pages: {error}"),
            ),
        );
    }
}

async fn execute_embedding_quanta(
    runtime: &RuntimeState,
    model: &EmbeddingModel,
    batch: TokenBatch,
    total_tokens: u64,
    request_bytes: u64,
) -> Result<Vectors, WireOperationError> {
    let mut scheduler = LaneScheduler::new(SchedulerConfig {
        byte_budget: request_bytes.max(1),
        bulk_quantum_tokens: runtime.jobs.bulk_quantum_tokens.max(1),
        max_concurrent_workers: 1,
        default_execution_ms: runtime.inline.estimated_execution_ms,
        ..SchedulerConfig::default()
    });
    scheduler
        .admit(
            &SystemClock,
            WorkRequest {
                queue_class: QueueClass::Bulk,
                deadline_ms: None,
                max_queue_ms: runtime.inline.max_queue_ms,
                request_bytes,
                token_cost: total_tokens.max(1),
                estimated_execution_ms: runtime.inline.estimated_execution_ms,
                payload: (),
            },
        )
        .map_err(|rejection| WireOperationError::from_stable(rejection.error, rejection.reason))?;

    let mut all_vectors = Vec::new();
    let mut cursor = 0_usize;
    while cursor < batch.items.len() {
        let Some(dispatch) = scheduler.next_dispatch(&SystemClock) else {
            tokio::task::yield_now().await;
            continue;
        };
        let mut quantum_tokens = 0_u64;
        let mut quantum_items = Vec::new();
        while cursor < batch.items.len() {
            let item_tokens = batch.items[cursor].len().max(1) as u64;
            if !quantum_items.is_empty()
                && quantum_tokens.saturating_add(item_tokens) > dispatch.quantum_tokens
            {
                break;
            }
            quantum_tokens = quantum_tokens.saturating_add(item_tokens);
            quantum_items.push(batch.items[cursor].clone());
            cursor += 1;
        }
        let mut vectors = execute_embedding(
            runtime,
            model,
            TokenBatch {
                items: quantum_items,
            },
        )
        .await?;
        scheduler.complete_dispatch(&dispatch);
        all_vectors.append(&mut vectors);
    }
    Ok(all_vectors)
}

fn embed_result_pages(
    state: &ModuleState,
    model: &EmbeddingModel,
    ids: Vec<String>,
    vectors: Vectors,
    tokenized: TokenizedBatch,
    table_epoch: u64,
    job_id: &str,
) -> Result<(Value, Vec<Vec<u8>>), WireOperationError> {
    let dims = vectors.first().map(Vec::len).unwrap_or(0) as u32;
    let response_vectors = ids
        .into_iter()
        .zip(vectors)
        .map(|(id, vector)| EmbedVector { id, vector })
        .collect::<Vec<_>>();
    let page_ranges = page_ranges(
        &response_vectors,
        &tokenized.real_token_counts,
        state.runtime.jobs.result_page_bytes.max(1),
    );
    let page_count = page_ranges.len() as u32;
    let mut pages = Vec::with_capacity(page_ranges.len());
    for (page_index, (start, end)) in page_ranges.iter().copied().enumerate() {
        let payload = EmbedResponsePayload {
            vectors: response_vectors[start..end].to_vec(),
            real_token_counts: tokenized.real_token_counts[start..end].to_vec(),
            truncation_disclosures: tokenized.disclosures[start..end].to_vec(),
        };
        let envelope = ResponseEnvelope {
            fingerprint: model.fingerprint.clone(),
            table_epoch,
            dims,
            provenance: ResponseProvenance {
                engine: model.engine_identity.clone(),
            },
            module_generation: state.module_generation,
            equivalent_to: Vec::new(),
            payload,
        };
        let mut value = serde_json::to_value(envelope).expect("embed job page serializes");
        if let Value::Object(map) = &mut value {
            map.insert("job_id".to_string(), Value::String(job_id.to_string()));
            map.insert(
                "state".to_string(),
                Value::String(JOB_STATE_DONE.to_string()),
            );
            map.insert("page".to_string(), Value::from(page_index as u64));
            map.insert("page_count".to_string(), Value::from(page_count));
            map.insert(
                "job_module_generation".to_string(),
                Value::from(state.module_generation),
            );
        }
        pages.push(serde_json::to_vec(&value).map_err(|error| {
            WireOperationError::from_stable(
                StableError::artifact_invalid(),
                format!("serialize embed job page: {error}"),
            )
        })?);
    }
    Ok((
        json!({
            "job_id": job_id,
            "state": JOB_STATE_DONE,
            "page_count": page_count,
            "dims": dims,
            "module_generation": state.module_generation,
        }),
        pages,
    ))
}

fn page_ranges(
    vectors: &[EmbedVector],
    token_counts: &[u32],
    max_bytes: usize,
) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = 0_usize;
    while start < vectors.len() {
        let mut end = start;
        let mut bytes = 0_usize;
        while end < vectors.len() {
            let item_bytes = vectors[end]
                .vector
                .len()
                .saturating_mul(std::mem::size_of::<f32>())
                .saturating_add(vectors[end].id.len())
                .saturating_add(
                    usize::try_from(token_counts.get(end).copied().unwrap_or(0)).unwrap_or(0),
                )
                .saturating_add(256);
            if end > start && bytes.saturating_add(item_bytes) > max_bytes {
                break;
            }
            bytes = bytes.saturating_add(item_bytes);
            end += 1;
        }
        ranges.push((start, end.max(start + 1).min(vectors.len())));
        start = ranges.last().map(|(_, end)| *end).unwrap_or(vectors.len());
    }
    ranges
}

fn fail_job_with_wire_error(
    state: &ModuleState,
    job_id: &str,
    transient: bool,
    error: WireOperationError,
) {
    let _ = state.store.fail_job(
        job_id,
        transient,
        &serde_json::to_value(error).expect("wire error serializes"),
        now_ms(),
    );
}

fn job_status_payload(state: &ModuleState, record: &JobRecord) -> Value {
    let mut payload = json!({
        "module_generation": state.module_generation,
        "job_id": record.job_id,
        "state": record.state,
        "request_key": record.request_key,
    });
    if let Value::Object(map) = &mut payload {
        if record.state == JOB_STATE_DONE {
            map.insert("page_count".to_string(), Value::from(record.page_count));
        }
        if record.state == JOB_STATE_FAILED_TRANSIENT || record.state == JOB_STATE_FAILED_PERMANENT
        {
            map.insert(
                "error".to_string(),
                record.error_json.clone().unwrap_or_else(|| {
                    serde_json::to_value(WireOperationError::from_stable(
                        StableError::engine_crashed(Some(100)),
                        "durable job failed without a stored typed error",
                    ))
                    .expect("fallback error serializes")
                }),
            );
        }
    }
    payload
}

async fn embed_result(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let params: EmbedResultParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid embed.result params: {error}"),
            )
        }
    };
    if let Err(error) = state.store.purge_expired_jobs(now_ms()) {
        return channel_error("store_failure", error.to_string());
    }
    let record = match state.store.get_job(&params.job_id) {
        Ok(Some(record)) => record,
        Ok(None) => return channel_error("invalid_request", "unknown or expired job_id"),
        Err(error) => return channel_error("store_failure", error.to_string()),
    };
    match record.state.as_str() {
        JOB_STATE_QUEUED | JOB_STATE_RUNNING => result_outcome(job_status_payload(&state, &record)),
        JOB_STATE_FAILED_TRANSIENT | JOB_STATE_FAILED_PERMANENT => {
            result_outcome(job_status_payload(&state, &record))
        }
        JOB_STATE_DONE => {
            let page = params.page.unwrap_or(0);
            if page >= record.page_count {
                return channel_error(
                    "invalid_request",
                    format!(
                        "embed.result page {page} is outside available page_count {}",
                        record.page_count
                    ),
                );
            }
            let bytes = match state.store.get_job_page(&record.job_id, page) {
                Ok(Some(bytes)) => bytes,
                Ok(None) => return channel_error("store_failure", "job result page is missing"),
                Err(error) => return channel_error("store_failure", error.to_string()),
            };
            let mut value: Value = match serde_json::from_slice(&bytes) {
                Ok(value) => value,
                Err(error) => return channel_error("store_failure", error.to_string()),
            };
            if let Value::Object(map) = &mut value {
                map.insert(
                    "module_generation".to_string(),
                    Value::from(state.module_generation),
                );
                map.insert(
                    "job_module_generation".to_string(),
                    Value::from(record.module_generation),
                );
            }
            result_outcome(value)
        }
        other => channel_error(
            "store_failure",
            format!("job {} has unknown state {other}", record.job_id),
        ),
    }
}

async fn embed_tokenized(
    state: Arc<ModuleState>,
    model: Arc<EmbeddingModel>,
    ids: Vec<String>,
    tokenized: TokenizedBatch,
    table_epoch: u64,
) -> HandlerOutcome {
    let vectors = match execute_embedding(&state.runtime, &model, tokenized.batch).await {
        Ok(vectors) => vectors,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };
    if vectors.len() != ids.len() {
        return result_outcome(error_payload(
            &state,
            WireOperationError::from_stable(
                StableError::engine_crashed(None),
                format!(
                    "engine returned {} vectors for {} requested items",
                    vectors.len(),
                    ids.len()
                ),
            ),
        ));
    }
    let dims = vectors.first().map(Vec::len).unwrap_or(0) as u32;
    let response_vectors = ids
        .into_iter()
        .zip(vectors)
        .map(|(id, vector)| EmbedVector { id, vector })
        .collect::<Vec<_>>();
    let payload = EmbedResponsePayload {
        vectors: response_vectors,
        real_token_counts: tokenized.real_token_counts,
        truncation_disclosures: tokenized.disclosures,
    };
    let envelope = ResponseEnvelope {
        fingerprint: model.fingerprint.clone(),
        table_epoch,
        dims,
        provenance: ResponseProvenance {
            engine: model.engine_identity.clone(),
        },
        module_generation: state.module_generation,
        equivalent_to: Vec::new(),
        payload,
    };
    result_outcome(serde_json::to_value(envelope).expect("embed envelope should serialize"))
}

async fn execute_embedding(
    runtime: &RuntimeState,
    model: &EmbeddingModel,
    batch: TokenBatch,
) -> Result<Vectors, WireOperationError> {
    let permit = runtime
        .execution
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| {
            WireOperationError::from_stable(
                StableError::queue_full(Some(100)),
                "inline embedding executor is closed",
            )
        })?;
    match &model.backend {
        EmbedBackend::Ort(engine) => {
            let engine = Arc::clone(engine);
            let loaded_model = model.loaded_model.clone();
            tokio::task::spawn_blocking(move || {
                let _permit = permit;
                let engine = engine.lock().map_err(|_| EngineError {
                    stage: EngineErrorStage::Inference,
                    risk_class: synapse_core::EngineRiskClass::AbortSafe,
                    message: "ORT engine mutex was poisoned during inference".to_string(),
                    retry_after_ms: Some(100),
                    safe_to_retry_same_request: true,
                })?;
                engine.embed_batch(&loaded_model, batch)
            })
            .await
            .map_err(|error| {
                WireOperationError::from_stable(
                    StableError::engine_crashed(Some(100)),
                    format!("embedding worker join failed: {error}"),
                )
            })?
            .map_err(engine_error_to_wire)
        }
        #[cfg(unix)]
        EmbedBackend::Llama(engine) => {
            let engine = Arc::clone(engine);
            let loaded_model = model.loaded_model.clone();
            tokio::task::spawn_blocking(move || {
                let _permit = permit;
                let engine = engine.lock().map_err(|_| EngineError {
                    stage: EngineErrorStage::Inference,
                    risk_class: synapse_core::EngineRiskClass::AbortCapable,
                    message: "llama worker engine mutex was poisoned during inference".to_string(),
                    retry_after_ms: Some(100),
                    safe_to_retry_same_request: true,
                })?;
                engine.embed_batch(&loaded_model, batch)
            })
            .await
            .map_err(|error| {
                WireOperationError::from_stable(
                    StableError::engine_crashed(Some(100)),
                    format!("embedding worker join failed: {error}"),
                )
            })?
            .map_err(engine_error_to_wire)
        }
    }
}

fn engine_error_to_wire(error: EngineError) -> WireOperationError {
    WireOperationError::from_stable(
        StableError::engine_crashed(error.retry_after_ms),
        error.message,
    )
}

fn batch_items(
    items: Vec<EmbedBatchItemParam>,
    texts: Vec<String>,
) -> Result<Vec<EmbedBatchItem>, String> {
    if !items.is_empty() && !texts.is_empty() {
        return Err("embed.batch accepts either items or texts, not both".to_string());
    }
    if !items.is_empty() {
        return Ok(items
            .into_iter()
            .enumerate()
            .map(|(index, item)| match item {
                EmbedBatchItemParam::Object { id, text } => EmbedBatchItem { id, text },
                EmbedBatchItemParam::Text(text) => EmbedBatchItem {
                    id: index.to_string(),
                    text,
                },
            })
            .collect());
    }
    Ok(texts
        .into_iter()
        .enumerate()
        .map(|(index, text)| EmbedBatchItem {
            id: index.to_string(),
            text,
        })
        .collect())
}

fn check_fingerprint_constraints(
    model: &EmbeddingModel,
    table_epoch: u64,
    target_fingerprint: Option<&str>,
    required_fingerprint: Option<&str>,
    allow_equivalent: bool,
    required_epoch: Option<u64>,
) -> Result<(), WireOperationError> {
    if let Some(required_epoch) = required_epoch {
        if required_epoch > table_epoch {
            return Err(WireOperationError::from_stable(
                StableError::migration_required(),
                format!(
                    "request requires alias table epoch {required_epoch}, but module is at epoch {table_epoch}"
                ),
            ));
        }
    }
    let requested = required_fingerprint.or(target_fingerprint);
    if let Some(requested) = requested {
        if requested != model.fingerprint.0 {
            let alias_table = AliasTable {
                table_epoch,
                rows: Vec::new(),
            };
            let equivalent = allow_equivalent
                && alias_table
                    .equivalent_fingerprints_at(&model.fingerprint, table_epoch)
                    .iter()
                    .any(|fingerprint| fingerprint.0 == requested);
            if !equivalent {
                return Err(WireOperationError::from_stable(
                    StableError::substitution_rejected(),
                    format!(
                        "requested fingerprint {requested} does not match loaded model fingerprint {}",
                        model.fingerprint.0
                    ),
                ));
            }
        }
    }
    Ok(())
}

async fn cache_pin(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let params: CachePinParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid cache.pin params: {error}"),
            )
        }
    };
    let module_id = params
        .module_id
        .as_deref()
        .unwrap_or(&state.module_id)
        .to_string();
    let result: Result<ModelCacheMeta, ModelCacheError> =
        if let Some(source_url) = params.source_url {
            state.model_cache.ingest(ModelCacheIngest {
                source_url,
                expected_digest: params.expected_digest.or(params.digest),
                format: params.format.unwrap_or_else(|| "unknown".to_string()),
                tokenizer_path: params.tokenizer_path,
                pin_module_id: Some(module_id),
            })
        } else if let Some(digest) = params.digest {
            state.model_cache.pin(&digest, &module_id)
        } else {
            return channel_error(
                "invalid_request",
                "cache.pin requires either source_url or digest",
            );
        };

    match result {
        Ok(meta) => result_outcome(json!({
            "module_generation": state.module_generation,
            "cache_root": state.model_cache.root().to_string_lossy(),
            "artifact": meta,
        })),
        Err(error) => result_outcome(error_payload(&state, cache_error_to_wire(error))),
    }
}

async fn cache_gc(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let params: CacheGcParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid cache.gc params: {error}"),
            )
        }
    };
    let now = now_ms();
    let grace_ms = params.grace_ms.unwrap_or(60_000);
    let result: Result<Vec<CacheGcOutcome>, ModelCacheError> = if let Some(digest) = params.digest {
        state
            .model_cache
            .gc_digest(&digest, &state.module_id, now, grace_ms)
            .map(|outcome| vec![outcome])
    } else {
        state.model_cache.gc_all(&state.module_id, now, grace_ms)
    };
    match result {
        Ok(outcomes) => result_outcome(json!({
            "module_generation": state.module_generation,
            "outcomes": outcomes,
        })),
        Err(error) => result_outcome(error_payload(&state, cache_error_to_wire(error))),
    }
}

fn cache_error_to_wire(error: ModelCacheError) -> WireOperationError {
    WireOperationError::from_stable(StableError::artifact_invalid(), error.to_string())
}

fn models_list_payload(state: &ModuleState, snapshot: CatalogSnapshot) -> Value {
    let mut models = snapshot.models;
    models.extend(state.runtime.catalog_entries());
    json!({
        "module_generation": state.module_generation,
        "table_epoch": snapshot.table_epoch,
        "models": models,
        "alias_rows": snapshot.alias_rows,
    })
}

fn error_payload(state: &ModuleState, error: WireOperationError) -> Value {
    json!({
        "module_generation": state.module_generation,
        "error": error,
    })
}

fn result_outcome(result: Value) -> HandlerOutcome {
    match serde_json::to_vec(&json!({ "result": result })) {
        Ok(body) => HandlerOutcome::Response(body),
        Err(error) => channel_error("encode_failed", error.to_string()),
    }
}

fn channel_error(code: impl Into<String>, message: impl Into<String>) -> HandlerOutcome {
    HandlerOutcome::Error {
        code: code.into(),
        message: message.into(),
    }
}

fn resolve_storage_descriptor(
    ack_storage: &Option<Value>,
    module_id: &str,
) -> Result<StorageDescriptor, ModuleError> {
    if let Some(value) = ack_storage {
        return serde_json::from_value(value.clone()).map_err(ModuleError::Json);
    }

    let path = sqlite_store_path(&std::env::temp_dir().to_string_lossy(), module_id);
    Ok(StorageDescriptor {
        module_id: module_id.to_string(),
        storage_namespace: "default".to_string(),
        isolation: Isolation::Module,
        backend: StorageBackend::Sqlite { path },
    })
}

fn management_operations() -> Vec<ManagementOperation> {
    use ManagementOperationKind::{Mutate, Query};

    let op = |name: &str, kind| ManagementOperation {
        name: name.to_string(),
        kind,
    };

    vec![
        op("embed.query", Query),
        op("embed.batch", Query),
        op("embed.result", Query),
        op("rerank.score", Query),
        op("microllm.oneshot", Query),
        op("model.load", Mutate),
        op("model.status", Query),
        op("models.list", Query),
        op("probe.start", Mutate),
        op("probe.status", Query),
        op("aliases.check_index", Query),
        op("cache.pin", Mutate),
        op("cache.gc", Mutate),
        op("admission.status", Query),
    ]
}

fn manifest(module_id: &str) -> ModuleManifest {
    ModuleManifest {
        module_id: module_id.to_string(),
        module_version: env!("CARGO_PKG_VERSION").to_string(),
        protocol_ver: PROTOCOL_VERSION,
        trust_tier: TrustTier::FirstParty,
        provides: vec![ProviderRole::ManagementSurface {
            operations: management_operations(),
            config_schema: json!({ "type": "object" }),
            observability: Vec::new(),
            identity_scope: vec![IdentityScope::Project, IdentityScope::Session],
        }],
        consumes: Vec::new(),
        scheduled_tasks: Vec::new(),
        bindings: Bindings {
            storage: StorageBinding {
                kind: StorageKind::Sqlite,
                scope: StorageScope::Project,
                owns_schema: true,
            },
            vault_grants: Vec::new(),
            identity: IdentityBinding {
                requires: vec![IdentityScope::Project],
                optional: vec![IdentityScope::Session],
            },
        },
    }
}

fn load_module_config() -> Result<ModuleConfig, ModuleError> {
    if let Ok(json) = env::var("SYNAPSE_CONFIG_JSON") {
        return serde_json::from_str(&strip_json_comments(&json)).map_err(ModuleError::Json);
    }
    if let Ok(json) = env::var("SYNAPSE_PRELOAD_MODELS") {
        let preload_models = serde_json::from_str(&strip_json_comments(&json))?;
        return Ok(ModuleConfig {
            preload_models,
            inline: InlineConfig::default(),
            jobs: JobConfig::default(),
        });
    }
    if let Ok(path) = env::var("SYNAPSE_CONFIG") {
        return load_module_config_file(Path::new(&path));
    }
    if let Some(home) = env::var_os("HOME") {
        let path = PathBuf::from(home)
            .join(".config")
            .join("cortexkit")
            .join("synapse.jsonc");
        if path.exists() {
            return load_module_config_file(&path);
        }
    }
    Ok(ModuleConfig::default())
}

fn load_module_config_file(path: &Path) -> Result<ModuleConfig, ModuleError> {
    let contents = fs::read_to_string(path)
        .map_err(|error| ModuleError::Config(format!("read {}: {error}", path.display())))?;
    serde_json::from_str(&strip_json_comments(&contents)).map_err(ModuleError::Json)
}

fn strip_json_comments(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    let mut in_string = false;
    let mut escaped = false;
    while let Some(ch) = chars.next() {
        if in_string {
            output.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        if ch == '"' {
            in_string = true;
            output.push(ch);
            continue;
        }
        if ch == '/' {
            match chars.peek().copied() {
                Some('/') => {
                    let _ = chars.next();
                    for next in chars.by_ref() {
                        if next == '\n' {
                            output.push('\n');
                            break;
                        }
                    }
                    continue;
                }
                Some('*') => {
                    let _ = chars.next();
                    let mut previous = '\0';
                    for next in chars.by_ref() {
                        if previous == '*' && next == '/' {
                            break;
                        }
                        previous = next;
                    }
                    continue;
                }
                _ => {}
            }
        }
        output.push(ch);
    }
    output
}

fn parse_pooling(value: &str) -> Result<WorkerPooling, ModuleError> {
    WorkerPooling::parse(value).ok_or_else(|| {
        ModuleError::Config(format!(
            "unsupported pooling '{value}'; expected mean, cls, or last"
        ))
    })
}

fn profile_pooling(pooling: WorkerPooling) -> PoolingStrategy {
    match pooling {
        WorkerPooling::Mean => PoolingStrategy::Mean,
        WorkerPooling::Cls => PoolingStrategy::Cls,
        WorkerPooling::Last => PoolingStrategy::LastToken,
    }
}

fn normalize_digest(value: &str) -> String {
    if value.starts_with("sha256:") {
        value.to_string()
    } else {
        format!("sha256:{value}")
    }
}

fn sha256_file(path: &Path) -> Result<String, ModuleError> {
    let mut file = fs::File::open(path)
        .map_err(|error| ModuleError::Config(format!("hash {}: {error}", path.display())))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| ModuleError::Config(format!("hash {}: {error}", path.display())))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn request_bytes_for_texts<'text>(texts: impl IntoIterator<Item = &'text str>) -> u64 {
    texts.into_iter().map(|text| text.len() as u64 + 128).sum()
}

fn table_epoch(state: &ModuleState) -> u64 {
    state
        .store
        .catalog_snapshot()
        .map(|snapshot| snapshot.table_epoch)
        .unwrap_or(0)
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}
