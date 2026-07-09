#![forbid(unsafe_code)]

use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

mod store;
pub mod worker_host;

use cortexkit_store_types::{sqlite_store_path, Isolation, StorageBackend, StorageDescriptor};
use reqwest::Url;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use store::{
    CatalogSnapshot, CertificationRow, JobAdmission, JobRecord, KnobAssignmentRow,
    ModelAssetLocator, ModelCatalogEntry, PerfRow, StoredModelConfig, SynapseStore,
    SynapseStoreError, JOB_STATE_DONE, JOB_STATE_FAILED_PERMANENT, JOB_STATE_FAILED_TRANSIENT,
    JOB_STATE_QUEUED, JOB_STATE_RUNNING,
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
    FlashAttentionSetting, GenerateEngine, GenerateOutput, GenerateRequest, LaneBudgetSnapshot,
    LaneScheduler, LoadedModel, MachineProfile, ModelCache, ModelCacheError, ModelCacheIngest,
    ModelCacheMeta, NormalizationMode, NumericDType, NumericProfile, NumericProfileId,
    PoolingStrategy, QueueClass, RerankEngine, RerankRequest, ResponseEnvelope, ResponseProvenance,
    RuntimeConfig, SanitizedTokenizer, SchedulerConfig, StableError, SystemMachineProfileCollector,
    ThreadPolicyClass, TokenBatch, TokenizationError, TokenizedBatch, TokenizerConfig,
    TruncationDisclosure, ValidatedArtifact, Vectors, WorkRequest, WorkerPooling,
};
use synapse_engine_ort::OrtEmbedEngine;
use thiserror::Error;
use tokio::sync::{Notify, Semaphore};

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
const DEFAULT_PROBE_MEAN_COSINE_THRESHOLD: f64 = 0.999;
const DEFAULT_PROBE_WORST_DECILE_RANK_OVERLAP_THRESHOLD: f64 = 0.9;
const DEFAULT_PROBE_ANE_PLACEMENT_THRESHOLD: f64 = 0.9;
const RERANK_PROBE_PEARSON_THRESHOLD: f64 = 0.999;
const GENERATE_PROBE_MIN_LABEL_MATCHES: usize = 7;
const BALANCED_QUIET_MIN_THROUGHPUT_RATIO: f64 = 0.5;
const PROBE_PERF_BATCH_TOKEN_BUDGET: usize = 1_024;
const PROBE_PERF_TARGET_TOTAL_TOKENS: u64 = 4_096;
const PROBE_PERF_MIN_BATCH_SAMPLES: usize = 3;
const PROBE_PERF_SINGLE_SAMPLES: usize = 20;
const SYNAPSE_OS_BUILD_OVERRIDE_ENV: &str = "SYNAPSE_OS_BUILD_OVERRIDE";

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

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum PerfKnob {
    Performance,
    #[default]
    Balanced,
    Quiet,
}

impl PerfKnob {
    fn as_str(self) -> &'static str {
        match self {
            Self::Performance => "performance",
            Self::Balanced => "balanced",
            Self::Quiet => "quiet",
        }
    }

    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "performance" => Ok(Self::Performance),
            "balanced" => Ok(Self::Balanced),
            "quiet" => Ok(Self::Quiet),
            other => Err(format!("unknown performance knob '{other}'")),
        }
    }
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
    machine_profile: MachineProfile,
    machine_profile_hash: String,
    runtime: Arc<RuntimeState>,
    model_cache: Arc<ModelCache>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct ModuleHealth {
    status: String,
    module_generation: u64,
    loaded_models: usize,
    machine_profile_hash: String,
    certification_stale: bool,
    performance_stale: bool,
    lanes: Vec<LaneHealth>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct LaneHealth {
    model_id: String,
    fingerprint: Fingerprint,
    certified: bool,
    certification_stale: bool,
    performance_stale: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    worker: Option<worker_host::WorkerHostHealth>,
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
}

#[derive(Clone, Debug, Default, Deserialize)]
struct ModuleConfig {
    #[serde(default)]
    preload_models: Vec<PreloadModelConfig>,
    #[serde(default)]
    inline: InlineConfig,
    #[serde(default)]
    jobs: JobConfig,
    #[serde(default)]
    probe: ProbeConfig,
    #[serde(default)]
    knob: PerfKnob,
    #[serde(default, alias = "dev_alias_admin", alias = "enable_alias_admin")]
    alias_admin_enabled: bool,
    #[serde(default)]
    dev: DevConfig,
}

#[derive(Clone, Debug, Deserialize)]
struct PreloadModelConfig {
    #[serde(default)]
    model_id: Option<String>,
    engine: String,
    #[serde(default, alias = "kind", alias = "capability")]
    task: Option<String>,
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

#[derive(Clone, Debug, Deserialize)]
struct ProbeConfig {
    #[serde(default = "default_probe_mean_cosine_threshold")]
    mean_cosine_threshold: f64,
    #[serde(default = "default_probe_worst_decile_rank_overlap_threshold")]
    worst_decile_rank_overlap_threshold: f64,
    #[serde(default = "default_probe_ane_placement_threshold")]
    ane_placement_threshold: f64,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            mean_cosine_threshold: default_probe_mean_cosine_threshold(),
            worst_decile_rank_overlap_threshold: default_probe_worst_decile_rank_overlap_threshold(
            ),
            ane_placement_threshold: default_probe_ane_placement_threshold(),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
struct DevConfig {
    #[serde(default, alias = "enable_alias_admin")]
    alias_admin_enabled: bool,
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

fn default_probe_mean_cosine_threshold() -> f64 {
    DEFAULT_PROBE_MEAN_COSINE_THRESHOLD
}

fn default_probe_worst_decile_rank_overlap_threshold() -> f64 {
    DEFAULT_PROBE_WORST_DECILE_RANK_OVERLAP_THRESHOLD
}

fn default_probe_ane_placement_threshold() -> f64 {
    DEFAULT_PROBE_ANE_PLACEMENT_THRESHOLD
}

struct RuntimeState {
    inline: InlineConfig,
    jobs: JobConfig,
    probe: ProbeConfig,
    knob: PerfKnob,
    alias_admin_enabled: bool,
    scheduler: Arc<Mutex<InlineScheduler>>,
    execution: Arc<Semaphore>,
    control_loads: Arc<Semaphore>,
    ort_engine: Arc<Mutex<OrtEmbedEngine>>,
    catalog: Arc<Mutex<BTreeMap<String, ModelSlot>>>,
    job_progress: Arc<Mutex<BTreeMap<String, ModelRuntimeState>>>,
}

struct ModelSlot {
    spec: StoredModelConfig,
    loaded: Option<Arc<EmbeddingModel>>,
    state: ModelRuntimeState,
    notify: Arc<Notify>,
    last_cold_load_ms: Option<f64>,
}

#[derive(Clone)]
struct ModelSlotSnapshot {
    spec: StoredModelConfig,
    loaded: Option<Arc<EmbeddingModel>>,
    state: ModelRuntimeState,
    notify: Arc<Notify>,
}

#[derive(Clone, Debug)]
enum ModelRuntimeState {
    Unloaded,
    Resolving,
    Downloading {
        bytes_done: u64,
        bytes_total: Option<u64>,
    },
    Validating,
    Loading,
    Ready,
    Failed(WireOperationError),
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
    task: ModelTask,
    loaded_model: LoadedModel,
    backend: EmbedBackend,
    tokenizer: SanitizedTokenizer,
    numeric_profile_id: NumericProfileId,
    fingerprint: Fingerprint,
    engine_identity: EngineIdentity,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ModelTask {
    Embed,
    Rerank,
    Generate,
}

impl ModelTask {
    fn as_str(self) -> &'static str {
        match self {
            Self::Embed => "embed",
            Self::Rerank => "rerank",
            Self::Generate => "generate",
        }
    }
}

#[derive(Clone)]
enum EmbedBackend {
    Ort(Arc<Mutex<OrtEmbedEngine>>),
    Worker(Arc<Mutex<worker_host::WorkerEngine>>),
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
struct RerankScoreParams {
    #[serde(default, alias = "model_id")]
    model: Option<String>,
    query: String,
    #[serde(default)]
    candidates: Vec<String>,
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

#[derive(Clone, Debug, Serialize)]
struct RerankScorePayload {
    scores: Vec<f32>,
    real_token_counts: Vec<u32>,
    truncation_disclosures: Vec<TruncationDisclosure>,
}

#[derive(Debug, Deserialize)]
struct MicroLlmOneshotParams {
    #[serde(default, alias = "model_id")]
    model: Option<String>,
    prompt: String,
    max_tokens: u32,
    #[serde(default)]
    grammar: Option<String>,
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

#[derive(Clone, Debug, Serialize)]
struct MicroLlmOneshotPayload {
    text: String,
    finish_reason: String,
    n_prompt: usize,
    n_gen: usize,
    real_token_counts: Vec<u32>,
    truncation_disclosures: Vec<TruncationDisclosure>,
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
    alias_table: AliasTable,
    request_bytes: u64,
    total_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct EmbedResultParams {
    job_id: String,
    #[serde(default)]
    page: Option<u32>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ModelLoadFiles {
    model: String,
    tokenizer: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ModelLoadParams {
    source: String,
    #[serde(default)]
    repo: Option<String>,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    path: Option<String>,
    files: ModelLoadFiles,
    #[serde(default)]
    expected_digest: Option<String>,
    engine: String,
    #[serde(default)]
    pooling: Option<String>,
    #[serde(default, alias = "kind", alias = "capability")]
    task: Option<String>,
    #[serde(default)]
    pin: bool,
    #[serde(default)]
    request_key: Option<String>,
    #[serde(default)]
    model_id: Option<String>,
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

#[derive(Debug, Deserialize)]
struct ModelStatusParams {
    #[serde(default)]
    job_id: Option<String>,
    #[serde(default)]
    model_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ModelUnloadParams {
    model_id: String,
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

#[derive(Debug, Deserialize)]
struct ProbeStartParams {
    #[serde(default)]
    request_key: Option<String>,
    #[serde(default)]
    models: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ProbeStatusParams {
    job_id: String,
}

#[derive(Debug, Deserialize)]
struct AliasesCheckIndexParams {
    index_fingerprint: String,
    #[serde(default)]
    provenance_set: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct AliasPairParams {
    #[serde(default, alias = "fingerprint_a")]
    left: Option<String>,
    #[serde(default, alias = "fingerprint_b")]
    right: Option<String>,
    #[serde(default)]
    evidence: Option<Value>,
}

impl AliasPairParams {
    fn fingerprints(self) -> Result<(Fingerprint, Fingerprint, Value), String> {
        let left = self
            .left
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| "alias pair requires fingerprint_a".to_string())?;
        let right = self
            .right
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| "alias pair requires fingerprint_b".to_string())?;
        Ok((
            Fingerprint(left),
            Fingerprint(right),
            self.evidence.unwrap_or_else(|| json!({})),
        ))
    }
}

#[derive(Debug, Deserialize)]
struct ProbeFixture {
    #[serde(default)]
    generation_command: Option<String>,
    items: Vec<ProbeFixtureItem>,
}

#[derive(Debug, Deserialize)]
struct ProbeFixtureItem {
    id: String,
    text: String,
    vector: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct RerankProbeFixture {
    #[serde(default)]
    generation_command: Option<String>,
    items: Vec<RerankProbeItem>,
}

#[derive(Debug, Deserialize)]
struct RerankProbeItem {
    id: String,
    query: String,
    candidates: Vec<String>,
    scores: Vec<f32>,
}

#[derive(Clone, Debug, Serialize)]
struct RerankProbeEvidence {
    pearson: f64,
    pairs: usize,
    requests: usize,
}

#[derive(Debug, Deserialize)]
struct GenerateProbeFixture {
    #[serde(default)]
    generation_command: Option<String>,
    items: Vec<GenerateProbeItem>,
}

#[derive(Debug, Deserialize)]
struct GenerateProbeItem {
    id: String,
    prompt: String,
    expected_label: String,
    max_tokens: u32,
}

#[derive(Clone, Debug, Serialize)]
struct GenerateProbeEvidence {
    label_matches: usize,
    items: usize,
}

#[derive(Clone, Debug, Serialize)]
struct ProbeEvidence {
    mean_cosine: f64,
    rank_overlap: f64,
    worst_decile: f64,
    items: usize,
}

struct ProbeLaneVectors {
    model: Arc<EmbeddingModel>,
    vectors: Vec<Vec<f32>>,
}

struct ProbeModelResult {
    lane_result: Value,
    certified_vectors: Option<Vec<Vec<f32>>>,
}

struct LaneMeasurementRows {
    current_certification: Option<CertificationRow>,
    latest_certification: Option<CertificationRow>,
    certification_stale: bool,
    current_performance: Option<PerfRow>,
    latest_performance: Option<PerfRow>,
    performance_stale: bool,
}

struct PerfBenchResult {
    throughput_tok_s: f64,
    cold_load_ms: f64,
    single_item_latency_p50_ms: f64,
    details: Value,
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
        let config = load_module_config()?;
        let catalog_models = sync_and_load_catalog_models(&store, &config)?;
        let model_cache = Arc::new(ModelCache::new(ModelCache::default_root()?));
        let runtime = Arc::new(RuntimeState::from_catalog(config, catalog_models)?);
        let machine_profile = machine_profile_with_overrides(MachineProfile::collect(
            &SystemMachineProfileCollector,
            runtime.engine_identities(),
        ));
        let machine_profile_hash = machine_profile.hash();
        Ok(Arc::new(ModuleState {
            module_id: self.inner.module_id.clone(),
            store,
            module_generation,
            machine_profile,
            machine_profile_hash,
            runtime,
            model_cache,
        }))
    }
}

impl RuntimeState {
    fn from_catalog(
        config: ModuleConfig,
        models: Vec<StoredModelConfig>,
    ) -> Result<Self, ModuleError> {
        let inline = config.inline;
        let jobs = config.jobs;
        let probe = config.probe;
        let knob = config.knob;
        let alias_admin_enabled = config.alias_admin_enabled || config.dev.alias_admin_enabled;
        let scheduler = Arc::new(Mutex::new(InlineScheduler { in_flight_bytes: 0 }));
        let execution = Arc::new(Semaphore::new(inline.max_concurrent_workers.max(1)));
        let catalog = models
            .into_iter()
            .map(|spec| {
                (
                    spec.model_id.clone(),
                    ModelSlot {
                        spec,
                        loaded: None,
                        state: ModelRuntimeState::Unloaded,
                        notify: Arc::new(Notify::new()),
                        last_cold_load_ms: None,
                    },
                )
            })
            .collect();
        Ok(Self {
            inline,
            jobs,
            probe,
            knob,
            alias_admin_enabled,
            scheduler,
            execution,
            control_loads: Arc::new(Semaphore::new(1)),
            ort_engine: Arc::new(Mutex::new(OrtEmbedEngine::new())),
            catalog: Arc::new(Mutex::new(catalog)),
            job_progress: Arc::new(Mutex::new(BTreeMap::new())),
        })
    }

    fn engine_identities(&self) -> Vec<EngineIdentity> {
        self.catalog
            .lock()
            .map(|catalog| {
                catalog
                    .values()
                    .map(|slot| slot.spec.engine_identity.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    fn default_model_id(&self) -> Option<String> {
        self.catalog
            .lock()
            .ok()
            .and_then(|catalog| catalog.keys().next().cloned())
    }

    fn loaded_models(&self) -> Vec<Arc<EmbeddingModel>> {
        self.catalog
            .lock()
            .map(|catalog| {
                catalog
                    .values()
                    .filter_map(|slot| slot.loaded.clone())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    }

    fn loaded_model_count(&self) -> usize {
        self.catalog
            .lock()
            .map(|catalog| {
                catalog
                    .values()
                    .filter(|slot| slot.loaded.is_some())
                    .count()
            })
            .unwrap_or(0)
    }

    fn catalog_entries(&self) -> Vec<ModelCatalogEntry> {
        self.catalog
            .lock()
            .map(|catalog| {
                catalog
                    .values()
                    .map(|slot| ModelCatalogEntry {
                        model_id: slot.spec.model_id.clone(),
                        state: model_runtime_state_name(&slot.state).to_string(),
                        fingerprints: vec![slot.spec.fingerprint.clone()],
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
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

fn sync_and_load_catalog_models(
    store: &SynapseStore,
    config: &ModuleConfig,
) -> Result<Vec<StoredModelConfig>, ModuleError> {
    let now = now_ms();
    for (index, preload) in config.preload_models.clone().into_iter().enumerate() {
        let spec = build_preload_catalog_model(index, preload, &config.inline, &config.jobs)?;
        store.upsert_model(&spec, now)?;
    }

    let mut normalized = Vec::new();
    for model in store.catalog_models()? {
        let refreshed = normalize_catalog_model(model.clone(), &config.inline, &config.jobs)?;
        if refreshed != model {
            store.upsert_model(&refreshed, now)?;
        }
        normalized.push(refreshed);
    }
    Ok(normalized)
}

fn build_preload_catalog_model(
    index: usize,
    preload: PreloadModelConfig,
    inline: &InlineConfig,
    jobs: &JobConfig,
) -> Result<StoredModelConfig, ModuleError> {
    let model_id = preload
        .model_id
        .clone()
        .unwrap_or_else(|| format!("{}-{index}", preload.engine));
    let engine_name = canonical_engine_name(&preload.engine);
    let task = parse_model_task(preload.task.as_deref(), &engine_name, &model_id)?;
    let pooling = parse_pooling(preload.pooling.as_deref().unwrap_or("mean"))?;
    let normalize = preload.normalize.unwrap_or(true);
    let max_tokens = preload.max_tokens.unwrap_or(512);
    let artifact_format = preload
        .format
        .clone()
        .unwrap_or_else(|| default_artifact_format(&engine_name));
    let artifact_digest = match preload.artifact_digest.clone() {
        Some(digest) => normalize_digest(&digest),
        None => format!("sha256:{}", sha256_file(&preload.model_path)?),
    };
    let tokenizer =
        SanitizedTokenizer::from_file(&preload.tokenizer_path, TokenizerConfig { max_tokens })?;
    build_stored_model_config(
        model_id,
        &engine_name,
        task,
        artifact_digest,
        artifact_format,
        format!("sha256:{}", tokenizer.sanitized_sha256()),
        ModelAssetLocator::LocalPath {
            path: preload.model_path.clone(),
        },
        ModelAssetLocator::LocalPath {
            path: preload.tokenizer_path.clone(),
        },
        local_file_url(&preload.model_path),
        local_file_url(&preload.tokenizer_path),
        pooling,
        normalize,
        max_tokens,
        preload
            .quant
            .clone()
            .unwrap_or_else(|| default_quant(&engine_name)),
        false,
        preload.worker_bin.clone(),
        preload.worker_runtime_dir.clone(),
        inline,
        jobs,
    )
}

fn normalize_catalog_model(
    model: StoredModelConfig,
    inline: &InlineConfig,
    jobs: &JobConfig,
) -> Result<StoredModelConfig, ModuleError> {
    let engine_name = canonical_engine_name(&model.engine);
    let task = parse_model_task(Some(&model.task), &engine_name, &model.model_id)?;
    let pooling = parse_pooling(&model.pooling)?;
    build_stored_model_config(
        model.model_id,
        &engine_name,
        task,
        normalize_digest(&model.artifact_digest),
        model.artifact_format,
        normalize_digest(&model.tokenizer_sanitized_digest),
        model.model_locator,
        model.tokenizer_locator,
        model.model_source_url,
        model.tokenizer_source_url,
        pooling,
        model.normalize,
        model.max_tokens,
        model.quant,
        model.pin,
        model.worker_bin,
        model.worker_runtime_dir,
        inline,
        jobs,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_stored_model_config(
    model_id: String,
    engine_name: &str,
    task: ModelTask,
    artifact_digest: String,
    artifact_format: String,
    tokenizer_sanitized_digest: String,
    model_locator: ModelAssetLocator,
    tokenizer_locator: ModelAssetLocator,
    model_source_url: String,
    tokenizer_source_url: String,
    pooling: WorkerPooling,
    normalize: bool,
    max_tokens: usize,
    quant: String,
    pin: bool,
    worker_bin: Option<PathBuf>,
    worker_runtime_dir: Option<PathBuf>,
    inline: &InlineConfig,
    jobs: &JobConfig,
) -> Result<StoredModelConfig, ModuleError> {
    let engine_identity = catalog_model_engine_identity(engine_name)?;
    let numeric_profile = NumericProfile {
        model_digest: artifact_digest.clone(),
        quant,
        engine: engine_identity.clone(),
        sanitized_tokenizer_digest: tokenizer_sanitized_digest.clone(),
        pooling: profile_pooling(pooling),
        normalization: if normalize {
            NormalizationMode::L2
        } else {
            NormalizationMode::None
        },
        dtype: match engine_name {
            "llama" | "ane" => NumericDType::F16,
            "mlx" => NumericDType::Bf16,
            _ => NumericDType::F32,
        },
        flash_attention: FlashAttentionSetting::Disabled,
        certified_shape: CertifiedShapeEnvelope {
            max_context_tokens: max_tokens.min(u32::MAX as usize) as u32,
            max_batch_tokens: inline.max_tokens.min(u32::MAX as u64) as u32,
            max_micro_batch_tokens: jobs.bulk_quantum_tokens.min(u32::MAX as u64) as u32,
            max_sequences: inline.max_items.min(u32::MAX as usize) as u32,
        },
        prompt_template: match task {
            ModelTask::Embed => None,
            ModelTask::Rerank => Some("synapse-rerank-bos-query-sep-doc-eos-v1".to_string()),
            ModelTask::Generate => Some("synapse-microllm-greedy-v1".to_string()),
        },
        prefix_template: None,
        thread_policy: ThreadPolicyClass::Balanced,
    };
    Ok(StoredModelConfig {
        model_id,
        engine: engine_name.to_string(),
        task: task.as_str().to_string(),
        artifact_digest,
        artifact_format,
        tokenizer_sanitized_digest,
        model_locator,
        tokenizer_locator,
        model_source_url,
        tokenizer_source_url,
        pooling: pooling.as_str().to_string(),
        normalize,
        max_tokens,
        quant: numeric_profile.quant.clone(),
        pin,
        engine_identity,
        numeric_profile_id: numeric_profile.numeric_profile_id(),
        fingerprint: numeric_profile.fingerprint(),
        worker_bin,
        worker_runtime_dir,
    })
}

fn canonical_engine_name(engine: &str) -> String {
    match engine.trim().to_ascii_lowercase().as_str() {
        "onnx" => "ort".to_string(),
        "llama.cpp" => "llama".to_string(),
        "coreml" | "neural_engine" => "ane".to_string(),
        other => other.to_string(),
    }
}

fn default_artifact_format(engine_name: &str) -> String {
    match engine_name {
        "llama" => "gguf".to_string(),
        "mlx" => "safetensors".to_string(),
        "ane" => "mlmodelc".to_string(),
        _ => "onnx".to_string(),
    }
}

fn default_quant(engine_name: &str) -> String {
    match engine_name {
        "llama" => "f16".to_string(),
        "mlx" => "bf16".to_string(),
        "ane" => "fp16".to_string(),
        _ => "fp32".to_string(),
    }
}

fn catalog_model_engine_identity(engine_name: &str) -> Result<EngineIdentity, ModuleError> {
    match engine_name {
        "ort" => Ok(OrtEmbedEngine::new().identity()),
        "llama" => Ok(worker_catalog_identity(
            "llama.cpp-worker",
            "protocol-v1",
            &[("transport", worker_catalog_transport())],
        )),
        "mlx" => Ok(worker_catalog_identity(
            "mlx-worker",
            "protocol-v1",
            &[
                ("transport", worker_catalog_transport()),
                ("numeric_profile", "bf16-distinct"),
            ],
        )),
        "ane" => Ok(worker_catalog_identity(
            "ane-coreml-worker",
            "protocol-v1",
            &[
                ("transport", worker_catalog_transport()),
                ("placement_gate", "neural-engine"),
            ],
        )),
        other => Err(ModuleError::Config(format!(
            "unsupported engine '{other}' for catalog model"
        ))),
    }
}

fn worker_catalog_transport() -> &'static str {
    if cfg!(windows) {
        "named-pipe-worker"
    } else {
        "unix-socket-worker"
    }
}

fn worker_catalog_identity(engine: &str, version: &str, flags: &[(&str, &str)]) -> EngineIdentity {
    let mut build_flags = BTreeMap::new();
    build_flags.insert("risk_class".to_string(), "abort_capable".to_string());
    for (key, value) in flags {
        build_flags.insert((*key).to_string(), (*value).to_string());
    }
    EngineIdentity {
        engine: engine.to_string(),
        version: version.to_string(),
        build_flags,
    }
}

fn local_file_url(path: &Path) -> String {
    format!("file://{}", path.to_string_lossy())
}

fn machine_profile_with_overrides(mut machine_profile: MachineProfile) -> MachineProfile {
    if let Ok(os_build) = env::var(SYNAPSE_OS_BUILD_OVERRIDE_ENV) {
        let os_build = os_build.trim();
        if !os_build.is_empty() {
            machine_profile.os_build = os_build.to_string();
        }
    }
    machine_profile
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
        let health = module_health(&state);
        let mut detail_parts = Vec::new();
        if health.certification_stale {
            detail_parts.push("certification_stale=true");
        }
        if health.performance_stale {
            detail_parts.push("performance_stale=true");
        }
        let detail = if detail_parts.is_empty() {
            "ok".to_string()
        } else {
            format!("ok; {}", detail_parts.join("; "))
        };
        HealthReport {
            status: subc_client_rs::HealthStatus::Ok,
            detail: Some(detail),
            metrics: Some(serde_json::to_value(&health).expect("module health should serialize")),
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
        "rerank.score" => rerank_score(state, request.params).await,
        "microllm.oneshot" => microllm_oneshot(state, request.params).await,
        "model.load" => model_load(state, request.params).await,
        "model.unload" => model_unload(state, request.params).await,
        "cache.pin" => cache_pin(state, request.params).await,
        "cache.gc" => cache_gc(state, request.params).await,
        "probe.start" => probe_start(state, request.params).await,
        "probe.status" => probe_status(state, request.params).await,
        "probe.report" => probe_report(state).await,
        "aliases.check_index" => aliases_check_index(state, request.params).await,
        "alias.retract" => alias_retract(state, request.params).await,
        "alias.declare" => alias_declare(state, request.params).await,
        "admission.status" => admission_status(state).await,
        "model.status" => model_status(state, request.params).await,
        other => channel_error(
            "unknown_method",
            format!("unknown method '{other}' for synapse management surface"),
        ),
    }
}

#[derive(Clone)]
struct ResolvedModelLoadSources {
    model_source_url: String,
    tokenizer_source_url: String,
}

fn model_runtime_state_name(state: &ModelRuntimeState) -> &'static str {
    match state {
        ModelRuntimeState::Unloaded => "unloaded",
        ModelRuntimeState::Resolving => "resolving",
        ModelRuntimeState::Downloading { .. } => "downloading",
        ModelRuntimeState::Validating => "validating",
        ModelRuntimeState::Loading => "loading",
        ModelRuntimeState::Ready => "ready",
        ModelRuntimeState::Failed(_) => "failed",
    }
}

fn model_slot_snapshot(runtime: &RuntimeState, model_id: &str) -> Option<ModelSlotSnapshot> {
    runtime
        .catalog
        .lock()
        .ok()?
        .get(model_id)
        .map(|slot| ModelSlotSnapshot {
            spec: slot.spec.clone(),
            loaded: slot.loaded.clone(),
            state: slot.state.clone(),
            notify: Arc::clone(&slot.notify),
        })
}

fn set_model_slot_state(runtime: &RuntimeState, model_id: &str, state: ModelRuntimeState) {
    if let Ok(mut catalog) = runtime.catalog.lock() {
        if let Some(slot) = catalog.get_mut(model_id) {
            let clear_loaded = !matches!(state, ModelRuntimeState::Ready);
            slot.state = state;
            if clear_loaded {
                slot.loaded = None;
            }
            slot.notify.notify_waiters();
        }
    }
}

fn model_cold_load_ms(runtime: &RuntimeState, model_id: &str) -> Option<f64> {
    runtime.catalog.lock().ok().and_then(|catalog| {
        catalog
            .get(model_id)
            .and_then(|slot| slot.last_cold_load_ms)
    })
}

fn set_model_slot_ready(
    runtime: &RuntimeState,
    model_id: &str,
    model: Arc<EmbeddingModel>,
    cold_load_ms: f64,
) {
    if let Ok(mut catalog) = runtime.catalog.lock() {
        if let Some(slot) = catalog.get_mut(model_id) {
            slot.loaded = Some(model);
            slot.state = ModelRuntimeState::Ready;
            slot.last_cold_load_ms = Some(cold_load_ms);
            slot.notify.notify_waiters();
        }
    }
}

fn set_job_progress(runtime: &RuntimeState, job_id: &str, state: ModelRuntimeState) {
    if let Ok(mut progress) = runtime.job_progress.lock() {
        progress.insert(job_id.to_string(), state);
    }
}

fn clear_job_progress(runtime: &RuntimeState, job_id: &str) {
    if let Ok(mut progress) = runtime.job_progress.lock() {
        progress.remove(job_id);
    }
}

fn job_progress_state(runtime: &RuntimeState, job_id: &str) -> Option<ModelRuntimeState> {
    runtime
        .job_progress
        .lock()
        .ok()
        .and_then(|progress| progress.get(job_id).cloned())
}

fn register_runtime_catalog_model(
    runtime: &RuntimeState,
    spec: StoredModelConfig,
) -> Result<(), WireOperationError> {
    let mut catalog = runtime.catalog.lock().map_err(|_| {
        WireOperationError::from_stable(
            StableError::model_loading(Some(100)),
            "model catalog state is unavailable",
        )
    })?;
    match catalog.get_mut(&spec.model_id) {
        Some(slot) if slot.spec.fingerprint != spec.fingerprint => {
            Err(WireOperationError::from_stable(
                StableError::artifact_invalid(),
                format!(
                    "model_id '{}' already refers to fingerprint {}",
                    spec.model_id, slot.spec.fingerprint.0
                ),
            ))
        }
        Some(slot) => {
            slot.spec = spec;
            if slot.loaded.is_none() {
                slot.state = ModelRuntimeState::Unloaded;
            }
            slot.notify.notify_waiters();
            Ok(())
        }
        None => {
            catalog.insert(
                spec.model_id.clone(),
                ModelSlot {
                    spec,
                    loaded: None,
                    state: ModelRuntimeState::Unloaded,
                    notify: Arc::new(Notify::new()),
                    last_cold_load_ms: None,
                },
            );
            Ok(())
        }
    }
}

fn model_status_payload(module_generation: u64, slot: &ModelSlotSnapshot) -> Value {
    let mut payload = json!({
        "module_generation": module_generation,
        "model_id": slot.spec.model_id,
        "fingerprint": slot.spec.fingerprint,
        "state": model_runtime_state_name(&slot.state),
        "engine": slot.spec.engine,
        "task": slot.spec.task,
    });
    if let Value::Object(map) = &mut payload {
        match &slot.state {
            ModelRuntimeState::Downloading {
                bytes_done,
                bytes_total,
            } => {
                map.insert("bytes_done".to_string(), Value::from(*bytes_done));
                if let Some(bytes_total) = bytes_total {
                    map.insert("bytes_total".to_string(), Value::from(*bytes_total));
                }
            }
            ModelRuntimeState::Failed(error) => {
                map.insert(
                    "error".to_string(),
                    serde_json::to_value(error).expect("model error serializes"),
                );
            }
            _ => {}
        }
    }
    payload
}

fn model_load_job_status_payload(state: &ModuleState, record: &JobRecord) -> Value {
    let mut payload = json!({
        "module_generation": state.module_generation,
        "job_id": record.job_id,
        "request_key": record.request_key,
    });
    if let Value::Object(map) = &mut payload {
        if record.state == JOB_STATE_DONE {
            map.insert("state".to_string(), Value::from("ready"));
            if let Some(Value::Object(result)) = record.result_json.clone() {
                for (key, value) in result {
                    map.insert(key, value);
                }
            }
            return payload;
        }
        if record.state == JOB_STATE_FAILED_TRANSIENT || record.state == JOB_STATE_FAILED_PERMANENT
        {
            map.insert("state".to_string(), Value::from("failed"));
            map.insert(
                "error".to_string(),
                record.error_json.clone().unwrap_or_else(|| {
                    serde_json::to_value(WireOperationError::from_stable(
                        StableError::model_loading(Some(100)),
                        "model load failed without a stored typed error",
                    ))
                    .expect("fallback model load error serializes")
                }),
            );
            return payload;
        }
        match job_progress_state(&state.runtime, &record.job_id) {
            Some(ModelRuntimeState::Downloading {
                bytes_done,
                bytes_total,
            }) => {
                map.insert("state".to_string(), Value::from("downloading"));
                map.insert("bytes_done".to_string(), Value::from(bytes_done));
                if let Some(bytes_total) = bytes_total {
                    map.insert("bytes_total".to_string(), Value::from(bytes_total));
                }
            }
            Some(ModelRuntimeState::Resolving) => {
                map.insert("state".to_string(), Value::from("resolving"));
            }
            Some(ModelRuntimeState::Validating) => {
                map.insert("state".to_string(), Value::from("validating"));
            }
            Some(ModelRuntimeState::Loading | ModelRuntimeState::Ready) => {
                map.insert("state".to_string(), Value::from("loading"));
            }
            Some(ModelRuntimeState::Unloaded | ModelRuntimeState::Failed(_)) | None => {
                map.insert(
                    "state".to_string(),
                    Value::from(if record.state == JOB_STATE_QUEUED {
                        "resolving"
                    } else {
                        "loading"
                    }),
                );
            }
        }
    }
    payload
}

async fn model_load(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let params: ModelLoadParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid model.load params: {error}"),
            )
        }
    };
    if let Err(message) = validate_model_load_request(&params) {
        return channel_error("invalid_request", message);
    }
    let now = now_ms();
    let request_key = params
        .request_key
        .clone()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| format!("model-load:{}:{now}", state.module_generation));
    let params_json = match serde_json::to_value(&params) {
        Ok(value) => value,
        Err(error) => return channel_error("invalid_request", error.to_string()),
    };
    let admission = match state.store.admit_job(
        &request_key,
        "model.load",
        state.module_generation,
        &params_json,
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
        let task_params = params.clone();
        set_job_progress(&state.runtime, &task_job_id, ModelRuntimeState::Resolving);
        tokio::spawn(async move {
            execute_model_load_job(task_state, task_job_id, task_params).await;
        });
    }
    result_outcome(model_load_job_status_payload(&state, &record))
}

async fn model_status(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let params: ModelStatusParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid model.status params: {error}"),
            )
        }
    };
    match (params.job_id, params.model_id) {
        (Some(job_id), None) => match state.store.get_job(&job_id) {
            Ok(Some(record)) if record.kind == "model.load" => {
                result_outcome(model_load_job_status_payload(&state, &record))
            }
            Ok(Some(_)) => channel_error(
                "invalid_request",
                "job_id does not refer to a model.load job",
            ),
            Ok(None) => channel_error("invalid_request", "unknown or expired job_id"),
            Err(error) => channel_error("store_failure", error.to_string()),
        },
        (None, Some(model_id)) => match model_slot_snapshot(&state.runtime, &model_id) {
            Some(slot) => result_outcome(model_status_payload(state.module_generation, &slot)),
            None => channel_error("invalid_request", "unknown model_id"),
        },
        _ => channel_error(
            "invalid_request",
            "model.status requires exactly one of job_id or model_id",
        ),
    }
}

async fn model_unload(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let params: ModelUnloadParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid model.unload params: {error}"),
            )
        }
    };
    let Some(snapshot) = model_slot_snapshot(&state.runtime, &params.model_id) else {
        return channel_error("invalid_request", "unknown model_id");
    };
    if matches!(
        snapshot.state,
        ModelRuntimeState::Resolving
            | ModelRuntimeState::Downloading { .. }
            | ModelRuntimeState::Validating
            | ModelRuntimeState::Loading
    ) {
        return result_outcome(error_payload(
            &state,
            WireOperationError::from_stable(
                StableError::model_loading(Some(250)),
                format!("model '{}' is still loading", params.model_id),
            ),
        ));
    }
    if let Some(loaded) = snapshot.loaded {
        let unload = tokio::task::spawn_blocking(move || unload_embedding_model_blocking(loaded))
            .await
            .map_err(|error| {
                WireOperationError::from_stable(
                    StableError::engine_crashed(Some(100)),
                    format!("model unload join failed: {error}"),
                )
            });
        match unload {
            Ok(Ok(())) => {}
            Ok(Err(error)) | Err(error) => {
                return result_outcome(error_payload(&state, error));
            }
        }
    }
    set_model_slot_state(
        &state.runtime,
        &params.model_id,
        ModelRuntimeState::Unloaded,
    );
    let slot = model_slot_snapshot(&state.runtime, &params.model_id)
        .expect("unloaded model remains registered");
    result_outcome(model_status_payload(state.module_generation, &slot))
}

async fn resolve_model_for_request(
    state: Arc<ModuleState>,
    requested: Option<&str>,
    task: ModelTask,
) -> Result<Arc<EmbeddingModel>, WireOperationError> {
    let model_id = if let Some(requested) = requested {
        requested.to_string()
    } else {
        match state.store.knob_assignment(
            &state.machine_profile_hash,
            task.as_str(),
            state.runtime.knob,
        ) {
            Ok(Some(assignment)) => assignment.model_id,
            Ok(None) => {
                let has_known_task = state
                    .runtime
                    .catalog
                    .lock()
                    .ok()
                    .map(|catalog| catalog.values().any(|slot| slot.spec.task == task.as_str()))
                    .unwrap_or(false);
                if has_known_task {
                    return Err(WireOperationError::from_stable(
                        StableError::probe_required(),
                        format!(
                            "task '{}' has no {} knob assignment on machine profile {}; run probe.start",
                            task.as_str(),
                            state.runtime.knob.as_str(),
                            state.machine_profile_hash,
                        ),
                    ));
                }
                let Some(default_model_id) = state.runtime.default_model_id() else {
                    return Err(WireOperationError::from_stable(
                        StableError::probe_required(),
                        "synapse requests require a registered model",
                    ));
                };
                default_model_id
            }
            Err(error) => {
                return Err(WireOperationError::from_stable(
                    StableError::engine_crashed(Some(100)),
                    format!("read knob assignment: {error}"),
                ))
            }
        }
    };
    let Some(snapshot) = model_slot_snapshot(&state.runtime, &model_id) else {
        return Err(WireOperationError::from_stable(
            StableError::model_loading(Some(250)),
            format!("model '{model_id}' is not registered"),
        ));
    };
    match (&snapshot.state, snapshot.loaded) {
        (ModelRuntimeState::Ready, Some(model)) => Ok(model),
        (ModelRuntimeState::Failed(error), _) if error.class == ErrorClass::Permanent => {
            Err(error.clone())
        }
        (
            ModelRuntimeState::Resolving
            | ModelRuntimeState::Downloading { .. }
            | ModelRuntimeState::Validating
            | ModelRuntimeState::Loading,
            _,
        ) => Err(WireOperationError::from_stable(
            StableError::model_loading(Some(250)),
            format!("model '{model_id}' is loading"),
        )),
        _ => {
            begin_background_catalog_load(Arc::clone(&state), model_id.clone());
            Err(WireOperationError::from_stable(
                StableError::model_loading(Some(250)),
                format!("model '{model_id}' is loading"),
            ))
        }
    }
}

fn begin_background_catalog_load(state: Arc<ModuleState>, model_id: String) {
    let should_spawn = {
        let Ok(mut catalog) = state.runtime.catalog.lock() else {
            return;
        };
        let Some(slot) = catalog.get_mut(&model_id) else {
            return;
        };
        if slot.loaded.is_some()
            || matches!(
                slot.state,
                ModelRuntimeState::Resolving
                    | ModelRuntimeState::Downloading { .. }
                    | ModelRuntimeState::Validating
                    | ModelRuntimeState::Loading
                    | ModelRuntimeState::Ready
            )
        {
            false
        } else {
            slot.state = ModelRuntimeState::Loading;
            slot.notify.notify_waiters();
            true
        }
    };
    if should_spawn {
        tokio::spawn(async move {
            let _ = load_catalog_model_task(state, model_id).await;
        });
    }
}

async fn ensure_model_loaded_for_control(
    state: Arc<ModuleState>,
    model_id: &str,
) -> Result<Arc<EmbeddingModel>, WireOperationError> {
    loop {
        let Some(snapshot) = model_slot_snapshot(&state.runtime, model_id) else {
            return Err(WireOperationError::from_stable(
                StableError::artifact_invalid(),
                format!("unknown model_id '{model_id}'"),
            ));
        };
        match (&snapshot.state, snapshot.loaded.clone()) {
            (ModelRuntimeState::Ready, Some(model)) => return Ok(model),
            (ModelRuntimeState::Failed(error), _) => return Err(error.clone()),
            (
                ModelRuntimeState::Resolving
                | ModelRuntimeState::Downloading { .. }
                | ModelRuntimeState::Validating
                | ModelRuntimeState::Loading,
                _,
            ) => snapshot.notify.notified().await,
            _ => {
                begin_background_catalog_load(Arc::clone(&state), model_id.to_string());
                snapshot.notify.notified().await;
            }
        }
    }
}

async fn load_catalog_model_task(
    state: Arc<ModuleState>,
    model_id: String,
) -> Result<Arc<EmbeddingModel>, WireOperationError> {
    let _permit = state
        .runtime
        .control_loads
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| {
            WireOperationError::from_stable(
                StableError::model_loading(Some(100)),
                "control load queue is closed",
            )
        })?;
    set_model_slot_state(&state.runtime, &model_id, ModelRuntimeState::Loading);
    let Some(snapshot) = model_slot_snapshot(&state.runtime, &model_id) else {
        return Err(WireOperationError::from_stable(
            StableError::artifact_invalid(),
            format!("unknown model_id '{model_id}'"),
        ));
    };
    let spec = snapshot.spec.clone();
    let ort_engine = Arc::clone(&state.runtime.ort_engine);
    let model_cache = Arc::clone(&state.model_cache);
    let load_started = std::time::Instant::now();
    let loaded = tokio::task::spawn_blocking(move || {
        load_catalog_model_blocking(spec, ort_engine, model_cache)
    })
    .await
    .map_err(|error| {
        WireOperationError::from_stable(
            StableError::engine_crashed(Some(100)),
            format!("model load join failed: {error}"),
        )
    })?;
    match loaded {
        Ok(model) => {
            let cold_load_ms = load_started.elapsed().as_secs_f64() * 1_000.0;
            let model = Arc::new(model);
            set_model_slot_ready(&state.runtime, &model_id, Arc::clone(&model), cold_load_ms);
            Ok(model)
        }
        Err(error) => {
            set_model_slot_state(
                &state.runtime,
                &model_id,
                ModelRuntimeState::Failed(error.clone()),
            );
            Err(error)
        }
    }
}

fn load_catalog_model_blocking(
    spec: StoredModelConfig,
    ort_engine: Arc<Mutex<OrtEmbedEngine>>,
    model_cache: Arc<ModelCache>,
) -> Result<EmbeddingModel, WireOperationError> {
    let task = parse_model_task(Some(&spec.task), &spec.engine, &spec.model_id)
        .map_err(|error| artifact_invalid_error(error.to_string()))?;
    let model_path = locator_path(&spec.model_locator, &model_cache)?;
    let tokenizer_path = locator_path(&spec.tokenizer_locator, &model_cache)?;
    let tokenizer = SanitizedTokenizer::from_file(
        &tokenizer_path.path,
        TokenizerConfig {
            max_tokens: spec.max_tokens,
        },
    )
    .map_err(|error| artifact_invalid_error(error.to_string()))?;
    let actual_tokenizer_digest = format!("sha256:{}", tokenizer.sanitized_sha256());
    if actual_tokenizer_digest != normalize_digest(&spec.tokenizer_sanitized_digest) {
        return Err(artifact_invalid_error(format!(
            "tokenizer digest mismatch for '{}': expected {}, got {}",
            spec.model_id, spec.tokenizer_sanitized_digest, actual_tokenizer_digest
        )));
    }
    let runtime_config = model_runtime_config(&spec, &model_path.path);
    let artifact = ValidatedArtifact {
        digest: spec.artifact_digest.clone(),
        format: spec.artifact_format.clone(),
    };
    let (backend, loaded_model) = match spec.engine.as_str() {
        "ort" => {
            let mut engine = ort_engine.lock().map_err(|_| {
                WireOperationError::from_stable(
                    StableError::engine_crashed(Some(100)),
                    "ORT engine mutex was poisoned during model load",
                )
            })?;
            let loaded_model = engine
                .load(&artifact, &runtime_config)
                .map_err(engine_error_to_wire)?;
            (EmbedBackend::Ort(Arc::clone(&ort_engine)), loaded_model)
        }
        "llama" | "mlx" | "ane" => load_worker_backend_blocking(&spec, &artifact, &runtime_config)?,
        other => {
            return Err(artifact_invalid_error(format!(
                "unsupported engine '{other}' for model '{}'",
                spec.model_id
            )))
        }
    };
    Ok(EmbeddingModel {
        model_id: spec.model_id.clone(),
        task,
        loaded_model,
        backend,
        tokenizer,
        numeric_profile_id: spec.numeric_profile_id.clone(),
        fingerprint: spec.fingerprint.clone(),
        engine_identity: spec.engine_identity.clone(),
    })
}

fn load_worker_backend_blocking(
    spec: &StoredModelConfig,
    artifact: &ValidatedArtifact,
    runtime_config: &RuntimeConfig,
) -> Result<(EmbedBackend, LoadedModel), WireOperationError> {
    use worker_host::{WorkerEngine, WorkerHostConfig};

    if matches!(spec.engine.as_str(), "mlx" | "ane") && !cfg!(target_os = "macos") {
        return Err(artifact_invalid_error(format!(
            "{} model '{}' is only supported on macOS",
            spec.engine, spec.model_id
        )));
    }

    let env_prefix = spec.engine.to_ascii_uppercase().replace('.', "_");
    let worker_bin_var = format!("SYNAPSE_{env_prefix}_WORKER_BIN");
    let worker_runtime_dir_var = format!("SYNAPSE_{env_prefix}_WORKER_RUNTIME_DIR");
    let worker_bin = spec
        .worker_bin
        .clone()
        .or_else(|| env::var_os(&worker_bin_var).map(PathBuf::from))
        .ok_or_else(|| {
            artifact_invalid_error(format!(
                "{} model '{}' requires worker_bin or {}",
                spec.engine, spec.model_id, worker_bin_var
            ))
        })?;
    let runtime_dir = spec
        .worker_runtime_dir
        .clone()
        .or_else(|| env::var_os(&worker_runtime_dir_var).map(PathBuf::from))
        .unwrap_or_else(|| env::temp_dir().join("synapse-workers"));
    let mut config = WorkerHostConfig::new(worker_bin, runtime_dir);
    config.worker_id = format!("synapse-{}-{}", spec.engine, spec.model_id);
    config.pooling =
        parse_pooling(&spec.pooling).map_err(|error| artifact_invalid_error(error.to_string()))?;
    config.normalize = spec.normalize;
    let mut engine = WorkerEngine::new(config).map_err(|error| {
        WireOperationError::from_stable(
            StableError::engine_crashed(Some(100)),
            format!(
                "create {} worker engine for '{}': {error}",
                spec.engine, spec.model_id
            ),
        )
    })?;
    let loaded_model =
        EmbedEngine::load(&mut engine, artifact, runtime_config).map_err(engine_error_to_wire)?;
    Ok((
        EmbedBackend::Worker(Arc::new(Mutex::new(engine))),
        loaded_model,
    ))
}

fn unload_embedding_model_blocking(model: Arc<EmbeddingModel>) -> Result<(), WireOperationError> {
    match &model.backend {
        EmbedBackend::Ort(engine) => {
            let mut engine = engine.lock().map_err(|_| {
                WireOperationError::from_stable(
                    StableError::engine_crashed(Some(100)),
                    "ORT engine mutex was poisoned during model unload",
                )
            })?;
            engine.unload(&model.loaded_model);
            Ok(())
        }
        EmbedBackend::Worker(engine) => {
            let mut engine = engine.lock().map_err(|_| {
                WireOperationError::from_stable(
                    StableError::engine_crashed(Some(100)),
                    "worker engine mutex was poisoned during model unload",
                )
            })?;
            EmbedEngine::unload(&mut *engine, &model.loaded_model);
            Ok(())
        }
    }
}

fn model_runtime_config(spec: &StoredModelConfig, model_path: &Path) -> RuntimeConfig {
    let mut runtime_config = RuntimeConfig::default();
    runtime_config.values.insert(
        "model_path".to_string(),
        model_path.to_string_lossy().to_string(),
    );
    runtime_config.values.insert(
        "artifact_path".to_string(),
        model_path.to_string_lossy().to_string(),
    );
    runtime_config
        .values
        .insert("pooling".to_string(), spec.pooling.clone());
    runtime_config.values.insert(
        "normalize".to_string(),
        if spec.normalize { "true" } else { "false" }.to_string(),
    );
    runtime_config
}

struct LocatedAsset {
    path: PathBuf,
    _guard: Option<synapse_core::ModelCacheReadGuard>,
}

fn locator_path(
    locator: &ModelAssetLocator,
    model_cache: &ModelCache,
) -> Result<LocatedAsset, WireOperationError> {
    match locator {
        ModelAssetLocator::LocalPath { path } => Ok(LocatedAsset {
            path: path.clone(),
            _guard: None,
        }),
        ModelAssetLocator::CacheDigest { digest } => {
            let guard = model_cache
                .acquire_read(digest)
                .map_err(model_cache_load_error)?;
            Ok(LocatedAsset {
                path: guard.blob_path().to_path_buf(),
                _guard: Some(guard),
            })
        }
    }
}

fn artifact_invalid_error(message: impl Into<String>) -> WireOperationError {
    WireOperationError::from_stable(StableError::artifact_invalid(), message)
}

fn transient_model_load_error(message: impl Into<String>) -> WireOperationError {
    WireOperationError::from_stable(StableError::model_loading(Some(1_000)), message)
}

fn model_cache_load_error(error: ModelCacheError) -> WireOperationError {
    match error {
        ModelCacheError::ArtifactInvalid(message) | ModelCacheError::InvalidSource(message) => {
            artifact_invalid_error(message)
        }
        ModelCacheError::Tokenizer(error) => artifact_invalid_error(error.to_string()),
        ModelCacheError::NotFound(digest) => transient_model_load_error(format!(
            "cache artifact {digest} is missing; re-submit model.load to reacquire it"
        )),
        ModelCacheError::Io {
            action,
            path,
            source,
        } => io_to_load_error(action, Path::new(&path), &source),
        ModelCacheError::Download { url, message } => {
            transient_model_load_error(format!("download {url}: {message}"))
        }
        ModelCacheError::Json(error) => transient_model_load_error(error.to_string()),
        ModelCacheError::Lease(error) => transient_model_load_error(error.to_string()),
    }
}

fn io_to_load_error(action: &str, path: &Path, source: &std::io::Error) -> WireOperationError {
    if source.raw_os_error() == Some(28) {
        return transient_model_load_error(format!(
            "disk is full while {action} {}: {source}",
            path.display()
        ));
    }
    transient_model_load_error(format!("{action} {}: {source}", path.display()))
}

async fn execute_model_load_job(state: Arc<ModuleState>, job_id: String, params: ModelLoadParams) {
    if !matches!(
        state
            .store
            .mark_job_running(&job_id, state.module_generation, now_ms()),
        Ok(true)
    ) {
        clear_job_progress(&state.runtime, &job_id);
        return;
    }

    let result = async {
        let sources = resolve_model_load_sources(&params).map_err(artifact_invalid_error)?;
        let temp_dir = env::temp_dir().join(format!(
            "synapse-model-load-{}-{}",
            std::process::id(),
            now_ms()
        ));
        fs::create_dir_all(&temp_dir)
            .map_err(|error| io_to_load_error("create temp directory", &temp_dir, &error))?;
        let model_path = temp_dir.join("model.bin");
        let tokenizer_path = temp_dir.join("tokenizer.json");

        set_job_progress(
            &state.runtime,
            &job_id,
            ModelRuntimeState::Downloading {
                bytes_done: 0,
                bytes_total: None,
            },
        );
        download_source_to_temp(
            &sources.model_source_url,
            &model_path,
            |bytes_done, bytes_total| {
                set_job_progress(
                    &state.runtime,
                    &job_id,
                    ModelRuntimeState::Downloading {
                        bytes_done,
                        bytes_total,
                    },
                );
            },
        )?;
        download_source_to_temp(&sources.tokenizer_source_url, &tokenizer_path, |_, _| {})?;

        set_job_progress(&state.runtime, &job_id, ModelRuntimeState::Validating);
        let engine_name = canonical_engine_name(&params.engine);
        validate_artifact_file(&model_path, &engine_name)?;

        let pin_module_id = params.pin.then(|| state.module_id.clone());
        let tokenizer_meta = state
            .model_cache
            .ingest(ModelCacheIngest {
                source_url: local_file_url(&tokenizer_path),
                expected_digest: None,
                format: "tokenizer_json".to_string(),
                tokenizer_path: None,
                pin_module_id: pin_module_id.clone(),
            })
            .map_err(model_cache_load_error)?;
        let model_meta = state
            .model_cache
            .ingest(ModelCacheIngest {
                source_url: local_file_url(&model_path),
                expected_digest: params.expected_digest.clone(),
                format: default_artifact_format(&engine_name),
                tokenizer_path: Some(tokenizer_path.clone()),
                pin_module_id,
            })
            .map_err(model_cache_load_error)?;
        let spec = build_loaded_catalog_model(
            &params,
            &engine_name,
            &sources,
            &model_meta,
            &tokenizer_meta,
            &state.runtime.inline,
            &state.runtime.jobs,
        )?;
        register_runtime_catalog_model(&state.runtime, spec.clone())?;
        state.store.upsert_model(&spec, now_ms()).map_err(|error| {
            transient_model_load_error(format!(
                "persist catalog entry for '{}': {error}",
                spec.model_id
            ))
        })?;
        set_job_progress(&state.runtime, &job_id, ModelRuntimeState::Loading);
        let loaded = ensure_model_loaded_for_control(Arc::clone(&state), &spec.model_id).await?;
        let result = json!({
            "model_id": loaded.model_id,
            "fingerprint": loaded.fingerprint,
        });
        let _ = fs::remove_dir_all(&temp_dir);
        Ok::<_, WireOperationError>(result)
    }
    .await;

    clear_job_progress(&state.runtime, &job_id);
    match result {
        Ok(result) => {
            if let Err(error) = state.store.complete_job(&job_id, &result, &[], now_ms()) {
                fail_job_with_wire_error(
                    &state,
                    &job_id,
                    true,
                    transient_model_load_error(format!("complete model.load job: {error}")),
                );
            }
        }
        Err(error) => {
            fail_job_with_wire_error(&state, &job_id, error.class == ErrorClass::Transient, error);
        }
    }
}

fn validate_model_load_request(params: &ModelLoadParams) -> Result<(), String> {
    if params.files.model.trim().is_empty() || params.files.tokenizer.trim().is_empty() {
        return Err("model.load requires files.model and files.tokenizer".to_string());
    }
    resolve_model_load_sources(params).map(|_| ())
}

fn resolve_model_load_sources(
    params: &ModelLoadParams,
) -> Result<ResolvedModelLoadSources, String> {
    match params.source.trim().to_ascii_lowercase().as_str() {
        "hf" => {
            let repo = params
                .repo
                .as_deref()
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| "model.load source=hf requires repo".to_string())?;
            Ok(ResolvedModelLoadSources {
                model_source_url: huggingface_resolve_url(repo, &params.files.model)?,
                tokenizer_source_url: huggingface_resolve_url(repo, &params.files.tokenizer)?,
            })
        }
        "url" => {
            let base = params
                .url
                .as_deref()
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| "model.load source=url requires url".to_string())?;
            Ok(ResolvedModelLoadSources {
                model_source_url: join_base_url(base, &params.files.model)?,
                tokenizer_source_url: join_base_url(base, &params.files.tokenizer)?,
            })
        }
        "file" => {
            let base = params
                .path
                .as_deref()
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| "model.load source=file requires path".to_string())?;
            Ok(ResolvedModelLoadSources {
                model_source_url: join_file_source(base, &params.files.model)?,
                tokenizer_source_url: join_file_source(base, &params.files.tokenizer)?,
            })
        }
        other => Err(format!("unsupported model.load source '{other}'")),
    }
}

fn huggingface_resolve_url(repo: &str, file: &str) -> Result<String, String> {
    let mut url = Url::parse("https://huggingface.co/")
        .map_err(|error| format!("build Hugging Face base URL: {error}"))?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| "build Hugging Face path segments".to_string())?;
        for segment in repo.split('/') {
            if !segment.is_empty() {
                segments.push(segment);
            }
        }
        segments.push("resolve");
        segments.push("main");
        for segment in file.split('/') {
            if !segment.is_empty() {
                segments.push(segment);
            }
        }
    }
    Ok(url.to_string())
}

fn join_base_url(base: &str, file: &str) -> Result<String, String> {
    let mut url =
        Url::parse(base).map_err(|error| format!("invalid url base '{base}': {error}"))?;
    let had_trailing_slash = base.ends_with('/');
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| format!("url '{base}' cannot accept path segments"))?;
        if !had_trailing_slash {
            segments.pop_if_empty();
        }
        for segment in file.split('/') {
            if !segment.is_empty() {
                segments.push(segment);
            }
        }
    }
    Ok(url.to_string())
}

fn join_file_source(base: &str, file: &str) -> Result<String, String> {
    let base_path = if let Some(path) = base.strip_prefix("file://") {
        PathBuf::from(path)
    } else {
        PathBuf::from(base)
    };
    Ok(local_file_url(&base_path.join(file)))
}

fn build_loaded_catalog_model(
    params: &ModelLoadParams,
    engine_name: &str,
    sources: &ResolvedModelLoadSources,
    model_meta: &ModelCacheMeta,
    tokenizer_meta: &ModelCacheMeta,
    inline: &InlineConfig,
    jobs: &JobConfig,
) -> Result<StoredModelConfig, WireOperationError> {
    let model_id = params.model_id.clone().unwrap_or_else(|| {
        derive_loaded_model_id(engine_name, &sources.model_source_url, &model_meta.digest)
    });
    let task = parse_model_task(params.task.as_deref(), engine_name, &model_id)
        .map_err(|error| artifact_invalid_error(error.to_string()))?;
    let pooling = parse_pooling(params.pooling.as_deref().unwrap_or("mean"))
        .map_err(|error| artifact_invalid_error(error.to_string()))?;
    let tokenizer_sanitized_digest =
        model_meta
            .sanitized_tokenizer_digest
            .clone()
            .ok_or_else(|| {
                artifact_invalid_error("model cache metadata is missing tokenizer digest")
            })?;
    build_stored_model_config(
        model_id,
        engine_name,
        task,
        model_meta.digest.clone(),
        default_artifact_format(engine_name),
        tokenizer_sanitized_digest,
        ModelAssetLocator::CacheDigest {
            digest: model_meta.digest.clone(),
        },
        ModelAssetLocator::CacheDigest {
            digest: tokenizer_meta.digest.clone(),
        },
        sources.model_source_url.clone(),
        sources.tokenizer_source_url.clone(),
        pooling,
        params.normalize.unwrap_or(true),
        params.max_tokens.unwrap_or(512),
        params
            .quant
            .clone()
            .unwrap_or_else(|| default_quant(engine_name)),
        params.pin,
        params.worker_bin.clone(),
        params.worker_runtime_dir.clone(),
        inline,
        jobs,
    )
    .map_err(|error| artifact_invalid_error(error.to_string()))
}

fn derive_loaded_model_id(engine_name: &str, model_source_url: &str, digest: &str) -> String {
    let base = model_source_url
        .rsplit('/')
        .find(|segment| !segment.is_empty())
        .unwrap_or(engine_name)
        .trim_end_matches(".onnx")
        .trim_end_matches(".gguf")
        .trim_end_matches(".json");
    let digest_suffix = digest
        .strip_prefix("sha256:")
        .unwrap_or(digest)
        .chars()
        .take(8)
        .collect::<String>();
    format!(
        "{}-{}",
        sanitize_model_id_component(base),
        digest_suffix.to_ascii_lowercase()
    )
}

fn sanitize_model_id_component(value: &str) -> String {
    let mut output = String::new();
    let mut last_dash = false;
    for ch in value.chars() {
        let ch = ch.to_ascii_lowercase();
        if ch.is_ascii_alphanumeric() {
            output.push(ch);
            last_dash = false;
        } else if !last_dash {
            output.push('-');
            last_dash = true;
        }
    }
    output.trim_matches('-').to_string()
}

fn validate_artifact_file(path: &Path, engine_name: &str) -> Result<(), WireOperationError> {
    let expected_format = default_artifact_format(engine_name);
    let mut header = [0_u8; 8];
    let mut file = fs::File::open(path)
        .map_err(|error| io_to_load_error("open downloaded artifact", path, &error))?;
    let read = file
        .read(&mut header)
        .map_err(|error| io_to_load_error("read downloaded artifact", path, &error))?;
    match expected_format.as_str() {
        "gguf" if read >= 4 && &header[..4] == b"GGUF" => Ok(()),
        "gguf" => Err(artifact_invalid_error(format!(
            "expected GGUF magic at {}",
            path.display()
        ))),
        "onnx" if read >= 1 && header[0] == 0x08 => Ok(()),
        "onnx" => Err(artifact_invalid_error(format!(
            "expected ONNX protobuf header at {}",
            path.display()
        ))),
        other => Err(artifact_invalid_error(format!(
            "unsupported artifact format '{other}' for {}",
            path.display()
        ))),
    }
}

fn download_source_to_temp(
    source_url: &str,
    destination: &Path,
    mut on_progress: impl FnMut(u64, Option<u64>),
) -> Result<(), WireOperationError> {
    let mut output = fs::File::create(destination)
        .map_err(|error| io_to_load_error("create download destination", destination, &error))?;
    if let Some(path) = source_url.strip_prefix("file://") {
        let path = Path::new(path);
        let total = fs::metadata(path).ok().map(|meta| meta.len());
        let mut input = fs::File::open(path)
            .map_err(|error| io_to_load_error("open source artifact", path, &error))?;
        copy_source_stream(
            &mut input,
            &mut output,
            total,
            destination,
            &mut on_progress,
        )?;
    } else {
        let client = reqwest::blocking::Client::new();
        let mut response = client
            .get(source_url)
            .send()
            .map_err(|error| transient_model_load_error(format!("download {source_url}: {error}")))?
            .error_for_status()
            .map_err(|error| {
                transient_model_load_error(format!("download {source_url}: {error}"))
            })?;
        let total = response.content_length();
        copy_source_stream(
            &mut response,
            &mut output,
            total,
            destination,
            &mut on_progress,
        )?;
    }
    output
        .flush()
        .map_err(|error| io_to_load_error("flush downloaded artifact", destination, &error))?;
    Ok(())
}

fn copy_source_stream(
    input: &mut impl Read,
    output: &mut fs::File,
    total: Option<u64>,
    destination: &Path,
    on_progress: &mut impl FnMut(u64, Option<u64>),
) -> Result<(), WireOperationError> {
    let mut buffer = [0_u8; 64 * 1024];
    let mut done = 0_u64;
    let delay_ms = env::var("SYNAPSE_TEST_MODEL_LOAD_CHUNK_DELAY_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0);
    loop {
        let read = input
            .read(&mut buffer)
            .map_err(|error| io_to_load_error("read source artifact", destination, &error))?;
        if read == 0 {
            break;
        }
        output
            .write_all(&buffer[..read])
            .map_err(|error| io_to_load_error("write downloaded artifact", destination, &error))?;
        done = done.saturating_add(read as u64);
        on_progress(done, total);
        if delay_ms > 0 {
            std::thread::sleep(Duration::from_millis(delay_ms));
        }
    }
    Ok(())
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
    let alias_table = match state.store.alias_table() {
        Ok(alias_table) => alias_table,
        Err(error) => return channel_error("store_failure", error.to_string()),
    };
    let model = match resolve_model_for_request(
        Arc::clone(&state),
        params.model.as_deref(),
        ModelTask::Embed,
    )
    .await
    {
        Ok(model) => model,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };
    if let Err(error) = ensure_model_certified(&state, &model) {
        return result_outcome(error_payload(&state, error));
    }
    if let Err(error) = check_fingerprint_constraints(
        &model,
        &alias_table,
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
    embed_tokenized(state, model, ids, tokenized, alias_table).await
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

    let alias_table = match state.store.alias_table() {
        Ok(alias_table) => alias_table,
        Err(error) => return channel_error("store_failure", error.to_string()),
    };
    let model = match resolve_model_for_request(
        Arc::clone(&state),
        params.model.as_deref(),
        ModelTask::Embed,
    )
    .await
    {
        Ok(model) => model,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };
    if let Err(error) = ensure_model_certified(&state, &model) {
        return result_outcome(error_payload(&state, error));
    }
    if let Err(error) = check_fingerprint_constraints(
        &model,
        &alias_table,
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
                alias_table,
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
    embed_tokenized(state, model, ids, tokenized, alias_table).await
}

async fn rerank_score(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let params: RerankScoreParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid rerank.score params: {error}"),
            )
        }
    };
    if params.candidates.is_empty() {
        return channel_error(
            "invalid_request",
            "rerank.score requires at least one candidate",
        );
    }
    if params.candidates.len() > state.runtime.inline.max_items {
        return result_outcome(error_payload(
            &state,
            WireOperationError::from_stable(
                StableError::queue_full(Some(state.runtime.inline.max_queue_ms)),
                format!(
                    "rerank.score candidate count {} exceeds inline budget {}",
                    params.candidates.len(),
                    state.runtime.inline.max_items
                ),
            ),
        ));
    }

    let alias_table = match state.store.alias_table() {
        Ok(alias_table) => alias_table,
        Err(error) => return channel_error("store_failure", error.to_string()),
    };
    let model = match resolve_model_for_request(
        Arc::clone(&state),
        params.model.as_deref(),
        ModelTask::Rerank,
    )
    .await
    {
        Ok(model) => model,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };
    if model.task != ModelTask::Rerank {
        return result_outcome(error_payload(
            &state,
            WireOperationError::from_stable(
                StableError::artifact_invalid(),
                format!(
                    "model '{}' is not configured for rerank.score",
                    model.model_id
                ),
            ),
        ));
    }
    if let Err(error) = ensure_model_certified(&state, &model) {
        return result_outcome(error_payload(&state, error));
    }
    if let Err(error) = check_fingerprint_constraints(
        &model,
        &alias_table,
        params.target_fingerprint.as_deref(),
        params.required_fingerprint.as_deref(),
        params.allow_equivalent,
        params.required_epoch,
    ) {
        return result_outcome(error_payload(&state, error));
    }

    let mut texts = Vec::with_capacity(params.candidates.len() + 1);
    texts.push(params.query.as_str());
    texts.extend(params.candidates.iter().map(String::as_str));
    let request_bytes = request_bytes_for_texts(texts.iter().copied());
    let tokenized = match model.tokenizer.tokenize_batch_without_special_tokens(texts) {
        Ok(tokenized) => tokenized,
        Err(error) => {
            return result_outcome(error_payload(
                &state,
                WireOperationError::from_stable(StableError::artifact_invalid(), error.to_string()),
            ))
        }
    };
    let mut token_items = tokenized.batch.items.clone();
    let query = token_items.remove(0);
    let candidate_token_counts = token_items
        .iter()
        .map(|candidate| {
            candidate
                .len()
                .saturating_add(query.len())
                .saturating_add(3) as u64
        })
        .sum::<u64>();
    if candidate_token_counts > state.runtime.inline.max_tokens {
        return result_outcome(error_payload(
            &state,
            WireOperationError::from_stable(
                StableError::queue_full(Some(state.runtime.inline.max_queue_ms)),
                format!(
                    "rerank.score token budget {candidate_token_counts} exceeds inline budget {}",
                    state.runtime.inline.max_tokens
                ),
            ),
        ));
    }
    let queue_class = if params.candidates.len() <= 20 {
        QueueClass::Interactive
    } else {
        QueueClass::Bulk
    };
    let _admission = match state.runtime.admit_inline(
        queue_class,
        request_bytes,
        params.deadline_ms,
        params.max_queue_ms,
    ) {
        Ok(admission) => admission,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };

    let scores = match execute_rerank(
        &state.runtime,
        &model,
        RerankRequest {
            query,
            candidates: token_items,
        },
    )
    .await
    {
        Ok(scores) => scores,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };
    if scores.scores.len() != params.candidates.len() {
        return result_outcome(error_payload(
            &state,
            WireOperationError::from_stable(
                StableError::engine_crashed(None),
                format!(
                    "engine returned {} rerank scores for {} candidates",
                    scores.scores.len(),
                    params.candidates.len()
                ),
            ),
        ));
    }
    let equivalent_to = equivalent_fingerprints(&alias_table, &model);
    let payload = RerankScorePayload {
        scores: scores.scores,
        real_token_counts: tokenized.real_token_counts,
        truncation_disclosures: tokenized.disclosures,
    };
    let envelope = ResponseEnvelope {
        fingerprint: model.fingerprint.clone(),
        table_epoch: alias_table.table_epoch,
        dims: 1,
        provenance: ResponseProvenance {
            engine: model.engine_identity.clone(),
        },
        module_generation: state.module_generation,
        equivalent_to,
        payload,
    };
    result_outcome(serde_json::to_value(envelope).expect("rerank envelope should serialize"))
}

async fn microllm_oneshot(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let params: MicroLlmOneshotParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid microllm.oneshot params: {error}"),
            )
        }
    };
    if params.max_tokens > 64 {
        return channel_error(
            "invalid_request",
            "microllm.oneshot max_tokens must be <= 64",
        );
    }
    if params.grammar.is_some() {
        return channel_error(
            "invalid_request",
            "microllm.oneshot grammar is not supported by the current llama-cpp-2 worker build",
        );
    }

    let alias_table = match state.store.alias_table() {
        Ok(alias_table) => alias_table,
        Err(error) => return channel_error("store_failure", error.to_string()),
    };
    let model = match resolve_model_for_request(
        Arc::clone(&state),
        params.model.as_deref(),
        ModelTask::Generate,
    )
    .await
    {
        Ok(model) => model,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };
    if model.task != ModelTask::Generate {
        return result_outcome(error_payload(
            &state,
            WireOperationError::from_stable(
                StableError::artifact_invalid(),
                format!(
                    "model '{}' is not configured for microllm.oneshot",
                    model.model_id
                ),
            ),
        ));
    }
    if let Err(error) = ensure_model_certified(&state, &model) {
        return result_outcome(error_payload(&state, error));
    }
    if let Err(error) = check_fingerprint_constraints(
        &model,
        &alias_table,
        params.target_fingerprint.as_deref(),
        params.required_fingerprint.as_deref(),
        params.allow_equivalent,
        params.required_epoch,
    ) {
        return result_outcome(error_payload(&state, error));
    }

    let request_bytes = request_bytes_for_texts([params.prompt.as_str()]);
    let tokenized = match model.tokenizer.tokenize_batch([params.prompt.as_str()]) {
        Ok(tokenized) => tokenized,
        Err(error) => {
            return result_outcome(error_payload(
                &state,
                WireOperationError::from_stable(StableError::artifact_invalid(), error.to_string()),
            ))
        }
    };
    let prompt_tokens = tokenized
        .real_token_counts
        .first()
        .copied()
        .unwrap_or_default() as u64;
    let total_tokens = prompt_tokens.saturating_add(u64::from(params.max_tokens));
    if total_tokens > state.runtime.inline.max_tokens {
        return result_outcome(error_payload(
            &state,
            WireOperationError::from_stable(
                StableError::queue_full(Some(state.runtime.inline.max_queue_ms)),
                format!(
                    "microllm.oneshot token budget {total_tokens} exceeds inline budget {}",
                    state.runtime.inline.max_tokens
                ),
            ),
        ));
    }
    let _admission = match state.runtime.admit_inline(
        QueueClass::Interactive,
        request_bytes,
        params.deadline_ms,
        params.max_queue_ms,
    ) {
        Ok(admission) => admission,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };

    let mut prompt_items = tokenized.batch.items.clone();
    let prompt = prompt_items.pop().unwrap_or_default();
    let output = match execute_generate(
        &state.runtime,
        &model,
        GenerateRequest {
            prompt,
            max_tokens: params.max_tokens,
            grammar: None,
        },
    )
    .await
    {
        Ok(output) => output,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };
    let equivalent_to = equivalent_fingerprints(&alias_table, &model);
    let payload = MicroLlmOneshotPayload {
        text: output.text,
        finish_reason: output.finish_reason,
        n_prompt: output.n_prompt,
        n_gen: output.n_gen,
        real_token_counts: tokenized.real_token_counts,
        truncation_disclosures: tokenized.disclosures,
    };
    let envelope = ResponseEnvelope {
        fingerprint: model.fingerprint.clone(),
        table_epoch: alias_table.table_epoch,
        dims: 0,
        provenance: ResponseProvenance {
            engine: model.engine_identity.clone(),
        },
        module_generation: state.module_generation,
        equivalent_to,
        payload,
    };
    result_outcome(serde_json::to_value(envelope).expect("microllm envelope should serialize"))
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
        work.alias_table,
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
    _total_tokens: u64,
    request_bytes: u64,
) -> Result<Vectors, WireOperationError> {
    let mut scheduler = LaneScheduler::new(SchedulerConfig {
        byte_budget: request_bytes.max(1),
        bulk_quantum_tokens: runtime.jobs.bulk_quantum_tokens.max(1),
        max_concurrent_workers: 1,
        default_execution_ms: runtime.inline.estimated_execution_ms,
        ..SchedulerConfig::default()
    });
    let scheduled_tokens = batch_token_cost(&batch);
    scheduler
        .admit(
            &SystemClock,
            WorkRequest {
                queue_class: QueueClass::Bulk,
                deadline_ms: None,
                max_queue_ms: runtime.inline.max_queue_ms,
                request_bytes,
                token_cost: scheduled_tokens,
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

fn batch_token_cost(batch: &TokenBatch) -> u64 {
    batch
        .items
        .iter()
        .map(|item| item.len().max(1) as u64)
        .sum::<u64>()
        .max(1)
}

fn embed_result_pages(
    state: &ModuleState,
    model: &EmbeddingModel,
    ids: Vec<String>,
    vectors: Vectors,
    tokenized: TokenizedBatch,
    alias_table: AliasTable,
    job_id: &str,
) -> Result<(Value, Vec<Vec<u8>>), WireOperationError> {
    let dims = vectors.first().map(Vec::len).unwrap_or(0) as u32;
    let equivalent_to = equivalent_fingerprints(&alias_table, model);
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
            table_epoch: alias_table.table_epoch,
            dims,
            provenance: ResponseProvenance {
                engine: model.engine_identity.clone(),
            },
            module_generation: state.module_generation,
            equivalent_to: equivalent_to.clone(),
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
    alias_table: AliasTable,
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
    let equivalent_to = equivalent_fingerprints(&alias_table, &model);
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
        table_epoch: alias_table.table_epoch,
        dims,
        provenance: ResponseProvenance {
            engine: model.engine_identity.clone(),
        },
        module_generation: state.module_generation,
        equivalent_to,
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
        EmbedBackend::Worker(engine) => {
            let engine = Arc::clone(engine);
            let loaded_model = model.loaded_model.clone();
            tokio::task::spawn_blocking(move || {
                let _permit = permit;
                let engine = engine.lock().map_err(|_| EngineError {
                    stage: EngineErrorStage::Inference,
                    risk_class: synapse_core::EngineRiskClass::AbortCapable,
                    message: "worker engine mutex was poisoned during inference".to_string(),
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

async fn execute_rerank(
    runtime: &RuntimeState,
    model: &EmbeddingModel,
    request: RerankRequest,
) -> Result<synapse_core::RerankScores, WireOperationError> {
    let permit = runtime
        .execution
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| {
            WireOperationError::from_stable(
                StableError::queue_full(Some(100)),
                "inline rerank executor is closed",
            )
        })?;
    match &model.backend {
        EmbedBackend::Ort(_) => Err(WireOperationError::from_stable(
            StableError::artifact_invalid(),
            format!("model '{}' does not support rerank.score", model.model_id),
        )),
        EmbedBackend::Worker(engine) => {
            let engine = Arc::clone(engine);
            let loaded_model = model.loaded_model.clone();
            tokio::task::spawn_blocking(move || {
                let _permit = permit;
                let engine = engine.lock().map_err(|_| EngineError {
                    stage: EngineErrorStage::Inference,
                    risk_class: synapse_core::EngineRiskClass::AbortCapable,
                    message: "worker engine mutex was poisoned during rerank".to_string(),
                    retry_after_ms: Some(100),
                    safe_to_retry_same_request: true,
                })?;
                engine.rerank(&loaded_model, request)
            })
            .await
            .map_err(|error| {
                WireOperationError::from_stable(
                    StableError::engine_crashed(Some(100)),
                    format!("rerank worker join failed: {error}"),
                )
            })?
            .map_err(engine_error_to_wire)
        }
    }
}

async fn execute_generate(
    runtime: &RuntimeState,
    model: &EmbeddingModel,
    request: GenerateRequest,
) -> Result<GenerateOutput, WireOperationError> {
    let permit = runtime
        .execution
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| {
            WireOperationError::from_stable(
                StableError::queue_full(Some(100)),
                "inline generate executor is closed",
            )
        })?;
    match &model.backend {
        EmbedBackend::Ort(_) => Err(WireOperationError::from_stable(
            StableError::artifact_invalid(),
            format!(
                "model '{}' does not support microllm.oneshot",
                model.model_id
            ),
        )),
        EmbedBackend::Worker(engine) => {
            let engine = Arc::clone(engine);
            let loaded_model = model.loaded_model.clone();
            tokio::task::spawn_blocking(move || {
                let _permit = permit;
                let engine = engine.lock().map_err(|_| EngineError {
                    stage: EngineErrorStage::Inference,
                    risk_class: synapse_core::EngineRiskClass::AbortCapable,
                    message: "worker engine mutex was poisoned during generate".to_string(),
                    retry_after_ms: Some(100),
                    safe_to_retry_same_request: true,
                })?;
                engine.generate(&loaded_model, request)
            })
            .await
            .map_err(|error| {
                WireOperationError::from_stable(
                    StableError::engine_crashed(Some(100)),
                    format!("generate worker join failed: {error}"),
                )
            })?
            .map_err(engine_error_to_wire)
        }
    }
}

fn engine_error_to_wire(error: EngineError) -> WireOperationError {
    if error.stage == EngineErrorStage::WorkerCrash && error.retry_after_ms.is_none() {
        return WireOperationError::from_stable(StableError::probe_required(), error.message);
    }
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
    alias_table: &AliasTable,
    target_fingerprint: Option<&str>,
    required_fingerprint: Option<&str>,
    allow_equivalent: bool,
    required_epoch: Option<u64>,
) -> Result<(), WireOperationError> {
    if let Some(required_epoch) = required_epoch {
        if required_epoch > alias_table.table_epoch {
            return Err(WireOperationError::from_stable(
                StableError::migration_required(),
                format!(
                    "request requires alias table epoch {required_epoch}, but module is at epoch {}",
                    alias_table.table_epoch
                ),
            ));
        }
    }
    let requested = required_fingerprint.or(target_fingerprint);
    if let Some(requested) = requested {
        if requested != model.fingerprint.0 {
            let equivalent = allow_equivalent
                && equivalent_fingerprints(alias_table, model)
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

fn equivalent_fingerprints(alias_table: &AliasTable, model: &EmbeddingModel) -> Vec<Fingerprint> {
    alias_table
        .equivalent_fingerprints_at(&model.fingerprint, now_ms())
        .into_iter()
        .collect()
}

fn ensure_model_certified(
    state: &ModuleState,
    model: &EmbeddingModel,
) -> Result<(), WireOperationError> {
    match state
        .store
        .get_cert_row(&state.machine_profile_hash, &model.fingerprint)
    {
        Ok(Some(_)) => Ok(()),
        Ok(None) => {
            let stale = state
                .store
                .has_stale_cert_row(&state.machine_profile_hash, &model.fingerprint)
                .unwrap_or(false);
            let message = if stale {
                format!(
                    "fingerprint {} has only stale certification rows for a different machine profile",
                    model.fingerprint.0
                )
            } else {
                format!(
                    "fingerprint {} is not certified on machine profile {}",
                    model.fingerprint.0, state.machine_profile_hash
                )
            };
            Err(WireOperationError::from_stable(
                StableError::not_certified(),
                message,
            ))
        }
        Err(error) => Err(WireOperationError::from_stable(
            StableError::engine_crashed(Some(100)),
            format!("read certification rows: {error}"),
        )),
    }
}

async fn probe_start(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let params: ProbeStartParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid probe.start params: {error}"),
            )
        }
    };
    let now = now_ms();
    let model_filter = params.models;
    let request_key = params
        .request_key
        .filter(|key| !key.trim().is_empty())
        .unwrap_or_else(|| format!("probe:{}:{now}", state.module_generation));
    let admission = match state.store.admit_job(
        &request_key,
        "probe",
        state.module_generation,
        &json!({ "models": model_filter.clone() }),
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
            execute_probe_job(task_state, task_job_id, model_filter).await;
        });
    }
    result_outcome(json!({
        "module_generation": state.module_generation,
        "job_id": record.job_id,
        "state": record.state,
    }))
}

async fn probe_status(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let params: ProbeStatusParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid probe.status params: {error}"),
            )
        }
    };
    match state.store.get_job(&params.job_id) {
        Ok(Some(record)) if record.kind == "probe" => {
            result_outcome(probe_status_payload(&state, &record))
        }
        Ok(Some(_)) => channel_error("invalid_request", "job_id does not refer to a probe job"),
        Ok(None) => channel_error("invalid_request", "unknown or expired job_id"),
        Err(error) => channel_error("store_failure", error.to_string()),
    }
}

async fn execute_probe_job(state: Arc<ModuleState>, job_id: String, model_filter: Vec<String>) {
    if !matches!(
        state
            .store
            .mark_job_running(&job_id, state.module_generation, now_ms()),
        Ok(true)
    ) {
        return;
    }

    let embed_fixture = match probe_fixture() {
        Ok(fixture) => fixture,
        Err(error) => {
            fail_job_with_wire_error(&state, &job_id, false, error);
            return;
        }
    };
    let rerank_fixture = match rerank_probe_fixture() {
        Ok(fixture) => fixture,
        Err(error) => {
            fail_job_with_wire_error(&state, &job_id, false, error);
            return;
        }
    };
    let generate_fixture = match generate_probe_fixture() {
        Ok(fixture) => fixture,
        Err(error) => {
            fail_job_with_wire_error(&state, &job_id, false, error);
            return;
        }
    };
    let selected_model_ids = state
        .runtime
        .catalog
        .lock()
        .map(|catalog| {
            catalog
                .keys()
                .filter(|model_id| {
                    model_filter.is_empty() || model_filter.iter().any(|id| id == *model_id)
                })
                .cloned()
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let mut selected_models = Vec::new();
    for model_id in selected_model_ids {
        let model = match ensure_model_loaded_for_control(Arc::clone(&state), &model_id).await {
            Ok(model) => model,
            Err(error) => {
                fail_job_with_wire_error(
                    &state,
                    &job_id,
                    error.class == ErrorClass::Transient,
                    error,
                );
                return;
            }
        };
        selected_models.push(model);
    }

    let mut lane_results = Vec::new();
    let mut certified_vectors = Vec::new();
    for model in selected_models {
        let probe_result = match model.task {
            ModelTask::Embed => {
                execute_embed_probe_for_model(&state, Arc::clone(&model), &embed_fixture).await
            }
            ModelTask::Rerank => {
                execute_rerank_probe_for_model(&state, Arc::clone(&model), &rerank_fixture).await
            }
            ModelTask::Generate => {
                execute_generate_probe_for_model(&state, Arc::clone(&model), &generate_fixture)
                    .await
            }
        };
        let probe_result = match probe_result {
            Ok(result) => result,
            Err(error) => {
                fail_job_with_wire_error(&state, &job_id, true, error);
                return;
            }
        };
        if let Some(vectors) = probe_result.certified_vectors {
            certified_vectors.push(ProbeLaneVectors {
                model: Arc::clone(&model),
                vectors,
            });
        }
        lane_results.push(probe_result.lane_result);
    }

    let mut alias_results = Vec::new();
    for left_index in 0..certified_vectors.len() {
        for right_index in left_index + 1..certified_vectors.len() {
            let left = &certified_vectors[left_index];
            let right = &certified_vectors[right_index];
            if left.model.fingerprint == right.model.fingerprint {
                continue;
            }
            let evidence = probe_evidence_between(&left.vectors, &right.vectors);
            let passed = evidence.mean_cosine >= state.runtime.probe.mean_cosine_threshold
                && evidence.worst_decile >= state.runtime.probe.worst_decile_rank_overlap_threshold;
            if !passed {
                continue;
            }
            let evidence_json = json!({
                "source": "probe",
                "left_model_id": left.model.model_id,
                "right_model_id": right.model.model_id,
                "metrics": evidence,
            });
            match state.store.declare_alias_pair(
                &left.model.fingerprint,
                &right.model.fingerprint,
                &evidence_json,
                now_ms(),
            ) {
                Ok((changed, table_epoch)) => alias_results.push(json!({
                    "fingerprint_a": left.model.fingerprint,
                    "fingerprint_b": right.model.fingerprint,
                    "changed": changed,
                    "table_epoch": table_epoch,
                })),
                Err(error) => {
                    fail_job_with_wire_error(
                        &state,
                        &job_id,
                        true,
                        WireOperationError::from_stable(
                            StableError::engine_crashed(Some(100)),
                            format!("write alias row: {error}"),
                        ),
                    );
                    return;
                }
            }
        }
    }

    let catalog_model_ids = state
        .runtime
        .catalog
        .lock()
        .map(|catalog| catalog.keys().cloned().collect::<BTreeSet<_>>())
        .unwrap_or_default();
    let perf_rows = match state.store.current_perf_rows(&state.machine_profile_hash) {
        Ok(rows) => rows
            .into_iter()
            .filter(|row| catalog_model_ids.contains(&row.model_id))
            .collect::<Vec<_>>(),
        Err(error) => {
            fail_job_with_wire_error(
                &state,
                &job_id,
                true,
                WireOperationError::from_stable(
                    StableError::engine_crashed(Some(100)),
                    format!("read performance rows: {error}"),
                ),
            );
            return;
        }
    };
    let knob_assignments = compute_knob_assignments(&perf_rows);
    if let Err(error) = state
        .store
        .replace_knob_assignments(&state.machine_profile_hash, &knob_assignments)
    {
        fail_job_with_wire_error(
            &state,
            &job_id,
            true,
            WireOperationError::from_stable(
                StableError::engine_crashed(Some(100)),
                format!("persist knob assignments: {error}"),
            ),
        );
        return;
    }
    let active_assignments = knob_assignments
        .iter()
        .filter(|assignment| assignment.knob == state.runtime.knob)
        .cloned()
        .collect::<Vec<_>>();
    let result = json!({
        "module_generation": state.module_generation,
        "machine_profile_hash": state.machine_profile_hash,
        "machine_profile": state.machine_profile,
        "current_knob": state.runtime.knob,
        "fixture": {
            "items": embed_fixture.items.len(),
            "first_id": embed_fixture.items.first().map(|item| item.id.clone()),
            "generation_command": embed_fixture.generation_command,
        },
        "fixtures": {
            "embed": {
                "items": embed_fixture.items.len(),
                "first_id": embed_fixture.items.first().map(|item| item.id.clone()),
                "generation_command": embed_fixture.generation_command,
            },
            "rerank": {
                "items": rerank_fixture.items.len(),
                "first_id": rerank_fixture.items.first().map(|item| item.id.clone()),
                "generation_command": rerank_fixture.generation_command,
            },
            "generate": {
                "items": generate_fixture.items.len(),
                "first_id": generate_fixture.items.first().map(|item| item.id.clone()),
                "generation_command": generate_fixture.generation_command,
            }
        },
        "lanes": lane_results,
        "aliases": alias_results,
        "knob_assignments": knob_assignments,
        "active_assignments": active_assignments,
    });
    if let Err(error) = state.store.complete_job(&job_id, &result, &[], now_ms()) {
        fail_job_with_wire_error(
            &state,
            &job_id,
            true,
            WireOperationError::from_stable(
                StableError::engine_crashed(Some(100)),
                format!("complete probe job: {error}"),
            ),
        );
    }
}

async fn execute_embed_probe_for_model(
    state: &ModuleState,
    model: Arc<EmbeddingModel>,
    fixture: &ProbeFixture,
) -> Result<ProbeModelResult, WireOperationError> {
    let texts = fixture
        .items
        .iter()
        .map(|item| item.text.as_str())
        .collect::<Vec<_>>();
    let tokenized = match model.tokenizer.tokenize_batch(texts) {
        Ok(tokenized) => tokenized,
        Err(error) => {
            return Ok(ProbeModelResult {
                lane_result: json!({
                    "model_id": model.model_id,
                    "task": "embed",
                    "fingerprint": model.fingerprint,
                    "numeric_profile_id": model.numeric_profile_id,
                    "status": "uncertified",
                    "error": error.to_string(),
                }),
                certified_vectors: None,
            })
        }
    };
    let vectors = match execute_embedding(&state.runtime, &model, tokenized.batch.clone()).await {
        Ok(vectors) => vectors,
        Err(error) => {
            return Ok(ProbeModelResult {
                lane_result: json!({
                    "model_id": model.model_id,
                    "task": "embed",
                    "fingerprint": model.fingerprint,
                    "numeric_profile_id": model.numeric_profile_id,
                    "status": "uncertified",
                    "error": error,
                }),
                certified_vectors: None,
            })
        }
    };
    let evidence = probe_evidence(&vectors, &fixture.items);
    let placement_share = ane_placement_share_for_model(&model)?;
    let quality_passed = evidence.mean_cosine >= state.runtime.probe.mean_cosine_threshold
        && evidence.worst_decile >= state.runtime.probe.worst_decile_rank_overlap_threshold;
    let placement_passed = if model.engine_identity.engine == "ane-coreml-worker" {
        placement_share.is_some_and(|share| share >= state.runtime.probe.ane_placement_threshold)
    } else {
        true
    };
    let passed = quality_passed && placement_passed;
    let certification_evidence = json!({
        "task": "embed",
        "metrics": evidence,
        "ane_placement_share": placement_share,
    });
    let performance = if passed {
        let cold_load_ms =
            model_cold_load_ms(&state.runtime, &model.model_id).ok_or_else(|| {
                WireOperationError::from_stable(
                    StableError::engine_crashed(Some(100)),
                    format!("missing cold-load measurement for '{}'", model.model_id),
                )
            })?;
        let perf = measure_embed_perf(&state.runtime, &model, &tokenized, cold_load_ms).await?;
        store_probe_cert_row(state, &model, certification_evidence.clone())?;
        Some(store_probe_perf_row(
            state,
            &model,
            ModelTask::Embed.as_str(),
            &perf,
        )?)
    } else {
        None
    };
    Ok(ProbeModelResult {
        lane_result: json!({
            "model_id": model.model_id,
            "task": "embed",
            "fingerprint": model.fingerprint,
            "numeric_profile_id": model.numeric_profile_id,
            "status": if passed { "certified" } else { "uncertified" },
            "evidence": evidence,
            "ane_placement_share": placement_share,
            "thresholds": {
                "mean_cosine": state.runtime.probe.mean_cosine_threshold,
                "worst_decile": state.runtime.probe.worst_decile_rank_overlap_threshold,
                "ane_placement_share": state.runtime.probe.ane_placement_threshold,
            },
            "performance": performance,
        }),
        certified_vectors: passed.then_some(vectors),
    })
}

fn ane_placement_share_for_model(
    model: &EmbeddingModel,
) -> Result<Option<f64>, WireOperationError> {
    if model.engine_identity.engine != "ane-coreml-worker" {
        return Ok(None);
    }
    match &model.backend {
        EmbedBackend::Worker(engine) => {
            let engine = engine.lock().map_err(|_| {
                WireOperationError::from_stable(
                    StableError::engine_crashed(Some(100)),
                    "worker engine mutex was poisoned during ANE placement ping",
                )
            })?;
            let ping = engine.ping().map_err(|error| {
                WireOperationError::from_stable(
                    StableError::engine_crashed(Some(100)),
                    format!("ANE placement ping failed: {error}"),
                )
            })?;
            Ok(ping.placement_share)
        }
        EmbedBackend::Ort(_) => Ok(None),
    }
}

async fn execute_rerank_probe_for_model(
    state: &ModuleState,
    model: Arc<EmbeddingModel>,
    fixture: &RerankProbeFixture,
) -> Result<ProbeModelResult, WireOperationError> {
    let mut actual = Vec::new();
    let mut reference = Vec::new();
    for item in &fixture.items {
        if item.candidates.len() != item.scores.len() {
            return Ok(ProbeModelResult {
                lane_result: json!({
                    "model_id": model.model_id,
                    "task": "rerank",
                    "fingerprint": model.fingerprint,
                    "numeric_profile_id": model.numeric_profile_id,
                    "status": "uncertified",
                    "error": format!("rerank fixture '{}' has {} candidates and {} scores", item.id, item.candidates.len(), item.scores.len()),
                }),
                certified_vectors: None,
            });
        }
        let mut texts = Vec::with_capacity(item.candidates.len() + 1);
        texts.push(item.query.as_str());
        texts.extend(item.candidates.iter().map(String::as_str));
        let tokenized = match model.tokenizer.tokenize_batch_without_special_tokens(texts) {
            Ok(tokenized) => tokenized,
            Err(error) => {
                return Ok(ProbeModelResult {
                    lane_result: json!({
                        "model_id": model.model_id,
                        "task": "rerank",
                        "fingerprint": model.fingerprint,
                        "numeric_profile_id": model.numeric_profile_id,
                        "status": "uncertified",
                        "error": error.to_string(),
                    }),
                    certified_vectors: None,
                })
            }
        };
        let mut token_items = tokenized.batch.items;
        let query = token_items.remove(0);
        let scores = match execute_rerank(
            &state.runtime,
            &model,
            RerankRequest {
                query,
                candidates: token_items,
            },
        )
        .await
        {
            Ok(scores) => scores,
            Err(error) => {
                return Ok(ProbeModelResult {
                    lane_result: json!({
                        "model_id": model.model_id,
                        "task": "rerank",
                        "fingerprint": model.fingerprint,
                        "numeric_profile_id": model.numeric_profile_id,
                        "status": "uncertified",
                        "error": error,
                    }),
                    certified_vectors: None,
                })
            }
        };
        actual.extend(scores.scores.into_iter().map(f64::from));
        reference.extend(item.scores.iter().copied().map(f64::from));
    }
    let pearson = pearson_correlation(&actual, &reference);
    let evidence = RerankProbeEvidence {
        pearson,
        pairs: actual.len(),
        requests: fixture.items.len(),
    };
    let passed = pearson >= RERANK_PROBE_PEARSON_THRESHOLD;
    let performance = if passed {
        let cold_load_ms =
            model_cold_load_ms(&state.runtime, &model.model_id).ok_or_else(|| {
                WireOperationError::from_stable(
                    StableError::engine_crashed(Some(100)),
                    format!("missing cold-load measurement for '{}'", model.model_id),
                )
            })?;
        let perf = measure_rerank_perf(&state.runtime, &model, fixture, cold_load_ms).await?;
        store_probe_cert_row(
            state,
            &model,
            json!({ "task": "rerank", "metrics": evidence }),
        )?;
        Some(store_probe_perf_row(
            state,
            &model,
            ModelTask::Rerank.as_str(),
            &perf,
        )?)
    } else {
        None
    };
    Ok(ProbeModelResult {
        lane_result: json!({
            "model_id": model.model_id,
            "task": "rerank",
            "fingerprint": model.fingerprint,
            "numeric_profile_id": model.numeric_profile_id,
            "status": if passed { "certified" } else { "uncertified" },
            "evidence": evidence,
            "thresholds": { "pearson": RERANK_PROBE_PEARSON_THRESHOLD },
            "performance": performance,
        }),
        certified_vectors: None,
    })
}

async fn execute_generate_probe_for_model(
    state: &ModuleState,
    model: Arc<EmbeddingModel>,
    fixture: &GenerateProbeFixture,
) -> Result<ProbeModelResult, WireOperationError> {
    let mut matches = 0_usize;
    let mut examples = Vec::new();
    for item in &fixture.items {
        let tokenized = match model.tokenizer.tokenize_batch([item.prompt.as_str()]) {
            Ok(tokenized) => tokenized,
            Err(error) => {
                return Ok(ProbeModelResult {
                    lane_result: json!({
                        "model_id": model.model_id,
                        "task": "generate",
                        "fingerprint": model.fingerprint,
                        "numeric_profile_id": model.numeric_profile_id,
                        "status": "uncertified",
                        "error": error.to_string(),
                    }),
                    certified_vectors: None,
                })
            }
        };
        let prompt = tokenized.batch.items.into_iter().next().unwrap_or_default();
        let output = match execute_generate(
            &state.runtime,
            &model,
            GenerateRequest {
                prompt,
                max_tokens: item.max_tokens.min(64),
                grammar: None,
            },
        )
        .await
        {
            Ok(output) => output,
            Err(error) => {
                return Ok(ProbeModelResult {
                    lane_result: json!({
                        "model_id": model.model_id,
                        "task": "generate",
                        "fingerprint": model.fingerprint,
                        "numeric_profile_id": model.numeric_profile_id,
                        "status": "uncertified",
                        "error": error,
                    }),
                    certified_vectors: None,
                })
            }
        };
        let actual_label = normalize_probe_label(&output.text);
        let expected_label = normalize_probe_label(&item.expected_label);
        if actual_label == expected_label {
            matches += 1;
        } else if examples.len() < 3 {
            examples.push(json!({
                "id": item.id,
                "expected": item.expected_label,
                "actual": output.text,
            }));
        }
    }
    let evidence = GenerateProbeEvidence {
        label_matches: matches,
        items: fixture.items.len(),
    };
    let passed = matches >= GENERATE_PROBE_MIN_LABEL_MATCHES;
    let performance = if passed {
        let cold_load_ms =
            model_cold_load_ms(&state.runtime, &model.model_id).ok_or_else(|| {
                WireOperationError::from_stable(
                    StableError::engine_crashed(Some(100)),
                    format!("missing cold-load measurement for '{}'", model.model_id),
                )
            })?;
        let perf = measure_generate_perf(&state.runtime, &model, fixture, cold_load_ms).await?;
        store_probe_cert_row(
            state,
            &model,
            json!({ "task": "generate", "metrics": evidence }),
        )?;
        Some(store_probe_perf_row(
            state,
            &model,
            ModelTask::Generate.as_str(),
            &perf,
        )?)
    } else {
        None
    };
    Ok(ProbeModelResult {
        lane_result: json!({
            "model_id": model.model_id,
            "task": "generate",
            "fingerprint": model.fingerprint,
            "numeric_profile_id": model.numeric_profile_id,
            "status": if passed { "certified" } else { "uncertified" },
            "evidence": evidence,
            "thresholds": { "label_matches": GENERATE_PROBE_MIN_LABEL_MATCHES },
            "mismatches": examples,
            "performance": performance,
        }),
        certified_vectors: None,
    })
}

fn store_probe_cert_row(
    state: &ModuleState,
    model: &EmbeddingModel,
    evidence: Value,
) -> Result<(), WireOperationError> {
    let row = CertificationRow {
        machine_profile_hash: state.machine_profile_hash.clone(),
        numeric_profile_id: model.numeric_profile_id.clone(),
        fingerprint: model.fingerprint.clone(),
        certified_at_ms: now_ms(),
        os_build: state.machine_profile.os_build.clone(),
        module_generation: state.module_generation,
        evidence,
    };
    state.store.store_cert_row(&row).map_err(|error| {
        WireOperationError::from_stable(
            StableError::engine_crashed(Some(100)),
            format!("write certification row: {error}"),
        )
    })
}

fn store_probe_perf_row(
    state: &ModuleState,
    model: &EmbeddingModel,
    workload: &str,
    perf: &PerfBenchResult,
) -> Result<PerfRow, WireOperationError> {
    let row = PerfRow {
        machine_profile_hash: state.machine_profile_hash.clone(),
        model_id: model.model_id.clone(),
        workload: workload.to_string(),
        numeric_profile_id: model.numeric_profile_id.clone(),
        fingerprint: model.fingerprint.clone(),
        engine: model.engine_identity.engine.clone(),
        measured_at_ms: now_ms(),
        os_build: state.machine_profile.os_build.clone(),
        module_generation: state.module_generation,
        throughput_tok_s: perf.throughput_tok_s,
        cold_load_ms: perf.cold_load_ms,
        single_item_latency_p50_ms: perf.single_item_latency_p50_ms,
        details: perf.details.clone(),
    };
    state.store.store_perf_row(&row).map_err(|error| {
        WireOperationError::from_stable(
            StableError::engine_crashed(Some(100)),
            format!("write performance row: {error}"),
        )
    })?;
    Ok(row)
}

async fn measure_embed_perf(
    runtime: &RuntimeState,
    model: &EmbeddingModel,
    tokenized: &TokenizedBatch,
    cold_load_ms: f64,
) -> Result<PerfBenchResult, WireOperationError> {
    if tokenized.batch.items.is_empty() {
        return Err(WireOperationError::from_stable(
            StableError::artifact_invalid(),
            format!("probe fixture has no embed items for '{}'", model.model_id),
        ));
    }
    let mut cursor = 0_usize;
    let mut total_tokens = 0_u64;
    let mut batch_samples = 0_usize;
    let started = std::time::Instant::now();
    while total_tokens < PROBE_PERF_TARGET_TOTAL_TOKENS
        || batch_samples < PROBE_PERF_MIN_BATCH_SAMPLES
    {
        let mut batch_items = Vec::new();
        let mut batch_tokens = 0_usize;
        while batch_tokens < PROBE_PERF_BATCH_TOKEN_BUDGET || batch_items.is_empty() {
            let index = cursor % tokenized.batch.items.len();
            let item_tokens = u64::from(
                tokenized
                    .real_token_counts
                    .get(index)
                    .copied()
                    .unwrap_or_default(),
            )
            .max(1);
            if !batch_items.is_empty()
                && batch_tokens.saturating_add(item_tokens as usize) > PROBE_PERF_BATCH_TOKEN_BUDGET
            {
                break;
            }
            batch_items.push(tokenized.batch.items[index].clone());
            batch_tokens = batch_tokens.saturating_add(item_tokens as usize);
            total_tokens = total_tokens.saturating_add(item_tokens);
            cursor += 1;
            if batch_tokens >= PROBE_PERF_BATCH_TOKEN_BUDGET {
                break;
            }
        }
        execute_embedding(runtime, model, TokenBatch { items: batch_items }).await?;
        batch_samples += 1;
    }
    let elapsed_secs = started.elapsed().as_secs_f64().max(f64::EPSILON);
    let throughput_tok_s = total_tokens as f64 / elapsed_secs;
    let mut latency_samples = Vec::with_capacity(PROBE_PERF_SINGLE_SAMPLES);
    for sample in 0..PROBE_PERF_SINGLE_SAMPLES {
        let index = sample % tokenized.batch.items.len();
        let started = std::time::Instant::now();
        execute_embedding(
            runtime,
            model,
            TokenBatch {
                items: vec![tokenized.batch.items[index].clone()],
            },
        )
        .await?;
        latency_samples.push(started.elapsed().as_secs_f64() * 1_000.0);
    }
    let single_item_latency_p50_ms = median_ms(&mut latency_samples);
    Ok(PerfBenchResult {
        throughput_tok_s,
        cold_load_ms,
        single_item_latency_p50_ms,
        details: json!({
            "batch_token_budget": PROBE_PERF_BATCH_TOKEN_BUDGET,
            "target_total_tokens": PROBE_PERF_TARGET_TOTAL_TOKENS,
            "throughput_total_tokens": total_tokens,
            "throughput_samples": batch_samples,
            "single_samples": PROBE_PERF_SINGLE_SAMPLES,
        }),
    })
}

async fn measure_rerank_perf(
    runtime: &RuntimeState,
    model: &EmbeddingModel,
    fixture: &RerankProbeFixture,
    cold_load_ms: f64,
) -> Result<PerfBenchResult, WireOperationError> {
    let mut requests = Vec::new();
    for item in &fixture.items {
        let mut texts = Vec::with_capacity(item.candidates.len() + 1);
        texts.push(item.query.as_str());
        texts.extend(item.candidates.iter().map(String::as_str));
        let tokenized = model
            .tokenizer
            .tokenize_batch_without_special_tokens(texts)
            .map_err(|error| {
                WireOperationError::from_stable(StableError::artifact_invalid(), error.to_string())
            })?;
        let mut token_items = tokenized.batch.items;
        let query = token_items.remove(0);
        let token_cost = token_items
            .iter()
            .map(|candidate| {
                candidate
                    .len()
                    .saturating_add(query.len())
                    .saturating_add(3) as u64
            })
            .sum::<u64>()
            .max(1);
        requests.push((
            RerankRequest {
                query,
                candidates: token_items,
            },
            token_cost,
        ));
    }
    if requests.is_empty() {
        return Err(WireOperationError::from_stable(
            StableError::artifact_invalid(),
            format!("probe fixture has no rerank items for '{}'", model.model_id),
        ));
    }
    let mut cursor = 0_usize;
    let mut total_tokens = 0_u64;
    let mut batch_samples = 0_usize;
    let started = std::time::Instant::now();
    while total_tokens < PROBE_PERF_TARGET_TOTAL_TOKENS
        || batch_samples < PROBE_PERF_MIN_BATCH_SAMPLES
    {
        let mut batch_tokens = 0_usize;
        while batch_tokens < PROBE_PERF_BATCH_TOKEN_BUDGET || batch_tokens == 0 {
            let (request, token_cost) = &requests[cursor % requests.len()];
            execute_rerank(runtime, model, request.clone()).await?;
            batch_tokens = batch_tokens.saturating_add(*token_cost as usize);
            total_tokens = total_tokens.saturating_add(*token_cost);
            cursor += 1;
            if batch_tokens >= PROBE_PERF_BATCH_TOKEN_BUDGET {
                break;
            }
        }
        batch_samples += 1;
    }
    let elapsed_secs = started.elapsed().as_secs_f64().max(f64::EPSILON);
    let throughput_tok_s = total_tokens as f64 / elapsed_secs;
    let mut latency_samples = Vec::with_capacity(PROBE_PERF_SINGLE_SAMPLES);
    for sample in 0..PROBE_PERF_SINGLE_SAMPLES {
        let (request, _) = &requests[sample % requests.len()];
        let started = std::time::Instant::now();
        execute_rerank(runtime, model, request.clone()).await?;
        latency_samples.push(started.elapsed().as_secs_f64() * 1_000.0);
    }
    let single_item_latency_p50_ms = median_ms(&mut latency_samples);
    Ok(PerfBenchResult {
        throughput_tok_s,
        cold_load_ms,
        single_item_latency_p50_ms,
        details: json!({
            "batch_token_budget": PROBE_PERF_BATCH_TOKEN_BUDGET,
            "target_total_tokens": PROBE_PERF_TARGET_TOTAL_TOKENS,
            "throughput_total_tokens": total_tokens,
            "throughput_samples": batch_samples,
            "single_samples": PROBE_PERF_SINGLE_SAMPLES,
        }),
    })
}

async fn measure_generate_perf(
    runtime: &RuntimeState,
    model: &EmbeddingModel,
    fixture: &GenerateProbeFixture,
    cold_load_ms: f64,
) -> Result<PerfBenchResult, WireOperationError> {
    let mut requests = Vec::new();
    for item in &fixture.items {
        let tokenized = model
            .tokenizer
            .tokenize_batch([item.prompt.as_str()])
            .map_err(|error| {
                WireOperationError::from_stable(StableError::artifact_invalid(), error.to_string())
            })?;
        let prompt = tokenized.batch.items.into_iter().next().unwrap_or_default();
        let prompt_tokens = prompt.len() as u64;
        let max_tokens = item.max_tokens.min(64);
        requests.push((
            GenerateRequest {
                prompt,
                max_tokens,
                grammar: None,
            },
            prompt_tokens.max(1),
        ));
    }
    if requests.is_empty() {
        return Err(WireOperationError::from_stable(
            StableError::artifact_invalid(),
            format!(
                "probe fixture has no generate items for '{}'",
                model.model_id
            ),
        ));
    }
    let mut cursor = 0_usize;
    let mut total_tokens = 0_u64;
    let mut batch_samples = 0_usize;
    let started = std::time::Instant::now();
    while total_tokens < PROBE_PERF_TARGET_TOTAL_TOKENS
        || batch_samples < PROBE_PERF_MIN_BATCH_SAMPLES
    {
        let mut batch_tokens = 0_usize;
        while batch_tokens < PROBE_PERF_BATCH_TOKEN_BUDGET || batch_tokens == 0 {
            let (request, prompt_tokens) = &requests[cursor % requests.len()];
            let output = execute_generate(runtime, model, request.clone()).await?;
            let estimated_tokens = prompt_tokens
                .saturating_add(u64::from(request.max_tokens))
                .max(1);
            batch_tokens = batch_tokens.saturating_add(estimated_tokens as usize);
            total_tokens = total_tokens
                .saturating_add(prompt_tokens.saturating_add(output.n_gen as u64).max(1));
            cursor += 1;
            if batch_tokens >= PROBE_PERF_BATCH_TOKEN_BUDGET {
                break;
            }
        }
        batch_samples += 1;
    }
    let elapsed_secs = started.elapsed().as_secs_f64().max(f64::EPSILON);
    let throughput_tok_s = total_tokens as f64 / elapsed_secs;
    let mut latency_samples = Vec::with_capacity(PROBE_PERF_SINGLE_SAMPLES);
    for sample in 0..PROBE_PERF_SINGLE_SAMPLES {
        let (request, _) = &requests[sample % requests.len()];
        let started = std::time::Instant::now();
        execute_generate(runtime, model, request.clone()).await?;
        latency_samples.push(started.elapsed().as_secs_f64() * 1_000.0);
    }
    let single_item_latency_p50_ms = median_ms(&mut latency_samples);
    Ok(PerfBenchResult {
        throughput_tok_s,
        cold_load_ms,
        single_item_latency_p50_ms,
        details: json!({
            "batch_token_budget": PROBE_PERF_BATCH_TOKEN_BUDGET,
            "target_total_tokens": PROBE_PERF_TARGET_TOTAL_TOKENS,
            "throughput_total_tokens": total_tokens,
            "throughput_samples": batch_samples,
            "single_samples": PROBE_PERF_SINGLE_SAMPLES,
        }),
    })
}

fn median_ms(samples: &mut [f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(f64::total_cmp);
    let mid = samples.len() / 2;
    if samples.len() % 2 == 1 {
        samples[mid]
    } else {
        (samples[mid - 1] + samples[mid]) / 2.0
    }
}

fn compute_knob_assignments(perf_rows: &[PerfRow]) -> Vec<KnobAssignmentRow> {
    let mut by_workload = BTreeMap::<String, Vec<&PerfRow>>::new();
    for row in perf_rows {
        by_workload
            .entry(row.workload.clone())
            .or_default()
            .push(row);
    }
    let mut assignments = Vec::new();
    for (workload, rows) in by_workload {
        let Some(performance_pick) = select_performance_row(&rows) else {
            continue;
        };
        let quiet_pick = select_quiet_row(&rows).unwrap_or(performance_pick);
        let balanced_pick = if quiet_pick.throughput_tok_s
            >= performance_pick.throughput_tok_s * BALANCED_QUIET_MIN_THROUGHPUT_RATIO
        {
            quiet_pick
        } else {
            performance_pick
        };
        for (knob, row) in [
            (PerfKnob::Performance, performance_pick),
            (PerfKnob::Balanced, balanced_pick),
            (PerfKnob::Quiet, quiet_pick),
        ] {
            assignments.push(KnobAssignmentRow {
                machine_profile_hash: row.machine_profile_hash.clone(),
                workload: workload.clone(),
                knob,
                model_id: row.model_id.clone(),
                numeric_profile_id: row.numeric_profile_id.clone(),
                fingerprint: row.fingerprint.clone(),
                engine: row.engine.clone(),
                measured_at_ms: row.measured_at_ms,
                os_build: row.os_build.clone(),
                module_generation: row.module_generation,
                throughput_tok_s: row.throughput_tok_s,
                single_item_latency_p50_ms: row.single_item_latency_p50_ms,
            });
        }
    }
    assignments.sort_by(|left, right| {
        left.workload
            .cmp(&right.workload)
            .then_with(|| left.knob.as_str().cmp(right.knob.as_str()))
            .then_with(|| left.model_id.cmp(&right.model_id))
    });
    assignments
}

fn select_performance_row<'a>(rows: &[&'a PerfRow]) -> Option<&'a PerfRow> {
    let mut candidates = rows.to_vec();
    candidates.sort_by(|left, right| {
        right
            .throughput_tok_s
            .total_cmp(&left.throughput_tok_s)
            .then_with(|| {
                left.single_item_latency_p50_ms
                    .total_cmp(&right.single_item_latency_p50_ms)
            })
            .then_with(|| left.model_id.cmp(&right.model_id))
    });
    candidates.into_iter().next()
}

fn select_quiet_row<'a>(rows: &[&'a PerfRow]) -> Option<&'a PerfRow> {
    let mut candidates = rows.to_vec();
    if candidates.iter().any(|row| is_ane_engine(&row.engine)) {
        candidates.retain(|row| is_ane_engine(&row.engine));
    }
    candidates.sort_by(|left, right| {
        engine_power_rank(&left.engine)
            .cmp(&engine_power_rank(&right.engine))
            .then_with(|| right.throughput_tok_s.total_cmp(&left.throughput_tok_s))
            .then_with(|| {
                left.single_item_latency_p50_ms
                    .total_cmp(&right.single_item_latency_p50_ms)
            })
            .then_with(|| left.model_id.cmp(&right.model_id))
    });
    candidates.into_iter().next()
}

fn is_ane_engine(engine: &str) -> bool {
    engine == "ane-coreml-worker"
}

/// Rank engines by expected power use for quiet mode: ANE first, CPU ORT second,
/// and Metal-family workers last. Synapse uses this static ordering in v1 because
/// the module does not yet have direct per-lane power measurements.
fn engine_power_rank(engine: &str) -> u8 {
    if is_ane_engine(engine) {
        0
    } else if engine == "ort" {
        1
    } else {
        2
    }
}

fn probe_status_payload(state: &ModuleState, record: &JobRecord) -> Value {
    let mut payload = job_status_payload(state, record);
    if let Value::Object(map) = &mut payload {
        if let Some(Value::Object(result)) = record.result_json.clone() {
            map.extend(result);
        }
    }
    payload
}

fn probe_fixture() -> Result<ProbeFixture, WireOperationError> {
    serde_json::from_str(include_str!("fixtures/probe_corpus_minilm_ort_fp32.json")).map_err(
        |error| {
            WireOperationError::from_stable(
                StableError::artifact_invalid(),
                format!("decode built-in probe fixture: {error}"),
            )
        },
    )
}

fn rerank_probe_fixture() -> Result<RerankProbeFixture, WireOperationError> {
    serde_json::from_str(include_str!("fixtures/probe_rerank_gte_modernbert_v1.json")).map_err(
        |error| {
            WireOperationError::from_stable(
                StableError::artifact_invalid(),
                format!("decode built-in rerank probe fixture: {error}"),
            )
        },
    )
}

fn generate_probe_fixture() -> Result<GenerateProbeFixture, WireOperationError> {
    serde_json::from_str(include_str!("fixtures/probe_generate_qwen3_0_6b_v1.json")).map_err(
        |error| {
            WireOperationError::from_stable(
                StableError::artifact_invalid(),
                format!("decode built-in generate probe fixture: {error}"),
            )
        },
    )
}

fn probe_evidence(vectors: &[Vec<f32>], items: &[ProbeFixtureItem]) -> ProbeEvidence {
    let reference = items
        .iter()
        .map(|item| item.vector.clone())
        .collect::<Vec<_>>();
    probe_evidence_between(vectors, &reference)
}

fn probe_evidence_between(left: &[Vec<f32>], right: &[Vec<f32>]) -> ProbeEvidence {
    let items = left.len().min(right.len());
    if items == 0 || left.len() != right.len() {
        return ProbeEvidence {
            mean_cosine: 0.0,
            rank_overlap: 0.0,
            worst_decile: 0.0,
            items: 0,
        };
    }
    let mean_cosine = left
        .iter()
        .zip(right)
        .map(|(left, right)| cosine(left, right))
        .sum::<f64>()
        / items as f64;
    let (rank_overlap, worst_decile) = rank_overlap_metrics(left, right);
    ProbeEvidence {
        mean_cosine,
        rank_overlap,
        worst_decile,
        items,
    }
}

fn rank_overlap_metrics(left: &[Vec<f32>], right: &[Vec<f32>]) -> (f64, f64) {
    let n = left.len().min(right.len());
    if n <= 2 {
        return (1.0, 1.0);
    }
    let k = (n / 10).max(1).min(n - 1);
    let mut overlaps = Vec::with_capacity(n);
    for query in 0..n {
        let top_left = top_k_neighbors(query, left, k);
        let top_right = top_k_neighbors(query, right, k);
        let hits = top_left
            .iter()
            .filter(|candidate| top_right.contains(candidate))
            .count();
        overlaps.push(hits as f64 / k as f64);
    }
    overlaps.sort_by(f64::total_cmp);
    let mean = overlaps.iter().sum::<f64>() / overlaps.len() as f64;
    let worst_len = overlaps.len().div_ceil(10).max(1);
    let worst = overlaps[..worst_len].iter().sum::<f64>() / worst_len as f64;
    (mean, worst)
}

fn top_k_neighbors(query: usize, vectors: &[Vec<f32>], k: usize) -> BTreeSet<usize> {
    let mut scored = vectors
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != query)
        .map(|(index, vector)| (cosine(&vectors[query], vector), index))
        .collect::<Vec<_>>();
    scored.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    scored.into_iter().take(k).map(|(_, index)| index).collect()
}

fn pearson_correlation(left: &[f64], right: &[f64]) -> f64 {
    if left.len() != right.len() || left.len() < 2 {
        return 0.0;
    }
    let left_mean = left.iter().sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().sum::<f64>() / right.len() as f64;
    let mut numerator = 0.0;
    let mut left_denominator = 0.0;
    let mut right_denominator = 0.0;
    for (left, right) in left.iter().zip(right) {
        let left_delta = left - left_mean;
        let right_delta = right - right_mean;
        numerator += left_delta * right_delta;
        left_denominator += left_delta * left_delta;
        right_denominator += right_delta * right_delta;
    }
    let denominator = left_denominator.sqrt() * right_denominator.sqrt();
    if denominator <= f64::EPSILON {
        0.0
    } else {
        numerator / denominator
    }
}

fn normalize_probe_label(output: &str) -> String {
    output
        .split(|ch: char| ch.is_ascii_punctuation() || ch.is_whitespace())
        .find(|part| !part.is_empty())
        .unwrap_or("")
        .to_ascii_lowercase()
}

fn cosine(left: &[f32], right: &[f32]) -> f64 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }
    let dot = left
        .iter()
        .zip(right)
        .map(|(left, right)| f64::from(*left) * f64::from(*right))
        .sum::<f64>();
    let left_norm = left
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt();
    let right_norm = right
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt();
    dot / (left_norm * right_norm + 1e-12)
}

async fn aliases_check_index(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let params: AliasesCheckIndexParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid aliases.check_index params: {error}"),
            )
        }
    };
    let alias_table = match state.store.alias_table() {
        Ok(alias_table) => alias_table,
        Err(error) => return channel_error("store_failure", error.to_string()),
    };
    let provenance_set = params
        .provenance_set
        .into_iter()
        .filter(|fingerprint| !fingerprint.trim().is_empty())
        .map(Fingerprint)
        .collect::<BTreeSet<_>>();
    let verdict = alias_table.check_index(&Fingerprint(params.index_fingerprint), &provenance_set);
    result_outcome(json!({
        "module_generation": state.module_generation,
        "table_epoch": alias_table.table_epoch,
        "verdict": verdict,
    }))
}

async fn alias_retract(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    mutate_alias_pair(state, params, AliasMutation::Retract).await
}

async fn alias_declare(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    mutate_alias_pair(state, params, AliasMutation::Declare).await
}

enum AliasMutation {
    Declare,
    Retract,
}

async fn mutate_alias_pair(
    state: Arc<ModuleState>,
    params: Value,
    mutation: AliasMutation,
) -> HandlerOutcome {
    if !state.runtime.alias_admin_enabled {
        return result_outcome(error_payload(
            &state,
            WireOperationError::from_stable(
                StableError::substitution_rejected(),
                "alias admin mutations require alias_admin_enabled config",
            ),
        ));
    }
    let params: AliasPairParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid alias mutation params: {error}"),
            )
        }
    };
    let (left, right, evidence) = match params.fingerprints() {
        Ok(pair) => pair,
        Err(message) => return channel_error("invalid_request", message),
    };
    let result = match mutation {
        AliasMutation::Declare => {
            state
                .store
                .declare_alias_pair(&left, &right, &evidence, now_ms())
        }
        AliasMutation::Retract => {
            state
                .store
                .retract_alias_pair(&left, &right, &evidence, now_ms())
        }
    };
    match result {
        Ok((changed, table_epoch)) => result_outcome(json!({
            "module_generation": state.module_generation,
            "changed": changed,
            "table_epoch": table_epoch,
        })),
        Err(error) => channel_error("store_failure", error.to_string()),
    }
}

fn lane_measurement_rows(state: &ModuleState, fingerprint: &Fingerprint) -> LaneMeasurementRows {
    let current_certification = state
        .store
        .get_cert_row(&state.machine_profile_hash, fingerprint)
        .ok()
        .flatten();
    let latest_certification = if current_certification.is_some() {
        current_certification.clone()
    } else {
        state.store.latest_cert_row(fingerprint).ok().flatten()
    };
    let current_performance = state
        .store
        .get_perf_row(&state.machine_profile_hash, fingerprint)
        .ok()
        .flatten();
    let latest_performance = if current_performance.is_some() {
        current_performance.clone()
    } else {
        state.store.latest_perf_row(fingerprint).ok().flatten()
    };
    LaneMeasurementRows {
        certification_stale: current_certification.is_none() && latest_certification.is_some(),
        performance_stale: current_performance.is_none() && latest_performance.is_some(),
        current_certification,
        latest_certification,
        current_performance,
        latest_performance,
    }
}

fn worker_health_from_slot(slot: &ModelSlotSnapshot) -> Option<worker_host::WorkerHostHealth> {
    slot.loaded
        .as_ref()
        .and_then(|model| worker_health_for_model(model))
}

fn lane_blocking_reason(
    slot: &ModelSlotSnapshot,
    measurements: &LaneMeasurementRows,
    worker_quarantined: bool,
) -> Option<&'static str> {
    if measurements.current_certification.is_some() {
        return None;
    }
    let failed_quarantined = matches!(
        &slot.state,
        ModelRuntimeState::Failed(error) if error.message.contains("quarantined")
    );
    if worker_quarantined || failed_quarantined {
        Some("quarantined")
    } else if !cfg!(target_os = "macos") && matches!(slot.spec.engine.as_str(), "mlx" | "ane") {
        Some("unsupported_platform")
    } else {
        Some("probe_required")
    }
}

fn certification_report_row(state: &ModuleState, row: &CertificationRow, stale: bool) -> Value {
    json!({
        "machine_profile_hash": row.machine_profile_hash,
        "numeric_profile_id": row.numeric_profile_id,
        "fingerprint": row.fingerprint,
        "certified_at_ms": row.certified_at_ms,
        "os_build": row.os_build,
        "module_generation": row.module_generation,
        "stale": stale,
        "stale_os_build": row.os_build != state.machine_profile.os_build,
        "evidence": row.evidence,
    })
}

fn performance_report_row(state: &ModuleState, row: &PerfRow, stale: bool) -> Value {
    json!({
        "machine_profile_hash": row.machine_profile_hash,
        "model_id": row.model_id,
        "workload": row.workload,
        "numeric_profile_id": row.numeric_profile_id,
        "fingerprint": row.fingerprint,
        "engine": row.engine,
        "measured_at_ms": row.measured_at_ms,
        "os_build": row.os_build,
        "module_generation": row.module_generation,
        "throughput_tok_s": row.throughput_tok_s,
        "cold_load_ms": row.cold_load_ms,
        "single_item_latency_p50_ms": row.single_item_latency_p50_ms,
        "stale": stale,
        "stale_os_build": row.os_build != state.machine_profile.os_build,
        "details": row.details,
    })
}

async fn probe_report(state: Arc<ModuleState>) -> HandlerOutcome {
    let slots = state
        .runtime
        .catalog
        .lock()
        .map(|catalog| {
            catalog
                .values()
                .map(|slot| ModelSlotSnapshot {
                    spec: slot.spec.clone(),
                    loaded: slot.loaded.clone(),
                    state: slot.state.clone(),
                    notify: Arc::clone(&slot.notify),
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let knob_assignments = match state.store.knob_assignments(&state.machine_profile_hash) {
        Ok(assignments) => assignments,
        Err(error) => return channel_error("store_failure", error.to_string()),
    };
    let active_assignments = knob_assignments
        .iter()
        .filter(|assignment| assignment.knob == state.runtime.knob)
        .cloned()
        .collect::<Vec<_>>();
    let mut lanes = Vec::with_capacity(slots.len());
    let mut certification_stale = false;
    let mut performance_stale = false;
    for slot in slots {
        let measurements = lane_measurement_rows(&state, &slot.spec.fingerprint);
        certification_stale |= measurements.certification_stale;
        performance_stale |= measurements.performance_stale;
        let worker = worker_health_from_slot(&slot);
        let worker_quarantined = worker
            .as_ref()
            .map(|health| health.quarantined_models > 0)
            .unwrap_or(false);
        let blocking_reason = lane_blocking_reason(&slot, &measurements, worker_quarantined);
        let certification = measurements
            .current_certification
            .as_ref()
            .or(measurements.latest_certification.as_ref())
            .map(|row| certification_report_row(&state, row, measurements.certification_stale));
        let performance = measurements
            .current_performance
            .as_ref()
            .or(measurements.latest_performance.as_ref())
            .map(|row| performance_report_row(&state, row, measurements.performance_stale));
        let error = match &slot.state {
            ModelRuntimeState::Failed(error) => Some(
                serde_json::to_value(error).expect("model error should serialize in probe.report"),
            ),
            _ => None,
        };
        lanes.push(json!({
            "model_id": slot.spec.model_id,
            "task": slot.spec.task,
            "engine": slot.spec.engine,
            "fingerprint": slot.spec.fingerprint,
            "numeric_profile_id": slot.spec.numeric_profile_id,
            "state": model_runtime_state_name(&slot.state),
            "certified": measurements.current_certification.is_some(),
            "certification_stale": measurements.certification_stale,
            "performance_stale": measurements.performance_stale,
            "blocking_reason": blocking_reason,
            "certification": certification,
            "performance": performance,
            "error": error,
            "worker": worker,
        }));
    }
    result_outcome(json!({
        "module_generation": state.module_generation,
        "machine_profile_hash": state.machine_profile_hash,
        "machine_profile": state.machine_profile,
        "current_knob": state.runtime.knob,
        "certification_stale": certification_stale,
        "performance_stale": performance_stale,
        "knob_assignments": knob_assignments,
        "active_assignments": active_assignments,
        "lanes": lanes,
    }))
}

async fn admission_status(state: Arc<ModuleState>) -> HandlerOutcome {
    let scheduler = match state.runtime.scheduler.lock() {
        Ok(scheduler) => scheduler,
        Err(_) => {
            return result_outcome(error_payload(
                &state,
                WireOperationError::from_stable(
                    StableError::queue_full(Some(100)),
                    "inline scheduler state is unavailable",
                ),
            ))
        }
    };
    let predicted_start_delay_ms = if state.runtime.execution.available_permits() == 0 {
        state.runtime.inline.estimated_execution_ms
    } else {
        0
    };
    let lanes = state
        .runtime
        .loaded_models()
        .into_iter()
        .map(|model| {
            let measurements = lane_measurement_rows(&state, &model.fingerprint);
            json!({
                "model_id": model.model_id,
                "fingerprint": model.fingerprint,
                "meeting_deadlines": predicted_start_delay_ms <= state.runtime.inline.max_queue_ms,
                "p50_start_delay_ms": predicted_start_delay_ms,
                "certified": measurements.current_certification.is_some(),
                "certification_stale": measurements.certification_stale,
                "performance_stale": measurements.performance_stale,
            })
        })
        .collect::<Vec<_>>();
    let certification_stale = lanes
        .iter()
        .any(|lane| lane["certification_stale"].as_bool().unwrap_or(false));
    let performance_stale = lanes
        .iter()
        .any(|lane| lane["performance_stale"].as_bool().unwrap_or(false));
    result_outcome(json!({
        "module_generation": state.module_generation,
        "machine_profile_hash": state.machine_profile_hash,
        "current_knob": state.runtime.knob,
        "inline_in_flight_bytes": scheduler.in_flight_bytes,
        "lanes": lanes,
        "certification_stale": certification_stale,
        "performance_stale": performance_stale,
    }))
}

fn worker_health_for_model(model: &EmbeddingModel) -> Option<worker_host::WorkerHostHealth> {
    match &model.backend {
        EmbedBackend::Worker(engine) => engine
            .lock()
            .ok()
            .and_then(|engine| engine.health_snapshot().ok()),
        EmbedBackend::Ort(_) => None,
    }
}

fn module_health(state: &ModuleState) -> ModuleHealth {
    let lanes = state
        .runtime
        .loaded_models()
        .into_iter()
        .map(|model| {
            let measurements = lane_measurement_rows(state, &model.fingerprint);
            LaneHealth {
                model_id: model.model_id.clone(),
                fingerprint: model.fingerprint.clone(),
                certified: measurements.current_certification.is_some(),
                certification_stale: measurements.certification_stale,
                performance_stale: measurements.performance_stale,
                worker: worker_health_for_model(&model),
            }
        })
        .collect::<Vec<_>>();
    ModuleHealth {
        status: "ok".to_string(),
        module_generation: state.module_generation,
        loaded_models: state.runtime.loaded_model_count(),
        machine_profile_hash: state.machine_profile_hash.clone(),
        certification_stale: lanes.iter().any(|lane| lane.certification_stale),
        performance_stale: lanes.iter().any(|lane| lane.performance_stale),
        lanes,
    }
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
    json!({
        "module_generation": state.module_generation,
        "table_epoch": snapshot.table_epoch,
        "models": state.runtime.catalog_entries(),
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
        op("model.unload", Mutate),
        op("models.list", Query),
        op("probe.start", Mutate),
        op("probe.status", Query),
        op("probe.report", Query),
        op("aliases.check_index", Query),
        op("alias.retract", Mutate),
        op("alias.declare", Mutate),
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
            probe: ProbeConfig::default(),
            knob: PerfKnob::default(),
            alias_admin_enabled: false,
            dev: DevConfig::default(),
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

fn parse_model_task(
    configured: Option<&str>,
    engine_name: &str,
    model_id: &str,
) -> Result<ModelTask, ModuleError> {
    let inferred;
    let value = match configured.map(str::trim).filter(|value| !value.is_empty()) {
        Some(value) => value,
        None => {
            let lower_model_id = model_id.to_ascii_lowercase();
            inferred = if engine_name == "llama" || engine_name == "llama.cpp" {
                if lower_model_id.contains("rerank") {
                    "rerank"
                } else if lower_model_id.contains("generate")
                    || lower_model_id.contains("microllm")
                    || lower_model_id.contains("qwen")
                {
                    "generate"
                } else {
                    "embed"
                }
            } else {
                "embed"
            };
            inferred
        }
    };
    match value.to_ascii_lowercase().as_str() {
        "embed" | "embedding" | "embeddings" => Ok(ModelTask::Embed),
        "rerank" | "reranker" | "rerank.score" => Ok(ModelTask::Rerank),
        "generate" | "generation" | "microllm" | "microllm.oneshot" => Ok(ModelTask::Generate),
        other => Err(ModuleError::Config(format!(
            "unsupported model task '{other}' for model '{model_id}'"
        ))),
    }
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
    let mut hasher = Sha256::new();
    if path.is_dir() {
        let mut files = Vec::new();
        collect_hash_files(path, &mut files)?;
        files.sort();
        for file in files {
            let relative = file
                .strip_prefix(path)
                .unwrap_or(&file)
                .to_string_lossy()
                .replace('\\', "/");
            hasher.update(relative.as_bytes());
            hasher.update([0]);
            hash_file_into(&file, &mut hasher)?;
        }
    } else {
        hash_file_into(path, &mut hasher)?;
    }
    Ok(hex::encode(hasher.finalize()))
}

fn collect_hash_files(path: &Path, files: &mut Vec<PathBuf>) -> Result<(), ModuleError> {
    for entry in fs::read_dir(path)
        .map_err(|error| ModuleError::Config(format!("hash {}: {error}", path.display())))?
    {
        let entry = entry
            .map_err(|error| ModuleError::Config(format!("hash {}: {error}", path.display())))?;
        let entry_path = entry.path();
        if entry_path.is_dir() {
            collect_hash_files(&entry_path, files)?;
        } else if entry_path.is_file() {
            files.push(entry_path);
        }
    }
    Ok(())
}

fn hash_file_into(path: &Path, hasher: &mut Sha256) -> Result<(), ModuleError> {
    let mut file = fs::File::open(path)
        .map_err(|error| ModuleError::Config(format!("hash {}: {error}", path.display())))?;
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
    Ok(())
}

fn request_bytes_for_texts<'text>(texts: impl IntoIterator<Item = &'text str>) -> u64 {
    texts.into_iter().map(|text| text.len() as u64 + 128).sum()
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_token_cost_tracks_actual_token_id_chunks() {
        let batch = TokenBatch {
            items: vec![vec![1, 2, 3], Vec::new(), vec![4, 5]],
        };

        assert_eq!(batch_token_cost(&batch), 6);
    }

    #[test]
    fn huggingface_resolve_url_uses_repo_segments_and_resolve_main() {
        let url = huggingface_resolve_url("Qdrant/all-MiniLM-L6-v2-onnx", "onnx/model.onnx")
            .expect("hf url should resolve");
        assert_eq!(
            url,
            "https://huggingface.co/Qdrant/all-MiniLM-L6-v2-onnx/resolve/main/onnx/model.onnx"
        );
    }

    #[test]
    fn mlx_and_ane_catalog_identities_are_distinct_worker_profiles() {
        let mlx = catalog_model_engine_identity("mlx").unwrap();
        let ane = catalog_model_engine_identity("ane").unwrap();
        assert_eq!(mlx.engine, "mlx-worker");
        assert_eq!(
            mlx.build_flags.get("numeric_profile").map(String::as_str),
            Some("bf16-distinct")
        );
        assert_eq!(ane.engine, "ane-coreml-worker");
        assert_eq!(
            ane.build_flags.get("placement_gate").map(String::as_str),
            Some("neural-engine")
        );
        assert_ne!(mlx, ane);
    }

    #[test]
    fn knob_assignments_prefer_throughput_but_allow_quiet_when_within_ratio() {
        let rows = vec![
            PerfRow {
                machine_profile_hash: "machine-a".to_string(),
                model_id: "metal-fast".to_string(),
                workload: "embed".to_string(),
                numeric_profile_id: NumericProfileId("np-fast".to_string()),
                fingerprint: Fingerprint("fp-fast".to_string()),
                engine: "mlx-worker".to_string(),
                measured_at_ms: 10,
                os_build: "24A1".to_string(),
                module_generation: 1,
                throughput_tok_s: 200.0,
                cold_load_ms: 20.0,
                single_item_latency_p50_ms: 9.0,
                details: json!({}),
            },
            PerfRow {
                machine_profile_hash: "machine-a".to_string(),
                model_id: "ane-quiet".to_string(),
                workload: "embed".to_string(),
                numeric_profile_id: NumericProfileId("np-quiet".to_string()),
                fingerprint: Fingerprint("fp-quiet".to_string()),
                engine: "ane-coreml-worker".to_string(),
                measured_at_ms: 11,
                os_build: "24A1".to_string(),
                module_generation: 1,
                throughput_tok_s: 120.0,
                cold_load_ms: 22.0,
                single_item_latency_p50_ms: 11.0,
                details: json!({}),
            },
        ];

        let assignments = compute_knob_assignments(&rows);
        let performance = assignments
            .iter()
            .find(|row| row.knob == PerfKnob::Performance)
            .expect("performance assignment");
        let balanced = assignments
            .iter()
            .find(|row| row.knob == PerfKnob::Balanced)
            .expect("balanced assignment");
        let quiet = assignments
            .iter()
            .find(|row| row.knob == PerfKnob::Quiet)
            .expect("quiet assignment");

        assert_eq!(performance.model_id, "metal-fast");
        assert_eq!(quiet.model_id, "ane-quiet");
        assert_eq!(balanced.model_id, "ane-quiet");
        assert!(
            quiet.throughput_tok_s
                >= performance.throughput_tok_s * BALANCED_QUIET_MIN_THROUGHPUT_RATIO
        );
    }
}
