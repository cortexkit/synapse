#![forbid(unsafe_code)]

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, VecDeque},
    env,
    ffi::OsString,
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, OnceLock,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
mod backend_check;
mod catalog;
mod catalog_migration;
mod catalog_probe;
mod diagnostics_findings;
mod engram_catalog;
mod event_outbox;
mod lane_pause;
pub use backend_check::check_backend;
use backend_check::{backend_reason, detected_catalog_backends, resolve_worker_binary_sibling};
use catalog_migration::migrate_compiled_catalog_digests;
#[cfg(test)]
mod detection_tests;
#[cfg(test)]
mod rerank_readout_tests;
#[cfg(all(test, unix))]
mod store_integration_tests;
pub use synapse_store as store;
use synapse_store::PerfKnob;
#[cfg(test)]
mod test_binaries;
pub use store::{
    RestoreImportReport, SynapseStore, SynapseStoreError, RECLAIM_FREELIST_MIN_BYTES,
    RECLAIM_FREELIST_PAGE_RATIO_DIVISOR,
};
mod media_jobs;
pub mod transcribe_host;
pub use synapse_worker_host::worker_host;
#[cfg(all(test, unix))]
mod worker_host_integration_tests;
/// Plain, compiled lane declarations and tasks for offline certification. The
/// validator never obtains catalog pins from a checkout or from record fields.
pub fn certification_catalog() -> synapse_certify::Result<synapse_certify::Catalog> {
    let compiled =
        catalog::compiled_catalog().map_err(|error| synapse_certify::Error(error.to_string()))?;
    Ok(synapse_certify::Catalog {
        lanes: compiled
            .models
            .iter()
            .flat_map(|model| {
                model.backends.iter().map(|backend| {
                    (
                        model.id.clone(),
                        backend.backend.clone(),
                        backend.engine.clone(),
                        backend.fingerprint.clone(),
                    )
                })
            })
            .collect(),
        tasks: compiled
            .models
            .iter()
            .map(|model| (model.id.clone(), model.task.clone()))
            .collect(),
    })
}

#[cfg(all(test, target_os = "macos"))]
mod catalog_token_identity_tests;

use cortexkit_lease::{FileLeaseStore, LeaseHandle, LeaseKey, LeaseStore};
use cortexkit_store_types::{sqlite_store_path, Isolation, StorageBackend, StorageDescriptor};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use store::{
    CheckpointItem, JobAdmission, JobAttemptClaim, JobRecord, ModelAssetLocator, ModelCatalogEntry,
    RecommendedBatch, StorageHealthInputs, StoredModelConfig, JOB_STATE_DONE,
    JOB_STATE_FAILED_PERMANENT, JOB_STATE_FAILED_TRANSIENT, JOB_STATE_QUEUED, JOB_STATE_RUNNING,
};
use subc_client_rs::{
    async_trait, BindDecision, ConnectionEnd, HandlerOutcome, HealthReport, ModuleHandler,
    RequestCtx, RouteBindRequest, RouteHandle, SubcModuleError,
};
use subc_protocol::{
    manifest::{
        build_provenance_from_source, BuildGitShaSource, Concurrency, GitTreeState, IdentityScope,
        ManagementOperation, ManagementOperationKind, ModuleManifest, ProviderRole,
    },
    ModuleHelloAckBody, Principal, PROTOCOL_VERSION, SUBC_MODULE_ID_ENV,
};
#[cfg(test)]
use synapse_core::worker_binary_file_name;
use synapse_core::{
    evaluate_cuda_floor, owned_cuda_engine_identity, worker_binary_env_var,
    worker_engine_names::{LLAMA_ENGINE, LLAMA_WORKER_ENGINE},
    worker_runtime_dir_env_var, AdmissionDecision, AdmissionRequest, CacheGcOutcome,
    CertifiedShapeEnvelope, Clock, CudaFloorDecision, EmbedEngine, EngineError, EngineErrorStage,
    EngineIdentity, ErrorClass, Fingerprint, FlashAttentionSetting, GenerateOutput,
    GenerateRequest, LaneBudgetSnapshot, LaneScheduler, LoadedModel, MachineProfile, ModelCache,
    ModelCacheError, ModelCacheIngest, ModelCacheMeta, NormalizationMode, NumericDType,
    NumericProfile, PoolingStrategy, QueueClass, RerankRequest, ResponseEnvelope,
    ResponseProvenance, RuntimeConfig, SanitizedTokenizer, SchedulerConfig, StableError,
    SystemMachineProfileCollector, ThreadPolicyClass, TokenBatch, TokenizationError,
    TokenizedBatch, TokenizerConfig, TruncationDisclosure, ValidatedArtifact, Vectors, WorkRequest,
    WorkerPooling, CUDA_WORKER_ENGINE, OWNED_CUDA_MINIMUM_DEVICE_CC, OWNED_CUDA_MINIMUM_DRIVER_API,
    OWNED_CUDA_PTX_VIRTUAL_ARCH,
};
use synapse_engine_owned::{
    engine_identity_with_projections, ModelFamily as OwnedFamily, OwnedDType,
    OwnedMetalEmbedEngine, ProjectionDtype, TokenizerPolicy as OwnedTokenizerPolicy,
    DEFAULT_ATTENTION_UNITS as OWNED_DEFAULT_ATTENTION_UNITS, PROJECTION_DTYPE_CONFIG_KEY,
};
use thiserror::Error;
use tokio::sync::{Notify, Semaphore};
use tracing::Instrument;
use url::Url;

pub const DEFAULT_MODULE_ID: &str = "synapse";

/// The components this module logs under, one per `tracing` target it emits.
///
/// The fleet logger derives a line's logger name from the target, rooted at the
/// module id: an event sent to target `perf` renders as `synapse.perf`, and
/// that dotted name is what the `CK_LOG` filter matches on
/// (`CK_LOG=error,synapse.perf=info`). Nothing registers this list at runtime —
/// the logger accepts any target, because refusing one inside a logging call
/// would be a worse failure than an undeclared name. The list is therefore
/// documentation, and the source of the manifest's `--loggers` declaration when
/// that lands.
pub const LOG_LOGGERS: &[&str] = &[
    "perf",        // Periodic activity and per-request completion metrics.
    "worker",      // Lines forwarded from supervised worker processes.
    "admission",   // Job admission, refusal, and completion decisions.
    "cert",        // Certification, staleness, and profile rotation events.
    "maintenance", // Garbage collection and temporary-data sweeps.
    "config",      // Configuration loading and validation failures.
];

const DEFAULT_INLINE_MAX_ITEMS: usize = 64;
const DEFAULT_INLINE_MAX_TOKENS: u64 = 8_192;
const DEFAULT_INLINE_BYTE_BUDGET: u64 = 64 * 1024 * 1024;
const DEFAULT_MAX_QUEUE_MS: u64 = 5_000;
const DEFAULT_DEADLINE_MS: u64 = 30_000;
const DEFAULT_ESTIMATED_EXECUTION_MS: u64 = 25;
const DEFAULT_MAX_CONCURRENT_WORKERS: usize = 2;
const DEFAULT_WORKER_LOAD_TIMEOUT_MS: u64 = 900_000;
const DEFAULT_WORKER_FORWARD_LINES_PER_SEC: u32 = 50;
const DEFAULT_TRANSIENT_RETRY_AFTER_MS: u64 = 100;
const DEFAULT_JOB_EXECUTION_TTL_MS: u64 = 24 * 60 * 60 * 1_000;
const DEFAULT_JOB_RESULT_RETENTION_TTL_MS: u64 = 24 * 60 * 60 * 1_000;
const DEFAULT_JOB_RESULT_PAGE_BYTES: usize = 512 * 1024;
const DEFAULT_JOB_BULK_QUANTUM_TOKENS: u64 = 3_072;
const DEFAULT_ENGINE_BATCH_TOKEN_BUDGET: u64 = 3_072;
const MAX_ENGINE_BATCH_ITEMS: usize = 8;
const SYNAPSE_OS_BUILD_OVERRIDE_ENV: &str = "SYNAPSE_OS_BUILD_OVERRIDE";
const SYNAPSE_CONFIG_PATH_ENV: &str = "SYNAPSE_CONFIG_PATH";
const SYNAPSE_EMBED_PROFILE_ENV: &str = "SYNAPSE_EMBED_PROFILE";
const DEFAULT_MICROLLM_MAX_TOKENS: u32 = 512;
const DEFAULT_CACHE_MAX_BYTES: u64 = 32 * 1024 * 1024 * 1024;
const SYNAPSE_SINGLETON_LEASE_SCOPE: &str = "singleton";

struct SynapseSingletonLease {
    _handle: Box<dyn LeaseHandle>,
}

/// Restore the owner-approved trust set from an Engram scratch database.
///
/// An explicit directory supports isolated drills. Production callers omit it
/// so the target is resolved by the same storage path function used when a
/// daemon acknowledgment does not provide a descriptor at module boot.
pub fn restore_import(
    capture_path: &Path,
    store_directory: Option<&Path>,
) -> Result<RestoreImportReport, ModuleError> {
    let descriptor = match store_directory {
        Some(directory) => StorageDescriptor {
            module_id: DEFAULT_MODULE_ID.to_string(),
            storage_namespace: "default".to_string(),
            isolation: Isolation::Module,
            backend: StorageBackend::Sqlite {
                path: directory.join("store.db").to_string_lossy().into_owned(),
            },
        },
        None => resolve_storage_descriptor(&None, DEFAULT_MODULE_ID)?,
    };
    SynapseStore::restore_from_capture(&descriptor, capture_path).map_err(ModuleError::Store)
}

pub async fn run_from_env() -> Result<(), ModuleError> {
    // This is the process's first read of the launch nonce, and it happens
    // before anything is spawned: the SDK reads the nonce pipe the daemon left
    // on descriptor 3 once, closes it and caches the value, so no worker this
    // module starts later can inherit the pipe. HELLO reuses the cached value.
    let module_id =
        module_id_from_environment(|key| env::var_os(key), LaunchNonceState::from_sdk())?;
    // Under supervision the daemon injects SUBC_MODULE_ID plus the retention
    // knobs (CK_LOG_MAX_AGE_DAYS, CK_LOG_ALARM_SEGMENT_MB) from its `log` config
    // block, and from_env reads all of them. Outside supervision it refuses for a
    // missing SUBC_MODULE_ID, while module_id_from_environment above deliberately
    // falls back to the default id for local runs -- so taking from_env's answer
    // unconditionally would turn a supported unsupervised launch into a startup
    // panic. That one typed error is answered with the id already resolved here;
    // every other arm, and the knob reading itself, stays the crate's.
    let logger_config = match cortexkit_log::Config::from_env() {
        Ok(config) => config,
        Err(cortexkit_log::InitError::ModuleIdNotInEnvironment) => {
            cortexkit_log::Config::for_module(&module_id)
        }
        Err(error) => {
            return Err(ModuleError::Config(format!(
                "initialize fleet logger: {error}"
            )))
        }
    };
    let _logger = cortexkit_log::init(logger_config)
        .map_err(|error| ModuleError::Config(format!("initialize fleet logger: {error}")))?;
    if let Err(error) = write_startup_engram_catalog(&module_id) {
        tracing::warn!(
            target: "lifecycle",
            module = %module_id,
            error = %error,
            "failed to write Engram catalog; startup continues"
        );
    }
    let _singleton = acquire_synapse_singleton_lease(&module_id)?;
    subc_connection_file_from_args()?;
    let handler = SynapseHandler::new(module_id.clone());
    // On the registration-channel launch path the SDK exits the process from
    // inside `on_connection_end` and never returns from `serve`, so the stop line
    // is written in that hook. This slot tells the code after `serve` whether the
    // hook already wrote it.
    let connection_end = Arc::clone(&handler.inner.connection_end);
    // One line when serving starts and one when it ends, so a restart that the
    // daemon or an operator caused can be read from this log alone. Without them
    // a module that restarts and serves nothing leaves no trace at all, and a
    // shutdown can't be told apart from a process that was killed outright:
    // only the second leaves no "stopped" line.
    let started = std::time::Instant::now();
    tracing::info!(
        target: "lifecycle",
        module = %module_id,
        pid = std::process::id(),
        version = env!("CARGO_PKG_VERSION"),
        "synapse started"
    );
    let outcome =
        match subc_client_rs::serve_from_env_with_handle(manifest(&module_id), handler.clone())
            .await
        {
            Ok((handle, serving)) => {
                // The SDK runs on_hello_ack, and with it the whole boot, before it returns
                // this handle, so the boot can't start the publisher; start it here instead.
                // A failed boot panics inside on_hello_ack, so the state is always set by now.
                match (handler.inner.state.get(), subc_connection_file_from_args()) {
                    (Some(state), Ok(path)) => event_outbox::start(state, handle, path),
                    (state, path) => tracing::warn!(
                        target: "lifecycle",
                        state_ready = state.is_some(),
                        connection_file = path.is_ok(),
                        "event publisher not started; committed events stay in the outbox"
                    ),
                }
                serving.await
            }
            Err(error) => Err(error),
        };
    let uptime_s = started.elapsed().as_secs();
    match &outcome {
        // `reason` tells a planned stop (`goodbye`) from a daemon that vanished
        // (`eof`, `reset`) and from this module closing the connection itself
        // (`closed`), which the bare stop line could not. On the error arm the
        // SDK reports no reason; the error itself says why.
        // The hook already logged the stop; only a return without one is logged here.
        Ok(()) if connection_end.get().is_some() => {}
        Ok(()) => tracing::info!(
            target: "lifecycle",
            module = %module_id,
            uptime_s,
            reason = connection_end_reason(None),
            "synapse stopped: serve loop returned"
        ),
        Err(error) => tracing::warn!(
            target: "lifecycle",
            module = %module_id,
            uptime_s,
            error = %error,
            "synapse stopped: serve loop failed"
        ),
    }
    outcome.map_err(ModuleError::Serve)
}

/// The name the `synapse stopped` line gives for how the daemon connection
/// ended. `unknown` only if the SDK returned without reporting an end, which
/// it does not do on a clean return.
fn connection_end_reason(end: Option<ConnectionEnd>) -> &'static str {
    end.map_or("unknown", ConnectionEnd::as_str)
}

/// What this process's launch nonce says about how it was started. The daemon
/// gives a nonce to every module it spawns, so its presence is what marks a
/// supervised launch; a local development run has none.
#[derive(Debug)]
enum LaunchNonceState {
    /// No nonce anywhere: an unsupervised local run.
    Absent,
    /// A nonce was read, from the descriptor or from the environment copy.
    Present,
    /// The daemon named a nonce descriptor that could not be read. The launch
    /// was still meant to be supervised, so it is treated like `Present`.
    Unreadable(String),
}

impl LaunchNonceState {
    /// Reads the nonce through the SDK's single reader. It takes descriptor 3
    /// when the daemon handed the nonce over that way and falls back to the
    /// `SUBC_LAUNCH_NONCE` environment copy otherwise (always, on Windows,
    /// where the daemon has no descriptor handoff). A second reader here would
    /// be wrong, not just redundant: the first read closes the descriptor, and
    /// a later independent read could take whatever reuses that number.
    fn from_sdk() -> Self {
        match subc_client_rs::launch_nonce() {
            Ok(Some(_)) => Self::Present,
            Ok(None) => Self::Absent,
            Err(error) => Self::Unreadable(error.to_string()),
        }
    }
}

fn module_id_from_environment(
    mut env_var: impl FnMut(&str) -> Option<OsString>,
    launch_nonce: LaunchNonceState,
) -> Result<String, ModuleError> {
    let module_id = env_var(SUBC_MODULE_ID_ENV)
        .and_then(|value| value.into_string().ok())
        .filter(|value| !value.trim().is_empty());
    if let Some(module_id) = module_id {
        return Ok(module_id);
    }
    // Under supervision, falling back to DEFAULT_MODULE_ID (even though it is
    // currently the true id) would make a missing daemon-supplied identity look valid.
    // The launch nonce distinguishes supervised launches from local development.
    match launch_nonce {
        LaunchNonceState::Absent => Ok(DEFAULT_MODULE_ID.to_string()),
        LaunchNonceState::Present => Err(ModuleError::Config(format!(
            "{SUBC_MODULE_ID_ENV} is required when a launch nonce is present"
        ))),
        LaunchNonceState::Unreadable(error) => Err(ModuleError::Config(format!(
            "{SUBC_MODULE_ID_ENV} is required when a launch nonce is present \
             (the launch nonce descriptor could not be read: {error})"
        ))),
    }
}

fn subc_connection_file_from_args() -> Result<PathBuf, ModuleError> {
    let mut args = env::args_os().skip(1);
    while let Some(argument) = args.next() {
        if argument == "--subc" {
            return args.next().map(PathBuf::from).ok_or_else(|| {
                ModuleError::Config("--subc requires a connection file path".to_string())
            });
        }
    }
    Err(ModuleError::Config(
        "missing required --subc connection file path".to_string(),
    ))
}

fn acquire_synapse_singleton_lease(module_id: &str) -> Result<SynapseSingletonLease, ModuleError> {
    let lease_root = synapse_lease_root()?;
    let store = FileLeaseStore::new(lease_root);
    let key = LeaseKey::new(module_id, "file", SYNAPSE_SINGLETON_LEASE_SCOPE);
    match store.acquire(&key) {
        Ok(handle) => Ok(SynapseSingletonLease { _handle: handle }),
        Err(cortexkit_lease::LeaseError::Held { .. }) => {
            let message = format!(
                "synapse singleton lease held: only one synapse module may run machine-wide \
                 (module={module_id}, scope={SYNAPSE_SINGLETON_LEASE_SCOPE})"
            );
            tracing::warn!(
                target: "admission",
                module = %module_id,
                scope = SYNAPSE_SINGLETON_LEASE_SCOPE,
                "synapse singleton lease held: only one synapse module may run machine-wide"
            );
            Err(ModuleError::SingletonHeld(message))
        }
        Err(cortexkit_lease::LeaseError::Io(error)) => {
            Err(ModuleError::Config(format!("singleton lease io: {error}")))
        }
    }
}

fn synapse_lease_root() -> Result<PathBuf, ModuleError> {
    synapse_core::resolve_data_root_from_process(synapse_core::DataRoot::Lease)
        .map_err(|error| ModuleError::Config(error.to_string()))
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
    #[error("singleton: {0}")]
    SingletonHeld(String),
}

#[derive(Clone)]
struct SynapseHandler {
    inner: Arc<SynapseHandlerInner>,
}

struct SynapseHandlerInner {
    module_id: String,
    state: OnceLock<Arc<ModuleState>>,
    initialization: OnceLock<tokio::sync::watch::Receiver<Option<Result<(), String>>>>,
    #[cfg(feature = "test-support")]
    initialize_count: std::sync::atomic::AtomicUsize,
    media_principals: Mutex<HashMap<RouteHandle, String>>,
    /// The `flow_id` from each bound route's scope, for routes the daemon
    /// opened on behalf of a flow (an automation acting for its owner).
    /// Recorded at bind; `flow_refusal` uses it to refuse methods on such
    /// routes.
    flow_routes: Mutex<HashMap<RouteHandle, String>>,
    diagnostics_routes: Mutex<HashMap<RouteHandle, bool>>,
    /// How the daemon connection ended, set once by the SDK's end-of-connection
    /// callback and read by the `synapse stopped` log line.
    connection_end: Arc<OnceLock<ConnectionEnd>>,
    /// When the handler was built, just before serving; the stop line reports
    /// uptime from it.
    started: std::time::Instant,
}

impl SynapseHandlerInner {
    /// Remembers whether a bound route's scope belongs to a flow. The stamp is
    /// fixed for the route's life, so recording it once at bind is enough.
    fn record_bind_scope(&self, req: &RouteBindRequest) {
        let diagnostics_allowed = req.scope.as_ref().is_none_or(|scope| {
            if scope.attributes.flow_id.is_some() {
                scope.owner_authorized
            } else {
                scope.kind == subc_protocol::scope::ScopeKind::Head
            }
        });
        if let Ok(mut routes) = self.diagnostics_routes.lock() {
            routes.insert(req.handle, diagnostics_allowed);
        }
        let flow_id = req
            .scope
            .as_ref()
            .and_then(|scope| scope.attributes.flow_id.clone());
        if let Ok(mut flows) = self.flow_routes.lock() {
            match flow_id {
                Some(flow_id) => {
                    flows.insert(req.handle, flow_id);
                }
                None => {
                    flows.remove(&req.handle);
                }
            }
        }
    }

    /// A flow acts for its owner without the owner present, so on a flow's
    /// route synapse serves only diagnostics and stored event bodies. Other
    /// queries can load models or run inference, so query classification alone is not a safe
    /// allowlist for unattended diagnosis.
    fn flow_refusal(&self, route: &RouteHandle, method: &str) -> Option<HandlerOutcome> {
        let flow_id = match self.flow_routes.lock() {
            Ok(flows) => flows.get(route).cloned()?,
            // A poisoned map can't say whether this route is a flow's, so
            // refuse rather than serve a mutation unchecked.
            Err(_) => "unknown".to_string(),
        };
        if matches!(method, "diagnostics.findings" | "events_get") {
            return None;
        }
        Some(channel_error(
            "flow_scope_refused",
            format!(
                "{method} is not available on a flow-scoped route (flow {flow_id}): \
                  flows may call only diagnostics.findings and events_get"
            ),
        ))
    }
}

fn module_state_machine_profile_hashes(profile: &MachineProfile) -> (String, String) {
    (profile.hash(), profile.revisioned_hash())
}

struct ModuleState {
    module_id: String,
    store: Arc<SynapseStore>,
    module_generation: u64,
    machine_profile: MachineProfile,
    machine_profile_hash: String,
    /// Legacy MachineProfile::hash() retained for the existing health field.
    legacy_machine_profile_hash: String,
    /// Revisioned hash used by catalog self-checks and rotation health.
    revisioned_machine_profile_hash: String,
    profile_activation_epoch: u64,
    runtime: Arc<RuntimeState>,
    model_cache: Arc<ModelCache>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct ModuleHealth {
    status: String,
    module_generation: u64,
    loaded_models: usize,
    machine_profile_hash: String,
    lanes: Vec<LaneHealth>,
    previous_revisioned_machine_profile_hash: Option<String>,
    current_revisioned_machine_profile_hash: String,
    profile_activation_epoch: u64,
    last_rotation_at_ms: Option<u64>,
    last_rotation_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct LaneHealth {
    model_id: String,
    fingerprint: Fingerprint,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    details: Option<Value>,
}

impl WireOperationError {
    fn from_stable(error: StableError, message: impl Into<String>) -> Self {
        let retry_after_ms = match error.class {
            ErrorClass::Transient => Some(
                error
                    .retry_after_ms
                    .unwrap_or(DEFAULT_TRANSIENT_RETRY_AFTER_MS),
            ),
            ErrorClass::Permanent => error.retry_after_ms,
        };
        Self {
            code: serde_json::to_value(error.code)
                .expect("stable error code serializes")
                .as_str()
                .expect("stable error code is a string")
                .to_string(),
            class: error.class,
            retry_after_ms,
            safe_to_retry_same_request: error.safe_to_retry_same_request,
            message: message.into(),
            details: error.details.map(Value::Object),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ModuleConfig {
    /// Discloses engine-bound token IDs for candidate certification, not consumer APIs.
    #[serde(default)]
    certify_observation: bool,
    #[serde(default = "default_hf_endpoint")]
    hf_endpoint: String,
    #[serde(default)]
    preload_models: Vec<PreloadModelConfig>,
    #[serde(default)]
    inline: InlineConfig,
    #[serde(default)]
    worker: WorkerConfig,
    #[serde(default)]
    ane_direct: AneDirectConfig,
    #[serde(default)]
    log: LogConfig,
    #[serde(default)]
    jobs: JobConfig,
    #[serde(default)]
    knob: PerfKnob,
    #[serde(default = "default_microllm_max_tokens")]
    microllm_max_tokens: u32,
    #[serde(default = "default_cache_max_bytes")]
    cache_max_bytes: u64,
}

impl Default for ModuleConfig {
    fn default() -> Self {
        Self {
            certify_observation: false,
            hf_endpoint: default_hf_endpoint(),
            preload_models: Vec::new(),
            inline: InlineConfig::default(),
            worker: WorkerConfig::default(),
            ane_direct: AneDirectConfig::default(),
            log: LogConfig::default(),
            jobs: JobConfig::default(),
            knob: PerfKnob::default(),
            microllm_max_tokens: default_microllm_max_tokens(),
            cache_max_bytes: default_cache_max_bytes(),
        }
    }
}

fn embedding_profile_enabled() -> bool {
    env::var_os(SYNAPSE_EMBED_PROFILE_ENV).is_some_and(|value| value.to_string_lossy() != "0")
}

fn ane_lane_timing(stage: &str, model_id: &str, job_id: Option<&str>, started: Instant) {
    if let Some(dir) = env::var_os("SYNAPSE_ANE_PROFILE_DIR") {
        use std::io::Write;
        let record = json!({"stage":stage, "model_id":model_id, "job_id":job_id,
            "ms":started.elapsed().as_secs_f64()*1000.0, "unix_us":std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_micros()});
        let path = PathBuf::from(dir).join(format!("module-{}.jsonl", std::process::id()));
        if let Ok(mut file) = fs::OpenOptions::new().create(true).append(true).open(path) {
            let _ = file.write_all(format!("{record}\n").as_bytes());
        }
    }
}

fn default_microllm_max_tokens() -> u32 {
    DEFAULT_MICROLLM_MAX_TOKENS
}

fn default_cache_max_bytes() -> u64 {
    DEFAULT_CACHE_MAX_BYTES
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct AneDirectConfig {
    // Packing is a per-device serving optimization, not part of numerical identity.
    #[serde(default)]
    tight_packing: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkerConfig {
    #[serde(default = "default_worker_load_timeout_ms")]
    load_timeout_ms: u64,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            load_timeout_ms: default_worker_load_timeout_ms(),
        }
    }
}

fn default_worker_load_timeout_ms() -> u64 {
    DEFAULT_WORKER_LOAD_TIMEOUT_MS
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LogConfig {
    #[serde(default)]
    perf_interval_secs: u64,
    #[serde(default = "default_worker_forward_lines_per_sec")]
    worker_forward_lines_per_sec: u32,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            perf_interval_secs: 0,
            worker_forward_lines_per_sec: default_worker_forward_lines_per_sec(),
        }
    }
}

fn default_worker_forward_lines_per_sec() -> u32 {
    DEFAULT_WORKER_FORWARD_LINES_PER_SEC
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PreloadModelConfig {
    #[serde(default)]
    model_id: Option<String>,
    engine: String,
    #[serde(default)]
    profile: Option<String>,
    #[serde(default)]
    prompt_template: Option<String>,
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
    backend: Option<String>,
    #[serde(default)]
    family: Option<String>,
    #[serde(default)]
    dtype: Option<String>,
    #[serde(default)]
    kernel_revision: Option<String>,
    #[serde(default)]
    ptx_virtual_arch: Option<String>,
    #[serde(default)]
    minimum_device_cc: Option<f32>,
    #[serde(default)]
    minimum_cuda_driver_api: Option<u32>,
    #[serde(default)]
    execution: Option<String>,
    #[serde(default)]
    attention_units: Option<usize>,
    #[serde(default)]
    worker_bin: Option<PathBuf>,
    #[serde(default)]
    worker_runtime_dir: Option<PathBuf>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
struct JobConfig {
    #[serde(default = "default_job_execution_ttl_ms")]
    execution_ttl_ms: u64,
    #[serde(default = "default_job_result_retention_ttl_ms")]
    result_retention_ttl_ms: u64,
    #[serde(default = "default_job_result_page_bytes")]
    result_page_bytes: usize,
    #[serde(default = "default_job_bulk_quantum_tokens")]
    bulk_quantum_tokens: u64,
}

impl Default for JobConfig {
    fn default() -> Self {
        Self {
            execution_ttl_ms: default_job_execution_ttl_ms(),
            result_retention_ttl_ms: default_job_result_retention_ttl_ms(),
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

fn default_job_execution_ttl_ms() -> u64 {
    DEFAULT_JOB_EXECUTION_TTL_MS
}

fn default_job_result_retention_ttl_ms() -> u64 {
    DEFAULT_JOB_RESULT_RETENTION_TTL_MS
}

fn default_job_result_page_bytes() -> usize {
    DEFAULT_JOB_RESULT_PAGE_BYTES
}

fn default_job_bulk_quantum_tokens() -> u64 {
    DEFAULT_JOB_BULK_QUANTUM_TOKENS
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct AdmissionRefusalCounter {
    count: u64,
    last_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct AdmissionTelemetrySnapshot {
    refusals: BTreeMap<String, AdmissionRefusalCounter>,
    jobs_minted: u64,
    jobs_completed: u64,
    jobs_failed: u64,
    jobs_inherited: u64,
    jobs_open: u64,
}

#[derive(Default)]
struct AdmissionTelemetry {
    jobs_minted: AtomicU64,
    jobs_completed: AtomicU64,
    jobs_failed: AtomicU64,
    jobs_inherited: AtomicU64,
    refusals: Mutex<BTreeMap<String, AdmissionRefusalCounter>>,
}

impl AdmissionTelemetry {
    fn record_refusal(&self, reason: &str) {
        if let Ok(mut refusals) = self.refusals.lock() {
            let counter = refusals
                .entry(reason.to_string())
                .or_insert(AdmissionRefusalCounter {
                    count: 0,
                    last_at_ms: 0,
                });
            counter.count = counter.count.saturating_add(1);
            counter.last_at_ms = now_ms();
        }
    }

    fn record_job_minted(&self) {
        self.jobs_minted.fetch_add(1, Ordering::Relaxed);
    }

    fn record_job_completed(&self) {
        self.jobs_completed.fetch_add(1, Ordering::Relaxed);
    }

    fn record_job_failed(&self) {
        self.jobs_failed.fetch_add(1, Ordering::Relaxed);
    }

    fn record_jobs_failed(&self, count: u64) {
        self.jobs_failed.fetch_add(count, Ordering::Relaxed);
    }

    fn record_jobs_inherited(&self, count: u64) {
        self.jobs_inherited.fetch_add(count, Ordering::Relaxed);
    }

    fn snapshot(&self) -> AdmissionTelemetrySnapshot {
        let jobs_minted = self.jobs_minted.load(Ordering::Relaxed);
        let jobs_completed = self.jobs_completed.load(Ordering::Relaxed);
        let jobs_failed = self.jobs_failed.load(Ordering::Relaxed);
        let jobs_inherited = self.jobs_inherited.load(Ordering::Relaxed);
        let jobs_open = jobs_minted
            .saturating_add(jobs_inherited)
            .saturating_sub(jobs_completed)
            .saturating_sub(jobs_failed);
        AdmissionTelemetrySnapshot {
            refusals: self
                .refusals
                .lock()
                .map(|refusals| refusals.clone())
                .unwrap_or_default(),
            jobs_minted,
            jobs_completed,
            jobs_failed,
            jobs_inherited,
            jobs_open,
        }
    }
}

struct RuntimeState {
    findings_changed: Arc<Notify>,
    findings_recompute: Mutex<()>,
    certify_observation: bool,
    media: media_jobs::Runtime,
    ane_direct: AneDirectConfig,
    hf_endpoint: String,
    release_catalog: catalog::Catalog,
    runnable_backends: BTreeSet<String>,
    backend_reasons: BTreeMap<String, &'static str>,
    backend_floor_observed: BTreeMap<String, Value>,
    catalog_locks: Mutex<BTreeMap<String, Arc<tokio::sync::Mutex<()>>>>,
    serving_gates: Mutex<BTreeMap<String, Arc<tokio::sync::RwLock<()>>>>,
    lane_execution_locks: Mutex<BTreeMap<String, Arc<tokio::sync::Mutex<()>>>>,
    /// Each lane's in-flight numerical self-check, for callers to join.
    lane_checks: Mutex<BTreeMap<String, tokio::sync::watch::Receiver<LaneCheckOutcome>>>,
    validated_readiness: Mutex<ValidatedReadinessCache>,
    #[cfg(test)]
    profile_check_passes_for_test: Mutex<bool>,
    #[cfg(test)]
    lane_check_tasks: std::sync::atomic::AtomicUsize,
    #[cfg(test)]
    model_load_tasks: std::sync::atomic::AtomicUsize,
    /// Holds a certification caller between its fast-path miss and joining
    /// the lane's check.
    #[cfg(test)]
    fast_path_gate: Mutex<Option<(Arc<Notify>, Arc<Notify>)>>,
    #[cfg(test)]
    resolver_ready_delay: Mutex<Option<Duration>>,
    download_locks: Mutex<BTreeMap<String, Arc<Mutex<()>>>>,
    download_bytes: Mutex<BTreeMap<String, (u64, u64)>>,
    catalog_disk: Mutex<()>,
    self_check_holders: Mutex<BTreeMap<String, String>>,
    catalog_jobs: Mutex<BTreeMap<String, String>>,
    inline: InlineConfig,
    jobs: JobConfig,
    worker_load_timeout: Duration,
    log: LogConfig,
    knob: PerfKnob,
    microllm_max_tokens: u32,
    cache_max_bytes: u64,
    scheduler: Arc<Mutex<InlineScheduler>>,
    execution: Arc<Semaphore>,
    execution_stats: Arc<Mutex<InlineExecutionStats>>,
    control_loads: Arc<Semaphore>,
    ane_supervisor: Mutex<Option<worker_host::ane_residency::AneResidencySupervisor>>,
    /// Captured at startup for production; unit tests must configure a path in
    /// their store fixture instead of falling back to the machine-wide lock.
    #[cfg(unix)]
    ane_lane_lock_path: Option<PathBuf>,
    #[cfg(all(test, unix))]
    ane_lane_test_root: Option<PathBuf>,
    lane_pause: lane_pause::LanePause,
    catalog: Arc<Mutex<BTreeMap<String, ModelSlot>>>,
    admission_telemetry: Arc<AdmissionTelemetry>,
    activity_telemetry: Arc<ActivityTelemetry>,
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

#[derive(Default)]
struct ValidatedReadinessCache {
    revision: u64,
    models: BTreeMap<String, ValidatedReadiness>,
}

struct ValidatedReadiness {
    model: std::sync::Weak<EmbeddingModel>,
    check_id: String,
    generation: u64,
    install_validated: bool,
}

#[derive(Clone, Debug)]
enum ModelRuntimeState {
    Unloaded,
    Validating,
    Loading,
    Ready,
    Failed(WireOperationError),
}

struct InlineScheduler {
    in_flight_bytes: u64,
}

const EXECUTION_WAIT_SAMPLE_LIMIT: usize = 256;

struct InlineExecutionStats {
    waiters: u64,
    in_flight: u64,
    wait_samples_ms: VecDeque<f64>,
}

#[derive(Default)]
struct ActivityTelemetry {
    by_model: Mutex<BTreeMap<String, u64>>,
    completed_tokens: AtomicU64,
    inline_sequence: AtomicU64,
    #[cfg(unix)]
    packed_preparations: Mutex<BTreeMap<String, Arc<PackedPreparation>>>,
}

#[cfg(unix)]
struct PackedPreparation {
    runtime: std::sync::Weak<RuntimeState>,
    model: std::sync::Weak<EmbeddingModel>,
    executor: tokio::runtime::Handle,
    state: Mutex<PackedPreparationState>,
}

#[cfg(unix)]
#[derive(Default)]
struct PackedPreparationState {
    running: bool,
    retry: bool,
    finished: bool,
}

fn packed_lane_enabled(runtime: &RuntimeState, model: &EmbeddingModel) -> bool {
    runtime.ane_direct.tight_packing
        && model.model_id == "qwen3-embedding-0.6b-ane"
        && model.task == ModelTask::Embed
        && is_direct_ane(model)
        && resolved_catalog_lane(runtime, &model.model_id)
            .is_some_and(|(_, backend)| model.fingerprint.0 == backend.fingerprint)
}

fn start_packed_preparation(runtime: &Arc<RuntimeState>, model: &Arc<EmbeddingModel>) {
    #[cfg(unix)]
    if packed_lane_enabled(runtime, model) {
        let telemetry = &runtime.activity_telemetry;
        {
            let mut preparations = telemetry.packed_preparations.lock().expect("preparations");
            if preparations
                .get(&model.model_id)
                .is_none_or(|p| !p.model.ptr_eq(&Arc::downgrade(model)))
            {
                preparations.insert(
                    model.model_id.clone(),
                    Arc::new(PackedPreparation {
                        runtime: Arc::downgrade(runtime),
                        model: Arc::downgrade(model),
                        executor: tokio::runtime::Handle::current(),
                        state: Mutex::new(PackedPreparationState::default()),
                    }),
                );
            }
        }
        telemetry.prepare_when_idle(&model.model_id);
    }
    #[cfg(not(unix))]
    let _ = (runtime, model);
}

impl ActivityTelemetry {
    #[cfg(unix)]
    fn prepare_when_idle(self: &Arc<Self>, model_id: &str) {
        if self
            .by_model
            .lock()
            .expect("activity")
            .contains_key(model_id)
        {
            return;
        }
        let Some(preparation) = self
            .packed_preparations
            .lock()
            .expect("preparations")
            .get(model_id)
            .cloned()
        else {
            return;
        };
        {
            let mut state = preparation.state.lock().expect("preparation state");
            if state.finished {
                return;
            }
            if state.running {
                // An idle transition during compilation must not lose a deferred retry.
                state.retry = true;
                return;
            }
            state.running = true;
        }
        let telemetry = self.clone();
        let model_id = model_id.to_string();
        let executor = preparation.executor.clone();
        executor.spawn(async move {
            let mut prepared = false;
            if let (Some(runtime), Some(model)) =
                (preparation.runtime.upgrade(), preparation.model.upgrade())
            {
                let pause_guard = lane_pause::admit(&runtime, &model_id);
                let current = pause_guard.is_ok() && model_slot_snapshot(&runtime, &model_id).is_some_and(|slot| {
                    matches!(slot.state, ModelRuntimeState::Ready)
                        && slot.loaded.is_some_and(|loaded| Arc::ptr_eq(&loaded, &model))
                });
                // A shared serving guard prevents unload without blocking new
                // requests on the lifecycle gate. Their own deadlines then bound
                // the worker-stream wait and report shape_compiling during a compile.
                if current && !telemetry.by_model.lock().expect("activity").contains_key(&model_id) {
                    if let Ok(_gate) = lane_serving_gate(&runtime, &model_id).try_read_owned() {
                        if let EmbedBackend::DirectAne(engine) = &model.backend {
                            match engine.serving.prepare_packed().await {
                                Ok(qualified) => prepared = qualified,
                                Err(error) => tracing::warn!(model_id, %error, "packed preparation deferred; serving standalone"),
                            }
                        }
                    }
                }
            }
            let retry = {
                let mut state = preparation.state.lock().expect("preparation state");
                state.running = false;
                state.finished = prepared;
                std::mem::take(&mut state.retry) && !prepared
            };
            if retry {
                telemetry.prepare_when_idle(&model_id);
            }
        });
    }

    fn begin(self: &Arc<Self>, model_id: &str) -> ActivityGuard {
        if let Ok(mut by_model) = self.by_model.lock() {
            let count = by_model.entry(model_id.to_string()).or_default();
            *count = count.saturating_add(1);
        }
        ActivityGuard {
            telemetry: Arc::clone(self),
            model_id: model_id.to_string(),
        }
    }

    fn record_completed_tokens(&self, tokens: u64) {
        self.completed_tokens.fetch_add(tokens, Ordering::Relaxed);
    }

    fn next_inline_job_id(&self, module_generation: u64) -> String {
        let sequence = self.inline_sequence.fetch_add(1, Ordering::Relaxed);
        format!("inline-{module_generation}-{sequence}")
    }
}

struct ActivityGuard {
    telemetry: Arc<ActivityTelemetry>,
    model_id: String,
}

impl Drop for ActivityGuard {
    fn drop(&mut self) {
        let mut idle = false;
        if let Ok(mut by_model) = self.telemetry.by_model.lock() {
            let remove = by_model.get_mut(&self.model_id).is_some_and(|count| {
                *count = count.saturating_sub(1);
                *count == 0
            });
            if remove {
                by_model.remove(&self.model_id);
                idle = true;
            }
        }
        #[cfg(unix)]
        if idle {
            self.telemetry.prepare_when_idle(&self.model_id);
        }
        #[cfg(not(unix))]
        let _ = idle;
    }
}

struct InlineExecutionPermit {
    _permit: tokio::sync::OwnedSemaphorePermit,
    stats: Arc<Mutex<InlineExecutionStats>>,
}

impl Drop for InlineExecutionPermit {
    fn drop(&mut self) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.in_flight = stats.in_flight.saturating_sub(1);
        }
    }
}

struct InlineAdmission {
    scheduler: Arc<Mutex<InlineScheduler>>,
    request_bytes: u64,
    deadline: tokio::time::Instant,
}

impl InlineAdmission {
    fn deadline(&self) -> tokio::time::Instant {
        self.deadline
    }
}

struct InlineWorkBudget {
    request_bytes: u64,
    deadline: Option<tokio::time::Instant>,
    job_id: String,
    started: Instant,
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
    fingerprint: Fingerprint,
    engine_identity: EngineIdentity,
    owned_tokenizer_policy: Option<OwnedTokenizerPolicy>,
}

#[derive(Clone, Debug, Default)]
struct ExecutionModelInfo {
    dims: Option<usize>,
    buckets: Option<Vec<usize>>,
    dtype: Option<String>,
}

#[cfg(feature = "test-support")]
mod test_deterministic;

impl EmbeddingModel {
    fn execution_info(&self) -> ExecutionModelInfo {
        match &self.backend {
            EmbedBackend::Owned(engine) => {
                if let Ok(engine) = engine.lock() {
                    if let Some(info) = engine.model_info(&self.loaded_model) {
                        return ExecutionModelInfo {
                            dims: Some(info.dims),
                            buckets: Some(info.buckets),
                            dtype: Some(info.dtype.as_str().to_string()),
                        };
                    }
                }
                ExecutionModelInfo::default()
            }
            EmbedBackend::Worker(engine) => {
                if let Ok(engine) = engine.lock() {
                    if let Some(info) = engine.model_info(&self.loaded_model) {
                        return ExecutionModelInfo {
                            dims: Some(info.dims),
                            buckets: info.buckets,
                            dtype: None,
                        };
                    }
                }
                ExecutionModelInfo::default()
            }
            #[cfg(unix)]
            EmbedBackend::DirectAne(engine) => ExecutionModelInfo {
                dims: Some(engine.serving.metadata.dims),
                buckets: Some(engine.serving.metadata.buckets.clone()),
                dtype: None,
            },
            #[cfg(feature = "test-support")]
            EmbedBackend::TestDeterministic(_) => ExecutionModelInfo {
                dims: Some(test_deterministic::dims()),
                ..Default::default()
            },
        }
    }
}

fn dtype_for_slot(spec: &StoredModelConfig, exec_dtype: Option<String>) -> Option<String> {
    if let Some(dtype) = exec_dtype {
        return Some(dtype);
    }
    if let Some(owned_dtype) = &spec.owned_dtype {
        return Some(owned_dtype.clone());
    }
    match spec.engine.as_str() {
        CUDA_WORKER_ENGINE => Some("f16".to_string()),
        _ => None,
    }
}

fn device_class_for_engine(engine: &str) -> Option<String> {
    match engine {
        "owned-metal" => Some("metal".to_string()),
        CUDA_WORKER_ENGINE | "cuda" => Some("cuda".to_string()),
        LLAMA_ENGINE | LLAMA_WORKER_ENGINE => Some("cpu".to_string()),
        _ => None,
    }
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

fn execution_lane(model: &EmbeddingModel) -> &'static str {
    execution_lane_for_engine(&model.engine_identity.engine)
}

/// The lane name operators see in `job done` log lines for each engine.
fn execution_lane_for_engine(engine: &str) -> &'static str {
    match engine {
        "owned-metal" => "metal",
        synapse_core::ANE_DIRECT_WORKER_ENGINE => "ane-direct",
        CUDA_WORKER_ENGINE => "cuda",
        synapse_core::VULKAN_WORKER_ENGINE => "vulkan",
        LLAMA_ENGINE | LLAMA_WORKER_ENGINE => "llama",
        _ => "unknown",
    }
}

fn record_admission_refusal(
    runtime: &RuntimeState,
    model_id: &str,
    job_id: Option<&str>,
    reason: &str,
) {
    runtime.admission_telemetry.record_refusal(reason);
    if let Some(job_id) = job_id {
        tracing::warn!(
            target: "admission",
            model_id,
            job_id,
            reason,
            "job refused"
        );
    } else {
        tracing::warn!(target: "admission", model_id, reason, "job refused");
    }
}

fn log_job_admitted(model_id: &str, job_id: &str) {
    tracing::info!(target: "admission", model_id, job_id, "job admitted");
}

fn log_job_done(model_id: &str, job_id: &str, lane: &str, tokens: u64, started: Instant) {
    let wall_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    tracing::info!(
        target: "perf",
        model_id,
        job_id,
        lane,
        tokens,
        wall_ms,
        "job done"
    );
}

fn log_job_failed(model_id: &str, job_id: &str, lane: &str, code: &str, wall_ms: u64) {
    tracing::info!(target: "perf", model_id, job_id, lane, code, wall_ms, "job failed");
}

fn failed_admitted_outcome(
    state: &ModuleState,
    model: &EmbeddingModel,
    job_id: &str,
    started: Instant,
    error: WireOperationError,
) -> HandlerOutcome {
    log_job_failed(
        &model.model_id,
        job_id,
        execution_lane(model),
        &error.code,
        started.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
    );
    result_outcome(error_payload(state, error))
}

#[derive(Clone)]
enum EmbedBackend {
    #[cfg(feature = "test-support")]
    TestDeterministic(Arc<test_deterministic::TestDeterministic>),
    #[cfg(unix)]
    DirectAne(Arc<worker_host::ane_residency::DirectAneEngine>),
    Owned(Arc<Mutex<OwnedMetalEmbedEngine>>),
    Worker(Arc<Mutex<worker_host::WorkerEngine>>),
}

#[derive(Clone, Debug, Serialize)]
struct EmbedVector {
    id: String,
    vector: Vec<f32>,
    content_sha256: String,
    submitted_sha256: String,
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
    required_epoch: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
struct RerankScorePayload {
    scores: Vec<f32>,
    real_token_counts: Vec<u32>,
    truncation_disclosures: Vec<TruncationDisclosure>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
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
    #[serde(default, rename = "allow_equivalent")]
    _allow_equivalent: bool,
    #[serde(default)]
    required_epoch: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
struct MicroLlmOneshotPayload {
    text: String,
    finish_reason: String,
    n_prompt: usize,
    n_gen: usize,
    generated_token_ids: Vec<u32>,
    real_token_counts: Vec<u32>,
    truncation_disclosures: Vec<TruncationDisclosure>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EmbedBatchItemParam {
    Object { id: String, text: String },
    Text(String),
}

#[derive(Clone, Debug, Deserialize)]
struct EmbedBatchItem {
    id: String,
    text: String,
}

struct EmbedBatchJobWork {
    model: Arc<EmbeddingModel>,
    request_digest: String,
    ids: Vec<String>,
    tokenized: TokenizedBatch,
    request_bytes: u64,
    total_tokens: u64,
}

struct PreparedJobPage {
    page_no: u32,
    bytes: Vec<u8>,
    checkpoints: Vec<CheckpointItem>,
}

#[derive(Debug, Deserialize)]
struct EmbedResultParams {
    job_id: String,
    #[serde(default)]
    page: Option<u32>,
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

struct SystemClock;

fn recommended_batch_for_engine(engine: &str, _max_tokens: usize) -> Option<RecommendedBatch> {
    match engine {
        // Every owned embed/rerank engine shares the module's batch planner,
        // which clamps a batch to these limits; clients size their calls from
        // this advice and refuse a lane that advertises none.
        "owned-metal"
        | CUDA_WORKER_ENGINE
        | "owned-vulkan"
        | synapse_core::ANE_DIRECT_WORKER_ENGINE => Some(RecommendedBatch {
            rows: MAX_ENGINE_BATCH_ITEMS,
            token_budget: DEFAULT_ENGINE_BATCH_TOKEN_BUDGET,
        }),
        _ => None,
    }
}

fn recommended_batch_for_model(
    spec: &StoredModelConfig,
    max_tokens: Option<usize>,
) -> Option<RecommendedBatch> {
    if spec.task == "transcribe" {
        None
    } else {
        recommended_batch_for_engine(&spec.engine, max_tokens.unwrap_or(spec.max_tokens))
    }
}

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
                initialization: OnceLock::new(),
                #[cfg(feature = "test-support")]
                initialize_count: std::sync::atomic::AtomicUsize::new(0),
                media_principals: Mutex::new(HashMap::new()),
                flow_routes: Mutex::new(HashMap::new()),
                diagnostics_routes: Mutex::new(HashMap::new()),
                connection_end: Arc::new(OnceLock::new()),
                started: std::time::Instant::now(),
            }),
        }
    }

    fn state(&self) -> Option<Arc<ModuleState>> {
        self.inner.state.get().cloned()
    }

    fn initialize(&self, ack: &ModuleHelloAckBody) -> Result<Arc<ModuleState>, ModuleError> {
        #[cfg(feature = "test-support")]
        self.inner.initialize_count.fetch_add(1, Ordering::SeqCst);
        let descriptor = resolve_storage_descriptor(&ack.storage, &self.inner.module_id)?;
        let store = Arc::new(SynapseStore::open(&descriptor)?);
        let module_generation = store.next_module_generation()?;
        let config = load_module_config()?;
        validate_hf_endpoint(&config.hf_endpoint).map_err(ModuleError::Config)?;
        let release_catalog = runtime_release_catalog()?;
        let model_cache = Arc::new(ModelCache::new(ModelCache::default_root()?));
        migrate_compiled_catalog_digests(&store, &model_cache)?;
        let catalog_models =
            sync_and_load_catalog_models(&store, &config, &release_catalog, &model_cache)?;
        // A machine-identity probe that cannot be established refuses the boot
        // rather than substituting a placeholder. A substituted value would
        // rotate the profile hash, fail every certified lane closed, and rotate
        // back on the next boot that happens to succeed -- a silent,
        // self-reverting identity change with nothing in the record to explain
        // it. The daemon surfaces this refusal and retries under its backoff.
        let collected = MachineProfile::collect(
            &SystemMachineProfileCollector,
            catalog_models
                .iter()
                .map(|model| model.engine_identity.clone()),
        )
        .map_err(|error| ModuleError::Config(error.to_string()))?;
        let machine_profile = machine_profile_with_overrides(collected);
        let (machine_profile_hash, revisioned_machine_profile_hash) =
            module_state_machine_profile_hashes(&machine_profile);
        let legacy_machine_profile_hash = machine_profile_hash.clone();
        let profile_activation =
            store.observe_profile(&machine_profile, now_ms(), module_generation)?;
        let profile_activation_epoch = profile_activation
            .state
            .profile_activation_epoch
            .ok_or_else(|| {
                ModuleError::Config("profile activation epoch is missing".to_string())
            })?;
        let runtime = Arc::new(RuntimeState::from_catalog(config, catalog_models)?);
        reconcile_startup_orphans(&store, module_generation, &runtime.admission_telemetry)?;
        let state = Arc::new(ModuleState {
            module_id: self.inner.module_id.clone(),
            store,
            module_generation,
            machine_profile,
            machine_profile_hash,
            legacy_machine_profile_hash,
            revisioned_machine_profile_hash,
            profile_activation_epoch,
            runtime,
            model_cache,
        });
        media_jobs::startup(&state)?;
        startup_catalog_runtime(&state)?;
        Ok(state)
    }
}

fn reconcile_startup_orphans(
    store: &SynapseStore,
    module_generation: u64,
    telemetry: &AdmissionTelemetry,
) -> Result<usize, SynapseStoreError> {
    let restart_error = WireOperationError::from_stable(
        StableError::module_restarted(),
        "module restarted before the durable job reached a terminal result",
    );
    let count = store.fail_prior_generation_incomplete_jobs(
        module_generation,
        &serde_json::to_value(&restart_error).expect("restart error serializes"),
        now_ms(),
    )?;
    if count > 0 {
        telemetry.record_jobs_inherited(count as u64);
        telemetry.record_jobs_failed(count as u64);
    }
    Ok(count)
}

impl RuntimeState {
    fn from_catalog(
        config: ModuleConfig,
        models: Vec<StoredModelConfig>,
    ) -> Result<Self, ModuleError> {
        validate_hf_endpoint(&config.hf_endpoint).map_err(ModuleError::Config)?;
        let release_catalog = runtime_release_catalog()?;
        let (runnable_backends, backend_reasons) = detected_catalog_backends(&config);
        let backend_floor_observed = catalog_floor_observations(&config, &runnable_backends);
        let hf_endpoint = config.hf_endpoint;
        let inline = config.inline;
        let jobs = config.jobs;
        let worker_load_timeout = Duration::from_millis(config.worker.load_timeout_ms);
        let log = config.log;
        let knob = config.knob;
        let microllm_max_tokens = config.microllm_max_tokens;
        let cache_max_bytes = config.cache_max_bytes;
        let scheduler = Arc::new(Mutex::new(InlineScheduler { in_flight_bytes: 0 }));
        let execution = Arc::new(Semaphore::new(inline.max_concurrent_workers.max(1)));
        let execution_stats = Arc::new(Mutex::new(InlineExecutionStats {
            waiters: 0,
            in_flight: 0,
            wait_samples_ms: VecDeque::new(),
        }));
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
            certify_observation: config.certify_observation,
            findings_changed: Arc::new(Notify::new()),
            findings_recompute: Mutex::new(()),
            media: media_jobs::Runtime::default(),
            ane_direct: config.ane_direct,
            hf_endpoint,
            release_catalog,
            runnable_backends,
            backend_floor_observed,
            backend_reasons,
            catalog_locks: Mutex::new(BTreeMap::new()),
            serving_gates: Mutex::new(BTreeMap::new()),
            lane_execution_locks: Mutex::new(BTreeMap::new()),
            lane_checks: Mutex::new(BTreeMap::new()),
            validated_readiness: Mutex::new(ValidatedReadinessCache::default()),
            #[cfg(test)]
            profile_check_passes_for_test: Mutex::new(false),
            #[cfg(test)]
            resolver_ready_delay: Mutex::new(None),
            #[cfg(test)]
            lane_check_tasks: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(test)]
            model_load_tasks: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(test)]
            fast_path_gate: Mutex::new(None),
            download_locks: Mutex::new(BTreeMap::new()),
            download_bytes: Mutex::new(BTreeMap::new()),
            catalog_disk: Mutex::new(()),
            self_check_holders: Mutex::new(BTreeMap::new()),
            catalog_jobs: Mutex::new(BTreeMap::new()),
            inline,
            jobs,
            worker_load_timeout,
            log,
            knob,
            microllm_max_tokens,
            cache_max_bytes,
            scheduler,
            execution,
            execution_stats,
            ane_supervisor: Mutex::new(None),
            #[cfg(all(unix, not(test)))]
            ane_lane_lock_path: worker_host::ane_residency::AneDirectLaneLock::default_path(),
            #[cfg(all(test, unix))]
            ane_lane_lock_path: None,
            #[cfg(all(test, unix))]
            ane_lane_test_root: None,
            lane_pause: lane_pause::LanePause::default(),
            control_loads: Arc::new(Semaphore::new(1)),
            catalog: Arc::new(Mutex::new(catalog)),
            admission_telemetry: Arc::new(AdmissionTelemetry::default()),
            activity_telemetry: Arc::new(ActivityTelemetry::default()),
        })
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

    fn admit_inline_until(
        &self,
        model_id: &str,
        job_id: Option<&str>,
        queue_class: QueueClass,
        request_bytes: u64,
        deadline: tokio::time::Instant,
        max_queue_ms: Option<u64>,
    ) -> Result<InlineAdmission, WireOperationError> {
        let mut admission = self.admit_inline(
            model_id,
            job_id,
            queue_class,
            request_bytes,
            Some(remaining_request_ms(deadline)),
            max_queue_ms,
        )?;
        // Admission accounts in whole milliseconds, but execution must retain
        // the original timer rather than shorten or restart the request budget.
        admission.deadline = deadline;
        Ok(admission)
    }

    fn admit_inline(
        &self,
        model_id: &str,
        job_id: Option<&str>,
        queue_class: QueueClass,
        request_bytes: u64,
        deadline_ms: Option<u64>,
        max_queue_ms: Option<u64>,
    ) -> Result<InlineAdmission, WireOperationError> {
        let clock = SystemClock;
        let now = clock.now_ms();
        let request_budget_ms = deadline_ms.unwrap_or(self.inline.deadline_ms);
        let deadline_at = Some(now.saturating_add(request_budget_ms));
        let deadline = tokio::time::Instant::now() + Duration::from_millis(request_budget_ms);
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
                if let Some(job_id) = job_id {
                    log_job_admitted(model_id, job_id);
                }
                Ok(InlineAdmission {
                    scheduler: Arc::clone(&self.scheduler),
                    request_bytes,
                    deadline,
                })
            }
            AdmissionDecision::Reject(rejection) => {
                let error = WireOperationError::from_stable(rejection.error, rejection.reason);
                record_admission_refusal(self, model_id, job_id, &error.code);
                Err(error)
            }
        }
    }
}

fn start_perf_sampler(state: Arc<ModuleState>) -> Option<tokio::task::JoinHandle<()>> {
    let interval_secs = state.runtime.log.perf_interval_secs;
    if interval_secs == 0 {
        return None;
    }
    Some(tokio::spawn(async move {
        let interval = Duration::from_secs(interval_secs);
        loop {
            tokio::time::sleep(interval).await;
            emit_activity_sample(&state, interval_secs).await;
        }
    }))
}

async fn emit_activity_sample(state: &ModuleState, interval_secs: u64) {
    let (in_flight, waiters) = state
        .runtime
        .execution_stats
        .lock()
        .map(|stats| (stats.in_flight, stats.waiters))
        .unwrap_or_default();
    let completed_tokens = state
        .runtime
        .activity_telemetry
        .completed_tokens
        .swap(0, Ordering::Relaxed);
    if in_flight == 0 {
        tracing::debug!(target: "perf", "activity idle");
        return;
    }
    let by_model = state
        .runtime
        .activity_telemetry
        .by_model
        .lock()
        .map(|counts| {
            counts
                .iter()
                .map(|(model_id, count)| format!("{model_id}:{count}"))
                .collect::<Vec<_>>()
                .join(",")
        })
        .unwrap_or_default();
    let workers = state
        .runtime
        .loaded_models()
        .into_iter()
        .filter_map(|model| match &model.backend {
            EmbedBackend::Worker(engine) => Some((model.model_id.clone(), Arc::clone(engine))),
            #[cfg(unix)]
            EmbedBackend::DirectAne(_) => None,
            #[cfg(feature = "test-support")]
            EmbedBackend::TestDeterministic(_) => None,
            EmbedBackend::Owned(_) => None,
        })
        .collect::<Vec<_>>();
    let worker_rss_mb = tokio::task::spawn_blocking(move || {
        workers
            .into_iter()
            .filter_map(|(model_id, engine)| {
                let engine = engine.try_lock().ok()?;
                let ping = engine.ping().ok()?;
                Some(format!("{model_id}:{}", ping.rss_mb))
            })
            .collect::<Vec<_>>()
            .join(",")
    })
    .await
    .unwrap_or_default();
    let tokens_per_s = if interval_secs == 0 {
        0.0
    } else {
        completed_tokens as f64 / interval_secs as f64
    };
    let queue_depth = waiters;
    tracing::debug!(
        target: "perf",
        in_flight,
        by_model,
        tokens_per_s = format_args!("{tokens_per_s:.3}"),
        queue_depth,
        waiters,
        worker_rss_mb,
        "activity"
    );
}

fn sync_and_load_catalog_models(
    store: &SynapseStore,
    config: &ModuleConfig,
    catalog: &catalog::Catalog,
    cache: &ModelCache,
) -> Result<Vec<StoredModelConfig>, ModuleError> {
    let now = now_ms();
    for (index, preload) in config.preload_models.clone().into_iter().enumerate() {
        let id = preload
            .model_id
            .clone()
            .unwrap_or_else(|| format!("{}-{index}", preload.engine));
        if catalog.is_reserved_id(&id) {
            return Err(ModuleError::Config(format!(
                "preload_models entry '{id}' uses a catalog-reserved id; use models.download"
            )));
        }
        let engine = canonical_engine_name(&preload.engine);
        let task = parse_model_task(preload.task.as_deref(), &engine, &id)?;
        if matches!(task, ModelTask::Embed | ModelTask::Rerank)
            && !store::OWNED_EMBED_RERANK_ENGINES.contains(&engine.as_str())
        {
            tracing::warn!(model_id=%id,%engine,"skipping preload_models entry with disallowed embed/rerank engine");
            continue;
        }
        let spec = build_preload_catalog_model(index, preload, &config.inline, &config.jobs)?;
        store.upsert_model(&spec, now)?;
    }

    let mut normalized = Vec::new();
    let reconciled = store.reconcile_persisted_registrations(
        &|id| catalog.is_reserved_id(id),
        &CatalogCache(cache),
    )?;
    for (id, engine) in &reconciled.deleted {
        tracing::warn!(model_id=%id,%engine,"deleted persisted registration with disallowed embed/rerank engine");
    }
    for id in &reconciled.skipped_reserved {
        tracing::warn!(model_id=%id,"skipping persisted catalog-reserved registration");
    }
    for model in reconciled.registrable {
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
    let catalog = preload
        .profile
        .as_deref()
        .filter(|profile| !is_test_preload_profile(&engine_name, profile))
        .map(CatalogProfile::load)
        .transpose()?;
    if let Some(catalog) = &catalog {
        catalog.validate(&engine_name, task, pooling, normalize, preload.max_tokens)?;
        if catalog.slug == "qwen3-reranker-0.6b" && preload.prompt_template.is_some() {
            return Err(ModuleError::Config(
                "catalog Qwen3 rerank refuses prompt_template".into(),
            ));
        }
    }
    let max_tokens = catalog
        .as_ref()
        .map_or_else(|| preload.max_tokens.unwrap_or(512), |_| 8192);
    let artifact_format = preload
        .format
        .clone()
        .unwrap_or_else(|| default_artifact_format(&engine_name));
    let artifact_digest = match preload.artifact_digest.clone() {
        Some(digest) => normalize_digest(&digest),
        None => format!("sha256:{}", sha256_file(&preload.model_path)?),
    };
    let owned = if let Some(catalog) = &catalog {
        Some(catalog.owned_config(preload.execution.as_deref(), preload.attention_units)?)
    } else {
        (engine_name == "owned-metal" || engine_name == CUDA_WORKER_ENGINE)
            .then(|| {
                if engine_name == CUDA_WORKER_ENGINE {
                    owned_cuda_catalog_config(
                        preload.family.as_deref(),
                        preload.dtype.as_deref(),
                        preload.execution.as_deref(),
                        preload.attention_units,
                        OwnedCudaDeclaredIdentity {
                            kernel_revision: preload.kernel_revision.as_deref(),
                            ptx_virtual_arch: preload.ptx_virtual_arch.as_deref(),
                            minimum_device_cc: preload.minimum_device_cc,
                            minimum_cuda_driver_api: preload.minimum_cuda_driver_api,
                        },
                    )
                } else {
                    owned_catalog_config(
                        &preload.model_path,
                        preload.family.as_deref(),
                        preload.dtype.as_deref(),
                        preload.execution.as_deref(),
                        preload.attention_units,
                        None,
                        Vec::new(),
                    )
                }
            })
            .transpose()?
    };
    let tokenizer_max_tokens = if catalog.is_some() {
        usize::MAX
    } else {
        owned_tokenizer_max_tokens(max_tokens, owned.as_ref())
    };
    let tokenizer = SanitizedTokenizer::from_file(
        &preload.tokenizer_path,
        TokenizerConfig {
            max_tokens: tokenizer_max_tokens,
        },
    )?;
    if let Some(catalog) = &catalog {
        catalog.validate_readout(&tokenizer)?;
    }
    let quant = preload.quant.clone().unwrap_or_else(|| {
        owned
            .as_ref()
            .map(|profile| profile.dtype.as_str().to_string())
            .unwrap_or_else(|| default_quant(&engine_name))
    });
    let declared_llama_backend = (engine_name == LLAMA_ENGINE)
        .then(|| preload.backend.clone())
        .flatten();
    let mut spec = build_stored_model_config(
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
        quant,
        false,
        preload.worker_bin.clone(),
        preload.worker_runtime_dir.clone(),
        Vec::new(),
        owned,
        inline,
        jobs,
    )?;
    #[cfg(feature = "test-support")]
    if preload
        .profile
        .as_deref()
        .is_some_and(|profile| test_deterministic::is_profile(&engine_name, profile))
    {
        spec.engine_identity
            .build_flags
            .insert("profile".into(), test_deterministic::PROFILE.into());
    }
    if let Some(backend) = declared_llama_backend {
        spec.engine_identity
            .build_flags
            .insert("backend".to_string(), backend);
    }
    Ok(spec)
}

fn normalize_catalog_model(
    model: StoredModelConfig,
    inline: &InlineConfig,
    jobs: &JobConfig,
) -> Result<StoredModelConfig, ModuleError> {
    #[cfg(feature = "test-support")]
    if model
        .engine_identity
        .build_flags
        .get("profile")
        .is_some_and(|profile| test_deterministic::is_profile(&model.engine, profile))
    {
        return Ok(model);
    }
    if model.engine_identity.build_flags.contains_key("profile") {
        let catalog = CatalogProfile::load(&model.engine_identity.build_flags["profile"])?;
        let task = parse_model_task(Some(&model.task), &model.engine, &model.model_id)?;
        let pooling = parse_pooling(&model.pooling)?;
        catalog.validate(
            &model.engine,
            task,
            pooling,
            model.normalize,
            Some(model.max_tokens),
        )?;
        let owned = catalog.owned_config(
            model.owned_execution.as_deref(),
            model.owned_attention_units,
        )?;
        return build_stored_model_config(
            model.model_id,
            &model.engine,
            task,
            model.artifact_digest,
            model.artifact_format,
            model.tokenizer_sanitized_digest,
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
            model.extra_locators,
            Some(owned),
            inline,
            jobs,
        );
    }
    let engine_name = canonical_engine_name(&model.engine);
    let task = parse_model_task(Some(&model.task), &engine_name, &model.model_id)?;
    let pooling = parse_pooling(&model.pooling)?;
    let owned = if engine_name == "owned-metal" {
        Some(OwnedCatalogConfig {
            family: OwnedFamily::parse(model.owned_family.as_deref().ok_or_else(|| {
                ModuleError::Config("owned-metal catalog entry is missing family".to_string())
            })?)
            .map_err(|error| ModuleError::Config(error.to_string()))?,
            dtype: OwnedDType::parse(model.owned_dtype.as_deref().ok_or_else(|| {
                ModuleError::Config("owned-metal catalog entry is missing dtype".to_string())
            })?)
            .map_err(|error| ModuleError::Config(error.to_string()))?,
            projections: stored_projection_dtype(&model.engine_identity),
            execution: model
                .owned_execution
                .clone()
                .unwrap_or_else(|| "explicit".to_string()),
            attention_units: model
                .owned_attention_units
                .unwrap_or(OWNED_DEFAULT_ATTENTION_UNITS),
            config_locator: model.config_locator.clone(),
            extra_locators: model.extra_locators.clone(),
            identity_override: None,
        })
    } else if engine_name == CUDA_WORKER_ENGINE {
        let mut profile = owned_cuda_catalog_config(
            model.owned_family.as_deref(),
            model.owned_dtype.as_deref(),
            model.owned_execution.as_deref(),
            model.owned_attention_units,
            OwnedCudaDeclaredIdentity {
                kernel_revision: model
                    .engine_identity
                    .build_flags
                    .get("kernel_revision")
                    .map(String::as_str),
                ptx_virtual_arch: model
                    .engine_identity
                    .build_flags
                    .get("ptx_virtual_arch")
                    .map(String::as_str),
                minimum_device_cc: model
                    .engine_identity
                    .build_flags
                    .get("minimum_device_cc")
                    .and_then(|value| value.parse().ok()),
                minimum_cuda_driver_api: model
                    .engine_identity
                    .build_flags
                    .get("minimum_cuda_driver_api")
                    .and_then(|value| value.parse().ok()),
            },
        )?;
        profile.config_locator = model.config_locator.clone();
        profile.extra_locators = model.extra_locators.clone();
        Some(profile)
    } else {
        None
    };
    let declared_llama_backend = (engine_name == LLAMA_ENGINE)
        .then(|| model.engine_identity.build_flags.get("backend").cloned())
        .flatten();
    let mut spec = build_stored_model_config(
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
        model.extra_locators,
        owned,
        inline,
        jobs,
    )?;
    if let Some(backend) = declared_llama_backend {
        spec.engine_identity
            .build_flags
            .insert("backend".to_string(), backend);
    }
    Ok(spec)
}

struct CatalogProfile {
    id: String,
    slug: String,
    manifest: Value,
    typed: synapse_parity::manifest::Manifest,
}

fn is_test_preload_profile(_engine: &str, _profile: &str) -> bool {
    #[cfg(feature = "test-support")]
    return test_deterministic::is_profile(_engine, _profile);
    #[cfg(not(feature = "test-support"))]
    false
}

fn release_profile(identity: &EngineIdentity) -> Option<&str> {
    identity
        .build_flags
        .get("profile")
        .map(String::as_str)
        .filter(|profile| !is_test_preload_profile(&identity.engine, profile))
}

impl CatalogProfile {
    fn load(id: &str) -> Result<Self, ModuleError> {
        let typed = synapse_parity::manifest::Manifest::from_slice(include_bytes!(
            "../../../bench/parity/models.json"
        ))
        .map_err(|error| ModuleError::Config(error.to_string()))?;
        let manifest = serde_json::to_value(&typed).expect("manifest serializes");
        let profile = manifest["profiles"].get(id).ok_or_else(|| {
            ModuleError::Config(format!("model_unsupported: unknown profile {id}"))
        })?;
        let slug = profile["model"]
            .as_str()
            .ok_or_else(|| ModuleError::Config("profile missing model".into()))?
            .to_string();
        let catalog = Self {
            id: id.into(),
            slug,
            manifest,
            typed,
        };
        catalog.validate_template()?;
        Ok(catalog)
    }

    fn validate_template(&self) -> Result<(), ModuleError> {
        if self.slug == "qwen3-reranker-0.6b" {
            let model = self
                .typed
                .model(&self.slug)
                .map_err(|error| ModuleError::Config(error.to_string()))?;
            let template = model
                .grammar()
                .map_err(|error| ModuleError::Config(error.to_string()))?
                .template
                .as_ref()
                .ok_or_else(|| {
                    ModuleError::Config("qwen_template_mismatch: missing template".into())
                })?;
            synapse_parity::oracle::check_template(
                template,
                include_str!("../../../bench/parity/oracles/qwen3-reranker-0.6b-README.md"),
            )
            .map_err(|error| ModuleError::Config(error.to_string()))?;
        }
        Ok(())
    }

    fn validate_readout(&self, tokenizer: &SanitizedTokenizer) -> Result<(), ModuleError> {
        if self.slug == "qwen3-reranker-0.6b" {
            for role in ["yes", "no"] {
                let token = &self.model()["grammar"]["readout"][role];
                let ids = tokenizer
                    .tokenizer()
                    .encode(token["text"].as_str().expect("readout text"), false)
                    .map_err(|error| ModuleError::Config(error.to_string()))?;
                if ids.get_ids() != [token["id"].as_u64().expect("readout id") as u32] {
                    return Err(ModuleError::Config(
                        "qwen_readout_mismatch: yes/no must resolve to their pinned single ids"
                            .into(),
                    ));
                }
            }
        }
        Ok(())
    }

    fn profile(&self) -> &Value {
        &self.manifest["profiles"][&self.id]
    }
    fn model(&self) -> &Value {
        &self.manifest["models"][&self.slug]
    }
    fn artifact_digest(&self) -> String {
        self.profile()["converted_package_digest"]
            .as_str()
            .unwrap_or_else(|| {
                self.model()["checkpoint_digest"]
                    .as_str()
                    .expect("manifest checkpoint digest")
            })
            .into()
    }
    fn validate(
        &self,
        engine: &str,
        task: ModelTask,
        pooling: WorkerPooling,
        normalize: bool,
        max_tokens: Option<usize>,
    ) -> Result<(), ModuleError> {
        let expected_pooling = match self.model()["grammar"]["pooling"].as_str() {
            Some("cls") => "cls",
            Some("masked_mean") => "mean",
            Some("last_non_pad") => "last",
            _ => return Err(ModuleError::Config("manifest pooling unsupported".into())),
        };
        if self.profile()["lane"] != engine
            || self.model()["operation"] != task.as_str()
            || pooling.as_str() != expected_pooling
            || normalize != (self.model()["output"]["normalization"] == "l2")
            || max_tokens.is_some_and(|limit| limit != 8192)
        {
            return Err(ModuleError::Config("catalog entry disagrees with manifest engine, task, pooling, normalize or max_tokens".into()));
        }
        Ok(())
    }

    fn owned_config(
        &self,
        execution: Option<&str>,
        attention_units: Option<usize>,
    ) -> Result<OwnedCatalogConfig, ModuleError> {
        if self.model()["operation"] == "transcribe" {
            return Err(ModuleError::Config(format!(
                "{}: transcribe lane has no embed owned config",
                self.slug
            )));
        }
        let family = if self.model()["architecture"]["family"] == "qwen3" {
            OwnedFamily::Qwen3
        } else {
            OwnedFamily::GteModernBert
        };
        let dtype = OwnedDType::parse(
            self.profile()["storage_dtype"]
                .as_str()
                .expect("manifest storage dtype"),
        )
        .map_err(|error| ModuleError::Config(error.to_string()))?;
        let lane = self.profile()["lane"].as_str().expect("manifest lane");
        let projections = if lane == "owned-metal" {
            catalog_projection_dtype(family, dtype)
        } else {
            ProjectionDtype::F32
        };
        let mut identity = if lane == "owned-metal" {
            engine_identity_with_projections(family, dtype, projections)
        } else if lane == "owned-cuda" {
            owned_cuda_engine_identity(
                family.as_str(),
                dtype.as_str(),
                synapse_core::CUDA_KERNEL_REVISION,
            )
        } else {
            catalog_model_engine_identity(lane)?
        };
        // StoredModelConfig already persists engine build flags. Store the profile
        // there so reloads retain the token-composition rules and the package
        // identity sent to the worker when loading weights.
        identity
            .build_flags
            .insert("profile".into(), self.id.clone());
        identity.build_flags.insert(
            "compute_dtype".into(),
            self.profile()["compute_dtype"]
                .as_str()
                .expect("manifest compute dtype")
                .into(),
        );
        identity
            .build_flags
            .insert("storage_dtype".into(), dtype.as_str().into());
        // `execution` picks between owned-metal's in-process modes, and the
        // catalog declares `explicit` on every backend. The CUDA engine only
        // runs inside its supervised worker, and owned_cuda_catalog_config
        // refuses anything else when the lane loads, so a CUDA profile always
        // records `supervised`. Execution is not part of the numeric profile,
        // so this does not change the lane fingerprint.
        let execution = if lane == CUDA_WORKER_ENGINE {
            "supervised"
        } else {
            execution.unwrap_or("explicit")
        };
        Ok(OwnedCatalogConfig {
            family,
            dtype,
            projections,
            execution: execution.into(),
            // A catalog profile always serves up to 8192 tokens, and the engine
            // refuses to load a model whose attention budget can't cover
            // max_tokens squared. The generic default (sized for legacy lanes)
            // is far smaller, so a profile preload that doesn't set the budget
            // would fail to load at all; default it to what the profile needs.
            attention_units: attention_units.unwrap_or(8192 * 8192),
            config_locator: None,
            extra_locators: Vec::new(),
            identity_override: Some(identity),
        })
    }

    fn transcribe_config(&self) -> Result<TranscribeCatalogConfig, ModuleError> {
        if self.model()["operation"] != "transcribe" {
            return Err(ModuleError::Config(format!(
                "{}: embed/rerank lane has no transcribe config",
                self.slug
            )));
        }
        let profile = transcribe_host::TranscribeProfile::pinned(&self.id)
            .map_err(|error| ModuleError::Config(error.to_string()))?;
        let dtype = OwnedDType::parse(
            self.profile()["storage_dtype"]
                .as_str()
                .expect("storage dtype"),
        )
        .map_err(|error| ModuleError::Config(error.to_string()))?;
        let lane = profile.lane.as_str();
        let revision = profile.kernel_revision.as_str();
        let mut identity = match lane {
            "owned-metal" => EngineIdentity {
                engine: lane.into(),
                version: synapse_engine_owned::ENGINE_VERSION.into(),
                build_flags: BTreeMap::from([
                    ("backend".into(), "metal-mpsgraph".into()),
                    ("family".into(), "parakeet-tdt".into()),
                    ("dtype".into(), dtype.as_str().into()),
                    ("kernel_revision".into(), revision.into()),
                    ("risk_class".into(), "abort_safe".into()),
                ]),
            },
            "owned-cuda" => owned_cuda_engine_identity("parakeet-tdt", dtype.as_str(), revision),
            _ => worker_catalog_identity(
                lane,
                revision,
                &[("family", "parakeet-tdt"), ("kernel_revision", revision)],
            ),
        };
        identity.build_flags.extend([
            ("profile".into(), self.id.clone()),
            (
                "compute_dtype".into(),
                self.profile()["compute_dtype"]
                    .as_str()
                    .expect("compute dtype")
                    .into(),
            ),
            ("storage_dtype".into(), dtype.as_str().into()),
        ]);
        Ok(TranscribeCatalogConfig {
            family: "parakeet-tdt",
            dtype,
            max_encoder_frames: profile.max_encoder_frames,
            execution: if lane == "owned-metal" {
                "explicit"
            } else {
                "supervised"
            }
            .into(),
            identity,
        })
    }

    fn transcribe_numeric_profile(&self, config: &TranscribeCatalogConfig) -> NumericProfile {
        let mut numeric = serde_json::from_value(json!({
            "model_digest": self.artifact_digest(), "quant": config.dtype.as_str(),
            "engine": config.identity, "sanitized_tokenizer_digest": "",
            "pooling": {"custom": "none"}, "normalization": "none",
            "dtype": config.dtype.as_str(), "flash_attention": "disabled",
            "certified_shape": {"max_context_tokens": config.max_encoder_frames,
                "max_batch_tokens": config.max_encoder_frames, "max_micro_batch_tokens": config.max_encoder_frames,
                "max_sequences": 1}, "thread_policy": "balanced",
        })).expect("transcribe numeric profile");
        self.apply_numeric_profile(&mut numeric);
        numeric
    }

    fn apply_numeric_profile(&self, numeric: &mut NumericProfile) {
        if self.model()["operation"] == "transcribe" {
            let config = self
                .transcribe_config()
                .expect("validated transcribe profile");
            numeric.model_digest = self.model()["checkpoint_digest"]
                .as_str()
                .expect("checkpoint digest")
                .into();
            numeric.operation = Some("transcribe".into());
            numeric.input_grammar = None;
            numeric.prompt_template = None;
            numeric.kernel_revision = Some(config.identity.build_flags["kernel_revision"].clone());
            numeric.speech_frontend = Some(synapse_speech::SPEECH_FRONTEND_REVISION.into());
            numeric.decode_rule = Some(synapse_speech::DECODE_RULE_REVISION.into());
            numeric.sanitized_tokenizer_digest = self.model()["tokenizer_digest"]
                .as_str()
                .expect("tokenizer digest")
                .into();
            numeric.pooling = PoolingStrategy::Custom("none".into());
            numeric.normalization = NormalizationMode::None;
            let frames = config.max_encoder_frames as u32;
            numeric.certified_shape = CertifiedShapeEnvelope {
                max_context_tokens: frames,
                max_batch_tokens: frames,
                max_micro_batch_tokens: frames,
                max_sequences: 1,
            };
            numeric.rotation = self.profile()["rotation"].as_str().map(str::to_string);
            numeric.converted_package_digest = self.profile()["converted_package_digest"]
                .as_str()
                .map(str::to_string);
            numeric.manifest_profile_digest =
                Some(sha256_hex(&synapse_parity::canonical::canonical_bytes(
                    &self.typed.profile_entry(&self.id).expect("profile entry"),
                )));
            return;
        }
        let input_grammar = format!(
            "synapse-input-grammar-v1:{}",
            self.typed
                .grammar_digest(&self.slug)
                .expect("manifest grammar")
        );
        let rotation = self.profile()["rotation"].as_str();
        let entry = self
            .typed
            .profile_entry(&self.id)
            .expect("manifest profile entry");
        numeric.model_digest = self.model()["checkpoint_digest"]
            .as_str()
            .expect("checkpoint digest")
            .into();
        numeric.operation = Some(
            self.model()["operation"]
                .as_str()
                .expect("operation")
                .into(),
        );
        numeric.input_grammar = Some(input_grammar.clone());
        numeric.prompt_template =
            (self.slug == "gte-reranker-modernbert-base").then_some(input_grammar);
        numeric.rotation = rotation.map(str::to_string);
        numeric.converted_package_digest = self.profile()["converted_package_digest"]
            .as_str()
            .map(str::to_string);
        numeric.manifest_profile_digest = Some(sha256_hex(
            &synapse_parity::canonical::canonical_bytes(&entry),
        ));
        numeric.kernel_revision = Some(match self.profile()["lane"].as_str().expect("lane") {
            // Catalog profile lanes record the graph revision of the projection
            // precision they run, so only lanes that opt in to f16 projections
            // get a new kernel revision (and with it a new fingerprint).
            "owned-metal" => synapse_engine_owned::metal_kernel_revision(catalog_projection_dtype(
                if self.model()["architecture"]["family"] == "qwen3" {
                    OwnedFamily::Qwen3
                } else {
                    OwnedFamily::GteModernBert
                },
                OwnedDType::parse(
                    self.profile()["storage_dtype"]
                        .as_str()
                        .expect("manifest storage dtype"),
                )
                .expect("manifest storage dtype"),
            )),
            "owned-cuda" => synapse_core::CUDA_KERNEL_REVISION.into(),
            "owned-vulkan" => synapse_core::VULKAN_KERNEL_REVISION.into(),
            "ane-direct-worker" => synapse_core::ANE_DIRECT_KERNEL_REVISION.into(),
            _ => unreachable!("manifest lane"),
        });
    }
}

#[derive(Clone, Debug)]
struct TranscribeCatalogConfig {
    family: &'static str,
    dtype: OwnedDType,
    max_encoder_frames: usize,
    execution: String,
    identity: EngineIdentity,
}

#[cfg(test)]
#[path = "catalog_lane_tests.rs"]
mod catalog_lane_tests;

#[derive(Clone, Debug)]
struct OwnedCatalogConfig {
    family: OwnedFamily,
    dtype: OwnedDType,
    /// ModernBERT weight-projection precision. Only catalog lanes opt in to f16
    /// (see `catalog_projection_dtype`); every other lane keeps f32 projections
    /// and an engine identity without a `projection_dtype` flag.
    projections: ProjectionDtype,
    execution: String,
    attention_units: usize,
    config_locator: Option<ModelAssetLocator>,
    extra_locators: Vec<ModelAssetLocator>,
    /// CUDA carries a backend-specific identity while reusing the catalog's
    /// family/dtype storage fields.
    identity_override: Option<EngineIdentity>,
}

/// Projection precision of a catalog Metal lane. Catalog lanes are new lanes, so
/// f16 gte-modernbert catalog lanes opt in to f16 weight projections. Existing
/// profile preloads never come through here and
/// keep the f32 projections their fingerprints were certified on.
fn catalog_projection_dtype(family: OwnedFamily, dtype: OwnedDType) -> ProjectionDtype {
    if family == OwnedFamily::GteModernBert && dtype == OwnedDType::F16 {
        ProjectionDtype::F16
    } else {
        ProjectionDtype::F32
    }
}

/// Projection precision a persisted lane was built with, read back from its
/// engine identity. A lane stored without the flag keeps f32 projections.
fn stored_projection_dtype(identity: &EngineIdentity) -> ProjectionDtype {
    match identity
        .build_flags
        .get(PROJECTION_DTYPE_CONFIG_KEY)
        .map(String::as_str)
    {
        Some("f16") => ProjectionDtype::F16,
        _ => ProjectionDtype::F32,
    }
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
    extra_locators: Vec<ModelAssetLocator>,
    owned: Option<OwnedCatalogConfig>,
    inline: &InlineConfig,
    jobs: &JobConfig,
) -> Result<StoredModelConfig, ModuleError> {
    #[cfg(feature = "test-support")]
    if engine_name == test_deterministic::NAME
        && !matches!(task, ModelTask::Embed | ModelTask::Rerank)
    {
        return Err(ModuleError::Config(
            "test-deterministic supports embedding and rerank only".into(),
        ));
    }
    if engine_name == "owned-metal" && !matches!(task, ModelTask::Embed | ModelTask::Rerank) {
        return Err(ModuleError::Config(
            "owned-metal supports embedding and rerank models only in wave 1".to_string(),
        ));
    }
    if matches!(
        engine_name,
        CUDA_WORKER_ENGINE | "owned-vulkan" | "ane-direct-worker"
    ) && !matches!(task, ModelTask::Embed | ModelTask::Rerank)
    {
        return Err(ModuleError::Config(
            "owned-cuda supports embedding and rerank models only".to_string(),
        ));
    }
    let engine_identity = owned
        .as_ref()
        .and_then(|profile| profile.identity_override.clone())
        .or_else(|| {
            owned.as_ref().map(|profile| {
                engine_identity_with_projections(profile.family, profile.dtype, profile.projections)
            })
        })
        .map_or_else(|| catalog_model_engine_identity(engine_name), Ok)?;
    let catalog = engine_identity
        .build_flags
        .get("profile")
        .map(|id| CatalogProfile::load(id))
        .transpose()?;
    if let Some(catalog) = &catalog {
        catalog.validate(engine_name, task, pooling, normalize, Some(max_tokens))?;
        let expected = catalog.artifact_digest();
        if normalize_digest(&artifact_digest) != normalize_digest(&expected) {
            return Err(ModuleError::Config(
                "package_digest_mismatch: catalog artifact digest disagrees with manifest".into(),
            ));
        }
    }
    let mut numeric_profile = NumericProfile {
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
        dtype: match owned.as_ref().map(|profile| profile.dtype) {
            Some(OwnedDType::F16) => NumericDType::F16,
            Some(OwnedDType::F32) => NumericDType::F32,
            None => match engine_name {
                LLAMA_ENGINE => NumericDType::F16,
                _ => NumericDType::F32,
            },
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
        operation: None,
        input_grammar: None,
        kernel_revision: None,
        rotation: None,
        converted_package_digest: None,
        manifest_profile_digest: None,
        speech_frontend: None,
        decode_rule: None,
    };
    if let Some(catalog) = &catalog {
        catalog.apply_numeric_profile(&mut numeric_profile);
    }
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
        owned_family: owned
            .as_ref()
            .map(|profile| profile.family.as_str().to_string()),
        owned_dtype: owned
            .as_ref()
            .map(|profile| profile.dtype.as_str().to_string()),
        owned_execution: owned.as_ref().map(|profile| profile.execution.clone()),
        owned_attention_units: owned.as_ref().map(|profile| profile.attention_units),
        config_locator: owned
            .as_ref()
            .and_then(|profile| profile.config_locator.clone()),
        extra_locators: if extra_locators.is_empty() {
            owned
                .as_ref()
                .map(|profile| profile.extra_locators.clone())
                .unwrap_or_default()
        } else {
            extra_locators
        },
        engine_identity,
        numeric_profile_id: numeric_profile.numeric_profile_id(),
        fingerprint: numeric_profile.fingerprint(),
        worker_bin,
        worker_runtime_dir,
    })
}

// Call only at new-registration boundaries, never while rebuilding stored
// configs: legacy declarations are part of persisted vector fingerprints.

fn owned_tokenizer_max_tokens(max_tokens: usize, owned: Option<&OwnedCatalogConfig>) -> usize {
    if owned.is_some_and(|profile| profile.family == OwnedFamily::Qwen3) {
        max_tokens.saturating_sub(1).max(1)
    } else {
        max_tokens
    }
}

fn owned_catalog_config(
    model_path: &Path,
    family: Option<&str>,
    dtype: Option<&str>,
    execution: Option<&str>,
    attention_units: Option<usize>,
    config_locator: Option<ModelAssetLocator>,
    extra_locators: Vec<ModelAssetLocator>,
) -> Result<OwnedCatalogConfig, ModuleError> {
    let detected = synapse_engine_owned::detect_family(model_path)
        .map_err(|error| ModuleError::Config(error.to_string()))?;
    if let Some(family) = family {
        let declared =
            OwnedFamily::parse(family).map_err(|error| ModuleError::Config(error.to_string()))?;
        if declared != detected {
            return Err(ModuleError::Config(format!(
                "declared owned-metal family {} does not match detected family {}",
                declared.as_str(),
                detected.as_str()
            )));
        }
    }
    let dtype = dtype
        .map(OwnedDType::parse)
        .transpose()
        .map_err(|error| ModuleError::Config(error.to_string()))?
        .unwrap_or_else(|| detected.recommended_dtype());
    let execution = execution.unwrap_or("explicit").to_ascii_lowercase();
    if !matches!(execution.as_str(), "explicit" | "lazy") {
        return Err(ModuleError::Config(format!(
            "unsupported owned-metal execution mode '{execution}'"
        )));
    }
    let attention_units = attention_units.unwrap_or(OWNED_DEFAULT_ATTENTION_UNITS);
    Ok(OwnedCatalogConfig {
        family: detected,
        dtype,
        projections: ProjectionDtype::F32,
        execution,
        attention_units,
        config_locator,
        extra_locators,
        identity_override: None,
    })
}

/// Declared CUDA build/floor identity for an owned-cuda catalog entry. Grouped
/// because these four values travel together: they are the PTX build identity
/// and support floors that the entry may declare, and each declared value must
/// match the compiled-in constant exactly (declaring is optional; lying is not).
struct OwnedCudaDeclaredIdentity<'a> {
    kernel_revision: Option<&'a str>,
    ptx_virtual_arch: Option<&'a str>,
    minimum_device_cc: Option<f32>,
    minimum_cuda_driver_api: Option<u32>,
}

fn owned_cuda_catalog_config(
    family: Option<&str>,
    dtype: Option<&str>,
    execution: Option<&str>,
    attention_units: Option<usize>,
    declared: OwnedCudaDeclaredIdentity<'_>,
) -> Result<OwnedCatalogConfig, ModuleError> {
    let OwnedCudaDeclaredIdentity {
        kernel_revision,
        ptx_virtual_arch,
        minimum_device_cc,
        minimum_cuda_driver_api,
    } = declared;
    let family = family.ok_or_else(|| {
        ModuleError::Config("owned-cuda catalog entry is missing family".to_string())
    })?;
    let family =
        OwnedFamily::parse(family).map_err(|error| ModuleError::Config(error.to_string()))?;
    let dtype = dtype.ok_or_else(|| {
        ModuleError::Config("owned-cuda catalog entry is missing dtype".to_string())
    })?;
    let dtype = OwnedDType::parse(dtype).map_err(|error| ModuleError::Config(error.to_string()))?;
    let execution = execution.unwrap_or("supervised").to_ascii_lowercase();
    if execution != "supervised" {
        return Err(ModuleError::Config(
            "owned-cuda requires supervised worker execution".to_string(),
        ));
    }
    let kernel_revision = kernel_revision.unwrap_or("cuda-kernel-v1");
    if kernel_revision.trim().is_empty() {
        return Err(ModuleError::Config(
            "owned-cuda kernel_revision must not be empty".to_string(),
        ));
    }
    if let Some(ptx_virtual_arch) = ptx_virtual_arch {
        if ptx_virtual_arch != OWNED_CUDA_PTX_VIRTUAL_ARCH {
            return Err(ModuleError::Config(format!(
                "owned-cuda requires PTX virtual architecture {}, got {ptx_virtual_arch}",
                OWNED_CUDA_PTX_VIRTUAL_ARCH
            )));
        }
    }
    if let Some(minimum_device_cc) = minimum_device_cc {
        if (minimum_device_cc - OWNED_CUDA_MINIMUM_DEVICE_CC).abs() > f32::EPSILON {
            return Err(ModuleError::Config(format!(
                "owned-cuda requires minimum device compute capability {}, got {minimum_device_cc}",
                OWNED_CUDA_MINIMUM_DEVICE_CC
            )));
        }
    }
    if let Some(minimum_cuda_driver_api) = minimum_cuda_driver_api {
        if minimum_cuda_driver_api != OWNED_CUDA_MINIMUM_DRIVER_API {
            return Err(ModuleError::Config(format!(
                "owned-cuda requires minimum CUDA driver API {}, got {minimum_cuda_driver_api}",
                OWNED_CUDA_MINIMUM_DRIVER_API
            )));
        }
    }
    Ok(OwnedCatalogConfig {
        family,
        dtype,
        projections: ProjectionDtype::F32,
        execution,
        attention_units: attention_units.unwrap_or(OWNED_DEFAULT_ATTENTION_UNITS),
        config_locator: None,
        extra_locators: Vec::new(),
        identity_override: Some(owned_cuda_engine_identity(
            family.as_str(),
            dtype.as_str(),
            kernel_revision,
        )),
    })
}

fn canonical_engine_name(engine: &str) -> String {
    match engine.trim().to_ascii_lowercase().as_str() {
        "llama.cpp" | LLAMA_WORKER_ENGINE => "llama".to_string(),
        // Catalog entries select this engine explicitly. Future hardware probes can
        // populate the same catalog value without changing request dispatch.
        "owned" | "metal" | "owned_metal" => "owned-metal".to_string(),
        CUDA_WORKER_ENGINE | "owned_cuda" | "cuda" => CUDA_WORKER_ENGINE.to_string(),
        "owned-decode" | "owned_metal_decode" | "owned-metal-decode" => {
            "owned-metal-decode".to_string()
        }
        other => other.to_string(),
    }
}

fn default_artifact_format(engine_name: &str) -> String {
    match engine_name {
        #[cfg(feature = "test-support")]
        test_deterministic::NAME => test_deterministic::NAME.into(),
        LLAMA_ENGINE => "gguf".to_string(),
        // Every owned engine loads a converted safetensors profile package.
        "owned-metal" | "owned-cuda" | "owned-vulkan" | synapse_core::ANE_DIRECT_WORKER_ENGINE => {
            "safetensors-package".to_string()
        }
        _ => "onnx".to_string(),
    }
}

fn default_quant(engine_name: &str) -> String {
    match engine_name {
        LLAMA_ENGINE => "f16".to_string(),
        "owned-metal" | "owned-cuda" => "f16".to_string(),
        _ => "fp32".to_string(),
    }
}

fn catalog_model_engine_identity(engine_name: &str) -> Result<EngineIdentity, ModuleError> {
    match engine_name {
        #[cfg(feature = "test-support")]
        test_deterministic::NAME => Ok(EmbedEngine::identity(
            &test_deterministic::TestDeterministic,
        )),
        LLAMA_ENGINE => Ok(worker_catalog_identity(
            LLAMA_WORKER_ENGINE,
            "protocol-v1",
            &[("transport", worker_catalog_transport())],
        )),
        // These two lanes exist only as catalog lanes, whose fingerprints are pinned
        // in the release manifest and must be identical on every platform that
        // runs them (Vulkan runs on Linux and Windows). The IPC transport differs
        // by OS and does not affect the vectors, so it stays out of the identity.
        // The legacy llama arm above keeps it, because live fingerprints depend on it.
        //
        // The version strings in this function ("protocol-v1", "protocol-v2") are
        // frozen fingerprint inputs, not the worker wire protocol: the workers now
        // speak protocol 3, which HELLO reports separately. They also surface as
        // `provenance.engine.version`. Changing one rotates every pinned fingerprint
        // on that lane and invalidates stored vectors, so leave them as they are.
        "owned-vulkan" | "ane-direct-worker" => {
            Ok(worker_catalog_identity(engine_name, "protocol-v2", &[]))
        }
        "owned-cuda" => Ok(owned_cuda_engine_identity(
            "unknown",
            "f16",
            "cuda-kernel-v1",
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

/// Run storage maintenance from the host's periodic health cadence. Keeping
/// this work on an existing heartbeat avoids a second timer loop in the module.
fn run_background_maintenance(state: &ModuleState) {
    if let Err(error) = state.store.purge_expired_jobs(now_ms()) {
        tracing::warn!(target: "maintenance", error = %error, "job purge sweep failed");
    }
}

#[async_trait]
impl ModuleHandler for SynapseHandler {
    async fn on_hello_ack(&self, ack: &ModuleHelloAckBody) {
        let mut completion = self
            .inner
            .initialization
            .get_or_init(|| {
                let handler = self.clone();
                let ack = ack.clone();
                let (tx, rx) = tokio::sync::watch::channel(None);
                // The task owns initialization independently of any HELLO_ACK waiter;
                // cancelling a callback must not start a second blocking boot.
                tokio::spawn(async move {
                    let result = tokio::task::spawn_blocking(move || {
                        let state = handler
                            .initialize(&ack)
                            .map_err(|error| error.to_string())?;
                        let _ = handler.inner.state.set(Arc::clone(&state));
                        diagnostics_findings::start(&state);
                        start_catalog_revalidation(Arc::clone(&state));
                        media_jobs::start_maintenance(&state);
                        let _ = start_perf_sampler(state);
                        Ok(())
                    })
                    .await
                    .unwrap_or_else(|error| Err(error.to_string()));
                    let _ = tx.send(Some(result));
                });
                rx
            })
            .clone();
        let result = completion
            .wait_for(Option::is_some)
            .await
            .map(|result| result.as_ref().expect("completed boot").clone())
            .unwrap_or_else(|error| Err(error.to_string()));
        if let Err(error) = result {
            panic!("synapse boot failed after HELLO_ACK: {error}");
        }
    }

    async fn on_bind(&self, req: &RouteBindRequest) -> BindDecision {
        if self.state().is_none() {
            return BindDecision::reject(
                "module_not_initialized",
                "synapse has not completed HELLO_ACK initialization",
            );
        }
        if let Some(principal) = req.principal.as_ref() {
            let owner = match principal {
                Principal::Direct => "principal:direct".to_string(),
                Principal::Reserved { module_id } => format!("principal:reserved:{module_id}"),
                Principal::Unverified => "principal:unverified".to_string(),
            };
            self.inner
                .media_principals
                .lock()
                .expect("media principals")
                .insert(req.handle, owner);
        }
        self.inner.record_bind_scope(req);
        BindDecision::accept()
    }

    async fn on_route_gone(&self, handle: &RouteHandle) {
        self.inner
            .media_principals
            .lock()
            .expect("media principals")
            .remove(handle);
        if let Ok(mut flows) = self.inner.flow_routes.lock() {
            flows.remove(handle);
        }
        if let Ok(mut routes) = self.inner.diagnostics_routes.lock() {
            routes.remove(handle);
        }
    }

    /// Writes the `synapse stopped` line. On the registration-channel launch path
    /// the SDK exits the process right after this hook returns, so code after
    /// `serve` never runs there. The SDK calls this at most once per connection,
    /// and this module serves exactly one connection per process.
    async fn on_connection_end(&self, end: ConnectionEnd) {
        if self.inner.connection_end.set(end).is_ok() {
            tracing::info!(
                target: "lifecycle",
                module = %self.inner.module_id,
                uptime_s = self.inner.started.elapsed().as_secs(),
                reason = connection_end_reason(Some(end)),
                "synapse stopped: connection ended"
            );
        }
    }

    async fn health(&self) -> HealthReport {
        let Some(state) = self.state() else {
            return HealthReport::ok();
        };
        run_background_maintenance(&state);
        let health = module_health(&state);
        HealthReport {
            status: subc_client_rs::HealthStatus::Ok,
            detail: Some("ok".to_string()),
            metrics: Some(serde_json::to_value(&health).expect("module health should serialize")),
        }
    }

    async fn handle(&self, ctx: RequestCtx, body: Vec<u8>) -> HandlerOutcome {
        let Some(state) = self.state() else {
            return channel_error(
                "module_not_initialized",
                "synapse has not completed HELLO_ACK initialization",
            );
        };

        let handle_started = Instant::now();
        let envelope: MethodEnvelope = match serde_json::from_slice(&body) {
            Ok(envelope) => envelope,
            Err(error) => {
                return channel_error(
                    "invalid_request",
                    format!("route request body is not decodable: {error}"),
                )
            }
        };
        // Only embedding batches are profiled here: the stage log describes the
        // embedding lane, and other methods would interleave unrelated rows.
        // The model is not resolved yet, so these two records carry the method
        // name in their `model_id` field.
        let profiled = envelope.method == "embed.batch";
        if profiled {
            ane_lane_timing("request_decode", "embed.batch", None, handle_started);
        }

        if let Some(refusal) = self
            .inner
            .flow_refusal(&ctx.route_handle(), &envelope.method)
        {
            return refusal;
        }
        if matches!(
            envelope.method.as_str(),
            "diagnostics.findings" | "events_get"
        ) && !self
            .inner
            .diagnostics_routes
            .lock()
            .ok()
            .and_then(|routes| routes.get(&ctx.route_handle()).copied())
            .unwrap_or(false)
        {
            return channel_error(
                "permission_refused",
                format!(
                    "{} requires a head, authorized flow, or unscoped management route",
                    envelope.method
                ),
            );
        }
        if envelope.method.starts_with("media.")
            || envelope.method == "role.describe"
            || envelope.method == "certify.transcribe_observation"
            || envelope.method == "certify.transcribe_measurements"
            || envelope.method == "certify.transcribe_observe_rerun"
        {
            let principal = self
                .inner
                .media_principals
                .lock()
                .expect("media principals")
                .get(&ctx.route_handle())
                .cloned();
            return media_jobs::handle(
                state,
                principal.as_deref(),
                &envelope.method,
                envelope.params,
            )
            .await;
        }
        let request_span = tracing::debug_span!("synapse_request", request_id = ctx.corr(),
            method = %envelope.method);
        let outcome = dispatch_request(state, envelope)
            .instrument(request_span)
            .await;
        if profiled {
            ane_lane_timing("handle_total", "embed.batch", None, handle_started);
        }
        outcome
    }
}

fn attach_certify_observation(response: &mut Value, observation: Option<Value>) {
    if let Some(observation) = observation {
        response["observation"] = observation;
    }
}

fn certify_observations(state: Arc<ModuleState>) -> HandlerOutcome {
    if !state.runtime.certify_observation {
        return channel_error(
            "certify_observation_disabled",
            "certification observation mode is disabled",
        );
    }
    result_outcome(certify_worker_snapshot(&state.runtime))
}

// Direct ANE placement evidence comes from the shared residency supervisor;
// transport counters alone cannot certify where its admitted layers execute.
fn certify_worker_snapshot(runtime: &RuntimeState) -> Value {
    let models = runtime.loaded_models();
    let mut available = !models.is_empty();
    let mut requests = BTreeMap::new();
    for model in models {
        match &model.backend {
            EmbedBackend::Owned(_) => {}
            #[cfg(unix)]
            EmbedBackend::DirectAne(engine) => {
                requests.insert(
                    worker_host::ane_residency::AneShapeWorker::worker_id(
                        engine.serving.channel.as_ref(),
                    )
                    .to_owned(),
                    engine.serving.channel.request_count(),
                );
            }
            EmbedBackend::Worker(engine) => {
                let count = engine.lock().ok().and_then(|engine| {
                    let identity = EmbedEngine::identity(&*engine);
                    if identity.engine == "ane-direct-worker" {
                        return None;
                    }
                    engine.request_count().ok()
                });
                if let Some((worker_id, count)) = count {
                    requests.insert(worker_id, count);
                } else {
                    available = false;
                }
            }
            _ => available = false,
        }
    }
    let observations = runtime
        .ane_supervisor
        .lock()
        .ok()
        .and_then(|s| s.as_ref().and_then(|s| s.observations()));
    let (inventories, admitted_count) = observations.unwrap_or_default();
    json!({"inventories": inventories, "worker_requests": requests, "admitted_count": admitted_count, "available": available})
}

async fn dispatch_request(state: Arc<ModuleState>, request: MethodEnvelope) -> HandlerOutcome {
    match request.method.as_str() {
        "diagnostics.findings" => diagnostics_findings::handle(&state, request.params),
        "lanes.pause" => lane_pause::pause(state, request.params).await,
        "lanes.resume" => lane_pause::resume(state).await,
        "lanes.status" => result_outcome(lane_pause::status(&state.runtime)),
        "models.catalog" => models_catalog(state, request.params).await,
        "models.download" => models_download(state, request.params).await,
        "models.download.cancel" => models_download_cancel(state, request.params).await,
        "models.remove" => models_remove(state, request.params).await,
        "models.list" => result_outcome(models_list_payload(&state)),
        "events_get" => event_outbox::events_get(&state, request.params),
        "certify.observations" => certify_observations(state),
        "embed.query" => embed_query(state, request.params).await,
        "embed.batch" => embed_batch(state, request.params).await,
        "embed.result" => embed_result(state, request.params).await,
        "rerank.score" => rerank_score(state, request.params).await,
        "microllm.oneshot" => microllm_oneshot(state, request.params).await,
        "model.unload" => model_unload(state, request.params).await,
        "cache.pin" => cache_pin(state, request.params).await,
        "cache.gc" => cache_gc(state, request.params).await,
        "admission.status" => admission_status(state).await,
        "model.status" => model_status(state, request.params).await,
        other => channel_error(
            "unknown_method",
            format!("unknown method '{other}' for synapse management surface"),
        ),
    }
}

fn model_runtime_state_name(state: &ModelRuntimeState) -> &'static str {
    match state {
        ModelRuntimeState::Unloaded => "unloaded",
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
            if diagnostics_findings::cached_quarantine(&slot.state)
                != diagnostics_findings::cached_quarantine(&state)
            {
                runtime.findings_changed.notify_one();
            }
            slot.state = state;
            if clear_loaded {
                slot.loaded = None;
            }
            slot.notify.notify_waiters();
        }
    }
}

fn set_model_slot_ready(
    runtime: &RuntimeState,
    model_id: &str,
    model: Arc<EmbeddingModel>,
    cold_load_ms: f64,
) {
    if let Ok(mut catalog) = runtime.catalog.lock() {
        if let Some(slot) = catalog.get_mut(model_id) {
            if diagnostics_findings::cached_quarantine(&slot.state) {
                runtime.findings_changed.notify_one();
            }
            slot.loaded = Some(model);
            slot.state = ModelRuntimeState::Ready;
            slot.last_cold_load_ms = Some(cold_load_ms);
            slot.notify.notify_waiters();
        }
    }
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
                if diagnostics_findings::cached_quarantine(&slot.state) {
                    runtime.findings_changed.notify_one();
                }
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
    if let ModelRuntimeState::Failed(error) = &slot.state {
        payload["error"] = serde_json::to_value(error).expect("model error serializes");
    }
    payload
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
            Ok(Some(record)) if record.kind == store::DOWNLOAD_JOB_KIND => {
                result_outcome(download_status_payload(&state, &record, true))
            }
            Ok(Some(_)) => channel_error(
                "invalid_request",
                "job_id does not refer to a models.download job",
            ),
            Ok(None) => channel_error("invalid_request", "unknown or expired job_id"),
            Err(error) => channel_error("store_failure", error.to_string()),
        },
        (None, Some(model_id)) => match model_slot_snapshot(&state.runtime, &model_id) {
            Some(slot) => result_outcome(model_status_payload(state.module_generation, &slot)),
            None => result_outcome(error_payload(&state, catalog_unknown(&model_id))),
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
    if let Some(entry) = state.runtime.release_catalog.entry(&params.model_id) {
        return result_outcome(error_payload(
            &state,
            catalog_wire_error(
                "invalid_request",
                json!({"catalog_id":entry.id,"lane_ids":entry.backends.iter().map(|b| catalog::lane_id(&entry.id,&b.backend)).collect::<Vec<_>>()}),
                "model.unload requires a lane id",
            ),
        ));
    }
    if model_slot_snapshot(&state.runtime, &params.model_id).is_some_and(|slot| {
        matches!(
            slot.state,
            ModelRuntimeState::Loading | ModelRuntimeState::Validating
        )
    }) {
        return result_outcome(error_payload(
            &state,
            WireOperationError::from_stable(
                StableError::model_loading(Some(250)),
                "model is still loading",
            ),
        ));
    }
    let lane_lock = catalog_lane_lock(&state.runtime, &params.model_id);
    let catalog_lane = resolved_catalog_lane(&state.runtime, &params.model_id).is_some();
    // Unload never waits: an in-flight inference (including one whose caller
    // has gone but whose reply is still draining) holds the serving gate.
    let catalog_guards = if catalog_lane {
        match lane_lock.try_lock_owned().ok().and_then(|guard| {
            lane_serving_gate(&state.runtime, &params.model_id)
                .try_write_owned()
                .ok()
                .map(|gate| (guard, gate))
        }) {
            Some(guards) => Some(guards),
            None => {
                return result_outcome(error_payload(
                    &state,
                    catalog_wire_error(
                        "model_in_use",
                        json!({"catalog_id":state.runtime.release_catalog.resolve_reserved(&params.model_id).map(|(e,_)| &e.id),"holders":state.runtime.self_check_holders.lock().expect("self-check holders").get(&params.model_id).map(|id| vec![json!({"kind":"self_check","check_id":id})]).unwrap_or_else(|| vec![json!({"kind":"request","lease_id":params.model_id})])}),
                        "catalog engine invocation is still in flight",
                    ),
                ))
            }
        }
    } else {
        None
    };
    if state
        .runtime
        .release_catalog
        .resolve_reserved(&params.model_id)
        .is_some_and(|(entry, _)| entry.task == "transcribe")
    {
        return match media_jobs::unload(&state.runtime, Some(&params.model_id), false).await {
            Ok(()) => result_outcome(json!({"model_id":params.model_id,"state":"unloaded"})),
            Err(error) => result_outcome(error_payload(&state, error)),
        };
    }
    let Some(snapshot) = model_slot_snapshot(&state.runtime, &params.model_id) else {
        return result_outcome(error_payload(&state, catalog_unknown(&params.model_id)));
    };
    if matches!(
        snapshot.state,
        ModelRuntimeState::Validating | ModelRuntimeState::Loading
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
        if resolved_catalog_lane(&state.runtime, &params.model_id).is_some()
            && Arc::strong_count(&loaded) > 2
        {
            return result_outcome(error_payload(
                &state,
                catalog_wire_error(
                    "model_in_use",
                    json!({"catalog_id":state.runtime.release_catalog.resolve_reserved(&params.model_id).map(|(e,_)| &e.id),"holders":[{"kind":"request","lease_id":params.model_id}]}),
                    "catalog lane holds a live request",
                ),
            ));
        }
        if catalog_lane {
            set_model_slot_state(
                &state.runtime,
                &params.model_id,
                ModelRuntimeState::Unloaded,
            );
        }
        let invocation_guards = catalog_guards;
        let unload = tokio::task::spawn_blocking(move || {
            let _invocation_guards = invocation_guards;
            unload_embedding_model_blocking(loaded)
        })
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
    if !catalog_lane {
        set_model_slot_state(
            &state.runtime,
            &params.model_id,
            ModelRuntimeState::Unloaded,
        );
    }
    let slot = model_slot_snapshot(&state.runtime, &params.model_id)
        .expect("unloaded model remains registered");
    result_outcome(model_status_payload(state.module_generation, &slot))
}

async fn resolve_model_for_request(
    state: Arc<ModuleState>,
    requested: Option<&str>,
    task: ModelTask,
) -> Result<Arc<EmbeddingModel>, WireOperationError> {
    let resolution = resolve_model_for_request_inner(Arc::clone(&state), requested, task).await;
    if let Err(error) = &resolution {
        record_admission_refusal(
            &state.runtime,
            requested.unwrap_or("default"),
            None,
            &error.code,
        );
    }
    resolution
}

async fn resolve_model_for_request_inner(
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
                            "task '{}' has no {} knob assignment on machine profile {}; specify a model id",
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
            StableError::new(
                synapse_core::StableErrorCode::UnknownModel,
                ErrorClass::Permanent,
                None,
                false,
            )
            .with_details(
                serde_json::from_value(json!({"model_id":model_id})).expect("details object"),
            ),
            format!("model '{model_id}' is not registered"),
        ));
    };
    match (&snapshot.state, snapshot.loaded) {
        (ModelRuntimeState::Ready, Some(model)) => Ok(model),
        (ModelRuntimeState::Failed(error), _) if error.class == ErrorClass::Permanent => {
            Err(error.clone())
        }
        (ModelRuntimeState::Validating | ModelRuntimeState::Loading, _) => {
            Err(WireOperationError::from_stable(
                StableError::model_loading(Some(250)),
                format!("model '{model_id}' is loading"),
            ))
        }
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
                ModelRuntimeState::Validating
                    | ModelRuntimeState::Loading
                    | ModelRuntimeState::Ready
            )
        {
            false
        } else {
            if diagnostics_findings::cached_quarantine(&slot.state) {
                state.runtime.findings_changed.notify_one();
            }
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

async fn ensure_model_loaded_until(
    state: Arc<ModuleState>,
    model_id: &str,
    deadline: tokio::time::Instant,
    timeout_ms: u64,
) -> Result<Arc<EmbeddingModel>, WireOperationError> {
    let Some(snapshot) = model_slot_snapshot(&state.runtime, model_id) else {
        return Err(WireOperationError::from_stable(
            StableError::artifact_invalid(),
            format!("unknown model_id '{model_id}'"),
        ));
    };
    match (&snapshot.state, snapshot.loaded.clone()) {
        (ModelRuntimeState::Ready, Some(model)) => Ok(model),
        (ModelRuntimeState::Failed(error), _) => Err(error.clone()),
        _ => {
            begin_background_catalog_load(Arc::clone(&state), model_id.to_string());
            wait_for_model_loaded(&state.runtime, model_id, deadline, timeout_ms).await
        }
    }
}

async fn wait_for_model_loaded(
    runtime: &RuntimeState,
    model_id: &str,
    deadline: tokio::time::Instant,
    timeout_ms: u64,
) -> Result<Arc<EmbeddingModel>, WireOperationError> {
    loop {
        drop(lane_pause::admit(runtime, model_id)?);
        let Some(snapshot) = model_slot_snapshot(runtime, model_id) else {
            return Err(WireOperationError::from_stable(
                StableError::artifact_invalid(),
                format!("unknown model_id '{model_id}'"),
            ));
        };
        match (&snapshot.state, snapshot.loaded.clone()) {
            (ModelRuntimeState::Ready, Some(model)) => return Ok(model),
            (ModelRuntimeState::Failed(error), _) => return Err(error.clone()),
            _ => {}
        }
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero()
            || tokio::time::timeout(remaining, snapshot.notify.notified())
                .await
                .is_err()
        {
            return Err(model_load_timeout_error(model_id, timeout_ms));
        }
    }
}

fn model_load_timeout_error(model_id: &str, timeout_ms: u64) -> WireOperationError {
    WireOperationError::from_stable(
        StableError::model_loading(Some(timeout_ms)),
        format!("timed out waiting for model '{model_id}' to load after {timeout_ms}ms"),
    )
}

async fn load_catalog_model_task(
    state: Arc<ModuleState>,
    model_id: String,
) -> Result<Arc<EmbeddingModel>, WireOperationError> {
    #[cfg(test)]
    state
        .runtime
        .model_load_tasks
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
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
    let _pause_guard = match lane_pause::admit(&state.runtime, &model_id) {
        Ok(guard) => guard,
        Err(error) => {
            set_model_slot_state(&state.runtime, &model_id, ModelRuntimeState::Unloaded);
            return Err(error);
        }
    };
    set_model_slot_state(&state.runtime, &model_id, ModelRuntimeState::Loading);
    let Some(snapshot) = model_slot_snapshot(&state.runtime, &model_id) else {
        return Err(WireOperationError::from_stable(
            StableError::artifact_invalid(),
            format!("unknown model_id '{model_id}'"),
        ));
    };
    let spec = snapshot.spec.clone();
    let model_cache = Arc::clone(&state.model_cache);
    let load_started = std::time::Instant::now();
    let microllm_max_tokens = state.runtime.microllm_max_tokens;
    let worker_load_timeout = state.runtime.worker_load_timeout;
    let worker_forward_lines_per_sec = state.runtime.log.worker_forward_lines_per_sec;
    let worker_load_context = WorkerLoadContext {
        findings_changed: Some(state.runtime.findings_changed.clone()),
        ..WorkerLoadContext::new(worker_load_timeout, worker_forward_lines_per_sec)
    };
    let ane_supervisor = direct_ane_supervisor(&state.runtime, &spec.engine);
    let is_catalog = state.runtime.release_catalog.is_reserved_id(&model_id);
    let fault_lane = model_id.clone();
    let loaded = tokio::task::spawn_blocking(move || {
        if is_catalog {
            catalog_call_fault(&fault_lane, "load")?;
        }
        load_catalog_model_blocking(
            CatalogAwareModelSpec {
                stored: spec,
                catalog_lane: is_catalog.then_some(fault_lane),
                worker_load_context,
            },
            model_cache,
            microllm_max_tokens,
            ane_supervisor?,
        )
    })
    .await
    .unwrap_or_else(|error| {
        Err(WireOperationError::from_stable(
            StableError::engine_crashed(Some(100)),
            format!("model load join failed: {error}"),
        ))
    });
    match loaded {
        Ok(model) => {
            let cold_load_ms = load_started.elapsed().as_secs_f64() * 1_000.0;
            let model = Arc::new(model);
            set_model_slot_ready(&state.runtime, &model_id, Arc::clone(&model), cold_load_ms);
            start_work_for_already_checked_lane(&state, &model);
            Ok(model)
        }
        Err(error) => {
            let error = if is_catalog {
                WireOperationError::from_stable(
                    StableError::engine_crashed(Some(250)),
                    error.message,
                )
            } else {
                error
            };
            set_model_slot_state(
                &state.runtime,
                &model_id,
                ModelRuntimeState::Failed(error.clone()),
            );
            Err(error)
        }
    }
}

fn stored_owned_profile(
    spec: &StoredModelConfig,
) -> Result<Option<OwnedCatalogConfig>, WireOperationError> {
    if spec.engine == "owned-metal" {
        let family = spec
            .owned_family
            .as_deref()
            .ok_or_else(|| artifact_invalid_error("owned-metal catalog entry is missing family"))?;
        let dtype = spec
            .owned_dtype
            .as_deref()
            .ok_or_else(|| artifact_invalid_error("owned-metal catalog entry is missing dtype"))?;
        return Ok(Some(OwnedCatalogConfig {
            family: OwnedFamily::parse(family)
                .map_err(|error| artifact_invalid_error(error.to_string()))?,
            dtype: OwnedDType::parse(dtype)
                .map_err(|error| artifact_invalid_error(error.to_string()))?,
            projections: stored_projection_dtype(&spec.engine_identity),
            execution: spec
                .owned_execution
                .clone()
                .unwrap_or_else(|| "explicit".to_string()),
            attention_units: spec
                .owned_attention_units
                .unwrap_or(OWNED_DEFAULT_ATTENTION_UNITS),
            config_locator: spec.config_locator.clone(),
            extra_locators: spec.extra_locators.clone(),
            identity_override: None,
        }));
    }
    if spec.engine == CUDA_WORKER_ENGINE {
        // A persisted owned-cuda row carries the same owned_*/config/extra
        // fields as metal; rebuild through the CUDA builder so the engine
        // identity and floors are re-derived from the stored build flags
        // instead of a second hand-rolled profile.
        let mut profile = owned_cuda_catalog_config(
            spec.owned_family.as_deref(),
            spec.owned_dtype.as_deref(),
            spec.owned_execution.as_deref(),
            spec.owned_attention_units,
            OwnedCudaDeclaredIdentity {
                kernel_revision: spec
                    .engine_identity
                    .build_flags
                    .get("kernel_revision")
                    .map(String::as_str),
                ptx_virtual_arch: spec
                    .engine_identity
                    .build_flags
                    .get("ptx_virtual_arch")
                    .map(String::as_str),
                minimum_device_cc: spec
                    .engine_identity
                    .build_flags
                    .get("minimum_device_cc")
                    .and_then(|value| value.parse().ok()),
                minimum_cuda_driver_api: spec
                    .engine_identity
                    .build_flags
                    .get("minimum_cuda_driver_api")
                    .and_then(|value| value.parse().ok()),
            },
        )
        .map_err(|error| artifact_invalid_error(error.to_string()))?;
        profile.config_locator = spec.config_locator.clone();
        profile.extra_locators = spec.extra_locators.clone();
        return Ok(Some(profile));
    }
    Ok(None)
}

fn assemble_owned_model_package(
    spec: &StoredModelConfig,
    model_path: &Path,
    model_cache: &ModelCache,
    profile: &OwnedCatalogConfig,
) -> Result<PathBuf, WireOperationError> {
    if model_path.is_dir()
        || model_path
            .parent()
            .is_some_and(|parent| parent.join("config.json").is_file())
    {
        return Ok(model_path.to_path_buf());
    }
    if !profile.extra_locators.is_empty() {
        return Err(artifact_invalid_error(format!(
            "sharded {} packages are reserved but not supported in wave 1",
            spec.engine
        )));
    }
    let config_locator = profile.config_locator.as_ref().ok_or_else(|| {
        artifact_invalid_error(format!(
            "{} model package is missing files.config",
            spec.engine
        ))
    })?;
    let config = locator_path(config_locator, model_cache)?;
    let package_key = spec.artifact_digest.trim_start_matches("sha256:");
    // Both owned backends resolve the same `config.json` + `model.safetensors`
    // layout from this root, keyed by digest, so metal's existing populated
    // packages are reused rather than re-copied for a cuda row.
    let packages_root = model_cache.root().join("owned-metal-models");
    let package_root = packages_root.join(package_key);
    if package_root.join("config.json").is_file()
        && package_root.join("model.safetensors").is_file()
    {
        record_owned_package_roles(&package_root, spec)?;
        return Ok(package_root);
    }
    fs::create_dir_all(&packages_root)
        .map_err(|error| io_to_load_error("create owned package root", &packages_root, &error))?;
    let temporary = packages_root.join(format!(".{package_key}.{}.tmp", std::process::id()));
    if temporary.exists() {
        fs::remove_dir_all(&temporary).map_err(|error| {
            io_to_load_error("remove stale owned package temp", &temporary, &error)
        })?;
    }
    fs::create_dir_all(&temporary)
        .map_err(|error| io_to_load_error("create owned package temp", &temporary, &error))?;
    fs::copy(model_path, temporary.join("model.safetensors"))
        .map_err(|error| io_to_load_error("copy owned model", model_path, &error))?;
    fs::copy(&config.path, temporary.join("config.json"))
        .map_err(|error| io_to_load_error("copy owned config", &config.path, &error))?;
    record_owned_package_roles(&temporary, spec)?;
    match fs::rename(&temporary, &package_root) {
        Ok(()) => {}
        Err(_) if package_root.is_dir() => {
            let _ = fs::remove_dir_all(&temporary);
        }
        Err(error) => {
            return Err(io_to_load_error(
                "publish owned model package",
                &package_root,
                &error,
            ))
        }
    }
    Ok(package_root)
}

fn record_owned_package_roles(
    package: &Path,
    spec: &StoredModelConfig,
) -> Result<(), WireOperationError> {
    let mut digests = Vec::new();
    for locator in std::iter::once(&spec.model_locator)
        .chain(std::iter::once(&spec.tokenizer_locator))
        .chain(spec.config_locator.iter())
        .chain(spec.extra_locators.iter())
    {
        let ModelAssetLocator::CacheDigest { digest } = locator else {
            return Ok(());
        };
        digests.push(digest.trim_start_matches("sha256:").to_string());
    }
    digests.sort();
    digests.dedup();
    let path = package.join("role-digests.json");
    let temporary = package.join("role-digests.json.tmp");
    fs::write(
        &temporary,
        serde_json::to_vec(&digests).expect("role digests"),
    )
    .map_err(|error| io_to_load_error("write owned package roots", &temporary, &error))?;
    fs::rename(&temporary, &path)
        .map_err(|error| io_to_load_error("publish owned package roots", &path, &error))
}

#[derive(Clone)]
struct WorkerLoadContext {
    findings_changed: Option<Arc<Notify>>,
    timeout: Duration,
    forward_lines_per_sec: u32,
}

impl WorkerLoadContext {
    fn new(timeout: Duration, forward_lines_per_sec: u32) -> Self {
        Self {
            timeout,
            findings_changed: None,
            forward_lines_per_sec,
        }
    }
}

// Catalog ownership comes from the release registry and stays outside the
// stored numerical profile, so quarantine policy cannot change fingerprints.
struct CatalogAwareModelSpec {
    stored: StoredModelConfig,
    catalog_lane: Option<String>,
    worker_load_context: WorkerLoadContext,
}

fn load_catalog_model_blocking(
    spec: CatalogAwareModelSpec,
    model_cache: Arc<ModelCache>,
    microllm_max_tokens: u32,
    ane_supervisor: Option<worker_host::ane_residency::AneResidencySupervisor>,
) -> Result<EmbeddingModel, WireOperationError> {
    let CatalogAwareModelSpec {
        stored: spec,
        catalog_lane,
        worker_load_context,
    } = spec;
    let task = parse_model_task(Some(&spec.task), &spec.engine, &spec.model_id)
        .map_err(|error| artifact_invalid_error(error.to_string()))?;
    if spec.engine == CUDA_WORKER_ENGINE {
        if cfg!(target_os = "macos") {
            return Err(artifact_invalid_error(format!(
                "owned-cuda model '{}' is not supported on macOS",
                spec.model_id
            )));
        }
        ensure_owned_cuda_floor(spec.worker_bin.as_deref())?;
    }
    let model_path = locator_path(&spec.model_locator, &model_cache)?;
    let tokenizer_path = locator_path(&spec.tokenizer_locator, &model_cache)?;
    let owned_profile = stored_owned_profile(&spec)?;
    // Keep every package member leased until the worker finishes loading it.
    let _extra_assets = spec
        .extra_locators
        .iter()
        .map(|locator| locator_path(locator, &model_cache))
        .collect::<Result<Vec<_>, _>>()?;
    let effective_model_path = if release_profile(&spec.engine_identity).is_some() {
        model_path.path.clone()
    } else if let Some(profile) = owned_profile.as_ref() {
        assemble_owned_model_package(&spec, &model_path.path, &model_cache, profile)?
    } else {
        model_path.path.clone()
    };
    let tokenizer = SanitizedTokenizer::from_file(
        &tokenizer_path.path,
        TokenizerConfig {
            max_tokens: if release_profile(&spec.engine_identity).is_some() {
                usize::MAX
            } else {
                owned_tokenizer_max_tokens(spec.max_tokens, owned_profile.as_ref())
            },
        },
    )
    .map_err(|error| artifact_invalid_error(error.to_string()))?;
    if let Some(id) = release_profile(&spec.engine_identity) {
        CatalogProfile::load(id)
            .and_then(|catalog| catalog.validate_readout(&tokenizer))
            .map_err(|error| artifact_invalid_error(error.to_string()))?;
    }
    let actual_tokenizer_digest = format!("sha256:{}", tokenizer.sanitized_sha256());
    if actual_tokenizer_digest != normalize_digest(&spec.tokenizer_sanitized_digest) {
        return Err(artifact_invalid_error(format!(
            "tokenizer digest mismatch for '{}': expected {}, got {}",
            spec.model_id, spec.tokenizer_sanitized_digest, actual_tokenizer_digest
        )));
    }
    let runtime_config = model_runtime_config(
        &spec,
        &effective_model_path,
        model_cache.root(),
        microllm_max_tokens,
    );
    let artifact = ValidatedArtifact {
        digest: spec.artifact_digest.clone(),
        format: spec.artifact_format.clone(),
    };
    let (backend, loaded_model, owned_tokenizer_policy) = match spec.engine.as_str() {
        #[cfg(feature = "test-support")]
        test_deterministic::NAME => {
            let mut engine = test_deterministic::TestDeterministic;
            let loaded = EmbedEngine::load(&mut engine, &artifact, &runtime_config)
                .map_err(engine_error_to_wire)?;
            (
                EmbedBackend::TestDeterministic(Arc::new(engine)),
                loaded,
                None,
            )
        }
        "owned-cuda" => {
            if cfg!(target_os = "macos") {
                return Err(artifact_invalid_error(format!(
                    "owned-cuda model '{}' is not supported on macOS",
                    spec.model_id
                )));
            }
            let (backend, loaded) = load_worker_backend_blocking(
                &spec,
                &artifact,
                &runtime_config,
                worker_load_context,
                ane_supervisor,
                catalog_lane.as_deref(),
            )?;
            (backend, loaded, None)
        }
        "owned-metal" => {
            if !cfg!(target_os = "macos") {
                return Err(artifact_invalid_error(format!(
                    "owned-metal model '{}' is only supported on macOS",
                    spec.model_id
                )));
            }
            let profile = owned_profile.as_ref().ok_or_else(|| {
                artifact_invalid_error("owned-metal catalog entry is missing runtime profile")
            })?;
            let mut engine = OwnedMetalEmbedEngine::new(profile.family, profile.dtype);
            let loaded_model = EmbedEngine::load(&mut engine, &artifact, &runtime_config)
                .map_err(engine_error_to_wire)?;
            if task == ModelTask::Rerank {
                engine
                    .validate_rerank(&loaded_model)
                    .map_err(engine_error_to_wire)?;
            }
            let policy = engine
                .tokenizer_policy(&loaded_model)
                .map_err(engine_error_to_wire)?;
            (
                EmbedBackend::Owned(Arc::new(Mutex::new(engine))),
                loaded_model,
                Some(policy),
            )
        }
        LLAMA_ENGINE | "owned-vulkan" | "ane-direct-worker" => {
            let (backend, loaded) = load_worker_backend_blocking(
                &spec,
                &artifact,
                &runtime_config,
                worker_load_context,
                ane_supervisor,
                catalog_lane.as_deref(),
            )?;
            (backend, loaded, None)
        }
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
        fingerprint: spec.fingerprint.clone(),
        engine_identity: spec.engine_identity.clone(),
        owned_tokenizer_policy,
    })
}

// Only the macOS direct-ANE backend calls this, but the mapping itself is
// platform-independent, and its tests run everywhere.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn ane_residency_error_to_wire(
    error: worker_host::ane_residency::AneResidencyError,
) -> WireOperationError {
    use worker_host::ane_residency::AneResidencyError;
    if matches!(error, AneResidencyError::DeadlineExceeded) {
        return WireOperationError::from_stable(
            StableError::deadline_exceeded(),
            error.to_string(),
        );
    }
    if let AneResidencyError::SequenceTooLong { n_tokens, max } = &error {
        return WireOperationError::from_stable(
            StableError::sequence_too_long(*n_tokens, *max, None),
            error.to_string(),
        );
    }
    if let AneResidencyError::PermitUnavailable(message) = &error {
        return WireOperationError::from_stable(StableError::queue_full(Some(100)), message);
    }
    if let AneResidencyError::ShapeCompiling {
        shape,
        retry_after_ms,
    } = &error
    {
        return WireOperationError {
            code: worker_host::ane_residency::ERR_ANE_SHAPE_COMPILING.to_owned(),
            class: ErrorClass::Transient,
            retry_after_ms: Some(*retry_after_ms),
            safe_to_retry_same_request: true,
            message: error.to_string(),
            details: Some(json!({ "shape": shape })),
        };
    }
    let engine_error = error.to_engine_error(EngineErrorStage::Inference);
    let safe_to_retry = engine_error.safe_to_retry_same_request;
    let mut wire = engine_error_to_wire(engine_error);
    wire.safe_to_retry_same_request = safe_to_retry;
    if let Some(code) = error.code() {
        wire.code = code.to_owned();
    }
    if error.code() == Some(worker_host::ane_residency::ERR_ANE_RESOURCES_EXHAUSTED) {
        wire.class = ErrorClass::Transient;
        wire.retry_after_ms = Some(worker_host::ane_residency::ANE_RESOURCES_RETRY_AFTER_MS);
        wire.safe_to_retry_same_request = true;
    } else if matches!(error, AneResidencyError::WorkerErr { .. }) {
        wire.class = ErrorClass::Permanent;
        wire.retry_after_ms = None;
    }
    if matches!(error, AneResidencyError::ResourcesExhausted { .. }) {
        wire.safe_to_retry_same_request = true;
    }
    wire
}

/// Maps a direct Neural Engine serving error to the wire. For
/// `shape_compiling` it also adds the lane id to `details` and names the lane
/// and the compiling shape in the message.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn ane_lane_error_to_wire(
    error: worker_host::ane_residency::AneResidencyError,
    lane: &str,
) -> WireOperationError {
    if let worker_host::ane_residency::AneResidencyError::Quarantined { retry_after_ms } = &error {
        let mut wire = WireOperationError::from_stable(
            StableError::engine_crashed(Some(*retry_after_ms)),
            error.to_string(),
        );
        wire.details = Some(json!({"lane_id": lane, "quarantined": true}));
        return wire;
    }
    if matches!(
        error,
        worker_host::ane_residency::AneResidencyError::SchedulerStopped
    ) {
        // The compile scheduler stopped: the lane stays closed until the
        // module restarts, so a retry only helps after that restart.
        let mut wire =
            WireOperationError::from_stable(StableError::engine_crashed(None), error.to_string());
        wire.details = Some(json!({"lane_id": lane, "scheduler_stopped": true}));
        return wire;
    }
    if let worker_host::ane_residency::AneResidencyError::WorkerClosed { worker_id } = &error {
        // The worker's old process may still own the Neural Engine, so it
        // stays closed until the module restarts; waiting cannot help.
        let mut wire =
            WireOperationError::from_stable(StableError::engine_crashed(None), error.to_string());
        wire.details =
            Some(json!({"lane_id": lane, "worker_id": worker_id, "worker_closed": true}));
        return wire;
    }
    let mut wire = ane_residency_error_to_wire(error);
    if wire.code == worker_host::ane_residency::ERR_ANE_SHAPE_COMPILING {
        if let Some(Value::Object(details)) = wire.details.as_mut() {
            details.insert("lane_id".into(), json!(lane));
            let shape = details.get("shape").cloned().unwrap_or(Value::Null);
            wire.message = format!("lane '{lane}' is compiling Neural Engine shape {shape}");
        }
    }
    wire
}

fn direct_ane_supervisor(
    runtime: &RuntimeState,
    engine: &str,
) -> Result<Option<worker_host::ane_residency::AneResidencySupervisor>, WireOperationError> {
    if engine != "ane-direct-worker" {
        return Ok(None);
    }
    if !cfg!(target_os = "macos") {
        return Err(artifact_invalid_error(
            "direct-ANE is only supported on macOS",
        ));
    }
    #[cfg(unix)]
    {
        let mut shared = runtime
            .ane_supervisor
            .lock()
            .map_err(|_| artifact_invalid_error("direct-ANE supervisor mutex poisoned"))?;
        if shared.is_none() {
            *shared =
                Some(acquire_runtime_ane_supervisor(runtime).map_err(ane_residency_error_to_wire)?);
        }
        if let Some(supervisor) = shared.as_ref() {
            supervisor
                .reacquire_lane()
                .map_err(ane_residency_error_to_wire)?;
        }
        Ok(shared.clone())
    }
    #[cfg(not(unix))]
    {
        let _ = runtime;
        Err(artifact_invalid_error(
            "direct-ANE is only supported on macOS",
        ))
    }
}

#[cfg(unix)]
fn runtime_ane_lock_path(
    runtime: &RuntimeState,
) -> Result<&Path, worker_host::ane_residency::AneResidencyError> {
    #[cfg(test)]
    {
        let root = runtime
            .ane_lane_test_root
            .as_ref()
            .expect("module tests must configure an ANE lock fixture root");
        let path = runtime
            .ane_lane_lock_path
            .as_ref()
            .expect("module tests must configure an ANE lock fixture path");
        assert!(
            path.starts_with(root)
                && !path
                    .components()
                    .any(|c| c == std::path::Component::ParentDir),
            "module test ANE lock must stay under its fixture root"
        );
        worker_host::ane_residency::assert_test_ane_lock_path(path);
    }
    runtime.ane_lane_lock_path.as_deref().ok_or_else(|| {
        worker_host::ane_residency::AneResidencyError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "HOME is not set, so the ANE lane lock has no location",
        ))
    })
}

#[cfg(unix)]
fn acquire_runtime_ane_supervisor(
    runtime: &RuntimeState,
) -> Result<
    worker_host::ane_residency::AneResidencySupervisor,
    worker_host::ane_residency::AneResidencyError,
> {
    use worker_host::ane_residency::{AneDirectLaneLock, AneResidencySupervisor};
    let mut supervisor = AneResidencySupervisor::with_lane_lock(
        Default::default(),
        AneDirectLaneLock::acquire(runtime_ane_lock_path(runtime)?)?,
    );
    if runtime.certify_observation {
        supervisor.enable_observations();
    }
    Ok(supervisor)
}

fn load_worker_backend_blocking(
    spec: &StoredModelConfig,
    artifact: &ValidatedArtifact,
    runtime_config: &RuntimeConfig,
    worker_load_context: WorkerLoadContext,
    ane_supervisor: Option<worker_host::ane_residency::AneResidencySupervisor>,
    catalog_lane: Option<&str>,
) -> Result<(EmbedBackend, LoadedModel), WireOperationError> {
    use worker_host::{WorkerEngine, WorkerHostConfig};

    let macos_available = cfg!(target_os = "macos");
    if spec.engine == "ane-direct-worker" && !macos_available {
        return Err(artifact_invalid_error(format!(
            "{} model '{}' is only supported on macOS",
            spec.engine, spec.model_id
        )));
    }

    let worker_bin_var = worker_binary_env_var(&spec.engine);
    let worker_runtime_dir_var = worker_runtime_dir_env_var(&spec.engine);
    let worker_bin = spec
        .worker_bin
        .clone()
        .or_else(|| env::var_os(&worker_bin_var).map(PathBuf::from))
        .or_else(|| resolve_worker_binary_sibling(&spec.engine))
        .ok_or_else(|| {
            artifact_invalid_error(format!(
                "{} model '{}' requires worker_bin, {}, or a sibling worker binary",
                spec.engine, spec.model_id, worker_bin_var
            ))
        })?;
    let runtime_dir = spec
        .worker_runtime_dir
        .clone()
        .or_else(|| env::var_os(&worker_runtime_dir_var).map(PathBuf::from))
        .unwrap_or_else(|| env::temp_dir().join("synapse-workers"));
    let mut config = WorkerHostConfig::new(worker_bin, runtime_dir);
    let _ = worker_load_context.timeout;
    config.worker_id = format!("synapse-{}-{}", spec.engine, spec.model_id);
    config.model_id = Some(spec.model_id.clone());
    config.catalog_lane = catalog_lane.map(str::to_owned);
    config.findings_changed = worker_load_context.findings_changed;
    config.worker_forward_lines_per_sec = worker_load_context.forward_lines_per_sec;
    config.engine_identity = Some(spec.engine_identity.clone());
    config.isolate_crash_key_by_worker_id = spec.engine == CUDA_WORKER_ENGINE;
    config.pooling =
        parse_pooling(&spec.pooling).map_err(|error| artifact_invalid_error(error.to_string()))?;
    config.normalize = spec.normalize;
    if spec.task == "generate" {
        config.request_timeout = Duration::from_secs(180);
    }
    #[cfg(unix)]
    if spec.engine == "ane-direct-worker" {
        let supervisor = ane_supervisor
            .ok_or_else(|| artifact_invalid_error("direct-ANE supervisor unavailable"))?;
        let engine = worker_host::ane_residency::DirectAneEngine::load(
            config,
            supervisor,
            artifact,
            runtime_config,
        )
        .map_err(ane_residency_error_to_wire)?;
        let loaded = LoadedModel {
            model_id: engine.serving.metadata.model_ref.clone(),
        };
        return Ok((EmbedBackend::DirectAne(Arc::new(engine)), loaded));
    }
    #[cfg(not(unix))]
    let _ = ane_supervisor;
    let engine = WorkerEngine::new(config);
    let mut engine = engine.map_err(|error| {
        WireOperationError::from_stable(
            StableError::engine_crashed(Some(100)),
            format!(
                "create {} worker engine for '{}': {error}",
                spec.engine, spec.model_id
            ),
        )
    })?;
    let loaded_model = EmbedEngine::load(&mut engine, artifact, runtime_config)
        .map_err(|error| worker_error_to_wire(error, catalog_lane))?;
    Ok((
        EmbedBackend::Worker(Arc::new(Mutex::new(engine))),
        loaded_model,
    ))
}

fn unload_embedding_model_blocking(model: Arc<EmbeddingModel>) -> Result<(), WireOperationError> {
    match &model.backend {
        #[cfg(feature = "test-support")]
        EmbedBackend::TestDeterministic(_) => Ok(()),
        EmbedBackend::Owned(engine) => {
            let mut engine = engine.lock().map_err(|_| {
                WireOperationError::from_stable(
                    StableError::engine_crashed(Some(100)),
                    "owned-metal engine mutex was poisoned during model unload",
                )
            })?;
            EmbedEngine::unload(&mut *engine, &model.loaded_model);
            Ok(())
        }
        #[cfg(unix)]
        EmbedBackend::DirectAne(engine) => engine.unload().map_err(ane_residency_error_to_wire),
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

fn model_runtime_config(
    spec: &StoredModelConfig,
    model_path: &Path,
    model_cache_root: &Path,
    microllm_max_tokens: u32,
) -> RuntimeConfig {
    let mut runtime_config = RuntimeConfig::default();
    if let Some(profile) = release_profile(&spec.engine_identity) {
        runtime_config
            .values
            .insert("profile".into(), profile.to_owned());
        runtime_config
            .values
            .insert("operation".into(), spec.task.clone());
    }
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
    runtime_config.values.insert(
        "microllm_max_tokens".to_string(),
        microllm_max_tokens.to_string(),
    );
    if spec.engine == LLAMA_ENGINE {
        let backend = spec
            .engine_identity
            .build_flags
            .get("backend")
            .cloned()
            .unwrap_or_else(|| {
                // Legacy catalog rows predate backend identity. The default worker
                // build selects Metal on macOS and CPU elsewhere, so this declaration
                // cannot mismatch the worker's default build on the same platform.
                default_llama_backend().to_string()
            });
        runtime_config.values.insert("backend".to_string(), backend);
    }
    if spec.engine == "owned-cuda" {
        runtime_config.values.insert(
            "backend".to_string(),
            spec.engine_identity
                .build_flags
                .get("backend")
                .cloned()
                .unwrap_or_else(|| "cuda-ptx".to_string()),
        );
        runtime_config.values.insert(
            "ptx_virtual_arch".to_string(),
            spec.engine_identity
                .build_flags
                .get("ptx_virtual_arch")
                .cloned()
                .unwrap_or_else(|| OWNED_CUDA_PTX_VIRTUAL_ARCH.to_string()),
        );
        runtime_config.values.insert(
            "minimum_device_cc".to_string(),
            spec.engine_identity
                .build_flags
                .get("minimum_device_cc")
                .cloned()
                .unwrap_or_else(|| OWNED_CUDA_MINIMUM_DEVICE_CC.to_string()),
        );
        runtime_config.values.insert(
            "minimum_cuda_driver_api".to_string(),
            spec.engine_identity
                .build_flags
                .get("minimum_cuda_driver_api")
                .cloned()
                .unwrap_or_else(|| OWNED_CUDA_MINIMUM_DRIVER_API.to_string()),
        );
    }
    if spec.engine == "owned-metal" {
        // The engine identity decides the graph: a lane whose identity records
        // f16 projections asks the engine for them, and a lane without the flag
        // gets the engine's default f32 projections.
        if let Some(projections) = spec
            .engine_identity
            .build_flags
            .get(PROJECTION_DTYPE_CONFIG_KEY)
        {
            runtime_config
                .values
                .insert(PROJECTION_DTYPE_CONFIG_KEY.to_string(), projections.clone());
        }
        runtime_config
            .values
            .insert("max_tokens".to_string(), spec.max_tokens.to_string());
        runtime_config.values.insert(
            "package_cache_root".to_string(),
            model_cache_root
                .join("owned-metal-packages")
                .to_string_lossy()
                .to_string(),
        );
        runtime_config.values.insert(
            "execution".to_string(),
            spec.owned_execution
                .clone()
                .unwrap_or_else(|| "explicit".to_string()),
        );
        runtime_config.values.insert(
            "attention_units".to_string(),
            spec.owned_attention_units
                .unwrap_or(OWNED_DEFAULT_ATTENTION_UNITS)
                .to_string(),
        );
    }
    runtime_config
}

// PREMISE this fallback depends on (not just the conclusion): a legacy catalog
// row without a declared backend was necessarily loaded by a default-build
// worker on this same platform, so "the platform default" and "what the
// artifact ran against" are the same value. That holds only while this mapping
// mirrors the llama worker's default build selection (metal on macOS, cpu
// elsewhere). If the worker's default build ever changes, this function must
// change with it in the same commit — otherwise legacy rows silently declare
// a backend their artifacts never ran under, and the worker-side equality
// check turns that drift into hard load refusals.
fn default_llama_backend() -> &'static str {
    if cfg!(target_os = "macos") {
        "metal"
    } else {
        "cpu"
    }
}

/// Build a llama runtime configuration through the same catalog path used by
/// production preloads, without loading a model artifact.
#[doc(hidden)]
pub fn llama_backend_contract_runtime_config(
    model_path: &Path,
    tokenizer_path: &Path,
    declared_backend: Option<&str>,
) -> Result<RuntimeConfig, String> {
    let preload: PreloadModelConfig = serde_json::from_value(json!({
        "model_id": "backend-contract-llama",
        "engine": LLAMA_ENGINE,
        "task": "embed",
        "model_path": model_path,
        "tokenizer_path": tokenizer_path,
        "format": "gguf",
        "pooling": "mean",
        "normalize": true,
        "max_tokens": 512,
        "quant": "f16",
        "backend": declared_backend
    }))
    .map_err(|error| error.to_string())?;
    let spec =
        build_preload_catalog_model(0, preload, &InlineConfig::default(), &JobConfig::default())
            .map_err(|error| error.to_string())?;
    let spec = normalize_catalog_model(spec, &InlineConfig::default(), &JobConfig::default())
        .map_err(|error| error.to_string())?;
    Ok(model_runtime_config(
        &spec,
        model_path,
        Path::new("/tmp/synapse-backend-contract-cache"),
        DEFAULT_MICROLLM_MAX_TOKENS,
    ))
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

fn cuda_floor_override() -> Option<(u32, u32, u32)> {
    let driver_api = ["SYNAPSE_CUDA_DRIVER_API", "CUDA_DRIVER_API"]
        .into_iter()
        .find_map(|name| {
            env::var(name)
                .ok()
                .and_then(|value| value.parse::<u32>().ok())
        });
    let compute = ["SYNAPSE_CUDA_COMPUTE_CAPABILITY", "CUDA_COMPUTE_CAPABILITY"]
        .into_iter()
        .find_map(|name| {
            env::var(name)
                .ok()
                .and_then(|value| parse_compute_capability(&value))
        });
    let (driver_api, (major, minor)) = (driver_api?, compute?);
    Some((driver_api, major, minor))
}

fn owned_cuda_floor_decision(worker: Option<&Path>) -> CudaFloorDecision {
    let packaging_driver = env::var("SYNAPSE_CUDA_PACKAGING_DRIVER").ok();
    let Some((driver_api, major, minor)) = cuda_floor_override() else {
        // The environment is the override; when it is silent, ask the worker.
        // The module deliberately does not link the CUDA driver, so the probe
        // has to run in the worker process and report its numbers back.
        return match owned_cuda_probe_floor(worker) {
            Ok(reading) => evaluate_cuda_floor(
                reading.driver_api,
                reading.compute_major,
                reading.compute_minor,
                packaging_driver,
            ),
            Err(_) => CudaFloorDecision::Unsupported {
                reason: synapse_core::CudaUnsupportedReason::HardwareUnavailable,
                observed: None,
            },
        };
    };
    evaluate_cuda_floor(driver_api, major, minor, packaging_driver)
}

/// A hardware reading reported by `ck-synapse-worker-cuda --probe-floor`.
#[derive(Clone, Copy, Debug)]
struct OwnedCudaFloorReading {
    driver_api: u32,
    compute_major: u32,
    compute_minor: u32,
}

fn resolve_catalog_worker(engine: &str, configured: Option<&Path>) -> Option<PathBuf> {
    configured
        .map(Path::to_path_buf)
        .or_else(|| env::var_os(worker_binary_env_var(engine)).map(PathBuf::from))
        .or_else(|| resolve_worker_binary_sibling(engine))
}

/// Catalog detection and CUDA loading reuse the worker's cached hardware-floor
/// probe. Refusals are cached as well, so loading cannot repeatedly stall on a
/// missing driver or an unsupported device until the module restarts.
fn owned_cuda_probe_floor(worker: Option<&Path>) -> Result<OwnedCudaFloorReading, String> {
    let worker = resolve_catalog_worker(CUDA_WORKER_ENGINE, worker)
        .ok_or_else(|| "worker_missing".to_string())?;
    let output = catalog_probe::cached_probe(&worker, catalog_probe::ProbeKind::Floor)
        .map_err(|error| format!("{}: {}", error.reason, error.detail))?;
    cuda_reading_from_output(output)
}

fn cuda_reading_from_output(
    output: catalog_probe::ProbeOutput,
) -> Result<OwnedCudaFloorReading, String> {
    let stderr = output.stderr.clone();
    if !output.status.success() && output.status.code() != Some(2) {
        return Err(format!(
            "CUDA floor probe exited {}; stderr: {stderr}",
            output.status
        ));
    }
    let parsed = catalog_probe::floor_result(output).map_err(|reason| {
        format!("invalid CUDA floor probe JSON or refusal: {reason}; stderr: {stderr}")
    })?;
    cuda_reading(&parsed).ok_or_else(|| "invalid CUDA floor probe hardware fields".to_string())
}

fn cuda_reading(parsed: &Value) -> Option<OwnedCudaFloorReading> {
    let observed = parsed.get("observed")?;
    Some(OwnedCudaFloorReading {
        driver_api: observed.get("driver_api")?.as_u64()?.try_into().ok()?,
        compute_major: observed
            .get("compute_capability")?
            .get("major")?
            .as_u64()?
            .try_into()
            .ok()?,
        compute_minor: observed
            .get("compute_capability")?
            .get("minor")?
            .as_u64()?
            .try_into()
            .ok()?,
    })
}

#[cfg(test)]
fn run_owned_cuda_probe(
    command: &mut std::process::Command,
    timeout: Duration,
) -> Result<OwnedCudaFloorReading, String> {
    let output = catalog_probe::run_probe(command, timeout)
        .map_err(|error| format!("{}: {}", error.reason, error.detail))?;
    cuda_reading_from_output(output)
}
fn parse_compute_capability(value: &str) -> Option<(u32, u32)> {
    let mut parts = value.trim().split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().unwrap_or("0").parse().ok()?;
    parts.next().is_none().then_some((major, minor))
}

fn owned_cuda_floor_observed(decision: &CudaFloorDecision, worker: Option<&Path>) -> Value {
    let error = match decision {
        CudaFloorDecision::Unsupported { observed: None, .. } => {
            owned_cuda_probe_floor(worker).err()
        }
        _ => None,
    };
    floor_observed_with_probe_error(decision, error.as_deref())
}

fn floor_observed_with_probe_error(decision: &CudaFloorDecision, error: Option<&str>) -> Value {
    match decision {
        CudaFloorDecision::Supported { observed }
        | CudaFloorDecision::Unsupported {
            observed: Some(observed),
            ..
        } => serde_json::to_value(observed).unwrap_or(Value::Null),
        CudaFloorDecision::Unsupported { observed: None, .. } => error
            .map(|stderr| json!({ "probe_stderr": stderr }))
            .unwrap_or(Value::Null),
    }
}

fn ensure_owned_cuda_floor(worker: Option<&Path>) -> Result<(), WireOperationError> {
    let decision = owned_cuda_floor_decision(worker);
    if decision.is_supported() {
        return Ok(());
    }
    let observed = owned_cuda_floor_observed(&decision, worker);
    Err(WireOperationError::from_stable(
        StableError::owned_cuda_unsupported(),
        format!(
            "owned-cuda floor refused before worker creation: decision={}, observed={}",
            decision.refusal_code().unwrap_or("owned_cuda_unsupported"),
            observed,
        ),
    ))
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
            "cache artifact {digest} is missing; reacquire the model assets"
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

fn huggingface_resolve_url(
    endpoint: &str,
    repo: &str,
    revision: &str,
    file: &str,
) -> Result<String, String> {
    validate_hf_endpoint(endpoint)?;
    if !pinned_revision(revision) {
        return Err("Hugging Face revision must be a 40-hex commit".into());
    }
    let mut url =
        Url::parse(endpoint).map_err(|error| format!("build Hugging Face base URL: {error}"))?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| "build Hugging Face path segments".to_string())?;
        segments.pop_if_empty();
        for segment in repo.split('/') {
            if !segment.is_empty() {
                segments.push(segment);
            }
        }
        segments.push("resolve");
        segments.push(revision);
        for segment in file.split('/') {
            if !segment.is_empty() {
                segments.push(segment);
            }
        }
    }
    Ok(url.to_string())
}

async fn embed_query(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let _pause_guard = match lane_pause::admit_params(&state, &params, ModelTask::Embed) {
        Ok(guard) => guard,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };
    let request_started = tokio::time::Instant::now();
    let params: EmbedQueryParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid embed.query params: {error}"),
            )
        }
    };
    let catalog_sequences = match preflight_catalog_sequences(
        &state,
        params.model.as_deref(),
        params.required_fingerprint.as_deref(),
        params.target_fingerprint.as_deref(),
        None,
        &[params.text.as_str()],
    ) {
        Ok(sequences) => sequences,
        Err(mut error) => {
            if let Some(details) = error.details.as_mut() {
                if error.code == "sequence_too_long" {
                    details["item_id"] = json!(params.id.as_deref().unwrap_or("query"));
                }
            }
            return result_outcome(error_payload(&state, error));
        }
    };
    let request_deadline = request_started
        + Duration::from_millis(
            params
                .deadline_ms
                .unwrap_or(state.runtime.inline.deadline_ms),
        );
    let model = match resolve_serving_model_until(
        Arc::clone(&state),
        params.model.as_deref(),
        ModelTask::Embed,
        params.required_fingerprint.as_deref(),
        params.target_fingerprint.as_deref(),
        request_deadline,
    )
    .await
    {
        Ok(model) => model,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };
    // Certification, admission and execution share the deadline used by resolution.
    let certify_deadline = Some(request_deadline);
    if let Err(error) = check_fingerprint_constraints(
        &model,
        params.target_fingerprint.as_deref(),
        params.required_fingerprint.as_deref(),
        params.required_epoch,
    ) {
        return result_outcome(error_payload(&state, error));
    }

    let request_bytes = request_bytes_for_texts([params.text.as_str()]);
    let job_id = state
        .runtime
        .activity_telemetry
        .next_inline_job_id(state.module_generation);
    let started = Instant::now();
    let admission = match state.runtime.admit_inline_until(
        &model.model_id,
        Some(&job_id),
        QueueClass::Interactive,
        request_bytes,
        request_deadline,
        params.max_queue_ms,
    ) {
        Ok(admission) => admission,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };
    let mut tokenized = match catalog_sequences
        .map(|sequences| catalog_embed_batch(&[params.text.as_str()], sequences))
        .map(Ok)
        .unwrap_or_else(|| model.tokenizer.tokenize_batch([params.text.as_str()]))
    {
        Ok(tokenized) => tokenized,
        Err(error) => {
            return failed_admitted_outcome(
                &state,
                &model,
                &job_id,
                started,
                WireOperationError::from_stable(StableError::artifact_invalid(), error.to_string()),
            )
        }
    };
    ane_lane_timing(
        "query_admission_and_tokenize",
        &model.model_id,
        Some(&job_id),
        started,
    );
    if let Err(mut error) = compose_catalog_embed(&model, &mut tokenized) {
        if let Some(details) = error.details.as_mut() {
            details["item_id"] = json!(params.id.as_deref().unwrap_or("query"));
        }
        return failed_admitted_outcome(&state, &model, &job_id, started, error);
    }
    if let Err(error) =
        ensure_profile_request_certified(state.clone(), &model, certify_deadline).await
    {
        return failed_admitted_outcome(&state, &model, &job_id, started, error);
    }
    apply_owned_tokenizer_policy(&model, &mut tokenized);
    let ids = vec![params.id.unwrap_or_else(|| "query".to_string())];
    embed_tokenized(
        state,
        model,
        ids,
        tokenized,
        false,
        InlineWorkBudget {
            request_bytes,
            deadline: Some(admission.deadline()),
            job_id,
            started,
        },
    )
    .await
}

fn sha256_text(text: &str) -> String {
    hex::encode(Sha256::digest(text.as_bytes()))
}

async fn embed_batch(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let _pause_guard = match lane_pause::admit_params(&state, &params, ModelTask::Embed) {
        Ok(guard) => guard,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };
    let resolution_started = Instant::now();
    let request_started = tokio::time::Instant::now();
    let params: EmbedBatchParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid embed.batch params: {error}"),
            )
        }
    };
    let mut job_params = json!({"model":params.model,"required_fingerprint":params.required_fingerprint,"target_fingerprint":params.target_fingerprint,"allow_equivalent":false,"required_epoch":params.required_epoch,"request_key":params.request_key});
    let items = match batch_items(params.items, params.texts) {
        Ok(items) => items,
        Err(message) => return channel_error("invalid_request", message),
    };
    if items.is_empty() {
        return channel_error("invalid_request", "embed.batch requires at least one item");
    }
    let text_refs = items
        .iter()
        .map(|item| item.text.as_str())
        .collect::<Vec<_>>();
    let catalog_sequences = match preflight_catalog_sequences(
        &state,
        params.model.as_deref(),
        params.required_fingerprint.as_deref(),
        params.target_fingerprint.as_deref(),
        None,
        &text_refs,
    ) {
        Ok(sequences) => sequences,
        Err(mut error) => {
            if error.code == "sequence_too_long" {
                if let Some(details) = error.details.as_mut() {
                    if let Some(index) = details["item_id"]
                        .as_str()
                        .and_then(|id| id.parse::<usize>().ok())
                    {
                        details["item_id"] = json!(items[index].id);
                    }
                }
            }
            return result_outcome(error_payload(&state, error));
        }
    };
    if params
        .model
        .as_deref()
        .is_none_or(|id| state.runtime.release_catalog.is_reserved_id(id))
    {
        let (entry, backend) = match select_catalog_lane(
            &state,
            params.model.as_deref(),
            ModelTask::Embed,
            params.required_fingerprint.as_deref(),
            params.target_fingerprint.as_deref(),
        ) {
            Ok(lane) => lane,
            Err(e) => return result_outcome(error_payload(&state, e)),
        };
        if items.len() > state.runtime.inline.max_items {
            job_params["items"] = json!(items
                .iter()
                .map(|i| json!({"id":i.id,"text":i.text}))
                .collect::<Vec<_>>());
            return submit_catalog_embed_job(state, entry, backend, job_params).await;
        }
    }

    let request_deadline = request_started
        + Duration::from_millis(
            params
                .deadline_ms
                .unwrap_or(state.runtime.inline.deadline_ms),
        );
    let model = match resolve_serving_model_until(
        Arc::clone(&state),
        params.model.as_deref(),
        ModelTask::Embed,
        params.required_fingerprint.as_deref(),
        params.target_fingerprint.as_deref(),
        request_deadline,
    )
    .await
    {
        Ok(model) => model,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };
    // Certification, admission and execution share the deadline used by resolution.
    let certify_deadline = Some(request_deadline);
    if let Err(error) = check_fingerprint_constraints(
        &model,
        params.target_fingerprint.as_deref(),
        params.required_fingerprint.as_deref(),
        params.required_epoch,
    ) {
        return result_outcome(error_payload(&state, error));
    }

    let request_bytes = request_bytes_for_texts(text_refs.iter().copied());
    ane_lane_timing("batch_resolve", &model.model_id, None, resolution_started);
    let tokenize_started = Instant::now();
    let mut tokenized = match catalog_sequences
        .map(|sequences| catalog_embed_batch(&text_refs, sequences))
        .map(Ok)
        .unwrap_or_else(|| model.tokenizer.tokenize_batch(text_refs))
    {
        Ok(tokenized) => tokenized,
        Err(error) => {
            return result_outcome(error_payload(
                &state,
                WireOperationError::from_stable(StableError::artifact_invalid(), error.to_string()),
            ))
        }
    };
    ane_lane_timing("batch_tokenize", &model.model_id, None, tokenize_started);
    let prepare_started = Instant::now();
    if let Err(mut error) = compose_catalog_embed(&model, &mut tokenized) {
        if let Some(details) = error.details.as_mut() {
            if let Some(index) = details["item_id"]
                .as_str()
                .and_then(|id| id.parse::<usize>().ok())
            {
                details["item_id"] = json!(items[index].id);
            }
        }
        return result_outcome(error_payload(&state, error));
    }
    if let Err(error) =
        ensure_profile_request_certified(state.clone(), &model, certify_deadline).await
    {
        return result_outcome(error_payload(&state, error));
    }
    apply_owned_tokenizer_policy(&model, &mut tokenized);
    let total_tokens = tokenized
        .real_token_counts
        .iter()
        .map(|tokens| u64::from(*tokens))
        .sum::<u64>();
    let digest_items = items
        .iter()
        .map(|item| (item.id.clone(), sha256_hex(item.text.as_bytes())))
        .collect::<Vec<_>>();
    let request_digest = compute_request_digest(
        "embed.batch",
        &model.model_id,
        &json!({
            "target_fingerprint": params.target_fingerprint,
            "required_fingerprint": params.required_fingerprint,
            "allow_equivalent": false,
            "required_epoch": params.required_epoch,
            // Preserve the historical local digest so retained request keys still replay.
            "accept_declared": false,
        }),
        &digest_items,
    );
    let ids = items.into_iter().map(|item| item.id).collect::<Vec<_>>();
    // Composition, tokenizer policy and the request digest, after tokenization.
    ane_lane_timing("batch_prepare", &model.model_id, None, prepare_started);

    if ids.len() > state.runtime.inline.max_items || total_tokens > state.runtime.inline.max_tokens
    {
        return submit_embed_batch_job(
            state,
            params.request_key,
            EmbedBatchJobWork {
                model,
                request_digest,
                ids,
                tokenized,
                request_bytes,
                total_tokens,
            },
        )
        .await;
    }

    let job_id = state
        .runtime
        .activity_telemetry
        .next_inline_job_id(state.module_generation);
    let started = Instant::now();
    let admission = match state.runtime.admit_inline_until(
        &model.model_id,
        Some(&job_id),
        QueueClass::Bulk,
        request_bytes,
        request_deadline,
        params.max_queue_ms,
    ) {
        Ok(admission) => admission,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };
    ane_lane_timing("batch_admission", &model.model_id, Some(&job_id), started);
    embed_tokenized(
        state,
        model,
        ids,
        tokenized,
        true,
        InlineWorkBudget {
            request_bytes,
            deadline: Some(admission.deadline()),
            job_id,
            started,
        },
    )
    .await
}

async fn rerank_score(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let _pause_guard = match lane_pause::admit_params(&state, &params, ModelTask::Rerank) {
        Ok(guard) => guard,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };
    let request_started = tokio::time::Instant::now();
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
    let candidates = params
        .candidates
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let catalog_pairs = match preflight_catalog_sequences(
        &state,
        params.model.as_deref(),
        params.required_fingerprint.as_deref(),
        params.target_fingerprint.as_deref(),
        Some(&params.query),
        &candidates,
    ) {
        Ok(pairs) => pairs,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };
    let catalog_request = catalog_pairs.is_some();
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

    let request_deadline = request_started
        + Duration::from_millis(
            params
                .deadline_ms
                .unwrap_or(state.runtime.inline.deadline_ms),
        );
    let model = match resolve_serving_model_until(
        Arc::clone(&state),
        params.model.as_deref(),
        ModelTask::Rerank,
        params.required_fingerprint.as_deref(),
        params.target_fingerprint.as_deref(),
        request_deadline,
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
    // Certification, admission and execution share the deadline used by resolution.
    let certify_deadline = Some(request_deadline);
    if let Err(error) = check_fingerprint_constraints(
        &model,
        params.target_fingerprint.as_deref(),
        params.required_fingerprint.as_deref(),
        params.required_epoch,
    ) {
        return result_outcome(error_payload(&state, error));
    }

    let owned_pairs =
        match catalog_pairs.map(Some).map(Ok).unwrap_or_else(|| {
            owned_rerank_pairs(&model, params.query.as_str(), &params.candidates)
        }) {
            Ok(pairs) => pairs,
            Err(error) => return result_outcome(error_payload(&state, error)),
        };
    if let Err(error) =
        ensure_profile_request_certified(state.clone(), &model, certify_deadline).await
    {
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
    let candidate_token_counts = if catalog_request {
        owned_pairs
            .as_ref()
            .expect("catalog composed pairs")
            .iter()
            .map(|pair| pair.len() as u64)
            .sum()
    } else {
        token_items
            .iter()
            .map(|candidate| {
                candidate
                    .len()
                    .saturating_add(query.len())
                    .saturating_add(3) as u64
            })
            .sum::<u64>()
    };
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
    let job_id = state
        .runtime
        .activity_telemetry
        .next_inline_job_id(state.module_generation);
    let started = Instant::now();
    let admission = match state.runtime.admit_inline_until(
        &model.model_id,
        Some(&job_id),
        queue_class,
        request_bytes,
        request_deadline,
        params.max_queue_ms,
    ) {
        Ok(admission) => admission,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };

    let observation = if state.runtime.certify_observation {
        let mut observation = match &owned_pairs {
            Some(pairs) => json!({"input_ids": pairs}),
            None => json!({"query_ids": query, "candidate_ids": token_items}),
        };
        if let Some(profile_id) = model.engine_identity.build_flags.get("profile") {
            if let Ok(profile) = CatalogProfile::load(profile_id) {
                let readout = &profile.model()["grammar"]["readout"];
                if readout["yes"]["id"].is_u64() && readout["no"]["id"].is_u64() {
                    observation["readout_ids"] = json!([readout["yes"]["id"], readout["no"]["id"]]);
                }
            }
        }
        Some(observation)
    } else {
        None
    };
    let scores = match execute_rerank(
        &state.runtime,
        &model,
        RerankRequest {
            query,
            candidates: token_items,
        },
        owned_pairs,
        Some(admission.deadline()),
        Some(&job_id),
    )
    .await
    {
        Ok(scores) => scores,
        Err(error) => return failed_admitted_outcome(&state, &model, &job_id, started, error),
    };
    if scores.scores.len() != params.candidates.len() {
        return failed_admitted_outcome(
            &state,
            &model,
            &job_id,
            started,
            WireOperationError::from_stable(
                StableError::engine_crashed(None),
                format!(
                    "engine returned {} rerank scores for {} candidates",
                    scores.scores.len(),
                    params.candidates.len()
                ),
            ),
        );
    }
    let equivalent_to = Vec::new();
    let payload = RerankScorePayload {
        scores: scores.scores,
        real_token_counts: tokenized.real_token_counts,
        truncation_disclosures: tokenized.disclosures,
    };
    let envelope = ResponseEnvelope {
        fingerprint: model.fingerprint.clone(),
        table_epoch: FROZEN_TABLE_EPOCH,
        dims: 1,
        provenance: ResponseProvenance {
            engine: model.engine_identity.clone(),
            owned_decode: Default::default(),
        },
        module_generation: state.module_generation,
        equivalent_to,
        payload,
    };
    log_job_done(
        &model.model_id,
        &job_id,
        execution_lane(&model),
        candidate_token_counts,
        started,
    );
    let mut response = serde_json::to_value(envelope).expect("rerank envelope should serialize");
    attach_certify_observation(&mut response, observation);
    result_outcome(response)
}

async fn microllm_oneshot(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let _pause_guard = match lane_pause::admit_params(&state, &params, ModelTask::Generate) {
        Ok(guard) => guard,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };
    let request_started = tokio::time::Instant::now();
    let params: MicroLlmOneshotParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid microllm.oneshot params: {error}"),
            )
        }
    };
    let ceiling = state.runtime.microllm_max_tokens;
    if params.max_tokens > ceiling {
        return channel_error(
            "invalid_request",
            format!(
                "microllm.oneshot max_tokens {} exceeds configured ceiling {}",
                params.max_tokens, ceiling
            ),
        );
    }
    if params
        .grammar
        .as_deref()
        .is_some_and(|grammar| !grammar.trim().is_empty())
    {
        return channel_error(
            "grammar_disabled",
            "microllm.oneshot constrained decoding is disabled in module config",
        );
    }

    let request_deadline = request_started
        + Duration::from_millis(
            params
                .deadline_ms
                .unwrap_or(state.runtime.inline.deadline_ms),
        );
    let model = match resolve_serving_model_until(
        Arc::clone(&state),
        params.model.as_deref(),
        ModelTask::Generate,
        params.required_fingerprint.as_deref(),
        params.target_fingerprint.as_deref(),
        request_deadline,
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
    if let Err(error) = check_fingerprint_constraints(
        &model,
        params.target_fingerprint.as_deref(),
        params.required_fingerprint.as_deref(),
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
    let job_id = state
        .runtime
        .activity_telemetry
        .next_inline_job_id(state.module_generation);
    let started = Instant::now();
    let admission = match state.runtime.admit_inline_until(
        &model.model_id,
        Some(&job_id),
        QueueClass::Interactive,
        request_bytes,
        request_deadline,
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
            grammar: None, // grammar requests are rejected before worker dispatch
        },
        Some(admission.deadline()),
        Some(&job_id),
    )
    .await
    {
        Ok(output) => output,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };
    let equivalent_to = Vec::new();
    let completed_tokens = (output.n_prompt as u64).saturating_add(output.n_gen as u64);
    let payload = MicroLlmOneshotPayload {
        text: output.text,
        finish_reason: output.finish_reason,
        n_prompt: output.n_prompt,
        n_gen: output.n_gen,
        generated_token_ids: output.generated_token_ids,
        real_token_counts: tokenized.real_token_counts,
        truncation_disclosures: tokenized.disclosures,
    };
    let envelope = ResponseEnvelope {
        fingerprint: model.fingerprint.clone(),
        table_epoch: FROZEN_TABLE_EPOCH,
        dims: 0,
        provenance: ResponseProvenance {
            engine: model.engine_identity.clone(),
            owned_decode: Default::default(),
        },
        module_generation: state.module_generation,
        equivalent_to,
        payload,
    };
    log_job_done(
        &model.model_id,
        &job_id,
        execution_lane(&model),
        completed_tokens,
        started,
    );
    result_outcome(serde_json::to_value(envelope).expect("microllm envelope should serialize"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn update_digest_bytes(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
}

fn update_digest_json(hasher: &mut Sha256, value: &Value) {
    match value {
        Value::Null => hasher.update([0]),
        Value::Bool(value) => hasher.update([1, u8::from(*value)]),
        Value::Number(value) => {
            hasher.update([2]);
            update_digest_bytes(hasher, value.to_string().as_bytes());
        }
        Value::String(value) => {
            hasher.update([3]);
            update_digest_bytes(hasher, value.as_bytes());
        }
        Value::Array(values) => {
            hasher.update([4]);
            hasher.update((values.len() as u64).to_be_bytes());
            for value in values {
                update_digest_json(hasher, value);
            }
        }
        Value::Object(values) => {
            hasher.update([5]);
            hasher.update((values.len() as u64).to_be_bytes());
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            for key in keys {
                update_digest_bytes(hasher, key.as_bytes());
                update_digest_json(hasher, &values[key]);
            }
        }
    }
}

fn compute_request_digest(
    op: &str,
    synapse_model_id: &str,
    constraints: &Value,
    items: &[(String, String)],
) -> String {
    let mut hasher = Sha256::new();
    update_digest_bytes(&mut hasher, b"synapse-request-digest-v1");
    update_digest_bytes(&mut hasher, op.as_bytes());
    update_digest_bytes(&mut hasher, synapse_model_id.as_bytes());
    // Keep these two bytes zero so existing durable request digests remain stable.
    hasher.update([0, 0]);
    update_digest_json(&mut hasher, constraints);
    hasher.update((items.len() as u64).to_be_bytes());
    for (item_id, content_hash) in items {
        update_digest_bytes(&mut hasher, item_id.as_bytes());
        update_digest_bytes(&mut hasher, content_hash.as_bytes());
    }
    hex::encode(hasher.finalize())
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
    let admission = match state.store.admit_job(&request_key, &work.request_digest, "embed.batch", state.module_generation, &json!({
        "model": work.model.model_id.clone(),
        "items": work.ids.len(),
        "request_bytes": work.request_bytes,
        "total_tokens": work.total_tokens,
    }), now, state.runtime.jobs.execution_ttl_ms, state.runtime.jobs.result_retention_ttl_ms) {
        Ok(admission) => admission,
        Err(SynapseStoreError::IdempotencyConflict { .. }) => {
            return result_outcome(error_payload(
                &state,
                WireOperationError::from_stable(
                    StableError::idempotency_conflict(),
                    format!("request_key '{request_key}' was already used for different request content"),
                ),
            ))
        }
        Err(error) => return channel_error("store_failure", error.to_string()),
    };

    let record = admission.record().clone();
    let job_minted = matches!(admission, JobAdmission::Admitted(_));
    if job_minted {
        state.runtime.admission_telemetry.record_job_minted();
        log_job_admitted(&work.model.model_id, &record.job_id);
        let task_state = Arc::clone(&state);
        let task_job_id = record.job_id.clone();
        tokio::spawn(async move {
            execute_embed_batch_job(task_state, task_job_id, work).await;
        });
    }

    result_outcome(job_status_payload(&state, &record))
}

async fn execute_embed_batch_job(
    state: Arc<ModuleState>,
    job_id: String,
    mut work: EmbedBatchJobWork,
) {
    let record = match state
        .store
        .claim_job_attempt(&job_id, state.module_generation, now_ms())
    {
        Ok(JobAttemptClaim::Claimed(record)) => record,
        Ok(JobAttemptClaim::Attached { .. } | JobAttemptClaim::NotClaimable(_)) => return,
        Err(_) => {
            log_stored_embed_failure(&state, &job_id, "store_failure");
            return;
        }
    };

    let _pause_guard = match lane_pause::admit(&state.runtime, &work.model.model_id) {
        Ok(guard) => guard,
        Err(error) => {
            fail_job_with_wire_error(&state, &job_id, true, error);
            return;
        }
    };

    let started = Instant::now();
    let requested_tokens = work.total_tokens;
    let committed_ids = match state.store.committed_item_ids(&record.request_digest) {
        Ok(ids) => ids,
        Err(error) => {
            fail_job_with_wire_error(
                &state,
                &job_id,
                true,
                WireOperationError::from_stable(
                    StableError::engine_crashed(Some(100)),
                    format!("read committed checkpoint ids: {error}"),
                ),
            );
            return;
        }
    };
    let pending_indices = work
        .ids
        .iter()
        .enumerate()
        .filter_map(|(index, id)| (!committed_ids.contains(id)).then_some(index))
        .collect::<Vec<_>>();
    work.ids = pending_indices
        .iter()
        .map(|index| work.ids[*index].clone())
        .collect();
    work.tokenized.batch.items = pending_indices
        .iter()
        .map(|index| work.tokenized.batch.items[*index].clone())
        .collect();
    work.tokenized.disclosures = pending_indices
        .iter()
        .map(|index| work.tokenized.disclosures[*index].clone())
        .collect();
    work.tokenized.real_token_counts = pending_indices
        .iter()
        .map(|index| work.tokenized.real_token_counts[*index])
        .collect();
    work.tokenized.embedded_texts = pending_indices
        .iter()
        .map(|index| work.tokenized.embedded_texts[*index].clone())
        .collect();
    work.tokenized.submitted_sha256s = pending_indices
        .iter()
        .map(|index| work.tokenized.submitted_sha256s[*index].clone())
        .collect();
    work.total_tokens = work
        .tokenized
        .real_token_counts
        .iter()
        .map(|tokens| u64::from(*tokens))
        .sum();

    if work.ids.is_empty() {
        let summary = json!({
            "job_id": job_id,
            "state": JOB_STATE_DONE,
            "page_count": record.page_count,
            "module_generation": state.module_generation,
        });
        match state.store.finish_job(&job_id, &summary, now_ms()) {
            Ok(()) => {
                state.runtime.admission_telemetry.record_job_completed();
                log_job_done(
                    &work.model.model_id,
                    &job_id,
                    execution_lane(&work.model),
                    requested_tokens,
                    started,
                );
            }
            Err(error) => fail_job_with_wire_error(
                &state,
                &job_id,
                true,
                WireOperationError::from_stable(
                    StableError::engine_crashed(Some(100)),
                    format!("finish resumed checkpoint-only job: {error}"),
                ),
            ),
        }
        return;
    }

    let vectors = match execute_embedding_quanta(
        &state.runtime,
        &work.model,
        work.tokenized.batch.clone(),
        work.total_tokens,
        work.request_bytes,
        None,
        Some(&job_id),
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
        &job_id,
        record.page_count,
    ) {
        Ok(pages) => pages,
        Err(error) => {
            fail_job_with_wire_error(&state, &job_id, false, error);
            return;
        }
    };
    for page in pages {
        if let Err(error) = state.store.commit_job_page(
            &job_id,
            page.page_no,
            &page.bytes,
            &page.checkpoints,
            now_ms(),
        ) {
            fail_job_with_wire_error(
                &state,
                &job_id,
                true,
                WireOperationError::from_stable(
                    StableError::engine_crashed(Some(100)),
                    format!("atomically commit job page: {error}"),
                ),
            );
            return;
        }
    }
    match state.store.finish_job(&job_id, &summary, now_ms()) {
        Ok(()) => {
            state.runtime.admission_telemetry.record_job_completed();
            log_job_done(
                &work.model.model_id,
                &job_id,
                execution_lane(&work.model),
                requested_tokens,
                started,
            );
        }
        Err(error) => fail_job_with_wire_error(
            &state,
            &job_id,
            true,
            WireOperationError::from_stable(
                StableError::engine_crashed(Some(100)),
                format!("finish completed job pages: {error}"),
            ),
        ),
    }
}

async fn execute_embedding_quanta(
    runtime: &RuntimeState,
    model: &EmbeddingModel,
    batch: TokenBatch,
    _total_tokens: u64,
    request_bytes: u64,
    deadline: Option<tokio::time::Instant>,
    job_id: Option<&str>,
) -> Result<Vectors, WireOperationError> {
    let profile = embedding_profile_enabled();
    let started = Instant::now();
    let item_count = batch.items.len();
    let scheduler_quantum_tokens = runtime.jobs.bulk_quantum_tokens.max(1);
    let engine_batch_tokens = scheduler_quantum_tokens.clamp(1, DEFAULT_ENGINE_BATCH_TOKEN_BUDGET);
    let engine_batches = plan_embedding_engine_batches(&batch, engine_batch_tokens);
    let mut scheduler = LaneScheduler::new(SchedulerConfig {
        byte_budget: request_bytes.max(1),
        bulk_quantum_tokens: scheduler_quantum_tokens,
        max_concurrent_workers: 1,
        default_execution_ms: runtime.inline.estimated_execution_ms,
        ..SchedulerConfig::default()
    });
    // One scheduler dispatch represents one bounded engine batch. Accounting is
    // deliberately independent of item lengths so length sorting can improve
    // padding without causing the scheduler to finish before all items run.
    let scheduled_tokens =
        scheduler_quantum_tokens.saturating_mul(engine_batches.len().max(1) as u64);
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

    let mut all_vectors = vec![Vec::new(); batch.items.len()];
    let mut batch_cursor = 0_usize;
    let mut dispatch_count = 0_usize;
    let mut scheduler_wait_ms = 0.0_f64;
    while batch_cursor < engine_batches.len() {
        let wait_started = Instant::now();
        let Some(dispatch) = scheduler.next_dispatch(&SystemClock) else {
            scheduler_wait_ms += wait_started.elapsed().as_secs_f64() * 1_000.0;
            tokio::task::yield_now().await;
            continue;
        };
        scheduler_wait_ms += wait_started.elapsed().as_secs_f64() * 1_000.0;
        ane_lane_timing(
            "bulk_scheduler_dispatch",
            &model.model_id,
            job_id,
            wait_started,
        );
        dispatch_count += 1;
        let indices = &engine_batches[batch_cursor];
        batch_cursor += 1;
        let quantum_tokens = indices
            .iter()
            .map(|&index| batch.items[index].len().max(1) as u64)
            .sum::<u64>();
        let quantum_items = indices
            .iter()
            .map(|&index| batch.items[index].clone())
            .collect::<Vec<_>>();
        let call_started = Instant::now();
        let quantum_item_count = quantum_items.len();
        let mut vectors = execute_embedding(
            runtime,
            model,
            TokenBatch {
                items: quantum_items,
            },
            deadline,
            job_id,
        )
        .await?;
        ane_lane_timing("bulk_engine_call", &model.model_id, job_id, call_started);
        if profile {
            tracing::debug!(
                target: "perf",
                dispatch = dispatch_count,
                items = quantum_item_count,
                tokens = quantum_tokens,
                scheduler_quantum = dispatch.quantum_tokens,
                engine_ms = format_args!("{:.3}", call_started.elapsed().as_secs_f64() * 1_000.0),
                "quanta"
            );
        }
        for (&index, vector) in indices.iter().zip(vectors.drain(..)) {
            all_vectors[index] = vector;
        }
        scheduler.complete_dispatch(&dispatch);
        // Give the async runtime a boundary between bounded engine calls. The
        // scheduler remains the source of class ordering and quantum fairness.
        tokio::task::yield_now().await;
    }
    ane_lane_timing("bulk_execution_total", &model.model_id, job_id, started);
    if profile {
        tracing::debug!(
            target: "perf",
            total_items = item_count,
            dispatches = dispatch_count,
            scheduler_wait_ms = format_args!("{scheduler_wait_ms:.3}"),
            total_ms = format_args!("{:.3}", started.elapsed().as_secs_f64() * 1_000.0),
            "quanta"
        );
    }
    Ok(all_vectors)
}

fn plan_embedding_engine_batches(batch: &TokenBatch, token_budget: u64) -> Vec<Vec<usize>> {
    let mut order = (0..batch.items.len()).collect::<Vec<_>>();
    order.sort_by_key(|&index| batch.items[index].len());
    let token_budget = token_budget.max(1);
    let mut batches = Vec::new();
    let mut start = 0_usize;
    while start < order.len() {
        let mut end = start;
        let mut tokens = 0_u64;
        while end < order.len() {
            let item_tokens = batch.items[order[end]].len().max(1) as u64;
            if end > start
                && (end - start >= MAX_ENGINE_BATCH_ITEMS
                    || tokens.saturating_add(item_tokens) > token_budget)
            {
                break;
            }
            tokens = tokens.saturating_add(item_tokens);
            end += 1;
        }
        batches.push(order[start..end].to_vec());
        start = end;
    }
    batches
}

#[cfg(test)]
fn batch_token_cost(batch: &TokenBatch) -> u64 {
    batch
        .items
        .iter()
        .map(|item| item.len().max(1) as u64)
        .sum::<u64>()
        .max(1)
}

#[allow(clippy::too_many_arguments)]
fn embed_result_pages(
    state: &ModuleState,
    model: &EmbeddingModel,
    ids: Vec<String>,
    vectors: Vectors,
    tokenized: TokenizedBatch,
    job_id: &str,
    first_page_no: u32,
) -> Result<(Value, Vec<PreparedJobPage>), WireOperationError> {
    let dims = vectors.first().map(Vec::len).unwrap_or(0) as u32;
    let equivalent_to = Vec::new();
    let response_vectors = ids
        .into_iter()
        .zip(vectors)
        .zip(&tokenized.embedded_texts)
        .zip(&tokenized.submitted_sha256s)
        .map(|(((id, vector), text), submitted_sha256)| EmbedVector {
            id,
            vector,
            content_sha256: sha256_text(text),
            submitted_sha256: submitted_sha256.clone(),
        })
        .collect::<Vec<_>>();
    let page_ranges = page_ranges(
        &response_vectors,
        &tokenized.real_token_counts,
        state.runtime.jobs.result_page_bytes.max(1),
    );
    let page_count = first_page_no.saturating_add(page_ranges.len() as u32);
    let mut pages = Vec::with_capacity(page_ranges.len());
    for (page_offset, (start, end)) in page_ranges.iter().copied().enumerate() {
        let page_no = first_page_no.saturating_add(page_offset as u32);
        let payload = EmbedResponsePayload {
            vectors: response_vectors[start..end].to_vec(),
            real_token_counts: tokenized.real_token_counts[start..end].to_vec(),
            truncation_disclosures: tokenized.disclosures[start..end].to_vec(),
        };
        let checkpoints = payload
            .vectors
            .iter()
            .map(|vector| {
                Ok(CheckpointItem {
                    item_id: vector.id.clone(),
                    result: serde_json::to_vec(vector).map_err(|error| {
                        WireOperationError::from_stable(
                            StableError::artifact_invalid(),
                            format!("serialize checkpoint item: {error}"),
                        )
                    })?,
                })
            })
            .collect::<Result<Vec<_>, WireOperationError>>()?;
        let envelope = ResponseEnvelope {
            fingerprint: model.fingerprint.clone(),
            table_epoch: FROZEN_TABLE_EPOCH,
            dims,
            provenance: ResponseProvenance {
                engine: model.engine_identity.clone(),
                owned_decode: Default::default(),
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
                Value::String(JOB_STATE_RUNNING.to_string()),
            );
            map.insert("page".to_string(), Value::from(page_no));
            map.insert("page_count".to_string(), Value::from(page_count));
            map.insert(
                "pages_available".to_string(),
                Value::from(page_no.saturating_add(1)),
            );
            map.insert(
                "job_module_generation".to_string(),
                Value::from(state.module_generation),
            );
        }
        pages.push(PreparedJobPage {
            page_no,
            bytes: serde_json::to_vec(&value).map_err(|error| {
                WireOperationError::from_stable(
                    StableError::artifact_invalid(),
                    format!("serialize embed job page: {error}"),
                )
            })?,
            checkpoints,
        });
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
    log_stored_embed_failure(state, job_id, &error.code);
    let _ = state.store.fail_job(
        job_id,
        transient,
        &serde_json::to_value(error).expect("wire error serializes"),
        now_ms(),
    );
    state.runtime.admission_telemetry.record_job_failed();
}

fn log_stored_embed_failure(state: &ModuleState, job_id: &str, code: &str) {
    let record = state.store.get_job(job_id).ok().flatten();
    // Emit a closing perf event only for embedding jobs, including failures
    // before an execution attempt claims the durable job record.
    if record.as_ref().is_some_and(|r| r.kind != "embed.batch") {
        return;
    }
    let model_id = record
        .as_ref()
        .and_then(|r| r.params_json.as_ref())
        .and_then(|p| p["model"].as_str())
        .unwrap_or("unknown");
    let lane = model_slot_snapshot(&state.runtime, model_id)
        .map(|s| execution_lane_for_engine(&s.spec.engine))
        .unwrap_or("unknown");
    let wall_ms = record
        .as_ref()
        .map(|r| now_ms().saturating_sub(r.created_ms))
        .unwrap_or(0);
    log_job_failed(model_id, job_id, lane, code, wall_ms);
}

fn job_status_payload(state: &ModuleState, record: &JobRecord) -> Value {
    let mut payload = json!({
        "module_generation": state.module_generation,
        "job_id": record.job_id,
        "state": record.state,
        "request_key": record.request_key,
        "pages_available": record.page_count,
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

    let requested_page = params.page.unwrap_or(0);
    if requested_page < record.page_count {
        let bytes = match state.store.get_job_page(&record.job_id, requested_page) {
            Ok(Some(bytes)) => bytes,
            Ok(None) => return channel_error("store_failure", "job result page is missing"),
            Err(error) => return channel_error("store_failure", error.to_string()),
        };
        let mut value: Value = match serde_json::from_slice(&bytes) {
            Ok(value) => value,
            Err(error) => return channel_error("store_failure", error.to_string()),
        };
        if let Value::Object(map) = &mut value {
            map.insert("job_id".to_string(), Value::String(record.job_id.clone()));
            map.insert(
                "module_generation".to_string(),
                Value::from(state.module_generation),
            );
            map.insert(
                "job_module_generation".to_string(),
                Value::from(record.module_generation),
            );
            map.insert("state".to_string(), Value::String(record.state.clone()));
            map.insert("page_count".to_string(), Value::from(record.page_count));
            map.insert(
                "pages_available".to_string(),
                Value::from(record.page_count),
            );
        }
        return result_outcome(value);
    }

    match record.state.as_str() {
        JOB_STATE_QUEUED | JOB_STATE_RUNNING => result_outcome(job_status_payload(&state, &record)),
        JOB_STATE_FAILED_TRANSIENT | JOB_STATE_FAILED_PERMANENT => {
            result_outcome(job_status_payload(&state, &record))
        }
        JOB_STATE_DONE => channel_error(
            "invalid_request",
            format!(
                "embed.result page {requested_page} is outside available page_count {}",
                record.page_count
            ),
        ),
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
    use_bulk_quanta: bool,
    budget: InlineWorkBudget,
) -> HandlerOutcome {
    let InlineWorkBudget {
        request_bytes,
        deadline,
        job_id,
        started,
    } = budget;
    let total_tokens = tokenized
        .real_token_counts
        .iter()
        .map(|&tokens| u64::from(tokens))
        .sum::<u64>();
    let observation = state.runtime.certify_observation.then(|| {
        json!({
            "input_ids": tokenized.batch.items,
        })
    });
    let vectors = match if use_bulk_quanta {
        execute_embedding_quanta(
            &state.runtime,
            &model,
            tokenized.batch,
            total_tokens,
            request_bytes,
            deadline,
            Some(&job_id),
        )
        .await
    } else {
        execute_embedding(
            &state.runtime,
            &model,
            tokenized.batch,
            deadline,
            Some(&job_id),
        )
        .await
    } {
        Ok(vectors) => vectors,
        Err(error) => return failed_admitted_outcome(&state, &model, &job_id, started, error),
    };
    if vectors.len() != ids.len() {
        return failed_admitted_outcome(
            &state,
            &model,
            &job_id,
            started,
            WireOperationError::from_stable(
                StableError::engine_crashed(None),
                format!(
                    "engine returned {} vectors for {} requested items",
                    vectors.len(),
                    ids.len()
                ),
            ),
        );
    }
    let reply_started = Instant::now();
    let dims = vectors.first().map(Vec::len).unwrap_or(0) as u32;
    let equivalent_to = Vec::new();
    let response_vectors = ids
        .into_iter()
        .zip(vectors)
        .zip(&tokenized.embedded_texts)
        .zip(&tokenized.submitted_sha256s)
        .map(|(((id, vector), text), submitted_sha256)| EmbedVector {
            id,
            vector,
            content_sha256: sha256_text(text),
            submitted_sha256: submitted_sha256.clone(),
        })
        .collect::<Vec<_>>();
    let payload = EmbedResponsePayload {
        vectors: response_vectors,
        real_token_counts: tokenized.real_token_counts,
        truncation_disclosures: tokenized.disclosures,
    };
    let envelope = ResponseEnvelope {
        fingerprint: model.fingerprint.clone(),
        table_epoch: FROZEN_TABLE_EPOCH,
        dims,
        provenance: ResponseProvenance {
            engine: model.engine_identity.clone(),
            owned_decode: Default::default(),
        },
        module_generation: state.module_generation,
        equivalent_to,
        payload,
    };
    log_job_done(
        &model.model_id,
        &job_id,
        execution_lane(&model),
        total_tokens,
        started,
    );
    let mut response = serde_json::to_value(envelope).expect("embed envelope should serialize");
    attach_certify_observation(&mut response, observation);
    // Building the reply value (row hashes, envelope, JSON value tree) and
    // encoding it to bytes are timed separately: the second is pure codec cost.
    ane_lane_timing("reply_build", &model.model_id, Some(&job_id), reply_started);
    let encode_started = Instant::now();
    let outcome = result_outcome(response);
    ane_lane_timing(
        "reply_encode",
        &model.model_id,
        Some(&job_id),
        encode_started,
    );
    outcome
}

fn apply_owned_tokenizer_policy(model: &EmbeddingModel, tokenized: &mut TokenizedBatch) {
    if model.engine_identity.build_flags.contains_key("profile") {
        return;
    }
    let Some(terminal) = model
        .owned_tokenizer_policy
        .and_then(|policy| policy.terminal_token_id)
    else {
        return;
    };
    for (index, ids) in tokenized.batch.items.iter_mut().enumerate() {
        let already_terminal = ids.last() == Some(&terminal);
        if already_terminal {
            ids.pop();
        }
        ids.truncate(model.tokenizer.max_tokens());
        ids.push(terminal);
        let effective = ids.len().min(u32::MAX as usize) as u32;
        tokenized.real_token_counts[index] = effective;
        tokenized.disclosures[index].effective_tokens = effective;
        if !already_terminal {
            tokenized.disclosures[index].submitted_tokens = tokenized.disclosures[index]
                .submitted_tokens
                .saturating_add(1);
        }
        tokenized.disclosures[index].truncated =
            tokenized.disclosures[index].submitted_tokens > effective;
    }
}

async fn acquire_execution_permit(
    runtime: &RuntimeState,
    deadline: Option<tokio::time::Instant>,
) -> Result<InlineExecutionPermit, WireOperationError> {
    acquire_execution_permit_from(
        runtime.execution.clone(),
        Arc::clone(&runtime.execution_stats),
        deadline,
    )
    .await
}

/// Hands a direct-ANE serving task the module's execution permits, which it
/// takes per inference exchange once it holds the worker stream (see
/// `lane_serving_gate` for the lock order).
#[cfg(unix)]
fn direct_ane_permit_source(runtime: &RuntimeState) -> worker_host::ane_residency::PermitSource {
    use worker_host::ane_residency::{AneResidencyError, ExchangePermit};
    let execution = runtime.execution.clone();
    let stats = Arc::clone(&runtime.execution_stats);
    Arc::new(move |deadline| {
        let execution = execution.clone();
        let stats = Arc::clone(&stats);
        Box::pin(async move {
            acquire_execution_permit_from(execution, stats, deadline)
                .await
                .map(|permit| Box::new(permit) as ExchangePermit)
                .map_err(|error| {
                    if error.code == "deadline_exceeded" {
                        AneResidencyError::DeadlineExceeded
                    } else {
                        AneResidencyError::PermitUnavailable(error.message)
                    }
                })
        })
    })
}

async fn acquire_execution_permit_from(
    execution: Arc<Semaphore>,
    execution_stats: Arc<Mutex<InlineExecutionStats>>,
    deadline: Option<tokio::time::Instant>,
) -> Result<InlineExecutionPermit, WireOperationError> {
    if let Ok(mut stats) = execution_stats.lock() {
        stats.waiters = stats.waiters.saturating_add(1);
    }
    let started = Instant::now();
    let permit_result = match deadline {
        Some(deadline) => {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            tokio::time::timeout(remaining, execution.acquire_owned()).await
        }
        None => Ok(execution.acquire_owned().await),
    };
    let wait_ms = started.elapsed().as_secs_f64() * 1_000.0;
    match permit_result {
        Ok(Ok(permit)) => {
            let mut stats = execution_stats.lock().map_err(|_| {
                WireOperationError::from_stable(
                    StableError::queue_full(Some(100)),
                    "inline execution statistics are unavailable",
                )
            })?;
            stats.waiters = stats.waiters.saturating_sub(1);
            stats.in_flight = stats.in_flight.saturating_add(1);
            if stats.wait_samples_ms.len() == EXECUTION_WAIT_SAMPLE_LIMIT {
                stats.wait_samples_ms.pop_front();
            }
            stats.wait_samples_ms.push_back(wait_ms);
            Ok(InlineExecutionPermit {
                _permit: permit,
                stats: Arc::clone(&execution_stats),
            })
        }
        Err(_) => {
            if let Ok(mut stats) = execution_stats.lock() {
                stats.waiters = stats.waiters.saturating_sub(1);
            }
            Err(WireOperationError::from_stable(
                StableError::deadline_exceeded(),
                "deadline exceeded waiting for inline execution permit",
            ))
        }
        Ok(Err(_)) => {
            if let Ok(mut stats) = execution_stats.lock() {
                stats.waiters = stats.waiters.saturating_sub(1);
            }
            Err(WireOperationError::from_stable(
                StableError::queue_full(Some(100)),
                "inline embedding executor is closed",
            ))
        }
    }
}

fn execution_wait_percentile(stats: &InlineExecutionStats, quantile: f64) -> f64 {
    if stats.wait_samples_ms.is_empty() {
        return 0.0;
    }
    let mut samples = stats.wait_samples_ms.iter().copied().collect::<Vec<_>>();
    samples.sort_by(f64::total_cmp);
    let index = ((samples.len() as f64 * quantile).ceil() as usize)
        .saturating_sub(1)
        .min(samples.len() - 1);
    samples[index]
}

async fn execute_embedding(
    runtime: &RuntimeState,
    model: &EmbeddingModel,
    batch: TokenBatch,
    deadline: Option<tokio::time::Instant>,
    job_id: Option<&str>,
) -> Result<Vectors, WireOperationError> {
    execute_embedding_with_catalog_guard(runtime, model, batch, deadline, job_id, None).await
}

fn is_direct_ane(model: &EmbeddingModel) -> bool {
    #[cfg(unix)]
    {
        matches!(model.backend, EmbedBackend::DirectAne(_))
    }
    #[cfg(not(unix))]
    {
        let _ = model;
        false
    }
}

/// Takes a catalog lane's serving-gate guard (or reuses a self-check's held
/// one) and, for engines that serve one call at a time, the lane execution
/// lock. Both come before the execution permit; see `lane_serving_gate`.
async fn lane_execution_guards(
    runtime: &RuntimeState,
    model: &EmbeddingModel,
    catalog_lane: bool,
    held_guard: Option<HeldLaneGate>,
    deadline: Option<tokio::time::Instant>,
) -> Result<
    (
        Option<LaneServingGuard>,
        Option<tokio::sync::OwnedMutexGuard<()>>,
    ),
    WireOperationError,
> {
    let serving = match held_guard {
        Some(held) => Some(LaneServingGuard::Exclusive(held)),
        None if catalog_lane || lane_pause::uses_ane_model(model) => {
            Some(acquire_lane_serving(runtime, &model.model_id, deadline).await?)
        }
        None => None,
    };
    let execution = if catalog_lane && !is_direct_ane(model) {
        Some(acquire_lane_execution(runtime, &model.model_id, deadline).await?)
    } else {
        None
    };
    Ok((serving, execution))
}

/// Direct-ANE takes its permit per inference exchange, after the worker
/// stream, so a request waiting behind a shape compile holds none. Every
/// other engine takes it here, before the engine call.
async fn upfront_execution_permit(
    runtime: &RuntimeState,
    model: &EmbeddingModel,
    deadline: Option<tokio::time::Instant>,
) -> Result<Option<InlineExecutionPermit>, WireOperationError> {
    if is_direct_ane(model) {
        Ok(None)
    } else {
        acquire_execution_permit(runtime, deadline).await.map(Some)
    }
}

// Dropping the caller's JoinHandle leaves the task reading the worker's reply
// running. Never abort that task after dispatch: it must drain the entire reply
// before releasing the shared connection, or the next request could consume it.
async fn await_worker_completion<T>(
    handle: tokio::task::JoinHandle<Result<T, EngineError>>,
    deadline: Option<tokio::time::Instant>,
    catalog_lane: Option<&str>,
) -> Result<T, WireOperationError> {
    let expired = || {
        WireOperationError::from_stable(
            StableError::deadline_exceeded(),
            "caller deadline expired; worker exchange continues draining",
        )
    };
    let joined = match deadline {
        Some(deadline) => tokio::time::timeout_at(deadline, handle)
            .await
            .map_err(|_| expired())?,
        None => handle.await,
    };
    if deadline.is_some_and(|deadline| tokio::time::Instant::now() >= deadline) {
        return Err(expired());
    }
    joined
        .map_err(|error| {
            WireOperationError::from_stable(
                StableError::engine_crashed(Some(100)),
                format!("worker owner join failed: {error}"),
            )
        })?
        .map_err(|error| worker_error_to_wire(error, catalog_lane))
}

fn worker_dispatch_deadline(deadline: Option<tokio::time::Instant>) -> Result<(), EngineError> {
    if deadline.is_some_and(|deadline| tokio::time::Instant::now() >= deadline) {
        return Err(EngineError {
            stage: EngineErrorStage::Inference,
            risk_class: synapse_core::EngineRiskClass::AbortSafe,
            message: "caller deadline expired before worker dispatch".into(),
            retry_after_ms: None,
            safe_to_retry_same_request: true,
        });
    }
    Ok(())
}

async fn execute_embedding_with_catalog_guard(
    runtime: &RuntimeState,
    model: &EmbeddingModel,
    batch: TokenBatch,
    deadline: Option<tokio::time::Instant>,
    job_id: Option<&str>,
    held_guard: Option<HeldLaneGate>,
) -> Result<Vectors, WireOperationError> {
    let profile = embedding_profile_enabled();
    let tokens = batch
        .items
        .iter()
        .map(|item| item.len().max(1) as u64)
        .sum::<u64>();
    let _activity = runtime.activity_telemetry.begin(&model.model_id);
    let lane_started = Instant::now();
    let catalog_lane = resolved_catalog_lane(runtime, &model.model_id).is_some();
    let (catalog_guard, lane_execution) =
        lane_execution_guards(runtime, model, catalog_lane, held_guard, deadline).await?;
    let fault_lane = catalog_lane.then(|| model.model_id.clone());
    ane_lane_timing("execution_lane_wait", &model.model_id, job_id, lane_started);
    let permit_started = Instant::now();
    let permit = upfront_execution_permit(runtime, model, deadline).await?;
    ane_lane_timing(
        "execution_permit_wait",
        &model.model_id,
        job_id,
        permit_started,
    );
    let result = match &model.backend {
        #[cfg(feature = "test-support")]
        EmbedBackend::TestDeterministic(engine) => {
            let engine = Arc::clone(engine);
            let loaded = model.loaded_model.clone();
            let model_id = model.model_id.clone();
            let job_id = job_id.map(str::to_owned);
            tokio::task::spawn_blocking(move || {
                let _permit = permit;
                let _catalog_guard = catalog_guard;
                let _lane_execution = lane_execution;
                // The test engine's own compute, so profiles can separate it
                // from the module's dispatch overhead around each engine call.
                let compute_started = Instant::now();
                let result = engine.embed_batch(&loaded, batch);
                ane_lane_timing(
                    "test_engine_compute",
                    &model_id,
                    job_id.as_deref(),
                    compute_started,
                );
                result
            })
            .await
            .map_err(|error| artifact_invalid_error(error.to_string()))?
            .map_err(engine_error_to_wire)
        }
        EmbedBackend::Owned(engine) => {
            let engine = Arc::clone(engine);
            let loaded_model = model.loaded_model.clone();
            let submitted_at = Instant::now();
            tokio::task::spawn_blocking(move || {
                let entered_at = Instant::now();
                if profile {
                    tracing::debug!(
                        target: "perf",
                        backend = "owned",
                        wait_ms = format_args!("{:.3}", submitted_at.elapsed().as_secs_f64() * 1_000.0),
                        "spawn_entry"
                    );
                }
                let _permit = permit;
                let _catalog_guard = catalog_guard;
                let _lane_execution = lane_execution;
                if let Some(id) = fault_lane.as_deref() { catalog_engine_fault(id,"serve")?; }
                let mutex_started = Instant::now();
                let engine = engine.lock().map_err(|_| EngineError {
                    stage: EngineErrorStage::Inference,
                    risk_class: synapse_core::EngineRiskClass::AbortSafe,
                    message: "owned-metal engine mutex was poisoned during inference".to_string(),
                    retry_after_ms: Some(100),
                    safe_to_retry_same_request: true,
                })?;
                if profile {
                    tracing::debug!(
                        target: "perf",
                        backend = "owned",
                        wait_ms = format_args!("{:.3}", mutex_started.elapsed().as_secs_f64() * 1_000.0),
                        "mutex_acquired"
                    );
                }
                let inference_started = Instant::now();
                let result = engine.embed_batch(&loaded_model, batch);
                if profile {
                    tracing::debug!(
                        target: "perf",
                        backend = "owned",
                        inference_ms = format_args!("{:.3}", inference_started.elapsed().as_secs_f64() * 1_000.0),
                        worker_ms = format_args!("{:.3}", entered_at.elapsed().as_secs_f64() * 1_000.0),
                        "engine_return"
                    );
                }
                result
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
        EmbedBackend::DirectAne(engine) => {
            if let Some(id) = fault_lane.as_deref() {
                catalog_engine_fault(id, "serve").map_err(engine_error_to_wire)?;
            }
            let roundtrip_started = Instant::now();
            let _ = (permit, lane_execution);
            let result = if packed_lane_enabled(runtime, model) {
                engine
                    .serving
                    .infer_packed_permitted(
                        batch.items,
                        deadline,
                        (catalog_guard, _activity),
                        Some(direct_ane_permit_source(runtime)),
                    )
                    .await
            } else {
                engine
                    .infer_permitted(
                        batch.items,
                        false,
                        deadline,
                        (catalog_guard, _activity),
                        Some(direct_ane_permit_source(runtime)),
                    )
                    .await
            };
            ane_lane_timing(
                "direct_ane_roundtrip",
                &model.model_id,
                job_id,
                roundtrip_started,
            );
            result.map_err(|error| ane_lane_error_to_wire(error, &model.model_id))
        }
        EmbedBackend::Worker(engine) => {
            let engine = Arc::clone(engine);
            let loaded_model = model.loaded_model.clone();
            let submitted_at = Instant::now();
            let job_id = job_id.map(str::to_owned);
            let handle = tokio::task::spawn_blocking(move || {
                let _activity = _activity;
                let entered_at = Instant::now();
                if profile {
                    tracing::debug!(
                        target: "perf",
                        backend = "worker",
                        wait_ms = format_args!("{:.3}", submitted_at.elapsed().as_secs_f64() * 1_000.0),
                        "spawn_entry"
                    );
                }
                let _permit = permit;
                let _catalog_guard = catalog_guard;
                let _lane_execution = lane_execution;
                if let Some(id) = fault_lane.as_deref() {
                    catalog_engine_fault(id, "serve")?;
                }
                let mutex_started = Instant::now();
                let engine = engine.lock().map_err(|_| EngineError {
                    stage: EngineErrorStage::Inference,
                    risk_class: synapse_core::EngineRiskClass::AbortCapable,
                    message: "worker engine mutex was poisoned during inference".to_string(),
                    retry_after_ms: Some(100),
                    safe_to_retry_same_request: true,
                })?;
                if profile {
                    tracing::debug!(
                        target: "perf",
                        backend = "worker",
                        wait_ms = format_args!("{:.3}", mutex_started.elapsed().as_secs_f64() * 1_000.0),
                        "mutex_acquired"
                    );
                }
                worker_dispatch_deadline(deadline)?;
                let inference_started = Instant::now();
                let result = engine.embed_batch_with_job(&loaded_model, batch, job_id.as_deref());
                if profile {
                    tracing::debug!(
                        target: "perf",
                        backend = "worker",
                        inference_ms = format_args!("{:.3}", inference_started.elapsed().as_secs_f64() * 1_000.0),
                        worker_ms = format_args!("{:.3}", entered_at.elapsed().as_secs_f64() * 1_000.0),
                        "engine_return"
                    );
                }
                result
            });
            await_worker_completion(
                handle,
                deadline,
                catalog_lane.then_some(model.model_id.as_str()),
            )
            .await
        }
    };
    if result.is_ok() {
        runtime.activity_telemetry.record_completed_tokens(tokens);
    }
    result
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RerankReadout {
    Legacy,
    CatalogLogits,
    CatalogProbability,
}

impl RerankReadout {
    fn select(catalog_owned: bool, identity: &EngineIdentity) -> Self {
        if !catalog_owned {
            return Self::Legacy;
        }
        // Native Metal returns the classifier's raw logit and needs sigmoid.
        // Profile-loaded Metal, ANE, CUDA and Vulkan already return probabilities.
        if identity.engine == "owned-metal" && !identity.build_flags.contains_key("profile") {
            Self::CatalogLogits
        } else {
            Self::CatalogProbability
        }
    }

    fn apply(self, mut scores: synapse_core::RerankScores) -> synapse_core::RerankScores {
        if self == Self::CatalogLogits {
            for score in &mut scores.scores {
                *score = 1.0 / (1.0 + (-*score).exp());
            }
        }
        scores
    }
}

async fn execute_rerank(
    runtime: &RuntimeState,
    model: &EmbeddingModel,
    request: RerankRequest,
    owned_pairs: Option<Vec<Vec<u32>>>,
    deadline: Option<tokio::time::Instant>,
    job_id: Option<&str>,
) -> Result<synapse_core::RerankScores, WireOperationError> {
    execute_rerank_with_catalog_guard(runtime, model, request, owned_pairs, deadline, job_id, None)
        .await
}

async fn execute_rerank_with_catalog_guard(
    runtime: &RuntimeState,
    model: &EmbeddingModel,
    request: RerankRequest,
    owned_pairs: Option<Vec<Vec<u32>>>,
    deadline: Option<tokio::time::Instant>,
    job_id: Option<&str>,
    held_guard: Option<HeldLaneGate>,
) -> Result<synapse_core::RerankScores, WireOperationError> {
    let tokens = request
        .candidates
        .iter()
        .map(|candidate| request.query.len().saturating_add(candidate.len()) as u64)
        .sum();
    let _activity = runtime.activity_telemetry.begin(&model.model_id);
    let catalog_lane = resolved_catalog_lane(runtime, &model.model_id).is_some();
    let (catalog_guard, lane_execution) =
        lane_execution_guards(runtime, model, catalog_lane, held_guard, deadline).await?;
    let fault_lane = catalog_lane.then(|| model.model_id.clone());
    let permit = upfront_execution_permit(runtime, model, deadline).await?;
    let result = match &model.backend {
        #[cfg(feature = "test-support")]
        EmbedBackend::TestDeterministic(engine) => {
            let _permit = permit;
            let _catalog_guard = catalog_guard;
            let _lane_execution = lane_execution;
            synapse_core::RerankEngine::rerank(&**engine, &model.loaded_model, request)
                .map_err(engine_error_to_wire)
        }
        #[cfg(unix)]
        EmbedBackend::DirectAne(engine) => {
            if let Some(id) = fault_lane.as_deref() {
                catalog_engine_fault(id, "serve").map_err(engine_error_to_wire)?;
            }
            let pairs = owned_pairs.ok_or_else(|| {
                artifact_invalid_error("direct-ANE rerank requires composed pairs")
            })?;
            let _ = (permit, lane_execution);
            let values = engine
                .infer_permitted(
                    pairs,
                    true,
                    deadline,
                    (catalog_guard, _activity),
                    Some(direct_ane_permit_source(runtime)),
                )
                .await
                .map_err(|error| ane_lane_error_to_wire(error, &model.model_id))?;
            Ok(synapse_core::RerankScores {
                scores: values.into_iter().map(|v| v[0]).collect(),
            })
        }
        EmbedBackend::Owned(engine) => {
            let pairs = owned_pairs.ok_or_else(|| {
                WireOperationError::from_stable(
                    StableError::artifact_invalid(),
                    format!(
                        "model '{}' has no module-framed rerank token IDs",
                        model.model_id
                    ),
                )
            })?;
            let engine = Arc::clone(engine);
            let loaded_model = model.loaded_model.clone();
            tokio::task::spawn_blocking(move || {
                let _permit = permit;
                let _catalog_guard = catalog_guard;
                let _lane_execution = lane_execution;
                if let Some(id) = fault_lane.as_deref() {
                    catalog_engine_fault(id, "serve")?;
                }
                let engine = engine.lock().map_err(|_| EngineError {
                    stage: EngineErrorStage::Inference,
                    risk_class: synapse_core::EngineRiskClass::AbortCapable,
                    message: "owned-metal engine mutex was poisoned during rerank".to_string(),
                    retry_after_ms: Some(100),
                    safe_to_retry_same_request: true,
                })?;
                engine.rerank_pairs(&loaded_model, pairs)
            })
            .await
            .map_err(|error| {
                WireOperationError::from_stable(
                    StableError::engine_crashed(Some(100)),
                    format!("owned-metal rerank join failed: {error}"),
                )
            })?
            .map_err(engine_error_to_wire)
        }
        EmbedBackend::Worker(engine) => {
            let engine = Arc::clone(engine);
            let loaded_model = model.loaded_model.clone();
            let job_id = job_id.map(str::to_owned);
            let handle = tokio::task::spawn_blocking(move || {
                let _activity = _activity;
                let _permit = permit;
                let _catalog_guard = catalog_guard;
                let _lane_execution = lane_execution;
                if let Some(id) = fault_lane.as_deref() {
                    catalog_engine_fault(id, "serve")?;
                }
                let engine = engine.lock().map_err(|_| EngineError {
                    stage: EngineErrorStage::Inference,
                    risk_class: synapse_core::EngineRiskClass::AbortCapable,
                    message: "worker engine mutex was poisoned during rerank".to_string(),
                    retry_after_ms: Some(100),
                    safe_to_retry_same_request: true,
                })?;
                let request = if let Some(pairs) = owned_pairs {
                    RerankRequest {
                        query: Vec::new(),
                        candidates: pairs,
                    }
                } else {
                    request
                };
                worker_dispatch_deadline(deadline)?;
                engine.rerank_with_job(&loaded_model, request, job_id.as_deref())
            });
            await_worker_completion(
                handle,
                deadline,
                catalog_lane.then_some(model.model_id.as_str()),
            )
            .await
        }
    };
    if result.is_ok() {
        runtime.activity_telemetry.record_completed_tokens(tokens);
    }
    // Serving and certification share this boundary. A lane resolved from the
    // release catalog must return probabilities; an ordinary preload keeps its
    // engine's bytes. Native catalog lanes have no profile flag, so that flag
    // cannot distinguish catalog serving from legacy serving.
    result.map(|scores| RerankReadout::select(catalog_lane, &model.engine_identity).apply(scores))
}

fn catalog_sequence_ceiling(sequences: &[Vec<u32>]) -> Result<(), WireOperationError> {
    if let Some((index, ids)) = sequences
        .iter()
        .enumerate()
        .find(|(_, ids)| ids.len() > 8192)
    {
        return Err(WireOperationError::from_stable(
            StableError::sequence_too_long(ids.len(), 8192, Some(&index.to_string())),
            "composed input exceeds the catalog lane's context limit",
        ));
    }
    Ok(())
}

// Catalog ownership comes from reserved IDs, not the optional profile flag:
// native Metal lanes have no profile flag and retain their pinned identities.
// Count the full composed input before loading a worker or admitting any work.
fn preflight_catalog_sequences(
    state: &ModuleState,
    model_id: Option<&str>,
    required_fingerprint: Option<&str>,
    target_fingerprint: Option<&str>,
    query: Option<&str>,
    texts: &[&str],
) -> Result<Option<Vec<Vec<u32>>>, WireOperationError> {
    if model_id.is_some_and(|id| !state.runtime.release_catalog.is_reserved_id(id)) {
        return Ok(None);
    }
    let task = if query.is_some() {
        ModelTask::Rerank
    } else {
        ModelTask::Embed
    };
    let (entry, backend) = select_catalog_lane(
        state,
        model_id,
        task,
        required_fingerprint,
        target_fingerprint,
    )?;
    let lane = catalog::lane_id(&entry.id, &backend.backend);
    let loaded = model_slot_snapshot(&state.runtime, &lane).and_then(|slot| slot.loaded);
    let cold_tokenizer;
    let tokenizer = if let Some(model) = loaded.as_ref() {
        &model.tokenizer
    } else {
        let files = entry.backend_files(&backend.backend);
        let file = files["tokenizer"];
        let path = state.model_cache.blob_path(&file.sha256);
        let install = current_catalog_install(state, &entry, &backend.backend)?.ok_or_else(||
            catalog_wire_error("model_not_installed", json!({"catalog_id":entry.id,"lane_id":lane,"download_op":"models.download","download_job_id":null}), "catalog installation disappeared"))?;
        use store::CatalogBlobCache;
        let size = CatalogCache(&state.model_cache)
            .blob_size(&file.sha256)
            .map_err(catalog_store_error)?;
        let digest = verified_catalog_file_digest(
            &state.store,
            &install,
            &file.path,
            &file.sha256,
            &path,
            |path| sha256_file(path).ok(),
        )?;
        if size != Some(file.size_bytes) || digest.as_deref() != Some(file.sha256.as_str()) {
            let error = catalog_artifact_error(file, digest, size);
            state
                .store
                .quarantine_catalog_blob(
                    &file.sha256,
                    &serde_json::to_value(&error).expect("wire error"),
                    &CatalogCache(&state.model_cache),
                    now_ms(),
                )
                .map_err(catalog_store_error)?;
            return Err(error);
        }
        cold_tokenizer = SanitizedTokenizer::from_file(
            path,
            TokenizerConfig {
                max_tokens: usize::MAX,
            },
        )
        .map_err(|error| artifact_invalid_error(error.to_string()))?;
        &cold_tokenizer
    };
    let profile = backend
        .profile
        .as_deref()
        .map(CatalogProfile::load)
        .transpose()
        .map_err(|error| artifact_invalid_error(error.to_string()))?;
    let composed = if let Some(query) = query {
        tokenize_rerank_pairs(tokenizer, profile.as_ref(), query, texts)?
    } else {
        let mut sequences = texts
            .iter()
            .map(|text| {
                tokenizer
                    .tokenizer()
                    .encode(*text, true)
                    .map(|encoding| encoding.get_ids().to_vec())
                    .map_err(|error| artifact_invalid_error(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if backend.family.as_deref() == Some("qwen3-0.6b") {
            let embed_profile = CatalogProfile::load("qwen3-embedding-0.6b.owned-metal")
                .map_err(|error| artifact_invalid_error(error.to_string()))?;
            let terminal = embed_profile.model()["grammar"]["terminal_tokens"][0]["id"]
                .as_u64()
                .expect("manifest EOS") as u32;
            for ids in &mut sequences {
                if ids.last() != Some(&terminal) {
                    ids.push(terminal);
                }
            }
        }
        sequences
    };
    catalog_sequence_ceiling(&composed)?;
    Ok(Some(composed))
}

fn catalog_embed_batch(texts: &[&str], sequences: Vec<Vec<u32>>) -> TokenizedBatch {
    let counts = sequences
        .iter()
        .map(|ids| ids.len() as u32)
        .collect::<Vec<_>>();
    TokenizedBatch {
        batch: TokenBatch { items: sequences },
        disclosures: counts
            .iter()
            .map(|count| TruncationDisclosure {
                submitted_tokens: *count,
                effective_tokens: *count,
                truncated: false,
            })
            .collect(),
        real_token_counts: counts,
        embedded_texts: texts.iter().map(|text| (*text).to_string()).collect(),
        submitted_sha256s: texts
            .iter()
            .map(|text| sha256_hex(text.as_bytes()))
            .collect(),
    }
}

fn compose_catalog_embed(
    model: &EmbeddingModel,
    tokenized: &mut TokenizedBatch,
) -> Result<(), WireOperationError> {
    let Some(id) = release_profile(&model.engine_identity) else {
        return Ok(());
    };
    let catalog =
        CatalogProfile::load(id).map_err(|error| artifact_invalid_error(error.to_string()))?;
    if catalog.slug == "qwen3-embedding-0.6b" {
        let terminal = catalog.model()["grammar"]["terminal_tokens"][0]["id"]
            .as_u64()
            .expect("manifest EOS") as u32;
        for (index, ids) in tokenized.batch.items.iter_mut().enumerate() {
            if ids.last() != Some(&terminal) {
                ids.push(terminal);
            }
            let count = ids.len().min(u32::MAX as usize) as u32;
            tokenized.real_token_counts[index] = count;
            tokenized.disclosures[index].submitted_tokens = count;
            tokenized.disclosures[index].effective_tokens = count;
        }
    }
    catalog_sequence_ceiling(&tokenized.batch.items)
}

fn owned_rerank_pairs(
    model: &EmbeddingModel,
    query: &str,
    candidates: &[String],
) -> Result<Option<Vec<Vec<u32>>>, WireOperationError> {
    let catalog = release_profile(&model.engine_identity)
        .map(CatalogProfile::load)
        .transpose()
        .map_err(|error| artifact_invalid_error(error.to_string()))?;
    if !matches!(&model.backend, EmbedBackend::Owned(_)) && catalog.is_none() {
        return Ok(None);
    }
    let candidates = candidates.iter().map(String::as_str).collect::<Vec<_>>();
    let pairs = tokenize_rerank_pairs(&model.tokenizer, catalog.as_ref(), query, &candidates)?;
    if catalog.is_some() {
        catalog_sequence_ceiling(&pairs)?;
    }
    Ok(Some(pairs))
}

fn tokenize_rerank_pairs(
    tokenizer: &SanitizedTokenizer,
    catalog: Option<&CatalogProfile>,
    query: &str,
    candidates: &[&str],
) -> Result<Vec<Vec<u32>>, WireOperationError> {
    if let Some(catalog) = catalog {
        if catalog.model()["grammar"]["kind"] == "template" {
            let template = &catalog.model()["grammar"]["template"];
            let encode = |text: &str| {
                tokenizer
                    .tokenizer()
                    .encode(text, false)
                    .map(|encoding| encoding.get_ids().to_vec())
                    .map_err(|error| artifact_invalid_error(error.to_string()))
            };
            let prefix = encode(template["prefix"].as_str().expect("template prefix"))?;
            let suffix = encode(template["suffix"].as_str().expect("template suffix"))?;
            let mut pairs = Vec::with_capacity(candidates.len());
            for candidate in candidates {
                let format = template["body_format"].as_str().expect("template body");
                let (head, tail) = format.split_once("{query}").expect("query placeholder");
                let (middle, end) = tail.split_once("{doc}").expect("doc placeholder");
                let head = head.replace(
                    "{instruction}",
                    template["instruction"]
                        .as_str()
                        .expect("template instruction"),
                );
                let body = format!("{head}{query}{middle}{candidate}{end}");
                let mut ids = prefix.clone();
                ids.extend(encode(&body)?);
                ids.extend_from_slice(&suffix);
                pairs.push(ids);
            }
            return Ok(pairs);
        }
    }
    let inputs = candidates
        .iter()
        .map(|candidate| (query, *candidate))
        .collect::<Vec<_>>();
    let encodings = tokenizer
        .tokenizer()
        .encode_batch(inputs, true)
        .map_err(|error| {
            WireOperationError::from_stable(
                StableError::artifact_invalid(),
                format!("encode owned-metal rerank pairs: {error}"),
            )
        })?;
    let pairs = encodings
        .into_iter()
        .map(|encoding| encoding.get_ids().to_vec())
        .collect::<Vec<_>>();
    if pairs.iter().any(Vec::is_empty) {
        return Err(WireOperationError::from_stable(
            StableError::artifact_invalid(),
            "owned-metal rerank pair tokenization produced an empty sequence",
        ));
    }
    Ok(pairs)
}

async fn execute_generate(
    runtime: &RuntimeState,
    model: &EmbeddingModel,
    request: GenerateRequest,
    deadline: Option<tokio::time::Instant>,
    job_id: Option<&str>,
) -> Result<GenerateOutput, WireOperationError> {
    let _activity = runtime.activity_telemetry.begin(&model.model_id);
    let catalog_lane = resolved_catalog_lane(runtime, &model.model_id).is_some();
    // Serving-gate read guard, then the lane execution lock, then the
    // execution permit (see `lane_serving_gate`).
    let (catalog_guard, lane_execution) =
        lane_execution_guards(runtime, model, catalog_lane, None, deadline).await?;
    let fault_lane = catalog_lane.then(|| model.model_id.clone());
    let permit = acquire_execution_permit(runtime, deadline).await?;
    let result = match &model.backend {
        #[cfg(unix)]
        EmbedBackend::DirectAne(_) => Err(artifact_invalid_error(
            "direct-ANE does not support generation",
        )),
        #[cfg(feature = "test-support")]
        EmbedBackend::TestDeterministic(_) => Err(artifact_invalid_error(
            "test-deterministic does not generate",
        )),
        EmbedBackend::Owned(_) => Err(WireOperationError::from_stable(
            StableError::artifact_invalid(),
            format!(
                "model '{}' does not support the legacy microllm.oneshot path",
                model.model_id
            ),
        )),
        EmbedBackend::Worker(engine) => {
            let engine = Arc::clone(engine);
            let loaded_model = model.loaded_model.clone();
            let job_id = job_id.map(str::to_owned);
            let handle = tokio::task::spawn_blocking(move || {
                let _activity = _activity;
                let _permit = permit;
                let _catalog_guard = catalog_guard;
                let _lane_execution = lane_execution;
                if let Some(id) = fault_lane.as_deref() {
                    catalog_engine_fault(id, "serve")?;
                }
                let engine = engine.lock().map_err(|_| EngineError {
                    stage: EngineErrorStage::Inference,
                    risk_class: synapse_core::EngineRiskClass::AbortCapable,
                    message: "worker engine mutex was poisoned during generate".to_string(),
                    retry_after_ms: Some(100),
                    safe_to_retry_same_request: true,
                })?;
                worker_dispatch_deadline(deadline)?;
                engine.generate_with_job(&loaded_model, request, job_id.as_deref())
            });
            await_worker_completion(
                handle,
                deadline,
                catalog_lane.then_some(model.model_id.as_str()),
            )
            .await
        }
    };
    if let Ok(output) = &result {
        runtime
            .activity_telemetry
            .record_completed_tokens((output.n_prompt as u64).saturating_add(output.n_gen as u64));
    }
    result
}

// The generic host uses WorkerCrash only for quarantine on serving calls.
// Ordinary transport failures use the operation's inference/load stage.
// Catalog serving retries after a cool-down; it does not require the caller to
// run a legacy worker probe and record approval before serving can resume.
fn worker_error_to_wire(error: EngineError, catalog_lane: Option<&str>) -> WireOperationError {
    if let Some(lane_id) = catalog_lane.filter(|_| error.stage == EngineErrorStage::WorkerCrash) {
        let retry_after_ms = error
            .retry_after_ms
            .unwrap_or(worker_host::CATALOG_QUARANTINE_COOLDOWN.as_millis() as u64);
        let mut wire = WireOperationError::from_stable(
            StableError::engine_crashed(Some(retry_after_ms)),
            error.message,
        );
        wire.details = Some(json!({ "lane_id": lane_id, "quarantined": true }));
        return wire;
    }
    engine_error_to_wire(error)
}

fn engine_error_to_wire(error: EngineError) -> WireOperationError {
    if error.stage == EngineErrorStage::Load {
        return artifact_invalid_error(error.message);
    }
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

// The alias table was removed, but consumers may have persisted the last epoch they saw.
const FROZEN_TABLE_EPOCH: u64 = 2;

fn check_fingerprint_constraints(
    model: &EmbeddingModel,
    target_fingerprint: Option<&str>,
    required_fingerprint: Option<&str>,
    required_epoch: Option<u64>,
) -> Result<(), WireOperationError> {
    if let Some(required_epoch) = required_epoch.filter(|epoch| *epoch > FROZEN_TABLE_EPOCH) {
        return Err(WireOperationError::from_stable(
            StableError::migration_required(),
            format!("request requires alias table epoch {required_epoch}, but module is at epoch {FROZEN_TABLE_EPOCH}"),
        ));
    }
    if let Some(requested) = required_fingerprint.or(target_fingerprint) {
        if requested != model.fingerprint.0 {
            return Err(WireOperationError::from_stable(
                StableError::substitution_rejected(),
                format!(
                    "requested fingerprint {requested} does not match loaded model fingerprint {}",
                    model.fingerprint.0
                ),
            ));
        }
    }
    Ok(())
}

async fn ensure_profile_request_certified(
    state: Arc<ModuleState>,
    model: &EmbeddingModel,
    deadline: Option<tokio::time::Instant>,
) -> Result<(), WireOperationError> {
    if release_profile(&model.engine_identity).is_some() {
        ensure_profile_preload_ready(state.clone(), &model.model_id, deadline).await?;
        check_profile_preload_certified(&state, model)?;
    }
    Ok(())
}

fn check_profile_preload_certified(
    state: &ModuleState,
    model: &EmbeddingModel,
) -> Result<(), WireOperationError> {
    if let Some(profile) = release_profile(&model.engine_identity) {
        if resolved_catalog_lane(&state.runtime, &model.model_id).is_none() {
            let (id, _) = profile_preload_check_key(state, model, profile)?;
            let status = profile_preload_check_status(state, &id)?;
            if status != "passed" {
                return Err(profile_preload_check_error(profile, &status));
            }
        }
    }
    Ok(())
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

fn registered_model_slots(state: &ModuleState) -> Vec<ModelSlotSnapshot> {
    state
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
        .unwrap_or_default()
        .into_iter()
        .filter(|slot| {
            resolved_catalog_lane(&state.runtime, &slot.spec.model_id).is_none_or(|(e, b)| {
                current_catalog_install(state, e, &b.backend)
                    .ok()
                    .flatten()
                    .is_some()
            })
        })
        .collect()
}

/// Report admission capacity plus process-local `{ refusals: { reason: { count,
/// last_at_ms } }, jobs_minted, jobs_completed, jobs_failed, jobs_inherited, jobs_open }`
/// counters. Refusal keys use stable identifiers from
/// the admission protocol so clients can rely on the same keys across releases.
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
    let execution_stats = match state.runtime.execution_stats.lock() {
        Ok(stats) => stats,
        Err(_) => {
            return result_outcome(error_payload(
                &state,
                WireOperationError::from_stable(
                    StableError::queue_full(Some(100)),
                    "inline execution statistics are unavailable",
                ),
            ))
        }
    };
    let execution_waiters = execution_stats.waiters;
    let execution_in_flight = execution_stats.in_flight;
    let execution_wait_p50_ms = execution_wait_percentile(&execution_stats, 0.50);
    let execution_wait_p95_ms = execution_wait_percentile(&execution_stats, 0.95);
    let mut lanes = state
        .runtime
        .loaded_models()
        .into_iter()
        .map(|model| {
            json!({
                "model_id": model.model_id,
                "fingerprint": model.fingerprint,
                "meeting_deadlines": predicted_start_delay_ms <= state.runtime.inline.max_queue_ms,
                "p50_start_delay_ms": predicted_start_delay_ms,
                "execution_waiters": execution_waiters,
                "inline_in_flight_executions": execution_in_flight,
                "execution_wait_p50_ms": execution_wait_p50_ms,
                "execution_wait_p95_ms": execution_wait_p95_ms,
                "certified": false,
            })
        })
        .collect::<Vec<_>>();
    for row in &mut lanes {
        let id = row["model_id"].as_str().unwrap_or("").to_string();
        if resolved_catalog_lane(&state.runtime, &id).is_some() {
            if let Err(e) = catalog_list_row(&state, row) {
                return result_outcome(error_payload(&state, e));
            }
            row["certification_required"] = json!(false);
            row["certification_status"] = json!("not_required");
        }
    }
    let telemetry = state.runtime.admission_telemetry.snapshot();
    result_outcome(json!({
        "module_generation": state.module_generation,
        "machine_profile_hash": state.machine_profile_hash,
        "current_knob": state.runtime.knob,
        "inline_in_flight_bytes": scheduler.in_flight_bytes,
        "execution_waiters": execution_waiters,
        "inline_in_flight_executions": execution_in_flight,
        "execution_wait_p50_ms": execution_wait_p50_ms,
        "execution_wait_p95_ms": execution_wait_p95_ms,
        "refusals": telemetry.refusals,
        "jobs_minted": telemetry.jobs_minted,
        "jobs_completed": telemetry.jobs_completed,
        "jobs_failed": telemetry.jobs_failed,
        "jobs_inherited": telemetry.jobs_inherited,
        "jobs_open": telemetry.jobs_open,
        "lanes": lanes,
        "catalog_lanes": registered_model_slots(&state).len(),
        "certified_lanes": 0,
    }))
}

fn worker_health_for_model(model: &EmbeddingModel) -> Option<worker_host::WorkerHostHealth> {
    match &model.backend {
        EmbedBackend::Worker(engine) => engine
            .lock()
            .ok()
            .and_then(|engine| engine.health_snapshot().ok()),
        #[cfg(unix)]
        EmbedBackend::DirectAne(_) => None,
        #[cfg(feature = "test-support")]
        EmbedBackend::TestDeterministic(_) => None,
        EmbedBackend::Owned(_) => None,
    }
}

fn module_health(state: &ModuleState) -> ModuleHealth {
    let lanes = state
        .runtime
        .loaded_models()
        .into_iter()
        .map(|model| LaneHealth {
            model_id: model.model_id.clone(),
            fingerprint: model.fingerprint.clone(),
            worker: worker_health_for_model(&model),
        })
        .collect::<Vec<_>>();
    let storage = state.store.storage_health_inputs().unwrap_or_else(|error| {
        tracing::warn!(target: "cert", error = %error, "failed to read profile health state");
        StorageHealthInputs {
            previous_revisioned_machine_profile_hash: None,
            current_revisioned_machine_profile_hash: Some(
                state.revisioned_machine_profile_hash.clone(),
            ),
            profile_activation_epoch: Some(state.profile_activation_epoch),
            last_rotation_at_ms: None,
            last_rotation_reason: Some("unknown_previous_snapshot".to_string()),
            rotation_event_count: 0,
        }
    });
    ModuleHealth {
        status: "ok".to_string(),
        module_generation: state.module_generation,
        loaded_models: state.runtime.loaded_model_count(),
        machine_profile_hash: state.legacy_machine_profile_hash.clone(),
        lanes,
        previous_revisioned_machine_profile_hash: storage.previous_revisioned_machine_profile_hash,
        current_revisioned_machine_profile_hash: storage
            .current_revisioned_machine_profile_hash
            .unwrap_or_else(|| state.revisioned_machine_profile_hash.clone()),
        profile_activation_epoch: storage
            .profile_activation_epoch
            .unwrap_or(state.profile_activation_epoch),
        last_rotation_at_ms: storage.last_rotation_at_ms,
        last_rotation_reason: storage.last_rotation_reason,
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
    let _catalog_disk = state
        .runtime
        .catalog_disk
        .lock()
        .expect("catalog disk lock");
    let roots = match catalog_cache_roots(&state) {
        Ok(r) => r,
        Err(e) => return channel_error("store_failure", e.to_string()),
    };
    let _catalog_roots = match roots
        .iter()
        .map(|d| state.model_cache.acquire_read(d))
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(r) => r,
        Err(e) => return result_outcome(error_payload(&state, cache_error_to_wire(e))),
    };
    let result: Result<Vec<CacheGcOutcome>, ModelCacheError> = (|| {
        let sweep_started = SystemTime::now();
        // Partial downloads left by a crashed ingest; files younger than the
        // 24-hour floor may belong to an ingest still in progress and stay.
        state
            .model_cache
            .cleanup_abandoned_ingest_temps(sweep_started)?;
        if let Some(digest) = params.digest {
            Ok(vec![state.model_cache.gc_digest(
                &digest,
                &state.module_id,
                now,
                grace_ms,
            )?])
        } else {
            state.model_cache.gc_to_watermark(
                &state.module_id,
                now,
                grace_ms,
                state.runtime.cache_max_bytes,
            )
        }
    })();
    match result {
        Ok(outcomes) => {
            let removed_any = outcomes
                .iter()
                .any(|outcome| matches!(outcome, CacheGcOutcome::Deleted { .. }));
            if removed_any {
                if let Err(error) = state.store.reclaim_freelist_if_needed() {
                    tracing::warn!(
                        target: "maintenance",
                        error = %error,
                        "reclaim freelist failed after cache gc sweep"
                    );
                }
            }
            result_outcome(json!({
                "module_generation": state.module_generation,
                "outcomes": outcomes,
            }))
        }
        Err(error) => result_outcome(error_payload(&state, cache_error_to_wire(error))),
    }
}

fn cache_error_to_wire(error: ModelCacheError) -> WireOperationError {
    // A held digest lease is the one transient member of this error set: pin and
    // gc now serialize on the same lease, so a pin that arrives while a reader
    // or a sweep holds it is refused for timing reasons alone and succeeds on a
    // retry. Mapping it to artifact_invalid like the rest would tell a caller
    // its artifact is permanently broken and must not be retried, which is the
    // opposite of the truth and would strand a healthy blob.
    if matches!(error, ModelCacheError::Lease(_)) {
        return WireOperationError::from_stable(
            StableError::queue_full(Some(250)),
            error.to_string(),
        );
    }
    WireOperationError::from_stable(StableError::artifact_invalid(), error.to_string())
}

fn module_catalog_entries(state: &ModuleState) -> Vec<ModelCatalogEntry> {
    let slots = state
        .runtime
        .catalog
        .lock()
        .map(|catalog| {
            catalog
                .values()
                .map(|slot| {
                    (
                        slot.spec.clone(),
                        slot.loaded.clone(),
                        slot.state.clone(),
                        slot.last_cold_load_ms,
                    )
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    slots
        .into_iter()
        .filter(|(spec, _, _, _)| {
            resolved_catalog_lane(&state.runtime, &spec.model_id).is_none_or(|(e, b)| {
                current_catalog_install(state, e, &b.backend)
                    .ok()
                    .flatten()
                    .is_some()
            })
        })
        .map(|(spec, loaded, runtime_state, last_cold_load_ms)| {
            let exec_info = loaded
                .as_ref()
                .map(|model| model.execution_info())
                .unwrap_or_default();
            let is_loaded = loaded.is_some();

            let mut bucket_ladder = if is_loaded { exec_info.buckets } else { None };
            if let Some(buckets) = bucket_ladder.as_mut() {
                buckets.sort_unstable();
                buckets.dedup();
            }

            let (max_tokens, max_tokens_source) = if spec.task == "transcribe" {
                (None, None)
            } else if release_profile(&spec.engine_identity).is_some() {
                (Some(8192), Some("manifest".to_string()))
            } else if let Some(ref buckets) = bucket_ladder {
                if let Some(&largest_bucket) = buckets.last() {
                    let source = if spec.engine == "owned-metal" {
                        "runtime_bucket"
                    } else {
                        "worker_bucket"
                    };
                    (Some(largest_bucket), Some(source.to_string()))
                } else {
                    (Some(spec.max_tokens), Some("catalog".to_string()))
                }
            } else if is_loaded {
                (Some(spec.max_tokens), Some("catalog".to_string()))
            } else {
                (Some(spec.max_tokens), Some("catalog_unloaded".to_string()))
            };

            let dims = if is_loaded { exec_info.dims } else { None };
            let dtype = dtype_for_slot(&spec, exec_info.dtype);
            let device_class = device_class_for_engine(&spec.engine);

            let certified = matches!(spec.task.as_str(), "embed" | "rerank").then_some(false);

            let warm_load_cost_hint_ms = last_cold_load_ms;

            let recommended_batch = recommended_batch_for_model(&spec, max_tokens);

            ModelCatalogEntry {
                model_id: spec.model_id,
                state: model_runtime_state_name(&runtime_state).to_string(),
                fingerprints: vec![spec.fingerprint],
                recommended_batch,
                max_tokens,
                max_tokens_source,
                bucket_ladder,
                dims,
                dtype,
                device_class,
                certified,
                warm_load_cost_hint_ms,
            }
        })
        .collect()
}

fn models_list_payload(state: &ModuleState) -> Value {
    // The durable download can become committed before its executor refreshes
    // the in-memory registry. Project catalog rows from current install records.
    if let Err(error) = sync_installed_catalog_slots(state) {
        return error_payload(state, error);
    }
    let rows = module_catalog_entries(state)
        .into_iter()
        .map(|entry| serde_json::to_value(entry).expect("catalog entry serializes"));
    let mut models = Vec::new();
    for mut row in rows {
        if let Some((entry, backend)) =
            resolved_catalog_lane(&state.runtime, row["model_id"].as_str().unwrap_or(""))
        {
            let installed = match current_catalog_install(state, entry, &backend.backend) {
                Ok(install) => install.is_some(),
                Err(error) => return error_payload(state, error),
            };
            if catalog_lane_backend_reason(
                &state.runtime,
                entry,
                backend,
                synapse_core::Platform::current(),
            )
            .is_some()
                || !installed
            {
                continue;
            }
        }
        if let Some(spec) =
            model_slot_snapshot(&state.runtime, row["model_id"].as_str().unwrap_or(""))
        {
            if let Some(profile) = spec.spec.engine_identity.build_flags.get("profile") {
                row["profile"] = json!(profile);
            }
        }
        if let Err(error) = catalog_list_row(state, &mut row) {
            return error_payload(state, error);
        }
        lane_pause::project(&state.runtime, &mut row);
        models.push(row);
    }
    models.sort_by(|left, right| left["model_id"].as_str().cmp(&right["model_id"].as_str()));
    json!({
        "module_generation": state.module_generation,
        "table_epoch": FROZEN_TABLE_EPOCH,
        "models": models,
        "alias_rows": [],
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

    default_storage_descriptor_with_environment(module_id, |key| env::var_os(key))
}

fn write_startup_engram_catalog(module_id: &str) -> Result<(), String> {
    let descriptor = default_storage_descriptor_with_environment(module_id, |key| env::var_os(key))
        .map_err(|error| error.to_string())?;
    let StorageBackend::Sqlite { path } = descriptor.backend else {
        return Err("native store resolver did not select SQLite".to_string());
    };
    let data_root = Path::new(&path)
        .parent()
        .ok_or_else(|| format!("store path has no parent: {path}"))?;
    engram_catalog::write(data_root)
        .map(|_| ())
        .map_err(|error| format!("{}: {error}", data_root.display()))
}

fn default_storage_descriptor_with_environment(
    module_id: &str,
    env_var: impl FnMut(&str) -> Option<OsString>,
) -> Result<StorageDescriptor, ModuleError> {
    let data_home = synapse_core::resolve_data_root(
        synapse_core::DataRoot::Store,
        synapse_core::Platform::current(),
        env_var,
    )
    .map_err(|error| ModuleError::Config(error.to_string()))?;
    let path = sqlite_store_path(&data_home.to_string_lossy(), module_id);
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
        // Wire descriptions are deliberately unset: the operation names are the
        // documented surface, and the daemon treats the field as optional
        // discovery metadata.
        description: None,
    };

    vec![
        op("diagnostics.findings", Query),
        op("events_get", Query),
        op("certify.observations", Query),
        op("certify.transcribe_observation", Query),
        op("certify.transcribe_measurements", Query),
        op("certify.transcribe_observe_rerun", Mutate),
        op("role.describe", Query),
        op("media.submit", Mutate),
        op("media.status", Query),
        op("media.result", Query),
        op("media.cancel", Mutate),
        op("media.upload_begin", Mutate),
        op("media.upload_append", Mutate),
        op("media.upload_status", Query),
        op("media.upload_commit", Mutate),
        op("media.read", Query),
        op("lanes.pause", Mutate),
        op("lanes.resume", Mutate),
        op("lanes.status", Query),
        op("embed.query", Query),
        op("embed.batch", Query),
        op("embed.result", Query),
        op("rerank.score", Query),
        op("microllm.oneshot", Query),
        op("model.status", Query),
        op("model.unload", Mutate),
        op("models.list", Query),
        op("models.catalog", Query),
        op("models.download", Mutate),
        op("models.download.cancel", Mutate),
        op("models.remove", Mutate),
        op("cache.pin", Mutate),
        op("cache.gc", Mutate),
        op("admission.status", Query),
    ]
}

fn manifest(module_id: &str) -> ModuleManifest {
    // ModuleManifest went #[non_exhaustive] in subc-protocol 0.15 so additive
    // wire fields stop breaking every constructor; the builder is the one
    // sanctioned construction path. Every deliberate claim below keeps its
    // reasoning from the literal-construction era.
    //
    // Trust tier and bindings are left undeclared: subc-protocol 0.19 made
    // them optional because the daemon reads neither on any production path,
    // and a declaration nothing consumes only drifts from the truth unnoticed.
    let media = load_module_config()
        .and_then(|config| RuntimeState::from_catalog(config, Vec::new()))
        .map(|runtime| media_jobs::declaration(&runtime))
        .unwrap_or_else(|_| json!({"groups":["jobs"],"jobs_survive_restart":false,"engines":[]}));
    ModuleManifest::builder(module_id, env!("CARGO_PKG_VERSION"))
        .role_declarations(BTreeMap::from([(
            "media-provider/v1".into(),
            media.as_object().expect("media declaration").clone(),
        )]))
        .protocol_ver(PROTOCOL_VERSION)
        .provides(vec![ProviderRole::ManagementSurface {
            operations: management_operations(),
            config_schema: json!({ "type": "object" }),
            observability: Vec::new(),
            identity_scope: vec![IdentityScope::Project, IdentityScope::Session],
            // A deliberate claim, not the enum default: synapse accepts
            // concurrent calls (machine-wide admission exists precisely to
            // absorb many clients at once) and owns all execution ordering
            // internally via the admission semaphore and the fair-share
            // scheduler. Serial would break concurrent embed traffic;
            // StatelessParallel would discard per-channel FIFO, which job
            // paging relies on.
            concurrency: Concurrency::ModuleManaged,
        }])
        // Unattended flows may read diagnoses, never infer or change serving state.
        // Bind handling uses the principal and daemon-stamped scope, not project identity.
        .capabilities(Some(subc_protocol::manifest::CapabilityDeclarations {
            provides: vec![
                "flow-scopes/v1".to_string(),
                "media-provider/v1".to_string(),
                subc_protocol::scope::BIND_NO_PROJECT_CAPABILITY.to_string(),
            ],
            requires: Vec::new(),
            must_never_reach: Vec::new(),
        }))
        .flow_operations(Some(vec![
            "diagnostics.findings".to_string(),
            "events_get".to_string(),
        ]))
        .events(Some(vec![subc_protocol::manifest::EventDeclaration::new(
            "findings_changed",
            1,
        )
        .with_headers(vec!["event_key".to_string()])]))
        // Examined and none declarable (Some([]) is the wire form of that claim;
        // None would mean the vocabulary is un-adopted): synapse mutates nothing
        // outside its own store and models directory, and observation-anchored
        // signals would claim watch points we do not maintain.
        .self_signals(Some(Vec::new()))
        // Declare the provenance facts this build actually has: the linked
        // subc-protocol crate version, the commit when the source tree was
        // clean (omitted with a reason when it was dirty or git was
        // unavailable), and the newest store migration this binary carries,
        // which a daemon can compare with the store's version to spot a stale
        // binary.
        .provenance(Some(declared_provenance()))
        .build()
}

/// Serialize the manifest with this binary's source revision, if available.
/// This lets placement check the build without connecting to the daemon.
pub fn manifest_json() -> String {
    serde_json::to_string(&manifest("synapse")).expect("the SDK module manifest is serializable")
}

/// Where this build's commit came from, as embedded by `build.rs`.
fn build_git_sha_source() -> BuildGitShaSource<'static> {
    match (
        option_env!("SYNAPSE_BUILD_REV"),
        option_env!("SYNAPSE_BUILD_TREE"),
    ) {
        (Some(revision), Some(tree)) => BuildGitShaSource::Git {
            revision,
            tree_state: if tree == "clean" {
                GitTreeState::Clean
            } else {
                GitTreeState::Dirty
            },
        },
        _ => BuildGitShaSource::NoGitDir,
    }
}

fn declared_provenance() -> subc_protocol::manifest::ManifestProvenance {
    build_provenance_from_source(
        build_git_sha_source(),
        None,
        Some(&store::newest_schema_version().to_string()),
    )
    // The only inputs the helper validates are the commit, which build.rs
    // emits as a full 40-character hex string, and the Cargo.lock digest, which
    // is passed as absent. A validation failure here would mean build.rs
    // emitted something malformed, not a runtime condition.
    .expect("build.rs emits a canonical commit, so provenance form validation passes")
}

fn load_module_config() -> Result<ModuleConfig, ModuleError> {
    load_module_config_with_environment(|key| env::var_os(key), env::current_dir().ok().as_deref())
}

fn load_module_config_with_environment(
    mut env_var: impl FnMut(&str) -> Option<OsString>,
    cwd: Option<&Path>,
) -> Result<ModuleConfig, ModuleError> {
    if let Some(path) = env_var(SYNAPSE_CONFIG_PATH_ENV) {
        return load_module_config_file(&PathBuf::from(path), ConfigTier::User);
    }
    let user_path = default_synapse_config_path(cortexkit_store_types::resolve_config_home())?;
    if let Some(cwd) = cwd {
        let project_path = cwd.join(".cortexkit").join("synapse.jsonc");
        if project_path.is_file() {
            return load_module_config_file(&project_path, ConfigTier::Project);
        }
    }
    if user_path.is_file() {
        return load_module_config_file(&user_path, ConfigTier::User);
    }
    Ok(ModuleConfig::default())
}

// The shared resolver owns precedence; reject cwd-relative roots before joining
// so a missing home cannot silently select a project's user-tier configuration.
fn default_synapse_config_path(config_home: String) -> Result<PathBuf, ModuleError> {
    let config_home = PathBuf::from(config_home);
    if !config_home.is_absolute() {
        return Err(ModuleError::Config(format!(
            "config home {} is relative; set XDG_CONFIG_HOME to an absolute path (or provide an absolute HOME / Windows APPDATA or USERPROFILE)",
            config_home.display()
        )));
    }
    Ok(config_home.join("cortexkit").join("synapse.jsonc"))
}

#[derive(Clone, Copy)]
enum ConfigTier {
    User,
    Project,
}

fn load_module_config_file(path: &Path, tier: ConfigTier) -> Result<ModuleConfig, ModuleError> {
    let contents = fs::read_to_string(path)
        .map_err(|error| ModuleError::Config(format!("read {}: {error}", path.display())))?;
    parse_module_config_json(&contents, &path.display().to_string(), tier)
}

fn parse_module_config_json(
    contents: &str,
    source: &str,
    _tier: ConfigTier,
) -> Result<ModuleConfig, ModuleError> {
    let stripped = strip_json_comments(contents);
    let value: Value = serde_json::from_str(&stripped).map_err(ModuleError::Json)?;
    if value
        .as_object()
        .is_some_and(|object| object.contains_key("remote_providers"))
    {
        return Err(ModuleError::Config(
            "remote_providers were removed; modules should call providers directly".to_string(),
        ));
    }
    const ALIAS_ADMIN_KEYS: [&str; 3] = [
        "alias_admin_enabled",
        "dev_alias_admin",
        "enable_alias_admin",
    ];
    for object in [&value, &value["dev"]] {
        if let Some(key) = ALIAS_ADMIN_KEYS
            .iter()
            .find(|key| object.get(**key).is_some())
        {
            return Err(ModuleError::Config(format!(
                "alias_admin_enabled was removed (found '{key}'); aliases are no longer supported"
            )));
        }
    }
    for key in ["grammar_enabled", "sidecar_spec", "decode_chain_k"] {
        if value.get(key).is_some() {
            return Err(ModuleError::Config(format!(
                "{key} was removed; owned decode serving was retired"
            )));
        }
    }
    if let Some(preloads) = value.get("preload_models").and_then(Value::as_array) {
        for (index, preload) in preloads.iter().enumerate() {
            let Some(engine) = preload.get("engine").and_then(Value::as_str) else {
                continue;
            };
            if canonical_engine_name(engine) == "owned-metal-decode" {
                let model_id = preload
                    .get("model_id")
                    .and_then(Value::as_str)
                    .map_or_else(|| format!("{engine}-{index}"), str::to_owned);
                return Err(ModuleError::Config(format!(
                    "preload model_id '{model_id}': owned decode serving was retired"
                )));
            }
        }
    }
    let config: ModuleConfig = serde_json::from_value(value).map_err(|error| {
        if let Some(field) = unknown_field_from_json_error(&error) {
            tracing::error!(
                target: "config",
                source,
                field,
                "synapse config parse error: unknown field"
            );
            ModuleError::Config(format!(
                "unknown config field '{field}' in {source} (deny_unknown_fields)"
            ))
        } else {
            tracing::error!(target: "config", source, error = %error, "synapse config parse error");
            ModuleError::Json(error)
        }
    })?;
    for (index, preload) in config.preload_models.iter().enumerate() {
        let engine = canonical_engine_name(&preload.engine);
        let model_id = preload
            .model_id
            .clone()
            .unwrap_or_else(|| format!("{}-{index}", preload.engine));
        if preload.profile.is_none()
            && matches!(
                parse_model_task(preload.task.as_deref(), &engine, &model_id)?,
                ModelTask::Embed | ModelTask::Rerank
            )
        {
            return Err(ModuleError::Config(format!(
                "preload model_id '{model_id}': free-form preloads for embed and rerank were removed; specify a profile"
            )));
        }
    }
    Ok(config)
}

fn unknown_field_from_json_error(error: &serde_json::Error) -> Option<String> {
    let message = error.to_string();
    let marker = "unknown field `";
    let start = message.find(marker)? + marker.len();
    let rest = &message[start..];
    let end = rest.find('`')?;
    Some(rest[..end].to_string())
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
            inferred = if engine_name == LLAMA_ENGINE || engine_name == "llama.cpp" {
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
    #[cfg(feature = "test-support")]
    mod retired_engine_tests;
    #[cfg(unix)]
    mod lane_pause_tests {
        include!("lane_pause_tests.rs");
    }
    #[cfg(unix)]
    #[tokio::test]
    async fn failing_admitted_embed_and_rerank_log_closing_perf_lines() {
        use tracing::instrument::WithSubscriber;
        #[derive(Clone)]
        struct Writer(Arc<Mutex<Vec<u8>>>);
        impl std::io::Write for Writer {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        for (method, profile_id, params) in [
            (
                "embed.query",
                "qwen3-embedding-0.6b.owned-metal",
                json!({"text":"a"}),
            ),
            (
                "embed.batch",
                "qwen3-embedding-0.6b.owned-metal",
                json!({"texts":["a","a"]}),
            ),
            (
                "rerank.score",
                "qwen3-reranker-0.6b.owned-metal",
                json!({"query":"a","candidates":["a"]}),
            ),
        ] {
            let (root, descriptor) = test_storage_descriptor("failed-closing-log");
            let state = catalog_test_state(&root, &descriptor, profile_id);
            // The fixture already has a measured certification row. Remove the
            // profile flag so serving uses that row instead of running model
            // inference to check numerical accuracy before admission.
            {
                let mut slots = state.runtime.catalog.lock().unwrap();
                let slot = slots.get_mut("catalog-sequence-test").unwrap();
                slot.spec.engine_identity.build_flags.remove("profile");
                Arc::get_mut(slot.loaded.as_mut().unwrap())
                    .unwrap()
                    .engine_identity
                    .build_flags
                    .remove("profile");
            }
            state.runtime.execution.close();
            let bytes = Arc::new(Mutex::new(Vec::new()));
            let writer = Writer(bytes.clone());
            let subscriber = tracing_subscriber::fmt()
                .with_ansi(false)
                .without_time()
                .with_target(true)
                .with_writer(move || writer.clone())
                .finish();
            let mut params = params;
            params["model"] = json!("catalog-sequence-test");
            let outcome = dispatch_request(
                state.clone(),
                MethodEnvelope {
                    method: method.into(),
                    params,
                },
            )
            .with_subscriber(subscriber)
            .await;
            let result = response_result(outcome, method);
            assert_eq!(result["error"]["code"], "queue_full", "{method}: {result}");
            let captured = String::from_utf8(bytes.lock().unwrap().clone()).unwrap();
            // A failed job must still close its log record: some line after
            // admission names the job and the lane it ran on.
            let job = captured
                .lines()
                .find(|line| line.contains("job admitted"))
                .and_then(|line| line.split_whitespace().find(|s| s.starts_with("job_id=")))
                .unwrap_or_else(|| panic!("{method} never admitted: {captured}"));
            assert!(
                captured
                    .lines()
                    .skip_while(|line| !line.contains("job admitted"))
                    .skip(1)
                    .any(|line| line.contains(job) && line.contains("lane=\"metal\"")),
                "{method}: no closing line names {job} and its lane: {captured}"
            );
            drop(state);
            fs::remove_dir_all(root).unwrap();
        }
    }
    #[cfg(unix)]
    #[tokio::test]
    async fn profile_catalog_execution_reuses_the_self_check_lane_guard() {
        let (root, _) = test_storage_descriptor("ane-check-guard");
        let mut spec = catalog_fixture_config("gte-modernbert-base.ane-direct-worker");
        spec.model_id = "gte-modernbert-base-ane".into();
        let mut model = catalog_test_model(&root, &spec);
        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let engine = tokio::task::spawn_blocking({
            let started = started.clone();
            let release = release.clone();
            move || worker_host::ane_residency::module_mock_engine(started, release)
        })
        .await
        .unwrap();
        Arc::get_mut(&mut model).unwrap().backend = EmbedBackend::DirectAne(engine);
        let runtime =
            Arc::new(RuntimeState::from_catalog(ModuleConfig::default(), vec![]).unwrap());
        let guard = Arc::new(
            lane_serving_gate(&runtime, &model.model_id)
                .write_owned()
                .await,
        );
        let task = tokio::spawn({
            let runtime = runtime.clone();
            let model = model.clone();
            let guard = guard.clone();
            async move {
                execute_embedding_with_catalog_guard(
                    &runtime,
                    &model,
                    TokenBatch {
                        items: vec![vec![1]],
                    },
                    None,
                    None,
                    Some(guard),
                )
                .await
            }
        });
        let dispatched = tokio::time::timeout(Duration::from_secs(5), started.notified()).await;
        release.notify_one();
        drop(guard);
        if dispatched.is_err() {
            task.abort();
        } else {
            task.await.unwrap().unwrap();
        }
        drop(model);
        drop(runtime);
        fs::remove_dir_all(root).unwrap();
        assert!(
            dispatched.is_ok(),
            "self-check must not reacquire its own serving-gate guard"
        );
    }

    #[cfg(unix)]
    #[test]
    fn profile_preload_serving_requires_numeric_self_check() {
        use synapse_parity::evaluator::{evaluate_subset, ObservedCase, Output};
        let (root, descriptor) = test_storage_descriptor("profile-preload-gate");
        let profile = "gte-modernbert-base.owned-vulkan";
        let state = catalog_test_state(&root, &descriptor, profile);
        let model = state.runtime.loaded_models().into_iter().next().unwrap();
        let error = check_profile_preload_certified(&state, &model).unwrap_err();
        assert_eq!(error.code, "self_check_failed");
        let refs = synapse_certify::self_check::load(profile).unwrap();
        let mut outputs = refs
            .fixtures
            .cases()
            .iter()
            .map(|case| {
                (
                    case.id.clone(),
                    ObservedCase {
                        output: case.output.clone(),
                        input_ids: case.input_ids.clone(),
                        readout: None,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        let evaluation = evaluate_subset(
            &refs.manifest,
            profile,
            &model.fingerprint.0,
            &refs.fixtures,
            &outputs,
        )
        .unwrap();
        assert!(evaluation.passed());
        let (id, key) = profile_preload_check_key(&state, &model, profile).unwrap();
        let generation = catalog_check_generation(&state, &id, &key).unwrap();
        complete_profile_preload_check(&state, profile, &id, generation, &evaluation).unwrap();
        assert!(check_profile_preload_certified(&state, &model).is_ok());
        let Output::Embedding(vector) = &mut outputs.get_mut("short-0").unwrap().output else {
            unreachable!()
        };
        vector.pop();
        let failed = evaluate_subset(
            &refs.manifest,
            profile,
            &model.fingerprint.0,
            &refs.fixtures,
            &outputs,
        )
        .unwrap();
        assert!(!failed.passed());
        let generation = catalog_check_generation(&state, &id, &key).unwrap();
        assert_eq!(
            complete_profile_preload_check(&state, profile, &id, generation, &failed)
                .unwrap_err()
                .code,
            "self_check_failed"
        );
        assert_eq!(profile_preload_check_status(&state, &id).unwrap(), "failed");
        assert_eq!(
            check_profile_preload_certified(&state, &model)
                .unwrap_err()
                .code,
            "self_check_failed"
        );
        drop(model);
        drop(state);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn certify_observation_off_preserves_response_bytes() {
        let mut response = serde_json::json!({"payload": {"vectors": [[1.0]]}});
        let before = serde_json::to_vec(&response).unwrap();
        super::attach_certify_observation(&mut response, None);
        assert_eq!(serde_json::to_vec(&response).unwrap(), before);
        assert!(!super::ModuleConfig::default().certify_observation);
    }

    #[cfg(unix)]
    async fn assert_direct_ane_absolute_deadline(cold: bool) {
        let (root, _) = test_storage_descriptor("ane-deadline");
        let mut spec = catalog_fixture_config("gte-modernbert-base.ane-direct-worker");
        // A catalog lane id, so execution also takes the lane's serving gate.
        spec.model_id = "gte-modernbert-base-ane".into();
        let mut model = catalog_test_model(&root, &spec);
        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let engine = tokio::task::spawn_blocking({
            let started = started.clone();
            let release = release.clone();
            move || {
                if cold {
                    worker_host::ane_residency::module_mock_engine_admission(started, release)
                } else {
                    worker_host::ane_residency::module_mock_engine(started, release)
                }
            }
        })
        .await
        .unwrap();
        if !cold {
            let worker: Arc<dyn worker_host::ane_residency::AneShapeWorker> =
                engine.serving.channel.clone();
            drop(
                engine
                    .serving
                    .supervisor
                    .lease(&worker, &engine.serving.metadata.model_ref, 128)
                    .await
                    .unwrap(),
            );
        }
        Arc::get_mut(&mut model).unwrap().backend = EmbedBackend::DirectAne(engine.clone());
        let runtime =
            Arc::new(RuntimeState::from_catalog(ModuleConfig::default(), vec![]).unwrap());
        let deadline = tokio::time::Instant::now() + Duration::from_millis(100);
        let task = tokio::spawn({
            let model = model.clone();
            let runtime = runtime.clone();
            async move {
                execute_embedding(
                    &runtime,
                    &model,
                    TokenBatch {
                        items: vec![vec![1]],
                    },
                    Some(deadline),
                    None,
                )
                .await
            }
        });
        tokio::time::timeout(Duration::from_secs(2), started.notified())
            .await
            .unwrap();
        tokio::time::sleep_until(deadline + Duration::from_millis(20)).await;
        let finished_before_reply = task.is_finished();
        let in_flight = runtime.execution_stats.lock().unwrap().in_flight;
        let gate_held = lane_serving_gate(&runtime, &model.model_id)
            .try_write()
            .is_err();
        release.notify_one();
        let result = task.await.unwrap();
        tokio::task::spawn_blocking({
            let engine = engine.clone();
            move || engine.unload()
        })
        .await
        .unwrap()
        .unwrap();
        // A deadline that expires during the request's own first-time compile
        // is reported as shape_compiling; one that expires during ordinary
        // inference stays deadline_exceeded.
        let error = result.unwrap_err();
        assert_eq!(
            error.code,
            if cold {
                "shape_compiling"
            } else {
                "deadline_exceeded"
            },
            "{error:?}"
        );
        assert!(
            finished_before_reply,
            "caller deadline must bound the result wait"
        );
        assert!(
            gate_held,
            "expired work must retain its serving-gate guard while its reply drains"
        );
        // An execution permit is held only around the inference exchange,
        // so a compile (cold) holds none and an inference (warm) holds one.
        assert_eq!(
            in_flight,
            if cold { 0 } else { 1 },
            "expired work must retain its permit while an inference reply drains"
        );
        assert_eq!(engine.serving.supervisor.stats().restarts, 0);
        assert_eq!(
            engine.serving.channel.request_count(),
            if cold { 1 } else { 2 },
            "late admission must not dispatch inference"
        );
        drop(model);
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn direct_ane_warm_reply_after_absolute_deadline_is_discarded() {
        assert_direct_ane_absolute_deadline(false).await;
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn direct_ane_cold_admission_after_absolute_deadline_is_drained_without_inference() {
        assert_direct_ane_absolute_deadline(true).await;
    }

    #[cfg(unix)]
    const ANE_TEST_LANE: &str = "gte-modernbert-base-ane";

    #[cfg(unix)]
    struct AneLaneFixture {
        root: PathBuf,
        state: Arc<ModuleState>,
        engine: Arc<worker_host::ane_residency::DirectAneEngine>,
        control: worker_host::ane_residency::ModuleMockControl,
    }

    /// A loaded direct-ANE catalog lane served by the mock worker, optionally
    /// with its numerical self-check already recorded as passed.
    #[cfg(unix)]
    async fn ane_lane_fixture(label: &str, self_check_passed: bool) -> AneLaneFixture {
        let (root, descriptor) = test_storage_descriptor(label);
        let store = Arc::new(SynapseStore::open(&descriptor).unwrap());
        let profile = test_machine_profile(&format!("{label}-os"));
        store.activate_profile(&profile, 1, 1000).unwrap();
        let state = test_module_state(store, profile);
        let mut spec = catalog_fixture_config("gte-modernbert-base.ane-direct-worker");
        spec.model_id = ANE_TEST_LANE.into();
        let mut model = catalog_test_model(&root, &spec);
        let (engine, control) =
            tokio::task::spawn_blocking(worker_host::ane_residency::module_mock_engine_controlled)
                .await
                .unwrap();
        Arc::get_mut(&mut model).unwrap().backend = EmbedBackend::DirectAne(engine.clone());
        if self_check_passed {
            let (entry, backend) = resolved_catalog_lane(&state.runtime, ANE_TEST_LANE).unwrap();
            let (id, key) = catalog_self_check_key(&state, entry, backend).unwrap();
            let generation = catalog_check_generation(&state, &id, &key).unwrap();
            assert!(catalog_complete_check(&state, &id, generation, "passed", None).unwrap());
        }
        state.runtime.catalog.lock().unwrap().insert(
            ANE_TEST_LANE.into(),
            ModelSlot {
                spec,
                loaded: Some(model),
                state: ModelRuntimeState::Ready,
                notify: Arc::new(Notify::new()),
                last_cold_load_ms: None,
            },
        );
        AneLaneFixture {
            root,
            state,
            engine,
            control,
        }
    }

    #[cfg(unix)]
    fn ane_lane_model(state: &ModuleState) -> Arc<EmbeddingModel> {
        model_slot_snapshot(&state.runtime, ANE_TEST_LANE)
            .and_then(|slot| slot.loaded)
            .expect("loaded test lane")
    }

    #[cfg(unix)]
    async fn embed_on_ane_lane(
        state: &ModuleState,
        tokens: usize,
        deadline: Option<tokio::time::Instant>,
    ) -> Result<Vectors, WireOperationError> {
        execute_embedding(
            &state.runtime,
            &ane_lane_model(state),
            TokenBatch {
                items: vec![vec![1; tokens]],
            },
            deadline,
            None,
        )
        .await
    }

    #[cfg(unix)]
    async fn finish_ane_lane_fixture(fixture: AneLaneFixture) {
        let AneLaneFixture {
            root,
            state,
            engine,
            control,
        } = fixture;
        drop((state, control));
        let _ = tokio::task::spawn_blocking(move || engine.unload()).await;
        let _ = fs::remove_dir_all(root);
    }

    /// What a short request does before it executes: the resolver's wait for
    /// the lane lock, then profile certification, both with a 6 s budget.
    #[cfg(unix)]
    async fn resolve_and_certify_ane_lane(
        state: &Arc<ModuleState>,
    ) -> Result<(), WireOperationError> {
        let (guard, _) = catalog_resolution_guard(&state.runtime, ANE_TEST_LANE, 6000).await?;
        drop(guard);
        ensure_profile_preload_ready(
            state.clone(),
            ANE_TEST_LANE,
            Some(tokio::time::Instant::now() + Duration::from_secs(6)),
        )
        .await
        .map(drop)
    }

    #[cfg(unix)]
    async fn installed_ready_ane_fixture() -> AneLaneFixture {
        let (root, state) = tokio::task::spawn_blocking(|| {
            installed_catalog_boundary_state("gte-modernbert-base", "ane")
        })
        .await
        .unwrap();
        let (mut engine, control) =
            tokio::task::spawn_blocking(worker_host::ane_residency::module_mock_engine_controlled)
                .await
                .unwrap();
        // Transport silence is a separate fault from a request waiting behind work.
        Arc::get_mut(&mut Arc::get_mut(&mut engine).unwrap().serving)
            .unwrap()
            .metadata
            .timeout = Duration::from_secs(10);
        let mut slots = state.runtime.catalog.lock().unwrap();
        let slot = slots.get_mut(ANE_TEST_LANE).unwrap();
        let mut model = slot.loaded.take().unwrap();
        Arc::get_mut(&mut model).unwrap().backend = EmbedBackend::DirectAne(engine.clone());
        slot.loaded = Some(model);
        drop(slots);
        AneLaneFixture {
            root,
            state,
            engine,
            control,
        }
    }

    #[cfg(unix)]
    async fn resolve_ready_ane_request(
        state: Arc<ModuleState>,
        budget: u64,
    ) -> Result<Vectors, WireOperationError> {
        let deadline = tokio::time::Instant::now() + Duration::from_millis(budget);
        let model = resolve_serving_model(
            state.clone(),
            Some(ANE_TEST_LANE),
            ModelTask::Embed,
            None,
            None,
            Some(budget),
        )
        .await?;
        ensure_profile_request_certified(state.clone(), &model, Some(deadline)).await?;
        execute_embedding(
            &state.runtime,
            &model,
            TokenBatch {
                items: vec![vec![1]],
            },
            Some(deadline),
            None,
        )
        .await
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn ready_catalog_exchange_waits_beyond_five_seconds_and_keeps_its_deadline() {
        let fixture = installed_ready_ane_fixture().await;
        resolve_ready_ane_request(fixture.state.clone(), 30000)
            .await
            .unwrap();
        let (started, release) = ane_gate();
        fixture
            .control
            .hold_next_inference(started.clone(), release.clone());
        let first = tokio::spawn(resolve_ready_ane_request(fixture.state.clone(), 30000));
        tokio::time::timeout(Duration::from_secs(5), started.notified())
            .await
            .unwrap();
        let long = tokio::spawn(resolve_ready_ane_request(fixture.state.clone(), 30000));
        let began = Instant::now();
        let short = resolve_ready_ane_request(fixture.state.clone(), 100).await;
        let elapsed = began.elapsed();
        tokio::time::sleep(Duration::from_millis(5200)).await;
        let answered_early = long.is_finished();
        release.notify_one();
        let first = first.await.unwrap();
        let long = long.await.unwrap();
        finish_ane_lane_fixture(fixture).await;
        assert!(
            !answered_early,
            "a certified lane must wait for the exchange, not a loading heuristic"
        );
        assert!(first.is_ok(), "{first:?}");
        assert!(long.is_ok(), "{long:?}");
        let error = short.unwrap_err();
        assert_eq!(error.code, "deadline_exceeded", "{error:?}");
        assert!(!error.safe_to_retry_same_request);
        // The deadline_exceeded code already rules out the five-second loading
        // cap; the bound only catches a hang, with room for a loaded runner.
        assert!(
            elapsed >= Duration::from_millis(100) && elapsed < Duration::from_secs(5),
            "{elapsed:?}"
        );
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn delayed_ready_catalog_resolver_uses_the_request_deadline() {
        let fixture = installed_ready_ane_fixture().await;
        *fixture.state.runtime.resolver_ready_delay.lock().unwrap() =
            Some(Duration::from_millis(5200));
        let began = Instant::now();
        let served = resolve_ready_ane_request(fixture.state.clone(), 30000).await;
        let elapsed = began.elapsed();
        // The resolver's readiness task retains the lane lifecycle lock after caller expiry.
        let lock = catalog_lane_lock(&fixture.state.runtime, ANE_TEST_LANE)
            .lock_owned()
            .await;
        drop(lock);
        finish_ane_lane_fixture(fixture).await;
        assert!(
            elapsed >= Duration::from_secs(5),
            "the readiness delay did not execute: {elapsed:?}"
        );
        assert!(
            served.is_ok(),
            "a delayed Ready resolver must serve within the request deadline: {served:?}"
        );
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn ready_catalog_lifecycle_wait_uses_the_request_deadline() {
        let fixture = installed_ready_ane_fixture().await;
        let guard = catalog_lane_lock(&fixture.state.runtime, ANE_TEST_LANE)
            .lock_owned()
            .await;
        let long = tokio::spawn(resolve_ready_ane_request(fixture.state.clone(), 30000));
        let short = resolve_ready_ane_request(fixture.state.clone(), 100).await;
        tokio::time::sleep(Duration::from_millis(5200)).await;
        let answered_early = long.is_finished();
        drop(guard);
        let long = long.await.unwrap();
        finish_ane_lane_fixture(fixture).await;
        assert!(
            !answered_early,
            "a Ready lifecycle waiter must not expire at five seconds"
        );
        assert!(long.is_ok(), "{long:?}");
        let error = short.unwrap_err();
        assert_eq!(error.code, "deadline_exceeded", "{error:?}");
        assert!(!error.safe_to_retry_same_request);
    }

    #[tokio::test(start_paused = true)]
    async fn request_deadline_inline_admission_preserves_fractional_budget() {
        let runtime = RuntimeState::from_catalog(ModuleConfig::default(), Vec::new()).unwrap();
        let deadline = tokio::time::Instant::now() + Duration::from_millis(1200);
        tokio::time::advance(Duration::from_micros(600_250)).await;
        let admission = runtime
            .admit_inline_until("test-model", None, QueueClass::Bulk, 1, deadline, None)
            .unwrap();
        assert_eq!(admission.deadline(), deadline);
    }

    #[tokio::test(start_paused = true)]
    async fn request_deadline_remaining_milliseconds_round_up() {
        let now = tokio::time::Instant::now();
        for (remaining_ns, expected_ms) in [
            (0, 0),
            (1, 1),
            (999_999, 1),
            (1_000_000, 1),
            (1_000_001, 2),
            (599_750_000, 600),
        ] {
            assert_eq!(
                remaining_request_ms(now + Duration::from_nanos(remaining_ns)),
                expected_ms,
                "remaining_ns={remaining_ns}"
            );
        }
        assert_eq!(remaining_request_ms(now - Duration::from_nanos(1)), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn request_deadline_profile_resolution_keeps_submillisecond_budget() {
        let (root, descriptor) = test_storage_descriptor("profile-resolution-deadline");
        let store = Arc::new(SynapseStore::open(&descriptor).unwrap());
        let profile = test_machine_profile("test-os");
        store.observe_profile(&profile, 10, 1).unwrap();
        let state = test_module_state(store, profile);
        {
            let mut catalog = state.runtime.catalog.lock().unwrap();
            let slot = catalog.get_mut("stuck-model").unwrap();
            slot.spec
                .engine_identity
                .build_flags
                .insert("profile".to_string(), "test".to_string());
            // An in-progress load stays pending without spawning a real worker.
            slot.state = ModelRuntimeState::Loading;
        }
        let deadline = tokio::time::Instant::now() + Duration::from_micros(500);
        {
            let resolution = resolve_serving_model_until(
                state.clone(),
                Some("stuck-model"),
                ModelTask::Embed,
                None,
                None,
                deadline,
            );
            tokio::pin!(resolution);
            tokio::select! {
                biased;
                _ = &mut resolution => panic!("model resolution expired before its deadline"),
                _ = std::future::ready(()) => {}
            }
            tokio::time::advance(Duration::from_millis(1)).await;
            let error = resolution.await.err().expect("the load must time out");
            assert_eq!(error.code, "model_loading");
            assert!(tokio::time::Instant::now() >= deadline);
        }
        drop(state);
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn catalog_batch_resolution_and_execution_share_one_deadline() {
        let fixture = installed_ready_ane_fixture().await;
        resolve_ready_ane_request(fixture.state.clone(), 30000)
            .await
            .unwrap();
        let (started, release) = ane_gate();
        fixture
            .control
            .hold_next_inference(started.clone(), release.clone());
        let first = tokio::spawn(resolve_ready_ane_request(fixture.state.clone(), 30000));
        tokio::time::timeout(Duration::from_secs(5), started.notified())
            .await
            .unwrap();
        // Resolution spends most of the budget. If execution restarted the
        // budget instead of inheriting what is left, the request would end
        // about 5.5 s later than its deadline, which a loaded runner cannot
        // mistake for the 6 s it should take.
        *fixture.state.runtime.resolver_ready_delay.lock().unwrap() =
            Some(Duration::from_millis(5500));
        let began = Instant::now();
        let result = response_result(
            embed_batch(
                fixture.state.clone(),
                json!({
                    "model": ANE_TEST_LANE, "texts": ["test"],
                    "deadline_ms": 6000, "accept_declared": true,
                }),
            )
            .await,
            "embed.batch",
        );
        let elapsed = began.elapsed();
        release.notify_one();
        let first = first.await.unwrap();
        finish_ane_lane_fixture(fixture).await;
        assert!(first.is_ok(), "{first:?}");
        assert_eq!(result["error"]["code"], "deadline_exceeded", "{result}");
        assert_eq!(result["error"]["safe_to_retry_same_request"], false);
        assert!(
            elapsed >= Duration::from_millis(6000) && elapsed < Duration::from_secs(11),
            "resolution must consume execution time, not restart the budget: {elapsed:?}"
        );
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn validated_catalog_readiness_is_fenced_by_model_key_and_generation() {
        let fixture = installed_ready_ane_fixture().await;
        let state = &fixture.state;
        let model = ane_lane_model(state);
        assert!(validated_model_readiness(state, &model).unwrap());
        let (id, key) = profile_check_identity(state, &model).unwrap();
        let previous_generation =
            state.runtime.validated_readiness.lock().unwrap().models[ANE_TEST_LANE].generation;
        let generation = catalog_check_generation(state, &id, &key).unwrap();
        assert!(generation > previous_generation);
        assert!(
            !validated_model_readiness(state, &model).unwrap(),
            "a running generation is not passed"
        );
        assert!(!catalog_complete_check(state, &id, previous_generation, "passed", None).unwrap());
        assert!(!validated_model_readiness(state, &model).unwrap());
        assert!(catalog_complete_check(state, &id, generation, "passed", None).unwrap());
        assert!(validated_model_readiness(state, &model).unwrap());
        state
            .runtime
            .validated_readiness
            .lock()
            .unwrap()
            .models
            .get_mut(ANE_TEST_LANE)
            .unwrap()
            .check_id = "obsolete-key".into();
        assert!(validated_model_readiness(state, &model).unwrap());
        assert_eq!(
            state.runtime.validated_readiness.lock().unwrap().models[ANE_TEST_LANE].check_id,
            id,
            "readiness must be validated for the current self-check key"
        );
        set_model_slot_state(&state.runtime, ANE_TEST_LANE, ModelRuntimeState::Unloaded);
        assert!(
            !validated_model_readiness(state, &model).unwrap(),
            "an unloaded slot cannot reuse readiness"
        );
        set_model_slot_state(&state.runtime, ANE_TEST_LANE, ModelRuntimeState::Ready);
        let mut replacement = catalog_test_model(
            &fixture.root,
            &model_slot_snapshot(&state.runtime, ANE_TEST_LANE)
                .unwrap()
                .spec,
        );
        Arc::get_mut(&mut replacement).unwrap().backend =
            EmbedBackend::DirectAne(fixture.engine.clone());
        state
            .runtime
            .catalog
            .lock()
            .unwrap()
            .get_mut(ANE_TEST_LANE)
            .unwrap()
            .loaded = Some(replacement.clone());
        assert!(
            !validated_model_readiness(state, &model).unwrap(),
            "a replaced instance cannot reuse readiness"
        );
        assert!(validated_model_readiness(state, &replacement).unwrap());
        drop(replacement);
        drop(model);
        finish_ane_lane_fixture(fixture).await;
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn ready_passed_catalog_lane_still_requires_its_initial_install_validation() {
        let fixture = installed_ready_ane_fixture().await;
        fixture
            .state
            .store
            .sqlite()
            .with_conn_fenced(|tx| {
                tx.execute("DELETE FROM catalog_install_members", [])?;
                tx.execute("DELETE FROM catalog_installs", [])?;
                Ok(())
            })
            .unwrap();
        let result = select_catalog_lane(
            &fixture.state,
            Some(ANE_TEST_LANE),
            ModelTask::Embed,
            None,
            None,
        );
        finish_ane_lane_fixture(fixture).await;
        assert_eq!(
            result.unwrap_err().code,
            "model_not_installed",
            "a preloaded Ready model must not bypass initial catalog installation validation"
        );
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn catalog_resolver_expiry_does_not_describe_a_failed_lane_as_loading() {
        let fixture = installed_ready_ane_fixture().await;
        let failure = artifact_invalid_error("catalog load failed");
        set_model_slot_state(
            &fixture.state.runtime,
            ANE_TEST_LANE,
            ModelRuntimeState::Failed(failure.clone()),
        );
        let expired = catalog_resolution_expired(
            &fixture.state.runtime,
            ANE_TEST_LANE,
            tokio::time::Instant::now(),
        );
        finish_ane_lane_fixture(fixture).await;
        assert_eq!(
            expired, failure,
            "a settled load failure must not become model_loading"
        );
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn validated_ready_catalog_selection_reuses_readiness_without_the_store_lock() {
        let fixture = installed_ready_ane_fixture().await;
        resolve_ready_ane_request(fixture.state.clone(), 30000)
            .await
            .unwrap();
        let held = Arc::new(Notify::new());
        let (release, wait) = std::sync::mpsc::channel();
        let blocker = tokio::task::spawn_blocking({
            let state = fixture.state.clone();
            let held = held.clone();
            move || {
                state
                    .store
                    .sqlite()
                    .with_conn(|_| {
                        held.notify_one();
                        wait.recv_timeout(Duration::from_secs(10)).unwrap();
                        Ok(())
                    })
                    .unwrap()
            }
        });
        tokio::time::timeout(Duration::from_secs(2), held.notified())
            .await
            .unwrap();
        let mut selected = tokio::task::spawn_blocking({
            let state = fixture.state.clone();
            move || select_catalog_lane(&state, Some(ANE_TEST_LANE), ModelTask::Embed, None, None)
        });
        let result = tokio::time::timeout(Duration::from_secs(1), &mut selected).await;
        release.send(()).unwrap();
        blocker.await.unwrap();
        if result.is_err() {
            let _ = selected.await;
        }
        finish_ane_lane_fixture(fixture).await;
        assert!(
            matches!(result, Ok(Ok(Ok(_)))),
            "validated Ready selection must not wait on store work: {result:?}"
        );
    }

    #[tokio::test(start_paused = true)]
    #[cfg(unix)]
    async fn unresolved_catalog_resolver_preserves_the_five_second_loading_cap() {
        for lifecycle in [ModelRuntimeState::Unloaded, ModelRuntimeState::Loading] {
            let fixture = installed_ready_ane_fixture().await;
            {
                let mut slots = fixture.state.runtime.catalog.lock().unwrap();
                let slot = slots.get_mut(ANE_TEST_LANE).unwrap();
                slot.state = lifecycle;
                slot.loaded = None;
            }
            *fixture.state.runtime.resolver_ready_delay.lock().unwrap() =
                Some(Duration::from_millis(6500));
            let began = tokio::time::Instant::now();
            let result = resolve_ready_ane_request(fixture.state.clone(), 30000).await;
            let elapsed = began.elapsed();
            let load_continues = catalog_lane_lock(&fixture.state.runtime, ANE_TEST_LANE)
                .try_lock_owned()
                .is_err();
            let lock = catalog_lane_lock(&fixture.state.runtime, ANE_TEST_LANE)
                .lock_owned()
                .await;
            drop(lock);
            finish_ane_lane_fixture(fixture).await;
            let error = result.unwrap_err();
            assert_eq!(
                error.code, "model_loading",
                "genuine lifecycle work retains the loading response: {error:?}"
            );
            assert_eq!(error.retry_after_ms, Some(250));
            assert!(error.safe_to_retry_same_request);
            assert!(
                load_continues,
                "the loading task must retain lifecycle ownership after the capped response"
            );
            assert_eq!(
                error.message,
                format!("catalog lane '{ANE_TEST_LANE}' is loading")
            );
            assert!(
                elapsed >= Duration::from_secs(5) && elapsed < Duration::from_secs(6),
                "{elapsed:?}"
            );
        }
    }

    #[tokio::test(start_paused = true)]
    #[cfg(unix)]
    async fn unresolved_catalog_resolver_uses_deadline_exceeded_when_the_request_budget_runs_out() {
        let fixture = installed_ready_ane_fixture().await;
        {
            let mut slots = fixture.state.runtime.catalog.lock().unwrap();
            let slot = slots.get_mut(ANE_TEST_LANE).unwrap();
            slot.state = ModelRuntimeState::Unloaded;
            slot.loaded = None;
        }
        *fixture.state.runtime.resolver_ready_delay.lock().unwrap() =
            Some(Duration::from_millis(300));
        let began = tokio::time::Instant::now();
        let result = resolve_ready_ane_request(fixture.state.clone(), 100).await;
        let elapsed = began.elapsed();
        let lock = catalog_lane_lock(&fixture.state.runtime, ANE_TEST_LANE)
            .lock_owned()
            .await;
        drop(lock);
        finish_ane_lane_fixture(fixture).await;
        let error = result.unwrap_err();
        assert_eq!(
            error.code, "deadline_exceeded",
            "the loading cap must not extend the request deadline: {error:?}"
        );
        assert!(!error.safe_to_retry_same_request);
        assert!(
            elapsed >= Duration::from_millis(100) && elapsed < Duration::from_secs(1),
            "{elapsed:?}"
        );
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn a_loading_catalog_resolver_that_observes_ready_uses_the_request_deadline() {
        let fixture = installed_ready_ane_fixture().await;
        let model = ane_lane_model(&fixture.state);
        {
            let mut slots = fixture.state.runtime.catalog.lock().unwrap();
            let slot = slots.get_mut(ANE_TEST_LANE).unwrap();
            slot.state = ModelRuntimeState::Loading;
            slot.loaded = None;
        }
        *fixture.state.runtime.resolver_ready_delay.lock().unwrap() =
            Some(Duration::from_millis(5200));
        let publish_ready = tokio::spawn({
            let state = fixture.state.clone();
            async move {
                tokio::time::sleep(Duration::from_millis(100)).await;
                let mut slots = state.runtime.catalog.lock().unwrap();
                let slot = slots.get_mut(ANE_TEST_LANE).unwrap();
                slot.loaded = Some(model);
                slot.state = ModelRuntimeState::Ready;
            }
        });
        let began = Instant::now();
        let result = resolve_ready_ane_request(fixture.state.clone(), 30000).await;
        let elapsed = began.elapsed();
        publish_ready.await.unwrap();
        finish_ane_lane_fixture(fixture).await;
        assert!(
            elapsed >= Duration::from_secs(5),
            "the readiness delay must execute: {elapsed:?}"
        );
        assert!(
            result.is_ok(),
            "a lane that became Ready must not be described as loading: {result:?}"
        );
    }

    #[tokio::test(start_paused = true)]
    #[cfg(unix)]
    async fn a_ready_catalog_resolver_that_must_reload_uses_the_loading_cap() {
        let fixture = installed_ready_ane_fixture().await;
        let lock = catalog_lane_lock(&fixture.state.runtime, ANE_TEST_LANE)
            .lock_owned()
            .await;
        *fixture.state.runtime.resolver_ready_delay.lock().unwrap() =
            Some(Duration::from_millis(6500));
        let request = tokio::spawn(resolve_ready_ane_request(fixture.state.clone(), 30000));
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            !request.is_finished(),
            "resolution must retain its lifecycle wait"
        );
        {
            let mut slots = fixture.state.runtime.catalog.lock().unwrap();
            let slot = slots.get_mut(ANE_TEST_LANE).unwrap();
            slot.loaded = None;
            slot.state = ModelRuntimeState::Unloaded;
        }
        let began = tokio::time::Instant::now();
        drop(lock);
        let result = request.await.unwrap();
        let elapsed = began.elapsed();
        let load_continues = catalog_lane_lock(&fixture.state.runtime, ANE_TEST_LANE)
            .try_lock_owned()
            .is_err();
        let lock = catalog_lane_lock(&fixture.state.runtime, ANE_TEST_LANE)
            .lock_owned()
            .await;
        drop(lock);
        finish_ane_lane_fixture(fixture).await;
        let error = result.unwrap_err();
        assert_eq!(
            error.code, "model_loading",
            "an unload before lifecycle acquisition requires a capped reload: {error:?}"
        );
        assert_eq!(error.retry_after_ms, Some(250));
        assert!(load_continues);
        assert!(
            elapsed >= Duration::from_secs(5) && elapsed < Duration::from_secs(6),
            "{elapsed:?}"
        );
    }

    #[cfg(unix)]
    fn ane_gate() -> (Arc<Notify>, Arc<Notify>) {
        (Arc::new(Notify::new()), Arc::new(Notify::new()))
    }

    #[cfg(unix)]
    const PACKING_TEST_LANE: &str = "qwen3-embedding-0.6b-ane";
    #[cfg(unix)]
    const PACKING_TEST_MODEL: &str = "ane-direct:qwen3-embedding-0.6b.ane-direct-worker:pinned";

    #[cfg(unix)]
    #[derive(Default)]
    struct PackingPeer {
        requests: Vec<synapse_core::WorkerRequest>,
        admission_gate: Option<(Arc<Notify>, Arc<Notify>)>,
        inference_gate: Option<(Arc<Notify>, Arc<Notify>)>,
        corrupt: bool,
    }

    #[cfg(unix)]
    async fn packing_test_serving(
        peer: Arc<Mutex<PackingPeer>>,
        supervisor: worker_host::ane_residency::AneResidencySupervisor,
        id: &str,
    ) -> Arc<worker_host::ane_residency::AneServing<tokio::net::UnixStream>> {
        use synapse_core::{worker_protocol::*, worker_transport::*};
        use worker_host::ane_residency::*;
        let connector: AneWorkerConnector<tokio::net::UnixStream> = Box::new(move || {
            let peer = peer.clone();
            Box::pin(async move {
                let (host, mut worker) = tokio::net::UnixStream::pair().unwrap();
                let task = tokio::spawn(async move {
                    let max = DEFAULT_MAX_FRAME_BYTES;
                    let mut resident = BTreeSet::new();
                    while let Ok(request) = read_json::<WorkerRequest, _>(&mut worker, max).await {
                        let raw = if request.carries_raw_frame() {
                            Some(read_raw(&mut worker, max).await.unwrap())
                        } else {
                            None
                        };
                        peer.lock().unwrap().requests.push(request.clone());
                        let inventory = || AnePlacementInventory {
                            executables: (0..28)
                                .map(|layer| AneExecutable {
                                    id: format!("layer-{layer}"),
                                    layers: vec![layer],
                                })
                                .collect(),
                            cpu_stages: vec!["cpu_tail".into()],
                        };
                        let mut output = None;
                        let response = match &request {
                            WorkerRequest::AneAdmitShape { req_id, shape, .. } => {
                                resident.insert((false, *shape));
                                WorkerResponse::Admitted {
                                    req_id: req_id.clone(),
                                    inventory: inventory(),
                                    cache_hit_layers: None,
                                    compiled_layers: None,
                                }
                            }
                            WorkerRequest::AneAdmitPacked { req_id, .. } => {
                                let gate = peer.lock().unwrap().admission_gate.take();
                                if let Some((started, release)) = gate {
                                    started.notify_one();
                                    release.notified().await;
                                }
                                resident.insert((true, 256));
                                WorkerResponse::Admitted {
                                    req_id: req_id.clone(),
                                    inventory: inventory(),
                                    cache_hit_layers: None,
                                    compiled_layers: None,
                                }
                            }
                            WorkerRequest::AneEvictShape { req_id, shape, .. } => {
                                resident.remove(&(false, *shape));
                                WorkerResponse::Evicted {
                                    req_id: req_id.clone(),
                                }
                            }
                            WorkerRequest::AneEvictPacked { req_id, .. } => {
                                resident.remove(&(true, 256));
                                WorkerResponse::Evicted {
                                    req_id: req_id.clone(),
                                }
                            }
                            WorkerRequest::EmbedBatch { req_id, items, .. }
                            | WorkerRequest::EmbedPacked { req_id, items, .. } => {
                                let gate = peer.lock().unwrap().inference_gate.take();
                                if let Some((started, release)) = gate {
                                    started.notify_one();
                                    release.notified().await;
                                }
                                let packed = matches!(request, WorkerRequest::EmbedPacked { .. });
                                if let WorkerRequest::EmbedPacked { passes, .. } = &request {
                                    synapse_core::validate_packed_passes(
                                        &items.iter().map(|item| item.n_tokens).collect::<Vec<_>>(),
                                        passes,
                                    )
                                    .unwrap();
                                    assert!(
                                        resident.contains(&(true, 256)),
                                        "packed dispatch must be resident"
                                    );
                                } else {
                                    assert!(
                                        items.iter().all(|item| resident.contains(&(
                                            false,
                                            *worker_host::ane_residency::ANE_SHAPE_LADDER
                                                .iter()
                                                .find(|&&width| width >= item.n_tokens)
                                                .unwrap()
                                        ))),
                                        "standalone dispatch must use its natural resident rung"
                                    );
                                }
                                let ids = decode_i32_frame(raw.as_ref().unwrap()).unwrap();
                                let mut offset = 0;
                                let values = items
                                    .iter()
                                    .map(|item| {
                                        let value = ids[offset] as f32;
                                        offset += item.n_tokens;
                                        value
                                            + if packed && peer.lock().unwrap().corrupt {
                                                1.0
                                            } else {
                                                0.0
                                            }
                                    })
                                    .collect::<Vec<_>>();
                                assert_eq!(offset, ids.len());
                                output = Some(encode_f32_frame(&values));
                                WorkerResponse::Vectors {
                                    req_id: req_id.clone(),
                                    dims: 1,
                                    n: items.len(),
                                }
                            }
                            _ => WorkerResponse::unsupported_request(&request),
                        };
                        if write_json(&mut worker, &response, max).await.is_err() {
                            break;
                        }
                        if let Some(output) = output {
                            if write_raw(&mut worker, &output, max).await.is_err() {
                                break;
                            }
                        }
                    }
                });
                Ok(AneWorkerSession {
                    pid: None,
                    stream: host,
                    packed: Some(synapse_core::PackedCapability::tight256()),
                    confirm_exit: Box::pin(async move {
                        task.await
                            .map_err(|error| AneResidencyError::Channel(error.to_string()))?;
                        Ok(())
                    }),
                })
            })
        });
        let channel = Arc::new(
            AneWorkerChannel::connect(id, DEFAULT_MAX_FRAME_BYTES, connector)
                .await
                .unwrap(),
        );
        Arc::new(
            AneServing::new(
                channel,
                supervisor,
                AneServingMetadata {
                    model_ref: PACKING_TEST_MODEL.into(),
                    dims: 1,
                    buckets: worker_host::ane_residency::ANE_SHAPE_LADDER.to_vec(),
                    pooling: WorkerPooling::Last,
                    normalize: true,
                    timeout: Duration::from_secs(3),
                },
            )
            .unwrap(),
        )
    }

    #[cfg(unix)]
    async fn packing_lane_fixture(
        config: ModuleConfig,
        passed: bool,
    ) -> (AneLaneFixture, Arc<Mutex<PackingPeer>>) {
        let (root, descriptor) = test_storage_descriptor("module-packing");
        let store = Arc::new(SynapseStore::open(&descriptor).unwrap());
        let profile = test_machine_profile("packing-os");
        store.activate_profile(&profile, 1, 1000).unwrap();
        let state = test_module_state_with_config(store, profile, config);
        let mut spec = catalog_fixture_config("qwen3-embedding-0.6b.ane-direct-worker");
        spec.model_id = PACKING_TEST_LANE.into();
        spec.fingerprint = Fingerprint(
            state
                .runtime
                .release_catalog
                .entry("qwen3-embedding-0.6b")
                .unwrap()
                .backend("ane")
                .unwrap()
                .fingerprint
                .clone(),
        );
        let mut model = catalog_test_model(&root, &spec);
        let (mut engine, control) =
            tokio::task::spawn_blocking(worker_host::ane_residency::module_mock_engine_controlled)
                .await
                .unwrap();
        engine.serving.retire().await.unwrap();
        let peer = Arc::new(Mutex::new(PackingPeer::default()));
        Arc::get_mut(&mut engine).unwrap().serving = packing_test_serving(
            peer.clone(),
            worker_host::ane_residency::AneResidencySupervisor::new(Default::default()),
            "module-packing",
        )
        .await;
        Arc::get_mut(&mut model).unwrap().backend = EmbedBackend::DirectAne(engine.clone());
        state.runtime.catalog.lock().unwrap().insert(
            PACKING_TEST_LANE.into(),
            ModelSlot {
                spec,
                loaded: Some(model),
                state: ModelRuntimeState::Ready,
                notify: Arc::new(Notify::new()),
                last_cold_load_ms: None,
            },
        );
        if passed {
            let model = model_slot_snapshot(&state.runtime, PACKING_TEST_LANE)
                .unwrap()
                .loaded
                .unwrap();
            let (id, key) = profile_check_identity(&state, &model).unwrap();
            let generation = catalog_check_generation(&state, &id, &key).unwrap();
            assert!(catalog_complete_check(&state, &id, generation, "passed", None).unwrap());
        }
        (
            AneLaneFixture {
                root,
                state,
                engine,
                control,
            },
            peer,
        )
    }

    #[cfg(unix)]
    async fn packing_call(
        state: &ModuleState,
        rows: Vec<Vec<u32>>,
        deadline: Option<tokio::time::Instant>,
    ) -> Result<Vectors, WireOperationError> {
        let model = model_slot_snapshot(&state.runtime, PACKING_TEST_LANE)
            .unwrap()
            .loaded
            .unwrap();
        execute_embedding(
            &state.runtime,
            &model,
            TokenBatch { items: rows },
            deadline,
            None,
        )
        .await
    }

    #[cfg(unix)]
    async fn packing_attempt_settled(state: &ModuleState) {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let preparation = state
                    .runtime
                    .activity_telemetry
                    .packed_preparations
                    .lock()
                    .unwrap()
                    .get(PACKING_TEST_LANE)
                    .unwrap()
                    .clone();
                if !preparation.state.lock().unwrap().running {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("preparation must finish or defer, not wait for budget");
    }

    #[cfg(unix)]
    fn packing_config() -> ModuleConfig {
        parse_module_config_json(
            r#"{"ane_direct":{"tight_packing":true}}"#,
            "test",
            ConfigTier::User,
        )
        .unwrap()
    }

    #[cfg(unix)]
    fn packing_test_numeric(config: &ModuleConfig) -> NumericProfile {
        let spec = catalog_fixture_config("qwen3-embedding-0.6b.ane-direct-worker");
        let mut numeric: NumericProfile = serde_json::from_value(json!({
            "model_digest": spec.artifact_digest,
            "quant": spec.quant,
            "engine": spec.engine_identity,
            "sanitized_tokenizer_digest": spec.tokenizer_sanitized_digest,
            "pooling": "last_token",
            "normalization": "l2",
            "dtype": spec.owned_dtype,
            "flash_attention": "disabled",
            "certified_shape": {
                "max_context_tokens": spec.max_tokens,
                "max_batch_tokens": config.inline.max_tokens,
                "max_micro_batch_tokens": config.jobs.bulk_quantum_tokens,
                "max_sequences": config.inline.max_items,
            },
            "thread_policy": "balanced",
        }))
        .unwrap();
        CatalogProfile::load("qwen3-embedding-0.6b.ane-direct-worker")
            .unwrap()
            .apply_numeric_profile(&mut numeric);
        assert_eq!(numeric.numeric_profile_id(), spec.numeric_profile_id);
        numeric
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn omitted_tight_packing_never_prepares_qualifies_or_dispatches_even_a_qualified_key() {
        use synapse_core::WorkerRequest;
        let config = parse_module_config_json(
            r#"{"jobs":{"bulk_quantum_tokens":3072}}"#,
            "existing install",
            ConfigTier::User,
        )
        .unwrap();
        let before = packing_test_numeric(&config).stable_bytes();
        assert_eq!(
            before,
            packing_test_numeric(&packing_config()).stable_bytes()
        );
        let (fixture, peer) = packing_lane_fixture(config, true).await;
        // Install a genuinely qualified resident key first, so disabling
        // tight_packing cannot pass merely because the worker lacked packing.
        assert!(fixture.engine.serving.prepare_packed().await.unwrap());
        peer.lock().unwrap().requests.clear();
        run_lane_check(&fixture.state, PACKING_TEST_LANE)
            .await
            .unwrap();
        assert!(fixture
            .state
            .runtime
            .activity_telemetry
            .packed_preparations
            .lock()
            .unwrap()
            .is_empty());
        assert_eq!(
            packing_call(&fixture.state, vec![vec![41; 127], vec![42; 129]], None)
                .await
                .unwrap(),
            vec![vec![41.0], vec![42.0]]
        );
        assert!(peer
            .lock()
            .unwrap()
            .requests
            .iter()
            .all(|request| !matches!(
                request,
                WorkerRequest::AneAdmitPacked { .. } | WorkerRequest::EmbedPacked { .. }
            )));
        assert_eq!(
            before,
            packing_test_numeric(&ModuleConfig::default()).stable_bytes()
        );
        finish_ane_lane_fixture(fixture).await;
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn tight_packing_prepares_after_self_check_and_preserves_bounded_dispatch() {
        use synapse_core::WorkerRequest;
        let (fixture, peer) = packing_lane_fixture(packing_config(), false).await;
        *fixture
            .state
            .runtime
            .profile_check_passes_for_test
            .lock()
            .unwrap() = true;
        assert!(fixture
            .state
            .runtime
            .activity_telemetry
            .packed_preparations
            .lock()
            .unwrap()
            .is_empty());
        run_lane_check(&fixture.state, PACKING_TEST_LANE)
            .await
            .unwrap();
        packing_attempt_settled(&fixture.state).await;
        assert!(profile_check_passed(&fixture.state, PACKING_TEST_LANE)
            .unwrap()
            .is_some());
        assert!(
            fixture
                .state
                .runtime
                .activity_telemetry
                .packed_preparations
                .lock()
                .unwrap()[PACKING_TEST_LANE]
                .state
                .lock()
                .unwrap()
                .finished,
            "packed preparation did not finish after the lane self-check"
        );
        assert!(catalog_lane_lock(&fixture.state.runtime, PACKING_TEST_LANE)
            .try_lock()
            .is_ok());
        peer.lock().unwrap().requests.clear();
        assert_eq!(
            packing_call(&fixture.state, vec![vec![9; 128]], None)
                .await
                .unwrap(),
            vec![vec![9.0]]
        );
        assert!(peer.lock().unwrap().requests.iter().any(|request| matches!(request, WorkerRequest::EmbedBatch { items, .. } if items.len() == 1 && items[0].n_tokens == 128)));
        assert!(!peer
            .lock()
            .unwrap()
            .requests
            .iter()
            .any(|request| matches!(request, WorkerRequest::EmbedPacked { .. })));
        peer.lock().unwrap().requests.clear();
        let model = model_slot_snapshot(&fixture.state.runtime, PACKING_TEST_LANE)
            .unwrap()
            .loaded
            .unwrap();
        let rows = (1..=17).map(|id| vec![id; 127]).collect::<Vec<_>>();
        let expected = (1..=17).map(|id| vec![id as f32]).collect::<Vec<_>>();
        assert_eq!(
            execute_embedding_quanta(
                &fixture.state.runtime,
                &model,
                TokenBatch { items: rows },
                2159,
                1,
                None,
                None
            )
            .await
            .unwrap(),
            expected
        );
        {
            let requests = peer.lock().unwrap();
            let executions = requests
                .requests
                .iter()
                .filter_map(|request| match request {
                    WorkerRequest::EmbedBatch { items, .. }
                    | WorkerRequest::EmbedPacked { items, .. } => Some(items),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(
                executions
                    .iter()
                    .map(|items| items.len())
                    .collect::<Vec<_>>(),
                vec![8, 8, 1]
            );
            assert!(executions.iter().all(|items| items
                .iter()
                .map(|item| item.n_tokens)
                .sum::<usize>()
                <= DEFAULT_ENGINE_BATCH_TOKEN_BUDGET as usize));
            assert_eq!(
                requests
                    .requests
                    .iter()
                    .filter(|request| matches!(request, WorkerRequest::EmbedPacked { .. }))
                    .count(),
                2
            );
        }
        drop(model);
        finish_ane_lane_fixture(fixture).await;
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn tight_packing_qualification_mismatch_disables_only_packing() {
        use synapse_core::WorkerRequest;
        let (fixture, peer) = packing_lane_fixture(packing_config(), true).await;
        peer.lock().unwrap().corrupt = true;
        run_lane_check(&fixture.state, PACKING_TEST_LANE)
            .await
            .unwrap();
        packing_attempt_settled(&fixture.state).await;
        assert!(
            peer.lock()
                .unwrap()
                .requests
                .iter()
                .any(|request| matches!(request, WorkerRequest::EmbedPacked { .. })),
            "packed qualification did not run after the lane self-check"
        );
        peer.lock().unwrap().requests.clear();
        for _ in 0..2 {
            assert_eq!(
                packing_call(&fixture.state, vec![vec![41; 127], vec![42; 129]], None)
                    .await
                    .unwrap(),
                vec![vec![41.0], vec![42.0]]
            );
            packing_attempt_settled(&fixture.state).await;
        }
        assert!(peer
            .lock()
            .unwrap()
            .requests
            .iter()
            .all(|request| !matches!(
                request,
                WorkerRequest::AneAdmitPacked { .. } | WorkerRequest::EmbedPacked { .. }
            )));
        assert!(profile_check_passed(&fixture.state, PACKING_TEST_LANE)
            .unwrap()
            .is_some());
        finish_ane_lane_fixture(fixture).await;
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn tight_packing_admission_contention_keeps_resident_single_deadline_shape_compiling() {
        use synapse_core::WorkerRequest;
        let (fixture, peer) = packing_lane_fixture(packing_config(), true).await;
        packing_call(&fixture.state, vec![vec![7]], None)
            .await
            .unwrap();
        let (started, release) = ane_gate();
        peer.lock().unwrap().admission_gate = Some((started.clone(), release.clone()));
        run_lane_check(&fixture.state, PACKING_TEST_LANE)
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), started.notified())
            .await
            .unwrap();
        assert!(catalog_lane_lock(&fixture.state.runtime, PACKING_TEST_LANE)
            .try_lock()
            .is_ok());
        assert!(lane_serving_gate(&fixture.state.runtime, PACKING_TEST_LANE)
            .try_read()
            .is_ok());
        let before = peer.lock().unwrap().requests.len();
        let request = packing_call(
            &fixture.state,
            vec![vec![8; 128]],
            Some(tokio::time::Instant::now() + Duration::from_millis(50)),
        )
        .await;
        assert_eq!(request.unwrap_err().code, "shape_compiling");
        assert_eq!(
            peer.lock().unwrap().requests.len(),
            before,
            "expired request must not dispatch while compilation holds the stream"
        );
        release.notify_one();
        packing_attempt_settled(&fixture.state).await;
        peer.lock().unwrap().requests.clear();
        assert_eq!(
            packing_call(&fixture.state, vec![vec![8; 128]], None)
                .await
                .unwrap(),
            vec![vec![8.0]]
        );
        assert!(peer
            .lock()
            .unwrap()
            .requests
            .iter()
            .all(|request| !matches!(
                request,
                WorkerRequest::AneAdmitPacked { .. } | WorkerRequest::EmbedPacked { .. }
            )));
        finish_ane_lane_fixture(fixture).await;
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn tight_packing_budget_defers_until_the_next_idle_transition() {
        use worker_host::ane_residency::*;
        let (fixture, peer) = packing_lane_fixture(packing_config(), true).await;
        let other = packing_test_serving(
            Arc::new(Mutex::new(PackingPeer::default())),
            fixture.engine.serving.supervisor.clone(),
            "other-worker",
        )
        .await;
        let worker: Arc<dyn AneShapeWorker> = other.channel.clone();
        let mut held = Vec::new();
        for width in [128, 256, 512] {
            held.push(
                other
                    .supervisor
                    .lease(&worker, PACKING_TEST_MODEL, width)
                    .await
                    .unwrap(),
            );
        }
        run_lane_check(&fixture.state, PACKING_TEST_LANE)
            .await
            .unwrap();
        packing_attempt_settled(&fixture.state).await;
        assert!(
            peer.lock().unwrap().requests.is_empty(),
            "full leased budget must skip even qualification references"
        );
        assert!(
            !fixture
                .state
                .runtime
                .activity_telemetry
                .packed_preparations
                .lock()
                .unwrap()[PACKING_TEST_LANE]
                .state
                .lock()
                .unwrap()
                .finished
        );
        drop(held);
        // Releasing budget alone does not compile on behalf of a request. Its
        // completed standalone exchange supplies the next idle transition.
        assert!(peer.lock().unwrap().requests.is_empty());
        packing_call(&fixture.state, vec![vec![1]], None)
            .await
            .unwrap();
        packing_attempt_settled(&fixture.state).await;
        assert!(
            fixture
                .state
                .runtime
                .activity_telemetry
                .packed_preparations
                .lock()
                .unwrap()[PACKING_TEST_LANE]
                .state
                .lock()
                .unwrap()
                .finished
        );
        assert_eq!(
            packing_call(&fixture.state, vec![vec![41; 127], vec![42; 129]], None)
                .await
                .unwrap(),
            vec![vec![41.0], vec![42.0]]
        );
        assert!(peer
            .lock()
            .unwrap()
            .requests
            .iter()
            .any(|request| matches!(request, synapse_core::WorkerRequest::EmbedPacked { .. })));
        other.retire().await.unwrap();
        finish_ane_lane_fixture(fixture).await;
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn tight_packing_waits_for_cancelled_request_drain_before_preparation() {
        use synapse_core::WorkerRequest;
        let (fixture, peer) = packing_lane_fixture(packing_config(), true).await;
        let (started, release) = ane_gate();
        peer.lock().unwrap().inference_gate = Some((started.clone(), release.clone()));
        let request = tokio::spawn({
            let state = fixture.state.clone();
            async move { packing_call(&state, vec![vec![1]], None).await }
        });
        tokio::time::timeout(Duration::from_secs(5), started.notified())
            .await
            .unwrap();
        run_lane_check(&fixture.state, PACKING_TEST_LANE)
            .await
            .unwrap();
        request.abort();
        assert!(request.await.unwrap_err().is_cancelled());
        assert_eq!(
            fixture
                .state
                .runtime
                .activity_telemetry
                .by_model
                .lock()
                .unwrap()[PACKING_TEST_LANE],
            1
        );
        assert!(
            !fixture
                .state
                .runtime
                .activity_telemetry
                .packed_preparations
                .lock()
                .unwrap()[PACKING_TEST_LANE]
                .state
                .lock()
                .unwrap()
                .running
        );
        assert!(
            !peer.lock().unwrap().requests.iter().any(|request| matches!(
                request,
                WorkerRequest::AneAdmitPacked { .. } | WorkerRequest::EmbedPacked { .. }
            ))
        );
        release.notify_one();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let finished = fixture
                    .state
                    .runtime
                    .activity_telemetry
                    .packed_preparations
                    .lock()
                    .unwrap()[PACKING_TEST_LANE]
                    .state
                    .lock()
                    .unwrap()
                    .finished;
                if finished {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(!fixture
            .state
            .runtime
            .activity_telemetry
            .by_model
            .lock()
            .unwrap()
            .contains_key(PACKING_TEST_LANE));
        finish_ane_lane_fixture(fixture).await;
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn tight_packing_unpinned_lane_stays_standalone_and_over_width_rows_fall_back() {
        use synapse_core::WorkerRequest;
        let (fixture, peer) = packing_lane_fixture(packing_config(), true).await;
        run_lane_check(&fixture.state, PACKING_TEST_LANE)
            .await
            .unwrap();
        packing_attempt_settled(&fixture.state).await;
        peer.lock().unwrap().requests.clear();
        assert_eq!(
            packing_call(
                &fixture.state,
                vec![vec![41; 127], vec![42; 129], vec![43; 257]],
                None
            )
            .await
            .unwrap(),
            vec![vec![41.0], vec![42.0], vec![43.0]]
        );
        assert!(peer.lock().unwrap().requests.iter().any(|request| matches!(request, WorkerRequest::EmbedPacked { items, .. } if items.len() == 2)), "qualified rows were not dispatched as packed after preparation");
        assert!(peer.lock().unwrap().requests.iter().any(|request| matches!(request, WorkerRequest::EmbedBatch { items, .. } if items.len() == 1 && items[0].n_tokens == 257)));
        assert!(fixture.engine.serving.prepare_packed().await.unwrap());
        fixture
            .state
            .runtime
            .activity_telemetry
            .packed_preparations
            .lock()
            .unwrap()
            .remove(PACKING_TEST_LANE);
        {
            let mut slots = fixture.state.runtime.catalog.lock().unwrap();
            Arc::get_mut(
                slots
                    .get_mut(PACKING_TEST_LANE)
                    .unwrap()
                    .loaded
                    .as_mut()
                    .unwrap(),
            )
            .unwrap()
            .fingerprint = Fingerprint("unqualified-profile".into());
        }
        peer.lock().unwrap().requests.clear();
        let model = model_slot_snapshot(&fixture.state.runtime, PACKING_TEST_LANE)
            .unwrap()
            .loaded
            .unwrap();
        start_packed_preparation(&fixture.state.runtime, &model);
        assert!(fixture
            .state
            .runtime
            .activity_telemetry
            .packed_preparations
            .lock()
            .unwrap()
            .is_empty());
        drop(model);
        assert_eq!(
            packing_call(&fixture.state, vec![vec![41; 127], vec![42; 129]], None)
                .await
                .unwrap(),
            vec![vec![41.0], vec![42.0]]
        );
        assert!(peer
            .lock()
            .unwrap()
            .requests
            .iter()
            .all(|request| !matches!(
                request,
                WorkerRequest::AneAdmitPacked { .. } | WorkerRequest::EmbedPacked { .. }
            )));
        finish_ane_lane_fixture(fixture).await;
    }

    /// After a restart the lane's self-check is usually already persisted as
    /// passed, so requests take the passed-check fast path and the lane check
    /// (which starts the prewarm after a fresh pass) never runs. The load must
    /// then start the prewarm itself, or the short shapes are only ever
    /// loaded by the requests that need them.
    #[tokio::test]
    #[cfg(unix)]
    async fn lane_loaded_with_a_persisted_passing_check_starts_the_prewarm() {
        use synapse_core::WorkerRequest;
        let (fixture, peer) = packing_lane_fixture(ModuleConfig::default(), true).await;
        let model = ensure_profile_preload_ready(fixture.state.clone(), PACKING_TEST_LANE, None)
            .await
            .unwrap();
        assert_eq!(
            fixture
                .state
                .runtime
                .lane_check_tasks
                .load(std::sync::atomic::Ordering::SeqCst),
            0,
            "with a persisted passing check, readiness must not start a lane check"
        );
        start_work_for_already_checked_lane(&fixture.state, &model);
        let admitted = || {
            peer.lock()
                .unwrap()
                .requests
                .iter()
                .filter_map(|request| match request {
                    WorkerRequest::AneAdmitShape { shape, .. } => Some(*shape),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        let prewarmed = tokio::time::timeout(Duration::from_secs(5), async {
            while admitted().len() < 3 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await;
        assert!(
            prewarmed.is_ok(),
            "the prewarm must load its shapes on a lane loaded already checked, saw {:?}",
            admitted()
        );
        assert_eq!(admitted(), vec![128, 256, 512]);
        drop(model);
        finish_ane_lane_fixture(fixture).await;
    }

    /// Whether a self-check (or unload) holds or is queued for the lane's
    /// serving gate: once a writer is queued, new readers cannot enter.
    #[cfg(unix)]
    fn ane_lane_writer_waiting(state: &ModuleState) -> bool {
        lane_serving_gate(&state.runtime, ANE_TEST_LANE)
            .try_read()
            .is_err()
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn resident_shape_request_is_not_refused_as_model_loading_while_another_shape_compiles() {
        let fixture = ane_lane_fixture("ane-compile-resident", true).await;
        let state = fixture.state.clone();
        embed_on_ane_lane(&state, 1, None).await.unwrap();
        let (started, release) = ane_gate();
        fixture
            .control
            .hold_next_admission(started.clone(), release.clone());
        let compile = tokio::spawn({
            let state = state.clone();
            async move { embed_on_ane_lane(&state, 600, None).await }
        });
        tokio::time::timeout(Duration::from_secs(2), started.notified())
            .await
            .unwrap();
        let resolved = resolve_and_certify_ane_lane(&state).await;
        let short = tokio::spawn({
            let state = state.clone();
            async move {
                let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
                embed_on_ane_lane(&state, 1, Some(deadline)).await
            }
        });
        tokio::time::sleep(Duration::from_millis(100)).await;
        let answered_during_compile = short.is_finished();
        release.notify_one();
        let short = short.await.unwrap();
        let compiled = compile.await.unwrap();
        let restarts = fixture.engine.serving.supervisor.stats().restarts;
        let compiles = fixture.control.compiles(1024);
        finish_ane_lane_fixture(fixture).await;
        assert!(
            resolved.is_ok(),
            "a loaded lane must resolve while a shape compiles: {resolved:?}"
        );
        assert!(
            !answered_during_compile,
            "the worker is serial: the short request runs after the compile"
        );
        assert!(short.is_ok(), "{short:?}");
        assert!(compiled.is_ok(), "{compiled:?}");
        assert_eq!(restarts, 0);
        assert_eq!(compiles, 1);
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn concurrent_requests_for_one_uncompiled_shape_compile_once_and_waiters_keep_their_deadline(
    ) {
        let fixture = ane_lane_fixture("ane-single-flight", true).await;
        let state = fixture.state.clone();
        let (started, release) = ane_gate();
        fixture
            .control
            .hold_next_admission(started.clone(), release.clone());
        let first = tokio::spawn({
            let state = state.clone();
            async move { embed_on_ane_lane(&state, 600, None).await }
        });
        tokio::time::timeout(Duration::from_secs(2), started.notified())
            .await
            .unwrap();
        let waiter = tokio::spawn({
            let state = state.clone();
            async move {
                let deadline = tokio::time::Instant::now() + Duration::from_millis(300);
                embed_on_ane_lane(&state, 600, Some(deadline)).await
            }
        });
        let waited = tokio::time::timeout(Duration::from_secs(2), waiter).await;
        release.notify_one();
        let first = first.await.unwrap();
        let later = embed_on_ane_lane(
            &state,
            600,
            Some(tokio::time::Instant::now() + Duration::from_secs(5)),
        )
        .await;
        let compiles = fixture.control.compiles(1024);
        finish_ane_lane_fixture(fixture).await;
        let error = match waited {
            Ok(Ok(Err(error))) => error,
            other => panic!("the waiter must answer at its own deadline: {other:?}"),
        };
        assert_eq!(error.code, "shape_compiling", "{error:?}");
        assert_eq!(error.class, ErrorClass::Transient);
        assert_eq!(error.retry_after_ms, Some(1000));
        assert!(error.safe_to_retry_same_request);
        assert_eq!(
            error.details,
            Some(json!({"lane_id": ANE_TEST_LANE, "shape": 1024}))
        );
        assert!(first.is_ok(), "{first:?}");
        assert!(later.is_ok(), "{later:?}");
        assert_eq!(
            compiles, 1,
            "one compile serves every request for the shape"
        );
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn cancelled_compile_neither_restarts_the_worker_nor_leaks_budget() {
        let fixture = ane_lane_fixture("ane-cancel-compile", true).await;
        let state = fixture.state.clone();
        embed_on_ane_lane(&state, 1, None).await.unwrap();
        let (started, release) = ane_gate();
        fixture
            .control
            .hold_next_admission(started.clone(), release.clone());
        let compile = tokio::spawn({
            let state = state.clone();
            async move { embed_on_ane_lane(&state, 600, None).await }
        });
        tokio::time::timeout(Duration::from_secs(2), started.notified())
            .await
            .unwrap();
        compile.abort();
        assert!(compile.await.unwrap_err().is_cancelled());
        let resolved = resolve_and_certify_ane_lane(&state).await;
        let short = tokio::spawn({
            let state = state.clone();
            async move {
                let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
                embed_on_ane_lane(&state, 1, Some(deadline)).await
            }
        });
        tokio::time::sleep(Duration::from_millis(100)).await;
        release.notify_one();
        let short = short.await.unwrap();
        let supervisor = fixture.engine.serving.supervisor.clone();
        // The abandoned compile finishes in its detached task; wait for it to
        // release everything it held.
        let gate = lane_serving_gate(&state.runtime, ANE_TEST_LANE);
        let drained = tokio::time::timeout(Duration::from_secs(5), async {
            while supervisor.reservation_snapshot().1 != 0 || gate.try_write().is_err() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await;
        let reservation = supervisor.reservation_snapshot();
        let resident = supervisor.resident_shapes();
        let restarts = supervisor.stats().restarts;
        let compiles = fixture.control.compiles(1024);
        finish_ane_lane_fixture(fixture).await;
        assert!(
            resolved.is_ok(),
            "a loaded lane must resolve while a cancelled compile drains: {resolved:?}"
        );
        assert!(short.is_ok(), "{short:?}");
        assert!(drained.is_ok(), "the cancelled compile never drained");
        assert_eq!(restarts, 0);
        assert_eq!(compiles, 1);
        assert_eq!(
            resident.values().flatten().copied().collect::<Vec<_>>(),
            vec![128, 1024]
        );
        // The budget holds exactly the two resident shapes (22 layer programs
        // each for this model) and no request still holds a lease on either.
        assert_eq!(reservation, (44, 0));
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn unload_and_self_check_never_overlap_an_in_flight_inference() {
        let fixture = ane_lane_fixture("ane-unload-in-flight", false).await;
        let state = fixture.state.clone();
        embed_on_ane_lane(&state, 1, None).await.unwrap();
        let (started, release) = ane_gate();
        fixture
            .control
            .hold_next_inference(started.clone(), release.clone());
        let in_flight = tokio::spawn({
            let state = state.clone();
            async move { embed_on_ane_lane(&state, 1, None).await }
        });
        tokio::time::timeout(Duration::from_secs(2), started.notified())
            .await
            .unwrap();
        // Cancel the caller: only the detached task still holds the inference.
        in_flight.abort();
        assert!(in_flight.await.unwrap_err().is_cancelled());
        let refused = response_result(
            model_unload(state.clone(), json!({"model_id": ANE_TEST_LANE})).await,
            "model.unload",
        );
        if refused["error"]["code"] != "model_in_use" {
            release.notify_one();
            finish_ane_lane_fixture(fixture).await;
            panic!("unload must refuse while an inference reply drains: {refused}");
        }
        let check = tokio::spawn({
            let state = state.clone();
            async move {
                ensure_profile_preload_ready(state, ANE_TEST_LANE, None)
                    .await
                    .map(drop)
            }
        });
        tokio::time::timeout(Duration::from_secs(5), async {
            while !ane_lane_writer_waiting(&state) {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("the self-check never queued for the lane");
        // The self-check is waiting for the in-flight reply; a request that
        // passed certification earlier must wait behind it, not reach the
        // worker alongside the check.
        let exchanges_before = fixture.engine.serving.channel.request_count();
        let behind_check = tokio::time::timeout(
            Duration::from_secs(2),
            embed_on_ane_lane(
                &state,
                1,
                Some(tokio::time::Instant::now() + Duration::from_millis(300)),
            ),
        )
        .await;
        let exchanges_during_check =
            fixture.engine.serving.channel.request_count() - exchanges_before;
        release.notify_one();
        let checked = tokio::time::timeout(Duration::from_secs(10), check).await;
        let unloaded = response_result(
            model_unload(state.clone(), json!({"model_id": ANE_TEST_LANE})).await,
            "model.unload",
        );
        finish_ane_lane_fixture(fixture).await;
        match behind_check {
            Ok(Err(error)) => assert_eq!(error.code, "deadline_exceeded", "{error:?}"),
            other => panic!("a request during the self-check must wait at its deadline: {other:?}"),
        }
        assert_eq!(
            exchanges_during_check, 0,
            "a request reached the worker while the self-check held the lane"
        );
        assert!(
            matches!(checked, Ok(Ok(_))),
            "the self-check never ran: {checked:?}"
        );
        assert!(unloaded.get("error").is_none(), "{unloaded}");
    }

    /// The request that finds the self-check pending starts it, but the check
    /// is lane lifecycle work: it keeps running after that request's deadline
    /// and lets later requests serve once it has passed.
    #[tokio::test]
    #[cfg(unix)]
    async fn a_request_that_starts_a_self_check_answers_model_loading_at_its_deadline() {
        let fixture = ane_lane_fixture("ane-check-starter", false).await;
        *fixture
            .state
            .runtime
            .profile_check_passes_for_test
            .lock()
            .unwrap() = true;
        let state = fixture.state.clone();
        // The self-check's first reference input pads to the 128-token shape,
        // which the mock worker must compile first; hold that compile open.
        let (started, release) = ane_gate();
        fixture
            .control
            .hold_next_admission(started.clone(), release.clone());
        let model = ane_lane_model(&state);
        // Drive only the caller's deadline with virtual time. Single polls keep
        // the clock from auto-advancing while the external mock worker waits.
        let (first, check_running) = {
            tokio::time::pause();
            let deadline = tokio::time::Instant::now() + Duration::from_millis(100);
            let certify = ensure_profile_request_certified(state.clone(), &model, Some(deadline));
            tokio::pin!(certify);
            let subscribed = std::future::poll_fn(|cx| {
                std::task::Poll::Ready(std::future::Future::poll(certify.as_mut(), cx))
            })
            .await;
            assert!(subscribed.is_pending(), "{subscribed:?}");
            tokio::time::advance(Duration::from_millis(99)).await;
            let before_deadline = std::future::poll_fn(|cx| {
                std::task::Poll::Ready(std::future::Future::poll(certify.as_mut(), cx))
            })
            .await;
            assert!(before_deadline.is_pending(), "{before_deadline:?}");
            // Tokio rounds timer deadlines up to its millisecond tick; cross
            // that tick without spending any wall-clock scheduling budget.
            tokio::time::advance(Duration::from_millis(2)).await;
            let first = std::future::poll_fn(|cx| {
                std::task::Poll::Ready(std::future::Future::poll(certify.as_mut(), cx))
            })
            .await;
            let check_running = state
                .runtime
                .lane_checks
                .lock()
                .unwrap()
                .get(ANE_TEST_LANE)
                .is_some_and(|outcome| outcome.borrow().is_none());
            tokio::time::resume();
            (first, check_running)
        };
        release.notify_one();
        until_lane_check_settles(&state).await;
        // A completed check must have crossed admission, even if the worker
        // reached it after the caller expired and the release permit was sent.
        let compile_reached =
            tokio::time::timeout(Duration::from_secs(2), started.notified()).await;
        let later = tokio::time::Instant::now() + Duration::from_secs(5);
        let second = ensure_profile_request_certified(state.clone(), &model, Some(later)).await;
        let served = embed_on_ane_lane(&state, 1, Some(later)).await;
        drop(model);
        finish_ane_lane_fixture(fixture).await;
        match first {
            std::task::Poll::Ready(Err(error)) => {
                assert_eq!(error.code, "model_loading", "{error:?}")
            }
            other => panic!("the starting request must answer at its deadline: {other:?}"),
        }
        assert!(
            compile_reached.is_ok(),
            "the check never ran its reference case"
        );
        assert!(check_running, "the check stopped with its starting request");
        assert!(second.is_ok(), "{second:?}");
        assert!(served.is_ok(), "{served:?}");
    }

    #[cfg(unix)]
    async fn until_lane_check_settles(state: &ModuleState) {
        tokio::time::timeout(Duration::from_secs(10), async {
            while !state.runtime.lane_checks.lock().unwrap().is_empty() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("the lane's self-check never finished");
    }

    /// Many short requests arrive while a self-check runs. They share the one
    /// check instead of each queueing their own task on the lane lock, so once
    /// the check passes nothing is left to run: the next request serves at
    /// once and unload is not refused.
    #[tokio::test]
    #[cfg(unix)]
    async fn concurrent_requests_during_a_self_check_share_it_and_leave_no_backlog() {
        let fixture = ane_lane_fixture("ane-check-single-flight", false).await;
        *fixture
            .state
            .runtime
            .profile_check_passes_for_test
            .lock()
            .unwrap() = true;
        let state = fixture.state.clone();
        let (started, release) = ane_gate();
        fixture
            .control
            .hold_next_admission(started.clone(), release.clone());
        let requests = (0..10)
            .map(|_| {
                let state = state.clone();
                let model = ane_lane_model(&state);
                tokio::spawn(async move {
                    let deadline = tokio::time::Instant::now() + Duration::from_millis(100);
                    ensure_profile_request_certified(state, &model, Some(deadline)).await
                })
            })
            .collect::<Vec<_>>();
        let mut answers = Vec::new();
        for request in requests {
            answers.push(request.await.unwrap());
        }
        let compile_reached =
            tokio::time::timeout(Duration::from_secs(2), started.notified()).await;
        let tasks_during_check = state
            .runtime
            .lane_check_tasks
            .load(std::sync::atomic::Ordering::SeqCst);
        release.notify_one();
        until_lane_check_settles(&state).await;
        // Cached certification must finish in its first poll, not join or start
        // lane work. A wall-clock bound would measure runner scheduling instead.
        let next = {
            let model = ane_lane_model(&state);
            tokio::time::pause();
            let deadline = tokio::time::Instant::now() + Duration::from_millis(50);
            let certify = ensure_profile_request_certified(state.clone(), &model, Some(deadline));
            tokio::pin!(certify);
            let next = std::future::poll_fn(|cx| {
                std::task::Poll::Ready(std::future::Future::poll(certify.as_mut(), cx))
            })
            .await;
            tokio::time::resume();
            next
        };
        let no_check_queued = state.runtime.lane_checks.lock().unwrap().is_empty();
        let served = embed_on_ane_lane(
            &state,
            1,
            Some(tokio::time::Instant::now() + Duration::from_secs(2)),
        )
        .await;
        let tasks_after = state
            .runtime
            .lane_check_tasks
            .load(std::sync::atomic::Ordering::SeqCst);
        let unloaded = response_result(
            model_unload(state.clone(), json!({"model_id": ANE_TEST_LANE})).await,
            "model.unload",
        );
        finish_ane_lane_fixture(fixture).await;
        for answer in &answers {
            match answer {
                Err(error) => assert_eq!(error.code, "model_loading", "{error:?}"),
                Ok(()) => panic!("a request certified while the check was held"),
            }
        }
        assert!(compile_reached.is_ok(), "the check never ran");
        assert_eq!(tasks_during_check, 1, "one check task per lane");
        assert!(matches!(next, std::task::Poll::Ready(Ok(()))), "{next:?}");
        assert!(no_check_queued, "cached certification queued lane work");
        assert!(served.is_ok(), "{served:?}");
        assert_eq!(tasks_after, 1, "a passed check is served without a task");
        assert!(unloaded.get("error").is_none(), "{unloaded}");
    }

    /// The joined check finishes after the caller's deadline while the caller
    /// is not running; when it resumes, both the outcome and its timer are
    /// ready. The request is still expired.
    #[tokio::test]
    #[cfg(unix)]
    async fn a_self_check_finishing_after_the_deadline_still_answers_model_loading() {
        let fixture = ane_lane_fixture("ane-check-late-join", false).await;
        *fixture
            .state
            .runtime
            .profile_check_passes_for_test
            .lock()
            .unwrap() = true;
        let state = fixture.state.clone();
        let (started, release) = ane_gate();
        fixture
            .control
            .hold_next_admission(started.clone(), release.clone());
        // Freeze the caller's clock so subscribing cannot consume its budget.
        // A zero-duration timeout can poll again after rescheduling; poll exactly
        // once to start the check and subscribe without waiting for its result.
        tokio::time::pause();
        let deadline = tokio::time::Instant::now() + Duration::from_millis(100);
        let certify = ensure_profile_preload_ready(state.clone(), ANE_TEST_LANE, Some(deadline));
        tokio::pin!(certify);
        let first_poll = std::future::poll_fn(|cx| {
            std::task::Poll::Ready(std::future::Future::poll(certify.as_mut(), cx))
        })
        .await;
        assert!(
            first_poll.is_pending(),
            "the check must still be pending: {:?}",
            first_poll.map(|answer| answer.map(drop))
        );
        tokio::time::resume();
        let mut outcome = state
            .runtime
            .lane_checks
            .lock()
            .unwrap()
            .get(ANE_TEST_LANE)
            .cloned()
            .expect("a check is in flight");
        tokio::time::sleep_until(deadline + Duration::from_millis(20)).await;
        release.notify_one();
        let published =
            tokio::time::timeout(Duration::from_secs(10), outcome.wait_for(Option::is_some))
                .await
                .map(|outcome| outcome.map(|outcome| outcome.clone().unwrap().is_ok()));
        // The release permit can precede admission on a busy runner. Once the
        // check has published, admission must have consumed it and signalled.
        tokio::time::timeout(Duration::from_secs(2), started.notified())
            .await
            .unwrap();
        let answered = certify.await.map(drop);
        finish_ane_lane_fixture(fixture).await;
        assert!(
            matches!(published, Ok(Ok(true))),
            "the check must pass: {published:?}"
        );
        match answered {
            Err(error) => assert_eq!(error.code, "model_loading", "{error:?}"),
            Ok(()) => panic!("a check finished after the deadline answered success"),
        }
    }

    /// A profile model that is not a catalog lane, requested without a
    /// deadline, still waits on its self-check only until the request's
    /// default inline deadline.
    #[cfg(unix)]
    #[tokio::test(start_paused = true)]
    async fn a_non_catalog_profile_request_waits_on_certification_only_until_its_deadline() {
        let (dir, descriptor) = test_storage_descriptor("profile-certify-deadline");
        let store = Arc::new(SynapseStore::open(&descriptor).unwrap());
        let profile = test_machine_profile("profile-certify-deadline-os");
        store.activate_profile(&profile, 1, 1000).unwrap();
        let mut config = ModuleConfig::default();
        config.inline.deadline_ms = 300;
        let state = test_module_state_with_config(store, profile, config);
        let mut spec = catalog_fixture_config("gte-modernbert-base.owned-vulkan");
        spec.model_id = "profile-certify-deadline".into();
        let model = catalog_test_model(&dir, &spec);
        state.runtime.catalog.lock().unwrap().insert(
            spec.model_id.clone(),
            ModelSlot {
                spec,
                loaded: Some(model),
                state: ModelRuntimeState::Ready,
                notify: Arc::new(Notify::new()),
                last_cold_load_ms: None,
            },
        );
        // Stands in for a self-check holding the model's lifecycle lock.
        let lifecycle = catalog_lane_lock(&state.runtime, "profile-certify-deadline")
            .lock_owned()
            .await;
        let began = tokio::time::Instant::now();
        let outcome = tokio::time::timeout(
            Duration::from_secs(3),
            embed_query(
                state.clone(),
                json!({"model": "profile-certify-deadline", "text": "a"}),
            ),
        )
        .await;
        let elapsed = began.elapsed();
        drop(lifecycle);
        drop(state);
        let _ = fs::remove_dir_all(dir);
        let result = response_result(
            outcome.expect("certification ignored the deadline"),
            "embed.query",
        );
        assert_eq!(result["error"]["code"], "model_loading", "{result}");
        assert!(elapsed < Duration::from_secs(1), "{elapsed:?}");
    }

    /// A catalog batch job's loading task queues for the lane's lifecycle lock.
    /// If the job's budget runs out first, the task leaves the queue instead
    /// of loading the lane later, possibly after an operator unloaded it.
    #[tokio::test]
    #[cfg(unix)]
    async fn a_catalog_job_whose_budget_expired_leaves_the_lane_lock_queue() {
        let fixture = ane_lane_fixture("ane-job-budget", true).await;
        let state = fixture.state.clone();
        let (entry, backend) = resolved_catalog_lane(&state.runtime, ANE_TEST_LANE)
            .map(|(entry, backend)| (entry.clone(), backend.clone()))
            .unwrap();
        // The operator unloaded the lane; a load or self-check holds the lock.
        set_model_slot_state(&state.runtime, ANE_TEST_LANE, ModelRuntimeState::Unloaded);
        let lifecycle = catalog_lane_lock(&state.runtime, ANE_TEST_LANE)
            .lock_owned()
            .await;
        let deadline = tokio::time::Instant::now() + Duration::from_millis(100);
        let job = tokio::spawn({
            let state = state.clone();
            async move {
                ensure_catalog_lane_ready(state, entry, backend, Some(deadline))
                    .await
                    .map(drop)
            }
        });
        let answered = tokio::time::timeout(Duration::from_secs(2), job).await;
        drop(lifecycle);
        let lock_free = catalog_lane_lock(&state.runtime, ANE_TEST_LANE)
            .try_lock()
            .is_ok();
        finish_ane_lane_fixture(fixture).await;
        match answered {
            Ok(Ok(Err(error))) => assert_eq!(error.code, "deadline_exceeded", "{error:?}"),
            other => panic!("the expired job must leave the queue: {other:?}"),
        }
        assert!(lock_free, "nothing may stay queued on the lane lock");
    }

    #[cfg(unix)]
    fn ane_lane_entry(state: &ModuleState) -> (catalog::CatalogEntry, catalog::CatalogBackend) {
        resolved_catalog_lane(&state.runtime, ANE_TEST_LANE)
            .map(|(entry, backend)| (entry.clone(), backend.clone()))
            .expect("test lane is a catalog lane")
    }

    /// An expired catalog job must not load the lane, whether the lane lock is
    /// granted to it in the same poll as its expired timer or is simply free.
    #[tokio::test(start_paused = true)]
    #[cfg(unix)]
    async fn an_expired_catalog_job_never_loads_the_lane_once_it_holds_the_lock() {
        let fixture = ane_lane_fixture("ane-job-late-grant", true).await;
        let state = fixture.state.clone();
        let (entry, backend) = ane_lane_entry(&state);
        // The operator unloaded the lane and still holds its lifecycle lock.
        set_model_slot_state(&state.runtime, ANE_TEST_LANE, ModelRuntimeState::Unloaded);
        let unload = catalog_lane_lock(&state.runtime, ANE_TEST_LANE)
            .lock_owned()
            .await;
        let deadline = tokio::time::Instant::now() + Duration::from_millis(50);
        let job = ensure_catalog_lane_ready(
            state.clone(),
            entry.clone(),
            backend.clone(),
            Some(deadline),
        );
        tokio::pin!(job);
        // One poll queues the job for the lane lock.
        assert!(tokio::time::timeout(Duration::ZERO, &mut job)
            .await
            .is_err());
        tokio::time::sleep_until(deadline + Duration::from_millis(20)).await;
        // Released after the deadline: the lock goes to the queued job.
        drop(unload);
        let granted = job.await.map(drop);
        let after_grant = model_slot_snapshot(&state.runtime, ANE_TEST_LANE).unwrap();
        // A job already out of budget that finds the lock free.
        let expired_at = tokio::time::Instant::now();
        tokio::time::sleep(Duration::from_millis(5)).await;
        let free = ensure_catalog_lane_ready(state.clone(), entry, backend, Some(expired_at))
            .await
            .map(drop);
        let after_free = model_slot_snapshot(&state.runtime, ANE_TEST_LANE).unwrap();
        let lock_free = catalog_lane_lock(&state.runtime, ANE_TEST_LANE)
            .try_lock()
            .is_ok();
        finish_ane_lane_fixture(fixture).await;
        for (case, answer) in [("late grant", &granted), ("free lock", &free)] {
            match answer {
                Err(error) => assert_eq!(error.code, "deadline_exceeded", "{case}: {error:?}"),
                Ok(()) => panic!("{case}: an expired job readied the lane"),
            }
        }
        for (case, slot) in [("late grant", &after_grant), ("free lock", &after_free)] {
            assert!(
                matches!(slot.state, ModelRuntimeState::Unloaded) && slot.loaded.is_none(),
                "{case}: an expired job touched the unloaded lane: {:?}",
                slot.state
            );
        }
        assert!(lock_free);
    }

    /// An admitted catalog job that is already out of budget is still failed
    /// with deadline_exceeded and removed from the catalog job registry.
    #[tokio::test]
    #[cfg(unix)]
    async fn an_expired_catalog_job_is_failed_and_unregistered() {
        let fixture = ane_lane_fixture("ane-job-finalize", true).await;
        let state = fixture.state.clone();
        let (entry, backend) = ane_lane_entry(&state);
        let params = json!({"request_key": "expired-catalog-job", "texts": ["a"]});
        // Admitted long enough ago that its one-millisecond budget has run out.
        let admission = state
            .store
            .admit_job(
                "expired-catalog-job",
                "sha256:expired-catalog-job",
                "embed.batch",
                state.module_generation,
                &params,
                now_ms() - 10_000,
                1,
                60_000,
            )
            .unwrap();
        let JobAdmission::Admitted(record) = admission else {
            panic!("the job was not admitted")
        };
        state
            .runtime
            .catalog_jobs
            .lock()
            .unwrap()
            .insert(record.job_id.clone(), entry.id.clone());
        run_catalog_embed_job(
            state.clone(),
            record.clone(),
            entry,
            backend,
            params,
            "sha256:expired-catalog-job".into(),
        )
        .await;
        let stored = state.store.get_job(&record.job_id).unwrap().unwrap();
        let registered = state
            .runtime
            .catalog_jobs
            .lock()
            .unwrap()
            .contains_key(&record.job_id);
        finish_ane_lane_fixture(fixture).await;
        assert!(stored.is_terminal_failed(), "{}", stored.state);
        assert_eq!(
            stored
                .error_json
                .as_ref()
                .map(|error| error["code"].clone()),
            Some(json!("deadline_exceeded"))
        );
        assert!(!registered, "the expired job stayed registered");
    }

    /// A caller sees the check pending, then the check passes and leaves the
    /// in-flight map before the caller joins. Joining must find the passed
    /// check rather than start a second one.
    #[tokio::test]
    #[cfg(unix)]
    async fn a_caller_that_joins_after_the_check_passed_does_not_start_another() {
        let fixture = ane_lane_fixture("ane-check-stale-fast-path", false).await;
        *fixture
            .state
            .runtime
            .profile_check_passes_for_test
            .lock()
            .unwrap() = true;
        let state = fixture.state.clone();
        let model = ane_lane_model(&state);
        // On this current-thread runtime, one poll registers the in-flight
        // check before its spawned task can run. The second caller can then
        // find no cached passed result and suspend before joining the check,
        // without waiting for a notification from the external worker.
        let first = ensure_profile_request_certified(state.clone(), &model, None);
        tokio::pin!(first);
        let first_poll = std::future::poll_fn(|cx| {
            std::task::Poll::Ready(std::future::Future::poll(first.as_mut(), cx))
        })
        .await;
        assert!(first_poll.is_pending(), "{first_poll:?}");
        let (missed, join) = ane_gate();
        *state.runtime.fast_path_gate.lock().unwrap() = Some((missed.clone(), join.clone()));
        let second = ensure_profile_request_certified(state.clone(), &model, None);
        tokio::pin!(second);
        let second_poll = std::future::poll_fn(|cx| {
            std::task::Poll::Ready(std::future::Future::poll(second.as_mut(), cx))
        })
        .await;
        assert!(second_poll.is_pending(), "{second_poll:?}");
        let missed = missed.notified();
        tokio::pin!(missed);
        let missed_poll = std::future::poll_fn(|cx| {
            std::task::Poll::Ready(std::future::Future::poll(missed.as_mut(), cx))
        })
        .await;
        assert!(
            missed_poll.is_ready(),
            "the second caller did not miss the fast path"
        );
        // Only now drive the worker. The second future stays suspended until
        // the passed check has left the in-flight map.
        let first = tokio::time::timeout(Duration::from_secs(10), first).await;
        until_lane_check_settles(&state).await;
        join.notify_one();
        let second = std::future::poll_fn(|cx| {
            std::task::Poll::Ready(std::future::Future::poll(second.as_mut(), cx))
        })
        .await;
        let tasks = state
            .runtime
            .lane_check_tasks
            .load(std::sync::atomic::Ordering::SeqCst);
        finish_ane_lane_fixture(fixture).await;
        assert!(matches!(first, Ok(Ok(()))), "{first:?}");
        assert!(
            matches!(second, std::task::Poll::Ready(Ok(()))),
            "{second:?}"
        );
        assert_eq!(tasks, 1, "a passed check was started again");
    }

    /// After a lane's self-check has passed, serving many-row batches runs the
    /// check no more, sends the worker nothing but the batches' own inference,
    /// and keeps the one engine and worker process the lane was loaded with.
    #[tokio::test]
    #[cfg(unix)]
    async fn serving_many_row_batches_after_the_check_passed_runs_no_check_and_one_worker() {
        let fixture = ane_lane_fixture("ane-batches-after-check", false).await;
        *fixture
            .state
            .runtime
            .profile_check_passes_for_test
            .lock()
            .unwrap() = true;
        let state = fixture.state.clone();
        let loaded = ane_lane_model(&state);
        let later = || Some(tokio::time::Instant::now() + Duration::from_secs(10));
        // The first request runs the pending check.
        ensure_profile_request_certified(state.clone(), &loaded, later())
            .await
            .unwrap();
        until_lane_check_settles(&state).await;
        let passed_recorded = profile_check_passed(&state, ANE_TEST_LANE)
            .unwrap()
            .is_some_and(|model| Arc::ptr_eq(&model, &loaded));
        let checks_after_pass = state
            .runtime
            .lane_check_tasks
            .load(std::sync::atomic::Ordering::SeqCst);
        let inferences_after_pass = fixture.control.inferences();
        const BATCHES: u64 = 4;
        for batch in 0..BATCHES {
            ensure_profile_request_certified(state.clone(), &loaded, later())
                .await
                .unwrap();
            let items = (0..64).map(|row| vec![1; 8 + row % 24]).collect::<Vec<_>>();
            let vectors =
                execute_embedding(&state.runtime, &loaded, TokenBatch { items }, later(), None)
                    .await
                    .unwrap_or_else(|error| panic!("batch {batch}: {error:?}"));
            assert_eq!(vectors.len(), 64);
        }
        let checks_after_batches = state
            .runtime
            .lane_check_tasks
            .load(std::sync::atomic::Ordering::SeqCst);
        let inferences_for_batches = fixture.control.inferences() - inferences_after_pass;
        let worker_starts = fixture.control.worker_starts();
        let restarts = fixture.engine.serving.supervisor.stats().restarts;
        let still_same_engine = model_slot_snapshot(&state.runtime, ANE_TEST_LANE)
            .and_then(|slot| slot.loaded)
            .is_some_and(|model| Arc::ptr_eq(&model, &loaded));
        drop(loaded);
        finish_ane_lane_fixture(fixture).await;
        assert!(
            passed_recorded,
            "the passed check was not recorded for the lane"
        );
        assert_eq!(checks_after_pass, 1);
        assert_eq!(
            checks_after_batches, 1,
            "the self-check ran again after passing"
        );
        // Every row pads to the same 128-token shape, so each batch is exactly
        // one engine inference and no extra (self-check) inference ran.
        assert_eq!(inferences_for_batches, BATCHES);
        assert_eq!(worker_starts, 1, "a second worker process was started");
        assert_eq!(restarts, 0);
        assert!(still_same_engine, "the lane was loaded again");
    }

    #[tokio::test(start_paused = true)]
    #[cfg(unix)]
    async fn profile_certification_waits_for_a_lane_self_check_only_until_its_deadline() {
        // A pending check, so the request must wait for the lane.
        let fixture = ane_lane_fixture("ane-certify-deadline", false).await;
        let state = fixture.state.clone();
        // Holding the lane's lifecycle lock simulates a load or self-check in
        // progress.
        let lifecycle = catalog_lane_lock(&state.runtime, ANE_TEST_LANE)
            .lock_owned()
            .await;
        let started = tokio::time::Instant::now();
        let waited = tokio::time::timeout(
            Duration::from_secs(2),
            ensure_profile_preload_ready(
                state.clone(),
                ANE_TEST_LANE,
                Some(tokio::time::Instant::now() + Duration::from_millis(150)),
            ),
        )
        .await
        .map(|result| result.map(drop));
        let elapsed = started.elapsed();
        drop(lifecycle);
        finish_ane_lane_fixture(fixture).await;
        match waited {
            Ok(Err(error)) => assert_eq!(error.code, "model_loading", "{error:?}"),
            other => panic!("certification must answer at the request deadline: {other:?}"),
        }
        assert!(elapsed < Duration::from_secs(1), "{elapsed:?}");
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn cancelled_direct_ane_keeps_module_permit_and_inflight_until_reply_drains() {
        for rerank in [false, true] {
            let (root, _) = test_storage_descriptor("ane-owned-guards");
            let spec = catalog_fixture_config("gte-modernbert-base.ane-direct-worker");
            let mut model = catalog_test_model(&root, &spec);
            let started = Arc::new(Notify::new());
            let release = Arc::new(Notify::new());
            let engine = tokio::task::spawn_blocking({
                let started = started.clone();
                let release = release.clone();
                move || worker_host::ane_residency::module_mock_engine(started, release)
            })
            .await
            .unwrap();
            Arc::get_mut(&mut model).unwrap().backend = EmbedBackend::DirectAne(engine.clone());
            let mut runtime = RuntimeState::from_catalog(ModuleConfig::default(), vec![]).unwrap();
            runtime.execution = Arc::new(Semaphore::new(1));
            let runtime = Arc::new(runtime);
            let task = tokio::spawn({
                let model = model.clone();
                let runtime = runtime.clone();
                async move {
                    if rerank {
                        execute_rerank(
                            &runtime,
                            &model,
                            RerankRequest::default(),
                            Some(vec![vec![1]]),
                            None,
                            None,
                        )
                        .await
                        .map(|_| ())
                    } else {
                        execute_embedding(
                            &runtime,
                            &model,
                            TokenBatch {
                                items: vec![vec![1]],
                            },
                            None,
                            None,
                        )
                        .await
                        .map(|_| ())
                    }
                }
            });
            tokio::time::timeout(Duration::from_secs(2), started.notified())
                .await
                .unwrap();
            task.abort();
            assert!(task.await.unwrap_err().is_cancelled());
            let available = runtime.execution.available_permits();
            let in_flight = runtime.execution_stats.lock().unwrap().in_flight;
            release.notify_one();
            tokio::time::timeout(Duration::from_secs(2), async {
                while runtime.execution.available_permits() == 0 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            assert_eq!(
                available, 0,
                "cancelled caller must not release an executing ANE task's permit"
            );
            assert_eq!(in_flight, 1);
            assert_eq!(runtime.execution_stats.lock().unwrap().in_flight, 0);
            tokio::task::spawn_blocking(move || engine.unload())
                .await
                .unwrap()
                .unwrap();
            drop(model);
            fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    #[cfg(unix)]
    fn cold_ane_supervisor_uses_the_runtime_fixture_lock_path() {
        let (root, descriptor) = test_storage_descriptor("ane-runtime-lock-path");
        let store = Arc::new(SynapseStore::open(&descriptor).unwrap());
        let profile = test_machine_profile("ane-runtime-lock-path-os");
        store.activate_profile(&profile, 1, 1000).unwrap();
        let state = test_module_state(store, profile);
        let path = runtime_ane_lock_path(&state.runtime).unwrap().to_path_buf();
        assert_eq!(
            path.parent().unwrap().canonicalize().unwrap(),
            root.canonicalize().unwrap()
        );
        assert!(state.runtime.ane_supervisor.lock().unwrap().is_none());
        let first = if cfg!(target_os = "macos") {
            direct_ane_supervisor(&state.runtime, "ane-direct-worker")
                .unwrap()
                .unwrap()
        } else {
            acquire_runtime_ane_supervisor(&state.runtime).unwrap()
        };
        assert!(
            worker_host::ane_residency::AneDirectLaneLock::acquire(&path).is_err(),
            "runtime acquisition must hold the configured fixture lock"
        );
        drop((first, state));
        let released = worker_host::ane_residency::AneDirectLaneLock::acquire(&path).unwrap();
        drop(released);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn unconfigured_ane_runtime_cannot_resolve_a_production_lock_in_tests() {
        let runtime = RuntimeState::from_catalog(ModuleConfig::default(), vec![]).unwrap();
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| runtime_ane_lock_path(
                &runtime
            )))
            .is_err(),
            "an unconfigured unit test must fail before resolving a production lock path"
        );
    }

    #[test]
    fn stopped_ane_scheduler_maps_to_engine_crashed_with_the_lane_id() {
        let wire = ane_lane_error_to_wire(
            worker_host::ane_residency::AneResidencyError::SchedulerStopped,
            "lane-a",
        );
        assert_eq!(wire.code, "engine_crashed");
        assert_eq!(
            wire.details,
            Some(json!({"lane_id": "lane-a", "scheduler_stopped": true}))
        );
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn direct_ane_models_reuse_one_runtime_supervisor() {
        let runtime = RuntimeState::from_catalog(ModuleConfig::default(), vec![]).unwrap();
        *runtime.ane_supervisor.lock().unwrap() = Some(
            worker_host::ane_residency::AneResidencySupervisor::new(Default::default()),
        );
        assert!(direct_ane_supervisor(&runtime, "owned-metal")
            .unwrap()
            .is_none());
        let mut first = direct_ane_supervisor(&runtime, "ane-direct-worker")
            .unwrap()
            .unwrap();
        let second = direct_ane_supervisor(&runtime, "ane-direct-worker")
            .unwrap()
            .unwrap();
        assert!(second.observations().is_none());
        first.enable_observations();
        assert_eq!(second.observations(), Some((vec![], 0)));
        assert_eq!(certify_worker_snapshot(&runtime)["admitted_count"], 0);
    }

    /// Every owned engine's worker loads a converted safetensors profile
    /// package. A missing arm falls through to the legacy "onnx" default,
    /// which the direct-ANE worker refuses outright and which would make a
    /// downloaded package fail the ONNX header check.
    #[test]
    fn owned_engines_default_to_the_safetensors_package_format() {
        for engine in [
            "owned-metal",
            "owned-cuda",
            "owned-vulkan",
            synapse_core::ANE_DIRECT_WORKER_ENGINE,
        ] {
            assert_eq!(
                default_artifact_format(engine),
                "safetensors-package",
                "{engine}"
            );
        }
    }

    #[test]
    fn ane_wire_errors_keep_resource_retry_and_uncertain_io_unsafe() {
        use worker_host::ane_residency::AneResidencyError;
        let wire = ane_residency_error_to_wire(AneResidencyError::WorkerErr {
            code: "ane_resources_exhausted".into(),
            msg: "hardware full".into(),
        });
        assert_eq!(wire.code, "ane_resources_exhausted");
        assert_eq!(wire.class, ErrorClass::Transient);
        assert_eq!(wire.retry_after_ms, Some(250));
        assert!(wire.safe_to_retry_same_request);
        let wire = ane_residency_error_to_wire(AneResidencyError::Channel(
            "partial inference response".into(),
        ));
        assert!(!wire.safe_to_retry_same_request);
    }

    #[test]
    fn certify_observation_query_refuses_when_disabled() {
        let (root, descriptor) = test_storage_descriptor("certify-disabled");
        let store = Arc::new(SynapseStore::open(&descriptor).unwrap());
        let profile = test_machine_profile("certify-disabled-os");
        store.observe_profile(&profile, 10, 1).unwrap();
        let state = test_module_state(store, profile);
        let HandlerOutcome::Error { code, .. } = certify_observations(state) else {
            panic!("disabled observation must refuse");
        };
        assert_eq!(code, "certify_observation_disabled");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn certify_observation_on_discloses_engine_ids() {
        let mut response = serde_json::json!({"payload": {"scores": [0.5]}});
        let observation =
            serde_json::json!({"input_ids": [[1, 2, 3]], "readout_ids": [9693, 2152]});
        super::attach_certify_observation(&mut response, Some(observation.clone()));
        assert_eq!(response["observation"], observation);
        let config: super::ModuleConfig =
            serde_json::from_value(serde_json::json!({"certify_observation": true})).unwrap();
        assert!(config.certify_observation);
    }
    use super::*;

    /// Builds a bind whose scope is decoded from the daemon's wire JSON, so the
    /// test also checks that the linked subc-protocol accepts a scope carrying
    /// `flow_id` (protocols before 0.29 refuse that unknown field).
    fn bind_with_scope(channel: u16, flow_id: Option<&str>) -> RouteBindRequest {
        let mut attributes = json!({"agent_id": "agent-1"});
        if let Some(flow_id) = flow_id {
            attributes["flow_id"] = json!(flow_id);
        }
        let stamp: subc_protocol::scope::ScopeStamp = serde_json::from_value(json!({
            "owner": {"kind": "reserved", "module_id": "prefrontal-core"},
            "ref": "head-1",
            "scope_epoch": 3,
            "kind": "worker",
            "attributes": attributes,
            "owner_authorized": true
        }))
        .expect("decode a scope stamp from wire JSON");
        RouteBindRequest::new(
            RouteHandle::detached(channel, 1),
            subc_protocol::RouteTarget::ManagementSurface {
                module_id: "synapse".into(),
            },
            subc_protocol::BindIdentity::new("synapse-test", "session-1"),
        )
        .with_scope(stamp)
    }

    fn refusal_code(outcome: Option<HandlerOutcome>) -> Option<String> {
        match outcome {
            Some(HandlerOutcome::Error { code, .. }) => Some(code),
            Some(_) => Some("non-error outcome".into()),
            None => None,
        }
    }

    #[tokio::test(start_paused = true)]
    async fn acknowledging_cached_quarantine_notifies_findings_release() {
        let spec = stuck_model_spec();
        let id = spec.model_id.clone();
        let runtime = RuntimeState::from_catalog(ModuleConfig::default(), vec![spec]).unwrap();
        set_model_slot_state(
            &runtime,
            &id,
            ModelRuntimeState::Failed(WireOperationError::from_stable(
                StableError::engine_crashed(Some(250)),
                "worker is quarantined",
            )),
        );
        assert!(tokio::time::timeout(
            Duration::from_millis(1),
            runtime.findings_changed.notified()
        )
        .await
        .is_ok());
        assert!(catalog_failed_load(&runtime, &id, true).is_some());
        assert!(tokio::time::timeout(
            Duration::from_millis(1),
            runtime.findings_changed.notified()
        )
        .await
        .is_ok());
        assert!(catalog_failed_load(&runtime, &id, true).is_none());
        assert!(tokio::time::timeout(
            Duration::from_millis(1),
            runtime.findings_changed.notified()
        )
        .await
        .is_err());
    }

    #[test]
    fn flow_routes_serve_diagnostics_and_event_bodies_only() {
        let handler = SynapseHandler::new("synapse".into());
        let inner = &handler.inner;
        let flow = bind_with_scope(1, Some("flow-7"));
        let plain = bind_with_scope(2, None);
        inner.record_bind_scope(&flow);
        inner.record_bind_scope(&plain);

        let operations = management_operations();
        let manifest = manifest("synapse");
        assert_eq!(
            manifest.flow_operations,
            Some(vec!["diagnostics.findings".into(), "events_get".into()])
        );
        assert_eq!(
            manifest.capabilities.as_ref().unwrap().provides,
            vec!["flow-scopes/v1", "media-provider/v1", "bind-no-project/v1"]
        );
        manifest.validate_flow_operations().unwrap();
        assert!(operations.iter().any(
            |op| op.name == "diagnostics.findings" && op.kind == ManagementOperationKind::Query
        ));
        assert!(operations
            .iter()
            .any(|op| op.name == "events_get" && op.kind == ManagementOperationKind::Query));
        assert_eq!(
            manifest.events,
            Some(vec![subc_protocol::manifest::EventDeclaration::new(
                "findings_changed",
                1
            )
            .with_headers(vec!["event_key".into()])])
        );
        assert!(operations.iter().any(|op| op.name == "embed.query"));
        for op in &operations {
            let on_flow = refusal_code(inner.flow_refusal(&flow.handle, &op.name));
            if matches!(op.name.as_str(), "diagnostics.findings" | "events_get") {
                assert_eq!(
                    on_flow, None,
                    "diagnosis and event bodies must be served to a flow"
                );
            } else {
                assert_eq!(
                    on_flow.as_deref(),
                    Some("flow_scope_refused"),
                    "{} must be refused on a flow route, even if it is a query",
                    op.name
                );
            }
            assert_eq!(
                refusal_code(inner.flow_refusal(&plain.handle, &op.name)),
                None,
                "{} on a route without flow_id must be unaffected",
                op.name
            );
        }
        // Undeclared names, including the short decode aliases dispatch also
        // accepts, are refused on a flow route rather than served by default.
        for undeclared in ["decode", "admit_session", "no.such.method"] {
            assert_eq!(
                refusal_code(inner.flow_refusal(&flow.handle, undeclared)).as_deref(),
                Some("flow_scope_refused"),
                "{undeclared}"
            );
        }
        let mut head = bind_with_scope(3, None);
        head.scope.as_mut().unwrap().kind = subc_protocol::scope::ScopeKind::Head;
        let mut unauthorized_flow = bind_with_scope(4, Some("flow-8"));
        unauthorized_flow.scope.as_mut().unwrap().owner_authorized = false;
        inner.record_bind_scope(&head);
        inner.record_bind_scope(&unauthorized_flow);
        let routes = inner.diagnostics_routes.lock().unwrap();
        assert_eq!(routes.get(&flow.handle), Some(&true));
        assert_eq!(
            routes.get(&plain.handle),
            Some(&false),
            "bare workers cannot read diagnostics"
        );
        assert_eq!(routes.get(&head.handle), Some(&true));
        assert_eq!(routes.get(&unauthorized_flow.handle), Some(&false));
    }

    #[tokio::test]
    async fn authorized_flow_can_fetch_committed_findings_body() {
        let (root, descriptor) = test_storage_descriptor("flow-events-get");
        let store = Arc::new(SynapseStore::open(&descriptor).unwrap());
        store
            .record_findings_digest("stale", "error", 1700000000000)
            .unwrap();
        let key: String = store
            .sqlite()
            .with_conn(|c| c.query_row("SELECT event_id FROM event_outbox", [], |r| r.get(0)))
            .unwrap();
        let profile = test_machine_profile("test-os");
        store.observe_profile(&profile, 1, 1).unwrap();
        let state = test_module_state(store.clone(), profile);
        let handler = SynapseHandler::new("synapse".into());
        assert!(handler.inner.state.set(state.clone()).is_ok());
        let route = bind_with_scope(5, Some("findings-flow"));
        handler.on_bind(&route).await;
        assert!(handler
            .inner
            .flow_refusal(&route.handle, "events_get")
            .is_none());
        assert_eq!(
            handler
                .inner
                .diagnostics_routes
                .lock()
                .unwrap()
                .get(&route.handle),
            Some(&true)
        );
        let response = response_result(
            dispatch_request(
                state.clone(),
                MethodEnvelope {
                    method: "events_get".into(),
                    params: json!({"event_key":key,"event_name":"findings_changed","version":1}),
                },
            )
            .await,
            "events_get",
        );
        let body: Value = serde_json::from_str(response["body"].as_str().unwrap()).unwrap();
        assert_eq!(body["event_id"], key);
        assert_eq!(body["worst_severity"], "error");
        drop(handler);
        drop(state);
        drop(store);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn the_flow_refusal_names_the_method_and_the_flow() {
        let handler = SynapseHandler::new("synapse".into());
        let flow = bind_with_scope(3, Some("flow-9"));
        handler.inner.record_bind_scope(&flow);
        match handler.inner.flow_refusal(&flow.handle, "model.unload") {
            Some(HandlerOutcome::Error { code, message }) => {
                assert_eq!(code, "flow_scope_refused");
                assert!(message.contains("model.unload"), "{message}");
                assert!(message.contains("flow-9"), "{message}");
            }
            _ => panic!("model.unload on a flow route must be refused"),
        }
        // A rebind of the same handle without flow_id clears the record.
        handler.inner.record_bind_scope(&bind_with_scope(3, None));
        assert!(handler
            .inner
            .flow_refusal(&flow.handle, "model.unload")
            .is_none());
    }

    #[test]
    fn cuda_floor_probe_retains_child_failure_and_rejects_bad_json() {
        let mut failed = std::process::Command::new(if cfg!(windows) { "cmd.exe" } else { "sh" });
        if cfg!(windows) {
            failed.args(["/D", "/C", "echo driver unavailable 1>&2 & exit /b 7"]);
        } else {
            failed.args(["-c", "echo 'driver unavailable' >&2; exit 7"]);
        }
        let error = run_owned_cuda_probe(&mut failed, Duration::from_secs(2)).unwrap_err();
        assert!(error.contains("driver unavailable"), "{error}");
        assert!(error.contains("exited"), "{error}");
        let mut malformed =
            std::process::Command::new(if cfg!(windows) { "cmd.exe" } else { "sh" });
        if cfg!(windows) {
            malformed.args(["/D", "/C", "echo invalid-json"]);
        } else {
            malformed.args(["-c", "echo invalid-json"]);
        }
        let error = run_owned_cuda_probe(&mut malformed, Duration::from_secs(2)).unwrap_err();
        assert!(error.contains("invalid CUDA floor probe JSON"), "{error}");
    }

    #[test]
    fn cuda_floor_probe_cache_is_keyed_per_worker_binary() {
        let header = if cfg!(windows) {
            "@echo off\r\n"
        } else {
            "#!/bin/sh\n"
        };
        let json = r#"{"status":"ok","code":null,"required":{},"observed":{"driver_api":13030,"compute_capability":{"major":8,"minor":9}}}"#;
        let success = if cfg!(windows) {
            format!("{header}echo {json}\r\n")
        } else {
            format!("{header}echo '{json}'\n")
        };
        let failure_script = format!(
            "{header}echo missing-library >&2\n{}\n",
            if cfg!(windows) { "exit /b 9" } else { "exit 9" }
        );
        let good_name = if cfg!(windows) {
            "ck-good.cmd"
        } else {
            "ck-good.sh"
        };
        let bad_name = if cfg!(windows) {
            "ck-bad.cmd"
        } else {
            "ck-bad.sh"
        };
        let good = synapse_core::dev_binary::ckdev_script(good_name, success.as_bytes()).unwrap();
        let bad =
            synapse_core::dev_binary::ckdev_script(bad_name, failure_script.as_bytes()).unwrap();

        // Restore bad's original failure script in case a previous run was interrupted
        if fs::read(&bad).ok().as_deref() != Some(failure_script.as_bytes()) {
            fs::write(&bad, &failure_script).unwrap();
        }

        let failure = owned_cuda_probe_floor(Some(&bad)).unwrap_err();
        assert!(failure.contains("missing-library"), "{failure}");
        let reading = owned_cuda_probe_floor(Some(&good)).unwrap();
        assert_eq!(
            (
                reading.driver_api,
                reading.compute_major,
                reading.compute_minor
            ),
            (13030, 8, 9)
        );
        // Failure caching is deliberate, but must not contaminate another worker.
        fs::write(&bad, &success).unwrap();
        assert_eq!(owned_cuda_probe_floor(Some(&bad)).unwrap_err(), failure);
        assert_eq!(
            owned_cuda_probe_floor(Some(&good)).unwrap().driver_api,
            13030
        );

        // Restore bad's script content so subsequent test runs reuse it without rewriting
        fs::write(&bad, &failure_script).unwrap();
    }

    #[test]
    fn cuda_floor_probe_slow_worker_does_not_block_other_workers() {
        let root = std::env::temp_dir().join(format!(
            "synapse-probe-slow-{}-{}",
            std::process::id(),
            TEST_STATE_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        let ready = root.join("ready");
        let release = root.join("release");
        std::env::set_var("SYNAPSE_PROBE_SLOW_READY", &ready);
        std::env::set_var("SYNAPSE_PROBE_SLOW_RELEASE", &release);

        let json = r#"{"status":"ok","code":null,"required":{},"observed":{"driver_api":13030,"compute_capability":{"major":8,"minor":9}}}"#;
        let stalled = if cfg!(windows) {
            format!(
                "@echo off\r\necho ready > \"%SYNAPSE_PROBE_SLOW_READY%\"\r\n:wait\r\nif exist \"%SYNAPSE_PROBE_SLOW_RELEASE%\" goto done\r\nping -n 2 127.0.0.1 >nul\r\ngoto wait\r\n:done\r\necho {json}\r\n"
            )
        } else {
            format!(
                "#!/bin/sh\nprintf ready > \"$SYNAPSE_PROBE_SLOW_READY\"\nwhile [ ! -f \"$SYNAPSE_PROBE_SLOW_RELEASE\" ]; do sleep 0.05; done\necho '{json}'\n"
            )
        };
        let success = if cfg!(windows) {
            format!("@echo off\r\necho {json}\r\n")
        } else {
            format!("#!/bin/sh\necho '{json}'\n")
        };
        let slow_name = if cfg!(windows) {
            "ck-slow.cmd"
        } else {
            "ck-slow.sh"
        };
        let quick_name = if cfg!(windows) {
            "ck-quick.cmd"
        } else {
            "ck-quick.sh"
        };
        let slow = synapse_core::dev_binary::ckdev_script(slow_name, stalled.as_bytes()).unwrap();
        let quick = synapse_core::dev_binary::ckdev_script(quick_name, success.as_bytes()).unwrap();

        let stalled = std::thread::spawn(move || owned_cuda_probe_floor(Some(&slow)));
        let ready_deadline = Instant::now() + Duration::from_secs(3);
        while !ready.exists() && Instant::now() < ready_deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        let confirmed_ready = ready.exists();
        let (tx, rx) = std::sync::mpsc::channel();
        let other = std::thread::spawn(move || {
            let _ = tx.send(owned_cuda_probe_floor(Some(&quick)));
        });
        // The quick probe must finish while the first worker is still blocked.
        let result = rx.recv_timeout(Duration::from_secs(3));
        fs::write(&release, "release").unwrap();
        let stalled_result = stalled.join().unwrap();
        other.join().unwrap();
        assert!(confirmed_ready, "stalled worker did not signal readiness");
        assert_eq!(stalled_result.unwrap().driver_api, 13030);
        assert_eq!(
            result
                .expect("unrelated probe blocked behind stalled worker")
                .unwrap()
                .driver_api,
            13030
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[ignore = "requires a staged CUDA worker and supported GPU; run explicitly"]
    fn cuda_floor_probe_matches_real_worker_binary_output() {
        let worker = env::var_os("SYNAPSE_TEST_CUDA_WORKER")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../target/release")
                    .join(if cfg!(windows) {
                        "ck-synapse-worker-cuda.exe"
                    } else {
                        "ck-synapse-worker-cuda"
                    })
            });
        assert!(
            worker.is_file(),
            "stage CUDA worker at {}",
            worker.display()
        );
        let scratch = env::temp_dir().join(format!("synapse-cuda-floor-{}", std::process::id()));
        let worker = synapse_core::dev_binary::ckdev_binary(worker, &scratch).unwrap();
        let reading = run_owned_cuda_probe(
            std::process::Command::new(&worker).arg("--probe-floor"),
            Duration::from_secs(10),
        )
        .expect("real worker probe");
        assert!(reading.driver_api >= synapse_core::OWNED_CUDA_MINIMUM_DRIVER_API);
        assert!(
            reading.compute_major as f32 + reading.compute_minor as f32 / 10.0
                >= synapse_core::OWNED_CUDA_MINIMUM_DEVICE_CC
        );
    }

    #[test]
    fn cuda_floor_failure_evidence_preserves_stderr_without_fabricating_hardware() {
        let unavailable = CudaFloorDecision::Unsupported {
            reason: synapse_core::CudaUnsupportedReason::HardwareUnavailable,
            observed: None,
        };
        let observed =
            floor_observed_with_probe_error(&unavailable, Some("CUDA driver unavailable"));
        assert_eq!(observed["probe_stderr"], "CUDA driver unavailable");
        assert!(observed.get("driver_api").is_none());
        let below_floor = evaluate_cuda_floor(11000, 8, 9, None);
        let observed = floor_observed_with_probe_error(&below_floor, Some("stale error"));
        assert_eq!(observed["driver_api"], 11000);
        assert!(observed.get("probe_stderr").is_none());
    }

    #[test]
    fn wire_contract_documents_every_management_operation() {
        const CONTRACT: &str = include_str!("../../../docs/wire-contract-v1.md");
        // The contract lists the registry as one backticked paragraph that
        // starts with this sentence. Normalise CRLF so a Windows checkout
        // still ends the paragraph at the blank line.
        let contract = CONTRACT.replace("\r\n", "\n");
        let documented = contract
            .split("The management registry in this snapshot is ")
            .nth(1)
            .expect("wire contract keeps its management registry inventory")
            .split("\n\n")
            .next()
            .unwrap()
            .split('`')
            .skip(1)
            .step_by(2)
            .map(str::to_owned)
            .collect::<BTreeSet<_>>();
        let registered = management_operations()
            .into_iter()
            .map(|operation| operation.name)
            .collect::<BTreeSet<_>>();
        assert_eq!(
            registered, documented,
            "the registered management operations and the wire contract's inventory differ"
        );
    }

    #[test]
    fn readme_lists_every_workspace_production_crate() {
        // A document that lists a declared surface silently omits members
        // added after it was written. The enumeration source here is the workspace
        // manifest's member list - the same source Cargo builds from - so a new
        // production crate cannot ship without joining the README sentence
        // this test reads.
        const README: &str = include_str!("../../../README.md");
        const WORKSPACE_MANIFEST: &str = include_str!("../../../Cargo.toml");

        let unlisted = WORKSPACE_MANIFEST
            .lines()
            .filter_map(|line| {
                let member = line.trim().trim_matches(|c| c == '"' || c == ',');
                member
                    .strip_prefix("crates/")
                    .map(|crate_name| crate_name.to_owned())
            })
            .filter(|crate_name| !README.contains(&format!("`{crate_name}`")))
            .collect::<Vec<_>>();
        assert!(
            unlisted.is_empty(),
            "README production-crate list is missing workspace members: {unlisted:?}"
        );
    }

    #[test]
    fn manifest_provenance_declares_only_facts_this_binary_knows() {
        // Print the compiled inputs so a reader of the test log sees the
        // artifact's stamp rather than the checkout's current git status.
        println!(
            "SYNAPSE_BUILD_REV={}; SYNAPSE_BUILD_TREE={}",
            option_env!("SYNAPSE_BUILD_REV").unwrap_or("absent"),
            option_env!("SYNAPSE_BUILD_TREE").unwrap_or("absent")
        );
        let provenance = manifest("synapse")
            .provenance
            .expect("an SDK module always has at least one honest provenance fact");

        // The referent is the linked subc-protocol crate -- the fleet's shared
        // wire vocabulary -- never synapse's own version, which would be a
        // real number from the wrong numbering space and would read as correct
        // to any check that inspects shape rather than meaning.
        assert_eq!(
            provenance.wire_crate_version.as_deref(),
            Some(subc_client_rs::SUBC_PROTOCOL_CRATE_VERSION),
            "wire_crate_version must name the linked SDK crate"
        );

        // The commit is declared only for a clean tree; a dirty tree or a
        // build without git declares it absent, with the reason. A sentinel
        // string like "unknown" would be a well-formed lie.
        match build_git_sha_source() {
            BuildGitShaSource::Git {
                revision,
                tree_state: GitTreeState::Clean,
            } => {
                assert_eq!(provenance.build_git_sha.as_deref(), Some(revision));
                assert_eq!(provenance.build_git_sha_absence_reason, None);
            }
            BuildGitShaSource::Git { .. } => {
                assert_eq!(provenance.build_git_sha, None);
                assert_eq!(
                    provenance.build_git_sha_absence_reason,
                    Some(subc_protocol::manifest::BuildGitShaAbsenceReason::DeclinedDirty)
                );
            }
            _ => {
                assert_eq!(provenance.build_git_sha, None);
                assert!(provenance.build_git_sha_absence_reason.is_some());
            }
        }
        assert_eq!(provenance.build_lock_digest, None);

        // Deliberately NOT asserting the number itself: restating a derived
        // value here would make this test agree with the code by construction
        // and pass whatever the migration list said. Shape and presence are
        // what this can check honestly; the derivation is what keeps the value
        // true.
        let schema_version = provenance
            .store_schema_version
            .as_deref()
            .expect("a module with a migration list can state its newest migration");
        assert!(
            schema_version
                .parse::<u32>()
                .is_ok_and(|version| version > 0),
            "store_schema_version must be a real migration number, got {schema_version:?}"
        );

        provenance
            .validate()
            .expect("declared provenance must satisfy the wire contract");
    }

    #[test]
    fn catalog_identity_names_match_worker_hello_constants() {
        let expected = [
            ("llama", LLAMA_WORKER_ENGINE),
            ("owned-cuda", CUDA_WORKER_ENGINE),
        ];
        for (engine_name, expected_engine) in expected {
            let identity = catalog_model_engine_identity(engine_name).expect("catalog identity");
            assert_eq!(
                identity.engine, expected_engine,
                "catalog engine {engine_name}"
            );
        }
    }

    fn stuck_model_spec() -> StoredModelConfig {
        StoredModelConfig {
            model_id: "stuck-model".to_string(),
            engine: LLAMA_ENGINE.to_string(),
            task: "embed".to_string(),
            artifact_digest: "artifact".to_string(),
            artifact_format: "gguf".to_string(),
            tokenizer_sanitized_digest: "tokenizer".to_string(),
            model_locator: ModelAssetLocator::LocalPath {
                path: PathBuf::from("/tmp/stuck-model.gguf"),
            },
            tokenizer_locator: ModelAssetLocator::LocalPath {
                path: PathBuf::from("/tmp/stuck-tokenizer.json"),
            },
            model_source_url: "file:///tmp/stuck-model.gguf".to_string(),
            tokenizer_source_url: "file:///tmp/stuck-tokenizer.json".to_string(),
            pooling: "mean".to_string(),
            normalize: true,
            max_tokens: 128,
            quant: "fp32".to_string(),
            pin: false,
            owned_family: None,
            owned_dtype: None,
            owned_execution: None,
            owned_attention_units: None,
            config_locator: None,
            extra_locators: Vec::new(),
            engine_identity: EngineIdentity {
                engine: LLAMA_ENGINE.to_string(),
                version: "test".to_string(),
                build_flags: BTreeMap::new(),
            },
            numeric_profile_id: synapse_core::NumericProfileId("test-profile".to_string()),
            fingerprint: Fingerprint("test-fingerprint".to_string()),
            worker_bin: None,
            worker_runtime_dir: None,
        }
    }

    static TEST_STATE_COUNTER: AtomicU64 = AtomicU64::new(0);

    pub(super) fn test_storage_descriptor(label: &str) -> (PathBuf, StorageDescriptor) {
        let root = std::env::temp_dir().join(format!(
            "synapse-module-{label}-{}-{}",
            std::process::id(),
            TEST_STATE_COUNTER.fetch_add(1, Ordering::Relaxed),
        ));
        let descriptor = StorageDescriptor {
            module_id: "synapse-test".to_string(),
            storage_namespace: "default".to_string(),
            isolation: Isolation::Module,
            backend: StorageBackend::Sqlite {
                path: root.join("store.db").to_string_lossy().to_string(),
            },
        };
        (root, descriptor)
    }

    pub(super) fn test_machine_profile(os_build: &str) -> MachineProfile {
        MachineProfile {
            os_build: os_build.to_string(),
            arch: "aarch64".to_string(),
            chip_model: "test-chip".to_string(),
            ram_class: "test-ram".to_string(),
            ane_subtype: None,
            engine_identities: Vec::new(),
        }
    }

    pub(super) fn test_module_state(
        store: Arc<SynapseStore>,
        profile: MachineProfile,
    ) -> Arc<ModuleState> {
        test_module_state_with_config(store, profile, ModuleConfig::default())
    }

    pub(super) fn response_result(outcome: HandlerOutcome, operation: &str) -> Value {
        let HandlerOutcome::Response(response) = outcome else {
            panic!("{operation} should return a response")
        };
        serde_json::from_slice::<Value>(&response).expect("operation response is JSON")["result"]
            .clone()
    }

    fn test_module_state_with_config(
        store: Arc<SynapseStore>,
        profile: MachineProfile,
        config: ModuleConfig,
    ) -> Arc<ModuleState> {
        let (machine_profile_hash, revisioned_machine_profile_hash) =
            module_state_machine_profile_hashes(&profile);
        let profile_activation_epoch = store
            .profile_state()
            .expect("test profile state reads")
            .profile_activation_epoch
            .expect("test profile is activated");
        let runtime = RuntimeState::from_catalog(config, vec![stuck_model_spec()])
            .expect("test runtime initializes");
        #[cfg(unix)]
        let runtime = {
            let mut runtime = runtime;
            // Cold loads and pause expiry must use the same per-store lock on
            // every test platform, including when no supervisor was installed.
            let database = store
                .sqlite()
                .with_conn(|conn| {
                    Ok(PathBuf::from(
                        conn.path().expect("test store has a database file"),
                    ))
                })
                .expect("test database path");
            let root = database
                .parent()
                .expect("test database has a fixture root")
                .to_path_buf();
            runtime.ane_lane_lock_path = Some(root.join("ane-direct.lock"));
            runtime.ane_lane_test_root = Some(root);
            runtime
        };
        let runtime = Arc::new(runtime);
        Arc::new(ModuleState {
            module_id: "synapse-test".to_string(),
            store,
            module_generation: 1,
            machine_profile: profile,
            machine_profile_hash: machine_profile_hash.clone(),
            legacy_machine_profile_hash: machine_profile_hash,
            revisioned_machine_profile_hash,
            profile_activation_epoch,
            runtime,
            model_cache: Arc::new(ModelCache::new(
                std::env::temp_dir().join("synapse-test-cache"),
            )),
        })
    }
    #[tokio::test]
    async fn perf_sampler_disabled_is_silent_and_enabled_names_active_model() {
        let log_root = std::env::temp_dir().join(format!(
            "synapse-perf-sampler-{}-{}",
            std::process::id(),
            TEST_STATE_COUNTER.fetch_add(1, Ordering::Relaxed),
        ));
        let log_handle = cortexkit_log::init(cortexkit_log::Config {
            module_id: "synapse".to_string(),
            logs_dir: log_root,
            bound: Vec::new(),
            spec: Some("debug".to_string()),
            retention: cortexkit_log::SegmentRetention::default(),
            redactor: None,
            clock: None,
        })
        .expect("test logger initializes");

        let (disabled_root, disabled_descriptor) = test_storage_descriptor("perf-disabled");
        let disabled_store = Arc::new(
            SynapseStore::open(&disabled_descriptor).expect("disabled sampler store opens"),
        );
        let disabled_profile = test_machine_profile("perf-disabled-os");
        disabled_store
            .observe_profile(&disabled_profile, 10, 1)
            .expect("disabled sampler profile activates");
        let disabled_state = test_module_state(disabled_store, disabled_profile);
        assert!(start_perf_sampler(disabled_state).is_none());
        assert!(!fs::read_to_string(log_handle.path())
            .unwrap_or_default()
            .contains("synapse.perf: activity"));

        let (enabled_root, enabled_descriptor) = test_storage_descriptor("perf-enabled");
        let enabled_store =
            Arc::new(SynapseStore::open(&enabled_descriptor).expect("enabled sampler store opens"));
        let enabled_profile = test_machine_profile("perf-enabled-os");
        enabled_store
            .observe_profile(&enabled_profile, 10, 1)
            .expect("enabled sampler profile activates");
        let mut config = ModuleConfig::default();
        config.log.perf_interval_secs = 1;
        let enabled_state = test_module_state_with_config(enabled_store, enabled_profile, config);
        enabled_state
            .runtime
            .execution_stats
            .lock()
            .expect("execution stats lock")
            .in_flight = 1;
        let _activity = enabled_state
            .runtime
            .activity_telemetry
            .begin("active-model");
        let sampler = start_perf_sampler(Arc::clone(&enabled_state)).expect("sampler starts");
        tokio::time::sleep(Duration::from_millis(1_200)).await;
        sampler.abort();
        // The aborted task still owns a clone of the state, and the state owns the
        // open store. Windows refuses to delete a directory holding an open file,
        // so wait for the task to finish unwinding and release every handle
        // before the cleanup below.
        let _ = sampler.await;
        drop(_activity);
        drop(enabled_state);

        let activity_lines = fs::read_to_string(log_handle.path())
            .expect("perf log reads")
            .lines()
            .filter(|line| line.contains("synapse.perf: activity "))
            .map(str::to_owned)
            .collect::<Vec<_>>();
        assert_eq!(activity_lines.len(), 1, "{activity_lines:#?}");
        assert!(activity_lines[0].contains("in_flight=1"));
        assert!(activity_lines[0].contains("by_model=active-model:1"));
        assert!(activity_lines[0].contains("queue_depth=0 waiters=0"));

        fs::remove_dir_all(disabled_root).expect("remove disabled sampler state");
        fs::remove_dir_all(enabled_root).expect("remove enabled sampler state");
    }

    #[test]
    fn a_held_cache_lease_is_reported_as_retryable_not_invalid() {
        // pin and gc serialize on one digest lease, so a pin refused for
        // contention is a timing failure, not a broken artifact. Reporting it
        // as artifact_invalid would tell the caller never to retry a blob that
        // is fine.
        let held = ModelCacheError::Lease(cortexkit_lease::LeaseError::Held {
            key: cortexkit_lease::LeaseKey::new("synapse", "model-cache", "digest"),
        });
        let wire = cache_error_to_wire(held);
        assert_eq!(wire.class, ErrorClass::Transient);
        assert!(
            wire.retry_after_ms.is_some(),
            "a transient refusal must carry a retry hint"
        );
        assert!(wire.safe_to_retry_same_request);

        let invalid = ModelCacheError::NotFound("digest".to_string());
        let wire = cache_error_to_wire(invalid);
        assert_eq!(
            wire.class,
            ErrorClass::Permanent,
            "non-lease cache errors keep the permanent classification"
        );
    }

    #[tokio::test]
    async fn admission_status_counts_registered_but_nonresident_profile_lanes() {
        let (root, descriptor) = test_storage_descriptor("admission-catalog-lanes");
        let store = Arc::new(SynapseStore::open(&descriptor).unwrap());
        let profile = test_machine_profile("test-os");
        store.activate_profile(&profile, 1, 1000).unwrap();
        let state = test_module_state(store, profile);
        state
            .runtime
            .catalog
            .lock()
            .unwrap()
            .get_mut("stuck-model")
            .unwrap()
            .spec = {
            let mut spec = catalog_fixture_config("gte-modernbert-base.owned-metal");
            spec.model_id = "stuck-model".into();
            spec
        };
        assert!(state.runtime.loaded_models().is_empty());
        let reply = response_result(admission_status(state).await, "admission.status");
        assert_eq!(reply["lanes"], json!([]));
        assert_eq!(reply["catalog_lanes"], 1);
        assert_eq!(reply["certified_lanes"], 0);
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn admission_status_counts_refused_admissions_by_wire_reason() {
        let (root, descriptor) = test_storage_descriptor("admission-refusal-counters");
        let store = Arc::new(SynapseStore::open(&descriptor).expect("test store opens"));
        let profile = test_machine_profile("test-os");
        store
            .observe_profile(&profile, 10, 1)
            .expect("test profile activates");
        let state = test_module_state(Arc::clone(&store), profile);
        let error = match state.runtime.admit_inline(
            "test-model",
            None,
            QueueClass::Interactive,
            state.runtime.inline.byte_budget.saturating_add(1),
            None,
            None,
        ) {
            Ok(_) => panic!("an over-budget request must be refused"),
            Err(error) => error,
        };
        assert_eq!(error.code, "queue_full");

        let HandlerOutcome::Response(response) = admission_status(Arc::clone(&state)).await else {
            panic!("admission.status should return a response")
        };
        let payload: Value = serde_json::from_slice(&response).expect("admission status is JSON");
        assert_eq!(payload["result"]["refusals"]["queue_full"]["count"], 1);
        assert!(payload["result"]["refusals"]["queue_full"]["last_at_ms"]
            .as_u64()
            .is_some_and(|value| value > 0));
        assert_eq!(payload["result"]["jobs_minted"], 0);

        drop(state);
        drop(store);
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn model_cache_gc_sweep_reclaims_store_freelist_when_blobs_deleted() {
        let (root, descriptor) = test_storage_descriptor("gc-reclaim");
        let store = Arc::new(SynapseStore::open(&descriptor).unwrap());
        let profile = test_machine_profile("gc-reclaim-os");
        store.observe_profile(&profile, 10, 1).unwrap();

        // Seed store with > 64 MiB of data and then delete it to create freelist pages.
        store
            .sqlite()
            .with_conn(|conn| {
                conn.execute(
                    "CREATE TABLE bloat_seed (id INTEGER PRIMARY KEY, data BLOB)",
                    [],
                )?;
                let chunk = vec![0xfeu8; 1024 * 1024];
                let mut stmt = conn.prepare("INSERT INTO bloat_seed (data) VALUES (?1)")?;
                for _ in 0..66 {
                    stmt.execute(rusqlite::params![&chunk])?;
                }
                drop(stmt);
                conn.execute("DELETE FROM bloat_seed", [])?;
                Ok(())
            })
            .unwrap();

        let freelist_before = store.freelist_count().unwrap();
        let page_count_before = store.page_count().unwrap();
        let page_size = store.page_size().unwrap();
        assert!(freelist_before * page_size >= RECLAIM_FREELIST_MIN_BYTES);
        assert!(freelist_before >= page_count_before / RECLAIM_FREELIST_PAGE_RATIO_DIVISOR);

        let cache_root = std::env::temp_dir().join(format!(
            "synapse-test-cache-gc-{}-{}",
            std::process::id(),
            TEST_STATE_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let model_cache = Arc::new(ModelCache::new(&cache_root));
        std::fs::create_dir_all(&cache_root).unwrap();
        let blob_source = cache_root.join("test-blob.bin");
        std::fs::write(&blob_source, b"test-model-artifact-payload").unwrap();
        model_cache
            .ingest(synapse_core::ModelCacheIngest {
                source_url: format!("file://{}", blob_source.display()),
                expected_digest: None,
                format: "bin".to_string(),
                tokenizer_path: None,
                pin_module_id: None,
            })
            .expect("ingest test artifact into model cache");

        let (machine_profile_hash, revisioned_machine_profile_hash) =
            module_state_machine_profile_hashes(&profile);
        let profile_activation_epoch = store
            .profile_state()
            .expect("test profile state reads")
            .profile_activation_epoch
            .expect("test profile is activated");
        let module_config = ModuleConfig {
            cache_max_bytes: 0,
            ..ModuleConfig::default()
        };
        let runtime = Arc::new(
            RuntimeState::from_catalog(module_config, vec![stuck_model_spec()])
                .expect("test runtime initializes"),
        );
        let state = Arc::new(ModuleState {
            module_id: "synapse-test".to_string(),
            store: Arc::clone(&store),
            module_generation: 1,
            machine_profile: profile,
            machine_profile_hash: machine_profile_hash.clone(),
            legacy_machine_profile_hash: machine_profile_hash,
            revisioned_machine_profile_hash,
            profile_activation_epoch,
            runtime,
            model_cache,
        });

        let outcome1 = cache_gc(
            Arc::clone(&state),
            json!({
                "grace_ms": 60_000
            }),
        )
        .await;
        let HandlerOutcome::Response(body1) = outcome1 else {
            panic!("expected Response outcome from cache_gc");
        };
        let payload1: Value = serde_json::from_slice(&body1).expect("json response");
        assert_eq!(payload1["result"]["outcomes"][0]["state"], "marked");
        // No delete occurred, so store freelist remains un-vacuumed.
        assert_eq!(store.freelist_count().unwrap(), freelist_before);

        // A subsequent GC sweep after grace expiration deletes the tombstoned artifact.
        let db_path = match &descriptor.backend {
            StorageBackend::Sqlite { path } => PathBuf::from(path),
            _ => unreachable!(),
        };
        let file_size_before = std::fs::metadata(&db_path).unwrap().len();

        let outcome2 = cache_gc(
            Arc::clone(&state),
            json!({
                "grace_ms": 0
            }),
        )
        .await;
        let HandlerOutcome::Response(body2) = outcome2 else {
            panic!("expected Response outcome from cache_gc");
        };
        let payload2: Value = serde_json::from_slice(&body2).expect("json response");
        assert_eq!(payload2["result"]["outcomes"][0]["state"], "deleted");

        // The GC sweep actually deleted tombstoned blobs, triggering guarded freelist reclaim!
        let freelist_after = store.freelist_count().unwrap();
        let file_size_after = std::fs::metadata(&db_path).unwrap().len();

        assert_eq!(freelist_after, 0);
        assert!(
            file_size_after < file_size_before / 10,
            "expected file to shrink: before={file_size_before}, after={file_size_after}"
        );

        drop(state);
        drop(store);
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(cache_root);
    }

    /// Module state over a fresh store and an empty model cache whose budget is
    /// `cache_max_bytes`. Returns the state, the cache and the directories to
    /// remove after the test.
    fn cache_gc_test_state(
        label: &str,
        cache_max_bytes: u64,
    ) -> (Arc<ModuleState>, Arc<ModelCache>, PathBuf, PathBuf) {
        let (root, descriptor) = test_storage_descriptor(label);
        let store = Arc::new(SynapseStore::open(&descriptor).unwrap());
        let profile = test_machine_profile(&format!("{label}-os"));
        store.observe_profile(&profile, 10, 1).unwrap();
        let cache_root = std::env::temp_dir().join(format!(
            "synapse-test-{label}-{}-{}",
            std::process::id(),
            TEST_STATE_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&cache_root).unwrap();
        let model_cache = Arc::new(ModelCache::new(&cache_root));
        let (machine_profile_hash, revisioned_machine_profile_hash) =
            module_state_machine_profile_hashes(&profile);
        let profile_activation_epoch = store
            .profile_state()
            .expect("test profile state reads")
            .profile_activation_epoch
            .expect("test profile is activated");
        let runtime = Arc::new(
            RuntimeState::from_catalog(
                ModuleConfig {
                    cache_max_bytes,
                    ..ModuleConfig::default()
                },
                vec![stuck_model_spec()],
            )
            .expect("test runtime initializes"),
        );
        let state = Arc::new(ModuleState {
            module_id: "synapse-test".to_string(),
            store,
            module_generation: 1,
            machine_profile: profile,
            machine_profile_hash: machine_profile_hash.clone(),
            legacy_machine_profile_hash: machine_profile_hash,
            revisioned_machine_profile_hash,
            profile_activation_epoch,
            runtime,
            model_cache: Arc::clone(&model_cache),
        });
        (state, model_cache, root, cache_root)
    }

    async fn run_cache_gc(state: &Arc<ModuleState>, params: Value) -> Value {
        let HandlerOutcome::Response(body) = cache_gc(Arc::clone(state), params).await else {
            panic!("expected Response outcome from cache_gc");
        };
        let payload: Value = serde_json::from_slice(&body).expect("json response");
        assert!(payload.get("error").is_none(), "cache_gc failed: {payload}");
        payload["result"].clone()
    }
    #[tokio::test]
    async fn cache_gc_removes_ingest_temps_older_than_a_day_only() {
        let (state, _model_cache, root, cache_root) =
            cache_gc_test_state("gc-ingest-temp", u64::MAX);
        let tmp = cache_root.join("tmp");
        std::fs::create_dir_all(&tmp).unwrap();
        let abandoned = tmp.join("ingest-1-1.tmp");
        let in_progress = tmp.join("ingest-2-2.tmp");
        std::fs::write(&abandoned, b"partial download").unwrap();
        std::fs::write(&in_progress, b"partial download").unwrap();
        let backdate = |path: &Path, age: Duration| {
            std::fs::File::options()
                .write(true)
                .open(path)
                .unwrap()
                .set_modified(SystemTime::now() - age)
                .unwrap();
        };
        backdate(&abandoned, Duration::from_secs(25 * 60 * 60));
        backdate(&in_progress, Duration::from_secs(23 * 60 * 60));

        run_cache_gc(&state, json!({ "grace_ms": 60_000 })).await;

        assert!(
            !abandoned.exists(),
            "a partial file older than 24 h is removed"
        );
        assert!(
            in_progress.is_file(),
            "a partial file younger than 24 h may belong to a running ingest"
        );
        drop(state);
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(cache_root);
    }
    #[tokio::test(start_paused = true)]
    async fn model_load_wait_timeout_fires_for_never_notified_slot() {
        let runtime = RuntimeState::from_catalog(ModuleConfig::default(), vec![stuck_model_spec()])
            .expect("test runtime should initialize");
        runtime
            .catalog
            .lock()
            .expect("catalog lock")
            .get_mut("stuck-model")
            .expect("stuck model is registered")
            .state = ModelRuntimeState::Loading;

        let start = tokio::time::Instant::now();
        let deadline = tokio::time::Instant::now() + Duration::from_millis(20);
        let error = match wait_for_model_loaded(&runtime, "stuck-model", deadline, 20).await {
            Ok(_) => panic!("a never-notified loading slot must time out"),
            Err(error) => error,
        };
        let elapsed = start.elapsed();
        assert_eq!(error.code, "model_loading");
        assert_eq!(error.class, ErrorClass::Transient);
        assert!(
            (Duration::from_millis(10)..=Duration::from_millis(500)).contains(&elapsed),
            "model-load timeout fired outside its bound: {elapsed:?}"
        );

        let zero_start = tokio::time::Instant::now();
        let zero_error =
            match wait_for_model_loaded(&runtime, "stuck-model", tokio::time::Instant::now(), 0)
                .await
            {
                Ok(_) => panic!("a zero deadline must still reject the stuck slot"),
                Err(error) => error,
            };
        assert_eq!(zero_error.code, "model_loading");
        assert!(
            zero_start.elapsed() <= Duration::from_millis(500),
            "zero-deadline model-load timeout took too long: {:?}",
            zero_start.elapsed()
        );
    }

    #[tokio::test(start_paused = true)]
    async fn execution_permit_timeout_fires_when_all_permits_are_held() {
        let runtime = RuntimeState::from_catalog(ModuleConfig::default(), Vec::new())
            .expect("test runtime should initialize");
        let permit_count = runtime.execution.available_permits();
        let mut held = Vec::with_capacity(permit_count);
        for _ in 0..permit_count {
            held.push(
                runtime
                    .execution
                    .clone()
                    .acquire_owned()
                    .await
                    .expect("test should hold an execution permit"),
            );
        }
        let start = tokio::time::Instant::now();
        let deadline = tokio::time::Instant::now() + Duration::from_millis(20);
        let error = acquire_execution_permit(&runtime, Some(deadline))
            .await
            .err()
            .expect("a saturated execution semaphore must time out");
        let elapsed = start.elapsed();
        assert_eq!(error.code, "deadline_exceeded");
        assert_eq!(error.class, ErrorClass::Transient);
        assert!(
            (Duration::from_millis(10)..=Duration::from_millis(500)).contains(&elapsed),
            "execution-permit timeout fired outside its bound: {elapsed:?}"
        );

        let zero_start = tokio::time::Instant::now();
        let zero_error = acquire_execution_permit(&runtime, Some(tokio::time::Instant::now()))
            .await
            .err()
            .expect("a saturated execution semaphore must reject a zero deadline");
        assert_eq!(zero_error.code, "deadline_exceeded");
        assert!(
            zero_start.elapsed() <= Duration::from_millis(500),
            "zero-deadline execution-permit timeout took too long: {:?}",
            zero_start.elapsed()
        );
        drop(held);
    }

    #[test]
    fn worker_binary_sibling_names_cover_every_worker_engine() {
        // Release archives unpack every binary at the archive root, so these
        // names are the sibling-resolution contract between the module and
        // the worker binaries it spawns.
        assert_eq!(
            worker_binary_file_name("llama"),
            Some("ck-synapse-worker-llama")
        );
        assert_eq!(
            worker_binary_file_name("owned-cuda"),
            Some("ck-synapse-worker-cuda")
        );
        assert_eq!(worker_binary_file_name("unknown-engine"), None);
    }

    #[test]
    fn engine_load_failures_are_permanent_artifact_errors() {
        let error = engine_error_to_wire(EngineError {
            stage: EngineErrorStage::Load,
            risk_class: synapse_core::EngineRiskClass::AbortSafe,
            message: "missing tensor; tried classifier.weight".to_string(),
            retry_after_ms: None,
            safe_to_retry_same_request: false,
        });

        assert_eq!(error.code, "artifact_invalid");
        assert_eq!(error.class, ErrorClass::Permanent);
        assert_eq!(error.retry_after_ms, None);
        assert!(!error.safe_to_retry_same_request);
    }

    #[test]
    fn transient_model_load_errors_always_include_retry_delay() {
        let explicit = transient_model_load_error("cache lease is contended");
        assert_eq!(explicit.class, ErrorClass::Transient);
        assert_eq!(explicit.retry_after_ms, Some(1_000));
        assert!(explicit.safe_to_retry_same_request);

        let normalized = WireOperationError::from_stable(
            StableError::model_loading(None),
            "model artifact is still downloading",
        );
        assert_eq!(normalized.class, ErrorClass::Transient);
        assert_eq!(
            normalized.retry_after_ms,
            Some(DEFAULT_TRANSIENT_RETRY_AFTER_MS)
        );
        assert!(normalized.safe_to_retry_same_request);
    }

    #[test]
    fn supervised_launch_without_subc_module_id_is_refused() {
        let error = module_id_from_environment(|_| None, LaunchNonceState::Present)
            .expect_err("a supervised launch must provide its module id");
        assert!(matches!(error, ModuleError::Config(_)));
        assert!(error.to_string().contains(SUBC_MODULE_ID_ENV));

        // A nonce descriptor the daemon named but that could not be read still
        // marks a supervised launch; it must not fall through to the default id.
        let unreadable = module_id_from_environment(
            |_| None,
            LaunchNonceState::Unreadable("descriptor 3 is not open".to_string()),
        )
        .expect_err("an unreadable nonce descriptor is still a supervised launch");
        assert!(unreadable.to_string().contains(SUBC_MODULE_ID_ENV));
        assert!(unreadable.to_string().contains("descriptor 3 is not open"));

        let unsupervised = module_id_from_environment(|_| None, LaunchNonceState::Absent)
            .expect("an unsupervised development run keeps the default module id");
        assert_eq!(unsupervised, DEFAULT_MODULE_ID);

        let supervised = module_id_from_environment(
            |key| (key == SUBC_MODULE_ID_ENV).then(|| OsString::from("synapse-test")),
            LaunchNonceState::Present,
        )
        .expect("a supervised launch with its module id is accepted");
        assert_eq!(supervised, "synapse-test");
    }

    #[test]
    fn stop_line_reason_names_how_the_connection_ended() {
        assert_eq!(
            connection_end_reason(Some(ConnectionEnd::Goodbye)),
            "goodbye"
        );
        assert_eq!(connection_end_reason(Some(ConnectionEnd::Eof)), "eof");
        assert_eq!(connection_end_reason(Some(ConnectionEnd::Reset)), "reset");
        assert_eq!(connection_end_reason(Some(ConnectionEnd::Closed)), "closed");
        assert_eq!(connection_end_reason(None), "unknown");
    }

    #[test]
    #[cfg(not(windows))]
    fn default_store_resolver_uses_xdg_then_home_data_directory() {
        let from_xdg =
            default_storage_descriptor_with_environment(DEFAULT_MODULE_ID, |key| match key {
                "XDG_DATA_HOME" => Some(OsString::from("/xdg-data")),
                "HOME" => Some(OsString::from("/home/operator")),
                _ => None,
            })
            .unwrap();
        assert_eq!(
            from_xdg.backend,
            StorageBackend::Sqlite {
                path: "/xdg-data/cortexkit/synapse/store.db".to_string()
            }
        );

        let from_home = default_storage_descriptor_with_environment(DEFAULT_MODULE_ID, |key| {
            (key == "HOME").then(|| OsString::from("/home/operator"))
        })
        .unwrap();
        let StorageBackend::Sqlite { path } = &from_home.backend else {
            panic!("expected a sqlite backend, got {:?}", from_home.backend);
        };
        let components = Path::new(path)
            .components()
            .map(|component| component.as_os_str().to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        let expected_tail = [
            "home",
            "operator",
            ".local",
            "share",
            "cortexkit",
            "synapse",
            "store.db",
        ];
        assert!(
            components.ends_with(
                &expected_tail
                    .iter()
                    .map(|segment| segment.to_string())
                    .collect::<Vec<_>>()
            ),
            "unexpected store path components: {components:?}"
        );
    }

    #[cfg(windows)]
    #[test]
    fn default_store_resolver_uses_localappdata_not_home_or_xdg() {
        let descriptor =
            default_storage_descriptor_with_environment(DEFAULT_MODULE_ID, |key| match key {
                "LOCALAPPDATA" => Some(OsString::from(r"C:\native-data")),
                "HOME" => Some(OsString::from(r"D:\home")),
                "XDG_DATA_HOME" => Some(OsString::from(r"E:\xdg")),
                _ => None,
            })
            .unwrap();
        assert_eq!(
            descriptor.backend,
            StorageBackend::Sqlite {
                path: sqlite_store_path(r"C:\native-data", DEFAULT_MODULE_ID),
            }
        );
    }

    #[test]
    fn daemon_storage_takes_precedence_over_native_store_default() {
        let supplied = StorageDescriptor {
            module_id: DEFAULT_MODULE_ID.to_string(),
            storage_namespace: "daemon".to_string(),
            isolation: Isolation::Module,
            backend: StorageBackend::Sqlite {
                path: "daemon-selected-store.db".to_string(),
            },
        };
        let resolved = resolve_storage_descriptor(
            &Some(serde_json::to_value(&supplied).unwrap()),
            DEFAULT_MODULE_ID,
        )
        .unwrap();
        assert_eq!(resolved.module_id, supplied.module_id);
        assert_eq!(resolved.storage_namespace, supplied.storage_namespace);
        assert_eq!(resolved.backend, supplied.backend);
    }

    #[test]
    fn xdg_config_home_is_read_before_home_and_unset_uses_home() {
        let root = std::env::temp_dir().join(format!(
            "synapse-module-config-home-{}-{}",
            std::process::id(),
            TEST_STATE_COUNTER.fetch_add(1, Ordering::Relaxed),
        ));
        let xdg_home = root.join("xdg");
        let home = root.join("home");
        let xdg_config = xdg_home.join("cortexkit").join("synapse.jsonc");
        let home_config = home.join(".config").join("cortexkit").join("synapse.jsonc");
        fs::create_dir_all(xdg_config.parent().expect("XDG config parent")).unwrap();
        fs::create_dir_all(home_config.parent().expect("HOME config parent")).unwrap();
        fs::write(&xdg_config, r#"{"cache_max_bytes": 111}"#).unwrap();
        fs::write(&home_config, r#"{"cache_max_bytes": 222}"#).unwrap();

        config_home_subprocess(Some(&xdg_home), Some(&home), "111");
        config_home_subprocess(None, Some(&home), "222");
        config_home_subprocess(Some(Path::new("")), Some(&home), "222");
        config_home_subprocess(None, None, "relative-error");
        fs::remove_dir_all(root).unwrap();
    }

    // The shared resolver reads process environment and has no injectable variant.
    // Isolate its inputs in a child test process rather than racing parallel tests.
    fn config_home_subprocess(xdg: Option<&Path>, home: Option<&Path>, expected: &str) {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command.args([
            "--exact",
            "tests::config_home_subprocess_probe",
            "--nocapture",
        ]);
        for key in [
            "XDG_CONFIG_HOME",
            "HOME",
            "APPDATA",
            "USERPROFILE",
            SYNAPSE_CONFIG_PATH_ENV,
        ] {
            command.env_remove(key);
        }
        if let Some(xdg) = xdg {
            command.env("XDG_CONFIG_HOME", xdg);
        }
        if let Some(home) = home {
            command.env("HOME", home);
        }
        let output = command
            .env("SYNAPSE_TEST_CONFIG_HOME_EXPECTED", expected)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
    }

    #[test]
    fn config_home_subprocess_probe() {
        let Ok(expected) = std::env::var("SYNAPSE_TEST_CONFIG_HOME_EXPECTED") else {
            return;
        };
        let result = load_module_config_with_environment(|key| env::var_os(key), None);
        if expected == "relative-error" {
            let error = result.expect_err("relative config home must be refused");
            assert!(matches!(error, ModuleError::Config(_)));
            assert!(error.to_string().contains("XDG_CONFIG_HOME"));
            assert!(error.to_string().contains("relative"));
        } else {
            assert_eq!(
                result.unwrap().cache_max_bytes,
                expected.parse::<u64>().unwrap()
            );
        }
    }

    #[test]
    fn relative_xdg_config_home_is_refused() {
        config_home_subprocess(
            Some(Path::new("relative-config")),
            Some(&std::env::temp_dir()),
            "relative-error",
        );
    }

    #[test]
    fn module_config_rejects_unknown_fields() {
        let error = parse_module_config_json(r#"{ "typo_field": true }"#, "test", ConfigTier::User)
            .expect_err("unknown fields should fail");
        assert!(matches!(error, ModuleError::Config(_)));
        assert!(error.to_string().contains("typo_field"));
    }

    #[test]
    fn module_config_tight_packing_is_opt_in_and_has_no_other_knobs() {
        for input in ["{}", r#"{"ane_direct":{}}"#] {
            let config = parse_module_config_json(input, "test", ConfigTier::User).unwrap();
            assert!(!config.ane_direct.tight_packing);
        }
        let config = parse_module_config_json(
            r#"{"ane_direct":{"tight_packing":true}}"#,
            "test",
            ConfigTier::User,
        )
        .unwrap();
        assert!(config.ane_direct.tight_packing);
        assert!(parse_module_config_json(
            r#"{"ane_direct":{"tight_packing":1}}"#,
            "test",
            ConfigTier::User,
        )
        .is_err());
        assert!(parse_module_config_json(
            r#"{"ane_direct":{"width":256}}"#,
            "test",
            ConfigTier::User,
        )
        .is_err());
    }

    #[test]
    fn remote_providers_removed_from_all_config_tiers() {
        for tier in [ConfigTier::User, ConfigTier::Project] {
            for providers in ["[]", "null", "[{}]"] {
                let error = parse_module_config_json(
                    &format!(r#"{{"remote_providers":{providers}}}"#),
                    "test",
                    tier,
                )
                .expect_err("remote provider configuration must fail closed");
                assert_eq!(
                    error.to_string(),
                    "config: remote_providers were removed; modules should call providers directly"
                );
            }
        }
    }

    #[test]
    fn free_form_preloads_removed_from_all_config_tiers() {
        for tier in [ConfigTier::User, ConfigTier::Project] {
            for task in ["embed", "rerank"] {
                let contents = format!(
                    r#"{{"preload_models":[{{"model_id":"removed-{task}","engine":"owned-metal","task":"{task}","model_path":"model","tokenizer_path":"tokenizer"}}]}}"#
                );
                let error = parse_module_config_json(&contents, "test", tier)
                    .expect_err("a free-form preload must fail closed");
                assert_eq!(error.to_string(), format!(
                    "config: preload model_id 'removed-{task}': free-form preloads for embed and rerank were removed; specify a profile"
                ));
            }
            parse_module_config_json(r#"{"preload_models":[{"model_id":"llama-generation","engine":"llama","task":"generate","model_path":"model","tokenizer_path":"tokenizer"}]}"#, "test", tier).expect("llama generation preload parses");
            let profiled = r#"{"preload_models":[{"engine":"owned-metal","profile":"kept-profile","model_path":"model","tokenizer_path":"tokenizer"}]}"#;
            parse_module_config_json(profiled, "test", tier).expect("kept preload parses");
        }
    }

    #[test]
    fn owned_decode_generate_preloads_are_refused_by_model_id() {
        let config = json!({"preload_models": [{
            "model_id": "qwen3-0.6b-decode-q8",
            "engine": "owned-metal-decode",
            "task": "generate",
            "model_path": "missing.safetensors",
            "tokenizer_path": "missing.json",
            "quantizer_revision": "retired-setting"
        }]});
        for tier in [ConfigTier::User, ConfigTier::Project] {
            let err = parse_module_config_json(&config.to_string(), "test", tier).unwrap_err();
            assert_eq!(
                err.to_string(),
                "config: preload model_id 'qwen3-0.6b-decode-q8': owned decode serving was retired"
            );
        }
    }

    #[test]
    fn retired_owned_decode_keys_are_refused_by_name() {
        for key in ["grammar_enabled", "sidecar_spec", "decode_chain_k"] {
            let config = format!(r#"{{"{key}":null}}"#);
            for tier in [ConfigTier::User, ConfigTier::Project] {
                let err = parse_module_config_json(&config, "test", tier).unwrap_err();
                assert_eq!(
                    err.to_string(),
                    format!("config: {key} was removed; owned decode serving was retired")
                );
            }
        }
    }

    #[test]
    fn alias_admin_removed_from_all_config_tiers() {
        for tier in [ConfigTier::User, ConfigTier::Project] {
            for key in [
                "alias_admin_enabled",
                "dev_alias_admin",
                "enable_alias_admin",
            ] {
                let setting = format!(r#"{{"{key}":false}}"#);
                for contents in [setting.clone(), format!(r#"{{"dev":{setting}}}"#)] {
                    let error = parse_module_config_json(&contents, "test", tier)
                        .expect_err("even a disabled retired key must fail closed");
                    assert_eq!(error.to_string(), format!(
                        "config: alias_admin_enabled was removed (found '{key}'); aliases are no longer supported"
                    ));
                }
            }
        }
    }

    #[test]
    fn embed_query_ignores_retired_accept_declared() {
        let params: EmbedQueryParams = serde_json::from_value(json!({
            "text": "query", "accept_declared": false
        }))
        .expect("published callers still send accept_declared on embed.query");
        assert_eq!(params.text, "query");
    }

    #[test]
    fn embed_batch_ignores_retired_accept_declared() {
        let params: EmbedBatchParams = serde_json::from_value(json!({
            "texts": ["query"], "accept_declared": false
        }))
        .expect("published callers still send accept_declared on embed.batch");
        assert_eq!(params.texts, ["query"]);
    }

    #[test]
    fn rerank_score_ignores_retired_accept_declared() {
        let params: RerankScoreParams = serde_json::from_value(json!({
            "query": "query", "candidates": ["document"], "accept_declared": false
        }))
        .expect("published callers still send accept_declared on rerank.score");
        assert_eq!(params.query, "query");
        assert_eq!(params.candidates, ["document"]);
    }

    #[test]
    fn module_config_parses_log_settings() {
        let default = parse_module_config_json(r#"{}"#, "test", ConfigTier::User)
            .expect("default log config should parse");
        assert_eq!(default.log.perf_interval_secs, 0);
        assert_eq!(
            default.log.worker_forward_lines_per_sec,
            DEFAULT_WORKER_FORWARD_LINES_PER_SEC
        );

        let configured = parse_module_config_json(
            r#"{"log":{"perf_interval_secs":5,"worker_forward_lines_per_sec":12}}"#,
            "test",
            ConfigTier::User,
        )
        .expect("log settings should parse");
        assert_eq!(configured.log.perf_interval_secs, 5);
        assert_eq!(configured.log.worker_forward_lines_per_sec, 12);

        let error = parse_module_config_json(
            r#"{"log":{"worker_forward_line_per_sec":12}}"#,
            "test",
            ConfigTier::User,
        )
        .expect_err("unknown log fields should fail");
        assert!(error.to_string().contains("worker_forward_line_per_sec"));
    }

    #[test]
    fn module_config_parses_worker_load_timeout() {
        let default = parse_module_config_json(r#"{}"#, "test", ConfigTier::User)
            .expect("default worker config should parse");
        assert_eq!(
            default.worker.load_timeout_ms,
            DEFAULT_WORKER_LOAD_TIMEOUT_MS
        );

        let configured = parse_module_config_json(
            r#"{"worker":{"load_timeout_ms":240000}}"#,
            "test",
            ConfigTier::User,
        )
        .expect("worker load timeout should parse");
        assert_eq!(configured.worker.load_timeout_ms, 240_000);

        let error = parse_module_config_json(
            r#"{"worker":{"load_timeout_mss":240000}}"#,
            "test",
            ConfigTier::User,
        )
        .expect_err("unknown worker fields should fail");
        assert!(error.to_string().contains("load_timeout_mss"));
    }

    #[test]
    fn module_config_parses_microllm_and_cache_fields() {
        let config = parse_module_config_json(
            r#"{
                "microllm_max_tokens": 128,
                "cache_max_bytes": 4096
            }"#,
            "test",
            ConfigTier::User,
        )
        .expect("valid config");
        assert_eq!(config.microllm_max_tokens, 128);
        assert_eq!(config.cache_max_bytes, 4096);
    }

    #[test]
    fn job_config_parses_split_ttls_and_rejects_legacy_alias() {
        let split = parse_module_config_json(
            r#"{
                "jobs": {
                    "execution_ttl_ms": 10,
                    "result_retention_ttl_ms": 20
                }
            }"#,
            "test",
            ConfigTier::User,
        )
        .unwrap();
        assert_eq!(split.jobs.execution_ttl_ms, 10);
        assert_eq!(split.jobs.result_retention_ttl_ms, 20);

        // Pre-release rename, no compatibility surface: the old key must fail
        // loudly (deny_unknown_fields) instead of being silently accepted.
        let legacy =
            parse_module_config_json(r#"{"jobs":{"ttl_ms":40}}"#, "test", ConfigTier::User);
        assert!(legacy.is_err(), "legacy ttl_ms key must be rejected");
        assert!(
            parse_module_config_json(
                r#"{"jobs":{"resume_deadline_ms":30}}"#,
                "test",
                ConfigTier::User,
            )
            .is_err(),
            "retired remote reauthentication deadline must be rejected"
        );
    }

    #[test]
    fn request_digest_is_canonical_and_binds_order_and_content() {
        let items = vec![
            ("a".to_string(), sha256_hex(b"first")),
            ("b".to_string(), sha256_hex(b"second")),
        ];
        let constraints_a: Value = serde_json::from_str(r#"{"z":1,"a":true}"#).unwrap();
        let constraints_b: Value = serde_json::from_str(r#"{"a":true,"z":1}"#).unwrap();
        let local = compute_request_digest("embed.batch", "model", &constraints_a, &items);
        assert_eq!(
            local,
            compute_request_digest("embed.batch", "model", &constraints_b, &items)
        );
        let mut reordered = items.clone();
        reordered.reverse();
        assert_ne!(
            local,
            compute_request_digest("embed.batch", "model", &constraints_a, &reordered)
        );
        let mut changed = items;
        changed[0].1 = sha256_hex(b"changed");
        assert_ne!(
            local,
            compute_request_digest("embed.batch", "model", &constraints_a, &changed)
        );
    }

    #[test]
    fn batch_token_cost_tracks_actual_token_id_chunks() {
        let batch = TokenBatch {
            items: vec![vec![1, 2, 3], Vec::new(), vec![4, 5]],
        };

        assert_eq!(batch_token_cost(&batch), 6);
    }

    #[test]
    fn engine_batch_plan_sorts_and_caps_uninterruptible_work() {
        let batch = TokenBatch {
            items: (0..16).map(|index| vec![index as u32; 300]).collect(),
        };
        let planned = plan_embedding_engine_batches(&batch, DEFAULT_ENGINE_BATCH_TOKEN_BUDGET);

        assert_eq!(planned.iter().map(Vec::len).collect::<Vec<_>>(), [8, 8]);
        let flattened = planned.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(flattened, (0..16).collect::<Vec<_>>());
    }

    #[test]
    fn engine_batch_plan_respects_token_budget_before_row_cap() {
        let batch = TokenBatch {
            items: (0..8).map(|index| vec![index as u32; 512]).collect(),
        };
        let planned = plan_embedding_engine_batches(&batch, DEFAULT_ENGINE_BATCH_TOKEN_BUDGET);

        assert_eq!(planned.iter().map(Vec::len).collect::<Vec<_>>(), [6, 2]);
    }

    #[test]
    fn huggingface_resolve_url_uses_repo_segments_and_pinned_revision() {
        let url = huggingface_resolve_url(
            "https://huggingface.co",
            "Qdrant/all-MiniLM-L6-v2-onnx",
            "0123456789012345678901234567890123456789",
            "onnx/model.onnx",
        )
        .expect("hf url should resolve");
        assert_eq!(
            url,
            "https://huggingface.co/Qdrant/all-MiniLM-L6-v2-onnx/resolve/0123456789012345678901234567890123456789/onnx/model.onnx"
        );
    }

    #[test]
    fn recommended_batch_policy_uses_engine_constants_and_omits_unknown_advice() {
        let owned = recommended_batch_for_engine("owned-metal", 512).unwrap();
        assert_eq!(owned.rows, MAX_ENGINE_BATCH_ITEMS);
        assert_eq!(owned.token_budget, DEFAULT_ENGINE_BATCH_TOKEN_BUDGET);

        let cuda = recommended_batch_for_engine(CUDA_WORKER_ENGINE, 2048)
            .expect("CUDA clients need usable batch advice");
        let wire = serde_json::to_value(cuda).expect("serialize CUDA batch advice");
        assert_eq!(wire["rows"], MAX_ENGINE_BATCH_ITEMS);
        assert_eq!(wire["token_budget"], DEFAULT_ENGINE_BATCH_TOKEN_BUDGET);

        for engine in ["owned-vulkan", synapse_core::ANE_DIRECT_WORKER_ENGINE] {
            let advice = recommended_batch_for_engine(engine, 8192)
                .unwrap_or_else(|| panic!("{engine} clients need usable batch advice"));
            assert_eq!(advice.rows, MAX_ENGINE_BATCH_ITEMS, "{engine}");
            assert_eq!(
                advice.token_budget, DEFAULT_ENGINE_BATCH_TOKEN_BUDGET,
                "{engine}"
            );
        }

        assert!(recommended_batch_for_engine(LLAMA_ENGINE, 512).is_none());
    }

    #[test]
    fn owned_rerank_catalog_acceptance_uses_a_distinct_processing_identity() {
        let owned = || OwnedCatalogConfig {
            family: OwnedFamily::GteModernBert,
            dtype: OwnedDType::F32,
            projections: ProjectionDtype::F32,
            execution: "explicit".to_string(),
            attention_units: OWNED_DEFAULT_ATTENTION_UNITS,
            config_locator: None,
            extra_locators: Vec::new(),
            identity_override: None,
        };
        let make_spec = |model_id: &str, task: ModelTask| {
            build_stored_model_config(
                model_id.to_string(),
                "owned-metal",
                task,
                "sha256:model".to_string(),
                "safetensors-package".to_string(),
                "sha256:tokenizer".to_string(),
                ModelAssetLocator::LocalPath {
                    path: PathBuf::from("/tmp/model"),
                },
                ModelAssetLocator::LocalPath {
                    path: PathBuf::from("/tmp/tokenizer"),
                },
                "file:///tmp/model".to_string(),
                "file:///tmp/tokenizer".to_string(),
                WorkerPooling::Mean,
                true,
                8192,
                "f32".to_string(),
                false,
                None,
                None,
                Vec::new(),
                Some(owned()),
                &InlineConfig::default(),
                &JobConfig::default(),
            )
        };

        let rerank = make_spec("gte-reranker", ModelTask::Rerank)
            .expect("owned ModernBERT rerank should be accepted");
        assert_eq!(rerank.task, "rerank");
        assert_eq!(
            rerank.numeric_profile_id,
            NumericProfile {
                model_digest: "sha256:model".to_string(),
                quant: "f32".to_string(),
                engine: rerank.engine_identity.clone(),
                sanitized_tokenizer_digest: "sha256:tokenizer".to_string(),
                pooling: PoolingStrategy::Mean,
                normalization: NormalizationMode::L2,
                dtype: NumericDType::F32,
                flash_attention: FlashAttentionSetting::Disabled,
                certified_shape: CertifiedShapeEnvelope {
                    max_context_tokens: 8192,
                    max_batch_tokens: InlineConfig::default().max_tokens as u32,
                    max_micro_batch_tokens: JobConfig::default().bulk_quantum_tokens as u32,
                    max_sequences: InlineConfig::default().max_items as u32,
                },
                prompt_template: Some("synapse-rerank-bos-query-sep-doc-eos-v1".to_string()),
                prefix_template: None,
                thread_policy: ThreadPolicyClass::Balanced,
                operation: None,
                input_grammar: None,
                kernel_revision: None,
                rotation: None,
                converted_package_digest: None,
                manifest_profile_digest: None,
                speech_frontend: None,
                decode_rule: None,
            }
            .numeric_profile_id()
        );
        let embed = make_spec("gte-embed", ModelTask::Embed)
            .expect("owned ModernBERT embedding should remain accepted");
        assert_ne!(rerank.fingerprint, embed.fingerprint);
    }

    #[test]
    fn owned_metal_still_rejects_generation_tasks_at_catalog_validation() {
        let error = build_stored_model_config(
            "owned-generation".to_string(),
            "owned-metal",
            ModelTask::Generate,
            "sha256:model".to_string(),
            "safetensors-package".to_string(),
            "sha256:tokenizer".to_string(),
            ModelAssetLocator::LocalPath {
                path: PathBuf::from("/tmp/model"),
            },
            ModelAssetLocator::LocalPath {
                path: PathBuf::from("/tmp/tokenizer"),
            },
            "file:///tmp/model".to_string(),
            "file:///tmp/tokenizer".to_string(),
            WorkerPooling::Mean,
            true,
            512,
            "f32".to_string(),
            false,
            None,
            None,
            Vec::new(),
            None,
            &InlineConfig::default(),
            &JobConfig::default(),
        )
        .expect_err("owned-metal generation must remain outside the embed/rerank lane");
        assert!(error
            .to_string()
            .contains("embedding and rerank models only"));
    }

    fn is_transcribe_profile(manifest: &Value, profile: &Value) -> bool {
        let model = profile["model"].as_str().unwrap();
        manifest["models"][model]["operation"] == "transcribe"
    }

    pub(super) fn catalog_fixture_config(id: &str) -> StoredModelConfig {
        let catalog = CatalogProfile::load(id).unwrap();
        let preload: PreloadModelConfig = serde_json::from_value(json!({
            "engine": catalog.profile()["lane"], "model_path": "package.safetensors", "tokenizer_path": "tokenizer.json"
        })).unwrap();
        let pooling = match catalog.model()["grammar"]["pooling"].as_str().unwrap() {
            "cls" => WorkerPooling::Cls,
            "masked_mean" => WorkerPooling::Mean,
            "last_non_pad" => WorkerPooling::Last,
            _ => unreachable!(),
        };
        build_stored_model_config(
            id.into(),
            catalog.profile()["lane"].as_str().unwrap(),
            if catalog.model()["operation"] == "embed" {
                ModelTask::Embed
            } else {
                ModelTask::Rerank
            },
            catalog.artifact_digest(),
            "safetensors".into(),
            "sha256:fixture-tokenizer".into(),
            ModelAssetLocator::LocalPath {
                path: "package.safetensors".into(),
            },
            ModelAssetLocator::LocalPath {
                path: "tokenizer.json".into(),
            },
            String::new(),
            String::new(),
            pooling,
            catalog.model()["output"]["normalization"] == "l2",
            8192,
            catalog.profile()["storage_dtype"].as_str().unwrap().into(),
            false,
            None,
            None,
            Vec::new(),
            Some(
                catalog
                    .owned_config(preload.execution.as_deref(), preload.attention_units)
                    .unwrap(),
            ),
            &InlineConfig::default(),
            &JobConfig::default(),
        )
        .unwrap()
    }

    #[test]
    fn certification_catalog_derives_only_compiled_lanes_and_keeps_their_pins() {
        let compiled = catalog::compiled_catalog().unwrap();
        let declared = certification_catalog().unwrap();
        let cells = synapse_certify::derive_cells(&declared.lanes);
        assert_eq!(cells.len(), 38);
        for cell in &cells {
            let backend = compiled
                .entry(&cell.model)
                .unwrap()
                .backend(synapse_certify::row_backend(&cell.row_id).unwrap())
                .unwrap();
            assert_eq!(cell.engine, backend.engine);
            assert_eq!(cell.fingerprint, backend.fingerprint);
        }
        assert!(!cells.iter().any(|cell| {
            (cell.row_id == "metal-m5" && cell.model == "qwen3-reranker-0.6b")
                || (cell.row_id == "ane-m5" && cell.model == "gte-reranker-modernbert-base")
        }));
    }

    fn catalog_rerank_numeric_result(reference: [f64; 2], scores: [f64; 2]) -> (bool, f64, f64) {
        let (dir, _) = test_storage_descriptor("catalog-rerank-tolerance");
        let mut spec = catalog_fixture_config("gte-reranker-modernbert-base.owned-metal");
        spec.model_id = "gte-reranker-modernbert-base-metal".into();
        let mut model = catalog_test_model(&dir, &spec);
        // The tokenizer fixture is profile-backed, but the native Metal engine
        // whose logits this test replays is loaded without a profile flag.
        Arc::get_mut(&mut model)
            .unwrap()
            .engine_identity
            .build_flags
            .remove("profile");
        Arc::get_mut(&mut model).unwrap().backend = EmbedBackend::Owned(Arc::new(Mutex::new(
            OwnedMetalEmbedEngine::new(OwnedFamily::GteModernBert, OwnedDType::F32),
        )));
        let mut entry = catalog::compiled_catalog()
            .unwrap()
            .entry("gte-reranker-modernbert-base")
            .unwrap()
            .clone();
        let backend = entry.backend("metal").unwrap().clone();
        assert!(backend.profile.is_none());
        let profile = CatalogProfile::load("gte-reranker-modernbert-base.owned-metal").unwrap();
        assert_eq!(
            profile.typed.profiles[&profile.id].rerank_tolerance_class,
            synapse_parity::manifest::ToleranceClass::Fp32
        );
        let check = entry.self_check.as_mut().unwrap();
        check.inputs = vec![
            catalog::SelfCheckInput::Rerank(catalog::RerankSelfCheckInput {
                query: "query".into(),
                candidates: vec!["first".into(), "second".into()],
            });
            4
        ];
        check.reference.scores = Some(vec![reference.to_vec(); 4]);
        let result = numerical_catalog_check(
            &model,
            &entry,
            &backend,
            |_| panic!("rerank must not call embedding inference"),
            |pairs| {
                assert_eq!(pairs.len(), 2);
                Ok(synapse_core::RerankScores {
                    scores: scores.iter().map(|p| (p / (1.0 - p)).ln() as f32).collect(),
                })
            },
        )
        .unwrap();
        drop(model);
        fs::remove_dir_all(dir).unwrap();
        result
    }

    #[test]
    fn catalog_rerank_fp32_score_bound_is_005() {
        let (passed, error, tolerance) = catalog_rerank_numeric_result([0.504, 0.5], [0.5, 0.5]);
        assert!(passed);
        assert!((error - 0.004).abs() < 1e-7);
        assert_eq!(tolerance, 0.005);
        let (passed, error, tolerance) = catalog_rerank_numeric_result([0.506, 0.5], [0.5, 0.5]);
        assert!(
            !passed,
            "an fp16 bound would incorrectly admit a 0.006 score error"
        );
        assert!((error - 0.006).abs() < 1e-7);
        assert_eq!(tolerance, 0.005);
    }

    #[test]
    fn catalog_rerank_fp32_order_gap_is_005() {
        let (passed, error, tolerance) =
            catalog_rerank_numeric_result([0.502, 0.498], [0.499, 0.501]);
        assert!(passed, "a reversed pair below the class gap is tolerated");
        assert!(error < 0.005);
        assert_eq!(tolerance, 0.005);
        let (passed, error, tolerance) =
            catalog_rerank_numeric_result([0.503, 0.497], [0.499, 0.501]);
        assert!(
            !passed,
            "a reversed pair above the class gap must refuse even within the score bound"
        );
        assert!(error < 0.005);
        assert_eq!(tolerance, 0.005);
    }

    #[test]
    fn profile_preload_attention_budget_covers_the_8192_token_ceiling() {
        let catalog = CatalogProfile::load("gte-modernbert-base.owned-metal").unwrap();
        assert_eq!(
            catalog.owned_config(None, None).unwrap().attention_units,
            8192 * 8192
        );
        assert_eq!(
            catalog
                .owned_config(None, Some(1234))
                .unwrap()
                .attention_units,
            1234,
            "an explicit budget is kept"
        );
    }

    #[test]
    fn catalog_only_worker_identities_are_platform_independent() {
        // Catalog fingerprints are pinned once per release and must match on
        // every OS that runs the lane; the worker transport differs by OS.
        for engine in ["owned-vulkan", "ane-direct-worker"] {
            let identity = catalog_model_engine_identity(engine).unwrap();
            assert!(
                !identity.build_flags.contains_key("transport"),
                "{engine}: {:?}",
                identity.build_flags
            );
        }
    }

    #[test]
    fn catalog_profiles_bind_identity_and_worker_package() {
        let manifest: Value =
            serde_json::from_slice(include_bytes!("../../../bench/parity/models.json")).unwrap();
        // Token-model (embed and rerank) profiles only; transcription
        // profiles have no token grammar and no preload lane.
        for (id, profile) in manifest["profiles"]
            .as_object()
            .unwrap()
            .iter()
            .filter(|(_, profile)| !is_transcribe_profile(&manifest, profile))
        {
            let spec = catalog_fixture_config(id);
            let catalog = CatalogProfile::load(id).unwrap();
            let mut numeric: NumericProfile = serde_json::from_value(json!({
                "model_digest": "unused", "quant": spec.quant, "engine": spec.engine_identity,
                "sanitized_tokenizer_digest": spec.tokenizer_sanitized_digest,
                "pooling": profile_pooling(parse_pooling(&spec.pooling).unwrap()),
                "normalization": if spec.normalize { "l2" } else { "none" },
                "dtype": profile["storage_dtype"], "flash_attention": "disabled",
                "certified_shape": {"max_context_tokens":8192,"max_batch_tokens":8192,"max_micro_batch_tokens":8192,"max_sequences":64},
                "thread_policy":"balanced"
            })).unwrap();
            catalog.apply_numeric_profile(&mut numeric);
            assert_eq!(numeric.model_digest, catalog.model()["checkpoint_digest"]);
            assert_eq!(
                numeric.manifest_profile_digest.as_deref(),
                manifest["digests"]["profiles"][id].as_str()
            );
            assert_eq!(
                numeric.input_grammar,
                Some(format!(
                    "synapse-input-grammar-v1:{}",
                    manifest["digests"]["grammar"][&catalog.slug]
                        .as_str()
                        .unwrap()
                ))
            );
            assert!(numeric.operation.is_some());
            assert!(numeric.kernel_revision.is_some());
            assert_eq!(
                numeric.prompt_template.is_some(),
                catalog.slug == "gte-reranker-modernbert-base"
            );
            assert_eq!(spec.artifact_digest, catalog.artifact_digest());
            let config = model_runtime_config(
                &spec,
                Path::new("package.safetensors"),
                Path::new("cache"),
                64,
            );
            assert_eq!(config.values["profile"], *id);
            assert_eq!(config.values["operation"], spec.task);
            assert_eq!(
                numeric.converted_package_digest.is_some(),
                profile["lane"] != "owned-metal"
            );
            if profile["lane"] != "owned-metal" {
                let mut wrong_package = spec.clone();
                wrong_package.artifact_digest = catalog.model()["checkpoint_digest"]
                    .as_str()
                    .unwrap()
                    .into();
                assert!(normalize_catalog_model(
                    wrong_package,
                    &InlineConfig::default(),
                    &JobConfig::default()
                )
                .unwrap_err()
                .to_string()
                .contains("package_digest_mismatch"));
            }
            let original = numeric.fingerprint();
            numeric.model_digest.push('0');
            assert_ne!(original, numeric.fingerprint());
            assert_eq!(
                spec.fingerprint,
                normalize_catalog_model(
                    spec.clone(),
                    &InlineConfig::default(),
                    &JobConfig::default()
                )
                .unwrap()
                .fingerprint
            );
            let expected = [
                "c0f400f352d41b549b864b9cbe59bc7c401e820c48c0585b98d1bc7d7c6eecd0",
                "27166bbf06d295c10dad348a8df8bf6774e209791bef402fc4847b24f96b1b39",
                // gte-modernbert-base.owned-metal: catalog profile lanes for f16
                // ModernBERT opt in to f16 weight projections (projection_dtype = f16,
                // kernel revision owned-metal-graph-5-bucket-2). Without the opt-in
                // this profile's fingerprint is
                // 3a0b02613e9f7cd7b502de500e28613df43025b59df2cd4b0ec604de42e5a2e2.
                "7f51a5b7679f7168f7e6b344eb79f81e88664e0600950002e336b208c46da1d1",
                "c86eceac46723880b0790de173899b8cd7a9cb2c560e1025ca4c1851c9a85195",
                "a006367b6adb645e44c82c9a56e973f3423fb4f9be2461ff80a38df68aa9c543",
                "510721ab2b99d667e5461764ac3c129e0244b4f1b32370a269608aa8e56bba20",
                "bdab90b6bb1d81696d679d3c267c95f90de418fa501a3824f42e0c3c217fe166",
                "29a1a109f29d1496b8924788a3f190a8df8dd8989054ef1b89459eb4300ce195",
                "df0b8ccadc3696d8e08cca52fda0236fc6a33a59134187c52517291ea94ac902",
                "896eb823c1ee5d9c31b42a12163684ab6b707968a619a7c0196775cc394f4148",
                "6553a78dce363808f967655b534db0c1536d837de3dcfba3a30e142a3cb847e9",
                "07f113383a8a0def9648161d12290d2326b67b4bdebbade0b64ee63e769b4382",
                "6e583acd1cacaf47a1c02873c52441f3b62417ed086fc88346d4d08a8389fbc0",
                "2c00ee65e9ae43600c62332a0d96f799b790c03339cf43ad286eaaca0f8fd6bd",
                "16ca0b783714705ca01235bfa7fe3a08da2615abc87cf081a1f0c7041edc2d5a",
                "47c91eb36c28f78cfe5c75a8143d28894794f5b2dc9ef7bc0d9eaa5721f9d768",
            ];
            let index = manifest["profiles"]
                .as_object()
                .unwrap()
                .iter()
                .filter(|(_, profile)| !is_transcribe_profile(&manifest, profile))
                .position(|(key, _)| key == id)
                .unwrap();
            assert_eq!(spec.fingerprint.0, expected[index], "{id}");
        }
    }

    #[test]
    fn only_catalog_f16_modernbert_lanes_opt_in_and_the_engine_is_told() {
        assert_eq!(
            catalog_projection_dtype(OwnedFamily::GteModernBert, OwnedDType::F16),
            ProjectionDtype::F16
        );
        for (family, dtype) in [
            (OwnedFamily::GteModernBert, OwnedDType::F32),
            (OwnedFamily::Qwen3, OwnedDType::F16),
            (OwnedFamily::MiniLm, OwnedDType::F16),
        ] {
            assert_eq!(
                catalog_projection_dtype(family, dtype),
                ProjectionDtype::F32
            );
        }
        let spec = |projections| {
            build_stored_model_config(
                "gte-lane".into(),
                "owned-metal",
                ModelTask::Embed,
                "sha256:model".into(),
                "safetensors".into(),
                "sha256:tokenizer".into(),
                ModelAssetLocator::LocalPath {
                    path: "unused".into(),
                },
                ModelAssetLocator::LocalPath {
                    path: "unused".into(),
                },
                String::new(),
                String::new(),
                WorkerPooling::Cls,
                true,
                8192,
                "f16".into(),
                false,
                None,
                None,
                Vec::new(),
                Some(OwnedCatalogConfig {
                    family: OwnedFamily::GteModernBert,
                    dtype: OwnedDType::F16,
                    projections,
                    execution: "explicit".into(),
                    attention_units: OWNED_DEFAULT_ATTENTION_UNITS,
                    config_locator: None,
                    extra_locators: Vec::new(),
                    identity_override: None,
                }),
                &InlineConfig::default(),
                &JobConfig::default(),
            )
            .unwrap()
        };
        let runtime = |spec: &StoredModelConfig| {
            model_runtime_config(spec, Path::new("model"), Path::new("cache"), 0)
        };
        let opted_in = spec(ProjectionDtype::F16);
        assert_eq!(
            opted_in.engine_identity.build_flags[PROJECTION_DTYPE_CONFIG_KEY],
            "f16"
        );
        assert_eq!(opted_in.engine_identity.build_flags["graph_revision"], "5");
        assert_eq!(
            runtime(&opted_in).values[PROJECTION_DTYPE_CONFIG_KEY],
            "f16"
        );
        assert_eq!(
            stored_projection_dtype(&opted_in.engine_identity),
            ProjectionDtype::F16
        );
        let default = spec(ProjectionDtype::F32);
        assert!(!default
            .engine_identity
            .build_flags
            .contains_key(PROJECTION_DTYPE_CONFIG_KEY));
        assert!(!runtime(&default)
            .values
            .contains_key(PROJECTION_DTYPE_CONFIG_KEY));
        assert_ne!(opted_in.fingerprint, default.fingerprint);
    }

    #[test]
    fn preload_profile_key_builds_every_catalog_lane_without_config_json() {
        let (dir, _) = test_storage_descriptor("catalog-preload");
        let manifest: Value =
            serde_json::from_slice(include_bytes!("../../../bench/parity/models.json")).unwrap();
        for (id, _) in manifest["profiles"]
            .as_object()
            .unwrap()
            .iter()
            .filter(|(_, profile)| !is_transcribe_profile(&manifest, profile))
        {
            let fixture = catalog_fixture_config(id);
            let model = catalog_test_model(&dir, &fixture);
            let preload: PreloadModelConfig = serde_json::from_value(json!({
                "model_id":id,"profile":id,"engine":fixture.engine,"task":fixture.task,
                "pooling":fixture.pooling,"normalize":fixture.normalize,"artifact_digest":fixture.artifact_digest,
                "model_path":dir.join("converted-package.safetensors"),"tokenizer_path":dir.join("catalog-tokenizer.json")
            })).unwrap();
            let stored = build_preload_catalog_model(
                0,
                preload,
                &InlineConfig::default(),
                &JobConfig::default(),
            )
            .unwrap();
            assert_eq!(stored.max_tokens, 8192);
            assert_eq!(stored.engine_identity.build_flags["profile"], *id);
            assert_eq!(stored.artifact_digest, fixture.artifact_digest);
            assert_eq!(model.tokenizer.max_tokens(), usize::MAX);
            let restored = normalize_catalog_model(
                stored.clone(),
                &InlineConfig::default(),
                &JobConfig::default(),
            )
            .unwrap();
            assert_eq!(restored.fingerprint, stored.fingerprint, "{id}");
            assert_eq!(restored.engine_identity, stored.engine_identity, "{id}");
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn catalog_config_rejects_manifest_disagreements() {
        let catalog = CatalogProfile::load("qwen3-reranker-0.6b.owned-metal").unwrap();
        assert!(catalog
            .validate(
                "owned-metal",
                ModelTask::Rerank,
                WorkerPooling::Last,
                false,
                None
            )
            .is_ok());
        for (engine, task, pooling, normalize, max) in [
            (
                "owned-cuda",
                ModelTask::Rerank,
                WorkerPooling::Last,
                false,
                None,
            ),
            (
                "owned-metal",
                ModelTask::Embed,
                WorkerPooling::Last,
                false,
                None,
            ),
            (
                "owned-metal",
                ModelTask::Rerank,
                WorkerPooling::Mean,
                false,
                None,
            ),
            (
                "owned-metal",
                ModelTask::Rerank,
                WorkerPooling::Last,
                true,
                None,
            ),
            (
                "owned-metal",
                ModelTask::Rerank,
                WorkerPooling::Last,
                false,
                Some(8191),
            ),
        ] {
            assert!(catalog
                .validate(engine, task, pooling, normalize, max)
                .is_err());
        }
        let preload = serde_json::from_value(json!({"profile": catalog.id, "engine":"owned-metal", "task":"rerank", "pooling":"last", "normalize":false, "prompt_template":"wrong", "model_path":"unused", "tokenizer_path":"unused"})).unwrap();
        assert!(build_preload_catalog_model(
            0,
            preload,
            &InlineConfig::default(),
            &JobConfig::default()
        )
        .unwrap_err()
        .to_string()
        .contains("prompt_template"));
    }

    pub(crate) fn catalog_test_model(dir: &Path, spec: &StoredModelConfig) -> Arc<EmbeddingModel> {
        use worker_host::{WorkerEngine, WorkerHostConfig};
        fs::create_dir_all(dir).unwrap();
        let tokenizer_path = dir.join("catalog-tokenizer.json");
        let mut tokenizer: Value = serde_json::from_str(r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"WhitespaceSplit"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"a":1,"yes":9693,"no":2152},"unk_token":"[UNK]"}}"#).unwrap();
        let catalog = CatalogProfile::load(&spec.engine_identity.build_flags["profile"]).unwrap();
        if catalog.slug == "qwen3-reranker-0.6b" {
            for role in ["yes", "no"] {
                tokenizer["model"]["vocab"][role] =
                    catalog.model()["grammar"]["readout"][role]["id"].clone();
            }
        }
        let occupied = tokenizer["model"]["vocab"]
            .as_object()
            .unwrap()
            .values()
            .map(|id| id.as_u64().unwrap())
            .collect::<std::collections::BTreeSet<_>>();
        for index in 0..=*occupied.last().unwrap() {
            if !occupied.contains(&index) {
                tokenizer["model"]["vocab"][format!("unused-{index}")] = json!(index);
            }
        }
        fs::write(&tokenizer_path, serde_json::to_vec(&tokenizer).unwrap()).unwrap();
        let worker = dir.join("must-not-start.sh");
        fs::write(
            &worker,
            format!(
                "#!/bin/sh\nprintf called > '{}'\nexit 1\n",
                dir.join("worker-called").display()
            ),
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&worker, fs::Permissions::from_mode(0o700)).unwrap();
        }
        let engine =
            WorkerEngine::new(WorkerHostConfig::new(worker, dir.join("worker-runtime"))).unwrap();
        let loaded = engine.insert_loaded_model_for_test("catalog-fixture".into(), 1, 8192, None);
        Arc::new(EmbeddingModel {
            model_id: spec.model_id.clone(),
            task: parse_model_task(Some(&spec.task), &spec.engine, &spec.model_id).unwrap(),
            loaded_model: loaded,
            backend: EmbedBackend::Worker(Arc::new(Mutex::new(engine))),
            tokenizer: SanitizedTokenizer::from_file(
                &tokenizer_path,
                TokenizerConfig {
                    max_tokens: usize::MAX,
                },
            )
            .unwrap(),
            fingerprint: spec.fingerprint.clone(),
            engine_identity: spec.engine_identity.clone(),
            owned_tokenizer_policy: None,
        })
    }

    // Only the Unix-only catalog worker tests call this; gate it with them so
    // Windows builds don't see it as dead code.
    #[cfg(unix)]
    fn catalog_test_state(
        dir: &Path,
        descriptor: &StorageDescriptor,
        id: &str,
    ) -> Arc<ModuleState> {
        let store = Arc::new(SynapseStore::open(descriptor).unwrap());
        let profile = test_machine_profile("catalog-test-os");
        store.activate_profile(&profile, 1, 1000).unwrap();
        let state = test_module_state(store.clone(), profile.clone());
        let mut spec = catalog_fixture_config(id);
        spec.model_id = "catalog-sequence-test".into();
        let model = catalog_test_model(dir, &spec);
        state.runtime.catalog.lock().unwrap().insert(
            spec.model_id.clone(),
            ModelSlot {
                spec,
                loaded: Some(model),
                state: ModelRuntimeState::Ready,
                notify: Arc::new(Notify::new()),
                last_cold_load_ms: None,
            },
        );
        state
    }

    #[cfg(unix)]
    fn catalog_boundary_worker(
        request: &synapse_core::WorkerRequest,
        raw: Option<&[u8]>,
    ) -> (synapse_core::WorkerResponse, Option<Vec<u8>>) {
        use synapse_core::{WorkerRequest, WorkerResponse};
        match request {
            WorkerRequest::EmbedBatch { req_id, items, .. } => {
                let ids = synapse_core::decode_i32_frame(raw.unwrap()).unwrap();
                assert_eq!(
                    ids.len(),
                    items.iter().map(|item| item.n_tokens).sum::<usize>()
                );
                (
                    WorkerResponse::Vectors {
                        req_id: req_id.clone(),
                        dims: 1,
                        n: items.len(),
                    },
                    Some(synapse_core::encode_f32_frame(&vec![1.0; items.len()])),
                )
            }
            WorkerRequest::Rerank {
                req_id, candidates, ..
            } => (
                WorkerResponse::Scores {
                    req_id: req_id.clone(),
                },
                Some(synapse_core::encode_f32_frame(&vec![0.5; candidates.len()])),
            ),
            WorkerRequest::RerankSequences {
                req_id, sequences, ..
            } => (
                WorkerResponse::Scores {
                    req_id: req_id.clone(),
                },
                Some(synapse_core::encode_f32_frame(&vec![0.5; sequences.len()])),
            ),
            other => (WorkerResponse::unsupported_request(other), None),
        }
    }

    #[cfg(unix)]
    fn installed_catalog_boundary_state(id: &str, name: &str) -> (PathBuf, Arc<ModuleState>) {
        use store::{CatalogInstallMember, CatalogInstallRecord, DownloadJobRequest};
        let (root, mut state) =
            catalog_runtime_tests::isolated_catalog_state("installed-catalog-boundary", true);
        Arc::get_mut(&mut Arc::get_mut(&mut state).unwrap().runtime)
            .unwrap()
            .runnable_backends = BTreeSet::from([name.into()]);
        let seed = {
            let entry = state.runtime.release_catalog.entry(id).unwrap();
            let backend = entry.backend(name).unwrap();
            catalog_fixture_config(
                backend
                    .profile
                    .as_deref()
                    .unwrap_or(&format!("{id}.owned-metal")),
            )
        };
        let mut model = catalog_test_model(&root, &seed);
        let tokenizer_path = root.join("catalog-tokenizer.json");
        if id.starts_with("gte-") {
            let mut tokenizer: Value =
                serde_json::from_slice(&fs::read(&tokenizer_path).unwrap()).unwrap();
            tokenizer["post_processor"] =
                json!({"type":"BertProcessing", "sep":["unused-2",2], "cls":["unused-3",3]});
            fs::write(&tokenizer_path, serde_json::to_vec(&tokenizer).unwrap()).unwrap();
        }
        // Use genuine catalog IDs and lane metadata, but bind the test catalog
        // to the synthetic tokenizer's real checksum for cold-path validation.
        let bytes = fs::read(&tokenizer_path).unwrap();
        let runtime = Arc::get_mut(&mut Arc::get_mut(&mut state).unwrap().runtime).unwrap();
        for file in &mut runtime
            .release_catalog
            .models
            .iter_mut()
            .find(|entry| entry.id == id)
            .unwrap()
            .files
        {
            if file.role == "tokenizer" {
                file.sha256 = sha256_hex(&bytes);
                file.size_bytes = bytes.len() as u64;
            }
        }
        let entry = state.runtime.release_catalog.entry(id).unwrap();
        let backend = entry.backend(name).unwrap();
        let spec = catalog_lane_spec(&state, entry, backend, false).unwrap();
        let inner = Arc::get_mut(&mut model).unwrap();
        inner.model_id = spec.model_id.clone();
        inner.engine_identity = spec.engine_identity.clone();
        inner.fingerprint = spec.fingerprint.clone();
        let owned = stored_owned_profile(&spec).unwrap();
        inner.tokenizer = SanitizedTokenizer::from_file(
            &tokenizer_path,
            TokenizerConfig {
                max_tokens: if backend.profile.is_some() {
                    usize::MAX
                } else {
                    owned_tokenizer_max_tokens(spec.max_tokens, owned.as_ref())
                },
            },
        )
        .unwrap();
        if id == "qwen3-embedding-0.6b" && backend.profile.is_none() {
            inner.owned_tokenizer_policy = Some(OwnedTokenizerPolicy {
                add_special_tokens: true,
                pad_token_id: 0,
                terminal_token_id: Some(151643),
            });
        }
        let EmbedBackend::Worker(engine) = &inner.backend else {
            unreachable!()
        };
        engine
            .lock()
            .unwrap()
            .attach_mock_worker_for_test(catalog_boundary_worker);
        let job = state
            .store
            .admit_download_job(&DownloadJobRequest {
                request_key: "fixture",
                request_digest: "fixture",
                module_generation: 1,
                params_json: &json!({}),
                now_ms: 1,
                result_retention_ttl_ms: 60000,
                entry_complete: false,
            })
            .unwrap();
        state
            .store
            .commit_download_job(
                &job.record().job_id,
                &[CatalogInstallRecord {
                    catalog_id: id.into(),
                    manifest_digest: entry.manifest_digest(),
                    backend: name.into(),
                    members: entry
                        .backend_files(name)
                        .values()
                        .map(|file| CatalogInstallMember {
                            path: file.path.clone(),
                            digest: file.sha256.clone(),
                        })
                        .collect(),
                }],
                2,
            )
            .unwrap();
        let blob = state
            .model_cache
            .blob_path(&entry.backend_files(name)["tokenizer"].sha256);
        fs::create_dir_all(blob.parent().unwrap()).unwrap();
        fs::copy(&tokenizer_path, blob).unwrap();
        let (check_id, key) = catalog_self_check_key(&state, entry, backend).unwrap();
        let generation = catalog_check_generation(&state, &check_id, &key).unwrap();
        assert!(catalog_complete_check(&state, &check_id, generation, "passed", None).unwrap());
        state.runtime.catalog.lock().unwrap().insert(
            spec.model_id.clone(),
            ModelSlot {
                spec,
                loaded: Some(model),
                state: ModelRuntimeState::Ready,
                notify: Arc::new(Notify::new()),
                last_cold_load_ms: None,
            },
        );
        (root, state)
    }

    #[cfg(unix)]
    fn boundary_text(count: usize) -> String {
        std::iter::repeat_n("a", count)
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[cfg(unix)]
    fn boundary_call(state: &Arc<ModuleState>, method: &str, params: Value) -> Value {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        response_result(
            runtime.block_on(dispatch_request(
                state.clone(),
                MethodEnvelope {
                    method: method.into(),
                    params,
                },
            )),
            method,
        )
    }

    #[cfg(unix)]
    fn assert_catalog_boundary_refusal(
        state: &Arc<ModuleState>,
        method: &str,
        params: Value,
        item: &str,
    ) {
        let before = certify_worker_snapshot(&state.runtime);
        let result = boundary_call(state, method, params);
        assert_eq!(
            result["error"]["code"], "sequence_too_long",
            "catalog pre-admission ceiling: {result}"
        );
        assert_eq!(result["error"]["class"], "permanent");
        assert_eq!(result["error"]["safe_to_retry_same_request"], false);
        assert!(result["error"].get("retry_after_ms").is_none());
        assert_eq!(
            result["error"]["details"],
            json!({"tokens":8193,"max_tokens":8192,"item_id":item})
        );
        assert!(result.get("job_id").is_none());
        assert!(result.get("truncation_disclosures").is_none());
        assert_eq!(certify_worker_snapshot(&state.runtime), before);
    }

    #[cfg(unix)]
    fn assert_catalog_cold_boundary_refusal(
        state: &Arc<ModuleState>,
        lane: &str,
        method: &str,
        params: Value,
        item: &str,
    ) {
        let model = state
            .runtime
            .catalog
            .lock()
            .unwrap()
            .get_mut(lane)
            .unwrap()
            .loaded
            .take()
            .unwrap();
        let EmbedBackend::Worker(engine) = &model.backend else {
            unreachable!()
        };
        let before = engine.lock().unwrap().request_count().unwrap();
        assert_catalog_boundary_refusal(state, method, params, item);
        assert_eq!(engine.lock().unwrap().request_count().unwrap(), before);
        assert!(state.runtime.catalog.lock().unwrap()[lane].loaded.is_none());
        state
            .runtime
            .catalog
            .lock()
            .unwrap()
            .get_mut(lane)
            .unwrap()
            .loaded = Some(model);
    }

    #[cfg(unix)]
    #[test]
    fn installed_catalog_embed_boundary_precedes_admission() {
        for id in ["gte-modernbert-base", "qwen3-embedding-0.6b"] {
            for name in ["metal", "ane", "cuda", "vulkan"] {
                if cfg!(target_os = "macos") && matches!(name, "cuda" | "vulkan") {
                    continue;
                }
                let (root, state) = installed_catalog_boundary_state(id, name);
                let lane = catalog::lane_id(id, name);
                let overhead = if id.starts_with("gte-") { 2 } else { 1 };
                assert_catalog_boundary_refusal(
                    &state,
                    "embed.query",
                    json!({"model":lane,"id":"query-8193","text":boundary_text(8193-overhead)}),
                    "query-8193",
                );
                let before = certify_worker_snapshot(&state.runtime);
                let result = boundary_call(
                    &state,
                    "embed.query",
                    json!({"model":lane,"text":boundary_text(8192-overhead)}),
                );
                assert_eq!(
                    result["vectors"][0]["vector"],
                    json!([1.0]),
                    "{id}/{name}: {result}"
                );
                assert_eq!(result["real_token_counts"], json!([8192]));
                assert_eq!(result["truncation_disclosures"][0]["truncated"], false);
                assert_ne!(certify_worker_snapshot(&state.runtime), before);
                assert_catalog_cold_boundary_refusal(
                    &state,
                    &lane,
                    "embed.query",
                    json!({"model":lane,"text":boundary_text(8193-overhead),"deadline_ms":0}),
                    "query",
                );
                drop(state);
                fs::remove_dir_all(root).unwrap();
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn installed_catalog_batch_boundary_precedes_job_diversion() {
        for id in ["gte-modernbert-base", "qwen3-embedding-0.6b"] {
            for name in ["metal", "ane", "cuda", "vulkan"] {
                if cfg!(target_os = "macos") && matches!(name, "cuda" | "vulkan") {
                    continue;
                }
                let (root, state) = installed_catalog_boundary_state(id, name);
                let lane = catalog::lane_id(id, name);
                let overhead = if id.starts_with("gte-") { 2 } else { 1 };
                let item = json!({"id":"row-8193","text":boundary_text(8193-overhead)});
                assert_catalog_boundary_refusal(
                    &state,
                    "embed.batch",
                    json!({"model":lane,"items":[item.clone()]}),
                    "row-8193",
                );
                let mut items =
                    vec![json!({"id":"short","text":"a"}); state.runtime.inline.max_items];
                items.push(item);
                assert_catalog_boundary_refusal(
                    &state,
                    "embed.batch",
                    json!({"model":lane,"items":items}),
                    "row-8193",
                );
                let before = certify_worker_snapshot(&state.runtime);
                let result = boundary_call(
                    &state,
                    "embed.batch",
                    json!({"model":lane,"items":[{"id":"row-8192","text":boundary_text(8192-overhead)}]}),
                );
                assert_eq!(
                    result["vectors"][0]["vector"],
                    json!([1.0]),
                    "{id}/{name}: {result}"
                );
                assert_eq!(result["real_token_counts"], json!([8192]));
                assert_eq!(result["truncation_disclosures"][0]["truncated"], false);
                assert_ne!(certify_worker_snapshot(&state.runtime), before);
                assert_catalog_cold_boundary_refusal(
                    &state,
                    &lane,
                    "embed.batch",
                    json!({"model":lane,"items":[{"id":"cold-row","text":boundary_text(8193-overhead)}]}),
                    "cold-row",
                );
                drop(state);
                fs::remove_dir_all(root).unwrap();
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn installed_catalog_rerank_boundary_precedes_inline_budget() {
        for (id, names) in [
            (
                "gte-reranker-modernbert-base",
                &["metal", "cuda", "vulkan"][..],
            ),
            ("qwen3-reranker-0.6b", &["ane", "cuda", "vulkan"][..]),
        ] {
            for name in names {
                if cfg!(target_os = "macos") && matches!(*name, "cuda" | "vulkan") {
                    continue;
                }
                let (root, state) = installed_catalog_boundary_state(id, name);
                let lane = catalog::lane_id(id, name);
                let overhead = if id.starts_with("gte-") {
                    5
                } else {
                    let model = state.runtime.catalog.lock().unwrap()[&lane]
                        .loaded
                        .clone()
                        .unwrap();
                    owned_rerank_pairs(&model, "a a", &[String::new()])
                        .unwrap()
                        .unwrap()[0]
                        .len()
                };
                assert_catalog_boundary_refusal(
                    &state,
                    "rerank.score",
                    json!({"model":lane,"query":"a a","candidates":[boundary_text(8193-overhead)]}),
                    "0",
                );
                let before = certify_worker_snapshot(&state.runtime);
                let result = boundary_call(
                    &state,
                    "rerank.score",
                    json!({"model":lane,"query":"a a","candidates":[boundary_text(8192-overhead)]}),
                );
                // The mock supplies 0.5 before the module's readout. Native
                // Metal treats it as a logit, so sigmoid(0.5) is 0.62245935;
                // profile backends treat it as a probability and pass it through.
                let expected = if id == "gte-reranker-modernbert-base" && *name == "metal" {
                    json!([0.622459352016449])
                } else {
                    json!([0.5])
                };
                assert_eq!(result["scores"], expected, "{id}/{name}: {result}");
                assert_ne!(certify_worker_snapshot(&state.runtime), before);
                assert_catalog_cold_boundary_refusal(
                    &state,
                    &lane,
                    "rerank.score",
                    json!({"model":lane,"query":"a a","candidates":["a",boundary_text(8193-overhead)]}),
                    "1",
                );
                drop(state);
                fs::remove_dir_all(root).unwrap();
            }
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn catalog_embed_ceiling_precedes_job_diversion_and_worker_call() {
        let (dir, descriptor) = test_storage_descriptor("catalog-embed-ceiling");
        let state = catalog_test_state(&dir, &descriptor, "gte-modernbert-base.owned-vulkan");
        let before = certify_worker_snapshot(&state.runtime);
        assert_eq!(before["available"], true);
        assert_eq!(before["worker_requests"].as_object().unwrap().len(), 1);
        assert_eq!(
            before["worker_requests"]
                .as_object()
                .unwrap()
                .values()
                .next()
                .unwrap(),
            0
        );
        let text = std::iter::repeat_n("a", 8193).collect::<Vec<_>>().join(" ");
        let result = response_result(
            embed_batch(
                state.clone(),
                json!({"model":"catalog-sequence-test", "items":[{"id":"row-8193", "text":text}]}),
            )
            .await,
            "embed.batch",
        );
        assert_eq!(result["error"]["code"], "sequence_too_long", "{result}");
        assert_eq!(result["error"]["class"], "permanent");
        assert_eq!(
            result["error"]["details"],
            json!({"tokens":8193,"max_tokens":8192,"item_id":"row-8193"})
        );
        assert!(result.get("job_id").is_none());
        assert!(result.get("truncation_disclosures").is_none());
        assert_eq!(certify_worker_snapshot(&state.runtime), before);
        assert!(!dir.join("worker-called").exists());
        let text = std::iter::repeat_n("a", 8192).collect::<Vec<_>>().join(" ");
        let result = response_result(
            embed_batch(
                state.clone(),
                json!({"model":"catalog-sequence-test", "texts":[text]}),
            )
            .await,
            "embed.batch",
        );
        assert_ne!(result["error"]["code"], "sequence_too_long", "{result}");
        assert_ne!(result["error"]["code"], "queue_full", "{result}");
        assert!(
            result["error"]["message"]
                .as_str()
                .unwrap()
                .contains("worker load requires runtime_config"),
            "8192-token input must reach the worker host: {result}"
        );
        let listed = models_list_payload(&state);
        let row = listed["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["model_id"] == "catalog-sequence-test")
            .unwrap();
        assert_eq!(row["max_tokens"], 8192);
        assert_eq!(row["profile"], "gte-modernbert-base.owned-vulkan");
        drop(state);
        fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn catalog_qwen_rerank_ceiling_precedes_both_token_budget_estimates() {
        let (dir, descriptor) = test_storage_descriptor("catalog-rerank-ceiling");
        let state = catalog_test_state(&dir, &descriptor, "qwen3-reranker-0.6b.owned-vulkan");
        let model = state.runtime.catalog.lock().unwrap()["catalog-sequence-test"]
            .loaded
            .clone()
            .unwrap();
        let overhead = owned_rerank_pairs(&model, "", &[String::new()])
            .unwrap()
            .unwrap()[0]
            .len();
        assert!(overhead > 3);
        for count in [8193 - overhead, 8193] {
            let doc = std::iter::repeat_n("a", count)
                .collect::<Vec<_>>()
                .join(" ");
            let result = response_result(
                rerank_score(
                    state.clone(),
                    json!({"model":"catalog-sequence-test", "query":"", "candidates":[doc]}),
                )
                .await,
                "rerank.score",
            );
            assert_eq!(result["error"]["code"], "sequence_too_long", "{result}");
            assert!(result.get("job_id").is_none());
            assert!(!dir.join("worker-called").exists());
        }
        let doc = std::iter::repeat_n("a", 8192 - overhead)
            .collect::<Vec<_>>()
            .join(" ");
        let pairs = owned_rerank_pairs(&model, "", std::slice::from_ref(&doc))
            .unwrap()
            .unwrap();
        assert_eq!(pairs[0].len(), 8192);
        assert_eq!(
            pairs[0].last(),
            Some(&0),
            "template tail, not EOS or readout ids"
        );
        let result = response_result(
            rerank_score(
                state.clone(),
                json!({"model":"catalog-sequence-test", "query":"", "candidates":[doc]}),
            )
            .await,
            "rerank.score",
        );
        assert_ne!(result["error"]["code"], "sequence_too_long", "{result}");
        assert!(
            result["error"]["message"]
                .as_str()
                .unwrap()
                .contains("worker load requires runtime_config"),
            "8192-token composed input must reach worker host: {result}"
        );
        drop(model);
        drop(state);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn catalog_fingerprint_constraints_reject_substitution_and_newer_epochs() {
        let (dir, _) = test_storage_descriptor("catalog-fingerprint-constraints");
        let spec = catalog_fixture_config("gte-modernbert-base.owned-cuda");
        let model = catalog_test_model(&dir, &spec);
        assert_eq!(
            check_fingerprint_constraints(&model, Some("metal-fingerprint"), None, Some(2))
                .unwrap_err()
                .code,
            "substitution_rejected"
        );
        assert_eq!(
            check_fingerprint_constraints(&model, None, None, Some(3))
                .unwrap_err()
                .code,
            "migration_required"
        );
        check_fingerprint_constraints(&model, Some(&spec.fingerprint.0), None, Some(2)).unwrap();
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn catalog_embed_composes_one_eos_and_bypasses_legacy_terminal_policy() {
        let (dir, _) = test_storage_descriptor("catalog-eos");
        let spec = catalog_fixture_config("qwen3-embedding-0.6b.owned-metal");
        let mut model = catalog_test_model(&dir, &spec);
        Arc::get_mut(&mut model).unwrap().owned_tokenizer_policy =
            Some(synapse_engine_owned::TokenizerPolicy {
                add_special_tokens: true,
                pad_token_id: 0,
                terminal_token_id: Some(999),
            });
        let mut tokenized = model.tokenizer.tokenize_batch(["a"]).unwrap();
        compose_catalog_embed(&model, &mut tokenized).unwrap();
        let eos = CatalogProfile::load(&spec.engine_identity.build_flags["profile"])
            .unwrap()
            .model()["grammar"]["terminal_tokens"][0]["id"]
            .as_u64()
            .unwrap() as u32;
        assert_eq!(tokenized.batch.items, vec![vec![1, eos]]);
        compose_catalog_embed(&model, &mut tokenized).unwrap();
        apply_owned_tokenizer_policy(&model, &mut tokenized);
        assert_eq!(tokenized.batch.items, vec![vec![1, eos]]);
        assert!(!tokenized.disclosures[0].truncated);
        Arc::get_mut(&mut model)
            .unwrap()
            .engine_identity
            .build_flags
            .remove("profile");
        apply_owned_tokenizer_policy(&model, &mut tokenized);
        assert_eq!(tokenized.batch.items[0].last(), Some(&999));
        drop(model);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn catalog_qwen_validates_readout_and_template() {
        let (dir, _) = test_storage_descriptor("catalog-qwen-validation");
        let spec = catalog_fixture_config("qwen3-reranker-0.6b.owned-metal");
        let model = catalog_test_model(&dir, &spec);
        let mut catalog = CatalogProfile::load("qwen3-reranker-0.6b.owned-metal").unwrap();
        catalog.validate_readout(&model.tokenizer).unwrap();
        let yes =
            catalog.manifest["models"][&catalog.slug]["grammar"]["readout"]["yes"]["id"].clone();
        let no =
            catalog.manifest["models"][&catalog.slug]["grammar"]["readout"]["no"]["id"].clone();
        catalog.manifest["models"][&catalog.slug]["grammar"]["readout"]["yes"]["id"] = no;
        catalog.manifest["models"][&catalog.slug]["grammar"]["readout"]["no"]["id"] = yes;
        assert!(catalog
            .validate_readout(&model.tokenizer)
            .unwrap_err()
            .to_string()
            .contains("qwen_readout_mismatch"));
        catalog
            .typed
            .models
            .get_mut(&catalog.slug)
            .unwrap()
            .grammar
            .as_mut()
            .unwrap()
            .template
            .as_mut()
            .unwrap()
            .prefix
            .push('x');
        assert!(catalog
            .validate_template()
            .unwrap_err()
            .to_string()
            .contains("qwen_template_mismatch"));
        drop(model);
        fs::remove_dir_all(dir).unwrap();
    }

    fn make_test_tokenizer(dir: &Path, max_tokens: usize) -> SanitizedTokenizer {
        let path = dir.join("tokenizer.json");
        std::fs::write(
            &path,
            r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0},"unk_token":"[UNK]"}}"#,
        )
        .expect("write test tokenizer");
        SanitizedTokenizer::from_file(&path, TokenizerConfig { max_tokens })
            .expect("load test tokenizer")
    }

    #[test]
    fn models_list_rows_carry_all_contract_fields_matching_enforced_sources() {
        use crate::worker_host::{WorkerEngine, WorkerHostConfig};

        let (storage_dir, descriptor) = test_storage_descriptor("models-list-enforced");
        let store = Arc::new(SynapseStore::open(&descriptor).expect("open test store"));
        let profile = test_machine_profile("test-os");
        store
            .activate_profile(&profile, 1, 1000)
            .expect("activate profile");
        let state = test_module_state(store, profile);

        let mut metal_spec = catalog_fixture_config("gte-modernbert-base.owned-metal");
        metal_spec.model_id = "test-metal-lane".to_string();
        metal_spec.engine = "owned-metal".to_string();
        metal_spec.task = "embed".to_string();
        metal_spec.max_tokens = 512;
        metal_spec.owned_dtype = Some("f16".to_string());
        metal_spec.fingerprint = Fingerprint(
            "metal-fingerprint-0000000000000000000000000000000000000000000000000000000000000000"
                .to_string(),
        );

        let mut metal_engine = OwnedMetalEmbedEngine::new(OwnedFamily::MiniLm, OwnedDType::F16);
        let metal_loaded = metal_engine.insert_test_model(
            "owned-metal:test:0".to_string(),
            384,
            vec![128, 256, 384, 512],
        );
        let metal_model = Arc::new(EmbeddingModel {
            model_id: metal_spec.model_id.clone(),
            task: ModelTask::Embed,
            loaded_model: metal_loaded,
            backend: EmbedBackend::Owned(Arc::new(Mutex::new(metal_engine))),
            tokenizer: make_test_tokenizer(&storage_dir, 512),
            fingerprint: metal_spec.fingerprint.clone(),
            engine_identity: metal_spec.engine_identity.clone(),
            owned_tokenizer_policy: None,
        });

        let mut ane_spec = catalog_fixture_config("qwen3-embedding-0.6b.owned-cuda");
        ane_spec.model_id = "test-ane-lane".to_string();
        ane_spec.engine = CUDA_WORKER_ENGINE.to_string();
        ane_spec.task = "embed".to_string();
        // A stale stored budget must not override the manifest's ceiling.
        ane_spec.max_tokens = 4096;
        ane_spec.fingerprint = Fingerprint(
            "ane-fingerprint-0000000000000000000000000000000000000000000000000000000000000000"
                .to_string(),
        );

        let worker_config = WorkerHostConfig::new(
            PathBuf::from("/bin/false"),
            storage_dir.join("test-ane-worker"),
        );
        let worker_engine = WorkerEngine::new(worker_config).expect("create worker engine");
        let ane_loaded = worker_engine.insert_loaded_model_for_test(
            "ane-host-model-0".to_string(),
            384,
            32,
            Some(vec![128, 256, 8192]),
        );
        let ane_model = Arc::new(EmbeddingModel {
            model_id: ane_spec.model_id.clone(),
            task: ModelTask::Embed,
            loaded_model: ane_loaded,
            backend: EmbedBackend::Worker(Arc::new(Mutex::new(worker_engine))),
            tokenizer: make_test_tokenizer(&storage_dir, 512),
            fingerprint: ane_spec.fingerprint.clone(),
            engine_identity: ane_spec.engine_identity.clone(),
            owned_tokenizer_policy: None,
        });

        let mut unloaded_spec = catalog_fixture_config("qwen3-embedding-0.6b.owned-cuda");
        unloaded_spec.model_id = "test-unloaded-lane".to_string();
        unloaded_spec.engine = CUDA_WORKER_ENGINE.to_string();
        unloaded_spec.task = "embed".to_string();
        unloaded_spec.max_tokens = 256;

        {
            let mut catalog = state.runtime.catalog.lock().expect("catalog locks");
            catalog.clear();
            catalog.insert(
                metal_spec.model_id.clone(),
                ModelSlot {
                    spec: metal_spec.clone(),
                    loaded: Some(metal_model),
                    state: ModelRuntimeState::Ready,
                    notify: Arc::new(Notify::new()),
                    last_cold_load_ms: Some(18.5),
                },
            );
            catalog.insert(
                ane_spec.model_id.clone(),
                ModelSlot {
                    spec: ane_spec.clone(),
                    loaded: Some(ane_model),
                    state: ModelRuntimeState::Ready,
                    notify: Arc::new(Notify::new()),
                    last_cold_load_ms: Some(32.0),
                },
            );
            catalog.insert(
                unloaded_spec.model_id.clone(),
                ModelSlot {
                    spec: unloaded_spec.clone(),
                    loaded: None,
                    state: ModelRuntimeState::Unloaded,
                    notify: Arc::new(Notify::new()),
                    last_cold_load_ms: None,
                },
            );
        }

        let payload = models_list_payload(&state);
        let models = payload["models"]
            .as_array()
            .expect("models array in payload");

        assert_eq!(models.len(), 3);
        for row in models {
            assert!(
                row["max_tokens"].as_u64().unwrap_or(0) > 0,
                "every models.list row must carry positive max_tokens: {row:?}"
            );
            assert!(
                row.get("max_tokens_source").is_some(),
                "every row must disclose max_tokens_source: {row:?}"
            );
            assert!(
                row.get("device_class").is_some(),
                "every row must identify device_class: {row:?}"
            );
            assert!(
                row.get("fingerprints").is_some(),
                "every row must carry fingerprints: {row:?}"
            );
            assert!(
                row.get("state").is_some(),
                "every row must carry state: {row:?}"
            );
            assert!(
                row.get("certified").is_some(),
                "every row must carry certified flag: {row:?}"
            );
        }

        let metal_row = models
            .iter()
            .find(|r| r["model_id"] == "test-metal-lane")
            .expect("metal row exists");
        assert_eq!(metal_row["dtype"], "f16");
        assert_eq!(metal_row["device_class"], "metal");
        assert_eq!(metal_row["certified"], false);
        assert_eq!(metal_row["warm_load_cost_hint_ms"], 18.5);
        assert_eq!(metal_row["recommended_batch"]["rows"], 8);
        assert_eq!(metal_row["max_tokens"], 8192);
        assert_eq!(metal_row["max_tokens_source"], "manifest");
        // Metal only reports live buckets and dimensions on macOS.
        if cfg!(target_os = "macos") {
            assert_eq!(metal_row["bucket_ladder"], json!([128, 256, 384, 512]));
            assert_eq!(metal_row["dims"], 384);
        } else {
            assert!(metal_row.get("bucket_ladder").is_none());
        }

        let ane_row = models
            .iter()
            .find(|r| r["model_id"] == "test-ane-lane")
            .expect("ane row exists");
        assert_eq!(ane_row["max_tokens"], 8192);
        assert_eq!(ane_row["max_tokens_source"], "manifest");
        assert_eq!(ane_row["bucket_ladder"], json!([128, 256, 8192]));
        assert_eq!(ane_row["max_tokens"], ane_row["bucket_ladder"][2]);
        assert_eq!(ane_row["dims"], 384);
        assert_eq!(ane_row["dtype"], "f16");
        assert_eq!(ane_row["device_class"], "cuda");
        assert_eq!(
            ane_row["certified"], false,
            "uncertified worker lane cannot advertise certified=true"
        );
        assert_eq!(ane_row["warm_load_cost_hint_ms"], 32.0);
        assert_eq!(ane_row["recommended_batch"]["rows"], 8);
        assert_eq!(
            ane_row["recommended_batch"]["token_budget"],
            DEFAULT_ENGINE_BATCH_TOKEN_BUDGET
        );

        let unloaded_row = models
            .iter()
            .find(|r| r["model_id"] == "test-unloaded-lane")
            .expect("unloaded row exists");
        assert_eq!(unloaded_row["max_tokens"], 8192);
        assert_eq!(unloaded_row["max_tokens_source"], "manifest");
        assert!(
            unloaded_row.get("bucket_ladder").is_none(),
            "unloaded lane must omit unknown bucket ladder"
        );
        assert!(
            unloaded_row.get("dims").is_none(),
            "unloaded lane must omit unknown dims"
        );
        assert_eq!(unloaded_row["device_class"], "cuda");
        assert_eq!(unloaded_row["certified"], false);
    }

    #[test]
    fn owned_cuda_catalog_profile_survives_restore_and_assembles_package() {
        let scratch = std::env::temp_dir().join(format!(
            "synapse-owned-cuda-load-{}-{}",
            std::process::id(),
            TEST_STATE_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&scratch).expect("create scratch dir");

        let model_src = scratch.join("model.safetensors");
        let tokenizer_src = scratch.join("tokenizer.json");
        let config_src = scratch.join("config.json");
        let header = br#"{"embedding.weight":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#;
        let mut model_bytes = (header.len() as u64).to_le_bytes().to_vec();
        model_bytes.extend_from_slice(header);
        model_bytes.extend_from_slice(&[0; 4]);
        std::fs::write(&model_src, &model_bytes).expect("write model");
        std::fs::write(&tokenizer_src, b"{}").expect("write tokenizer");
        std::fs::write(&config_src, b"{}").expect("write config");
        let model_digest = format!("sha256:{}", sha256_hex(&model_bytes));
        let config_digest = format!("sha256:{}", sha256_hex(b"{}"));
        let mut spec = catalog_fixture_config("qwen3-embedding-0.6b.owned-cuda");
        spec.artifact_format = "safetensors-package".into();
        assert_eq!(spec.engine, CUDA_WORKER_ENGINE);
        assert_eq!(spec.owned_family.as_deref(), Some("qwen3-0.6b"));
        assert_eq!(spec.owned_dtype.as_deref(), Some("f16"));
        assert_eq!(spec.owned_execution.as_deref(), Some("supervised"));

        // Restart round-trip: the stored row is re-read through
        // normalize_catalog_model and must rehydrate the same owned profile.
        let mut restored = normalize_catalog_model(
            spec.clone(),
            &InlineConfig::default(),
            &JobConfig::default(),
        )
        .expect("stored row must normalize");
        assert_eq!(restored.owned_family, spec.owned_family);
        assert_eq!(restored.owned_dtype, spec.owned_dtype);
        assert_eq!(restored.config_locator, spec.config_locator);
        assert_eq!(restored.artifact_format, "safetensors-package");
        assert_eq!(restored.fingerprint, spec.fingerprint);

        // An installed lane supplies the separate cached config file.
        restored.config_locator = Some(ModelAssetLocator::CacheDigest {
            digest: config_digest.clone(),
        });
        let rehydrated = stored_owned_profile(&restored)
            .expect("stored owned-cuda profile must rehydrate")
            .expect("owned-cuda row must yield a profile");
        assert_eq!(rehydrated.family, OwnedFamily::Qwen3);
        assert_eq!(rehydrated.dtype, OwnedDType::F16);
        assert_eq!(rehydrated.execution, "supervised");
        assert_eq!(
            rehydrated.config_locator,
            Some(ModelAssetLocator::CacheDigest {
                digest: config_digest.clone()
            })
        );
        // Terminal token policy: Qwen3 reserves one token.
        assert_eq!(
            owned_tokenizer_max_tokens(spec.max_tokens, Some(&rehydrated)),
            spec.max_tokens - 1
        );

        // The package the CUDA worker loads: assemble from the cache so the
        // engine sees a directory holding config.json + model.safetensors.
        let cache_root = scratch.join("cache");
        std::fs::create_dir_all(cache_root.join("blobs")).expect("create cache blobs");
        let model_cache = ModelCache::new(&cache_root);
        std::fs::write(model_cache.blob_path(&model_digest), &model_bytes)
            .expect("stage model blob");
        std::fs::write(
            model_cache.blob_path(&config_digest),
            std::fs::read(&config_src).expect("read config"),
        )
        .expect("stage config blob");
        let package = assemble_owned_model_package(
            &restored,
            model_cache.blob_path(&model_digest).as_path(),
            &model_cache,
            &rehydrated,
        )
        .expect("owned-cuda package must assemble");
        assert!(package.is_dir(), "package must be a directory");
        assert!(package.join("config.json").is_file());
        assert!(package.join("model.safetensors").is_file());

        // resolve_model_root is the contract the CUDA engine enforces on the
        // effective path: a bare file resolves to its parent only when the
        // parent holds config.json, which the assembled package guarantees.
        let runtime_config = model_runtime_config(
            &restored,
            &package,
            model_cache.root(),
            DEFAULT_MICROLLM_MAX_TOKENS,
        );
        assert_eq!(
            runtime_config.values["model_path"],
            package.to_string_lossy()
        );
        assert_eq!(runtime_config.values["backend"], "cuda-ptx");
        assert_eq!(
            runtime_config.values["ptx_virtual_arch"],
            OWNED_CUDA_PTX_VIRTUAL_ARCH
        );

        let _ = std::fs::remove_dir_all(&scratch);
    }

    #[test]
    fn models_list_every_row_carries_max_tokens_and_matches_enforced_source() {
        let (_storage_dir, descriptor) = test_storage_descriptor("models-list-source-match");
        let store = Arc::new(SynapseStore::open(&descriptor).expect("open test store"));
        let profile = test_machine_profile("test-os");
        store
            .activate_profile(&profile, 1, 1000)
            .expect("activate profile");
        let state = test_module_state(store, profile);

        let payload = models_list_payload(&state);
        let models = payload["models"]
            .as_array()
            .expect("models array in payload");

        assert!(
            !models.is_empty(),
            "models.list must contain at least one model"
        );
        for row in models {
            let max_tokens = row["max_tokens"].as_u64();
            assert!(
                max_tokens.is_some_and(|val| val > 0),
                "row {} missing positive max_tokens: {row:?}",
                row["model_id"]
            );
            if row["model_id"] == "stuck-model" {
                assert_eq!(
                    max_tokens,
                    Some(128),
                    "stuck-model max_tokens must match catalog value (128)"
                );
            }
        }
    }

    #[tokio::test]
    async fn admission_telemetry_tracks_startup_reconciliation_of_orphaned_jobs() {
        let (root, descriptor) = test_storage_descriptor("telemetry-orphans");
        let store = Arc::new(SynapseStore::open(&descriptor).unwrap());
        let profile = test_machine_profile("telemetry-orphans-os");
        store.observe_profile(&profile, 10, 1).unwrap();

        // Seed store with a job in running state from a prior process (generation 1)
        let admission = store
            .admit_job(
                "orphan-key-1",
                "orphan-digest-1",
                "embed.batch",
                1,
                &json!({"model": "test"}),
                1000,
                10_000,
                10_000,
            )
            .unwrap();
        let job_id = admission.record().job_id.clone();
        assert!(store.mark_job_running(&job_id, 1, 1001).unwrap());

        // New process startup reconciliation with generation 2
        let telemetry = AdmissionTelemetry::default();
        let reconciled =
            reconcile_startup_orphans(&store, 2, &telemetry).expect("reconcile succeeds");
        assert_eq!(reconciled, 1);

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.jobs_inherited, 1);
        assert_eq!(snapshot.jobs_failed, 1);
        assert_eq!(snapshot.jobs_minted, 0);
        assert_eq!(snapshot.jobs_completed, 0);
        assert_eq!(snapshot.jobs_open, 0);

        let record = store.get_job(&job_id).unwrap().expect("job exists");
        assert_eq!(record.state, JOB_STATE_FAILED_TRANSIENT);

        drop(store);
        let _ = std::fs::remove_dir_all(root);
    }
}

#[cfg(test)]
mod catalog_runtime_tests {
    use super::*;
    use crate::backend_check::direct_ane_catalog_available;

    /// Waits for a fixture download to reach a terminal state and asserts it
    /// committed. The bound only guards against a hang, so it matches the
    /// end-to-end catalog suite's 15 s: the download path makes several
    /// durable store writes, and a busy Windows runner has taken over 5 s for
    /// them. A timeout reports the job's last state, so a real stall is
    /// distinguishable from a slow runner.
    async fn wait_for_committed_download(state: &Arc<ModuleState>, job: &str) {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
        loop {
            let record = state.store.get_job(job).unwrap().unwrap();
            if !store::is_download_non_terminal(&record.state) {
                assert_eq!(record.state, "committed", "{record:?}");
                return;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "download {job} still {} after 15 s: {record:?}",
                record.state
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
    use crate::tests::{
        catalog_test_model, response_result, test_machine_profile, test_module_state,
        test_storage_descriptor,
    };

    #[tokio::test]
    async fn catalog_ane_without_direct_worker_refuses_before_install_or_load() {
        let (root, state) = isolated_catalog_state("ane-unavailable", true);
        assert!(!direct_ane_catalog_available(true, None));
        assert!(!direct_ane_catalog_available(
            false,
            Some(&root.join("missing"))
        ));
        // Even an existing binary cannot advertise ANE off macOS.
        assert!(!direct_ane_catalog_available(
            false,
            Some(&std::env::current_exe().unwrap())
        ));
        assert!(direct_ane_catalog_available(
            true,
            Some(&std::env::current_exe().unwrap())
        ));
        for (id, task) in [
            ("gte-modernbert-base", ModelTask::Embed),
            ("qwen3-embedding-0.6b", ModelTask::Embed),
            ("qwen3-reranker-0.6b", ModelTask::Rerank),
        ] {
            let error = resolve_serving_model(
                state.clone(),
                Some(&catalog::lane_id(id, "ane")),
                task,
                None,
                None,
                None,
            )
            .await
            .err()
            .unwrap();
            assert_eq!(error.code, "backend_unavailable", "{id}");
        }
        assert!(state.runtime.catalog.lock().unwrap().is_empty());
        drop(state);
        fs::remove_dir_all(root).unwrap();
    }

    /// Every lane the release catalog can install must satisfy what its engine
    /// and its clients require, on any host: the engine's artifact format, and
    /// batch advice in `models.list` (clients size their calls from it and
    /// refuse a lane without it). Both broke the first live install of the
    /// direct-ANE lane, and no hardware test exercised that path.
    /// Every engine that may serve embedding or reranking needs a named lane
    /// in the `job done` log line; the direct-ANE lane logged `lane=unknown`
    /// because this table was not extended with it.
    #[test]
    fn every_serving_engine_logs_a_named_lane() {
        for engine in store::OWNED_EMBED_RERANK_ENGINES {
            if *engine == "test-deterministic" {
                continue;
            }
            assert_ne!(
                execution_lane_for_engine(engine),
                "unknown",
                "engine {engine} logs lane=unknown"
            );
        }
    }

    #[test]
    fn every_release_catalog_lane_meets_its_engine_and_client_contract() {
        let (root, state) = isolated_catalog_state("catalog-client-contract", true);
        let mut lanes = 0;
        for entry in &state.runtime.release_catalog.models {
            for backend in &entry.backends {
                let spec = catalog_lane_spec(&state, entry, backend, false).unwrap();
                let lane = catalog::lane_id(&entry.id, &backend.backend);
                assert_eq!(
                    spec.artifact_format,
                    default_artifact_format(&spec.engine),
                    "{lane}: engine {} receives the wrong artifact format",
                    spec.engine
                );
                if entry.task == "transcribe" {
                    assert!(recommended_batch_for_model(&spec, None).is_none(), "{lane}");
                    let mut row = json!({"model_id": lane, "max_tokens": spec.max_tokens,
                        "recommended_batch": {"rows":8,"token_budget":8192}});
                    catalog_list_row(&state, &mut row).unwrap();
                    assert_eq!(
                        row["max_duration_ms"],
                        if backend.backend == "ane" {
                            81_850
                        } else {
                            600_000
                        },
                        "{lane}"
                    );
                    assert!(row.get("max_tokens").is_none(), "{lane}");
                    assert!(row.get("recommended_batch").is_none(), "{lane}");
                } else {
                    let advice = recommended_batch_for_model(&spec, Some(spec.max_tokens))
                        .unwrap_or_else(|| {
                            panic!("{lane}: engine {} has no batch advice", spec.engine)
                        });
                    assert!(advice.rows > 0 && advice.token_budget > 0, "{lane}");
                }
                lanes += 1;
            }
        }
        assert_eq!(lanes, 18, "every declared catalog lane must be checked");
        drop(state);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn catalog_cuda_lane_specs_rehydrate_a_supervised_worker_profile() {
        // The catalog declares `execution: explicit` on every backend, an
        // owned-metal in-process mode. Loading an owned-cuda lane rebuilds its
        // profile through owned_cuda_catalog_config, which accepts only
        // supervised worker execution, so the stored spec must carry that.
        let (root, state) = isolated_catalog_state("cuda-specs", true);
        let mut lanes = 0;
        for entry in &state.runtime.release_catalog.models {
            for backend in &entry.backends {
                if backend.engine != CUDA_WORKER_ENGINE || entry.task == "transcribe" {
                    continue;
                }
                assert_eq!(backend.execution.as_deref(), Some("explicit"));
                let lane = catalog::lane_id(&entry.id, &backend.backend);
                let spec = catalog_lane_spec(&state, entry, backend, false).unwrap();
                assert_eq!(
                    spec.owned_execution.as_deref(),
                    Some("supervised"),
                    "{lane}"
                );
                let profile = stored_owned_profile(&spec)
                    .unwrap_or_else(|error| panic!("{lane}: {error:?}"))
                    .unwrap_or_else(|| panic!("{lane}: owned-cuda spec must yield a profile"));
                assert_eq!(profile.execution, "supervised", "{lane}");
                lanes += 1;
            }
        }
        assert_eq!(lanes, 4, "every owned-cuda catalog lane must be checked");
        // The certification runner preloads the same profiles with
        // `execution: explicit`; that path goes through owned_config as well.
        let preload = CatalogProfile::load("gte-modernbert-base.owned-cuda")
            .unwrap()
            .owned_config(Some("explicit"), None)
            .unwrap();
        assert_eq!(preload.execution, "supervised");
        drop(state);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn catalog_ane_specs_use_manifest_preload_identity_and_package() {
        let (root, state) = isolated_catalog_state("ane-specs", true);
        for id in [
            "gte-modernbert-base",
            "qwen3-embedding-0.6b",
            "qwen3-reranker-0.6b",
        ] {
            let entry = state.runtime.release_catalog.entry(id).unwrap();
            let backend = entry.backend("ane").unwrap();
            let spec = catalog_lane_spec(&state, entry, backend, false).unwrap();
            let profile = CatalogProfile::load(backend.profile.as_deref().unwrap()).unwrap();
            assert_eq!(spec.engine, "ane-direct-worker");
            // The direct worker refuses anything but a converted package.
            assert_eq!(spec.artifact_format, "safetensors-package");
            assert_eq!(spec.engine_identity.build_flags["profile"], profile.id);
            assert_eq!(spec.artifact_digest, profile.artifact_digest());
            assert_eq!(
                spec.model_locator,
                ModelAssetLocator::CacheDigest {
                    digest: profile.artifact_digest()
                }
            );
            assert_eq!(spec.max_tokens, 8192);
            let (_, key) = catalog_self_check_key(&state, entry, backend).unwrap();
            assert_eq!(
                key["fixture_revision"],
                synapse_certify::self_check::seal_digest(&profile.id).unwrap()
            );
            assert_eq!(
                key["engine_identity"],
                serde_json::to_value(&spec.engine_identity).unwrap()
            );
        }
        drop(state);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn catalog_gpu_specs_use_manifest_preload_identity_and_package() {
        let (root, state) = isolated_catalog_state("gpu-specs", false);
        for entry in state
            .runtime
            .release_catalog
            .models
            .iter()
            .filter(|entry| entry.task != "transcribe")
        {
            for name in ["cuda", "vulkan"] {
                let backend = entry.backend(name).unwrap();
                let profile = CatalogProfile::load(backend.profile.as_deref().unwrap()).unwrap();
                let spec =
                    catalog_profile_lane_spec(&state, entry, backend, &profile.id, false).unwrap();
                assert_eq!(spec.engine, format!("owned-{name}"));
                assert_eq!(spec.model_id, catalog::lane_id(&entry.id, name));
                assert_eq!(spec.artifact_format, "safetensors-package");
                assert_eq!(spec.artifact_digest, profile.artifact_digest());
                assert_eq!(
                    spec.model_locator,
                    ModelAssetLocator::CacheDigest {
                        digest: profile.artifact_digest()
                    }
                );
                assert_eq!(
                    spec.engine_identity,
                    profile
                        .owned_config(
                            backend.execution.as_deref(),
                            backend.attention_units.map(|units| units as usize)
                        )
                        .unwrap()
                        .identity_override
                        .unwrap()
                );
                assert_eq!(spec.engine_identity.build_flags["profile"], profile.id);
                assert_eq!(spec.max_tokens, 8192);
                assert_eq!(spec.quant, "f16");
            }
        }
        drop(state);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn release_catalog_gpu_sequences_refuse_before_worker_spawn() {
        let (root, state) = isolated_catalog_state("gpu-release-ceiling", false);
        for name in ["cuda", "vulkan"] {
            let entry = state
                .runtime
                .release_catalog
                .entry("gte-modernbert-base")
                .unwrap();
            let backend = entry.backend(name).unwrap();
            let spec = catalog_lane_spec(&state, entry, backend, false).unwrap();
            let dir = root.join(name);
            let model = catalog_test_model(&dir, &spec);
            let text = std::iter::repeat_n("a", 8193).collect::<Vec<_>>().join(" ");
            let mut tokenized = model.tokenizer.tokenize_batch([text.as_str()]).unwrap();
            let error = compose_catalog_embed(&model, &mut tokenized).unwrap_err();
            assert_eq!(error.code, "sequence_too_long", "{name}: {error:?}");
            assert!(!dir.join("worker-called").exists());
            // The boundary control distinguishes the ceiling from a path that
            // refuses all inputs regardless of their composed token count.
            let text = std::iter::repeat_n("a", 8192).collect::<Vec<_>>().join(" ");
            let mut tokenized = model.tokenizer.tokenize_batch([text.as_str()]).unwrap();
            compose_catalog_embed(&model, &mut tokenized).unwrap();
        }
        drop(state);
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn release_catalog_feature_off_gpu_workers_refuse_below_admission() {
        let scratch = env::temp_dir().join(format!(
            "synapse-catalog-load-isolated-{}",
            std::process::id()
        ));
        let original = std::env::current_exe().unwrap();
        let executable = synapse_core::dev_binary::ckdev_binary(&original, &scratch).unwrap();
        let mut command =
            synapse_core::without_launch_nonce(std::process::Command::new(executable));
        command.args([
            "--exact",
            "catalog_runtime_tests::release_catalog_gpu_worker_load_fixture",
            "--ignored",
            "--nocapture",
        ]);
        for key in [
            "SYNAPSE_TEST_RUNNABLE_BACKENDS",
            "SYNAPSE_TEST_CATALOG",
            "SYNAPSE_CUDA_DRIVER_API",
            "CUDA_DRIVER_API",
            "SYNAPSE_CUDA_COMPUTE_CAPABILITY",
            "CUDA_COMPUTE_CAPABILITY",
            "SYNAPSE_CUDA_PACKAGING_DRIVER",
        ] {
            command.env_remove(key);
        }
        for name in ["cuda", "vulkan"] {
            let key = format!("SYNAPSE_TEST_{}_WORKER", name.to_ascii_uppercase());
            let source = env::var_os(&key).map(PathBuf::from).unwrap_or_else(|| {
                test_binaries::built_binary(&format!("ck-synapse-worker-{name}"))
            });
            command.env(key, source);
        }
        let output = command.output().unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success() && stdout.contains("1 passed; 0 failed"),
            "{stdout}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let _ = fs::remove_dir_all(scratch);
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    #[ignore = "run in an isolated process by release_catalog_feature_off_gpu_workers_refuse_below_admission"]
    fn release_catalog_gpu_worker_load_fixture() {
        assert!(cuda_floor_override().is_none());
        let (root, state) = isolated_catalog_state("gpu", false);
        for name in ["cuda", "vulkan"] {
            let source = env::var_os(format!("SYNAPSE_TEST_{}_WORKER", name.to_ascii_uppercase()))
                .map(PathBuf::from)
                .expect("isolated parent supplies staged worker paths");
            let worker = synapse_core::dev_binary::ckdev_binary(&source, root.join(name)).unwrap();
            let version = synapse_core::without_launch_nonce(std::process::Command::new(&worker))
                .arg("--version")
                .output()
                .unwrap();
            assert!(
                version.status.success()
                    && String::from_utf8_lossy(&version.stdout).contains("features=none"),
                "stage a feature-off {name} worker: {}",
                source.display()
            );
            for entry in state
                .runtime
                .release_catalog
                .models
                .iter()
                .filter(|entry| entry.task != "transcribe")
            {
                let backend = entry.backend(name).unwrap();
                let mut spec = catalog_lane_spec(&state, entry, backend, false).unwrap();
                let dir = root.join(&spec.model_id);
                let model = catalog_test_model(&dir, &spec);
                spec.tokenizer_locator = ModelAssetLocator::LocalPath {
                    path: dir.join("catalog-tokenizer.json"),
                };
                spec.tokenizer_sanitized_digest =
                    format!("sha256:{}", model.tokenizer.sanitized_sha256());
                spec.model_locator = ModelAssetLocator::LocalPath {
                    path: dir.join("absent-package.safetensors"),
                };
                spec.worker_bin = Some(worker.clone());
                let serving = root.join("s");
                spec.worker_runtime_dir = Some(serving.clone());
                let error = load_catalog_model_blocking(
                    CatalogAwareModelSpec {
                        stored: spec,
                        catalog_lane: None,
                        worker_load_context: WorkerLoadContext::new(Duration::from_secs(10), 0),
                    },
                    state.model_cache.clone(),
                    8192,
                    None,
                )
                .err()
                .expect("feature-off worker must refuse");
                if name == "vulkan" {
                    // This LOAD error can only arrive after protocol-v2 HELLO.
                    // Device discovery must refuse before opening the absent package.
                    assert!(
                        error.message.contains("vulkan_no_device"),
                        "{}: {error:?}",
                        entry.id
                    );
                } else {
                    assert_eq!(
                        error.code, "owned_cuda_unsupported",
                        "{}: {error:?}",
                        entry.id
                    );
                    assert!(
                        error
                            .message
                            .contains("owned-cuda floor refused before worker creation")
                            && error.message.contains("cuda_no_driver"),
                        "{}: {error:?}",
                        entry.id
                    );
                    assert!(
                        !serving.exists(),
                        "CUDA floor refusal must precede serving-worker spawn"
                    );
                }
            }
        }
        drop(state);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[ignore = "downloads pinned real tokenizers; integrator must compare Linux and Windows output before merge"]
    fn mint_release_catalog_fingerprints() {
        let (root, state) = isolated_catalog_state("mint-catalog-fingerprints", false);
        for entry in &state.runtime.release_catalog.models {
            let file = &entry.backend_files("cuda")["tokenizer"];
            state
                .model_cache
                .ingest(ModelCacheIngest {
                    source_url: format!(
                        "https://huggingface.co/{}/resolve/{}/{}",
                        entry.upstream.hf_repo, entry.upstream.revision, file.path
                    ),
                    expected_digest: Some(format!("sha256:{}", file.sha256)),
                    format: "json".into(),
                    tokenizer_path: None,
                    pin_module_id: None,
                })
                .unwrap();
            let tokenizer_path = state.model_cache.blob_path(&file.sha256);
            assert_eq!(
                fs::metadata(&tokenizer_path).unwrap().len(),
                file.size_bytes
            );
            assert_eq!(sha256_file(&tokenizer_path).unwrap(), file.sha256);
            let tokenizer = SanitizedTokenizer::from_file(
                &tokenizer_path,
                TokenizerConfig {
                    max_tokens: usize::MAX,
                },
            )
            .unwrap();
            for name in ["cuda", "vulkan"] {
                let backend = entry.backend(name).unwrap();
                let profile_id = backend.profile.as_deref().unwrap();
                let profile = CatalogProfile::load(profile_id).unwrap();
                profile.validate_readout(&tokenizer).unwrap();
                // Fingerprints bind the converted package's declared checksum and
                // the real tokenizer, not a device or a worker transport. No model
                // conversion or GPU is needed to mint the cross-platform pins.
                let mut spec =
                    catalog_profile_lane_spec(&state, entry, backend, profile_id, false).unwrap();
                spec.tokenizer_sanitized_digest =
                    format!("sha256:{}", tokenizer.sanitized_sha256());
                let spec =
                    normalize_catalog_model(spec, &InlineConfig::default(), &JobConfig::default())
                        .unwrap();
                println!("{} {}", spec.model_id, spec.fingerprint.0);
            }
        }
        drop(state);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[ignore = "requires original pinned HF snapshots; converts packages but does not run accelerators"]
    fn catalog_original_snapshots_rebuild_ane_lane_pins() {
        let hub =
            PathBuf::from(env::var_os("SYNAPSE_PINNED_HF_CACHE").expect("SYNAPSE_PINNED_HF_CACHE"));
        let (root, state) = isolated_catalog_state("ane-real-specs", true);
        let packages = env::var_os("ANE_TEST_PACKAGES").map(PathBuf::from);
        fs::create_dir_all(state.model_cache.blob_path("placeholder").parent().unwrap()).unwrap();
        for id in [
            "gte-modernbert-base",
            "qwen3-embedding-0.6b",
            "qwen3-reranker-0.6b",
        ] {
            let entry = state.runtime.release_catalog.entry(id).unwrap();
            let backend = entry.backend("ane").unwrap();
            let snapshot = hub.join(format!(
                "models--{}/snapshots/{}",
                entry.upstream.hf_repo.replace('/', "--"),
                entry.upstream.revision
            ));
            for file in entry.backend_files("ane").values() {
                let original = snapshot.join(&file.path);
                assert_eq!(
                    sha256_file(&original).unwrap(),
                    file.sha256,
                    "{id} {}",
                    file.path
                );
                fs::hard_link(
                    original.canonicalize().unwrap(),
                    state.model_cache.blob_path(&file.sha256),
                )
                .unwrap();
            }
            let package_digest = CatalogProfile::load(backend.profile.as_deref().unwrap())
                .unwrap()
                .artifact_digest();
            if let Some(directory) = &packages {
                let package = directory.join(format!("{id}.safetensors"));
                if package.is_file() {
                    state
                        .model_cache
                        .ingest(ModelCacheIngest {
                            source_url: local_file_url(&package),
                            expected_digest: Some(package_digest.clone()),
                            format: "safetensors".into(),
                            tokenizer_path: None,
                            pin_module_id: None,
                        })
                        .unwrap();
                }
            }
            let spec = catalog_lane_spec(&state, entry, backend, true).unwrap();
            // Save only converted weights whose SHA-256 matches the package
            // checksum in bench/parity/models.json. The later capacity test must
            // load those exact converted bytes, not weights from another model
            // revision or conversion method that could have different resource use.
            if let Some(directory) = &packages {
                fs::create_dir_all(directory).unwrap();
                fs::copy(
                    state.model_cache.blob_path(&package_digest),
                    directory.join(format!("{id}.safetensors")),
                )
                .unwrap();
            }
            assert_eq!(spec.fingerprint.0, backend.fingerprint, "{id}");
            // Compare catalog installation with a startup preload: a model loaded
            // from configuration before requests arrive. Hardware certification
            // uses that startup path, so both must produce the same fingerprint
            // and engine identity for the same weights and tokenizer.
            let preload: PreloadModelConfig = serde_json::from_value(json!({
                "model_id": spec.model_id, "engine": spec.engine, "profile": backend.profile,
                "task": spec.task, "pooling": spec.pooling, "normalize": spec.normalize,
                "model_path": state.model_cache.blob_path(&spec.artifact_digest),
                "tokenizer_path": state.model_cache.blob_path(&entry.backend_files("ane")["tokenizer"].sha256)
            })).unwrap();
            let rebuilt = build_preload_catalog_model(
                0,
                preload,
                &InlineConfig::default(),
                &JobConfig::default(),
            )
            .unwrap();
            assert_eq!(spec.fingerprint, rebuilt.fingerprint, "{id}");
            assert_eq!(spec.engine_identity, rebuilt.engine_identity, "{id}");
        }
        drop(state);
        fs::remove_dir_all(root).unwrap();
    }

    pub(super) fn isolated_catalog_state(
        label: &str,
        runnable: bool,
    ) -> (PathBuf, Arc<ModuleState>) {
        let (root, descriptor) = test_storage_descriptor(label);
        let store = Arc::new(SynapseStore::open(&descriptor).unwrap());
        let profile = test_machine_profile("catalog-test-os");
        store.observe_profile(&profile, 1, 1).unwrap();
        let mut state = test_module_state(store, profile);
        let inner = Arc::get_mut(&mut state).unwrap();
        inner.model_cache = Arc::new(ModelCache::new(root.join("cache")));
        let runtime = Arc::get_mut(&mut inner.runtime).unwrap();
        runtime.catalog.lock().unwrap().clear();
        runtime.runnable_backends = if runnable {
            BTreeSet::from(["metal".into()])
        } else {
            BTreeSet::new()
        };
        (root, state)
    }

    #[cfg(unix)]
    #[test]
    fn startup_catalog_lane_load_executes_environment_hard_link_without_preload() {
        let original = env::current_exe().unwrap();
        let built_dir = original.parent().unwrap().parent().unwrap();
        let source = test_binaries::built_binary("synapse-worker-timeout-mock");
        assert!(
            source.is_file(),
            "build synapse-worker-timeout-mock before module lib tests"
        );
        let scratch = built_dir.join(format!("synapse-env-worker-{}", std::process::id()));
        let executable =
            synapse_core::dev_binary::ckdev_binary_hard_link(&original, &scratch).unwrap();
        let mut command =
            synapse_core::without_launch_nonce(std::process::Command::new(executable));
        command.args([
            "--exact",
            "catalog_runtime_tests::startup_environment_worker_fixture",
            "--ignored",
            "--nocapture",
        ]);
        command.env("SYNAPSE_TEST_STARTUP_WORKER", source);
        for key in [
            "SYNAPSE_TEST_CATALOG",
            "SYNAPSE_TEST_RUNNABLE_BACKENDS",
            "SYNAPSE_TIMEOUT_WORKER_EMBED_DIMS",
            "SYNAPSE_TIMEOUT_WORKER_EMBED_N",
            "SYNAPSE_TIMEOUT_WORKER_NO_BUCKETS",
            "SYNAPSE_TIMEOUT_WORKER_BUCKETS",
            "SYNAPSE_TIMEOUT_WORKER_ABORT_ON_EMBED",
        ] {
            command.env_remove(key);
        }
        let output = command.output().unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success() && stdout.contains("1 passed; 0 failed"),
            "{stdout}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let _ = fs::remove_dir_all(scratch);
    }

    #[cfg(unix)]
    #[test]
    #[ignore = "run in an isolated process by startup_catalog_lane_load_executes_environment_hard_link_without_preload"]
    fn startup_environment_worker_fixture() {
        use std::os::unix::fs::MetadataExt;
        assert!(ModuleConfig::default().preload_models.is_empty());
        let (root, state) = isolated_catalog_state("startup-env-worker", false);
        assert!(state.runtime.loaded_models().is_empty());
        let source = PathBuf::from(env::var_os("SYNAPSE_TEST_STARTUP_WORKER").unwrap());
        let aliases = source
            .parent()
            .unwrap()
            .join(format!("startup-worker-alias-{}", std::process::id()));
        let worker = synapse_core::dev_binary::ckdev_binary_hard_link(&source, &aliases).unwrap();
        let source_meta = fs::metadata(&source).unwrap();
        let alias_meta = fs::metadata(&worker).unwrap();
        assert_eq!(
            (source_meta.dev(), source_meta.ino()),
            (alias_meta.dev(), alias_meta.ino())
        );
        let log = root.join("worker-started");
        env::set_var("SYNAPSE_TIMEOUT_WORKER_START_LOG", &log);
        env::set_var(worker_binary_env_var("owned-vulkan"), &worker);
        let entry = state
            .runtime
            .release_catalog
            .entry("gte-modernbert-base")
            .unwrap();
        let backend = entry.backend("vulkan").unwrap();
        let mut spec = catalog_lane_spec(&state, entry, backend, false).unwrap();
        assert!(spec.worker_bin.is_none());
        let fixture_dir = root.join("model");
        let fixture = catalog_test_model(&fixture_dir, &spec);
        spec.tokenizer_locator = ModelAssetLocator::LocalPath {
            path: fixture_dir.join("catalog-tokenizer.json"),
        };
        spec.tokenizer_sanitized_digest =
            format!("sha256:{}", fixture.tokenizer.sanitized_sha256());
        let package = fixture_dir.join("package.safetensors");
        fs::write(&package, b"mock worker does not interpret package bytes").unwrap();
        spec.model_locator = ModelAssetLocator::LocalPath { path: package };
        let runtime_dir = PathBuf::from("/tmp").join(format!("sew-{}", std::process::id()));
        spec.worker_runtime_dir = Some(runtime_dir.clone());
        drop(fixture);
        let model = load_catalog_model_blocking(
            CatalogAwareModelSpec {
                stored: spec,
                catalog_lane: None,
                worker_load_context: WorkerLoadContext::new(Duration::from_secs(10), 0),
            },
            state.model_cache.clone(),
            8192,
            None,
        )
        .unwrap();
        assert_eq!(model.model_id, "gte-modernbert-base-vulkan");
        let executed = PathBuf::from(fs::read_to_string(&log).unwrap())
            .canonicalize()
            .unwrap();
        assert_eq!(executed, worker.canonicalize().unwrap());
        drop(model);
        drop(state);
        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(aliases).unwrap();
        fs::remove_dir_all(runtime_dir).unwrap();
    }

    #[test]
    fn catalog_endpoint_and_asset_guards_reject_unpinned_network_io() {
        for value in [
            "https://example.org/a",
            "https://example.org?x=1",
            "https://user@example.org",
            "file:///tmp/models",
            "https://example.org#fragment",
        ] {
            assert!(validate_hf_endpoint(value).is_err(), "accepted {value}");
        }
        assert!(validate_hf_endpoint("http://127.0.0.1:1234").is_ok());
        let sha = "a".repeat(64);
        assert!(validate_resolved_asset(
            "https://elsewhere.org/model",
            None,
            "https://huggingface.co"
        )
        .is_err());
        assert!(validate_resolved_asset(
            "https://huggingface.co/repo/resolve/main/model",
            Some(&sha),
            "https://huggingface.co"
        )
        .is_err());
        assert!(validate_resolved_asset(
            "http://127.0.0.1:1234/repo/resolve/main/model",
            Some(&sha),
            "http://127.0.0.1:1234"
        )
        .is_err());
        assert!(validate_resolved_asset(
            "https://cdn.example.org/body",
            Some(&sha),
            "https://huggingface.co"
        )
        .is_ok());
        assert!(
            validate_resolved_asset("file:///tmp/model", None, "https://huggingface.co").is_ok()
        );
        let url = huggingface_resolve_url(
            "http://127.0.0.1:1234",
            "owner/repo",
            &"a".repeat(40),
            "dir/a b.json",
        )
        .unwrap();
        assert_eq!(
            url,
            format!(
                "http://127.0.0.1:1234/owner/repo/resolve/{}/dir/a%20b.json",
                "a".repeat(40)
            )
        );
        assert!(
            huggingface_resolve_url("https://huggingface.co", "owner/repo", "main", "model")
                .is_err()
        );
    }

    #[tokio::test]
    async fn catalog_default_resolution_refuses_before_loading_and_ignores_knobs() {
        let (root, state) = isolated_catalog_state("default-resolution", true);
        let not_installed =
            resolve_serving_model(state.clone(), None, ModelTask::Embed, None, None, None)
                .await
                .err()
                .unwrap();
        assert_eq!(not_installed.code, "model_not_installed");
        assert_eq!(
            not_installed.details.unwrap()["catalog_id"],
            "gte-modernbert-base"
        );
        let mismatch = resolve_serving_model(
            state.clone(),
            Some("gte-modernbert-base"),
            ModelTask::Rerank,
            None,
            None,
            None,
        )
        .await
        .err()
        .unwrap();
        assert_eq!(mismatch.code, "invalid_request");
        let fingerprint = resolve_serving_model(
            state.clone(),
            Some("gte-modernbert-base"),
            ModelTask::Embed,
            Some(&"1".repeat(64)),
            None,
            None,
        )
        .await
        .err()
        .unwrap();
        assert_eq!(fingerprint.code, "substitution_rejected");
        let unknown = resolve_serving_model(
            state.clone(),
            Some("gte-modernbert-base-ane"),
            ModelTask::Embed,
            None,
            None,
            None,
        )
        .await
        .err()
        .unwrap();
        // The catalog declares gte-modernbert-base-ane, but this test runtime
        // advertises only Metal. A missing direct Neural Engine worker must
        // return backend_unavailable, not the unknown_model error for an
        // undeclared model or a fallback to Metal.
        assert_eq!(unknown.code, "backend_unavailable");
        assert!(state.runtime.catalog.lock().unwrap().is_empty());
        drop(state);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn catalog_unavailable_backend_precedes_missing_install() {
        let (root, state) = isolated_catalog_state("unavailable-before-install", false);
        let error = select_catalog_lane(
            &state,
            Some("gte-modernbert-base-metal"),
            ModelTask::Embed,
            None,
            None,
        )
        .unwrap_err();
        assert_eq!(
            error.code, "backend_unavailable",
            "backend refusal must precede missing installation"
        );
        let entry = state
            .runtime
            .release_catalog
            .entry("gte-modernbert-base")
            .unwrap();
        assert!(current_catalog_install(&state, entry, "metal")
            .unwrap()
            .is_none());
        drop(state);
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn catalog_load_rejects_fingerprint_mismatch_before_engine_invocation() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let (root, mut state) = isolated_catalog_state("fingerprint-mismatch", true);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let backend_name = if cfg!(target_os = "macos") {
            "metal"
        } else {
            "cuda"
        };
        let catalog_id = "fingerprint-mismatch-fixture";
        let bodies = std::sync::Arc::new(BTreeMap::from([
            ("model.safetensors".to_string(), b"test model bytes".to_vec()),
            (
                "tokenizer.json".to_string(),
                br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"WhitespaceSplit"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"a":1},"unk_token":"[UNK]"}}"#.to_vec(),
            ),
            ("config.json".to_string(), b"{}".to_vec()),
        ]));
        let runtime = Arc::get_mut(&mut Arc::get_mut(&mut state).unwrap().runtime).unwrap();
        runtime.hf_endpoint = endpoint;
        runtime.runnable_backends = BTreeSet::from([backend_name.into()]);
        let entry = runtime
            .release_catalog
            .models
            .iter_mut()
            .find(|entry| entry.id == "gte-modernbert-base")
            .unwrap();
        entry.id = catalog_id.into();
        let backend = entry
            .backends
            .iter_mut()
            .find(|backend| backend.backend == backend_name)
            .unwrap();
        backend.profile = None;
        backend.fingerprint = "0".repeat(64);
        for file in &mut entry.files {
            file.backends = vec![backend_name.into()];
            let body = &bodies[&file.path];
            file.sha256 = sha256_hex(body);
            file.size_bytes = body.len() as u64;
        }
        let pinned_fingerprint = backend.fingerprint.clone();
        let lane = catalog::lane_id(&entry.id, &backend.backend);

        let server_bodies = bodies.clone();
        let server = tokio::spawn(async move {
            for _ in 0..server_bodies.len() {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut buffer = [0u8; 4096];
                let size = stream.read(&mut buffer).await.unwrap();
                let request = String::from_utf8_lossy(&buffer[..size]);
                let path = request
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .and_then(|path| path.rsplit('/').next())
                    .unwrap();
                let body = &server_bodies[path];
                stream
                    .write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        )
                        .as_bytes(),
                    )
                    .await
                    .unwrap();
                stream.write_all(body).await.unwrap();
            }
        });

        let accepted = response_result(
            models_download(
                state.clone(),
                json!({"catalog_id":catalog_id,"request_key":"fingerprint-mismatch"}),
            )
            .await,
            "models.download",
        );
        assert_eq!(accepted["state"], "queued");
        wait_for_committed_download(&state, accepted["job_id"].as_str().unwrap()).await;

        let load_attempts_path = root.join("catalog-load-attempts");
        let lane_env = "SYNAPSE_TEST_CATALOG_FAULT_LANE";
        let attempts_env = "SYNAPSE_TEST_CATALOG_LOAD_ATTEMPTS";
        let previous_lane = std::env::var_os(lane_env);
        let previous_attempts = std::env::var_os(attempts_env);
        std::env::set_var(lane_env, &lane);
        std::env::set_var(attempts_env, &load_attempts_path);
        let result = resolve_serving_model(
            state.clone(),
            Some(&lane),
            ModelTask::Embed,
            None,
            None,
            Some(5_000),
        )
        .await;
        let lane_state = model_slot_snapshot(&state.runtime, &lane).map(|slot| slot.state);
        let load_attempts = fs::read_to_string(&load_attempts_path).unwrap_or_default();
        if let Some(value) = previous_lane {
            std::env::set_var(lane_env, value);
        } else {
            std::env::remove_var(lane_env);
        }
        if let Some(value) = previous_attempts {
            std::env::set_var(attempts_env, value);
        } else {
            std::env::remove_var(attempts_env);
        }

        server.abort();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        while Arc::strong_count(&state) > 1 {
            assert!(tokio::time::Instant::now() < deadline);
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        drop(state);
        fs::remove_dir_all(root).unwrap();

        assert!(
            load_attempts.is_empty(),
            "catalog engine load attempts should stay empty, got {load_attempts:?}"
        );
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("a mismatched catalog pin must refuse the lane"),
        };
        assert_eq!(error.code, "artifact_invalid", "{error:?}");
        let details = error.details.expect("fingerprint mismatch details");
        assert_eq!(details["expected_fingerprint"], pinned_fingerprint);
        let actual_fingerprint = details["actual_fingerprint"].as_str().unwrap();
        assert_eq!(actual_fingerprint.len(), 64);
        assert!(actual_fingerprint
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit()));
        assert_ne!(actual_fingerprint, pinned_fingerprint);
        assert!(matches!(lane_state, Some(ModelRuntimeState::Failed(_))));
    }

    #[tokio::test]
    async fn catalog_no_accelerator_lists_every_entry_and_refuses_downloads() {
        let (root, state) = isolated_catalog_state("no-accelerator", false);
        let result = response_result(
            models_catalog(state.clone(), json!({})).await,
            "models.catalog",
        );
        assert_eq!(result["models"].as_array().unwrap().len(), 5);
        for row in result["models"].as_array().unwrap() {
            assert_eq!(row["download_bytes"], 0);
            assert_eq!(row["install_state"], "not_installed");
        }
        let result = response_result(
            models_download(state.clone(), json!({"catalog_id":"gte-modernbert-base"})).await,
            "models.download",
        );
        assert_eq!(result["error"]["code"], "backend_unavailable");
        let result = response_result(
            models_download(
                state.clone(),
                json!({"catalog_id":"gte-modernbert-base-ane"}),
            )
            .await,
            "models.download",
        );
        assert_eq!(result["error"]["code"], "invalid_request");
        let result = response_result(
            models_remove(state.clone(), json!({"catalog_id":"minilm"})).await,
            "models.remove",
        );
        assert_eq!(result["error"]["code"], "unknown_model");
        drop(state);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn catalog_self_check_id_matches_the_runtime_key_for_all_seven_fields() {
        let (root, state) = isolated_catalog_state("self-check-pure-id", false);
        for entry in &state.runtime.release_catalog.models {
            for backend in &entry.backends {
                let (id, key) = catalog_self_check_key(&state, entry, backend).unwrap();
                let pure_id = catalog_self_check_id(
                    key["catalog_id"].as_str().unwrap(),
                    key["manifest_digest"].as_str().unwrap(),
                    key["backend"].as_str().unwrap(),
                    key["fingerprint"].as_str().unwrap(),
                    &key["engine_identity"],
                    key["os_build"].as_str().unwrap(),
                    key["fixture_revision"].as_str().unwrap(),
                )
                .unwrap();
                assert_eq!(pure_id, id);
                assert_eq!(id, sha256_hex(catalog::jcs(&key).unwrap().as_bytes()));
                for field in [
                    "catalog_id",
                    "manifest_digest",
                    "backend",
                    "fingerprint",
                    "engine_identity",
                    "os_build",
                    "fixture_revision",
                ] {
                    let mut changed = key.clone();
                    changed[field] = json!("different");
                    assert_ne!(id, sha256_hex(catalog::jcs(&changed).unwrap().as_bytes()));
                }
            }
        }
        drop(state);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn catalog_self_check_generation_fences_late_completion_and_projects_mirrors() {
        let (root, state) = isolated_catalog_state("self-check-generation", true);
        let entry = state
            .runtime
            .release_catalog
            .entry("gte-modernbert-base")
            .unwrap();
        let backend = &entry.backends[0];
        let (id, key) = catalog_self_check_key(&state, entry, backend).unwrap();
        let first = catalog_check_generation(&state, &id, &key).unwrap();
        assert!(catalog_complete_check(&state, &id, first, "pending", None).unwrap());
        assert!(!catalog_complete_check(&state, &id, first, "passed", None).unwrap());
        let second = catalog_check_generation(&state, &id, &key).unwrap();
        assert!(second > first);
        assert!(!catalog_complete_check(&state, &id, first, "passed", None).unwrap());
        assert!(
            catalog_complete_check(&state, &id, second, "failed", Some("numerical_mismatch"))
                .unwrap()
        );
        let mut row = json!({"model_id":"gte-modernbert-base-metal"});
        catalog_list_row(&state, &mut row).unwrap();
        assert_eq!(row["certified"], false);
        assert_eq!(row["serving_admission"], "disabled");
        assert_eq!(row["serving_admission_reason"], "self_check_failed");
        assert_eq!(row["self_check"]["reason"], "numerical_mismatch");
        assert!(row["self_check"]["checked_at_ms"].as_u64().is_some());
        drop(state);
        fs::remove_dir_all(root).unwrap();
    }

    /// After an OS update the self-check key changes, so a lane that served
    /// before reads `pending`. Startup must pick it up for revalidation (a
    /// consumer gating on `certified` would never load it), but must not
    /// auto-load a fresh install that never passed.
    #[test]
    fn startup_revalidates_only_lanes_that_passed_under_an_earlier_key() {
        let (root, state) = isolated_catalog_state("revalidate-after-os-update", true);
        let entry = state
            .runtime
            .release_catalog
            .entry("gte-modernbert-base")
            .unwrap()
            .clone();
        let backend = entry.backend("metal").unwrap().clone();
        state
            .store
            .sqlite()
            .with_conn_fenced(|tx| {
                tx.execute(
                    "INSERT INTO catalog_installs VALUES (?1,?2,?3)",
                    rusqlite::params![entry.id, entry.manifest_digest(), backend.backend],
                )?;
                Ok(())
            })
            .unwrap();
        let selected = |state: &ModuleState| {
            catalog_lanes_needing_revalidation(state)
                .unwrap()
                .into_iter()
                .map(|(entry, backend)| catalog::lane_id(&entry.id, &backend.backend))
                .collect::<Vec<_>>()
        };
        assert!(
            selected(&state).is_empty(),
            "a fresh install that never passed loads on first use, not at startup"
        );
        // The same lane passed before the OS update, under the old build.
        let (id, key) = catalog_self_check_key(&state, &entry, &backend).unwrap();
        let mut old_key = key.clone();
        old_key["os_build"] = json!("previous-os-build");
        let old_id = format!("{id}-previous-os-build");
        let generation = catalog_check_generation(&state, &old_id, &old_key).unwrap();
        assert!(catalog_complete_check(&state, &old_id, generation, "passed", None).unwrap());
        assert_eq!(
            selected(&state),
            vec!["gte-modernbert-base-metal".to_string()]
        );
        // Once it passes under the current key, there is nothing left to do.
        let generation = catalog_check_generation(&state, &id, &key).unwrap();
        assert!(catalog_complete_check(&state, &id, generation, "passed", None).unwrap());
        assert!(selected(&state).is_empty());
        drop(state);
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn catalog_download_streams_pinned_files_commits_and_reuses_without_network() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let (root, mut state) = isolated_catalog_state("download-stream", true);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let body = b"catalog fixture bytes";
        let revision = "a".repeat(40);
        let runtime = Arc::get_mut(&mut Arc::get_mut(&mut state).unwrap().runtime).unwrap();
        runtime.hf_endpoint = endpoint;
        let entry = runtime
            .release_catalog
            .models
            .iter_mut()
            .find(|e| e.id == "gte-modernbert-base")
            .unwrap();
        entry.upstream.revision = revision.clone();
        for file in &mut entry.files {
            file.sha256 = sha256_hex(body);
            file.size_bytes = body.len() as u64;
        }
        let requests = Arc::new(AtomicU64::new(0));
        let seen = requests.clone();
        let server = tokio::spawn(async move {
            loop {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut buffer = [0u8; 4096];
                let size = stream.read(&mut buffer).await.unwrap();
                let request = String::from_utf8_lossy(&buffer[..size]);
                assert!(
                    request.contains(&format!("/resolve/{revision}/")),
                    "{request}"
                );
                seen.fetch_add(1, Ordering::SeqCst);
                stream
                    .write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        )
                        .as_bytes(),
                    )
                    .await
                    .unwrap();
                stream.write_all(body).await.unwrap();
            }
        });
        let accepted = response_result(
            models_download(
                state.clone(),
                json!({"catalog_id":"gte-modernbert-base","request_key":"first"}),
            )
            .await,
            "models.download",
        );
        assert_eq!(accepted["state"], "queued");
        let job = accepted["job_id"].as_str().unwrap();
        wait_for_committed_download(&state, job).await;
        assert_eq!(requests.load(Ordering::SeqCst), 1);
        let same = response_result(
            models_download(
                state.clone(),
                json!({"catalog_id":"gte-modernbert-base","request_key":"first"}),
            )
            .await,
            "models.download",
        );
        assert_eq!(same["job_id"], job);
        assert_eq!(same["state"], "committed");
        let status = response_result(
            model_status(state.clone(), json!({"job_id":job})).await,
            "model.status",
        );
        assert_eq!(
            status,
            json!({"job_id":job,"state":"committed","kind":"models.download"})
        );
        let removed = response_result(
            models_remove(state.clone(), json!({"catalog_id":"gte-modernbert-base"})).await,
            "models.remove",
        );
        assert_eq!(removed["freed_bytes"], body.len() as u64);
        assert_eq!(removed["removed_manifests"].as_array().unwrap().len(), 1);
        assert_eq!(requests.load(Ordering::SeqCst), 1);
        server.abort();
        // The detached download task still holds a state clone, and with it
        // the store connection, for a moment after the job reads committed.
        // Windows refuses to delete a directory with an open file, so wait
        // until this test holds the only reference before removing it.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        while Arc::strong_count(&state) > 1 {
            assert!(
                tokio::time::Instant::now() < deadline,
                "{} module-state references still held",
                Arc::strong_count(&state)
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        drop(state);
        fs::remove_dir_all(root).unwrap();
    }
}

// Compiled release entries require a complete install and their own numerical
// self-check. User-registered models' probe results and performance preferences
// cannot authorize a different backend for a compiled release entry.
fn default_hf_endpoint() -> String {
    "https://huggingface.co".into()
}
fn pinned_revision(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|b| b.is_ascii_hexdigit())
}
fn validate_hf_endpoint(endpoint: &str) -> Result<(), String> {
    let url = Url::parse(endpoint).map_err(|e| format!("hf_endpoint: {e}"))?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !matches!(url.path(), "" | "/")
        || url.query().is_some()
        || url.fragment().is_some()
        || !url.username().is_empty()
        || url.password().is_some()
    {
        return Err(
            "hf_endpoint must be an absolute HTTP(S) origin without path, query or userinfo".into(),
        );
    }
    Ok(())
}
fn validate_resolved_asset(
    source: &str,
    digest: Option<&str>,
    endpoint: &str,
) -> Result<(), String> {
    let Ok(url) = Url::parse(source) else {
        return Ok(());
    };
    if !matches!(url.scheme(), "http" | "https") {
        return Ok(());
    }
    if !digest.is_some_and(|d| {
        let d = d.strip_prefix("sha256:").unwrap_or(d);
        d.len() == 64 && d.bytes().all(|b| b.is_ascii_hexdigit())
    }) {
        return Err("every HTTP(S) asset requires a sha256 digest".into());
    }
    let configured = Url::parse(endpoint).map_err(|e| e.to_string())?;
    let same_origin = |other: &Url| {
        url.host_str() == other.host_str()
            && url.port_or_known_default() == other.port_or_known_default()
    };
    if url.host_str() == Some("huggingface.co") && url.port_or_known_default() == Some(443)
        || same_origin(&configured)
    {
        let segments = url
            .path_segments()
            .ok_or("invalid Hugging Face URL")?
            .collect::<Vec<_>>();
        if !segments
            .windows(2)
            .any(|pair| pair[0] == "resolve" && pinned_revision(pair[1]))
        {
            return Err("Hugging Face resolve URLs require a pinned 40-hex revision".into());
        }
    }
    Ok(())
}
fn runtime_release_catalog() -> Result<catalog::Catalog, ModuleError> {
    #[cfg(feature = "test-support")]
    if let Ok(path) = env::var("SYNAPSE_TEST_CATALOG") {
        let text = fs::read_to_string(path)
            .map_err(|e| ModuleError::Config(format!("test catalog: {e}")))?;
        return catalog::load_schema_valid(&text).map_err(|e| ModuleError::Config(e.to_string()));
    }
    catalog::compiled_catalog()
        .cloned()
        .map_err(|e| ModuleError::Config(e.to_string()))
}
fn catalog_floor_observations(
    config: &ModuleConfig,
    runnable: &BTreeSet<String>,
) -> BTreeMap<String, Value> {
    let mut observed = BTreeMap::new();
    for (backend, engine) in [("cuda", CUDA_WORKER_ENGINE), ("vulkan", "owned-vulkan")] {
        if !runnable.contains(backend) {
            continue;
        }
        let configured = config
            .preload_models
            .iter()
            .find(|m| m.engine == engine && m.worker_bin.is_some())
            .and_then(|m| m.worker_bin.as_deref());
        if let Some(worker) = resolve_catalog_worker(engine, configured) {
            if let Ok(output) =
                catalog_probe::cached_probe(&worker, catalog_probe::ProbeKind::Floor)
            {
                if let Ok(envelope) = catalog_probe::floor_result(output) {
                    observed.insert(backend.into(), envelope["observed"].clone());
                }
            }
        }
    }
    observed
}

#[derive(Debug)]
struct CatalogLaneRefusal {
    code: &'static str,
    details: String,
}

fn transcribe_profile_floor_reason(
    profile: &Value,
    observed: &Value,
) -> Option<CatalogLaneRefusal> {
    let required: &[&str] = match profile["lane"].as_str() {
        Some("owned-cuda") => &[
            "cuda_min_driver_api",
            "cuda_min_compute_major",
            "cuda_min_compute_minor",
        ],
        Some("owned-vulkan") => &[
            "vulkan_min_device_local_bytes",
            "vulkan_min_storage_buffer_range",
        ],
        _ => return None,
    };
    for field in required {
        if profile[*field].as_u64().is_none() {
            return Some(CatalogLaneRefusal {
                code: "probe_failed",
                details: format!("missing manifest floor {field}"),
            });
        }
    }
    let mut readings = Vec::new();
    match profile["lane"].as_str()? {
        "owned-cuda" => {
            for (floor, reading, label) in [
                (
                    "cuda_min_driver_api",
                    &observed["driver_api"],
                    "CUDA driver API",
                ),
                (
                    "cuda_min_compute_major",
                    &observed["compute_capability"]["major"],
                    "CUDA compute major",
                ),
                (
                    "cuda_min_compute_minor",
                    &observed["compute_capability"]["minor"],
                    "CUDA compute minor",
                ),
            ] {
                if reading.as_u64().is_none() {
                    return Some(CatalogLaneRefusal {
                        code: "probe_failed",
                        details: format!("missing {label} reading for {floor}"),
                    });
                }
            }
            let major = profile["cuda_min_compute_major"].as_u64()?;
            let minor = profile["cuda_min_compute_minor"].as_u64()?;
            let actual_major = observed["compute_capability"]["major"].as_u64()?;
            let actual_minor = observed["compute_capability"]["minor"].as_u64()?;
            if (actual_major, actual_minor) < (major, minor) {
                return Some(CatalogLaneRefusal { code: "device_below_floor", details: format!("CUDA compute capability {actual_major}.{actual_minor} below this lane's {major}.{minor} floor") });
            }
            readings.push((
                "CUDA driver API",
                profile["cuda_min_driver_api"].as_u64()?,
                observed["driver_api"].as_u64(),
            ));
        }
        "owned-vulkan" => {
            let heap = observed["device_local_heaps"].as_array().and_then(|heaps| {
                heaps
                    .iter()
                    .map(Value::as_u64)
                    .collect::<Option<Vec<_>>>()
                    .and_then(|h| h.into_iter().max())
            });
            readings.push((
                "device-local memory bytes",
                profile["vulkan_min_device_local_bytes"].as_u64()?,
                heap,
            ));
            readings.push((
                "storage buffer range bytes",
                profile["vulkan_min_storage_buffer_range"].as_u64()?,
                observed["max_storage_buffer_range"].as_u64(),
            ));
        }
        _ => return None,
    }
    for (label, floor, actual) in readings {
        let Some(actual) = actual else {
            return Some(CatalogLaneRefusal {
                code: "probe_failed",
                details: format!("missing {label} reading for this lane's {floor} floor"),
            });
        };
        if actual < floor {
            return Some(CatalogLaneRefusal {
                code: "device_below_floor",
                details: format!("{label} {actual} below this lane's {floor} floor"),
            });
        }
    }
    None
}

fn catalog_lane_backend_reason(
    runtime: &RuntimeState,
    entry: &catalog::CatalogEntry,
    backend: &catalog::CatalogBackend,
    platform: synapse_core::Platform,
) -> Option<CatalogLaneRefusal> {
    if let Some(code) = catalog_backend_reason(runtime, &backend.backend, platform) {
        return Some(CatalogLaneRefusal {
            code,
            details: String::new(),
        });
    }
    if entry.task != "transcribe" || !matches!(backend.backend.as_str(), "cuda" | "vulkan") {
        return None;
    }
    let profile = match CatalogProfile::load(backend.profile.as_deref().unwrap_or("")) {
        Ok(profile) => profile,
        Err(error) => {
            return Some(CatalogLaneRefusal {
                code: "probe_failed",
                details: error.to_string(),
            })
        }
    };
    transcribe_profile_floor_reason(
        profile.profile(),
        runtime
            .backend_floor_observed
            .get(&backend.backend)
            .unwrap_or(&Value::Null),
    )
}

fn catalog_backend_reason(
    runtime: &RuntimeState,
    backend: &str,
    platform: synapse_core::Platform,
) -> Option<&'static str> {
    backend_reason(
        &runtime.runnable_backends,
        &runtime.backend_reasons,
        backend,
        platform,
    )
}
fn catalog_wire_error(
    code: &str,
    details: Value,
    message: impl Into<String>,
) -> WireOperationError {
    let stable_code: synapse_core::StableErrorCode =
        serde_json::from_value(json!(code)).expect("known stable code");
    let mut error = WireOperationError::from_stable(
        StableError::new(
            stable_code,
            if code == "download_failed" {
                ErrorClass::Transient
            } else {
                ErrorClass::Permanent
            },
            if code == "download_failed" {
                Some(1000)
            } else {
                None
            },
            code == "download_failed",
        ),
        message,
    );
    error.details = Some(details);
    error
}
fn catalog_store_error(error: impl std::fmt::Display) -> WireOperationError {
    WireOperationError {
        code: "store_failure".into(),
        class: ErrorClass::Transient,
        retry_after_ms: Some(250),
        safe_to_retry_same_request: true,
        message: error.to_string(),
        details: None,
    }
}
fn catalog_unknown(id: &str) -> WireOperationError {
    catalog_wire_error(
        "unknown_model",
        json!({"model_id":id}),
        format!("unknown model '{id}'"),
    )
}
fn catalog_request_digest(entry: &catalog::CatalogEntry) -> String {
    sha256_hex(
        catalog::jcs(&json!({"catalog_id":entry.id,"manifest_digest":entry.manifest_digest()}))
            .expect("integer manifest")
            .as_bytes(),
    )
}
fn catalog_entry_for_management<'a>(
    runtime: &'a RuntimeState,
    id: &str,
) -> Result<&'a catalog::CatalogEntry, WireOperationError> {
    if let Some((entry, Some(_))) = runtime.release_catalog.resolve_reserved(id) {
        return Err(catalog_wire_error(
            "invalid_request",
            json!({"model_id":id,"catalog_id":entry.id}),
            "use a catalog id, not a derived lane id",
        ));
    }
    runtime
        .release_catalog
        .entry(id)
        .ok_or_else(|| catalog_unknown(id))
}
fn catalog_lane_lock(runtime: &RuntimeState, id: &str) -> Arc<tokio::sync::Mutex<()>> {
    runtime
        .catalog_locks
        .lock()
        .expect("catalog lock map")
        .entry(id.into())
        .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
        .clone()
}

// A catalog lane uses three locks, always taken in this order:
// 1. `catalog_lane_lock`: lane lifecycle (verify and load, numerical self-check,
//    unload, failed-load acknowledgement). Serving inference never takes it, so
//    request resolution waits on it only for real lifecycle work.
// 2. `lane_serving_gate`: in-flight lifetime. Serving holds a read guard until
//    the engine reply has drained (a cancelled caller's detached direct-ANE task
//    keeps it). The self-check holds the write guard while it runs, and unload
//    needs `try_write`, so neither overlaps an inference.
// 3. `lane_execution_lock`: same-lane serialisation for engines that run one
//    call at a time, taken before the execution permit so a queued request
//    does not hold a permit. Direct-ANE lanes skip it: their worker stream
//    serialises calls, and they take the permit after that stream.
// No path takes the lane lock while it holds a serving-gate guard.
fn lane_serving_gate(runtime: &RuntimeState, id: &str) -> Arc<tokio::sync::RwLock<()>> {
    runtime
        .serving_gates
        .lock()
        .expect("serving gate map")
        .entry(id.into())
        .or_insert_with(|| Arc::new(tokio::sync::RwLock::new(())))
        .clone()
}

fn lane_execution_lock(runtime: &RuntimeState, id: &str) -> Arc<tokio::sync::Mutex<()>> {
    runtime
        .lane_execution_locks
        .lock()
        .expect("lane execution lock map")
        .entry(id.into())
        .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
        .clone()
}

/// The exclusive serving-gate guard a self-check holds. The check runs its
/// reference inputs through the normal execution path, which reuses this guard
/// instead of waiting for a shared one that the check itself blocks.
type HeldLaneGate = Arc<tokio::sync::OwnedRwLockWriteGuard<()>>;

/// What one catalog-lane execution holds on its lane's serving gate.
#[allow(dead_code)] // Held only for its drop; never read.
enum LaneServingGuard {
    Shared(tokio::sync::OwnedRwLockReadGuard<()>),
    Exclusive(HeldLaneGate),
}

async fn acquire_lane_serving(
    runtime: &RuntimeState,
    lane: &str,
    deadline: Option<tokio::time::Instant>,
) -> Result<LaneServingGuard, WireOperationError> {
    let gate = lane_serving_gate(runtime, lane);
    let guard = match deadline {
        Some(deadline) => tokio::time::timeout_at(deadline, gate.read_owned())
            .await
            .map_err(|_| {
                // Only a self-check or an unload holds the write guard. Inline
                // requests meet a self-check earlier, in certification, and are
                // answered model_loading there; admitted jobs must never fail
                // with model_loading, so this wait reports the budget instead.
                WireOperationError::from_stable(
                    StableError::deadline_exceeded(),
                    format!("deadline exceeded waiting for catalog lane '{lane}'"),
                )
            })?,
        None => gate.read_owned().await,
    };
    Ok(LaneServingGuard::Shared(guard))
}

async fn acquire_lane_execution(
    runtime: &RuntimeState,
    lane: &str,
    deadline: Option<tokio::time::Instant>,
) -> Result<tokio::sync::OwnedMutexGuard<()>, WireOperationError> {
    let lock = lane_execution_lock(runtime, lane);
    match deadline {
        Some(deadline) => tokio::time::timeout_at(deadline, lock.lock_owned())
            .await
            .map_err(|_| {
                WireOperationError::from_stable(
                    StableError::deadline_exceeded(),
                    format!("deadline exceeded waiting for catalog lane '{lane}'"),
                )
            }),
        None => Ok(lock.lock_owned().await),
    }
}
fn download_publish_lock(runtime: &RuntimeState, id: &str) -> Arc<Mutex<()>> {
    runtime
        .download_locks
        .lock()
        .expect("download lock map")
        .entry(id.into())
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}
fn catalog_directory_bytes(root: &Path) -> std::io::Result<u64> {
    let mut pending = vec![root.to_path_buf()];
    let mut bytes = 0u64;
    while let Some(path) = pending.pop() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let kind = entry.file_type()?;
            if kind.is_dir() {
                pending.push(entry.path());
            } else if kind.is_file() {
                bytes = bytes.saturating_add(entry.metadata()?.len());
            }
        }
    }
    Ok(bytes)
}

fn remove_owned_package_directory(path: &Path) -> std::io::Result<u64> {
    let name = path.file_name().expect("package name").to_string_lossy();
    let orphan = path.with_file_name(format!(
        ".{name}.removing-{}-{}",
        std::process::id(),
        now_ms()
    ));
    fs::rename(path, &orphan)?;
    let bytes = catalog_directory_bytes(&orphan)?;
    fs::remove_dir_all(orphan)?;
    Ok(bytes)
}

fn sweep_owned_package_orphans(cache: &Path) -> std::io::Result<u64> {
    let mut freed = 0u64;
    for directory in ["owned-metal-models", "owned-metal-packages"] {
        let entries = match fs::read_dir(cache.join(directory)) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error),
        };
        for entry in entries {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if entry.file_type()?.is_dir() && name.starts_with('.') && name.contains(".removing-") {
                let bytes = catalog_directory_bytes(&entry.path())?;
                fs::remove_dir_all(entry.path())?;
                freed = freed.saturating_add(bytes);
            }
        }
    }
    Ok(freed)
}

fn reclaim_unrooted_owned_packages(
    tx: &rusqlite::Transaction<'_>,
    cache: &ModelCache,
) -> rusqlite::Result<u64> {
    use store::CatalogBlobCache;
    let io = |error: std::io::Error| rusqlite::Error::ToSqlConversionFailure(Box::new(error));
    let entries = match fs::read_dir(cache.root().join("owned-metal-models")) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(io(error)),
    };
    let mut freed = 0u64;
    for entry in entries {
        let entry = entry.map_err(io)?;
        if !entry.file_type().map_err(io)?.is_dir()
            || entry.file_name().to_string_lossy().starts_with('.')
        {
            continue;
        }
        let descriptor = match fs::read(entry.path().join("role-digests.json")) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(io(error)),
        };
        let Ok(digests) = serde_json::from_slice::<Vec<String>>(&descriptor) else {
            continue;
        };
        if digests.is_empty()
            || digests.iter().any(|digest| {
                digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit())
            })
        {
            continue;
        }
        let mut rooted = false;
        for digest in &digests {
            if catalog_blob_referenced(tx, digest)?
                || CatalogCache(cache).is_pinned(digest).map_err(io)?
            {
                rooted = true;
                break;
            }
        }
        if rooted {
            continue;
        }
        // Compiled keys hash the assembled model's canonical path, not its blob
        // path. Reclaim them while that path still exists, then hide the package
        // before deleting it so interrupted removal cannot be mistaken for a cache hit.
        let compiled = synapse_engine_owned::remove_compiled_packages(
            &cache.root().join("owned-metal-packages"),
            &entry.path(),
        )
        .saturating_add(synapse_engine_owned::remove_compiled_packages(
            &cache.root().join("owned-metal-packages"),
            &entry.path().join("model.safetensors"),
        ));
        freed = freed
            .saturating_add(compiled)
            .saturating_add(remove_owned_package_directory(&entry.path()).map_err(io)?);
    }
    Ok(freed)
}

struct CatalogCache<'a>(&'a ModelCache);
impl store::CatalogBlobCache for CatalogCache<'_> {
    fn blob_size(&self, digest: &str) -> std::io::Result<Option<u64>> {
        match fs::metadata(self.0.blob_path(digest)) {
            Ok(m) => Ok(Some(m.len())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }
    fn is_pinned(&self, digest: &str) -> std::io::Result<bool> {
        match self.0.read_meta(digest) {
            Ok(m) => Ok(!m.pins.is_empty()),
            Err(ModelCacheError::NotFound(_)) => Ok(false),
            Err(e) => Err(std::io::Error::other(e.to_string())),
        }
    }
    fn remove_pins(&self, digest: &str) -> std::io::Result<()> {
        match self.0.read_meta(digest) {
            Ok(mut m) => {
                m.pins.clear();
                fs::write(self.0.meta_path(digest), serde_json::to_vec(&m)?)
            }
            Err(ModelCacheError::NotFound(_)) => Ok(()),
            Err(e) => Err(std::io::Error::other(e.to_string())),
        }
    }
    fn delete_blob(&self, digest: &str) -> std::io::Result<u64> {
        let path = self.0.blob_path(digest);
        let bytes = self.blob_size(digest)?.unwrap_or(0);
        let packages = synapse_engine_owned::remove_compiled_packages(self.0.root(), &path);
        remove_catalog_file(&path)?;
        remove_catalog_file(&self.0.meta_path(digest))?;
        Ok(bytes.saturating_add(packages))
    }
    fn delete_staging(&self, id: &str) -> std::io::Result<()> {
        let path = catalog_staging(self.0, id);
        match fs::remove_dir_all(path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e),
        }
    }
}
fn remove_catalog_file(path: &Path) -> std::io::Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}
fn catalog_staging(cache: &ModelCache, id: &str) -> PathBuf {
    cache.root().join("catalog-staging").join(id)
}
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelsCatalogParams {
    query: Option<String>,
    task: Option<String>,
    runnable_here: Option<bool>,
    installed: Option<bool>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelsDownloadParams {
    catalog_id: String,
    request_key: Option<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelsRemoveParams {
    catalog_id: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelsCancelParams {
    job_id: String,
}
fn current_catalog_install(
    state: &ModuleState,
    entry: &catalog::CatalogEntry,
    backend: &str,
) -> Result<Option<store::CatalogInstallRecord>, WireOperationError> {
    Ok(state
        .store
        .catalog_installs(&entry.id)
        .map_err(catalog_store_error)?
        .into_iter()
        .find(|r| r.manifest_digest == entry.manifest_digest() && r.backend == backend))
}
fn catalog_entry_complete(
    state: &ModuleState,
    entry: &catalog::CatalogEntry,
) -> Result<bool, WireOperationError> {
    use store::CatalogBlobCache;
    for backend in entry.backends.iter().filter(|b| {
        catalog_lane_backend_reason(&state.runtime, entry, b, synapse_core::Platform::current())
            .is_none()
    }) {
        if current_catalog_install(state, entry, &backend.backend)?.is_none() {
            return Ok(false);
        }
        for (_, file) in entry.backend_named_files(&backend.backend) {
            if CatalogCache(&state.model_cache)
                .blob_size(&file.sha256)
                .map_err(catalog_store_error)?
                != Some(file.size_bytes)
            {
                return Ok(false);
            }
        }
    }
    Ok(true)
}
async fn models_catalog(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let params: ModelsCatalogParams = match serde_json::from_value(params) {
        Ok(p) => p,
        Err(e) => return channel_error("invalid_request", e.to_string()),
    };
    let result = (|| {
        let mut rows = Vec::new();
        for entry in &state.runtime.release_catalog.models {
            let query = params.query.as_deref().unwrap_or("").trim().to_lowercase();
            if ![&entry.id, &entry.name, &entry.description]
                .iter()
                .any(|s| s.to_lowercase().contains(&query))
                || params.task.as_deref().is_some_and(|t| t != entry.task)
            {
                continue;
            }
            let runnable = entry
                .backends
                .iter()
                .filter(|b| {
                    catalog_lane_backend_reason(
                        &state.runtime,
                        entry,
                        b,
                        synapse_core::Platform::current(),
                    )
                    .is_none()
                })
                .map(|b| b.backend.as_str())
                .collect::<Vec<_>>();
            if params
                .runnable_here
                .is_some_and(|r| r != !runnable.is_empty())
            {
                continue;
            }
            let installs = state
                .store
                .catalog_installs(&entry.id)
                .map_err(catalog_store_error)?;
            let manifest = entry.manifest_digest();
            let installed = !runnable.is_empty()
                && runnable.iter().all(|b| {
                    installs
                        .iter()
                        .any(|i| i.manifest_digest == manifest && i.backend == *b)
                });
            if params.installed.is_some_and(|wanted| wanted != installed) {
                continue;
            }
            let install_state = if installed {
                "installed"
            } else if installs.iter().any(|i| i.manifest_digest != manifest) {
                "stale"
            } else {
                "not_installed"
            };
            let mut backends = Vec::new();
            for b in &entry.backends {
                let installed = installs
                    .iter()
                    .any(|i| i.manifest_digest == manifest && i.backend == b.backend);
                let check = if installed {
                    catalog_self_check_projection(&state, entry, b)?
                } else {
                    Value::Null
                };
                let refusal = catalog_lane_backend_reason(
                    &state.runtime,
                    entry,
                    b,
                    synapse_core::Platform::current(),
                );
                let mut row = json!({"backend":b.backend,"lane_id":catalog::lane_id(&entry.id,&b.backend),"fingerprint":b.fingerprint,"runnable":refusal.is_none(),"reason":refusal.as_ref().map(|r| r.code),"installed":installed,"self_check":check});
                if let Some(refusal) = refusal.filter(|r| !r.details.is_empty()) {
                    row["reason_details"] = json!(refusal.details);
                }
                if let Some(duration) = b.effective_max_duration_ms() {
                    row["max_duration_ms"] = json!(duration);
                }
                backends.push(row);
            }
            rows.push(json!({"id":entry.id,"task":entry.task,"name":entry.name,"description":entry.description,"default_for_task":entry.default_for_task,"upstream":entry.upstream,"manifest_digest":manifest,"download_bytes":entry.download_bytes(&runnable),"install_state":install_state,"backends":backends}));
        }
        rows.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
        Ok::<_, WireOperationError>(
            json!({"catalog_revision":state.runtime.release_catalog.catalog_revision,"models":rows}),
        )
    })();
    match result {
        Ok(v) => result_outcome(v),
        Err(e) => result_outcome(error_payload(&state, e)),
    }
}
fn download_status_payload(state: &ModuleState, record: &JobRecord, kind: bool) -> Value {
    let mut value = json!({"job_id":record.job_id,"state":record.state});
    if kind {
        value["kind"] = json!(store::DOWNLOAD_JOB_KIND);
    }
    if record.state == "downloading" {
        let (done, total) = state
            .runtime
            .download_bytes
            .lock()
            .expect("download counters")
            .get(&record.job_id)
            .copied()
            .unwrap_or_default();
        value["bytes_done"] = json!(done);
        value["bytes_total"] = json!(total);
    }
    if record.state == "failed" {
        value["error"] = record.error_json.clone().unwrap_or(Value::Null);
    }
    value
}
async fn models_download(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let params: ModelsDownloadParams = match serde_json::from_value(params) {
        Ok(p) => p,
        Err(e) => return channel_error("invalid_request", e.to_string()),
    };
    let admitted = (|| {
        let _disk = state
            .runtime
            .catalog_disk
            .lock()
            .expect("catalog disk lock");
        let entry = catalog_entry_for_management(&state.runtime, &params.catalog_id)?;
        if !entry.backends.iter().any(|b| {
            catalog_lane_backend_reason(&state.runtime, entry, b, synapse_core::Platform::current())
                .is_none()
        }) {
            return Err(catalog_backend_unavailable(&state, entry, None));
        }
        if entry.task == "transcribe" {
            for backend in entry.backends.iter().filter(|b| {
                catalog_lane_backend_reason(
                    &state.runtime,
                    entry,
                    b,
                    synapse_core::Platform::current(),
                )
                .is_none()
            }) {
                let profile =
                    CatalogProfile::load(backend.profile.as_deref().expect("speech profile"))
                        .map_err(|error| artifact_invalid_error(error.to_string()))?;
                let config = profile
                    .transcribe_config()
                    .map_err(|error| artifact_invalid_error(error.to_string()))?;
                validate_catalog_lane_fingerprint(
                    entry,
                    backend,
                    &profile.transcribe_numeric_profile(&config).fingerprint(),
                )?;
            }
        }
        let request_key = params
            .request_key
            .as_deref()
            .map(str::trim)
            .filter(|k| !k.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| format!("models.download:{}:{}", entry.id, entry.manifest_digest()));
        state.store.admit_download_job(&store::DownloadJobRequest { request_key:&request_key, request_digest:&catalog_request_digest(entry), module_generation:state.module_generation, params_json:&json!({"catalog_id":entry.id,"manifest_digest":entry.manifest_digest()}), now_ms:now_ms(), result_retention_ttl_ms:state.runtime.jobs.result_retention_ttl_ms, entry_complete:catalog_entry_complete(&state,entry)? }).map_err(|e| match e { SynapseStoreError::IdempotencyConflict { .. } => WireOperationError::from_stable(StableError::idempotency_conflict(),e.to_string()), _ => catalog_store_error(e) })
    })();
    match admitted {
        Ok(admission) => {
            let response = download_status_payload(&state, admission.record(), false);
            if let store::DownloadAdmission::Created(record) = admission {
                state.runtime.admission_telemetry.record_job_minted();
                let task_state = state.clone();
                tokio::spawn(async move {
                    execute_catalog_download(task_state, record).await;
                });
            }
            result_outcome(response)
        }
        Err(e) => result_outcome(error_payload(&state, e)),
    }
}
async fn models_download_cancel(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let params: ModelsCancelParams = match serde_json::from_value(params) {
        Ok(p) => p,
        Err(e) => return channel_error("invalid_request", e.to_string()),
    };
    let lock = download_publish_lock(&state.runtime, &params.job_id);
    let _publish = lock.lock().expect("publish lock");
    let _disk = state
        .runtime
        .catalog_disk
        .lock()
        .expect("catalog disk lock");
    match state.store.end_download_job(
        &params.job_id,
        store::DownloadEnd::Cancelled,
        &CatalogCache(&state.model_cache),
        now_ms(),
    ) {
        Ok(store::DownloadTransition::Applied { record, .. }) => {
            state.runtime.admission_telemetry.record_job_failed();
            result_outcome(download_status_payload(&state, &record, false))
        }
        Ok(store::DownloadTransition::Unchanged(Some(record))) => {
            result_outcome(download_status_payload(&state, &record, false))
        }
        Ok(_) => channel_error("invalid_request", "unknown or expired job_id"),
        Err(e) => channel_error("store_failure", e.to_string()),
    }
}

fn catalog_backend_unavailable(
    state: &ModuleState,
    entry: &catalog::CatalogEntry,
    selected: Option<&catalog::CatalogBackend>,
) -> WireOperationError {
    let backends = entry
        .backends
        .iter()
        .filter(|b| selected.is_none_or(|s| s.backend == b.backend))
        .map(|b| {
            let refusal = catalog_lane_backend_reason(
                &state.runtime,
                entry,
                b,
                synapse_core::Platform::current(),
            );
            let mut row = json!({"backend":b.backend,"reason":refusal.as_ref().map(|r| r.code)});
            if let Some(refusal) = refusal.filter(|r| !r.details.is_empty()) {
                row["reason_details"] = json!(refusal.details);
            }
            row
        })
        .collect::<Vec<_>>();
    catalog_wire_error(
        "backend_unavailable",
        json!({"catalog_id":entry.id,"lane_id":selected.map(|b| catalog::lane_id(&entry.id,&b.backend)),"backends":backends}),
        "no selected catalog backend is runnable here",
    )
}
fn catalog_artifact_error(
    file: &catalog::CatalogFile,
    actual_digest: Option<String>,
    actual_size: Option<u64>,
) -> WireOperationError {
    catalog_wire_error(
        "artifact_invalid",
        json!({"file":file.path,"expected_sha256":file.sha256,"actual_sha256":actual_digest,"expected_size":file.size_bytes,"actual_size":actual_size,"recovery_op":"models.download"}),
        "catalog artifact does not match the pinned manifest",
    )
}
fn download_error(
    file: &str,
    reason: &str,
    status: Option<u16>,
    message: impl Into<String>,
) -> WireOperationError {
    catalog_wire_error(
        "download_failed",
        json!({"file":file,"reason":reason,"http_status":status}),
        message,
    )
}
/// Pause a test download without holding its publish or catalog_disk mutex.
/// Cancellation takes both mutexes, so holding either here prevents a test from
/// cancelling the paused job before releasing the barrier.
fn catalog_test_barrier(kind: &str, job: &str) {
    #[cfg(feature = "test-support")]
    if let Ok(root) = env::var(format!(
        "SYNAPSE_TEST_DOWNLOAD_{}_BARRIER",
        kind.to_ascii_uppercase().replace('-', "_")
    )) {
        let root = PathBuf::from(root);
        let _ = fs::create_dir_all(&root);
        let _ = fs::write(root.join(format!("{job}.ready")), b"ready");
        while !root.join(format!("{job}.release")).exists() {
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    let _ = (kind, job);
}
fn catalog_validation_timing(lane: &str, stage: &str, started: std::time::Instant) {
    #[cfg(feature = "test-support")]
    if let Ok(path) = env::var("SYNAPSE_TEST_CATALOG_VALIDATION_TIMINGS") {
        if let Ok(mut file) = fs::OpenOptions::new().create(true).append(true).open(path) {
            let _ = writeln!(
                file,
                "{}",
                json!({"lane":lane,"stage":stage,"elapsed_ms":started.elapsed().as_millis()})
            );
        }
    }
    let _ = (lane, stage, started);
}

fn catalog_call_fault(id: &str, call: &str) -> Result<(), WireOperationError> {
    #[cfg(feature = "test-support")]
    if env::var("SYNAPSE_TEST_CATALOG_FAULT_LANE").ok().as_deref() == Some(id) {
        if call == "load" {
            if let Ok(path) = env::var("SYNAPSE_TEST_CATALOG_LOAD_ATTEMPTS") {
                fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .expect("test catalog load counter")
                    .write_all(format!("{id}\n").as_bytes())
                    .expect("test catalog load counter write");
            }
        }
        match env::var("SYNAPSE_TEST_CATALOG_FAULT").ok().as_deref() {
            Some(value) if value == format!("{call}:stall-crash") => {
                std::thread::sleep(Duration::from_secs(6));
                return Err(WireOperationError::from_stable(
                    StableError::engine_crashed(Some(250)),
                    "injected catalog engine crash",
                ));
            }
            Some(value) if value == format!("{call}:stall") => {
                std::thread::sleep(Duration::from_secs(10))
            }
            Some(value) if value == format!("{call}:crash") => {
                return Err(WireOperationError::from_stable(
                    StableError::engine_crashed(Some(250)),
                    "injected catalog engine crash",
                ))
            }
            _ => {}
        }
    }
    let _ = (id, call);
    Ok(())
}
async fn execute_catalog_download(state: Arc<ModuleState>, record: JobRecord) {
    let result = fetch_catalog_download(&state, &record).await;
    if let Err(error) = result {
        if let Err(error) = catalog_download_blocking(move || {
            let lock = download_publish_lock(&state.runtime, &record.job_id);
            let _publish = lock.lock().expect("publish lock");
            let _disk = state
                .runtime
                .catalog_disk
                .lock()
                .expect("catalog disk lock");
            let value = serde_json::to_value(error).expect("wire error");
            match state.store.end_download_job(
                &record.job_id,
                store::DownloadEnd::Failed { error_json: &value },
                &CatalogCache(&state.model_cache),
                now_ms(),
            ) {
                Ok(store::DownloadTransition::Applied { .. }) => {
                    state.runtime.admission_telemetry.record_job_failed()
                }
                Ok(_) => {}
                Err(error) => tracing::warn!(%error, "download failure transaction failed"),
            }
            Ok(())
        })
        .await
        {
            tracing::warn!(error = %error.message, "download failure cleanup task failed");
        }
    }
}

// Download disk transactions and hashing must not occupy a Tokio runtime worker:
// management frames need to remain responsive while a large artifact is written or published.
async fn catalog_download_blocking<T: Send + 'static>(
    work: impl FnOnce() -> Result<T, WireOperationError> + Send + 'static,
) -> Result<T, WireOperationError> {
    tokio::task::spawn_blocking(work).await.map_err(|error| {
        WireOperationError::from_stable(
            StableError::engine_crashed(Some(250)),
            format!("catalog download blocking task failed: {error}"),
        )
    })?
}
async fn fetch_catalog_download(
    state: &Arc<ModuleState>,
    record: &JobRecord,
) -> Result<(), WireOperationError> {
    use store::CatalogBlobCache;
    let plan_state = state.clone();
    let plan_record = record.clone();
    let (entry, manifest, targets, files, staging) = catalog_download_blocking(move || {
        let entry = plan_state
            .runtime
            .release_catalog
            .entry(
                plan_record
                    .params_json
                    .as_ref()
                    .and_then(|p| p["catalog_id"].as_str())
                    .unwrap_or(""),
            )
            .ok_or_else(|| catalog_unknown("download entry"))?
            .clone();
        let manifest = entry.manifest_digest();
        let targets = entry
            .backends
            .iter()
            .filter(|b| {
                catalog_lane_backend_reason(
                    &plan_state.runtime,
                    &entry,
                    b,
                    synapse_core::Platform::current(),
                )
                .is_none()
            })
            .map(|b| b.backend.clone())
            .collect::<Vec<_>>();
        let files = entry
            .files
            .iter()
            .filter(|f| f.backends.iter().any(|b| targets.contains(b)))
            .cloned()
            .collect::<Vec<_>>();
        let mut planned = BTreeSet::new();
        let total = files
            .iter()
            .filter(|f| {
                planned.insert(f.sha256.clone())
                    && CatalogCache(&plan_state.model_cache)
                        .blob_size(&f.sha256)
                        .ok()
                        .flatten()
                        != Some(f.size_bytes)
            })
            .map(|f| f.size_bytes)
            .sum::<u64>();
        plan_state
            .runtime
            .download_bytes
            .lock()
            .expect("download counters")
            .insert(plan_record.job_id.clone(), (0, total));
        let staging = catalog_staging(&plan_state.model_cache, &plan_record.job_id);
        fs::create_dir_all(&staging)
            .map_err(|e| download_error("", "storage_full", None, e.to_string()))?;
        Ok((entry, manifest, targets, files, staging))
    })
    .await?;
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| download_error("", "network", None, e.to_string()))?;
    for file in &files {
        let reuse_state = state.clone();
        let reuse_record = record.clone();
        let reuse_file = file.clone();
        let (reuse, active) = catalog_download_blocking(move || {
            let lock = download_publish_lock(&reuse_state.runtime, &reuse_record.job_id);
            let reuse = {
                let _publish = lock.lock().expect("publish lock");
                let _disk = reuse_state
                    .runtime
                    .catalog_disk
                    .lock()
                    .expect("catalog disk lock");
                let reuse = reuse_state
                    .store
                    .root_reused_download_blob(
                        &reuse_record.job_id,
                        &reuse_file.sha256,
                        reuse_file.size_bytes,
                        &CatalogCache(&reuse_state.model_cache),
                    )
                    .map_err(catalog_store_error)?;
                if matches!(reuse, store::DownloadReuse::Absent) {
                    if let Some(size) = CatalogCache(&reuse_state.model_cache)
                        .blob_size(&reuse_file.sha256)
                        .map_err(catalog_store_error)?
                    {
                        let error = catalog_artifact_error(&reuse_file, None, Some(size));
                        reuse_state
                            .store
                            .quarantine_catalog_blob(
                                &reuse_file.sha256,
                                &serde_json::to_value(error).expect("wire error"),
                                &CatalogCache(&reuse_state.model_cache),
                                now_ms(),
                            )
                            .map_err(catalog_store_error)?;
                    }
                }
                reuse
            };
            if matches!(reuse, store::DownloadReuse::Rooted) {
                catalog_test_barrier("pre-publish", &reuse_record.job_id);
            }
            if matches!(reuse, store::DownloadReuse::Absent)
                && !reuse_state
                    .store
                    .advance_download_job(&reuse_record.job_id, "downloading", now_ms())
                    .map_err(catalog_store_error)?
            {
                return Ok((reuse, false));
            }
            Ok((reuse, true))
        })
        .await?;
        if !active {
            return Ok(());
        }
        match reuse {
            store::DownloadReuse::Stopped(_) => return Ok(()),
            store::DownloadReuse::Rooted => continue,
            store::DownloadReuse::Absent => {}
        }
        let source = huggingface_resolve_url(
            &state.runtime.hf_endpoint,
            &entry.upstream.hf_repo,
            &entry.upstream.revision,
            &file.path,
        )
        .map_err(artifact_invalid_error)?;
        validate_resolved_asset(&source, Some(&file.sha256), &state.runtime.hf_endpoint)
            .map_err(artifact_invalid_error)?;
        let mut response =
            tokio::time::timeout(Duration::from_secs(60), client.get(&source).send())
                .await
                .map_err(|_| download_error(&file.path, "network", None, "download idle timeout"))?
                .map_err(|e| download_error(&file.path, "network", None, e.to_string()))?;
        if !response.status().is_success() {
            return Err(download_error(
                &file.path,
                "http_status",
                Some(response.status().as_u16()),
                "download HTTP refusal",
            ));
        }
        let path = staging.join(&file.sha256);
        let output_path = path.clone();
        let output_file = file.clone();
        let mut output = catalog_download_blocking(move || {
            fs::File::create(output_path)
                .map_err(|e| download_error(&output_file.path, "storage_full", None, e.to_string()))
        })
        .await?;
        let mut hasher = Sha256::new();
        let mut size = 0u64;
        loop {
            let body = tokio::time::timeout(Duration::from_secs(60), response.chunk())
                .await
                .map_err(|_| download_error(&file.path, "network", None, "download idle timeout"))?
                .map_err(|e| download_error(&file.path, "network", None, e.to_string()))?;
            let Some(body) = body else {
                break;
            };
            let write_state = state.clone();
            let write_record = record.clone();
            let write_file = file.clone();
            let written = catalog_download_blocking(move || {
                if !write_state
                    .store
                    .get_job(&write_record.job_id)
                    .map_err(catalog_store_error)?
                    .is_some_and(|j| store::is_download_non_terminal(&j.state))
                {
                    return Ok(None);
                }
                output.write_all(&body).map_err(|e| {
                    download_error(&write_file.path, "storage_full", None, e.to_string())
                })?;
                hasher.update(&body);
                size = size.saturating_add(body.len() as u64);
                if let Some(counts) = write_state
                    .runtime
                    .download_bytes
                    .lock()
                    .expect("download counters")
                    .get_mut(&write_record.job_id)
                {
                    counts.0 = counts.0.saturating_add(body.len() as u64);
                }
                Ok(Some((output, hasher, size)))
            })
            .await?;
            let Some(written) = written else {
                return Ok(());
            };
            (output, hasher, size) = written;
        }
        let publish_state = state.clone();
        let publish_record = record.clone();
        let publish_file = file.clone();
        let published = catalog_download_blocking(move || {
            output.sync_all().map_err(|e| {
                download_error(&publish_file.path, "storage_full", None, e.to_string())
            })?;
            drop(output);
            if !publish_state
                .store
                .advance_download_job(&publish_record.job_id, "verifying", now_ms())
                .map_err(catalog_store_error)?
            {
                return Ok(false);
            }
            let digest = hex::encode(hasher.finalize());
            if digest != publish_file.sha256 || size != publish_file.size_bytes {
                return Err(catalog_artifact_error(
                    &publish_file,
                    Some(digest),
                    Some(size),
                ));
            }
            catalog_test_barrier("pre-publish", &publish_record.job_id);
            let lock = download_publish_lock(&publish_state.runtime, &publish_record.job_id);
            let _publish = lock.lock().expect("publish lock");
            let _disk = publish_state
                .runtime
                .catalog_disk
                .lock()
                .expect("catalog disk lock");
            match publish_state
                .store
                .record_download_publication(
                    &publish_record.job_id,
                    &publish_file.sha256,
                    &CatalogCache(&publish_state.model_cache),
                )
                .map_err(catalog_store_error)?
            {
                store::DownloadPublication::Stopped(_) => return Ok(false),
                store::DownloadPublication::Proceed { .. } => {
                    publish_state
                        .model_cache
                        .ingest(ModelCacheIngest {
                            source_url: local_file_url(&path),
                            expected_digest: Some(publish_file.sha256.clone()),
                            format: if publish_file.role == "model" {
                                "safetensors"
                            } else {
                                "json"
                            }
                            .into(),
                            tokenizer_path: None,
                            pin_module_id: None,
                        })
                        .map_err(model_cache_load_error)?;
                }
            }
            Ok(true)
        })
        .await?;
        if !published {
            return Ok(());
        }
    }
    let commit_state = state.clone();
    let commit_record = record.clone();
    catalog_download_blocking(move || {
        catalog_test_barrier("pre-commit", &commit_record.job_id);
        let lock = download_publish_lock(&commit_state.runtime, &commit_record.job_id);
        let _publish = lock.lock().expect("publish lock");
        let _disk = commit_state
            .runtime
            .catalog_disk
            .lock()
            .expect("catalog disk lock");
        let installs = targets
            .iter()
            .map(|b| store::CatalogInstallRecord {
                catalog_id: entry.id.clone(),
                manifest_digest: manifest.clone(),
                backend: b.clone(),
                members: entry
                    .backend_named_files(b)
                    .map(|(_, f)| store::CatalogInstallMember {
                        path: f.path.clone(),
                        digest: f.sha256.clone(),
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        if let store::DownloadTransition::Applied { .. } = commit_state
            .store
            .commit_download_job(&commit_record.job_id, &installs, now_ms())
            .map_err(catalog_store_error)?
        {
            commit_state
                .runtime
                .admission_telemetry
                .record_job_completed();
            CatalogCache(&commit_state.model_cache)
                .delete_staging(&commit_record.job_id)
                .map_err(catalog_store_error)?;
            sync_installed_catalog_slots(&commit_state)?;
        }
        Ok(())
    })
    .await
}

fn catalog_blob_referenced(tx: &rusqlite::Transaction<'_>, digest: &str) -> rusqlite::Result<bool> {
    let roots: i64 = tx.query_row("SELECT (SELECT COUNT(*) FROM catalog_install_members WHERE digest=?1) + (SELECT COUNT(*) FROM download_acquisitions WHERE digest=?1)", [digest], |r| r.get(0))?;
    if roots > 0 {
        return Ok(true);
    }
    let mut query = tx.prepare("SELECT config_json FROM models")?;
    let configs = query.query_map([], |r| r.get::<_, Vec<u8>>(0))?;
    for config in configs {
        let value: Value = serde_json::from_slice(&config?).unwrap_or(Value::Null);
        fn contains(value: &Value, digest: &str) -> bool {
            match value {
                Value::String(s) => s.strip_prefix("sha256:").unwrap_or(s) == digest,
                Value::Array(a) => a.iter().any(|v| contains(v, digest)),
                Value::Object(o) => o.values().any(|v| contains(v, digest)),
                _ => false,
            }
        }
        if contains(&value, digest) {
            return Ok(true);
        }
    }
    Ok(false)
}
fn catalog_holders(
    state: &ModuleState,
    entry: &catalog::CatalogEntry,
) -> Result<Vec<Value>, WireOperationError> {
    let mut holders = Vec::new();
    for backend in &entry.backends {
        let id = catalog::lane_id(&entry.id, &backend.backend);
        if let Some(slot) = model_slot_snapshot(&state.runtime, &id) {
            if slot.loaded.is_some() {
                holders.push(json!({"kind":"loaded","model_id":id}));
            }
            if matches!(
                slot.state,
                ModelRuntimeState::Loading | ModelRuntimeState::Validating
            ) {
                holders.push(json!({"kind":"request","lease_id":format!("catalog-load:{id}")}));
            }
        }
        if let Some(check_id) = state
            .runtime
            .self_check_holders
            .lock()
            .expect("self-check holders")
            .get(&id)
        {
            holders.push(json!({"kind":"self_check","check_id":check_id}));
        }
    }
    if let Some(job) = state
        .store
        .active_download_job(&catalog_request_digest(entry))
        .map_err(catalog_store_error)?
    {
        holders.push(json!({"kind":"job","job_id":job.job_id}));
    }
    for (job, id) in state
        .runtime
        .catalog_jobs
        .lock()
        .expect("catalog jobs")
        .iter()
    {
        if id == &entry.id {
            holders.push(json!({"kind":"job","job_id":job}));
        }
    }
    Ok(holders)
}
async fn models_remove(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    use store::CatalogBlobCache;
    let params: ModelsRemoveParams = match serde_json::from_value(params) {
        Ok(p) => p,
        Err(e) => return channel_error("invalid_request", e.to_string()),
    };
    let result = (|| {
        let _disk = state
            .runtime
            .catalog_disk
            .lock()
            .expect("catalog disk lock");
        let entry = catalog_entry_for_management(&state.runtime, &params.catalog_id)?;
        let mut holders = catalog_holders(&state, entry)?;
        holders.extend(media_jobs::holders(&state, entry)?);
        if !holders.is_empty() {
            return Err(catalog_wire_error(
                "model_in_use",
                json!({"catalog_id":entry.id,"holders":holders}),
                "catalog model is in use",
            ));
        }
        let installs = state
            .store
            .catalog_installs(&entry.id)
            .map_err(catalog_store_error)?;
        let manifests = installs
            .iter()
            .map(|i| i.manifest_digest.clone())
            .collect::<BTreeSet<_>>();
        invalidate_validated_readiness(&state.runtime);
        let freed = state
            .store
            .sqlite()
            .with_conn_fenced(|tx| {
                let mut q = tx.prepare(
                    "SELECT DISTINCT digest FROM catalog_install_members WHERE catalog_id=?1",
                )?;
                let digests = q
                    .query_map([&entry.id], |r| r.get::<_, String>(0))?
                    .collect::<Result<Vec<_>, _>>()?;
                drop(q);
                tx.execute(
                    "DELETE FROM catalog_install_members WHERE catalog_id=?1",
                    [&entry.id],
                )?;
                tx.execute(
                    "DELETE FROM catalog_installs WHERE catalog_id=?1",
                    [&entry.id],
                )?;
                tx.execute(
                    "DELETE FROM catalog_self_checks WHERE catalog_id=?1",
                    [&entry.id],
                )?;
                let mut freed = 0u64;
                for digest in digests {
                    if !catalog_blob_referenced(tx, &digest)?
                        && !CatalogCache(&state.model_cache)
                            .is_pinned(&digest)
                            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?
                    {
                        freed = freed.saturating_add(
                            CatalogCache(&state.model_cache)
                                .delete_blob(&digest)
                                .map_err(|e| {
                                    rusqlite::Error::ToSqlConversionFailure(Box::new(e))
                                })?,
                        );
                    }
                }
                freed =
                    freed.saturating_add(reclaim_unrooted_owned_packages(tx, &state.model_cache)?);
                Ok(freed)
            })
            .map_err(catalog_store_error)?;
        invalidate_validated_readiness(&state.runtime);
        let mut slots = state.runtime.catalog.lock().expect("runtime catalog");
        for b in &entry.backends {
            slots.remove(&catalog::lane_id(&entry.id, &b.backend));
        }
        Ok(json!({"catalog_id":entry.id,"removed_manifests":manifests,"freed_bytes":freed}))
    })();
    match result {
        Ok(v) => result_outcome(v),
        Err(e) => result_outcome(error_payload(&state, e)),
    }
}
fn catalog_lane_spec(
    state: &ModuleState,
    entry: &catalog::CatalogEntry,
    backend: &catalog::CatalogBackend,
    verify: bool,
) -> Result<StoredModelConfig, WireOperationError> {
    if let Some(profile) = backend.profile.as_deref() {
        return catalog_profile_lane_spec(state, entry, backend, profile, verify);
    }
    let files = entry.backend_files(&backend.backend);
    let tokenizer = files["tokenizer"];
    let max_tokens = backend.max_tokens.expect("validated max_tokens") as usize;
    let tokenizer_path = state.model_cache.blob_path(&tokenizer.sha256);
    let sanitized_digest = if verify {
        format!(
            "sha256:{}",
            SanitizedTokenizer::from_file(&tokenizer_path, TokenizerConfig { max_tokens })
                .map_err(|e| artifact_invalid_error(e.to_string()))?
                .sanitized_sha256()
        )
    } else {
        format!("sha256:{}", "0".repeat(64))
    };
    catalog_native_lane_spec(state, entry, backend, sanitized_digest)
}

fn catalog_native_lane_spec(
    state: &ModuleState,
    entry: &catalog::CatalogEntry,
    backend: &catalog::CatalogBackend,
    sanitized_digest: String,
) -> Result<StoredModelConfig, WireOperationError> {
    let files = entry.backend_files(&backend.backend);
    let model = files["model"];
    let tokenizer = files["tokenizer"];
    let max_tokens = backend.max_tokens.expect("validated max_tokens") as usize;
    let tokenizer_path = state.model_cache.blob_path(&tokenizer.sha256);
    let family = OwnedFamily::parse(backend.family.as_deref().expect("validated family"))
        .map_err(|e| artifact_invalid_error(e.to_string()))?;
    let dtype = OwnedDType::parse(backend.dtype.as_deref().expect("validated dtype"))
        .map_err(|e| artifact_invalid_error(e.to_string()))?;
    let mut identity =
        engine_identity_with_projections(family, dtype, catalog_projection_dtype(family, dtype));
    if entry.task == "rerank"
        && RerankReadout::select(true, &identity) == RerankReadout::CatalogLogits
    {
        // The native catalog lane now serves sigmoid scores instead of logits.
        // Bind that numeric change to its fingerprint without changing the raw
        // engine head or any legacy preload identity.
        identity
            .build_flags
            .insert("rerank_readout".into(), "sigmoid_classifier_logit".into());
    }
    let owned = OwnedCatalogConfig {
        family,
        dtype,
        projections: catalog_projection_dtype(family, dtype),
        execution: backend.execution.clone().expect("validated execution"),
        attention_units: backend.attention_units.expect("validated attention units") as usize,
        config_locator: files.get("config").map(|f| ModelAssetLocator::CacheDigest {
            digest: format!("sha256:{}", f.sha256),
        }),
        extra_locators: Vec::new(),
        identity_override: Some(identity),
    };
    let roles = files
        .iter()
        .map(|(role, file)| (role.to_string(), format!("sha256:{}", file.sha256)))
        .collect::<Vec<_>>();
    let digest = format!(
        "sha256:{}",
        sha256_hex(&serde_json::to_vec(&roles).expect("role digests"))
    );
    // Test catalogs can exercise backend selection without non-Metal hardware.
    // Each declared backend still has its own files and invocation mutex.
    build_stored_model_config(
        catalog::lane_id(&entry.id, &backend.backend),
        "owned-metal",
        parse_model_task(Some(&entry.task), "owned-metal", &entry.id)
            .map_err(|e| artifact_invalid_error(e.to_string()))?,
        digest,
        "safetensors-package".into(),
        sanitized_digest,
        ModelAssetLocator::CacheDigest {
            digest: format!("sha256:{}", model.sha256),
        },
        ModelAssetLocator::CacheDigest {
            digest: format!("sha256:{}", tokenizer.sha256),
        },
        local_file_url(&state.model_cache.blob_path(&model.sha256)),
        local_file_url(&tokenizer_path),
        parse_pooling(backend.pooling.as_deref().unwrap_or("cls"))
            .map_err(|e| artifact_invalid_error(e.to_string()))?,
        backend.normalize.unwrap_or(false),
        max_tokens,
        backend.dtype.clone().expect("validated dtype"),
        false,
        None,
        None,
        Vec::new(),
        Some(owned),
        &InlineConfig::default(),
        &JobConfig::default(),
    )
    .map_err(|e| artifact_invalid_error(e.to_string()))
}

fn catalog_profile_lane_spec(
    state: &ModuleState,
    entry: &catalog::CatalogEntry,
    backend: &catalog::CatalogBackend,
    profile_id: &str,
    verify: bool,
) -> Result<StoredModelConfig, WireOperationError> {
    let profile = CatalogProfile::load(profile_id)
        .map_err(|error| artifact_invalid_error(error.to_string()))?;
    let files = entry.backend_files(&backend.backend);
    let source = state.model_cache.blob_path(&files["model"].sha256);
    let tokenizer_path = state.model_cache.blob_path(&files["tokenizer"].sha256);
    let digest = profile.artifact_digest();
    let package_path = state.model_cache.blob_path(&digest);
    if verify
        && sha256_file(&package_path).ok().as_deref() != Some(digest.trim_start_matches("sha256:"))
    {
        // Downloads contain original Hugging Face model weights whose SHA-256
        // checksums are recorded in bench/parity/models.json. Convert them with
        // the same function used by hardware certification and verify the
        // converted package's checksum before caching it. The normal model cache
        // lets reloads and garbage collection manage those same bytes.
        let package =
            synapse_parity::convert::convert_profile_file(&profile.typed, profile_id, &source)
                .map_err(|error| artifact_invalid_error(error.to_string()))?;
        let temporary = state.model_cache.root().join(format!(
            "catalog-convert-{}-{}.safetensors",
            std::process::id(),
            now_ms()
        ));
        fs::write(&temporary, package)
            .map_err(|error| artifact_invalid_error(error.to_string()))?;
        let ingested = state.model_cache.ingest(synapse_core::ModelCacheIngest {
            source_url: local_file_url(&temporary),
            expected_digest: Some(digest.clone()),
            format: "safetensors".into(),
            tokenizer_path: None,
            pin_module_id: None,
        });
        let _ = fs::remove_file(&temporary);
        ingested.map_err(|error| artifact_invalid_error(error.to_string()))?;
    }
    if entry.task == "transcribe" {
        let config = profile
            .transcribe_config()
            .map_err(|error| artifact_invalid_error(error.to_string()))?;
        if verify
            && sha256_file(&tokenizer_path).ok().as_deref()
                != Some(files["tokenizer"].sha256.as_str())
        {
            return Err(artifact_invalid_error(
                "transcribe tokenizer digest mismatch",
            ));
        }
        let numeric = profile.transcribe_numeric_profile(&config);
        validate_catalog_lane_fingerprint(entry, backend, &numeric.fingerprint())?;
        return Ok(StoredModelConfig {
            model_id: catalog::lane_id(&entry.id, &backend.backend),
            engine: backend.engine.clone(),
            task: "transcribe".into(),
            artifact_digest: digest.clone(),
            artifact_format: default_artifact_format(&backend.engine),
            tokenizer_sanitized_digest: numeric.sanitized_tokenizer_digest.clone(),
            model_locator: ModelAssetLocator::CacheDigest { digest },
            tokenizer_locator: ModelAssetLocator::CacheDigest {
                digest: format!("sha256:{}", files["tokenizer"].sha256),
            },
            model_source_url: local_file_url(&package_path),
            tokenizer_source_url: local_file_url(&tokenizer_path),
            pooling: "none".into(),
            normalize: false,
            // Speech admission uses the profile's duration and frame limits, not
            // StoredModelConfig's required token limit. Zero is unused for transcribe.
            max_tokens: 0,
            quant: config.dtype.as_str().into(),
            pin: false,
            owned_family: Some(config.family.into()),
            owned_dtype: Some(config.dtype.as_str().into()),
            owned_execution: Some(config.execution),
            owned_attention_units: None,
            config_locator: files.get("config").map(|f| ModelAssetLocator::CacheDigest {
                digest: format!("sha256:{}", f.sha256),
            }),
            extra_locators: Vec::new(),
            engine_identity: config.identity,
            numeric_profile_id: numeric.numeric_profile_id(),
            fingerprint: numeric.fingerprint(),
            worker_bin: None,
            worker_runtime_dir: None,
        });
    }
    let sanitized_digest = if verify {
        // Count the tokens actually sent to the model, including special tokens
        // and any query/document template, before enforcing the 8192-token limit.
        // Truncating during tokenization could hide an oversized input instead
        // of producing the required refusal for an 8193-token sequence.
        let tokenizer = SanitizedTokenizer::from_file(
            &tokenizer_path,
            TokenizerConfig {
                max_tokens: usize::MAX,
            },
        )
        .map_err(|error| artifact_invalid_error(error.to_string()))?;
        profile
            .validate_readout(&tokenizer)
            .map_err(|error| artifact_invalid_error(error.to_string()))?;
        format!("sha256:{}", tokenizer.sanitized_sha256())
    } else {
        format!("sha256:{}", "0".repeat(64))
    };
    let pooling = match profile.model()["grammar"]["pooling"].as_str() {
        Some("cls") => WorkerPooling::Cls,
        Some("masked_mean") => WorkerPooling::Mean,
        _ => WorkerPooling::Last,
    };
    let owned = profile
        .owned_config(
            backend.execution.as_deref(),
            backend.attention_units.map(|units| units as usize),
        )
        .map_err(|error| artifact_invalid_error(error.to_string()))?;
    build_stored_model_config(
        catalog::lane_id(&entry.id, &backend.backend),
        &backend.engine,
        parse_model_task(Some(&entry.task), &backend.engine, &entry.id)
            .map_err(|error| artifact_invalid_error(error.to_string()))?,
        digest.clone(),
        // The worker receives the converted profile package, not the original
        // checkpoint, so it must be labelled the way every owned engine expects.
        default_artifact_format(&backend.engine),
        sanitized_digest,
        ModelAssetLocator::CacheDigest { digest },
        ModelAssetLocator::CacheDigest {
            digest: format!("sha256:{}", files["tokenizer"].sha256),
        },
        local_file_url(&package_path),
        local_file_url(&tokenizer_path),
        pooling,
        profile.model()["output"]["normalization"] == "l2",
        8192,
        owned.dtype.as_str().into(),
        false,
        None,
        None,
        Vec::new(),
        Some(owned),
        &InlineConfig::default(),
        &JobConfig::default(),
    )
    .map_err(|error| artifact_invalid_error(error.to_string()))
}
fn sync_installed_catalog_slots(state: &ModuleState) -> Result<(), WireOperationError> {
    for entry in &state.runtime.release_catalog.models {
        for backend in &entry.backends {
            if catalog_lane_backend_reason(
                &state.runtime,
                entry,
                backend,
                synapse_core::Platform::current(),
            )
            .is_some()
                || current_catalog_install(state, entry, &backend.backend)?.is_none()
            {
                continue;
            }
            let id = catalog::lane_id(&entry.id, &backend.backend);
            if model_slot_snapshot(&state.runtime, &id).is_some() {
                continue;
            }
            let mut spec = catalog_lane_spec(state, entry, backend, false)?;
            // Unloaded rows expose the fingerprint declared by the catalog backend.
            // Loading recomputes it from verified artifact and tokenizer bytes and
            // rejects a different value before invoking the engine.
            spec.fingerprint = Fingerprint(backend.fingerprint.clone());
            register_runtime_catalog_model(&state.runtime, spec)?;
        }
    }
    Ok(())
}
/// Installed catalog lanes that passed a self-check before, but have no result
/// under their current key. The key includes the OS build, so an OS update
/// leaves every lane in use here, and `models.list` reports it `certified:
/// false` until something loads it. A consumer that gates on that flag never
/// sends the request that would load it, so startup checks these lanes itself.
/// Lanes that never passed (fresh installs) are left to load on first use.
fn catalog_lanes_needing_revalidation(
    state: &ModuleState,
) -> Result<Vec<(catalog::CatalogEntry, catalog::CatalogBackend)>, WireOperationError> {
    let mut lanes = Vec::new();
    for entry in &state.runtime.release_catalog.models {
        for backend in &entry.backends {
            if catalog_lane_backend_reason(
                &state.runtime,
                entry,
                backend,
                synapse_core::Platform::current(),
            )
            .is_some()
                || current_catalog_install(state, entry, &backend.backend)?.is_none()
                || catalog_self_check_projection(state, entry, backend)?["state"] != "pending"
            {
                continue;
            }
            let passed_before: bool = state
                .store
                .sqlite()
                .with_conn(|conn| {
                    conn.query_row(
                        "SELECT EXISTS(SELECT 1 FROM catalog_self_checks \
                         WHERE catalog_id=?1 AND backend=?2 AND state='passed')",
                        [&entry.id, &backend.backend],
                        |row| row.get(0),
                    )
                })
                .map_err(catalog_store_error)?;
            if passed_before {
                lanes.push((entry.clone(), backend.clone()));
            }
        }
    }
    Ok(lanes)
}

/// Loads and self-checks, one at a time, every lane
/// `catalog_lanes_needing_revalidation` names. It goes through the same
/// single-flight path a request takes, so a request arriving meanwhile joins
/// the check instead of starting a second one, and lanes on Apple's Neural
/// Engine still go through the supervisor that caps how many compiled programs
/// the chip holds and runs one compile at a time.
fn start_catalog_revalidation(state: Arc<ModuleState>) {
    // Detached: it ends once each named lane is checked, and a module stop
    // drops it with the runtime.
    tokio::spawn(async move {
        let lanes = match catalog_lanes_needing_revalidation(&state) {
            Ok(lanes) => lanes,
            Err(error) => {
                tracing::warn!(target: "synapse.catalog", error = %error.message,
                    "catalog revalidation skipped: could not list lanes");
                diagnostics_findings::refresh(&state);
                return;
            }
        };
        for (entry, backend) in lanes {
            let lane = catalog::lane_id(&entry.id, &backend.backend);
            tracing::info!(target: "synapse.catalog", model_id = %lane,
                "catalog self-check scheduled: no result for the current key");
            match ensure_catalog_lane_ready(state.clone(), entry, backend, None).await {
                Ok(_) => tracing::info!(target: "synapse.catalog", model_id = %lane,
                    "catalog self-check passed at startup"),
                Err(error) => tracing::warn!(target: "synapse.catalog", model_id = %lane,
                    code = %error.code, error = %error.message,
                    "catalog self-check at startup did not pass"),
            }
        }
        diagnostics_findings::refresh(&state);
    });
}

pub(crate) fn catalog_self_check_id(
    catalog_id: &str,
    manifest_digest: &str,
    backend: &str,
    fingerprint: &str,
    engine_identity: &Value,
    os_build: &str,
    fixture_revision: &str,
) -> Result<String, String> {
    let key = json!({
        "catalog_id": catalog_id,
        "manifest_digest": manifest_digest,
        "backend": backend,
        "fingerprint": fingerprint,
        "engine_identity": engine_identity,
        "os_build": os_build,
        "fixture_revision": fixture_revision,
    });
    Ok(sha256_hex(catalog::jcs(&key)?.as_bytes()))
}

// Catalog/media integration supplies lane identity and a host-loading closure.
// Keep speech checking separate from embedding's token-count admission runtime.
#[allow(dead_code)]
mod transcribe_self_check {
    use super::*;
    use transcribe_host::{TranscribeHost, TranscribeHostError};

    #[derive(Clone)]
    pub(crate) struct Lane {
        pub catalog_id: String,
        pub backend: String,
        pub profile: String,
        pub fingerprint: String,
        pub manifest_digest: String,
        pub engine_identity: Value,
    }

    pub(super) fn key(
        state: &ModuleState,
        lane: &Lane,
    ) -> Result<(String, Value), WireOperationError> {
        let revision = format!(
            "{}:{}",
            lane.profile,
            synapse_certify::self_check::speech::revision()
        );
        let id = catalog_self_check_id(
            &lane.catalog_id,
            &lane.manifest_digest,
            &lane.backend,
            &lane.fingerprint,
            &lane.engine_identity,
            &state.machine_profile.os_build,
            &revision,
        )
        .map_err(catalog_store_error)?;
        Ok((
            id,
            json!({"catalog_id": lane.catalog_id, "manifest_digest": lane.manifest_digest,
            "backend": lane.backend, "fingerprint": lane.fingerprint,
            "engine_identity": lane.engine_identity, "os_build": state.machine_profile.os_build,
            "fixture_revision": revision}),
        ))
    }

    fn failed(lane: &Lane, id: &str, reason: &str) -> WireOperationError {
        catalog_wire_error(
            "self_check_failed",
            json!({
                "model_id": catalog::lane_id(&lane.catalog_id, &lane.backend),
                "backend": lane.backend, "check_id": id, "reason": reason,
            }),
            "speech numerical self-check failed",
        )
    }

    fn crashed(reason: impl Into<String>) -> WireOperationError {
        WireOperationError::from_stable(StableError::engine_crashed(Some(1000)), reason)
    }

    pub(crate) fn projection(
        state: &ModuleState,
        lane: &Lane,
    ) -> Result<Value, WireOperationError> {
        use rusqlite::OptionalExtension;
        let (id, _) = key(state, lane)?;
        let row = state.store.sqlite().with_conn(|conn| {
            conn.query_row("SELECT state, checked_at_ms, reason FROM catalog_self_checks WHERE check_id=?1", [&id],
                |r| Ok((r.get::<_, String>(0)?, r.get::<_, Option<u64>>(1)?, r.get::<_, Option<String>>(2)?))).optional()
        }).map_err(catalog_store_error)?;
        let (status, checked, reason) = row.unwrap_or(("pending".into(), None, None));
        Ok(json!({"check_id": id, "state": status, "checked_at_ms": checked, "reason": reason}))
    }

    /// Run in a blocking task that owns the engine, not in the request future.
    /// The lane lock, host and shape lease live until inference drains, even past the ceiling.
    /// Catalog removal/download clears the existing persisted rows; key rotation
    /// naturally makes a previous decision inapplicable.
    pub(crate) fn ensure(
        state: &ModuleState,
        lane: &Lane,
        load: impl FnOnce() -> Result<TranscribeHost, TranscribeHostError>,
        phase: impl FnMut(&'static str),
    ) -> Result<TranscribeHost, WireOperationError> {
        ensure_using(
            state,
            lane,
            load,
            phase,
            synapse_certify::self_check::speech::load,
        )
    }

    pub(crate) fn wall_ms(
        state: &ModuleState,
        lane: &Lane,
    ) -> Result<Option<u64>, WireOperationError> {
        use rusqlite::OptionalExtension;
        let (id, _) = key(state, lane)?;
        state.store.sqlite().with_conn(|conn| {
            conn.query_row("SELECT self_check_wall_ms FROM catalog_self_checks WHERE check_id=?1 AND state='passed'", [&id], |r| r.get(0)).optional().map(Option::flatten)
        }).map_err(catalog_store_error)
    }

    fn ensure_using(
        state: &ModuleState,
        lane: &Lane,
        load: impl FnOnce() -> Result<TranscribeHost, TranscribeHostError>,
        mut phase: impl FnMut(&'static str),
        references: impl FnOnce(
            &str,
        ) -> synapse_certify::Result<
            Vec<synapse_certify::self_check::speech::Case>,
        >,
    ) -> Result<TranscribeHost, WireOperationError> {
        let model_id = catalog::lane_id(&lane.catalog_id, &lane.backend);
        let _guard = catalog_lane_lock(&state.runtime, &model_id).blocking_lock_owned();
        let (id, key) = key(state, lane)?;
        let saved = projection(state, lane)?;
        if saved["state"] == "failed" {
            return Err(failed(
                lane,
                &id,
                saved["reason"].as_str().unwrap_or("reference mismatch"),
            ));
        }
        let cached = saved["state"] == "passed" && wall_ms(state, lane)?.is_some();
        let generation = if cached {
            None
        } else {
            Some(catalog_check_generation(state, &id, &key)?)
        };
        let references = match references(&lane.profile) {
            Ok(references) => references,
            Err(error) => {
                let reason = error.to_string();
                let generation = match generation {
                    Some(generation) => generation,
                    None => catalog_check_generation(state, &id, &key)?,
                };
                if !catalog_complete_check(state, &id, generation, "failed", Some(&reason))? {
                    return Err(crashed("speech self-check generation changed"));
                }
                return Err(failed(lane, &id, &reason));
            }
        };
        let mut elapsed_ms = None;
        let execution = (|| {
            phase("loading");
            // Loading a worker host validates its reported transcription revisions.
            let host = load().map_err(|e| crashed(e.to_string()))?;
            if host.profile().id != lane.profile {
                return Err(crashed("self-check host profile differs from lane"));
            }
            if cached {
                return Ok((host, None));
            }
            let mismatch = {
                let frames = references
                    .iter()
                    .map(|c| c.valid_encoder_frames)
                    .max()
                    .expect("sealed cases");
                if host.profile().lane == "ane-direct-worker" {
                    phase("shape_compiling");
                }
                let admission = host.admit_self_check(frames).map_err(|e| {
                    #[cfg(unix)]
                    if let TranscribeHostError::Ane(error) = e {
                        let wire = ane_residency_error_to_wire(error);
                        if wire.code == "shape_compiling" {
                            return wire;
                        }
                        return crashed(wire.message);
                    }
                    crashed(e.to_string())
                })?;
                phase("self_check");
                let start = Instant::now();
                let mut mismatch = None;
                for case in &references {
                    let result = admission.transcribe(&case.wav);
                    // The ceiling includes WAV decode, frontend, engine and words,
                    // but starts only after load/revision and shape admission.
                    if start.elapsed() > Duration::from_millis(2000) {
                        return Err(crashed("speech self-check exceeded 2000 ms"));
                    }
                    let result = result.map_err(|e| crashed(e.to_string()))?;
                    let triples = result
                        .output
                        .triples
                        .iter()
                        .map(|t| [t.token_id, t.frame_index, t.duration])
                        .collect::<Vec<_>>();
                    if let Err(error) = synapse_certify::self_check::speech::compare(
                        case,
                        &triples,
                        &result.transcript.text,
                    ) {
                        mismatch = Some(error.to_string());
                        break;
                    }
                }
                elapsed_ms = Some(u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX));
                mismatch
            };
            Ok((host, mismatch))
        })();
        match execution {
            Ok((host, mismatch)) => {
                let status = if mismatch.is_some() {
                    "failed"
                } else {
                    "passed"
                };
                if let Some(generation) = generation {
                    state.store.sqlite().with_conn_fenced(|tx| {
                        tx.execute("UPDATE catalog_self_checks SET self_check_wall_ms=?3 WHERE check_id=?1 AND generation=?2 AND state='running'", rusqlite::params![id,generation,elapsed_ms])?;
                        Ok(())
                    }).map_err(catalog_store_error)?;
                    if !catalog_complete_check(state, &id, generation, status, mismatch.as_deref())?
                    {
                        return Err(crashed("speech self-check generation changed"));
                    }
                }
                if let Some(reason) = mismatch {
                    return Err(failed(lane, &id, &reason));
                }
                phase("ready");
                Ok(host)
            }
            Err(error) => {
                if let Some(generation) = generation {
                    catalog_complete_check(state, &id, generation, "pending", None)?;
                }
                Err(error)
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use transcribe_host::TranscribeProfile;

        fn fixture(label: &str, backend: &str) -> (PathBuf, Arc<ModuleState>, Lane) {
            let (root, descriptor) = crate::tests::test_storage_descriptor(label);
            let store = Arc::new(SynapseStore::open(&descriptor).unwrap());
            let profile = crate::tests::test_machine_profile("speech-check-os");
            store.activate_profile(&profile, 1, 1000).unwrap();
            let state = crate::tests::test_module_state(store, profile);
            let engine = if backend == "metal" {
                "owned-metal"
            } else {
                "owned-cuda"
            };
            let lane = Lane {
                catalog_id: "parakeet-tdt-0.6b-v3".into(),
                backend: backend.into(),
                profile: format!("parakeet-tdt-0.6b-v3.{engine}"),
                fingerprint: "test-pin".into(),
                manifest_digest: "test-manifest".into(),
                engine_identity: json!({"engine":engine}),
            };
            (root, state, lane)
        }

        #[derive(Clone, Copy)]
        enum Behavior {
            Pass,
            Mismatch,
            Runtime,
            Slow,
        }

        fn output(
            request: &synapse_core::TranscribeRequest,
            behavior: Behavior,
            index: usize,
        ) -> Result<synapse_core::TranscribeOutput, EngineError> {
            assert!(
                !request.observe,
                "certification mode must not leak into a self-check"
            );
            assert_eq!(request.mel.len(), request.mel_frames * 128);
            assert!(request.mel.iter().all(|n| n.is_finite()));
            if matches!(behavior, Behavior::Runtime) {
                return Err(EngineError {
                    stage: EngineErrorStage::Inference,
                    risk_class: synapse_core::EngineRiskClass::AbortSafe,
                    message: "runtime fixture".into(),
                    retry_after_ms: None,
                    safe_to_retry_same_request: false,
                });
            }
            if matches!(behavior, Behavior::Slow) {
                std::thread::sleep(Duration::from_millis(2001));
            }
            let cases =
                synapse_certify::self_check::speech::load("parakeet-tdt-0.6b-v3.owned-metal")
                    .unwrap();
            let case = &cases[index];
            assert_eq!(
                request.valid_mel_frames.div_ceil(8),
                case.valid_encoder_frames
            );
            let mut triples = case
                .triples
                .iter()
                .map(|t| synapse_core::TranscriptRecord {
                    token_id: t[0],
                    frame_index: t[1],
                    duration: t[2],
                })
                .collect::<Vec<_>>();
            if matches!(behavior, Behavior::Mismatch) {
                triples.push(synapse_core::TranscriptRecord {
                    token_id: 380,
                    frame_index: 0,
                    duration: 1,
                });
            }
            Ok(synapse_core::TranscribeOutput {
                valid_encoder_frames: case.valid_encoder_frames,
                triples,
                observation: None,
            })
        }

        fn host(lane: &Lane, behavior: Behavior, calls: Arc<AtomicUsize>) -> TranscribeHost {
            let profile = TranscribeProfile::pinned(&lane.profile).unwrap();
            #[cfg(unix)]
            if lane.backend == "cuda" {
                let engine = worker_host::WorkerEngine::new(worker_host::WorkerHostConfig::new(
                    "/bin/false",
                    "unused",
                ))
                .unwrap();
                let model =
                    engine.insert_loaded_model_for_test("speech-check".into(), 1024, 0, None);
                engine.attach_mock_worker_for_test(move |request, raw| {
                    let synapse_core::WorkerRequest::Transcribe {
                        req_id,
                        mel_frames,
                        valid_mel_frames,
                        observe,
                        ..
                    } = request
                    else {
                        panic!("expected TRANSCRIBE")
                    };
                    let request = synapse_core::TranscribeRequest {
                        mel: synapse_core::decode_f32_frame(raw.unwrap()).unwrap(),
                        mel_frames: *mel_frames,
                        valid_mel_frames: *valid_mel_frames,
                        observe: *observe,
                    };
                    let index = calls.fetch_add(1, Ordering::SeqCst);
                    match output(&request, behavior, index) {
                        Ok(out) => (
                            synapse_core::WorkerResponse::Transcript {
                                req_id: req_id.clone(),
                                valid_encoder_frames: out.valid_encoder_frames,
                                count: out.triples.len(),
                                peak_device_memory_bytes: None,
                                peak_storage_buffer_bytes: None,
                            },
                            Some(synapse_core::encode_transcript_frame(&out.triples)),
                        ),
                        Err(_) => (
                            synapse_core::WorkerResponse::Err {
                                req_id: Some(req_id.clone()),
                                code: "engine_crashed".into(),
                                msg: "runtime fixture".into(),
                            },
                            None,
                        ),
                    }
                });
                return TranscribeHost::worker_for_test(profile, engine, model);
            }
            TranscribeHost::in_process_for_test(profile, move |request| {
                output(&request, behavior, calls.fetch_add(1, Ordering::SeqCst))
            })
        }

        fn outcomes(backend: &str) {
            for (name, behavior, code) in [
                ("pass", Behavior::Pass, None),
                ("mismatch", Behavior::Mismatch, Some("self_check_failed")),
                ("runtime", Behavior::Runtime, Some("engine_crashed")),
                ("slow", Behavior::Slow, Some("engine_crashed")),
            ] {
                let (root, state, lane) = fixture(&format!("speech-{backend}-{name}"), backend);
                let calls = Arc::new(AtomicUsize::new(0));
                let mut phases = Vec::new();
                let result = ensure(
                    &state,
                    &lane,
                    || Ok(host(&lane, behavior, calls.clone())),
                    |p| phases.push(p),
                );
                if let Some(code) = code {
                    let error = result.err().unwrap();
                    assert_eq!(error.code, code);
                    if code == "self_check_failed" {
                        assert_eq!(error.class, ErrorClass::Permanent);
                        assert!(error.retry_after_ms.is_none());
                        let details = error.details.unwrap();
                        assert_eq!(details["backend"], backend);
                        assert_eq!(
                            details["model_id"],
                            catalog::lane_id(&lane.catalog_id, backend)
                        );
                        assert_eq!(
                            details["check_id"],
                            projection(&state, &lane).unwrap()["check_id"]
                        );
                        assert!(details["reason"].as_str().unwrap().contains("silence-1s"));
                        assert_eq!(projection(&state, &lane).unwrap()["state"], "failed");
                        assert_eq!(
                            ensure(
                                &state,
                                &lane,
                                || panic!("permanent failures never load again"),
                                |_| {}
                            )
                            .err()
                            .unwrap()
                            .code,
                            code
                        );
                    } else {
                        assert_eq!(error.class, ErrorClass::Transient);
                        assert!(error.retry_after_ms.is_some());
                        assert_eq!(projection(&state, &lane).unwrap()["state"], "pending");
                    }
                } else {
                    drop(result.unwrap());
                    assert_eq!(calls.load(Ordering::SeqCst), 3);
                    assert_eq!(phases, ["loading", "self_check", "ready"]);
                    assert_eq!(projection(&state, &lane).unwrap()["state"], "passed");
                    let (check_id, _) = key(&state, &lane).unwrap();
                    let generation = || {
                        state
                            .store
                            .sqlite()
                            .with_conn(|conn| {
                                conn.query_row(
                                    "SELECT generation FROM catalog_self_checks WHERE check_id=?1",
                                    [&check_id],
                                    |row| row.get::<_, u64>(0),
                                )
                            })
                            .unwrap()
                    };
                    let checked_generation = generation();
                    let checked_at = projection(&state, &lane).unwrap()["checked_at_ms"].clone();
                    let checked_wall = wall_ms(&state, &lane).unwrap().unwrap();
                    assert!(checked_wall < 2000);
                    assert_ne!(json!(checked_wall), checked_at);
                    drop(
                        ensure(
                            &state,
                            &lane,
                            || Ok(host(&lane, Behavior::Runtime, calls.clone())),
                            |_| {},
                        )
                        .unwrap(),
                    );
                    assert_eq!(
                        calls.load(Ordering::SeqCst),
                        3,
                        "cached pass runs no inference"
                    );
                    assert_eq!(wall_ms(&state, &lane).unwrap(), Some(checked_wall));
                    assert_eq!(
                        generation(),
                        checked_generation,
                        "reusing evidence starts no new check"
                    );
                    assert_eq!(
                        projection(&state, &lane).unwrap()["checked_at_ms"],
                        checked_at
                    );
                }
                drop(state);
                fs::remove_dir_all(root).unwrap();
            }
        }

        #[test]
        fn speech_self_check_in_process_outcomes() {
            outcomes("metal");
        }

        #[cfg(unix)]
        #[test]
        fn speech_self_check_mock_worker_outcomes() {
            outcomes("cuda");
        }

        #[test]
        fn speech_self_check_reference_failure_persists_until_key_change_or_download() {
            reference_failures("metal");
            #[cfg(unix)]
            reference_failures("cuda");
        }

        fn reference_failures(backend: &str) {
            for invalid in [false, true] {
                let (root, state, mut lane) = fixture(
                    if invalid {
                        "speech-seal"
                    } else {
                        "speech-absent"
                    },
                    backend,
                );
                let error = ensure_using(
                    &state,
                    &lane,
                    || panic!("bad references never infer"),
                    |_| {},
                    |_| {
                        if invalid {
                            Err(synapse_certify::Error("invalid seal".into()))
                        } else {
                            synapse_certify::self_check::speech::load("absent")
                        }
                    },
                )
                .err()
                .unwrap();
                assert_eq!(error.code, "self_check_failed");
                assert!(error.retry_after_ms.is_none());
                assert_eq!(
                    ensure(
                        &state,
                        &lane,
                        || panic!("failed reference stays failed"),
                        |_| {}
                    )
                    .err()
                    .unwrap(),
                    error
                );
                let old = projection(&state, &lane).unwrap()["check_id"].clone();
                lane.fingerprint = "rotated".into();
                assert_ne!(projection(&state, &lane).unwrap()["check_id"], old);
                drop(
                    ensure(
                        &state,
                        &lane,
                        || Ok(host(&lane, Behavior::Pass, Arc::new(AtomicUsize::new(0)))),
                        |_| {},
                    )
                    .unwrap(),
                );
                lane.fingerprint = "test-pin".into();
                // This is the same catalog-id reset used by model removal/download.
                state
                    .store
                    .sqlite()
                    .with_conn_fenced(|tx| {
                        tx.execute(
                            "DELETE FROM catalog_self_checks WHERE catalog_id=?1",
                            [&lane.catalog_id],
                        )
                    })
                    .unwrap();
                drop(
                    ensure(
                        &state,
                        &lane,
                        || Ok(host(&lane, Behavior::Pass, Arc::new(AtomicUsize::new(0)))),
                        |_| {},
                    )
                    .unwrap(),
                );
                drop(state);
                fs::remove_dir_all(root).unwrap();
            }
        }

        #[test]
        fn speech_self_check_load_failure_is_retryable_and_precedes_execution() {
            load_failure("metal");
            #[cfg(unix)]
            load_failure("cuda");
        }

        fn load_failure(backend: &str) {
            let (root, state, lane) = fixture("speech-load", backend);
            let error = ensure(
                &state,
                &lane,
                || Err(TranscribeHostError::Profile("load fixture".into())),
                |_| {},
            )
            .err()
            .unwrap();
            assert_eq!(error.code, "engine_crashed");
            assert!(error.retry_after_ms.is_some());
            assert_eq!(projection(&state, &lane).unwrap()["state"], "pending");
            drop(
                ensure(
                    &state,
                    &lane,
                    || {
                        if backend == "metal" {
                            std::thread::sleep(Duration::from_millis(2001));
                        }
                        Ok(host(&lane, Behavior::Pass, Arc::new(AtomicUsize::new(0))))
                    },
                    |_| {},
                )
                .unwrap(),
            );
            drop(state);
            fs::remove_dir_all(root).unwrap();
        }

        /// Uses the actual Metal engine rather than a test callback.
        #[cfg(target_os = "macos")]
        #[test]
        #[ignore = "requires pinned model and SYNAPSE_METAL_SPEECH_GPU=1 hardware opt-in"]
        fn speech_self_check_real_metal() {
            assert_eq!(env::var("SYNAPSE_METAL_SPEECH_GPU").as_deref(), Ok("1"));
            let (root, state, lane) = fixture("speech-real-metal", "metal");
            let cfg = RuntimeConfig {
                values: [(
                    "model_path".into(),
                    env::var("SYNAPSE_METAL_SPEECH_MODEL").unwrap(),
                )]
                .into(),
            };
            drop(
                ensure(
                    &state,
                    &lane,
                    || {
                        TranscribeHost::load_metal(
                            TranscribeProfile::pinned(&lane.profile).unwrap(),
                            &synapse_core::ValidatedArtifact {
                                format: "safetensors-package".into(),
                                digest: "test".into(),
                            },
                            &cfg,
                            transcribe_host::ObservationMode::Certification,
                        )
                    },
                    |_| {},
                )
                .unwrap(),
            );
            drop(state);
            fs::remove_dir_all(root).unwrap();
        }
    }
}

fn catalog_self_check_key(
    state: &ModuleState,
    entry: &catalog::CatalogEntry,
    backend: &catalog::CatalogBackend,
) -> Result<(String, Value), WireOperationError> {
    if entry.task == "transcribe" {
        return transcribe_self_check::key(state, &catalog_transcribe_lane(entry, backend)?);
    }
    let identity = if let Some(profile) = backend.profile.as_deref() {
        CatalogProfile::load(profile)
            .and_then(|profile| profile.owned_config(backend.execution.as_deref(), None))
            .map_err(|error| artifact_invalid_error(error.to_string()))?
            .identity_override
            .expect("profile identity")
    } else {
        let family = OwnedFamily::parse(backend.family.as_deref().expect("family"))
            .map_err(|e| artifact_invalid_error(e.to_string()))?;
        let dtype = OwnedDType::parse(backend.dtype.as_deref().expect("dtype"))
            .map_err(|e| artifact_invalid_error(e.to_string()))?;
        engine_identity_with_projections(family, dtype, catalog_projection_dtype(family, dtype))
    };
    #[cfg(feature = "test-support")]
    let identity = {
        let mut identity = identity;
        if let Ok(version) = env::var("SYNAPSE_TEST_ENGINE_IDENTITY") {
            identity.version = version;
        }
        identity
    };
    let fixture_revision = if let Some(profile) = backend.profile.as_deref() {
        synapse_certify::self_check::seal_digest(profile)
            .map_err(|error| artifact_invalid_error(error.to_string()))?
    } else {
        entry
            .self_check
            .as_ref()
            .expect("self-check")
            .fixture_revision
            .to_string()
    };
    let key = json!({"catalog_id":entry.id,"manifest_digest":entry.manifest_digest(),"backend":backend.backend,"fingerprint":backend.fingerprint,"engine_identity":identity,"os_build":state.machine_profile.os_build,"fixture_revision":fixture_revision});
    let id = catalog_self_check_id(
        &entry.id,
        &entry.manifest_digest(),
        &backend.backend,
        &backend.fingerprint,
        &key["engine_identity"],
        &state.machine_profile.os_build,
        &fixture_revision,
    )
    .map_err(catalog_store_error)?;
    Ok((id, key))
}
fn catalog_self_check_projection(
    state: &ModuleState,
    entry: &catalog::CatalogEntry,
    backend: &catalog::CatalogBackend,
) -> Result<Value, WireOperationError> {
    if entry.task == "transcribe" {
        return transcribe_self_check::projection(state, &catalog_transcribe_lane(entry, backend)?);
    }
    use rusqlite::OptionalExtension;
    let (id, _) = catalog_self_check_key(state, entry, backend)?;
    let row = state
        .store
        .sqlite()
        .with_conn(|conn| {
            conn.query_row(
                "SELECT state, checked_at_ms, reason FROM catalog_self_checks WHERE check_id=?1",
                [&id],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, Option<u64>>(1)?,
                        r.get::<_, Option<String>>(2)?,
                    ))
                },
            )
            .optional()
        })
        .map_err(catalog_store_error)?;
    let (status, checked, reason) = row.unwrap_or(("pending".into(), None, None));
    Ok(json!({"state":status,"checked_at_ms":checked,"reason":reason}))
}
fn catalog_self_check_failed(
    entry: &catalog::CatalogEntry,
    backend: &catalog::CatalogBackend,
    check_id: &str,
    reason: &Value,
) -> WireOperationError {
    catalog_wire_error(
        "self_check_failed",
        json!({"model_id":catalog::lane_id(&entry.id,&backend.backend),"backend":backend.backend,"check_id":check_id,"reason":reason}),
        "catalog numerical self-check failed",
    )
}
fn catalog_check_generation(
    state: &ModuleState,
    id: &str,
    key: &Value,
) -> Result<u64, WireOperationError> {
    invalidate_validated_readiness(&state.runtime);
    let result = state.store.sqlite().with_conn_fenced(|tx| {
        tx.execute("UPDATE self_check_run_seq SET value=value+1 WHERE id=0",[])?;
        let generation: u64 = tx.query_row("SELECT value FROM self_check_run_seq WHERE id=0",[],|r| r.get(0))?;
        tx.execute("INSERT INTO catalog_self_checks (check_id,catalog_id,manifest_digest,backend,fingerprint,engine_identity_json,os_build,fixture_revision,state,generation,checked_at_ms,reason) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,'running',?9,NULL,NULL) ON CONFLICT(check_id) DO UPDATE SET state='running',generation=excluded.generation,checked_at_ms=NULL,reason=NULL",rusqlite::params![id,key["catalog_id"].as_str(),key["manifest_digest"].as_str(),key["backend"].as_str(),key["fingerprint"].as_str(),key["engine_identity"].to_string(),key["os_build"].as_str(),key["fixture_revision"].as_str(),generation])?;
        Ok(generation)
    }).map_err(catalog_store_error);
    invalidate_validated_readiness(&state.runtime);
    result
}
fn catalog_complete_check(
    state: &ModuleState,
    id: &str,
    generation: u64,
    status: &str,
    reason: Option<&str>,
) -> Result<bool, WireOperationError> {
    invalidate_validated_readiness(&state.runtime);
    let result = state.store.sqlite().with_conn_fenced(|tx| {
        if status == "pending" {
            tx.execute("UPDATE self_check_run_seq SET value=value+1 WHERE id=0",[])?;
            Ok(tx.execute("UPDATE catalog_self_checks SET state='pending',generation=(SELECT value FROM self_check_run_seq WHERE id=0),checked_at_ms=NULL,reason=NULL WHERE check_id=?1 AND state='running' AND generation=?2",rusqlite::params![id,generation])? > 0)
        } else {
            Ok(tx.execute("UPDATE catalog_self_checks SET state=?3,checked_at_ms=?4,reason=?5 WHERE check_id=?1 AND state='running' AND generation=?2",rusqlite::params![id,generation,status,now_ms(),reason])? > 0)
        }
    }).map_err(catalog_store_error);
    invalidate_validated_readiness(&state.runtime);
    if matches!(result, Ok(true)) {
        state.runtime.findings_changed.notify_one();
    }
    result
}
fn owned_numerical_catalog_check(
    model: &EmbeddingModel,
    entry: &catalog::CatalogEntry,
    backend: &catalog::CatalogBackend,
) -> Result<(bool, f64, f64), WireOperationError> {
    catalog_call_fault(&model.model_id, "self_check")?;
    let EmbedBackend::Owned(engine) = &model.backend else {
        return Err(artifact_invalid_error(
            "catalog self-check requires an owned engine",
        ));
    };
    let engine = engine
        .lock()
        .map_err(|_| transient_model_load_error("owned engine mutex poisoned"))?;
    numerical_catalog_check(
        model,
        entry,
        backend,
        |batch| engine.embed_batch(&model.loaded_model, batch),
        |pairs| engine.rerank_pairs(&model.loaded_model, pairs),
    )
}

fn numerical_catalog_check(
    model: &EmbeddingModel,
    entry: &catalog::CatalogEntry,
    backend: &catalog::CatalogBackend,
    embed_batch: impl Fn(TokenBatch) -> Result<Vec<Vec<f32>>, EngineError>,
    rerank_pairs: impl Fn(Vec<Vec<u32>>) -> Result<synapse_core::RerankScores, EngineError>,
) -> Result<(bool, f64, f64), WireOperationError> {
    let check = entry.self_check.as_ref().expect("self-check");
    if entry.task == "embed" {
        let texts = check
            .inputs
            .iter()
            .map(|i| match i {
                catalog::SelfCheckInput::Text(s) => s.as_str(),
                _ => unreachable!(),
            })
            .collect::<Vec<_>>();
        let mut tokenized = model
            .tokenizer
            .tokenize_batch(texts)
            .map_err(|e| artifact_invalid_error(e.to_string()))?;
        apply_owned_tokenizer_policy(model, &mut tokenized);
        let vectors = embed_batch(tokenized.batch).map_err(engine_error_to_wire)?;
        let references = check.reference.vectors.as_ref().expect("embed references");
        let shape = vectors.len() == 8
            && vectors.iter().all(|v| {
                v.len() == backend.dims.expect("dims") as usize && v.iter().all(|x| x.is_finite())
            });
        let min = if shape {
            vectors
                .iter()
                .zip(references)
                .map(|(v, r)| cosine(v, r))
                .fold(1.0f64, f64::min)
        } else {
            -1.0
        };
        Ok((shape && min.is_finite() && min >= 0.999, min, 0.999))
    } else {
        let profile_id = format!("{}.{}", entry.id, backend.engine);
        let profile = CatalogProfile::load(&profile_id)
            .map_err(|error| artifact_invalid_error(error.to_string()))?;
        let tolerance = synapse_parity::evaluator::rerank_tolerance(
            profile.typed.profiles[&profile_id].rerank_tolerance_class,
        );
        let mut max = 0.0f64;
        let mut passed = true;
        for (input, reference) in check
            .inputs
            .iter()
            .zip(check.reference.scores.as_ref().expect("rerank references"))
        {
            let catalog::SelfCheckInput::Rerank(input) = input else {
                unreachable!()
            };
            let pairs =
                owned_rerank_pairs(model, &input.query, &input.candidates)?.expect("owned pairs");
            let raw = rerank_pairs(pairs).map_err(engine_error_to_wire)?;
            let raw_finite = raw.scores.iter().all(|score| score.is_finite());
            let scores = RerankReadout::select(true, &model.engine_identity)
                .apply(raw)
                .scores
                .into_iter()
                .map(f64::from)
                .collect::<Vec<_>>();
            if scores.len() != reference.len() || !raw_finite {
                passed = false;
                continue;
            }
            for (s, r) in scores.iter().zip(reference) {
                max = max.max((s - r).abs());
            }
            for i in 0..scores.len() {
                for j in i + 1..scores.len() {
                    if (reference[i] - reference[j]).abs() >= tolerance
                        && (scores[i] - scores[j]) * (reference[i] - reference[j]) < 0.0
                    {
                        passed = false;
                    }
                }
            }
        }
        Ok((passed && max <= tolerance, max, tolerance))
    }
}

fn resolved_catalog_lane<'a>(
    runtime: &'a RuntimeState,
    id: &str,
) -> Option<(&'a catalog::CatalogEntry, &'a catalog::CatalogBackend)> {
    let (entry, backend) = runtime.release_catalog.resolve_reserved(id)?;
    Some((entry, entry.backend(backend?)?))
}
fn select_catalog_lane(
    state: &ModuleState,
    requested: Option<&str>,
    task: ModelTask,
    required: Option<&str>,
    target: Option<&str>,
) -> Result<(catalog::CatalogEntry, catalog::CatalogBackend), WireOperationError> {
    let requested = requested
        .map(str::to_string)
        .or_else(|| {
            state
                .runtime
                .release_catalog
                .models
                .iter()
                .find(|e| e.task == task.as_str() && e.default_for_task)
                .map(|e| e.id.clone())
        })
        .ok_or_else(|| catalog_unknown(task.as_str()))?;
    let (entry, pinned) = state
        .runtime
        .release_catalog
        .resolve_reserved(&requested)
        .ok_or_else(|| catalog_unknown(&requested))?;
    let pinned_backend = pinned
        .map(|b| entry.backend(b).ok_or_else(|| catalog_unknown(&requested)))
        .transpose()?;
    if entry.task != task.as_str() {
        return Err(catalog_wire_error(
            "invalid_request",
            json!({"catalog_id":entry.id,"requested_task":task.as_str(),"catalog_task":entry.task}),
            "catalog task does not match request",
        ));
    }
    let fingerprint = required.or(target);
    let matched = fingerprint.map(|f| entry.backends.iter().find(|b| b.fingerprint == f));
    if matched.is_some_and(|b| b.is_none())
        || pinned_backend.is_some_and(|b| fingerprint.is_some_and(|f| f != b.fingerprint))
        || required.zip(target).is_some_and(|(r, t)| r != t)
    {
        return Err(catalog_wire_error(
            "substitution_rejected",
            json!({"lane_id":pinned_backend.map(|b| catalog::lane_id(&entry.id,&b.backend)),"required_fingerprint":fingerprint}),
            "catalog fingerprints cannot be substituted",
        ));
    }
    let selected = pinned_backend.or(matched.flatten()).or_else(|| {
        entry
            .backends
            .iter()
            .find(|b| state.runtime.runnable_backends.contains(&b.backend))
    });
    let backend = selected.ok_or_else(|| catalog_backend_unavailable(state, entry, None))?;
    if catalog_backend_reason(
        &state.runtime,
        &backend.backend,
        synapse_core::Platform::current(),
    )
    .is_some()
    {
        return Err(catalog_backend_unavailable(state, entry, Some(backend)));
    }
    let lane = catalog::lane_id(&entry.id, &backend.backend);
    let ready_model = model_slot_snapshot(&state.runtime, &lane)
        .filter(|slot| matches!(slot.state, ModelRuntimeState::Ready))
        .and_then(|slot| slot.loaded);
    let ready_passed = ready_model
        .as_ref()
        .map(|model| validated_model_readiness(state, model))
        .transpose()?
        .unwrap_or(false);
    if ready_passed {
        // A numerical check can pass for a config preload even when the catalog's
        // pinned files were never installed. Skip the install-row lookup only after
        // this selector has verified it for this model instance. Removal cannot
        // delete the installation while the model is resident.
        if state
            .runtime
            .validated_readiness
            .lock()
            .expect("validated readiness")
            .models
            .get(&lane)
            .is_some_and(|passed| {
                passed.install_validated
                    && passed.model.upgrade().is_some_and(|current| {
                        Arc::ptr_eq(&current, ready_model.as_ref().expect("passed Ready model"))
                    })
            })
        {
            return Ok((entry.clone(), backend.clone()));
        }
    }
    if current_catalog_install(state, entry, &backend.backend)?.is_none() {
        let job = state
            .store
            .active_download_job(&catalog_request_digest(entry))
            .map_err(catalog_store_error)?;
        return Err(catalog_wire_error(
            "model_not_installed",
            json!({"catalog_id":entry.id,"lane_id":catalog::lane_id(&entry.id,&backend.backend),"download_op":"models.download","download_job_id":job.map(|j| j.job_id)}),
            "install this model with models.download",
        ));
    }
    if ready_passed {
        let mut cache = state
            .runtime
            .validated_readiness
            .lock()
            .expect("validated readiness");
        if let Some(passed) = cache.models.get_mut(&lane) {
            if passed.model.upgrade().is_some_and(|current| {
                Arc::ptr_eq(&current, ready_model.as_ref().expect("passed Ready model"))
            }) {
                passed.install_validated = true;
            }
        }
    }
    let check = catalog_self_check_projection(state, entry, backend)?;
    if check["state"] == "failed" {
        return Err(catalog_self_check_failed(
            entry,
            backend,
            &catalog_self_check_key(state, entry, backend)?.0,
            &check["reason"],
        ));
    }
    Ok((entry.clone(), backend.clone()))
}
fn profile_preload_check_error(profile: &str, reason: &str) -> WireOperationError {
    catalog_wire_error(
        "self_check_failed",
        json!({"profile": profile, "reason": reason}),
        format!("profile preload numerical self-check failed for {profile}: {reason}"),
    )
}

fn profile_preload_check_key(
    state: &ModuleState,
    model: &EmbeddingModel,
    profile: &str,
) -> Result<(String, Value), WireOperationError> {
    let seal = synapse_certify::self_check::seal_digest(profile)
        .map_err(|error| profile_preload_check_error(profile, &error.to_string()))?;
    let catalog = CatalogProfile::load(profile)
        .map_err(|error| profile_preload_check_error(profile, &error.to_string()))?;
    let key = json!({"catalog_id": model.model_id, "manifest_digest": catalog.typed.manifest_digest(), "backend": catalog.profile()["lane"], "fingerprint": model.fingerprint.0, "engine_identity": model.engine_identity, "os_build": state.machine_profile.os_build, "fixture_revision": seal});
    Ok((
        sha256_hex(catalog::jcs(&key).map_err(catalog_store_error)?.as_bytes()),
        key,
    ))
}

fn profile_preload_check_status(
    state: &ModuleState,
    id: &str,
) -> Result<String, WireOperationError> {
    use rusqlite::OptionalExtension;
    state
        .store
        .sqlite()
        .with_conn(|conn| {
            conn.query_row(
                "SELECT state FROM catalog_self_checks WHERE check_id=?1",
                [id],
                |row| row.get::<_, String>(0),
            )
            .optional()
        })
        .map(|row| row.unwrap_or_else(|| "pending".into()))
        .map_err(catalog_store_error)
}

fn complete_profile_preload_check(
    state: &ModuleState,
    profile: &str,
    id: &str,
    generation: u64,
    evaluation: &synapse_parity::evaluator::SubsetEvaluation,
) -> Result<(), WireOperationError> {
    let passed = evaluation.passed();
    let report = serde_json::to_string(evaluation).map_err(catalog_store_error)?;
    if !catalog_complete_check(
        state,
        id,
        generation,
        if passed { "passed" } else { "failed" },
        Some(&report),
    )? {
        return Err(profile_preload_check_error(
            profile,
            "self-check generation changed",
        ));
    }
    if passed {
        Ok(())
    } else {
        Err(profile_preload_check_error(profile, &report))
    }
}

async fn numerical_profile_preload_check(
    state: &ModuleState,
    model: &EmbeddingModel,
    profile: &str,
    references: synapse_certify::self_check::References,
    held_guard: Option<HeldLaneGate>,
) -> Result<synapse_parity::evaluator::SubsetEvaluation, WireOperationError> {
    use synapse_parity::evaluator::{ObservedCase, Output};
    let mut outputs = BTreeMap::new();
    for case in &references.cases {
        let id = case["id"].as_str().expect("sealed case id");
        let (output, input_ids, readout) = if model.task == ModelTask::Embed {
            let text = case["text"].as_str().expect("sealed embed text");
            let mut tokenized = model
                .tokenizer
                .tokenize_batch(vec![text])
                .map_err(|error| artifact_invalid_error(error.to_string()))?;
            compose_catalog_embed(model, &mut tokenized)?;
            apply_owned_tokenizer_policy(model, &mut tokenized);
            let input_ids = tokenized.batch.items[0].clone();
            let vectors = execute_embedding_with_catalog_guard(
                &state.runtime,
                model,
                tokenized.batch,
                None,
                None,
                held_guard.clone(),
            )
            .await?;
            let vector = vectors
                .into_iter()
                .next()
                .ok_or_else(|| profile_preload_check_error(profile, "missing self-check vector"))?;
            (
                Output::Embedding(vector.into_iter().map(f64::from).collect()),
                input_ids,
                None,
            )
        } else {
            let query = case["query"].as_str().expect("sealed rerank query");
            let candidates = vec![case["document"]
                .as_str()
                .expect("sealed rerank document")
                .to_string()];
            let pairs = owned_rerank_pairs(model, query, &candidates)?
                .ok_or_else(|| profile_preload_check_error(profile, "missing composed pairs"))?;
            let input_ids = pairs[0].clone();
            let scores = execute_rerank_with_catalog_guard(
                &state.runtime,
                model,
                RerankRequest {
                    query: vec![],
                    candidates: pairs.clone(),
                },
                Some(pairs),
                None,
                None,
                held_guard.clone(),
            )
            .await?;
            let score =
                scores.scores.into_iter().next().ok_or_else(|| {
                    profile_preload_check_error(profile, "missing self-check score")
                })?;
            let readout = &references
                .manifest
                .model(&references.manifest.profiles[profile].model)
                .expect("sealed model")
                .grammar()
                .expect("a rerank model has a token grammar")
                .readout;
            let readout = readout
                .yes
                .as_ref()
                .zip(readout.no.as_ref())
                .map(|(yes, no)| (yes.id, no.id));
            (Output::Score(f64::from(score)), input_ids, readout)
        };
        outputs.insert(
            id.to_string(),
            ObservedCase {
                output,
                input_ids,
                readout,
            },
        );
    }
    synapse_parity::evaluator::evaluate_subset(
        &references.manifest,
        profile,
        &model.fingerprint.0,
        &references.fixtures,
        &outputs,
    )
    .map_err(|error| profile_preload_check_error(profile, &error.to_string()))
}

async fn ensure_profile_preload_ready(
    state: Arc<ModuleState>,
    model_id: &str,
    deadline: Option<tokio::time::Instant>,
) -> Result<Arc<EmbeddingModel>, WireOperationError> {
    // Once the check has passed, serving takes no lane lock and starts nothing.
    if let Some(model) = profile_check_passed(&state, model_id)? {
        return Ok(model);
    }
    let loading = || {
        WireOperationError::from_stable(
            StableError::model_loading(Some(250)),
            format!("catalog lane '{model_id}' is running its numerical self-check"),
        )
    };
    #[cfg(test)]
    {
        let gate = state.runtime.fast_path_gate.lock().unwrap().take();
        if let Some((reached, proceed)) = gate {
            reached.notify_one();
            proceed.notified().await;
        }
    }
    let mut outcome = join_lane_check(&state, model_id);
    let finished = async move {
        outcome
            .wait_for(Option::is_some)
            .await
            .map(|outcome| outcome.clone().expect("waited for an outcome"))
    };
    let joined = match deadline {
        Some(deadline) => tokio::time::timeout_at(deadline, finished)
            .await
            .map_err(|_| loading())?,
        None => finished.await,
    };
    // The check and the timer can both be ready in the same poll; a check that
    // finished after the deadline still leaves this request expired.
    if deadline.is_some_and(|deadline| tokio::time::Instant::now() >= deadline) {
        return Err(loading());
    }
    joined.map_err(|_| {
        WireOperationError::from_stable(
            StableError::engine_crashed(Some(250)),
            format!("catalog lane '{model_id}' self-check task ended without an outcome"),
        )
    })?
}

/// The loaded model when its numerical self-check has already passed.
fn profile_check_passed(
    state: &ModuleState,
    model_id: &str,
) -> Result<Option<Arc<EmbeddingModel>>, WireOperationError> {
    let Some(model) = model_slot_snapshot(&state.runtime, model_id)
        .filter(|slot| matches!(slot.state, ModelRuntimeState::Ready))
        .and_then(|slot| slot.loaded)
    else {
        return Ok(None);
    };
    Ok(validated_model_readiness(state, &model)?.then_some(model))
}

fn invalidate_validated_readiness(runtime: &RuntimeState) {
    let mut cache = runtime
        .validated_readiness
        .lock()
        .expect("validated readiness");
    cache.revision = cache.revision.wrapping_add(1);
    cache.models.clear();
}

/// Reuse a passed check only for its current key and the same Ready model instance.
/// Check-generation writes invalidate the cache before and after their transaction,
/// so a store read racing a new check cannot publish an obsolete passed generation.
fn validated_model_readiness(
    state: &ModuleState,
    model: &Arc<EmbeddingModel>,
) -> Result<bool, WireOperationError> {
    use rusqlite::OptionalExtension;
    let key_started = Instant::now();
    tracing::debug!(model_id = %model.model_id, stage = "readiness_key_start", "catalog readiness");
    let id = if let Some((entry, backend)) = resolved_catalog_lane(&state.runtime, &model.model_id)
    {
        catalog_self_check_key(state, entry, backend)?.0
    } else {
        profile_check_identity(state, model)?.0
    };
    tracing::debug!(model_id = %model.model_id, stage = "readiness_key_finish",
        wall_ms = key_started.elapsed().as_secs_f64() * 1000.0, "catalog readiness");
    let same_ready_model = || {
        model_slot_snapshot(&state.runtime, &model.model_id).is_some_and(|slot| {
            matches!(slot.state, ModelRuntimeState::Ready)
                && slot
                    .loaded
                    .is_some_and(|current| Arc::ptr_eq(&current, model))
        })
    };
    if !same_ready_model() {
        return Ok(false);
    }
    let revision = {
        let cache = state
            .runtime
            .validated_readiness
            .lock()
            .expect("validated readiness");
        if let Some(passed) = cache.models.get(&model.model_id) {
            if passed.check_id == id
                && passed
                    .model
                    .upgrade()
                    .is_some_and(|current| Arc::ptr_eq(&current, model))
            {
                tracing::debug!(model_id = %model.model_id, check_id = %id,
                    generation = passed.generation, stage = "validated_readiness_reuse",
                    "catalog readiness");
                return Ok(true);
            }
        }
        cache.revision
    };
    let started = Instant::now();
    tracing::debug!(model_id = %model.model_id, stage = "readiness_store_start", "catalog readiness");
    let passed_generation =
        state
            .store
            .sqlite()
            .with_conn(|conn| {
                conn.query_row(
            "SELECT generation FROM catalog_self_checks WHERE check_id=?1 AND state='passed'",
            [&id], |row| row.get::<_, u64>(0),
        ).optional()
            })
            .map_err(catalog_store_error)?;
    tracing::debug!(model_id = %model.model_id, check_id = %id, stage = "readiness_store",
        wall_ms = started.elapsed().as_secs_f64() * 1000.0, "catalog readiness");
    let Some(generation) = passed_generation else {
        return Ok(false);
    };
    let mut cache = state
        .runtime
        .validated_readiness
        .lock()
        .expect("validated readiness");
    if cache.revision != revision || !same_ready_model() {
        return Ok(false);
    }
    cache.models.insert(
        model.model_id.clone(),
        ValidatedReadiness {
            model: Arc::downgrade(model),
            check_id: id,
            generation,
            install_validated: false,
        },
    );
    Ok(true)
}

type LaneCheckOutcome = Option<Result<Arc<EmbeddingModel>, WireOperationError>>;

/// Removes a lane's in-flight check entry when its task ends, even by panic.
struct LaneCheckEntry {
    runtime: Arc<RuntimeState>,
    lane: String,
}

impl Drop for LaneCheckEntry {
    fn drop(&mut self) {
        if let Ok(mut checks) = self.runtime.lane_checks.lock() {
            checks.remove(&self.lane);
        }
    }
}

/// Subscribes to the lane's in-flight lifecycle check, starting it if none
/// is running. At most one check task runs per lane: callers that give up
/// leave nothing queued behind it.
fn join_lane_check(
    state: &Arc<ModuleState>,
    model_id: &str,
) -> tokio::sync::watch::Receiver<LaneCheckOutcome> {
    let mut checks = state.runtime.lane_checks.lock().expect("lane checks");
    if let Some(outcome) = checks.get(model_id) {
        return outcome.clone();
    }
    // The caller's fast path may have seen the check pending just before it
    // passed and left this map. Decide again here, where no new check can
    // start concurrently, so a passed check is never run a second time.
    match profile_check_passed(state, model_id) {
        Ok(Some(model)) => return tokio::sync::watch::channel(Some(Ok(model))).1,
        Ok(None) => {}
        Err(error) => return tokio::sync::watch::channel(Some(Err(error))).1,
    }
    let (publish, outcome) = tokio::sync::watch::channel(None);
    checks.insert(model_id.to_string(), outcome.clone());
    drop(checks);
    #[cfg(test)]
    state
        .runtime
        .lane_check_tasks
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let entry = LaneCheckEntry {
        runtime: state.runtime.clone(),
        lane: model_id.to_string(),
    };
    let state = state.clone();
    let model_id = model_id.to_string();
    tokio::spawn(async move {
        let result = run_lane_check(&state, &model_id).await;
        // Leave the map before publishing, so a caller that sees this outcome
        // and asks again takes the fast path or starts a fresh check.
        drop(entry);
        let _ = publish.send(Some(result));
    });
    outcome
}

/// The lane lifecycle work behind certification: under the lane lock, run the
/// numerical self-check unless it has already been decided.
async fn run_lane_check(
    state: &Arc<ModuleState>,
    model_id: &str,
) -> Result<Arc<EmbeddingModel>, WireOperationError> {
    let guard = catalog_lane_lock(&state.runtime, model_id)
        .lock_owned()
        .await;
    // Never load here. Requests load lanes through resolution; a check that
    // was queued before an unload must not bring the lane back.
    let model = model_slot_snapshot(&state.runtime, model_id)
        .filter(|slot| matches!(slot.state, ModelRuntimeState::Ready))
        .and_then(|slot| slot.loaded)
        .ok_or_else(|| {
            WireOperationError::from_stable(
                StableError::model_loading(Some(250)),
                format!("model '{model_id}' is not loaded"),
            )
        })?;
    let model = check_profile_model(state, model, guard).await?;
    start_ready_lane_work(&state.runtime, &model);
    Ok(model)
}

/// Background work for a lane whose numerical self-check has passed: packed
/// preparation and the direct-ANE prewarm. Each does nothing for a lane it
/// does not apply to and ignores a repeated start for the same lane load.
fn start_ready_lane_work(runtime: &Arc<RuntimeState>, model: &Arc<EmbeddingModel>) {
    start_packed_preparation(runtime, model);
    start_direct_ane_prewarm(runtime, model);
}

/// Whether [`start_direct_ane_prewarm`] applies to `model`: a direct-ANE
/// catalog lane.
fn direct_ane_prewarm_enabled(runtime: &RuntimeState, model: &EmbeddingModel) -> bool {
    is_direct_ane(model) && resolved_catalog_lane(runtime, &model.model_id).is_some()
}

/// Called when a lane finishes loading. A lane whose self-check already
/// passed under its current key (persisted from an earlier run, which is the
/// usual case after a restart) is ready at once: requests take the
/// passed-check fast path and `run_lane_check` never runs for this load. The
/// work `run_lane_check` starts after a passing check is started here
/// instead, once the load owner has released the lane's lifecycle lock. A
/// lane whose check has not passed gets that work from `run_lane_check`.
fn start_work_for_already_checked_lane(state: &Arc<ModuleState>, model: &Arc<EmbeddingModel>) {
    if !packed_lane_enabled(&state.runtime, model)
        && !direct_ane_prewarm_enabled(&state.runtime, model)
    {
        return;
    }
    let state = state.clone();
    let model = model.clone();
    tokio::spawn(async move {
        let guard = catalog_lane_lock(&state.runtime, &model.model_id)
            .lock_owned()
            .await;
        drop(guard);
        if profile_check_passed(&state, &model.model_id).is_ok_and(|m| m.is_some()) {
            start_ready_lane_work(&state.runtime, &model);
        }
    });
}

/// Once a direct-ANE catalog lane is ready (its self-check passed), its
/// common short shapes are loaded in the background at the lowest priority,
/// so that later requests at those lengths usually find them resident. The
/// supervisor loads them only while no request waits and only into free
/// budget, and ignores a repeated call for the same worker.
fn start_direct_ane_prewarm(runtime: &Arc<RuntimeState>, model: &EmbeddingModel) {
    #[cfg(unix)]
    if let EmbedBackend::DirectAne(engine) = &model.backend {
        if direct_ane_prewarm_enabled(runtime, model) {
            // Packed preparation that gave up while a prewarm admission held
            // the worker or its budget is retried when that admission ends;
            // the launcher does nothing for a lane without packing.
            let weak = Arc::downgrade(runtime);
            let model_id = model.model_id.clone();
            let notify: worker_host::ane_residency::PrewarmNotify = Arc::new(move || {
                if let Some(runtime) = weak.upgrade() {
                    runtime.activity_telemetry.prepare_when_idle(&model_id);
                }
            });
            if let Err(error) = engine.prewarm(&model.model_id, Some(notify)) {
                tracing::warn!(target: "worker", lane = %model.model_id, %error,
                    "direct-ANE prewarm could not start");
            }
        }
    }
    #[cfg(not(unix))]
    let _ = (runtime, model);
}

/// The numerical self-check's persisted id and key for a profile model.
fn profile_check_identity(
    state: &ModuleState,
    model: &EmbeddingModel,
) -> Result<(String, Value), WireOperationError> {
    let profile = model
        .engine_identity
        .build_flags
        .get("profile")
        .expect("profile preload");
    if let Some((entry, backend)) = resolved_catalog_lane(&state.runtime, &model.model_id) {
        catalog_self_check_key(state, entry, backend)
    } else {
        profile_preload_check_key(state, model, profile)
    }
}

// Catalog installs and startup preloads must compare model output against the
// same load-time fp32 reference inputs and expected outputs. These cases are
// copied from bench/parity, checked by SHA-256 and embedded in synapse-certify.
// Use the production execution path so validation exercises the same worker
// and hardware routing as serving requests, not a separate implementation.
async fn check_profile_model(
    state: &ModuleState,
    model: Arc<EmbeddingModel>,
    _lane: tokio::sync::OwnedMutexGuard<()>,
) -> Result<Arc<EmbeddingModel>, WireOperationError> {
    let profile = model
        .engine_identity
        .build_flags
        .get("profile")
        .expect("profile preload");
    let (id, key) = profile_check_identity(state, &model)?;
    let references = match synapse_certify::self_check::load(profile) {
        Ok(references) => references,
        Err(error) => {
            let generation = catalog_check_generation(state, &id, &key)?;
            catalog_complete_check(state, &id, generation, "failed", Some(&error.to_string()))?;
            return Err(profile_preload_check_error(profile, &error.to_string()));
        }
    };
    match profile_preload_check_status(state, &id)?.as_str() {
        "passed" => return Ok(model),
        "failed" => {
            return Err(profile_preload_check_error(
                profile,
                "persisted numerical failure",
            ))
        }
        _ => {}
    }
    let generation = catalog_check_generation(state, &id, &key)?;
    // Exclusive for the whole check: waits for in-flight inference to drain
    // and keeps new requests out until the reference cases have run.
    let gate = Arc::new(
        lane_serving_gate(&state.runtime, &model.model_id)
            .write_owned()
            .await,
    );
    let evaluation =
        match numerical_profile_preload_check(state, &model, profile, references, Some(gate)).await
        {
            Ok(evaluation) => evaluation,
            Err(error) => {
                let reason = serde_json::to_string(&error).expect("wire error serializes");
                catalog_complete_check(state, &id, generation, "failed", Some(&reason))?;
                return Err(profile_preload_check_error(profile, &reason));
            }
        };
    // A mock worker cannot reproduce the reference outputs, so tests that need
    // a passing numerical self-check still run every reference case and then
    // record the check as passed instead of evaluating the outputs.
    #[cfg(test)]
    if *state
        .runtime
        .profile_check_passes_for_test
        .lock()
        .expect("verdict override")
    {
        catalog_complete_check(state, &id, generation, "passed", None)?;
        return Ok(model);
    }
    complete_profile_preload_check(state, profile, &id, generation, &evaluation)?;
    Ok(model)
}

#[cfg(test)]
async fn resolve_serving_model(
    state: Arc<ModuleState>,
    requested: Option<&str>,
    task: ModelTask,
    required: Option<&str>,
    target: Option<&str>,
    deadline_ms: Option<u64>,
) -> Result<Arc<EmbeddingModel>, WireOperationError> {
    let deadline = tokio::time::Instant::now()
        + Duration::from_millis(deadline_ms.unwrap_or(state.runtime.inline.deadline_ms));
    resolve_serving_model_until(state, requested, task, required, target, deadline).await
}

async fn resolve_serving_model_until(
    state: Arc<ModuleState>,
    requested: Option<&str>,
    task: ModelTask,
    required: Option<&str>,
    target: Option<&str>,
    deadline: tokio::time::Instant,
) -> Result<Arc<EmbeddingModel>, WireOperationError> {
    if let Some(model_id) = requested {
        if model_slot_snapshot(&state.runtime, model_id).is_some_and(|slot| {
            !state.runtime.release_catalog.is_reserved_id(model_id)
                && slot
                    .spec
                    .engine_identity
                    .build_flags
                    .contains_key("profile")
        }) {
            return ensure_model_loaded_until(
                state,
                model_id,
                deadline,
                remaining_request_ms(deadline),
            )
            .await;
        }
    }
    if task == ModelTask::Generate
        || requested.is_some_and(|id| !state.runtime.release_catalog.is_reserved_id(id))
    {
        return resolve_model_for_request(state, requested, task).await;
    }
    let (entry, backend) = select_catalog_lane(&state, requested, task, required, target)?;
    let lane = catalog::lane_id(&entry.id, &backend.backend);
    if tokio::time::Instant::now() >= deadline {
        return Err(catalog_resolution_expired(&state.runtime, &lane, deadline));
    }
    let mut loading_capped = !catalog_lane_is_ready(&state.runtime, &lane)
        && !catalog_lane_check_running(&state.runtime, &lane);
    // Only unresolved loading work has the wire contract's five-second cap.
    // Both lifecycle queueing and the detached load consume this one cap, which
    // never extends the request deadline. Ready lanes and running numerical checks
    // use the request deadline instead.
    let mut resolution_deadline = if loading_capped {
        deadline.min(tokio::time::Instant::now() + Duration::from_secs(5))
    } else {
        deadline
    };
    tracing::debug!(model_id = %lane, stage = "resolver_budget", loading_capped,
        remaining_ms = remaining_request_ms(resolution_deadline), "catalog resolver");
    let started = std::time::Instant::now();
    let mut locked =
        catalog_resolution_guard_until(&state.runtime, &lane, resolution_deadline, deadline).await;
    if locked.is_err()
        && loading_capped
        && catalog_lane_is_ready(&state.runtime, &lane)
        && tokio::time::Instant::now() < deadline
    {
        locked = catalog_resolution_guard_until(&state.runtime, &lane, deadline, deadline).await;
    }
    let (guard, waited) = locked?;
    ane_lane_timing("resolver_lane_wait", &lane, None, started);
    tracing::debug!(model_id = %lane, stage = "lifecycle_acquired", waited,
        wall_ms = started.elapsed().as_secs_f64() * 1000.0, "catalog resolver");
    if let Some(error) = catalog_failed_load(&state.runtime, &lane, !waited) {
        return Err(error);
    }
    // The lifecycle lock may have been held by an unload after selection saw
    // Ready. Re-evaluate the load budget under that lock before starting work.
    let needs_load = !catalog_lane_is_ready(&state.runtime, &lane)
        && !catalog_lane_check_running(&state.runtime, &lane);
    if needs_load && !loading_capped {
        loading_capped = true;
        resolution_deadline = deadline.min(tokio::time::Instant::now() + Duration::from_secs(5));
    } else if !needs_load {
        loading_capped = false;
        resolution_deadline = deadline;
    }
    if tokio::time::Instant::now() >= deadline {
        return Err(catalog_resolution_expired(&state.runtime, &lane, deadline));
    }
    let task_state = state.clone();
    // Only the lock owner starts detached work. A timed-out waiter must not
    // stay queued and start a new load after the attempt it was waiting for fails.
    let owner = tracing::debug_span!("catalog_readiness_owner", model_id = %lane);
    tracing::debug!(model_id = %lane, stage = "readiness_spawn", "catalog resolver");
    let mut handle = tokio::spawn(
        async move {
            tracing::debug!(stage = "readiness_start", "catalog resolver");
            let result = ensure_catalog_lane_ready_owned(task_state, entry, backend, guard).await;
            tracing::debug!(
                stage = "readiness_finish",
                wall_ms = started.elapsed().as_secs_f64() * 1000.0,
                code = result.as_ref().err().map(|error| error.code.as_str()),
                "catalog resolver"
            );
            result
        }
        .instrument(owner),
    );
    let mut readiness = tokio::time::timeout_at(resolution_deadline, &mut handle).await;
    // A cold attempt can publish Ready before its bookkeeping task settles.
    // Once loading has ended, delayed bookkeeping must not masquerade as a load.
    if readiness.is_err()
        && loading_capped
        && catalog_lane_is_ready(&state.runtime, &lane)
        && tokio::time::Instant::now() < deadline
    {
        readiness = tokio::time::timeout_at(deadline, &mut handle).await;
    }
    match readiness {
        Ok(Ok(result)) => {
            if tokio::time::Instant::now() >= deadline {
                Err(catalog_resolution_expired(&state.runtime, &lane, deadline))
            } else {
                result
            }
        }
        Ok(Err(e)) => Err(WireOperationError::from_stable(
            StableError::engine_crashed(Some(250)),
            format!("catalog load task failed: {e}"),
        )),
        Err(_) => Err(catalog_resolution_expired(&state.runtime, &lane, deadline)),
    }
}

#[cfg(all(test, unix))]
async fn catalog_resolution_guard(
    runtime: &RuntimeState,
    lane: &str,
    budget: u64,
) -> Result<(tokio::sync::OwnedMutexGuard<()>, bool), WireOperationError> {
    let deadline = tokio::time::Instant::now() + Duration::from_millis(budget);
    catalog_resolution_guard_until(runtime, lane, deadline, deadline).await
}

/// A timed-out waiter leaves the lifecycle queue and cannot start another load.
/// The loading cap and the request deadline share this same queueing budget.
async fn catalog_resolution_guard_until(
    runtime: &RuntimeState,
    lane: &str,
    resolution_deadline: tokio::time::Instant,
    request_deadline: tokio::time::Instant,
) -> Result<(tokio::sync::OwnedMutexGuard<()>, bool), WireOperationError> {
    let lock = catalog_lane_lock(runtime, lane);
    tracing::debug!(model_id = %lane, stage = "lifecycle_wait", "catalog resolver");
    match lock.clone().try_lock_owned() {
        Ok(guard) => Ok((guard, false)),
        Err(_) => Ok((
            tokio::time::timeout_at(resolution_deadline, lock.lock_owned())
                .await
                .map_err(|_| catalog_resolution_expired(runtime, lane, request_deadline))?,
            true,
        )),
    }
}

fn remaining_request_ms(deadline: tokio::time::Instant) -> u64 {
    // A fractional millisecond is still usable budget. Round up only for
    // millisecond accounting; timers continue to use the absolute deadline.
    deadline
        .saturating_duration_since(tokio::time::Instant::now())
        .as_nanos()
        .div_ceil(1_000_000)
        .min(u64::MAX as u128) as u64
}

fn catalog_lane_is_ready(runtime: &RuntimeState, lane: &str) -> bool {
    model_slot_snapshot(runtime, lane)
        .is_some_and(|slot| matches!(slot.state, ModelRuntimeState::Ready) && slot.loaded.is_some())
}

fn catalog_lane_check_running(runtime: &RuntimeState, lane: &str) -> bool {
    runtime
        .lane_checks
        .lock()
        .expect("lane checks")
        .get(lane)
        .is_some_and(|outcome| outcome.borrow().is_none())
}

fn catalog_resolution_expired(
    runtime: &RuntimeState,
    lane: &str,
    deadline: tokio::time::Instant,
) -> WireOperationError {
    let slot = model_slot_snapshot(runtime, lane);
    if let Some(ModelSlotSnapshot {
        state: ModelRuntimeState::Failed(error),
        ..
    }) = &slot
    {
        tracing::debug!(model_id = %lane, stage = "resolver_expired", branch = "failed",
            code = %error.code, "catalog resolver");
        return error.clone();
    }
    let ready = slot.is_some_and(|slot| {
        matches!(slot.state, ModelRuntimeState::Ready) && slot.loaded.is_some()
    });
    let checking = catalog_lane_check_running(runtime, lane);
    let error = if !checking && (ready || tokio::time::Instant::now() >= deadline) {
        WireOperationError::from_stable(
            StableError::deadline_exceeded(),
            "catalog admission deadline expired",
        )
    } else if checking {
        WireOperationError::from_stable(
            StableError::model_loading(Some(250)),
            format!("model '{lane}' is running its numerical self-check"),
        )
    } else {
        WireOperationError::from_stable(
            StableError::model_loading(Some(250)),
            format!("catalog lane '{lane}' is loading"),
        )
    };
    tracing::debug!(model_id = %lane, stage = "resolver_expired", ready, checking,
        code = %error.code, "catalog resolver");
    error
}

fn catalog_failed_load(
    runtime: &RuntimeState,
    lane: &str,
    acknowledge: bool,
) -> Option<WireOperationError> {
    let mut catalog = runtime.catalog.lock().expect("runtime catalog");
    let slot = catalog.get_mut(lane)?;
    let ModelRuntimeState::Failed(error) = &slot.state else {
        return None;
    };
    let error = error.clone();
    // Waiters all observe the completed attempt's error. A fresh request
    // acknowledges it, so a later retry can start one new attempt.
    if acknowledge {
        if diagnostics_findings::cached_quarantine(&slot.state) {
            runtime.findings_changed.notify_one();
        }
        slot.state = ModelRuntimeState::Unloaded;
        slot.notify.notify_waiters();
    }
    Some(error)
}
struct CatalogInvocation {
    runtime: Arc<RuntimeState>,
    lane: String,
    _guard: tokio::sync::OwnedMutexGuard<()>,
    _gate: tokio::sync::OwnedRwLockWriteGuard<()>,
}
impl Drop for CatalogInvocation {
    fn drop(&mut self) {
        self.runtime
            .self_check_holders
            .lock()
            .expect("self-check holders")
            .remove(&self.lane);
    }
}
async fn ensure_catalog_lane_ready(
    state: Arc<ModuleState>,
    entry: catalog::CatalogEntry,
    backend: catalog::CatalogBackend,
    deadline: Option<tokio::time::Instant>,
) -> Result<Arc<EmbeddingModel>, WireOperationError> {
    let lane = catalog::lane_id(&entry.id, &backend.backend);
    let lock = catalog_lane_lock(&state.runtime, &lane);
    // A job whose budget has run out must leave the queue rather than load
    // the lane later, possibly after an operator has unloaded it.
    let (guard, waited) = match lock.clone().try_lock_owned() {
        Ok(guard) => (guard, false),
        Err(_) => match deadline {
            Some(deadline) => (
                tokio::time::timeout_at(deadline, lock.lock_owned())
                    .await
                    .map_err(|_| {
                        WireOperationError::from_stable(
                            StableError::deadline_exceeded(),
                            "catalog batch execution deadline expired",
                        )
                    })?,
                true,
            ),
            None => (lock.lock_owned().await, true),
        },
    };
    // The lock can be granted in the same poll as the expired timer, or be
    // free for a job that was already out of budget. Either way the job must
    // not go on to load the lane.
    if deadline.is_some_and(|deadline| tokio::time::Instant::now() >= deadline) {
        drop(guard);
        return Err(WireOperationError::from_stable(
            StableError::deadline_exceeded(),
            "catalog batch execution deadline expired",
        ));
    }
    if let Some(error) = catalog_failed_load(&state.runtime, &lane, !waited) {
        return Err(error);
    }
    let profile_backed = backend.profile.is_some();
    let model = ensure_catalog_lane_ready_owned(state.clone(), entry, backend, guard).await?;
    if profile_backed {
        // Never loads, and the job's own timer bounds this wait; a deadline
        // here could fail an admitted job with model_loading.
        ensure_profile_preload_ready(state, &lane, None).await
    } else {
        Ok(model)
    }
}

// Nanosecond ctime is essential: restoring mtime after an in-place write must
// not make corrupted bytes eligible for the persisted verification fast path.
fn catalog_file_stamp(path: &Path) -> Option<[String; 5]> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let metadata = std::fs::metadata(path).ok()?;
        if !metadata.is_file() {
            return None;
        }
        Some([
            metadata.dev().to_string(),
            metadata.ino().to_string(),
            metadata.size().to_string(),
            (i128::from(metadata.mtime()) * 1_000_000_000 + i128::from(metadata.mtime_nsec()))
                .to_string(),
            (i128::from(metadata.ctime()) * 1_000_000_000 + i128::from(metadata.ctime_nsec()))
                .to_string(),
        ])
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        None
    }
}

fn verified_catalog_file_digest(
    store: &SynapseStore,
    install: &store::CatalogInstallRecord,
    member_path: &str,
    expected: &str,
    path: &Path,
    hash: impl FnOnce(&Path) -> Option<String>,
) -> Result<Option<String>, WireOperationError> {
    let before = catalog_file_stamp(path);
    if let Some(stamp) = &before {
        if store
            .catalog_file_verified(install, member_path, expected, stamp)
            .map_err(catalog_store_error)?
        {
            return Ok(Some(expected.to_owned()));
        }
    }
    let digest = hash(path);
    let after = catalog_file_stamp(path);
    // A file modified while hashing cannot certify the bytes subsequently loaded.
    if before != after {
        return Ok(None);
    }
    if digest.as_deref() == Some(expected) {
        if let Some(stamp) = &after {
            store
                .record_catalog_file_verification(install, member_path, expected, stamp)
                .map_err(catalog_store_error)?;
        }
    }
    Ok(digest)
}

async fn ensure_catalog_lane_ready_owned(
    state: Arc<ModuleState>,
    entry: catalog::CatalogEntry,
    backend: catalog::CatalogBackend,
    guard: tokio::sync::OwnedMutexGuard<()>,
) -> Result<Arc<EmbeddingModel>, WireOperationError> {
    use store::CatalogBlobCache;
    let lane = catalog::lane_id(&entry.id, &backend.backend);
    #[cfg(test)]
    {
        let delay = state.runtime.resolver_ready_delay.lock().unwrap().take();
        if let Some(delay) = delay {
            tokio::time::sleep(delay).await;
        }
    }
    let key_started = Instant::now();
    if let Some(model) = model_slot_snapshot(&state.runtime, &lane)
        .filter(|slot| matches!(slot.state, ModelRuntimeState::Ready))
        .and_then(|slot| slot.loaded)
    {
        if validated_model_readiness(&state, &model)? {
            tracing::debug!(model_id = %lane, branch = "ready_passed", stage = "readiness_return",
                wall_ms = key_started.elapsed().as_secs_f64() * 1000.0, "catalog resolver");
            return Ok(model);
        }
    }
    tracing::debug!(model_id = %lane, branch = "consult_readiness", stage = "readiness_key",
        wall_ms = key_started.elapsed().as_secs_f64() * 1000.0, "catalog resolver");
    if model_slot_snapshot(&state.runtime, &lane).is_none() {
        sync_installed_catalog_slots(&state)?;
    }
    let (check_id, key) = catalog_self_check_key(&state, &entry, &backend)?;
    let projection_started = Instant::now();
    let projection = catalog_self_check_projection(&state, &entry, &backend)?;
    tracing::debug!(model_id = %lane, stage = "readiness_projection",
        wall_ms = projection_started.elapsed().as_secs_f64() * 1000.0, "catalog resolver");
    if projection["state"] == "failed" {
        return Err(catalog_self_check_failed(
            &entry,
            &backend,
            &check_id,
            &projection["reason"],
        ));
    }
    let model = if let Some(model) =
        model_slot_snapshot(&state.runtime, &lane).and_then(|s| s.loaded)
    {
        model
    } else {
        tracing::debug!(model_id = %lane, branch = "cold_load", stage = "validation_start", "catalog resolver");
        set_model_slot_state(&state.runtime, &lane, ModelRuntimeState::Validating);
        let verify_state = state.clone();
        let verify_entry = entry.clone();
        let verify_backend = backend.clone();
        let verify_lane = lane.clone();
        let verified = tokio::task::spawn_blocking(move || {
            let state = verify_state;
            let entry = verify_entry;
            let backend = verify_backend;
            let lane = verify_lane;
            let lock_started = std::time::Instant::now();
            let _disk = state.runtime.catalog_disk.lock().expect("catalog disk lock");
            catalog_validation_timing(&lane, "disk_lock", lock_started);
            let install = current_catalog_install(&state,&entry,&backend.backend)?.ok_or_else(|| catalog_wire_error("model_not_installed",json!({"catalog_id":entry.id,"lane_id":lane,"download_op":"models.download","download_job_id":null}),"catalog installation disappeared"))?;
            for (_, file) in entry.backend_named_files(&backend.backend) {
                let member = install
                    .members
                    .iter()
                    .find(|m| m.path == file.path)
                    .ok_or_else(|| catalog_artifact_error(file, None, None))?;
                let size = CatalogCache(&state.model_cache)
                    .blob_size(&member.digest)
                    .map_err(catalog_store_error)?;
                let path = state.model_cache.blob_path(&member.digest);
                let hash_started = std::time::Instant::now();
                let digest = verified_catalog_file_digest(
                    &state.store, &install, &file.path, &file.sha256, &path,
                    |path| sha256_file(path).ok(),
                )?;
                catalog_validation_timing(&lane, &file.path, hash_started);
                if size != Some(file.size_bytes)
                    || digest.as_deref() != Some(member.digest.as_str())
                    || member.digest != file.sha256
                {
                    let error = catalog_artifact_error(file, digest, size);
                    state
                        .store
                        .quarantine_catalog_blob(
                            &member.digest,
                            &serde_json::to_value(&error).expect("wire error"),
                            &CatalogCache(&state.model_cache),
                            now_ms(),
                        )
                        .map_err(catalog_store_error)?;
                    return Err(error);
                }
            }
            let parameters_started = std::time::Instant::now();
            let spec = catalog_lane_spec(&state, &entry, &backend, true)?;
            catalog_validation_timing(&lane, "parameters", parameters_started);
            validate_catalog_lane_fingerprint(&entry, &backend, &spec.fingerprint)?;
            if let Some(slot) = state
                .runtime
                .catalog
                .lock()
                .expect("runtime catalog")
                .get_mut(&lane)
            {
                slot.spec = spec;
                if diagnostics_findings::cached_quarantine(&slot.state) {
                    state.runtime.findings_changed.notify_one();
                }
                slot.state = ModelRuntimeState::Loading;
            }
            Ok(())
        }).await.unwrap_or_else(|error| Err(WireOperationError::from_stable(
            StableError::engine_crashed(Some(250)), format!("catalog verification task failed: {error}"),
        )));
        if let Err(error) = verified {
            set_model_slot_state(
                &state.runtime,
                &lane,
                ModelRuntimeState::Failed(error.clone()),
            );
            return Err(error);
        }
        let load_state = state.clone();
        let load_lane = lane.clone();
        // The lane guard belongs to this task, not the waiting request. A request
        // timeout cannot allow another load while a blocking engine load runs.
        let load_started = Instant::now();
        let loaded = load_catalog_model_task(load_state, load_lane).await?;
        tracing::debug!(model_id = %lane, stage = "load_finish",
            wall_ms = load_started.elapsed().as_secs_f64() * 1000.0, "catalog resolver");
        loaded
    };
    if backend.profile.is_some() {
        // Delay inference on the load-time reference cases until serving has
        // counted the full model input, including special tokens. A request over
        // 8192 tokens must be refused before compiling Neural Engine programs,
        // even the programs for a short reference case.
        return Ok(model);
    }
    if projection["state"] == "passed" {
        return Ok(model);
    }
    let generation = catalog_check_generation(&state, &check_id, &key)?;
    // No serving inference may overlap the check; see `lane_serving_gate`.
    let gate = lane_serving_gate(&state.runtime, &lane).write_owned().await;
    state
        .runtime
        .self_check_holders
        .lock()
        .expect("self-check holders")
        .insert(lane.clone(), check_id.clone());
    let invocation = CatalogInvocation {
        runtime: state.runtime.clone(),
        lane: lane.clone(),
        _guard: guard,
        _gate: gate,
    };
    let call_model = model.clone();
    let call_entry = entry.clone();
    let call_backend = backend.clone();
    let handle = tokio::task::spawn_blocking(move || {
        let _invocation = invocation;
        let result = owned_numerical_catalog_check(&call_model, &call_entry, &call_backend);
        (result, _invocation)
    });
    match tokio::time::timeout(Duration::from_millis(2000), handle).await {
        Ok(Ok((Ok((passed, observed, threshold)), _invocation))) => {
            let reason = if passed {
                None
            } else {
                Some("numerical_mismatch")
            };
            if !catalog_complete_check(
                &state,
                &check_id,
                generation,
                if passed { "passed" } else { "failed" },
                reason,
            )? {
                return Err(WireOperationError::from_stable(
                    StableError::engine_crashed(Some(250)),
                    "self-check generation changed",
                ));
            }
            if passed {
                Ok(model)
            } else {
                tracing::warn!(model_id=%lane,backend=%backend.backend,%check_id,observed,threshold,key=%key,"catalog numerical self-check failed");
                Err(catalog_self_check_failed(
                    &entry,
                    &backend,
                    &check_id,
                    &json!("numerical_mismatch"),
                ))
            }
        }
        other => {
            catalog_complete_check(&state, &check_id, generation, "pending", None)?;
            let message = match other {
                Err(_) => "catalog self-check exceeded 2000 ms".to_string(),
                Ok(Err(e)) => e.to_string(),
                Ok(Ok((Err(e), _invocation))) => e.message,
                _ => unreachable!(),
            };
            Err(WireOperationError::from_stable(
                StableError::engine_crashed(Some(250)),
                message,
            ))
        }
    }
}

fn startup_catalog_runtime(state: &ModuleState) -> Result<(), ModuleError> {
    sweep_owned_package_orphans(state.model_cache.root())
        .map_err(|error| ModuleError::Config(format!("sweep removed owned packages: {error}")))?;
    state.store.sqlite().with_conn_fenced(|tx| {
        tx.execute("UPDATE catalog_self_checks SET state='pending',checked_at_ms=NULL,reason=NULL WHERE state='running'",[])?;
        Ok(())
    }).map_err(|e| ModuleError::Config(e.to_string()))?;
    let error = json!({"code":"module_restarted","class":"transient","retry_after_ms":250,"safe_to_retry_same_request":true,"message":"module restarted during download"});
    let failed = state.store.fail_interrupted_download_jobs(
        state.module_generation,
        &error,
        &CatalogCache(&state.model_cache),
        now_ms(),
    )?;
    state
        .runtime
        .admission_telemetry
        .record_jobs_inherited(failed.len() as u64);
    state
        .runtime
        .admission_telemetry
        .record_jobs_failed(failed.len() as u64);
    sync_installed_catalog_slots(state).map_err(|e| ModuleError::Config(e.message))
}
fn validate_catalog_lane_fingerprint(
    entry: &catalog::CatalogEntry,
    backend: &catalog::CatalogBackend,
    fingerprint: &Fingerprint,
) -> Result<(), WireOperationError> {
    if fingerprint.0 != backend.fingerprint {
        return Err(catalog_wire_error(
            "artifact_invalid",
            json!({"model_id":catalog::lane_id(&entry.id,&backend.backend),"expected_fingerprint":backend.fingerprint,"actual_fingerprint":fingerprint}),
            "catalog fingerprint differs from verified artifact parameters",
        ));
    }
    Ok(())
}

fn catalog_transcribe_lane(
    entry: &catalog::CatalogEntry,
    backend: &catalog::CatalogBackend,
) -> Result<transcribe_self_check::Lane, WireOperationError> {
    let profile = backend.profile.as_deref().expect("transcribe profile");
    let config = CatalogProfile::load(profile)
        .and_then(|p| p.transcribe_config())
        .map_err(|error| artifact_invalid_error(error.to_string()))?;
    Ok(transcribe_self_check::Lane {
        catalog_id: entry.id.clone(),
        backend: backend.backend.clone(),
        profile: profile.into(),
        fingerprint: backend.fingerprint.clone(),
        manifest_digest: entry.manifest_digest(),
        engine_identity: serde_json::to_value(config.identity).expect("identity"),
    })
}

fn catalog_list_row(state: &ModuleState, row: &mut Value) -> Result<(), WireOperationError> {
    let id = row["model_id"].as_str().unwrap_or("").to_string();
    if let Some((entry, backend)) = resolved_catalog_lane(&state.runtime, &id) {
        if entry.task == "transcribe" {
            let object = row.as_object_mut().expect("catalog row");
            for field in [
                "max_tokens",
                "max_tokens_source",
                "recommended_batch",
                "bucket_ladder",
                "dims",
                "pooling",
                "normalize",
                "attention_units",
            ] {
                object.remove(field);
            }
            row["max_duration_ms"] = json!(backend
                .effective_max_duration_ms()
                .expect("transcribe duration"));
        }
        let check = catalog_self_check_projection(state, entry, backend)?;
        row["certified"] = json!(check["state"] == "passed");
        row["serving_admission"] = json!(if check["state"] == "failed" {
            "disabled"
        } else {
            "enabled"
        });
        if check["state"] == "failed" {
            row["serving_admission_reason"] = json!("self_check_failed");
        } else {
            row.as_object_mut()
                .expect("catalog row")
                .remove("serving_admission_reason");
        }
        row["self_check"] = check;
        row["fingerprints"] = json!([backend.fingerprint]);
    } else if model_slot_snapshot(&state.runtime, &id)
        .is_some_and(|s| matches!(s.spec.task.as_str(), "embed" | "rerank"))
    {
        row["self_check"] = json!({"state":"not_applicable","checked_at_ms":null,"reason":null});
    }
    Ok(())
}

fn catalog_engine_fault(id: &str, call: &str) -> Result<(), EngineError> {
    catalog_call_fault(id, call).map_err(|error| EngineError {
        stage: EngineErrorStage::Inference,
        risk_class: synapse_core::EngineRiskClass::AbortCapable,
        message: error.message,
        retry_after_ms: Some(250),
        safe_to_retry_same_request: true,
    })
}

fn catalog_cache_roots(state: &ModuleState) -> Result<Vec<String>, SynapseStoreError> {
    let mut roots = state.store.sqlite().with_conn(|conn| {
        let mut query = conn.prepare("SELECT digest FROM catalog_install_members UNION SELECT digest FROM download_acquisitions")?;
        let roots = query.query_map([],|r| r.get::<_,String>(0))?.collect::<Result<Vec<_>,_>>()?;
        Ok(roots)
    })?;
    for model in state.store.catalog_models()? {
        for locator in std::iter::once(&model.model_locator)
            .chain(std::iter::once(&model.tokenizer_locator))
            .chain(model.config_locator.iter())
            .chain(model.extra_locators.iter())
        {
            if let ModelAssetLocator::CacheDigest { digest } = locator {
                roots.push(digest.clone());
            }
        }
    }
    roots.sort();
    roots.dedup();
    Ok(roots)
}

/// Runs an admitted catalog embed job: readies its lane within the job's
/// execution budget, executes it, and always finalizes the job record and its
/// `catalog_jobs` registration, whatever the outcome.
async fn run_catalog_embed_job(
    state: Arc<ModuleState>,
    record: store::JobRecord,
    entry: catalog::CatalogEntry,
    backend: catalog::CatalogBackend,
    params: Value,
    digest: String,
) {
    let job = record.job_id.clone();
    let prepared = async {
        let _pause_guard = lane_pause::admit(
            &state.runtime,
            &catalog::lane_id(&entry.id, &backend.backend),
        )?;
        let budget = record
            .execution_expires_ms
            .unwrap_or(now_ms())
            .saturating_sub(now_ms());
        if budget == 0 {
            return Err(WireOperationError::from_stable(
                StableError::deadline_exceeded(),
                "catalog batch execution deadline expired",
            ));
        }
        let parsed: EmbedBatchParams =
            serde_json::from_value(params).map_err(|e| artifact_invalid_error(e.to_string()))?;
        let items = batch_items(parsed.items, parsed.texts).map_err(artifact_invalid_error)?;
        let texts = items
            .iter()
            .map(|item| item.text.as_str())
            .collect::<Vec<_>>();
        let lane = catalog::lane_id(&entry.id, &backend.backend);
        let sequences = preflight_catalog_sequences(
            &state,
            Some(&lane),
            parsed.required_fingerprint.as_deref(),
            parsed.target_fingerprint.as_deref(),
            None,
            &texts,
        )?
        .expect("catalog job lane");
        let task_state = state.clone();
        let deadline = tokio::time::Instant::now() + Duration::from_millis(budget);
        let loading = tokio::spawn(async move {
            ensure_catalog_lane_ready(task_state, entry, backend, Some(deadline)).await
        });
        let model = tokio::time::timeout(Duration::from_millis(budget), loading)
            .await
            .map_err(|_| {
                WireOperationError::from_stable(
                    StableError::deadline_exceeded(),
                    "catalog batch execution deadline expired",
                )
            })?
            .map_err(|e| {
                WireOperationError::from_stable(
                    StableError::engine_crashed(Some(250)),
                    e.to_string(),
                )
            })??;
        let request_bytes = request_bytes_for_texts(texts.iter().copied());
        let mut tokenized = catalog_embed_batch(&texts, sequences);
        compose_catalog_embed(&model, &mut tokenized)?;
        apply_owned_tokenizer_policy(&model, &mut tokenized);
        let total_tokens = tokenized
            .real_token_counts
            .iter()
            .map(|n| u64::from(*n))
            .sum();
        check_fingerprint_constraints(
            &model,
            parsed.target_fingerprint.as_deref(),
            parsed.required_fingerprint.as_deref(),
            parsed.required_epoch,
        )?;
        Ok::<_, WireOperationError>(EmbedBatchJobWork {
            model,
            request_digest: digest,
            ids: items.into_iter().map(|i| i.id).collect(),
            tokenized,
            request_bytes,
            total_tokens,
        })
    }
    .await;
    match prepared {
        Ok(work) => execute_embed_batch_job(state.clone(), job.clone(), work).await,
        Err(e) => fail_job_with_wire_error(&state, &job, e.class == ErrorClass::Transient, e),
    }
    state
        .runtime
        .catalog_jobs
        .lock()
        .expect("catalog jobs")
        .remove(&job);
}

async fn submit_catalog_embed_job(
    state: Arc<ModuleState>,
    entry: catalog::CatalogEntry,
    backend: catalog::CatalogBackend,
    params: Value,
) -> HandlerOutcome {
    let Some(key) = params["request_key"]
        .as_str()
        .filter(|s| !s.trim().is_empty())
    else {
        return channel_error(
            "invalid_request",
            "job-shaped embed.batch requires a non-empty request_key",
        );
    };
    let digest = compute_request_digest(
        "embed.batch",
        &catalog::lane_id(&entry.id, &backend.backend),
        &params,
        &[],
    );
    let admission = {
        let _disk = state
            .runtime
            .catalog_disk
            .lock()
            .expect("catalog disk lock");
        if let Err(e) = select_catalog_lane(
            &state,
            Some(&catalog::lane_id(&entry.id, &backend.backend)),
            ModelTask::Embed,
            Some(&backend.fingerprint),
            None,
        ) {
            return result_outcome(error_payload(&state, e));
        }
        match state.store.admit_job(
            key,
            &digest,
            "embed.batch",
            state.module_generation,
            &params,
            now_ms(),
            state.runtime.jobs.execution_ttl_ms,
            state.runtime.jobs.result_retention_ttl_ms,
        ) {
            Ok(admission) => {
                if matches!(admission, JobAdmission::Admitted(_)) {
                    state
                        .runtime
                        .catalog_jobs
                        .lock()
                        .expect("catalog jobs")
                        .insert(admission.record().job_id.clone(), entry.id.clone());
                }
                admission
            }
            Err(e @ SynapseStoreError::IdempotencyConflict { .. }) => {
                return result_outcome(error_payload(
                    &state,
                    WireOperationError::from_stable(
                        StableError::idempotency_conflict(),
                        e.to_string(),
                    ),
                ))
            }
            Err(e) => return channel_error("store_failure", e.to_string()),
        }
    };
    let record = admission.record().clone();
    let response = job_status_payload(&state, &record);
    if matches!(admission, JobAdmission::Admitted(_)) {
        state.runtime.admission_telemetry.record_job_minted();
        tokio::spawn(run_catalog_embed_job(
            state, record, entry, backend, params, digest,
        ));
    }
    result_outcome(response)
}
