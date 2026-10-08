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

mod ane_artifact;
mod catalog;
// Provider adapters stay module-private so credentials and remote identity checks
// cannot be bypassed by a second public call path.
/// Certification probes, immutable fixture batteries and oracles,
/// scheduler-evidence ingestion, the approval-backed serving predicate, and
/// validation of all twelve owned-metal-decode release requirements. The source
/// lives under
/// `crates/synapse-module/owned-decode-certification/`; the `#[path]`
/// attribute wires that directory into the crate as a module.
#[path = "../owned-decode-certification/mod.rs"]
pub mod owned_decode_certification;
/// Module-owned schemas and checked-in records for the production owned-decode
/// lane. Loaded by catalog validation and CI probes. See
/// `owned_decode_contracts::load_manifest_dir`.
pub mod owned_decode_contracts;
/// Grammar compilation and the dedicated DECODE scheduler for the owned-decode
/// lane: JSON-schema-subset parsing and validation, checked-in grammar limits,
/// the byte-level constrained automaton, the `token-id-json-constraint-v1`
/// representation, and the `QueueClass::Decode` scheduler with quantum
/// sequencing. The source lives under
/// `crates/synapse-module/owned-decode-grammar-scheduler/`; the `#[path]`
/// attribute wires that directory into the crate as a module.
#[path = "../owned-decode-grammar-scheduler/mod.rs"]
pub mod owned_decode_grammar_scheduler;
/// Module-side request processing and lane routing for the owned-metal-decode
/// lane: catalog validation, family registration, identity computation, Q8
/// ingest orchestration, certification access, approval-backed lane selection
/// and fallback,
/// provenance, and end-to-end `microllm.oneshot` orchestration. The source lives
/// under `crates/synapse-module/owned-decode-routing/`; the `#[path]` attribute
/// wires that directory into the crate as a module.
#[path = "../owned-decode-routing/mod.rs"]
pub mod owned_decode_routing;
/// Runtime-owned sidecar normalization and bounded target-token hint banking.
/// Sidecar completions never receive authority to commit target decode state.
#[path = "../owned-decode-sidecar/mod.rs"]
pub mod owned_decode_sidecar;
#[allow(dead_code)]
mod remote;
mod rollback;
mod store;
pub use store::{
    maintenance_vacuum_count, RestoreImportReport, SynapseStore, SynapseStoreError,
    RECLAIM_FREELIST_MIN_BYTES, RECLAIM_FREELIST_PAGE_RATIO_DIVISOR,
};
pub mod worker_host;

use cortexkit_lease::{FileLeaseStore, LeaseHandle, LeaseKey, LeaseStore};
use cortexkit_store_types::{sqlite_store_path, Isolation, StorageBackend, StorageDescriptor};
use owned_decode_certification::probe::ProbeSnapshot;
use remote::{
    config::{validate_remote_providers, ConfiguredProvider, RemoteProviderConfig, RemoteTask},
    gateway::{RemoteEmbedVector, RemoteGateway, RemoteGatewayError},
    runtime::RemoteClass,
    vault::SubcVaultCredentialClient,
    ContinuityCheck,
};
use reqwest::Url;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use store::{
    ApprovalCertificationHealth, AssuranceClass, CatalogSnapshot, CertificationClass,
    CertificationKey, CertificationRow, CertificationStatus, CheckpointItem,
    ClassScopedCertificationRow, EvidenceRequirementsDivergence, JobAdmission, JobAttemptClaim,
    JobRecord, KnobAssignmentRow, ModelAssetLocator, ModelCatalogEntry,
    OwnedDecodeAdmissionEvaluation, OwnedDecodeAdmissionRefusal, OwnedDecodeCertificationRow,
    OwnedDecodeMatchInputs, PerfRow, ProbeWriteOutcome, RecommendedBatch, StorageHealthInputs,
    StoredModelConfig, CERT_EVIDENCE_SCHEMA_REVISION, JOB_STATE_DONE, JOB_STATE_FAILED_PERMANENT,
    JOB_STATE_FAILED_TRANSIENT, JOB_STATE_PAUSED_NEEDS_REAUTH, JOB_STATE_QUEUED, JOB_STATE_RUNNING,
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
use synapse_core::{
    evaluate_cuda_floor, owned_cuda_engine_identity, worker_binary_env_var,
    worker_binary_file_name,
    worker_engine_names::{
        ANE_WORKER_ENGINE, DECODE_WORKER_ENGINE, LLAMA_ENGINE, LLAMA_WORKER_ENGINE,
    },
    worker_runtime_dir_env_var, AdmissionDecision, AdmissionRequest, AliasTable, CacheGcOutcome,
    CertifiedShapeEnvelope, Clock, CudaFloorDecision, EmbedEngine, EngineError, EngineErrorStage,
    EngineIdentity, ErrorClass, Fingerprint, FlashAttentionSetting, GenerateEngine, GenerateOutput,
    GenerateRequest, LaneBudgetSnapshot, LaneScheduler, LoadedModel, MachineProfile, ModelCache,
    ModelCacheError, ModelCacheIngest, ModelCacheMeta, NormalizationMode, NumericDType,
    NumericProfile, NumericProfileId, PoolingStrategy, QueueClass, RerankRequest, ResponseEnvelope,
    ResponseProvenance, RuntimeConfig, SanitizedTokenizer, SchedulerConfig, SidecarSpec,
    StableError, SystemMachineProfileCollector, ThreadPolicyClass, TokenBatch, TokenizationError,
    TokenizedBatch, TokenizerConfig, TruncationDisclosure, ValidatedArtifact, Vectors, WorkRequest,
    WorkerPooling, CUDA_WORKER_ENGINE, MACHINE_PROFILE_HASH_REVISION, OWNED_CUDA_MINIMUM_DEVICE_CC,
    OWNED_CUDA_MINIMUM_DRIVER_API, OWNED_CUDA_PTX_VIRTUAL_ARCH,
};
use synapse_engine_owned::{
    engine_identity as owned_engine_identity, ModelFamily as OwnedFamily, OwnedDType,
    OwnedMetalEmbedEngine, TokenizerPolicy as OwnedTokenizerPolicy,
    DEFAULT_ATTENTION_UNITS as OWNED_DEFAULT_ATTENTION_UNITS,
};
use thiserror::Error;
use tokio::sync::{Notify, Semaphore};

impl owned_decode_routing::lane::AdmissionBoundaryReader for SynapseStore {
    fn admission_boundary_matches(
        &self,
        snapshot: &owned_decode_routing::lane::AdmissionBoundarySnapshot,
    ) -> Result<bool, String> {
        self.owned_decode_dispatch_admission_matches(
            snapshot.profile_activation_epoch,
            &snapshot.model_id,
            &snapshot.decode_fingerprint,
            &snapshot.approval_semantic_digest,
            snapshot.approval_generation,
        )
        .map_err(|error| error.to_string())
    }
}

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
const OWNED_DECODE_PROBE_TIMEOUT_MS: u64 = 900_000;
const DEFAULT_TRANSIENT_RETRY_AFTER_MS: u64 = 100;
const DEFAULT_JOB_EXECUTION_TTL_MS: u64 = 24 * 60 * 60 * 1_000;
const DEFAULT_JOB_RESULT_RETENTION_TTL_MS: u64 = 24 * 60 * 60 * 1_000;
const DEFAULT_RESUME_DEADLINE_MS: u64 = 24 * 60 * 60 * 1_000;
const DEFAULT_JOB_RESULT_PAGE_BYTES: usize = 512 * 1024;
const DEFAULT_JOB_BULK_QUANTUM_TOKENS: u64 = 3_072;
const DEFAULT_ENGINE_BATCH_TOKEN_BUDGET: u64 = 3_072;
const MAX_ENGINE_BATCH_ITEMS: usize = 8;
const DEFAULT_PROBE_MEAN_COSINE_THRESHOLD: f64 = 0.999;
const DEFAULT_PROBE_WORST_DECILE_RANK_OVERLAP_THRESHOLD: f64 = 0.9;
const DEFAULT_PROBE_ANE_PLACEMENT_THRESHOLD: f64 = 0.9;
const DEFAULT_DECODE_CHAIN_K: u32 = 1;
const MAX_DECODE_CHAIN_K: u32 = 16;
const RERANK_PROBE_PEARSON_THRESHOLD: f64 = 0.999;
const BALANCED_QUIET_MIN_THROUGHPUT_RATIO: f64 = 0.5;
const PROBE_PERF_BATCH_TOKEN_BUDGET: usize = 1_024;
const PROBE_PERF_TARGET_TOTAL_TOKENS: u64 = 4_096;
const PROBE_PERF_MIN_BATCH_SAMPLES: usize = 3;
const PROBE_PERF_SINGLE_SAMPLES: usize = 20;
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
/// Restored approvals are imported disabled and require operator re-enable
/// before serving resumes.
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
    let _singleton = acquire_synapse_singleton_lease(&module_id)?;
    let connection_file = subc_connection_file_from_args()?;
    let handler = SynapseHandler::new(module_id.clone(), connection_file);
    // `serve` takes the handler, so keep a handle on the one slot the stop line
    // needs: how the daemon connection ended, which the SDK reports to the
    // handler just before `serve` returns.
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
    let outcome = subc_client_rs::serve(manifest(&module_id), handler).await;
    let uptime_s = started.elapsed().as_secs();
    match &outcome {
        // `reason` tells a planned stop (`goodbye`) from a daemon that vanished
        // (`eof`, `reset`) and from this module closing the connection itself
        // (`closed`), which the bare stop line could not. On the error arm the
        // SDK reports no reason; the error itself says why.
        Ok(()) => tracing::info!(
            target: "lifecycle",
            module = %module_id,
            uptime_s,
            reason = connection_end_reason(connection_end.get().copied()),
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
    if let Ok(root) = env::var("CORTEXKIT_LEASE_ROOT") {
        return Ok(PathBuf::from(root));
    }
    let home = env::var_os("HOME").ok_or_else(|| {
        ModuleError::Config("HOME is unset; cannot resolve cortexkit lease root".to_string())
    })?;
    Ok(PathBuf::from(home)
        .join(".local")
        .join("share")
        .join("cortexkit")
        .join("leases"))
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
    connection_file: PathBuf,
    state: OnceLock<Arc<ModuleState>>,
    approval_operators: Mutex<HashMap<RouteHandle, String>>,
    /// The `flow_id` from each bound route's scope, for routes the daemon
    /// opened on behalf of a flow (an automation acting for its owner).
    /// Recorded at bind; `flow_refusal` uses it to refuse methods on such
    /// routes.
    flow_routes: Mutex<HashMap<RouteHandle, String>>,
    /// How the daemon connection ended, set once by the SDK's end-of-connection
    /// callback and read by the `synapse stopped` log line.
    connection_end: Arc<OnceLock<ConnectionEnd>>,
}

impl SynapseHandlerInner {
    /// Remembers whether a bound route's scope belongs to a flow. The stamp is
    /// fixed for the route's life, so recording it once at bind is enough.
    fn record_bind_scope(&self, req: &RouteBindRequest) {
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
    /// route synapse serves only the operations it declares as queries, which
    /// give every caller the same answer. Every other method (the mutations
    /// that change what the whole machine serves, and any name not declared at
    /// all) is refused by name. Without this, a flow would load, remove or
    /// approve models as its owner.
    fn flow_refusal(&self, route: &RouteHandle, method: &str) -> Option<HandlerOutcome> {
        let flow_id = match self.flow_routes.lock() {
            Ok(flows) => flows.get(route).cloned()?,
            // A poisoned map can't say whether this route is a flow's, so
            // refuse rather than serve a mutation unchecked.
            Err(_) => "unknown".to_string(),
        };
        let declared_query = management_operations()
            .iter()
            .any(|op| op.name == method && op.kind == ManagementOperationKind::Query);
        if declared_query {
            return None;
        }
        Some(channel_error(
            "flow_scope_refused",
            format!(
                "{method} is not available on a flow-scoped route (flow {flow_id}): \
                 flows may call only synapse's query operations"
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
    /// Existing profile hash used by legacy certification and performance rows.
    machine_profile_hash: String,
    /// Legacy MachineProfile::hash() retained for the existing health field.
    legacy_machine_profile_hash: String,
    /// Revisioned hash used by owned-decode admission and evidence identity.
    revisioned_machine_profile_hash: String,
    profile_activation_epoch: u64,
    runtime: Arc<RuntimeState>,
    model_cache: Arc<ModelCache>,
    continuity_check: Arc<dyn ContinuityCheck>,
    remote_gateway: Arc<RemoteGateway>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct ModuleHealth {
    status: String,
    module_generation: u64,
    loaded_models: usize,
    machine_profile_hash: String,
    certification: CertificationHealth,
    certification_stale: bool,
    performance_stale: bool,
    lanes: Vec<LaneHealth>,
    previous_revisioned_machine_profile_hash: Option<String>,
    current_revisioned_machine_profile_hash: String,
    profile_activation_epoch: u64,
    last_rotation_at_ms: Option<u64>,
    last_rotation_reason: Option<String>,
    re_certification_state: String,
    evidence_requirements_divergence: Vec<EvidenceRequirementsDivergence>,
    approval_certification_outcomes: Vec<ApprovalCertificationHealth>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct CertificationHealth {
    certification_stale: bool,
    stale_since_ms: Option<u64>,
    lanes: Vec<CertificationHealthLane>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct CertificationHealthLane {
    model_id: String,
    workload: String,
    certified: bool,
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

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ApprovalMigrationParams {
    seed_revision: String,
    schema_revision: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ApprovalEnableParams {
    model_id: String,
    decode_fingerprint: String,
    grammar_enabled: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ApprovalDisableParams {
    model_id: String,
    decode_fingerprint: String,
    reason: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ApprovalEmergencyRollbackParams {
    reason: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OwnedDecodeSessionAdmissionParams {
    catalog_fingerprint: String,
    caller_id: String,
    context_ceiling_tokens: u32,
    generation: synapse_core::GenerationConfiguration,
    kv_configuration: owned_decode_routing::admission::SessionKvConfiguration,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OwnedDecodeSessionDecodeParams {
    session_id: String,
    req_id: String,
    prompt: String,
    max_tokens: u32,
    #[serde(default)]
    grammar: Option<String>,
    #[serde(default)]
    deadline_ms: Option<u64>,
    #[serde(default)]
    max_queue_ms: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OwnedDecodeSessionSnapshotParams {
    session_id: String,
    position_tokens: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OwnedDecodeSessionContinueParams {
    session_id: String,
    retained_kv_session_id: String,
    /// Present when an aborted stream left a retained token prefix; `req_id`
    /// lets the supervisor continue from that prefix without replaying or
    /// skipping committed tokens.
    #[serde(default)]
    req_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OwnedDecodeSessionAbortParams {
    session_id: String,
    req_id: String,
    retain_kv: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OwnedDecodeSessionCloseParams {
    session_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OwnedDecodeSessionStatusParams {
    session_id: String,
    req_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OwnedDecodeServingControlParams {
    catalog_fingerprint: String,
    reason: String,
}

#[derive(Clone, Debug)]
struct OwnedDecodeWireSession {
    catalog_fingerprint: String,
    model_id: String,
    routing_session_id: owned_decode_routing::admission::SessionId,
    kv_configuration: owned_decode_routing::admission::SessionKvConfiguration,
    active_request: Option<String>,
    retained_kv_session_id: Option<String>,
    retained_position: Option<u32>,
    /// Wall-clock time (`now_ms`) at which close or an artifact revoke ended the
    /// session; `None` while it is open. Background maintenance evicts the entry
    /// once it has been closed for longer than
    /// [`CLOSED_OWNED_DECODE_SESSION_RETENTION_MS`].
    closed_at_ms: Option<u64>,
}

impl OwnedDecodeWireSession {
    fn is_closed(&self) -> bool {
        self.closed_at_ms.is_some()
    }
}

/// How long a closed decode session stays registered before background
/// maintenance forgets it. Keeping it for a while is what lets a repeated
/// `owned_decode.close` stay idempotent and lets `owned_decode.session_status`
/// report the terminal state to a client that lost the terminal frame. Both of
/// those recoveries happen within seconds of the close, so 15 minutes is a
/// generous margin while still stopping the session map from growing by one
/// entry per admitted session for the life of the module.
const CLOSED_OWNED_DECODE_SESSION_RETENTION_MS: u64 = 15 * 60 * 1_000;

#[derive(Clone, Copy, Debug)]
struct PendingSessionAbort {
    retain_kv: bool,
}

struct OwnedDecodeWireState {
    sessions: BTreeMap<String, OwnedDecodeWireSession>,
    residency: Option<owned_decode_routing::admission::ResidencyRouter>,
    streams: owned_decode_worker::StreamingSupervisor,
    scheduler: owned_decode_grammar_scheduler::scheduler::DecodeScheduler,
    pending_aborts: BTreeMap<(String, String), PendingSessionAbort>,
    next_session_sequence: u64,
}

impl Default for OwnedDecodeWireState {
    fn default() -> Self {
        Self {
            sessions: BTreeMap::new(),
            residency: None,
            streams: owned_decode_worker::StreamingSupervisor::default(),
            scheduler:
                owned_decode_grammar_scheduler::scheduler::DecodeScheduler::shipped_embed_load_v1(),
            pending_aborts: BTreeMap::new(),
            next_session_sequence: 0,
        }
    }
}

impl OwnedDecodeWireState {
    /// Forget sessions that have been closed for longer than the retention
    /// window, together with their request stream records, and return how many
    /// sessions were removed. Open sessions are never removed, however long ago
    /// they were admitted.
    fn evict_expired_closed_sessions(&mut self, now_ms: u64) -> usize {
        let expired = self
            .sessions
            .iter()
            .filter(|(_, session)| {
                session.closed_at_ms.is_some_and(|closed_at_ms| {
                    now_ms.saturating_sub(closed_at_ms) > CLOSED_OWNED_DECODE_SESSION_RETENTION_MS
                })
            })
            .map(|(session_id, _)| session_id.clone())
            .collect::<Vec<_>>();
        for session_id in &expired {
            self.sessions.remove(session_id);
            // Status looks the session up first, so once it is gone its stream
            // records are unreachable and would only accumulate.
            self.streams.forget_session(session_id);
        }
        expired.len()
    }
}

struct ServingAdmissionMaterial {
    machine: owned_decode_routing::admission::MachineTuple,
    envelope: owned_decode_routing::admission::PlatformEnvelope,
    artifact: owned_decode_routing::admission::ArtifactReservation,
    model_id: String,
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
    log: LogConfig,
    #[serde(default)]
    jobs: JobConfig,
    #[serde(default)]
    probe: ProbeConfig,
    #[serde(default)]
    knob: PerfKnob,
    #[serde(default, alias = "dev_alias_admin", alias = "enable_alias_admin")]
    alias_admin_enabled: bool,
    #[serde(default = "default_microllm_max_tokens")]
    microllm_max_tokens: u32,
    #[serde(default)]
    grammar_enabled: bool,
    /// Dormant semantic sidecar configuration. Launch still requires a compiled,
    /// identity-matching `synapse-json-schema-v1` constraint.
    #[serde(default)]
    sidecar_spec: SidecarSpec,
    /// Free-text owned decode chain span. Grammar requests always use K=1
    /// because host-side token masking requires a per-token boundary; values
    /// above one change free-text execution shape only after certification
    /// covers the configured span.
    #[serde(default = "default_decode_chain_k")]
    decode_chain_k: u32,
    #[serde(default = "default_cache_max_bytes")]
    cache_max_bytes: u64,
    #[serde(default)]
    dev: DevConfig,
    #[serde(default)]
    remote_providers: Vec<RemoteProviderConfig>,
}

impl Default for ModuleConfig {
    fn default() -> Self {
        Self {
            certify_observation: false,
            hf_endpoint: default_hf_endpoint(),
            preload_models: Vec::new(),
            inline: InlineConfig::default(),
            worker: WorkerConfig::default(),
            log: LogConfig::default(),
            jobs: JobConfig::default(),
            probe: ProbeConfig::default(),
            knob: PerfKnob::default(),
            alias_admin_enabled: false,
            microllm_max_tokens: default_microllm_max_tokens(),
            grammar_enabled: false,
            sidecar_spec: SidecarSpec::default(),
            decode_chain_k: default_decode_chain_k(),
            cache_max_bytes: default_cache_max_bytes(),
            dev: DevConfig::default(),
            remote_providers: Vec::new(),
        }
    }
}

fn embedding_profile_enabled() -> bool {
    env::var_os(SYNAPSE_EMBED_PROFILE_ENV).is_some_and(|value| value.to_string_lossy() != "0")
}

fn default_microllm_max_tokens() -> u32 {
    DEFAULT_MICROLLM_MAX_TOKENS
}

fn default_cache_max_bytes() -> u64 {
    DEFAULT_CACHE_MAX_BYTES
}

fn default_decode_chain_k() -> u32 {
    DEFAULT_DECODE_CHAIN_K
}

fn validate_decode_chain_k(value: u32) -> Result<(), ModuleError> {
    if (1..=MAX_DECODE_CHAIN_K).contains(&value) {
        Ok(())
    } else {
        Err(ModuleError::Config(format!(
            "decode_chain_k must be between 1 and {MAX_DECODE_CHAIN_K}, got {value}"
        )))
    }
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
    arithmetic_identity_revision: Option<String>,
    #[serde(default)]
    metallib_revision: Option<String>,
    #[serde(default)]
    quantizer_revision: Option<String>,
    #[serde(default)]
    kernel_revision: Option<String>,
    #[serde(default)]
    ptx_virtual_arch: Option<String>,
    #[serde(default)]
    minimum_device_cc: Option<f32>,
    #[serde(default)]
    minimum_cuda_driver_api: Option<u32>,
    #[serde(default)]
    derived_digest: Option<String>,
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
    #[allow(dead_code)]
    #[serde(default = "default_resume_deadline_ms")]
    resume_deadline_ms: u64,
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
            resume_deadline_ms: default_resume_deadline_ms(),
            result_page_bytes: default_job_result_page_bytes(),
            bulk_quantum_tokens: default_job_bulk_quantum_tokens(),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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

fn default_job_execution_ttl_ms() -> u64 {
    DEFAULT_JOB_EXECUTION_TTL_MS
}

fn default_job_result_retention_ttl_ms() -> u64 {
    DEFAULT_JOB_RESULT_RETENTION_TTL_MS
}

fn default_resume_deadline_ms() -> u64 {
    DEFAULT_RESUME_DEADLINE_MS
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

    fn record_job_inherited(&self) {
        self.jobs_inherited.fetch_add(1, Ordering::Relaxed);
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
    certify_observation: bool,
    hf_endpoint: String,
    release_catalog: catalog::Catalog,
    runnable_backends: BTreeSet<String>,
    catalog_locks: Mutex<BTreeMap<String, Arc<tokio::sync::Mutex<()>>>>,
    download_locks: Mutex<BTreeMap<String, Arc<Mutex<()>>>>,
    download_bytes: Mutex<BTreeMap<String, (u64, u64)>>,
    catalog_disk: Mutex<()>,
    self_check_holders: Mutex<BTreeMap<String, String>>,
    catalog_jobs: Mutex<BTreeMap<String, String>>,
    inline: InlineConfig,
    jobs: JobConfig,
    probe: ProbeConfig,
    worker_load_timeout: Duration,
    log: LogConfig,
    knob: PerfKnob,
    alias_admin_enabled: bool,
    microllm_max_tokens: u32,
    grammar_enabled: bool,
    sidecar_spec: SidecarSpec,
    decode_chain_k: u32,
    cache_max_bytes: u64,
    scheduler: Arc<Mutex<InlineScheduler>>,
    execution: Arc<Semaphore>,
    execution_stats: Arc<Mutex<InlineExecutionStats>>,
    control_loads: Arc<Semaphore>,
    ane_supervisor: Mutex<Option<worker_host::ane_residency::AneResidencySupervisor>>,
    catalog: Arc<Mutex<BTreeMap<String, ModelSlot>>>,
    job_progress: Arc<Mutex<BTreeMap<String, ModelRuntimeState>>>,
    owned_decode_q8: Arc<Mutex<owned_decode_routing::q8ingest::Q8IngestRegistry>>,
    owned_decode_dispatches:
        Arc<Mutex<BTreeMap<String, Arc<Mutex<worker_host::SupervisedDecodeDispatch>>>>>,
    /// Tracks module-owned decode sessions so serving admission, interrupted-stream
    /// recovery, and scheduler updates use the same durable session state.
    owned_decode_sessions: Arc<Mutex<OwnedDecodeWireState>>,
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
}

impl ActivityTelemetry {
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
        if let Ok(mut by_model) = self.telemetry.by_model.lock() {
            let remove = by_model.get_mut(&self.model_id).is_some_and(|count| {
                *count = count.saturating_sub(1);
                *count == 0
            });
            if remove {
                by_model.remove(&self.model_id);
            }
        }
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
    numeric_profile_id: NumericProfileId,
    fingerprint: Fingerprint,
    certification_fingerprint: Fingerprint,
    engine_identity: EngineIdentity,
    owned_tokenizer_policy: Option<OwnedTokenizerPolicy>,
    /// Platform-gated owned-decode execution refusal discovered while resolving
    /// the catalog identity. Routing consumes this before lane selection.
    owned_decode_resolution_refusal: Option<owned_decode_routing::error::OwnedDecodeError>,
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
                dims: Some(test_deterministic::DIMS),
                ..Default::default()
            },
            EmbedBackend::OwnedDecode => ExecutionModelInfo::default(),
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
        "ane" | ANE_WORKER_ENGINE => Some("f16".to_string()),
        CUDA_WORKER_ENGINE => Some("f16".to_string()),
        _ => None,
    }
}

fn device_class_for_engine(engine: &str) -> Option<String> {
    match engine {
        "owned-metal" | "owned-metal-decode" => Some("metal".to_string()),
        "ane" | ANE_WORKER_ENGINE => Some("ane".to_string()),
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
    match model.engine_identity.engine.as_str() {
        "owned-metal" => "metal",
        ANE_WORKER_ENGINE => "ane",
        CUDA_WORKER_ENGINE => "cuda",
        LLAMA_ENGINE | LLAMA_WORKER_ENGINE => "llama",
        DECODE_WORKER_ENGINE => "decode",
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

#[derive(Clone)]
enum EmbedBackend {
    #[cfg(feature = "test-support")]
    TestDeterministic(Arc<test_deterministic::TestDeterministic>),
    #[cfg(unix)]
    DirectAne(Arc<worker_host::ane_residency::DirectAneEngine>),
    Owned(Arc<Mutex<OwnedMetalEmbedEngine>>),
    /// Owned decode is loaded for each supervised generation so its generation
    /// supervisor, rather than the generic worker host, enforces the crash limit.
    OwnedDecode,
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
    allow_equivalent: bool,
    #[serde(default)]
    required_epoch: Option<u64>,
    #[serde(default)]
    accept_declared: bool,
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
    #[serde(default)]
    accept_declared: bool,
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
    #[serde(default)]
    accept_declared: bool,
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
    #[serde(default)]
    allow_equivalent: bool,
    #[serde(default)]
    required_epoch: Option<u64>,
    #[serde(default, rename = "accept_declared")]
    _accept_declared: bool,
    /// Set only by the session API; once a persistent session reserves
    /// owned-decode KV state, force the same worker identity instead of switching
    /// to a separate llama runtime.
    #[serde(skip)]
    owned_only: bool,
}

#[derive(Clone, Debug, Serialize)]
struct MicroLlmOneshotPayload {
    text: String,
    finish_reason: String,
    n_prompt: usize,
    n_gen: usize,
    /// Raw committed IDs let the session surface preserve the worker's exact
    /// progress history instead of retokenizing rendered text.
    generated_token_ids: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    runtime_config_digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    derived_digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_id: Option<String>,
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
    alias_table: AliasTable,
    request_bytes: u64,
    total_tokens: u64,
}

struct RemoteEmbedBatchJobWork {
    profile: Arc<remote::config::ConfiguredRemoteProfile>,
    request_digest: String,
    items: Vec<EmbedBatchItem>,
    deadline_ms: u64,
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
struct JobResumeParams {
    job_id: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
enum ModelLoadFileSpec {
    Legacy(String),
    Detailed { url: String, sha256: String },
}

impl ModelLoadFileSpec {
    fn locator(&self) -> &str {
        match self {
            Self::Legacy(value) => value,
            Self::Detailed { url, .. } => url,
        }
    }

    fn expected_digest(&self) -> Option<String> {
        match self {
            Self::Legacy(_) => None,
            Self::Detailed { sha256, .. } => Some(normalize_digest(sha256)),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ModelLoadFiles {
    model: ModelLoadFileSpec,
    tokenizer: ModelLoadFileSpec,
    #[serde(default)]
    config: Option<ModelLoadFileSpec>,
    #[serde(default)]
    extra: Vec<ModelLoadFileSpec>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ModelLoadParams {
    source: String,
    #[serde(default)]
    revision: Option<String>,
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
    deadline_ms: Option<u64>,
    #[serde(default)]
    model_id: Option<String>,
    #[serde(default)]
    normalize: Option<bool>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    quant: Option<String>,
    #[serde(default)]
    family: Option<String>,
    #[serde(default)]
    dtype: Option<String>,
    #[serde(default)]
    execution: Option<String>,
    #[serde(default)]
    attention_units: Option<usize>,
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
    deadline_ms: Option<u64>,
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
    comment: Option<String>,
    #[serde(default)]
    generation_command: Option<String>,
    #[serde(default)]
    family: Option<String>,
    #[serde(default)]
    reference_model: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    dims: Option<usize>,
    #[serde(default)]
    pooling: Option<String>,
    #[serde(default)]
    normalize: Option<bool>,
    #[serde(default)]
    ort_version: Option<String>,
    #[serde(default)]
    model_sha256: Option<String>,
    #[serde(default)]
    tokenizer_sha256: Option<String>,
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

#[derive(Clone, Debug, Deserialize)]
struct GenerateProbeFixture {
    family: String,
    dtype: String,
    quant: String,
    model: String,
    model_revision: String,
    #[serde(default)]
    generation_command: Option<String>,
    generation_command_sha256: String,
    provenance: Value,
    #[serde(default)]
    structural_band: GenerateStructuralBand,
    items: Vec<GenerateProbeItem>,
}

#[derive(Clone, Debug, Deserialize)]
struct GenerateProbeItem {
    id: String,
    prompt: String,
    expected_token_ids: Vec<u32>,
    max_new_tokens: u32,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct GenerateStructuralBand {
    max_forks: usize,
    top2_gap_ceiling: f64,
    #[serde(default)]
    allowed_forks: Vec<GenerateAllowedFork>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct GenerateAllowedFork {
    id: String,
    token_index: usize,
    oracle_token: u32,
    alternate_token: u32,
    oracle_top2: [u32; 2],
    top2_gap: f64,
}

#[derive(Clone, Debug, Serialize)]
struct GenerateProbeEvidence {
    token_exact_matches: usize,
    accepted_structural_forks: usize,
    max_certified_forks: usize,
    items: usize,
    tokens_compared: usize,
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

#[derive(Clone, Debug, Eq, PartialEq)]
struct ProbeReferenceKey {
    family: String,
    model: String,
}

struct LaneMeasurementRows {
    current_certification: Option<CertificationRow>,
    latest_certification: Option<CertificationRow>,
    current_probe: Option<CertificationRow>,
    latest_probe: Option<CertificationRow>,
    certification_stale: bool,
    current_performance: Option<PerfRow>,
    latest_performance: Option<PerfRow>,
    performance_stale: bool,
}

struct CatalogLaneMeasurements {
    slot: ModelSlotSnapshot,
    certification_fingerprint: Fingerprint,
    measurements: LaneMeasurementRows,
}

struct CatalogMeasurementSummary {
    lanes: Vec<CatalogLaneMeasurements>,
    certification_stale: bool,
    performance_stale: bool,
    certified_lanes: usize,
}

struct PerfBenchResult {
    throughput_tok_s: f64,
    cold_load_ms: f64,
    single_item_latency_p50_ms: f64,
    details: Value,
}

struct SystemClock;

/// Use eight rows for ANE because the paired-sweep benchmark identified eight
/// as the recommended fixed batch size. With fixed sequence buckets, the
/// corresponding token budget is the row count multiplied by the model's
/// maximum sequence length; see `bench/results/mc-paired-sweep-20260720.md`
/// for the measurements.
fn recommended_batch_for_engine(engine: &str, max_tokens: usize) -> Option<RecommendedBatch> {
    match engine {
        "owned-metal" | CUDA_WORKER_ENGINE => Some(RecommendedBatch {
            rows: MAX_ENGINE_BATCH_ITEMS,
            token_budget: DEFAULT_ENGINE_BATCH_TOKEN_BUDGET,
        }),
        "ane" | ANE_WORKER_ENGINE => {
            let rows = MAX_ENGINE_BATCH_ITEMS;
            Some(RecommendedBatch {
                rows,
                token_budget: (max_tokens.max(1) as u64).saturating_mul(rows as u64),
            })
        }
        _ => None,
    }
}

impl Clock for SystemClock {
    fn now_ms(&self) -> u64 {
        now_ms()
    }
}

impl SynapseHandler {
    fn new(module_id: String, connection_file: PathBuf) -> Self {
        Self {
            inner: Arc::new(SynapseHandlerInner {
                module_id,
                connection_file,
                state: OnceLock::new(),
                approval_operators: Mutex::new(HashMap::new()),
                flow_routes: Mutex::new(HashMap::new()),
                connection_end: Arc::new(OnceLock::new()),
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
        let config = load_module_config()?;
        let configured_remote =
            validate_remote_providers(&config.remote_providers).map_err(ModuleError::Config)?;
        bind_remote_provider_urls(&store, &configured_remote)?;
        validate_hf_endpoint(&config.hf_endpoint).map_err(ModuleError::Config)?;
        let release_catalog = runtime_release_catalog()?;
        let model_cache = Arc::new(ModelCache::new(ModelCache::default_root()?));
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
        let vault_client = Arc::new(SubcVaultCredentialClient::new(
            self.inner.connection_file.clone(),
        ));
        let remote_gateway = Arc::new(
            RemoteGateway::new(
                Arc::clone(&store),
                configured_remote,
                vault_client,
                machine_profile_hash.clone(),
            )
            .map_err(|error| ModuleError::Config(error.message))?,
        );
        let runtime = Arc::new(RuntimeState::from_catalog(config, catalog_models)?);
        reconcile_startup_orphans(&store, module_generation, &runtime.admission_telemetry)?;
        let continuity_check: Arc<dyn ContinuityCheck> = remote_gateway.continuity.clone();
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
            continuity_check,
            remote_gateway,
        });
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
        let runnable_backends = detected_catalog_backends();
        let hf_endpoint = config.hf_endpoint;
        let inline = config.inline;
        let jobs = config.jobs;
        let probe = config.probe;
        let worker_load_timeout = Duration::from_millis(config.worker.load_timeout_ms);
        let log = config.log;
        let knob = config.knob;
        let alias_admin_enabled = config.alias_admin_enabled || config.dev.alias_admin_enabled;
        let microllm_max_tokens = config.microllm_max_tokens;
        let grammar_enabled = config.grammar_enabled;
        config
            .sidecar_spec
            .validate()
            .map_err(|error| ModuleError::Config(error.to_string()))?;
        let sidecar_spec = config.sidecar_spec;
        validate_decode_chain_k(config.decode_chain_k)?;
        let decode_chain_k = config.decode_chain_k;
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
            hf_endpoint,
            release_catalog,
            runnable_backends,
            catalog_locks: Mutex::new(BTreeMap::new()),
            download_locks: Mutex::new(BTreeMap::new()),
            download_bytes: Mutex::new(BTreeMap::new()),
            catalog_disk: Mutex::new(()),
            self_check_holders: Mutex::new(BTreeMap::new()),
            catalog_jobs: Mutex::new(BTreeMap::new()),
            inline,
            jobs,
            probe,
            worker_load_timeout,
            log,
            knob,
            alias_admin_enabled,
            microllm_max_tokens,
            grammar_enabled,
            sidecar_spec,
            decode_chain_k,
            cache_max_bytes,
            scheduler,
            execution,
            execution_stats,
            ane_supervisor: Mutex::new(None),
            control_loads: Arc::new(Semaphore::new(1)),
            catalog: Arc::new(Mutex::new(catalog)),
            job_progress: Arc::new(Mutex::new(BTreeMap::new())),
            owned_decode_q8: Arc::new(Mutex::new(
                owned_decode_routing::q8ingest::Q8IngestRegistry::new(),
            )),
            owned_decode_dispatches: Arc::new(Mutex::new(BTreeMap::new())),
            owned_decode_sessions: Arc::new(Mutex::new(OwnedDecodeWireState::default())),
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
            EmbedBackend::Owned(_) | EmbedBackend::OwnedDecode => None,
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

fn bind_remote_provider_urls(
    store: &SynapseStore,
    providers: &[ConfiguredProvider],
) -> Result<(), ModuleError> {
    let now = now_ms();
    let mut active_hashes = Vec::new();
    for provider in providers {
        for profile in &provider.models {
            store.bind_remote_profile_url(
                &profile.remote_profile_hash,
                provider.base_url.as_str(),
            )?;
            active_hashes.push(profile.remote_profile_hash.clone());
        }
    }
    store.sweep_remote_url_bindings(&active_hashes, now, 7 * 24 * 60 * 60 * 1_000)?;
    Ok(())
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
    if let Some(backend) = declared_llama_backend {
        spec.engine_identity
            .build_flags
            .insert("backend".to_string(), backend);
    }
    if engine_name == "owned-metal-decode" {
        use owned_decode_routing::identity::WeightQuant;

        let family = preload.family.ok_or_else(|| {
            ModuleError::Config("owned-metal-decode catalog entry is missing family".to_string())
        })?;
        owned_decode_routing::family::Family::parse(&family)
            .map_err(|error| ModuleError::Config(error.as_str().to_string()))?;
        let dtype = preload.dtype.unwrap_or_else(|| "f16".to_string());
        if dtype != "f16" {
            return Err(ModuleError::Config(format!(
                "owned-metal-decode activation dtype '{dtype}' is unsupported"
            )));
        }
        let weight_quant = WeightQuant::parse(&spec.quant)
            .map_err(|error| ModuleError::Config(error.as_str().to_string()))?;
        let q8_identity = match weight_quant {
            WeightQuant::F16 => {
                if preload.quantizer_revision.is_some() || preload.derived_digest.is_some() {
                    return Err(ModuleError::Config(
                        "owned-metal-decode f16 entry must not declare Q8 identity".to_string(),
                    ));
                }
                None
            }
            WeightQuant::Q8_0 => Some((
                preload.quantizer_revision.ok_or_else(|| {
                    ModuleError::Config(
                        "owned-metal-decode q8_0 entry is missing quantizer_revision".to_string(),
                    )
                })?,
                preload.derived_digest.ok_or_else(|| {
                    ModuleError::Config(
                        "owned-metal-decode q8_0 entry is missing derived_digest".to_string(),
                    )
                })?,
            )),
        };
        spec.owned_family = Some(family);
        spec.owned_dtype = Some(dtype);
        spec.owned_execution = Some(
            preload
                .execution
                .unwrap_or_else(|| "supervised".to_string()),
        );
        if let Some(revision) = preload.arithmetic_identity_revision {
            spec.engine_identity
                .build_flags
                .insert("arithmetic_identity_revision".to_string(), revision);
        }
        if let Some(revision) = preload.metallib_revision {
            spec.engine_identity
                .build_flags
                .insert("metallib_revision".to_string(), revision);
        }
        if let Some((quantizer_revision, derived_digest)) = q8_identity {
            spec.engine_identity
                .build_flags
                .insert("quantizer_revision".to_string(), quantizer_revision);
            spec.engine_identity
                .build_flags
                .insert("derived_digest".to_string(), derived_digest);
        }
    }
    Ok(spec)
}

fn normalize_catalog_model(
    model: StoredModelConfig,
    inline: &InlineConfig,
    jobs: &JobConfig,
) -> Result<StoredModelConfig, ModuleError> {
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
    let decode_metadata = (engine_name == "owned-metal-decode").then(|| {
        (
            model.owned_family.clone(),
            model.owned_dtype.clone(),
            model.owned_execution.clone(),
            model.engine_identity.build_flags.clone(),
        )
    });
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
    if let Some((family, dtype, execution, build_flags)) = decode_metadata {
        spec.owned_family = family;
        spec.owned_dtype = dtype.or_else(|| Some("f16".to_string()));
        spec.owned_execution = execution.or_else(|| Some("supervised".to_string()));
        spec.engine_identity.build_flags.extend(build_flags);
    }
    Ok(spec)
}

struct CatalogProfile {
    id: String,
    slug: String,
    manifest: Value,
    typed: synapse_parity::manifest::Manifest,
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
            let template = model.grammar.template.as_ref().ok_or_else(|| {
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
        let mut identity = if lane == "owned-metal" {
            owned_engine_identity(family, dtype)
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
        Ok(OwnedCatalogConfig {
            family,
            dtype,
            execution: execution.unwrap_or("explicit").into(),
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

    fn apply_numeric_profile(&self, numeric: &mut NumericProfile) {
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
        numeric.kernel_revision = Some(
            match self.profile()["lane"].as_str().expect("lane") {
                "owned-metal" => synapse_core::METAL_KERNEL_REVISION,
                "owned-cuda" => synapse_core::CUDA_KERNEL_REVISION,
                "owned-vulkan" => synapse_core::VULKAN_KERNEL_REVISION,
                "ane-direct-worker" => synapse_core::ANE_DIRECT_KERNEL_REVISION,
                _ => unreachable!("manifest lane"),
            }
            .into(),
        );
    }
}

#[derive(Clone, Debug)]
struct OwnedCatalogConfig {
    family: OwnedFamily,
    dtype: OwnedDType,
    execution: String,
    attention_units: usize,
    config_locator: Option<ModelAssetLocator>,
    extra_locators: Vec<ModelAssetLocator>,
    /// CUDA carries a backend-specific identity while reusing the catalog's
    /// family/dtype storage fields.
    identity_override: Option<EngineIdentity>,
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
    if engine_name == "owned-metal-decode" && task != ModelTask::Generate {
        return Err(ModuleError::Config(
            "owned-metal-decode supports generation models only".to_string(),
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
            owned
                .as_ref()
                .map(|profile| owned_engine_identity(profile.family, profile.dtype))
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
                LLAMA_ENGINE | "ane" => NumericDType::F16,
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
        "onnx" => "ort".to_string(),
        "llama.cpp" | LLAMA_WORKER_ENGINE => "llama".to_string(),
        "coreml" | "neural_engine" | ANE_WORKER_ENGINE => "ane".to_string(),
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
        "ane" => "mlmodelc".to_string(),
        // Every owned engine loads a converted safetensors profile package.
        "owned-metal" | "owned-cuda" | "owned-vulkan" | synapse_core::ANE_DIRECT_WORKER_ENGINE => {
            "safetensors-package".to_string()
        }
        "owned-metal-decode" => "owned-safetensors".to_string(),
        _ => "onnx".to_string(),
    }
}

fn default_quant(engine_name: &str) -> String {
    match engine_name {
        LLAMA_ENGINE => "f16".to_string(),
        "ane" => "fp16".to_string(),
        "owned-metal" | "owned-cuda" | "owned-metal-decode" => "f16".to_string(),
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
        "ane" => Ok(worker_catalog_identity(
            ANE_WORKER_ENGINE,
            "protocol-v1",
            &[
                ("transport", worker_catalog_transport()),
                ("placement_gate", "neural-engine"),
            ],
        )),
        // These two lanes exist only as catalog lanes, whose fingerprints are pinned
        // in the release manifest and must be identical on every platform that
        // runs them (Vulkan runs on Linux and Windows). The IPC transport differs
        // by OS and does not affect the vectors, so it stays out of the identity.
        // The legacy arms above keep it, because live fingerprints depend on it.
        "owned-vulkan" | "ane-direct-worker" => {
            Ok(worker_catalog_identity(engine_name, "protocol-v2", &[]))
        }
        "owned-cuda" => Ok(owned_cuda_engine_identity(
            "unknown",
            "f16",
            "cuda-kernel-v1",
        )),
        "owned-metal-decode" => Ok(worker_catalog_identity(
            DECODE_WORKER_ENGINE,
            "owned-metal-decode-worker-v1",
            &[
                ("transport", worker_catalog_transport()),
                ("lane", "decode"),
                ("risk_class", "abort_capable"),
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

/// Run storage maintenance from the host's periodic health cadence. Keeping
/// this work on an existing heartbeat avoids a second timer loop in the module.
fn run_background_maintenance(state: &ModuleState) {
    run_background_maintenance_at(state, now_ms());
}

/// The maintenance pass with the wall-clock reading supplied by the caller, so
/// tests can run it at a chosen time instead of waiting for time to pass.
fn run_background_maintenance_at(state: &ModuleState, now: u64) {
    if let Err(error) = state.store.purge_expired_jobs(now) {
        tracing::warn!(target: "maintenance", error = %error, "job purge sweep failed");
    }
    let active_hashes = state
        .remote_gateway
        .profiles()
        .iter()
        .map(|profile| profile.remote_profile_hash.clone())
        .collect::<Vec<_>>();
    if let Err(error) =
        state
            .store
            .sweep_remote_url_bindings(&active_hashes, now, 7 * 24 * 60 * 60 * 1_000)
    {
        tracing::warn!(target: "maintenance", error = %error, "URL binding sweep failed");
    }
    match state.runtime.owned_decode_sessions.lock() {
        Ok(mut sessions) => {
            sessions.evict_expired_closed_sessions(now);
        }
        Err(_) => {
            tracing::warn!(target: "maintenance", "closed decode session sweep skipped: session lock poisoned");
        }
    }
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
        let _ = self.inner.state.set(Arc::clone(&state));
        let _ = start_perf_sampler(state);
    }

    async fn on_bind(&self, req: &RouteBindRequest) -> BindDecision {
        if self.state().is_none() {
            return BindDecision::reject(
                "module_not_initialized",
                "synapse has not completed HELLO_ACK initialization",
            );
        }
        if let Ok(mut operators) = self.inner.approval_operators.lock() {
            let approved_by = match req.principal.as_ref() {
                Some(Principal::Direct) => Some("principal:direct".to_string()),
                Some(Principal::Reserved { module_id }) => {
                    Some(format!("principal:reserved:{module_id}"))
                }
                Some(Principal::Unverified) | None => None,
            };
            if let Some(approved_by) = approved_by {
                operators.insert(req.handle, approved_by);
            } else {
                operators.remove(&req.handle);
            }
        }
        self.inner.record_bind_scope(req);
        BindDecision::accept()
    }

    async fn on_route_gone(&self, handle: &RouteHandle) {
        if let Ok(mut operators) = self.inner.approval_operators.lock() {
            operators.remove(handle);
        }
        if let Ok(mut flows) = self.inner.flow_routes.lock() {
            flows.remove(handle);
        }
    }

    /// Remembers how the daemon connection ended so the `synapse stopped` line
    /// can say it. The SDK calls this at most once per connection, and this
    /// module serves exactly one connection per process.
    async fn on_connection_end(&self, end: ConnectionEnd) {
        let _ = self.inner.connection_end.set(end);
    }

    /// Keep the daemon health status `ok` while publishing certification metrics as
    /// `{ certification: { certification_stale, stale_since_ms, lanes: [{ model_id,
    /// workload, certified }] } }`. Health reads stored rows only; it never runs probes.
    async fn health(&self) -> HealthReport {
        let Some(state) = self.state() else {
            return HealthReport::ok();
        };
        run_background_maintenance(&state);
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

    async fn handle(&self, ctx: RequestCtx, body: Vec<u8>) -> HandlerOutcome {
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

        if let Some(refusal) = self
            .inner
            .flow_refusal(&ctx.route_handle(), &envelope.method)
        {
            return refusal;
        }
        let approved_by = self
            .inner
            .approval_operators
            .lock()
            .ok()
            .and_then(|operators| operators.get(&ctx.route_handle()).cloned());
        dispatch_request(state, envelope, approved_by.as_deref()).await
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

async fn dispatch_request(
    state: Arc<ModuleState>,
    request: MethodEnvelope,
    approved_by: Option<&str>,
) -> HandlerOutcome {
    match request.method.as_str() {
        "models.catalog" => models_catalog(state, request.params).await,
        "models.download" => models_download(state, request.params).await,
        "models.download.cancel" => models_download_cancel(state, request.params).await,
        "models.remove" => models_remove(state, request.params).await,
        "models.list" => match state.store.catalog_snapshot() {
            Ok(snapshot) => result_outcome(models_list_payload(&state, snapshot)),
            Err(error) => channel_error("store_failure", error.to_string()),
        },
        "certify.observations" => certify_observations(state),
        "embed.query" => embed_query(state, request.params).await,
        "embed.batch" => embed_batch(state, request.params).await,
        "embed.result" => embed_result(state, request.params).await,
        "job.resume" => job_resume(state, request.params).await,
        "rerank.score" => rerank_score(state, request.params).await,
        "microllm.oneshot" => microllm_oneshot(state, request.params).await,
        "owned_decode.admit_session" | "admit_session" => {
            owned_decode_admit_session(state, request.params).await
        }
        "owned_decode.decode" | "decode" => {
            owned_decode_session_decode(state, request.params).await
        }
        "owned_decode.snapshot" | "snapshot" => {
            owned_decode_session_snapshot(state, request.params).await
        }
        "owned_decode.continue" | "continue" => {
            owned_decode_session_continue(state, request.params).await
        }
        "owned_decode.abort" | "abort" => owned_decode_session_abort(state, request.params).await,
        "owned_decode.close" | "close" => owned_decode_session_close(state, request.params).await,
        "owned_decode.session_status" | "session_status" => {
            owned_decode_session_status(state, request.params).await
        }
        "owned_decode.disable" | "disable" => owned_decode_disable(state, request.params).await,
        "owned_decode.revoke" | "revoke" => owned_decode_revoke(state, request.params).await,
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
        "approvals.migrate_owned_decode" => {
            approvals_migrate_owned_decode(state, request.params).await
        }
        "approvals.enable" => match approved_by {
            Some(approved_by) => approval_enable(state, request.params, approved_by).await,
            None => channel_error(
                "operator_identity_unavailable",
                "approvals.enable requires an authenticated operator identity",
            ),
        },
        "approvals.disable" => approval_disable(state, request.params).await,
        "approvals.emergency_rollback" => approvals_emergency_rollback(state, request.params).await,
        "model.status" => model_status(state, request.params).await,
        other => channel_error(
            "unknown_method",
            format!("unknown method '{other}' for synapse management surface"),
        ),
    }
}

fn owned_decode_failure(
    state: &ModuleState,
    code: impl Into<String>,
    message: impl Into<String>,
) -> HandlerOutcome {
    result_outcome(json!({
        "module_generation": state.module_generation,
        "error": {
            "code": code.into(),
            "class": "permanent",
            "safe_to_retry_same_request": false,
            "message": message.into(),
        }
    }))
}

fn owned_decode_admission_failure(
    state: &ModuleState,
    code: impl Into<String>,
    message: impl Into<String>,
) -> HandlerOutcome {
    let code = code.into();
    state.runtime.admission_telemetry.record_refusal(&code);
    owned_decode_failure(state, code, message)
}

fn map_serving_refusal(refusal: store::ServingRefusal) -> synapse_core::OwnedDecodeRefusal {
    match refusal {
        store::ServingRefusal::ArtifactUnapproved => {
            synapse_core::OwnedDecodeRefusal::ArtifactUnapproved
        }
        store::ServingRefusal::ArtifactDisabled => {
            synapse_core::OwnedDecodeRefusal::ArtifactDisabled
        }
        store::ServingRefusal::ArtifactRevoked => synapse_core::OwnedDecodeRefusal::ArtifactRevoked,
        store::ServingRefusal::CertificationMismatch => {
            synapse_core::OwnedDecodeRefusal::ArtifactMismatch
        }
        store::ServingRefusal::RetainedStateInvalidated => {
            synapse_core::OwnedDecodeRefusal::RetainedKvUnavailable
        }
    }
}

fn map_admission_refusal(
    refusal: &owned_decode_routing::admission::AdmissionRefusal,
) -> synapse_core::OwnedDecodeRefusal {
    use owned_decode_routing::admission::AdmissionRefusal;

    match refusal {
        AdmissionRefusal::UnsupportedPlatformTuple => {
            synapse_core::OwnedDecodeRefusal::UnsupportedMachine
        }
        AdmissionRefusal::InvalidContextCeiling { .. } => {
            synapse_core::OwnedDecodeRefusal::InvalidContextCeiling
        }
        AdmissionRefusal::InsufficientReservedLaneMemory { .. } => {
            synapse_core::OwnedDecodeRefusal::InsufficientMemory
        }
        AdmissionRefusal::IncompatibleArtifact
        | AdmissionRefusal::ArtifactSwapActiveSessions { .. } => {
            synapse_core::OwnedDecodeRefusal::IncompatibleResidentArtifact
        }
        AdmissionRefusal::UnsupportedGenerationConfig => {
            synapse_core::OwnedDecodeRefusal::SamplingUnsupported
        }
        AdmissionRefusal::InvalidKvConfiguration => {
            synapse_core::OwnedDecodeRefusal::InvalidKvConfiguration
        }
        AdmissionRefusal::InvalidKvAlignment { .. } => {
            synapse_core::OwnedDecodeRefusal::InvalidKvAlignment
        }
        AdmissionRefusal::UnknownSession { .. } => {
            synapse_core::OwnedDecodeRefusal::RetainedKvUnavailable
        }
    }
}

fn current_unified_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        synapse_core::without_launch_nonce(std::process::Command::new("sysctl"))
            .args(["-n", "hw.memsize"])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .and_then(|value| value.trim().parse().ok())
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

fn serving_admission_material(
    state: &ModuleState,
    catalog_fingerprint: &str,
) -> Result<ServingAdmissionMaterial, (synapse_core::OwnedDecodeRefusal, String)> {
    use owned_decode_certification::CertificationGateResult;
    use owned_decode_routing::admission::{ArtifactReservation, MachineTuple, PlatformEnvelope};
    use store::ServingApprovalState;

    let approval = state
        .store
        .serving_approval(catalog_fingerprint)
        .map_err(|error| {
            (
                synapse_core::OwnedDecodeRefusal::ArtifactMismatch,
                format!("serving approval is unreadable: {error}"),
            )
        })?
        .ok_or_else(|| {
            (
                synapse_core::OwnedDecodeRefusal::ArtifactUnapproved,
                "no serving approval exists for the catalog fingerprint".to_string(),
            )
        })?;
    match approval.state {
        ServingApprovalState::Enabled => {}
        ServingApprovalState::Disabled => {
            return Err((
                synapse_core::OwnedDecodeRefusal::ArtifactDisabled,
                "the catalog fingerprint is disabled".to_string(),
            ))
        }
        ServingApprovalState::Revoked => {
            return Err((
                synapse_core::OwnedDecodeRefusal::ArtifactRevoked,
                "the catalog fingerprint is revoked".to_string(),
            ))
        }
    }
    let certification = state
        .store
        .serving_certification(&approval.certification_record_id)
        .map_err(|error| {
            (
                synapse_core::OwnedDecodeRefusal::ArtifactMismatch,
                format!("serving certification is unreadable: {error}"),
            )
        })?
        .ok_or_else(|| {
            (
                synapse_core::OwnedDecodeRefusal::ArtifactMismatch,
                "the serving approval references no certification record".to_string(),
            )
        })?;
    let record = certification.record;
    if record.unit.catalog_fingerprint != catalog_fingerprint {
        return Err((
            synapse_core::OwnedDecodeRefusal::ArtifactMismatch,
            "the serving certification does not match the requested catalog fingerprint"
                .to_string(),
        ));
    }
    if record.machine_evidence.machine.machine_profile_hash != state.revisioned_machine_profile_hash
        || record.machine_evidence.machine.macos_build != state.machine_profile.os_build
    {
        return Err((
            synapse_core::OwnedDecodeRefusal::UnsupportedMachine,
            "the serving certification was recorded for a different machine tuple".to_string(),
        ));
    }
    let unified_memory_bytes = current_unified_memory_bytes().ok_or_else(|| {
        (
            synapse_core::OwnedDecodeRefusal::UnsupportedMachine,
            "the module cannot read the machine unified-memory capacity".to_string(),
        )
    })?;
    let platform = record
        .gate_results
        .iter()
        .find_map(|result| match result {
            CertificationGateResult::PlatformEnvelope(result) => Some(result),
            _ => None,
        })
        .ok_or_else(|| {
            (
                synapse_core::OwnedDecodeRefusal::ArtifactMismatch,
                "the serving certification has no platform-envelope result".to_string(),
            )
        })?;
    let artifact = ArtifactReservation::new(
        record.unit.catalog_fingerprint.clone(),
        platform.artifact_weight_bytes,
        platform.kv_bytes_per_token,
    )
    .map_err(|error| {
        (
            synapse_core::OwnedDecodeRefusal::ArtifactMismatch,
            format!("certified artifact reservation is invalid: {error}"),
        )
    })?;
    let envelope = PlatformEnvelope::new(
        platform.machine_profile_hash.clone(),
        platform.macos_build.clone(),
        platform.unified_memory_bytes,
        platform.reserved_embed_rerank_bytes,
        artifact.clone(),
    )
    .map_err(|error| {
        (
            synapse_core::OwnedDecodeRefusal::ArtifactMismatch,
            format!("certified platform envelope is invalid: {error}"),
        )
    })?;
    Ok(ServingAdmissionMaterial {
        machine: MachineTuple::new(
            state.revisioned_machine_profile_hash.clone(),
            state.machine_profile.os_build.clone(),
            unified_memory_bytes,
        ),
        envelope,
        artifact,
        model_id: record.artifact_lineage.model_id,
    })
}

/// Resolve the model a serving approval's catalog runs, from the certification
/// record the approval references. Unlike [`serving_admission_material`] this
/// does not require the approval to be enabled, because `owned_decode.disable`
/// needs the model after the approval has left the enabled state.
fn serving_catalog_model_id(
    state: &ModuleState,
    approval: &store::ServingApprovalRecord,
) -> Option<String> {
    let certification = match state
        .store
        .serving_certification(&approval.certification_record_id)
    {
        Ok(Some(certification)) => certification,
        Ok(None) => {
            tracing::warn!(
                catalog_fingerprint = %approval.catalog_fingerprint,
                "serving approval references no certification record; cannot resolve its model"
            );
            return None;
        }
        Err(error) => {
            tracing::warn!(
                catalog_fingerprint = %approval.catalog_fingerprint,
                error = %error,
                "serving certification is unreadable; cannot resolve its model"
            );
            return None;
        }
    };
    let record = certification.record;
    if record.unit.catalog_fingerprint != approval.catalog_fingerprint {
        tracing::warn!(
            catalog_fingerprint = %approval.catalog_fingerprint,
            "serving certification does not match its approval's catalog fingerprint"
        );
        return None;
    }
    Some(record.artifact_lineage.model_id)
}
fn session_request_key(session_id: &str, req_id: &str) -> String {
    format!("{session_id}:{req_id}")
}

fn absolute_serving_boundary(previous: u64, request_committed: u32) -> u64 {
    previous.saturating_add(request_committed.into())
}

async fn owned_decode_admit_session(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let params: OwnedDecodeSessionAdmissionParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid owned_decode.admit_session params: {error}"),
            )
        }
    };
    if let Err(refusal) = synapse_core::validate_greedy_generation(&params.generation) {
        return owned_decode_admission_failure(
            &state,
            refusal.as_str(),
            "wave-1 sessions require greedy_top1",
        );
    }
    let material = match serving_admission_material(&state, &params.catalog_fingerprint) {
        Ok(material) => material,
        Err((refusal, message)) => {
            return owned_decode_admission_failure(&state, refusal.as_str(), message)
        }
    };
    let mut sessions = match state.runtime.owned_decode_sessions.lock() {
        Ok(sessions) => sessions,
        Err(_) => {
            return owned_decode_admission_failure(
                &state,
                "owned_decode_unavailable",
                "the decode-session coordinator is unavailable",
            )
        }
    };
    if sessions
        .residency
        .as_ref()
        .is_none_or(|router| router.active_session_count() == 0)
    {
        sessions.residency = Some(owned_decode_routing::admission::ResidencyRouter::new(
            material.machine.clone(),
            material.envelope.clone(),
        ));
    }
    let routing_request = owned_decode_routing::admission::AdmissionRequest::new(
        params.caller_id,
        material.artifact.clone(),
        params.context_ceiling_tokens,
        owned_decode_routing::admission::GenerationConfiguration::greedy_top1(),
        params.kv_configuration,
    );
    let routing_admission = match sessions
        .residency
        .as_mut()
        .expect("decode residency is initialized")
        .admit_session(routing_request)
    {
        Ok(admission) => admission,
        Err(refusal) => {
            let wire = map_admission_refusal(&refusal);
            return owned_decode_admission_failure(&state, wire.as_str(), refusal.to_string());
        }
    };
    let session_id = format!(
        "owned-decode-{}-{}-{}",
        state.module_generation,
        now_ms(),
        sessions.next_session_sequence
    );
    sessions.next_session_sequence = sessions.next_session_sequence.saturating_add(1);
    let durable_admission =
        state
            .store
            .admit_serving_session(&session_id, &params.catalog_fingerprint, now_ms());
    let approval_generation = match durable_admission {
        Ok(store::ServingSessionAdmission::Admitted {
            approval_generation,
            ..
        }) => approval_generation,
        Ok(store::ServingSessionAdmission::Refused { reason }) => {
            let _ = sessions
                .residency
                .as_mut()
                .expect("decode residency is initialized")
                .close_session(routing_admission.session_id);
            let refusal = map_serving_refusal(reason);
            return owned_decode_admission_failure(
                &state,
                refusal.as_str(),
                "serving approval refused admission",
            );
        }
        Err(error) => {
            let _ = sessions
                .residency
                .as_mut()
                .expect("decode residency is initialized")
                .close_session(routing_admission.session_id);
            return owned_decode_failure(
                &state,
                "store_failure",
                format!("could not persist session admission: {error}"),
            );
        }
    };
    sessions.sessions.insert(
        session_id.clone(),
        OwnedDecodeWireSession {
            catalog_fingerprint: params.catalog_fingerprint,
            model_id: material.model_id,
            routing_session_id: routing_admission.session_id,
            kv_configuration: params.kv_configuration,
            active_request: None,
            retained_kv_session_id: None,
            retained_position: None,
            closed_at_ms: None,
        },
    );
    let receipt = routing_admission.receipt;
    result_outcome(json!({
        "session_id": session_id,
        "catalog_fingerprint": receipt.catalog_fingerprint,
        "approval_generation": approval_generation,
        "reservation": {
            "reserved_embed_rerank_bytes": receipt.reserved_embed_rerank_bytes,
            "reserved_artifact_weight_bytes": receipt.reserved_artifact_weight_bytes,
            "reserved_session_kv_bytes": receipt.reserved_session_kv_bytes,
            "context_ceiling_tokens": receipt.context_ceiling_tokens,
        },
    }))
}

async fn owned_decode_session_snapshot(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let params: OwnedDecodeSessionSnapshotParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid owned_decode.snapshot params: {error}"),
            )
        }
    };
    let mut sessions = match state.runtime.owned_decode_sessions.lock() {
        Ok(sessions) => sessions,
        Err(_) => {
            return owned_decode_failure(
                &state,
                "owned_decode_unavailable",
                "decode sessions are unavailable",
            )
        }
    };
    let Some(session) = sessions.sessions.get(&params.session_id).cloned() else {
        return owned_decode_failure(
            &state,
            "unknown_session",
            "the decode session does not exist",
        );
    };
    if session.is_closed() || session.active_request.is_some() {
        return owned_decode_failure(
            &state,
            synapse_core::OwnedDecodeRefusal::SessionStillInFlight.as_str(),
            "a session can snapshot only at an idle committed boundary",
        );
    }
    let boundary = synapse_core::KvReuseBoundary {
        position: params.position_tokens,
        block_size: session.kv_configuration.block_size_tokens,
        recurrent_state_grain: session.kv_configuration.recurrent_state_grain_tokens,
    };
    if let Err(refusal) = synapse_core::validate_kv_reuse_boundary(boundary) {
        return owned_decode_failure(
            &state,
            refusal.as_str(),
            "snapshot position is not a valid KV boundary",
        );
    }
    let route = match sessions
        .residency
        .as_ref()
        .expect("admitted decode session has residency")
        .route_continuation(session.routing_session_id, params.position_tokens)
    {
        Ok(route) => route,
        Err(refusal) => {
            let wire = map_admission_refusal(&refusal);
            return owned_decode_failure(&state, wire.as_str(), refusal.to_string());
        }
    };
    let retained_kv_session_id = format!(
        "{}:retained:{}:{}",
        params.session_id, params.position_tokens, sessions.next_session_sequence
    );
    sessions.next_session_sequence = sessions.next_session_sequence.saturating_add(1);
    match state.store.retain_serving_state(
        &retained_kv_session_id,
        &session.catalog_fingerprint,
        now_ms(),
    ) {
        Ok(store::ServingContinuationAdmission::Admitted { .. }) => {}
        Ok(store::ServingContinuationAdmission::Refused { reason }) => {
            let refusal = map_serving_refusal(reason);
            return owned_decode_failure(
                &state,
                refusal.as_str(),
                "serving control refused KV retention",
            );
        }
        Err(error) => {
            return owned_decode_failure(
                &state,
                "store_failure",
                format!("could not persist retained KV state: {error}"),
            )
        }
    }
    let session = sessions
        .sessions
        .get_mut(&params.session_id)
        .expect("session remains registered while snapshotting");
    session.retained_kv_session_id = Some(retained_kv_session_id.clone());
    session.retained_position = Some(params.position_tokens);
    result_outcome(json!({
        "session_id": params.session_id,
        "retained_kv_session_id": retained_kv_session_id,
        "retained_position": route.retained_prefix_tokens,
        "reused_blocks": route.reused_blocks,
    }))
}

async fn owned_decode_session_continue(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let params: OwnedDecodeSessionContinueParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid owned_decode.continue params: {error}"),
            )
        }
    };
    match state
        .store
        .admit_serving_continuation(&params.retained_kv_session_id)
    {
        Ok(store::ServingContinuationAdmission::Admitted { .. }) => {}
        Ok(store::ServingContinuationAdmission::Refused { reason }) => {
            let refusal = map_serving_refusal(reason);
            return owned_decode_failure(
                &state,
                refusal.as_str(),
                "serving control refused continuation",
            );
        }
        Err(error) => {
            return owned_decode_failure(
                &state,
                "store_failure",
                format!("could not read retained KV state: {error}"),
            )
        }
    }
    let sessions = match state.runtime.owned_decode_sessions.lock() {
        Ok(sessions) => sessions,
        Err(_) => {
            return owned_decode_failure(
                &state,
                "owned_decode_unavailable",
                "decode sessions are unavailable",
            )
        }
    };
    let Some(session) = sessions.sessions.get(&params.session_id) else {
        return owned_decode_failure(
            &state,
            "unknown_session",
            "the decode session does not exist",
        );
    };
    if session.is_closed() || session.active_request.is_some() {
        return owned_decode_failure(
            &state,
            synapse_core::OwnedDecodeRefusal::SessionStillInFlight.as_str(),
            "a continuation requires an idle retained session",
        );
    }
    if session.retained_kv_session_id.as_deref() != Some(&params.retained_kv_session_id) {
        return owned_decode_failure(
            &state,
            synapse_core::OwnedDecodeRefusal::RetainedKvUnavailable.as_str(),
            "the retained KV state belongs to a different session",
        );
    }
    let position = session
        .retained_position
        .expect("retained session has a position");
    if let Some(req_id) = params.req_id.as_deref() {
        let prefix = match sessions.streams.continuation_prefix(
            &params.session_id,
            req_id,
            &params.retained_kv_session_id,
            synapse_core::ArtifactServingState::Approved,
        ) {
            Ok(prefix) => prefix,
            Err(error) => {
                return owned_decode_failure(
                    &state,
                    synapse_core::OwnedDecodeRefusal::RetainedKvUnavailable.as_str(),
                    error.to_string(),
                )
            }
        };
        if prefix.retained_position != position {
            return owned_decode_failure(
                &state,
                synapse_core::OwnedDecodeRefusal::RetainedKvUnavailable.as_str(),
                "the retained stream prefix does not match the session snapshot",
            );
        }
    }
    let route = match sessions
        .residency
        .as_ref()
        .expect("admitted decode session has residency")
        .route_continuation(session.routing_session_id, position)
    {
        Ok(route) => route,
        Err(refusal) => {
            let wire = map_admission_refusal(&refusal);
            return owned_decode_failure(&state, wire.as_str(), refusal.to_string());
        }
    };
    result_outcome(json!({
        "session_id": params.session_id,
        "retained_kv_session_id": params.retained_kv_session_id,
        "retained_position": route.retained_prefix_tokens,
        "reused_blocks": route.reused_blocks,
    }))
}

async fn owned_decode_session_abort(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let params: OwnedDecodeSessionAbortParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid owned_decode.abort params: {error}"),
            )
        }
    };
    let mut sessions = match state.runtime.owned_decode_sessions.lock() {
        Ok(sessions) => sessions,
        Err(_) => {
            return owned_decode_failure(
                &state,
                "owned_decode_unavailable",
                "decode sessions are unavailable",
            )
        }
    };
    let Some(session) = sessions.sessions.get(&params.session_id) else {
        return owned_decode_failure(
            &state,
            "unknown_session",
            "the decode session does not exist",
        );
    };
    if session.active_request.as_deref() != Some(&params.req_id) {
        return owned_decode_failure(
            &state,
            synapse_core::OwnedDecodeRefusal::SessionStillInFlight.as_str(),
            "the request is not an active decode in this session",
        );
    }
    let op_id = session_request_key(&params.session_id, &params.req_id);
    let cancellation = sessions.scheduler.request_cancel(&op_id, now_ms());
    sessions.pending_aborts.insert(
        (params.session_id.clone(), params.req_id.clone()),
        PendingSessionAbort {
            retain_kv: params.retain_kv,
        },
    );
    let disposition = match cancellation {
        owned_decode_grammar_scheduler::scheduler::CancelResult::RemovedQueued => "removed_queued",
        owned_decode_grammar_scheduler::scheduler::CancelResult::DeferredToBoundary => {
            "deferred_to_boundary"
        }
        owned_decode_grammar_scheduler::scheduler::CancelResult::NotFound => "deferred_to_boundary",
    };
    result_outcome(json!({
        "session_id": params.session_id,
        "req_id": params.req_id,
        "abort": "requested",
        "disposition": disposition,
    }))
}

async fn owned_decode_session_status(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let params: OwnedDecodeSessionStatusParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid owned_decode.session_status params: {error}"),
            )
        }
    };
    let sessions = match state.runtime.owned_decode_sessions.lock() {
        Ok(sessions) => sessions,
        Err(_) => {
            return owned_decode_failure(
                &state,
                "owned_decode_unavailable",
                "decode sessions are unavailable",
            )
        }
    };
    let Some(session) = sessions.sessions.get(&params.session_id) else {
        return owned_decode_failure(
            &state,
            "unknown_session",
            "the decode session does not exist",
        );
    };
    if let Ok(status) = sessions
        .streams
        .session_status(&params.session_id, &params.req_id)
    {
        return result_outcome(serde_json::to_value(status).expect("session status serializes"));
    }
    if session.active_request.as_deref() == Some(&params.req_id) {
        return result_outcome(json!({
            "session_id": params.session_id,
            "req_id": params.req_id,
            "committed_token_count": 0,
            "state": "in_flight",
        }));
    }
    owned_decode_failure(
        &state,
        "unknown_request",
        "the request is not registered for this session",
    )
}

#[derive(Debug)]
struct OwnedDecodeWorkerStream {
    generated_token_ids: Vec<u32>,
    identity: synapse_core::OneshotEnvelopeIdentity,
    generation_id: String,
}

fn required_string(value: &Value, field: &str) -> Result<String, String> {
    value
        .as_str()
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("owned worker response is missing {field}"))
}

fn parse_owned_decode_worker_stream(body: &[u8]) -> Result<OwnedDecodeWorkerStream, String> {
    let body: Value = serde_json::from_slice(body)
        .map_err(|error| format!("owned worker response is not JSON: {error}"))?;
    let result = body
        .get("result")
        .ok_or_else(|| "owned worker response has no result".to_string())?;
    let provenance = result
        .get("provenance")
        .ok_or_else(|| "owned worker response has no provenance".to_string())?;
    if required_string(
        provenance.get("lane").unwrap_or(&Value::Null),
        "provenance.lane",
    )? != "decode"
        || required_string(
            provenance.get("worker").unwrap_or(&Value::Null),
            "provenance.worker",
        )? != "supervised"
    {
        return Err(
            "owned session execution did not stay on the supervised decode lane".to_string(),
        );
    }
    let generated_token_ids = result
        .get("generated_token_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| "owned worker response has no generated token IDs".to_string())?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| "owned worker response has an invalid token ID".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let n_gen = result
        .get("n_gen")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| "owned worker response has no generation count".to_string())?;
    if n_gen != generated_token_ids.len() {
        return Err("owned worker generation count disagrees with committed token IDs".to_string());
    }
    let decode_fingerprint = required_string(
        provenance.get("decode_fingerprint").unwrap_or(&Value::Null),
        "provenance.decode_fingerprint",
    )?;
    let processing_fingerprint = required_string(
        provenance
            .get("processing_fingerprint")
            .unwrap_or(&Value::Null),
        "provenance.processing_fingerprint",
    )?;
    let runtime_config_digest = required_string(
        result.get("runtime_config_digest").unwrap_or(&Value::Null),
        "runtime_config_digest",
    )?;
    let worker_generation = provenance
        .get("worker_generation")
        .and_then(Value::as_u64)
        .ok_or_else(|| "owned worker response has no worker generation".to_string())?;
    let generation_id = required_string(
        result.get("generation_id").unwrap_or(&Value::Null),
        "generation_id",
    )?;
    let derived_digest = result
        .get("derived_digest")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    Ok(OwnedDecodeWorkerStream {
        generated_token_ids,
        identity: synapse_core::OneshotEnvelopeIdentity {
            decode_fingerprint: synapse_core::Fingerprint(decode_fingerprint),
            processing_fingerprint: synapse_core::Fingerprint(processing_fingerprint),
            runtime_config_digest,
            worker_generation,
            derived_digest,
        },
        generation_id,
    })
}

fn clear_owned_decode_request(state: &ModuleState, session_id: &str, req_id: &str) {
    if let Ok(mut sessions) = state.runtime.owned_decode_sessions.lock() {
        let op_id = session_request_key(session_id, req_id);
        sessions.scheduler.remove_op(&op_id);
        sessions
            .pending_aborts
            .remove(&(session_id.to_string(), req_id.to_string()));
        if let Some(session) = sessions.sessions.get_mut(session_id) {
            if session.active_request.as_deref() == Some(req_id) {
                session.active_request = None;
            }
        }
    }
}

/// Consume an abort at a committed session boundary.
///
/// This is called after every progress frame and again after the progress loop because a
/// zero-token worker result has no frame boundary at which to observe a pending abort.
fn take_pending_session_abort(
    state: &ModuleState,
    sessions: &mut OwnedDecodeWireState,
    session_id: &str,
    req_id: &str,
    committed: u32,
    op_id: &str,
    frames: &mut Vec<synapse_core::FrameEnvelope>,
) -> Option<HandlerOutcome> {
    let abort = sessions
        .pending_aborts
        .remove(&(session_id.to_string(), req_id.to_string()))?;
    let retention = if abort.retain_kv {
        let retained_kv_session_id = format!(
            "{session_id}:retained:{committed}:{}",
            sessions.next_session_sequence
        );
        sessions.next_session_sequence = sessions.next_session_sequence.saturating_add(1);
        let catalog_fingerprint = sessions
            .sessions
            .get(session_id)
            .expect("active session exists")
            .catalog_fingerprint
            .clone();
        match state.store.retain_serving_state(
            &retained_kv_session_id,
            &catalog_fingerprint,
            now_ms(),
        ) {
            Ok(store::ServingContinuationAdmission::Admitted { .. }) => {
                owned_decode_worker::RetentionPreflight::Ready {
                    retained_kv_session_id,
                    retained_position: committed,
                }
            }
            Ok(store::ServingContinuationAdmission::Refused { .. }) | Err(_) => {
                owned_decode_worker::RetentionPreflight::Refused
            }
        }
    } else {
        owned_decode_worker::RetentionPreflight::NotRequested
    };
    match sessions.streams.abort(session_id, req_id, retention, None) {
        Ok(outcome) => {
            if let Some(prefix) = outcome.retained_prefix {
                if let Some(session) = sessions.sessions.get_mut(session_id) {
                    session.retained_kv_session_id = Some(prefix.retained_kv_session_id);
                    session.retained_position = Some(prefix.retained_position);
                }
            }
            frames.push(outcome.terminal);
            sessions.scheduler.remove_op(op_id);
            if let Some(session) = sessions.sessions.get_mut(session_id) {
                session.active_request = None;
            }
            Some(result_outcome(json!({
                "session_id": session_id,
                "req_id": req_id,
                "cancelled": outcome.cancellation,
                "frames": frames,
            })))
        }
        Err(error) => Some(owned_decode_failure(
            state,
            "worker_protocol_error",
            error.to_string(),
        )),
    }
}

async fn owned_decode_session_decode(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let params: OwnedDecodeSessionDecodeParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid owned_decode.decode params: {error}"),
            )
        }
    };
    if params.req_id.trim().is_empty() || params.max_tokens == 0 {
        return channel_error(
            "invalid_request",
            "owned_decode.decode requires a non-empty req_id and positive max_tokens",
        );
    }
    let model_id = {
        let mut sessions = match state.runtime.owned_decode_sessions.lock() {
            Ok(sessions) => sessions,
            Err(_) => {
                return owned_decode_failure(
                    &state,
                    "owned_decode_unavailable",
                    "decode sessions are unavailable",
                )
            }
        };
        let Some(session) = sessions.sessions.get(&params.session_id) else {
            return owned_decode_failure(
                &state,
                "unknown_session",
                "the decode session does not exist",
            );
        };
        if session.is_closed() || session.active_request.is_some() {
            return owned_decode_failure(
                &state,
                synapse_core::OwnedDecodeRefusal::SessionStillInFlight.as_str(),
                "the decode session is not idle",
            );
        }
        let model_id = session.model_id.clone();
        let op_id = session_request_key(&params.session_id, &params.req_id);
        let admitted_at_ms = now_ms();
        sessions
            .scheduler
            .admit_decode(owned_decode_grammar_scheduler::scheduler::DecodeOp {
                op_id: op_id.clone(),
                generation_id: params.req_id.clone(),
                admitted_at_ms,
                anchor_ms: admitted_at_ms,
                committed_tokens: 0,
                max_tokens: params.max_tokens,
                resident: false,
                cancelled_at_ms: None,
                deadline_at_ms: params
                    .deadline_ms
                    .and_then(|deadline| admitted_at_ms.checked_add(deadline)),
            });
        let selected = sessions.scheduler.arbitrate(admitted_at_ms);
        if selected
            .as_ref()
            .and_then(|selected| selected.op_id.as_deref())
            != Some(op_id.as_str())
        {
            sessions.scheduler.remove_op(&op_id);
            return owned_decode_failure(
                &state,
                "decode_scheduler_busy",
                "decode work was not selected at a committed scheduler boundary",
            );
        }
        sessions
            .sessions
            .get_mut(&params.session_id)
            .expect("idle session remains registered")
            .active_request = Some(params.req_id.clone());
        model_id
    };

    let route_params = MicroLlmOneshotParams {
        model: Some(model_id),
        prompt: params.prompt.clone(),
        max_tokens: params.max_tokens,
        grammar: params.grammar.clone(),
        deadline_ms: params.deadline_ms,
        max_queue_ms: params.max_queue_ms,
        target_fingerprint: None,
        required_fingerprint: None,
        allow_equivalent: false,
        required_epoch: None,
        _accept_declared: false,
        owned_only: true,
    };
    let worker_stream = match route_owned_decode_wire(
        Arc::clone(&state),
        &route_params,
        route_params.model.as_deref().expect("session model is set"),
        params.req_id.clone(),
        Instant::now(),
    )
    .await
    {
        HandlerOutcome::Response(body) => match parse_owned_decode_worker_stream(&body) {
            Ok(stream) => stream,
            Err(error) => {
                clear_owned_decode_request(&state, &params.session_id, &params.req_id);
                return owned_decode_failure(&state, "worker_protocol_error", error);
            }
        },
        outcome => {
            clear_owned_decode_request(&state, &params.session_id, &params.req_id);
            return outcome;
        }
    };
    if worker_stream.generated_token_ids.len() > params.max_tokens as usize {
        clear_owned_decode_request(&state, &params.session_id, &params.req_id);
        return owned_decode_failure(
            &state,
            "worker_protocol_error",
            "worker exceeded the admitted generation limit",
        );
    }
    let committed_before_request = match state.store.serving_session(&params.session_id) {
        Ok(Some(session)) => session.committed_token_count,
        Ok(None) => {
            clear_owned_decode_request(&state, &params.session_id, &params.req_id);
            return owned_decode_failure(
                &state,
                "store_failure",
                "the durable decode session does not exist",
            );
        }
        Err(error) => {
            clear_owned_decode_request(&state, &params.session_id, &params.req_id);
            return owned_decode_failure(
                &state,
                "store_failure",
                format!("could not read the decode session boundary: {error}"),
            );
        }
    };

    let mut sessions = match state.runtime.owned_decode_sessions.lock() {
        Ok(sessions) => sessions,
        Err(_) => {
            return owned_decode_failure(
                &state,
                "owned_decode_unavailable",
                "decode sessions are unavailable",
            )
        }
    };
    let op_id = session_request_key(&params.session_id, &params.req_id);
    let Some(session) = sessions.sessions.get(&params.session_id) else {
        sessions.scheduler.remove_op(&op_id);
        return owned_decode_failure(
            &state,
            "unknown_session",
            "the decode session closed before the worker responded",
        );
    };
    if session.active_request.as_deref() != Some(&params.req_id) {
        sessions.scheduler.remove_op(&op_id);
        return owned_decode_failure(
            &state,
            "request_cancelled",
            "the decode request was removed before the worker responded",
        );
    }
    let grammar_constrained = params
        .grammar
        .as_deref()
        .is_some_and(|grammar| !grammar.trim().is_empty());
    let chain_k = if grammar_constrained {
        1
    } else {
        state.runtime.decode_chain_k
    };
    let request = owned_decode_worker::StreamRequest {
        req_id: params.req_id.clone(),
        session_id: params.session_id.clone(),
        generation_id: worker_stream.generation_id.clone(),
        identity: worker_stream.identity.clone(),
        decode_mode: synapse_core::DecodeMode::Serial,
        grammar_constrained,
        chain_k,
    };
    if let Err(error) = sessions.streams.begin(request) {
        sessions.scheduler.remove_op(&op_id);
        if let Some(session) = sessions.sessions.get_mut(&params.session_id) {
            session.active_request = None;
        }
        return owned_decode_failure(&state, "worker_protocol_error", error.to_string());
    }

    let mut frames = Vec::new();
    let mut committed = 0_u32;
    let mut sequence = synapse_core::StreamSequence::FIRST.0;
    let quantum = sessions.scheduler.config().production_n as usize;
    let chunks = worker_stream
        .generated_token_ids
        .chunks(quantum)
        .collect::<Vec<_>>();
    for (index, token_ids) in chunks.iter().enumerate() {
        committed = committed.saturating_add(token_ids.len() as u32);
        if let Err(error) = sessions
            .scheduler
            .try_commit_quantum(&op_id, committed, now_ms())
        {
            sessions.scheduler.remove_op(&op_id);
            if let Some(session) = sessions.sessions.get_mut(&params.session_id) {
                session.active_request = None;
            }
            return owned_decode_failure(
                &state,
                "scheduler_boundary_error",
                format!("scheduler boundary rejected the committed span: {error:?}"),
            );
        }
        match state.store.commit_serving_session_boundary(
            &params.session_id,
            absolute_serving_boundary(committed_before_request, committed),
            now_ms(),
        ) {
            Ok(store::ServingBoundaryOutcome::Continue { .. }) => {}
            Ok(store::ServingBoundaryOutcome::Terminated {
                unload_artifact, ..
            }) => {
                // Publish this progress before the terminal so status and stream
                // history agree on every token that crossed the boundary.
                //
                // Note what the prefix is, because it is easy to read the wrong
                // way: the store did NOT commit it before revocation. The same
                // transaction that observes `Revoked` also writes the new
                // committed count and then returns `Terminated`
                // (`store.rs:3905-3933`), so this prefix is the quantum
                // generated after revocation and committed by the transaction
                // that saw it. That is the documented boundary behaviour -- a
                // revoke fences admission and truncates emission at the next
                // COMMITTED boundary, not mid-quantum -- and publishing it is
                // correct precisely because the store already counted it.
                let progress = synapse_core::FrameEnvelope::new(
                    &params.req_id,
                    &params.session_id,
                    synapse_core::StreamSequence(sequence),
                    synapse_core::WorkerFrame::Progress {
                        progress: synapse_core::ProgressFrame {
                            committed_token_ids: token_ids.to_vec(),
                            committed_token_count: committed,
                            boundary: synapse_core::ProgressBoundary::Continuing,
                        },
                    },
                );
                if let Err(error) = sessions.streams.observe_frame(&progress) {
                    return owned_decode_failure(
                        &state,
                        "worker_protocol_error",
                        error.to_string(),
                    );
                }
                frames.push(progress);
                sequence = sequence.saturating_add(1);
                let terminal = synapse_core::TerminalEnvelope {
                    req_id: params.req_id.clone(),
                    session_id: params.session_id.clone(),
                    committed_token_count: committed,
                    tokens_emitted: committed,
                    identity: worker_stream.identity.clone(),
                    terminal_state: synapse_core::TerminalState::ArtifactRevoked,
                    decode_mode: synapse_core::DecodeMode::Serial,
                    speculative_telemetry: None,
                };
                let frame = synapse_core::FrameEnvelope::new(
                    &params.req_id,
                    &params.session_id,
                    synapse_core::StreamSequence(sequence),
                    synapse_core::WorkerFrame::Error { terminal },
                );
                if let Err(error) = sessions.streams.observe_frame(&frame) {
                    return owned_decode_failure(
                        &state,
                        "worker_protocol_error",
                        error.to_string(),
                    );
                }
                frames.push(frame);
                sessions.scheduler.remove_op(&op_id);
                let routing_session_id = sessions
                    .sessions
                    .get(&params.session_id)
                    .expect("revoked session remains registered")
                    .routing_session_id;
                if let Some(router) = sessions.residency.as_mut() {
                    let _ = router.close_session(routing_session_id);
                }
                let model_id = sessions
                    .sessions
                    .get(&params.session_id)
                    .expect("revoked session remains registered")
                    .model_id
                    .clone();
                if let Some(session) = sessions.sessions.get_mut(&params.session_id) {
                    session.active_request = None;
                    session.closed_at_ms = Some(now_ms());
                }
                if unload_artifact {
                    if let Ok(mut dispatches) = state.runtime.owned_decode_dispatches.lock() {
                        dispatches.remove(&model_id);
                    }
                }
                return result_outcome(json!({
                    "session_id": params.session_id,
                    "req_id": params.req_id,
                    "frames": frames,
                }));
            }
            Err(error) => {
                sessions.scheduler.remove_op(&op_id);
                if let Some(session) = sessions.sessions.get_mut(&params.session_id) {
                    session.active_request = None;
                }
                return owned_decode_failure(
                    &state,
                    "store_failure",
                    format!("could not commit the decode boundary: {error}"),
                );
            }
        }
        let frame = synapse_core::FrameEnvelope::new(
            &params.req_id,
            &params.session_id,
            synapse_core::StreamSequence(sequence),
            synapse_core::WorkerFrame::Progress {
                progress: synapse_core::ProgressFrame {
                    committed_token_ids: token_ids.to_vec(),
                    committed_token_count: committed,
                    boundary: synapse_core::ProgressBoundary::Continuing,
                },
            },
        );
        if let Err(error) = sessions.streams.observe_frame(&frame) {
            return owned_decode_failure(&state, "worker_protocol_error", error.to_string());
        }
        frames.push(frame);
        sequence = sequence.saturating_add(1);

        if let Some(outcome) = take_pending_session_abort(
            &state,
            &mut sessions,
            &params.session_id,
            &params.req_id,
            committed,
            &op_id,
            &mut frames,
        ) {
            return outcome;
        }
        if index + 1 < chunks.len() {
            sessions.scheduler.requeue_continuation(&op_id);
            let selected = sessions.scheduler.arbitrate(now_ms());
            if selected
                .as_ref()
                .and_then(|selected| selected.op_id.as_deref())
                != Some(op_id.as_str())
            {
                sessions.scheduler.remove_op(&op_id);
                if let Some(session) = sessions.sessions.get_mut(&params.session_id) {
                    session.active_request = None;
                }
                return owned_decode_failure(
                    &state,
                    "decode_scheduler_busy",
                    "a later decode quantum lost scheduler ownership",
                );
            }
        }
    }
    // A zero-token worker result has no progress loop to detect an abort requested
    // before completion, so apply abort handling at the already-committed boundary.
    if let Some(outcome) = take_pending_session_abort(
        &state,
        &mut sessions,
        &params.session_id,
        &params.req_id,
        committed,
        &op_id,
        &mut frames,
    ) {
        return outcome;
    }
    let terminal = synapse_core::TerminalEnvelope {
        req_id: params.req_id.clone(),
        session_id: params.session_id.clone(),
        committed_token_count: committed,
        tokens_emitted: committed,
        identity: worker_stream.identity,
        terminal_state: synapse_core::TerminalState::Completed,
        decode_mode: synapse_core::DecodeMode::Serial,
        speculative_telemetry: None,
    };
    let frame = synapse_core::FrameEnvelope::new(
        &params.req_id,
        &params.session_id,
        synapse_core::StreamSequence(sequence),
        synapse_core::WorkerFrame::Final { terminal },
    );
    if let Err(error) = sessions.streams.observe_frame(&frame) {
        return owned_decode_failure(&state, "worker_protocol_error", error.to_string());
    }
    frames.push(frame);
    sessions.scheduler.remove_op(&op_id);
    if let Some(session) = sessions.sessions.get_mut(&params.session_id) {
        session.active_request = None;
    }
    result_outcome(json!({
        "session_id": params.session_id,
        "req_id": params.req_id,
        "frames": frames,
    }))
}

async fn owned_decode_session_close(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let params: OwnedDecodeSessionCloseParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid owned_decode.close params: {error}"),
            )
        }
    };
    let mut sessions = match state.runtime.owned_decode_sessions.lock() {
        Ok(sessions) => sessions,
        Err(_) => {
            return owned_decode_failure(
                &state,
                "owned_decode_unavailable",
                "decode sessions are unavailable",
            )
        }
    };
    let Some(session) = sessions.sessions.get(&params.session_id).cloned() else {
        return owned_decode_failure(
            &state,
            "unknown_session",
            "the decode session does not exist",
        );
    };
    if session.active_request.is_some() {
        return owned_decode_failure(
            &state,
            synapse_core::OwnedDecodeRefusal::SessionStillInFlight.as_str(),
            "an active decode must stop at a boundary before close",
        );
    }
    let mut unload_artifact = false;
    if !session.is_closed() {
        match state
            .store
            .complete_serving_session(&params.session_id, now_ms())
        {
            Ok(completion) => unload_artifact = completion.unload_artifact,
            Err(error) => {
                return owned_decode_failure(
                    &state,
                    "store_failure",
                    format!("could not complete serving session: {error}"),
                );
            }
        }
        if let Some(router) = sessions.residency.as_mut() {
            if let Err(error) = router.close_session(session.routing_session_id) {
                return owned_decode_failure(&state, "routing_error", error.to_string());
            }
        }
        if let Some(session) = sessions.sessions.get_mut(&params.session_id) {
            session.closed_at_ms = Some(now_ms());
        }
    }
    drop(sessions);
    if unload_artifact {
        if let Ok(mut dispatches) = state.runtime.owned_decode_dispatches.lock() {
            dispatches.remove(&session.model_id);
        }
    }
    result_outcome(json!({
        "session_id": params.session_id,
        "closed": true,
        "unload_artifact": unload_artifact,
    }))
}

async fn owned_decode_disable(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let params: OwnedDecodeServingControlParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid owned_decode.disable params: {error}"),
            )
        }
    };
    match state
        .store
        .disable_serving_catalog(&params.catalog_fingerprint, &params.reason, now_ms())
    {
        Ok(outcome) => {
            // The store's certification record is the authority for which model
            // a catalog serves. The in-memory session map is not: closed
            // sessions are evicted after a retention window, the map starts
            // empty after a module restart, and certification can load a worker
            // before any session is admitted.
            if outcome.unload_artifact {
                if let Some(model_id) = serving_catalog_model_id(&state, &outcome.approval) {
                    if let Ok(mut dispatches) = state.runtime.owned_decode_dispatches.lock() {
                        dispatches.remove(&model_id);
                    }
                }
            }
            result_outcome(
                serde_json::to_value(outcome).expect("serving disable outcome serializes"),
            )
        }
        Err(error) => owned_decode_failure(
            &state,
            "store_failure",
            format!("could not disable serving artifact: {error}"),
        ),
    }
}

async fn owned_decode_revoke(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let params: OwnedDecodeServingControlParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid owned_decode.revoke params: {error}"),
            )
        }
    };
    let outcome = match state.store.revoke_serving_catalog(
        &params.catalog_fingerprint,
        &params.reason,
        now_ms(),
    ) {
        Ok(outcome) => outcome,
        Err(error) => {
            return owned_decode_failure(
                &state,
                "store_failure",
                format!("could not revoke serving artifact: {error}"),
            )
        }
    };
    if let Ok(mut sessions) = state.runtime.owned_decode_sessions.lock() {
        let active_ops = sessions
            .sessions
            .iter()
            .filter_map(|(session_id, session)| {
                (session.catalog_fingerprint == params.catalog_fingerprint)
                    .then_some((session_id.clone(), session.active_request.clone()))
            })
            .collect::<Vec<_>>();
        for (session_id, req_id) in active_ops {
            if let Some(req_id) = req_id {
                sessions
                    .scheduler
                    .request_revoke(&session_request_key(&session_id, &req_id), now_ms());
            }
        }
    }
    if outcome.unload_artifact {
        if let Ok(mut dispatches) = state.runtime.owned_decode_dispatches.lock() {
            dispatches.clear();
        }
    }
    result_outcome(serde_json::to_value(outcome).expect("serving revoke outcome serializes"))
}

#[derive(Clone)]
struct ResolvedModelLoadAsset {
    source_url: String,
    expected_digest: Option<String>,
}

#[derive(Clone)]
struct ResolvedModelLoadSources {
    model: ResolvedModelLoadAsset,
    tokenizer: ResolvedModelLoadAsset,
    config: Option<ResolvedModelLoadAsset>,
    extra: Vec<ResolvedModelLoadAsset>,
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
    if params
        .model_id
        .as_deref()
        .is_some_and(|id| state.runtime.release_catalog.is_reserved_id(id))
    {
        return result_outcome(error_payload(
            &state,
            catalog_wire_error(
                "invalid_request",
                json!({"model_id": params.model_id}),
                "catalog ids must be installed with models.download",
            ),
        ));
    }
    if let Err(message) = validate_model_load_request(&params).and_then(|_| {
        resolve_model_load_sources_at(&params, &state.runtime.hf_endpoint).map(|_| ())
    }) {
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
    let request_digest =
        compute_request_digest("model.load", "management", None, None, &params_json, &[]);
    let admission = match state.store.admit_job(
        &request_key,
        &request_digest,
        "model.load",
        state.module_generation,
        None,
        &params_json,
        now,
        state.runtime.jobs.execution_ttl_ms,
        state.runtime.jobs.result_retention_ttl_ms,
    ) {
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
        if let Some(model_id) = params.model_id.as_deref() {
            log_job_admitted(model_id, &record.job_id);
        }
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
            Ok(Some(record)) if record.kind == store::DOWNLOAD_JOB_KIND => {
                result_outcome(download_status_payload(&state, &record, true))
            }
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
            ModelRuntimeState::Loading
                | ModelRuntimeState::Resolving
                | ModelRuntimeState::Validating
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
    let mut catalog_guard = if catalog_lane {
        match lane_lock.try_lock_owned() {
            Ok(guard) => Some(guard),
            Err(_) => {
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
    let Some(snapshot) = model_slot_snapshot(&state.runtime, &params.model_id) else {
        return result_outcome(error_payload(&state, catalog_unknown(&params.model_id)));
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
        let invocation_guard = catalog_guard.take();
        let unload = tokio::task::spawn_blocking(move || {
            let _invocation_guard = invocation_guard;
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
    if let Ok(mut dispatches) = state.runtime.owned_decode_dispatches.lock() {
        dispatches.remove(&params.model_id);
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

/// Timeout ceiling for control-path model-load waits. Matches the worker
/// load timeout (DEFAULT_WORKER_LOAD_TIMEOUT_MS) as the ANE precedent.
const MODEL_LOAD_CONTROL_TIMEOUT_MS: u64 = DEFAULT_WORKER_LOAD_TIMEOUT_MS;

async fn ensure_model_loaded_for_control(
    state: Arc<ModuleState>,
    model_id: &str,
    request_deadline_ms: Option<u64>,
) -> Result<Arc<EmbeddingModel>, WireOperationError> {
    let timeout_ms = request_deadline_ms.unwrap_or(MODEL_LOAD_CONTROL_TIMEOUT_MS);
    let deadline = tokio::time::Instant::now() + Duration::from_millis(timeout_ms);
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
    let model_cache = Arc::clone(&state.model_cache);
    let load_started = std::time::Instant::now();
    let microllm_max_tokens = state.runtime.microllm_max_tokens;
    let worker_load_timeout = state.runtime.worker_load_timeout;
    let worker_forward_lines_per_sec = state.runtime.log.worker_forward_lines_per_sec;
    let owned_decode_q8 = Arc::clone(&state.runtime.owned_decode_q8);
    let ane_supervisor = direct_ane_supervisor(&state.runtime, &spec.engine);
    let is_catalog = state.runtime.release_catalog.is_reserved_id(&model_id);
    let fault_lane = model_id.clone();
    let loaded = tokio::task::spawn_blocking(move || {
        if is_catalog {
            catalog_call_fault(&fault_lane, "load")?;
        }
        load_catalog_model_blocking(
            spec,
            model_cache,
            microllm_max_tokens,
            worker_load_timeout,
            worker_forward_lines_per_sec,
            owned_decode_q8,
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

fn load_catalog_model_blocking(
    spec: StoredModelConfig,
    model_cache: Arc<ModelCache>,
    microllm_max_tokens: u32,
    worker_load_timeout: Duration,
    worker_forward_lines_per_sec: u32,
    owned_decode_q8: Arc<Mutex<owned_decode_routing::q8ingest::Q8IngestRegistry>>,
    ane_supervisor: Option<worker_host::ane_residency::AneResidencySupervisor>,
) -> Result<EmbeddingModel, WireOperationError> {
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
    let extra_assets = spec
        .extra_locators
        .iter()
        .map(|locator| locator_path(locator, &model_cache))
        .collect::<Result<Vec<_>, _>>()?;
    let ane_artifacts = if spec.engine == "ane"
        && cfg!(target_os = "macos")
        && matches!(spec.artifact_format.as_str(), "mlmodelc" | "coreml")
    {
        Some(materialize_ane_artifacts(
            &spec,
            &model_path,
            &extra_assets,
            model_cache.root(),
        )?)
    } else {
        None
    };
    let effective_model_path = if let Some(artifacts) = ane_artifacts.as_ref() {
        artifacts
            .first()
            .expect("ANE artifact set always contains the primary model")
            .path
            .clone()
    } else if spec.engine_identity.build_flags.contains_key("profile") {
        model_path.path.clone()
    } else if let Some(profile) = owned_profile.as_ref() {
        assemble_owned_model_package(&spec, &model_path.path, &model_cache, profile)?
    } else {
        model_path.path.clone()
    };
    let tokenizer = SanitizedTokenizer::from_file(
        &tokenizer_path.path,
        TokenizerConfig {
            max_tokens: if spec.engine_identity.build_flags.contains_key("profile") {
                usize::MAX
            } else {
                owned_tokenizer_max_tokens(spec.max_tokens, owned_profile.as_ref())
            },
        },
    )
    .map_err(|error| artifact_invalid_error(error.to_string()))?;
    if let Some(id) = spec.engine_identity.build_flags.get("profile") {
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
    if spec.engine == "owned-metal-decode" {
        let entry = owned_decode_catalog_entry(&spec)
            .map_err(|error| artifact_invalid_error(error.as_str()))?;
        ingest_owned_decode_q8(
            &entry,
            &model_path.path,
            model_cache.root(),
            &owned_decode_q8,
        )?;
    }
    let runtime_config = model_runtime_config(
        &spec,
        &effective_model_path,
        &extra_assets,
        model_cache.root(),
        microllm_max_tokens,
        ane_artifacts.as_deref(),
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
                worker_load_timeout,
                worker_forward_lines_per_sec,
                ane_supervisor,
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
        // An owned-decode catalog entry has platform-independent identity data.
        // Resolve that identity on every target; the platform gate is a typed
        // routing refusal so a substitutable request can select llama instead.
        "owned-metal-decode" => (
            EmbedBackend::OwnedDecode,
            LoadedModel {
                model_id: format!("owned-decode:{}", spec.model_id),
            },
            None,
        ),
        LLAMA_ENGINE | "ane" | "owned-vulkan" | "ane-direct-worker" => {
            let (backend, loaded) = load_worker_backend_blocking(
                &spec,
                &artifact,
                &runtime_config,
                worker_load_timeout,
                worker_forward_lines_per_sec,
                ane_supervisor,
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
    let certification_fingerprint = if spec.engine == "owned-metal-decode" {
        owned_decode_catalog_entry(&spec)
            .and_then(|entry| entry.decode_identity_inputs().decode_fingerprint())
            .map_err(|error| artifact_invalid_error(error.as_str()))?
    } else {
        spec.fingerprint.clone()
    };
    Ok(EmbeddingModel {
        model_id: spec.model_id.clone(),
        task,
        loaded_model,
        backend,
        tokenizer,
        numeric_profile_id: spec.numeric_profile_id.clone(),
        fingerprint: spec.fingerprint.clone(),
        certification_fingerprint,
        engine_identity: spec.engine_identity.clone(),
        owned_tokenizer_policy,
        owned_decode_resolution_refusal: owned_decode_resolution_refusal(&spec),
    })
}

fn ingest_owned_decode_q8(
    entry: &owned_decode_routing::CatalogEntry,
    source_path: &Path,
    cache_root: &Path,
    registry: &Arc<Mutex<owned_decode_routing::q8ingest::Q8IngestRegistry>>,
) -> Result<(), WireOperationError> {
    use owned_decode_routing::{error::OwnedDecodeError, q8artifact::derive_and_cache_q8_blocks};

    let Some(identity) = entry.q8.as_ref() else {
        return Ok(());
    };
    let configured_cache_root = env::var_os("SYNAPSE_OWNED_DECODE_Q8_CACHE_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| cache_root.to_path_buf());
    let artifact = derive_and_cache_q8_blocks(
        source_path,
        &configured_cache_root,
        entry.family,
        &entry.artifact_source_digest,
        &identity.quantizer_revision,
    )
    .map_err(|error| artifact_invalid_error(format!("derive Q8 artifact: {error:#}")))?;
    let mut registry = registry.lock().map_err(|_| {
        WireOperationError::from_stable(
            StableError::engine_crashed(Some(100)),
            "owned-decode Q8 ingest registry mutex was poisoned",
        )
    })?;
    registry.register_expected_digest(
        &entry.artifact_source_digest,
        &identity.quantizer_revision,
        &identity.derived_digest,
    );
    match registry.load_or_ingest(
        &entry.artifact_source_digest,
        &identity.quantizer_revision,
        "q8_0",
        &[],
        |_| artifact.derived_digest,
    ) {
        Ok(_) | Err(OwnedDecodeError::ArtifactPoisoned | OwnedDecodeError::NotCertified) => Ok(()),
        Err(error) => Err(artifact_invalid_error(format!(
            "Q8 ingest failed: {}",
            error.as_str()
        ))),
    }
}

/// Resolve a supervised worker binary when the spec and the engine-specific
/// env var are both absent: look for the worker shipped beside the running
/// module binary (release archives unpack every binary at the archive root).
/// Returns `None` when no sibling exists, so the existing refusal path keeps
/// its message.
fn resolve_worker_binary_sibling(engine: &str) -> Option<PathBuf> {
    let file_name = worker_binary_file_name(engine)?;
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    let candidate = dir.join(file_name);
    #[cfg(windows)]
    let candidate = candidate.with_extension("exe");
    candidate.is_file().then_some(candidate)
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
            let mut supervisor = worker_host::ane_residency::AneResidencySupervisor::acquire_lane(
                Default::default(),
            )
            .map_err(ane_residency_error_to_wire)?;
            if runtime.certify_observation {
                supervisor.enable_observations();
            }
            *shared = Some(supervisor);
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

fn load_worker_backend_blocking(
    spec: &StoredModelConfig,
    artifact: &ValidatedArtifact,
    runtime_config: &RuntimeConfig,
    worker_load_timeout: Duration,
    worker_forward_lines_per_sec: u32,
    ane_supervisor: Option<worker_host::ane_residency::AneResidencySupervisor>,
) -> Result<(EmbedBackend, LoadedModel), WireOperationError> {
    use worker_host::{WorkerEngine, WorkerHostConfig};

    if matches!(spec.engine.as_str(), "ane" | "ane-direct-worker") && !cfg!(target_os = "macos") {
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
    config.load_timeout = worker_load_timeout;
    config.worker_id = format!("synapse-{}-{}", spec.engine, spec.model_id);
    config.model_id = Some(spec.model_id.clone());
    config.worker_forward_lines_per_sec = worker_forward_lines_per_sec;
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

fn materialize_ane_artifacts(
    spec: &StoredModelConfig,
    model: &LocatedAsset,
    extra_assets: &[LocatedAsset],
    cache_root: &Path,
) -> Result<Vec<ane_artifact::MaterializedCoreMlArtifact>, WireOperationError> {
    let assets = std::iter::once((&spec.model_locator, model))
        .chain(spec.extra_locators.iter().zip(extra_assets.iter()));
    let mut materialized = Vec::with_capacity(1 + extra_assets.len());
    for (locator, asset) in assets {
        let source_digest = model_asset_digest(locator, &spec.artifact_digest);
        let artifact =
            ane_artifact::materialize_core_ml_artifact(&asset.path, &source_digest, cache_root)
                .map_err(|error| {
                    artifact_invalid_error(format!(
                        "materialize ANE Core ML artifact {}: {error:#}",
                        asset.path.display()
                    ))
                })?;
        tracing::debug!(
            target: "maintenance",
            source = %asset.path.display(),
            materialized = %artifact.path.display(),
            source_digest,
            reused = artifact.reused,
            "ANE Core ML artifact ready"
        );
        materialized.push(artifact);
    }
    Ok(materialized)
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
        EmbedBackend::OwnedDecode => Ok(()),
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
    extra_assets: &[LocatedAsset],
    model_cache_root: &Path,
    microllm_max_tokens: u32,
    ane_artifacts: Option<&[ane_artifact::MaterializedCoreMlArtifact]>,
) -> RuntimeConfig {
    let mut runtime_config = RuntimeConfig::default();
    if let Some(profile) = spec.engine_identity.build_flags.get("profile") {
        runtime_config
            .values
            .insert("profile".into(), profile.clone());
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
    if spec.engine == "ane" {
        let (paths, digests) = if let Some(artifacts) = ane_artifacts {
            (
                artifacts
                    .iter()
                    .map(|artifact| artifact.path.to_string_lossy().to_string())
                    .collect(),
                artifacts
                    .iter()
                    .map(|artifact| artifact.digest.clone())
                    .collect(),
            )
        } else {
            let mut paths = vec![model_path.to_string_lossy().to_string()];
            paths.extend(
                extra_assets
                    .iter()
                    .map(|asset| asset.path.to_string_lossy().to_string()),
            );
            let mut digests = vec![model_asset_digest(
                &spec.model_locator,
                &spec.artifact_digest,
            )];
            digests.extend(
                spec.extra_locators
                    .iter()
                    .map(|locator| model_asset_digest(locator, &spec.artifact_digest)),
            );
            (paths, digests)
        };
        runtime_config.values.insert(
            "artifact_paths".to_string(),
            serde_json::to_string(&paths).expect("ANE artifact paths serialize"),
        );
        runtime_config.values.insert(
            "artifact_digests".to_string(),
            serde_json::to_string(&digests).expect("ANE artifact digests serialize"),
        );
    }
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
        &[],
        Path::new("/tmp/synapse-backend-contract-cache"),
        DEFAULT_MICROLLM_MAX_TOKENS,
        None,
    ))
}

struct LocatedAsset {
    path: PathBuf,
    _guard: Option<synapse_core::ModelCacheReadGuard>,
}

fn model_asset_digest(locator: &ModelAssetLocator, fallback: &str) -> String {
    match locator {
        ModelAssetLocator::CacheDigest { digest } => digest.clone(),
        ModelAssetLocator::LocalPath { .. } => fallback.to_string(),
    }
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

fn owned_cuda_floor_decision(worker: Option<&Path>) -> CudaFloorDecision {
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
    let packaging_driver = env::var("SYNAPSE_CUDA_PACKAGING_DRIVER").ok();
    let (Some(driver_api), Some((major, minor))) = (driver_api, compute) else {
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

type OwnedCudaProbeEntry = Arc<OnceLock<Result<OwnedCudaFloorReading, String>>>;

static OWNED_CUDA_PROBE: std::sync::LazyLock<Mutex<HashMap<PathBuf, OwnedCudaProbeEntry>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

/// Cache successes and failures per worker; a complete environment override skips it.
fn owned_cuda_probe_floor(worker: Option<&Path>) -> Result<OwnedCudaFloorReading, String> {
    let worker = worker
        .map(Path::to_path_buf)
        .or_else(|| env::var_os(worker_binary_env_var(CUDA_WORKER_ENGINE)).map(PathBuf::from))
        .or_else(|| resolve_worker_binary_sibling(CUDA_WORKER_ENGINE))
        .ok_or_else(|| "CUDA floor probe worker binary not found".to_string())?;
    let worker = fs::canonicalize(&worker).unwrap_or(worker);
    let entry = OWNED_CUDA_PROBE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .entry(worker.clone())
        .or_default()
        .clone();
    // Only callers for this worker wait; unrelated workers can probe concurrently.
    // Failed probes stay cached deliberately until module restart.
    entry
        .get_or_init(|| {
            let mut command =
                synapse_core::without_launch_nonce(std::process::Command::new(&worker));
            command.arg("--probe-floor");
            run_owned_cuda_probe(&mut command, Duration::from_secs(10))
        })
        .clone()
}

fn run_owned_cuda_probe(
    command: &mut std::process::Command,
    timeout: Duration,
) -> Result<OwnedCudaFloorReading, String> {
    let deadline = std::time::Instant::now() + timeout;
    command
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let mut child = command
        .spawn()
        .map_err(|error| format!("spawn CUDA floor probe: {error}"))?;
    let stdout = child.stdout.take().expect("piped stdout");
    let mut stderr = child.stderr.take().expect("piped stderr");
    let (stdout_tx, stdout_rx) = std::sync::mpsc::sync_channel(1);
    let (stderr_tx, stderr_rx) = std::sync::mpsc::sync_channel(1);
    std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let result = stdout.take(4097).read_to_end(&mut bytes).map(|_| bytes);
        let _ = stdout_tx.send(result);
    });
    std::thread::spawn(move || {
        let mut tail = Vec::new();
        let mut chunk = [0_u8; 4096];
        while let Ok(count) = stderr.read(&mut chunk) {
            if count == 0 {
                break;
            }
            let discard = (tail.len() + count).saturating_sub(4096);
            tail.drain(..discard);
            tail.extend_from_slice(&chunk[..count]);
        }
        let _ = stderr_tx.send(String::from_utf8_lossy(&tail).into_owned());
    });
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) if std::time::Instant::now() < deadline => {
                std::thread::sleep(
                    Duration::from_millis(20)
                        .min(deadline.saturating_duration_since(std::time::Instant::now())),
                );
            }
            other => {
                let _ = child.kill();
                let _ = child.wait();
                break Err(match other {
                    Err(error) => format!("wait for CUDA floor probe: {error}"),
                    _ => "CUDA floor probe timed out".to_string(),
                });
            }
        }
    };
    // Bound pipe completion too: a descendant may still hold an inherited pipe.
    let stderr = stderr_rx
        .recv_timeout(deadline.saturating_duration_since(std::time::Instant::now()))
        .unwrap_or_default();
    let fail = |reason: String| format!("{reason}; stderr: {stderr}");
    let status = status.map_err(fail)?;
    if !status.success() {
        return Err(fail(format!("CUDA floor probe exited {status}")));
    }
    let stdout = stdout_rx
        .recv_timeout(deadline.saturating_duration_since(std::time::Instant::now()))
        .map_err(|error| fail(format!("CUDA floor probe stdout: {error}")))?
        .map_err(|error| fail(format!("read CUDA floor probe stdout: {error}")))?;
    if stdout.len() > 4096 {
        return Err(fail(
            "CUDA floor probe stdout exceeds 4096 bytes".to_string(),
        ));
    }
    let parsed: Value = serde_json::from_slice(&stdout)
        .map_err(|error| fail(format!("invalid CUDA floor probe JSON: {error}")))?;
    let reading = || {
        Some(OwnedCudaFloorReading {
            driver_api: parsed.get("driver_api")?.as_u64()?.try_into().ok()?,
            compute_major: parsed
                .get("compute_capability")?
                .get("major")?
                .as_u64()?
                .try_into()
                .ok()?,
            compute_minor: parsed
                .get("compute_capability")?
                .get("minor")?
                .as_u64()?
                .try_into()
                .ok()?,
        })
    };
    reading().ok_or_else(|| fail("invalid CUDA floor probe hardware fields".to_string()))
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

async fn owned_cuda_evidence(
    state: &ModuleState,
    model: &EmbeddingModel,
) -> Result<Option<Value>, WireOperationError> {
    if model.engine_identity.engine != CUDA_WORKER_ENGINE {
        return Ok(None);
    }
    let worker = state
        .runtime
        .catalog
        .lock()
        .ok()
        .and_then(|catalog| {
            catalog
                .get(&model.model_id)
                .map(|slot| slot.spec.worker_bin.clone())
        })
        .ok_or_else(|| {
            artifact_invalid_error(format!("missing catalog entry for '{}'", model.model_id))
        })?;
    let (decision, observed) = tokio::task::spawn_blocking(move || {
        let decision = owned_cuda_floor_decision(worker.as_deref());
        let observed = owned_cuda_floor_observed(&decision, worker.as_deref());
        (decision, observed)
    })
    .await
    .map_err(|error| transient_model_load_error(format!("CUDA evidence task failed: {error}")))?;
    Ok(Some(json!({
        "engine": CUDA_WORKER_ENGINE,
        "backend": model.engine_identity.build_flags.get("backend"),
        "ptx_virtual_arch": model.engine_identity.build_flags.get("ptx_virtual_arch").cloned().unwrap_or_else(|| OWNED_CUDA_PTX_VIRTUAL_ARCH.to_string()),
        "minimum_device_cc": model.engine_identity.build_flags.get("minimum_device_cc").cloned().unwrap_or_else(|| OWNED_CUDA_MINIMUM_DEVICE_CC.to_string()),
        "minimum_cuda_driver_api": model.engine_identity.build_flags.get("minimum_cuda_driver_api").cloned().unwrap_or_else(|| OWNED_CUDA_MINIMUM_DRIVER_API.to_string()),
        "floor_state": if decision.is_supported() { "supported" } else { "unsupported" },
        "floor_refusal": decision.refusal_code(),
        "observed": observed,
        "cuda_cache_path": env::var("CUDA_CACHE_PATH").ok(),
        "worker_host_load_timeout_ms": state.runtime.worker_load_timeout.as_millis(),
        "worker_host_load_timeout_source": "worker.load_timeout_ms",
        "cold_load_ms": model_cold_load_ms(&state.runtime, &model.model_id),
        "warm_load_ms": Value::Null,
        "device_memory": Value::Null,
        "resident_process_count": 1,
    })))
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

struct ModelLoadScratch {
    path: PathBuf,
}

impl ModelLoadScratch {
    fn create(path: PathBuf) -> std::io::Result<Self> {
        fs::create_dir_all(&path)?;
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ModelLoadScratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn model_load_scratch_path(job_id: &str) -> PathBuf {
    env::temp_dir().join(format!(
        "synapse-model-load-{}-{job_id}",
        std::process::id()
    ))
}

fn model_load_owned_profile(
    engine: &str,
    root: &Path,
    params: &ModelLoadParams,
    config: Option<&ModelCacheMeta>,
    extra: &[ModelCacheMeta],
) -> Result<Option<OwnedCatalogConfig>, WireOperationError> {
    if engine != "owned-metal" && engine != CUDA_WORKER_ENGINE {
        return Ok(None);
    }
    let config = config.ok_or_else(|| {
        artifact_invalid_error(format!("{engine} model.load requires files.config"))
    })?;
    let locator = Some(ModelAssetLocator::CacheDigest {
        digest: config.digest.clone(),
    });
    let extras = extra
        .iter()
        .map(|meta| ModelAssetLocator::CacheDigest {
            digest: meta.digest.clone(),
        })
        .collect();
    let profile = if engine == "owned-metal" {
        owned_catalog_config(
            root,
            params.family.as_deref(),
            params.dtype.as_deref(),
            params.execution.as_deref(),
            params.attention_units,
            locator,
            extras,
        )
    } else {
        owned_cuda_catalog_config(
            params.family.as_deref(),
            params.dtype.as_deref(),
            params.execution.as_deref(),
            params.attention_units,
            OwnedCudaDeclaredIdentity {
                kernel_revision: None,
                ptx_virtual_arch: None,
                minimum_device_cc: None,
                minimum_cuda_driver_api: None,
            },
        )
        .map(|mut profile| {
            profile.config_locator = locator;
            profile.extra_locators = extras;
            profile
        })
    }
    .map_err(|error| artifact_invalid_error(error.to_string()))?;
    Ok(Some(profile))
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
        let sources = resolve_model_load_sources_at(&params, &state.runtime.hf_endpoint)
            .map_err(artifact_invalid_error)?;
        let temp_dir = model_load_scratch_path(&job_id);
        let scratch = ModelLoadScratch::create(temp_dir.clone())
            .map_err(|error| io_to_load_error("create temp directory", &temp_dir, &error))?;
        let temp_dir = scratch.path();
        let model_path = temp_dir.join("model.bin");
        let tokenizer_path = temp_dir.join("tokenizer.json");
        let config_path = sources
            .config
            .as_ref()
            .map(|_| temp_dir.join("config.json"));
        let extra_paths = sources
            .extra
            .iter()
            .enumerate()
            .map(|(index, _)| temp_dir.join(format!("extra-{index}.artifact")))
            .collect::<Vec<_>>();

        set_job_progress(
            &state.runtime,
            &job_id,
            ModelRuntimeState::Downloading {
                bytes_done: 0,
                bytes_total: None,
            },
        );
        download_source_to_temp(
            &sources.model.source_url,
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
        download_source_to_temp(&sources.tokenizer.source_url, &tokenizer_path, |_, _| {})?;
        if let (Some(source), Some(path)) = (&sources.config, &config_path) {
            download_source_to_temp(&source.source_url, path, |_, _| {})?;
        }
        for (source, path) in sources.extra.iter().zip(&extra_paths) {
            download_source_to_temp(&source.source_url, path, |_, _| {})?;
        }

        set_job_progress(&state.runtime, &job_id, ModelRuntimeState::Validating);
        let engine_name = canonical_engine_name(&params.engine);
        validate_artifact_file(&model_path, &engine_name)?;
        for extra_path in &extra_paths {
            validate_artifact_file(extra_path, &engine_name)?;
        }

        let pin_module_id = params.pin.then(|| state.module_id.clone());
        let tokenizer_meta = state
            .model_cache
            .ingest(ModelCacheIngest {
                source_url: local_file_url(&tokenizer_path),
                expected_digest: sources.tokenizer.expected_digest.clone(),
                format: "tokenizer_json".to_string(),
                tokenizer_path: None,
                pin_module_id: pin_module_id.clone(),
            })
            .map_err(model_cache_load_error)?;
        let model_meta = state
            .model_cache
            .ingest(ModelCacheIngest {
                source_url: local_file_url(&model_path),
                expected_digest: sources
                    .model
                    .expected_digest
                    .clone()
                    .or_else(|| params.expected_digest.clone()),
                format: default_artifact_format(&engine_name),
                tokenizer_path: Some(tokenizer_path.clone()),
                pin_module_id: pin_module_id.clone(),
            })
            .map_err(model_cache_load_error)?;
        let config_meta = config_path
            .as_ref()
            .map(|path| {
                state.model_cache.ingest(ModelCacheIngest {
                    source_url: local_file_url(path),
                    expected_digest: sources
                        .config
                        .as_ref()
                        .and_then(|source| source.expected_digest.clone()),
                    format: "json".to_string(),
                    tokenizer_path: None,
                    pin_module_id: pin_module_id.clone(),
                })
            })
            .transpose()
            .map_err(model_cache_load_error)?;
        let extra_metas = extra_paths
            .iter()
            .zip(&sources.extra)
            .map(|(path, source)| {
                state.model_cache.ingest(ModelCacheIngest {
                    source_url: local_file_url(path),
                    expected_digest: source.expected_digest.clone(),
                    format: if engine_name == "owned-metal" {
                        "safetensors".to_string()
                    } else {
                        default_artifact_format(&engine_name)
                    },
                    tokenizer_path: None,
                    pin_module_id: pin_module_id.clone(),
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(model_cache_load_error)?;
        let package_digest = package_digest(
            &model_meta,
            &tokenizer_meta,
            config_meta.as_ref(),
            &extra_metas,
        );
        let owned = model_load_owned_profile(
            &engine_name,
            temp_dir,
            &params,
            config_meta.as_ref(),
            &extra_metas,
        )?;
        let spec = build_loaded_catalog_model(
            &params,
            &engine_name,
            &sources,
            &model_meta,
            &tokenizer_meta,
            package_digest,
            extra_metas
                .iter()
                .map(|meta| ModelAssetLocator::CacheDigest {
                    digest: meta.digest.clone(),
                })
                .collect(),
            owned,
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
        let loaded =
            ensure_model_loaded_for_control(Arc::clone(&state), &spec.model_id, params.deadline_ms)
                .await?;
        let result = json!({
            "model_id": loaded.model_id,
            "fingerprint": loaded.fingerprint,
        });
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
            } else {
                state.runtime.admission_telemetry.record_job_completed();
            }
        }
        Err(error) => {
            fail_job_with_wire_error(&state, &job_id, error.class == ErrorClass::Transient, error);
        }
    }
}

fn validate_model_load_request(params: &ModelLoadParams) -> Result<(), String> {
    if params.files.model.locator().trim().is_empty()
        || params.files.tokenizer.locator().trim().is_empty()
    {
        return Err("model.load requires files.model and files.tokenizer".to_string());
    }
    let engine = canonical_engine_name(&params.engine);
    let task = parse_model_task(
        params.task.as_deref(),
        &engine,
        params.model_id.as_deref().unwrap_or("model"),
    )
    .map_err(|error| error.to_string())?;
    if matches!(task, ModelTask::Embed | ModelTask::Rerank)
        && !store::OWNED_EMBED_RERANK_ENGINES.contains(&engine.as_str())
    {
        return Err(format!(
            "engine '{engine}' is not an owned embed/rerank engine"
        ));
    }
    if params.source == "hf" && !params.revision.as_deref().is_some_and(pinned_revision) {
        return Err("model.load source=hf requires a 40-hex revision".to_string());
    }
    Ok(())
}

#[cfg(test)]
fn resolve_model_load_sources(
    params: &ModelLoadParams,
) -> Result<ResolvedModelLoadSources, String> {
    resolve_model_load_sources_at(params, &default_hf_endpoint())
}

fn resolve_model_load_sources_at(
    params: &ModelLoadParams,
    endpoint: &str,
) -> Result<ResolvedModelLoadSources, String> {
    let resolve = |spec: &ModelLoadFileSpec| -> Result<ResolvedModelLoadAsset, String> {
        let locator = spec.locator();
        let source_url = if matches!(spec, ModelLoadFileSpec::Detailed { .. })
            && (locator.starts_with("https://")
                || locator.starts_with("http://")
                || locator.starts_with("file://"))
        {
            locator.to_string()
        } else {
            match params.source.trim().to_ascii_lowercase().as_str() {
                "hf" => {
                    let repo = params
                        .repo
                        .as_deref()
                        .filter(|value| !value.trim().is_empty())
                        .ok_or_else(|| "model.load source=hf requires repo".to_string())?;
                    huggingface_resolve_url(
                        endpoint,
                        repo,
                        params.revision.as_deref().unwrap_or(""),
                        locator,
                    )?
                }
                "url" => {
                    let base = params
                        .url
                        .as_deref()
                        .filter(|value| !value.trim().is_empty())
                        .ok_or_else(|| "model.load source=url requires url".to_string())?;
                    join_base_url(base, locator)?
                }
                "file" => {
                    let base = params
                        .path
                        .as_deref()
                        .filter(|value| !value.trim().is_empty())
                        .ok_or_else(|| "model.load source=file requires path".to_string())?;
                    join_file_source(base, locator)?
                }
                other => return Err(format!("unsupported model.load source '{other}'")),
            }
        };
        let expected_digest = spec.expected_digest().or_else(|| {
            std::ptr::eq(spec, &params.files.model)
                .then(|| params.expected_digest.clone())
                .flatten()
        });
        validate_resolved_asset(&source_url, expected_digest.as_deref(), endpoint)?;
        Ok(ResolvedModelLoadAsset {
            source_url,
            expected_digest,
        })
    };

    Ok(ResolvedModelLoadSources {
        model: resolve(&params.files.model)?,
        tokenizer: resolve(&params.files.tokenizer)?,
        config: params.files.config.as_ref().map(resolve).transpose()?,
        extra: params
            .files
            .extra
            .iter()
            .map(resolve)
            .collect::<Result<Vec<_>, _>>()?,
    })
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

fn package_digest(
    model: &ModelCacheMeta,
    tokenizer: &ModelCacheMeta,
    config: Option<&ModelCacheMeta>,
    extra: &[ModelCacheMeta],
) -> String {
    let mut roles = vec![
        ("model".to_string(), model.digest.clone()),
        ("tokenizer".to_string(), tokenizer.digest.clone()),
    ];
    if let Some(config) = config {
        roles.push(("config".to_string(), config.digest.clone()));
    }
    roles.extend(extra.iter().map(|meta| {
        let role = meta
            .source_url
            .rsplit('/')
            .find(|segment| !segment.is_empty())
            .unwrap_or("extra");
        (format!("extra:{role}"), meta.digest.clone())
    }));
    roles.sort_by(|left, right| left.0.cmp(&right.0));
    format!(
        "sha256:{}",
        sha256_hex(&serde_json::to_vec(&roles).expect("package digest tuple serializes"))
    )
}

#[allow(clippy::too_many_arguments)]
fn build_loaded_catalog_model(
    params: &ModelLoadParams,
    engine_name: &str,
    sources: &ResolvedModelLoadSources,
    model_meta: &ModelCacheMeta,
    tokenizer_meta: &ModelCacheMeta,
    package_digest: String,
    extra_locators: Vec<ModelAssetLocator>,
    owned: Option<OwnedCatalogConfig>,
    inline: &InlineConfig,
    jobs: &JobConfig,
) -> Result<StoredModelConfig, WireOperationError> {
    let model_id = params.model_id.clone().unwrap_or_else(|| {
        derive_loaded_model_id(engine_name, &sources.model.source_url, &model_meta.digest)
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
        if owned.is_some() || engine_name == "ane" {
            package_digest
        } else {
            model_meta.digest.clone()
        },
        default_artifact_format(engine_name),
        tokenizer_sanitized_digest,
        ModelAssetLocator::CacheDigest {
            digest: model_meta.digest.clone(),
        },
        ModelAssetLocator::CacheDigest {
            digest: tokenizer_meta.digest.clone(),
        },
        sources.model.source_url.clone(),
        sources.tokenizer.source_url.clone(),
        pooling,
        params.normalize.unwrap_or(true),
        params.max_tokens.unwrap_or(512),
        params.quant.clone().unwrap_or_else(|| {
            owned
                .as_ref()
                .map(|profile| profile.dtype.as_str().to_string())
                .unwrap_or_else(|| default_quant(engine_name))
        }),
        params.pin,
        params.worker_bin.clone(),
        params.worker_runtime_dir.clone(),
        extra_locators,
        owned,
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
        #[cfg(feature = "test-support")]
        test_deterministic::NAME if read == 8 && &header == b"SYNTEST1" => Ok(()),
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
        "safetensors-package" if read == 8 => {
            let header_len = u64::from_le_bytes(header);
            let file_len = file.metadata().map(|metadata| metadata.len()).unwrap_or(0);
            if header_len > 0 && header_len.saturating_add(8) <= file_len {
                Ok(())
            } else {
                Err(artifact_invalid_error(format!(
                    "invalid safetensors header at {}",
                    path.display()
                )))
            }
        }
        "mlmodelc" if read >= 4 && &header[..4] == b"PK\x03\x04" => Ok(()),
        "mlmodelc" => Err(artifact_invalid_error(format!(
            "expected a zipped .mlmodelc bundle at {}",
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
    let resolution_started = Instant::now();
    let mut params: EmbedQueryParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid embed.query params: {error}"),
            )
        }
    };
    if params
        .model
        .as_deref()
        .is_some_and(|model_id| state.remote_gateway.is_remote(model_id))
    {
        return remote_embed_query(state, params).await;
    }
    let alias_table = match state.store.alias_table() {
        Ok(alias_table) => alias_table,
        Err(error) => return channel_error("store_failure", error.to_string()),
    };
    let model = match resolve_serving_model(
        Arc::clone(&state),
        params.model.as_deref(),
        ModelTask::Embed,
        params.required_fingerprint.as_deref(),
        params.target_fingerprint.as_deref(),
        params.deadline_ms,
    )
    .await
    {
        Ok(model) => model,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };
    if resolved_catalog_lane(&state.runtime, &model.model_id).is_some() {
        let budget = params
            .deadline_ms
            .unwrap_or(state.runtime.inline.deadline_ms);
        params.deadline_ms =
            Some(budget.saturating_sub(resolution_started.elapsed().as_millis() as u64));
    }
    if let Err(error) = ensure_pre_tokenization_certified(
        &state,
        &model,
        CertificationClass::Embedding,
        params.accept_declared,
    ) {
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
    let job_id = state
        .runtime
        .activity_telemetry
        .next_inline_job_id(state.module_generation);
    let started = Instant::now();
    let admission = match state.runtime.admit_inline(
        &model.model_id,
        Some(&job_id),
        QueueClass::Interactive,
        request_bytes,
        params.deadline_ms,
        params.max_queue_ms,
    ) {
        Ok(admission) => admission,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };
    let mut tokenized = match model.tokenizer.tokenize_batch([params.text.as_str()]) {
        Ok(tokenized) => tokenized,
        Err(error) => {
            return result_outcome(error_payload(
                &state,
                WireOperationError::from_stable(StableError::artifact_invalid(), error.to_string()),
            ))
        }
    };
    if let Err(mut error) = compose_catalog_embed(&model, &mut tokenized) {
        if let Some(details) = error.details.as_mut() {
            details["item_id"] = json!(params.id.as_deref().unwrap_or("query"));
        }
        return result_outcome(error_payload(&state, error));
    }
    if let Err(error) = ensure_profile_request_certified(state.clone(), &model).await {
        return result_outcome(error_payload(&state, error));
    }
    apply_owned_tokenizer_policy(&model, &mut tokenized);
    let ids = vec![params.id.unwrap_or_else(|| "query".to_string())];
    embed_tokenized(
        state,
        model,
        ids,
        tokenized,
        alias_table,
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

async fn remote_embed_query(state: Arc<ModuleState>, params: EmbedQueryParams) -> HandlerOutcome {
    let model_id = params
        .model
        .as_deref()
        .expect("remote query has an explicit model");
    let profile = state
        .remote_gateway
        .profile(model_id)
        .expect("remote query profile was checked before dispatch");
    if profile.task != RemoteTask::Embed {
        return result_outcome(error_payload(
            &state,
            WireOperationError::from_stable(
                StableError::op_not_supported_for_remote(),
                "the named remote profile does not support embed.query",
            ),
        ));
    }
    if !params.accept_declared {
        return result_outcome(error_payload(
            &state,
            WireOperationError::from_stable(
                StableError::declared_identity_not_accepted(),
                "remote profiles require accept_declared=true",
            ),
        ));
    }
    if let Err(error) = check_remote_fingerprint_constraints(
        &profile,
        params.target_fingerprint.as_deref(),
        params.required_fingerprint.as_deref(),
        params.required_epoch,
        &state,
    ) {
        return result_outcome(error_payload(&state, error));
    }
    let id = params.id.unwrap_or_else(|| "query".to_string());
    if let Err(error) = state.remote_gateway.ensure_certified(&profile).await {
        return remote_error_outcome(&state, error);
    }
    let request_bytes = request_bytes_for_texts([params.text.as_str()]);
    let job_id = state
        .runtime
        .activity_telemetry
        .next_inline_job_id(state.module_generation);
    let started = Instant::now();
    let deadline_ms = params
        .deadline_ms
        .unwrap_or(state.runtime.inline.deadline_ms);
    let estimated_ms = match state.remote_gateway.predicted_finish_ms(
        &profile,
        params.text.split_whitespace().count().max(1) as u64,
        now_ms(),
    ) {
        Ok(estimate) => estimate,
        Err(error) => return remote_error_outcome(&state, error),
    };
    if estimated_ms > deadline_ms {
        return result_outcome(error_payload(
            &state,
            WireOperationError::from_stable(
                StableError::deadline_exceeded(),
                format!(
                    "predicted remote finish {estimated_ms}ms exceeds deadline {deadline_ms}ms"
                ),
            ),
        ));
    }
    let _admission = match state.runtime.admit_inline(
        &profile.synapse_model_id,
        Some(&job_id),
        QueueClass::Interactive,
        request_bytes,
        params.deadline_ms,
        params.max_queue_ms,
    ) {
        Ok(admission) => admission,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };
    let original_count = params.text.split_whitespace().count().max(1) as u32;
    match state
        .remote_gateway
        .embed(
            &profile,
            &[params.text],
            RemoteClass::Interactive,
            deadline_ms,
        )
        .await
    {
        Ok(result) => remote_embed_success(
            &state,
            &profile,
            vec![id],
            vec![original_count],
            result,
            &job_id,
            started,
        ),
        Err(error) => remote_error_outcome(&state, error),
    }
}

fn remote_embed_success(
    state: &ModuleState,
    profile: &remote::config::ConfiguredRemoteProfile,
    ids: Vec<String>,
    original_token_counts: Vec<u32>,
    result: remote::gateway::RemoteEmbeddingResult,
    job_id: &str,
    started: Instant,
) -> HandlerOutcome {
    let tokens = result
        .token_counts
        .iter()
        .map(|count| u64::from(*count))
        .sum();
    let disclosures = original_token_counts
        .iter()
        .zip(&result.token_counts)
        .map(|(submitted, effective)| TruncationDisclosure {
            submitted_tokens: *submitted,
            effective_tokens: *effective,
            truncated: effective < submitted,
        })
        .collect::<Vec<_>>();
    let vectors = ids
        .into_iter()
        .zip(result.vectors)
        .zip(result.submitted_texts)
        .zip(result.submitted_sha256s)
        .map(
            |(((id, vector), text), submitted_sha256)| RemoteEmbedVector {
                id,
                vector,
                content_sha256: sha256_text(&text),
                submitted_sha256,
            },
        )
        .collect::<Vec<_>>();
    let dims = profile.dims.min(u32::MAX as usize) as u32;
    let table_epoch = state
        .store
        .alias_table()
        .map(|table| table.table_epoch)
        .unwrap_or(0);
    let mut envelope = json!({
        "fingerprint": profile.fingerprint,
        "table_epoch": table_epoch,
        "dims": dims,
        "provenance": state.remote_gateway.provenance(profile),
        "module_generation": state.module_generation,
        "equivalent_to": [],
        "assurance": "declared",
        "identity_revision": profile.identity_revision,
        "payload": {
            "vectors": vectors,
            "real_token_counts": result.token_counts,
            "truncation_disclosures": disclosures,
        },
    });
    if let Some(provider_request_id) = result.provider_request_id {
        envelope["provider_request_id"] = Value::String(provider_request_id);
    }
    state
        .runtime
        .activity_telemetry
        .record_completed_tokens(tokens);
    log_job_done(&profile.synapse_model_id, job_id, "remote", tokens, started);
    result_outcome(envelope)
}

fn remote_error_outcome(state: &ModuleState, error: RemoteGatewayError) -> HandlerOutcome {
    let mut payload = json!({
        "module_generation": state.module_generation,
        "error": WireOperationError::from_stable(error.stable, error.message),
    });
    if let Some(provider_request_id) = error.provider_request_id {
        payload["provider_request_id"] = Value::String(provider_request_id);
    }
    result_outcome(payload)
}

fn check_remote_fingerprint_constraints(
    profile: &remote::config::ConfiguredRemoteProfile,
    target_fingerprint: Option<&str>,
    required_fingerprint: Option<&str>,
    required_epoch: Option<u64>,
    state: &ModuleState,
) -> Result<(), WireOperationError> {
    if target_fingerprint
        .into_iter()
        .chain(required_fingerprint)
        .any(|required| required != profile.fingerprint.0)
    {
        return Err(WireOperationError::from_stable(
            StableError::substitution_rejected(),
            "the remote profile fingerprint does not satisfy the request constraint",
        ));
    }
    if let Some(required_epoch) = required_epoch {
        let current_epoch = state
            .store
            .alias_table()
            .map_err(|error| {
                WireOperationError::from_stable(
                    StableError::engine_crashed(Some(100)),
                    format!("read alias table: {error}"),
                )
            })?
            .table_epoch;
        if current_epoch != required_epoch {
            return Err(WireOperationError::from_stable(
                StableError::substitution_rejected(),
                format!("required table epoch {required_epoch} does not match current epoch {current_epoch}"),
            ));
        }
    }
    Ok(())
}

fn sha256_text(text: &str) -> String {
    hex::encode(Sha256::digest(text.as_bytes()))
}

async fn embed_batch(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let resolution_started = Instant::now();
    let mut params: EmbedBatchParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid embed.batch params: {error}"),
            )
        }
    };
    if params
        .model
        .as_deref()
        .is_some_and(|model_id| state.remote_gateway.is_remote(model_id))
    {
        return remote_embed_batch(state, params).await;
    }
    let mut job_params = json!({"model":params.model,"required_fingerprint":params.required_fingerprint,"target_fingerprint":params.target_fingerprint,"allow_equivalent":params.allow_equivalent,"required_epoch":params.required_epoch,"accept_declared":params.accept_declared,"request_key":params.request_key});
    let items = match batch_items(params.items, params.texts) {
        Ok(items) => items,
        Err(message) => return channel_error("invalid_request", message),
    };
    if items.is_empty() {
        return channel_error("invalid_request", "embed.batch requires at least one item");
    }
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

    let alias_table = match state.store.alias_table() {
        Ok(alias_table) => alias_table,
        Err(error) => return channel_error("store_failure", error.to_string()),
    };
    let model = match resolve_serving_model(
        Arc::clone(&state),
        params.model.as_deref(),
        ModelTask::Embed,
        params.required_fingerprint.as_deref(),
        params.target_fingerprint.as_deref(),
        params.deadline_ms,
    )
    .await
    {
        Ok(model) => model,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };
    if resolved_catalog_lane(&state.runtime, &model.model_id).is_some() {
        let budget = params
            .deadline_ms
            .unwrap_or(state.runtime.inline.deadline_ms);
        params.deadline_ms =
            Some(budget.saturating_sub(resolution_started.elapsed().as_millis() as u64));
    }
    if let Err(error) = ensure_pre_tokenization_certified(
        &state,
        &model,
        CertificationClass::Embedding,
        params.accept_declared,
    ) {
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
    let mut tokenized = match model.tokenizer.tokenize_batch(text_refs) {
        Ok(tokenized) => tokenized,
        Err(error) => {
            return result_outcome(error_payload(
                &state,
                WireOperationError::from_stable(StableError::artifact_invalid(), error.to_string()),
            ))
        }
    };
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
    if let Err(error) = ensure_profile_request_certified(state.clone(), &model).await {
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
        None,
        None,
        &json!({
            "target_fingerprint": params.target_fingerprint,
            "required_fingerprint": params.required_fingerprint,
            "allow_equivalent": params.allow_equivalent,
            "required_epoch": params.required_epoch,
            "accept_declared": params.accept_declared,
        }),
        &digest_items,
    );
    let ids = items.into_iter().map(|item| item.id).collect::<Vec<_>>();

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
                alias_table,
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
    let admission = match state.runtime.admit_inline(
        &model.model_id,
        Some(&job_id),
        QueueClass::Bulk,
        request_bytes,
        params.deadline_ms,
        params.max_queue_ms,
    ) {
        Ok(admission) => admission,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };
    embed_tokenized(
        state,
        model,
        ids,
        tokenized,
        alias_table,
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

async fn remote_embed_batch(state: Arc<ModuleState>, params: EmbedBatchParams) -> HandlerOutcome {
    let model_id = params
        .model
        .as_deref()
        .expect("remote batch has an explicit model");
    let profile = state
        .remote_gateway
        .profile(model_id)
        .expect("remote batch profile was checked before dispatch");
    if profile.task != RemoteTask::Embed {
        return result_outcome(error_payload(
            &state,
            WireOperationError::from_stable(
                StableError::op_not_supported_for_remote(),
                "the named remote profile does not support embed.batch",
            ),
        ));
    }
    if !params.accept_declared {
        return result_outcome(error_payload(
            &state,
            WireOperationError::from_stable(
                StableError::declared_identity_not_accepted(),
                "remote profiles require accept_declared=true",
            ),
        ));
    }
    let items = match batch_items(params.items, params.texts) {
        Ok(items) if !items.is_empty() => items,
        Ok(_) => return channel_error("invalid_request", "embed.batch requires at least one item"),
        Err(message) => return channel_error("invalid_request", message),
    };
    if let Err(error) = check_remote_fingerprint_constraints(
        &profile,
        params.target_fingerprint.as_deref(),
        params.required_fingerprint.as_deref(),
        params.required_epoch,
        &state,
    ) {
        return result_outcome(error_payload(&state, error));
    }
    if let Err(error) = state.remote_gateway.ensure_certified(&profile).await {
        return remote_error_outcome(&state, error);
    }
    let counts = items
        .iter()
        .map(|item| {
            item.text
                .split_whitespace()
                .count()
                .max(1)
                .min(u32::MAX as usize) as u32
        })
        .collect::<Vec<_>>();
    let total_tokens = counts.iter().map(|count| u64::from(*count)).sum::<u64>();
    let request_bytes = request_bytes_for_texts(items.iter().map(|item| item.text.as_str()));
    let digest_items = items
        .iter()
        .map(|item| (item.id.clone(), sha256_text(&item.text)))
        .collect::<Vec<_>>();
    let request_digest = compute_request_digest(
        "embed.batch",
        &profile.synapse_model_id,
        Some(&profile.remote_profile_hash),
        state.remote_gateway.logical_handle(&profile).as_deref(),
        &json!({
            "target_fingerprint": params.target_fingerprint,
            "required_fingerprint": params.required_fingerprint,
            "allow_equivalent": params.allow_equivalent,
            "required_epoch": params.required_epoch,
            "accept_declared": true,
        }),
        &digest_items,
    );
    let deadline_ms = params
        .deadline_ms
        .unwrap_or(state.runtime.inline.deadline_ms);
    let estimated_ms =
        match state
            .remote_gateway
            .predicted_finish_ms(&profile, total_tokens, now_ms())
        {
            Ok(estimate) => estimate,
            Err(error) => return remote_error_outcome(&state, error),
        };
    if estimated_ms > deadline_ms {
        return result_outcome(error_payload(
            &state,
            WireOperationError::from_stable(
                StableError::deadline_exceeded(),
                format!(
                    "predicted remote finish {estimated_ms}ms exceeds deadline {deadline_ms}ms"
                ),
            ),
        ));
    }
    if items.len() > state.runtime.inline.max_items
        || total_tokens > state.runtime.inline.max_tokens
    {
        return submit_remote_embed_batch_job(
            state,
            params.request_key,
            RemoteEmbedBatchJobWork {
                profile,
                request_digest,
                items,
                deadline_ms,
            },
        )
        .await;
    }
    let job_id = state
        .runtime
        .activity_telemetry
        .next_inline_job_id(state.module_generation);
    let started = Instant::now();
    let _admission = match state.runtime.admit_inline(
        &profile.synapse_model_id,
        Some(&job_id),
        QueueClass::Bulk,
        request_bytes,
        params.deadline_ms,
        params.max_queue_ms,
    ) {
        Ok(admission) => admission,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };
    let ids = items.iter().map(|item| item.id.clone()).collect::<Vec<_>>();
    let texts = items.into_iter().map(|item| item.text).collect::<Vec<_>>();
    match state
        .remote_gateway
        .embed(&profile, &texts, RemoteClass::Bulk, deadline_ms)
        .await
    {
        Ok(result) => remote_embed_success(&state, &profile, ids, counts, result, &job_id, started),
        Err(error) => remote_error_outcome(&state, error),
    }
}

async fn submit_remote_embed_batch_job(
    state: Arc<ModuleState>,
    request_key: Option<String>,
    work: RemoteEmbedBatchJobWork,
) -> HandlerOutcome {
    let Some(request_key) = request_key.filter(|key| !key.trim().is_empty()) else {
        return channel_error(
            "invalid_request",
            "job-shaped remote embed.batch requires a non-empty request_key",
        );
    };
    let now = now_ms();
    let logical_handle = state.remote_gateway.logical_handle(&work.profile);
    let admission = match state.store.admit_job(
        &request_key,
        &work.request_digest,
        "embed.batch",
        state.module_generation,
        logical_handle.as_deref(),
        &json!({
            "remote_profile_hash": work.profile.remote_profile_hash,
            "model": work.profile.synapse_model_id,
            "items": work.items.iter().map(|item| json!({"id": item.id, "text": item.text})).collect::<Vec<_>>(),
            "deadline_ms": work.deadline_ms,
            "accept_declared": true,
        }),
        now,
        state.runtime.jobs.execution_ttl_ms,
        state.runtime.jobs.result_retention_ttl_ms,
    ) {
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
        log_job_admitted(&work.profile.synapse_model_id, &record.job_id);
        spawn_remote_embed_batch_job(Arc::clone(&state), record.job_id.clone(), work);
    }
    result_outcome(job_status_payload(&state, &record))
}

fn spawn_remote_embed_batch_job(
    state: Arc<ModuleState>,
    job_id: String,
    work: RemoteEmbedBatchJobWork,
) {
    tokio::spawn(async move {
        execute_remote_embed_batch_job(state, job_id, work).await;
    });
}

async fn execute_remote_embed_batch_job(
    state: Arc<ModuleState>,
    job_id: String,
    work: RemoteEmbedBatchJobWork,
) {
    let record = match state
        .store
        .claim_job_attempt(&job_id, state.module_generation, now_ms())
    {
        Ok(JobAttemptClaim::Claimed(record)) => record,
        Ok(JobAttemptClaim::Attached { .. } | JobAttemptClaim::NotClaimable(_)) | Err(_) => return,
    };
    if let Err(error) = state.remote_gateway.ensure_certified(&work.profile).await {
        fail_job_with_wire_error(
            &state,
            &job_id,
            error.stable.class == ErrorClass::Transient,
            WireOperationError::from_stable(error.stable, error.message),
        );
        return;
    }
    let started = Instant::now();
    let mut completed_tokens = 0_u64;
    let committed = match state.store.committed_item_ids(&record.request_digest) {
        Ok(committed) => committed,
        Err(error) => {
            fail_job_with_wire_error(
                &state,
                &job_id,
                true,
                WireOperationError::from_stable(
                    StableError::engine_crashed(Some(100)),
                    format!("read remote checkpoints: {error}"),
                ),
            );
            return;
        }
    };
    let pending = work
        .items
        .into_iter()
        .filter(|item| !committed.contains(&item.id))
        .collect::<Vec<_>>();
    let chunk_size = state.runtime.inline.max_items.max(1);
    let mut page_no = record.page_count;
    for chunk in pending.chunks(chunk_size) {
        let ids = chunk.iter().map(|item| item.id.clone()).collect::<Vec<_>>();
        let texts = chunk
            .iter()
            .map(|item| item.text.clone())
            .collect::<Vec<_>>();
        let original_counts = texts
            .iter()
            .map(|text| {
                text.split_whitespace()
                    .count()
                    .max(1)
                    .min(u32::MAX as usize) as u32
            })
            .collect::<Vec<_>>();
        let result = match state
            .remote_gateway
            .embed(&work.profile, &texts, RemoteClass::Bulk, work.deadline_ms)
            .await
        {
            Ok(result) => result,
            Err(error) if error.stable == StableError::needs_reauth() => {
                if let Some(handle) = state.remote_gateway.logical_handle(&work.profile) {
                    let _ = state.store.pause_job_needs_reauth(
                        &job_id,
                        &handle,
                        now_ms(),
                        state.runtime.jobs.resume_deadline_ms,
                    );
                } else {
                    fail_job_with_wire_error(
                        &state,
                        &job_id,
                        false,
                        WireOperationError::from_stable(error.stable, error.message),
                    );
                }
                return;
            }
            Err(error) => {
                fail_job_with_wire_error(
                    &state,
                    &job_id,
                    error.stable.class == ErrorClass::Transient,
                    WireOperationError::from_stable(error.stable, error.message),
                );
                return;
            }
        };
        let provider_request_id = result.provider_request_id.clone();
        completed_tokens = completed_tokens.saturating_add(
            result
                .token_counts
                .iter()
                .map(|count| u64::from(*count))
                .sum::<u64>(),
        );
        let disclosures = original_counts
            .iter()
            .zip(&result.token_counts)
            .map(|(submitted, effective)| TruncationDisclosure {
                submitted_tokens: *submitted,
                effective_tokens: *effective,
                truncated: effective < submitted,
            })
            .collect::<Vec<_>>();
        let vectors = ids
            .iter()
            .cloned()
            .zip(result.vectors)
            .zip(result.submitted_texts)
            .zip(result.submitted_sha256s)
            .map(
                |(((id, vector), text), submitted_sha256)| RemoteEmbedVector {
                    id,
                    vector,
                    content_sha256: sha256_text(&text),
                    submitted_sha256,
                },
            )
            .collect::<Vec<_>>();
        let mut page_value = json!({
            "fingerprint": work.profile.fingerprint,
            "table_epoch": state.store.alias_table().map(|table| table.table_epoch).unwrap_or(0),
            "dims": work.profile.dims,
            "provenance": state.remote_gateway.provenance(&work.profile),
            "module_generation": state.module_generation,
            "equivalent_to": [],
            "assurance": "declared",
            "identity_revision": work.profile.identity_revision,
            "payload": {
                "vectors": vectors,
                "real_token_counts": result.token_counts,
                "truncation_disclosures": disclosures,
            },
        });
        if let Some(provider_request_id) = provider_request_id.as_ref() {
            page_value["provider_request_ids"] = json!([provider_request_id]);
        }
        let page_bytes = match serde_json::to_vec(&page_value) {
            Ok(bytes) => bytes,
            Err(error) => {
                fail_job_with_wire_error(
                    &state,
                    &job_id,
                    false,
                    WireOperationError::from_stable(
                        StableError::engine_crashed(None),
                        format!("serialize remote result page: {error}"),
                    ),
                );
                return;
            }
        };
        let checkpoints = vectors
            .iter()
            .map(|vector| CheckpointItem {
                item_id: vector.id.clone(),
                result: serde_json::to_vec(vector).expect("remote checkpoint serializes"),
                provider_request_id: provider_request_id.clone(),
            })
            .collect::<Vec<_>>();
        if let Err(error) =
            state
                .store
                .commit_job_page(&job_id, page_no, &page_bytes, &checkpoints, now_ms())
        {
            fail_job_with_wire_error(
                &state,
                &job_id,
                true,
                WireOperationError::from_stable(
                    StableError::engine_crashed(Some(100)),
                    format!("atomically commit remote checkpoint page: {error}"),
                ),
            );
            return;
        }
        page_no = page_no.saturating_add(1);
    }
    let summary = json!({
        "job_id": job_id,
        "state": JOB_STATE_DONE,
        "page_count": page_no,
        "module_generation": state.module_generation,
        "fingerprint": work.profile.fingerprint,
        "assurance": "declared",
        "identity_revision": work.profile.identity_revision,
        "provenance": state.remote_gateway.provenance(&work.profile),
    });
    match state.store.finish_job(&job_id, &summary, now_ms()) {
        Ok(()) => {
            state.runtime.admission_telemetry.record_job_completed();
            state
                .runtime
                .activity_telemetry
                .record_completed_tokens(completed_tokens);
            log_job_done(
                &work.profile.synapse_model_id,
                &job_id,
                "remote",
                completed_tokens,
                started,
            );
        }
        Err(error) => fail_job_with_wire_error(
            &state,
            &job_id,
            true,
            WireOperationError::from_stable(
                StableError::engine_crashed(Some(100)),
                format!("finish remote checkpoint job: {error}"),
            ),
        ),
    }
}

async fn rerank_score(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let resolution_started = Instant::now();
    let mut params: RerankScoreParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid rerank.score params: {error}"),
            )
        }
    };
    if params
        .model
        .as_deref()
        .is_some_and(|model_id| state.remote_gateway.is_remote(model_id))
    {
        return result_outcome(error_payload(
            &state,
            WireOperationError::from_stable(
                StableError::op_not_supported_for_remote(),
                "rerank.score is not supported for remote profiles in gateway v1",
            ),
        ));
    }
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
    let model = match resolve_serving_model(
        Arc::clone(&state),
        params.model.as_deref(),
        ModelTask::Rerank,
        params.required_fingerprint.as_deref(),
        params.target_fingerprint.as_deref(),
        params.deadline_ms,
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
    if resolved_catalog_lane(&state.runtime, &model.model_id).is_some() {
        let budget = params
            .deadline_ms
            .unwrap_or(state.runtime.inline.deadline_ms);
        params.deadline_ms =
            Some(budget.saturating_sub(resolution_started.elapsed().as_millis() as u64));
    }
    if let Err(error) = ensure_pre_tokenization_certified(
        &state,
        &model,
        CertificationClass::Rerank,
        params.accept_declared,
    ) {
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

    let owned_pairs = match owned_rerank_pairs(&model, params.query.as_str(), &params.candidates) {
        Ok(pairs) => pairs,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };
    if let Err(error) = ensure_profile_request_certified(state.clone(), &model).await {
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
    let job_id = state
        .runtime
        .activity_telemetry
        .next_inline_job_id(state.module_generation);
    let started = Instant::now();
    let admission = match state.runtime.admit_inline(
        &model.model_id,
        Some(&job_id),
        queue_class,
        request_bytes,
        params.deadline_ms,
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
            remote: None,
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
    let params: MicroLlmOneshotParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid microllm.oneshot params: {error}"),
            )
        }
    };
    if params
        .model
        .as_deref()
        .is_some_and(|model_id| state.remote_gateway.is_remote(model_id))
    {
        return result_outcome(error_payload(
            &state,
            WireOperationError::from_stable(
                StableError::op_not_supported_for_remote(),
                "microllm.oneshot is not supported for remote profiles in gateway v1",
            ),
        ));
    }
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
    if let Some(model_id) = params.model.as_deref() {
        let owned_decode = state.runtime.catalog.lock().ok().is_some_and(|catalog| {
            catalog.get(model_id).is_some_and(|slot| {
                slot.spec.engine == "owned-metal-decode"
                    || slot.spec.engine_identity.engine == "owned-metal-decode"
            })
        });
        if owned_decode {
            let job_id = state
                .runtime
                .activity_telemetry
                .next_inline_job_id(state.module_generation);
            return route_owned_decode_wire(
                Arc::clone(&state),
                &params,
                model_id,
                job_id,
                Instant::now(),
            )
            .await;
        }
    }
    match params.grammar.as_deref() {
        None | Some("") => {}
        Some(raw) if raw.trim().is_empty() => {}
        Some(_) if !state.runtime.grammar_enabled => {
            return channel_error(
                "grammar_disabled",
                "microllm.oneshot constrained decoding is disabled in module config",
            );
        }
        Some(_) => {
            // Constrained requests are owned-decode-only because the legacy
            // llama worker must never receive raw grammar. Until this machine
            // has a certified and explicitly enabled owned-decode grammar lane,
            // fail closed with the routing contract's stable error ID.
            return channel_error(
                "grammar_disabled",
                "no certified and enabled owned-decode grammar lane is available",
            );
        }
    }

    let alias_table = match state.store.alias_table() {
        Ok(alias_table) => alias_table,
        Err(error) => return channel_error("store_failure", error.to_string()),
    };
    let model = match resolve_serving_model(
        Arc::clone(&state),
        params.model.as_deref(),
        ModelTask::Generate,
        params.required_fingerprint.as_deref(),
        params.target_fingerprint.as_deref(),
        params.deadline_ms,
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
    let job_id = state
        .runtime
        .activity_telemetry
        .next_inline_job_id(state.module_generation);
    let started = Instant::now();
    let admission = match state.runtime.admit_inline(
        &model.model_id,
        Some(&job_id),
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
    let equivalent_to = equivalent_fingerprints(&alias_table, &model);
    let completed_tokens = (output.n_prompt as u64).saturating_add(output.n_gen as u64);
    let payload = MicroLlmOneshotPayload {
        text: output.text,
        finish_reason: output.finish_reason,
        n_prompt: output.n_prompt,
        n_gen: output.n_gen,
        generated_token_ids: output.generated_token_ids,
        runtime_config_digest: None,
        derived_digest: None,
        generation_id: None,
        real_token_counts: tokenized.real_token_counts,
        truncation_disclosures: tokenized.disclosures,
    };
    let envelope = ResponseEnvelope {
        fingerprint: model.fingerprint.clone(),
        table_epoch: alias_table.table_epoch,
        dims: 0,
        provenance: ResponseProvenance {
            engine: model.engine_identity.clone(),
            remote: None,
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

#[derive(Clone)]
struct PersistentDecodeCertification {
    store: Arc<SynapseStore>,
    match_inputs: OwnedDecodeMatchInputs,
}

impl owned_decode_routing::certification::CertificationAccess for PersistentDecodeCertification {
    fn is_unconstrained_certified(
        &self,
        key: &owned_decode_routing::certification::UnconstrainedCertKey,
    ) -> bool {
        if key.machine_profile_hash != self.match_inputs.revisioned_machine_profile_hash {
            return false;
        }
        let mut inputs = self.match_inputs.clone();
        inputs.decode_fingerprint = key.decode_fingerprint.0.clone();
        inputs.constraint_runtime_identities.clear();
        self.store
            .get_owned_decode_cert_row_matching(&inputs)
            .ok()
            .flatten()
            .is_some()
    }

    fn is_constrained_certified(
        &self,
        key: &owned_decode_routing::certification::ConstrainedCertKey,
    ) -> bool {
        if key.machine_profile_hash != self.match_inputs.revisioned_machine_profile_hash {
            return false;
        }
        let mut inputs = self.match_inputs.clone();
        inputs.decode_fingerprint = key.decode_fingerprint.0.clone();
        inputs.constraint_runtime_identities = vec![key.constraint_runtime_identity.clone()];
        self.store
            .get_owned_decode_cert_row_matching(&inputs)
            .ok()
            .flatten()
            .is_some()
    }
}

// Staged certification evidence helper retained for migration/fleet reporting; remove this allow when that slice consumes it.
#[allow(dead_code)]
fn worker_path_certification(evidence: &Value) -> bool {
    let battery = evidence["worker_path"]["fixture_battery"].as_str();
    evidence["worker_path"]["transport"].as_str() == Some(worker_catalog_transport())
        && evidence["worker_path"]["protocol"].as_str()
            == Some(owned_decode_worker::identity::WORKER_PROTOCOL_ID)
        && matches!(battery, Some("20x64-token-exact" | "20x64-structural-band"))
}

fn sidecar_hint_bank_source(
    state: Arc<ModuleState>,
    target_prompt: &str,
    grammar: Option<&str>,
    target_tokenizer: &SanitizedTokenizer,
    compiled: Option<&owned_decode_grammar_scheduler::grammar_compile::CompiledConstraint>,
    worker_constraint: Option<&owned_decode_worker::protocol::TokenIdJsonConstraint>,
    deadline_ms: u64,
) -> Option<Box<dyn owned_decode_worker::worker::HintBankSource + Send>> {
    let sidecar_spec = state.runtime.sidecar_spec.clone();
    let compiled = compiled?;
    let worker_constraint = worker_constraint?;
    if !sidecar_spec.enabled
        || sidecar_spec.strategy != synapse_core::SidecarStrategy::WholeObject
        || compiled.constraint.canonical_schema_digest != worker_constraint.canonical_schema_digest
    {
        return None;
    }
    let grammar = grammar?.trim();
    if grammar.is_empty() {
        return None;
    }
    let schema = frozen_sidecar_object_schema(
        compiled.automaton.schema(),
        compiled.constraint.canonical_schema_digest.clone(),
    );
    let sidecar_prompt = format!(
        "Return exactly one JSON object that satisfies this JSON Schema. Return no prose.\nSchema:\n{grammar}\n\nRequest:\n{target_prompt}"
    );
    let policy = owned_decode_sidecar::RenderPolicy::compact();
    let render_policy_digest = policy.digest();
    let schema_identity = compiled.constraint.canonical_schema_digest.clone();
    let target_tokenizer = target_tokenizer.clone();
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    let _sidecar_job = tokio::spawn(async move {
        let Ok(model) = ensure_model_loaded_for_control(
            Arc::clone(&state),
            &sidecar_spec.model_id,
            Some(deadline_ms),
        )
        .await
        else {
            return;
        };
        if model.task != ModelTask::Generate {
            return;
        }
        let Ok(tokenized) = model.tokenizer.tokenize_batch([sidecar_prompt.as_str()]) else {
            return;
        };
        let Some(prompt) = tokenized.batch.items.first().cloned() else {
            return;
        };
        let Ok(output) = execute_generate(
            &state.runtime,
            &model,
            GenerateRequest {
                prompt,
                max_tokens: sidecar_spec.max_new_tokens,
                grammar: None,
            },
            Some(tokio::time::Instant::now() + Duration::from_millis(deadline_ms)),
            None,
        )
        .await
        else {
            return;
        };
        // The frozen whole-object result contract has no scalar-root rendering
        // variant. Such a valid grammar still launches the configured sidecar,
        // but its completion cannot publish a bank until that contract grows a
        // matching root representation.
        let Some(schema) = schema else {
            return;
        };
        let Ok(prepared) =
            owned_decode_sidecar::prepare_whole_object(&schema, output.text.as_bytes(), &policy)
        else {
            return;
        };
        let Ok(bank) = owned_decode_sidecar::build_default_hint_bank(
            &target_tokenizer,
            schema_identity,
            render_policy_digest,
            &prepared,
            owned_decode_sidecar::SidecarWorkBounds::default(),
            now_ms(),
        ) else {
            return;
        };
        let _ = sender.try_send(bank);
    });
    Some(Box::new(
        owned_decode_worker::worker::SidecarHintBankPickup::pending(receiver),
    ))
}

fn frozen_sidecar_object_schema(
    schema: &owned_decode_grammar_scheduler::grammar_schema::Schema,
    schema_identity: String,
) -> Option<owned_decode_sidecar::FrozenObjectSchema> {
    let owned_decode_sidecar::FrozenSchema::Object { properties } =
        frozen_sidecar_schema_node(schema, 0)?
    else {
        return None;
    };
    Some(owned_decode_sidecar::FrozenObjectSchema {
        schema_identity,
        properties,
    })
}

fn frozen_sidecar_schema_node(
    schema: &owned_decode_grammar_scheduler::grammar_schema::Schema,
    index: usize,
) -> Option<owned_decode_sidecar::FrozenSchema> {
    use owned_decode_grammar_scheduler::grammar_schema::{NodeKind, SchemaType};
    use owned_decode_sidecar::{FrozenProperty, FrozenScalarType};

    let node = schema.node(index);
    match (&node.ty, &node.kind) {
        (
            SchemaType::Object,
            NodeKind::Object {
                properties,
                required,
            },
        ) => Some(owned_decode_sidecar::FrozenSchema::Object {
            properties: properties
                .iter()
                .map(|(name, child)| {
                    Some(FrozenProperty {
                        name: name.clone(),
                        required: required.contains(name),
                        schema: frozen_sidecar_schema_node(schema, *child)?,
                    })
                })
                .collect::<Option<Vec<_>>>()?,
        }),
        (SchemaType::Array, NodeKind::Array { items }) => {
            Some(owned_decode_sidecar::FrozenSchema::Array {
                items: Box::new(frozen_sidecar_schema_node(schema, *items)?),
            })
        }
        (schema_type, NodeKind::Scalar { enumeration }) => {
            let scalar_type = match schema_type {
                SchemaType::String => FrozenScalarType::String,
                SchemaType::Number => FrozenScalarType::Number,
                SchemaType::Integer => FrozenScalarType::Integer,
                SchemaType::Boolean => FrozenScalarType::Boolean,
                SchemaType::Null => FrozenScalarType::Null,
                SchemaType::Object | SchemaType::Array => return None,
            };
            let enumeration = match enumeration {
                Some(values) => Some(
                    values
                        .iter()
                        .map(|value| serde_json::from_str(&value.json_text()).ok())
                        .collect::<Option<Vec<Value>>>()?,
                ),
                None => None,
            };
            Some(owned_decode_sidecar::FrozenSchema::Scalar {
                scalar_type,
                enumeration,
            })
        }
        _ => None,
    }
}

struct WireDecodeDispatch {
    owned: Option<Arc<Mutex<worker_host::SupervisedDecodeDispatch>>>,
    prompt: Vec<u32>,
    constraint: Option<owned_decode_worker::protocol::TokenIdJsonConstraint>,
    /// Present only for an enabled request whose compiled constraint and
    /// sidecar schema share the canonical identity.
    sidecar_hint_bank_source: Option<Box<dyn owned_decode_worker::worker::HintBankSource + Send>>,
    deadline_ms: u64,
    llama: Option<Arc<EmbeddingModel>>,
    llama_output: Option<GenerateOutput>,
}

impl owned_decode_routing::DecodeDispatch for WireDecodeDispatch {
    fn dispatch(
        &mut self,
        command: &owned_decode_routing::DispatchedCommand,
    ) -> Result<owned_decode_routing::ExecutionSuccess, owned_decode_routing::error::OwnedDecodeError>
    {
        use owned_decode_routing::lane::LaneKind;
        use owned_decode_routing::provenance::FinishReason;

        if command.lane == LaneKind::OwnedDecode {
            let mut dispatch = self
                .owned
                .as_ref()
                .ok_or(owned_decode_routing::error::OwnedDecodeError::Unavailable)?
                .lock()
                .map_err(|_| owned_decode_routing::error::OwnedDecodeError::Unavailable)?;
            dispatch.set_request(
                self.prompt.clone(),
                self.constraint.clone(),
                self.deadline_ms,
            );
            // Request-level routing installs the request-scoped source only
            // after successful compilation and identity validation. Polling the
            // source remains non-blocking while a sidecar job is pending.
            dispatch.set_hint_bank_source(
                self.sidecar_hint_bank_source
                    .take()
                    .unwrap_or_else(|| Box::new(owned_decode_worker::worker::NoHintBankSource)),
            );
            return owned_decode_routing::DecodeDispatch::dispatch(&mut *dispatch, command);
        }

        let model = self
            .llama
            .as_ref()
            .ok_or(owned_decode_routing::error::OwnedDecodeError::Unavailable)?;
        let EmbedBackend::Worker(engine) = &model.backend else {
            return Err(owned_decode_routing::error::OwnedDecodeError::Unsupported);
        };
        let engine = engine
            .lock()
            .map_err(|_| owned_decode_routing::error::OwnedDecodeError::Unavailable)?;
        let output = engine
            .generate(
                &model.loaded_model,
                GenerateRequest {
                    prompt: self.prompt.clone(),
                    max_tokens: command.max_tokens,
                    grammar: None,
                },
            )
            .map_err(|_| owned_decode_routing::error::OwnedDecodeError::Unavailable)?;
        let (finish_reason, lane_finish_reason) = match output.finish_reason.as_str() {
            "stop" | "stop_token" => (FinishReason::StopToken, None),
            "length" | "max_tokens" => (FinishReason::MaxTokens, None),
            "cancelled" => (FinishReason::Cancelled, None),
            other => (FinishReason::StopToken, Some(other.to_string())),
        };
        let success = owned_decode_routing::ExecutionSuccess {
            generated_token_ids: output.generated_token_ids.clone(),
            finish_reason,
            lane_finish_reason,
            worker_generation: 0,
            last_completed_quantum_sequence: 0,
            crash_retry_count: 0,
            failure_classifications: Vec::new(),
        };
        self.llama_output = Some(output);
        Ok(success)
    }
}

/// Classify an owned-decode resolution refusal without discarding the catalog
/// identity. Metal execution is unavailable off macOS, but its catalog row and
/// fingerprints remain valid routing data everywhere.
fn owned_decode_resolution_refusal_for_platform(
    engine: &str,
    platform_supports_owned_decode: bool,
) -> Option<owned_decode_routing::error::OwnedDecodeError> {
    (engine == "owned-metal-decode" && !platform_supports_owned_decode)
        .then_some(owned_decode_routing::error::OwnedDecodeError::Unsupported)
}

fn owned_decode_resolution_refusal(
    spec: &StoredModelConfig,
) -> Option<owned_decode_routing::error::OwnedDecodeError> {
    owned_decode_resolution_refusal_for_platform(&spec.engine, cfg!(target_os = "macos"))
}

fn owned_decode_catalog_entry(
    spec: &StoredModelConfig,
) -> Result<owned_decode_routing::CatalogEntry, owned_decode_routing::error::OwnedDecodeError> {
    use owned_decode_routing::identity::{ActivationDType, Q8Identity, WeightQuant};

    let family_name = spec.owned_family.as_deref().unwrap_or_default();
    let family = owned_decode_routing::family::Family::parse(family_name)?;
    let activation_dtype = ActivationDType::parse(spec.owned_dtype.as_deref().unwrap_or_default())?;
    let weight_quant = WeightQuant::parse(&spec.quant)?;
    let flag = |name: &str| spec.engine_identity.build_flags.get(name).cloned();
    let q8 = if weight_quant.is_q8() {
        Some(Q8Identity {
            quantizer_revision: flag("quantizer_revision")
                .filter(|value| !value.trim().is_empty())
                .ok_or(owned_decode_routing::error::OwnedDecodeError::Unsupported)?,
            derived_digest: flag("derived_digest")
                .filter(|value| !value.trim().is_empty())
                .ok_or(owned_decode_routing::error::OwnedDecodeError::Unsupported)?,
        })
    } else {
        None
    };
    Ok(owned_decode_routing::CatalogEntry {
        entry_id: spec.model_id.clone(),
        engine: owned_decode_routing::CATALOG_ENGINE.to_string(),
        task: owned_decode_routing::CATALOG_TASK.to_string(),
        lane: owned_decode_routing::CATALOG_LANE.to_string(),
        worker: owned_decode_routing::CATALOG_WORKER.to_string(),
        risk_class: owned_decode_routing::CATALOG_RISK_CLASS.to_string(),
        family,
        activation_dtype,
        weight_quant,
        arithmetic_identity_revision: flag("arithmetic_identity_revision")
            .unwrap_or_else(|| "owned-decode-arithmetic-v1".to_string()),
        metallib_revision: flag("metallib_revision")
            .unwrap_or_else(|| "owned-decode-metallib-v1".to_string()),
        max_context_tokens: flag("max_context_tokens")
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or_else(|| spec.max_tokens.min(u32::MAX as usize) as u32),
        artifact_source_digest: spec.artifact_digest.clone(),
        q8,
        owned_family: spec.owned_family.clone(),
        owned_dtype: Some("f16".to_string()),
        quant: Some(spec.quant.clone()),
    })
}

fn owned_decode_processing_fingerprint(
    entry: &owned_decode_routing::CatalogEntry,
) -> Result<Fingerprint, owned_decode_routing::error::OwnedDecodeError> {
    use owned_decode_routing::identity::{PrefillEngineClass, ProcessingIdentityInputs};

    let decode_fingerprint = entry.decode_identity_inputs().decode_fingerprint()?;
    let families = owned_decode_routing::family::FamilyRegistry::production();
    let registration = families.get(entry.family)?;
    Ok(ProcessingIdentityInputs {
        decode_fingerprint,
        prefill_engine_class: PrefillEngineClass::Gpu,
        tokenizer_sanitized_digest: registration.tokenizer_sanitized_digest.clone(),
        prompt_template_revision: registration.prompt_template_revision.clone(),
        special_token_policy_revision: registration.special_token_policy_revision.clone(),
        stop_token_policy_revision: registration.stop_token_policy_revision.clone(),
        detokenizer_revision: registration.detokenizer_revision.clone(),
    }
    .processing_fingerprint())
}

fn owned_decode_runtime_identity(
    spec: &StoredModelConfig,
    entry: &owned_decode_routing::CatalogEntry,
    decode_chain_k: u32,
) -> (String, u32) {
    use owned_decode_routing::identity::RuntimeConfigManifest;

    let scheduler: owned_decode_contracts::SchedulerManifest = serde_json::from_str(include_str!(
        "../owned-decode-manifests/decode-sched-manifest-v1.json"
    ))
    .expect("checked-in decode scheduler manifest parses");
    let flag = |name: &str| spec.engine_identity.build_flags.get(name);
    let manifest = RuntimeConfigManifest {
        worker_revision: spec.engine_identity.version.clone(),
        protocol_revision: owned_decode_worker::identity::WORKER_PROTOCOL_ID.to_string(),
        metallib_revision: entry.metallib_revision.clone(),
        chain_k: decode_chain_k,
        batched_verification: flag("batched_verification").is_some_and(|value| value == "true"),
        resident_limit: 1,
        attention_kv_reservation_units: spec
            .owned_attention_units
            .unwrap_or(entry.max_context_tokens as usize)
            as u64,
        lfm2_conv_cache_reservation_bytes: flag("lfm2_conv_cache_reservation_bytes")
            .and_then(|value| value.parse().ok())
            .unwrap_or(0),
        context_manifest_revision: "decode-context-buckets-v1".to_string(),
        crash_policy_revision: "two-strike-crash-budget-v1".to_string(),
        quarantine_duration_ms: owned_decode_worker::budget::BudgetPolicy::default()
            .quarantine_duration_ms,
        scheduler: scheduler.runtime.clone(),
    };
    let manifest_digest = manifest.digest();
    let runtime_config_digest = if entry.family == owned_decode_routing::family::Family::Qwen3_0_6b
        && entry.weight_quant == owned_decode_routing::identity::WeightQuant::F16
    {
        // Reuse the existing runtime manifest digest for the Qwen3 0.6B F16
        // lane so its established runtime identity does not change. Other
        // family/quant combinations include those values in the digest so
        // every expanded lane has a distinct runtime identity.
        manifest_digest
    } else {
        sha256_hex(
            &serde_json::to_vec(&json!({
                "runtime_manifest_digest": manifest_digest,
                "family": entry.family.as_str(),
                "activation_dtype": entry.activation_dtype.as_str(),
                "weight_quant": entry.weight_quant.as_str(),
            }))
            .expect("owned-decode lane runtime identity serializes"),
        )
    };
    (runtime_config_digest, scheduler.runtime.production_n)
}

fn owned_decode_worker_runtime_dir(spec: &StoredModelConfig) -> PathBuf {
    spec.worker_runtime_dir
        .clone()
        .or_else(|| env::var_os("SYNAPSE_OWNED_DECODE_WORKER_RUNTIME_DIR").map(PathBuf::from))
        .unwrap_or_else(|| env::temp_dir().join("synapse-owned-decode-workers"))
}

fn owned_decode_budget_store_path(spec: &StoredModelConfig) -> PathBuf {
    owned_decode_worker_runtime_dir(spec).join(format!("{}-crash-budget.json", spec.model_id))
}

fn owned_decode_quarantined(
    state: &ModuleState,
    spec: &StoredModelConfig,
    decode_fingerprint: &Fingerprint,
    runtime_config_digest: &str,
) -> bool {
    use owned_decode_worker::budget::{CrashBudget as OwnedCrashBudget, FileBudgetStore};
    use owned_decode_worker::identity::QuarantineKey;

    let Ok(store) = FileBudgetStore::open(owned_decode_budget_store_path(spec)) else {
        return true;
    };
    let budget = OwnedCrashBudget::new(store, owned_decode_worker::budget::BudgetPolicy::default());
    let key = QuarantineKey::new(
        &state.machine_profile_hash,
        &decode_fingerprint.0,
        runtime_config_digest,
    );
    // The clock is load-bearing: `is_quarantined` answers `now < until`, so a
    // zero here makes any positive expiry true forever and the routing gate can
    // never clear, while the dispatch path in worker_host asks the same
    // predicate with a real clock and sees the quarantine expire. Two call
    // sites of one predicate must not disagree about time.
    budget.is_quarantined(&key, now_ms())
}

#[cfg(target_os = "macos")]
fn owned_decode_vocabulary_digest(
    tokenizer: &SanitizedTokenizer,
) -> Result<String, WireOperationError> {
    use synapse_engine_owned::owned_decode_engine::TokenVocabulary;

    let vocabulary = TokenVocabulary::from_tokenizer(tokenizer.tokenizer())
        .map_err(|error| artifact_invalid_error(error.to_string()))?;
    let mut hasher = Sha256::new();
    for token_id in 0..vocabulary.len() {
        if let Some(piece) = vocabulary.token_piece(token_id as u32) {
            hasher.update(piece);
        }
        hasher.update([0]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// The owned decode engines are Metal-only; on other platforms the owned lane
/// refuses before any grammar compilation, so this path is unreachable in
/// practice — it exists so the wire handler compiles on every target and fails
/// closed if ever reached.
#[cfg(not(target_os = "macos"))]
fn owned_decode_vocabulary_digest(
    _tokenizer: &SanitizedTokenizer,
) -> Result<String, WireOperationError> {
    Err(WireOperationError::from_stable(
        StableError::artifact_invalid(),
        "owned decode is unsupported on this platform (owned_decode_unsupported)",
    ))
}

fn worker_constraint(
    compiled: &owned_decode_grammar_scheduler::TokenIdJsonConstraintV1,
) -> owned_decode_worker::protocol::TokenIdJsonConstraint {
    owned_decode_worker::protocol::TokenIdJsonConstraint {
        encoding_id: compiled.representation_revision.clone(),
        constraint_runtime_identity: compiled.constraint_runtime_identity.digest(),
        constraint_fingerprint: compiled.constraint_fingerprint.0.clone(),
        grammar_subset_revision: compiled
            .constraint_runtime_identity
            .grammar_subset_revision
            .clone(),
        grammar_compiler_revision: compiled
            .constraint_runtime_identity
            .grammar_compiler_revision
            .clone(),
        tokenizer_vocabulary_digest: compiled.tokenizer_vocabulary_digest.clone(),
        limits_manifest_id: compiled.limits_manifest_id.clone(),
        worker_constraint_runtime_revision: compiled
            .constraint_runtime_identity
            .worker_constraint_runtime_revision
            .clone(),
        canonical_schema_digest: compiled.canonical_schema_digest.clone(),
        initial_state_encoding: compiled.initial_state_encoding.clone(),
        initial_state_digest: compiled.initial_state_digest.clone(),
        compiled_automaton_digest: compiled.compiled_automaton_digest.clone(),
        automaton_bytes: compiled.automaton_bytes.clone(),
    }
}

/// A worker-setup outcome before `DecodeDispatch::dispatch` runs. Typed
/// owned-lane refusals must return through `OwnedDecodeRouter`, while unrelated
/// setup faults keep their existing wire error.
#[derive(Debug)]
enum OwnedDecodeDispatchPreparationError {
    Refused(owned_decode_routing::error::OwnedDecodeError),
    Wire(WireOperationError),
}

fn build_supervised_decode_dispatch_for_chain_k(
    state: &ModuleState,
    spec: &StoredModelConfig,
    entry: &owned_decode_routing::CatalogEntry,
    prompt_ids: Vec<u32>,
    constraint: Option<owned_decode_worker::protocol::TokenIdJsonConstraint>,
    deadline_ms: u64,
    decode_chain_k: u32,
) -> Result<worker_host::SupervisedDecodeDispatch, OwnedDecodeDispatchPreparationError> {
    use owned_decode_routing::error::OwnedDecodeError;
    use owned_decode_worker::{
        budget::BudgetPolicy,
        identity::QuarantineKey,
        protocol::{GenerateStart, Sampling},
        supervisor::TerminalControl,
        validation::WorkerStartContext,
    };
    use worker_host::{OwnedDecodeWorkerFactory, SupervisedDecodeDispatch, WorkerHostConfig};

    // The supervised owned worker is a Metal executable. Classify a non-macOS
    // setup as an owned-lane refusal before dispatch so selection can choose the
    // configured llama lane instead of treating its startup failure as terminal.
    if !cfg!(target_os = "macos") {
        return Err(OwnedDecodeDispatchPreparationError::Refused(
            OwnedDecodeError::Unsupported,
        ));
    }

    let worker_bin = spec
        .worker_bin
        .clone()
        .or_else(|| env::var_os("SYNAPSE_OWNED_DECODE_WORKER_BIN").map(PathBuf::from))
        .ok_or(OwnedDecodeDispatchPreparationError::Refused(
            OwnedDecodeError::Unavailable,
        ))?;
    if !worker_bin.is_file() {
        return Err(OwnedDecodeDispatchPreparationError::Refused(
            OwnedDecodeError::Unavailable,
        ));
    }
    let model_path = locator_path(&spec.model_locator, &state.model_cache)
        .map_err(OwnedDecodeDispatchPreparationError::Wire)?;
    let tokenizer_path = locator_path(&spec.tokenizer_locator, &state.model_cache)
        .map_err(OwnedDecodeDispatchPreparationError::Wire)?;
    let mut runtime_config = model_runtime_config(
        spec,
        &model_path.path,
        &[],
        state.model_cache.root(),
        state.runtime.microllm_max_tokens,
        None,
    );
    let decode_fingerprint = entry
        .decode_identity_inputs()
        .decode_fingerprint()
        .map_err(OwnedDecodeDispatchPreparationError::Refused)?;
    let processing_fingerprint = owned_decode_processing_fingerprint(entry)
        .map_err(OwnedDecodeDispatchPreparationError::Refused)?;
    let (runtime_config_digest, production_n) =
        owned_decode_runtime_identity(spec, entry, decode_chain_k);
    for (key, value) in [
        ("family", entry.family.as_str().to_string()),
        ("weight_quant", entry.weight_quant.as_str().to_string()),
        ("context_bucket", entry.max_context_tokens.to_string()),
        ("production_n", production_n.to_string()),
        ("decode_chain_k", decode_chain_k.to_string()),
        (
            "tokenizer_path",
            tokenizer_path.path.to_string_lossy().to_string(),
        ),
        ("decode_fingerprint", decode_fingerprint.0.clone()),
        ("processing_fingerprint", processing_fingerprint.0),
        ("runtime_config_digest", runtime_config_digest.clone()),
    ] {
        runtime_config.values.insert(key.to_string(), value);
    }
    let artifact = ValidatedArtifact {
        digest: spec
            .artifact_digest
            .strip_prefix("sha256:")
            .unwrap_or(&spec.artifact_digest)
            .to_string(),
        format: spec.artifact_format.clone(),
    };
    let runtime_dir = owned_decode_worker_runtime_dir(spec);
    let mut host_config = WorkerHostConfig::new(worker_bin, runtime_dir);
    host_config.worker_id = format!("synapse-owned-decode-{}", spec.model_id);
    host_config.model_id = Some(spec.model_id.clone());
    host_config.worker_forward_lines_per_sec = state.runtime.log.worker_forward_lines_per_sec;
    host_config.load_timeout = state.runtime.worker_load_timeout;
    host_config.request_timeout = Duration::from_millis(deadline_ms.max(1));
    let factory = OwnedDecodeWorkerFactory::new(host_config, artifact, runtime_config);
    let key = QuarantineKey::new(
        &state.machine_profile_hash,
        &decode_fingerprint.0,
        &runtime_config_digest,
    );
    let start = GenerateStart {
        generation_id: String::new(),
        loaded_model_ref: String::new(),
        decode_fingerprint: decode_fingerprint.0.clone(),
        runtime_config_digest: runtime_config_digest.clone(),
        prompt_ids,
        stop_ids: Vec::new(),
        max_tokens: 1,
        sampling: Sampling::greedy_top1(),
        constraint: constraint.clone(),
    };
    let context = WorkerStartContext {
        loaded_model_ref: String::new(),
        decode_fingerprint: decode_fingerprint.0,
        runtime_config_digest,
        expected_constraint: constraint,
    };
    SupervisedDecodeDispatch::new(
        factory,
        owned_decode_budget_store_path(spec),
        BudgetPolicy::default(),
        production_n,
        key,
        start,
        context,
        TerminalControl {
            deadline_at: Some(deadline_ms),
            cancel_at: None,
        },
    )
    .map_err(|_| OwnedDecodeDispatchPreparationError::Refused(OwnedDecodeError::Unavailable))
}

fn build_supervised_decode_dispatch(
    state: &ModuleState,
    spec: &StoredModelConfig,
    entry: &owned_decode_routing::CatalogEntry,
    prompt_ids: Vec<u32>,
    constraint: Option<owned_decode_worker::protocol::TokenIdJsonConstraint>,
    deadline_ms: u64,
) -> Result<worker_host::SupervisedDecodeDispatch, OwnedDecodeDispatchPreparationError> {
    build_supervised_decode_dispatch_for_chain_k(
        state,
        spec,
        entry,
        prompt_ids,
        constraint,
        deadline_ms,
        state.runtime.decode_chain_k,
    )
}

fn cached_supervised_decode_dispatch(
    state: &ModuleState,
    spec: &StoredModelConfig,
    entry: &owned_decode_routing::CatalogEntry,
    prompt: Vec<u32>,
    constraint: Option<owned_decode_worker::protocol::TokenIdJsonConstraint>,
    deadline_ms: u64,
) -> Result<Arc<Mutex<worker_host::SupervisedDecodeDispatch>>, OwnedDecodeDispatchPreparationError>
{
    cached_supervised_decode_dispatch_for_chain_k(
        state,
        spec,
        entry,
        prompt,
        constraint,
        deadline_ms,
        state.runtime.decode_chain_k,
    )
}

fn cached_supervised_decode_dispatch_for_chain_k(
    state: &ModuleState,
    spec: &StoredModelConfig,
    entry: &owned_decode_routing::CatalogEntry,
    prompt: Vec<u32>,
    constraint: Option<owned_decode_worker::protocol::TokenIdJsonConstraint>,
    deadline_ms: u64,
    decode_chain_k: u32,
) -> Result<Arc<Mutex<worker_host::SupervisedDecodeDispatch>>, OwnedDecodeDispatchPreparationError>
{
    // The shared cache represents the configured production request shape, so
    // the worker loaded by certification is also available to the first served
    // request. The K=1 comparison shape for a configured K>1 probe is private:
    // it has a distinct runtime identity and must not replace production cache.
    let cache_configured_shape = decode_chain_k == state.runtime.decode_chain_k;
    if cache_configured_shape {
        if let Some(dispatch) = state
            .runtime
            .owned_decode_dispatches
            .lock()
            .map_err(|_| {
                OwnedDecodeDispatchPreparationError::Wire(WireOperationError::from_stable(
                    StableError::engine_crashed(Some(100)),
                    "owned-decode dispatch cache is unavailable",
                ))
            })?
            .get(&spec.model_id)
            .cloned()
        {
            return Ok(dispatch);
        }
    }
    let created = if cache_configured_shape {
        build_supervised_decode_dispatch(state, spec, entry, prompt, constraint, deadline_ms)?
    } else {
        build_supervised_decode_dispatch_for_chain_k(
            state,
            spec,
            entry,
            prompt,
            constraint,
            deadline_ms,
            decode_chain_k,
        )?
    };
    let created = Arc::new(Mutex::new(created));
    if !cache_configured_shape {
        return Ok(created);
    }
    let mut dispatches = state.runtime.owned_decode_dispatches.lock().map_err(|_| {
        OwnedDecodeDispatchPreparationError::Wire(WireOperationError::from_stable(
            StableError::engine_crashed(Some(100)),
            "owned-decode dispatch cache is unavailable",
        ))
    })?;
    Ok(dispatches
        .entry(spec.model_id.clone())
        .or_insert_with(|| created.clone())
        .clone())
}

async fn dispatch_supervised_decode(
    dispatch: Arc<Mutex<worker_host::SupervisedDecodeDispatch>>,
    prompt: Vec<u32>,
    constraint: Option<owned_decode_worker::protocol::TokenIdJsonConstraint>,
    deadline_ms: u64,
    command: owned_decode_routing::DispatchedCommand,
) -> Result<owned_decode_routing::ExecutionSuccess, WireOperationError> {
    tokio::task::spawn_blocking(move || {
        let mut dispatch = dispatch.lock().map_err(|_| {
            WireOperationError::from_stable(
                StableError::engine_crashed(Some(100)),
                "owned-decode dispatch cache is unavailable",
            )
        })?;
        dispatch.set_request(prompt, constraint, deadline_ms);
        owned_decode_routing::DecodeDispatch::dispatch(&mut *dispatch, &command).map_err(|error| {
            WireOperationError::from_stable(
                StableError::engine_crashed(Some(100)),
                format!(
                    "owned-decode worker-path dispatch failed: {}",
                    error.as_str()
                ),
            )
        })
    })
    .await
    .map_err(|error| {
        WireOperationError::from_stable(
            StableError::engine_crashed(Some(100)),
            format!("owned-decode worker-path task failed: {error}"),
        )
    })?
}

struct OwnedDecodeEnvironmentInputs {
    processing_fingerprint: Fingerprint,
    runtime_config_digest: String,
    constraint_runtime_identity: Option<String>,
    llama: Option<owned_decode_routing::lane::LlamaLane>,
    equivalent_fingerprints: BTreeSet<Fingerprint>,
}

fn owned_decode_environment(
    state: &ModuleState,
    spec: &StoredModelConfig,
    entry: &owned_decode_routing::CatalogEntry,
    decode_fingerprint: &Fingerprint,
    inputs: OwnedDecodeEnvironmentInputs,
) -> owned_decode_routing::RoutingEnvironment {
    use owned_decode_routing::certification::{
        CertificationAccess, ConstrainedCertKey, UnconstrainedCertKey,
    };

    let OwnedDecodeEnvironmentInputs {
        processing_fingerprint,
        runtime_config_digest,
        constraint_runtime_identity,
        llama,
        equivalent_fingerprints,
    } = inputs;
    let constraint_runtime_identities = constraint_runtime_identity
        .clone()
        .into_iter()
        .collect::<Vec<_>>();
    let worker_path_evidence = owned_decode_worker_path_identity();
    let match_inputs = OwnedDecodeMatchInputs {
        revisioned_machine_profile_hash: state.revisioned_machine_profile_hash.clone(),
        profile_activation_epoch: state.profile_activation_epoch,
        model_id: entry.entry_id.clone(),
        decode_fingerprint: decode_fingerprint.0.clone(),
        processing_fingerprint: processing_fingerprint.0,
        runtime_config_digest,
        constraint_runtime_identities,
        worker_path_evidence,
        evidence_schema_revision: store::CERT_EVIDENCE_SCHEMA_REVISION.to_string(),
        g_dec_manifest_revision: store::G_DEC_MANIFEST_REVISION.to_string(),
    };
    let certification = PersistentDecodeCertification {
        store: Arc::clone(&state.store),
        match_inputs: match_inputs.clone(),
    };
    let unconstrained_certified = CertificationAccess::is_unconstrained_certified(
        &certification,
        &UnconstrainedCertKey {
            machine_profile_hash: state.revisioned_machine_profile_hash.clone(),
            decode_fingerprint: decode_fingerprint.clone(),
        },
    );
    let constrained_certified = constraint_runtime_identity.as_ref().is_none_or(|identity| {
        CertificationAccess::is_constrained_certified(
            &certification,
            &ConstrainedCertKey {
                machine_profile_hash: state.revisioned_machine_profile_hash.clone(),
                decode_fingerprint: decode_fingerprint.clone(),
                constraint_runtime_identity: identity.clone(),
            },
        )
    });
    let quarantined = owned_decode_quarantined(
        state,
        spec,
        decode_fingerprint,
        &match_inputs.runtime_config_digest,
    );
    let artifacts_trusted = if entry.weight_quant.is_q8() {
        entry.q8.as_ref().is_some_and(|q8| {
            state
                .runtime
                .owned_decode_q8
                .lock()
                .ok()
                .is_some_and(|registry| {
                    registry
                        .entry(&entry.artifact_source_digest, &q8.quantizer_revision)
                        .is_some_and(|artifact| {
                            artifact.trust_state
                                == owned_decode_routing::q8ingest::TrustState::Trusted
                        })
                })
        })
    } else {
        true
    };
    let scheduler: owned_decode_contracts::SchedulerManifest = serde_json::from_str(include_str!(
        "../owned-decode-manifests/decode-sched-manifest-v1.json"
    ))
    .expect("checked-in scheduler manifest parses");
    let scheduler_status = owned_decode_certification::ingest_scheduler_evidence(&scheduler);
    let scheduler_evidence_committed =
        owned_decode_certification::scheduler_evidence_committed(&scheduler_status);
    let wire_bindings = owned_decode_contracts::WireErrorBindingsManifest {
        manifest_revision: "owned-decode-wire-error-bindings-v1".to_string(),
        schema_revision: "owned-decode-contracts-v1".to_string(),
        request_contract_revision: "wire-contract-v1".to_string(),
        deadline_error_id: "deadline_exceeded".to_string(),
        cancellation_error_id: "cancelled".to_string(),
        wire_changelog: Vec::new(),
    };
    let admission_evaluation = state
        .store
        .owned_decode_admission_evaluation(&match_inputs)
        .unwrap_or(OwnedDecodeAdmissionEvaluation::Refused {
            approval: Box::new(None),
            refusal: OwnedDecodeAdmissionRefusal::NotCertified,
        });
    let admission = admission_evaluation.admission().cloned();
    let approval = admission_evaluation.approval();
    let approval_grammar_enabled = approval.is_some_and(|approval| approval.grammar_enabled);
    let disabled_reason = match admission_evaluation.refusal() {
        Some(OwnedDecodeAdmissionRefusal::ApprovalDisabled { disabled_reason }) => {
            Some(disabled_reason.clone())
        }
        _ => None,
    };
    let serving_inputs = owned_decode_routing::lane::ServingPredicateInputs {
        approval_present: approval.is_some(),
        approval_enabled: approval.is_some_and(|approval| approval.enabled),
        approval_identity_matches: approval.is_some_and(|approval| {
            approval.model_id == match_inputs.model_id
                && approval.decode_fingerprint == match_inputs.decode_fingerprint
        }),
        current_profile_matches: admission.is_some(),
        current_epoch_valid: state.profile_activation_epoch > 0,
        certification_matches: admission.is_some()
            && unconstrained_certified
            && constrained_certified,
        evidence_revisions_compatible: admission.is_some(),
        gates_complete: admission.is_some(),
        processing_fingerprint_matches: admission.is_some(),
        runtime_config_digest_matches: admission.is_some(),
        worker_path_matches: admission.is_some(),
        constrained_identities_match: admission.is_some(),
        artifacts_trusted,
        // Subsumed rather than unchecked, which is not obvious from the
        // literal. This arm's contract is "the exact runtime and processing
        // identities are installed", and `admission` is the fenced
        // certification match, which compares processing_fingerprint,
        // runtime_config_digest, constraint_runtime_identities and
        // worker_path_evidence against the stored row before it resolves
        // (`store.rs:7636-7648`). A row that disagrees on any of them does
        // not match, so `admission.is_some()` -- already required by the
        // four arms above -- is the identity check. Passing `admission`
        // here too would add a fifth reading of one fact rather than a
        // check; the honest value is the constant, with the reason stated.
        identities_installed: true,
        quarantined,
        wire_bindings_literal: owned_decode_certification::wire_bindings_are_literal(
            &wire_bindings,
        ),
        scheduler_evidence_committed,
    };
    let serving = owned_decode_routing::lane::serving_predicate(&serving_inputs);
    let mut environment = owned_decode_routing::RoutingEnvironment::with_serving_evaluated(
        state.revisioned_machine_profile_hash.clone(),
        state.runtime.grammar_enabled,
        approval_grammar_enabled,
        serving,
        quarantined,
        llama,
        equivalent_fingerprints,
        constraint_runtime_identity,
    );
    if let Some(disabled_reason) = disabled_reason {
        environment = environment.with_serving_refusal_message(disabled_reason);
    }
    if let Some(admission) = admission {
        let boundary_reader: Arc<dyn owned_decode_routing::lane::AdmissionBoundaryReader> =
            state.store.clone();
        environment = environment.with_admission_boundary(
            owned_decode_routing::lane::AdmissionBoundarySnapshot {
                profile_activation_epoch: admission.profile_activation_epoch,
                model_id: admission.approval.model_id,
                decode_fingerprint: admission.approval.decode_fingerprint,
                approval_semantic_digest: admission.approval.semantic_digest,
                approval_generation: admission.approval.generation,
            },
            boundary_reader,
        );
    }
    environment
}

fn owned_decode_worker_path_identity() -> Value {
    json!({
        "transport": worker_catalog_transport(),
        "protocol": owned_decode_worker::identity::WORKER_PROTOCOL_ID,
        "fixture_battery": "20x64-structural-band",
    })
}

fn owned_decode_gate_evidence() -> Value {
    Value::Array(
        (1..=12)
            .map(|number| {
                json!({
                    "id": format!("G-DEC-{number:02}"),
                    "status": "passed",
                    "manifest_revision": store::G_DEC_MANIFEST_REVISION,
                })
            })
            .collect(),
    )
}

fn owned_decode_probe_match_inputs(
    state: &ModuleState,
    model: &EmbeddingModel,
    profile_hash: String,
    profile_epoch: u64,
    constraint_runtime_identities: Vec<String>,
) -> Result<OwnedDecodeMatchInputs, WireOperationError> {
    let spec = state
        .runtime
        .catalog
        .lock()
        .ok()
        .and_then(|catalog| catalog.get(&model.model_id).map(|slot| slot.spec.clone()))
        .ok_or_else(|| artifact_invalid_error("missing catalog entry for owned-decode probe"))?;
    let entry = owned_decode_catalog_entry(&spec)
        .map_err(|error| artifact_invalid_error(error.as_str()))?;
    let decode_fingerprint = entry
        .decode_identity_inputs()
        .decode_fingerprint()
        .map_err(|error| artifact_invalid_error(error.as_str()))?;
    let processing_fingerprint = owned_decode_processing_fingerprint(&entry)
        .map_err(|error| artifact_invalid_error(error.as_str()))?;
    let (runtime_config_digest, _) =
        owned_decode_runtime_identity(&spec, &entry, state.runtime.decode_chain_k);
    let constraint_runtime_identities = if constraint_runtime_identities.is_empty() {
        Vec::new()
    } else {
        vec![owned_decode_probe_constraint_identity(
            model,
            &decode_fingerprint,
        )?]
    };
    Ok(OwnedDecodeMatchInputs {
        revisioned_machine_profile_hash: profile_hash,
        profile_activation_epoch: profile_epoch,
        model_id: entry.entry_id,
        decode_fingerprint: decode_fingerprint.0,
        processing_fingerprint: processing_fingerprint.0,
        runtime_config_digest,
        constraint_runtime_identities,
        worker_path_evidence: owned_decode_worker_path_identity(),
        evidence_schema_revision: store::CERT_EVIDENCE_SCHEMA_REVISION.to_string(),
        g_dec_manifest_revision: store::G_DEC_MANIFEST_REVISION.to_string(),
    })
}

fn owned_decode_probe_constraint_identity(
    model: &EmbeddingModel,
    decode_fingerprint: &Fingerprint,
) -> Result<String, WireOperationError> {
    let vocabulary_digest = owned_decode_vocabulary_digest(&model.tokenizer)?;
    let compiled = owned_decode_grammar_scheduler::compile_grammar(
        r#"{"type":"null"}"#,
        &owned_decode_grammar_scheduler::CompileContext {
            base_decode_fingerprint: decode_fingerprint.clone(),
            tokenizer_vocabulary_digest: vocabulary_digest,
        },
        &owned_decode_grammar_scheduler::GrammarSubsetManifest::default(),
    )
    .map_err(|error| artifact_invalid_error(error.message))?;
    Ok(compiled.constraint.constraint_runtime_identity.digest())
}

async fn route_owned_decode_wire(
    state: Arc<ModuleState>,
    params: &MicroLlmOneshotParams,
    model_id: &str,
    job_id: String,
    started: Instant,
) -> HandlerOutcome {
    use owned_decode_routing::lane::{LaneKind, LlamaLane};
    use owned_decode_routing::request::{OneshotRequest, SamplingMode};

    let spec = match state
        .runtime
        .catalog
        .lock()
        .ok()
        .and_then(|catalog| catalog.get(model_id).map(|slot| slot.spec.clone()))
    {
        Some(spec) => spec,
        None => return channel_error("invalid_request", format!("unknown model '{model_id}'")),
    };
    let entry = match owned_decode_catalog_entry(&spec) {
        Ok(entry) => entry,
        Err(error) => {
            return channel_error(
                error.as_str(),
                format!("owned-decode catalog entry '{model_id}' is unsupported"),
            )
        }
    };
    let owned_model =
        match ensure_model_loaded_for_control(Arc::clone(&state), model_id, params.deadline_ms)
            .await
        {
            Ok(model) => model,
            Err(error) => return result_outcome(error_payload(&state, error)),
        };
    let resolution_owned_refusal = owned_model.owned_decode_resolution_refusal;
    let alias_table = match state.store.alias_table() {
        Ok(table) => table,
        Err(error) => return channel_error("store_failure", error.to_string()),
    };
    if params
        .required_epoch
        .is_some_and(|required| required > alias_table.table_epoch)
    {
        return result_outcome(error_payload(
            &state,
            WireOperationError::from_stable(
                StableError::migration_required(),
                "requested alias table epoch is newer than the module table",
            ),
        ));
    }
    let tokenizer_path = match locator_path(&spec.tokenizer_locator, &state.model_cache) {
        Ok(path) => path,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };
    let tokenizer = match SanitizedTokenizer::from_file(
        &tokenizer_path.path,
        TokenizerConfig {
            max_tokens: spec.max_tokens,
        },
    ) {
        Ok(tokenizer) => tokenizer,
        Err(error) => return channel_error("artifact_invalid", error.to_string()),
    };
    let tokenized = match tokenizer.tokenize_batch([params.prompt.as_str()]) {
        Ok(tokenized) => tokenized,
        Err(error) => return channel_error("invalid_request", error.to_string()),
    };
    let prompt = tokenized.batch.items.first().cloned().unwrap_or_default();
    let decode_fingerprint = match entry.decode_identity_inputs().decode_fingerprint() {
        Ok(fingerprint) => fingerprint,
        Err(error) => return channel_error(error.as_str(), "invalid decode identity"),
    };
    let processing_fingerprint = match owned_decode_processing_fingerprint(&entry) {
        Ok(fingerprint) => fingerprint,
        Err(error) => return channel_error(error.as_str(), "invalid processing identity"),
    };
    let constrained = params
        .grammar
        .as_deref()
        .is_some_and(|grammar| !grammar.trim().is_empty());
    let approval_grammar_enabled = state
        .store
        .get_approval(&entry.entry_id, &decode_fingerprint.0)
        .ok()
        .flatten()
        .is_some_and(|approval| approval.grammar_enabled);
    if constrained && (!state.runtime.grammar_enabled || !approval_grammar_enabled) {
        record_admission_refusal(&state.runtime, model_id, Some(&job_id), "grammar_disabled");
        return channel_error(
            "grammar_disabled",
            "constrained owned-decode requests require both runtime and approval grammar enablement",
        );
    }
    let compiled_constraint = if resolution_owned_refusal.is_some() {
        // The platform refusal must reach lane selection before Metal-only
        // grammar setup, which cannot construct a tokenizer vocabulary here.
        None
    } else {
        match params
            .grammar
            .as_deref()
            .filter(|grammar| !grammar.trim().is_empty())
        {
            Some(grammar) => {
                let vocabulary_digest = match owned_decode_vocabulary_digest(&tokenizer) {
                    Ok(digest) => digest,
                    Err(error) => return result_outcome(error_payload(&state, error)),
                };
                match owned_decode_grammar_scheduler::compile_grammar(
                    grammar,
                    &owned_decode_grammar_scheduler::CompileContext {
                        base_decode_fingerprint: decode_fingerprint.clone(),
                        tokenizer_vocabulary_digest: vocabulary_digest,
                    },
                    &owned_decode_grammar_scheduler::GrammarSubsetManifest::default(),
                ) {
                    Ok(compiled) => Some(compiled),
                    Err(error) => return channel_error(error.wire_error().as_str(), error.message),
                }
            }
            None => None,
        }
    };
    let constraint_runtime_identity = compiled_constraint
        .as_ref()
        .map(|compiled| compiled.constraint.constraint_runtime_identity.digest());
    let worker_constraint = compiled_constraint
        .as_ref()
        .map(|compiled| worker_constraint(&compiled.constraint));
    let (runtime_config_digest, _) =
        owned_decode_runtime_identity(&spec, &entry, state.runtime.decode_chain_k);

    let llama_spec = if !constrained {
        state.runtime.catalog.lock().ok().and_then(|catalog| {
            catalog
                .values()
                .map(|slot| &slot.spec)
                .find(|candidate| {
                    candidate.model_id != model_id
                        && candidate.engine == LLAMA_ENGINE
                        && candidate.task == ModelTask::Generate.as_str()
                })
                .cloned()
        })
    } else {
        None
    };
    let llama_model = if let Some(fallback) = llama_spec.as_ref() {
        match ensure_model_loaded_for_control(
            Arc::clone(&state),
            &fallback.model_id,
            params.deadline_ms,
        )
        .await
        {
            Ok(model) => Some(model),
            Err(error) => return result_outcome(error_payload(&state, error)),
        }
    } else {
        None
    };
    let llama_lane = llama_spec.as_ref().map(|fallback| LlamaLane {
        decode_fingerprint: fallback.fingerprint.clone(),
        processing_fingerprint: fallback.fingerprint.clone(),
    });
    let equivalent_fingerprints = alias_table
        .equivalent_fingerprints_at(&decode_fingerprint, now_ms())
        .into_iter()
        .collect::<BTreeSet<_>>();
    let environment = owned_decode_environment(
        &state,
        &spec,
        &entry,
        &decode_fingerprint,
        OwnedDecodeEnvironmentInputs {
            processing_fingerprint: processing_fingerprint.clone(),
            runtime_config_digest: runtime_config_digest.clone(),
            constraint_runtime_identity: constraint_runtime_identity.clone(),
            llama: llama_lane,
            equivalent_fingerprints,
        },
    )
    .with_decode_chain_k(state.runtime.decode_chain_k);
    let environment = match resolution_owned_refusal {
        Some(refusal) => environment.with_resolution_owned_refusal(refusal),
        None => environment,
    };

    let certification = PersistentDecodeCertification {
        store: Arc::clone(&state.store),
        match_inputs: OwnedDecodeMatchInputs {
            revisioned_machine_profile_hash: state.revisioned_machine_profile_hash.clone(),
            profile_activation_epoch: state.profile_activation_epoch,
            model_id: entry.entry_id.clone(),
            decode_fingerprint: decode_fingerprint.0.clone(),
            processing_fingerprint: processing_fingerprint.0.clone(),
            runtime_config_digest: runtime_config_digest.clone(),
            constraint_runtime_identities: constraint_runtime_identity
                .clone()
                .into_iter()
                .collect(),
            worker_path_evidence: owned_decode_worker_path_identity(),
            evidence_schema_revision: store::CERT_EVIDENCE_SCHEMA_REVISION.to_string(),
            g_dec_manifest_revision: store::G_DEC_MANIFEST_REVISION.to_string(),
        },
    };
    let q8 = match state.runtime.owned_decode_q8.lock() {
        Ok(registry) => registry.clone(),
        Err(_) => {
            return channel_error(
                "owned_decode_unavailable",
                "owned-decode Q8 ingest registry is unavailable",
            )
        }
    };
    let context_buckets: owned_decode_contracts::ContextBucketsManifest = serde_json::from_str(
        include_str!("../owned-decode-manifests/decode-context-buckets-v1.json"),
    )
    .expect("checked-in decode context buckets parse");
    let router = owned_decode_routing::OwnedDecodeRouter::new(
        owned_decode_routing::family::FamilyRegistry::production(),
        context_buckets,
        q8,
        Box::new(certification),
    );
    let request = OneshotRequest {
        family: entry.family,
        weight_quant: entry.weight_quant,
        prompt_token_count: prompt.len().min(u32::MAX as usize) as u32,
        max_tokens: params.max_tokens,
        sampling: SamplingMode::GreedyTop1,
        grammar: if resolution_owned_refusal.is_some() && constrained {
            // Preserve owned-only request shape while the platform refusal is
            // routed. Grammar compilation above is intentionally skipped.
            Some(Value::Null)
        } else {
            params
                .grammar
                .as_deref()
                .filter(|grammar| !grammar.trim().is_empty())
                .and_then(|grammar| serde_json::from_str(grammar).ok())
        },
        required_fingerprint: params.required_fingerprint.clone().map(Fingerprint),
        allow_equivalent: params.allow_equivalent,
        target_fingerprint: params.target_fingerprint.clone().map(Fingerprint),
        required_processing_fingerprint: None,
        owned_only: params.owned_only,
    };
    let total_tokens = u64::from(request.prompt_token_count) + u64::from(params.max_tokens);
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
    let admission = match state.runtime.admit_inline(
        model_id,
        Some(&job_id),
        QueueClass::Interactive,
        request_bytes_for_texts([params.prompt.as_str()]),
        params.deadline_ms,
        params.max_queue_ms,
    ) {
        Ok(admission) => admission,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };
    let permit = match acquire_execution_permit(&state.runtime, Some(admission.deadline())).await {
        Ok(permit) => permit,
        Err(error) => return result_outcome(error_payload(&state, error)),
    };
    let _activity = state.runtime.activity_telemetry.begin(model_id);
    let deadline_ms = params
        .deadline_ms
        .unwrap_or(state.runtime.inline.deadline_ms)
        .max(1);
    let (owned, pre_dispatch_owned_refusal) = if resolution_owned_refusal.is_some() {
        // Resolution already classified this platform's owned lane as unavailable;
        // do not attempt to construct a Metal worker before lane selection.
        (None, None)
    } else {
        match cached_supervised_decode_dispatch(
            &state,
            &spec,
            &entry,
            prompt.clone(),
            worker_constraint.clone(),
            deadline_ms,
        ) {
            Ok(dispatch) => (Some(dispatch), None),
            Err(OwnedDecodeDispatchPreparationError::Refused(refusal)) => (None, Some(refusal)),
            Err(OwnedDecodeDispatchPreparationError::Wire(error)) => {
                return result_outcome(error_payload(&state, error));
            }
        }
    };
    let environment = match pre_dispatch_owned_refusal {
        Some(refusal) => environment.with_pre_dispatch_owned_refusal(refusal),
        None => environment,
    };
    let sidecar_hint_bank_source = sidecar_hint_bank_source(
        Arc::clone(&state),
        &params.prompt,
        params.grammar.as_deref(),
        &tokenizer,
        compiled_constraint.as_ref(),
        worker_constraint.as_ref(),
        deadline_ms,
    );
    let dispatch = WireDecodeDispatch {
        owned,
        prompt,
        constraint: worker_constraint,
        sidecar_hint_bank_source,
        deadline_ms,
        llama: llama_model.clone(),
        llama_output: None,
    };
    let generation_id = format!("{}-{}", state.module_generation, now_ms());
    // The worker and the terminal envelope must use the same generation identity.
    // Keep a copy because the routing closure owns the dispatch-side value.
    let generation_id_for_payload = generation_id.clone();
    let n_prompt = request.prompt_token_count as usize;
    let routed = tokio::task::spawn_blocking(move || {
        let mut dispatch = dispatch;
        let routed = router.route_oneshot(
            &environment,
            &entry,
            &request,
            &generation_id,
            &mut dispatch,
        );
        (routed, dispatch)
    })
    .await;
    drop(permit);
    drop(admission);
    let (routed, dispatch) = match routed {
        Ok(result) => result,
        Err(error) => {
            return result_outcome(error_payload(
                &state,
                WireOperationError::from_stable(
                    StableError::engine_crashed(Some(100)),
                    format!("owned-decode dispatch join failed: {error}"),
                ),
            ))
        }
    };
    let mut routed = match routed {
        Ok(response) => response,
        Err(failure) => {
            record_admission_refusal(&state.runtime, model_id, Some(&job_id), failure.wire_id());
            let detail = failure.message.as_deref();
            return channel_error(
                failure.wire_id(),
                match (failure.underlying_owned_decode_refusal_id, detail) {
                    (Some(underlying), Some(detail)) => format!(
                        "owned-decode request refused: {} (underlying {}; {detail})",
                        failure.wire_id(),
                        underlying.as_str()
                    ),
                    (Some(underlying), None) => format!(
                        "owned-decode request refused: {} (underlying {})",
                        failure.wire_id(),
                        underlying.as_str()
                    ),
                    (None, Some(detail)) => {
                        format!(
                            "owned-decode request refused: {} ({detail})",
                            failure.wire_id()
                        )
                    }
                    (None, None) => format!("owned-decode request refused: {}", failure.wire_id()),
                },
            );
        }
    };
    if let Some(compiled) = compiled_constraint.as_ref() {
        routed.provenance = routed.provenance.clone().with_constraint(
            compiled.constraint.constraint_runtime_identity.digest(),
            compiled.constraint.constraint_fingerprint.clone(),
            compiled
                .constraint
                .constraint_runtime_identity
                .grammar_compiler_revision
                .clone(),
        );
    }
    let text = if routed.lane == LaneKind::Llama {
        dispatch
            .llama_output
            .as_ref()
            .map(|output| output.text.clone())
            .unwrap_or_default()
    } else {
        match tokenizer.decode(&routed.generated_token_ids) {
            Ok(text) => text,
            Err(error) => return channel_error("artifact_invalid", error.to_string()),
        }
    };
    let completion_lane = match routed.lane {
        LaneKind::OwnedDecode => "decode",
        LaneKind::Llama => "llama",
    };
    let completion_tokens =
        (n_prompt as u64).saturating_add(routed.generated_token_ids.len() as u64);
    let selected_model = if routed.lane == LaneKind::Llama {
        llama_model.as_deref()
    } else {
        None
    };
    let fingerprint = selected_model
        .map(|model| model.fingerprint.clone())
        .unwrap_or_else(|| spec.fingerprint.clone());
    let engine = selected_model
        .map(|model| model.engine_identity.clone())
        .unwrap_or_else(|| spec.engine_identity.clone());
    let equivalent_to = alias_table
        .equivalent_fingerprints_at(&fingerprint, now_ms())
        .into_iter()
        .collect();
    let provenance = &routed.provenance;
    let payload = MicroLlmOneshotPayload {
        text,
        finish_reason: routed.finish_reason.as_str().to_string(),
        n_prompt,
        n_gen: routed.generated_token_ids.len(),
        generated_token_ids: routed.generated_token_ids.clone(),
        runtime_config_digest: Some(runtime_config_digest),
        derived_digest: spec
            .engine_identity
            .build_flags
            .get("derived_digest")
            .cloned(),
        generation_id: Some(generation_id_for_payload),
        real_token_counts: tokenized.real_token_counts,
        truncation_disclosures: tokenized.disclosures,
    };
    let envelope = ResponseEnvelope {
        fingerprint,
        table_epoch: alias_table.table_epoch,
        dims: 0,
        provenance: ResponseProvenance {
            engine,
            remote: None,
            owned_decode: synapse_core::OwnedDecodeResponseProvenance {
                lane: Some(provenance.lane.clone()),
                worker: Some(provenance.worker.clone()),
                risk_class: Some(provenance.risk_class.clone()),
                decode_fingerprint: Some(provenance.decode_fingerprint.clone()),
                processing_fingerprint: Some(provenance.processing_fingerprint.clone()),
                fallback_reason: provenance.fallback_reason.clone(),
                lane_finish_reason: provenance.lane_finish_reason.clone(),
                worker_generation: (provenance.worker_generation != 0)
                    .then_some(provenance.worker_generation),
                last_completed_quantum_sequence: (provenance.last_completed_quantum_sequence != 0)
                    .then_some(provenance.last_completed_quantum_sequence),
                crash_retry_count: provenance.crash_retry_count,
                failure_classifications: provenance.failure_classifications.clone(),
                constraint_runtime_identity: provenance.constraint_runtime_identity.clone(),
                constraint_fingerprint: provenance.constraint_fingerprint.clone(),
                grammar_compiler_revision: provenance.grammar_compiler_revision.clone(),
                chain_k: provenance.chain_k,
                underlying_owned_decode_refusal_id: provenance
                    .underlying_owned_decode_refusal_id
                    .clone(),
            },
        },
        module_generation: state.module_generation,
        equivalent_to,
        payload,
    };
    state
        .runtime
        .activity_telemetry
        .record_completed_tokens(completion_tokens);
    log_job_done(
        model_id,
        &job_id,
        completion_lane,
        completion_tokens,
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
    remote_profile_hash: Option<&str>,
    logical_handle: Option<&str>,
    constraints: &Value,
    items: &[(String, String)],
) -> String {
    let mut hasher = Sha256::new();
    update_digest_bytes(&mut hasher, b"synapse-request-digest-v1");
    update_digest_bytes(&mut hasher, op.as_bytes());
    update_digest_bytes(&mut hasher, synapse_model_id.as_bytes());
    hasher.update([u8::from(remote_profile_hash.is_some())]);
    if let Some(remote_profile_hash) = remote_profile_hash {
        update_digest_bytes(&mut hasher, remote_profile_hash.as_bytes());
    }
    hasher.update([u8::from(logical_handle.is_some())]);
    if let Some(logical_handle) = logical_handle {
        update_digest_bytes(&mut hasher, logical_handle.as_bytes());
    }
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
    let admission = match state.store.admit_job(
        &request_key,
        &work.request_digest,
        "embed.batch",
        state.module_generation,
        None,
        &json!({
            "model": work.model.model_id.clone(),
            "items": work.ids.len(),
            "request_bytes": work.request_bytes,
            "total_tokens": work.total_tokens,
        }),
        now,
        state.runtime.jobs.execution_ttl_ms,
        state.runtime.jobs.result_retention_ttl_ms,
    ) {
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
        Ok(JobAttemptClaim::Attached { .. } | JobAttemptClaim::NotClaimable(_)) | Err(_) => return,
    };

    if !enforce_checkpoint_continuity(
        &state,
        &job_id,
        &record.request_digest,
        &work.model.model_id,
        record.logical_handle.as_deref(),
    )
    .await
    {
        return;
    }

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
        work.alias_table,
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

async fn apply_checkpoint_continuity(
    store: &SynapseStore,
    continuity_check: &dyn ContinuityCheck,
    job_id: &str,
    request_digest: &str,
    synapse_model_id: &str,
    logical_handle: Option<&str>,
    now_ms: u64,
) -> Result<bool, SynapseStoreError> {
    if store.checkpoint_count(request_digest)? == 0 {
        return Ok(true);
    }
    match continuity_check
        .check(request_digest, synapse_model_id, logical_handle)
        .await
    {
        Ok(()) => Ok(true),
        Err(error) => {
            store.quarantine_job_for_continuity(
                job_id,
                &format!("continuity check failed before appending checkpoints: {error}"),
                now_ms,
            )?;
            Ok(false)
        }
    }
}

async fn enforce_checkpoint_continuity(
    state: &ModuleState,
    job_id: &str,
    request_digest: &str,
    synapse_model_id: &str,
    logical_handle: Option<&str>,
) -> bool {
    match apply_checkpoint_continuity(
        &state.store,
        state.continuity_check.as_ref(),
        job_id,
        request_digest,
        synapse_model_id,
        logical_handle,
        now_ms(),
    )
    .await
    {
        Ok(allowed) => {
            if !allowed {
                state.runtime.admission_telemetry.record_job_failed();
            }
            allowed
        }
        Err(error) => {
            fail_job_with_wire_error(
                state,
                job_id,
                true,
                WireOperationError::from_stable(
                    StableError::engine_crashed(Some(100)),
                    format!("inspect checkpoint continuity trigger: {error}"),
                ),
            );
            false
        }
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
    alias_table: AliasTable,
    job_id: &str,
    first_page_no: u32,
) -> Result<(Value, Vec<PreparedJobPage>), WireOperationError> {
    let dims = vectors.first().map(Vec::len).unwrap_or(0) as u32;
    let equivalent_to = equivalent_fingerprints(&alias_table, model);
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
                    provider_request_id: None,
                })
            })
            .collect::<Result<Vec<_>, WireOperationError>>()?;
        let envelope = ResponseEnvelope {
            fingerprint: model.fingerprint.clone(),
            table_epoch: alias_table.table_epoch,
            dims,
            provenance: ResponseProvenance {
                engine: model.engine_identity.clone(),
                remote: None,
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
    let _ = state.store.fail_job(
        job_id,
        transient,
        &serde_json::to_value(error).expect("wire error serializes"),
        now_ms(),
    );
    state.runtime.admission_telemetry.record_job_failed();
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
        if record.state == JOB_STATE_PAUSED_NEEDS_REAUTH {
            if let Some(logical_handle) = &record.logical_handle {
                map.insert(
                    "logical_handle".to_string(),
                    Value::String(logical_handle.clone()),
                );
            }
            if let Some(paused_at_ms) = record.paused_at_ms {
                map.insert("paused_at_ms".to_string(), Value::from(paused_at_ms));
            }
            if let Some(resume_deadline_ms) = record.resume_deadline_ms {
                map.insert(
                    "resume_deadline_ms".to_string(),
                    Value::from(resume_deadline_ms),
                );
            }
            map.insert("action".to_string(), Value::String("reauth".to_string()));
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

async fn job_resume(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let params: JobResumeParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid job.resume params: {error}"),
            )
        }
    };
    let now = now_ms();
    let prior_generation = state
        .store
        .get_job(&params.job_id)
        .ok()
        .flatten()
        .map(|record| record.module_generation);
    let resumed = match state.store.resume_paused_job(
        &params.job_id,
        state.module_generation,
        now,
        state.runtime.jobs.execution_ttl_ms,
    ) {
        Ok(resumed) => resumed,
        Err(error) => return channel_error("store_failure", error.to_string()),
    };
    if resumed && prior_generation.is_some_and(|gen| gen < state.module_generation) {
        state.runtime.admission_telemetry.record_job_inherited();
    }
    let record = match state.store.get_job(&params.job_id) {
        Ok(Some(record)) => record,
        Ok(None) => return channel_error("invalid_request", "unknown or expired job_id"),
        Err(error) => return channel_error("store_failure", error.to_string()),
    };
    // Re-spawn execution for the resumed job. Only remote jobs pause for
    // re-authentication (vault_locked / needs_reauth), so only remote batch
    // jobs need re-dispatch here.
    if resumed
        && record.kind == "embed.batch"
        && record
            .params_json
            .as_ref()
            .and_then(|params| params.get("model"))
            .and_then(Value::as_str)
            .is_some_and(|model_id| state.remote_gateway.is_remote(model_id))
    {
        if let Err(error) =
            respawn_resumed_remote_job(&state, &record, record.logical_handle.as_deref())
        {
            fail_job_with_wire_error(&state, &record.job_id, false, error);
        }
    }
    let record = match state.store.get_job(&params.job_id) {
        Ok(Some(record)) => record,
        Ok(None) => return channel_error("store_failure", "resumed job disappeared"),
        Err(error) => return channel_error("store_failure", error.to_string()),
    };
    result_outcome(job_status_payload(&state, &record))
}

/// Re-dispatch a resumed remote job from its stored request parameters. Paused
/// jobs are not selected by a background queue consumer, so resumption must
/// explicitly start the same execution task used for initial admission.
fn respawn_resumed_remote_job(
    state: &Arc<ModuleState>,
    record: &JobRecord,
    logical_handle: Option<&str>,
) -> Result<(), WireOperationError> {
    let params_json = record.params_json.as_ref().ok_or_else(|| {
        WireOperationError::from_stable(
            StableError::artifact_invalid(),
            "resumed remote job has no stored request parameters",
        )
    })?;
    let model_id = params_json
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            WireOperationError::from_stable(
                StableError::artifact_invalid(),
                "resumed remote job parameters are missing model",
            )
        })?;
    let profile = state.remote_gateway.profile(model_id).ok_or_else(|| {
        WireOperationError::from_stable(
            StableError::artifact_invalid(),
            format!("remote profile not found for model '{model_id}'"),
        )
    })?;
    if let Some(logical_handle) = logical_handle {
        let gateway_handle = state.remote_gateway.logical_handle(&profile);
        if gateway_handle.as_deref() != Some(logical_handle) {
            return Err(WireOperationError::from_stable(
                StableError::artifact_invalid(),
                "resumed job credential handle no longer matches the configured provider",
            ));
        }
    }
    let items_value = params_json.get("items").ok_or_else(|| {
        WireOperationError::from_stable(
            StableError::artifact_invalid(),
            "resumed remote job parameters are missing items",
        )
    })?;
    let items: Vec<EmbedBatchItem> =
        serde_json::from_value(items_value.clone()).map_err(|error| {
            WireOperationError::from_stable(
                StableError::artifact_invalid(),
                format!("resumed remote job items are invalid: {error}"),
            )
        })?;
    let deadline_ms = params_json
        .get("deadline_ms")
        .and_then(Value::as_u64)
        .unwrap_or(state.runtime.inline.deadline_ms);
    spawn_remote_embed_batch_job(
        Arc::clone(state),
        record.job_id.clone(),
        RemoteEmbedBatchJobWork {
            profile,
            request_digest: record.request_digest.clone(),
            items,
            deadline_ms,
        },
    );
    Ok(())
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
        JOB_STATE_QUEUED | JOB_STATE_RUNNING | JOB_STATE_PAUSED_NEEDS_REAUTH => {
            result_outcome(job_status_payload(&state, &record))
        }
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
    alias_table: AliasTable,
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
        table_epoch: alias_table.table_epoch,
        dims,
        provenance: ResponseProvenance {
            engine: model.engine_identity.clone(),
            remote: None,
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
    result_outcome(response)
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
    if let Ok(mut stats) = runtime.execution_stats.lock() {
        stats.waiters = stats.waiters.saturating_add(1);
    }
    let started = Instant::now();
    let permit_result = match deadline {
        Some(deadline) => {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            tokio::time::timeout(remaining, runtime.execution.clone().acquire_owned()).await
        }
        None => Ok(runtime.execution.clone().acquire_owned().await),
    };
    let wait_ms = started.elapsed().as_secs_f64() * 1_000.0;
    match permit_result {
        Ok(Ok(permit)) => {
            let mut stats = runtime.execution_stats.lock().map_err(|_| {
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
                stats: Arc::clone(&runtime.execution_stats),
            })
        }
        Err(_) => {
            if let Ok(mut stats) = runtime.execution_stats.lock() {
                stats.waiters = stats.waiters.saturating_sub(1);
            }
            Err(WireOperationError::from_stable(
                StableError::deadline_exceeded(),
                "deadline exceeded waiting for inline execution permit",
            ))
        }
        Ok(Err(_)) => {
            if let Ok(mut stats) = runtime.execution_stats.lock() {
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

async fn execute_embedding_with_catalog_guard(
    runtime: &RuntimeState,
    model: &EmbeddingModel,
    batch: TokenBatch,
    deadline: Option<tokio::time::Instant>,
    job_id: Option<&str>,
    held_guard: Option<Arc<tokio::sync::OwnedMutexGuard<()>>>,
) -> Result<Vectors, WireOperationError> {
    let profile = embedding_profile_enabled();
    let tokens = batch
        .items
        .iter()
        .map(|item| item.len().max(1) as u64)
        .sum::<u64>();
    let _activity = runtime.activity_telemetry.begin(&model.model_id);
    let catalog_lane = resolved_catalog_lane(runtime, &model.model_id).is_some();
    let catalog_guard = if held_guard.is_some() {
        held_guard
    } else if catalog_lane {
        Some(Arc::new(
            catalog_lane_lock(runtime, &model.model_id)
                .lock_owned()
                .await,
        ))
    } else {
        None
    };
    let fault_lane = catalog_lane.then(|| model.model_id.clone());
    // Take the lane lock before an execution permit, in every execute_* path.
    // A profile self-check holds the lane lock while it executes; if callers
    // took a permit first, requests queued on that lane could hold every
    // permit while waiting for the lock, and the self-check could never run.
    let permit = acquire_execution_permit(runtime, deadline).await?;
    let result = match &model.backend {
        #[cfg(feature = "test-support")]
        EmbedBackend::TestDeterministic(engine) => {
            let engine = Arc::clone(engine);
            let loaded = model.loaded_model.clone();
            tokio::task::spawn_blocking(move || {
                let _permit = permit;
                let _catalog_guard = catalog_guard;
                engine.embed_batch(&loaded, batch)
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
            let values = engine
                .infer_guarded(
                    batch.items,
                    false,
                    deadline,
                    (permit, catalog_guard, _activity),
                )
                .await
                .map_err(ane_residency_error_to_wire)?;
            Ok(values)
        }
        EmbedBackend::OwnedDecode => Err(WireOperationError::from_stable(
            StableError::artifact_invalid(),
            format!("model '{}' does not support embedding", model.model_id),
        )),
        EmbedBackend::Worker(engine) => {
            let engine = Arc::clone(engine);
            let loaded_model = model.loaded_model.clone();
            let submitted_at = Instant::now();
            let job_id = job_id.map(str::to_owned);
            tokio::task::spawn_blocking(move || {
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
                if let Some(id) = fault_lane.as_deref() { catalog_engine_fault(id,"serve")?; }
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
                let inference_started = Instant::now();
                let result =
                    engine.embed_batch_with_job(&loaded_model, batch, job_id.as_deref());
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
    };
    if result.is_ok() {
        runtime.activity_telemetry.record_completed_tokens(tokens);
    }
    result
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
    held_guard: Option<Arc<tokio::sync::OwnedMutexGuard<()>>>,
) -> Result<synapse_core::RerankScores, WireOperationError> {
    let tokens = request
        .candidates
        .iter()
        .map(|candidate| request.query.len().saturating_add(candidate.len()) as u64)
        .sum();
    let _activity = runtime.activity_telemetry.begin(&model.model_id);
    let catalog_lane = resolved_catalog_lane(runtime, &model.model_id).is_some();
    let catalog_guard = if held_guard.is_some() {
        held_guard
    } else if catalog_lane {
        Some(Arc::new(
            catalog_lane_lock(runtime, &model.model_id)
                .lock_owned()
                .await,
        ))
    } else {
        None
    };
    let fault_lane = catalog_lane.then(|| model.model_id.clone());
    let permit = acquire_execution_permit(runtime, deadline).await?;
    let result = match &model.backend {
        #[cfg(feature = "test-support")]
        EmbedBackend::TestDeterministic(engine) => {
            let _permit = permit;
            let _catalog_guard = catalog_guard;
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
            let values = engine
                .infer_guarded(pairs, true, deadline, (permit, catalog_guard, _activity))
                .await
                .map_err(ane_residency_error_to_wire)?;
            Ok(synapse_core::RerankScores {
                scores: values.into_iter().map(|v| v[0]).collect(),
            })
        }
        EmbedBackend::OwnedDecode => Err(WireOperationError::from_stable(
            StableError::artifact_invalid(),
            format!("model '{}' does not support rerank.score", model.model_id),
        )),
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
            tokio::task::spawn_blocking(move || {
                let _permit = permit;
                let _catalog_guard = catalog_guard;
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
                engine.rerank_with_job(&loaded_model, request, job_id.as_deref())
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
    };
    if result.is_ok() {
        runtime.activity_telemetry.record_completed_tokens(tokens);
    }
    result
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

fn compose_catalog_embed(
    model: &EmbeddingModel,
    tokenized: &mut TokenizedBatch,
) -> Result<(), WireOperationError> {
    let Some(id) = model.engine_identity.build_flags.get("profile") else {
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
    let catalog = model
        .engine_identity
        .build_flags
        .get("profile")
        .map(|id| CatalogProfile::load(id))
        .transpose()
        .map_err(|error| artifact_invalid_error(error.to_string()))?;
    if !matches!(&model.backend, EmbedBackend::Owned(_)) && catalog.is_none() {
        return Ok(None);
    }
    if let Some(catalog) = &catalog {
        if catalog.model()["grammar"]["kind"] == "template" {
            let template = &catalog.model()["grammar"]["template"];
            let encode = |text: &str| {
                model
                    .tokenizer
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
            catalog_sequence_ceiling(&pairs)?;
            return Ok(Some(pairs));
        }
    }
    let inputs = candidates
        .iter()
        .map(|candidate| (query, candidate.as_str()))
        .collect::<Vec<_>>();
    let encodings = model
        .tokenizer
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
    if catalog.is_some() {
        catalog_sequence_ceiling(&pairs)?;
    }
    Ok(Some(pairs))
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
    // Lane lock before permit, the same order as execute_embedding.
    let catalog_guard = if catalog_lane {
        Some(
            catalog_lane_lock(runtime, &model.model_id)
                .lock_owned()
                .await,
        )
    } else {
        None
    };
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
        EmbedBackend::Owned(_) | EmbedBackend::OwnedDecode => Err(WireOperationError::from_stable(
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
            tokio::task::spawn_blocking(move || {
                let _permit = permit;
                let _catalog_guard = catalog_guard;
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
                engine.generate_with_job(&loaded_model, request, job_id.as_deref())
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
    };
    if let Ok(output) = &result {
        runtime
            .activity_telemetry
            .record_completed_tokens((output.n_prompt as u64).saturating_add(output.n_gen as u64));
    }
    result
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
            let equivalent = !model.engine_identity.build_flags.contains_key("profile")
                && allow_equivalent
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

// Legacy probes retain their original ordering. A profile's numerical check is
// deferred until after composed-token validation so an oversized input cannot
// trigger self-check inference before its sequence_too_long refusal.
fn ensure_pre_tokenization_certified(
    state: &ModuleState,
    model: &EmbeddingModel,
    class: CertificationClass,
    accept_declared: bool,
) -> Result<(), WireOperationError> {
    if model.engine_identity.build_flags.contains_key("profile") {
        Ok(())
    } else {
        ensure_model_certified(state, model, class, accept_declared)
    }
}

async fn ensure_profile_request_certified(
    state: Arc<ModuleState>,
    model: &EmbeddingModel,
) -> Result<(), WireOperationError> {
    if model.engine_identity.build_flags.contains_key("profile") {
        ensure_profile_preload_ready(state.clone(), &model.model_id, None).await?;
        ensure_model_certified(
            &state,
            model,
            if model.task == ModelTask::Embed {
                CertificationClass::Embedding
            } else {
                CertificationClass::Rerank
            },
            false,
        )?;
    }
    Ok(())
}

fn ensure_model_certified(
    state: &ModuleState,
    model: &EmbeddingModel,
    certification_class: CertificationClass,
    accept_declared: bool,
) -> Result<(), WireOperationError> {
    if resolved_catalog_lane(&state.runtime, &model.model_id).is_some() {
        return Ok(());
    }
    if let Some(profile) = model.engine_identity.build_flags.get("profile") {
        let (id, _) = profile_preload_check_key(state, model, profile)?;
        return match profile_preload_check_status(state, &id)?.as_str() {
            "passed" => Ok(()),
            status => Err(profile_preload_check_error(profile, status)),
        };
    }
    // Owned-CUDA has no declared or inherited certification path. A measured
    // row must match this exact machine-profile hash before serving.
    let result = if model.engine_identity.engine == CUDA_WORKER_ENGINE {
        match state.store.get_cert_row(
            certification_class,
            &state.machine_profile_hash,
            &model.certification_fingerprint,
        ) {
            Ok(Some(_)) => Ok(()),
            Ok(None) => Err(WireOperationError::from_stable(
                StableError::not_certified(),
                format!(
                    "owned-cuda fingerprint {} is not certified on machine profile {}",
                    model.certification_fingerprint.0, state.machine_profile_hash
                ),
            )),
            Err(error) => Err(WireOperationError::from_stable(
                StableError::engine_crashed(Some(100)),
                format!("read owned-cuda certification row: {error}"),
            )),
        }
    } else {
        ensure_fingerprint_certified(
            &state.store,
            certification_class,
            &state.machine_profile_hash,
            &model.certification_fingerprint,
            &model.model_id,
            accept_declared,
        )
    };
    if let Err(error) = &result {
        record_admission_refusal(&state.runtime, &model.model_id, None, &error.code);
        if error.code == "not_certified"
            && state
                .store
                .has_stale_cert_row(
                    certification_class,
                    &state.machine_profile_hash,
                    &model.certification_fingerprint,
                )
                .unwrap_or(false)
        {
            let _ = state
                .store
                .observe_certification_stale(&state.revisioned_machine_profile_hash, now_ms());
        }
    }
    result
}

fn ensure_fingerprint_certified(
    store: &SynapseStore,
    certification_class: CertificationClass,
    machine_profile_hash: &str,
    fingerprint: &Fingerprint,
    model_id: &str,
    accept_declared: bool,
) -> Result<(), WireOperationError> {
    match store.get_cert_row(certification_class, machine_profile_hash, fingerprint) {
        Ok(Some(_)) => return Ok(()),
        Ok(None) => {}
        Err(error) => {
            return Err(WireOperationError::from_stable(
                StableError::engine_crashed(Some(100)),
                format!("read measured certification rows: {error}"),
            ))
        }
    }

    match declared_certification_for_request(store, fingerprint, model_id, accept_declared) {
        Ok(Some(_)) => Ok(()),
        Ok(None) => {
            let failed_probe = store
                .get_probe_row(certification_class, machine_profile_hash, fingerprint)
                .map_err(|error| {
                    WireOperationError::from_stable(
                        StableError::engine_crashed(Some(100)),
                        format!("read probe outcome row: {error}"),
                    )
                })?
                .filter(|row| row.status == CertificationStatus::Uncertified);
            if let Some(row) = failed_probe {
                let reason = row
                    .evidence
                    .get("blocking_reason")
                    .and_then(Value::as_str)
                    .unwrap_or("probe_failed");
                return Err(WireOperationError::from_stable(
                    StableError::not_certified(),
                    format!(
                        "fingerprint {} is uncertified on machine profile {}: {}",
                        fingerprint.0, machine_profile_hash, reason
                    ),
                ));
            }
            let stale = store
                .has_stale_cert_row(certification_class, machine_profile_hash, fingerprint)
                .unwrap_or(false);
            let message = if stale {
                format!(
                    "fingerprint {} has only stale certification rows for a different machine profile",
                    fingerprint.0
                )
            } else {
                format!(
                    "fingerprint {} is not certified on machine profile {}",
                    fingerprint.0, machine_profile_hash
                )
            };
            Err(WireOperationError::from_stable(
                StableError::not_certified(),
                message,
            ))
        }
        Err(error) => Err(error),
    }
}

fn declared_certification_for_request(
    store: &SynapseStore,
    fingerprint: &Fingerprint,
    model_id: &str,
    accept_declared: bool,
) -> Result<Option<CertificationRow>, WireOperationError> {
    match store.declared_cert_row_for_fingerprint(fingerprint) {
        Ok(Some(row)) if accept_declared => Ok(Some(row)),
        Ok(Some(_)) => Err(WireOperationError::from_stable(
            StableError::declared_identity_not_accepted(),
            format!(
                "model '{model_id}' has declared identity assurance; set accept_declared=true to opt in"
            ),
        )),
        Ok(None) => Ok(None),
        Err(error) => Err(WireOperationError::from_stable(
            StableError::engine_crashed(Some(100)),
            format!("read declared certification rows: {error}"),
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
    if params
        .models
        .iter()
        .any(|id| state.runtime.release_catalog.is_reserved_id(id))
    {
        return result_outcome(error_payload(
            &state,
            catalog_wire_error(
                "invalid_request",
                json!({"models":params.models}),
                "catalog lanes use self-check, not probe.start",
            ),
        ));
    }
    let model_filter = params.models;
    let request_key = params
        .request_key
        .filter(|key| !key.trim().is_empty())
        .unwrap_or_else(|| format!("probe:{}:{now}", state.module_generation));
    let params_json = json!({
        "models": model_filter.clone(),
        "deadline_ms": params.deadline_ms,
    });
    let request_digest =
        compute_request_digest("probe", "management", None, None, &params_json, &[]);
    let admission = match state.store.admit_job(
        &request_key,
        &request_digest,
        "probe",
        state.module_generation,
        None,
        &params_json,
        now,
        state.runtime.jobs.execution_ttl_ms,
        state.runtime.jobs.result_retention_ttl_ms,
    ) {
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
        let supervisor_state = Arc::clone(&state);
        let supervisor_job_id = record.job_id.clone();
        let task_state = Arc::clone(&state);
        let task_job_id = record.job_id.clone();
        let deadline_ms = params.deadline_ms;
        tokio::spawn(async move {
            let task = tokio::spawn(async move {
                execute_probe_job(task_state, task_job_id, model_filter, deadline_ms).await;
            });
            if let Err(error) = task.await {
                fail_job_with_wire_error(
                    &supervisor_state,
                    &supervisor_job_id,
                    true,
                    WireOperationError::from_stable(
                        StableError::engine_crashed(Some(100)),
                        format!("probe execution task failed: {error}"),
                    ),
                );
            }
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

async fn execute_probe_job(
    state: Arc<ModuleState>,
    job_id: String,
    model_filter: Vec<String>,
    deadline_ms: Option<u64>,
) {
    if !matches!(
        state
            .store
            .mark_job_running(&job_id, state.module_generation, now_ms()),
        Ok(true)
    ) {
        return;
    }

    let embed_fixtures = match probe_fixtures() {
        Ok(fixtures) => fixtures,
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
    let generate_fixtures = match generate_probe_fixtures() {
        Ok(fixtures) => fixtures,
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
                .filter(|model_id| !state.runtime.release_catalog.is_reserved_id(model_id))
                .filter(|model_id| {
                    model_filter.is_empty() || model_filter.iter().any(|id| id == *model_id)
                })
                .cloned()
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let mut selected_models = Vec::new();
    let mut lane_results = Vec::new();
    for model_id in selected_model_ids {
        let model = match ensure_model_loaded_for_control(
            Arc::clone(&state),
            &model_id,
            deadline_ms,
        )
        .await
        {
            Ok(model) => model,
            Err(error) => {
                let spec = model_slot_snapshot(&state.runtime, &model_id).map(|slot| slot.spec);
                let blocking_reason = match error.code.as_str() {
                    "owned_cuda_unsupported" => "owned_cuda_unsupported",
                    "artifact_invalid" if error.message.contains("macOS") => "backend_unavailable",
                    "not_certified" => "not_certified",
                    _ => "load_failed",
                };
                lane_results.push(json!({
                        "model_id": model_id,
                        "cell_id": spec.as_ref().map(|spec| format!("{}/{}/{}", spec.engine, spec.owned_family.as_deref().unwrap_or("unknown"), spec.quant)),
                        "engine": spec.as_ref().map(|spec| spec.engine.clone()),
                        "backend": spec.as_ref().and_then(|spec| spec.engine_identity.build_flags.get("backend").cloned()),
                        "fingerprint": spec.as_ref().map(|spec| spec.fingerprint.clone()),
                        "status": if blocking_reason == "owned_cuda_unsupported" || blocking_reason == "backend_unavailable" { "unsupported" } else { "uncertified" },
                        "blocking_reason": blocking_reason,
                        "error": error,
                    }));
                continue;
            }
        };
        selected_models.push(model);
    }

    for profile in state
        .remote_gateway
        .profiles()
        .into_iter()
        .filter(|profile| {
            model_filter.is_empty()
                || model_filter
                    .iter()
                    .any(|model_id| model_id == &profile.synapse_model_id)
        })
    {
        match state
            .remote_gateway
            .calibrate(&profile, state.module_generation, now_ms())
            .await
        {
            Ok(()) => lane_results.push(json!({
                "model_id": profile.synapse_model_id,
                "fingerprint": profile.fingerprint,
                "assurance": "declared",
                "identity_revision": profile.identity_revision,
                "passed": true,
                "status": "certified",
                "probe": "remote_sentinel_calibration",
            })),
            Err(error) => {
                fail_job_with_wire_error(
                    &state,
                    &job_id,
                    error.stable.class == ErrorClass::Transient,
                    WireOperationError::from_stable(error.stable, error.message),
                );
                return;
            }
        }
    }
    let mut certified_vectors = Vec::new();
    for model in selected_models {
        let probe_result = match model.task {
            ModelTask::Embed => {
                execute_embed_probe_for_model(&state, Arc::clone(&model), &embed_fixtures).await
            }
            ModelTask::Rerank => {
                execute_rerank_probe_for_model(&state, Arc::clone(&model), &rerank_fixture).await
            }
            ModelTask::Generate => {
                execute_generate_probe_for_model(&state, Arc::clone(&model), &generate_fixtures)
                    .await
            }
        };
        let probe_result = match probe_result {
            Ok(result) => result,
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
    let mut routable_perf_rows = Vec::with_capacity(perf_rows.len());
    for row in perf_rows {
        let certification_required =
            row.workload != ModelTask::Generate.as_str() || row.engine == "owned-metal";
        let certification_class = match row.workload.as_str() {
            "embed" => Some(CertificationClass::Embedding),
            "rerank" => Some(CertificationClass::Rerank),
            _ => None,
        };
        let certified = if certification_required {
            match certification_class.map(|class| {
                state
                    .store
                    .get_cert_row(class, &state.machine_profile_hash, &row.fingerprint)
            }) {
                Some(Ok(row)) => row.is_some(),
                Some(Err(error)) => {
                    fail_job_with_wire_error(
                        &state,
                        &job_id,
                        true,
                        WireOperationError::from_stable(
                            StableError::engine_crashed(Some(100)),
                            format!("read certification rows for knob mapping: {error}"),
                        ),
                    );
                    return;
                }
                None => false,
            }
        } else {
            true
        };
        if certified {
            routable_perf_rows.push(row);
        }
    }
    let knob_assignments = compute_knob_assignments(&routable_perf_rows);
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
        "machine_profile_hash_revision": MACHINE_PROFILE_HASH_REVISION,
        "machine_profile": state.machine_profile,
        "current_knob": state.runtime.knob,
        "fixture": {
            "items": embed_fixtures.first().map_or(0, |fixture| fixture.items.len()),
            "first_id": embed_fixtures
                .first()
                .and_then(|fixture| fixture.items.first())
                .map(|item| item.id.clone()),
            "generation_command": embed_fixtures
                .first()
                .and_then(|fixture| fixture.generation_command.clone()),
            "sets": embed_fixtures
                .iter()
                .map(probe_fixture_provenance)
                .collect::<Vec<_>>(),
        },
        "fixtures": {
            "embed": {
                "items": embed_fixtures.first().map_or(0, |fixture| fixture.items.len()),
                "first_id": embed_fixtures
                    .first()
                    .and_then(|fixture| fixture.items.first())
                    .map(|item| item.id.clone()),
                "generation_command": embed_fixtures
                    .first()
                    .and_then(|fixture| fixture.generation_command.clone()),
                "sets": embed_fixtures
                    .iter()
                    .map(probe_fixture_provenance)
                    .collect::<Vec<_>>(),
            },
            "rerank": {
                "items": rerank_fixture.items.len(),
                "first_id": rerank_fixture.items.first().map(|item| item.id.clone()),
                "generation_command": rerank_fixture.generation_command,
            },
            "generate": generate_fixtures
                .iter()
                .map(generate_fixture_provenance)
                .collect::<Vec<_>>()
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
    } else {
        state.runtime.admission_telemetry.record_job_completed();
    }
}

async fn execute_embed_probe_for_model(
    state: &ModuleState,
    model: Arc<EmbeddingModel>,
    fixtures: &[ProbeFixture],
) -> Result<ProbeModelResult, WireOperationError> {
    let reference_key = probe_reference_key(&model);
    #[cfg(feature = "test-support")]
    let test_fixtures = test_deterministic_probe_fixtures();
    #[cfg(feature = "test-support")]
    let fixtures = if model.engine_identity.engine == test_deterministic::NAME {
        test_fixtures.as_slice()
    } else {
        fixtures
    };
    let reference_candidates = fixtures
        .iter()
        .filter(|fixture| probe_fixture_matches_key(fixture, &reference_key))
        .collect::<Vec<_>>();
    let Some(text_fixture) = reference_candidates.first().copied() else {
        let evidence = json!({
            "task": "embed",
            "gate": "reference_fixture",
            "blocking_reason": "reference_fixture_missing",
            "reference": {
                "family": reference_key.family.clone(),
                "model": reference_key.model.clone(),
            },
        });
        store_probe_outcome_row(
            state,
            &model,
            CertificationStatus::Uncertified,
            evidence.clone(),
        )?;
        return Ok(ProbeModelResult {
            lane_result: json!({
                "model_id": model.model_id,
                "task": "embed",
                "fingerprint": model.fingerprint,
                "numeric_profile_id": model.numeric_profile_id,
                "status": "uncertified",
                "blocking_reason": "reference_fixture_missing",
                "evidence": evidence,
                "performance": Value::Null,
            }),
            certified_vectors: None,
        });
    };
    let texts = text_fixture
        .items
        .iter()
        .map(|item| item.text.as_str())
        .collect::<Vec<_>>();
    let mut tokenized = match model.tokenizer.tokenize_batch(texts) {
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
    compose_catalog_embed(&model, &mut tokenized)?;
    apply_owned_tokenizer_policy(&model, &mut tokenized);
    let vectors = match execute_embedding(
        &state.runtime,
        &model,
        tokenized.batch.clone(),
        None,
        None,
    )
    .await
    {
        Ok(vectors) => vectors,
        Err(error) => return Err(error),
    };
    let actual_dims = vectors.first().map(Vec::len);
    let actual_item_count = vectors.len();
    let vectors_have_one_dimension = vectors
        .iter()
        .all(|vector| Some(vector.len()) == actual_dims);
    let fixture = reference_candidates.into_iter().find(|fixture| {
        vectors_have_one_dimension
            && actual_item_count == fixture.items.len()
            && fixture_reference_dims(fixture) == actual_dims
    });
    let Some(fixture) = fixture else {
        let evidence = json!({
            "task": "embed",
            "gate": "reference_fixture",
            "blocking_reason": "reference_fixture_missing",
            "reference": {
                "family": reference_key.family.clone(),
                "model": reference_key.model.clone(),
                "available_dims": fixtures
                    .iter()
                    .filter(|fixture| probe_fixture_matches_key(fixture, &reference_key))
                    .filter_map(fixture_reference_dims)
                    .collect::<Vec<_>>(),
            },
            "actual_dims": actual_dims,
            "actual_items": actual_item_count,
        });
        store_probe_outcome_row(
            state,
            &model,
            CertificationStatus::Uncertified,
            evidence.clone(),
        )?;
        return Ok(ProbeModelResult {
            lane_result: json!({
                "model_id": model.model_id,
                "task": "embed",
                "fingerprint": model.fingerprint,
                "numeric_profile_id": model.numeric_profile_id,
                "status": "uncertified",
                "blocking_reason": "reference_fixture_missing",
                "evidence": evidence,
                "performance": Value::Null,
            }),
            certified_vectors: None,
        });
    };
    let evidence = probe_evidence(&vectors, &fixture.items);
    let placement_share = ane_placement_share_for_model(&model).await?;
    let quality_passed = evidence.mean_cosine >= state.runtime.probe.mean_cosine_threshold
        && evidence.worst_decile >= state.runtime.probe.worst_decile_rank_overlap_threshold;
    let placement_passed = if model.engine_identity.engine == "ane-coreml-worker" {
        placement_share.is_some_and(|share| share >= state.runtime.probe.ane_placement_threshold)
    } else {
        true
    };
    let passed = quality_passed && placement_passed;
    let cuda_evidence = owned_cuda_evidence(state, &model).await?;
    let certification_evidence = json!({
        "task": "embed",
        "metrics": evidence,
        "ane_placement_share": placement_share,
        "cuda": cuda_evidence,
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
            "cuda": cuda_evidence,
            "performance": performance,
        }),
        certified_vectors: passed.then_some(vectors),
    })
}

async fn ane_placement_share_for_model(
    model: &EmbeddingModel,
) -> Result<Option<f64>, WireOperationError> {
    if model.engine_identity.engine != "ane-coreml-worker" {
        return Ok(None);
    }
    match &model.backend {
        EmbedBackend::Worker(engine) => {
            // WorkerEngine::ping bridges to its private runtime with block_on;
            // keep that synchronous bridge off the module's async runtime.
            let engine = Arc::clone(engine);
            let ping = tokio::task::spawn_blocking(move || {
                let engine = engine.lock().map_err(|_| {
                    worker_host::WorkerHostError::Protocol(
                        "worker engine mutex was poisoned during ANE placement ping".to_string(),
                    )
                })?;
                engine.ping()
            })
            .await
            .map_err(|error| {
                WireOperationError::from_stable(
                    StableError::engine_crashed(Some(100)),
                    format!("ANE placement ping join failed: {error}"),
                )
            })?
            .map_err(|error| {
                engine_error_to_wire(error.to_engine_error(EngineErrorStage::Inference))
            })?;
            Ok(ping.placement_share)
        }
        #[cfg(unix)]
        EmbedBackend::DirectAne(_) => Ok(None),
        #[cfg(feature = "test-support")]
        EmbedBackend::TestDeterministic(_) => Ok(None),
        EmbedBackend::Owned(_) | EmbedBackend::OwnedDecode => Ok(None),
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
        let owned_pairs = owned_rerank_pairs(&model, item.query.as_str(), &item.candidates)?;
        let scores = match execute_rerank(
            &state.runtime,
            &model,
            RerankRequest {
                query,
                candidates: token_items,
            },
            owned_pairs,
            None,
            None,
        )
        .await
        {
            Ok(scores) => scores,
            Err(error) => return Err(error),
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
    fixtures: &[GenerateProbeFixture],
) -> Result<ProbeModelResult, WireOperationError> {
    use owned_decode_routing::lane::LaneKind;

    if !microllm_certification_required(&model) {
        return Ok(ProbeModelResult {
            lane_result: json!({
                "model_id": model.model_id,
                "task": "generate",
                "fingerprint": model.fingerprint,
                "numeric_profile_id": model.numeric_profile_id,
                "status": "not_required",
                "certification_required": false,
                "reason": "worker_lane_uses_existing_dispatch_path",
            }),
            certified_vectors: None,
        });
    }

    let spec = state
        .runtime
        .catalog
        .lock()
        .ok()
        .and_then(|catalog| catalog.get(&model.model_id).map(|slot| slot.spec.clone()))
        .ok_or_else(|| {
            WireOperationError::from_stable(
                StableError::artifact_invalid(),
                format!("missing catalog entry for '{}'", model.model_id),
            )
        })?;
    let entry = owned_decode_catalog_entry(&spec)
        .map_err(|error| artifact_invalid_error(error.as_str()))?;
    let decode_fingerprint = entry
        .decode_identity_inputs()
        .decode_fingerprint()
        .map_err(|error| artifact_invalid_error(error.as_str()))?;
    let processing_fingerprint = owned_decode_processing_fingerprint(&entry)
        .map_err(|error| artifact_invalid_error(error.as_str()))?;
    let (runtime_config_digest, _) =
        owned_decode_runtime_identity(&spec, &entry, state.runtime.decode_chain_k);
    let constrained_runtime_identity =
        owned_decode_probe_constraint_identity(&model, &decode_fingerprint)?;
    let probe_snapshot_inputs = OwnedDecodeMatchInputs {
        revisioned_machine_profile_hash: state.revisioned_machine_profile_hash.clone(),
        profile_activation_epoch: state.profile_activation_epoch,
        model_id: entry.entry_id.clone(),
        decode_fingerprint: decode_fingerprint.0.clone(),
        processing_fingerprint: processing_fingerprint.0.clone(),
        runtime_config_digest,
        constraint_runtime_identities: vec![constrained_runtime_identity],
        worker_path_evidence: owned_decode_worker_path_identity(),
        evidence_schema_revision: store::CERT_EVIDENCE_SCHEMA_REVISION.to_string(),
        g_dec_manifest_revision: store::G_DEC_MANIFEST_REVISION.to_string(),
    };
    let probe_snapshot = ProbeSnapshot::capture(&probe_snapshot_inputs)
        .map_err(|error| artifact_invalid_error(error.as_str()))?;
    let Some(fixture) = fixtures.iter().find(|fixture| {
        spec.owned_family.as_deref() == Some(fixture.family.as_str())
            && spec.owned_dtype.as_deref() == Some(fixture.dtype.as_str())
            && spec.quant == fixture.quant
    }) else {
        let evidence = json!({
            "task": "generate",
            "gate": "structural_band",
            "blocking_reason": "fixture_unavailable",
            "available_fixtures": fixtures
                .iter()
                .map(generate_fixture_provenance)
                .collect::<Vec<_>>(),
            "model_family": spec.owned_family,
            "model_dtype": spec.owned_dtype,
            "model_quant": spec.quant,
        });
        store_owned_probe_outcome(
            state,
            &model,
            &probe_snapshot,
            CertificationStatus::Uncertified,
            evidence.clone(),
        )?;
        return Ok(ProbeModelResult {
            lane_result: json!({
                "model_id": model.model_id,
                "task": "generate",
                "fingerprint": decode_fingerprint,
                "numeric_profile_id": model.numeric_profile_id,
                "status": "uncertified",
                "certification_required": true,
                "blocking_reason": "fixture_unavailable",
                "evidence": evidence,
                "performance": Value::Null,
            }),
            certified_vectors: None,
        });
    };

    if let Some(q8_identity) = entry.q8.as_ref() {
        let trust_state = state
            .runtime
            .owned_decode_q8
            .lock()
            .ok()
            .and_then(|registry| {
                registry
                    .entry(
                        &entry.artifact_source_digest,
                        &q8_identity.quantizer_revision,
                    )
                    .map(|artifact| artifact.trust_state)
            });
        if trust_state != Some(owned_decode_routing::q8ingest::TrustState::Trusted) {
            let blocking_reason =
                if trust_state == Some(owned_decode_routing::q8ingest::TrustState::Poisoned) {
                    "artifact_poisoned"
                } else {
                    "owned_decode_not_certified"
                };
            let evidence = json!({
                "task": "generate",
                "gate": "q8_artifact_trust",
                "blocking_reason": blocking_reason,
                "fixture": generate_fixture_provenance(fixture),
            });
            store_owned_probe_outcome(
                state,
                &model,
                &probe_snapshot,
                CertificationStatus::Uncertified,
                evidence.clone(),
            )?;
            return Ok(ProbeModelResult {
                lane_result: json!({
                    "model_id": model.model_id,
                    "task": "generate",
                    "fingerprint": decode_fingerprint,
                    "numeric_profile_id": model.numeric_profile_id,
                    "status": "uncertified",
                    "certification_required": true,
                    "blocking_reason": blocking_reason,
                    "evidence": evidence,
                    "performance": Value::Null,
                }),
                certified_vectors: None,
            });
        }
    }

    let mut exact_matches = 0_usize;
    let mut accepted_structural_forks = Vec::new();
    let mut tokens_compared = 0_usize;
    let mut mismatches = Vec::new();
    let mut chain_shape_mismatches = Vec::new();
    let mut throughput_samples = Vec::with_capacity(fixture.items.len());
    let mut latency_samples = Vec::with_capacity(fixture.items.len());
    let chain_shapes = if state.runtime.decode_chain_k > 1 {
        vec![1, state.runtime.decode_chain_k]
    } else {
        vec![1]
    };
    let mut baseline_outputs = Vec::with_capacity(fixture.items.len());
    let mut last_worker_dispatch: Option<Arc<Mutex<worker_host::SupervisedDecodeDispatch>>> = None;

    for (shape_index, chain_k) in chain_shapes.into_iter().enumerate() {
        let mut worker_dispatch: Option<Arc<Mutex<worker_host::SupervisedDecodeDispatch>>> = None;
        for (index, item) in fixture.items.iter().enumerate() {
            let tokenized = model
                .tokenizer
                .tokenize_batch([item.prompt.as_str()])
                .map_err(|error| {
                    WireOperationError::from_stable(
                        StableError::artifact_invalid(),
                        format!("owned-decode probe tokenization failed: {error}"),
                    )
                })?;
            let prompt = tokenized.batch.items.into_iter().next().unwrap_or_default();
            let prompt_token_count = prompt.len().min(u32::MAX as usize) as u32;
            if worker_dispatch.is_none() {
                worker_dispatch = Some(
                    cached_supervised_decode_dispatch_for_chain_k(
                        state,
                        &spec,
                        &entry,
                        prompt.clone(),
                        None,
                        OWNED_DECODE_PROBE_TIMEOUT_MS,
                        chain_k,
                    )
                    .map_err(|error| match error {
                        OwnedDecodeDispatchPreparationError::Refused(refusal) => {
                            artifact_invalid_error(format!(
                                "owned-decode certification cannot prepare worker: {}",
                                refusal.as_str()
                            ))
                        }
                        OwnedDecodeDispatchPreparationError::Wire(error) => error,
                    })?,
                );
            }
            let dispatch = worker_dispatch
                .as_ref()
                .ok_or_else(|| {
                    WireOperationError::from_stable(
                        StableError::artifact_invalid(),
                        "owned-decode certification requires a supervised worker binary",
                    )
                })?
                .clone();
            let started = std::time::Instant::now();
            let output = dispatch_supervised_decode(
                dispatch,
                prompt,
                None,
                OWNED_DECODE_PROBE_TIMEOUT_MS,
                owned_decode_routing::DispatchedCommand {
                    lane: LaneKind::OwnedDecode,
                    decode_fingerprint: decode_fingerprint.clone(),
                    processing_fingerprint: processing_fingerprint.clone(),
                    prompt_token_count,
                    max_tokens: item.max_new_tokens,
                    generation_id: format!(
                        "probe-{}-{shape_index}-{index}",
                        state.module_generation
                    ),
                    constrained: false,
                    chain_k,
                },
            )
            .await?;

            if shape_index == 0 {
                let elapsed_secs = started.elapsed().as_secs_f64().max(f64::EPSILON);
                latency_samples.push(elapsed_secs * 1_000.0);
                throughput_samples.push(output.generated_token_ids.len() as f64 / elapsed_secs);
                tokens_compared = tokens_compared.saturating_add(item.expected_token_ids.len());
                baseline_outputs.push(output.generated_token_ids.clone());
                if output.generated_token_ids == item.expected_token_ids {
                    exact_matches += 1;
                } else if accepted_structural_forks.len() < fixture.structural_band.max_forks {
                    if let Some(fork) = certified_generate_fork(
                        item,
                        &output.generated_token_ids,
                        &fixture.structural_band,
                    ) {
                        accepted_structural_forks.push(fork);
                    } else {
                        mismatches.push(decode_token_mismatch(item, &output.generated_token_ids));
                    }
                } else {
                    mismatches.push(decode_token_mismatch(item, &output.generated_token_ids));
                }
            } else if baseline_outputs.get(index) != Some(&output.generated_token_ids) {
                chain_shape_mismatches.push(json!({
                    "prompt_index": index,
                    "configured_chain_k": chain_k,
                    "baseline_k": 1,
                    "expected_token_ids": baseline_outputs.get(index),
                    "actual_token_ids": output.generated_token_ids,
                }));
            }
        }
        last_worker_dispatch = worker_dispatch;
    }

    let worker_dispatch = last_worker_dispatch.ok_or_else(|| {
        WireOperationError::from_stable(
            StableError::artifact_invalid(),
            "owned-decode certification requires a supervised worker binary",
        )
    })?;

    let vocabulary_digest = owned_decode_vocabulary_digest(&model.tokenizer)?;
    let constrained_schema = r#"{"type":"null"}"#;
    let compiled = owned_decode_grammar_scheduler::compile_grammar(
        constrained_schema,
        &owned_decode_grammar_scheduler::CompileContext {
            base_decode_fingerprint: decode_fingerprint.clone(),
            tokenizer_vocabulary_digest: vocabulary_digest,
        },
        &owned_decode_grammar_scheduler::GrammarSubsetManifest::default(),
    )
    .map_err(|error| {
        WireOperationError::from_stable(
            StableError::artifact_invalid(),
            format!(
                "compile owned-decode certification constraint: {}",
                error.message
            ),
        )
    })?;
    let constrained_runtime_identity = compiled.constraint.constraint_runtime_identity.digest();
    let constrained_prompt = model
        .tokenizer
        .tokenize("Respond with exactly the JSON literal null and nothing else:\n")
        .map_err(|error| artifact_invalid_error(error.to_string()))?
        .ids;
    let constrained_dispatch = worker_dispatch;
    let constrained = dispatch_supervised_decode(
        constrained_dispatch,
        constrained_prompt.clone(),
        Some(worker_constraint(&compiled.constraint)),
        OWNED_DECODE_PROBE_TIMEOUT_MS,
        owned_decode_routing::DispatchedCommand {
            lane: LaneKind::OwnedDecode,
            decode_fingerprint: decode_fingerprint.clone(),
            processing_fingerprint,
            prompt_token_count: constrained_prompt.len().min(u32::MAX as usize) as u32,
            max_tokens: 64,
            generation_id: format!("probe-{}-constrained", state.module_generation),
            constrained: true,
            chain_k: 1,
        },
    )
    .await;
    let constrained_schema_valid = constrained
        .as_ref()
        .ok()
        .and_then(|output| model.tokenizer.decode(&output.generated_token_ids).ok())
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .is_some_and(|value| value.is_null());

    let evidence = GenerateProbeEvidence {
        token_exact_matches: exact_matches,
        accepted_structural_forks: accepted_structural_forks.len(),
        max_certified_forks: fixture.structural_band.max_forks,
        items: fixture.items.len(),
        tokens_compared,
    };
    let fixture_passed = exact_matches + accepted_structural_forks.len() == fixture.items.len()
        && mismatches.is_empty()
        && chain_shape_mismatches.is_empty();
    let passed = fixture_passed && constrained_schema_valid;
    let certification_evidence = json!({
        "task": "generate",
        "gate": "structural_band",
        "blocking_reason": if passed {
            Value::Null
        } else if !chain_shape_mismatches.is_empty() {
            json!("configured_chain_shape_diverged_from_k1")
        } else if !fixture_passed {
            json!("token_mismatch_outside_structural_band")
        } else {
            json!("constrained_worker_path_failed")
        },
        "metrics": evidence,
         "accepted_forks": accepted_structural_forks,
         "mismatches": mismatches,
         "chain_shape_mismatches": chain_shape_mismatches,
         "chain_shapes": if state.runtime.decode_chain_k > 1 {
             vec![1, state.runtime.decode_chain_k]
         } else {
             vec![1]
         },
         "fixture": generate_fixture_provenance(fixture),
        "worker_path": {
            "transport": worker_catalog_transport(),
            "protocol": owned_decode_worker::identity::WORKER_PROTOCOL_ID,
            "fixture_battery": "20x64-structural-band",
            "prompt_count": fixture.items.len(),
            "constrained_schema_valid": constrained_schema_valid,
            "constrained_runtime_identities": if constrained_schema_valid {
                vec![constrained_runtime_identity]
            } else {
                Vec::<String>::new()
            },
        },
    });
    let certification_evidence = if passed {
        let mut evidence = certification_evidence;
        evidence["g_dec"] = owned_decode_gate_evidence();
        evidence
    } else {
        certification_evidence
    };
    store_owned_probe_outcome(
        state,
        &model,
        &probe_snapshot,
        if passed {
            CertificationStatus::Certified
        } else {
            CertificationStatus::Uncertified
        },
        certification_evidence.clone(),
    )?;

    let performance = if passed {
        let cold_load_ms =
            model_cold_load_ms(&state.runtime, &model.model_id).ok_or_else(|| {
                WireOperationError::from_stable(
                    StableError::engine_crashed(Some(100)),
                    format!("missing cold-load measurement for '{}'", model.model_id),
                )
            })?;
        let perf = PerfBenchResult {
            throughput_tok_s: median_value(&mut throughput_samples),
            cold_load_ms,
            single_item_latency_p50_ms: median_value(&mut latency_samples),
            details: json!({
                "mode": "supervised_worker_socket_single_stream",
                "statistic": "median_over_fixtures",
                "fixture_samples": fixture.items.len(),
                "generated_tokens_per_fixture": fixture.items.first().map(|item| item.expected_token_ids.len()),
            }),
        };
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
            "fingerprint": decode_fingerprint,
            "numeric_profile_id": model.numeric_profile_id,
            "status": if passed { "certified" } else { "uncertified" },
            "certification_required": true,
            "blocking_reason": certification_evidence["blocking_reason"],
            "evidence": certification_evidence,
            "performance": performance,
        }),
        certified_vectors: None,
    })
}

fn microllm_certification_required(model: &EmbeddingModel) -> bool {
    engine_requires_microllm_certification(&model.engine_identity.engine)
}

fn engine_requires_microllm_certification(engine: &str) -> bool {
    engine == "owned-metal-decode"
}

fn generate_fixture_provenance(fixture: &GenerateProbeFixture) -> Value {
    json!({
        "family": fixture.family,
        "dtype": fixture.dtype,
        "quant": fixture.quant,
        "model": fixture.model,
        "model_revision": fixture.model_revision,
        "generation_command": fixture.generation_command,
        "generation_command_sha256": fixture.generation_command_sha256,
        "provenance": fixture.provenance,
        "structural_band": {
            "max_forks": fixture.structural_band.max_forks,
            "top2_gap_ceiling": fixture.structural_band.top2_gap_ceiling,
            "allowed_forks": fixture.structural_band.allowed_forks,
        },
        "items": fixture.items.len(),
    })
}

fn certified_generate_fork<'a>(
    item: &GenerateProbeItem,
    actual: &[u32],
    structural_band: &'a GenerateStructuralBand,
) -> Option<&'a GenerateAllowedFork> {
    let token_index = item
        .expected_token_ids
        .iter()
        .zip(actual)
        .position(|(expected, actual)| expected != actual)
        .or_else(|| {
            (item.expected_token_ids.len() != actual.len())
                .then(|| actual.len().min(item.expected_token_ids.len()))
        })?;
    let oracle_token = *item.expected_token_ids.get(token_index)?;
    let alternate_token = *actual.get(token_index)?;
    structural_band.allowed_forks.iter().find(|fork| {
        fork.id == item.id
            && fork.token_index == token_index
            && fork.oracle_token == oracle_token
            && fork.alternate_token == alternate_token
            && fork.oracle_top2.contains(&oracle_token)
            && fork.oracle_top2.contains(&alternate_token)
            && fork.top2_gap <= structural_band.top2_gap_ceiling
    })
}

fn decode_token_mismatch(item: &GenerateProbeItem, actual: &[u32]) -> Value {
    let divergence_index = item
        .expected_token_ids
        .iter()
        .zip(actual)
        .position(|(expected, actual)| expected != actual)
        .unwrap_or_else(|| item.expected_token_ids.len().min(actual.len()));
    json!({
        "id": item.id,
        "prompt": item.prompt,
        "divergence_token_index": divergence_index,
        "expected_token_id": item.expected_token_ids.get(divergence_index),
        "actual_token_id": actual.get(divergence_index),
        "expected_token_ids": item.expected_token_ids,
        "actual_token_ids": actual,
    })
}

fn store_owned_probe_outcome(
    state: &ModuleState,
    model: &EmbeddingModel,
    snapshot: &ProbeSnapshot,
    status: CertificationStatus,
    evidence: Value,
) -> Result<ProbeWriteOutcome, WireOperationError> {
    let snapshot_inputs = snapshot.to_match_inputs();
    let profile_state = state.store.profile_state().map_err(|error| {
        WireOperationError::from_stable(
            StableError::engine_crashed(Some(100)),
            format!("read terminal probe profile state: {error}"),
        )
    })?;
    let terminal = owned_decode_probe_match_inputs(
        state,
        model,
        profile_state
            .revisioned_machine_profile_hash
            .ok_or_else(|| artifact_invalid_error("terminal probe profile hash is missing"))?,
        profile_state
            .profile_activation_epoch
            .ok_or_else(|| artifact_invalid_error("terminal probe profile epoch is missing"))?,
        snapshot_inputs.constraint_runtime_identities.clone(),
    )?;
    let row = OwnedDecodeCertificationRow {
        status,
        revisioned_machine_profile_hash: terminal.revisioned_machine_profile_hash.clone(),
        profile_activation_epoch: terminal.profile_activation_epoch,
        model_id: terminal.model_id.clone(),
        decode_fingerprint: terminal.decode_fingerprint.clone(),
        processing_fingerprint: terminal.processing_fingerprint.clone(),
        runtime_config_digest: terminal.runtime_config_digest.clone(),
        constraint_runtime_identities: terminal.constraint_runtime_identities.clone(),
        worker_path_evidence: terminal.worker_path_evidence.clone(),
        evidence_schema_revision: terminal.evidence_schema_revision.clone(),
        g_dec_manifest_revision: terminal.g_dec_manifest_revision.clone(),
        numeric_profile_id: Some(model.numeric_profile_id.clone()),
        fingerprint: Fingerprint(terminal.decode_fingerprint.clone()),
        certified_at_ms: now_ms(),
        os_build: state.machine_profile.os_build.clone(),
        module_generation: state.module_generation,
        evidence,
    };
    let mut unconstrained_row = row.clone();
    unconstrained_row.constraint_runtime_identities.clear();
    state
        .store
        .store_owned_decode_cert_rows_if_current(
            &snapshot_inputs,
            &terminal,
            &[row, unconstrained_row],
            now_ms(),
        )
        .map_err(|error| {
            WireOperationError::from_stable(
                StableError::engine_crashed(Some(100)),
                format!("write owned-decode probe evidence: {error}"),
            )
        })
}

fn store_probe_cert_row(
    state: &ModuleState,
    model: &EmbeddingModel,
    evidence: Value,
) -> Result<(), WireOperationError> {
    store_probe_outcome_row(state, model, CertificationStatus::Certified, evidence)
}

fn store_probe_outcome_row(
    state: &ModuleState,
    model: &EmbeddingModel,
    status: CertificationStatus,
    evidence: Value,
) -> Result<(), WireOperationError> {
    store_probe_outcome_for_fingerprint(state, model, &model.fingerprint, status, evidence)
}

fn store_probe_outcome_for_fingerprint(
    state: &ModuleState,
    model: &EmbeddingModel,
    fingerprint: &Fingerprint,
    status: CertificationStatus,
    evidence: Value,
) -> Result<(), WireOperationError> {
    let certification_class = match model.task {
        ModelTask::Embed => CertificationClass::Embedding,
        ModelTask::Rerank => CertificationClass::Rerank,
        ModelTask::Generate => {
            return Err(WireOperationError::from_stable(
                StableError::engine_crashed(Some(100)),
                "generation probes cannot write embedding or rerank certification rows",
            ))
        }
    };
    let row = ClassScopedCertificationRow {
        certification_class,
        assurance_class: AssuranceClass::Measured,
        status,
        key_hash: state.machine_profile_hash.clone(),
        machine_profile_hash: Some(state.machine_profile_hash.clone()),
        remote_profile_hash: None,
        identity_revision: None,
        numeric_profile_id: Some(model.numeric_profile_id.clone()),
        fingerprint: fingerprint.clone(),
        certified_at_ms: now_ms(),
        os_build: state.machine_profile.os_build.clone(),
        module_generation: state.module_generation,
        evidence,
    };
    state
        .store
        .store_class_scoped_cert_row(&row)
        .map_err(|error| {
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
        execute_embedding(
            runtime,
            model,
            TokenBatch { items: batch_items },
            None,
            None,
        )
        .await?;
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
            None,
            None,
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
        let owned_pairs = owned_rerank_pairs(model, item.query.as_str(), &item.candidates)?;
        requests.push((
            RerankRequest {
                query,
                candidates: token_items,
            },
            token_cost,
            owned_pairs,
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
            let (request, token_cost, owned_pairs) = &requests[cursor % requests.len()];
            execute_rerank(
                runtime,
                model,
                request.clone(),
                owned_pairs.clone(),
                None,
                None,
            )
            .await?;
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
        let (request, _, owned_pairs) = &requests[sample % requests.len()];
        let started = std::time::Instant::now();
        execute_rerank(
            runtime,
            model,
            request.clone(),
            owned_pairs.clone(),
            None,
            None,
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

fn median_ms(samples: &mut [f64]) -> f64 {
    median_value(samples)
}

fn median_value(samples: &mut [f64]) -> f64 {
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

fn probe_fixtures() -> Result<Vec<ProbeFixture>, WireOperationError> {
    let mut minilm: ProbeFixture = serde_json::from_str(include_str!(
        "fixtures/probe_corpus_minilm_ort_fp32.json"
    ))
    .map_err(|error| {
        WireOperationError::from_stable(
            StableError::artifact_invalid(),
            format!("decode built-in MiniLM probe fixture: {error}"),
        )
    })?;
    // Keep the original MiniLM fixture bytes unchanged while assigning its
    // reference identity at load time for family-safe fixture selection.
    minilm.family = Some("minilm".to_string());
    minilm.reference_model = Some("minilm".to_string());
    minilm.dims = fixture_reference_dims(&minilm);

    let gte: ProbeFixture = serde_json::from_str(include_str!(
        "fixtures/probe_corpus_gte_modernbert_ort_fp32.json"
    ))
    .map_err(|error| {
        WireOperationError::from_stable(
            StableError::artifact_invalid(),
            format!("decode built-in GTE ModernBERT probe fixture: {error}"),
        )
    })?;

    // Qwen3-Embedding reference vectors for the owned-cuda and owned-metal
    // Qwen3 lanes: candle-transformers CPU f32, last-token pool/L2, generated
    // independently of the kernels under test. Without this set, owned-cuda
    // Qwen3 certification dead-ends at `reference_fixture_missing`.
    let mut qwen3: ProbeFixture = serde_json::from_str(include_str!(
        "fixtures/probe_corpus_qwen3_embedding_fp32.json"
    ))
    .map_err(|error| {
        WireOperationError::from_stable(
            StableError::artifact_invalid(),
            format!("decode built-in Qwen3 embedding probe fixture: {error}"),
        )
    })?;
    qwen3.dims = fixture_reference_dims(&qwen3);

    Ok(vec![minilm, gte, qwen3])
}

fn probe_reference_key(model: &EmbeddingModel) -> ProbeReferenceKey {
    #[cfg(feature = "test-support")]
    if model.engine_identity.engine == test_deterministic::NAME {
        return ProbeReferenceKey {
            family: test_deterministic::NAME.into(),
            model: model.model_id.clone(),
        };
    }
    let model_id = model.model_id.to_ascii_lowercase();
    let family = model
        .engine_identity
        .build_flags
        .get("family")
        .cloned()
        .or_else(|| {
            if model_id.contains("gte-modernbert") || model_id.contains("modernbert") {
                Some("gte-modernbert".to_string())
            } else if model_id.contains("minilm") {
                Some("minilm".to_string())
            } else if model_id.contains("qwen3-embedding") {
                Some("qwen3-0.6b".to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string());
    let reference_model = if model_id.contains("gte-modernbert-base") {
        "gte-modernbert-base".to_string()
    } else if model_id.contains("minilm") {
        "minilm".to_string()
    } else if model_id.contains("qwen3-embedding") {
        "qwen3-embedding-0.6b".to_string()
    } else {
        model.model_id.clone()
    };
    ProbeReferenceKey {
        family,
        model: reference_model,
    }
}

#[cfg(feature = "test-support")]
fn test_deterministic_probe_fixtures() -> Vec<ProbeFixture> {
    // Fixed reference values for one and two unknown tokens (ID 0) from
    // the fixture tokenizer, independent of the engine's inference implementation.
    let mut one = vec![0.0_f32; test_deterministic::DIMS];
    one[0] = std::f32::consts::FRAC_1_SQRT_2;
    one[1] = std::f32::consts::FRAC_1_SQRT_2;
    let mut two = vec![0.0_f32; test_deterministic::DIMS];
    two[0] = 1.0 / 5.0_f32.sqrt();
    two[1] = 2.0 / 5.0_f32.sqrt();
    [
        "test-minilm",
        "test-minilm-a",
        "test-minilm-b",
        "test-minilm-loaded",
    ]
    .into_iter()
    .map(|id| {
        serde_json::from_value(json!({
            "family": test_deterministic::NAME, "reference_model": id,
            "dims": test_deterministic::DIMS,
            "items": [
                {"id": "one", "text": "probe", "vector": one},
                {"id": "two", "text": "probe probe", "vector": two}
            ]
        }))
        .expect("static deterministic probe fixture")
    })
    .collect()
}

fn probe_fixture_matches_key(fixture: &ProbeFixture, key: &ProbeReferenceKey) -> bool {
    fixture.family.as_deref() == Some(key.family.as_str())
        && fixture.reference_model.as_deref() == Some(key.model.as_str())
}

fn fixture_reference_dims(fixture: &ProbeFixture) -> Option<usize> {
    fixture
        .dims
        .or_else(|| fixture.items.first().map(|item| item.vector.len()))
}

fn probe_fixture_provenance(fixture: &ProbeFixture) -> Value {
    json!({
        "comment": fixture.comment,
        "family": fixture.family,
        "reference_model": fixture.reference_model,
        "model": fixture.model,
        "dims": fixture_reference_dims(fixture),
        "pooling": fixture.pooling,
        "normalize": fixture.normalize,
        "ort_version": fixture.ort_version,
        "model_sha256": fixture.model_sha256,
        "tokenizer_sha256": fixture.tokenizer_sha256,
        "items": fixture.items.len(),
        "first_id": fixture.items.first().map(|item| item.id.clone()),
        "generation_command": fixture.generation_command,
    })
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

fn generate_probe_fixtures() -> Result<Vec<GenerateProbeFixture>, WireOperationError> {
    [
        include_str!("fixtures/probe_decode_qwen3_0_6b_f16_v1.json"),
        include_str!("fixtures/probe_decode_lfm2_1_2b_f16_v1.json"),
        include_str!("fixtures/probe_decode_qwen3_0_6b_q8_0_v1.json"),
        include_str!("fixtures/probe_decode_lfm2_1_2b_q8_0_v1.json"),
    ]
    .into_iter()
    .map(|fixture| {
        serde_json::from_str(fixture).map_err(|error| {
            WireOperationError::from_stable(
                StableError::artifact_invalid(),
                format!("decode built-in generate probe fixture: {error}"),
            )
        })
    })
    .collect()
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

async fn approvals_migrate_owned_decode(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let params: ApprovalMigrationParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid approval migration params: {error}"),
            )
        }
    };
    match state
        .store
        .migrate_owned_decode_approvals(&params.seed_revision, &params.schema_revision)
    {
        Ok(result) => result_outcome(json!({
            "outcome": result.outcome,
            "seed_revision": result.seed_revision,
            "rows": result.rows,
            "marker": result.marker,
            "rendering": result.rendering(),
        })),
        Err(SynapseStoreError::ApprovalMigrationStateCorrupt(reason)) => {
            channel_error("approval_migration_state_corrupt", reason)
        }
        Err(error) => channel_error("store_failure", error.to_string()),
    }
}

async fn approval_enable(
    state: Arc<ModuleState>,
    params: Value,
    approved_by: &str,
) -> HandlerOutcome {
    let params: ApprovalEnableParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid approval enable params: {error}"),
            )
        }
    };
    match state.store.enable_or_create_approval(
        &params.model_id,
        &params.decode_fingerprint,
        params.grammar_enabled,
        approved_by,
        now_ms(),
    ) {
        Ok(row) => result_outcome(json!({
            "model_id": row.model_id,
            "decode_fingerprint": row.decode_fingerprint,
            "enabled": row.enabled,
            "grammar_enabled": row.grammar_enabled,
            "approved_by": row.approved_by,
            "approved_at_ms": row.approved_at_ms,
            "semantic_digest": row.semantic_digest,
            "generation": row.generation,
        })),
        Err(SynapseStoreError::Decode(reason)) => channel_error("invalid_request", reason),
        Err(error) => channel_error("store_failure", error.to_string()),
    }
}

async fn approval_disable(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let params: ApprovalDisableParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid approval disable params: {error}"),
            )
        }
    };
    match rollback::disable_exact_approval(
        &state.store,
        &params.model_id,
        &params.decode_fingerprint,
        &params.reason,
        now_ms(),
    ) {
        Ok(row) => result_outcome(json!({
            "model_id": row.model_id,
            "decode_fingerprint": row.decode_fingerprint,
            "enabled": row.enabled,
            "disabled_reason": row.disabled_reason,
        })),
        Err(error) => channel_error("store_failure", error.to_string()),
    }
}

async fn approvals_emergency_rollback(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let params: ApprovalEmergencyRollbackParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return channel_error(
                "invalid_request",
                format!("invalid emergency rollback params: {error}"),
            )
        }
    };
    match rollback::disable_all_approvals(&state.store, &params.reason, now_ms()) {
        Ok(disabled) => result_outcome(json!({
            "disabled": disabled,
            "reason": params.reason,
        })),
        Err(error) => channel_error("store_failure", error.to_string()),
    }
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

fn certification_class_for_task(task: &str) -> Option<CertificationClass> {
    match task {
        "embed" => Some(CertificationClass::Embedding),
        "rerank" => Some(CertificationClass::Rerank),
        _ => None,
    }
}

fn owned_measurement_report_row(row: OwnedDecodeCertificationRow) -> CertificationRow {
    CertificationRow {
        assurance_class: AssuranceClass::Measured,
        status: row.status,
        key: CertificationKey::Measured {
            machine_profile_hash: row.revisioned_machine_profile_hash,
        },
        numeric_profile_id: row
            .numeric_profile_id
            .unwrap_or_else(|| NumericProfileId(String::new())),
        fingerprint: row.fingerprint,
        certified_at_ms: row.certified_at_ms,
        os_build: row.os_build,
        module_generation: row.module_generation,
        evidence: row.evidence,
    }
}

fn lane_measurement_rows(
    state: &ModuleState,
    model_id: &str,
    task: &str,
    engine: &str,
    fingerprint: &Fingerprint,
) -> LaneMeasurementRows {
    if resolved_catalog_lane(&state.runtime, model_id).is_some() {
        return LaneMeasurementRows {
            current_certification: None,
            latest_certification: None,
            current_probe: None,
            latest_probe: None,
            current_performance: None,
            latest_performance: None,
            certification_stale: false,
            performance_stale: false,
        };
    }
    let (current_certification, latest_certification, current_probe, latest_probe) =
        if engine == DECODE_WORKER_ENGINE {
            let current_probe = state
                .store
                .get_owned_decode_measurement_row(
                    &state.revisioned_machine_profile_hash,
                    state.profile_activation_epoch,
                    model_id,
                    &fingerprint.0,
                    CERT_EVIDENCE_SCHEMA_REVISION,
                    &[],
                )
                .ok()
                .flatten()
                .map(owned_measurement_report_row);
            let current_certification = current_probe
                .as_ref()
                .filter(|row| row.status == CertificationStatus::Certified)
                .cloned();
            // `latest` must be a DIFFERENT fact from `current`, or every
            // staleness reading derived from the pair is unsatisfiable: the
            // consumers compute `current.is_none() && latest.is_some()`, which
            // is a constant false when both name the same row. So look the lane
            // up again without the profile scope -- "was this ever certified"
            // against "is it certified here" -- which is what separates a lane
            // whose profile rotated from one that was never probed.
            let latest = if current_certification.is_some() {
                current_certification.clone()
            } else {
                state
                    .store
                    .latest_owned_decode_measurement_row(
                        model_id,
                        &fingerprint.0,
                        CERT_EVIDENCE_SCHEMA_REVISION,
                        &[],
                    )
                    .ok()
                    .flatten()
                    .map(owned_measurement_report_row)
            };
            (current_certification, latest.clone(), current_probe, latest)
        } else if let Some(certification_class) = certification_class_for_task(task) {
            let current_certification = state
                .store
                .get_cert_row(
                    certification_class,
                    &state.machine_profile_hash,
                    fingerprint,
                )
                .ok()
                .flatten();
            let latest_certification = if current_certification.is_some() {
                current_certification.clone()
            } else {
                state
                    .store
                    .latest_cert_row(certification_class, fingerprint)
                    .ok()
                    .flatten()
            };
            let current_probe = state
                .store
                .get_probe_row(
                    certification_class,
                    &state.machine_profile_hash,
                    fingerprint,
                )
                .ok()
                .flatten();
            let latest_probe = if current_probe.is_some() {
                current_probe.clone()
            } else {
                state
                    .store
                    .latest_probe_row(certification_class, fingerprint)
                    .ok()
                    .flatten()
            };
            (
                current_certification,
                latest_certification,
                current_probe,
                latest_probe,
            )
        } else {
            (None, None, None, None)
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
        current_probe,
        latest_probe,
        current_performance,
        latest_performance,
    }
}

fn catalog_measurement_summary(state: &ModuleState) -> CatalogMeasurementSummary {
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
    let lanes = slots
        .into_iter()
        .filter(|slot| {
            resolved_catalog_lane(&state.runtime, &slot.spec.model_id).is_none_or(|(e, b)| {
                current_catalog_install(state, e, &b.backend)
                    .ok()
                    .flatten()
                    .is_some()
            })
        })
        .map(|slot| {
            let certification_fingerprint = slot
                .loaded
                .as_ref()
                .map(|model| model.certification_fingerprint.clone())
                .or_else(|| {
                    (slot.spec.engine == "owned-metal-decode")
                        .then(|| owned_decode_catalog_entry(&slot.spec).ok())
                        .flatten()
                        .and_then(|entry| entry.decode_identity_inputs().decode_fingerprint().ok())
                })
                .unwrap_or_else(|| slot.spec.fingerprint.clone());
            let measurements = lane_measurement_rows(
                state,
                &slot.spec.model_id,
                &slot.spec.task,
                &slot.spec.engine,
                &certification_fingerprint,
            );
            CatalogLaneMeasurements {
                slot,
                certification_fingerprint,
                measurements,
            }
        })
        .collect::<Vec<_>>();
    CatalogMeasurementSummary {
        certification_stale: lanes
            .iter()
            .any(|lane| lane.measurements.certification_stale),
        performance_stale: lanes.iter().any(|lane| lane.measurements.performance_stale),
        certified_lanes: lanes
            .iter()
            .filter(|lane| lane.measurements.current_certification.is_some())
            .count(),
        lanes,
    }
}

fn lane_certification_status(
    certification_required: bool,
    evidence_certified: bool,
) -> &'static str {
    if !certification_required {
        "not_required"
    } else if evidence_certified {
        "certified"
    } else {
        "uncertified"
    }
}

/// Projects approval + certification state into the lane summary's
/// `serving_admission` field.
///
/// Deliberately independent of worker residency: decode workers spawn lazily,
/// so "is a worker currently loaded" is a liveness fact (already visible in
/// the summary's worker block), not an admission fact. Folding it in here made
/// enabled-and-certified lanes render "disabled" after every restart until
/// their first request arrived - the same conflation this projection exists to
/// remove, in the opposite direction.
fn serving_admission_projection(
    owned_decode_lane: bool,
    evidence_certified: bool,
    approval: Option<(bool, Option<String>)>,
) -> (Option<&'static str>, Option<String>) {
    if !owned_decode_lane {
        return (None, None);
    }
    match approval {
        Some((true, _)) if evidence_certified => (Some("enabled"), None),
        Some((true, _)) => (Some("disabled"), Some("not_certified".to_string())),
        Some((false, disabled_reason)) => (
            Some("disabled"),
            disabled_reason.or_else(|| Some("approval_disabled".to_string())),
        ),
        None => (Some("disabled"), Some("approval_absent".to_string())),
    }
}

fn worker_health_from_slot(slot: &ModelSlotSnapshot) -> Option<worker_host::WorkerHostHealth> {
    slot.loaded
        .as_ref()
        .and_then(|model| worker_health_for_model(model))
}

fn lane_requires_certification(slot: &ModelSlotSnapshot) -> bool {
    slot.spec.task != ModelTask::Generate.as_str()
        || matches!(
            slot.spec.engine.as_str(),
            "owned-metal" | "owned-metal-decode"
        )
}

fn lane_blocking_reason(
    slot: &ModelSlotSnapshot,
    measurements: &LaneMeasurementRows,
    worker_quarantined: bool,
) -> Option<&'static str> {
    if !lane_requires_certification(slot) || measurements.current_certification.is_some() {
        return None;
    }
    if let Some(reason) = measurements
        .current_probe
        .as_ref()
        .and_then(|row| row.evidence.get("blocking_reason"))
        .and_then(Value::as_str)
    {
        return Some(match reason {
            "token_mismatch" => "token_mismatch",
            "fixture_unavailable" => "fixture_unavailable",
            "tokenization_failed" => "tokenization_failed",
            "generation_failed" => "generation_failed",
            "reference_fixture_missing" => "reference_fixture_missing",
            "owned_cuda_unsupported" => "owned_cuda_unsupported",
            "insufficient_vram" => "insufficient_vram",
            "backend_unavailable" => "backend_unavailable",
            _ => "probe_failed",
        });
    }
    let failed_cuda_floor = matches!(
        &slot.state,
        ModelRuntimeState::Failed(error) if error.message.contains("owned-cuda floor refused")
    );
    if failed_cuda_floor {
        return Some("owned_cuda_unsupported");
    }
    let failed_quarantined = matches!(
        &slot.state,
        ModelRuntimeState::Failed(error) if error.message.contains("quarantined")
    );
    if worker_quarantined || failed_quarantined {
        Some("quarantined")
    } else if !cfg!(target_os = "macos") && slot.spec.engine.as_str() == "ane" {
        Some("unsupported_platform")
    } else {
        Some("probe_required")
    }
}

fn certification_report_row(state: &ModuleState, row: &CertificationRow, stale: bool) -> Value {
    let (machine_profile_hash, remote_profile_hash, identity_revision) = match &row.key {
        CertificationKey::Measured {
            machine_profile_hash,
        } => (Some(machine_profile_hash.as_str()), None, None),
        CertificationKey::Declared {
            machine_profile_hash,
            remote_profile_hash,
            identity_revision,
        } => (
            Some(machine_profile_hash.as_str()),
            Some(remote_profile_hash.as_str()),
            Some(identity_revision.as_str()),
        ),
    };
    json!({
        "assurance_class": row.assurance_class,
        "status": row.status,
        "machine_profile_hash": machine_profile_hash,
        "remote_profile_hash": remote_profile_hash,
        "identity_revision": identity_revision,
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
    let CatalogMeasurementSummary {
        lanes: catalog_lanes,
        certification_stale,
        performance_stale,
        ..
    } = catalog_measurement_summary(&state);
    let knob_assignments = match state.store.knob_assignments(&state.machine_profile_hash) {
        Ok(assignments) => assignments,
        Err(error) => return channel_error("store_failure", error.to_string()),
    };
    let active_assignments = knob_assignments
        .iter()
        .filter(|assignment| assignment.knob == state.runtime.knob)
        .cloned()
        .collect::<Vec<_>>();
    let mut lanes = Vec::with_capacity(catalog_lanes.len());
    let mut omission_records = Vec::new();
    for catalog_lane in catalog_lanes {
        let CatalogLaneMeasurements {
            slot,
            certification_fingerprint,
            measurements,
        } = catalog_lane;
        if let Some((entry, backend)) = resolved_catalog_lane(&state.runtime, &slot.spec.model_id) {
            let mut row = json!({"model_id":slot.spec.model_id,"task":entry.task,"engine":slot.spec.engine,"backend":backend.backend,"fingerprint":backend.fingerprint,"state":model_runtime_state_name(&slot.state),"certification_required":false,"certification_status":"not_required","certification_stale":false,"performance_stale":false});
            if let Err(e) = catalog_list_row(&state, &mut row) {
                return result_outcome(error_payload(&state, e));
            }
            lanes.push(row);
            continue;
        }
        let worker = worker_health_from_slot(&slot);
        let worker_quarantined = worker
            .as_ref()
            .map(|health| health.quarantined_models > 0)
            .unwrap_or(false);
        let certification_required = lane_requires_certification(&slot);
        let blocking_reason = lane_blocking_reason(&slot, &measurements, worker_quarantined);
        let probe_stale =
            measurements.current_probe.is_none() && measurements.latest_probe.is_some();
        let certification = measurements
            .current_probe
            .as_ref()
            .or(measurements.latest_probe.as_ref())
            .or(measurements.current_certification.as_ref())
            .or(measurements.latest_certification.as_ref())
            .map(|row| certification_report_row(&state, row, probe_stale));
        let certification_status = lane_certification_status(
            certification_required,
            measurements.current_certification.is_some(),
        );
        let (serving_admission, serving_admission_reason) =
            if slot.spec.engine != DECODE_WORKER_ENGINE {
                (None, None)
            } else {
                match state
                    .store
                    .get_approval(&slot.spec.model_id, &certification_fingerprint.0)
                {
                    Ok(approval) => serving_admission_projection(
                        true,
                        measurements.current_certification.is_some(),
                        approval.map(|approval| (approval.enabled, approval.disabled_reason)),
                    ),
                    Err(_) => (Some("disabled"), Some("approval_unavailable".to_string())),
                }
            };
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
        let backend = slot
            .spec
            .engine_identity
            .build_flags
            .get("backend")
            .cloned();
        let support_state = if blocking_reason.is_some_and(|reason| {
            matches!(
                reason,
                "owned_cuda_unsupported" | "backend_unavailable" | "insufficient_vram"
            )
        }) {
            "unsupported"
        } else if measurements.current_certification.is_some() {
            "certified"
        } else {
            "uncertified"
        };
        let selected = active_assignments
            .iter()
            .any(|assignment| assignment.model_id == slot.spec.model_id);
        let recommendation = json!({
            "selected": selected,
            "policy": "nonmac-lane-order-v1",
            "reason": if selected { "active_machine_profile_assignment" } else { "not_selected" },
            "machine_profile_hash": state.machine_profile_hash,
        });
        let omission = blocking_reason.filter(|reason| *reason != "probe_required").map(|reason| {
            let record = json!({
                "model_id": slot.spec.model_id,
                "cell_id": format!("{}/{}/{}", slot.spec.engine, slot.spec.owned_family.as_deref().unwrap_or("unknown"), slot.spec.quant),
                "reason": reason,
                "machine_profile_hash": state.machine_profile_hash,
            });
            omission_records.push(record.clone());
            record
        });
        lanes.push(json!({
            "model_id": slot.spec.model_id,
            "cell_id": format!("{}/{}/{}", slot.spec.engine, slot.spec.owned_family.as_deref().unwrap_or("unknown"), slot.spec.quant),
            "task": slot.spec.task,
            "engine": slot.spec.engine,
            "backend": backend,
            "fingerprint": certification_fingerprint,
            "numeric_profile_id": slot.spec.numeric_profile_id,
            "state": model_runtime_state_name(&slot.state),
            "support_state": support_state,
            "certification_required": certification_required,
            "certification_status": certification_status,
            "serving_admission": serving_admission,
            "serving_admission_reason": serving_admission_reason,
            "certified": measurements.current_certification.is_some(),
            "certification_stale": measurements.certification_stale,
            "performance_stale": measurements.performance_stale,
            "blocking_reason": blocking_reason,
            "compatibility": {
                "task": slot.spec.task,
                "family": slot.spec.owned_family,
                "fingerprint": certification_fingerprint,
                "dtype_or_quantization": slot.spec.quant,
            },
            "workload_eligibility": { "eligible": support_state != "unsupported" },
            "recommendation": recommendation,
            "omission": omission,
            "certification": certification,
            "performance": performance,
            "error": error,
            "worker": worker,
        }));
    }
    result_outcome(json!({
        "module_generation": state.module_generation,
        "machine_profile_hash": state.machine_profile_hash,
        "machine_profile_hash_revision": MACHINE_PROFILE_HASH_REVISION,
        "machine_profile": state.machine_profile,
        "current_knob": state.runtime.knob,
        "certification_stale": certification_stale,
        "performance_stale": performance_stale,
        "knob_assignments": knob_assignments,
        "active_assignments": active_assignments,
        "omission_records": omission_records,
        "recommendation_policy": "nonmac-lane-order-v1",
        "lanes": lanes,
    }))
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
    let catalog_measurements = catalog_measurement_summary(&state);
    let mut lanes = state
        .runtime
        .loaded_models()
        .into_iter()
        .map(|model| {
            let measurements = lane_measurement_rows(
                &state,
                &model.model_id,
                model.task.as_str(),
                &model.engine_identity.engine,
                &model.certification_fingerprint,
            );
            json!({
                "model_id": model.model_id,
                "fingerprint": model.fingerprint,
                "meeting_deadlines": predicted_start_delay_ms <= state.runtime.inline.max_queue_ms,
                "p50_start_delay_ms": predicted_start_delay_ms,
                "execution_waiters": execution_waiters,
                "inline_in_flight_executions": execution_in_flight,
                "execution_wait_p50_ms": execution_wait_p50_ms,
                "execution_wait_p95_ms": execution_wait_p95_ms,
                "certified": measurements.current_certification.is_some(),
                "certification_stale": measurements.certification_stale,
                "performance_stale": measurements.performance_stale,
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
        "catalog_lanes": catalog_measurements.lanes.len(),
        "certified_lanes": catalog_measurements.certified_lanes,
        "certification_stale": catalog_measurements.certification_stale,
        "performance_stale": catalog_measurements.performance_stale,
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
        EmbedBackend::Owned(_) | EmbedBackend::OwnedDecode => None,
    }
}

fn certification_health(
    state: &ModuleState,
    persisted_stale_since_ms: Option<u64>,
) -> CertificationHealth {
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
    let mut certification_stale = false;
    let lanes = slots
        .into_iter()
        .map(|slot| {
            let certification_fingerprint = slot
                .loaded
                .as_ref()
                .map(|model| model.certification_fingerprint.clone())
                .or_else(|| {
                    (slot.spec.engine == "owned-metal-decode")
                        .then(|| owned_decode_catalog_entry(&slot.spec).ok())
                        .flatten()
                        .and_then(|entry| entry.decode_identity_inputs().decode_fingerprint().ok())
                })
                .unwrap_or_else(|| slot.spec.fingerprint.clone());
            let measurements = lane_measurement_rows(
                state,
                &slot.spec.model_id,
                &slot.spec.task,
                &slot.spec.engine,
                &certification_fingerprint,
            );
            certification_stale |= measurements.certification_stale;
            CertificationHealthLane {
                model_id: slot.spec.model_id,
                workload: slot.spec.task,
                certified: measurements.current_certification.is_some(),
            }
        })
        .collect();
    CertificationHealth {
        certification_stale,
        stale_since_ms: certification_stale
            .then_some(persisted_stale_since_ms)
            .flatten(),
        lanes,
    }
}

fn module_health(state: &ModuleState) -> ModuleHealth {
    let lanes = state
        .runtime
        .loaded_models()
        .into_iter()
        .map(|model| {
            let measurements = lane_measurement_rows(
                state,
                &model.model_id,
                model.task.as_str(),
                &model.engine_identity.engine,
                &model.certification_fingerprint,
            );
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
    let storage = state.store.storage_health_inputs().unwrap_or_else(|error| {
        tracing::warn!(target: "cert", error = %error, "failed to read profile health state");
        StorageHealthInputs {
            previous_revisioned_machine_profile_hash: None,
            current_revisioned_machine_profile_hash: Some(
                state.revisioned_machine_profile_hash.clone(),
            ),
            profile_activation_epoch: Some(state.profile_activation_epoch),
            certification_stale_since_ms: None,
            last_rotation_at_ms: None,
            last_rotation_reason: Some("unknown_previous_snapshot".to_string()),
            rotation_event_count: 0,
            re_certification_state: "failed".to_string(),
            evidence_requirements_divergence: Vec::new(),
            approval_certification_outcomes: Vec::new(),
        }
    });
    let certification = certification_health(state, storage.certification_stale_since_ms);
    ModuleHealth {
        status: "ok".to_string(),
        module_generation: state.module_generation,
        loaded_models: state.runtime.loaded_model_count(),
        machine_profile_hash: state.legacy_machine_profile_hash.clone(),
        certification_stale: certification.certification_stale,
        certification,
        performance_stale: lanes.iter().any(|lane| lane.performance_stale),
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
        re_certification_state: storage.re_certification_state,
        evidence_requirements_divergence: storage.evidence_requirements_divergence,
        approval_certification_outcomes: storage.approval_certification_outcomes,
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
        ane_artifact::cleanup_abandoned_temps(state.model_cache.root(), sweep_started)
            .map_err(ane_materialization_cache_error)?;
        // Partial downloads left by a crashed ingest; files younger than the
        // 24-hour floor may belong to an ingest still in progress and stay.
        state
            .model_cache
            .cleanup_abandoned_ingest_temps(sweep_started)?;
        let outcomes = if let Some(digest) = params.digest {
            vec![state
                .model_cache
                .gc_digest(&digest, &state.module_id, now, grace_ms)?]
        } else {
            // Materialized bundles and derived Q8 decode weights consume the same
            // cache budget as their source blobs. Lowering the blob watermark by
            // their current size makes the existing mark/delete pass run whenever
            // the combined cache is over budget; source deletion below reclaims
            // the matching derivatives.
            let materialized_bytes = derived_artifact_bytes(state.model_cache.root())?;
            state.model_cache.gc_to_watermark(
                &state.module_id,
                now,
                grace_ms,
                state
                    .runtime
                    .cache_max_bytes
                    .saturating_sub(materialized_bytes),
            )?
        };
        for outcome in &outcomes {
            if let CacheGcOutcome::Deleted { digest } = outcome {
                ane_artifact::remove_for_source_digest(state.model_cache.root(), digest)
                    .map_err(ane_materialization_cache_error)?;
                owned_decode_routing::q8artifact::remove_for_source_digest(
                    state.model_cache.root(),
                    digest,
                )
                .map_err(q8_artifact_cache_error)?;
            }
        }
        Ok(outcomes)
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

fn ane_materialization_cache_error(error: anyhow::Error) -> ModelCacheError {
    ModelCacheError::ArtifactInvalid(format!("ANE materialization cache: {error:#}"))
}

fn q8_artifact_cache_error(error: anyhow::Error) -> ModelCacheError {
    ModelCacheError::ArtifactInvalid(format!("owned-decode Q8 cache: {error:#}"))
}

/// Bytes of every artifact derived from a cached source blob and stored below
/// the model cache root: Core ML bundles and owned-decode Q8 weights.
fn derived_artifact_bytes(cache_root: &Path) -> Result<u64, ModelCacheError> {
    let ane_bytes = ane_artifact::total_materialized_bytes(cache_root)
        .map_err(ane_materialization_cache_error)?;
    let q8_bytes = owned_decode_routing::q8artifact::total_cached_bytes(cache_root)
        .map_err(q8_artifact_cache_error)?;
    Ok(ane_bytes.saturating_add(q8_bytes))
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

            let (max_tokens, max_tokens_source) =
                if spec.engine_identity.build_flags.contains_key("profile") {
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

            let certification_fingerprint = loaded
                .as_ref()
                .map(|model| model.certification_fingerprint.clone())
                .or_else(|| {
                    (spec.engine == "owned-metal-decode")
                        .then(|| owned_decode_catalog_entry(&spec).ok())
                        .flatten()
                        .and_then(|entry| entry.decode_identity_inputs().decode_fingerprint().ok())
                })
                .unwrap_or_else(|| spec.fingerprint.clone());

            let certified =
                if spec.engine == DECODE_WORKER_ENGINE || spec.engine == "owned-metal-decode" {
                    let has_cert = state
                        .store
                        .get_owned_decode_measurement_row(
                            &state.revisioned_machine_profile_hash,
                            state.profile_activation_epoch,
                            &spec.model_id,
                            &certification_fingerprint.0,
                            CERT_EVIDENCE_SCHEMA_REVISION,
                            &[],
                        )
                        // A store error renders identically to "no certified
                        // row": both become false. That direction is
                        // deliberate and fail-closed -- an unreadable store
                        // must never publish `certified: true`. What it costs
                        // is an operator who cannot tell "not certified" from
                        // "database unreadable", so make the error OBSERVABLE
                        // rather than changing what the flag means. Absence is
                        // already spoken for by lane classes with no
                        // certification concept; giving it a second producer
                        // here would trade a silent error for an ambiguous
                        // field.
                        .inspect_err(|error| {
                            tracing::warn!(
                                target: "synapse.catalog",
                                model_id = %spec.model_id,
                                %error,
                                "certification row unreadable; reporting certified=false"
                            );
                        })
                        .ok()
                        .flatten()
                        .is_some_and(|row| row.status == CertificationStatus::Certified);
                    Some(has_cert)
                } else if let Some(class) = certification_class_for_task(&spec.task) {
                    let has_cert = state
                        .store
                        .get_cert_row(
                            class,
                            &state.machine_profile_hash,
                            &certification_fingerprint,
                        )
                        // Same fail-closed swallow as the decode arm above, and
                        // the same remedy: report it rather than encode it.
                        .inspect_err(|error| {
                            tracing::warn!(
                                target: "synapse.catalog",
                                model_id = %spec.model_id,
                                %error,
                                "certification row unreadable; reporting certified=false"
                            );
                        })
                        .ok()
                        .flatten()
                        .is_some();
                    Some(has_cert)
                } else if spec.engine == "owned-metal" && spec.task == "generate" {
                    Some(false)
                } else {
                    // Lane classes with no certification concept (such as worker-backed llama generate)
                    // omit the field rather than fabricating true or false.
                    None
                };

            // The same projection probe.report uses, so the two surfaces cannot
            // disagree about a lane in the same second. `certified` above is
            // evidence-for-this-machine; this is approved-to-serve, and a lane
            // holding the first without the second refuses.
            let (serving_admission, serving_admission_reason) =
                if spec.engine != DECODE_WORKER_ENGINE {
                    (None, None)
                } else {
                    match state
                        .store
                        .get_approval(&spec.model_id, &certification_fingerprint.0)
                    {
                        Ok(approval) => serving_admission_projection(
                            true,
                            certified.unwrap_or(false),
                            approval.map(|approval| (approval.enabled, approval.disabled_reason)),
                        ),
                        Err(_) => (Some("disabled"), Some("approval_unavailable".to_string())),
                    }
                };

            let warm_load_cost_hint_ms = last_cold_load_ms.or_else(|| {
                if resolved_catalog_lane(&state.runtime, &spec.model_id).is_some() {
                    return None;
                }
                state
                    .store
                    .get_perf_row(&state.machine_profile_hash, &spec.fingerprint)
                    .ok()
                    .flatten()
                    .map(|perf| perf.cold_load_ms)
            });

            let recommended_batch =
                recommended_batch_for_engine(&spec.engine, max_tokens.unwrap_or(spec.max_tokens));

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
                serving_admission: serving_admission.map(str::to_string),
                serving_admission_reason,
                warm_load_cost_hint_ms,
            }
        })
        .collect()
}

fn models_list_payload(state: &ModuleState, snapshot: CatalogSnapshot) -> Value {
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
            if !state.runtime.runnable_backends.contains(&backend.backend) || !installed {
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
        models.push(row);
    }
    models.extend(state.remote_gateway.catalog_entries());
    models.sort_by(|left, right| left["model_id"].as_str().cmp(&right["model_id"].as_str()));
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

    default_storage_descriptor_with_environment(module_id, |key| env::var_os(key))
}

fn default_storage_descriptor_with_environment(
    module_id: &str,
    mut env_var: impl FnMut(&str) -> Option<OsString>,
) -> Result<StorageDescriptor, ModuleError> {
    let data_home = env_var("XDG_DATA_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            env_var("HOME")
                .filter(|value| !value.is_empty())
                .map(PathBuf::from)
                .map(|home| home.join(".local").join("share"))
        })
        .ok_or_else(|| {
            ModuleError::Config(
                "XDG_DATA_HOME and HOME are unset; cannot resolve Synapse store".to_string(),
            )
        })?;
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
        op("certify.observations", Query),
        op("embed.query", Query),
        op("embed.batch", Query),
        op("embed.result", Query),
        op("job.resume", Mutate),
        op("rerank.score", Query),
        op("microllm.oneshot", Query),
        op("owned_decode.admit_session", Mutate),
        op("owned_decode.decode", Mutate),
        op("owned_decode.snapshot", Mutate),
        op("owned_decode.continue", Mutate),
        op("owned_decode.abort", Mutate),
        op("owned_decode.close", Mutate),
        op("owned_decode.session_status", Query),
        op("owned_decode.disable", Mutate),
        op("owned_decode.revoke", Mutate),
        op("model.load", Mutate),
        op("model.status", Query),
        op("model.unload", Mutate),
        op("models.list", Query),
        op("models.catalog", Query),
        op("models.download", Mutate),
        op("models.download.cancel", Mutate),
        op("models.remove", Mutate),
        op("probe.start", Mutate),
        op("probe.status", Query),
        op("probe.report", Query),
        op("aliases.check_index", Query),
        op("alias.retract", Mutate),
        op("alias.declare", Mutate),
        op("cache.pin", Mutate),
        op("cache.gc", Mutate),
        op("admission.status", Query),
        op("approvals.migrate_owned_decode", Mutate),
        op("approvals.enable", Mutate),
        op("approvals.disable", Mutate),
        op("approvals.emergency_rollback", Mutate),
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
    ModuleManifest::builder(module_id, env!("CARGO_PKG_VERSION"))
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
        // Capability grammar is not adopted yet: leaving the block unset keeps the
        // pre-capability manifest contract, and consumers keep addressing synapse
        // by module id and operation name.
        .capabilities(None)
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
            let mut project = load_module_config_file(&project_path, ConfigTier::Project)?;
            if user_path.is_file() {
                project.remote_providers =
                    load_module_config_file(&user_path, ConfigTier::User)?.remote_providers;
            }
            return Ok(project);
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
    tier: ConfigTier,
) -> Result<ModuleConfig, ModuleError> {
    let stripped = strip_json_comments(contents);
    let value: Value = serde_json::from_str(&stripped).map_err(ModuleError::Json)?;
    if matches!(tier, ConfigTier::Project)
        && value
            .as_object()
            .is_some_and(|object| object.contains_key("remote_providers"))
    {
        return Err(ModuleError::Config(
            "remote_providers is user-tier only and may not appear in project-tier config"
                .to_string(),
        ));
    }
    if matches!(tier, ConfigTier::Project)
        && value
            .as_object()
            .is_some_and(|object| object.contains_key("decode_chain_k"))
    {
        return Err(ModuleError::Config(
            "decode_chain_k is user-tier only and may not appear in project-tier config"
                .to_string(),
        ));
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
    validate_decode_chain_k(config.decode_chain_k)?;
    validate_remote_providers(&config.remote_providers).map_err(ModuleError::Config)?;
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
            catalog_lane_lock(&runtime, &model.model_id)
                .lock_owned()
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
            "self-check must not reacquire its own catalog lane lock"
        );
    }
    #[tokio::test]
    async fn profile_less_preload_keeps_probe_required() {
        let (root, descriptor) = test_storage_descriptor("legacy-preload-gate");
        let store = Arc::new(SynapseStore::open(&descriptor).unwrap());
        let profile = test_machine_profile("legacy-preload-os");
        store.observe_profile(&profile, 10, 1).unwrap();
        let state = test_module_state(store, profile);
        let error = resolve_model_for_request(state, None, ModelTask::Embed)
            .await
            .err()
            .unwrap();
        assert_eq!(error.code, "probe_required");
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn profile_preload_serving_requires_numeric_self_check() {
        use synapse_parity::evaluator::{evaluate_subset, ObservedCase, Output};
        let (root, descriptor) = test_storage_descriptor("profile-preload-gate");
        let profile = "gte-modernbert-base.owned-vulkan";
        let state = catalog_test_state(&root, &descriptor, profile);
        let model = state.runtime.loaded_models().into_iter().next().unwrap();
        let error = ensure_model_certified(&state, &model, CertificationClass::Embedding, true)
            .unwrap_err();
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
        assert!(
            ensure_model_certified(&state, &model, CertificationClass::Embedding, true).is_ok()
        );
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
            ensure_model_certified(&state, &model, CertificationClass::Embedding, true)
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
        let spec = catalog_fixture_config("gte-modernbert-base.ane-direct-worker");
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
        release.notify_one();
        let result = task.await.unwrap();
        tokio::task::spawn_blocking({
            let engine = engine.clone();
            move || engine.unload()
        })
        .await
        .unwrap()
        .unwrap();
        assert_eq!(result.unwrap_err().code, "deadline_exceeded");
        assert!(
            finished_before_reply,
            "caller deadline must bound the result wait"
        );
        assert_eq!(
            in_flight, 1,
            "expired work must retain guards while its reply drains"
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
            subc_protocol::BindIdentity::new(PathBuf::from("/"), "synapse-test", "session-1"),
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

    #[test]
    fn flow_routes_get_queries_only_and_other_routes_are_unchanged() {
        let handler = SynapseHandler::new("synapse".into(), PathBuf::new());
        let inner = &handler.inner;
        let flow = bind_with_scope(1, Some("flow-7"));
        let plain = bind_with_scope(2, None);
        inner.record_bind_scope(&flow);
        inner.record_bind_scope(&plain);

        let operations = management_operations();
        let queries = operations
            .iter()
            .filter(|op| op.kind == ManagementOperationKind::Query)
            .count();
        let mutations = operations.len() - queries;
        // Denominators: a list with no queries or no mutations would make the
        // loop below vacuous.
        assert!(
            queries > 0 && mutations > 0,
            "{queries} queries, {mutations} mutations"
        );
        for op in &operations {
            let on_flow = refusal_code(inner.flow_refusal(&flow.handle, &op.name));
            if op.kind == ManagementOperationKind::Query {
                assert_eq!(on_flow, None, "query {} must be served to a flow", op.name);
            } else {
                assert_eq!(
                    on_flow.as_deref(),
                    Some("flow_scope_refused"),
                    "mutation {} must be refused on a flow route",
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
    }

    #[test]
    fn the_flow_refusal_names_the_method_and_the_flow() {
        let handler = SynapseHandler::new("synapse".into(), PathBuf::new());
        let flow = bind_with_scope(3, Some("flow-9"));
        handler.inner.record_bind_scope(&flow);
        match handler.inner.flow_refusal(&flow.handle, "model.load") {
            Some(HandlerOutcome::Error { code, message }) => {
                assert_eq!(code, "flow_scope_refused");
                assert!(message.contains("model.load"), "{message}");
                assert!(message.contains("flow-9"), "{message}");
            }
            _ => panic!("model.load on a flow route must be refused"),
        }
        // A rebind of the same handle without flow_id clears the record.
        handler.inner.record_bind_scope(&bind_with_scope(3, None));
        assert!(handler
            .inner
            .flow_refusal(&flow.handle, "model.load")
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
        let root = std::env::temp_dir().join(format!(
            "synapse-probe-key-{}-{}",
            std::process::id(),
            TEST_STATE_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        let good = root.join(if cfg!(windows) { "good.cmd" } else { "good.sh" });
        let bad = root.join(if cfg!(windows) { "bad.cmd" } else { "bad.sh" });
        let header = if cfg!(windows) {
            "@echo off\r\n"
        } else {
            "#!/bin/sh\n"
        };
        let json = r#"{"driver_api":13030,"compute_capability":{"major":8,"minor":9}}"#;
        let success = if cfg!(windows) {
            format!("{header}echo {json}\r\n")
        } else {
            format!("{header}echo '{json}'\n")
        };
        fs::write(&good, &success).unwrap();
        fs::write(
            &bad,
            format!(
                "{header}echo missing-library >&2\n{}\n",
                if cfg!(windows) { "exit /b 9" } else { "exit 9" }
            ),
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            for path in [&good, &bad] {
                fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
            }
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
        fs::write(&bad, success).unwrap();
        assert_eq!(owned_cuda_probe_floor(Some(&bad)).unwrap_err(), failure);
        assert_eq!(
            owned_cuda_probe_floor(Some(&good)).unwrap().driver_api,
            13030
        );
        fs::remove_dir_all(root).unwrap();
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
        let slow = root.join(if cfg!(windows) { "slow.cmd" } else { "slow.sh" });
        let quick = root.join(if cfg!(windows) {
            "quick.cmd"
        } else {
            "quick.sh"
        });
        let json = r#"{"driver_api":13030,"compute_capability":{"major":8,"minor":9}}"#;
        let stalled = if cfg!(windows) {
            format!(
                "@echo off\r\necho ready >\"{}\"\r\n:wait\r\nif exist \"{}\" goto done\r\nping -n 2 127.0.0.1 >nul\r\ngoto wait\r\n:done\r\necho {json}\r\n",
                ready.display(), release.display()
            )
        } else {
            format!(
                "#!/bin/sh\nprintf ready >'{}'\nwhile [ ! -f '{}' ]; do sleep 0.05; done\necho '{json}'\n",
                ready.display(), release.display()
            )
        };
        let success = if cfg!(windows) {
            format!("@echo off\r\necho {json}\r\n")
        } else {
            format!("#!/bin/sh\necho '{json}'\n")
        };
        fs::write(&slow, stalled).unwrap();
        fs::write(&quick, success).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            for path in [&slow, &quick] {
                fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
            }
        }
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
    fn probe_report_separates_certification_from_serving_admission() {
        let (serving_admission, serving_admission_reason) = serving_admission_projection(
            true,
            true,
            Some((false, Some("held for operator review".to_string()))),
        );
        let lfm2_lane = json!({
            "certification_status": lane_certification_status(true, true),
            "serving_admission": serving_admission,
            "serving_admission_reason": serving_admission_reason,
        });
        assert_eq!(lfm2_lane["certification_status"], "certified");
        assert_eq!(lfm2_lane["serving_admission"], "disabled");
        assert_eq!(
            lfm2_lane["serving_admission_reason"],
            "held for operator review"
        );

        let (serving_admission, serving_admission_reason) =
            serving_admission_projection(true, false, Some((true, None)));
        let uncertified_lane = json!({
            "certification_status": lane_certification_status(true, false),
            "serving_admission": serving_admission,
            "serving_admission_reason": serving_admission_reason,
        });
        assert_eq!(uncertified_lane["certification_status"], "uncertified");

        let (serving_admission, serving_admission_reason) =
            serving_admission_projection(true, true, Some((true, None)));
        let enabled_lane = json!({
            "certification_status": lane_certification_status(true, true),
            "serving_admission": serving_admission,
            "serving_admission_reason": serving_admission_reason,
        });
        // Residency is no longer an input by construction: enabled-and-certified
        // renders enabled whether or not a worker has spawned yet (the shape every
        // decode lane is in right after a restart).
        assert_eq!(enabled_lane["certification_status"], "certified");
        assert_eq!(enabled_lane["serving_admission"], "enabled");
    }

    #[test]
    fn wire_contract_documents_every_management_operation() {
        const CONTRACT: &str = include_str!("../../../docs/wire-contract-v1.md");
        const NEWLY_DOCUMENTED_OPERATIONS: [&str; 16] = [
            "owned_decode.admit_session",
            "owned_decode.decode",
            "owned_decode.snapshot",
            "owned_decode.continue",
            "owned_decode.abort",
            "owned_decode.close",
            "owned_decode.session_status",
            "owned_decode.disable",
            "owned_decode.revoke",
            "model.unload",
            "alias.declare",
            "alias.retract",
            "approvals.migrate_owned_decode",
            "approvals.enable",
            "approvals.disable",
            "approvals.emergency_rollback",
        ];

        let operation_names = management_operations()
            .into_iter()
            .map(|operation| operation.name)
            .collect::<BTreeSet<_>>();
        let undocumented = operation_names
            .iter()
            .filter(|name| !CONTRACT.contains(&format!("`{name}`")))
            .collect::<Vec<_>>();
        assert!(
            undocumented.is_empty(),
            "wire contract is missing registered operations: {undocumented:?}"
        );

        for name in NEWLY_DOCUMENTED_OPERATIONS {
            assert!(
                operation_names.contains(name),
                "coverage sentinel is no longer registered: {name}"
            );
            assert!(
                CONTRACT.contains(&format!("`{name}`")),
                "wire contract is missing newly documented operation: {name}"
            );
        }
    }

    #[test]
    fn wire_contract_documents_every_stable_error_code() {
        const CONTRACT: &str = include_str!("../../../docs/wire-contract-v1.md");

        let undocumented_error_codes = synapse_core::StableErrorCode::ALL
            .into_iter()
            .filter_map(|error_code| {
                let wire_code = serde_json::to_value(error_code)
                    .expect("stable error code serializes")
                    .as_str()
                    .expect("stable error code serializes as a string")
                    .to_owned();
                (!CONTRACT.contains(&format!("`{wire_code}`"))).then_some(wire_code)
            })
            .collect::<Vec<_>>();
        assert!(
            undocumented_error_codes.is_empty(),
            "wire contract is missing stable error codes: {undocumented_error_codes:?}"
        );
    }

    #[test]
    fn wire_contract_documents_every_model_catalog_field() {
        const CONTRACT: &str = include_str!("../../../docs/wire-contract-v1.md");
        const MODEL_CATALOG_FIELDS: [&str; 9] = [
            "recommended_batch",
            "max_tokens",
            "max_tokens_source",
            "bucket_ladder",
            "dims",
            "dtype",
            "device_class",
            "certified",
            "warm_load_cost_hint_ms",
        ];

        for field in MODEL_CATALOG_FIELDS {
            assert!(
                CONTRACT.contains(&format!("`{field}`")),
                "wire contract is missing model catalog field: {field}"
            );
        }
    }

    #[test]
    fn wire_contract_documents_every_embed_vector_field() {
        const CONTRACT: &str = include_str!("../../../docs/wire-contract-v1.md");
        const EMBED_VECTOR_FIELDS: [&str; 4] =
            ["id", "vector", "content_sha256", "submitted_sha256"];

        for field in EMBED_VECTOR_FIELDS {
            assert!(
                CONTRACT.contains(&format!("`{field}`")),
                "wire contract is missing embed vector field: {field}"
            );
        }
    }

    #[test]
    fn readme_lists_every_workspace_production_crate() {
        // Third instance of the enumeration-gap class (issues #2, #5, #8): a
        // document describing a declared surface silently omits members added
        // after it was written. The enumeration source here is the workspace
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
    fn sidecar_config_is_default_off() {
        assert!(!ModuleConfig::default().sidecar_spec.enabled);
    }

    #[test]
    fn frozen_sidecar_schema_preserves_canonical_property_order() {
        let schema = owned_decode_grammar_scheduler::grammar_schema::parse_schema(
            r#"{"type":"object","properties":{"second":{"type":"integer"},"first":{"type":"string","enum":["ok"]}},"required":["first"],"additionalProperties":false}"#,
            &owned_decode_grammar_scheduler::grammar_limits::GrammarLimits::default(),
        )
        .expect("schema parses");
        let frozen = frozen_sidecar_object_schema(&schema, "schema-v1".to_string())
            .expect("object schema is sidecar-renderable");
        assert_eq!(frozen.schema_identity, "schema-v1");
        assert_eq!(
            frozen
                .properties
                .iter()
                .map(|property| property.name.as_str())
                .collect::<Vec<_>>(),
            ["first", "second"]
        );
        assert!(frozen.properties[0].required);
        assert!(!frozen.properties[1].required);
    }

    #[test]
    fn catalog_identity_names_match_worker_hello_constants() {
        let expected = [
            ("llama", LLAMA_WORKER_ENGINE),
            ("ane", ANE_WORKER_ENGINE),
            ("owned-metal-decode", DECODE_WORKER_ENGINE),
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
            engine: "ort".to_string(),
            task: "embed".to_string(),
            artifact_digest: "artifact".to_string(),
            artifact_format: "onnx".to_string(),
            tokenizer_sanitized_digest: "tokenizer".to_string(),
            model_locator: ModelAssetLocator::LocalPath {
                path: PathBuf::from("/tmp/stuck-model.onnx"),
            },
            tokenizer_locator: ModelAssetLocator::LocalPath {
                path: PathBuf::from("/tmp/stuck-tokenizer.json"),
            },
            model_source_url: "file:///tmp/stuck-model.onnx".to_string(),
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
                engine: "ort".to_string(),
                version: "test".to_string(),
                build_flags: BTreeMap::new(),
            },
            numeric_profile_id: NumericProfileId("test-profile".to_string()),
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
        let runtime = Arc::new(
            RuntimeState::from_catalog(config, vec![stuck_model_spec()])
                .expect("test runtime initializes"),
        );
        let remote_gateway = Arc::new(
            RemoteGateway::new(
                Arc::clone(&store),
                Vec::new(),
                Arc::new(SubcVaultCredentialClient::new(PathBuf::from("unused-subc"))),
                machine_profile_hash.clone(),
            )
            .expect("empty remote gateway initializes"),
        );
        let continuity_check: Arc<dyn ContinuityCheck> = remote_gateway.continuity.clone();
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
            continuity_check,
            remote_gateway,
        })
    }

    const PENDING_ABORT_SESSION_ID: &str = "pending-abort-session";
    const PENDING_ABORT_REQ_ID: &str = "pending-abort-request";
    const PENDING_ABORT_COMMITTED: u32 = 2;
    const PENDING_ABORT_SEQUENCE: u64 = 11;

    fn pending_abort_identity() -> synapse_core::OneshotEnvelopeIdentity {
        synapse_core::OneshotEnvelopeIdentity {
            decode_fingerprint: Fingerprint("decode-fingerprint".to_string()),
            processing_fingerprint: Fingerprint("processing-fingerprint".to_string()),
            runtime_config_digest: "runtime-digest".to_string(),
            worker_generation: 7,
            derived_digest: Some("derived-digest".to_string()),
        }
    }

    fn pending_abort_test_state(
        label: &str,
        approved_catalog: bool,
    ) -> (PathBuf, Arc<ModuleState>, String) {
        let (root, descriptor) = test_storage_descriptor(label);
        let store = Arc::new(SynapseStore::open(&descriptor).expect("pending-abort store opens"));
        let profile = test_machine_profile(label);
        store
            .observe_profile(&profile, 10, 1)
            .expect("pending-abort profile activates");
        let catalog_fingerprint = if approved_catalog {
            store::configure_serving_catalog_for_test(&store)
        } else {
            "d".repeat(64)
        };
        let state = test_module_state(store, profile);
        let op_id = session_request_key(PENDING_ABORT_SESSION_ID, PENDING_ABORT_REQ_ID);
        {
            let mut sessions = state
                .runtime
                .owned_decode_sessions
                .lock()
                .expect("decode sessions lock");
            sessions.next_session_sequence = PENDING_ABORT_SEQUENCE;
            sessions.sessions.insert(
                PENDING_ABORT_SESSION_ID.to_string(),
                OwnedDecodeWireSession {
                    catalog_fingerprint,
                    model_id: "test-model".to_string(),
                    routing_session_id: owned_decode_routing::admission::SessionId(1),
                    kv_configuration: owned_decode_routing::admission::SessionKvConfiguration::new(
                        256, 256,
                    )
                    .expect("test KV configuration is valid"),
                    active_request: Some(PENDING_ABORT_REQ_ID.to_string()),
                    retained_kv_session_id: None,
                    retained_position: None,
                    closed_at_ms: None,
                },
            );
            sessions
                .streams
                .begin(owned_decode_worker::StreamRequest {
                    req_id: PENDING_ABORT_REQ_ID.to_string(),
                    session_id: PENDING_ABORT_SESSION_ID.to_string(),
                    generation_id: "generation-1".to_string(),
                    identity: pending_abort_identity(),
                    decode_mode: synapse_core::DecodeMode::Serial,
                    grammar_constrained: false,
                    chain_k: 1,
                })
                .expect("pending-abort stream begins");
            sessions
                .streams
                .observe_frame(&synapse_core::FrameEnvelope::new(
                    PENDING_ABORT_REQ_ID,
                    PENDING_ABORT_SESSION_ID,
                    synapse_core::StreamSequence::FIRST,
                    synapse_core::WorkerFrame::Progress {
                        progress: synapse_core::ProgressFrame {
                            committed_token_ids: vec![17, 23],
                            committed_token_count: PENDING_ABORT_COMMITTED,
                            boundary: synapse_core::ProgressBoundary::Continuing,
                        },
                    },
                ))
                .expect("pending-abort progress is observed");
            sessions
                .scheduler
                .admit_decode(owned_decode_grammar_scheduler::scheduler::DecodeOp {
                    op_id: op_id.clone(),
                    generation_id: PENDING_ABORT_REQ_ID.to_string(),
                    admitted_at_ms: 1,
                    anchor_ms: 1,
                    committed_tokens: PENDING_ABORT_COMMITTED,
                    max_tokens: 16,
                    resident: true,
                    cancelled_at_ms: None,
                    deadline_at_ms: None,
                });
            assert_eq!(
                sessions
                    .scheduler
                    .arbitrate(1)
                    .and_then(|selected| selected.op_id),
                Some(op_id.clone())
            );
        }
        (root, state, op_id)
    }

    fn assert_abort_terminal(frame: &synapse_core::FrameEnvelope) {
        let synapse_core::WorkerFrame::Error { terminal } = &frame.frame else {
            panic!("pending abort should push an error terminal")
        };
        assert_eq!(
            terminal.terminal_state,
            synapse_core::TerminalState::Aborted
        );
        assert_eq!(terminal.committed_token_count, PENDING_ABORT_COMMITTED);
    }

    #[test]
    fn pending_session_abort_retains_approved_kv_prefix() {
        let (root, state, op_id) = pending_abort_test_state("abort-retained", true);
        let retained_id = format!(
            "{PENDING_ABORT_SESSION_ID}:retained:{PENDING_ABORT_COMMITTED}:{PENDING_ABORT_SEQUENCE}"
        );
        let mut frames = Vec::new();
        let result = {
            let mut sessions = state
                .runtime
                .owned_decode_sessions
                .lock()
                .expect("decode sessions lock");
            sessions.pending_aborts.insert(
                (
                    PENDING_ABORT_SESSION_ID.to_string(),
                    PENDING_ABORT_REQ_ID.to_string(),
                ),
                PendingSessionAbort { retain_kv: true },
            );
            let outcome = take_pending_session_abort(
                &state,
                &mut sessions,
                PENDING_ABORT_SESSION_ID,
                PENDING_ABORT_REQ_ID,
                PENDING_ABORT_COMMITTED,
                &op_id,
                &mut frames,
            )
            .expect("pending abort diverts decode");
            let session = sessions
                .sessions
                .get(PENDING_ABORT_SESSION_ID)
                .expect("session remains registered");
            assert_eq!(sessions.next_session_sequence, PENDING_ABORT_SEQUENCE + 1);
            assert_eq!(
                session.retained_kv_session_id.as_deref(),
                Some(retained_id.as_str())
            );
            assert_eq!(session.retained_position, Some(PENDING_ABORT_COMMITTED));
            assert_eq!(session.active_request, None);
            assert!(sessions.scheduler.op(&op_id).is_none());
            response_result(outcome, "take pending retained abort")
        };

        assert_eq!(result["cancelled"]["committed_token_count"], 2);
        assert_eq!(frames.len(), 1);
        assert_abort_terminal(&frames[0]);
        let retained = state
            .store
            .retained_serving_state(&retained_id)
            .expect("retained state reads")
            .expect("approved retention writes a store row");
        assert_eq!(retained.state_id, retained_id);
        assert!(retained.valid);

        drop(state);
        fs::remove_dir_all(root).expect("remove pending-abort state");
    }

    #[test]
    fn pending_session_abort_refused_retention_aborts_without_prefix() {
        let (root, state, op_id) = pending_abort_test_state("abort-refused", false);
        let refused_id = format!(
            "{PENDING_ABORT_SESSION_ID}:retained:{PENDING_ABORT_COMMITTED}:{PENDING_ABORT_SEQUENCE}"
        );
        let mut frames = Vec::new();
        {
            let mut sessions = state
                .runtime
                .owned_decode_sessions
                .lock()
                .expect("decode sessions lock");
            sessions.pending_aborts.insert(
                (
                    PENDING_ABORT_SESSION_ID.to_string(),
                    PENDING_ABORT_REQ_ID.to_string(),
                ),
                PendingSessionAbort { retain_kv: true },
            );
            let outcome = take_pending_session_abort(
                &state,
                &mut sessions,
                PENDING_ABORT_SESSION_ID,
                PENDING_ABORT_REQ_ID,
                PENDING_ABORT_COMMITTED,
                &op_id,
                &mut frames,
            )
            .expect("pending abort diverts decode");
            let result = response_result(outcome, "take pending refused abort");
            assert_eq!(result["cancelled"]["committed_token_count"], 2);
            let session = sessions
                .sessions
                .get(PENDING_ABORT_SESSION_ID)
                .expect("session remains registered");
            assert_eq!(sessions.next_session_sequence, PENDING_ABORT_SEQUENCE + 1);
            assert_eq!(session.retained_kv_session_id, None);
            assert_eq!(session.retained_position, None);
            assert_eq!(session.active_request, None);
            assert!(sessions.scheduler.op(&op_id).is_none());
        }

        assert_eq!(frames.len(), 1);
        assert_abort_terminal(&frames[0]);
        assert!(state
            .store
            .retained_serving_state(&refused_id)
            .expect("retained state reads")
            .is_none());

        drop(state);
        fs::remove_dir_all(root).expect("remove pending-abort state");
    }

    #[test]
    fn pending_session_abort_without_retention_aborts_cleanly() {
        let (root, state, op_id) = pending_abort_test_state("abort-not-requested", false);
        let unrequested_id = format!(
            "{PENDING_ABORT_SESSION_ID}:retained:{PENDING_ABORT_COMMITTED}:{PENDING_ABORT_SEQUENCE}"
        );
        let mut frames = Vec::new();
        {
            let mut sessions = state
                .runtime
                .owned_decode_sessions
                .lock()
                .expect("decode sessions lock");
            sessions.pending_aborts.insert(
                (
                    PENDING_ABORT_SESSION_ID.to_string(),
                    PENDING_ABORT_REQ_ID.to_string(),
                ),
                PendingSessionAbort { retain_kv: false },
            );
            let outcome = take_pending_session_abort(
                &state,
                &mut sessions,
                PENDING_ABORT_SESSION_ID,
                PENDING_ABORT_REQ_ID,
                PENDING_ABORT_COMMITTED,
                &op_id,
                &mut frames,
            )
            .expect("pending abort diverts decode");
            let result = response_result(outcome, "take pending unretained abort");
            assert_eq!(result["cancelled"]["committed_token_count"], 2);
            let session = sessions
                .sessions
                .get(PENDING_ABORT_SESSION_ID)
                .expect("session remains registered");
            assert_eq!(sessions.next_session_sequence, PENDING_ABORT_SEQUENCE);
            assert_eq!(session.retained_kv_session_id, None);
            assert_eq!(session.retained_position, None);
            assert_eq!(session.active_request, None);
            assert!(sessions.scheduler.op(&op_id).is_none());
        }

        assert_eq!(frames.len(), 1);
        assert_abort_terminal(&frames[0]);
        assert!(state
            .store
            .retained_serving_state(&unrequested_id)
            .expect("retained state reads")
            .is_none());

        drop(state);
        fs::remove_dir_all(root).expect("remove pending-abort state");
    }

    #[test]
    fn missing_pending_session_abort_does_not_divert_decode() {
        let (root, state, op_id) = pending_abort_test_state("abort-missing", false);
        let mut frames = Vec::new();
        {
            let mut sessions = state
                .runtime
                .owned_decode_sessions
                .lock()
                .expect("decode sessions lock");
            assert!(take_pending_session_abort(
                &state,
                &mut sessions,
                PENDING_ABORT_SESSION_ID,
                PENDING_ABORT_REQ_ID,
                PENDING_ABORT_COMMITTED,
                &op_id,
                &mut frames,
            )
            .is_none());
            let session = sessions
                .sessions
                .get(PENDING_ABORT_SESSION_ID)
                .expect("session remains registered");
            assert_eq!(sessions.next_session_sequence, PENDING_ABORT_SEQUENCE);
            assert_eq!(
                session.active_request.as_deref(),
                Some(PENDING_ABORT_REQ_ID)
            );
            assert!(sessions.scheduler.op(&op_id).is_some());
        }
        assert!(frames.is_empty());

        drop(state);
        fs::remove_dir_all(root).expect("remove pending-abort state");
    }

    const RETAINED_SESSION_ID: &str = "retained-closed-session";
    const RETAINED_REQ_ID: &str = "retained-closed-request";

    /// A module state whose store holds one approved, certified serving catalog.
    fn owned_decode_serving_test_state(label: &str) -> (PathBuf, Arc<ModuleState>, String) {
        let (root, descriptor) = test_storage_descriptor(label);
        let store = Arc::new(SynapseStore::open(&descriptor).expect("serving store opens"));
        let profile = test_machine_profile(label);
        store
            .observe_profile(&profile, 10, 1)
            .expect("serving profile activates");
        let catalog_fingerprint = store::configure_serving_catalog_for_test(&store);
        let state = test_module_state(store, profile);
        (root, state, catalog_fingerprint)
    }

    /// The model the test catalog's certification record names, read straight
    /// from the stored certification.
    fn certified_model_id(state: &ModuleState, catalog_fingerprint: &str) -> String {
        let approval = state
            .store
            .serving_approval(catalog_fingerprint)
            .expect("serving approval reads")
            .expect("test catalog is approved");
        state
            .store
            .serving_certification(&approval.certification_record_id)
            .expect("serving certification reads")
            .expect("approval references a certification")
            .record
            .artifact_lineage
            .model_id
    }

    /// Register an open session in both the durable ledger and the in-memory
    /// map, with a completed request stream so `session_status` has a terminal
    /// state to report.
    fn register_completed_session(
        state: &ModuleState,
        catalog_fingerprint: &str,
        model_id: &str,
        admitted_at_ms: u64,
    ) {
        assert!(matches!(
            state
                .store
                .admit_serving_session(RETAINED_SESSION_ID, catalog_fingerprint, admitted_at_ms)
                .expect("serving session admission persists"),
            store::ServingSessionAdmission::Admitted { .. }
        ));
        let mut sessions = state
            .runtime
            .owned_decode_sessions
            .lock()
            .expect("decode sessions lock");
        sessions.sessions.insert(
            RETAINED_SESSION_ID.to_string(),
            OwnedDecodeWireSession {
                catalog_fingerprint: catalog_fingerprint.to_string(),
                model_id: model_id.to_string(),
                routing_session_id: owned_decode_routing::admission::SessionId(1),
                kv_configuration: owned_decode_routing::admission::SessionKvConfiguration::new(
                    256, 256,
                )
                .expect("test KV configuration is valid"),
                active_request: None,
                retained_kv_session_id: None,
                retained_position: None,
                closed_at_ms: None,
            },
        );
        sessions
            .streams
            .begin(owned_decode_worker::StreamRequest {
                req_id: RETAINED_REQ_ID.to_string(),
                session_id: RETAINED_SESSION_ID.to_string(),
                generation_id: "generation-1".to_string(),
                identity: pending_abort_identity(),
                decode_mode: synapse_core::DecodeMode::Serial,
                grammar_constrained: false,
                chain_k: 1,
            })
            .expect("request stream begins");
        sessions
            .streams
            .observe_frame(&synapse_core::FrameEnvelope::new(
                RETAINED_REQ_ID,
                RETAINED_SESSION_ID,
                synapse_core::StreamSequence::FIRST,
                synapse_core::WorkerFrame::Final {
                    terminal: synapse_core::TerminalEnvelope {
                        req_id: RETAINED_REQ_ID.to_string(),
                        session_id: RETAINED_SESSION_ID.to_string(),
                        committed_token_count: 0,
                        tokens_emitted: 0,
                        identity: pending_abort_identity(),
                        terminal_state: synapse_core::TerminalState::Completed,
                        decode_mode: synapse_core::DecodeMode::Serial,
                        speculative_telemetry: None,
                    },
                },
            ))
            .expect("terminal frame is observed");
    }

    async fn close_retained_session(state: &Arc<ModuleState>) -> Value {
        response_result(
            owned_decode_session_close(
                Arc::clone(state),
                json!({ "session_id": RETAINED_SESSION_ID }),
            )
            .await,
            "owned_decode.close",
        )
    }

    async fn retained_session_status(state: &Arc<ModuleState>) -> Value {
        response_result(
            owned_decode_session_status(
                Arc::clone(state),
                json!({ "session_id": RETAINED_SESSION_ID, "req_id": RETAINED_REQ_ID }),
            )
            .await,
            "owned_decode.session_status",
        )
    }

    fn retained_session_closed_at_ms(state: &ModuleState) -> u64 {
        state
            .runtime
            .owned_decode_sessions
            .lock()
            .expect("decode sessions lock")
            .sessions
            .get(RETAINED_SESSION_ID)
            .expect("closed session is still registered")
            .closed_at_ms
            .expect("close records when the session closed")
    }

    fn retained_session_registered(state: &ModuleState) -> bool {
        state
            .runtime
            .owned_decode_sessions
            .lock()
            .expect("decode sessions lock")
            .sessions
            .contains_key(RETAINED_SESSION_ID)
    }

    /// A dispatch that never starts a worker; the tests only check whether the
    /// dispatch cache still holds it.
    fn idle_decode_dispatch(
        root: &Path,
        name: &str,
    ) -> Arc<Mutex<worker_host::SupervisedDecodeDispatch>> {
        use owned_decode_worker::{
            budget::BudgetPolicy,
            identity::QuarantineKey,
            protocol::{GenerateStart, Sampling},
            supervisor::TerminalControl,
            validation::WorkerStartContext,
        };
        let factory = worker_host::OwnedDecodeWorkerFactory::new(
            worker_host::WorkerHostConfig::new("missing-owned-decode-worker", root),
            ValidatedArtifact {
                digest: "idle-digest".to_string(),
                format: "owned-safetensors".to_string(),
            },
            RuntimeConfig {
                values: BTreeMap::new(),
            },
        );
        let dispatch = worker_host::SupervisedDecodeDispatch::new(
            factory,
            root.join(format!("{name}-budget.json")),
            BudgetPolicy::default(),
            16,
            QuarantineKey::new("idle-machine", "idle-fingerprint", "idle-runtime"),
            GenerateStart {
                generation_id: String::new(),
                loaded_model_ref: String::new(),
                decode_fingerprint: "idle-fingerprint".to_string(),
                runtime_config_digest: "idle-runtime".to_string(),
                prompt_ids: vec![1],
                stop_ids: Vec::new(),
                max_tokens: 1,
                sampling: Sampling::greedy_top1(),
                constraint: None,
            },
            WorkerStartContext {
                loaded_model_ref: String::new(),
                decode_fingerprint: "idle-fingerprint".to_string(),
                runtime_config_digest: "idle-runtime".to_string(),
                expected_constraint: None,
            },
            TerminalControl::default(),
        )
        .expect("idle dispatch opens its budget store");
        Arc::new(Mutex::new(dispatch))
    }

    /// Cache a dispatch for the certified model and one for an unrelated model,
    /// so a test can tell a targeted unload from clearing the whole cache.
    fn cache_certified_and_unrelated_dispatches(
        state: &ModuleState,
        root: &Path,
        certified_model_id: &str,
    ) {
        let mut dispatches = state
            .runtime
            .owned_decode_dispatches
            .lock()
            .expect("dispatch cache lock");
        dispatches.insert(
            certified_model_id.to_string(),
            idle_decode_dispatch(root, "certified"),
        );
        dispatches.insert(
            "unrelated-model".to_string(),
            idle_decode_dispatch(root, "unrelated"),
        );
    }

    fn cached_dispatch_models(state: &ModuleState) -> Vec<String> {
        state
            .runtime
            .owned_decode_dispatches
            .lock()
            .expect("dispatch cache lock")
            .keys()
            .cloned()
            .collect()
    }

    async fn disable_catalog(state: &Arc<ModuleState>, catalog_fingerprint: &str) -> Value {
        response_result(
            owned_decode_disable(
                Arc::clone(state),
                json!({
                    "catalog_fingerprint": catalog_fingerprint,
                    "reason": "operator disable",
                }),
            )
            .await,
            "owned_decode.disable",
        )
    }

    #[tokio::test]
    async fn maintenance_evicts_session_closed_longer_than_retention() {
        let (root, state, catalog) = owned_decode_serving_test_state("evict-expired-closed");
        let model_id = certified_model_id(&state, &catalog);
        register_completed_session(&state, &catalog, &model_id, now_ms());
        assert_eq!(close_retained_session(&state).await["closed"], true);
        let closed_at_ms = retained_session_closed_at_ms(&state);

        run_background_maintenance_at(
            &state,
            closed_at_ms + CLOSED_OWNED_DECODE_SESSION_RETENTION_MS + 1,
        );

        assert!(!retained_session_registered(&state));
        assert!(
            state
                .runtime
                .owned_decode_sessions
                .lock()
                .expect("decode sessions lock")
                .streams
                .session_status(RETAINED_SESSION_ID, RETAINED_REQ_ID)
                .is_err(),
            "an evicted session's request stream record must be dropped with it"
        );
        assert_eq!(
            retained_session_status(&state).await["error"]["code"],
            "unknown_session"
        );
        assert_eq!(
            close_retained_session(&state).await["error"]["code"],
            "unknown_session"
        );

        drop(state);
        fs::remove_dir_all(root).expect("remove eviction state");
    }

    #[tokio::test]
    async fn maintenance_keeps_session_closed_within_retention() {
        let (root, state, catalog) = owned_decode_serving_test_state("keep-recent-closed");
        let model_id = certified_model_id(&state, &catalog);
        register_completed_session(&state, &catalog, &model_id, now_ms());
        assert_eq!(close_retained_session(&state).await["closed"], true);
        let closed_at_ms = retained_session_closed_at_ms(&state);

        // Exactly at the retention boundary the session is still kept.
        run_background_maintenance_at(
            &state,
            closed_at_ms + CLOSED_OWNED_DECODE_SESSION_RETENTION_MS,
        );

        assert!(retained_session_registered(&state));
        let status = retained_session_status(&state).await;
        assert!(status.get("error").is_none(), "status failed: {status}");
        assert_eq!(status["session_id"], RETAINED_SESSION_ID);
        assert_eq!(
            status["state"],
            json!({ "state": "terminal", "terminal_state": "completed" })
        );
        let repeated_close = close_retained_session(&state).await;
        assert_eq!(
            repeated_close["closed"], true,
            "repeated close: {repeated_close}"
        );
        assert_eq!(
            retained_session_closed_at_ms(&state),
            closed_at_ms,
            "a repeated close must not restart the retention window"
        );

        drop(state);
        fs::remove_dir_all(root).expect("remove retention state");
    }

    #[tokio::test]
    async fn maintenance_never_evicts_open_session() {
        let (root, state, catalog) = owned_decode_serving_test_state("keep-open");
        let model_id = certified_model_id(&state, &catalog);
        // Admitted at the very start of the clock and swept at its very end.
        register_completed_session(&state, &catalog, &model_id, 1);

        run_background_maintenance_at(&state, u64::MAX);

        assert!(retained_session_registered(&state));
        let status = retained_session_status(&state).await;
        assert!(status.get("error").is_none(), "status failed: {status}");

        drop(state);
        fs::remove_dir_all(root).expect("remove open-session state");
    }

    #[tokio::test]
    async fn disable_unloads_model_after_its_closed_session_was_evicted() {
        let (root, state, catalog) = owned_decode_serving_test_state("disable-after-evict");
        let model_id = certified_model_id(&state, &catalog);
        register_completed_session(&state, &catalog, &model_id, now_ms());
        assert_eq!(close_retained_session(&state).await["closed"], true);
        let closed_at_ms = retained_session_closed_at_ms(&state);
        run_background_maintenance_at(
            &state,
            closed_at_ms + CLOSED_OWNED_DECODE_SESSION_RETENTION_MS + 1,
        );
        assert!(!retained_session_registered(&state));
        cache_certified_and_unrelated_dispatches(&state, &root, &model_id);

        let outcome = disable_catalog(&state, &catalog).await;

        assert_eq!(outcome["unload_artifact"], true, "disable: {outcome}");
        assert_eq!(
            cached_dispatch_models(&state),
            vec!["unrelated-model".to_string()]
        );

        drop(state);
        fs::remove_dir_all(root).expect("remove disable-after-evict state");
    }

    #[tokio::test]
    async fn disable_unloads_model_no_session_ever_used() {
        let (root, state, catalog) = owned_decode_serving_test_state("disable-no-session");
        let model_id = certified_model_id(&state, &catalog);
        // A dispatch cached by certification before any session was admitted.
        cache_certified_and_unrelated_dispatches(&state, &root, &model_id);

        let outcome = disable_catalog(&state, &catalog).await;

        assert_eq!(outcome["unload_artifact"], true, "disable: {outcome}");
        assert_eq!(
            cached_dispatch_models(&state),
            vec!["unrelated-model".to_string()]
        );

        drop(state);
        fs::remove_dir_all(root).expect("remove disable-no-session state");
    }

    #[tokio::test]
    async fn disable_unloads_model_of_recently_closed_session() {
        let (root, state, catalog) = owned_decode_serving_test_state("disable-recent-closed");
        let model_id = certified_model_id(&state, &catalog);
        register_completed_session(&state, &catalog, &model_id, now_ms());
        assert_eq!(close_retained_session(&state).await["closed"], true);
        cache_certified_and_unrelated_dispatches(&state, &root, &model_id);

        let outcome = disable_catalog(&state, &catalog).await;

        assert_eq!(outcome["unload_artifact"], true, "disable: {outcome}");
        assert_eq!(
            cached_dispatch_models(&state),
            vec!["unrelated-model".to_string()]
        );
        assert!(retained_session_registered(&state));

        drop(state);
        fs::remove_dir_all(root).expect("remove disable-recent-closed state");
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

    #[tokio::test]
    async fn health_reports_certification_staleness_for_rotated_profiles() {
        let (root, descriptor) = test_storage_descriptor("health-certification-staleness");
        let store = Arc::new(SynapseStore::open(&descriptor).expect("test store opens"));
        let profile_a = test_machine_profile("test-os-a");
        store
            .observe_profile(&profile_a, 10, 1)
            .expect("first profile activates");
        let fingerprint = Fingerprint("test-fingerprint".to_string());
        store
            .store_class_scoped_cert_row(&ClassScopedCertificationRow {
                certification_class: CertificationClass::Embedding,
                assurance_class: AssuranceClass::Measured,
                status: CertificationStatus::Certified,
                key_hash: profile_a.hash(),
                machine_profile_hash: Some(profile_a.hash()),
                remote_profile_hash: None,
                identity_revision: None,
                numeric_profile_id: Some(NumericProfileId("test-profile".to_string())),
                fingerprint: fingerprint.clone(),
                certified_at_ms: 11,
                os_build: profile_a.os_build.clone(),
                module_generation: 1,
                evidence: json!({}),
            })
            .expect("current certification stores");

        let healthy_handler = SynapseHandler::new("synapse-test".to_string(), PathBuf::new());
        assert!(healthy_handler
            .inner
            .state
            .set(test_module_state(Arc::clone(&store), profile_a.clone()))
            .is_ok());
        let healthy = healthy_handler.health().await;
        assert!(matches!(healthy.status, subc_client_rs::HealthStatus::Ok));
        let healthy_metrics = healthy.metrics.expect("health has metrics");
        assert_eq!(
            healthy_metrics["certification"]["certification_stale"],
            Value::Bool(false)
        );
        assert_eq!(
            healthy_metrics["certification"]["lanes"],
            json!([{"model_id": "stuck-model", "workload": "embed", "certified": true}])
        );

        let profile_b = test_machine_profile("test-os-b");
        store
            .observe_profile(&profile_b, 12, 1)
            .expect("rotated profile activates");
        assert_eq!(
            store
                .observe_certification_stale(&profile_b.revisioned_hash(), 13)
                .expect("staleness records"),
            Some(13)
        );
        let stale_handler = SynapseHandler::new("synapse-test".to_string(), PathBuf::new());
        assert!(stale_handler
            .inner
            .state
            .set(test_module_state(Arc::clone(&store), profile_b))
            .is_ok());
        let stale = stale_handler.health().await;
        assert!(matches!(stale.status, subc_client_rs::HealthStatus::Ok));
        let stale_metrics = stale.metrics.expect("health has metrics");
        assert_eq!(
            stale_metrics["certification"]["certification_stale"],
            Value::Bool(true)
        );
        assert_eq!(stale_metrics["certification"]["stale_since_ms"], 13);
        assert_eq!(
            stale_metrics["certification"]["lanes"],
            json!([{"model_id": "stuck-model", "workload": "embed", "certified": false}])
        );

        drop(stale_handler);
        drop(healthy_handler);
        drop(store);
        let _ = std::fs::remove_dir_all(root);
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

    #[test]
    fn owned_decode_quarantine_precheck_honours_the_wall_clock() {
        // The routing precheck and the dispatch path in worker_host ask the
        // same predicate. `is_quarantined` answers `now < until`, so a zero
        // clock in the precheck reports every past expiry as still blocking
        // while dispatch sees it cleared. This pins the precheck to a real
        // clock by asserting the two answers agree for an expired record.
        use owned_decode_worker::budget::{
            BudgetPolicy, CrashBudget as OwnedCrashBudget, FileBudgetStore,
        };
        use owned_decode_worker::error::FailureClassification;
        use owned_decode_worker::identity::QuarantineKey;

        let root = std::env::temp_dir().join(format!(
            "synapse-quarantine-clock-{}-{}",
            std::process::id(),
            now_ms()
        ));
        std::fs::create_dir_all(&root).expect("budget directory");
        let store = FileBudgetStore::open(root.join("budget.json")).expect("budget store opens");
        let policy = BudgetPolicy::default();
        let mut budget = OwnedCrashBudget::new(store, policy);
        let key = QuarantineKey::new("profile-hash", "decode-fingerprint", "config-digest");

        // Charge far enough in the past that the quarantine window has closed.
        let charged_at = now_ms()
            .saturating_sub(policy.quarantine_duration_ms)
            .saturating_sub(60_000);
        let mut outcome = None;
        for _ in 0..policy.max_strikes {
            outcome = Some(
                budget
                    .charge(&key, FailureClassification::Crash, charged_at)
                    .expect("charge persists"),
            );
        }
        assert!(
            outcome.expect("at least one charge").quarantined,
            "the budget must be exhausted for this fixture to mean anything"
        );

        assert!(
            !budget.is_quarantined(&key, now_ms()),
            "an expired quarantine must not block under the real clock"
        );
        assert!(
            budget.is_quarantined(&key, 0),
            "a zero clock reports the same expired quarantine as blocking; this \
             is why the precheck must not pass zero"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn admission_status_reports_catalog_certification_health_without_resident_lanes() {
        let (root, descriptor) = test_storage_descriptor("admission-catalog-certification-health");
        let store = Arc::new(SynapseStore::open(&descriptor).expect("test store opens"));
        let profile_a = test_machine_profile("test-os-a");
        store
            .observe_profile(&profile_a, 10, 1)
            .expect("first profile activates");
        let fingerprint = Fingerprint("test-fingerprint".to_string());
        store
            .store_class_scoped_cert_row(&ClassScopedCertificationRow {
                certification_class: CertificationClass::Embedding,
                assurance_class: AssuranceClass::Measured,
                status: CertificationStatus::Certified,
                key_hash: profile_a.hash(),
                machine_profile_hash: Some(profile_a.hash()),
                remote_profile_hash: None,
                identity_revision: None,
                numeric_profile_id: Some(NumericProfileId("test-profile".to_string())),
                fingerprint: fingerprint.clone(),
                certified_at_ms: 11,
                os_build: profile_a.os_build.clone(),
                module_generation: 1,
                evidence: json!({}),
            })
            .expect("current certification stores");
        store
            .store_perf_row(&PerfRow {
                machine_profile_hash: profile_a.hash(),
                model_id: "stuck-model".to_string(),
                workload: "embed".to_string(),
                numeric_profile_id: NumericProfileId("test-profile".to_string()),
                fingerprint,
                engine: "ort".to_string(),
                measured_at_ms: 11,
                os_build: profile_a.os_build.clone(),
                module_generation: 1,
                throughput_tok_s: 1.0,
                cold_load_ms: 1.0,
                single_item_latency_p50_ms: 1.0,
                details: json!({}),
            })
            .expect("current performance row stores");

        let healthy_state = test_module_state(Arc::clone(&store), profile_a);
        assert!(healthy_state.runtime.loaded_models().is_empty());
        let healthy_admission = response_result(
            admission_status(Arc::clone(&healthy_state)).await,
            "admission.status",
        );
        let healthy_probe = response_result(
            probe_report(Arc::clone(&healthy_state)).await,
            "probe.report",
        );
        assert_eq!(healthy_admission["lanes"], json!([]));
        assert_eq!(healthy_admission["catalog_lanes"], 1);
        assert_eq!(healthy_admission["certified_lanes"], 1);
        assert_eq!(healthy_admission["certification_stale"], json!(false));
        assert_eq!(
            healthy_admission["certification_stale"],
            healthy_probe["certification_stale"]
        );
        assert_eq!(
            healthy_admission["performance_stale"],
            healthy_probe["performance_stale"]
        );

        let profile_b = test_machine_profile("test-os-b");
        store
            .observe_profile(&profile_b, 12, 1)
            .expect("rotated profile activates");
        let stale_state = test_module_state(Arc::clone(&store), profile_b);
        assert!(stale_state.runtime.loaded_models().is_empty());
        let stale_admission = response_result(
            admission_status(Arc::clone(&stale_state)).await,
            "admission.status",
        );
        let stale_probe =
            response_result(probe_report(Arc::clone(&stale_state)).await, "probe.report");
        assert_eq!(stale_admission["lanes"], json!([]));
        assert_eq!(stale_admission["catalog_lanes"], 1);
        assert_eq!(stale_admission["certified_lanes"], 0);
        assert_eq!(stale_admission["certification_stale"], json!(true));
        assert_eq!(stale_admission["performance_stale"], json!(true));
        assert_eq!(
            stale_admission["certification_stale"],
            stale_probe["certification_stale"]
        );
        assert_eq!(
            stale_admission["performance_stale"],
            stale_probe["performance_stale"]
        );

        drop(stale_state);
        drop(healthy_state);
        drop(store);
        let _ = std::fs::remove_dir_all(root);
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
            .store
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
        let meta = model_cache
            .ingest(synapse_core::ModelCacheIngest {
                source_url: format!("file://{}", blob_source.display()),
                expected_digest: None,
                format: "bin".to_string(),
                tokenizer_path: None,
                pin_module_id: None,
            })
            .expect("ingest test artifact into model cache");
        let materialized =
            ane_artifact::materialized_entry_path(&cache_root, &meta.digest).unwrap();
        std::fs::create_dir_all(materialized.join("artifact")).unwrap();
        std::fs::write(
            materialized.join("artifact/weights.bin"),
            b"derived Core ML bytes",
        )
        .unwrap();

        let (machine_profile_hash, revisioned_machine_profile_hash) =
            module_state_machine_profile_hashes(&profile);
        let profile_activation_epoch = store
            .profile_state()
            .expect("test profile state reads")
            .profile_activation_epoch
            .expect("test profile is activated");
        let module_config = ModuleConfig {
            cache_max_bytes: std::fs::metadata(model_cache.blob_path(&meta.digest))
                .unwrap()
                .len(),
            ..ModuleConfig::default()
        };
        let runtime = Arc::new(
            RuntimeState::from_catalog(module_config, vec![stuck_model_spec()])
                .expect("test runtime initializes"),
        );
        let remote_gateway = Arc::new(
            RemoteGateway::new(
                Arc::clone(&store),
                Vec::new(),
                Arc::new(SubcVaultCredentialClient::new(PathBuf::from("unused-subc"))),
                machine_profile_hash.clone(),
            )
            .expect("empty remote gateway initializes"),
        );
        let continuity_check: Arc<dyn ContinuityCheck> = remote_gateway.continuity.clone();
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
            continuity_check,
            remote_gateway,
        });

        // The source blob alone is exactly at the watermark, so this sweep runs
        // only when the derivative is included in the cache budget.
        let vacuum_count_before = maintenance_vacuum_count();
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
        assert!(
            materialized.is_dir(),
            "marking a source must retain its Core ML derivative during grace"
        );
        // No delete occurred, so store freelist remains un-vacuumed.
        assert_eq!(store.freelist_count().unwrap(), freelist_before);
        assert_eq!(maintenance_vacuum_count(), vacuum_count_before);

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
        assert!(
            !materialized.exists(),
            "deleting a source must reclaim its Core ML derivative"
        );

        // The GC sweep actually deleted tombstoned blobs, triggering guarded freelist reclaim!
        let freelist_after = store.freelist_count().unwrap();
        let file_size_after = std::fs::metadata(&db_path).unwrap().len();
        let vacuum_count_after = maintenance_vacuum_count();

        assert_eq!(freelist_after, 0);
        assert!(
            file_size_after < file_size_before / 10,
            "expected file to shrink: before={file_size_before}, after={file_size_after}"
        );
        assert_eq!(vacuum_count_after, vacuum_count_before + 1);

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
        let remote_gateway = Arc::new(
            RemoteGateway::new(
                Arc::clone(&store),
                Vec::new(),
                Arc::new(SubcVaultCredentialClient::new(PathBuf::from("unused-subc"))),
                machine_profile_hash.clone(),
            )
            .expect("empty remote gateway initializes"),
        );
        let continuity_check: Arc<dyn ContinuityCheck> = remote_gateway.continuity.clone();
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
            continuity_check,
            remote_gateway,
        });
        (state, model_cache, root, cache_root)
    }

    fn ingest_cache_gc_blob(model_cache: &ModelCache, name: &str, payload: &[u8]) -> String {
        let source = model_cache.root().join(name);
        std::fs::write(&source, payload).unwrap();
        model_cache
            .ingest(synapse_core::ModelCacheIngest {
                source_url: format!("file://{}", source.display()),
                expected_digest: None,
                format: "bin".to_string(),
                tokenizer_path: None,
                pin_module_id: None,
            })
            .expect("ingest test artifact into model cache")
            .digest
    }

    /// A published Q8 artifact directory as `derive_and_cache_q8_blocks` leaves
    /// it: an object file and a lineage sidecar naming its source digest.
    fn write_q8_artifact(
        cache_root: &Path,
        key: &str,
        source_digest: &str,
        bytes: usize,
    ) -> PathBuf {
        let dir = cache_root
            .join(owned_decode_routing::q8artifact::CACHE_DIRECTORY)
            .join(key);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("weights.q8_0"), vec![7_u8; bytes]).unwrap();
        std::fs::write(
            dir.join("lineage.json"),
            serde_json::to_vec(&json!({ "source_manifest_digest": source_digest })).unwrap(),
        )
        .unwrap();
        dir
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
    async fn cache_gc_source_deletion_reclaims_its_q8_artifact_only() {
        let (state, model_cache, root, cache_root) = cache_gc_test_state("gc-q8-source", u64::MAX);
        let deleted = ingest_cache_gc_blob(&model_cache, "deleted.bin", b"deleted source");
        let kept = ingest_cache_gc_blob(&model_cache, "kept.bin", b"kept source");
        let deleted_q8 = write_q8_artifact(&cache_root, &"1".repeat(64), &deleted, 16);
        let kept_q8 = write_q8_artifact(&cache_root, &"2".repeat(64), &kept, 16);

        for grace_ms in [60_000, 0] {
            run_cache_gc(&state, json!({ "digest": deleted, "grace_ms": grace_ms })).await;
        }

        assert!(!model_cache.blob_path(&deleted).exists());
        assert!(
            !deleted_q8.exists(),
            "deleting a source must reclaim its Q8 derivative"
        );
        assert!(
            kept_q8.join("weights.q8_0").is_file(),
            "a Q8 artifact of another source must survive"
        );
        drop(state);
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(cache_root);
    }

    #[tokio::test]
    async fn cache_gc_watermark_counts_q8_artifact_bytes() {
        // The blob alone is exactly at the budget; only the Q8 derivative
        // pushes the combined cache over it.
        let payload = b"q8 watermark source blob";
        let (state, model_cache, root, cache_root) =
            cache_gc_test_state("gc-q8-watermark", payload.len() as u64);
        let digest = ingest_cache_gc_blob(&model_cache, "source.bin", payload);
        let q8 = write_q8_artifact(&cache_root, &"3".repeat(64), &digest, 64);

        let first = run_cache_gc(&state, json!({ "grace_ms": 60_000 })).await;
        assert_eq!(
            first["outcomes"][0]["state"], "marked",
            "Q8 bytes must count toward the cache budget: {first}"
        );
        let second = run_cache_gc(&state, json!({ "grace_ms": 0 })).await;
        assert_eq!(second["outcomes"][0]["state"], "deleted");
        assert!(!q8.exists());
        drop(state);
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(cache_root);
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
    #[tokio::test]
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

        let start = Instant::now();
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

        let zero_start = Instant::now();
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

    #[tokio::test]
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
        let start = Instant::now();
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

        let zero_start = Instant::now();
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
        assert_eq!(
            worker_binary_file_name("owned-metal-decode"),
            Some("ck-synapse-worker-decode")
        );
        assert_eq!(worker_binary_file_name("ort"), None);
        assert_eq!(worker_binary_file_name("unknown-engine"), None);
    }

    #[test]
    fn embed_probe_fixtures_cover_every_owned_family_with_reference_identity() {
        let fixtures = probe_fixtures().expect("shipped embed fixtures should parse");
        assert_eq!(fixtures.len(), 3);
        let identities = fixtures
            .iter()
            .map(|fixture| {
                (
                    fixture.family.clone().unwrap_or_default(),
                    fixture.reference_model.clone().unwrap_or_default(),
                    fixture_reference_dims(fixture),
                )
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(
            identities,
            BTreeSet::from([
                ("minilm".to_string(), "minilm".to_string(), Some(384)),
                (
                    "gte-modernbert".to_string(),
                    "gte-modernbert-base".to_string(),
                    Some(768)
                ),
                (
                    "qwen3-0.6b".to_string(),
                    "qwen3-embedding-0.6b".to_string(),
                    Some(1024)
                ),
            ])
        );
        // The owned-CUDA Qwen3 identity resolves to the Qwen3 fixture set; a
        // missing set here is exactly the `reference_fixture_missing`
        // certification dead-end this fixture closes.
        let qwen3 = fixtures
            .iter()
            .find(|fixture| fixture.family.as_deref() == Some("qwen3-0.6b"))
            .expect("qwen3 fixture present");
        assert_eq!(qwen3.items.len(), 64);
        assert!(qwen3.items.iter().all(|item| item.vector.len() == 1024));
    }

    #[test]
    fn decode_certification_fixtures_are_the_pinned_twenty_by_sixty_four_oracles() {
        let fixtures = generate_probe_fixtures().expect("shipped decode fixtures should parse");
        assert_eq!(fixtures.len(), 4);
        let lanes = fixtures
            .iter()
            .map(|fixture| (fixture.family.as_str(), fixture.quant.as_str()))
            .collect::<BTreeSet<_>>();
        assert_eq!(
            lanes,
            BTreeSet::from([
                ("qwen3-0.6b", "f16"),
                ("qwen3-0.6b", "q8_0"),
                ("lfm2-1.2b", "f16"),
                ("lfm2-1.2b", "q8_0"),
            ])
        );
        for fixture in &fixtures {
            assert_eq!(fixture.dtype, "f16");
            assert_eq!(fixture.items.len(), 20);
            assert!(fixture
                .items
                .iter()
                .all(|item| { item.max_new_tokens == 64 && item.expected_token_ids.len() <= 64 }));
            let command = fixture.generation_command.as_deref().unwrap();
            assert_eq!(
                fixture.generation_command_sha256,
                sha256_hex(command.as_bytes())
            );
            assert_eq!(
                fixture.provenance["generation_command_sha256"],
                fixture.generation_command_sha256
            );
        }
        assert_eq!(
            fixtures
                .iter()
                .find(|fixture| fixture.family == "qwen3-0.6b" && fixture.quant == "f16")
                .unwrap()
                .provenance["source_tokens_sha256"],
            "c3080813c45c364a73cbb6dce122afbba20e761b2189a31d0055ecf435232af1"
        );
        assert_eq!(
            fixtures
                .iter()
                .find(|fixture| fixture.family == "lfm2-1.2b" && fixture.quant == "f16")
                .unwrap()
                .structural_band
                .max_forks,
            2
        );
        assert!(fixtures
            .iter()
            .filter(|fixture| fixture.quant == "q8_0")
            .all(|fixture| fixture.structural_band.max_forks == 0));
    }

    #[test]
    fn structural_band_accepts_only_a_pinned_top_two_fork() {
        let fixtures = generate_probe_fixtures().expect("shipped decode fixtures should parse");
        let fixture = fixtures
            .iter()
            .find(|fixture| fixture.family == "lfm2-1.2b" && fixture.quant == "f16")
            .unwrap();
        let item = fixture
            .items
            .iter()
            .find(|item| item.id == "completion-15")
            .unwrap();
        let mut accepted = item.expected_token_ids.clone();
        accepted[17] = 523;
        assert!(certified_generate_fork(item, &accepted, &fixture.structural_band).is_some());
        accepted[17] = 524;
        assert!(certified_generate_fork(item, &accepted, &fixture.structural_band).is_none());
    }

    #[test]
    fn corrupted_decode_fixture_records_the_first_diverging_prompt_and_token() {
        let fixtures = generate_probe_fixtures().expect("shipped decode fixtures should parse");
        let item = &fixtures[0].items[0];
        let mut corrupted = item.expected_token_ids.clone();
        corrupted[7] = corrupted[7].wrapping_add(1);

        let mismatch = decode_token_mismatch(item, &corrupted);
        assert_eq!(mismatch["id"], item.id);
        assert_eq!(mismatch["prompt"], item.prompt);
        assert_eq!(mismatch["divergence_token_index"], 7);
        assert_eq!(mismatch["expected_token_id"], item.expected_token_ids[7]);
        assert_eq!(mismatch["actual_token_id"], corrupted[7]);
    }

    #[test]
    fn worker_path_certification_accepts_exact_and_structural_batteries() {
        let evidence = |battery| {
            json!({
                "worker_path": {
                    "transport": worker_catalog_transport(),
                    "protocol": owned_decode_worker::identity::WORKER_PROTOCOL_ID,
                    "fixture_battery": battery,
                }
            })
        };
        assert!(worker_path_certification(&evidence("20x64-token-exact")));
        assert!(worker_path_certification(&evidence(
            "20x64-structural-band"
        )));
        assert!(!worker_path_certification(&evidence("unverified")));
    }

    #[test]
    fn only_owned_microllm_lanes_require_decode_certification() {
        assert!(engine_requires_microllm_certification("owned-metal-decode"));
        assert!(!engine_requires_microllm_certification("owned-metal"));
        assert!(!engine_requires_microllm_certification("llama"));
    }

    #[test]
    fn simulated_non_macos_owned_decode_resolution_routes_through_lane_selection() {
        use owned_decode_routing::{
            error::OwnedDecodeError,
            family::Family,
            identity::WeightQuant,
            lane::{
                select_lane, FallbackReason, LaneOutcome, LaneSelectionContext, LlamaLane,
                OwnedEvaluation,
            },
            request::{OneshotRequest, SamplingMode},
        };

        let refusal = owned_decode_resolution_refusal_for_platform("owned-metal-decode", false);
        assert_eq!(refusal, Some(OwnedDecodeError::Unsupported));
        assert_eq!(
            owned_decode_resolution_refusal_for_platform("owned-metal-decode", true),
            None
        );
        let request = OneshotRequest {
            family: Family::Qwen3_0_6b,
            weight_quant: WeightQuant::F16,
            prompt_token_count: 1,
            max_tokens: 1,
            sampling: SamplingMode::GreedyTop1,
            grammar: None,
            required_fingerprint: None,
            allow_equivalent: false,
            target_fingerprint: None,
            required_processing_fingerprint: None,
            owned_only: false,
        };
        let outcome = select_lane(&LaneSelectionContext {
            request: &request,
            owned_decode_fingerprint: Fingerprint("owned-decode".to_string()),
            owned_processing_fingerprint: Fingerprint("owned-processing".to_string()),
            owned: OwnedEvaluation::Refused(refusal.expect("unsupported platform refusal")),
            llama: Some(LlamaLane {
                decode_fingerprint: Fingerprint("llama-decode".to_string()),
                processing_fingerprint: Fingerprint("llama-processing".to_string()),
            }),
            equivalent_fingerprints: BTreeSet::new(),
        });

        assert_eq!(
            outcome,
            LaneOutcome::Llama {
                fallback_reason: FallbackReason::OwnedRefusal(OwnedDecodeError::Unsupported),
            }
        );
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
    fn concurrent_model_load_jobs_use_distinct_scratch_paths() {
        assert_ne!(
            model_load_scratch_path("job_first"),
            model_load_scratch_path("job_second")
        );
    }

    #[test]
    fn second_decode_request_extends_the_absolute_session_boundary() {
        assert_eq!(absolute_serving_boundary(11, 4), 15);
    }

    #[test]
    fn model_load_scratch_removes_downloads_when_scope_exits() {
        let path = env::temp_dir().join(format!(
            "synapse-model-load-scratch-test-{}-{}",
            std::process::id(),
            now_ms()
        ));
        {
            let scratch =
                ModelLoadScratch::create(path.clone()).expect("create model-load scratch");
            fs::write(scratch.path().join("model.bin"), b"downloaded model")
                .expect("write scratch download");
        }

        assert!(!path.exists(), "scratch directory survived guard drop");
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
        // The HOME arm joins path segments, so on Windows the separators come out
        // mixed; compare components rather than the rendered string.
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
    fn project_tier_rejects_decode_chain_k_with_security_boundary_error() {
        let error =
            parse_module_config_json(r#"{"decode_chain_k": 8}"#, "project", ConfigTier::Project)
                .expect_err("project tier must not control machine decode shape");
        assert_eq!(
            error.to_string(),
            "config: decode_chain_k is user-tier only and may not appear in project-tier config"
        );
    }

    #[test]
    fn project_tier_rejects_remote_providers_with_security_boundary_error() {
        let error = parse_module_config_json(
            r#"{"remote_providers": []}"#,
            "project",
            ConfigTier::Project,
        )
        .expect_err("project tier must not control remote credentials or endpoints");
        assert_eq!(
            error.to_string(),
            "config: remote_providers is user-tier only and may not appear in project-tier config"
        );
    }

    #[test]
    fn user_tier_parses_declared_remote_provider() {
        let config = parse_module_config_json(
            r#"{
                "remote_providers": [{
                    "name": "mock",
                    "base_url": "http://127.0.0.1:8080/v1",
                    "adapter": {"kind": "openai_compatible"},
                    "auth": {"kind": "none"},
                    "models": [{
                        "synapse_model_id": "remote-embed",
                        "task": "embed",
                        "model": "mock-embed",
                        "identity_revision": "r1",
                        "dims": 3,
                        "input_profile_id": "whitespace-v1"
                    }]
                }]
            }"#,
            "user",
            ConfigTier::User,
        )
        .expect("user tier may configure remote providers");
        assert_eq!(config.remote_providers.len(), 1);
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
    fn module_config_parses_decode_chain_k_with_safe_default_and_bounds() {
        let default = parse_module_config_json(r#"{}"#, "test", ConfigTier::User)
            .expect("default chain span should parse");
        assert_eq!(default.decode_chain_k, DEFAULT_DECODE_CHAIN_K);

        let configured =
            parse_module_config_json(r#"{"decode_chain_k": 16}"#, "test", ConfigTier::User)
                .expect("maximum chain span should parse");
        assert_eq!(configured.decode_chain_k, 16);

        for value in [0, 17] {
            let error = parse_module_config_json(
                &format!(r#"{{"decode_chain_k": {value}}}"#),
                "test",
                ConfigTier::User,
            )
            .expect_err("chain span outside 1..=16 must fail");
            assert!(error.to_string().contains("decode_chain_k"));
        }
    }

    #[test]
    fn module_config_parses_microllm_and_cache_fields() {
        let config = parse_module_config_json(
            r#"{
                "microllm_max_tokens": 128,
                "grammar_enabled": true,
                "cache_max_bytes": 4096
            }"#,
            "test",
            ConfigTier::User,
        )
        .expect("valid config");
        assert_eq!(config.microllm_max_tokens, 128);
        assert!(config.grammar_enabled);
        assert_eq!(config.cache_max_bytes, 4096);
    }

    #[test]
    fn job_config_parses_split_ttls_and_rejects_legacy_alias() {
        let split = parse_module_config_json(
            r#"{
                "jobs": {
                    "execution_ttl_ms": 10,
                    "result_retention_ttl_ms": 20,
                    "resume_deadline_ms": 30
                }
            }"#,
            "test",
            ConfigTier::User,
        )
        .unwrap();
        assert_eq!(split.jobs.execution_ttl_ms, 10);
        assert_eq!(split.jobs.result_retention_ttl_ms, 20);
        assert_eq!(split.jobs.resume_deadline_ms, 30);

        // Pre-release rename, no compatibility surface: the old key must fail
        // loudly (deny_unknown_fields) instead of being silently accepted.
        let legacy =
            parse_module_config_json(r#"{"jobs":{"ttl_ms":40}}"#, "test", ConfigTier::User);
        assert!(legacy.is_err(), "legacy ttl_ms key must be rejected");
    }

    #[test]
    fn request_digest_is_canonical_and_binds_order_content_and_remote_identity() {
        let items = vec![
            ("a".to_string(), sha256_hex(b"first")),
            ("b".to_string(), sha256_hex(b"second")),
        ];
        let constraints_a: Value = serde_json::from_str(r#"{"z":1,"a":true}"#).unwrap();
        let constraints_b: Value = serde_json::from_str(r#"{"a":true,"z":1}"#).unwrap();
        let local =
            compute_request_digest("embed.batch", "model", None, None, &constraints_a, &items);
        assert_eq!(
            local,
            compute_request_digest("embed.batch", "model", None, None, &constraints_b, &items,)
        );
        let mut reordered = items.clone();
        reordered.reverse();
        assert_ne!(
            local,
            compute_request_digest(
                "embed.batch",
                "model",
                None,
                None,
                &constraints_a,
                &reordered,
            )
        );
        assert_ne!(
            local,
            compute_request_digest(
                "embed.batch",
                "model",
                Some("remote-profile"),
                Some("vault/provider"),
                &constraints_a,
                &items,
            )
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
    fn ane_catalog_identity_carries_the_neural_engine_placement_gate() {
        let ane = catalog_model_engine_identity("ane").unwrap();
        assert_eq!(ane.engine, "ane-coreml-worker");
        assert_eq!(
            ane.build_flags.get("placement_gate").map(String::as_str),
            Some("neural-engine")
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

        let ane = recommended_batch_for_engine("ane", 512).unwrap();
        assert_eq!(ane.rows, MAX_ENGINE_BATCH_ITEMS);
        assert_eq!(ane.token_budget, 512 * MAX_ENGINE_BATCH_ITEMS as u64);

        assert!(recommended_batch_for_engine("ort", 512).is_none());
        assert!(recommended_batch_for_engine(LLAMA_ENGINE, 512).is_none());
    }

    #[test]
    fn owned_rerank_catalog_acceptance_uses_a_distinct_processing_identity() {
        let owned = || OwnedCatalogConfig {
            family: OwnedFamily::GteModernBert,
            dtype: OwnedDType::F32,
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

    #[test]
    fn knob_assignments_prefer_throughput_but_allow_quiet_when_within_ratio() {
        let rows = vec![
            PerfRow {
                machine_profile_hash: "machine-a".to_string(),
                model_id: "metal-fast".to_string(),
                workload: "embed".to_string(),
                numeric_profile_id: NumericProfileId("np-fast".to_string()),
                fingerprint: Fingerprint("fp-fast".to_string()),
                engine: "owned-metal".to_string(),
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
                engine: ANE_WORKER_ENGINE.to_string(),
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

    #[test]
    fn owned_decode_worker_result_preserves_raw_ids_and_terminal_identity() {
        let response = json!({
            "result": {
                "provenance": {
                    "lane": "decode",
                    "worker": "supervised",
                    "decode_fingerprint": "decode-fingerprint",
                    "processing_fingerprint": "processing-fingerprint",
                    "worker_generation": 9,
                },
                "generated_token_ids": [17, 23],
                "n_gen": 2,
                "runtime_config_digest": "runtime-digest",
                "derived_digest": "derived-digest",
                "generation_id": "generation-9",
            }
        });
        let stream = parse_owned_decode_worker_stream(
            &serde_json::to_vec(&response).expect("test response serializes"),
        )
        .expect("supervised owned response should map to a stream identity");

        assert_eq!(stream.generated_token_ids, vec![17, 23]);
        assert_eq!(stream.identity.decode_fingerprint.0, "decode-fingerprint");
        assert_eq!(
            stream.identity.processing_fingerprint.0,
            "processing-fingerprint"
        );
        assert_eq!(stream.identity.runtime_config_digest, "runtime-digest");
        assert_eq!(stream.identity.worker_generation, 9);
        assert_eq!(
            stream.identity.derived_digest.as_deref(),
            Some("derived-digest")
        );
        assert_eq!(stream.generation_id, "generation-9");
    }

    fn catalog_fixture_config(id: &str) -> StoredModelConfig {
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
        for (id, profile) in manifest["profiles"].as_object().unwrap() {
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
                &[],
                Path::new("cache"),
                64,
                None,
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
                "3a0b02613e9f7cd7b502de500e28613df43025b59df2cd4b0ec604de42e5a2e2",
                "7808303bba4061bbd0ee205c22a0a53d039c12b61a380dc7d2c2bf6f240a0934",
                "a006367b6adb645e44c82c9a56e973f3423fb4f9be2461ff80a38df68aa9c543",
                "510721ab2b99d667e5461764ac3c129e0244b4f1b32370a269608aa8e56bba20",
                "bdab90b6bb1d81696d679d3c267c95f90de418fa501a3824f42e0c3c217fe166",
                "0e6fbb6174c877f4aead2478b182da31106f10a70019afbb0662ab572c1b8258",
                "df0b8ccadc3696d8e08cca52fda0236fc6a33a59134187c52517291ea94ac902",
                "896eb823c1ee5d9c31b42a12163684ab6b707968a619a7c0196775cc394f4148",
                "6553a78dce363808f967655b534db0c1536d837de3dcfba3a30e142a3cb847e9",
                "fb2d8058714b0b766e5c6847344a3b1651091730388fa38966c7c67a2b88eb13",
                "6e583acd1cacaf47a1c02873c52441f3b62417ed086fc88346d4d08a8389fbc0",
                "2c00ee65e9ae43600c62332a0d96f799b790c03339cf43ad286eaaca0f8fd6bd",
                "16ca0b783714705ca01235bfa7fe3a08da2615abc87cf081a1f0c7041edc2d5a",
                "fb2a4fcb16f3fe2deaf26c0ed11568b1e1c83db5523a666ad1c497ee1580cfbd",
            ];
            let index = manifest["profiles"]
                .as_object()
                .unwrap()
                .keys()
                .position(|key| key == id)
                .unwrap();
            assert_eq!(spec.fingerprint.0, expected[index], "{id}");
        }
    }

    #[test]
    fn preload_fingerprints_rebuild_from_committed_inputs() {
        for bytes in [
            include_bytes!("../../../bench/parity/preload/gte-modernbert-base-f16.json").as_slice(),
            include_bytes!("../../../bench/parity/preload/gte-reranker-modernbert-base-f32.json")
                .as_slice(),
        ] {
            let fixture: Value = serde_json::from_slice(bytes).unwrap();
            let inline: InlineConfig = serde_json::from_value(fixture["inline"].clone()).unwrap();
            let jobs: JobConfig = serde_json::from_value(fixture["jobs"].clone()).unwrap();
            let owned = OwnedCatalogConfig {
                family: OwnedFamily::parse(fixture["owned_family"].as_str().unwrap()).unwrap(),
                dtype: OwnedDType::parse(fixture["owned_dtype"].as_str().unwrap()).unwrap(),
                execution: "explicit".into(),
                attention_units: OWNED_DEFAULT_ATTENTION_UNITS,
                config_locator: None,
                extra_locators: Vec::new(),
                identity_override: None,
            };
            let spec = build_stored_model_config(
                fixture["model_id"].as_str().unwrap().into(),
                "owned-metal",
                parse_model_task(fixture["task"].as_str(), "owned-metal", "fixture").unwrap(),
                fixture["artifact_digest"].as_str().unwrap().into(),
                "safetensors".into(),
                fixture["sanitized_tokenizer_digest"]
                    .as_str()
                    .unwrap()
                    .into(),
                ModelAssetLocator::LocalPath {
                    path: "unused".into(),
                },
                ModelAssetLocator::LocalPath {
                    path: "unused".into(),
                },
                String::new(),
                String::new(),
                parse_pooling(fixture["pooling"].as_str().unwrap()).unwrap(),
                fixture["normalize"].as_bool().unwrap(),
                fixture["max_tokens"].as_u64().unwrap() as usize,
                fixture["quant"].as_str().unwrap().into(),
                false,
                None,
                None,
                Vec::new(),
                Some(owned),
                &inline,
                &jobs,
            )
            .unwrap();
            let expected = fixture["expected_fingerprint"]
                .as_str()
                .expect("preload fixture must pin expected_fingerprint");
            assert_eq!(spec.fingerprint.0, expected);
            assert!(!spec.engine_identity.build_flags.contains_key("profile"));
        }
    }

    #[test]
    fn preload_profile_key_builds_every_catalog_lane_without_config_json() {
        let (dir, _) = test_storage_descriptor("catalog-preload");
        let manifest: Value =
            serde_json::from_slice(include_bytes!("../../../bench/parity/models.json")).unwrap();
        for id in manifest["profiles"].as_object().unwrap().keys() {
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

    fn catalog_test_model(dir: &Path, spec: &StoredModelConfig) -> Arc<EmbeddingModel> {
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
            numeric_profile_id: spec.numeric_profile_id.clone(),
            fingerprint: spec.fingerprint.clone(),
            certification_fingerprint: spec.fingerprint.clone(),
            engine_identity: spec.engine_identity.clone(),
            owned_tokenizer_policy: None,
            owned_decode_resolution_refusal: None,
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
        store
            .store_class_scoped_cert_row(&ClassScopedCertificationRow {
                certification_class: if spec.task == "embed" {
                    CertificationClass::Embedding
                } else {
                    CertificationClass::Rerank
                },
                assurance_class: AssuranceClass::Measured,
                status: CertificationStatus::Certified,
                key_hash: state.machine_profile_hash.clone(),
                machine_profile_hash: Some(state.machine_profile_hash.clone()),
                remote_profile_hash: None,
                identity_revision: None,
                numeric_profile_id: Some(spec.numeric_profile_id.clone()),
                fingerprint: spec.fingerprint.clone(),
                certified_at_ms: 11,
                os_build: profile.os_build,
                module_generation: 1,
                evidence: json!({}),
            })
            .unwrap();
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
        let snapshot = state.store.catalog_snapshot().unwrap();
        let listed = models_list_payload(&state, snapshot);
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
    fn catalog_aliases_cannot_substitute_another_backend() {
        let (dir, _) = test_storage_descriptor("catalog-aliases");
        for (profile, foreign) in [
            (
                "gte-modernbert-base.ane-direct-worker",
                "coreml-fingerprint",
            ),
            ("gte-modernbert-base.owned-cuda", "metal-fingerprint"),
        ] {
            let spec = catalog_fixture_config(profile);
            let model = catalog_test_model(&dir, &spec);
            let alias = AliasTable {
                table_epoch: 1,
                rows: vec![synapse_core::AliasRow::with_evidence(
                    spec.fingerprint.clone(),
                    Fingerprint(foreign.into()),
                    0,
                    None,
                    json!({}),
                )],
            };
            assert!(equivalent_fingerprints(&alias, &model)
                .iter()
                .any(|fp| fp.0 == foreign));
            assert_eq!(
                check_fingerprint_constraints(&model, &alias, Some(foreign), None, true, None)
                    .unwrap_err()
                    .code,
                "substitution_rejected"
            );
            check_fingerprint_constraints(
                &model,
                &alias,
                Some(&spec.fingerprint.0),
                None,
                true,
                None,
            )
            .unwrap();
        }
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

    #[tokio::test(flavor = "current_thread")]
    async fn cuda_certification_evidence_uses_model_worker_without_blocking() {
        let (storage_dir, descriptor) = test_storage_descriptor("cuda-evidence");
        let store = Arc::new(SynapseStore::open(&descriptor).expect("open test store"));
        let profile = test_machine_profile("test-os");
        store
            .activate_profile(&profile, 1, 1000)
            .expect("activate profile");
        let state = test_module_state(store, profile);
        let worker = storage_dir.join(if cfg!(windows) {
            "probe.cmd"
        } else {
            "probe.sh"
        });
        let output = r#"{"driver_api":13030,"compute_capability":{"major":8,"minor":9}}"#;
        let script = if cfg!(windows) {
            format!("@echo off\r\nping -n 3 127.0.0.1 >nul\r\necho {output}\r\n")
        } else {
            format!("#!/bin/sh\nsleep 2\nprintf '%s\\n' '{output}'\n")
        };
        fs::write(&worker, script).expect("write probe");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&worker, fs::Permissions::from_mode(0o700)).unwrap();
        }
        let mut spec = stuck_model_spec();
        spec.engine_identity.engine = CUDA_WORKER_ENGINE.to_string();
        spec.worker_bin = Some(worker);
        state
            .runtime
            .catalog
            .lock()
            .unwrap()
            .get_mut(&spec.model_id)
            .unwrap()
            .spec = spec.clone();
        // This test exercises evidence collection only; no engine inference occurs.
        let mut engine = OwnedMetalEmbedEngine::new(OwnedFamily::MiniLm, OwnedDType::F16);
        let loaded_model = engine.insert_test_model("evidence-test".to_string(), 384, vec![128]);
        let model = EmbeddingModel {
            model_id: spec.model_id,
            task: ModelTask::Embed,
            loaded_model,
            backend: EmbedBackend::Owned(Arc::new(Mutex::new(engine))),
            tokenizer: make_test_tokenizer(&storage_dir, 128),
            numeric_profile_id: spec.numeric_profile_id,
            fingerprint: spec.fingerprint.clone(),
            certification_fingerprint: spec.fingerprint,
            engine_identity: spec.engine_identity,
            owned_tokenizer_policy: None,
            owned_decode_resolution_refusal: None,
        };
        let evidence = owned_cuda_evidence(&state, &model);
        tokio::pin!(evidence);
        // A blocking probe on this single-thread executor would complete before
        // the timer can run. The model-specific probe must instead remain pending.
        tokio::select! {
            biased;
            result = &mut evidence => panic!("probe completed before executor progressed: {result:?}"),
            _ = tokio::time::sleep(Duration::from_millis(100)) => {}
        }
        let evidence = evidence
            .await
            .expect("evidence task")
            .expect("CUDA evidence");
        assert_eq!(evidence["floor_state"], "supported");
        assert_eq!(evidence["observed"]["driver_api"], 13030);
        // Windows keeps the tokenizer file mapped while the process runs, so
        // scratch removal is best effort here; assertions above are the contract.
        let _ = fs::remove_dir_all(storage_dir);
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

        let mut metal_spec = stuck_model_spec();
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
            numeric_profile_id: metal_spec.numeric_profile_id.clone(),
            fingerprint: metal_spec.fingerprint.clone(),
            certification_fingerprint: metal_spec.fingerprint.clone(),
            engine_identity: metal_spec.engine_identity.clone(),
            owned_tokenizer_policy: None,
            owned_decode_resolution_refusal: None,
        });

        state
            .store
            .store
            .with_conn(|conn| {
                conn.execute(
                    "INSERT INTO cert_rows (
                        certification_class, assurance_class, status, key_hash,
                        machine_profile_hash, remote_profile_hash, identity_revision,
                        numeric_profile_id, fingerprint, certified_at_ms, os_build,
                        module_generation, evidence_json
                     ) VALUES (?1, 'measured', 'certified', ?2, ?2, NULL, NULL, 'test-profile', ?3, 1000, 'test-os', 1, '{}')",
                    rusqlite::params!["embedding", state.machine_profile_hash, &metal_spec.fingerprint.0],
                )?;
                Ok(())
            })
            .expect("insert cert row for metal lane");

        let mut ane_spec = stuck_model_spec();
        ane_spec.model_id = "test-ane-lane".to_string();
        ane_spec.engine = "ane".to_string();
        ane_spec.task = "embed".to_string();
        // Catalog config carries 4096 (the batch budget drift defect) to verify
        // the published ceiling prefers the worker's real loaded ceiling (512).
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
            Some(vec![128, 256, 512]),
        );
        let ane_model = Arc::new(EmbeddingModel {
            model_id: ane_spec.model_id.clone(),
            task: ModelTask::Embed,
            loaded_model: ane_loaded,
            backend: EmbedBackend::Worker(Arc::new(Mutex::new(worker_engine))),
            tokenizer: make_test_tokenizer(&storage_dir, 512),
            numeric_profile_id: ane_spec.numeric_profile_id.clone(),
            fingerprint: ane_spec.fingerprint.clone(),
            certification_fingerprint: ane_spec.fingerprint.clone(),
            engine_identity: ane_spec.engine_identity.clone(),
            owned_tokenizer_policy: None,
            owned_decode_resolution_refusal: None,
        });

        let mut unloaded_spec = stuck_model_spec();
        unloaded_spec.model_id = "test-unloaded-lane".to_string();
        unloaded_spec.engine = "ane".to_string();
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

        let snapshot = state
            .store
            .catalog_snapshot()
            .expect("catalog snapshot reads");
        let payload = models_list_payload(&state, snapshot);
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
        assert_eq!(metal_row["certified"], true);
        assert_eq!(metal_row["warm_load_cost_hint_ms"], 18.5);
        assert_eq!(metal_row["recommended_batch"]["rows"], 8);
        // The Metal engine only reports live shapes where it exists. Off macOS
        // it holds no model, so the row falls back to catalog config: asserting
        // the bucket-derived values everywhere would assert a fiction.
        if cfg!(target_os = "macos") {
            assert_eq!(metal_row["max_tokens"], 512);
            assert_eq!(metal_row["max_tokens_source"], "runtime_bucket");
            assert_eq!(metal_row["bucket_ladder"], json!([128, 256, 384, 512]));
            assert_eq!(metal_row["dims"], 384);
        } else {
            assert_eq!(metal_row["max_tokens"], 512);
            assert_eq!(metal_row["max_tokens_source"], "catalog");
            assert!(metal_row.get("bucket_ladder").is_none());
        }

        let ane_row = models
            .iter()
            .find(|r| r["model_id"] == "test-ane-lane")
            .expect("ane row exists");
        assert_eq!(
            ane_row["max_tokens"], 512,
            "ANE ceiling must come from worker's largest bucket, not misleading catalog budget"
        );
        assert_eq!(ane_row["max_tokens_source"], "worker_bucket");
        assert_eq!(ane_row["bucket_ladder"], json!([128, 256, 512]));
        assert_eq!(ane_row["dims"], 384);
        assert_eq!(ane_row["dtype"], "f16");
        assert_eq!(ane_row["device_class"], "ane");
        assert_eq!(
            ane_row["certified"], false,
            "uncertified ANE lane cannot advertise certified=true"
        );
        assert_eq!(ane_row["warm_load_cost_hint_ms"], 32.0);
        assert_eq!(ane_row["recommended_batch"]["rows"], 8);
        assert_eq!(ane_row["recommended_batch"]["token_budget"], 4096);

        // Enforced ceiling invariant: verify that the advertised ceiling (512) would
        // never permit a row length that the ANE worker refuses.
        let ladder = ane_row["bucket_ladder"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect::<Vec<_>>();
        let advertised_ceiling = ane_row["max_tokens"].as_u64().unwrap();

        // Any length up to the ceiling finds a covering bucket in the ladder:
        for length in [1, 100, 128, 200, 256, 400, 512] {
            assert!(
                length <= advertised_ceiling,
                "length should be within ceiling"
            );
            assert!(
                ladder.iter().any(|&bucket| bucket >= length),
                "ANE worker will accept length {length} because covering bucket exists"
            );
        }
        // Any length exceeding the ceiling has no covering bucket:
        for length in [513, 600, 1024, 4096] {
            assert!(
                length > advertised_ceiling,
                "length exceeds advertised ceiling"
            );
            assert!(
                !ladder.iter().any(|&bucket| bucket >= length),
                "ANE worker would refuse length {length} as no covering bucket exists"
            );
        }

        let unloaded_row = models
            .iter()
            .find(|r| r["model_id"] == "test-unloaded-lane")
            .expect("unloaded row exists");
        assert_eq!(unloaded_row["max_tokens"], 256);
        assert_eq!(unloaded_row["max_tokens_source"], "catalog_unloaded");
        assert!(
            unloaded_row.get("bucket_ladder").is_none(),
            "unloaded lane must omit unknown bucket ladder"
        );
        assert!(
            unloaded_row.get("dims").is_none(),
            "unloaded lane must omit unknown dims"
        );
        assert_eq!(unloaded_row["device_class"], "ane");
        assert_eq!(unloaded_row["certified"], false);
    }

    /// Regresses the native CUDA model.load path: a source=file owned-cuda load
    /// with family=qwen3/dtype=f16/execution=supervised must persist a complete
    /// owned profile, survive the restart normalization round-trip, and reach a
    /// package directory whose config.json the CUDA engine resolves.
    #[test]
    fn owned_cuda_model_load_persists_owned_profile_and_assembles_package() {
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
        // model.load construction: the same params execute_model_load_job
        // parses, and the same owned profile it must now build for owned-cuda.
        // source=file joins the top-level path with each file locator.
        let params: ModelLoadParams = serde_json::from_value(json!({
            "source": "file",
            "path": scratch.to_string_lossy(),
            "engine": CUDA_WORKER_ENGINE,
            "task": "embed",
            "model_id": "native-qwen3-load",
            "family": "qwen3",
            "dtype": "f16",
            "execution": "supervised",
            "max_tokens": 512,
            "files": {
                "model": "model.safetensors",
                "tokenizer": "tokenizer.json",
                "config": "config.json"
            }
        }))
        .expect("model.load params must parse");
        let engine_name = canonical_engine_name(&params.engine);
        assert_eq!(engine_name, CUDA_WORKER_ENGINE);

        let sources = resolve_model_load_sources(&params).expect("sources must resolve");
        let model_digest = format!("sha256:{}", sha256_hex(&model_bytes));
        let tokenizer_digest = format!(
            "sha256:{}",
            sha256_hex(&std::fs::read(&tokenizer_src).expect("read tokenizer"))
        );
        let config_digest = format!(
            "sha256:{}",
            sha256_hex(&std::fs::read(&config_src).expect("read config"))
        );

        let meta = |digest: String,
                    source_url: String,
                    format: String,
                    tokenizer_digest: Option<String>| ModelCacheMeta {
            digest,
            source_url,
            format,
            sanitized_tokenizer_digest: tokenizer_digest,
            validation_state: synapse_core::CacheValidationState::Valid,
            pins: Vec::new(),
            tombstone: None,
        };
        let model_meta = meta(
            model_digest.clone(),
            local_file_url(&model_src),
            default_artifact_format(&engine_name),
            Some(tokenizer_digest.clone()),
        );
        let tokenizer_meta = meta(
            tokenizer_digest.clone(),
            local_file_url(&tokenizer_src),
            "tokenizer_json".to_string(),
            None,
        );
        let config_meta = meta(
            config_digest.clone(),
            local_file_url(&config_src),
            "json".to_string(),
            None,
        );

        let owned_profile =
            model_load_owned_profile(&engine_name, &scratch, &params, Some(&config_meta), &[])
                .expect("load profile must build")
                .expect("CUDA load must retain its profile");

        let package_digest = package_digest(&model_meta, &tokenizer_meta, Some(&config_meta), &[]);
        let spec = build_loaded_catalog_model(
            &params,
            &engine_name,
            &sources,
            &model_meta,
            &tokenizer_meta,
            package_digest,
            Vec::new(),
            Some(owned_profile),
            &InlineConfig::default(),
            &JobConfig::default(),
        )
        .expect("catalog model must build");
        // The persisted row is the defect: pre-fix these were all None because
        // no owned profile reached build_stored_model_config.
        assert_eq!(spec.engine, CUDA_WORKER_ENGINE);
        assert_eq!(spec.owned_family.as_deref(), Some("qwen3-0.6b"));
        assert_eq!(spec.owned_dtype.as_deref(), Some("f16"));
        assert_eq!(spec.owned_execution.as_deref(), Some("supervised"));
        assert_eq!(
            spec.config_locator,
            Some(ModelAssetLocator::CacheDigest {
                digest: config_digest.clone()
            })
        );
        assert_eq!(spec.artifact_format, "safetensors-package");

        // Restart round-trip: the stored row is re-read through
        // normalize_catalog_model and must rehydrate the same owned profile.
        let restored = normalize_catalog_model(
            spec.clone(),
            &InlineConfig::default(),
            &JobConfig::default(),
        )
        .expect("stored row must normalize");
        assert_eq!(restored.owned_family, spec.owned_family);
        assert_eq!(restored.owned_dtype, spec.owned_dtype);
        assert_eq!(restored.config_locator, spec.config_locator);
        assert_eq!(restored.fingerprint, spec.fingerprint);

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
        // Terminal token policy: qwen3 reserves one token, so a 512-token
        // budget must become 511 through the owned profile.
        assert_eq!(
            owned_tokenizer_max_tokens(spec.max_tokens, Some(&rehydrated)),
            511
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
            &[],
            model_cache.root(),
            DEFAULT_MICROLLM_MAX_TOKENS,
            None,
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

    /// The projection must actually reach the descriptor. The unit test above
    /// pins the projection and the serialization; neither notices if
    /// `module_catalog_entries` stops wiring them together, which is the one
    /// edit that would silently restore the original defect.
    #[test]
    fn module_catalog_entries_wire_serving_admission_for_decode_lanes() {
        let (storage_dir, descriptor) = test_storage_descriptor("catalog-serving-admission");
        let store = Arc::new(SynapseStore::open(&descriptor).expect("store opens"));
        let profile = test_machine_profile("test-os");
        store
            .activate_profile(&profile, 1, 1000)
            .expect("activate profile");
        let state = test_module_state(store, profile);
        let _ = &storage_dir;

        let mut decode_spec = stuck_model_spec();
        decode_spec.model_id = "test-decode-lane".to_string();
        decode_spec.engine = DECODE_WORKER_ENGINE.to_string();
        decode_spec.task = "generate".to_string();
        decode_spec.fingerprint = Fingerprint("d".repeat(64));

        {
            let mut catalog = state.runtime.catalog.lock().expect("catalog locks");
            catalog.clear();
            catalog.insert(
                decode_spec.model_id.clone(),
                ModelSlot {
                    spec: decode_spec.clone(),
                    loaded: None,
                    state: ModelRuntimeState::Unloaded,
                    notify: Arc::new(Notify::new()),
                    last_cold_load_ms: None,
                },
            );
        }

        let entries = module_catalog_entries(&state);
        let row = entries
            .iter()
            .find(|entry| entry.model_id == "test-decode-lane")
            .expect("the decode lane appears in the catalog");

        // No approval row was written, so the lane refuses; the descriptor must
        // say so rather than leaving a consumer with `certified` alone.
        assert_eq!(
            row.serving_admission.as_deref(),
            Some("disabled"),
            "a decode lane with no approval must report disabled: {row:?}"
        );
        assert_eq!(
            row.serving_admission_reason.as_deref(),
            Some("approval_absent")
        );
    }

    /// `models.list` must carry the approval fact, not only the certification
    /// fact. A decode lane can be certified for this machine and still refuse
    /// every request because no operator approved it, and before this field the
    /// only thing the descriptor said about that lane was `certified: true` --
    /// so a consumer picking a lane read a lane that refuses as ready.
    ///
    /// Asserts BOTH rows rather than one: the disabled lane must say disabled
    /// while still reporting `certified: true` (otherwise the fix would be
    /// indistinguishable from making `certified` mean serving), and the enabled
    /// lane must say enabled (otherwise a field stuck at "disabled" would pass).
    #[test]
    fn models_list_reports_serving_admission_beside_certified() {
        let approved = serving_admission_projection(true, true, Some((true, None)));
        assert_eq!(approved, (Some("enabled"), None));

        let certified_but_unapproved = serving_admission_projection(true, true, None);
        assert_eq!(
            certified_but_unapproved,
            (Some("disabled"), Some("approval_absent".to_string())),
            "a certified lane with no approval row must read disabled, because it refuses"
        );

        let approved_but_uncertified =
            serving_admission_projection(true, false, Some((true, None)));
        assert_eq!(
            approved_but_uncertified,
            (Some("disabled"), Some("not_certified".to_string()))
        );

        // A lane class with no approval concept omits the field rather than
        // defaulting it, so absence can never be read as "enabled".
        assert_eq!(
            serving_admission_projection(false, true, Some((true, None))),
            (None, None)
        );

        // And the descriptor must actually serialize what the projection says.
        let entry = ModelCatalogEntry {
            model_id: "owned-model".to_string(),
            state: "unloaded".to_string(),
            fingerprints: vec![Fingerprint("f".repeat(64))],
            recommended_batch: None,
            max_tokens: Some(512),
            max_tokens_source: Some("catalog".to_string()),
            bucket_ladder: None,
            dims: None,
            dtype: None,
            device_class: None,
            certified: Some(true),
            serving_admission: certified_but_unapproved.0.map(str::to_string),
            serving_admission_reason: certified_but_unapproved.1.clone(),
            warm_load_cost_hint_ms: None,
        };
        let value = serde_json::to_value(&entry).expect("entry serializes");
        assert_eq!(value["certified"], json!(true));
        assert_eq!(value["serving_admission"], json!("disabled"));
        assert_eq!(value["serving_admission_reason"], json!("approval_absent"));
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

        let snapshot = state
            .store
            .catalog_snapshot()
            .expect("catalog snapshot reads");
        let payload = models_list_payload(&state, snapshot);
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

    #[test]
    fn models_list_generate_lane_without_evidence_does_not_report_certified_true() {
        let (_storage_dir, descriptor) = test_storage_descriptor("models-list-generate-uncert");
        let store = Arc::new(SynapseStore::open(&descriptor).expect("open test store"));
        let profile = test_machine_profile("test-os");
        store
            .activate_profile(&profile, 1, 1000)
            .expect("activate profile");
        let state = test_module_state(store, profile);

        let mut llama_spec = stuck_model_spec();
        llama_spec.model_id = "test-llama-generate".to_string();
        llama_spec.engine = "llama".to_string();
        llama_spec.task = "generate".to_string();

        let mut decode_spec = stuck_model_spec();
        decode_spec.model_id = "test-decode-generate".to_string();
        decode_spec.engine = DECODE_WORKER_ENGINE.to_string();
        decode_spec.task = "generate".to_string();

        {
            let mut catalog = state.runtime.catalog.lock().expect("catalog locks");
            catalog.clear();
            catalog.insert(
                llama_spec.model_id.clone(),
                ModelSlot {
                    spec: llama_spec.clone(),
                    loaded: None,
                    state: ModelRuntimeState::Ready,
                    notify: Arc::new(Notify::new()),
                    last_cold_load_ms: None,
                },
            );
            catalog.insert(
                decode_spec.model_id.clone(),
                ModelSlot {
                    spec: decode_spec.clone(),
                    loaded: None,
                    state: ModelRuntimeState::Ready,
                    notify: Arc::new(Notify::new()),
                    last_cold_load_ms: None,
                },
            );
        }

        let snapshot = state
            .store
            .catalog_snapshot()
            .expect("catalog snapshot reads");
        let payload = models_list_payload(&state, snapshot);
        let models = payload["models"]
            .as_array()
            .expect("models array in payload");

        let llama_row = models
            .iter()
            .find(|r| r["model_id"] == "test-llama-generate")
            .expect("llama row exists");
        assert_ne!(
            llama_row.get("certified"),
            Some(&json!(true)),
            "generate lane without evidence must not report certified=true"
        );
        assert!(
            llama_row.get("certified").is_none(),
            "legacy generate lane without certification concept must omit certified field"
        );

        let decode_row = models
            .iter()
            .find(|r| r["model_id"] == "test-decode-generate")
            .expect("decode row exists");
        assert_ne!(
            decode_row.get("certified"),
            Some(&json!(true)),
            "uncertified decode lane must not report certified=true"
        );
        assert_eq!(
            decode_row["certified"], false,
            "uncertified decode lane must report certified=false"
        );
    }

    #[tokio::test]
    async fn admission_telemetry_tracks_in_process_job_completion() {
        let (root, descriptor) = test_storage_descriptor("telemetry-job-complete");
        let store = Arc::new(SynapseStore::open(&descriptor).unwrap());
        let profile = test_machine_profile("telemetry-complete-os");
        store.observe_profile(&profile, 10, 1).unwrap();
        let state = test_module_state(store, profile);

        let outcome = probe_start(
            Arc::clone(&state),
            json!({"request_key": "probe-complete-1"}),
        )
        .await;
        let result = response_result(outcome, "probe.start");
        let job_id = result["job_id"].as_str().expect("job_id in result");

        let mut done = false;
        for _ in 0..100 {
            tokio::time::sleep(Duration::from_millis(10)).await;
            let status_outcome = probe_status(Arc::clone(&state), json!({"job_id": job_id})).await;
            let status_result = response_result(status_outcome, "probe.status");
            if status_result["state"] == "done" {
                done = true;
                break;
            }
        }
        assert!(done, "probe job did not finish in time");

        let status_outcome = admission_status(Arc::clone(&state)).await;
        let status = response_result(status_outcome, "admission.status");
        assert_eq!(status["jobs_open"], 0, "jobs_open == 1");
        assert_eq!(status["jobs_completed"], 1);
        assert_eq!(status["jobs_minted"], 1);

        drop(state);
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn admission_telemetry_tracks_in_process_job_failure() {
        let (root, descriptor) = test_storage_descriptor("telemetry-job-fail");
        let store = Arc::new(SynapseStore::open(&descriptor).unwrap());
        let profile = test_machine_profile("telemetry-fail-os");
        store.observe_profile(&profile, 10, 1).unwrap();
        let state = test_module_state(store, profile);

        let outcome = model_load(
            Arc::clone(&state),
            json!({
                "request_key": "load-fail-1",
                "model_id": "nonexistent-model",
                "source": "file",
                "path": "/tmp/nonexistent-model-dir-12345",
                "files": {
                    "model": {
                        "url": "model.safetensors",
                        "sha256": "0000000000000000000000000000000000000000000000000000000000000000"
                    },
                    "tokenizer": {
                        "url": "tokenizer.json",
                        "sha256": "0000000000000000000000000000000000000000000000000000000000000000"
                    }
                },
                "engine": "owned-metal"
            }),
        )
        .await;
        let result = response_result(outcome, "model.load");
        let job_id = result["job_id"].as_str().expect("job_id in result");

        let mut failed = false;
        for _ in 0..100 {
            tokio::time::sleep(Duration::from_millis(10)).await;
            if let Ok(Some(record)) = state.store.get_job(job_id) {
                if record.state == "failed_transient" || record.state == "failed_permanent" {
                    failed = true;
                    break;
                }
            }
        }
        if !failed {
            let record = state.store.get_job(job_id).unwrap();
            panic!("job state was: {record:?}");
        }

        let status_outcome = admission_status(Arc::clone(&state)).await;
        let status = response_result(status_outcome, "admission.status");
        assert_eq!(status["jobs_minted"], 1);
        assert_eq!(status["jobs_failed"], 1);
        assert_eq!(status["jobs_completed"], 0);
        assert_eq!(status["jobs_open"], 0);

        drop(state);
        let _ = std::fs::remove_dir_all(root);
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
                None,
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

    #[test]
    fn admission_telemetry_jobs_open_never_underflows() {
        let telemetry = AdmissionTelemetry::default();
        telemetry.record_job_completed();
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.jobs_minted, 0);
        assert_eq!(snapshot.jobs_completed, 1);
        assert_eq!(snapshot.jobs_open, 0);
    }
}

#[cfg(test)]
mod routing_identity_dump_tests {
    use super::*;

    /// Prints routing identities for preload-spec JSON supplied in
    /// `SYNAPSE_OWNED_DECODE_PRELOAD_SPEC_JSON`. The value may be one preload
    /// object or an array; fleet certification supplies all four family/quant
    /// lanes so their independent identities are recorded in one dump.
    #[test]
    #[ignore = "requires readable fleet preload specs and tokenizer artifacts"]
    fn dump_owned_decode_routing_identity_from_preload_spec() {
        let preload_json = env::var("SYNAPSE_OWNED_DECODE_PRELOAD_SPEC_JSON")
            .expect("set SYNAPSE_OWNED_DECODE_PRELOAD_SPEC_JSON to preload-model JSON");
        let value: Value = serde_json::from_str(&preload_json).expect("preload JSON must parse");
        let preload_values = match value {
            Value::Array(values) => values,
            value @ Value::Object(_) => vec![value],
            _ => panic!("preload JSON must be one object or an array of objects"),
        };
        let identities = preload_values
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                let preload: PreloadModelConfig =
                    serde_json::from_value(value).expect("preload spec must parse");
                let spec = build_preload_catalog_model(
                    index,
                    preload,
                    &InlineConfig::default(),
                    &JobConfig::default(),
                )
                .expect("preload spec must build a catalog model");
                let entry = owned_decode_catalog_entry(&spec)
                    .expect("preload spec must build a decode entry");
                let decode_fingerprint = entry
                    .decode_identity_inputs()
                    .decode_fingerprint()
                    .expect("decode identity must be valid");
                let processing_fingerprint = owned_decode_processing_fingerprint(&entry)
                    .expect("processing identity must be valid");
                let (runtime_config_digest, _) =
                    owned_decode_runtime_identity(&spec, &entry, DEFAULT_DECODE_CHAIN_K);
                let tokenizer_path = match &spec.tokenizer_locator {
                    ModelAssetLocator::LocalPath { path } => path,
                    ModelAssetLocator::CacheDigest { .. } => {
                        panic!("identity dump requires a local tokenizer path")
                    }
                };
                let tokenizer = SanitizedTokenizer::from_file(
                    tokenizer_path,
                    TokenizerConfig {
                        max_tokens: spec.max_tokens,
                    },
                )
                .expect("preload tokenizer must load");
                let tokenizer_vocabulary_digest = owned_decode_vocabulary_digest(&tokenizer)
                    .expect("tokenizer vocabulary must load");
                let constraint = owned_decode_grammar_scheduler::compile_grammar(
                    r#"{"type":"string"}"#,
                    &owned_decode_grammar_scheduler::CompileContext {
                        base_decode_fingerprint: decode_fingerprint.clone(),
                        tokenizer_vocabulary_digest,
                    },
                    &owned_decode_grammar_scheduler::GrammarSubsetManifest::default(),
                )
                .expect("default grammar subset must compile a string schema");
                serde_json::json!({
                    "entry_id": entry.entry_id,
                    "family": entry.family.as_str(),
                    "activation_dtype": entry.activation_dtype.as_str(),
                    "weight_quant": entry.weight_quant.as_str(),
                    "quantizer_revision": entry.q8.as_ref().map(|q8| q8.quantizer_revision.as_str()),
                    "derived_digest": entry.q8.as_ref().map(|q8| q8.derived_digest.as_str()),
                    "decode_fingerprint": decode_fingerprint.0,
                    "processing_fingerprint": processing_fingerprint.0,
                    "runtime_config_digest": runtime_config_digest,
                    "constraint_runtime_identity": constraint
                        .constraint
                        .constraint_runtime_identity
                        .digest(),
                })
            })
            .collect::<Vec<_>>();
        println!(
            "{}",
            serde_json::to_string_pretty(&identities).expect("identity tuples serialize")
        );
    }
}

#[cfg(test)]
mod catalog_runtime_tests {
    use super::*;
    use crate::tests::{
        response_result, test_machine_profile, test_module_state, test_storage_descriptor,
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

    fn isolated_catalog_state(label: &str, runnable: bool) -> (PathBuf, Arc<ModuleState>) {
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

    #[test]
    fn model_load_uses_each_resolved_asset_scheme_for_digest_guards() {
        let mut params: ModelLoadParams = serde_json::from_value(json!({"source":"file","path":"/tmp","engine":"owned-metal","task":"embed","files":{"model":{"url":"https://cdn.example.org/model","sha256":""},"tokenizer":"tokenizer.json"}})).unwrap();
        assert!(resolve_model_load_sources(&params).is_err());
        params.files.model = ModelLoadFileSpec::Detailed {
            url: "https://cdn.example.org/model".into(),
            sha256: "a".repeat(64),
        };
        assert!(resolve_model_load_sources(&params).is_ok());
        params.expected_digest = Some("a".repeat(64));
        params.files.tokenizer = ModelLoadFileSpec::Detailed {
            url: "https://cdn.example.org/tokenizer".into(),
            sha256: String::new(),
        };
        assert!(resolve_model_load_sources(&params).is_err());
        params.files.tokenizer = ModelLoadFileSpec::Legacy("tokenizer.json".into());
        params.engine = "ort".into();
        assert!(validate_model_load_request(&params).is_err());
        params.engine = "llama".into();
        assert!(validate_model_load_request(&params).is_err());
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
            Some(&"0".repeat(64)),
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

    #[tokio::test]
    async fn catalog_no_accelerator_lists_every_entry_and_refuses_downloads() {
        let (root, state) = isolated_catalog_state("no-accelerator", false);
        let result = response_result(
            models_catalog(state.clone(), json!({})).await,
            "models.catalog",
        );
        assert_eq!(result["models"].as_array().unwrap().len(), 4);
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
        let result = response_result(model_load(state.clone(),json!({"source":"file","path":"/missing","engine":"owned-metal","model_id":"gte-modernbert-base-ane","files":{"model":"m","tokenizer":"t"}})).await,"model.load");
        assert_eq!(result["error"]["code"], "invalid_request");
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
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            let record = state.store.get_job(job).unwrap().unwrap();
            if !store::is_download_non_terminal(&record.state) {
                assert_eq!(record.state, "committed", "{record:?}");
                break;
            }
            assert!(tokio::time::Instant::now() < deadline);
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
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
fn detected_catalog_backends() -> BTreeSet<String> {
    #[cfg(feature = "test-support")]
    if let Ok(value) = env::var("SYNAPSE_TEST_RUNNABLE_BACKENDS") {
        return value
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();
    }
    #[cfg(target_os = "macos")]
    let backends = {
        let mut backends = BTreeSet::new();
        if metal::Device::system_default().is_some() {
            backends.insert("metal".into());
        }
        if direct_ane_catalog_available(true, catalog_direct_ane_worker().as_deref()) {
            backends.insert("ane".into());
        }
        backends
    };
    #[cfg(not(target_os = "macos"))]
    let backends = BTreeSet::new();
    backends
}
#[cfg(target_os = "macos")]
fn catalog_direct_ane_worker() -> Option<PathBuf> {
    env::var_os(worker_binary_env_var("ane-direct-worker"))
        .map(PathBuf::from)
        .or_else(|| resolve_worker_binary_sibling("ane-direct-worker"))
}
// These catalog models use ck-synapse-worker-ane-direct, which calls Apple's
// Neural Engine API directly. The separate worker that accesses the Neural
// Engine through Apple's Core ML framework does not satisfy this requirement.
// Require macOS and the direct worker binary before offering this backend.
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
fn direct_ane_catalog_available(macos: bool, worker: Option<&Path>) -> bool {
    macos && worker.is_some_and(Path::is_file)
}
fn catalog_backend_reason(runtime: &RuntimeState, backend: &str) -> Option<&'static str> {
    if runtime.runnable_backends.contains(backend) {
        None
    } else if !cfg!(target_os = "macos") && matches!(backend, "metal" | "ane") {
        Some("not_supported_on_platform")
    } else if backend == "metal" {
        Some("device_missing")
    } else {
        Some("worker_missing")
    }
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
    for backend in entry
        .backends
        .iter()
        .filter(|b| state.runtime.runnable_backends.contains(&b.backend))
    {
        if current_catalog_install(state, entry, &backend.backend)?.is_none() {
            return Ok(false);
        }
        for file in entry.backend_files(&backend.backend).values() {
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
                .filter(|b| state.runtime.runnable_backends.contains(&b.backend))
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
                backends.push(json!({"backend":b.backend,"lane_id":catalog::lane_id(&entry.id,&b.backend),"fingerprint":b.fingerprint,"runnable":catalog_backend_reason(&state.runtime,&b.backend).is_none(),"reason":catalog_backend_reason(&state.runtime,&b.backend),"installed":installed,"self_check":check}));
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
        if !entry
            .backends
            .iter()
            .any(|b| state.runtime.runnable_backends.contains(&b.backend))
        {
            return Err(catalog_backend_unavailable(&state, entry, None));
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
    let backends = entry.backends.iter().filter(|b| selected.is_none_or(|s| s.backend == b.backend)).map(|b| json!({"backend":b.backend,"reason":catalog_backend_reason(&state.runtime,&b.backend)})).collect::<Vec<_>>();
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
            .filter(|b| plan_state.runtime.runnable_backends.contains(&b.backend))
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
                    .backend_files(b)
                    .values()
                    .map(|f| store::CatalogInstallMember {
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
                ModelRuntimeState::Loading
                    | ModelRuntimeState::Resolving
                    | ModelRuntimeState::Validating
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
        let holders = catalog_holders(&state, entry)?;
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
        let freed = state
            .store
            .store
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
    let model = files["model"];
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
    let owned = OwnedCatalogConfig {
        family: OwnedFamily::parse(backend.family.as_deref().expect("validated family"))
            .map_err(|e| artifact_invalid_error(e.to_string()))?,
        dtype: OwnedDType::parse(backend.dtype.as_deref().expect("validated dtype"))
            .map_err(|e| artifact_invalid_error(e.to_string()))?,
        execution: backend.execution.clone().expect("validated execution"),
        attention_units: backend.attention_units.expect("validated attention units") as usize,
        config_locator: files.get("config").map(|f| ModelAssetLocator::CacheDigest {
            digest: format!("sha256:{}", f.sha256),
        }),
        extra_locators: Vec::new(),
        identity_override: None,
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
        "safetensors".into(),
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
            if !state.runtime.runnable_backends.contains(&backend.backend)
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
fn catalog_self_check_key(
    state: &ModuleState,
    entry: &catalog::CatalogEntry,
    backend: &catalog::CatalogBackend,
) -> Result<(String, Value), WireOperationError> {
    let identity = if let Some(profile) = backend.profile.as_deref() {
        CatalogProfile::load(profile)
            .and_then(|profile| profile.owned_config(backend.execution.as_deref(), None))
            .map_err(|error| artifact_invalid_error(error.to_string()))?
            .identity_override
            .expect("profile identity")
    } else {
        owned_engine_identity(
            OwnedFamily::parse(backend.family.as_deref().expect("family"))
                .map_err(|e| artifact_invalid_error(e.to_string()))?,
            OwnedDType::parse(backend.dtype.as_deref().expect("dtype"))
                .map_err(|e| artifact_invalid_error(e.to_string()))?,
        )
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
    let id = sha256_hex(catalog::jcs(&key).map_err(catalog_store_error)?.as_bytes());
    Ok((id, key))
}
fn catalog_self_check_projection(
    state: &ModuleState,
    entry: &catalog::CatalogEntry,
    backend: &catalog::CatalogBackend,
) -> Result<Value, WireOperationError> {
    use rusqlite::OptionalExtension;
    let (id, _) = catalog_self_check_key(state, entry, backend)?;
    let row = state
        .store
        .store
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
    state.store.store.with_conn_fenced(|tx| {
        tx.execute("UPDATE self_check_run_seq SET value=value+1 WHERE id=0",[])?;
        let generation: u64 = tx.query_row("SELECT value FROM self_check_run_seq WHERE id=0",[],|r| r.get(0))?;
        tx.execute("INSERT INTO catalog_self_checks (check_id,catalog_id,manifest_digest,backend,fingerprint,engine_identity_json,os_build,fixture_revision,state,generation,checked_at_ms,reason) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,'running',?9,NULL,NULL) ON CONFLICT(check_id) DO UPDATE SET state='running',generation=excluded.generation,checked_at_ms=NULL,reason=NULL",rusqlite::params![id,key["catalog_id"].as_str(),key["manifest_digest"].as_str(),key["backend"].as_str(),key["fingerprint"].as_str(),key["engine_identity"].to_string(),key["os_build"].as_str(),key["fixture_revision"].as_str(),generation])?;
        Ok(generation)
    }).map_err(catalog_store_error)
}
fn catalog_complete_check(
    state: &ModuleState,
    id: &str,
    generation: u64,
    status: &str,
    reason: Option<&str>,
) -> Result<bool, WireOperationError> {
    state.store.store.with_conn_fenced(|tx| {
        if status == "pending" {
            tx.execute("UPDATE self_check_run_seq SET value=value+1 WHERE id=0",[])?;
            Ok(tx.execute("UPDATE catalog_self_checks SET state='pending',generation=(SELECT value FROM self_check_run_seq WHERE id=0),checked_at_ms=NULL,reason=NULL WHERE check_id=?1 AND state='running' AND generation=?2",rusqlite::params![id,generation])? > 0)
        } else {
            Ok(tx.execute("UPDATE catalog_self_checks SET state=?3,checked_at_ms=?4,reason=?5 WHERE check_id=?1 AND state='running' AND generation=?2",rusqlite::params![id,generation,status,now_ms(),reason])? > 0)
        }
    }).map_err(catalog_store_error)
}
fn numerical_catalog_check(
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
        let vectors = engine
            .embed_batch(&model.loaded_model, tokenized.batch)
            .map_err(engine_error_to_wire)?;
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
        let tolerance = backend.rerank_abs_tolerance.expect("tolerance");
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
            let raw = engine
                .rerank_pairs(&model.loaded_model, pairs)
                .map_err(engine_error_to_wire)?;
            let scores = raw
                .scores
                .iter()
                .map(|x| 1.0 / (1.0 + (-f64::from(*x)).exp()))
                .collect::<Vec<_>>();
            if scores.len() != reference.len() || raw.scores.iter().any(|s| !s.is_finite()) {
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
    if catalog_backend_reason(&state.runtime, &backend.backend).is_some() {
        return Err(catalog_backend_unavailable(state, entry, Some(backend)));
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
        .store
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
    held_guard: Option<Arc<tokio::sync::OwnedMutexGuard<()>>>,
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
            let readout = references
                .manifest
                .model(&references.manifest.profiles[profile].model)
                .expect("sealed model")
                .grammar
                .readout
                .yes
                .as_ref()
                .zip(
                    references
                        .manifest
                        .model(&references.manifest.profiles[profile].model)
                        .expect("sealed model")
                        .grammar
                        .readout
                        .no
                        .as_ref(),
                )
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
    deadline_ms: Option<u64>,
) -> Result<Arc<EmbeddingModel>, WireOperationError> {
    let lock = catalog_lane_lock(&state.runtime, model_id);
    let guard = Arc::new(lock.lock_owned().await);
    let model = ensure_model_loaded_for_control(state.clone(), model_id, deadline_ms).await?;
    check_profile_model(&state, model, guard).await
}

// Catalog installs and startup preloads must compare model output against the
// same load-time fp32 reference inputs and expected outputs. These cases are
// copied from bench/parity, checked by SHA-256 and embedded in synapse-certify.
// Use the production execution path so validation exercises the same worker
// and hardware routing as serving requests, not a separate implementation.
async fn check_profile_model(
    state: &ModuleState,
    model: Arc<EmbeddingModel>,
    guard: Arc<tokio::sync::OwnedMutexGuard<()>>,
) -> Result<Arc<EmbeddingModel>, WireOperationError> {
    let profile = model
        .engine_identity
        .build_flags
        .get("profile")
        .expect("profile preload");
    let (id, key) =
        if let Some((entry, backend)) = resolved_catalog_lane(&state.runtime, &model.model_id) {
            catalog_self_check_key(state, entry, backend)?
        } else {
            profile_preload_check_key(state, &model, profile)?
        };
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
    let evaluation = match numerical_profile_preload_check(
        state,
        &model,
        profile,
        references,
        Some(guard),
    )
    .await
    {
        Ok(evaluation) => evaluation,
        Err(error) => {
            let reason = serde_json::to_string(&error).expect("wire error serializes");
            catalog_complete_check(state, &id, generation, "failed", Some(&reason))?;
            return Err(profile_preload_check_error(profile, &reason));
        }
    };
    complete_profile_preload_check(state, profile, &id, generation, &evaluation)?;
    Ok(model)
}

async fn resolve_serving_model(
    state: Arc<ModuleState>,
    requested: Option<&str>,
    task: ModelTask,
    required: Option<&str>,
    target: Option<&str>,
    deadline_ms: Option<u64>,
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
            return ensure_model_loaded_for_control(state, model_id, deadline_ms).await;
        }
    }
    if task == ModelTask::Generate
        || requested.is_some_and(|id| !state.runtime.release_catalog.is_reserved_id(id))
    {
        return resolve_model_for_request(state, requested, task).await;
    }
    let (entry, backend) = select_catalog_lane(&state, requested, task, required, target)?;
    let lane = catalog::lane_id(&entry.id, &backend.backend);
    let budget = deadline_ms.unwrap_or(state.runtime.inline.deadline_ms);
    if budget == 0 {
        return Err(catalog_loading_response(&lane, budget));
    }
    let started = std::time::Instant::now();
    let lock = catalog_lane_lock(&state.runtime, &lane);
    let (guard, waited) = match lock.clone().try_lock_owned() {
        Ok(guard) => (guard, false),
        Err(_) => (
            tokio::time::timeout(Duration::from_millis(budget.min(5000)), lock.lock_owned())
                .await
                .map_err(|_| catalog_loading_response(&lane, budget))?,
            true,
        ),
    };
    if let Some(error) = catalog_failed_load(&state.runtime, &lane, !waited) {
        return Err(error);
    }
    let remaining = budget
        .min(5000)
        .saturating_sub(started.elapsed().as_millis().min(u64::MAX as u128) as u64);
    let task_state = state.clone();
    // Only the lock owner starts detached work. A timed-out waiter must not
    // stay queued and start a new load after the attempt it was waiting for fails.
    let handle = tokio::spawn(async move {
        ensure_catalog_lane_ready_owned(task_state, entry, backend, guard).await
    });
    match tokio::time::timeout(Duration::from_millis(remaining), handle).await {
        Ok(Ok(result)) => result,
        Ok(Err(e)) => Err(WireOperationError::from_stable(
            StableError::engine_crashed(Some(250)),
            format!("catalog load task failed: {e}"),
        )),
        Err(_) => Err(catalog_loading_response(&lane, budget)),
    }
}

fn catalog_loading_response(lane: &str, budget: u64) -> WireOperationError {
    if budget <= 5000 {
        WireOperationError::from_stable(
            StableError::deadline_exceeded(),
            "catalog admission deadline expired",
        )
    } else {
        WireOperationError::from_stable(
            StableError::model_loading(Some(250)),
            format!("catalog lane '{lane}' is loading"),
        )
    }
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
        slot.state = ModelRuntimeState::Unloaded;
        slot.notify.notify_waiters();
    }
    Some(error)
}
struct CatalogInvocation {
    runtime: Arc<RuntimeState>,
    lane: String,
    _guard: tokio::sync::OwnedMutexGuard<()>,
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
) -> Result<Arc<EmbeddingModel>, WireOperationError> {
    let lane = catalog::lane_id(&entry.id, &backend.backend);
    let lock = catalog_lane_lock(&state.runtime, &lane);
    let (guard, waited) = match lock.clone().try_lock_owned() {
        Ok(guard) => (guard, false),
        Err(_) => (lock.lock_owned().await, true),
    };
    if let Some(error) = catalog_failed_load(&state.runtime, &lane, !waited) {
        return Err(error);
    }
    let profile_backed = backend.profile.is_some();
    let model = ensure_catalog_lane_ready_owned(state.clone(), entry, backend, guard).await?;
    if profile_backed {
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
    if model_slot_snapshot(&state.runtime, &lane).is_none() {
        sync_installed_catalog_slots(&state)?;
    }
    let (check_id, key) = catalog_self_check_key(&state, &entry, &backend)?;
    let projection = catalog_self_check_projection(&state, &entry, &backend)?;
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
            for file in entry.backend_files(&backend.backend).values() {
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
            if spec.fingerprint.0 != backend.fingerprint {
                return Err(catalog_wire_error(
                    "artifact_invalid",
                    json!({"model_id":lane,"expected_fingerprint":backend.fingerprint,"actual_fingerprint":spec.fingerprint}),
                    "catalog fingerprint differs from verified artifact parameters",
                ));
            }
            if let Some(slot) = state
                .runtime
                .catalog
                .lock()
                .expect("runtime catalog")
                .get_mut(&lane)
            {
                slot.spec = spec;
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
        let loaded = load_catalog_model_task(load_state, load_lane).await?;
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
    };
    let call_model = model.clone();
    let call_entry = entry.clone();
    let call_backend = backend.clone();
    let handle = tokio::task::spawn_blocking(move || {
        let _invocation = invocation;
        let result = numerical_catalog_check(&call_model, &call_entry, &call_backend);
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
    state.store.store.with_conn_fenced(|tx| {
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
fn catalog_list_row(state: &ModuleState, row: &mut Value) -> Result<(), WireOperationError> {
    let id = row["model_id"].as_str().unwrap_or("").to_string();
    if let Some((entry, backend)) = resolved_catalog_lane(&state.runtime, &id) {
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
    let mut roots = state.store.store.with_conn(|conn| {
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
        None,
        None,
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
            None,
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
        tokio::spawn(async move {
            let job = record.job_id.clone();
            let prepared = async {
                let budget = record
                    .execution_expires_ms
                    .unwrap_or(now_ms())
                    .saturating_sub(now_ms());
                let task_state = state.clone();
                let loading = tokio::spawn(async move {
                    ensure_catalog_lane_ready(task_state, entry, backend).await
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
                let parsed: EmbedBatchParams = serde_json::from_value(params)
                    .map_err(|e| artifact_invalid_error(e.to_string()))?;
                let items =
                    batch_items(parsed.items, parsed.texts).map_err(artifact_invalid_error)?;
                let request_bytes = request_bytes_for_texts(items.iter().map(|i| i.text.as_str()));
                let mut tokenized = model
                    .tokenizer
                    .tokenize_batch(items.iter().map(|i| i.text.as_str()))
                    .map_err(|e| artifact_invalid_error(e.to_string()))?;
                compose_catalog_embed(&model, &mut tokenized)?;
                apply_owned_tokenizer_policy(&model, &mut tokenized);
                let total_tokens = tokenized
                    .real_token_counts
                    .iter()
                    .map(|n| u64::from(*n))
                    .sum();
                let alias_table = state.store.alias_table().map_err(catalog_store_error)?;
                check_fingerprint_constraints(
                    &model,
                    &alias_table,
                    parsed.target_fingerprint.as_deref(),
                    parsed.required_fingerprint.as_deref(),
                    false,
                    parsed.required_epoch,
                )?;
                Ok::<_, WireOperationError>(EmbedBatchJobWork {
                    model,
                    request_digest: digest,
                    ids: items.into_iter().map(|i| i.id).collect(),
                    tokenized,
                    alias_table,
                    request_bytes,
                    total_tokens,
                })
            }
            .await;
            match prepared {
                Ok(work) => execute_embed_batch_job(state.clone(), job.clone(), work).await,
                Err(e) => {
                    fail_job_with_wire_error(&state, &job, e.class == ErrorClass::Transient, e)
                }
            }
            state
                .runtime
                .catalog_jobs
                .lock()
                .expect("catalog jobs")
                .remove(&job);
        });
    }
    result_outcome(response)
}
