use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::io;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use owned_decode_worker::{
    budget::{BudgetPolicy as OwnedBudgetPolicy, CrashBudget as OwnedCrashBudget, FileBudgetStore},
    error::DecodeError,
    identity::QuarantineKey,
    protocol::{
        DecodeTransportRequest, FrameEnvelope, GenerateCancel, GenerateContinue,
        GenerateInstallHintBank, GenerateProgress, GenerateStart, HintBankInstalled,
    },
    supervisor::{
        Clock as OwnedClock, GenerationRequest as OwnedGenerationRequest, Supervisor,
        TerminalControl,
    },
    validation::{StartAuthorization, WorkerStartContext},
    worker::{
        CancelAck, DecodeWorker, HintBankSource, NoHintBankSource, SteppedFrame, WorkerFactory,
        WorkerFault, WorkerStartFailure,
    },
};
use serde::{Deserialize, Serialize};
use synapse_core::worker_transport::accept_worker_handshake_with_hello;
use synapse_core::{
    decode_f32_frame, encode_i32_frame, is_owned_worker_hello_engine, prepare_listener, read_json,
    worker_engine_names::LLAMA_WORKER_ENGINE, write_json, EmbedEngine, EngineError,
    EngineErrorStage, EngineIdentity, EngineRiskClass, ExpectedHelloBinding, GenerateEngine,
    GenerateOutput, GenerateRequest, LoadedModel, ProgressBoundary, RerankEngine, RerankRequest,
    RerankScores, RuntimeConfig, TokenBatch, TokenIds, TransportError, ValidatedArtifact, Vector,
    Vectors, WorkerCandidate, WorkerPooling, WorkerRequest, WorkerResponse, WorkerSequence,
    WorkerTokenItem, WorkerTransportStream, DEFAULT_MAX_FRAME_BYTES,
};
use thiserror::Error;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::{Child, Command};
use tokio::runtime::Runtime;
use tokio::time::timeout;

const STDERR_RING_BYTES: usize = 8 * 1024;

#[cfg(all(test, unix))]
mod quarantine_tests;

// Catalog users cannot manually probe and approve a worker to restore serving.
// A finite ten-minute cool-down lets a desktop lane recover without that step.
pub(crate) const CATALOG_QUARANTINE_COOLDOWN: Duration = Duration::from_secs(600);
mod liveness;
#[cfg(all(test, unix))]
use synapse_core::{read_raw, write_raw};
#[cfg(all(test, unix))]
mod framing_tests;
#[cfg(all(test, unix))]
mod progress_tests;

#[derive(Clone, Debug)]
pub struct CrashBudget {
    pub max_crashes: usize,
}

impl Default for CrashBudget {
    fn default() -> Self {
        Self { max_crashes: 2 }
    }
}

/// Crash/quarantine authority for a worker host.
///
/// Catalog workers use the host's per-model consecutive faults. Owned decode
/// persists its own supervisor budget by machine/decode/runtime identity, so the
/// generic host must not charge a second fault counter for the same worker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CrashAuthority {
    #[default]
    WorkerHost,
    OwnedDecodeSupervisor,
}

#[derive(Clone, Debug)]
pub struct WorkerHostConfig {
    pub worker_bin: PathBuf,
    pub runtime_dir: PathBuf,
    pub worker_id: String,
    pub max_frame: u32,
    pub handshake_timeout: Duration,
    /// Owned-decode envelope only; catalog requests use fixed idle windows.
    pub request_timeout: Duration,
    pub crash_budget: CrashBudget,
    pub crash_authority: CrashAuthority,
    pub extra_args: Vec<String>,
    pub pooling: WorkerPooling,
    pub normalize: bool,
    /// Stable catalog model identity attached to forwarded worker diagnostics.
    pub model_id: Option<String>,
    /// Selected lane ID from the release catalog's reserved-ID lookup, not a
    /// profile flag or arbitrary model-name convention. None for free-form models.
    pub catalog_lane: Option<String>,
    /// Shared stdout/stderr forwarding budget for this worker process.
    pub worker_forward_lines_per_sec: u32,
    /// Optional catalog identity used by engines that share this generic host.
    pub engine_identity: Option<EngineIdentity>,
    /// Owned-CUDA has one process per stable model specification. Including
    /// the worker id in the crash key keeps equal artifacts isolated.
    pub isolate_crash_key_by_worker_id: bool,
    /// Manifest digest and kernel revision an owned worker's HELLO must carry.
    /// Set only for lanes bound to the model manifest; `None` keeps the
    /// handshake that accepts a HELLO without either field.
    pub expected_hello_binding: Option<ExpectedHelloBinding>,
    /// Direct-ANE lane lock the spawned worker inherits as its stdin, so the
    /// lock stays held until this worker has exited too. See
    /// [`ane_residency::AneDirectLaneLock`].
    pub inherited_lane_lock: Option<Arc<std::fs::File>>,
}

impl WorkerHostConfig {
    pub fn new(worker_bin: impl Into<PathBuf>, runtime_dir: impl Into<PathBuf>) -> Self {
        Self {
            worker_bin: worker_bin.into(),
            runtime_dir: runtime_dir.into(),
            worker_id: "default".to_string(),
            max_frame: DEFAULT_MAX_FRAME_BYTES,
            handshake_timeout: Duration::from_secs(5),
            request_timeout: Duration::from_secs(30),

            crash_budget: CrashBudget::default(),
            crash_authority: CrashAuthority::WorkerHost,
            extra_args: Vec::new(),
            pooling: WorkerPooling::Mean,
            normalize: true,
            model_id: None,
            catalog_lane: None,
            worker_forward_lines_per_sec: 50,
            engine_identity: None,
            isolate_crash_key_by_worker_id: false,
            expected_hello_binding: None,
            inherited_lane_lock: None,
        }
    }
}

#[derive(Debug, Error)]
pub enum WorkerHostError {
    #[error("worker I/O: {0}")]
    Io(#[from] io::Error),
    #[error("worker JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("worker protocol: {0}")]
    Protocol(String),
    #[error("worker protocol version {advertised:?} is unsupported; required {required}")]
    ProtocolVersion {
        advertised: Option<u8>,
        required: u8,
    },
    #[error("worker returned {code}: {msg}")]
    WorkerErr { code: String, msg: String },
    /// The worker's HELLO was refused because its manifest digest or kernel
    /// revision differs from the lane's (`manifest_mismatch`,
    /// `kernel_revision_mismatch`). Distinct from a worker `ERR` at LOAD.
    #[error("worker HELLO refused with {code}: {msg}")]
    HelloRefused { code: String, msg: String },
    #[error("engine_crashed at {stage}: {detail}")]
    EngineCrashed {
        stage: String,
        detail: String,
        stderr_tail: String,
    },
    #[error("model/config is quarantined after repeated worker crashes: {key}")]
    Quarantined {
        key: String,
        retry_after_ms: Option<u64>,
    },
}

impl WorkerHostError {
    pub fn to_engine_error(&self, stage: EngineErrorStage) -> EngineError {
        EngineError {
            stage: if matches!(self, Self::Quarantined { .. }) {
                EngineErrorStage::WorkerCrash
            } else {
                stage
            },
            risk_class: EngineRiskClass::AbortCapable,
            message: self.to_string(),
            retry_after_ms: match self {
                Self::WorkerErr { code, .. }
                    if code == ane_residency::ERR_ANE_RESOURCES_EXHAUSTED =>
                {
                    Some(ane_residency::ANE_RESOURCES_RETRY_AFTER_MS)
                }
                Self::Quarantined { retry_after_ms, .. } => *retry_after_ms,
                Self::WorkerErr { .. } | Self::HelloRefused { .. } => None,
                _ => Some(250),
            },
            safe_to_retry_same_request: matches!(
                self,
                Self::EngineCrashed { .. }
                    | Self::Quarantined {
                        retry_after_ms: Some(_),
                        ..
                    }
            ) || matches!(self, Self::WorkerErr {code, ..} if code == ane_residency::ERR_ANE_RESOURCES_EXHAUSTED),
        }
    }

    /// The typed code of a worker `ERR` or of a HELLO refusal.
    pub fn code(&self) -> Option<&str> {
        match self {
            Self::WorkerErr { code, .. } | Self::HelloRefused { code, .. } => Some(code),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
struct LogRing {
    bytes: Arc<Mutex<Vec<u8>>>,
}

impl LogRing {
    fn new() -> Self {
        Self {
            bytes: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn push(&self, bytes: &[u8]) {
        let mut guard = self.bytes.lock().expect("stderr ring mutex poisoned");
        guard.extend_from_slice(bytes);
        if guard.len() > STDERR_RING_BYTES {
            let drain = guard.len() - STDERR_RING_BYTES;
            guard.drain(0..drain);
        }
    }

    fn tail(&self) -> String {
        let guard = self.bytes.lock().expect("stderr ring mutex poisoned");
        String::from_utf8_lossy(&guard).into_owned()
    }
}

struct WorkerConnection {
    stream: WorkerTransportStream,
    child: Child,
    logs: LogRing,
    worker_generation: u64,
    packed: Option<synapse_core::PackedCapability>,
    unusable: bool,
}

struct GenericExchangeGuard<'a> {
    unusable: &'a mut bool,
    completed: bool,
}

impl Drop for GenericExchangeGuard<'_> {
    fn drop(&mut self) {
        if !self.completed {
            // A dropped future may have written a request or read a partial
            // reply. Fence reuse now; the next call tears down only this owned
            // child as lifecycle, not a crash charged to the later caller.
            *self.unusable = true;
        }
    }
}

#[derive(Default)]
struct WorkerLogContext {
    job_id: Option<String>,
}

struct WorkerForwardLimiter {
    cap: u32,
    window_started: Instant,
    forwarded: u32,
    dropped: u64,
}

impl WorkerForwardLimiter {
    fn new(cap: u32) -> Self {
        Self {
            cap,
            window_started: Instant::now(),
            forwarded: 0,
            dropped: 0,
        }
    }

    fn admit(&mut self, now: Instant) -> Option<u64> {
        if now.duration_since(self.window_started) >= Duration::from_secs(1) {
            self.window_started = now;
            self.forwarded = 0;
        }
        if self.forwarded >= self.cap {
            self.dropped = self.dropped.saturating_add(1);
            return None;
        }
        self.forwarded = self.forwarded.saturating_add(1);
        Some(std::mem::take(&mut self.dropped))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkerModelInfo {
    pub dims: usize,
    pub cold_load_ms: u64,
    pub buckets: Option<Vec<usize>>,
}

#[derive(Clone, Debug)]
struct LoadedWorkerModel {
    crash_key: String,
    artifact: ValidatedArtifact,
    runtime_config: RuntimeConfig,
    worker_model_ref: Option<String>,
    dims: usize,
    cold_load_ms: u64,
    buckets: Option<Vec<usize>>,
}

#[derive(Debug)]
struct OwnedDecodeAdapterState {
    generation_id: String,
    session_id: String,
    generated_ids: Vec<u32>,
    next_stream_sequence: u64,
    quantum_sequence: u32,
    max_tokens: u32,
    constraint_identity: Option<String>,
}

#[derive(Debug)]
enum OwnedDecodeCommandResponse {
    Frames(Vec<FrameEnvelope>),
    Cancelled(synapse_core::CancelledTransportResponse),
    HintBankInstalled {
        req_id: String,
        installation: HintBankInstalled,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct WorkerHostHealth {
    pub worker_connected: bool,
    pub tracked_models: usize,
    pub loaded_worker_models: usize,
    pub crash_count_window: usize,
    pub quarantined_models: usize,
    pub degraded: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub placement_share: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WorkerPing {
    pub rss_mb: u64,
    pub models_loaded: usize,
    pub placement_share: Option<f64>,
}

pub struct WorkerHost {
    config: WorkerHostConfig,
    connection: Option<WorkerConnection>,
    loaded_models: HashMap<String, LoadedWorkerModel>,
    quarantined: HashSet<String>,
    quarantine_until: HashMap<String, Instant>,
    #[cfg(test)]
    clock: Option<Instant>,
    #[cfg(test)]
    test_idle_window: Option<Duration>,
    #[cfg(test)]
    test_setup_window: Option<Duration>,
    // Consecutive compute faults survive reloads and health PINGs. Only a
    // fully drained successful serving call clears them, never caller expiry.
    crashes: HashMap<String, usize>,
    last_placement_share: Option<f64>,
    request_counter: u64,
    model_counter: u64,
    owned_decode_stream: Option<OwnedDecodeAdapterState>,
    log_context: Arc<Mutex<WorkerLogContext>>,
    forward_limiter: Arc<Mutex<WorkerForwardLimiter>>,
}

impl WorkerHost {
    pub fn new(config: WorkerHostConfig) -> Self {
        let forward_limiter = Arc::new(Mutex::new(WorkerForwardLimiter::new(
            config.worker_forward_lines_per_sec,
        )));
        Self {
            config,
            connection: None,
            loaded_models: HashMap::new(),
            quarantined: HashSet::new(),
            quarantine_until: HashMap::new(),
            #[cfg(test)]
            clock: None,
            #[cfg(test)]
            test_idle_window: None,
            #[cfg(test)]
            test_setup_window: None,
            crashes: HashMap::new(),
            last_placement_share: None,
            request_counter: 0,
            model_counter: 0,
            owned_decode_stream: None,
            log_context: Arc::new(Mutex::new(WorkerLogContext::default())),
            forward_limiter,
        }
    }

    pub fn request_count(&self) -> u64 {
        self.request_counter
    }

    /// Capability from the currently connected worker's validated HELLO.
    pub fn packed_capability(&self) -> Option<&synapse_core::PackedCapability> {
        self.connection
            .as_ref()
            .and_then(|connection| connection.packed.as_ref())
    }

    fn set_log_job_id(&self, job_id: Option<&str>) {
        if let Ok(mut context) = self.log_context.lock() {
            context.job_id = job_id.map(str::to_owned);
        }
    }

    pub async fn load_model(
        &mut self,
        artifact: &ValidatedArtifact,
        cfg: &RuntimeConfig,
    ) -> Result<LoadedModel, WorkerHostError> {
        let artifact_path = artifact_path(cfg)?;
        let crash_key = crash_key(
            &artifact_path,
            cfg,
            self.config
                .isolate_crash_key_by_worker_id
                .then_some(self.config.worker_id.as_str()),
        );
        if let Some(error) = self.quarantine_error(&crash_key) {
            return Err(error);
        }

        let (worker_model_ref, dims, cold_load_ms, buckets) = self
            .load_worker_model_ref(artifact, cfg, &crash_key)
            .await?;
        let stable_model_id = format!("host-model-{}", self.model_counter);
        self.model_counter = self.model_counter.saturating_add(1);
        self.loaded_models.insert(
            stable_model_id.clone(),
            LoadedWorkerModel {
                crash_key,
                artifact: artifact.clone(),
                runtime_config: cfg.clone(),
                worker_model_ref: Some(worker_model_ref),
                dims,
                cold_load_ms,
                buckets,
            },
        );
        Ok(LoadedModel {
            model_id: stable_model_id,
        })
    }

    pub async fn embed_batch(
        &mut self,
        model: &LoadedModel,
        batch: TokenBatch,
    ) -> Result<Vectors, WorkerHostError> {
        let (worker_model_ref, crash_key) = self.ensure_worker_model(model).await?;
        let (items, ids) = flatten_batch(&batch)?;
        let req_id = self.next_req_id("embed");
        let request = WorkerRequest::EmbedBatch {
            req_id: req_id.clone(),
            model_ref: worker_model_ref,
            pooling: self.config.pooling,
            normalize: self.config.normalize,
            items,
        };
        match self
            .send_request(request, Some(encode_i32_frame(&ids)), true)
            .await
        {
            Ok((
                WorkerResponse::Vectors {
                    req_id: got,
                    dims,
                    n,
                },
                Some(raw),
            )) => {
                ensure_req_id(&req_id, &got)?;
                decode_vectors(&raw, n, dims)
            }
            Ok((WorkerResponse::Err { code, msg, .. }, _)) => {
                Err(WorkerHostError::WorkerErr { code, msg })
            }
            Ok((other, _)) => Err(WorkerHostError::Protocol(format!(
                "EMBED_BATCH returned unexpected response {other:?}"
            ))),
            Err(error @ WorkerHostError::EngineCrashed { .. }) => {
                self.record_crash_and_maybe_restart(crash_key).await;
                Err(error)
            }
            Err(error) => Err(error),
        }
    }

    pub async fn rerank(
        &mut self,
        model: &LoadedModel,
        request: RerankRequest,
    ) -> Result<RerankScores, WorkerHostError> {
        if self.serves_owned_rerank() && !request.query.is_empty() {
            return Err(WorkerHostError::Protocol(
                "owned rerank lanes take fully composed sequences as candidates; query must be empty"
                    .to_string(),
            ));
        }
        let (worker_model_ref, crash_key) = self.ensure_worker_model(model).await?;
        let req_id = self.next_req_id("rerank");
        let (worker_request, ids) =
            self.rerank_request(req_id.clone(), worker_model_ref, &request)?;
        let wire_type = worker_request.wire_type();
        match self
            .send_request(worker_request, Some(encode_i32_frame(&ids)), true)
            .await
        {
            Ok((WorkerResponse::Scores { req_id: got }, Some(raw))) => {
                ensure_req_id(&req_id, &got)?;
                Ok(RerankScores {
                    scores: decode_f32_frame(&raw)
                        .map_err(|error| WorkerHostError::Protocol(error.to_string()))?,
                })
            }
            Ok((WorkerResponse::Err { code, msg, .. }, _)) => {
                Err(WorkerHostError::WorkerErr { code, msg })
            }
            Ok((other, _)) => Err(WorkerHostError::Protocol(format!(
                "{wire_type} returned unexpected response {other:?}"
            ))),
            Err(error @ WorkerHostError::EngineCrashed { .. }) => {
                self.record_crash_and_maybe_restart(crash_key).await;
                Err(error)
            }
            Err(error) => Err(error),
        }
    }

    /// Owned workers (CUDA, Vulkan, direct ANE) score `RERANK_SEQUENCES`;
    /// every other lane, llama included, keeps the segment-frame `RERANK`.
    fn serves_owned_rerank(&self) -> bool {
        self.config
            .engine_identity
            .as_ref()
            .is_some_and(|identity| is_owned_worker_hello_engine(&identity.engine))
    }

    /// Builds the rerank request for this lane and its raw i32 frame.
    ///
    /// Owned lanes receive each candidate as one fully composed sequence
    /// (query, candidate and every special or template token, built by the
    /// module from the manifest grammar), concatenated in candidate order. The
    /// llama lane receives the query segment followed by the candidate
    /// segments and assembles the pairs itself.
    fn rerank_request(
        &self,
        req_id: String,
        model_ref: String,
        request: &RerankRequest,
    ) -> Result<(WorkerRequest, Vec<i32>), WorkerHostError> {
        let owned = self.serves_owned_rerank();
        let mut ids = if owned {
            Vec::new()
        } else {
            token_ids_to_i32(&request.query)?
        };
        for candidate in &request.candidates {
            ids.extend(token_ids_to_i32(candidate)?);
        }
        let worker_request = if owned {
            WorkerRequest::RerankSequences {
                req_id,
                model_ref,
                sequences: request
                    .candidates
                    .iter()
                    .map(|sequence| WorkerSequence {
                        n_tokens: sequence.len(),
                    })
                    .collect(),
            }
        } else {
            WorkerRequest::Rerank {
                req_id,
                model_ref,
                query_n_tokens: request.query.len(),
                candidates: request
                    .candidates
                    .iter()
                    .map(|candidate| WorkerCandidate {
                        n_tokens: candidate.len(),
                    })
                    .collect(),
            }
        };
        Ok((worker_request, ids))
    }

    pub async fn generate(
        &mut self,
        model: &LoadedModel,
        request: GenerateRequest,
    ) -> Result<GenerateOutput, WorkerHostError> {
        let (worker_model_ref, crash_key) = self.ensure_worker_model(model).await?;
        let req_id = self.next_req_id("generate");
        let ids = token_ids_to_i32(&request.prompt)?;
        let worker_request = WorkerRequest::Generate {
            req_id: req_id.clone(),
            model_ref: worker_model_ref,
            max_tokens: request.max_tokens,
            grammar: request.grammar.clone(),
        };
        match self
            .send_request(worker_request, Some(encode_i32_frame(&ids)), false)
            .await
        {
            Ok((
                WorkerResponse::Text {
                    req_id: got,
                    text,
                    n_prompt,
                    n_gen,
                    finish_reason,
                    generated_token_ids,
                },
                _,
            )) => {
                ensure_req_id(&req_id, &got)?;
                Ok(GenerateOutput {
                    text,
                    finish_reason,
                    n_prompt,
                    n_gen,
                    generated_token_ids,
                })
            }
            Ok((WorkerResponse::Err { code, msg, .. }, _)) => {
                Err(WorkerHostError::WorkerErr { code, msg })
            }
            Ok((other, _)) => Err(WorkerHostError::Protocol(format!(
                "GENERATE returned unexpected response {other:?}"
            ))),
            Err(error @ WorkerHostError::EngineCrashed { .. }) => {
                self.record_crash_and_maybe_restart(crash_key).await;
                Err(error)
            }
            Err(error) => Err(error),
        }
    }

    /// Start one resident owned-decode generation. The generic host owns only
    /// process/nonce/framing supervision; the S3 supervisor remains the sole
    /// crash-budget and quarantine authority.
    pub async fn owned_decode_start(
        &mut self,
        model: &LoadedModel,
        mut start: GenerateStart,
    ) -> Result<(u64, Vec<FrameEnvelope>), WorkerHostError> {
        let (worker_model_ref, _) = self.ensure_worker_model(model).await?;
        start.loaded_model_ref = worker_model_ref;
        let worker_generation = self
            .connection
            .as_ref()
            .map(|connection| connection.worker_generation)
            .ok_or_else(|| {
                WorkerHostError::Protocol("owned worker is not connected".to_string())
            })?;
        let req_id = self.next_req_id("generate_start");
        self.owned_decode_stream = Some(OwnedDecodeAdapterState {
            generation_id: start.generation_id.clone(),
            session_id: start.generation_id.clone(),
            generated_ids: Vec::new(),
            next_stream_sequence: synapse_core::StreamSequence::FIRST.0,
            quantum_sequence: 1,
            max_tokens: start.max_tokens,
            constraint_identity: start
                .constraint
                .as_ref()
                .map(|constraint| constraint.constraint_fingerprint.clone()),
        });
        match self
            .send_owned_request(DecodeTransportRequest::GenerateStart {
                req_id,
                start: Box::new(start),
            })
            .await?
        {
            OwnedDecodeCommandResponse::Frames(frames) => Ok((worker_generation, frames)),
            other => Err(WorkerHostError::Protocol(format!(
                "GENERATE_START returned unexpected response {other:?}"
            ))),
        }
    }

    pub async fn owned_decode_continue(
        &mut self,
        continuation: GenerateContinue,
    ) -> Result<Vec<FrameEnvelope>, WorkerHostError> {
        let req_id = self.next_req_id("generate_continue");
        let adapter = self.owned_decode_stream.as_mut().ok_or_else(|| {
            WorkerHostError::Protocol("GENERATE_CONTINUE has no active generation".to_string())
        })?;
        adapter.quantum_sequence = adapter
            .quantum_sequence
            .checked_add(1)
            .ok_or_else(|| WorkerHostError::Protocol("quantum sequence overflow".to_string()))?;
        match self
            .send_owned_request(DecodeTransportRequest::GenerateContinue {
                req_id,
                continuation,
            })
            .await?
        {
            OwnedDecodeCommandResponse::Frames(frames) => Ok(frames),
            other => Err(WorkerHostError::Protocol(format!(
                "GENERATE_CONTINUE returned unexpected response {other:?}"
            ))),
        }
    }

    /// Install a ready, bounded sidecar bank while the worker is paused at a
    /// progress boundary. This reuses the resident worker transport rather than
    /// creating a second mailbox or worker session.
    pub async fn owned_decode_install_hint_bank(
        &mut self,
        installation: GenerateInstallHintBank,
    ) -> Result<HintBankInstalled, WorkerHostError> {
        let req_id = self.next_req_id("generate_install_hint_bank");
        match self
            .send_owned_request(DecodeTransportRequest::GenerateInstallHintBank {
                req_id: req_id.clone(),
                installation,
            })
            .await?
        {
            OwnedDecodeCommandResponse::HintBankInstalled {
                req_id: got,
                installation,
            } => {
                ensure_req_id(&req_id, &got)?;
                Ok(installation)
            }
            OwnedDecodeCommandResponse::Frames(frames) => {
                let frame = frames.into_iter().last().ok_or_else(|| {
                    WorkerHostError::Protocol(
                        "GENERATE_INSTALL_HINT_BANK returned no frames".to_string(),
                    )
                })?;
                Err(WorkerHostError::WorkerErr {
                    code: match frame.frame {
                        owned_decode_worker::protocol::WorkerFrame::Error { id } => id,
                        other => format!("unexpected_{other:?}"),
                    },
                    msg: "owned worker rejected sidecar hint bank".to_string(),
                })
            }
            other => Err(WorkerHostError::Protocol(format!(
                "GENERATE_INSTALL_HINT_BANK returned unexpected response {other:?}"
            ))),
        }
    }

    pub async fn owned_decode_cancel(
        &mut self,
        cancellation: GenerateCancel,
    ) -> Result<u32, WorkerHostError> {
        let req_id = self.next_req_id("generate_cancel");
        match self
            .send_owned_request(DecodeTransportRequest::GenerateCancel {
                req_id: req_id.clone(),
                cancellation,
            })
            .await?
        {
            OwnedDecodeCommandResponse::Cancelled(cancelled) => {
                ensure_req_id(&req_id, &cancelled.req_id)?;
                Ok(cancelled.committed_token_count)
            }
            OwnedDecodeCommandResponse::Frames(frames) => {
                let frame = frames.into_iter().last().ok_or_else(|| {
                    WorkerHostError::Protocol("GENERATE_CANCEL returned no frames".to_string())
                })?;
                Err(WorkerHostError::WorkerErr {
                    code: match frame.frame {
                        owned_decode_worker::protocol::WorkerFrame::Error { id } => id,
                        other => format!("unexpected_{other:?}"),
                    },
                    msg: "owned worker rejected cancellation".to_string(),
                })
            }
            other => Err(WorkerHostError::Protocol(format!(
                "GENERATE_CANCEL returned unexpected response {other:?}"
            ))),
        }
    }

    pub async fn unload(&mut self, model: &LoadedModel) -> Result<(), WorkerHostError> {
        self.close_unusable_connection().await;
        let Some(loaded) = self.loaded_models.remove(&model.model_id) else {
            return Ok(());
        };
        if self.config.catalog_lane.is_some() {
            self.quarantined.remove(&loaded.crash_key);
            self.quarantine_until.remove(&loaded.crash_key);
            self.crashes.remove(&loaded.crash_key);
        }
        let Some(worker_model_ref) = loaded.worker_model_ref else {
            return Ok(());
        };
        let req_id = self.next_req_id("unload");
        let request = WorkerRequest::Unload {
            req_id: req_id.clone(),
            model_ref: worker_model_ref,
        };
        match self.send_request(request, None, false).await {
            Ok((WorkerResponse::Unloaded { req_id: got }, _)) => {
                ensure_req_id(&req_id, &got)?;
                Ok(())
            }
            Ok((WorkerResponse::Err { code, msg, .. }, _)) => {
                Err(WorkerHostError::WorkerErr { code, msg })
            }
            Ok((other, _)) => Err(WorkerHostError::Protocol(format!(
                "UNLOAD returned unexpected response {other:?}"
            ))),
            Err(error @ WorkerHostError::EngineCrashed { .. }) => {
                self.record_crash_and_maybe_restart(loaded.crash_key).await;
                Err(error)
            }
            Err(error) => Err(error),
        }
    }

    pub async fn ping(&mut self) -> Result<WorkerPing, WorkerHostError> {
        let req_id = self.next_req_id("ping");
        let response = self
            .send_request(
                WorkerRequest::Ping {
                    req_id: req_id.clone(),
                },
                None,
                false,
            )
            .await;
        let (response, _) = match response {
            Ok(response) => response,
            Err(error @ WorkerHostError::EngineCrashed { .. }) => {
                // A placement ping is part of ANE probe execution. Treat a
                // dead worker like any other supervised request so its crash
                // budget and lazy restart state stay consistent.
                let crash_keys = self
                    .loaded_models
                    .values()
                    .map(|model| model.crash_key.clone())
                    .collect::<Vec<_>>();
                for key in crash_keys {
                    self.record_crash_and_maybe_restart(key).await;
                }
                return Err(error);
            }
            Err(error) => return Err(error),
        };
        match response {
            WorkerResponse::Pong {
                req_id: got,
                rss_mb,
                models_loaded,
                placement_share,
                ..
            } => {
                ensure_req_id(&req_id, &got)?;
                self.last_placement_share = placement_share;
                Ok(WorkerPing {
                    rss_mb,
                    models_loaded,
                    placement_share,
                })
            }
            WorkerResponse::Err { code, msg, .. } => Err(WorkerHostError::WorkerErr { code, msg }),
            other => Err(WorkerHostError::Protocol(format!(
                "PING returned unexpected response {other:?}"
            ))),
        }
    }

    fn next_req_id(&mut self, prefix: &str) -> String {
        let req_id = format!("{prefix}-{}", self.request_counter);
        self.request_counter += 1;
        req_id
    }

    pub fn model_info(&self, model_id: &str) -> Option<WorkerModelInfo> {
        self.loaded_models.get(model_id).map(|m| WorkerModelInfo {
            dims: m.dims,
            cold_load_ms: m.cold_load_ms,
            buckets: m.buckets.clone(),
        })
    }

    pub fn health_snapshot(&mut self) -> WorkerHostHealth {
        let crash_count_window = self.crashes.values().sum();
        let loaded_worker_models = self
            .loaded_models
            .values()
            .filter(|loaded| loaded.worker_model_ref.is_some())
            .count();
        WorkerHostHealth {
            worker_connected: self
                .connection
                .as_ref()
                .is_some_and(|connection| !connection.unusable),
            tracked_models: self.loaded_models.len(),
            loaded_worker_models,
            crash_count_window,
            quarantined_models: self.quarantined.len(),
            degraded: crash_count_window > 0 || !self.quarantined.is_empty(),
            placement_share: self.last_placement_share,
        }
    }

    async fn ensure_worker_model(
        &mut self,
        model: &LoadedModel,
    ) -> Result<(String, String), WorkerHostError> {
        self.close_unusable_connection().await;
        let loaded = self
            .loaded_models
            .get(&model.model_id)
            .cloned()
            .ok_or_else(|| {
                WorkerHostError::Protocol(format!("unknown loaded model {}", model.model_id))
            })?;
        if let Some(error) = self.quarantine_error(&loaded.crash_key) {
            return Err(error);
        }
        if self.connection.is_some() {
            if let Some(worker_model_ref) = loaded.worker_model_ref {
                return Ok((worker_model_ref, loaded.crash_key));
            }
        }

        let (worker_model_ref, dims, cold_load_ms, buckets) = self
            .load_worker_model_ref(&loaded.artifact, &loaded.runtime_config, &loaded.crash_key)
            .await?;
        if let Some(entry) = self.loaded_models.get_mut(&model.model_id) {
            entry.worker_model_ref = Some(worker_model_ref.clone());
            entry.dims = dims;
            entry.cold_load_ms = cold_load_ms;
            entry.buckets = buckets;
        }
        Ok((worker_model_ref, loaded.crash_key))
    }

    async fn load_worker_model_ref(
        &mut self,
        artifact: &ValidatedArtifact,
        cfg: &RuntimeConfig,
        crash_key: &str,
    ) -> Result<(String, usize, u64, Option<Vec<usize>>), WorkerHostError> {
        if let Some(error) = self.quarantine_error(crash_key) {
            return Err(error);
        }
        let artifact_path = artifact_path(cfg)?;
        let req_id = self.next_req_id("load");
        let request = WorkerRequest::Load {
            req_id: req_id.clone(),
            artifact_path: artifact_path.to_string_lossy().to_string(),
            artifact_digest: artifact.digest.clone(),
            format: artifact.format.clone(),
            runtime_config: cfg.values.clone(),
        };
        match self.send_request(request, None, false).await {
            Ok((
                WorkerResponse::Loaded {
                    req_id: got,
                    model_ref,
                    dims,
                    cold_load_ms,
                    buckets,
                },
                _,
            )) => {
                ensure_req_id(&req_id, &got)?;
                Ok((model_ref, dims, cold_load_ms, buckets))
            }
            Ok((WorkerResponse::Err { code, msg, .. }, _)) => {
                Err(WorkerHostError::WorkerErr { code, msg })
            }
            Ok((other, _)) => Err(WorkerHostError::Protocol(format!(
                "LOAD returned unexpected response {other:?}"
            ))),
            Err(error @ WorkerHostError::EngineCrashed { .. }) => {
                self.record_crash_and_maybe_restart(crash_key.to_string())
                    .await;
                Err(error)
            }
            Err(error) => Err(error),
        }
    }

    async fn ensure_worker(&mut self) -> Result<(), WorkerHostError> {
        self.close_unusable_connection().await;
        if self.connection.is_some() {
            return Ok(());
        }
        self.start_worker().await
    }

    async fn close_unusable_connection(&mut self) {
        if self
            .connection
            .as_ref()
            .is_some_and(|connection| connection.unusable)
        {
            self.kill_current().await;
        }
    }

    async fn start_worker(&mut self) -> Result<(), WorkerHostError> {
        let (endpoint, listener) =
            prepare_listener(&self.config.runtime_dir, &self.config.worker_id)?;
        let nonce = nonce_hex16();
        let mut command =
            synapse_core::without_launch_nonce_tokio(Command::new(&self.config.worker_bin));
        append_worker_spawn_args(&mut command, &endpoint, &nonce);
        command
            .env("SYNAPSE_WORKER_ID", &self.config.worker_id)
            .stderr(Stdio::piped())
            .stdout(Stdio::piped())
            .kill_on_drop(true);
        for arg in &self.config.extra_args {
            command.arg(arg);
        }
        if let Some(lock) = &self.config.inherited_lane_lock {
            command.stdin(ane_residency::inheritable_lock_stdio(lock)?);
        }
        let expected_engine = self
            .config
            .engine_identity
            .as_ref()
            .map(|identity| identity.engine.as_str());
        if let Some(expected_engine) = expected_engine {
            // The timeout test worker exercises several catalog engines; pass
            // the expected identity without weakening the host-side handshake.
            command.env("SYNAPSE_WORKER_EXPECTED_ENGINE", expected_engine);
        }
        if let Some(binding) = &self.config.expected_hello_binding {
            // Like the expected engine, these exist for the timeout test
            // worker, which echoes them into its HELLO. Real owned workers
            // send the digest and revision they were built with.
            command
                .env(
                    "SYNAPSE_WORKER_EXPECTED_MANIFEST_DIGEST",
                    &binding.manifest_digest,
                )
                .env(
                    "SYNAPSE_WORKER_EXPECTED_KERNEL_REVISION",
                    &binding.kernel_revision,
                );
        }
        let mut child = command.spawn().map_err(|error| {
            WorkerHostError::Protocol(format!(
                "spawn worker {}: {error}",
                self.config.worker_bin.display()
            ))
        })?;
        let logs = LogRing::new();
        let model_id = self
            .config
            .model_id
            .clone()
            .unwrap_or_else(|| self.config.worker_id.clone());
        if let Some(stderr) = child.stderr.take() {
            spawn_pipe_reader(
                "stderr",
                stderr,
                logs.clone(),
                self.config.worker_id.clone(),
                model_id.clone(),
                Arc::clone(&self.log_context),
                Arc::clone(&self.forward_limiter),
            );
        }
        if let Some(stdout) = child.stdout.take() {
            spawn_pipe_reader(
                "stdout",
                stdout,
                logs.clone(),
                self.config.worker_id.clone(),
                model_id,
                Arc::clone(&self.log_context),
                Arc::clone(&self.forward_limiter),
            );
        }

        let required_protocol_version =
            (self.config.crash_authority == CrashAuthority::OwnedDecodeSupervisor).then_some(2);
        let (stream, hello) = match accept_worker_handshake_with_hello(
            listener,
            &nonce,
            self.config.max_frame,
            self.config.handshake_timeout,
            expected_engine,
            required_protocol_version,
            self.config.expected_hello_binding.as_ref(),
        )
        .await
        {
            Ok(stream) => stream,
            Err(error) => {
                let _ = child.kill().await;
                let _ = child.wait().await;
                return Err(error.into());
            }
        };
        self.connection = Some(WorkerConnection {
            stream,
            child,
            logs,
            worker_generation: worker_generation_from_nonce(&nonce)?,
            packed: hello.packed,
            unusable: false,
        });
        Ok(())
    }

    async fn send_request(
        &mut self,
        request: WorkerRequest,
        raw: Option<Vec<u8>>,
        _expect_raw_response: bool,
    ) -> Result<(WorkerResponse, Option<Vec<u8>>), WorkerHostError> {
        self.ensure_worker().await?;
        let cuda = self
            .config
            .engine_identity
            .as_ref()
            .is_some_and(|identity| {
                identity.engine == synapse_core::worker_engine_names::CUDA_WORKER_ENGINE
            });
        let idle_override = {
            #[cfg(test)]
            {
                self.test_idle_window
                    .map(|inference| (inference, self.test_setup_window.unwrap_or(inference)))
            }
            #[cfg(not(test))]
            {
                None
            }
        };
        let connection = self.connection.as_mut().expect("connected worker");
        let generation = connection.worker_generation;
        let mut exchange_guard = GenericExchangeGuard {
            unusable: &mut connection.unusable,
            completed: false,
        };
        let result = liveness::exchange(
            &mut connection.stream,
            &request,
            raw.as_deref(),
            self.config.max_frame,
            cuda,
            idle_override,
        )
        .await;
        exchange_guard.completed = result.is_ok();
        drop(exchange_guard);
        match result {
            Ok(value) => {
                // Only drained serving success proves the compute path healthy.
                // Internal LOAD/reload, PING and shape setup must not hide a GPU
                // that crashes on every inference. Caller expiry does not reset
                // here; a detached owner's eventual serving success does.
                if matches!(
                    &value.0,
                    WorkerResponse::Vectors { .. }
                        | WorkerResponse::Scores { .. }
                        | WorkerResponse::Text { .. }
                ) && self
                    .connection
                    .as_ref()
                    .is_some_and(|c| c.worker_generation == generation)
                {
                    let model_ref = match &request {
                        WorkerRequest::EmbedBatch { model_ref, .. }
                        | WorkerRequest::EmbedPacked { model_ref, .. }
                        | WorkerRequest::Rerank { model_ref, .. }
                        | WorkerRequest::RerankSequences { model_ref, .. }
                        | WorkerRequest::Generate { model_ref, .. } => Some(model_ref),
                        _ => None,
                    };
                    if let Some(model_ref) = model_ref {
                        for model in self.loaded_models.values() {
                            if model.worker_model_ref.as_ref() == Some(model_ref) {
                                self.crashes.remove(&model.crash_key);
                            }
                        }
                    }
                }
                Ok(value)
            }
            Err(error) => {
                let stderr_tail = self.kill_current().await;
                Err(WorkerHostError::EngineCrashed {
                    stage: request_stage(&request).into(),
                    detail: format!("worker_generation={generation}: {error}"),
                    stderr_tail,
                })
            }
        }
    }

    async fn send_owned_request(
        &mut self,
        request: DecodeTransportRequest,
    ) -> Result<OwnedDecodeCommandResponse, WorkerHostError> {
        self.ensure_worker().await?;
        let max_frame = self.config.max_frame;
        let result = timeout(self.config.request_timeout, async {
            let connection = self
                .connection
                .as_mut()
                .expect("connection exists after ensure_worker");
            write_json(&mut connection.stream, &request, max_frame).await?;
            read_owned_decode_response(
                &mut connection.stream,
                max_frame,
                &request,
                &mut self.owned_decode_stream,
            )
            .await
        })
        .await;
        match result {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(error @ WorkerHostError::Protocol(_)))
            | Ok(Err(error @ WorkerHostError::ProtocolVersion { .. })) => Err(error),
            Ok(Err(error @ WorkerHostError::WorkerErr { .. })) => Err(error),
            Ok(Err(error)) => {
                let stderr_tail = self.kill_current().await;
                Err(WorkerHostError::EngineCrashed {
                    stage: "owned_decode_transport".to_string(),
                    detail: error.to_string(),
                    stderr_tail,
                })
            }
            Err(_) => {
                let stderr_tail = self.kill_current().await;
                Err(WorkerHostError::EngineCrashed {
                    stage: "timeout".to_string(),
                    detail: format!(
                        "owned decode request exceeded {} ms",
                        self.config.request_timeout.as_millis()
                    ),
                    stderr_tail,
                })
            }
        }
    }

    async fn kill_current(&mut self) -> String {
        let Some(mut connection) = self.connection.take() else {
            self.forget_worker_model_refs();
            return String::new();
        };
        let _ = connection.child.kill().await;
        let _ = connection.child.wait().await;
        self.forget_worker_model_refs();
        connection.logs.tail()
    }

    async fn record_crash_and_maybe_restart(&mut self, key: String) {
        if self.config.crash_authority == CrashAuthority::OwnedDecodeSupervisor {
            return;
        }
        let quarantined = self.record_crash(key);
        if !quarantined {
            let _ = self.start_worker().await;
        }
    }

    fn now(&self) -> Instant {
        #[cfg(test)]
        if let Some(now) = self.clock {
            return now;
        }
        Instant::now()
    }

    fn quarantine_error(&mut self, key: &str) -> Option<WorkerHostError> {
        if !self.quarantined.contains(key) {
            return None;
        }
        let retry_after_ms = if self.config.catalog_lane.is_some() {
            let until = *self
                .quarantine_until
                .get(key)
                .expect("catalog quarantine has expiry");
            let now = self.now();
            if now >= until {
                self.quarantined.remove(key);
                self.quarantine_until.remove(key);
                self.crashes.remove(key);
                return None;
            }
            Some(
                u64::try_from(until.duration_since(now).as_millis())
                    .unwrap_or(u64::MAX)
                    .max(1),
            )
        } else {
            None
        };
        Some(WorkerHostError::Quarantined {
            key: key.into(),
            retry_after_ms,
        })
    }

    fn record_crash(&mut self, key: String) -> bool {
        let now = self.now();
        let entries = self.crashes.entry(key.clone()).or_default();
        *entries += 1;
        if *entries >= self.config.crash_budget.max_crashes {
            if self.config.catalog_lane.is_some() {
                self.quarantine_until
                    .insert(key.clone(), now + CATALOG_QUARANTINE_COOLDOWN);
            }
            self.quarantined.insert(key);
            true
        } else {
            false
        }
    }

    fn forget_worker_model_refs(&mut self) {
        for loaded in self.loaded_models.values_mut() {
            loaded.worker_model_ref = None;
        }
    }
}

async fn read_owned_decode_response(
    stream: &mut WorkerTransportStream,
    max_frame: u32,
    request: &DecodeTransportRequest,
    state: &mut Option<OwnedDecodeAdapterState>,
) -> Result<OwnedDecodeCommandResponse, WorkerHostError> {
    let expected_req_id = match request {
        DecodeTransportRequest::GenerateStart { req_id, .. }
        | DecodeTransportRequest::GenerateContinue { req_id, .. }
        | DecodeTransportRequest::GenerateInstallHintBank { req_id, .. }
        | DecodeTransportRequest::GenerateCancel { req_id, .. } => req_id,
    };
    let mut frames = Vec::new();
    loop {
        let value: serde_json::Value = read_json(stream, max_frame).await?;
        if let Ok(envelope) = serde_json::from_value::<synapse_core::FrameEnvelope>(value.clone()) {
            let adapter = state.as_mut().ok_or_else(|| {
                WorkerHostError::Protocol("owned decode frame has no active generation".to_string())
            })?;
            if envelope.protocol != synapse_core::OWNED_DECODE_ENVELOPE_V2_SCHEMA
                || envelope.protocol_version
                    != synapse_core::OWNED_DECODE_ENVELOPE_V2_PROTOCOL_VERSION
                || envelope.req_id != *expected_req_id
                || envelope.session_id != adapter.session_id
                || envelope.stream_seq.0 != adapter.next_stream_sequence
            {
                return Err(WorkerHostError::Protocol(format!(
                    "invalid owned decode v2 frame header for request {expected_req_id}"
                )));
            }
            adapter.next_stream_sequence = adapter
                .next_stream_sequence
                .checked_add(1)
                .ok_or_else(|| WorkerHostError::Protocol("stream sequence overflow".to_string()))?;
            match envelope.frame {
                synapse_core::WorkerFrame::Progress { progress } => {
                    adapter
                        .generated_ids
                        .extend_from_slice(&progress.committed_token_ids);
                    let committed_token_count = u32::try_from(adapter.generated_ids.len())
                        .map_err(|_| {
                            WorkerHostError::Protocol("token count overflow".to_string())
                        })?;
                    if committed_token_count != progress.committed_token_count {
                        return Err(WorkerHostError::Protocol(
                            "owned decode progress accounting mismatch".to_string(),
                        ));
                    }
                    frames.push(FrameEnvelope::new(
                        owned_decode_worker::protocol::WorkerFrame::Progress(GenerateProgress {
                            generation_id: adapter.generation_id.clone(),
                            quantum_sequence: adapter.quantum_sequence,
                            committed_token_count,
                            boundary: progress.boundary,
                        }),
                    ));
                    if progress.boundary == ProgressBoundary::Yield {
                        return Ok(OwnedDecodeCommandResponse::Frames(frames));
                    }
                }
                synapse_core::WorkerFrame::Final { terminal } => {
                    if terminal.req_id != *expected_req_id
                        || terminal.session_id != adapter.session_id
                        || terminal.committed_token_count != adapter.generated_ids.len() as u32
                        || terminal.tokens_emitted != terminal.committed_token_count
                        || terminal.terminal_state != synapse_core::TerminalState::Completed
                    {
                        return Err(WorkerHostError::Protocol(
                            "owned decode terminal accounting mismatch".to_string(),
                        ));
                    }
                    let finish_reason = if terminal.committed_token_count >= adapter.max_tokens {
                        owned_decode_worker::protocol::FinishReason::MaxTokens
                    } else if adapter.constraint_identity.is_some() {
                        owned_decode_worker::protocol::FinishReason::GrammarComplete
                    } else {
                        owned_decode_worker::protocol::FinishReason::StopToken
                    };
                    frames.push(FrameEnvelope::new(
                        owned_decode_worker::protocol::WorkerFrame::Final(
                            owned_decode_worker::protocol::FinalResponse {
                                generation_id: adapter.generation_id.clone(),
                                generated_ids: adapter.generated_ids.clone(),
                                committed_token_count: terminal.committed_token_count,
                                decode_fingerprint: terminal.identity.decode_fingerprint.0,
                                runtime_config_digest: terminal.identity.runtime_config_digest,
                                worker_generation: terminal.identity.worker_generation,
                                finish_reason,
                                constraint_identity: adapter.constraint_identity.clone(),
                                constraint_complete: adapter.constraint_identity.is_some(),
                                last_completed_sequence: adapter.quantum_sequence,
                                hint_verification: Default::default(),
                            },
                        ),
                    ));
                    *state = None;
                    return Ok(OwnedDecodeCommandResponse::Frames(frames));
                }
                synapse_core::WorkerFrame::Error { terminal } => {
                    if terminal.req_id != *expected_req_id
                        || terminal.session_id != adapter.session_id
                        || terminal.committed_token_count != adapter.generated_ids.len() as u32
                        || terminal.tokens_emitted != terminal.committed_token_count
                    {
                        return Err(WorkerHostError::Protocol(
                            "owned decode error accounting mismatch".to_string(),
                        ));
                    }
                    let id = match terminal.terminal_state {
                        synapse_core::TerminalState::Aborted => DecodeError::Cancelled.as_str(),
                        synapse_core::TerminalState::ArtifactDisabled
                        | synapse_core::TerminalState::ArtifactRevoked => {
                            DecodeError::ArtifactPoisoned.as_str()
                        }
                        synapse_core::TerminalState::Failed
                        | synapse_core::TerminalState::Completed => {
                            DecodeError::ProtocolMismatch.as_str()
                        }
                    };
                    frames.push(FrameEnvelope::new(
                        owned_decode_worker::protocol::WorkerFrame::Error { id: id.to_string() },
                    ));
                    *state = None;
                    return Ok(OwnedDecodeCommandResponse::Frames(frames));
                }
            }
            continue;
        }

        if matches!(request, DecodeTransportRequest::GenerateCancel { .. }) {
            if let Ok(cancelled) =
                serde_json::from_value::<synapse_core::CancelledTransportResponse>(value.clone())
            {
                *state = None;
                return Ok(OwnedDecodeCommandResponse::Cancelled(cancelled));
            }
        }
        if matches!(
            request,
            DecodeTransportRequest::GenerateInstallHintBank { .. }
        ) {
            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct InstalledResponse {
                #[serde(rename = "type")]
                response_type: String,
                req_id: String,
                installation: HintBankInstalled,
            }
            if let Ok(installed) = serde_json::from_value::<InstalledResponse>(value.clone()) {
                if installed.response_type == "HINT_BANK_INSTALLED" {
                    return Ok(OwnedDecodeCommandResponse::HintBankInstalled {
                        req_id: installed.req_id,
                        installation: installed.installation,
                    });
                }
            }
        }
        if let Ok(WorkerResponse::Err { code, msg, .. }) =
            serde_json::from_value::<WorkerResponse>(value.clone())
        {
            return Err(WorkerHostError::WorkerErr { code, msg });
        }
        return Err(WorkerHostError::Protocol(format!(
            "worker returned malformed owned decode response: {value}"
        )));
    }
}

pub struct WorkerEngine {
    /// Present from construction until [`Drop`], which moves the runtime to a
    /// teardown thread: both `Runtime::block_on` and `Runtime::drop` panic on
    /// a thread that is currently driving another tokio runtime, and engine
    /// values dropped from module state teardown are dropped on exactly such
    /// a thread.
    runtime: Option<Runtime>,
    host: Arc<Mutex<WorkerHost>>,
    /// How long [`Drop`] waits for the teardown thread. Always
    /// [`TEARDOWN_BUDGET`] in production; a field only so a test can prove the
    /// wait happens without spending the real budget to do it.
    teardown_budget: Duration,
}

impl WorkerEngine {
    pub fn new(config: WorkerHostConfig) -> Result<Self, WorkerHostError> {
        let runtime = Runtime::new()
            .map_err(|error| WorkerHostError::Protocol(format!("create tokio runtime: {error}")))?;
        Ok(Self {
            runtime: Some(runtime),
            host: Arc::new(Mutex::new(WorkerHost::new(config))),
            teardown_budget: TEARDOWN_BUDGET,
        })
    }

    /// The engine runtime. Present until `Drop` takes it; unreachable after,
    /// since every caller holds `&self` to a live (not-yet-dropped) engine.
    fn runtime(&self) -> &Runtime {
        self.runtime
            .as_ref()
            .expect("worker engine runtime is present until drop")
    }

    fn lock_host(&self) -> Result<std::sync::MutexGuard<'_, WorkerHost>, WorkerHostError> {
        self.host
            .lock()
            .map_err(|_| WorkerHostError::Protocol("worker host mutex poisoned".to_string()))
    }

    pub fn request_count(&self) -> Result<(String, u64), WorkerHostError> {
        let host = self.lock_host()?;
        Ok((host.config.worker_id.clone(), host.request_count()))
    }

    pub fn health_snapshot(&self) -> Result<WorkerHostHealth, WorkerHostError> {
        let mut host = self.lock_host()?;
        Ok(host.health_snapshot())
    }

    pub fn ping(&self) -> Result<WorkerPing, WorkerHostError> {
        let mut host = self.lock_host()?;
        self.runtime().block_on(host.ping())
    }

    pub fn model_info(&self, model: &LoadedModel) -> Option<WorkerModelInfo> {
        let host = self.lock_host().ok()?;
        host.model_info(&model.model_id)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn insert_loaded_model_for_test(
        &self,
        model_id: String,
        dims: usize,
        cold_load_ms: u64,
        buckets: Option<Vec<usize>>,
    ) -> LoadedModel {
        let mut host = self.lock_host().expect("worker host locks");
        host.loaded_models.insert(
            model_id.clone(),
            LoadedWorkerModel {
                crash_key: "test-crash-key".to_string(),
                artifact: ValidatedArtifact {
                    digest: "sha256:test".to_string(),
                    format: "coreml".to_string(),
                },
                runtime_config: RuntimeConfig::default(),
                worker_model_ref: Some("worker-ref-0".to_string()),
                dims,
                cold_load_ms,
                buckets,
            },
        );
        LoadedModel { model_id }
    }

    #[cfg(all(test, unix))]
    pub(crate) fn attach_mock_worker_for_test<F>(&self, respond: F)
    where
        F: Fn(&WorkerRequest, Option<&[u8]>) -> (WorkerResponse, Option<Vec<u8>>) + Send + 'static,
    {
        self.runtime().block_on(async {
            tests::attach_mock_worker(&mut self.lock_host().unwrap(), respond);
        });
    }

    pub(crate) fn embed_batch_with_job(
        &self,
        model: &LoadedModel,
        batch: TokenBatch,
        job_id: Option<&str>,
    ) -> Result<Vectors, EngineError> {
        let mut host = self
            .lock_host()
            .map_err(|error| error.to_engine_error(EngineErrorStage::Inference))?;
        host.set_log_job_id(job_id);
        let result = self.runtime().block_on(host.embed_batch(model, batch));
        host.set_log_job_id(None);
        result.map_err(|error| error.to_engine_error(EngineErrorStage::Inference))
    }

    pub(crate) fn rerank_with_job(
        &self,
        model: &LoadedModel,
        request: RerankRequest,
        job_id: Option<&str>,
    ) -> Result<RerankScores, EngineError> {
        let mut host = self
            .lock_host()
            .map_err(|error| error.to_engine_error(EngineErrorStage::Inference))?;
        host.set_log_job_id(job_id);
        let result = self.runtime().block_on(host.rerank(model, request));
        host.set_log_job_id(None);
        result.map_err(|error| error.to_engine_error(EngineErrorStage::Inference))
    }

    pub(crate) fn generate_with_job(
        &self,
        model: &LoadedModel,
        request: GenerateRequest,
        job_id: Option<&str>,
    ) -> Result<GenerateOutput, EngineError> {
        let mut host = self
            .lock_host()
            .map_err(|error| error.to_engine_error(EngineErrorStage::Inference))?;
        host.set_log_job_id(job_id);
        let result = self.runtime().block_on(host.generate(model, request));
        host.set_log_job_id(None);
        result.map_err(|error| error.to_engine_error(EngineErrorStage::Inference))
    }

    fn owned_decode_start(
        &self,
        model: &LoadedModel,
        start: GenerateStart,
    ) -> Result<(u64, Vec<FrameEnvelope>), WorkerHostError> {
        let mut host = self.lock_host()?;
        self.runtime()
            .block_on(host.owned_decode_start(model, start))
    }

    fn owned_decode_continue(
        &self,
        continuation: GenerateContinue,
    ) -> Result<Vec<FrameEnvelope>, WorkerHostError> {
        let mut host = self.lock_host()?;
        self.runtime()
            .block_on(host.owned_decode_continue(continuation))
    }

    fn owned_decode_install_hint_bank(
        &self,
        installation: GenerateInstallHintBank,
    ) -> Result<HintBankInstalled, WorkerHostError> {
        let mut host = self.lock_host()?;
        self.runtime()
            .block_on(host.owned_decode_install_hint_bank(installation))
    }

    fn owned_decode_cancel(&self, cancellation: GenerateCancel) -> Result<u32, WorkerHostError> {
        let mut host = self.lock_host()?;
        self.runtime()
            .block_on(host.owned_decode_cancel(cancellation))
    }

    fn owned_decode_worker_generation(&self) -> Result<u64, WorkerHostError> {
        self.lock_host()?
            .connection
            .as_ref()
            .map(|connection| connection.worker_generation)
            .ok_or_else(|| {
                WorkerHostError::Protocol(
                    "owned-decode worker is not connected after model load".to_string(),
                )
            })
    }

    fn owned_decode_kill(&self) {
        if let Ok(mut host) = self.lock_host() {
            let _ = self.runtime().block_on(host.kill_current());
        }
    }

    /// Pause draining has already fenced inference. Confirm process exit before
    /// releasing the inherited hardware lock; an UNLOAD acknowledgement alone
    /// leaves the worker process (and its descriptor) alive.
    pub(crate) fn shutdown_for_pause(&self) -> Result<(), WorkerHostError> {
        let mut host = self.lock_host()?;
        self.runtime().block_on(async {
            if let Some(connection) = host.connection.as_mut() {
                if connection.child.try_wait()?.is_none() {
                    connection.child.start_kill()?;
                }
                connection.child.wait().await?;
            }
            host.connection.take();
            host.forget_worker_model_refs();
            host.config.inherited_lane_lock.take();
            Ok(())
        })
    }
}

/// How long a drop waits for its worker child to be killed and reaped.
///
/// `kill_current` sends a kill and awaits the child's exit, which a live worker
/// completes in milliseconds. The budget exists so a wedged child or a held host
/// mutex cannot stall shutdown indefinitely, not to accommodate a slow one.
const TEARDOWN_BUDGET: Duration = Duration::from_secs(5);

impl Drop for WorkerEngine {
    fn drop(&mut self) {
        let Some(runtime) = self.runtime.take() else {
            return;
        };
        let host = Arc::clone(&self.host);
        let (finished_tx, finished) = std::sync::mpsc::sync_channel(1);
        let teardown = move || {
            if let Ok(mut host) = host.lock() {
                let _ = runtime.block_on(host.kill_current());
            }
            drop(runtime);
            let _ = finished_tx.send(());
        };
        let budget = self.teardown_budget;
        if tokio::runtime::Handle::try_current().is_ok() {
            // Dropped on a thread that is driving a tokio runtime (module
            // state teardown): block_on here — or even dropping the engine
            // runtime — panics, and a panic in Drop aborts the rest of the
            // state teardown. Hand the blocking kill to a dedicated thread.
            std::thread::spawn(teardown);
            // Then WAIT for that thread, bounded. Spawning without waiting made
            // shutdown depend on the process outliving a thread nothing joins:
            // on an orderly exit the kill raced `main` returning, and the child
            // was reaped only because it independently exits on socket EOF.
            // That backstop works and is measured, but it leaves our own
            // teardown unobservable and reports nothing when it does not fire.
            // The teardown thread drives its OWN runtime, so waiting here cannot
            // deadlock against the runtime this thread is driving.
            if finished.recv_timeout(budget).is_err() {
                tracing::warn!(
                    target: "synapse.worker",
                    budget_ms = budget.as_millis() as u64,
                    "worker teardown did not finish within its budget; the child is left to exit on socket EOF"
                );
            }
        } else {
            teardown();
        }
    }
}

impl WorkerEngine {
    fn configured_identity(&self) -> Option<EngineIdentity> {
        self.host
            .lock()
            .ok()
            .and_then(|host| host.config.engine_identity.clone())
    }
}

impl EmbedEngine for WorkerEngine {
    fn identity(&self) -> EngineIdentity {
        if let Some(identity) = self.configured_identity() {
            return identity;
        }
        let mut build_flags = BTreeMap::new();
        build_flags.insert("risk_class".to_string(), "abort_capable".to_string());
        build_flags.insert(
            "transport".to_string(),
            worker_transport_label().to_string(),
        );
        EngineIdentity {
            engine: LLAMA_WORKER_ENGINE.to_string(),
            version: "protocol-v1".to_string(),
            build_flags,
        }
    }

    fn load(
        &mut self,
        artifact: &ValidatedArtifact,
        cfg: &RuntimeConfig,
    ) -> Result<LoadedModel, EngineError> {
        let mut host = self
            .lock_host()
            .map_err(|error| error.to_engine_error(EngineErrorStage::Load))?;
        self.runtime()
            .block_on(host.load_model(artifact, cfg))
            .map_err(|error| error.to_engine_error(EngineErrorStage::Load))
    }

    fn embed_batch(&self, model: &LoadedModel, batch: TokenBatch) -> Result<Vectors, EngineError> {
        self.embed_batch_with_job(model, batch, None)
    }

    fn embed_one(&self, model: &LoadedModel, ids: TokenIds) -> Result<Vector, EngineError> {
        let mut vectors = self.embed_batch(model, TokenBatch { items: vec![ids] })?;
        vectors.pop().ok_or_else(|| EngineError {
            stage: EngineErrorStage::Inference,
            risk_class: EngineRiskClass::AbortCapable,
            message: "worker returned no vector for single-item batch".to_string(),
            retry_after_ms: None,
            safe_to_retry_same_request: true,
        })
    }

    fn unload(&mut self, model: &LoadedModel) {
        if let Ok(mut host) = self.lock_host() {
            let _ = self.runtime().block_on(host.unload(model));
        }
    }
}

impl RerankEngine for WorkerEngine {
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
        self.rerank_with_job(model, request, None)
    }

    fn unload(&mut self, model: &LoadedModel) {
        <Self as EmbedEngine>::unload(self, model);
    }
}

impl GenerateEngine for WorkerEngine {
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

    fn generate(
        &self,
        model: &LoadedModel,
        request: GenerateRequest,
    ) -> Result<GenerateOutput, EngineError> {
        self.generate_with_job(model, request, None)
    }

    fn unload(&mut self, model: &LoadedModel) {
        <Self as EmbedEngine>::unload(self, model);
    }
}

/// Process factory consumed by the S3 owned-decode supervisor. Each spawn owns
/// a fresh transport session and reloads the immutable model key. The generic
/// host's rolling crash window is disabled, leaving the S3 store-backed budget
/// as the single quarantine authority.
#[derive(Clone)]
pub struct OwnedDecodeWorkerFactory {
    config: WorkerHostConfig,
    artifact: ValidatedArtifact,
    runtime_config: RuntimeConfig,
    idle: Arc<Mutex<Option<ReusableOwnedDecodeWorker>>>,
}

struct ReusableOwnedDecodeWorker {
    engine: WorkerEngine,
    model: LoadedModel,
    worker_generation: u64,
}

impl OwnedDecodeWorkerFactory {
    pub fn new(
        mut config: WorkerHostConfig,
        artifact: ValidatedArtifact,
        runtime_config: RuntimeConfig,
    ) -> Self {
        config.crash_authority = CrashAuthority::OwnedDecodeSupervisor;
        Self {
            config,
            artifact,
            runtime_config,
            idle: Arc::new(Mutex::new(None)),
        }
    }
}

/// The supervisor's clock for a production dispatch: wall-clock milliseconds
/// since the Unix epoch.
///
/// The supervisor reads one clock for two purposes: request deadlines, and the
/// crash budget's `quarantined_until`, which is persisted to disk and read back
/// by the routing precheck (with wall-clock `now_ms()`) and by later processes.
/// A value persisted from a per-process monotonic clock means nothing to any of
/// those readers, so the quarantine would be invisible to routing and lost on
/// restart. Deadlines are set from this same clock in `set_request`, so they
/// stay comparable with every reading the supervisor takes; the only cost is
/// that a wall-clock step during one request can move its deadline.
struct WallDispatchClock;

impl OwnedClock for WallDispatchClock {
    fn now(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|elapsed| elapsed.as_millis().try_into().unwrap_or(u64::MAX))
            .unwrap_or(0)
    }
}

/// Real routing dispatch for the owned lane. Routing fixes the selected lane;
/// this adapter drives the S3 supervisor, which in turn owns the only crash
/// budget, one-redispatch rule, and persistent quarantine key.
pub struct SupervisedDecodeDispatch {
    supervisor: Supervisor<FileBudgetStore>,
    factory: OwnedDecodeWorkerFactory,
    key: QuarantineKey,
    start: GenerateStart,
    context: WorkerStartContext,
    control: TerminalControl,
    clock: WallDispatchClock,
    hint_bank_source: Box<dyn HintBankSource + Send>,
}

impl SupervisedDecodeDispatch {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        factory: OwnedDecodeWorkerFactory,
        budget_store_path: impl AsRef<Path>,
        budget_policy: OwnedBudgetPolicy,
        production_n: u32,
        key: QuarantineKey,
        start: GenerateStart,
        context: WorkerStartContext,
        control: TerminalControl,
    ) -> io::Result<Self> {
        let budget_store = FileBudgetStore::open(budget_store_path)?;
        Ok(Self {
            supervisor: Supervisor::new(
                OwnedCrashBudget::new(budget_store, budget_policy),
                production_n,
            ),
            factory,
            key,
            start,
            context,
            control,
            clock: WallDispatchClock,
            hint_bank_source: Box::new(NoHintBankSource),
        })
    }

    /// Replace the request-local inputs while retaining the supervised worker
    /// pool and persistent crash-budget authority.
    pub fn set_request(
        &mut self,
        prompt_ids: Vec<u32>,
        constraint: Option<owned_decode_worker::protocol::TokenIdJsonConstraint>,
        deadline_ms: u64,
    ) {
        self.start.prompt_ids = prompt_ids;
        self.start.constraint.clone_from(&constraint);
        self.context.expected_constraint = constraint;
        self.control.deadline_at = Some(self.clock.now().saturating_add(deadline_ms));
        self.hint_bank_source = Box::new(NoHintBankSource);
    }

    /// Replace the request-local completion source without changing the worker
    /// pool or its crash-budget authority.
    pub fn set_hint_bank_source(&mut self, hint_bank_source: Box<dyn HintBankSource + Send>) {
        self.hint_bank_source = hint_bank_source;
    }

    #[must_use]
    pub fn crash_budget_remaining(&self) -> u32 {
        self.supervisor.budget().remaining(&self.key)
    }

    #[must_use]
    pub fn is_quarantined(&self) -> bool {
        self.supervisor
            .budget()
            .is_quarantined(&self.key, self.clock.now())
    }
}

impl crate::owned_decode_routing::DecodeDispatch for SupervisedDecodeDispatch {
    fn dispatch(
        &mut self,
        command: &crate::owned_decode_routing::DispatchedCommand,
    ) -> Result<
        crate::owned_decode_routing::ExecutionSuccess,
        crate::owned_decode_routing::error::OwnedDecodeError,
    > {
        if command.lane != crate::owned_decode_routing::lane::LaneKind::OwnedDecode {
            return Err(crate::owned_decode_routing::error::OwnedDecodeError::Unsupported);
        }
        self.start.generation_id.clone_from(&command.generation_id);
        self.start
            .decode_fingerprint
            .clone_from(&command.decode_fingerprint.0);
        self.start.max_tokens = command.max_tokens;
        let request = OwnedGenerationRequest {
            key: self.key.clone(),
            start: self.start.clone(),
        };
        let outcome = self.supervisor.run_generation_with_hint_bank(
            &request,
            &mut self.factory,
            &self.context,
            &self.control,
            &self.clock,
            &mut *self.hint_bank_source,
        );
        let success = outcome.result.map_err(map_decode_error)?;
        Ok(crate::owned_decode_routing::ExecutionSuccess {
            generated_token_ids: success.generated_ids,
            finish_reason: map_finish_reason(success.finish_reason),
            lane_finish_reason: None,
            worker_generation: success.worker_generation,
            last_completed_quantum_sequence: success.last_completed_sequence,
            crash_retry_count: outcome.provenance.crash_retry_count,
            failure_classifications: outcome
                .provenance
                .failure_classifications
                .into_iter()
                .map(|classification| classification.as_str().to_string())
                .collect(),
        })
    }
}

fn map_finish_reason(
    finish_reason: owned_decode_worker::protocol::FinishReason,
) -> crate::owned_decode_routing::provenance::FinishReason {
    match finish_reason {
        owned_decode_worker::protocol::FinishReason::StopToken => {
            crate::owned_decode_routing::provenance::FinishReason::StopToken
        }
        owned_decode_worker::protocol::FinishReason::MaxTokens => {
            crate::owned_decode_routing::provenance::FinishReason::MaxTokens
        }
        owned_decode_worker::protocol::FinishReason::GrammarComplete => {
            crate::owned_decode_routing::provenance::FinishReason::GrammarComplete
        }
        owned_decode_worker::protocol::FinishReason::Cancelled => {
            crate::owned_decode_routing::provenance::FinishReason::Cancelled
        }
    }
}

fn map_decode_error(error: DecodeError) -> crate::owned_decode_routing::error::OwnedDecodeError {
    use crate::owned_decode_routing::error::OwnedDecodeError as RoutingError;
    match error {
        DecodeError::NotCertified => RoutingError::NotCertified,
        DecodeError::CertificationFailed => RoutingError::CertificationFailed,
        DecodeError::Quarantined => RoutingError::Quarantined,
        DecodeError::ArtifactPoisoned => RoutingError::ArtifactPoisoned,
        DecodeError::Unavailable => RoutingError::Unavailable,
        DecodeError::Unsupported => RoutingError::Unsupported,
        DecodeError::ProtocolMismatch => RoutingError::ProtocolMismatch,
        DecodeError::RuntimeConfigMismatch => RoutingError::RuntimeConfigMismatch,
        DecodeError::ConstraintVersionMismatch => RoutingError::ConstraintVersionMismatch,
        DecodeError::SamplingUnsupported => RoutingError::SamplingUnsupported,
        DecodeError::ContextCapacityExceeded => RoutingError::ContextCapacityExceeded,
        DecodeError::GrammarDisabled => RoutingError::GrammarDisabled,
        DecodeError::GrammarParseFailed => RoutingError::GrammarParseFailed,
        DecodeError::GrammarFeatureUnsupported => RoutingError::GrammarFeatureUnsupported,
        DecodeError::GrammarUnsatisfiable => RoutingError::GrammarUnsatisfiable,
        DecodeError::GrammarStopBeforeCompletion => RoutingError::GrammarStopBeforeCompletion,
        DecodeError::GrammarMaxTokensExhausted => RoutingError::GrammarMaxTokensExhausted,
        DecodeError::DeadlineExceeded => RoutingError::DeadlineExceeded,
        DecodeError::Cancelled => RoutingError::Cancelled,
    }
}

/// A startup failure charges the lane's crash budget and the request is refused
/// as `owned_decode_unavailable`, which says nothing about the cause. The error is
/// logged here, where it still exists, so an operator reading the log can tell a
/// missing artifact from a failed spawn or a worker that died during load.
fn decode_startup_failure(
    stage: &'static str,
    artifact_digest: &str,
    error: &dyn std::fmt::Debug,
) -> WorkerFault {
    tracing::warn!(
        target: "worker",
        stage,
        artifact_digest,
        error = ?error,
        "owned decode worker failed to start"
    );
    WorkerFault::StartupFailure
}

struct OwnedDecodeWorkerSession {
    engine: Option<WorkerEngine>,
    model: LoadedModel,
    worker_generation: u64,
    pending: VecDeque<owned_decode_worker::protocol::WorkerFrame>,
    idle: Arc<Mutex<Option<ReusableOwnedDecodeWorker>>>,
    reusable: bool,
}

impl WorkerFactory for OwnedDecodeWorkerFactory {
    fn spawn(&mut self) -> Result<Box<dyn DecodeWorker>, WorkerFault> {
        if let Some(worker) = self.idle.lock().ok().and_then(|mut idle| idle.take()) {
            return Ok(Box::new(OwnedDecodeWorkerSession {
                engine: Some(worker.engine),
                model: worker.model,
                worker_generation: worker.worker_generation,
                pending: VecDeque::new(),
                idle: Arc::clone(&self.idle),
                reusable: true,
            }));
        }
        let digest = self.artifact.digest.as_str();
        let mut engine = WorkerEngine::new(self.config.clone())
            .map_err(|error| decode_startup_failure("spawn the worker", digest, &error))?;
        let model = GenerateEngine::load(&mut engine, &self.artifact, &self.runtime_config)
            .map_err(|error| decode_startup_failure("load the model", digest, &error))?;
        let worker_generation = engine.owned_decode_worker_generation().map_err(|error| {
            decode_startup_failure("read the worker generation", digest, &error)
        })?;
        Ok(Box::new(OwnedDecodeWorkerSession {
            engine: Some(engine),
            model,
            worker_generation,
            pending: VecDeque::new(),
            idle: Arc::clone(&self.idle),
            reusable: true,
        }))
    }
}

impl DecodeWorker for OwnedDecodeWorkerSession {
    fn worker_generation(&self) -> u64 {
        self.worker_generation
    }

    fn start(
        &mut self,
        start: &GenerateStart,
        context: &WorkerStartContext,
        production_n: u32,
    ) -> Result<StartAuthorization, WorkerStartFailure> {
        let authorization =
            owned_decode_worker::validation::validate_start(start, context, production_n)
                .map_err(WorkerStartFailure::from)?;
        let response = self
            .engine
            .as_ref()
            .ok_or(WorkerStartFailure::Fault(WorkerFault::Crash))?
            .owned_decode_start(&self.model, start.clone());
        let (worker_generation, envelopes) = match response {
            Ok(response) => response,
            Err(error) => {
                self.reusable = matches!(
                    error,
                    WorkerHostError::Protocol(_)
                        | WorkerHostError::ProtocolVersion { .. }
                        | WorkerHostError::Json(_)
                        | WorkerHostError::WorkerErr { .. }
                );
                return Err(owned_host_start_failure(&error));
            }
        };
        for envelope in &envelopes {
            owned_decode_worker::protocol::validate_frame_structure(envelope)?;
        }
        if worker_generation != self.worker_generation {
            self.reusable = false;
            return Err(WorkerStartFailure::Fault(WorkerFault::Crash));
        }
        self.pending
            .extend(envelopes.into_iter().map(|envelope| envelope.frame));
        Ok(authorization)
    }

    fn step(&mut self) -> Result<SteppedFrame, WorkerFault> {
        let frame = self.pending.pop_front().ok_or(WorkerFault::Crash)?;
        Ok(SteppedFrame {
            worker_generation: self.worker_generation,
            frame,
        })
    }

    fn install_hint_bank(
        &mut self,
        installation: &GenerateInstallHintBank,
    ) -> Result<(), WorkerFault> {
        let response = self
            .engine
            .as_ref()
            .ok_or(WorkerFault::Crash)?
            .owned_decode_install_hint_bank(installation.clone());
        let installed = match response {
            Ok(installed) => installed,
            Err(error) => {
                self.reusable = matches!(
                    error,
                    WorkerHostError::Protocol(_) | WorkerHostError::ProtocolVersion { .. }
                );
                return Err(owned_host_fault(&error));
            }
        };
        if installed.generation_id != installation.generation_id
            || installed.bank_content_digest != installation.bank.content_digest()
        {
            self.reusable = false;
            return Err(WorkerFault::Crash);
        }
        Ok(())
    }

    fn send_continue(&mut self, continuation: &GenerateContinue) -> Result<(), WorkerFault> {
        let response = self
            .engine
            .as_ref()
            .ok_or(WorkerFault::Crash)?
            .owned_decode_continue(continuation.clone());
        let envelopes = match response {
            Ok(envelopes) => envelopes,
            Err(error) => {
                self.reusable = matches!(
                    error,
                    WorkerHostError::Protocol(_) | WorkerHostError::ProtocolVersion { .. }
                );
                return Err(owned_host_fault(&error));
            }
        };
        for envelope in &envelopes {
            owned_decode_worker::protocol::validate_frame_structure(envelope)
                .map_err(|_| WorkerFault::Crash)?;
        }
        self.pending
            .extend(envelopes.into_iter().map(|envelope| envelope.frame));
        Ok(())
    }

    fn send_cancel(&mut self, cancellation: &GenerateCancel) -> Result<CancelAck, WorkerFault> {
        let response = self
            .engine
            .as_ref()
            .ok_or(WorkerFault::FailedCancellation)?
            .owned_decode_cancel(cancellation.clone());
        let committed_token_count = match response {
            Ok(committed_token_count) => committed_token_count,
            Err(
                error @ (WorkerHostError::Protocol(_) | WorkerHostError::ProtocolVersion { .. }),
            ) => {
                self.reusable = true;
                return Err(owned_host_fault(&error));
            }
            Err(_) => {
                self.reusable = false;
                return Err(WorkerFault::FailedCancellation);
            }
        };
        Ok(CancelAck::Acknowledged {
            committed_token_count,
        })
    }

    fn kill(&mut self) {
        self.reusable = false;
        if let Some(engine) = self.engine.as_ref() {
            engine.owned_decode_kill();
        }
        self.pending.clear();
    }
}

impl Drop for OwnedDecodeWorkerSession {
    fn drop(&mut self) {
        let Some(engine) = self.engine.take() else {
            return;
        };
        if !self.reusable {
            return;
        }
        if let Ok(mut idle) = self.idle.lock() {
            *idle = Some(ReusableOwnedDecodeWorker {
                engine,
                model: self.model.clone(),
                worker_generation: self.worker_generation,
            });
        }
    }
}

fn owned_host_start_failure(error: &WorkerHostError) -> WorkerStartFailure {
    match error {
        WorkerHostError::Protocol(_)
        | WorkerHostError::ProtocolVersion { .. }
        | WorkerHostError::Json(_) => WorkerStartFailure::Typed(DecodeError::ProtocolMismatch),
        WorkerHostError::WorkerErr { code, .. } => WorkerStartFailure::Typed(
            DecodeError::from_id(code).unwrap_or(DecodeError::ProtocolMismatch),
        ),
        _ => WorkerStartFailure::Fault(owned_host_fault(error)),
    }
}

fn owned_host_fault(error: &WorkerHostError) -> WorkerFault {
    match error {
        WorkerHostError::EngineCrashed { stage, .. } if stage == "timeout" => WorkerFault::Timeout,
        WorkerHostError::Protocol(_)
        | WorkerHostError::ProtocolVersion { .. }
        | WorkerHostError::Json(_)
        | WorkerHostError::WorkerErr { .. }
        | WorkerHostError::HelloRefused { .. } => WorkerFault::Protocol,
        WorkerHostError::EngineCrashed { .. }
        | WorkerHostError::Io(_)
        | WorkerHostError::Quarantined { .. } => WorkerFault::Crash,
    }
}

fn spawn_pipe_reader<R>(
    stream: &'static str,
    mut reader: R,
    ring: LogRing,
    worker_id: String,
    model_id: String,
    context: Arc<Mutex<WorkerLogContext>>,
    limiter: Arc<Mutex<WorkerForwardLimiter>>,
) where
    R: AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let prefix = format!("[{stream}] ");
        let mut buffer = [0_u8; 512];
        let mut partial = Vec::new();
        loop {
            match reader.read(&mut buffer).await {
                Ok(0) => break,
                Ok(n) => {
                    ring.push(prefix.as_bytes());
                    ring.push(&buffer[..n]);
                    partial.extend_from_slice(&buffer[..n]);
                    while let Some(newline) = partial.iter().position(|byte| *byte == b'\n') {
                        let mut complete = partial.drain(..=newline).collect::<Vec<_>>();
                        complete.pop();
                        if complete.last() == Some(&b'\r') {
                            complete.pop();
                        }
                        forward_worker_line(
                            &worker_id,
                            stream,
                            &model_id,
                            &String::from_utf8_lossy(&complete),
                            &context,
                            &limiter,
                        );
                    }
                }
                Err(_) => break,
            }
        }
    });
}

fn forward_worker_line(
    worker_id: &str,
    stream: &str,
    model_id: &str,
    line: &str,
    context: &Mutex<WorkerLogContext>,
    limiter: &Mutex<WorkerForwardLimiter>,
) {
    let dropped = match limiter.lock() {
        Ok(mut limiter) => match limiter.admit(Instant::now()) {
            Some(dropped) => dropped,
            None => return,
        },
        Err(_) => return,
    };
    let job_id = context
        .lock()
        .ok()
        .and_then(|context| context.job_id.clone());
    match (job_id.as_deref(), dropped) {
        (Some(job_id), 0) => tracing::info!(
            target: "worker",
            worker = worker_id,
            stream,
            model_id,
            job_id,
            "{line}"
        ),
        (Some(job_id), dropped) => tracing::info!(
            target: "worker",
            worker = worker_id,
            stream,
            model_id,
            job_id,
            dropped,
            "{line}"
        ),
        (None, 0) => tracing::info!(
            target: "worker",
            worker = worker_id,
            stream,
            model_id,
            "{line}"
        ),
        (None, dropped) => tracing::info!(
            target: "worker",
            worker = worker_id,
            stream,
            model_id,
            dropped,
            "{line}"
        ),
    }
}

fn worker_transport_label() -> &'static str {
    if cfg!(windows) {
        "named-pipe-worker"
    } else {
        "unix-socket-worker"
    }
}

#[cfg(unix)]
fn append_worker_spawn_args(command: &mut Command, endpoint: &Path, nonce: &str) {
    command
        .arg("--socket")
        .arg(endpoint)
        .arg("--nonce")
        .arg(nonce);
}

#[cfg(windows)]
fn append_worker_spawn_args(command: &mut Command, endpoint: &str, nonce: &str) {
    command
        .arg("--pipe")
        .arg(endpoint)
        .arg("--nonce")
        .arg(nonce);
}

impl From<TransportError> for WorkerHostError {
    fn from(error: TransportError) -> Self {
        match error {
            TransportError::Io(error) => Self::Io(error),
            TransportError::Protocol(message) => Self::Protocol(message),
            TransportError::UnsupportedProtocolVersion {
                advertised,
                required,
            } => Self::ProtocolVersion {
                advertised,
                required,
            },
            TransportError::HelloBindingMismatch(mismatch) => Self::HelloRefused {
                code: mismatch.code.to_string(),
                msg: mismatch.to_string(),
            },
        }
    }
}

fn ensure_req_id(expected: &str, got: &str) -> Result<(), WorkerHostError> {
    if expected == got {
        Ok(())
    } else {
        Err(WorkerHostError::Protocol(format!(
            "response req_id mismatch: expected {expected}, got {got}"
        )))
    }
}

fn artifact_path(cfg: &RuntimeConfig) -> Result<PathBuf, WorkerHostError> {
    cfg.values
        .get("artifact_path")
        .or_else(|| cfg.values.get("model_path"))
        .map(PathBuf::from)
        .ok_or_else(|| {
            WorkerHostError::Protocol(
                "worker load requires runtime_config artifact_path or model_path".to_string(),
            )
        })
}

fn crash_key(path: &Path, cfg: &RuntimeConfig, worker_id: Option<&str>) -> String {
    let mut values = cfg.values.clone();
    values.insert(
        "artifact_path".to_string(),
        path.to_string_lossy().to_string(),
    );
    if let Some(worker_id) = worker_id {
        values.insert("stable_model_worker_id".to_string(), worker_id.to_string());
    }
    serde_json::to_string(&values).unwrap_or_else(|_| path.to_string_lossy().to_string())
}

fn flatten_batch(batch: &TokenBatch) -> Result<(Vec<WorkerTokenItem>, Vec<i32>), WorkerHostError> {
    let mut items = Vec::with_capacity(batch.items.len());
    let mut ids = Vec::new();
    for (index, token_ids) in batch.items.iter().enumerate() {
        items.push(WorkerTokenItem {
            id: index.to_string(),
            n_tokens: token_ids.len(),
        });
        ids.extend(token_ids_to_i32(token_ids)?);
    }
    Ok((items, ids))
}

fn token_ids_to_i32(token_ids: &[u32]) -> Result<Vec<i32>, WorkerHostError> {
    token_ids
        .iter()
        .copied()
        .map(|token| {
            i32::try_from(token).map_err(|_| {
                WorkerHostError::Protocol(format!("token id {token} does not fit into i32"))
            })
        })
        .collect()
}

fn decode_vectors(raw: &[u8], n: usize, dims: usize) -> Result<Vectors, WorkerHostError> {
    let flat =
        decode_f32_frame(raw).map_err(|error| WorkerHostError::Protocol(error.to_string()))?;
    let expected = n
        .checked_mul(dims)
        .ok_or_else(|| WorkerHostError::Protocol("vector shape overflow".to_string()))?;
    if flat.len() != expected {
        return Err(WorkerHostError::Protocol(format!(
            "raw vector frame has {} floats, expected {expected}",
            flat.len()
        )));
    }
    Ok(flat
        .chunks(dims)
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>())
}

fn request_stage(request: &WorkerRequest) -> &'static str {
    match request {
        WorkerRequest::Load { .. } => "load",
        WorkerRequest::EmbedBatch { .. } => "embed_batch",
        WorkerRequest::EmbedPacked { .. } => "embed_packed",
        WorkerRequest::AneAdmitPacked { .. } => "ane_admit_packed",
        WorkerRequest::AneEvictPacked { .. } => "ane_evict_packed",
        WorkerRequest::Rerank { .. } => "rerank",
        WorkerRequest::RerankSequences { .. } => "rerank_sequences",
        WorkerRequest::AneAdmitShape { .. } => "ane_admit_shape",
        WorkerRequest::AneEvictShape { .. } => "ane_evict_shape",
        WorkerRequest::Generate { .. } => "generate",
        WorkerRequest::Unload { .. } => "unload",
        WorkerRequest::Ping { .. } => "ping",
        WorkerRequest::Shutdown {} => "shutdown",
    }
}

fn nonce_hex16() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0);
    let value = now ^ u64::from(std::process::id()) ^ COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{value:016x}")
}

fn worker_generation_from_nonce(nonce: &str) -> Result<u64, WorkerHostError> {
    if nonce.len() != 16 || !nonce.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(WorkerHostError::Protocol(
            "worker nonce must be 8-byte hex".to_string(),
        ));
    }
    u64::from_str_radix(nonce, 16)
        .map_err(|error| WorkerHostError::Protocol(format!("parse worker generation: {error}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[cfg(unix)]
    fn pause_shutdown_reaps_coreml_worker_before_releasing_inherited_lock() {
        const PROBE: &str = "SYNAPSE_PAUSE_SHUTDOWN_LOCK_PROBE";
        if std::env::var_os(PROBE).is_none() {
            // Isolate the file descriptor from concurrent test threads that fork.
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "worker_host::tests::pause_shutdown_reaps_coreml_worker_before_releasing_inherited_lock", "--test-threads=1"])
                .env(PROBE, "1").output().unwrap();
            if !output.status.success() {
                let combined = format!(
                    "{}{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
                // Lead with the child's own panic message: the full libtest output
                // starts with "running 1 test", which hides why the child failed.
                let reason = combined
                    .lines()
                    .skip_while(|line| !line.contains("panicked at"))
                    .nth(1)
                    .unwrap_or("isolated child failed without a panic message");
                panic!("{reason}\n{combined}");
            }
            return;
        }
        use ane_residency::{inheritable_lock_stdio, AneDirectLaneLock};
        let root = std::env::temp_dir().join(format!("synapse-pause-shutdown-{}", nonce_hex16()));
        let path = root.join("ane.lock");
        let holder = AneDirectLaneLock::acquire(&path).unwrap();
        let mut config = WorkerHostConfig::new("unused-worker", &root);
        config.inherited_lane_lock = Some(holder.inheritable());
        let engine = WorkerEngine::new(config).unwrap();
        let peer = engine.runtime().block_on(async {
            let (stream, peer) = tokio::net::UnixStream::pair().unwrap();
            let child = Command::new("sleep")
                .arg("30")
                .kill_on_drop(true)
                .stdin(inheritable_lock_stdio(&holder.inheritable()).unwrap())
                .spawn()
                .unwrap();
            engine.lock_host().unwrap().connection = Some(WorkerConnection {
                stream,
                child,
                logs: LogRing::new(),
                worker_generation: 1,
                packed: None,
                unusable: false,
            });
            peer
        });
        drop(holder);
        assert!(AneDirectLaneLock::acquire(&path).is_err());
        engine.shutdown_for_pause().unwrap();
        let replacement = AneDirectLaneLock::acquire(&path)
            .expect("confirmed exit must free the inherited descriptor");
        drop((peer, engine, replacement));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn nonce_is_hex16() {
        let nonce = nonce_hex16();
        assert_eq!(nonce.len(), 16);
        assert!(nonce.chars().all(|ch| ch.is_ascii_hexdigit()));
    }

    /// Dropping a `WorkerEngine` from a thread that is driving a tokio
    /// runtime must not panic. The fleet hit exactly this: module state
    /// teardown under `serve_with_handle` dropped engine values held in
    /// collections, and the old Drop called `Runtime::block_on` (and then
    /// implicitly `Runtime::drop`) on the async worker thread — both panic
    /// there, and a panic in Drop truncates the rest of state teardown.
    #[tokio::test]
    async fn worker_engine_drop_inside_async_context_does_not_panic() {
        let engine =
            WorkerEngine::new(WorkerHostConfig::new("unused-worker", std::env::temp_dir()))
                .expect("engine constructs");
        // Directly dropping in the async context reproduces the fleet panic
        // with the old Drop; with the teardown-thread Drop it must succeed.
        drop(engine);
    }

    /// The engine must WAIT for its teardown thread, not merely start it.
    ///
    /// Holding the host mutex stalls teardown deterministically, which stands in
    /// for the real stall (a wedged child that will not reap). If `Drop` returns
    /// promptly the wait is gone, and shutdown is back to racing `main` with the
    /// child reaped only by socket EOF. The assertion is on ELAPSED TIME rather
    /// than on an error, because a drop that skipped the wait still succeeds.
    #[tokio::test]
    async fn worker_engine_drop_waits_for_its_teardown_thread() {
        let budget = Duration::from_millis(400);
        let mut engine =
            WorkerEngine::new(WorkerHostConfig::new("unused-worker", std::env::temp_dir()))
                .expect("engine constructs");
        engine.teardown_budget = budget;
        let host = Arc::clone(&engine.host);

        let held = host.lock().expect("host mutex");
        let started = Instant::now();
        drop(engine);
        let waited = started.elapsed();
        drop(held);

        assert!(
            waited >= budget,
            "drop returned in {waited:?} without waiting out its {budget:?} budget, so the \
             teardown thread was started and abandoned"
        );
        // An upper bound as well: waiting materially past the budget would mean
        // the bound is not doing its job either.
        assert!(
            waited < budget * 5,
            "drop waited {waited:?}, far past its {budget:?} budget"
        );
    }

    /// The companion: with nothing stalling teardown, the drop must return
    /// quickly rather than always paying the budget. Without this, the test
    /// above would pass over an implementation that slept unconditionally.
    #[tokio::test]
    async fn worker_engine_drop_returns_promptly_when_teardown_is_free() {
        let mut engine =
            WorkerEngine::new(WorkerHostConfig::new("unused-worker", std::env::temp_dir()))
                .expect("engine constructs");
        engine.teardown_budget = Duration::from_secs(30);

        let started = Instant::now();
        drop(engine);
        let waited = started.elapsed();

        assert!(
            waited < Duration::from_secs(5),
            "an unobstructed teardown took {waited:?}; drop is paying its budget rather than \
             waiting for completion"
        );
    }

    fn wall_now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|elapsed| elapsed.as_millis().try_into().unwrap_or(u64::MAX))
            .unwrap_or(0)
    }

    fn owned_quarantine_key() -> QuarantineKey {
        QuarantineKey::new("clock-profile", "clock-fingerprint", "clock-runtime")
    }

    fn budget_file(label: &str) -> PathBuf {
        std::env::temp_dir()
            .join(format!(
                "synapse-owned-clock-{label}-{}-{}",
                std::process::id(),
                nonce_hex16()
            ))
            .join("budget.json")
    }

    /// A production `SupervisedDecodeDispatch` whose worker can never start:
    /// the runtime config names no artifact, so every spawn is a startup
    /// failure that the supervisor charges to the crash budget. This drives
    /// real strikes through the production supervisor and clock without a
    /// model or worker binary.
    fn failing_dispatch(budget_path: &Path) -> SupervisedDecodeDispatch {
        let factory = OwnedDecodeWorkerFactory::new(
            WorkerHostConfig::new("missing-owned-decode-worker", std::env::temp_dir()),
            ValidatedArtifact {
                digest: "clock-digest".to_string(),
                format: "owned-safetensors".to_string(),
            },
            RuntimeConfig {
                values: BTreeMap::new(),
            },
        );
        let start = GenerateStart {
            generation_id: String::new(),
            loaded_model_ref: String::new(),
            decode_fingerprint: "clock-fingerprint".to_string(),
            runtime_config_digest: "clock-runtime".to_string(),
            prompt_ids: vec![1],
            stop_ids: Vec::new(),
            max_tokens: 4,
            sampling: owned_decode_worker::protocol::Sampling::greedy_top1(),
            constraint: None,
        };
        let context = WorkerStartContext {
            loaded_model_ref: String::new(),
            decode_fingerprint: "clock-fingerprint".to_string(),
            runtime_config_digest: "clock-runtime".to_string(),
            expected_constraint: None,
        };
        let mut dispatch = SupervisedDecodeDispatch::new(
            factory,
            budget_path,
            OwnedBudgetPolicy::default(),
            16,
            owned_quarantine_key(),
            start,
            context,
            TerminalControl::default(),
        )
        .expect("supervised dispatch opens its budget store");
        dispatch.set_request(vec![1], None, 60_000);
        dispatch
    }

    fn dispatch_once(
        dispatch: &mut SupervisedDecodeDispatch,
        generation_id: &str,
    ) -> Result<
        crate::owned_decode_routing::ExecutionSuccess,
        crate::owned_decode_routing::error::OwnedDecodeError,
    > {
        use crate::owned_decode_routing::DecodeDispatch;
        dispatch.dispatch(&crate::owned_decode_routing::DispatchedCommand {
            lane: crate::owned_decode_routing::lane::LaneKind::OwnedDecode,
            decode_fingerprint: synapse_core::Fingerprint("clock-fingerprint".to_string()),
            processing_fingerprint: synapse_core::Fingerprint("clock-processing".to_string()),
            prompt_token_count: 1,
            max_tokens: 4,
            generation_id: generation_id.to_string(),
            constrained: false,
            chain_k: 1,
        })
    }

    /// Charge strikes through the dispatch path until the key is quarantined.
    fn strike_until_quarantined(dispatch: &mut SupervisedDecodeDispatch) {
        use crate::owned_decode_routing::error::OwnedDecodeError;
        let max_strikes = OwnedBudgetPolicy::default().max_strikes;
        for strike in 0..max_strikes {
            let result = dispatch_once(dispatch, &format!("clock-strike-{strike}"));
            let expected = if strike + 1 == max_strikes {
                OwnedDecodeError::Quarantined
            } else {
                OwnedDecodeError::Unavailable
            };
            assert_eq!(result.err(), Some(expected), "strike {strike}");
        }
        assert_eq!(dispatch.crash_budget_remaining(), 0);
    }

    /// The routing precheck reopens the budget file and asks `is_quarantined`
    /// with wall-clock `now_ms()`. A quarantine charged by dispatch must be
    /// visible to that exact question immediately, and must lift once the wall
    /// clock passes `quarantined_until`.
    #[test]
    fn dispatch_quarantine_is_visible_to_the_wall_clock_precheck() {
        let path = budget_file("precheck");
        let mut dispatch = failing_dispatch(&path);
        let before = wall_now_ms();
        strike_until_quarantined(&mut dispatch);
        let after = wall_now_ms();

        let precheck = OwnedCrashBudget::new(
            FileBudgetStore::open(&path).expect("precheck reopens the budget"),
            OwnedBudgetPolicy::default(),
        );
        let key = owned_quarantine_key();
        assert!(
            precheck.is_quarantined(&key, wall_now_ms()),
            "the routing precheck must see the quarantine dispatch just charged; persisted \
             record: {:?}",
            precheck.record(&key)
        );

        // The persisted expiry is a wall-clock instant: charge time plus the
        // policy duration, where the charge happened between `before` and
        // `after`.
        let duration = OwnedBudgetPolicy::default().quarantine_duration_ms;
        let until = precheck
            .record(&key)
            .quarantined_until
            .expect("an exhausted key records its expiry");
        assert!(
            (before + duration..=after + duration).contains(&until),
            "quarantined_until {until} is not wall-clock charge time plus {duration} \
             (charged between {before} and {after})"
        );

        // Expiry, with the wall-clock reading injected rather than slept for.
        assert!(precheck.is_quarantined(&key, until - 1));
        assert!(
            !precheck.is_quarantined(&key, until),
            "the quarantine lifts once the wall clock reaches its expiry"
        );

        drop(dispatch);
        let _ = std::fs::remove_dir_all(path.parent().expect("budget directory"));
    }

    /// A module restart must not lose the quarantine: a fresh process reopens
    /// the budget file with a fresh dispatch, and both the wall-clock predicate
    /// and the new dispatch still refuse the key without charging it again.
    #[test]
    fn dispatch_quarantine_survives_a_store_reopen() {
        use crate::owned_decode_routing::error::OwnedDecodeError;

        let path = budget_file("restart");
        {
            let mut first_process = failing_dispatch(&path);
            strike_until_quarantined(&mut first_process);
        }

        let reopened = FileBudgetStore::open(&path).expect("reopen the budget file");
        let budget = OwnedCrashBudget::new(reopened, OwnedBudgetPolicy::default());
        let key = owned_quarantine_key();
        assert!(
            budget.is_quarantined(&key, wall_now_ms()),
            "a reopened budget must still hold the quarantine at wall-clock now; record: {:?}",
            budget.record(&key)
        );

        let mut second_process = failing_dispatch(&path);
        assert!(second_process.is_quarantined());
        assert_eq!(
            dispatch_once(&mut second_process, "after-restart").err(),
            Some(OwnedDecodeError::Quarantined)
        );
        let strikes = OwnedCrashBudget::new(
            FileBudgetStore::open(&path).expect("reopen the budget file"),
            OwnedBudgetPolicy::default(),
        )
        .record(&key)
        .strikes;
        assert_eq!(
            strikes,
            OwnedBudgetPolicy::default().max_strikes,
            "a refused quarantined dispatch charges nothing"
        );

        drop(second_process);
        let _ = std::fs::remove_dir_all(path.parent().expect("budget directory"));
    }

    /// The converse across a restart: a quarantine whose wall-clock expiry has
    /// passed must not block a freshly created dispatch. A clock that restarts
    /// at zero with each dispatch would read any wall-clock expiry as far in
    /// the future.
    #[test]
    fn expired_wall_clock_quarantine_does_not_block_a_fresh_dispatch() {
        let path = budget_file("expired");
        let policy = OwnedBudgetPolicy::default();
        {
            let mut budget = OwnedCrashBudget::new(
                FileBudgetStore::open(&path).expect("open the budget file"),
                policy,
            );
            let charged_at = wall_now_ms() - policy.quarantine_duration_ms - 60_000;
            for _ in 0..policy.max_strikes {
                budget
                    .charge(
                        &owned_quarantine_key(),
                        owned_decode_worker::error::FailureClassification::Crash,
                        charged_at,
                    )
                    .expect("charge persists");
            }
        }

        let dispatch = failing_dispatch(&path);
        assert!(
            !dispatch.is_quarantined(),
            "a quarantine that expired in wall-clock time must not block a new dispatch"
        );

        drop(dispatch);
        let _ = std::fs::remove_dir_all(path.parent().expect("budget directory"));
    }

    #[tokio::test]
    async fn owned_decode_supervisor_is_the_only_crash_budget_authority() {
        let mut config = WorkerHostConfig::new("unused-worker", std::env::temp_dir());
        config.crash_authority = CrashAuthority::OwnedDecodeSupervisor;
        let mut host = WorkerHost::new(config);
        host.record_crash_and_maybe_restart("owned-key".to_string())
            .await;
        assert!(host.crashes.is_empty());
        assert!(host.quarantined.is_empty());
        assert!(host.connection.is_none());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn rejects_stale_worker_nonce() {
        use synapse_core::worker_framing::write_json_frame;
        use synapse_core::worker_socket_path;
        use synapse_core::{WorkerHello, WORKER_PROTOCOL_VERSION};
        use tokio::net::UnixStream;

        let tmp = PathBuf::from(format!("/tmp/synh-{}", nonce_hex16()));
        let path = worker_socket_path(&tmp, "nonce-test");
        let listener = synapse_core::bind_listener(&path).unwrap();
        let client = tokio::spawn(async move {
            let mut stream = UnixStream::connect(&path).await.unwrap();
            let hello = WorkerHello {
                v: WORKER_PROTOCOL_VERSION,
                nonce: "wrongnonce00000".to_string(),
                engine: EngineIdentity {
                    engine: "test".to_string(),
                    version: "0".to_string(),
                    build_flags: BTreeMap::new(),
                },
                pid: 1,
                max_frame: DEFAULT_MAX_FRAME_BYTES,
                manifest_digest: None,
                kernel_revision: None,
                packed: None,
            };
            write_json_frame(&mut stream, &hello, DEFAULT_MAX_FRAME_BYTES)
                .await
                .unwrap();
        });
        let error: WorkerHostError = synapse_core::accept_worker_handshake(
            listener,
            "expectednonce000",
            DEFAULT_MAX_FRAME_BYTES,
            Duration::from_secs(1),
        )
        .await
        .expect_err("wrong nonce must be rejected")
        .into();
        assert!(
            matches!(error, WorkerHostError::Protocol(message) if message.contains("rejected worker HELLO"))
        );
        client.await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn owned_decode_handshake_refuses_a_pre_v2_worker() {
        use synapse_core::worker_framing::write_json_frame;
        use tokio::net::UnixStream;

        let tmp = PathBuf::from(format!("/tmp/synh-v2-{}", nonce_hex16()));
        let path = synapse_core::worker_socket_path(&tmp, "owned-v1-test");
        let listener = synapse_core::bind_listener(&path).unwrap();
        let client_path = path.clone();
        let client = tokio::spawn(async move {
            let mut stream = UnixStream::connect(&client_path).await.unwrap();
            let hello = serde_json::json!({
                "v": synapse_core::WORKER_PROTOCOL_VERSION,
                "nonce": "0123456789abcdef",
                "engine": { "engine": "decode", "version": "0", "build_flags": {} },
                "pid": 1,
                "max_frame": DEFAULT_MAX_FRAME_BYTES,
                "protocol_version": 1,
            });
            write_json_frame(&mut stream, &hello, DEFAULT_MAX_FRAME_BYTES)
                .await
                .unwrap();
        });
        let error: WorkerHostError =
            synapse_core::accept_worker_handshake_with_engine_and_protocol_version(
                listener,
                "0123456789abcdef",
                DEFAULT_MAX_FRAME_BYTES,
                Duration::from_secs(1),
                Some("decode"),
                Some(2),
                None,
            )
            .await
            .expect_err("owned decode must reject a pre-v2 worker")
            .into();

        assert!(matches!(
            error,
            WorkerHostError::ProtocolVersion {
                advertised: Some(1),
                required: 2
            }
        ));
        client.await.unwrap();
        let _ = std::fs::remove_file(path);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn malformed_owned_response_is_a_protocol_error() {
        let (mut module, mut worker) = tokio::net::UnixStream::pair().unwrap();
        synapse_core::write_json(
            &mut worker,
            &serde_json::json!({ "kind": "final", "not": "an envelope" }),
            DEFAULT_MAX_FRAME_BYTES,
        )
        .await
        .unwrap();
        let request = DecodeTransportRequest::GenerateCancel {
            req_id: "cancel-1".to_string(),
            cancellation: GenerateCancel {
                generation_id: "generation-1".to_string(),
            },
        };
        let mut state = Some(OwnedDecodeAdapterState {
            generation_id: "generation-1".to_string(),
            session_id: "generation-1".to_string(),
            generated_ids: Vec::new(),
            next_stream_sequence: 1,
            quantum_sequence: 1,
            max_tokens: 16,
            constraint_identity: None,
        });

        let error =
            read_owned_decode_response(&mut module, DEFAULT_MAX_FRAME_BYTES, &request, &mut state)
                .await
                .expect_err("malformed response must fail");
        assert!(matches!(error, WorkerHostError::Protocol(_)));
        assert!(matches!(
            owned_host_start_failure(&error),
            WorkerStartFailure::Typed(DecodeError::ProtocolMismatch)
        ));
    }

    #[test]
    fn owned_cuda_model_ids_isolate_crash_keys_for_equal_specs() {
        let path = PathBuf::from("/models/shared.safetensors");
        let mut runtime = RuntimeConfig::default();
        runtime
            .values
            .insert("runtime_revision".to_string(), "v1".to_string());
        let first = crash_key(&path, &runtime, Some("synapse-owned-cuda-first"));
        let second = crash_key(&path, &runtime, Some("synapse-owned-cuda-second"));
        assert_ne!(first, second);
        assert!(first.contains("stable_model_worker_id"));
    }

    #[test]
    fn fault_health_does_not_recover_with_elapsed_time() {
        let mut config = WorkerHostConfig::new("/bin/false", "/tmp/synh-health");
        config.crash_budget = CrashBudget { max_crashes: 3 };
        let mut host = WorkerHost::new(config);

        assert!(!host.health_snapshot().degraded);
        assert!(!host.record_crash("model-a".to_string()));
        let degraded = host.health_snapshot();
        assert!(degraded.degraded);
        assert_eq!(degraded.crash_count_window, 1);

        std::thread::sleep(Duration::from_millis(10));
        let recovered = host.health_snapshot();
        assert!(recovered.degraded);
        assert_eq!(recovered.crash_count_window, 1);
    }

    #[test]
    fn worker_crash_preserves_models_for_lazy_reload() {
        let mut host = WorkerHost::new(WorkerHostConfig::new(
            "/bin/false",
            "/tmp/synh-preserve-models",
        ));
        let mut runtime_config = RuntimeConfig::default();
        runtime_config
            .values
            .insert("artifact_path".to_string(), "/tmp/model.gguf".to_string());
        host.loaded_models.insert(
            "stable-model".to_string(),
            LoadedWorkerModel {
                crash_key: "model-key".to_string(),
                artifact: ValidatedArtifact {
                    digest: "sha256:test".to_string(),
                    format: "gguf".to_string(),
                },
                runtime_config,
                worker_model_ref: Some("worker-model-0".to_string()),
                dims: 0,
                cold_load_ms: 0,
                buckets: None,
            },
        );

        host.forget_worker_model_refs();

        let tracked = host.loaded_models.get("stable-model").unwrap();
        assert_eq!(tracked.crash_key, "model-key");
        assert_eq!(tracked.artifact.format, "gguf");
        assert!(tracked.worker_model_ref.is_none());
    }

    #[cfg(unix)]
    type SeenRequests = Vec<(WorkerRequest, Option<Vec<u8>>)>;

    /// Connects `host` to an in-process mock worker that answers each request
    /// with `respond`. The returned task yields every request (and its raw
    /// frame) once the host drops the connection.
    #[cfg(unix)]
    pub(super) fn attach_mock_worker<F>(
        host: &mut WorkerHost,
        respond: F,
    ) -> tokio::task::JoinHandle<SeenRequests>
    where
        F: Fn(&WorkerRequest, Option<&[u8]>) -> (WorkerResponse, Option<Vec<u8>>) + Send + 'static,
    {
        let (module, mut worker) = tokio::net::UnixStream::pair().unwrap();
        // The mock below speaks the protocol in-process; the host still owns a
        // child process per connection, so give it a harmless one.
        let child = Command::new("sleep")
            .arg("30")
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        host.connection = Some(WorkerConnection {
            stream: module,
            child,
            logs: LogRing::new(),
            worker_generation: 1,
            packed: None,
            unusable: false,
        });
        tokio::spawn(async move {
            let mut seen = Vec::new();
            while let Ok(request) =
                read_json::<WorkerRequest, _>(&mut worker, DEFAULT_MAX_FRAME_BYTES).await
            {
                let raw = if request.carries_raw_frame() {
                    Some(
                        read_raw(&mut worker, DEFAULT_MAX_FRAME_BYTES)
                            .await
                            .unwrap(),
                    )
                } else {
                    None
                };
                let (response, raw_out) = respond(&request, raw.as_deref());
                write_json(&mut worker, &response, DEFAULT_MAX_FRAME_BYTES)
                    .await
                    .unwrap();
                if let Some(raw_out) = raw_out {
                    write_raw(&mut worker, &raw_out, DEFAULT_MAX_FRAME_BYTES)
                        .await
                        .unwrap();
                }
                seen.push((request, raw));
            }
            seen
        })
    }

    #[cfg(unix)]
    fn host_for_engine(engine: &str) -> WorkerHost {
        let mut config = WorkerHostConfig::new("/bin/false", std::env::temp_dir());
        config.engine_identity = Some(EngineIdentity {
            engine: engine.to_string(),
            version: "test".to_string(),
            build_flags: BTreeMap::new(),
        });
        WorkerHost::new(config)
    }

    #[cfg(unix)]
    fn runtime_config(extra: &[(&str, &str)]) -> RuntimeConfig {
        let mut config = RuntimeConfig::default();
        config
            .values
            .insert("artifact_path".to_string(), "/tmp/model.bin".to_string());
        for (key, value) in extra {
            config.values.insert(key.to_string(), value.to_string());
        }
        config
    }

    #[cfg(unix)]
    fn artifact() -> ValidatedArtifact {
        ValidatedArtifact {
            digest: "sha256:package".to_string(),
            format: "safetensors-package".to_string(),
        }
    }

    /// Answers LOAD with LOADED (or with the ERR code named by the
    /// `test_error` runtime key), and scores each rerank input by the sum of
    /// its ids so a test can tell which ids were scored in which order.
    #[cfg(unix)]
    fn rerank_mock(
        request: &WorkerRequest,
        raw: Option<&[u8]>,
    ) -> (WorkerResponse, Option<Vec<u8>>) {
        match request {
            WorkerRequest::Load {
                req_id,
                runtime_config,
                ..
            } => match runtime_config.get("test_error") {
                Some(code) => (
                    WorkerResponse::Err {
                        req_id: Some(req_id.clone()),
                        code: code.clone(),
                        msg: "refused before reading weights".to_string(),
                    },
                    None,
                ),
                None => (
                    WorkerResponse::Loaded {
                        req_id: req_id.clone(),
                        model_ref: "mock:0".to_string(),
                        dims: 0,
                        cold_load_ms: 0,
                        buckets: None,
                    },
                    None,
                ),
            },
            WorkerRequest::RerankSequences {
                req_id, sequences, ..
            } => {
                let ids = synapse_core::decode_i32_frame(raw.unwrap()).unwrap();
                let mut offset = 0;
                let scores = sequences
                    .iter()
                    .map(|sequence| {
                        let slice = &ids[offset..offset + sequence.n_tokens];
                        offset += sequence.n_tokens;
                        slice.iter().sum::<i32>() as f32
                    })
                    .collect::<Vec<_>>();
                assert_eq!(offset, ids.len(), "raw frame holds exactly the sequences");
                (
                    WorkerResponse::Scores {
                        req_id: req_id.clone(),
                    },
                    Some(synapse_core::encode_f32_frame(&scores)),
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
            other => (WorkerResponse::unsupported_request(other), None),
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn owned_lanes_send_composed_sequences_and_keep_candidate_order() {
        let sequences: Vec<Vec<u32>> = vec![
            vec![1, 500, 2, 900, 901, 2],
            vec![1, 500, 2, 7, 2],
            vec![1, 500, 2, 40, 41, 42, 43, 2],
            vec![1, 500, 2, 3000, 2],
        ];
        for engine in synapse_core::OWNED_WORKER_HELLO_ENGINES {
            let mut host = host_for_engine(engine);
            let mock = attach_mock_worker(&mut host, rerank_mock);
            assert_eq!(host.request_count(), 0);
            let model = host
                .load_model(&artifact(), &runtime_config(&[]))
                .await
                .unwrap();
            let scores = host
                .rerank(
                    &model,
                    RerankRequest {
                        query: Vec::new(),
                        candidates: sequences.clone(),
                    },
                )
                .await
                .unwrap();
            let expected: Vec<f32> = sequences
                .iter()
                .map(|sequence| sequence.iter().sum::<u32>() as f32)
                .collect();
            assert_eq!(
                scores.scores, expected,
                "{engine} scores keep candidate order"
            );
            let requests = host.request_count();
            drop(host);

            let seen = mock.await.unwrap();
            assert_eq!(requests, seen.len() as u64);
            assert_eq!(requests, 2);
            let (request, raw) = &seen[1];
            let WorkerRequest::RerankSequences {
                sequences: sent, ..
            } = request
            else {
                panic!("{engine} must receive RERANK_SEQUENCES, got {request:?}");
            };
            assert_eq!(sent.len(), sequences.len());
            let ids = synapse_core::decode_i32_frame(raw.as_deref().unwrap()).unwrap();
            let mut offset = 0;
            for (sent, fixture) in sent.iter().zip(&sequences) {
                let delivered = &ids[offset..offset + sent.n_tokens];
                offset += sent.n_tokens;
                let fixture: Vec<i32> = fixture.iter().map(|id| *id as i32).collect();
                assert_eq!(
                    delivered,
                    fixture.as_slice(),
                    "{engine} sequences arrive unchanged"
                );
            }
            assert_eq!(offset, ids.len());
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn llama_lane_keeps_the_segment_rerank_frame() {
        let mut host = host_for_engine(LLAMA_WORKER_ENGINE);
        let mock = attach_mock_worker(&mut host, rerank_mock);
        let model = host
            .load_model(&artifact(), &runtime_config(&[]))
            .await
            .unwrap();
        host.rerank(
            &model,
            RerankRequest {
                query: vec![10, 11],
                candidates: vec![vec![20], vec![30, 31, 32]],
            },
        )
        .await
        .unwrap();
        drop(host);
        let seen = mock.await.unwrap();
        let (request, raw) = &seen[1];
        assert!(matches!(
            request,
            WorkerRequest::Rerank { query_n_tokens: 2, candidates, .. } if candidates.len() == 2
        ));
        assert_eq!(
            synapse_core::decode_i32_frame(raw.as_deref().unwrap()).unwrap(),
            vec![10, 11, 20, 30, 31, 32]
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn owned_rerank_refuses_an_uncomposed_query_segment() {
        let mut host = host_for_engine("owned-vulkan");
        let _mock = attach_mock_worker(&mut host, rerank_mock);
        let model = host
            .load_model(&artifact(), &runtime_config(&[]))
            .await
            .unwrap();
        let error = host
            .rerank(
                &model,
                RerankRequest {
                    query: vec![1, 2],
                    candidates: vec![vec![3]],
                },
            )
            .await
            .expect_err("owned lanes never assemble pairs themselves");
        assert!(
            matches!(error, WorkerHostError::Protocol(message) if message.contains("composed"))
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn load_forwards_profile_and_operation_and_surfaces_typed_refusals() {
        let mut host = host_for_engine("ane-direct-worker");
        let mock = attach_mock_worker(&mut host, rerank_mock);
        host.load_model(
            &artifact(),
            &runtime_config(&[
                (
                    synapse_core::LOAD_RUNTIME_PROFILE,
                    "gte-reranker-modernbert-base.ane-direct-worker",
                ),
                (synapse_core::LOAD_RUNTIME_OPERATION, "rerank"),
            ]),
        )
        .await
        .unwrap();
        for code in [
            synapse_core::ERR_MODEL_UNSUPPORTED,
            synapse_core::ERR_OPERATION_MISMATCH,
            synapse_core::ERR_PACKAGE_DIGEST_MISMATCH,
            synapse_core::ERR_HEAD_TENSOR_MISSING,
        ] {
            let error = host
                .load_model(&artifact(), &runtime_config(&[("test_error", code)]))
                .await
                .expect_err("the worker refuses this LOAD");
            assert!(
                matches!(&error, WorkerHostError::WorkerErr { code: got, .. } if got == code),
                "{code}: {error}"
            );
            assert_eq!(error.code(), Some(code));
            assert!(error.to_string().contains(code));
        }
        drop(host);
        let seen = mock.await.unwrap();
        let WorkerRequest::Load {
            runtime_config,
            artifact_digest,
            ..
        } = &seen[0].0
        else {
            panic!("first request is LOAD");
        };
        assert_eq!(
            runtime_config.get("profile").map(String::as_str),
            Some("gte-reranker-modernbert-base.ane-direct-worker")
        );
        assert_eq!(
            runtime_config.get("operation").map(String::as_str),
            Some("rerank")
        );
        assert_eq!(artifact_digest, "sha256:package");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn hello_binding_refusal_is_distinct_from_load_refusals() {
        use synapse_core::worker_framing::write_json_frame;
        use tokio::net::UnixStream;

        let tmp = PathBuf::from(format!("/tmp/synh-bind-{}", nonce_hex16()));
        let path = synapse_core::worker_socket_path(&tmp, "binding-test");
        let listener = synapse_core::bind_listener(&path).unwrap();
        let client_path = path.clone();
        let client = tokio::spawn(async move {
            let mut stream = UnixStream::connect(&client_path).await.unwrap();
            let hello = serde_json::json!({
                "v": synapse_core::WORKER_PROTOCOL_VERSION,
                "nonce": "0123456789abcdef",
                "engine": { "engine": "owned-cuda", "version": "0", "build_flags": {} },
                "pid": 1,
                "max_frame": DEFAULT_MAX_FRAME_BYTES,
                "manifest_digest": "b".repeat(64),
                "kernel_revision": "cuda-kernel-v1",
            });
            write_json_frame(&mut stream, &hello, DEFAULT_MAX_FRAME_BYTES)
                .await
                .unwrap();
        });
        let expected = ExpectedHelloBinding {
            manifest_digest: "a".repeat(64),
            kernel_revision: "cuda-kernel-v1".to_string(),
        };
        let error: WorkerHostError =
            synapse_core::accept_worker_handshake_with_engine_and_protocol_version(
                listener,
                "0123456789abcdef",
                DEFAULT_MAX_FRAME_BYTES,
                Duration::from_secs(1),
                Some("owned-cuda"),
                None,
                Some(&expected),
            )
            .await
            .expect_err("a worker built from another manifest is refused")
            .into();
        assert!(matches!(
            &error,
            WorkerHostError::HelloRefused { code, .. } if code == synapse_core::ERR_MANIFEST_MISMATCH
        ));
        assert_eq!(error.code(), Some(synapse_core::ERR_MANIFEST_MISMATCH));
        assert!(error
            .to_engine_error(EngineErrorStage::Load)
            .retry_after_ms
            .is_none());
        client.await.unwrap();
        let _ = std::fs::remove_file(path);
    }
}

/// Direct-ANE shape residency: the module-side budget for compiled shapes
/// across every direct-ANE worker, and the lock that keeps the lane to one
/// module process.
///
/// A direct-ANE worker compiles one executable set per (model, ladder shape)
/// and keeps it resident on the Neural Engine. Too many resident shapes
/// exhaust the ANE, so the supervisor owns a budget of
/// [`ANE_RESIDENT_SHAPES_PER_MODEL`] shapes per model ref and
/// [`ANE_RESIDENT_SHAPES_TOTAL`] overall, plus a module-wide
/// [`ANE_RESIDENT_EXECUTABLES_TOTAL`] layer-executable budget, enforced with
/// `ANE_ADMIT_SHAPE` and `ANE_EVICT_SHAPE`. A shape counts against both budgets
/// from its admit request until its `EVICTED` ack. While a request runs on a shape it holds a lease,
/// and a leased shape is never evicted.
///
/// A request runs its rungs one at a time in ascending order and releases each
/// rung's lease before asking for the next one ([`AneResidencySupervisor::run_by_rung`]),
/// so no request holds a lease while it waits. A request whose shape needs a
/// cold compile waits in the compile queue: at most one shape compiles at a
/// time across all workers, and the queued compile with the smallest start
/// key (arrival plus predicted cost) goes next. When the budget is full, the
/// request at the head of that queue evicts the least recently used shape
/// that has no lease.
///
/// Worker connections must not be held while waiting for admission: the
/// request at the head may need to evict a shape on another worker. That is why
/// workers are driven through [`AneShapeWorker`], whose connection is held only
/// for a single exchange ([`AneWorkerChannel`]).
pub mod ane_residency {
    #[cfg(test)]
    mod quarantine_tests;
    use std::collections::{BTreeMap, BTreeSet, VecDeque};
    use std::future::Future;
    use std::io;
    #[cfg(unix)]
    use std::path::Path;
    use std::path::PathBuf;
    use std::pin::Pin;
    use std::process::Stdio;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex, MutexGuard};

    #[cfg(test)]
    use synapse_core::{read_json, read_raw, write_json, write_raw};
    use synapse_core::{
        AnePlacementInventory, TransportError, WorkerRequest, WorkerResponse, ERR_ANE_LANE_BUSY,
    };
    use thiserror::Error;
    use tokio::io::{AsyncRead, AsyncWrite};
    use tokio::sync::Notify;

    /// Padded sequence shapes a direct-ANE worker compiles; a sequence runs on
    /// the smallest rung at least its length.
    pub const ANE_SHAPE_LADDER: [usize; 7] = [128, 256, 512, 1024, 2048, 4096, 8192];
    pub const ANE_RESIDENT_SHAPES_PER_MODEL: usize = 4;
    pub const ANE_RESIDENT_SHAPES_TOTAL: usize = 8;
    /// Only M5 Max was measured: Mac17,6, OS build 26A434 failed at 115-118
    /// loaded layer executables across processes, including gte-modernbert-base
    /// at sequence lengths 4096 and 8192.
    /// Reserve 15 slots below the lowest observed failure; other ANE users can
    /// still exhaust resources, so the worker's transient refusal remains a backstop.
    pub const ANE_RESIDENT_EXECUTABLES_TOTAL: usize = 100;
    /// Bound admission queue waits; worker admit/evict exchanges finish separately
    /// so a timed-out waiter cannot leave an unaccounted partially admitted shape.
    pub const ANE_ADMISSION_WAIT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(600);

    // A closed cohort of single-flight joiners gets a scheduling turn, not an
    // indefinite residency pin. Late requests cannot renew this 250 ms grace.
    const ADMISSION_HANDOFF: std::time::Duration = std::time::Duration::from_millis(250);
    #[cfg(test)]
    mod admission_trace;
    mod scheduler;
    pub const ERR_ANE_RESOURCES_EXHAUSTED: &str = "ane_resources_exhausted";
    pub const ANE_RESOURCES_RETRY_AFTER_MS: u64 = 250;
    /// A request is blocked on a shape compile: its deadline expired while it
    /// waited for one, or the compile queue predicts its shape will be ready
    /// only after that deadline. The model is loaded; a retry joins the same
    /// compile or finds the shape resident.
    pub const ERR_ANE_SHAPE_COMPILING: &str = "shape_compiling";
    /// The smallest `retry_after_ms` a `shape_compiling` answer carries; the
    /// answer otherwise carries the predicted time until the shape is ready.
    pub const ANE_SHAPE_COMPILING_RETRY_AFTER_MS: u64 = 1000;
    /// The wire code for a direct-ANE lane whose compile scheduler stopped.
    pub const ERR_ANE_SCHEDULER_STOPPED: &str = "engine_crashed";
    /// A queued cold compile with a known cost waits this long before it may
    /// start, so compiles requested together are started cheapest first
    /// instead of in the order their requests happened to arrive.
    const COMPILE_GATHER: std::time::Duration = std::time::Duration::from_millis(50);
    /// A queued compile that no request is waiting for any more is dropped
    /// once this multiple of the retry delay it last promised has passed.
    /// Otherwise demand from a caller that never retries keeps compiling and
    /// evicting shapes that other requests use.
    const ORPHAN_EXPIRY_FACTOR: u32 = 2;
    /// Lock file, relative to the home directory, held by the one module
    /// process allowed to run direct-ANE workers.
    pub const ANE_DIRECT_LOCK_HOME_PATH: &str = "Library/Caches/ck-synapse/ane-direct.lock";

    pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

    /// Whatever the module holds while one inference exchange runs (its
    /// execution permit); dropped when the exchange ends.
    pub type ExchangePermit = Box<dyn std::any::Any + Send>;
    /// Grants an [`ExchangePermit`] by the given request deadline. The serving
    /// task calls it once per inference exchange, after taking the worker
    /// stream, never while waiting for admission.
    pub type PermitSource = Arc<
        dyn Fn(
                Option<tokio::time::Instant>,
            ) -> BoxFuture<'static, Result<ExchangePermit, AneResidencyError>>
            + Send
            + Sync,
    >;

    /// What a serving request is waiting for right now. When its deadline
    /// expires, the answer is decided by the wait that actually expired, not
    /// by whatever the worker happens to be doing when the error is mapped.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum WaitPhase {
        /// Budget queue, eviction, execution permit or inference I/O.
        Other,
        /// The request waited for its own shape's compile, by this request or
        /// by another one it joined.
        Compile(usize),
        /// Waiting for the compile scheduler's answer to request `id`; when
        /// the caller's own deadline expires, the scheduler decides the answer.
        Scheduled { id: u64, shape: usize },
        /// Waiting for the worker stream since the given instant. The wait was
        /// on a compile if an admit exchange held the stream at any moment
        /// between that instant and the deadline.
        Stream(tokio::time::Instant),
        /// The deadline already expired; `Some(shape)` means on a compile.
        Expired(Option<usize>),
    }

    #[derive(Debug)]
    struct RequestWaits {
        phase: Mutex<WaitPhase>,
        /// The supervisor and worker this request leases from, so an expiry
        /// on the worker stream can ask for the running compile's remaining
        /// time.
        link: Mutex<Option<(std::sync::Weak<Inner>, String)>>,
        abandoned: std::sync::atomic::AtomicBool,
        abandonment: Notify,
    }

    impl RequestWaits {
        fn new() -> Self {
            Self {
                phase: Mutex::new(WaitPhase::Other),
                link: Mutex::new(None),
                abandoned: std::sync::atomic::AtomicBool::new(false),
                abandonment: Notify::new(),
            }
        }

        fn lock(&self) -> MutexGuard<'_, WaitPhase> {
            self.phase
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
        }

        fn link(&self) -> Option<(Arc<Inner>, String)> {
            let link = self
                .link
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone()?;
            Some((link.0.upgrade()?, link.1))
        }

        /// When the compile now running on this request's worker is predicted
        /// to finish, if one is running.
        fn running_compile_ready(&self) -> Option<tokio::time::Instant> {
            let (inner, worker_id) = self.link()?;
            let now = tokio::time::Instant::now();
            let state = inner.lock();
            state
                .slots
                .values()
                .filter(|slot| {
                    slot.worker.worker_id() == worker_id && slot.state == SlotState::Admitting
                })
                .map(|slot| now + slot.compile_remaining(now))
                .max()
        }
    }

    tokio::task_local! {
        /// The wait phase of the serving request whose task this is.
        static REQUEST_WAITS: Arc<RequestWaits>;
    }

    tokio::task_local! {
        /// Where the worker channel reports the moment a compile operation's
        /// admission was written to the worker, after any wait for the
        /// worker stream. The scheduler counts the running compile and learns
        /// compile times from that moment.
        static COMPILE_DISPATCHED: Arc<scheduler::DispatchCell>;
    }

    // The caller may disappear while dispatched I/O drains in its detached task.
    // Only work that has not reached the wire is cancelled by this signal.
    struct RequestAbandonment(Arc<RequestWaits>);
    impl Drop for RequestAbandonment {
        fn drop(&mut self) {
            self.0.abandoned.store(true, Ordering::Release);
            self.0.abandonment.notify_waiters();
        }
    }

    fn ensure_request_live() -> Result<(), AneResidencyError> {
        if REQUEST_WAITS
            .try_with(|waits| waits.abandoned.load(Ordering::Acquire))
            .unwrap_or(false)
        {
            return Err(AneResidencyError::Abandoned);
        }
        Ok(())
    }

    async fn cancellable_wait<F: Future>(future: F) -> Result<F::Output, AneResidencyError> {
        let cancellation = async {
            if let Ok(waits) = REQUEST_WAITS.try_with(Arc::clone) {
                let notified = waits.abandonment.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                if !waits.abandoned.load(Ordering::Acquire) {
                    notified.await;
                }
            } else {
                std::future::pending::<()>().await;
            }
        };
        tokio::select! { result = future => Ok(result), _ = cancellation => Err(AneResidencyError::Abandoned) }
    }

    /// `shape_compiling` for `shape`, whose retry delay is the time left until
    /// the predicted `ready` instant, but never below the documented minimum.
    fn shape_compiling(shape: usize, ready: Option<tokio::time::Instant>) -> AneResidencyError {
        let predicted_ms = ready.map_or(0, |ready| {
            let left = ready.saturating_duration_since(tokio::time::Instant::now());
            u64::try_from(left.as_millis()).unwrap_or(u64::MAX)
        });
        AneResidencyError::ShapeCompiling {
            shape,
            retry_after_ms: predicted_ms.max(ANE_SHAPE_COMPILING_RETRY_AFTER_MS),
        }
    }

    /// Records the current wait of this task's serving request, unless its
    /// deadline has already been classified.
    fn enter_wait(phase: WaitPhase) {
        let _ = REQUEST_WAITS.try_with(|waits| {
            let mut current = waits.lock();
            if !matches!(*current, WaitPhase::Expired(_)) {
                *current = phase;
            }
        });
    }

    /// The deadline expired during a wait whose cause is `cause`. The first
    /// classification recorded for the request wins. A compile cause carries
    /// the predicted time until the compile now running on the request's
    /// worker finishes.
    fn deadline_expired(cause: Option<usize>) -> AneResidencyError {
        let recorded = REQUEST_WAITS
            .try_with(|waits| {
                let mut current = waits.lock();
                match *current {
                    WaitPhase::Expired(first) => first,
                    _ => {
                        *current = WaitPhase::Expired(cause);
                        cause
                    }
                }
            })
            .unwrap_or(cause);
        match recorded {
            Some(shape) => shape_compiling(
                shape,
                REQUEST_WAITS
                    .try_with(|waits| waits.running_compile_ready())
                    .unwrap_or(None),
            ),
            None => AneResidencyError::DeadlineExceeded,
        }
    }

    /// The deadline expired during the last recorded lease wait.
    fn lease_deadline_expired() -> AneResidencyError {
        let cause = REQUEST_WAITS
            .try_with(|waits| match *waits.lock() {
                WaitPhase::Compile(shape) | WaitPhase::Scheduled { shape, .. } => Some(shape),
                WaitPhase::Expired(first) => first,
                _ => None,
            })
            .unwrap_or(None);
        deadline_expired(cause)
    }

    /// The smallest ladder rung that holds `n_tokens`, or `None` above 8192.
    pub fn ladder_rung(n_tokens: usize) -> Option<usize> {
        ANE_SHAPE_LADDER
            .iter()
            .copied()
            .find(|rung| *rung >= n_tokens)
    }

    #[derive(Debug, Error)]
    pub enum AneResidencyError {
        #[error("direct-ANE request abandoned by its caller")]
        Abandoned,
        #[error("direct-ANE request deadline exceeded")]
        DeadlineExceeded,
        #[error("model/config is quarantined after repeated worker crashes; retry after {retry_after_ms}ms")]
        Quarantined { retry_after_ms: u64 },
        #[error("sequence of {n_tokens} tokens exceeds the largest direct-ANE shape {max}")]
        SequenceTooLong { n_tokens: usize, max: usize },
        /// The worker answered `ERR`.
        #[error("worker returned {code}: {msg}")]
        WorkerErr { code: String, msg: String },
        #[error("ane_resources_exhausted: {msg}; retry after {retry_after_ms}ms")]
        ResourcesExhausted { msg: String, retry_after_ms: u64 },
        #[error("direct-ANE worker channel: {0}")]
        Channel(String),
        #[error("ane_lane_busy: another live process holds {}", path.display())]
        LaneBusy { path: PathBuf },
        #[error("direct-ANE lane lock: {0}")]
        Io(#[from] io::Error),
        #[error("shape_compiling: the direct-ANE worker is compiling shape {shape}")]
        ShapeCompiling { shape: usize, retry_after_ms: u64 },
        /// The module's execution permit could not be granted for a reason
        /// other than the request deadline (the executor is shutting down).
        #[error("execution permit unavailable: {0}")]
        PermitUnavailable(String),
        /// The compile scheduler's loop ended (a bug: it panicked or was
        /// stopped). Direct-ANE lanes stay closed until the module restarts.
        #[error("engine_crashed: the direct-ANE compile scheduler stopped; the lane is closed until the module restarts")]
        SchedulerStopped,
        /// The worker's old process could not be confirmed to have exited, so
        /// it may still own the Neural Engine: the worker stays closed until
        /// the module restarts, and its requests are refused at once.
        #[error("engine_crashed: direct-ANE worker {worker_id} could not confirm that its old process exited; it is closed until the module restarts")]
        WorkerClosed { worker_id: String },
        /// A compile, eviction or recovery task panicked or was cancelled.
        #[error("direct-ANE worker operation failed: {0}")]
        Panicked(String),
    }

    impl AneResidencyError {
        pub fn to_engine_error(
            &self,
            stage: synapse_core::EngineErrorStage,
        ) -> synapse_core::EngineError {
            synapse_core::EngineError {
                stage,
                risk_class: synapse_core::EngineRiskClass::AbortCapable,
                message: self.to_string(),
                retry_after_ms: match self {
                    Self::Quarantined { retry_after_ms }
                    | Self::ResourcesExhausted { retry_after_ms, .. }
                    | Self::ShapeCompiling { retry_after_ms, .. } => Some(*retry_after_ms),
                    _ => None,
                },
                safe_to_retry_same_request: matches!(
                    self,
                    Self::ResourcesExhausted { .. }
                        | Self::ShapeCompiling { .. }
                        | Self::Quarantined { .. }
                ),
            }
        }
        pub fn code(&self) -> Option<&str> {
            match self {
                Self::WorkerErr { code, .. } => Some(code),
                Self::LaneBusy { .. } => Some(ERR_ANE_LANE_BUSY),
                Self::ResourcesExhausted { .. } => Some(ERR_ANE_RESOURCES_EXHAUSTED),
                Self::ShapeCompiling { .. } => Some(ERR_ANE_SHAPE_COMPILING),
                Self::SchedulerStopped => Some(ERR_ANE_SCHEDULER_STOPPED),
                _ => None,
            }
        }
    }

    impl From<TransportError> for AneResidencyError {
        fn from(error: TransportError) -> Self {
            Self::Channel(error.to_string())
        }
    }

    /// One direct-ANE worker the supervisor admits and evicts shapes on.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
    pub enum AneProgram {
        #[default]
        Single,
        Packed {
            program_revision: u32,
        },
    }

    pub trait AneShapeWorker: Send + Sync {
        /// Stable id of this worker; shapes are dropped per worker on restart.
        fn worker_id(&self) -> &str;
        /// A scheduling hint only: the stream remains locked for the full RPC.
        /// Unknown transports conservatively report busy and remain fallbacks.
        fn stream_idle(&self) -> bool {
            false
        }
        fn admit_program<'a>(
            &'a self,
            generation: u64,
            model_ref: &'a str,
            shape: usize,
            program: AneProgram,
        ) -> BoxFuture<'a, Result<AnePlacementInventory, AneResidencyError>> {
            match program {
                AneProgram::Single => self.admit_shape_at_generation(generation, model_ref, shape),
                AneProgram::Packed { .. } => Box::pin(async {
                    Err(AneResidencyError::WorkerErr {
                        code: "unsupported_request".into(),
                        msg: "packed program unavailable".into(),
                    })
                }),
            }
        }
        fn evict_program<'a>(
            &'a self,
            generation: u64,
            model_ref: &'a str,
            shape: usize,
            program: AneProgram,
        ) -> BoxFuture<'a, Result<(), AneResidencyError>> {
            match program {
                AneProgram::Single => self.evict_shape_at_generation(generation, model_ref, shape),
                AneProgram::Packed { .. } => Box::pin(async {
                    Err(AneResidencyError::WorkerErr {
                        code: "unsupported_request".into(),
                        msg: "packed program unavailable".into(),
                    })
                }),
            }
        }
        /// The test hold on expired-wait classification for this worker.
        #[cfg(test)]
        fn classify_gate(&self) -> Option<Arc<ClassifyGate>> {
            None
        }
        /// Identifies the owning process, not just the stable worker name.
        fn generation(&self) -> u64 {
            0
        }
        /// Whether a restart could not confirm that the old worker process
        /// exited. Its hardware ownership is then unknown, so the worker
        /// stays closed rather than being retried.
        fn exit_unconfirmed(&self) -> bool {
            false
        }
        /// The worker process's id, for logs, when known.
        fn process_id(&self) -> Option<u32> {
            None
        }
        /// Restarts the worker and sends on `exited` as soon as the old
        /// process's exit is confirmed, before the replacement starts, so the
        /// scheduler can release what the old process held. This default
        /// signals only once the whole restart has finished with the exit
        /// confirmed.
        fn restart_with_exit_signal<'a>(
            &'a self,
            exited: tokio::sync::oneshot::Sender<()>,
        ) -> BoxFuture<'a, Result<(), AneResidencyError>> {
            Box::pin(async move {
                let result = self.restart().await;
                if result.is_ok() || !self.exit_unconfirmed() {
                    let _ = exited.send(());
                }
                result
            })
        }
        fn evict_shape_at_generation<'a>(
            &'a self,
            generation: u64,
            model_ref: &'a str,
            shape: usize,
        ) -> BoxFuture<'a, Result<(), AneResidencyError>> {
            if self.generation() != generation {
                Box::pin(async {
                    Err(AneResidencyError::Channel("stale worker generation".into()))
                })
            } else {
                self.evict_shape(model_ref, shape)
            }
        }
        /// Full layer count reserved before compilation, from the model profile in
        /// the repository's versioned bench/parity/models.json manifest.
        fn executable_count(&self, model_ref: &str) -> Result<usize, AneResidencyError> {
            let layers = pinned_model_param(model_ref, "num_hidden_layers")
                .and_then(|layers| usize::try_from(layers).ok())
                .filter(|layers| *layers > 0);
            layers.ok_or_else(|| AneResidencyError::WorkerErr {
                code: "model_unsupported".into(),
                msg: "no pinned layer count for direct-ANE model reference".into(),
            })
        }
        /// Expected standalone compile time of `shape` with `layers` layers,
        /// before the module has measured any compile. It only orders the
        /// compile queue and seeds its predictions; zero means unknown, and
        /// such compiles are queued in arrival order and never refused early.
        fn compile_cost_seed(
            &self,
            model_ref: &str,
            shape: usize,
            layers: usize,
        ) -> std::time::Duration {
            pinned_compile_cost_seed(model_ref, shape, layers)
        }
        fn admit_shape_at_generation<'a>(
            &'a self,
            generation: u64,
            model_ref: &'a str,
            shape: usize,
        ) -> BoxFuture<'a, Result<AnePlacementInventory, AneResidencyError>> {
            if self.generation() != generation {
                Box::pin(async {
                    Err(AneResidencyError::Channel("stale worker generation".into()))
                })
            } else {
                self.admit_shape(model_ref, shape)
            }
        }
        /// Sends `ANE_ADMIT_SHAPE` and waits for `ADMITTED`.
        fn admit_shape<'a>(
            &'a self,
            model_ref: &'a str,
            shape: usize,
        ) -> BoxFuture<'a, Result<AnePlacementInventory, AneResidencyError>>;
        /// Sends `ANE_EVICT_SHAPE` and waits for `EVICTED`.
        fn evict_shape<'a>(
            &'a self,
            model_ref: &'a str,
            shape: usize,
        ) -> BoxFuture<'a, Result<(), AneResidencyError>>;
        /// Replaces the worker process with a fresh one that has no shapes.
        fn restart(&self) -> BoxFuture<'_, Result<(), AneResidencyError>>;
    }

    /// The seed estimate for a model in the pinned manifest: its layer count
    /// times the measured per-layer time for its hidden size and `shape`.
    fn pinned_compile_cost_seed(
        model_ref: &str,
        shape: usize,
        layers: usize,
    ) -> std::time::Duration {
        let hidden = pinned_model_param(model_ref, "hidden_size")
            .and_then(|hidden| usize::try_from(hidden).ok());
        seed_compile_cost(hidden, layers, shape)
    }

    /// One architecture parameter of the model behind a direct-ANE model
    /// reference, from the repository's pinned bench/parity/models.json.
    fn pinned_model_param(model_ref: &str, name: &str) -> Option<u64> {
        static MANIFEST: std::sync::OnceLock<serde_json::Value> = std::sync::OnceLock::new();
        let manifest = MANIFEST.get_or_init(|| {
            serde_json::from_str(include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../bench/parity/models.json"
            )))
            .expect("pinned parity manifest")
        });
        let profile = model_ref
            .strip_prefix("ane-direct:")
            .and_then(|value| value.split(':').next())?;
        let model = manifest["profiles"][profile]["model"].as_str()?;
        manifest["models"][model]["architecture"]["params"][name].as_u64()
    }

    /// Median compile-and-load seconds per layer, by hidden size and ladder
    /// width, measured with nothing else compiling on an M4 Mac mini (macOS
    /// 27.0, October 2026). The 768-wide row at 4096 and 8192 and the
    /// 1024-wide row at 8192 were not measured on the M4: they are M5 Max
    /// standalone times divided by 1.27, the M5/M4 ratio measured for the
    /// 1024-wide model at 4096. The 1024-wide 8192 value on the M5 was itself
    /// an estimate. The compile queue corrects all of these with measured
    /// compiles; they only need to rank shapes and give a first prediction.
    const SEED_SECONDS_PER_LAYER: [(usize, [(usize, f64); 5]); 2] = [
        (
            768,
            [
                (128, 0.21),
                (512, 0.37),
                (2048, 0.86),
                (4096, 2.03),
                (8192, 4.13),
            ],
        ),
        (
            1024,
            [
                (128, 0.38),
                (512, 0.76),
                (2048, 1.49),
                (4096, 3.02),
                (8192, 6.47),
            ],
        ),
    ];

    /// `layers` times the table's per-layer seconds for the nearest measured
    /// hidden size (the wider row when the size is unknown), interpolated
    /// linearly between measured widths and clamped at both ends.
    fn seed_compile_cost(
        hidden: Option<usize>,
        layers: usize,
        width: usize,
    ) -> std::time::Duration {
        let (_, row) = SEED_SECONDS_PER_LAYER
            .iter()
            .min_by_key(|(size, _)| {
                hidden.map_or(usize::MAX - size, |hidden| size.abs_diff(hidden))
            })
            .expect("the seed table has rows");
        let per_layer = match row.iter().position(|(rung, _)| *rung >= width) {
            Some(0) => row[0].1,
            None => row[row.len() - 1].1,
            Some(upper) => {
                let (low_width, low) = row[upper - 1];
                let (high_width, high) = row[upper];
                let fraction = (width - low_width) as f64 / (high_width - low_width) as f64;
                low + (high - low) * fraction
            }
        };
        std::time::Duration::from_secs_f64(per_layer * layers as f64)
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct AneResidencyLimits {
        pub per_model: usize,
        pub total: usize,
    }

    impl Default for AneResidencyLimits {
        fn default() -> Self {
            Self {
                per_model: ANE_RESIDENT_SHAPES_PER_MODEL,
                total: ANE_RESIDENT_SHAPES_TOTAL,
            }
        }
    }

    /// Counters sampled after every `ADMITTED` and every `EVICTED`.
    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    pub struct AneResidencyStats {
        pub admitted: u64,
        pub evicted: u64,
        pub samples: u64,
        pub max_resident_per_model: usize,
        pub max_resident_total: usize,
        pub restarts: u64,
    }

    // Test-only observation separates complete lease latency from the compile RPC,
    // which starts after the connection lock and excludes queued inference work.
    #[cfg(test)]
    #[derive(Clone, serde::Serialize)]
    struct AdmissionWait {
        shape: usize,
        elapsed_ms: f64,
        admitted: bool,
    }
    #[cfg(test)]
    type AdmissionWaits = Arc<Mutex<Vec<AdmissionWait>>>;
    #[cfg(test)]
    tokio::task_local! {
        static ADMISSION_WAITS: AdmissionWaits;
        static COMPILE_RPC_MS: Arc<Mutex<Option<f64>>>;
    }

    #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct ShapeKey {
        worker_id: String,
        model_ref: String,
        shape: usize,
        program: AneProgram,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum SlotState {
        Admitting,
        Resident,
        Evicting,
        /// The compile failed with an unknown outcome (transport fault,
        /// timeout or panic): it may still be running in the old process, so
        /// the slot holds the compile gate until that process's exit is
        /// confirmed.
        Failed,
        /// The worker answered the admission with an error: nothing is
        /// compiling, so the gate is free, but the reservation stays charged
        /// until the worker's recovery confirms its exit.
        Rejected,
        /// The worker's exit could not be confirmed: the reservation stays
        /// charged, but no longer holds the compile gate.
        Abandoned,
    }

    struct Slot {
        /// Distinguishes this residency from an earlier one of the same shape,
        /// so a lease or ack from before a restart never touches the new slot.
        id: u64,
        worker: Arc<dyn AneShapeWorker>,
        state: SlotState,
        leases: u32,
        last_used: u64,
        inventory: Option<Arc<AnePlacementInventory>>,
        executables: usize,
        generation: u64,
        packed_qualified: bool,
        joined: BTreeSet<u64>,
        handoff_until: Option<tokio::time::Instant>,
        /// The predicted duration of the compile behind an `Admitting` slot,
        /// and the worker's seed estimate for it.
        compile_cost: std::time::Duration,
        compile_seed: std::time::Duration,
        /// The queue entry this compile came from, so an admission withdrawn
        /// before it reached the worker can be queued again in its old place.
        queued_as: Option<QueuedPlace>,
        /// The request whose turn reserved this slot; it gets the compile's
        /// error if the compile fails.
        owner: Option<u64>,
        /// The retry delay promised while this compile ran, so a withdrawn
        /// compile goes back into the queue with it.
        promise: Option<(tokio::time::Instant, std::time::Duration)>,
        /// When the compile's admission reached the worker.
        dispatched_at: Option<tokio::time::Instant>,
    }

    /// Where a compile stood in the queue: its creation order and start key.
    #[derive(Clone, Copy, Debug)]
    struct QueuedPlace {
        seq: u64,
        enqueued_at: tokio::time::Instant,
        start_key: tokio::time::Instant,
    }

    impl Slot {
        /// Whether the 250 ms handoff to the requests that waited for this
        /// shape's compile is open at `now`.
        fn handing_off(&self, now: tokio::time::Instant) -> bool {
            self.state == SlotState::Resident
                && !self.joined.is_empty()
                && self.handoff_until.is_some_and(|until| now < until)
        }

        /// Predicted time left in this slot's compile, counted from the moment
        /// it reached the worker (all of it until then). A compile running past
        /// its prediction, or failed with an unknown outcome, is still given
        /// one more layer's worth of time.
        fn compile_remaining(&self, now: tokio::time::Instant) -> std::time::Duration {
            if self.compile_cost.is_zero() {
                return std::time::Duration::ZERO;
            }
            let layers = u32::try_from(self.executables.max(1)).unwrap_or(u32::MAX);
            let layer = self.compile_cost / layers;
            match (self.state, self.dispatched_at) {
                (SlotState::Admitting, None) => self.compile_cost,
                (SlotState::Admitting, Some(dispatched)) => self
                    .compile_cost
                    .saturating_sub(now.saturating_duration_since(dispatched))
                    .max(layer),
                (SlotState::Failed, _) => layer,
                _ => std::time::Duration::ZERO,
            }
        }
    }

    /// A cold compile waiting for its turn. At most one compile runs at a
    /// time across all direct-ANE workers: Apple's compiler serves one layer
    /// at a time machine-wide, so compiles run side by side gain no
    /// throughput, and a small shape beside a wide one waits one wide layer
    /// per layer of its own. The entry with the smallest `start_key` starts
    /// next.
    struct QueuedCompile {
        /// Creation order; breaks ties between equal start keys.
        seq: u64,
        enqueued_at: tokio::time::Instant,
        /// `enqueued_at + cost`, fixed at creation: cheapest first, with
        /// aging. An entry of cost C can only be passed by entries that
        /// arrive less than C after it, so a wide compile still gets a turn.
        start_key: tokio::time::Instant,
        /// The worker's seed estimate. The duration used in predictions is
        /// recomputed from it on every read ([`State::estimate`]), so an
        /// entry queued before the first measured compile is repriced once
        /// the machine factor is known; only `start_key` stays fixed.
        seed: std::time::Duration,
        worker: Arc<dyn AneShapeWorker>,
        executables: usize,
        /// When the last request waiting for this entry left, and the retry
        /// delay it was given. The entry outlives its requests so a retry
        /// finds the shape compiled, but only for [`ORPHAN_EXPIRY_FACTOR`]
        /// times that delay unless another request attaches.
        promise: Option<(tokio::time::Instant, std::time::Duration)>,
    }

    impl QueuedCompile {
        fn order(&self) -> (tokio::time::Instant, u64) {
            (self.start_key, self.seq)
        }

        /// When this entry may start: compiles with a known cost wait a short
        /// gathering window so compiles requested together start cheapest first.
        fn gathered_at(&self) -> tokio::time::Instant {
            if self.seed.is_zero() {
                self.enqueued_at
            } else {
                self.enqueued_at + COMPILE_GATHER
            }
        }

        /// The instant after which an entry no request waits for is dropped.
        fn orphan_expiry(&self) -> Option<tokio::time::Instant> {
            self.promise
                .map(|(promised_at, retry)| promised_at + retry * ORPHAN_EXPIRY_FACTOR)
        }

        fn place(&self) -> QueuedPlace {
            QueuedPlace {
                seq: self.seq,
                enqueued_at: self.enqueued_at,
                start_key: self.start_key,
            }
        }
    }

    #[derive(Default)]
    struct State {
        slots: BTreeMap<ShapeKey, Slot>,
        recovering: BTreeSet<String>,
        retired: BTreeSet<String>,
        waiters: VecDeque<u64>,
        /// The shape each queued request asked for, recorded on its first
        /// decision. A request whose shape already has a slot (resident,
        /// being admitted or being evicted) needs no budget, so it neither
        /// holds nor waits for the head of the FIFO.
        waiter_keys: BTreeMap<u64, ShapeKey>,
        /// Set while the queue head waits for an eviction victim and none is
        /// free: the id of its earliest request (or of the last request seen,
        /// for a compile kept only for a promise). Later requests then lease
        /// resident shapes in queue order through the scheduler, so a stream
        /// of short leases cannot starve the head.
        starved_head: Option<u64>,
        /// Each model's smallest published ladder rung. Eviction picks it only
        /// when no other unleased resident shape can be evicted, which keeps
        /// short requests (queries) from recompiling after a long one.
        smallest_rungs: BTreeMap<String, usize>,
        next_slot: u64,
        clock: u64,
        stats: AneResidencyStats,
        /// Cold compiles waiting for their turn, one per shape key.
        entries: BTreeMap<ShapeKey, QueuedCompile>,
        next_entry: u64,
        /// The latest bound (request deadline or wait timeout) of each queued
        /// request that was told its compile would be ready in time. A newer
        /// compile is not started ahead of it if that would break the bound.
        waiter_bounds: BTreeMap<u64, tokio::time::Instant>,
        /// Measured compile time per (model, shape, program).
        measured_costs: BTreeMap<(String, usize, AneProgram), std::time::Duration>,
        /// Moving average of measured compile time over the worker's seed
        /// estimate, so a slower machine corrects shapes not measured yet.
        machine_factor: Option<f64>,
        // Kept outside residency slots so eviction and confirmed-exit recovery
        // cannot erase certification evidence. Disabled supervisors retain nothing.
        inventory_sink: Option<Vec<Arc<AnePlacementInventory>>>,
        #[cfg(test)]
        recorded_samples: Vec<BTreeMap<String, Vec<usize>>>,
    }

    impl State {
        fn tick(&mut self) -> u64 {
            self.clock += 1;
            self.clock
        }

        fn reserved_executables(&self) -> usize {
            // Admitting and evicting shapes retain their full reservation until the worker ack.
            self.slots.values().map(|slot| slot.executables).sum()
        }

        fn model_count(&self, model_ref: &str) -> usize {
            self.slots
                .keys()
                .filter(|key| key.model_ref == model_ref)
                .count()
        }

        fn is_pinned(&self, key: &ShapeKey) -> bool {
            key.program == AneProgram::Single
                && self.smallest_rungs.get(&key.model_ref) == Some(&key.shape)
        }

        /// Removes a request from the queue, wherever it is.
        fn leave_queue(&mut self, ticket: u64) -> bool {
            let queued = self.waiters.len();
            self.waiters.retain(|waiter| *waiter != ticket);
            self.waiter_keys.remove(&ticket);
            self.waiter_bounds.remove(&ticket);
            for slot in self.slots.values_mut() {
                slot.joined.remove(&ticket);
            }
            if self.starved_head == Some(ticket) {
                self.starved_head = None;
            }
            self.waiters.len() != queued
        }

        /// Whether `worker_id` cannot compile now: it is being recovered (a
        /// retiring worker is too, until its exit is confirmed). Its queued
        /// compiles keep their place but are not elected, so they cannot hold
        /// up other workers' compiles. A retired worker's queued compiles are
        /// dropped when it retires, and the scheduler answers its later
        /// requests with the retirement before queuing anything.
        fn worker_unavailable(&self, worker_id: &str) -> bool {
            self.recovering.contains(worker_id)
        }

        /// Predicted compile time of `key`: what it took last time on this
        /// machine, else the worker's seed scaled by the machine factor.
        fn estimate(&self, key: &ShapeKey, seed: std::time::Duration) -> std::time::Duration {
            if seed.is_zero() {
                return seed;
            }
            if let Some(measured) =
                self.measured_costs
                    .get(&(key.model_ref.clone(), key.shape, key.program))
            {
                return *measured;
            }
            seed.mul_f64(self.machine_factor.unwrap_or(1.0))
        }

        /// Records a standalone compile so later predictions use it. Only
        /// single-shape programs are learned: the seed describes one shape,
        /// so a packed program measured against it would skew the factor.
        fn learn(&mut self, key: &ShapeKey, seed: std::time::Duration, took: std::time::Duration) {
            if seed.is_zero() || took.is_zero() || key.program != AneProgram::Single {
                return;
            }
            self.measured_costs
                .insert((key.model_ref.clone(), key.shape, key.program), took);
            let ratio = took.as_secs_f64() / seed.as_secs_f64();
            self.machine_factor = Some(match self.machine_factor {
                Some(factor) => (factor + ratio) / 2.0,
                None => ratio,
            });
        }

        /// Whether the module-wide compile gate is held: a compile is
        /// running, or one failed and its worker's exit is not confirmed yet
        /// (that compile may still be running in the old process). A failed
        /// slot is removed only by recovery or retirement after confirmed exit.
        fn compile_running(&self) -> bool {
            self.slots
                .values()
                .any(|slot| matches!(slot.state, SlotState::Admitting | SlotState::Failed))
        }

        fn running_remaining(&self, now: tokio::time::Instant) -> std::time::Duration {
            self.slots
                .values()
                .map(|slot| slot.compile_remaining(now))
                .sum()
        }

        /// The tightest bound of the requests waiting for `key`'s entry.
        fn entry_bound(&self, key: &ShapeKey) -> Option<tokio::time::Instant> {
            self.waiter_keys
                .iter()
                .filter(|(_, waiting)| *waiting == key)
                .filter_map(|(ticket, _)| self.waiter_bounds.get(ticket).copied())
                .min()
        }

        /// Whether a request is queued for `key`.
        fn requested(&self, key: &ShapeKey) -> bool {
            self.waiter_keys.values().any(|waiting| waiting == key)
        }

        /// When `key` is predicted to be resident: the rest of the running
        /// compile, then every queued compile ordered before it, then its own,
        /// each at its current estimate. Entries of workers that cannot compile
        /// now are not elected, so they are not counted ahead. `None` when
        /// `key` is neither compiling nor queued.
        fn predicted_ready(
            &self,
            key: &ShapeKey,
            now: tokio::time::Instant,
        ) -> Option<tokio::time::Instant> {
            if let Some(slot) = self.slots.get(key) {
                return (slot.state == SlotState::Admitting)
                    .then(|| now + slot.compile_remaining(now));
            }
            let entry = self.entries.get(key)?;
            let ahead: std::time::Duration = self
                .entries
                .iter()
                .filter(|(other, queued)| {
                    queued.order() < entry.order() && !self.worker_unavailable(&other.worker_id)
                })
                .map(|(other, queued)| self.estimate(other, queued.seed))
                .sum();
            Some(now + self.running_remaining(now) + ahead + self.estimate(key, entry.seed))
        }

        /// The shape freed to retry an admission the worker refused for lack
        /// of Neural Engine resources: any unleased resident shape outside a
        /// handoff and outside a recovering worker, by eviction preference.
        fn resource_victim(
            &self,
            now: tokio::time::Instant,
        ) -> Option<(ShapeKey, u64, Arc<dyn AneShapeWorker>, u64)> {
            self.slots
                .iter()
                .filter(|(_, slot)| {
                    slot.state == SlotState::Resident
                        && slot.leases == 0
                        && !slot.handing_off(now)
                        && !self.recovering.contains(slot.worker.worker_id())
                })
                .min_by_key(|(candidate, slot)| self.eviction_rank(candidate, slot))
                .map(|(candidate, slot)| {
                    (
                        candidate.clone(),
                        slot.id,
                        slot.worker.clone(),
                        slot.generation,
                    )
                })
        }

        /// A worker's queued compiles rejoin the queue when its recovery
        /// ends. They keep their start keys, but, like a newcomer, one is moved
        /// behind any queued compile whose accepted bound it would otherwise
        /// break. The returning compiles are placed together, in their order:
        /// each one is checked with the cost of those already placed ahead of
        /// the same compile, so several cannot together break a bound that
        /// each alone would keep. Call while the worker is still marked
        /// recovering, so the predictions it checks do not count these
        /// entries yet.
        fn rejoin(&mut self, worker_id: &str, now: tokio::time::Instant) {
            let mut keys: Vec<ShapeKey> = self
                .entries
                .keys()
                .filter(|key| key.worker_id == worker_id)
                .cloned()
                .collect();
            keys.sort_by_key(|key| self.entries[key].order());
            let mut placed: Vec<((tokio::time::Instant, u64), std::time::Duration)> = Vec::new();
            for key in keys {
                let entry = &self.entries[&key];
                let cost = self.estimate(&key, entry.seed);
                let mut order = entry.order();
                let mut others: Vec<(ShapeKey, (tokio::time::Instant, u64))> = self
                    .entries
                    .iter()
                    .filter(|(other, _)| other.worker_id != worker_id)
                    .map(|(other, queued)| (other.clone(), queued.order()))
                    .collect();
                others.sort_by_key(|(_, other_order)| *other_order);
                let mut moved = false;
                for (other, other_order) in others {
                    if order >= other_order {
                        continue;
                    }
                    let promised = self.entry_bound(&other);
                    let ready = self.predicted_ready(&other, now);
                    if let (Some(bound), Some(ready)) = (promised, ready) {
                        let returned_ahead: std::time::Duration = placed
                            .iter()
                            .filter(|(placed_order, _)| *placed_order < other_order)
                            .map(|(_, placed_cost)| *placed_cost)
                            .sum();
                        if ready + returned_ahead + cost > bound {
                            self.next_entry += 1;
                            order = (other_order.0, self.next_entry);
                            moved = true;
                        }
                    }
                }
                if moved {
                    let entry = self.entries.get_mut(&key).expect("listed above");
                    entry.start_key = order.0;
                    entry.seq = order.1;
                }
                placed.push((order, cost));
            }
        }

        /// Queues a cold compile for `key`. Its start key is `now + cost`,
        /// unless starting it ahead of an entry would push that entry past
        /// the bound its requests were promised; then it goes behind it.
        fn create_entry(
            &mut self,
            key: &ShapeKey,
            worker: &Arc<dyn AneShapeWorker>,
            executables: usize,
            now: tokio::time::Instant,
        ) {
            let seed = worker.compile_cost_seed(&key.model_ref, key.shape, executables);
            let cost = self.estimate(key, seed);
            let mut start_key = now + cost;
            let mut queued: Vec<_> = self.entries.iter().collect();
            queued.sort_by_key(|(_, entry)| entry.order());
            for (other, entry) in queued {
                if start_key >= entry.start_key {
                    continue;
                }
                let promised = self.entry_bound(other);
                let ready = self.predicted_ready(other, now);
                if let (Some(bound), Some(ready)) = (promised, ready) {
                    if ready + cost > bound {
                        start_key = entry.start_key;
                    }
                }
            }
            self.next_entry += 1;
            self.entries.insert(
                key.clone(),
                QueuedCompile {
                    seq: self.next_entry,
                    enqueued_at: now,
                    start_key,
                    seed,
                    worker: worker.clone(),
                    executables,
                    promise: None,
                },
            );
        }

        fn eviction_rank(&self, key: &ShapeKey, slot: &Slot) -> (bool, bool, bool, u64) {
            (
                !slot.worker.stream_idle(),
                self.waiter_keys.values().any(|waiting| waiting == key),
                self.is_pinned(key),
                slot.last_used,
            )
        }

        fn sample(&mut self) {
            self.stats.samples += 1;
            #[cfg(test)]
            {
                let mut map: BTreeMap<String, Vec<usize>> = BTreeMap::new();
                for key in self.slots.keys() {
                    map.entry(key.model_ref.clone())
                        .or_default()
                        .push(key.shape);
                }
                self.recorded_samples.push(map);
            }
            let mut per_model: BTreeMap<&str, usize> = BTreeMap::new();
            for key in self.slots.keys() {
                *per_model.entry(key.model_ref.as_str()).or_default() += 1;
            }
            let largest = per_model.values().copied().max().unwrap_or(0);
            self.stats.max_resident_per_model = self.stats.max_resident_per_model.max(largest);
            self.stats.max_resident_total = self.stats.max_resident_total.max(self.slots.len());
        }
    }

    struct Inner {
        limits: AneResidencyLimits,
        executable_budget: usize,
        wait_timeout: std::time::Duration,
        state: Mutex<State>,
        changed: Notify,
        /// The scheduler's mailbox, created when the supervisor is first used.
        scheduler: Mutex<Option<tokio::sync::mpsc::UnboundedSender<scheduler::Msg>>>,
        /// Run the scheduler on a thread of its own (the production
        /// supervisor, shared by every direct-ANE engine, each of which has its
        /// own runtime) rather than on the first caller's runtime (tests, whose
        /// paused clocks the scheduler must share).
        dedicated_scheduler: bool,
        /// Set when the scheduler's loop ends: every request then gets
        /// `SchedulerStopped` (the lane is closed) instead of hanging.
        closed: std::sync::atomic::AtomicBool,
        next_waiter: AtomicU64,
        /// Whether the scheduler wants to hear about dropped leases (a starved
        /// head, a draining recovery or retirement, or an open handoff).
        watch_leases: std::sync::atomic::AtomicBool,
        /// Lets a test abort the scheduler task to check that its requests
        /// fail fast.
        #[cfg(test)]
        scheduler_task: Mutex<Option<tokio::task::AbortHandle>>,
        #[cfg(test)]
        admission_gate: Option<Arc<tokio::sync::Mutex<()>>>,
        #[cfg(test)]
        lease_decision_gate: Mutex<Option<(Arc<Notify>, Arc<Notify>)>>,
        /// Holds the scheduler once, right after a pass's decisions and before
        /// it arms its wake-up timer, so a test can move the clock between the
        /// two.
        #[cfg(test)]
        wait_decision_gate: Mutex<Option<(Arc<Notify>, Arc<Notify>)>>,
        /// Makes the scheduler take this long over each `Nudge` message, so a
        /// test can keep its mailbox busier than it can drain.
        #[cfg(test)]
        slow_nudges: Mutex<Option<std::time::Duration>>,
        /// Given to the scheduler when it starts; it is dropped with the
        /// scheduler, so a test can see the scheduler end.
        #[cfg(test)]
        scheduler_exit_signal: Mutex<Option<std::sync::mpsc::Sender<()>>>,
        #[cfg(unix)]
        lane_lock: Mutex<Option<AneDirectLaneLock>>,
        #[cfg(unix)]
        lane_path: Option<PathBuf>,
    }

    impl Inner {
        fn lock(&self) -> MutexGuard<'_, State> {
            self.state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
        }

        fn new(limits: AneResidencyLimits, dedicated_scheduler: bool) -> Self {
            Self {
                limits,
                executable_budget: ANE_RESIDENT_EXECUTABLES_TOTAL,
                wait_timeout: ANE_ADMISSION_WAIT_TIMEOUT,
                state: Mutex::new(State::default()),
                changed: Notify::new(),
                scheduler: Mutex::new(None),
                dedicated_scheduler,
                closed: std::sync::atomic::AtomicBool::new(false),
                next_waiter: AtomicU64::new(0),
                watch_leases: std::sync::atomic::AtomicBool::new(false),
                #[cfg(test)]
                scheduler_task: Mutex::new(None),
                #[cfg(test)]
                admission_gate: None,
                #[cfg(test)]
                lease_decision_gate: Mutex::new(None),
                #[cfg(test)]
                wait_decision_gate: Mutex::new(None),
                #[cfg(test)]
                slow_nudges: Mutex::new(None),
                #[cfg(test)]
                scheduler_exit_signal: Mutex::new(None),
                #[cfg(unix)]
                lane_lock: Mutex::new(None),
                #[cfg(unix)]
                lane_path: None,
            }
        }

        /// Sends `message` to the scheduler, starting it on first use.
        fn send(self: &Arc<Self>, message: scheduler::Msg) -> Result<(), AneResidencyError> {
            if self.closed.load(Ordering::Acquire) {
                return Err(AneResidencyError::SchedulerStopped);
            }
            let mut mailbox = self
                .scheduler
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if mailbox.is_none() {
                let (task, sender) = scheduler::Scheduler::new(self);
                if self.dedicated_scheduler || tokio::runtime::Handle::try_current().is_err() {
                    let started = std::thread::Builder::new()
                        .name("ane-compile-scheduler".into())
                        .spawn(move || {
                            match tokio::runtime::Builder::new_current_thread()
                                .enable_all()
                                .build()
                            {
                                Ok(runtime) => runtime.block_on(task.run()),
                                Err(error) => tracing::error!(
                                    target: "worker",
                                    error = %error,
                                    "direct-ANE compile scheduler could not start a runtime"
                                ),
                            }
                        });
                    if let Err(error) = started {
                        self.closed.store(true, Ordering::Release);
                        return Err(AneResidencyError::Io(error));
                    }
                } else {
                    let handle = tokio::spawn(task.run());
                    #[cfg(test)]
                    {
                        *self.scheduler_task.lock().unwrap() = Some(handle.abort_handle());
                    }
                    #[cfg(not(test))]
                    drop(handle);
                }
                *mailbox = Some(sender);
            }
            let sender = mailbox.as_ref().expect("started above");
            if sender.send(message).is_err() {
                self.closed.store(true, Ordering::Release);
                return Err(AneResidencyError::SchedulerStopped);
            }
            Ok(())
        }

        /// Wakes the scheduler for a change requests do not report in
        /// detail. Does nothing before the scheduler has started.
        fn nudge(&self) {
            if let Some(sender) = self
                .scheduler
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .as_ref()
            {
                let _ = sender.send(scheduler::Msg::Nudge);
            }
        }
    }

    /// A request's place in the handoff cohort of a shape it waited for.
    /// Dropping it removes the request from the cohort if it has not leased
    /// through it (leasing removes it already), and wakes the scheduler when
    /// that leaves the cohort empty, so the handoff barrier ends at once.
    struct CohortMembership {
        inner: Arc<Inner>,
        key: ShapeKey,
        id: u64,
    }

    impl Drop for CohortMembership {
        fn drop(&mut self) {
            let emptied = {
                let mut state = self.inner.lock();
                state
                    .slots
                    .get_mut(&self.key)
                    .is_some_and(|slot| slot.joined.remove(&self.id) && slot.joined.is_empty())
            };
            if emptied {
                self.inner.nudge();
            }
        }
    }

    /// Wakes the scheduler when a request stops waiting for its answer, so a
    /// compile or eviction nobody needs any more can be withdrawn promptly.
    struct DepartureNudge {
        inner: Arc<Inner>,
        armed: bool,
    }

    impl Drop for DepartureNudge {
        fn drop(&mut self) {
            if self.armed {
                self.inner.nudge();
            }
        }
    }

    /// Lets a serving request's expiry find the supervisor and worker it
    /// waits on, for a fresh prediction.
    fn link_request(inner: &Arc<Inner>, worker_id: &str) {
        let _ = REQUEST_WAITS.try_with(|waits| {
            *waits
                .link
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) =
                Some((Arc::downgrade(inner), worker_id.to_owned()));
        });
    }

    /// Owns the direct-ANE residency budget. Cheap to clone; clones share it.
    #[derive(Clone)]
    pub struct AneResidencySupervisor {
        inner: Arc<Inner>,
    }

    impl AneResidencySupervisor {
        /// A supervisor that holds no lane lock (the caller holds it, or tests).
        pub fn new(limits: AneResidencyLimits) -> Self {
            Self {
                inner: Arc::new(Inner::new(limits, false)),
            }
        }

        /// A supervisor that initially owns the lane lock; a drained pause may
        /// release it and re-acquire it before admitting new work. This is the
        /// module's supervisor, shared by engines with runtimes of their own,
        /// so its scheduler runs on a thread of its own.
        #[cfg(unix)]
        pub fn with_lane_lock(limits: AneResidencyLimits, lock: AneDirectLaneLock) -> Self {
            let mut inner = Inner::new(limits, true);
            inner.lane_path = Some(lock.path().to_path_buf());
            inner.lane_lock = Mutex::new(Some(lock));
            Self {
                inner: Arc::new(inner),
            }
        }

        /// Takes the lane lock at `~/Library/Caches/ck-synapse/ane-direct.lock`;
        /// fails with `ane_lane_busy` while another process or its workers hold it.
        #[cfg(unix)]
        pub fn acquire_lane(limits: AneResidencyLimits) -> Result<Self, AneResidencyError> {
            let path = AneDirectLaneLock::default_path().ok_or_else(|| {
                AneResidencyError::Io(io::Error::new(
                    io::ErrorKind::NotFound,
                    "HOME is not set, so the direct-ANE lane lock has no location",
                ))
            })?;
            Ok(Self::with_lane_lock(
                limits,
                AneDirectLaneLock::acquire(&path)?,
            ))
        }

        /// The held lane lock; spawned direct-ANE workers must inherit it
        /// (see [`inheritable_lock_stdio`]).
        #[cfg(unix)]
        pub fn inherited_lane_lock(&self) -> Result<Option<Arc<std::fs::File>>, AneResidencyError> {
            let lock = self.inner.lane_lock.lock().expect("lane lock");
            if lock.is_none() {
                if let Some(path) = &self.inner.lane_path {
                    return Err(AneResidencyError::LaneBusy { path: path.clone() });
                }
            }
            Ok(lock.as_ref().map(AneDirectLaneLock::inheritable))
        }

        /// Core ML can borrow an owned descriptor without depending on direct-ANE
        /// ownership. An absent or released lock is not a Core ML load refusal.
        #[cfg(unix)]
        pub(crate) fn held_lane_lock(&self) -> Option<Arc<std::fs::File>> {
            self.inner
                .lane_lock
                .lock()
                .expect("lane lock")
                .as_ref()
                .map(AneDirectLaneLock::inheritable)
        }

        /// Release only after every owner has confirmed shutdown. A retained
        /// parent-side descriptor means shutdown has not yet freed the hardware.
        #[cfg(unix)]
        pub fn release_lane(&self) -> Result<(), AneResidencyError> {
            let mut lock = self.inner.lane_lock.lock().expect("lane lock");
            if let Some(held) = lock.as_ref() {
                if Arc::strong_count(&held.file) != 1 {
                    return Err(AneResidencyError::LaneBusy {
                        path: held.path.clone(),
                    });
                }
            }
            lock.take();
            Ok(())
        }

        /// Reopen the lane lock path through a fresh file description before new loads.
        #[cfg(unix)]
        pub fn reacquire_lane(&self) -> Result<(), AneResidencyError> {
            let mut lock = self.inner.lane_lock.lock().expect("lane lock");
            if lock.is_none() {
                if let Some(path) = &self.inner.lane_path {
                    *lock = Some(AneDirectLaneLock::acquire(path)?);
                }
            }
            Ok(())
        }

        /// Enable capture before the first admission; existing events are not reconstructed.
        pub fn enable_observations(&mut self) {
            self.inner
                .lock()
                .inventory_sink
                .get_or_insert_with(Vec::new);
        }

        /// Retained ADMITTED events and the independently maintained admission total.
        pub fn observations(&self) -> Option<(Vec<AnePlacementInventory>, u64)> {
            let state = self.inner.lock();
            state.inventory_sink.as_ref().map(|sink| {
                (
                    sink.iter().map(|inventory| (**inventory).clone()).collect(),
                    state.stats.admitted,
                )
            })
        }

        /// Recover a transport-faulted owner after the inference task releases its
        /// lease. The scheduler drains the worker's leases and in-flight compile
        /// (bounded by the admission wait timeout), restarts it, and answers
        /// whether the replacement came up, or why not.
        pub async fn recover(
            &self,
            worker: &Arc<dyn AneShapeWorker>,
        ) -> Result<(), AneResidencyError> {
            self.recover_at_generation(worker, worker.generation())
                .await
        }

        /// Ignore faults from an owner generation that has already been replaced.
        ///
        /// The wait is bounded end to end: the drain (admission wait timeout),
        /// the scheduler's wait for the old process's exit
        /// (`EXIT_CONFIRMATION_GIVE_UP`, after which it answers that the worker
        /// is closed), and the replacement's start and model restore (another
        /// admission wait timeout).
        pub async fn recover_at_generation(
            &self,
            worker: &Arc<dyn AneShapeWorker>,
            generation: u64,
        ) -> Result<(), AneResidencyError> {
            let (reply, answer) = tokio::sync::oneshot::channel();
            if self
                .inner
                .send(scheduler::Msg::Recover {
                    worker: worker.clone(),
                    generation,
                    reply,
                })
                .is_err()
            {
                return Err(AneResidencyError::SchedulerStopped);
            }
            let bound = self.inner.wait_timeout * 2 + scheduler::EXIT_CONFIRMATION_GIVE_UP;
            match tokio::time::timeout(bound, answer).await {
                Ok(Ok(outcome)) => outcome,
                Ok(Err(_)) => Err(AneResidencyError::SchedulerStopped),
                Err(_) => Err(AneResidencyError::Channel(format!(
                    "worker recovery did not finish within {} s",
                    bound.as_secs()
                ))),
            }
        }

        /// Stop new leases and retain reservations until the owner has confirmed
        /// exit. If `shutdown` fails, the reservations stay charged but no
        /// longer hold the compile gate.
        pub async fn retire<F>(
            &self,
            worker: &Arc<dyn AneShapeWorker>,
            shutdown: F,
        ) -> Result<(), AneResidencyError>
        where
            F: Future<Output = Result<(), AneResidencyError>>,
        {
            let (reply, drained) = tokio::sync::oneshot::channel();
            if self
                .inner
                .send(scheduler::Msg::Retire {
                    worker: worker.clone(),
                    reply,
                })
                .is_ok()
            {
                let _ = drained.await;
            }
            let result = shutdown.await;
            let (reply, done) = tokio::sync::oneshot::channel();
            if self
                .inner
                .send(scheduler::Msg::RetireDone {
                    worker_id: worker.worker_id().to_owned(),
                    exit_confirmed: result.is_ok(),
                    reply,
                })
                .is_ok()
            {
                let _ = done.await;
            }
            result
        }

        pub fn limits(&self) -> AneResidencyLimits {
            self.inner.limits
        }

        pub fn stats(&self) -> AneResidencyStats {
            self.inner.lock().stats.clone()
        }

        /// Executables reserved against the budget, and leases currently held.
        #[cfg(all(test, unix))]
        pub(crate) fn reservation_snapshot(&self) -> (usize, u32) {
            let state = self.inner.lock();
            (
                state.reserved_executables(),
                state.slots.values().map(|slot| slot.leases).sum(),
            )
        }

        /// Shapes counted against the budget, per model ref.
        pub fn resident_shapes(&self) -> BTreeMap<String, Vec<usize>> {
            let state = self.inner.lock();
            let mut shapes: BTreeMap<String, Vec<usize>> = BTreeMap::new();
            for key in state.slots.keys() {
                shapes
                    .entry(key.model_ref.clone())
                    .or_default()
                    .push(key.shape);
            }
            shapes
        }

        /// Waits for `shape` of `model_ref` to be resident on `worker` and
        /// leases it. The lease must be dropped before the caller waits for
        /// anything else in the supervisor.
        pub async fn lease(
            &self,
            worker: &Arc<dyn AneShapeWorker>,
            model_ref: &str,
            shape: usize,
        ) -> Result<AneShapeLease, AneResidencyError> {
            self.lease_until(worker, model_ref, shape, None).await
        }

        /// Deadline expiry stops queued admissions; active RPCs finish accounting
        /// before their late result is discarded.
        pub async fn lease_until(
            &self,
            worker: &Arc<dyn AneShapeWorker>,
            model_ref: &str,
            shape: usize,
            request_deadline: Option<tokio::time::Instant>,
        ) -> Result<AneShapeLease, AneResidencyError> {
            #[cfg(test)]
            let started = std::time::Instant::now();
            let result = self
                .lease_inner(
                    worker,
                    model_ref,
                    shape,
                    AneProgram::Single,
                    request_deadline,
                )
                .await;
            #[cfg(test)]
            let _ = ADMISSION_WAITS.try_with(|waits| {
                waits.lock().unwrap().push(AdmissionWait {
                    shape,
                    elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                    admitted: result.is_ok(),
                })
            });
            result
        }

        async fn lease_inner(
            &self,
            worker: &Arc<dyn AneShapeWorker>,
            model_ref: &str,
            shape: usize,
            program: AneProgram,
            request_deadline: Option<tokio::time::Instant>,
        ) -> Result<AneShapeLease, AneResidencyError> {
            let key = ShapeKey {
                worker_id: worker.worker_id().to_owned(),
                model_ref: model_ref.to_string(),
                shape,
                program,
            };
            let executables = if matches!(program, AneProgram::Packed { .. }) {
                28
            } else {
                worker.executable_count(model_ref)?
            };
            if executables > self.inner.executable_budget {
                return Err(resource_refusal(AneResidencyError::Channel(
                    "shape exceeds executable budget".into(),
                )));
            }
            let deadline = tokio::time::Instant::now() + self.inner.wait_timeout;
            // The latest moment this request can still use its shape.
            let bound = request_deadline.map_or(deadline, |request| request.min(deadline));
            link_request(&self.inner, worker.worker_id());
            let mut cohort = None;
            // Leaves the handoff cohort if this request stops before leasing
            // through it, so the barrier does not wait for a member that left.
            let mut membership: Option<CohortMembership> = None;
            loop {
                if request_deadline.is_some_and(|deadline| tokio::time::Instant::now() >= deadline)
                {
                    #[cfg(test)]
                    pass_worker_gate(worker).await;
                    return Err(lease_deadline_expired());
                }
                ensure_request_live()?;
                #[cfg(test)]
                {
                    let gate = self.inner.lease_decision_gate.lock().unwrap().take();
                    if let Some((checked, continue_grant)) = gate {
                        checked.notify_one();
                        continue_grant.notified().await;
                    }
                }
                ensure_request_live()?;
                let member = membership.take();
                let leased = self.try_lease(&key, worker, cohort.take());
                drop(member);
                if let Some(lease) = leased? {
                    #[cfg(test)]
                    admission_trace::event("lease_grant", model_ref, shape);
                    return Ok(lease);
                }
                // Not resident (or a barrier applies): ask the scheduler and
                // wait for its one answer. Going away just drops the receiver.
                let id = self.inner.next_waiter.fetch_add(1, Ordering::Relaxed) + 1;
                let (reply, answer) = tokio::sync::oneshot::channel();
                self.inner.send(scheduler::Msg::Want(scheduler::Want {
                    id,
                    key: key.clone(),
                    worker: worker.clone(),
                    executables,
                    bound: Some(bound),
                    deadline: request_deadline,
                    reply: scheduler::Reply::Lease(reply),
                    trace: scheduler::Trace::current(),
                }))?;
                enter_wait(WaitPhase::Scheduled { id, shape });
                let left = DepartureNudge {
                    inner: self.inner.clone(),
                    armed: true,
                };
                let mut answer = answer;
                let answer = match cancellable_wait(&mut answer).await {
                    // A request whose turn started the compile keeps its serving
                    // task (and with it the lane's serving gate) until the
                    // compile's reply has drained, even after its caller went
                    // away; it then discards the result.
                    Err(AneResidencyError::Abandoned) if self.owns_running_compile(&key, id) => {
                        let _ = answer.await;
                        Err(AneResidencyError::Abandoned)
                    }
                    other => other,
                };
                let mut left = left;
                left.armed = matches!(answer, Err(_) | Ok(Err(_)));
                match answer {
                    Err(error) => return Err(error),
                    Ok(Err(_)) => return Err(AneResidencyError::SchedulerStopped),
                    Ok(Ok(scheduler::Answer::Lease { lease, compiled })) => {
                        enter_wait(if compiled {
                            WaitPhase::Compile(shape)
                        } else {
                            WaitPhase::Other
                        });
                        #[cfg(test)]
                        admission_trace::event("lease_grant", model_ref, shape);
                        return Ok(lease);
                    }
                    Ok(Ok(scheduler::Answer::Ready)) => {
                        enter_wait(WaitPhase::Compile(shape));
                        cohort = Some(id);
                    }
                    Ok(Ok(scheduler::Answer::Joined(member))) => {
                        enter_wait(WaitPhase::Compile(shape));
                        cohort = Some(id);
                        membership = Some(member);
                    }
                    Ok(Ok(scheduler::Answer::Failed(error))) => {
                        #[cfg(test)]
                        if matches!(
                            error,
                            AneResidencyError::ShapeCompiling { .. }
                                | AneResidencyError::DeadlineExceeded
                        ) {
                            pass_worker_gate(worker).await;
                        }
                        return Err(error);
                    }
                }
            }
        }

        fn packed_key(worker: &Arc<dyn AneShapeWorker>, model_ref: &str) -> ShapeKey {
            ShapeKey {
                worker_id: worker.worker_id().into(),
                model_ref: model_ref.into(),
                shape: synapse_core::PACKED_WIDTH,
                program: AneProgram::Packed {
                    program_revision: synapse_core::PACKED_PROGRAM_REVISION,
                },
            }
        }

        /// Borrow only an already compiled program whose output matched standalone
        /// output bit-for-bit on the qualification fixtures. A missing program
        /// returns None instead of triggering compilation. A stopped scheduler
        /// closes the lane for packed requests too, as for `try_lease`.
        fn try_packed_lease(
            &self,
            worker: &Arc<dyn AneShapeWorker>,
            model_ref: &str,
        ) -> Result<Option<AneShapeLease>, AneResidencyError> {
            if self.inner.closed.load(Ordering::Acquire) {
                return Err(AneResidencyError::SchedulerStopped);
            }
            let key = Self::packed_key(worker, model_ref);
            let mut state = self.inner.lock();
            // Packed arrivals have no budget-queue ticket. Do not let their fast
            // path repeatedly lease the only program a starved admission can evict.
            if state.starved_head.is_some() {
                return Ok(None);
            }
            if state.recovering.contains(worker.worker_id())
                || state.retired.contains(worker.worker_id())
            {
                return Ok(None);
            }
            let now = state.tick();
            let Some(slot) = state.slots.get_mut(&key) else {
                return Ok(None);
            };
            if slot.state != SlotState::Resident
                || !slot.packed_qualified
                || slot.generation != worker.generation()
            {
                return Ok(None);
            }
            slot.leases += 1;
            slot.last_used = now;
            Ok(Some(AneShapeLease {
                inner: self.inner.clone(),
                key,
                slot_id: slot.id,
                inventory: slot.inventory.clone(),
            }))
        }

        fn packed_resident(
            &self,
            worker: &Arc<dyn AneShapeWorker>,
            model_ref: &str,
        ) -> Result<bool, AneResidencyError> {
            let lease = self.try_packed_lease(worker, model_ref)?;
            Ok(lease.is_some())
        }

        fn preparation_possible(&self, worker: &Arc<dyn AneShapeWorker>) -> bool {
            let state = self.inner.lock();
            if !state.waiters.is_empty()
                || state.recovering.contains(worker.worker_id())
                || state.retired.contains(worker.worker_id())
            {
                return false;
            }
            if state.slots.values().any(|slot| {
                slot.worker.worker_id() == worker.worker_id()
                    && (slot.leases != 0 || slot.state == SlotState::Admitting)
            }) {
                return false;
            }
            let reclaimable: usize = state
                .slots
                .values()
                .filter(|slot| {
                    slot.state == SlotState::Resident
                        && slot.leases == 0
                        && !state.recovering.contains(slot.worker.worker_id())
                })
                .map(|slot| slot.executables)
                .sum();
            state.reserved_executables().saturating_add(28)
                <= self.inner.executable_budget.saturating_add(reclaimable)
        }

        /// Preparation does not wait for the shared executable budget behind live
        /// requests or for this worker's own work: the scheduler answers `None`
        /// at once unless the packed program can be compiled now (evicting if
        /// needed), and retry happens after the lane's last in-flight request
        /// completes.
        async fn prepare_packed_lease(
            &self,
            worker: &Arc<dyn AneShapeWorker>,
            model_ref: &str,
        ) -> Result<Option<AneShapeLease>, AneResidencyError> {
            let key = Self::packed_key(worker, model_ref);
            let id = self.inner.next_waiter.fetch_add(1, Ordering::Relaxed) + 1;
            let (reply, answer) = tokio::sync::oneshot::channel();
            self.inner.send(scheduler::Msg::Want(scheduler::Want {
                id,
                key,
                worker: worker.clone(),
                executables: 28,
                bound: None,
                deadline: None,
                reply: scheduler::Reply::Prepare(reply),
                trace: scheduler::Trace::current(),
            }))?;
            answer
                .await
                .map_err(|_| AneResidencyError::SchedulerStopped)?
        }

        fn qualify_packed(&self, lease: &AneShapeLease) {
            if let Some(slot) = self
                .inner
                .lock()
                .slots
                .get_mut(&lease.key)
                .filter(|slot| slot.id == lease.slot_id)
            {
                slot.packed_qualified = true;
            }
        }

        /// Runs a request whose sequences have `lengths` tokens, one ladder
        /// rung at a time in ascending order. `run` receives the rung and the
        /// indices of the sequences padded to it, and must return one output
        /// per index; it runs while that rung's lease is held, and the lease is
        /// released before the next rung is requested. Outputs come back in
        /// the order of `lengths`.
        pub async fn run_by_rung<T, F, Fut>(
            &self,
            worker: &Arc<dyn AneShapeWorker>,
            model_ref: &str,
            lengths: &[usize],
            run: F,
        ) -> Result<Vec<T>, AneResidencyError>
        where
            F: FnMut(usize, Vec<usize>) -> Fut,
            Fut: Future<Output = Result<Vec<T>, AneResidencyError>>,
        {
            self.run_by_published_rung(worker, model_ref, lengths, &ANE_SHAPE_LADDER, run)
                .await
        }

        /// Runs against the ladder advertised by the loaded worker, preserving
        /// the same lease lifetime and output ordering as `run_by_rung`.
        pub async fn run_by_published_rung<T, F, Fut>(
            &self,
            worker: &Arc<dyn AneShapeWorker>,
            model_ref: &str,
            lengths: &[usize],
            ladder: &[usize],
            run: F,
        ) -> Result<Vec<T>, AneResidencyError>
        where
            F: FnMut(usize, Vec<usize>) -> Fut,
            Fut: Future<Output = Result<Vec<T>, AneResidencyError>>,
        {
            self.run_by_published_rung_until(worker, model_ref, lengths, ladder, None, run)
                .await
        }

        pub async fn run_by_published_rung_until<T, F, Fut>(
            &self,
            worker: &Arc<dyn AneShapeWorker>,
            model_ref: &str,
            lengths: &[usize],
            ladder: &[usize],
            deadline: Option<tokio::time::Instant>,
            mut run: F,
        ) -> Result<Vec<T>, AneResidencyError>
        where
            F: FnMut(usize, Vec<usize>) -> Fut,
            Fut: Future<Output = Result<Vec<T>, AneResidencyError>>,
        {
            if ladder.is_empty()
                || ladder[0] == 0
                || ladder.windows(2).any(|pair| pair[0] >= pair[1])
            {
                return Err(AneResidencyError::Channel(
                    "invalid published bucket ladder".into(),
                ));
            }
            self.inner
                .lock()
                .smallest_rungs
                .insert(model_ref.to_string(), ladder[0]);
            let mut rungs: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
            for (index, &n_tokens) in lengths.iter().enumerate() {
                let rung = ladder
                    .iter()
                    .copied()
                    .find(|rung| *rung >= n_tokens)
                    .ok_or(AneResidencyError::SequenceTooLong {
                        n_tokens,
                        max: ladder[ladder.len() - 1],
                    })?;
                rungs.entry(rung).or_default().push(index);
            }
            let mut outputs: Vec<Option<T>> = (0..lengths.len()).map(|_| None).collect();
            for (rung, indices) in rungs {
                ensure_request_live()?;
                let lease = self.lease_until(worker, model_ref, rung, deadline).await?;
                if deadline.is_some_and(|deadline| tokio::time::Instant::now() >= deadline) {
                    drop(lease);
                    return Err(lease_deadline_expired());
                }
                ensure_request_live()?;
                let results = run(rung, indices.clone()).await;
                drop(lease);
                let results = results?;
                if results.len() != indices.len() {
                    return Err(AneResidencyError::Channel(format!(
                        "rung {rung} returned {} outputs for {} sequences",
                        results.len(),
                        indices.len()
                    )));
                }
                for (index, value) in indices.into_iter().zip(results) {
                    outputs[index] = Some(value);
                }
            }
            Ok(outputs
                .into_iter()
                .map(|value| value.expect("every sequence belongs to exactly one rung"))
                .collect())
        }

        /// The lease fast path: one lock of the state, no message. Leases
        /// `key` if it is resident, of the worker's current generation, and
        /// no scheduler barrier applies. A barrier applies while a starved
        /// head waits for a victim (requests then lease in queue order through
        /// the scheduler), except for `cohort`, a request the scheduler told
        /// its compile is ready, which may lease during the 250 ms handoff.
        fn try_lease(
            &self,
            key: &ShapeKey,
            worker: &Arc<dyn AneShapeWorker>,
            cohort: Option<u64>,
        ) -> Result<Option<AneShapeLease>, AneResidencyError> {
            if self.inner.closed.load(Ordering::Acquire) {
                return Err(AneResidencyError::SchedulerStopped);
            }
            let mut state = self.inner.lock();
            // A retired worker is also marked recovering until its exit is
            // confirmed, and has no slots after; the scheduler then answers
            // its requests with the retirement.
            if state.recovering.contains(worker.worker_id()) {
                return Ok(None);
            }
            let now = tokio::time::Instant::now();
            let starved = state.starved_head.is_some();
            let tick = state.tick();
            let Some(slot) = state.slots.get_mut(key) else {
                return Ok(None);
            };
            if slot.state != SlotState::Resident || slot.generation != worker.generation() {
                return Ok(None);
            }
            let member =
                cohort.is_some_and(|id| slot.joined.contains(&id) && slot.handing_off(now));
            if !member && starved {
                return Ok(None);
            }
            slot.leases += 1;
            slot.last_used = tick;
            if let Some(id) = cohort.filter(|_| member) {
                slot.joined.remove(&id);
            }
            let lease = AneShapeLease {
                inner: self.inner.clone(),
                key: key.clone(),
                slot_id: slot.id,
                inventory: slot.inventory.clone(),
            };
            let handoff_ended = member && slot.joined.is_empty();
            drop(state);
            if handoff_ended {
                // The last cohort member leased: the head waiting behind the
                // handoff may take its turn now.
                self.inner.nudge();
            }
            Ok(Some(lease))
        }

        /// Whether request `id` reserved the compile of `key` and that compile
        /// has already reached the worker. Before then a cancelled owner simply
        /// leaves, and the scheduler withdraws the compile if nobody else waits
        /// for it and no promise is live, so nothing reaches the wire.
        fn owns_running_compile(&self, key: &ShapeKey, id: u64) -> bool {
            self.inner.lock().slots.get(key).is_some_and(|slot| {
                slot.state == SlotState::Admitting
                    && slot.owner == Some(id)
                    && slot.dispatched_at.is_some()
            })
        }

        /// Test only: evicts a slot the test has marked `Evicting`, through
        /// the scheduler; returns whether the worker acknowledged it.
        #[cfg(test)]
        async fn evict(
            &self,
            _owner: Arc<dyn AneShapeWorker>,
            key: ShapeKey,
            slot_id: u64,
        ) -> bool {
            let (reply, answer) = tokio::sync::oneshot::channel();
            if self
                .inner
                .send(scheduler::Msg::Evict {
                    key,
                    slot_id,
                    reply,
                })
                .is_err()
            {
                return false;
            }
            answer.await.unwrap_or(false)
        }
    }

    fn resource_refusal(error: AneResidencyError) -> AneResidencyError {
        AneResidencyError::ResourcesExhausted {
            msg: error.to_string(),
            retry_after_ms: ANE_RESOURCES_RETRY_AFTER_MS,
        }
    }

    /// Recovers `worker` through the supervisor's scheduler (see
    /// [`AneResidencySupervisor::recover`]).
    #[cfg(test)]
    async fn recover_worker(inner: &Arc<Inner>, worker: &Arc<dyn AneShapeWorker>) -> bool {
        AneResidencySupervisor {
            inner: inner.clone(),
        }
        .recover(worker)
        .await
        .is_ok()
    }

    /// A held lease on one resident shape; dropping it releases the lease.
    pub struct AneShapeLease {
        inner: Arc<Inner>,
        key: ShapeKey,
        slot_id: u64,
        inventory: Option<Arc<AnePlacementInventory>>,
    }

    impl AneShapeLease {
        pub fn model_ref(&self) -> &str {
            &self.key.model_ref
        }

        pub fn shape(&self) -> usize {
            self.key.shape
        }

        /// The placement inventory the worker reported when it admitted the shape.
        pub fn inventory(&self) -> Option<&AnePlacementInventory> {
            self.inventory.as_deref()
        }
    }

    impl Drop for AneShapeLease {
        fn drop(&mut self) {
            let mut state = self.inner.lock();
            let now = state.tick();
            if let Some(slot) = state
                .slots
                .get_mut(&self.key)
                .filter(|slot| slot.id == self.slot_id)
            {
                slot.leases = slot.leases.saturating_sub(1);
                slot.last_used = now;
            }
            drop(state);
            self.inner.changed.notify_waiters();
            // Only a scheduler waiting on leases (a starved head, a draining
            // recovery or retirement, an open handoff) needs to hear about
            // this; the inference hot path sends nothing otherwise.
            if self.inner.watch_leases.load(Ordering::Acquire) {
                self.inner.nudge();
            }
        }
    }

    /// Opens a fresh connection to a direct-ANE worker process (spawning it
    /// and completing its HELLO).
    pub type AneWorkerConnector<S> = Box<
        dyn Fn() -> BoxFuture<'static, Result<AneWorkerSession<S>, AneResidencyError>>
            + Send
            + Sync,
    >;

    /// A stream plus its owning process's exit confirmation. After the stream
    /// closes, `confirm_exit` must wait for the child's exit status, not merely
    /// send a termination signal. In-memory tests model a process with a server
    /// task and confirm that task ended instead.
    pub struct AneWorkerSession<S> {
        /// The worker process's id, for logs, when known.
        pub pid: Option<u32>,
        pub stream: S,
        pub packed: Option<synapse_core::PackedCapability>,
        pub confirm_exit: BoxFuture<'static, Result<(), AneResidencyError>>,
    }

    #[cfg(test)]
    struct CompileRpcClock(std::time::Instant);
    #[cfg(test)]
    impl Drop for CompileRpcClock {
        fn drop(&mut self) {
            let _ = COMPILE_RPC_MS.try_with(|elapsed| {
                *elapsed.lock().unwrap() = Some(self.0.elapsed().as_secs_f64() * 1000.0)
            });
        }
    }

    // Dropping a partially completed exchange leaves unread framing on the socket.
    // Fault it before the stream mutex is released so no next request can consume it.
    struct ExchangeFaultGuard<'a> {
        faulted: &'a std::sync::atomic::AtomicBool,
        completed: bool,
    }
    impl Drop for ExchangeFaultGuard<'_> {
        fn drop(&mut self) {
            if !self.completed {
                self.faulted.store(true, Ordering::Relaxed);
            }
        }
    }

    /// Where a test may hold an expired wait before it is classified.
    #[cfg(test)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum GateSite {
        /// The caller of `infer_permitted`, after its own timer expired.
        Caller,
        /// The serving task's own lease, stream or permit wait.
        Inner,
    }

    /// Holds expired waits at each `GateSite` until the test releases it, and
    /// counts how many have arrived there.
    #[cfg(test)]
    pub struct ClassifyGate {
        arrived: [AtomicU64; 2],
        changed: Notify,
        open: [tokio::sync::Semaphore; 2],
    }

    #[cfg(test)]
    impl ClassifyGate {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                arrived: [AtomicU64::new(0), AtomicU64::new(0)],
                changed: Notify::new(),
                open: [
                    tokio::sync::Semaphore::new(0),
                    tokio::sync::Semaphore::new(0),
                ],
            })
        }

        fn index(site: GateSite) -> usize {
            match site {
                GateSite::Caller => 0,
                GateSite::Inner => 1,
            }
        }

        async fn pass(&self, site: GateSite) {
            self.arrived[Self::index(site)].fetch_add(1, Ordering::SeqCst);
            self.changed.notify_waiters();
            // Closed means released: every waiter at this site proceeds.
            let _ = self.open[Self::index(site)].acquire().await;
        }

        /// Lets every wait held (now or later) at `site` be classified.
        fn release(&self, site: GateSite) {
            self.open[Self::index(site)].close();
        }

        /// Waits until at least `count` expired waits have reached `site`.
        async fn reached(&self, site: GateSite, count: u64) {
            loop {
                let changed = self.changed.notified();
                tokio::pin!(changed);
                changed.as_mut().enable();
                if self.arrived[Self::index(site)].load(Ordering::SeqCst) >= count {
                    return;
                }
                changed.await;
            }
        }
    }

    #[cfg(test)]
    async fn pass_worker_gate(worker: &Arc<dyn AneShapeWorker>) {
        if let Some(gate) = worker.classify_gate() {
            gate.pass(GateSite::Inner).await;
        }
    }

    /// Admission exchanges kept in a channel's history. Admits are rare and
    /// each holds the stream for a whole compile, so 16 covers far more than
    /// any request's stream wait in practice.
    const ADMIT_HISTORY: usize = 16;

    /// One admission exchange's hold on the worker stream.
    #[derive(Clone, Copy, Debug)]
    struct AdmitHold {
        id: u64,
        shape: usize,
        start: tokio::time::Instant,
        /// `None` while the admit still holds the stream.
        end: Option<tokio::time::Instant>,
    }

    /// When admission (compile) exchanges held the worker stream, so a stream
    /// wait that expires can be classified by what it actually waited behind.
    #[derive(Default)]
    struct StreamHolder {
        /// The shape an admission exchange holding the stream now is compiling.
        compiling: Option<usize>,
        /// The most recent admit holds, oldest first; at most `ADMIT_HISTORY`.
        admits: VecDeque<AdmitHold>,
        /// The latest end, and its shape, among holds already dropped from
        /// `admits`. A wait that began before this instant may have overlapped
        /// a forgotten hold.
        forgotten: Option<(tokio::time::Instant, usize)>,
        next_id: u64,
    }

    impl StreamHolder {
        /// Records an admit that took the stream at `start`.
        fn begin_admit(&mut self, shape: usize, start: tokio::time::Instant) -> u64 {
            self.next_id += 1;
            let id = self.next_id;
            self.admits.push_back(AdmitHold {
                id,
                shape,
                start,
                end: None,
            });
            if self.admits.len() > ADMIT_HISTORY {
                // Only the newest hold can still be open: the stream has one
                // holder at a time, so the dropped hold has ended.
                if let Some(old) = self.admits.pop_front() {
                    let end = old.end.unwrap_or(old.start);
                    if self.forgotten.is_none_or(|(latest, _)| end >= latest) {
                        self.forgotten = Some((end, old.shape));
                    }
                }
            }
            id
        }

        fn end_admit(&mut self, id: u64, end: tokio::time::Instant) {
            if let Some(hold) = self.admits.iter_mut().find(|hold| hold.id == id) {
                hold.end = Some(end);
            }
        }

        /// The shape of an admit that held the stream at some moment of
        /// `[start, deadline]`, if any. When the history no longer reaches back
        /// to `start`, the answer is unknown and errs towards a compile. Both
        /// possible answers tell the client its request was not run; erring
        /// this way gives it `shape_compiling`, which says the same request may
        /// be sent again, instead of `deadline_exceeded`, which says it may not.
        fn compile_during(
            &self,
            start: tokio::time::Instant,
            deadline: tokio::time::Instant,
        ) -> Option<usize> {
            let now = tokio::time::Instant::now();
            self.admits
                .iter()
                .rev()
                .find(|hold| hold.start <= deadline && hold.end.unwrap_or(now) >= start)
                .map(|hold| hold.shape)
                .or_else(|| {
                    self.forgotten
                        .filter(|(end, _)| *end >= start)
                        .map(|(_, shape)| shape)
                })
        }
    }

    /// Marks one exchange as the stream holder for as long as it runs,
    /// including when it is dropped part way through, and records admit holds.
    struct HolderMark<'a> {
        holder: &'a Mutex<StreamHolder>,
        admit: Option<u64>,
    }
    impl<'a> HolderMark<'a> {
        fn hold(holder: &'a Mutex<StreamHolder>, compiling: Option<usize>) -> Self {
            let mut state = holder
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state.compiling = compiling;
            let admit =
                compiling.map(|shape| state.begin_admit(shape, tokio::time::Instant::now()));
            Self { holder, admit }
        }
    }
    impl Drop for HolderMark<'_> {
        fn drop(&mut self) {
            let mut state = self
                .holder
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state.compiling = None;
            if let Some(id) = self.admit {
                state.end_admit(id, tokio::time::Instant::now());
            }
        }
    }

    #[derive(Default)]
    struct ChannelFaults {
        strikes: usize,
        last_generation: Option<u64>,
        until: Option<std::time::Instant>,
        #[cfg(test)]
        clock: Option<std::time::Instant>,
    }

    impl ChannelFaults {
        fn now(&self) -> std::time::Instant {
            #[cfg(test)]
            if let Some(now) = self.clock {
                return now;
            }
            std::time::Instant::now()
        }
        fn remaining(&mut self) -> Option<u64> {
            let until = self.until?;
            let now = self.now();
            if now >= until {
                self.strikes = 0;
                self.last_generation = None;
                self.until = None;
                None
            } else {
                Some(
                    u64::try_from(until.duration_since(now).as_millis())
                        .unwrap_or(u64::MAX)
                        .max(1),
                )
            }
        }
    }

    /// A serial direct-ANE connection. Waiting for residency capacity must not
    /// lock the stream and block runnable inference; only wire exchanges and
    /// recovery that waits for the old process to exit hold that lock.
    pub struct AneWorkerChannel<S> {
        worker_id: String,
        max_frame: u32,
        stream: tokio::sync::Mutex<Option<AneWorkerSession<S>>>,
        connect: AneWorkerConnector<S>,
        loaded_models: Mutex<BTreeMap<String, WorkerRequest>>,
        faulted: std::sync::atomic::AtomicBool,
        // Successful LOAD/PING/compile cannot hide broken serving. Only a
        // drained serving result clears consecutive faults on this generation.
        faults: Mutex<ChannelFaults>,
        exit_unconfirmed: std::sync::atomic::AtomicBool,
        retired: std::sync::atomic::AtomicBool,
        packed_enabled: std::sync::atomic::AtomicBool,
        packed_reason_logged: std::sync::atomic::AtomicBool,
        request_counter: AtomicU64,
        stop_exchanges: Notify,
        generation: AtomicU64,
        shape_rpc_timeout: std::time::Duration,
        owner_exit_timeout: std::time::Duration,
        holder: Mutex<StreamHolder>,
        /// The current worker process's id, when the connector reports it.
        pid: Mutex<Option<u32>>,
        #[cfg(test)]
        classify_gate: Mutex<Option<Arc<ClassifyGate>>>,
        /// Replaces the pinned compile-cost seed in tests. Mock workers compile
        /// instantly, so test channels start with a zero seed (no early
        /// refusals, arrival order); the real-hardware stress clears it.
        #[cfg(test)]
        test_compile_seed: Mutex<Option<std::time::Duration>>,
    }

    impl<S> AneWorkerChannel<S>
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        pub async fn connect(
            worker_id: impl Into<String>,
            max_frame: u32,
            connect: AneWorkerConnector<S>,
        ) -> Result<Self, AneResidencyError> {
            let stream = connect().await?;
            let stream_pid = stream.pid;
            let packed_enabled =
                stream.packed.as_ref() == Some(&synapse_core::PackedCapability::tight256());
            Ok(Self {
                worker_id: worker_id.into(),
                max_frame,
                stream: tokio::sync::Mutex::new(Some(stream)),
                connect,
                loaded_models: Mutex::new(BTreeMap::new()),
                faulted: std::sync::atomic::AtomicBool::new(false),
                faults: Mutex::new(ChannelFaults::default()),
                exit_unconfirmed: std::sync::atomic::AtomicBool::new(false),
                retired: std::sync::atomic::AtomicBool::new(false),
                packed_enabled: std::sync::atomic::AtomicBool::new(packed_enabled),
                packed_reason_logged: std::sync::atomic::AtomicBool::new(false),
                request_counter: AtomicU64::new(0),
                stop_exchanges: Notify::new(),
                generation: AtomicU64::new(0),
                // Compiling/loading all 22 GTE layers at length 8192 took 115.495s
                // in debug; 600s leaves headroom for that but bounds a silent worker.
                shape_rpc_timeout: std::time::Duration::from_secs(600),
                owner_exit_timeout: std::time::Duration::from_secs(30),
                holder: Mutex::new(StreamHolder::default()),
                pid: Mutex::new(stream_pid),
                #[cfg(test)]
                classify_gate: Mutex::new(None),
                #[cfg(test)]
                test_compile_seed: Mutex::new(Some(std::time::Duration::ZERO)),
            })
        }

        /// Prevent replacement spawning while retirement drains existing leases.
        fn charge_fault(&self, generation: u64) {
            let mut faults = self.faults.lock().unwrap_or_else(|p| p.into_inner());
            if faults.last_generation != Some(generation) {
                faults.last_generation = Some(generation);
                faults.strikes += 1;
                if faults.strikes >= 2 {
                    faults.until = Some(faults.now() + super::CATALOG_QUARANTINE_COOLDOWN);
                }
            }
        }

        fn cooldown_remaining(&self) -> Option<u64> {
            self.faults
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .remaining()
        }

        pub fn begin_retirement(&self) {
            self.retired.store(true, Ordering::Release);
        }

        /// Permanently close this channel and confirm the process has exited.
        pub async fn shutdown(&self) -> Result<(), AneResidencyError> {
            self.begin_retirement();
            self.faulted.store(true, Ordering::Relaxed);
            self.stop_exchanges.notify_waiters();
            let mut guard = self.stream.lock().await;
            if self.exit_unconfirmed.load(Ordering::Relaxed) {
                return Err(AneResidencyError::Channel(
                    "owner exit remains unconfirmed".into(),
                ));
            }
            if let Some(old) = guard.take() {
                self.exit_unconfirmed.store(true, Ordering::Relaxed);
                drop(old.stream);
                tokio::time::timeout(self.owner_exit_timeout, old.confirm_exit)
                    .await
                    .map_err(|_| {
                        AneResidencyError::Channel(
                            "owner exit confirmation deadline exceeded".into(),
                        )
                    })??;
                self.exit_unconfirmed.store(false, Ordering::Relaxed);
            }
            Ok(())
        }

        pub fn request_count(&self) -> u64 {
            self.request_counter.load(Ordering::Relaxed)
        }

        pub fn next_req_id(&self, prefix: &str) -> String {
            format!(
                "{prefix}-{}",
                self.request_counter.fetch_add(1, Ordering::Relaxed)
            )
        }

        fn packing_available(&self) -> bool {
            let available = self.packed_enabled.load(Ordering::Acquire);
            if !available && !self.packed_reason_logged.swap(true, Ordering::AcqRel) {
                tracing::info!(target: "worker", worker_id = %self.worker_id, "tight packing disabled: worker has no compatible HELLO capability or refused packed execution");
            }
            available
        }

        fn disable_packing(&self, reason: &str) {
            if self.packed_enabled.swap(false, Ordering::AcqRel) {
                self.packed_reason_logged.store(true, Ordering::Release);
                tracing::warn!(target: "worker", worker_id = %self.worker_id, reason, "tight packing disabled; standalone serving remains available");
            }
        }

        /// Sends one request (and its raw frame) and reads the response (and
        /// its raw frame). A transport failure marks the connection unusable; the
        /// worker is unusable until [`AneShapeWorker::restart`].
        pub async fn exchange(
            &self,
            request: &WorkerRequest,
            raw: Option<&[u8]>,
        ) -> Result<(WorkerResponse, Option<Vec<u8>>), AneResidencyError> {
            self.exchange_in_generation(request, raw, self.generation())
                .await
        }

        async fn exchange_in_generation(
            &self,
            request: &WorkerRequest,
            raw: Option<&[u8]>,
            generation: u64,
        ) -> Result<(WorkerResponse, Option<Vec<u8>>), AneResidencyError> {
            // Waiting for a stream already used by another request is not
            // worker inactivity. The background shape-admission task owns the
            // exchange through its reply even if the waiting caller cancels.
            let mut guard = cancellable_wait(self.stream.lock()).await?;
            #[cfg(test)]
            admission_trace::rpc_event("stream_grant", request);
            self.exchange_locked(&mut guard, request, raw, generation, None)
                .await
        }

        /// Answers a caller whose deadline expired while its serving task was
        /// still waiting: the cause is that task's current wait, unless the
        /// task already classified its own expiry.
        async fn expired_wait(
            &self,
            waits: &RequestWaits,
            deadline: tokio::time::Instant,
        ) -> AneResidencyError {
            let scheduled = match *waits.lock() {
                WaitPhase::Scheduled { id, .. } => Some(id),
                _ => None,
            };
            if let Some(id) = scheduled {
                // The request waits on the compile scheduler: it decides the
                // answer, once, with its current prediction, and keeps the
                // compile queued for exactly the delay it gives.
                let answer = match waits.link() {
                    Some((inner, _)) => {
                        let (reply, answer) = tokio::sync::oneshot::channel();
                        match inner.send(scheduler::Msg::Expired { id, reply }) {
                            Ok(()) => answer
                                .await
                                .map(scheduler::AnswerKind::into_error)
                                .unwrap_or(AneResidencyError::SchedulerStopped),
                            Err(error) => error,
                        }
                    }
                    None => AneResidencyError::DeadlineExceeded,
                };
                let cause = match &answer {
                    AneResidencyError::ShapeCompiling { shape, .. } => Some(*shape),
                    _ => None,
                };
                let mut phase = waits.lock();
                if !matches!(*phase, WaitPhase::Expired(_)) {
                    *phase = WaitPhase::Expired(cause);
                }
                return answer;
            }
            let mut phase = waits.lock();
            let cause = match *phase {
                WaitPhase::Expired(first) => first,
                WaitPhase::Compile(shape) => Some(shape),
                WaitPhase::Scheduled { shape, .. } => Some(shape),
                WaitPhase::Stream(start) => self.compile_during(start, deadline),
                WaitPhase::Other => None,
            };
            *phase = WaitPhase::Expired(cause);
            drop(phase);
            match cause {
                Some(shape) => shape_compiling(shape, waits.running_compile_ready()),
                None => AneResidencyError::DeadlineExceeded,
            }
        }

        /// Lets a test hold an expired wait between its deadline expiring and
        /// the expiry being classified, separately for the caller and for the
        /// serving task's own waits.
        #[cfg(test)]
        async fn pass_classify_gate(&self, site: GateSite) {
            let gate = self.classify_gate.lock().unwrap().clone();
            if let Some(gate) = gate {
                gate.pass(site).await;
            }
        }

        fn stream_holder(&self) -> MutexGuard<'_, StreamHolder> {
            self.holder
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
        }

        /// The shape being compiled while an admission exchange holds the stream.
        pub fn compiling_shape(&self) -> Option<usize> {
            self.stream_holder().compiling
        }

        /// The shape of an admit that held this stream at some moment of a
        /// stream wait `[start, deadline]` (see `StreamHolder::compile_during`).
        fn compile_during(
            &self,
            start: tokio::time::Instant,
            deadline: tokio::time::Instant,
        ) -> Option<usize> {
            self.stream_holder().compile_during(start, deadline)
        }

        /// One inference exchange. Waiting for the stream (which an admission
        /// may hold for a whole compile) is bounded only by the request
        /// deadline: giving up before anything is written leaves the
        /// connection healthy, so it must not count as a transport fault. The
        /// execution permit is taken after the stream lock, so a request
        /// waiting behind a compile holds no permit. `io_timeout` bounds the
        /// write and read only.
        pub async fn exchange_inference<P>(
            &self,
            request: &WorkerRequest,
            raw: Option<&[u8]>,
            deadline: Option<tokio::time::Instant>,
            io_timeout: std::time::Duration,
            permit: P,
        ) -> Result<(WorkerResponse, Option<Vec<u8>>), AneResidencyError>
        where
            P: Future<Output = Result<ExchangePermit, AneResidencyError>>,
        {
            self.exchange_inference_checked(request, raw, deadline, io_timeout, None, permit)
                .await
        }

        async fn exchange_inference_checked<P>(
            &self,
            request: &WorkerRequest,
            raw: Option<&[u8]>,
            deadline: Option<tokio::time::Instant>,
            io_timeout: std::time::Duration,
            expected_dims: Option<usize>,
            permit: P,
        ) -> Result<(WorkerResponse, Option<Vec<u8>>), AneResidencyError>
        where
            P: Future<Output = Result<ExchangePermit, AneResidencyError>>,
        {
            let generation = self.generation();
            let expired = |deadline: Option<tokio::time::Instant>| {
                deadline.is_some_and(|deadline| tokio::time::Instant::now() >= deadline)
            };
            let stream_wait = tokio::time::Instant::now();
            enter_wait(WaitPhase::Stream(stream_wait));
            let mut guard = match deadline {
                Some(deadline) => {
                    match tokio::time::timeout_at(deadline, cancellable_wait(self.stream.lock()))
                        .await
                    {
                        Ok(guard) => guard?,
                        Err(_) => {
                            #[cfg(test)]
                            self.pass_classify_gate(GateSite::Inner).await;
                            return Err(deadline_expired(
                                self.compile_during(stream_wait, deadline),
                            ));
                        }
                    }
                }
                None => cancellable_wait(self.stream.lock()).await?,
            };
            // A timer can expire in the same poll that grants the lock. Nothing
            // may be written after the deadline, and the admits that held the
            // stream during this wait decide whether it was blocked on a compile.
            if let Some(deadline) = deadline.filter(|_| expired(deadline)) {
                drop(guard);
                #[cfg(test)]
                self.pass_classify_gate(GateSite::Inner).await;
                return Err(deadline_expired(self.compile_during(stream_wait, deadline)));
            }
            enter_wait(WaitPhase::Other);
            let _permit = match deadline {
                Some(deadline) => {
                    match tokio::time::timeout_at(deadline, cancellable_wait(permit)).await {
                        Ok(permit) => permit??,
                        Err(_) => {
                            drop(guard);
                            #[cfg(test)]
                            self.pass_classify_gate(GateSite::Inner).await;
                            return Err(deadline_expired(None));
                        }
                    }
                }
                None => cancellable_wait(permit).await??,
            };
            if expired(deadline) {
                return Err(deadline_expired(None));
            }
            let result = self
                .exchange_locked(&mut guard, request, raw, generation, Some(io_timeout))
                .await;
            let result = result.and_then(|(response, raw)| {
                let serving = match &response {
                    WorkerResponse::Vectors { dims, .. } => {
                        if expected_dims.is_some_and(|expected| expected != *dims) {
                            return Err(AneResidencyError::Channel(
                                "inference output dimensions do not match the model".into(),
                            ));
                        }
                        true
                    }
                    WorkerResponse::Scores { .. } => true,
                    _ => false,
                };
                if serving {
                    let values =
                        synapse_core::decode_f32_frame(raw.as_deref().ok_or_else(|| {
                            AneResidencyError::Channel("missing inference raw frame".into())
                        })?)
                        .map_err(|error| AneResidencyError::Channel(error.to_string()))?;
                    if values.iter().any(|value| !value.is_finite()) {
                        return Err(AneResidencyError::Channel(
                            "nonfinite inference output".into(),
                        ));
                    }
                    // Validation remains inside the stream lock: a later request
                    // or recovery cannot turn this generation's rejected result
                    // into a healthy completion and erase its preceding fault.
                    if self.generation() == generation && !self.faulted.load(Ordering::Acquire) {
                        let mut faults = self.faults.lock().unwrap_or_else(|p| p.into_inner());
                        faults.strikes = 0;
                        faults.last_generation = None;
                        faults.until = None;
                    }
                }
                Ok((response, raw))
            });
            if matches!(&result, Err(AneResidencyError::Channel(_)))
                && self.generation() == generation
                && !self.faulted.swap(true, Ordering::AcqRel)
            {
                self.charge_fault(generation);
            }
            result
        }

        async fn exchange_locked(
            &self,
            guard: &mut Option<AneWorkerSession<S>>,
            request: &WorkerRequest,
            raw: Option<&[u8]>,
            generation: u64,
            idle_override: Option<std::time::Duration>,
        ) -> Result<(WorkerResponse, Option<Vec<u8>>), AneResidencyError> {
            if let Some(retry_after_ms) = self.cooldown_remaining() {
                return Err(AneResidencyError::Quarantined { retry_after_ms });
            }
            let _holder = HolderMark::hold(
                &self.holder,
                match request {
                    WorkerRequest::AneAdmitShape { shape, .. } => Some(*shape),
                    WorkerRequest::AneAdmitPacked { .. } => Some(synapse_core::PACKED_WIDTH),
                    _ => None,
                },
            );
            if self.generation() != generation {
                return Err(AneResidencyError::Channel("stale worker generation".into()));
            }
            let stopped = self.stop_exchanges.notified();
            tokio::pin!(stopped);
            stopped.as_mut().enable();
            if self.faulted.load(Ordering::Relaxed) {
                return Err(AneResidencyError::Channel(
                    "worker connection requires confirmed-exit restart".into(),
                ));
            }
            let session = guard.as_mut().ok_or_else(|| {
                AneResidencyError::Channel(format!("worker {} is not connected", self.worker_id))
            })?;
            if matches!(
                request,
                WorkerRequest::AneAdmitPacked { .. }
                    | WorkerRequest::AneEvictPacked { .. }
                    | WorkerRequest::EmbedPacked { .. }
            ) && session.packed.as_ref() != Some(&synapse_core::PackedCapability::tight256())
            {
                return Err(AneResidencyError::WorkerErr {
                    code: "unsupported_request".into(),
                    msg: "HELLO lacks packed capability".into(),
                });
            }
            let stream = &mut session.stream;
            let max_frame = self.max_frame;
            ensure_request_live()?;
            let mut exchange_guard = ExchangeFaultGuard {
                faulted: &self.faulted,
                completed: false,
            };
            if matches!(
                request,
                WorkerRequest::AneAdmitShape { .. } | WorkerRequest::AneAdmitPacked { .. }
            ) {
                scheduler::mark_compile_dispatched();
            }
            #[cfg(test)]
            let compile_clock = matches!(request, WorkerRequest::AneAdmitShape { .. })
                .then(|| CompileRpcClock(std::time::Instant::now()));
            #[cfg(test)]
            let rpc_clock = admission_trace::Rpc::start(request);
            let result = tokio::select! {
                _=stopped=>Err(AneResidencyError::Channel("worker recovery interrupted an active exchange".into())),
                result=async {
                    let test_idle = idle_override.or_else(|| matches!(request,
                        WorkerRequest::AneAdmitShape { .. } | WorkerRequest::AneAdmitPacked { .. }
                        | WorkerRequest::AneEvictShape { .. } | WorkerRequest::AneEvictPacked { .. })
                        .then_some(self.shape_rpc_timeout));
                    super::liveness::exchange(stream, request, raw, max_frame, false, test_idle.map(|window| (window, window))).await
                        .map_err(|error| AneResidencyError::Channel(error.to_string()))
                }=>result
            };
            #[cfg(test)]
            drop(compile_clock);
            #[cfg(test)]
            drop(rpc_clock);
            match result {
                Ok((response, raw)) => {
                    exchange_guard.completed = true;
                    if matches!(
                        request,
                        WorkerRequest::AneAdmitPacked { .. } | WorkerRequest::EmbedPacked { .. }
                    ) && matches!(&response, WorkerResponse::Err { code, .. } if code == "unsupported_request")
                    {
                        self.disable_packing("worker answered unsupported_request");
                    }
                    let mut models = self.loaded_models.lock().unwrap_or_else(|p| p.into_inner());
                    match (request, &response) {
                        (
                            WorkerRequest::Load { req_id, .. },
                            WorkerResponse::Loaded {
                                req_id: got,
                                model_ref,
                                ..
                            },
                        ) if req_id == got => {
                            models.insert(model_ref.clone(), request.clone());
                        }
                        (
                            WorkerRequest::Unload { req_id, model_ref },
                            WorkerResponse::Unloaded { req_id: got },
                        ) if req_id == got => {
                            models.remove(model_ref);
                        }
                        _ => {}
                    }
                    Ok((response, raw))
                }
                Err(error) => {
                    if self.generation() == generation && !self.faulted.swap(true, Ordering::AcqRel)
                    {
                        self.charge_fault(generation);
                    }
                    Err(error)
                }
            }
        }
    }

    impl<S> AneShapeWorker for AneWorkerChannel<S>
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        fn worker_id(&self) -> &str {
            &self.worker_id
        }
        fn stream_idle(&self) -> bool {
            self.stream.try_lock().is_ok()
        }
        #[cfg(test)]
        fn classify_gate(&self) -> Option<Arc<ClassifyGate>> {
            self.classify_gate.lock().unwrap().clone()
        }
        fn compile_cost_seed(
            &self,
            model_ref: &str,
            shape: usize,
            layers: usize,
        ) -> std::time::Duration {
            #[cfg(test)]
            if let Some(seed) = *self.test_compile_seed.lock().unwrap() {
                return seed;
            }
            pinned_compile_cost_seed(model_ref, shape, layers)
        }
        fn generation(&self) -> u64 {
            self.generation.load(Ordering::Acquire)
        }
        fn exit_unconfirmed(&self) -> bool {
            self.exit_unconfirmed.load(Ordering::Acquire)
        }

        fn admit_shape<'a>(
            &'a self,
            model_ref: &'a str,
            shape: usize,
        ) -> BoxFuture<'a, Result<AnePlacementInventory, AneResidencyError>> {
            self.admit_shape_at_generation(self.generation(), model_ref, shape)
        }
        fn admit_shape_at_generation<'a>(
            &'a self,
            generation: u64,
            model_ref: &'a str,
            shape: usize,
        ) -> BoxFuture<'a, Result<AnePlacementInventory, AneResidencyError>> {
            Box::pin(async move {
                let req_id = self.next_req_id("admit");
                let request = WorkerRequest::AneAdmitShape {
                    req_id: req_id.clone(),
                    model_ref: model_ref.to_string(),
                    shape,
                };
                match self
                    .exchange_in_generation(&request, None, generation)
                    .await?
                {
                    (
                        WorkerResponse::Admitted {
                            req_id: got,
                            inventory,
                        },
                        _,
                    ) if got == req_id => Ok(inventory),
                    (WorkerResponse::Err { code, msg, .. }, _) => {
                        Err(AneResidencyError::WorkerErr { code, msg })
                    }
                    (other, _) => Err(AneResidencyError::Channel(format!(
                        "ANE_ADMIT_SHAPE returned unexpected response {other:?}"
                    ))),
                }
            })
        }

        fn admit_program<'a>(
            &'a self,
            generation: u64,
            model_ref: &'a str,
            shape: usize,
            program: AneProgram,
        ) -> BoxFuture<'a, Result<AnePlacementInventory, AneResidencyError>> {
            if program == AneProgram::Single {
                return self.admit_shape_at_generation(generation, model_ref, shape);
            }
            Box::pin(async move {
                if !self.packing_available() {
                    return Err(AneResidencyError::WorkerErr {
                        code: "unsupported_request".into(),
                        msg: "packed capability unavailable".into(),
                    });
                }
                let AneProgram::Packed { program_revision } = program else {
                    unreachable!()
                };
                let req_id = self.next_req_id("admit-packed");
                match self
                    .exchange_in_generation(
                        &WorkerRequest::AneAdmitPacked {
                            req_id: req_id.clone(),
                            model_ref: model_ref.into(),
                            program_revision,
                        },
                        None,
                        generation,
                    )
                    .await?
                {
                    (
                        WorkerResponse::Admitted {
                            req_id: got,
                            inventory,
                        },
                        _,
                    ) if got == req_id => Ok(inventory),
                    (WorkerResponse::Err { code, msg, .. }, _) => {
                        Err(AneResidencyError::WorkerErr { code, msg })
                    }
                    (other, _) => Err(AneResidencyError::Channel(format!(
                        "unexpected packed admission: {other:?}"
                    ))),
                }
            })
        }

        fn evict_program<'a>(
            &'a self,
            generation: u64,
            model_ref: &'a str,
            shape: usize,
            program: AneProgram,
        ) -> BoxFuture<'a, Result<(), AneResidencyError>> {
            if program == AneProgram::Single {
                return self.evict_shape_at_generation(generation, model_ref, shape);
            }
            Box::pin(async move {
                let AneProgram::Packed { program_revision } = program else {
                    unreachable!()
                };
                let req_id = self.next_req_id("evict-packed");
                match self
                    .exchange_in_generation(
                        &WorkerRequest::AneEvictPacked {
                            req_id: req_id.clone(),
                            model_ref: model_ref.into(),
                            program_revision,
                        },
                        None,
                        generation,
                    )
                    .await?
                {
                    (WorkerResponse::Evicted { req_id: got }, _) if got == req_id => Ok(()),
                    (WorkerResponse::Err { code, msg, .. }, _) => {
                        Err(AneResidencyError::WorkerErr { code, msg })
                    }
                    (other, _) => Err(AneResidencyError::Channel(format!(
                        "unexpected packed eviction: {other:?}"
                    ))),
                }
            })
        }

        fn evict_shape<'a>(
            &'a self,
            model_ref: &'a str,
            shape: usize,
        ) -> BoxFuture<'a, Result<(), AneResidencyError>> {
            self.evict_shape_at_generation(self.generation(), model_ref, shape)
        }

        fn evict_shape_at_generation<'a>(
            &'a self,
            generation: u64,
            model_ref: &'a str,
            shape: usize,
        ) -> BoxFuture<'a, Result<(), AneResidencyError>> {
            Box::pin(async move {
                let req_id = self.next_req_id("evict");
                let request = WorkerRequest::AneEvictShape {
                    req_id: req_id.clone(),
                    model_ref: model_ref.to_string(),
                    shape,
                };
                match self
                    .exchange_in_generation(&request, None, generation)
                    .await?
                {
                    (WorkerResponse::Evicted { req_id: got }, _) if got == req_id => Ok(()),
                    (WorkerResponse::Err { code, msg, .. }, _) => {
                        Err(AneResidencyError::WorkerErr { code, msg })
                    }
                    (other, _) => Err(AneResidencyError::Channel(format!(
                        "ANE_EVICT_SHAPE returned unexpected response {other:?}"
                    ))),
                }
            })
        }

        fn restart(&self) -> BoxFuture<'_, Result<(), AneResidencyError>> {
            let (exited, _) = tokio::sync::oneshot::channel();
            self.restart_with_exit_signal(exited)
        }

        fn process_id(&self) -> Option<u32> {
            *self
                .pid
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
        }

        fn restart_with_exit_signal<'a>(
            &'a self,
            exited: tokio::sync::oneshot::Sender<()>,
        ) -> BoxFuture<'a, Result<(), AneResidencyError>> {
            Box::pin(async move {
                // Stop active I/O before taking the stream lock. A lease or a
                // nonresponsive request cannot prevent escalation after drain timeout.
                self.faulted.store(true, Ordering::Relaxed);
                self.stop_exchanges.notify_waiters();
                let mut guard = self.stream.lock().await;
                if self.retired.load(Ordering::Acquire) {
                    return Err(AneResidencyError::Channel("worker is retired".into()));
                }
                if self.exit_unconfirmed.load(Ordering::Relaxed) {
                    return Err(AneResidencyError::Channel(
                        "old owner exit remains unconfirmed".into(),
                    ));
                }
                self.generation.fetch_add(1, Ordering::AcqRel);
                if let Some(old) = guard.take() {
                    self.exit_unconfirmed.store(true, Ordering::Relaxed);
                    drop(old.stream);
                    tokio::time::timeout(self.owner_exit_timeout,old.confirm_exit).await.map_err(|_|AneResidencyError::Channel("owner exit confirmation deadline exceeded; ownership remains closed".into()))??;
                    self.exit_unconfirmed.store(false, Ordering::Relaxed);
                }
                if self.retired.load(Ordering::Acquire) {
                    return Err(AneResidencyError::Channel(
                        "worker retired during owner exit confirmation".into(),
                    ));
                }
                // The old process is gone: what it held can be released before
                // the replacement starts.
                let _ = exited.send(());
                if self.cooldown_remaining().is_some() {
                    // The old process has exited, so the supervisor may release
                    // its executable reservations. During the fault cool-down,
                    // stop here instead of spawning and recompiling a replacement.
                    return Ok(());
                }
                let mut fresh = tokio::time::timeout(self.shape_rpc_timeout, (self.connect)())
                    .await
                    .map_err(|_| {
                        AneResidencyError::Channel(
                            "replacement connection deadline exceeded".into(),
                        )
                    })??;
                let models = self
                    .loaded_models
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .clone();
                let reload = async {
                    for (model_ref, mut request) in models {
                        let req_id = self.next_req_id("reload");
                        if let WorkerRequest::Load { req_id: id, .. } = &mut request {
                            *id = req_id.clone();
                        }
                        let (response, _) = super::liveness::exchange(
                            &mut fresh.stream,
                            &request,
                            None,
                            self.max_frame,
                            false,
                            None,
                        )
                        .await
                        .map_err(|error| AneResidencyError::Channel(error.to_string()))?;
                        match response {
                            WorkerResponse::Loaded {
                                req_id: got,
                                model_ref: got_ref,
                                ..
                            } if got == req_id && got_ref == model_ref => {}
                            other => {
                                return Err(AneResidencyError::Channel(format!(
                                    "model reload returned {other:?}"
                                )))
                            }
                        }
                    }
                    Ok::<_, AneResidencyError>(())
                }
                .await;
                if let Err(error) = reload {
                    if matches!(
                        error,
                        AneResidencyError::Channel(_) | AneResidencyError::Io(_)
                    ) {
                        self.charge_fault(self.generation());
                    }
                    self.exit_unconfirmed.store(true, Ordering::Relaxed);
                    drop(fresh.stream);
                    tokio::time::timeout(self.owner_exit_timeout, fresh.confirm_exit)
                        .await
                        .map_err(|_| {
                            AneResidencyError::Channel("replacement owner exit unconfirmed".into())
                        })??;
                    self.exit_unconfirmed.store(false, Ordering::Relaxed);
                    return Err(error);
                }
                self.packed_enabled.store(
                    fresh.packed.as_ref() == Some(&synapse_core::PackedCapability::tight256()),
                    Ordering::Release,
                );
                self.packed_reason_logged.store(false, Ordering::Release);
                *self
                    .pid
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()) = fresh.pid;
                *guard = Some(fresh);
                self.faulted.store(false, Ordering::Relaxed);
                Ok(())
            })
        }
    }

    /// A `Stdio` for a direct-ANE worker's stdin that duplicates the locked
    /// descriptor. The duplicate shares the lock, so the lock is released only
    /// when every copy of the open file description has been closed, including
    /// worker stdin and unrelated forked children's copies before exec.
    pub fn inheritable_lock_stdio(lock: &std::fs::File) -> io::Result<Stdio> {
        Ok(Stdio::from(lock.try_clone()?))
    }

    /// The advisory lock (`flock`) that allows one module process at a time to
    /// run direct-ANE workers.
    #[cfg(unix)]
    #[derive(Debug)]
    pub struct AneDirectLaneLock {
        file: Arc<std::fs::File>,
        path: PathBuf,
    }

    /// Unit tests may inspect the production path to check this guard, but must
    /// never open it: another process may be using the machine's Neural Engine.
    #[cfg(all(test, unix))]
    pub(crate) fn assert_test_ane_lock_path(path: &Path) {
        fn resolved(path: &Path) -> PathBuf {
            path.parent()
                .and_then(|parent| parent.canonicalize().ok())
                .zip(path.file_name())
                .map(|(parent, name)| parent.join(name))
                .unwrap_or_else(|| path.to_path_buf())
        }
        if let Some(production) = AneDirectLaneLock::default_path() {
            assert_ne!(
                resolved(path),
                resolved(&production),
                "module tests must not open the production ANE lane lock"
            );
        }
    }

    #[cfg(unix)]
    impl AneDirectLaneLock {
        /// `~/Library/Caches/ck-synapse/ane-direct.lock`.
        pub fn default_path() -> Option<PathBuf> {
            std::env::var_os("HOME").map(|home| PathBuf::from(home).join(ANE_DIRECT_LOCK_HOME_PATH))
        }

        /// Takes the lock without waiting. Fails with `ane_lane_busy` while an
        /// open file description holds the lock, including copies inherited by
        /// direct-ANE workers or unrelated children between fork and exec.
        // `File::try_lock` is a `flock` and avoids the unsafe code a direct call would need.
        pub fn acquire(path: &Path) -> Result<Self, AneResidencyError> {
            #[cfg(test)]
            assert_test_ane_lock_path(path);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(path)?;
            match file.try_lock() {
                Ok(()) => Ok(Self {
                    file: Arc::new(file),
                    path: path.to_path_buf(),
                }),
                Err(std::fs::TryLockError::WouldBlock) => Err(AneResidencyError::LaneBusy {
                    path: path.to_path_buf(),
                }),
                Err(std::fs::TryLockError::Error(error)) => Err(error.into()),
            }
        }

        pub fn path(&self) -> &Path {
            &self.path
        }

        /// The locked file, for [`super::WorkerHostConfig::inherited_lane_lock`].
        pub fn inheritable(&self) -> Arc<std::fs::File> {
            self.file.clone()
        }
    }

    /// Spawns and connects a direct-ANE worker process for a worker channel.
    /// The connector lives as long as its channel, and the channel lives in
    /// the supervisor's slots, so it holds the supervisor only weakly: a
    /// strong reference made a cycle that kept the supervisor, its lane lock
    /// and its scheduler thread alive after the last engine was dropped.
    #[cfg(unix)]
    fn direct_worker_connector(
        config: super::WorkerHostConfig,
        supervisor: &AneResidencySupervisor,
    ) -> AneWorkerConnector<tokio::net::UnixStream> {
        let supervisor = Arc::downgrade(&supervisor.inner);
        Box::new(move || {
            let mut config = config.clone();
            let inherited = match supervisor.upgrade() {
                Some(inner) => AneResidencySupervisor { inner }.inherited_lane_lock(),
                None => Err(AneResidencyError::SchedulerStopped),
            };
            Box::pin(async move {
                config.inherited_lane_lock = inherited?;
                let mut host = super::WorkerHost::new(config);
                host.ping()
                    .await
                    .map_err(|error| AneResidencyError::Channel(error.to_string()))?;
                let super::WorkerConnection {
                    stream,
                    mut child,
                    packed,
                    ..
                } = host.connection.take().ok_or_else(|| {
                    AneResidencyError::Channel("spawned worker has no connection".into())
                })?;
                Ok(AneWorkerSession {
                    pid: child.id(),
                    packed,
                    stream,
                    confirm_exit: Box::pin(async move {
                        match tokio::time::timeout(std::time::Duration::from_secs(2), child.wait())
                            .await
                        {
                            Ok(status) => {
                                status.map_err(AneResidencyError::Io)?;
                            }
                            Err(_) => {
                                child.start_kill().map_err(AneResidencyError::Io)?;
                                child.wait().await.map_err(AneResidencyError::Io)?;
                            }
                        }
                        Ok(())
                    }),
                })
            })
        })
    }

    /// The production owner of one direct-ANE worker and its I/O runtime.
    #[cfg(unix)]
    pub struct DirectAneEngine {
        pub serving: Arc<AneServing<tokio::net::UnixStream>>,
        runtime: Option<Arc<tokio::runtime::Runtime>>,
    }

    #[cfg(unix)]
    impl DirectAneEngine {
        pub fn load(
            mut config: super::WorkerHostConfig,
            supervisor: AneResidencySupervisor,
            artifact: &synapse_core::ValidatedArtifact,
            runtime_config: &synapse_core::RuntimeConfig,
        ) -> Result<Self, AneResidencyError> {
            let artifact_path = runtime_config
                .values
                .get("artifact_path")
                .cloned()
                .ok_or_else(|| AneResidencyError::Channel("missing artifact path".into()))?;
            config.worker_id = format!("{}-{}", config.worker_id, super::nonce_hex16());
            let runtime = Arc::new(
                tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(1)
                    .enable_all()
                    .build()
                    .map_err(AneResidencyError::Io)?,
            );
            let id = config.worker_id.clone();
            let max_frame = config.max_frame;
            let pooling = config.pooling;
            let normalize = config.normalize;
            let timeout = config.request_timeout;
            let load_timeout = super::liveness::SETUP_IDLE;
            let connector = direct_worker_connector(config, &supervisor);
            let channel =
                Arc::new(runtime.block_on(AneWorkerChannel::connect(id, max_frame, connector))?);
            let request = WorkerRequest::Load {
                req_id: channel.next_req_id("load"),
                artifact_path,
                artifact_digest: artifact.digest.clone(),
                format: artifact.format.clone(),
                runtime_config: runtime_config.values.clone(),
            };
            let loaded = runtime.block_on(async {
                let (response, _) =
                    tokio::time::timeout(load_timeout, channel.exchange(&request, None))
                        .await
                        .map_err(|_| {
                            AneResidencyError::Channel("LOAD deadline exceeded".into())
                        })??;
                match response {
                    WorkerResponse::Loaded {
                        req_id,
                        model_ref,
                        dims,
                        buckets: Some(buckets),
                        ..
                    } if request.req_id() == Some(req_id.as_str()) => AneServing::new(
                        channel.clone(),
                        supervisor.clone(),
                        AneServingMetadata {
                            model_ref,
                            buckets,
                            dims,
                            pooling,
                            normalize,
                            timeout,
                        },
                    ),
                    WorkerResponse::Err { code, msg, .. } => {
                        Err(AneResidencyError::WorkerErr { code, msg })
                    }
                    other => Err(AneResidencyError::Channel(format!(
                        "invalid LOAD response: {other:?}"
                    ))),
                }
            });
            match loaded {
                Ok(serving) => Ok(Self {
                    serving: Arc::new(serving),
                    runtime: Some(runtime),
                }),
                Err(error) => {
                    runtime.block_on(channel.shutdown())?;
                    Err(error)
                }
            }
        }

        pub async fn infer(
            &self,
            sequences: Vec<Vec<u32>>,
            rerank: bool,
            deadline: Option<tokio::time::Instant>,
        ) -> Result<Vec<Vec<f32>>, AneResidencyError> {
            self.infer_guarded(sequences, rerank, deadline, ()).await
        }

        pub async fn infer_guarded<G: Send + 'static>(
            &self,
            sequences: Vec<Vec<u32>>,
            rerank: bool,
            deadline: Option<tokio::time::Instant>,
            guards: G,
        ) -> Result<Vec<Vec<f32>>, AneResidencyError> {
            self.infer_permitted(sequences, rerank, deadline, guards, None)
                .await
        }

        /// Like `infer_guarded`, but each inference exchange first takes a
        /// permit from `permits`, after the worker stream is held.
        pub async fn infer_permitted<G: Send + 'static>(
            &self,
            sequences: Vec<Vec<u32>>,
            rerank: bool,
            deadline: Option<tokio::time::Instant>,
            guards: G,
            permits: Option<PermitSource>,
        ) -> Result<Vec<Vec<f32>>, AneResidencyError> {
            let serving = self.serving.clone();
            self.runtime
                .as_ref()
                .expect("live ANE runtime")
                .spawn(async move {
                    serving
                        .infer_permitted(sequences, rerank, deadline, guards, permits)
                        .await
                })
                .await
                .map_err(|error| AneResidencyError::Channel(error.to_string()))?
        }

        pub fn unload(&self) -> Result<(), AneResidencyError> {
            self.runtime
                .as_ref()
                .expect("live ANE runtime")
                .block_on(self.serving.retire())
        }
    }

    #[cfg(unix)]
    impl Drop for DirectAneEngine {
        fn drop(&mut self) {
            let Some(runtime) = self.runtime.take() else {
                return;
            };
            let serving = self.serving.clone();
            let (done, wait) = std::sync::mpsc::sync_channel(1);
            // Runtime teardown and child reaping cannot block a Tokio executor.
            std::thread::spawn(move || {
                serving.channel.begin_retirement();
                if let Err(error) = runtime.block_on(serving.retire()) {
                    tracing::error!(target: "worker", %error, "direct-ANE teardown could not confirm exit");
                }
                drop(serving);
                drop(runtime);
                let _ = done.send(());
            });
            let _ = wait.recv_timeout(std::time::Duration::from_secs(5));
        }
    }

    /// Metadata returned by LOAD, with the request policy supplied by the module.
    pub struct AneServingMetadata {
        pub model_ref: String,
        pub buckets: Vec<usize>,
        pub dims: usize,
        pub pooling: synapse_core::WorkerPooling,
        pub normalize: bool,
        pub timeout: std::time::Duration,
    }

    /// A serving connection whose inference tasks own their leases through the
    /// entire response read, even when their callers abandon the result.
    pub struct AneServing<S> {
        pub channel: Arc<AneWorkerChannel<S>>,
        pub supervisor: AneResidencySupervisor,
        pub metadata: AneServingMetadata,
        active_requests: Arc<std::sync::atomic::AtomicUsize>,
        preparing_packed: std::sync::atomic::AtomicBool,
        #[cfg(test)]
        recovery_gate: Mutex<Option<(Arc<Notify>, Arc<Notify>)>>,
    }

    struct RequestActivity(Arc<std::sync::atomic::AtomicUsize>);
    impl Drop for RequestActivity {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::AcqRel);
        }
    }

    struct PreparationActivity<'a>(&'a std::sync::atomic::AtomicBool);
    impl Drop for PreparationActivity<'_> {
        fn drop(&mut self) {
            self.0.store(false, Ordering::Release);
        }
    }

    fn packed_subset(passes: &[Vec<usize>]) -> (Vec<usize>, Vec<Vec<usize>>) {
        let mut indices = passes.iter().flatten().copied().collect::<Vec<_>>();
        indices.sort_unstable();
        let remapped = passes
            .iter()
            .map(|pass| {
                pass.iter()
                    .map(|i| indices.binary_search(i).expect("planned row"))
                    .collect()
            })
            .collect();
        (indices, remapped)
    }

    impl<S: AsyncRead + AsyncWrite + Unpin + Send + 'static> AneServing<S> {
        pub fn new(
            channel: Arc<AneWorkerChannel<S>>,
            supervisor: AneResidencySupervisor,
            metadata: AneServingMetadata,
        ) -> Result<Self, AneResidencyError> {
            if metadata.buckets.last() != Some(&8192)
                || metadata.buckets.first().is_none_or(|n| *n == 0)
                || metadata.buckets.windows(2).any(|pair| pair[0] >= pair[1])
                || metadata.dims == 0
            {
                return Err(AneResidencyError::Channel(
                    "LOAD returned invalid bucket or dimension metadata".into(),
                ));
            }
            Ok(Self {
                channel,
                supervisor,
                metadata,
                active_requests: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                preparing_packed: std::sync::atomic::AtomicBool::new(false),
                #[cfg(test)]
                recovery_gate: Mutex::new(None),
            })
        }

        #[cfg(test)]
        async fn hold_before_recovery(&self) {
            let gate = self.recovery_gate.lock().unwrap().take();
            if let Some((fault_observed, continue_recovery)) = gate {
                fault_observed.notify_one();
                continue_recovery.notified().await;
            }
        }

        pub async fn retire(&self) -> Result<(), AneResidencyError> {
            self.channel.begin_retirement();
            let worker: Arc<dyn AneShapeWorker> = self.channel.clone();
            self.supervisor
                .retire(&worker, self.channel.shutdown())
                .await
        }

        pub async fn infer(
            self: &Arc<Self>,
            sequences: Vec<Vec<u32>>,
            rerank: bool,
        ) -> Result<Vec<Vec<f32>>, AneResidencyError> {
            self.infer_guarded(sequences, rerank, None, ()).await
        }

        /// The detached task retains module scheduling guards until I/O and recovery finish.
        pub async fn infer_guarded<G: Send + 'static>(
            self: &Arc<Self>,
            sequences: Vec<Vec<u32>>,
            rerank: bool,
            deadline: Option<tokio::time::Instant>,
            guards: G,
        ) -> Result<Vec<Vec<f32>>, AneResidencyError> {
            self.infer_permitted(sequences, rerank, deadline, guards, None)
                .await
        }

        /// The detached task retains `guards` until I/O and recovery finish.
        /// Each inference exchange takes a permit from `permits` only once it
        /// holds the worker stream. A deadline that expires while the worker
        /// is compiling a shape is reported as `ShapeCompiling`.
        pub async fn infer_permitted<G: Send + 'static>(
            self: &Arc<Self>,
            sequences: Vec<Vec<u32>>,
            rerank: bool,
            deadline: Option<tokio::time::Instant>,
            guards: G,
            permits: Option<PermitSource>,
        ) -> Result<Vec<Vec<f32>>, AneResidencyError> {
            self.infer_mode(sequences, rerank, deadline, guards, permits, false)
                .await
        }

        /// Explicit packed-inference entry point. Only programs prepared by
        /// prepare_packed are eligible; requests never compile or run qualification.
        pub async fn infer_packed(
            self: &Arc<Self>,
            sequences: Vec<Vec<u32>>,
        ) -> Result<Vec<Vec<f32>>, AneResidencyError> {
            self.infer_packed_permitted(sequences, None, (), None).await
        }

        pub async fn infer_packed_permitted<G: Send + 'static>(
            self: &Arc<Self>,
            sequences: Vec<Vec<u32>>,
            deadline: Option<tokio::time::Instant>,
            guards: G,
            permits: Option<PermitSource>,
        ) -> Result<Vec<Vec<f32>>, AneResidencyError> {
            self.infer_mode(sequences, false, deadline, guards, permits, true)
                .await
        }

        /// Compile the packed program and compare fixed fixtures with independent
        /// standalone output at each row's smallest fitting shape. False means
        /// contention prevented preparation or packing was disabled; standalone
        /// serving remains usable.
        pub async fn prepare_packed(self: &Arc<Self>) -> Result<bool, AneResidencyError> {
            let serving = self.clone();
            tokio::spawn(async move { serving.prepare_packed_inner().await })
                .await
                .map_err(|e| {
                    AneResidencyError::Channel(format!("packed preparation task failed: {e}"))
                })?
        }

        async fn prepare_packed_inner(self: &Arc<Self>) -> Result<bool, AneResidencyError> {
            if !self
                .metadata
                .model_ref
                .starts_with("ane-direct:qwen3-embedding-0.6b.ane-direct-worker:")
            {
                return Ok(false);
            }
            if !self.channel.packing_available()
                || self.active_requests.load(Ordering::Acquire) != 0
                || self.preparing_packed.swap(true, Ordering::AcqRel)
            {
                return Ok(false);
            }
            let _preparing = PreparationActivity(&self.preparing_packed);
            let worker: Arc<dyn AneShapeWorker> = self.channel.clone();
            if self
                .supervisor
                .packed_resident(&worker, &self.metadata.model_ref)?
            {
                return Ok(true);
            }
            if !self.supervisor.preparation_possible(&worker) {
                return Ok(false);
            }
            let fixtures = synapse_core::packed_qualification_fixtures_v1();
            let mut references = Vec::new();
            for fixture in &fixtures {
                match self.infer(fixture.clone(), false).await {
                    Ok(vectors) => references.push(vectors),
                    Err(error) if error.code() == Some(ERR_ANE_RESOURCES_EXHAUSTED) => {
                        return Ok(false)
                    }
                    Err(error) => return Err(error),
                }
            }
            if self.active_requests.load(Ordering::Acquire) != 0 {
                return Ok(false);
            }
            let lease = match self
                .supervisor
                .prepare_packed_lease(&worker, &self.metadata.model_ref)
                .await
            {
                Ok(Some(lease)) => lease,
                Ok(None) => return Ok(false),
                Err(error) if error.code() == Some("unsupported_request") => {
                    self.channel
                        .disable_packing("worker refused packed admission");
                    return Ok(false);
                }
                Err(error) => return Err(error),
            };
            for (fixture, reference) in fixtures.iter().zip(&references) {
                let lengths = fixture.iter().map(Vec::len).collect::<Vec<_>>();
                let plan = synapse_core::plan_packed(&lengths)
                    .map_err(|e| AneResidencyError::Channel(e.to_string()))?;
                let (indices, passes) = packed_subset(&plan.passes);
                let selected = indices.iter().map(|&i| fixture[i].clone()).collect();
                let fault_generation = worker.generation();
                let vectors = match self
                    .exchange_sequences_mode(selected, false, None, None, Some(passes))
                    .await
                {
                    Ok(vectors) => vectors,
                    Err(error) if error.code() == Some("unsupported_request") => {
                        self.channel
                            .disable_packing("worker refused packed qualification");
                        return Ok(false);
                    }
                    Err(error @ AneResidencyError::Channel(_)) => {
                        // Recovery waits for leases to drain. Release qualification's
                        // own lease before joining the generation-scoped recovery.
                        drop(lease);
                        #[cfg(test)]
                        self.hold_before_recovery().await;
                        let _ = self
                            .supervisor
                            .recover_at_generation(&worker, fault_generation)
                            .await;
                        return Err(error);
                    }
                    Err(error) => return Err(error),
                };
                for (&index, vector) in indices.iter().zip(&vectors) {
                    if !vector
                        .iter()
                        .map(|v| v.to_bits())
                        .eq(reference[index].iter().map(|v| v.to_bits()))
                    {
                        self.channel.disable_packing(&format!(
                            "qualification-v1 mismatch at row {index}, length {}",
                            lengths[index]
                        ));
                        return Ok(false);
                    }
                }
            }
            self.supervisor.qualify_packed(&lease);
            Ok(true)
        }

        async fn infer_mode<G: Send + 'static>(
            self: &Arc<Self>,
            sequences: Vec<Vec<u32>>,
            rerank: bool,
            deadline: Option<tokio::time::Instant>,
            guards: G,
            permits: Option<PermitSource>,
            packing: bool,
        ) -> Result<Vec<Vec<f32>>, AneResidencyError> {
            if self.channel.retired.load(Ordering::Acquire) {
                return Err(AneResidencyError::Channel("worker is retired".into()));
            }
            // Validate the entire request before either admission or transport I/O.
            if let Some(sequence) = sequences.iter().find(|sequence| sequence.len() > 8192) {
                return Err(AneResidencyError::SequenceTooLong {
                    n_tokens: sequence.len(),
                    max: 8192,
                });
            }
            if sequences.iter().any(Vec::is_empty) {
                return Err(AneResidencyError::WorkerErr {
                    code: "invalid_request".into(),
                    msg: "empty token sequence".into(),
                });
            }
            let serving = self.clone();
            self.active_requests.fetch_add(1, Ordering::AcqRel);
            let activity = RequestActivity(self.active_requests.clone());
            let waits = Arc::new(RequestWaits::new());
            let _abandonment = RequestAbandonment(waits.clone());
            let task = tokio::spawn(REQUEST_WAITS.scope(waits.clone(), async move {
                let _guards = guards;
                let _activity = activity;
                let lengths: Vec<_> = sequences.iter().map(Vec::len).collect();
                let worker: Arc<dyn AneShapeWorker> = serving.channel.clone();
                if let Some(retry_after_ms) = serving.channel.cooldown_remaining() {
                    return Err(AneResidencyError::Quarantined { retry_after_ms });
                }
                if serving.channel.faulted.load(Ordering::Acquire) {
                    match serving.supervisor.recover(&worker).await {
                        Ok(()) => {}
                        Err(
                            error @ (AneResidencyError::WorkerClosed { .. }
                            | AneResidencyError::SchedulerStopped),
                        ) => return Err(error),
                        Err(_) => {
                            return Err(AneResidencyError::Channel(
                                "worker recovery could not confirm exit".into(),
                            ))
                        }
                    }
                }
                let fault_generation = Arc::new(AtomicU64::new(worker.generation()));
                let result = if packing && !rerank {
                    serving
                        .run_mixed(&worker, sequences, deadline, permits, &fault_generation)
                        .await
                } else {
                    serving
                        .supervisor
                        .run_by_published_rung_until(
                            &worker,
                            &serving.metadata.model_ref,
                            &lengths,
                            &serving.metadata.buckets,
                            deadline,
                            |_, indices| {
                                fault_generation.store(worker.generation(), Ordering::Release);
                                let serving = serving.clone();
                                let permits = permits.clone();
                                let selected: Vec<_> = indices
                                    .iter()
                                    .map(|&index| sequences[index].clone())
                                    .collect();
                                async move {
                                    serving
                                        .exchange_sequences(selected, rerank, deadline, permits)
                                        .await
                                }
                            },
                        )
                        .await
                };
                if matches!(&result, Err(AneResidencyError::Channel(_))) {
                    #[cfg(test)]
                    serving.hold_before_recovery().await;
                    let _ = serving
                        .supervisor
                        .recover_at_generation(&worker, fault_generation.load(Ordering::Acquire))
                        .await;
                }
                result
            }));
            let joined = match deadline {
                Some(deadline) => match tokio::time::timeout_at(deadline, task).await {
                    Ok(joined) => joined,
                    Err(_) => {
                        #[cfg(test)]
                        self.channel.pass_classify_gate(GateSite::Caller).await;
                        return Err(self.channel.expired_wait(&waits, deadline).await);
                    }
                },
                None => task.await,
            };
            joined.map_err(|error| {
                AneResidencyError::Channel(format!("inference task failed: {error}"))
            })?
        }

        async fn exchange_sequences(
            &self,
            sequences: Vec<Vec<u32>>,
            rerank: bool,
            deadline: Option<tokio::time::Instant>,
            permits: Option<PermitSource>,
        ) -> Result<Vec<Vec<f32>>, AneResidencyError> {
            self.exchange_sequences_mode(sequences, rerank, deadline, permits, None)
                .await
        }

        async fn run_mixed(
            self: &Arc<Self>,
            worker: &Arc<dyn AneShapeWorker>,
            sequences: Vec<Vec<u32>>,
            deadline: Option<tokio::time::Instant>,
            permits: Option<PermitSource>,
            fault_generation: &AtomicU64,
        ) -> Result<Vec<Vec<f32>>, AneResidencyError> {
            let lengths = sequences.iter().map(Vec::len).collect::<Vec<_>>();
            ensure_request_live()?;
            let mut output = vec![Vec::new(); sequences.len()];
            let mut singles: Vec<_> = (0..sequences.len()).collect();
            if self.channel.packing_available()
                && self
                    .supervisor
                    .packed_resident(worker, &self.metadata.model_ref)?
            {
                let plan = synapse_core::plan_packed(&lengths)
                    .map_err(|e| AneResidencyError::Channel(e.to_string()))?;
                // Recheck that the packed program is still resident when borrowing it.
                // If it was evicted after planning, run its planned rows individually
                // instead of compiling another packed program on the request path.
                if !plan.passes.is_empty() {
                    #[cfg(test)]
                    {
                        let gate = self
                            .supervisor
                            .inner
                            .lease_decision_gate
                            .lock()
                            .unwrap()
                            .take();
                        if let Some((planned, continue_lease)) = gate {
                            planned.notify_one();
                            continue_lease.notified().await;
                        }
                    }
                    ensure_request_live()?;
                    if let Some(lease) = self
                        .supervisor
                        .try_packed_lease(worker, &self.metadata.model_ref)?
                    {
                        fault_generation.store(worker.generation(), Ordering::Release);
                        let (indices, passes) = packed_subset(&plan.passes);
                        let selected = indices.iter().map(|&i| sequences[i].clone()).collect();
                        let result = self
                            .exchange_sequences_mode(
                                selected,
                                false,
                                deadline,
                                permits.clone(),
                                Some(passes),
                            )
                            .await;
                        drop(lease);
                        match result {
                            Ok(vectors) => {
                                for (&index, vector) in indices.iter().zip(vectors) {
                                    output[index] = vector;
                                }
                                singles = plan.singles;
                            }
                            Err(error) if error.code() == Some("unsupported_request") => {
                                self.channel
                                    .disable_packing("worker refused packed execution");
                            }
                            Err(error) => return Err(error),
                        }
                    }
                }
            }
            if !singles.is_empty() {
                let selected_lengths = singles.iter().map(|&i| lengths[i]).collect::<Vec<_>>();
                let vectors = self
                    .supervisor
                    .run_by_published_rung_until(
                        worker,
                        &self.metadata.model_ref,
                        &selected_lengths,
                        &self.metadata.buckets,
                        deadline,
                        |_, indices| {
                            fault_generation.store(worker.generation(), Ordering::Release);
                            let selected = indices
                                .iter()
                                .map(|&i| sequences[singles[i]].clone())
                                .collect();
                            let permits = permits.clone();
                            async move {
                                self.exchange_sequences(selected, false, deadline, permits)
                                    .await
                            }
                        },
                    )
                    .await?;
                for (index, vector) in singles.into_iter().zip(vectors) {
                    output[index] = vector;
                }
            }
            Ok(output)
        }

        async fn exchange_sequences_mode(
            &self,
            sequences: Vec<Vec<u32>>,
            rerank: bool,
            deadline: Option<tokio::time::Instant>,
            permits: Option<PermitSource>,
            passes: Option<Vec<Vec<usize>>>,
        ) -> Result<Vec<Vec<f32>>, AneResidencyError> {
            if deadline.is_some_and(|deadline| tokio::time::Instant::now() >= deadline) {
                return Err(lease_deadline_expired());
            }
            use synapse_core::{
                decode_f32_frame, encode_i32_frame, WorkerSequence, WorkerTokenItem,
            };
            let ids: Vec<i32> = sequences
                .iter()
                .flatten()
                .map(|&id| {
                    i32::try_from(id).map_err(|_| AneResidencyError::WorkerErr {
                        code: "invalid_request".into(),
                        msg: "token ID exceeds i32".into(),
                    })
                })
                .collect::<Result<_, _>>()?;
            let req_id = self
                .channel
                .next_req_id(if rerank { "rerank" } else { "embed" });
            let request = if let Some(passes) = passes {
                WorkerRequest::EmbedPacked {
                    req_id: req_id.clone(),
                    model_ref: self.metadata.model_ref.clone(),
                    program_revision: synapse_core::PACKED_PROGRAM_REVISION,
                    items: sequences
                        .iter()
                        .enumerate()
                        .map(|(i, s)| WorkerTokenItem {
                            id: i.to_string(),
                            n_tokens: s.len(),
                        })
                        .collect(),
                    passes,
                }
            } else if rerank {
                WorkerRequest::RerankSequences {
                    req_id: req_id.clone(),
                    model_ref: self.metadata.model_ref.clone(),
                    sequences: sequences
                        .iter()
                        .map(|s| WorkerSequence { n_tokens: s.len() })
                        .collect(),
                }
            } else {
                WorkerRequest::EmbedBatch {
                    req_id: req_id.clone(),
                    model_ref: self.metadata.model_ref.clone(),
                    pooling: self.metadata.pooling,
                    normalize: self.metadata.normalize,
                    items: sequences
                        .iter()
                        .enumerate()
                        .map(|(index, s)| WorkerTokenItem {
                            id: index.to_string(),
                            n_tokens: s.len(),
                        })
                        .collect(),
                }
            };
            let raw = encode_i32_frame(&ids);
            let permit = async move {
                match permits {
                    Some(permits) => permits(deadline).await,
                    None => Ok(Box::new(()) as ExchangePermit),
                }
            };
            let (response, raw) = self
                .channel
                .exchange_inference_checked(
                    &request,
                    Some(&raw),
                    deadline,
                    self.metadata.timeout,
                    Some(if rerank { 1 } else { self.metadata.dims }),
                    permit,
                )
                .await?;
            let dims = if rerank { 1 } else { self.metadata.dims };
            match response {
                WorkerResponse::Vectors {
                    req_id: got,
                    n,
                    dims: got_dims,
                } if !rerank && got == req_id && n == sequences.len() && got_dims == dims => {}
                WorkerResponse::Scores { req_id: got } if rerank && got == req_id => {}
                WorkerResponse::Err {
                    req_id: Some(got),
                    code,
                    msg,
                } if got == req_id => return Err(AneResidencyError::WorkerErr { code, msg }),
                other => {
                    return Err(AneResidencyError::Channel(format!(
                        "unexpected inference response: {other:?}"
                    )))
                }
            }
            let raw = raw
                .ok_or_else(|| AneResidencyError::Channel("missing inference raw frame".into()))?;
            let values = decode_f32_frame(&raw)
                .map_err(|error| AneResidencyError::Channel(error.to_string()))?;
            if values.len() != sequences.len().saturating_mul(dims)
                || values.iter().any(|v| !v.is_finite())
            {
                return Err(AneResidencyError::Channel(
                    "invalid inference output count or value".into(),
                ));
            }
            if deadline.is_some_and(|deadline| tokio::time::Instant::now() >= deadline) {
                return Err(deadline_expired(None));
            }
            Ok(values.chunks_exact(dims).map(<[f32]>::to_vec).collect())
        }
    }

    #[cfg(all(test, target_os = "macos"))]
    mod hardware_tests {
        use super::*;
        use crate::worker_host::{WorkerConnection, WorkerHost, WorkerHostConfig};
        use serde_json::{json, Value};
        use std::time::Duration;
        use synapse_core::{
            decode_f32_frame, encode_i32_frame, WorkerPooling, WorkerSequence, WorkerTokenItem,
            DEFAULT_MAX_FRAME_BYTES,
        };
        use tokio::process::Child;
        type RealChannel = Arc<AneWorkerChannel<tokio::net::UnixStream>>;
        const SLUGS: [&str; 4] = [
            "gte-modernbert-base",
            "gte-reranker-modernbert-base",
            "qwen3-embedding-0.6b",
            "qwen3-reranker-0.6b",
        ];

        struct Factory {
            root: PathBuf,
            children: Arc<Mutex<Vec<Child>>>,
            lock: Arc<std::fs::File>,
        }
        impl Factory {
            fn new(root: PathBuf, lock: &AneDirectLaneLock) -> Self {
                std::fs::create_dir_all(&root).unwrap();
                Self {
                    root,
                    children: Arc::new(Mutex::new(Vec::new())),
                    lock: lock.file.clone(),
                }
            }
            async fn connect(&self, id: &str) -> RealChannel {
                let children = self.children.clone();
                let lock = self.lock.clone();
                let root = self.root.clone();
                let worker_id = id.to_owned();
                let connector: AneWorkerConnector<tokio::net::UnixStream> = Box::new(move || {
                    let children = children.clone();
                    let lock = lock.clone();
                    let root = root.clone();
                    let id = worker_id.clone();
                    Box::pin(async move {
                        let binary = std::env::var_os("ANE_TEST_WORKER").expect("ANE_TEST_WORKER");
                        let binary =
                            synapse_core::dev_binary::ckdev_binary(PathBuf::from(binary), &root)
                                .unwrap();
                        let mut config = WorkerHostConfig::new(binary, root);
                        config.worker_id = id;
                        config.inherited_lane_lock = Some(lock);
                        let mut host = WorkerHost::new(config);
                        host.ping()
                            .await
                            .map_err(|error| AneResidencyError::Channel(format!("{error:?}")))?;
                        let WorkerConnection { stream, child, .. } =
                            host.connection.take().unwrap();
                        let pid = child.id().expect("new child pid");
                        children.lock().unwrap().push(child);
                        Ok(AneWorkerSession {
                            pid: None,
                            packed: None,
                            stream,
                            confirm_exit: Box::pin(async move {
                                let mut child = {
                                    let mut children = children.lock().unwrap();
                                    let index = children
                                        .iter()
                                        .position(|child| child.id() == Some(pid))
                                        .ok_or_else(|| {
                                            AneResidencyError::Channel(
                                                "old worker child handle missing".into(),
                                            )
                                        })?;
                                    children.remove(index)
                                };
                                match tokio::time::timeout(
                                    std::time::Duration::from_secs(10),
                                    child.wait(),
                                )
                                .await
                                {
                                    Ok(status) => {
                                        status.map_err(|error| {
                                            AneResidencyError::Channel(format!(
                                                "worker exit wait: {error}"
                                            ))
                                        })?;
                                    }
                                    Err(_) => {
                                        child.start_kill().map_err(|error| {
                                            AneResidencyError::Channel(format!(
                                                "worker termination: {error}"
                                            ))
                                        })?;
                                        child.wait().await.map_err(|error| {
                                            AneResidencyError::Channel(format!(
                                                "worker exit confirmation: {error}"
                                            ))
                                        })?;
                                    }
                                }
                                Ok(())
                            }),
                        })
                    })
                });
                let channel = AneWorkerChannel::connect(id, DEFAULT_MAX_FRAME_BYTES, connector)
                    .await
                    .unwrap();
                // Real workers compile for real: use the pinned cost seeds, as
                // production does, so the compile queue orders and predicts.
                *channel.test_compile_seed.lock().unwrap() = None;
                Arc::new(channel)
            }
            fn pids(&self) -> Vec<u32> {
                self.children
                    .lock()
                    .unwrap()
                    .iter()
                    .filter_map(Child::id)
                    .collect()
            }
            async fn stop(&self) {
                let children = std::mem::take(&mut *self.children.lock().unwrap());
                for mut child in children {
                    let _ = child.kill().await;
                    let _ = child.wait().await;
                }
            }
        }
        fn manifest() -> Value {
            serde_json::from_slice(include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../bench/parity/models.json"
            )))
            .unwrap()
        }
        fn fixture(slug: &str) -> Value {
            serde_json::from_slice(&std::fs::read(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("../../bench/parity/fixtures/{slug}/{slug}.ref-v1.transformers-5.16.1.seed-0.json"))).unwrap()).unwrap()
        }
        fn fixture_ids(slug: &str, id: &str) -> Vec<i32> {
            let value = fixture(slug);
            serde_json::from_value(
                value["cases"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|case| case["id"] == id)
                    .unwrap()["input_ids"]
                    .clone(),
            )
            .unwrap()
        }
        async fn load(channel: &RealChannel, slug: &str) -> String {
            let manifest = manifest();
            let profile = format!("{slug}.ane-direct-worker");
            let digest = manifest["profiles"][&profile]["converted_package_digest"]
                .as_str()
                .unwrap();
            let root =
                PathBuf::from(std::env::var_os("ANE_TEST_PACKAGES").expect("ANE_TEST_PACKAGES"));
            let operation = if slug.contains("reranker") {
                "rerank"
            } else {
                "embed"
            };
            let request = WorkerRequest::Load {
                req_id: channel.next_req_id("load"),
                artifact_path: root
                    .join(format!("{slug}.safetensors"))
                    .to_string_lossy()
                    .into_owned(),
                artifact_digest: digest.to_owned(),
                format: "safetensors-package".into(),
                runtime_config: [
                    ("profile".into(), profile),
                    ("operation".into(), operation.into()),
                ]
                .into_iter()
                .collect(),
            };
            match channel.exchange(&request, None).await.unwrap().0 {
                WorkerResponse::Loaded { model_ref, .. } => model_ref,
                other => panic!("real LOAD failed: {other:?}"),
            }
        }
        fn signal(pid: u32, number: i32) {
            assert!(
                synapse_core::without_launch_nonce(std::process::Command::new("/bin/kill"))
                    .args([format!("-{number}"), pid.to_string()])
                    .status()
                    .unwrap()
                    .success()
            );
        }
        #[derive(Default)]
        struct Ledger {
            compile_samples: BTreeMap<String, Vec<Value>>,
            admitted: usize,
            evicted: usize,
            leased_evicts: usize,
            shape_not_admitted: usize,
            no_ane_resources: usize,
        }
        struct Observed {
            channel: RealChannel,
            supervisor: std::sync::Weak<Inner>,
            ledger: Arc<Mutex<Ledger>>,
            slug: String,
            model_ref: String,
        }
        impl AneShapeWorker for Observed {
            fn stream_idle(&self) -> bool {
                self.channel.stream_idle()
            }
            fn worker_id(&self) -> &str {
                self.channel.worker_id()
            }
            fn generation(&self) -> u64 {
                self.channel.generation()
            }
            fn admit_shape<'a>(
                &'a self,
                model_ref: &'a str,
                shape: usize,
            ) -> BoxFuture<'a, Result<AnePlacementInventory, AneResidencyError>> {
                self.admit_shape_at_generation(self.generation(), model_ref, shape)
            }
            fn admit_shape_at_generation<'a>(
                &'a self,
                generation: u64,
                model_ref: &'a str,
                shape: usize,
            ) -> BoxFuture<'a, Result<AnePlacementInventory, AneResidencyError>> {
                Box::pin(async move {
                    let elapsed = Arc::new(Mutex::new(None));
                    let result = COMPILE_RPC_MS
                        .scope(
                            elapsed.clone(),
                            self.channel
                                .admit_shape_at_generation(generation, model_ref, shape),
                        )
                        .await;
                    // No compilation RPC is written until the worker socket mutex
                    // is acquired; a refusal before that point has zero dispatched time.
                    let milliseconds = elapsed.lock().unwrap().unwrap_or(0.0);
                    self.ledger.lock().unwrap().compile_samples.entry(self.slug.clone()).or_default().push(json!({"shape":shape,"elapsed_ms":milliseconds,"admitted":result.is_ok()}));
                    match &result {
                        Ok(inventory) => {
                            let model = &manifest()["models"][&self.slug];
                            let layers = model["architecture"]["params"]["num_hidden_layers"]
                                .as_u64()
                                .unwrap() as usize;
                            let mut covered: Vec<u32> = inventory
                                .executables
                                .iter()
                                .flat_map(|entry| entry.layers.iter().copied())
                                .collect();
                            covered.sort();
                            if covered != (0..layers as u32).collect::<Vec<_>>()
                                || inventory.cpu_stages.iter().any(|stage| {
                                    ![
                                        "token_embedding",
                                        "mask_position",
                                        "rotation_in",
                                        "rotation_out",
                                        "final_norm",
                                        "pooling",
                                        "gte_classifier_head",
                                        "qwen_yes_no_readout",
                                    ]
                                    .contains(&stage.as_str())
                                })
                            {
                                return Err(AneResidencyError::Channel(
                                    "invalid ADMITTED placement inventory".into(),
                                ));
                            }
                            self.ledger.lock().unwrap().admitted += 1;
                        }
                        Err(error) if error.code() == Some(ERR_ANE_RESOURCES_EXHAUSTED) => {
                            self.ledger.lock().unwrap().no_ane_resources += 1
                        }
                        _ => (),
                    }
                    result
                })
            }
            fn evict_shape<'a>(
                &'a self,
                model_ref: &'a str,
                shape: usize,
            ) -> BoxFuture<'a, Result<(), AneResidencyError>> {
                self.evict_shape_at_generation(self.generation(), model_ref, shape)
            }
            fn evict_shape_at_generation<'a>(
                &'a self,
                generation: u64,
                model_ref: &'a str,
                shape: usize,
            ) -> BoxFuture<'a, Result<(), AneResidencyError>> {
                Box::pin(async move {
                    if self
                        .supervisor
                        .upgrade()
                        .unwrap()
                        .lock()
                        .slots
                        .get(&ShapeKey {
                            worker_id: self.worker_id().to_owned(),
                            model_ref: model_ref.into(),
                            shape,
                            program: AneProgram::Single,
                        })
                        .is_some_and(|slot| slot.leases != 0)
                    {
                        self.ledger.lock().unwrap().leased_evicts += 1;
                    }
                    self.channel
                        .evict_shape_at_generation(generation, model_ref, shape)
                        .await?;
                    self.ledger.lock().unwrap().evicted += 1;
                    Ok(())
                })
            }
            fn restart<'a>(&'a self) -> BoxFuture<'a, Result<(), AneResidencyError>> {
                Box::pin(async move { self.channel.restart().await })
            }
        }
        async fn infer(
            worker: &Observed,
            ids: Vec<i32>,
            rerank: bool,
        ) -> Result<Vec<f32>, AneResidencyError> {
            let item = WorkerTokenItem {
                id: "candidate".into(),
                n_tokens: ids.len(),
            };
            let request = if rerank {
                WorkerRequest::RerankSequences {
                    req_id: worker.channel.next_req_id("score"),
                    model_ref: worker.model_ref.clone(),
                    sequences: vec![WorkerSequence {
                        n_tokens: ids.len(),
                    }],
                }
            } else {
                WorkerRequest::EmbedBatch {
                    req_id: worker.channel.next_req_id("embed"),
                    model_ref: worker.model_ref.clone(),
                    items: vec![item],
                    pooling: WorkerPooling::Mean,
                    normalize: true,
                }
            };
            let (response, raw) = worker
                .channel
                .exchange(&request, Some(&encode_i32_frame(&ids)))
                .await?;
            match response {
                WorkerResponse::Scores { .. } | WorkerResponse::Vectors { .. } => {
                    let result = decode_f32_frame(&raw.unwrap())
                        .map_err(|e| AneResidencyError::Channel(e.to_string()))?;
                    assert!(!result.is_empty() && result.iter().all(|value| value.is_finite()));
                    Ok(result)
                }
                WorkerResponse::Err { code, msg, .. } => {
                    if code == "shape_not_admitted" {
                        worker.ledger.lock().unwrap().shape_not_admitted += 1;
                    }
                    Err(AneResidencyError::WorkerErr { code, msg })
                }
                other => Err(AneResidencyError::Channel(format!(
                    "unexpected inference response: {other:?}"
                ))),
            }
        }
        fn output(command: &str, args: &[&str]) -> String {
            let result = synapse_core::without_launch_nonce(std::process::Command::new(command))
                .args(args)
                .output()
                .unwrap();
            assert!(result.status.success(), "metadata command {command} failed");
            String::from_utf8(result.stdout).unwrap().trim().to_owned()
        }
        fn load_averages() -> Vec<f64> {
            output("sysctl", &["-n", "vm.loadavg"])
                .split_whitespace()
                .filter_map(|part| part.parse().ok())
                .collect()
        }
        // This small validator supports the assertions used by the committed
        // development schema; newly added assertions must not pass unchecked.
        fn schema_validate(value: &Value, schema: &Value) -> Result<(), String> {
            for key in schema.as_object().ok_or("schema must be an object")?.keys() {
                if ![
                    "$schema",
                    "title",
                    "type",
                    "required",
                    "additionalProperties",
                    "properties",
                    "items",
                    "const",
                    "enum",
                    "minimum",
                    "minLength",
                    "maxLength",
                    "minItems",
                    "maxItems",
                    "uniqueItems",
                ]
                .contains(&key.as_str())
                {
                    return Err(format!("unsupported schema assertion {key}"));
                }
            }
            if let Some(kind) = schema["type"].as_str() {
                let valid = match kind {
                    "object" => value.is_object(),
                    "array" => value.is_array(),
                    "string" => value.is_string(),
                    "integer" => value.is_u64() || value.is_i64(),
                    "number" => value.is_number(),
                    "boolean" => value.is_boolean(),
                    other => return Err(format!("unsupported schema type {other}")),
                };
                if !valid {
                    return Err(format!("expected {kind}"));
                }
            }
            if let Some(expected) = schema.get("const") {
                if value != expected {
                    return Err("const mismatch".into());
                }
            }
            if let Some(choices) = schema["enum"].as_array() {
                if !choices.contains(value) {
                    return Err("enum mismatch".into());
                }
            }
            if let Some(minimum) = schema["minimum"].as_f64() {
                if value.as_f64().unwrap() < minimum {
                    return Err("below minimum".into());
                }
            }
            if let Some(text) = value.as_str() {
                for (key, less) in [("minLength", true), ("maxLength", false)] {
                    if let Some(limit) = schema[key].as_u64() {
                        let n = text.chars().count() as u64;
                        if (less && n < limit) || (!less && n > limit) {
                            return Err(format!("{key} violated"));
                        }
                    }
                }
            }
            if let Some(map) = value.as_object() {
                if let Some(required) = schema["required"].as_array() {
                    for key in required {
                        if !map.contains_key(key.as_str().unwrap()) {
                            return Err(format!("missing {key}"));
                        }
                    }
                }
                for (key, child) in map {
                    if let Some(child_schema) = schema["properties"].get(key) {
                        schema_validate(child, child_schema)
                            .map_err(|error| format!("{key}: {error}"))?;
                    } else if schema["additionalProperties"] == false {
                        return Err(format!("unexpected {key}"));
                    } else if schema["additionalProperties"].is_object() {
                        schema_validate(child, &schema["additionalProperties"])?;
                    }
                }
            }
            if let Some(items) = value.as_array() {
                if let Some(n) = schema["minItems"].as_u64() {
                    if items.len() < n as usize {
                        return Err("too few items".into());
                    }
                }
                if let Some(n) = schema["maxItems"].as_u64() {
                    if items.len() > n as usize {
                        return Err("too many items".into());
                    }
                }
                if schema["uniqueItems"] == true
                    && items
                        .iter()
                        .enumerate()
                        .any(|(i, item)| items[..i].contains(item))
                {
                    return Err("duplicate items".into());
                }
                if let Some(item_schema) = schema.get("items") {
                    for item in items {
                        schema_validate(item, item_schema)?;
                    }
                }
            }
            Ok(())
        }
        fn stress_schema() -> Value {
            serde_json::from_slice(include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../synapse-worker-ane-direct/tests/fixtures/ane-direct-stress.schema.json"
            )))
            .unwrap()
        }
        #[test]
        fn stress_schema_rejects_missing_counts_wrong_types_and_invalid_shapes() {
            let mut valid = json!({"schema":2,"case":"residency_safety","wall_time_ms":1,"admission_wait_bound_ms":2100000,"admission_timeline":[],"kind":"development","harness_profile":"debug","source_commit":"1".repeat(40),"machine":{"model_identifier":"test"},"os":{"version":"test","build":"test"},"load_1_5_15_start":[0,0,0],"load_1_5_15_end":[0,0,0],"request_count":37,"completed_count":37,"request_errors":[],"sample_count":1,"samples":[{"m":[128]}],"max_resident_per_model":1,"max_resident_total":1,"admitted_count":1,"evicted_count":0,"shape_not_admitted_count":0,"leased_evict_count":0,"no_ane_resources":{"count":0},"ane_lane_busy":{"status":"tested"},"metal_ranking_check":{"status":"skipped","reason":"test"},"reranker_pool_check":{"finite":"passed","byte_identical_repeats":"passed"}});
            let schema = stress_schema();
            assert!(
                schema_validate(&valid, &schema).is_ok(),
                "version 2 reports may omit the older optional timing summaries"
            );
            valid["admission_timeline"] = json!([{"event":"queue_entry","at_ms":0,"request_id":"r","model":"m","shape":128,"ticket":0}]);
            valid["compile_duration_ms_by_model"] = json!({"gte-modernbert-base":{"count":1,"min":10,"median":10,"max":10,"samples":[{"shape":128,"elapsed_ms":10,"admitted":true}]}});
            valid["request_wait_times"]=json!((0..37).map(|index|json!({"model":"gte-modernbert-base","request_id":format!("request-{index}"),"admission_wait_ms":12,"waits":[{"shape":128,"elapsed_ms":12,"admitted":true}]})).collect::<Vec<_>>());
            assert!(schema_validate(&valid, &schema).is_ok());
            let mut missing = valid.clone();
            missing.as_object_mut().unwrap().remove("sample_count");
            assert!(schema_validate(&missing, &schema).is_err());
            for (field, bad) in [
                ("load_1_5_15_start", json!(["wrong", 0, 0])),
                ("load_1_5_15_end", json!([0, 0])),
                ("samples", json!([{"m":[17]}])),
                ("samples", json!([{"m":[128,128]}])),
                ("completed_count", json!("37")),
                ("schema", json!(1)),
                ("case", json!("unknown")),
                ("wall_time_ms", json!(-1)),
                ("admission_wait_bound_ms", json!(600000)),
                ("admission_timeline", json!([{"event":"rpc_start"}])),
                (
                    "admission_timeline",
                    json!([{"event":"payload","at_ms":0,"request_id":"r","model":"m","shape":128}]),
                ),
                (
                    "admission_timeline",
                    json!([{"event":"queue_entry","at_ms":-1,"request_id":"r","model":"m","shape":128}]),
                ),
                (
                    "admission_timeline",
                    json!([{"event":"queue_entry","at_ms":0,"request_id":"r","model":"m","shape":17}]),
                ),
                (
                    "admission_timeline",
                    json!([{"event":"queue_entry","at_ms":0,"request_id":"r","model":"m","shape":128,"ticket":-1}]),
                ),
                (
                    "admission_timeline",
                    json!([{"event":"queue_entry","at_ms":0,"request_id":"r","model":"m","shape":128,"payload":"forbidden"}]),
                ),
                ("sample_count", json!(-1)),
                ("source_commit", json!("short")),
                ("request_wait_times", json!([])),
                (
                    "compile_duration_ms_by_model",
                    json!({"m":{"count":1,"min":-1,"median":0,"max":0,"samples":[]}}),
                ),
                (
                    "compile_duration_ms_by_model",
                    json!({"m":{"count":1,"min":1,"median":1,"max":1,"samples":[{"shape":128,"elapsed_ms":1,"admitted":"true"}]}}),
                ),
            ] {
                let mut malformed = valid.clone();
                malformed[field] = bad;
                assert!(
                    schema_validate(&malformed, &schema).is_err(),
                    "{field} must be rejected"
                );
            }
            let mut extra = valid;
            extra["unknown"] = json!(true);
            assert!(schema_validate(&extra, &schema).is_err());
        }
        type RequestWaits = Arc<Mutex<BTreeMap<(String, String), AdmissionWaits>>>;

        fn spawn_stress_request(
            tasks: &mut tokio::task::JoinSet<Result<(), AneResidencyError>>,
            waits: &RequestWaits,
            timeline: &Arc<admission_trace::Timeline>,
            model: String,
            request_id: String,
            request: impl std::future::Future<Output = Result<(), AneResidencyError>> + Send + 'static,
        ) {
            let samples = AdmissionWaits::default();
            let context =
                admission_trace::Context::new(timeline.clone(), format!("{model}/{request_id}"));
            waits
                .lock()
                .unwrap()
                .insert((model, request_id), samples.clone());
            tasks.spawn(admission_trace::scope(
                Some(context),
                ADMISSION_WAITS.scope(samples, request),
            ));
        }

        fn compile_duration_summary(samples: &BTreeMap<String, Vec<Value>>) -> Value {
            let mut summary = serde_json::Map::new();
            for (model, records) in samples {
                let mut elapsed = records
                    .iter()
                    .map(|record| record["elapsed_ms"].as_f64().unwrap())
                    .collect::<Vec<_>>();
                elapsed.sort_by(f64::total_cmp);
                let count = elapsed.len();
                if count > 0 {
                    summary.insert(model.clone(),json!({"count":count,"min":elapsed[0],"median":(elapsed[(count-1)/2]+elapsed[count/2])/2.0,"max":elapsed[count-1],"samples":records}));
                }
            }
            Value::Object(summary)
        }

        fn request_wait_summary(waits: &RequestWaits) -> Vec<Value> {
            waits.lock().unwrap().iter().map(|((model,request_id),samples)|{
                let samples=samples.lock().unwrap();
                json!({"model":model,"request_id":request_id,"admission_wait_ms":samples.iter().map(|sample|sample.elapsed_ms).sum::<f64>(),"waits":*samples})
            }).collect()
        }

        #[test]
        fn compile_duration_summary_reports_independent_even_and_odd_medians() {
            let records = |elapsed: &[f64]| {
                elapsed
                    .iter()
                    .map(|elapsed| json!({"shape":128,"elapsed_ms":elapsed,"admitted":true}))
                    .collect::<Vec<_>>()
            };
            let summary = compile_duration_summary(&BTreeMap::from([
                ("gte".into(), records(&[9.0, 1.0, 7.0, 3.0])),
                ("qwen".into(), records(&[8.0, 4.0, 2.0])),
            ]));
            assert_eq!(summary["gte"]["min"], 1.0);
            assert_eq!(summary["gte"]["median"], 5.0);
            assert_eq!(summary["gte"]["max"], 9.0);
            assert_eq!(summary["gte"]["count"], 4);
            assert_eq!(summary["qwen"]["min"], 2.0);
            assert_eq!(summary["qwen"]["median"], 4.0);
            assert_eq!(summary["qwen"]["max"], 8.0);
            assert_eq!(summary["qwen"]["count"], 3);
        }

        fn write_stress(out: &Path, report: &Value) {
            schema_validate(report, &stress_schema()).unwrap();
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(out, serde_json::to_vec_pretty(report).unwrap()).unwrap();
            println!("ANE_STRESS_OUTPUT={}", out.display());
        }
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        #[ignore = "real four-worker stress; load1<16 and load5<20 required"]
        async fn real_direct_ane_residency_safety() {
            run_direct_ane_stress("residency_safety").await;
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        #[ignore = "cold-throughput characterization; exclusive ANE access required"]
        async fn real_direct_ane_cold_throughput() {
            run_direct_ane_stress("cold_throughput").await;
        }

        /// A real worker whose shape compiles are timed against one epoch, so
        /// the compile-scheduling case can show which compiles overlapped.
        struct TimedWorker {
            channel: RealChannel,
            slug: String,
            epoch: std::time::Instant,
            compiles: Arc<Mutex<Vec<Value>>>,
        }
        impl AneShapeWorker for TimedWorker {
            fn stream_idle(&self) -> bool {
                self.channel.stream_idle()
            }
            fn worker_id(&self) -> &str {
                self.channel.worker_id()
            }
            fn generation(&self) -> u64 {
                self.channel.generation()
            }
            fn admit_shape<'a>(
                &'a self,
                model_ref: &'a str,
                shape: usize,
            ) -> BoxFuture<'a, Result<AnePlacementInventory, AneResidencyError>> {
                self.admit_shape_at_generation(self.generation(), model_ref, shape)
            }
            fn admit_shape_at_generation<'a>(
                &'a self,
                generation: u64,
                model_ref: &'a str,
                shape: usize,
            ) -> BoxFuture<'a, Result<AnePlacementInventory, AneResidencyError>> {
                Box::pin(async move {
                    let start = self.epoch.elapsed().as_secs_f64();
                    let result = self
                        .channel
                        .admit_shape_at_generation(generation, model_ref, shape)
                        .await;
                    let end = self.epoch.elapsed().as_secs_f64();
                    self.compiles.lock().unwrap().push(json!({
                        "slug": self.slug,
                        "shape": shape,
                        "start_s": start,
                        "end_s": end,
                        "admitted": result.is_ok(),
                        "code": result.as_ref().err().and_then(AneResidencyError::code),
                    }));
                    result
                })
            }
            fn evict_shape<'a>(
                &'a self,
                model_ref: &'a str,
                shape: usize,
            ) -> BoxFuture<'a, Result<(), AneResidencyError>> {
                self.channel.evict_shape(model_ref, shape)
            }
            fn evict_shape_at_generation<'a>(
                &'a self,
                generation: u64,
                model_ref: &'a str,
                shape: usize,
            ) -> BoxFuture<'a, Result<(), AneResidencyError>> {
                self.channel
                    .evict_shape_at_generation(generation, model_ref, shape)
            }
            fn restart<'a>(&'a self) -> BoxFuture<'a, Result<(), AneResidencyError>> {
                Box::pin(async move { self.channel.restart().await })
            }
        }

        /// One request of a compile-scheduling arm: model slug, shape, when to
        /// ask (seconds after the arm's epoch) and an optional deadline in
        /// seconds after asking.
        type ScheduledRequest = (&'static str, usize, f64, Option<f64>);

        /// Runs one arm on fresh workers and a fresh supervisor (production
        /// limits and 600 s bound), so every shape compiles cold, as after a
        /// restart. `sequential` asks each request only after the previous one
        /// was answered (standalone timings); otherwise requests are asked at
        /// their own times. Returns every answer and every compile, timed.
        async fn compile_scheduling_arm(
            lock: &AneDirectLaneLock,
            root: &Path,
            name: &str,
            sequential: bool,
            requests: &[ScheduledRequest],
        ) -> Value {
            let factory = Factory::new(root.join(name), lock);
            let supervisor = AneResidencySupervisor::new(AneResidencyLimits::default());
            let compiles = Arc::new(Mutex::new(Vec::new()));
            let mut slugs: Vec<&str> = requests.iter().map(|request| request.0).collect();
            slugs.sort();
            slugs.dedup();
            let mut loaded = Vec::new();
            for slug in &slugs {
                let channel = factory.connect(&format!("{name}-{slug}")).await;
                let model_ref = load(&channel, slug).await;
                loaded.push((*slug, channel, model_ref));
            }
            let epoch = std::time::Instant::now();
            let workers: BTreeMap<&str, (Arc<dyn AneShapeWorker>, String)> = loaded
                .into_iter()
                .map(|(slug, channel, model_ref)| {
                    let worker: Arc<dyn AneShapeWorker> = Arc::new(TimedWorker {
                        channel,
                        slug: slug.into(),
                        epoch,
                        compiles: compiles.clone(),
                    });
                    (slug, (worker, model_ref))
                })
                .collect();
            let ask = |request: ScheduledRequest| {
                let (slug, shape, at, deadline) = request;
                let (worker, model_ref) = workers[slug].clone();
                let supervisor = supervisor.clone();
                async move {
                    if !sequential {
                        tokio::time::sleep_until(
                            tokio::time::Instant::from_std(epoch) + Duration::from_secs_f64(at),
                        )
                        .await;
                    }
                    let asked = epoch.elapsed().as_secs_f64();
                    let request_deadline = deadline.map(|seconds| {
                        tokio::time::Instant::now() + Duration::from_secs_f64(seconds)
                    });
                    let result = supervisor
                        .lease_until(&worker, &model_ref, shape, request_deadline)
                        .await;
                    let answered = epoch.elapsed().as_secs_f64();
                    let (code, retry_after_ms) = match &result {
                        Ok(_) => (None, None),
                        Err(error) => (
                            Some(error.code().unwrap_or("other").to_owned()),
                            error
                                .to_engine_error(synapse_core::EngineErrorStage::Inference)
                                .retry_after_ms,
                        ),
                    };
                    drop(result);
                    json!({
                        "slug": slug,
                        "shape": shape,
                        "deadline_s": deadline,
                        "asked_s": asked,
                        "answered_s": answered,
                        "code": code,
                        "retry_after_ms": retry_after_ms,
                    })
                }
            };
            let mut answers = Vec::new();
            if sequential {
                for request in requests {
                    answers.push(ask(*request).await);
                }
            } else {
                let mut tasks = Vec::new();
                for request in requests {
                    tasks.push(tokio::spawn(ask(*request)));
                }
                for task in tasks {
                    answers.push(task.await.unwrap());
                }
            }
            // A refused request leaves its compile queued for the retry; wait
            // until the queue is empty so that compile is timed too.
            let drained = tokio::time::timeout(Duration::from_secs(900), async {
                loop {
                    let idle = {
                        let state = supervisor.inner.lock();
                        state.entries.is_empty() && !state.compile_running()
                    };
                    if idle {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
            })
            .await
            .is_ok();
            factory.stop().await;
            let compiles = compiles.lock().unwrap().clone();
            json!({
                "arm": name,
                "sequential": sequential,
                "queue_drained": drained,
                "answers": answers,
                "compiles": compiles,
            })
        }

        /// Hardware check of the compile queue, started by the operator on a
        /// Mac with exclusive Neural Engine access (run_stress.py --case
        /// compile-scheduling):
        /// standalone compile times, then a small shape asked together with
        /// and after a wide one, six shapes asked at once, and a request
        /// whose deadline the queue predicts it cannot meet. It writes a JSON
        /// report (ANE_STRESS_OUT) and asserts only what must hold on any
        /// machine: no two compiles overlap and the Neural Engine never ran
        /// out of resources. Latency is judged by the operator's runner, which
        /// compares each arm with the standalone times in the same report.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        #[ignore = "compile-scheduling hardware check; exclusive ANE access required"]
        async fn real_direct_ane_compile_scheduling() {
            const GTE: &str = "gte-modernbert-base";
            const QWEN: &str = "qwen3-embedding-0.6b";
            let load_start = load_averages();
            let root = PathBuf::from(format!(
                "../../target/ane-compile-scheduling-{}",
                std::process::id()
            ));
            std::fs::create_dir_all(&root).unwrap();
            let lock = AneDirectLaneLock::acquire(&root.join("lane.lock")).unwrap();
            let standalone: Vec<ScheduledRequest> = [
                (GTE, 128),
                (GTE, 512),
                (GTE, 2048),
                (QWEN, 128),
                (QWEN, 512),
                (QWEN, 2048),
                (QWEN, 4096),
            ]
            .into_iter()
            .map(|(slug, shape)| (slug, shape, 0.0, None))
            .collect();
            let batch: Vec<ScheduledRequest> = standalone[..6]
                .iter()
                .map(|(slug, shape, _, _)| (*slug, *shape, 0.0, None))
                .collect();
            let arms: [(&str, bool, Vec<ScheduledRequest>); 5] = [
                ("standalone", true, standalone),
                (
                    "together",
                    false,
                    vec![(QWEN, 4096, 0.0, None), (GTE, 128, 0.001, None)],
                ),
                (
                    "late",
                    false,
                    vec![(QWEN, 4096, 0.0, None), (GTE, 128, 10.0, None)],
                ),
                ("batch", false, batch),
                (
                    "refusal",
                    false,
                    vec![(QWEN, 4096, 0.0, None), (GTE, 512, 1.0, Some(30.0))],
                ),
            ];
            let mut results = Vec::new();
            for (name, sequential, requests) in arms {
                println!("COMPILE_SCHEDULING_ARM={name}");
                results
                    .push(compile_scheduling_arm(&lock, &root, name, sequential, &requests).await);
            }
            let report = json!({
                "schema": 1,
                "case": "compile_scheduling",
                "os": {
                    "version": output("sw_vers", &["-productVersion"]),
                    "build": output("sw_vers", &["-buildVersion"]),
                    "model": output("sysctl", &["-n", "hw.model"]),
                },
                "load_start": load_start,
                "load_end": load_averages(),
                "arms": results,
            });
            let out = std::env::var_os("ANE_STRESS_OUT")
                .map(PathBuf::from)
                .unwrap_or_else(|| root.join("compile-scheduling.json"));
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(&out, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
            println!("ANE_STRESS_OUTPUT={}", out.display());
            for arm in report["arms"].as_array().unwrap() {
                let mut spans: Vec<(f64, f64)> = arm["compiles"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|compile| {
                        (
                            compile["start_s"].as_f64().unwrap(),
                            compile["end_s"].as_f64().unwrap(),
                        )
                    })
                    .collect();
                spans.sort_by(|left, right| left.0.total_cmp(&right.0));
                assert!(
                    spans.windows(2).all(|pair| pair[1].0 >= pair[0].1),
                    "{}: two shape compiles overlapped: {spans:?}",
                    arm["arm"]
                );
                let codes: Vec<&Value> = arm["answers"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .chain(arm["compiles"].as_array().unwrap())
                    .map(|entry| &entry["code"])
                    .collect();
                assert!(
                    !codes
                        .iter()
                        .any(|code| code.as_str() == Some(ERR_ANE_RESOURCES_EXHAUSTED)),
                    "{}: the Neural Engine ran out of resources",
                    arm["arm"]
                );
            }
        }

        async fn run_direct_ane_stress(case: &str) {
            let timeline = admission_trace::Timeline::new();
            let load_start = load_averages();
            assert_eq!(load_start.len(), 3);
            assert!(
                load_start[0] < 16.0 && load_start[1] < 20.0,
                "stress deferred: load={load_start:?}"
            );
            let root = PathBuf::from(format!("../../target/ane-stress-{}", std::process::id()));
            std::fs::create_dir_all(&root).unwrap();
            let lock = AneDirectLaneLock::acquire(&root.join("lane.lock")).unwrap();
            let factory = Factory::new(root.clone(), &lock);
            let mut supervisor =
                AneResidencySupervisor::with_lane_lock(AneResidencyLimits::default(), lock);
            // QE's five measured RPCs took 492.198 s and ended at 528.815 s;
            // GE then needed 118.356 s, reaching 647.171 s before ten unmeasured
            // keys. Allow 3 * (528.815 + 118.356) = 1941.513 s, rounded up to
            // 2100 s for safety coverage, not a throughput promise. Both cases
            // use this test-only backstop; production remains 600 s.
            Arc::get_mut(&mut supervisor.inner).unwrap().wait_timeout = Duration::from_secs(2100);
            // Serialize shape compiles only in this test to isolate concurrency effects from residency limits.
            if std::env::var_os("ANE_STRESS_SERIALIZE_ADMISSION").is_some() {
                Arc::get_mut(&mut supervisor.inner).unwrap().admission_gate =
                    Some(Arc::new(tokio::sync::Mutex::new(())));
            }
            let ledger = Arc::new(Mutex::new(Ledger::default()));
            let mut workers = Vec::new();
            for (index, slug) in SLUGS.iter().enumerate() {
                let channel = factory.connect(&format!("w{index}")).await;
                let model_ref = load(&channel, slug).await;
                workers.push(Arc::new(Observed {
                    channel,
                    supervisor: Arc::downgrade(&supervisor.inner),
                    ledger: ledger.clone(),
                    slug: (*slug).into(),
                    model_ref,
                }));
            }
            let barrier = Arc::new(tokio::sync::Barrier::new(38));
            let mut tasks = tokio::task::JoinSet::new();
            let request_waits = RequestWaits::default();
            for worker in &workers {
                for shape in [128, 256, 512, 1024, 2048, 4096, 8192] {
                    let worker = worker.clone();
                    let supervisor = supervisor.clone();
                    let barrier = barrier.clone();
                    spawn_stress_request(
                        &mut tasks,
                        &request_waits,
                        &timeline,
                        worker.slug.clone(),
                        format!("admit-{shape}"),
                        async move {
                            barrier.wait().await;
                            let channel: Arc<dyn AneShapeWorker> = worker.clone();
                            let lease =
                                supervisor.lease(&channel, &worker.model_ref, shape).await?;
                            assert!(!lease.inventory().unwrap().executables.is_empty());
                            drop(lease);
                            Ok::<_, AneResidencyError>(())
                        },
                    );
                }
            }
            for index in 0..8 {
                let worker = workers[0].clone();
                let supervisor = supervisor.clone();
                let barrier = barrier.clone();
                spawn_stress_request(
                    &mut tasks,
                    &request_waits,
                    &timeline,
                    worker.slug.clone(),
                    format!("embed-{index}"),
                    async move {
                        barrier.wait().await;
                        let ids = [
                            fixture_ids(SLUGS[0], "boundary-128"),
                            fixture_ids(SLUGS[0], "boundary-129"),
                        ];
                        let channel: Arc<dyn AneShapeWorker> = worker.clone();
                        supervisor
                            .run_by_rung(&channel, &worker.model_ref, &[128, 129], |_, indices| {
                                let worker = worker.clone();
                                let input = ids[indices[0]].clone();
                                async move { infer(&worker, input, false).await.map(|_| vec![()]) }
                            })
                            .await
                            .map(|_| ())
                    },
                );
            }
            let pool_results: Arc<Mutex<Vec<(usize, f32)>>> = Arc::new(Mutex::new(Vec::new()));
            let pool_result_task = pool_results.clone();
            let worker = workers[1].clone();
            let supervisor_pool = supervisor.clone();
            let barrier_pool = barrier.clone();
            spawn_stress_request(
                &mut tasks,
                &request_waits,
                &timeline,
                worker.slug.clone(),
                "reranker-pool".into(),
                async move {
                    barrier_pool.wait().await;
                    let mut long = fixture_ids(SLUGS[1], "long-8192");
                    let eos = *long.last().unwrap();
                    long.truncate(1535);
                    long.push(eos);
                    let ids = [
                        fixture_ids(SLUGS[1], "short-0"),
                        fixture_ids(SLUGS[1], "boundary-511"),
                        long,
                    ];
                    let channel: Arc<dyn AneShapeWorker> = worker.clone();
                    supervisor_pool
                        .run_by_rung(
                            &channel,
                            &worker.model_ref,
                            &[ids[0].len(), ids[1].len(), ids[2].len()],
                            |shape, indices| {
                                let worker = worker.clone();
                                let input = ids[indices[0]].clone();
                                let pool_results = pool_result_task.clone();
                                async move {
                                    let first = infer(&worker, input.clone(), true).await?;
                                    let second = infer(&worker, input, true).await?;
                                    assert_eq!(
                                        first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                        second.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                                    );
                                    assert_eq!(first.len(), 1);
                                    pool_results.lock().unwrap().push((shape, first[0]));
                                    Ok::<_, AneResidencyError>(vec![()])
                                }
                            },
                        )
                        .await
                        .map(|_| ())
                },
            );
            assert_eq!(tasks.len(), 37);
            barrier.wait().await;
            let mut completed = 0;
            let mut errors = Vec::new();
            while let Some(result) = tasks.join_next().await {
                match result {
                    Ok(Ok(())) => completed += 1,
                    other => errors.push(format!("{other:?}")),
                }
            }
            let stats = supervisor.stats();
            let samples = supervisor.inner.lock().recorded_samples.clone();
            let mut pool_results = pool_results.lock().unwrap().clone();
            pool_results.sort_by_key(|entry| entry.0);
            let pool_passed =
                pool_results.iter().map(|entry| entry.0).collect::<Vec<_>>() == [128, 512, 2048];
            let clean = {
                let ledger = ledger.lock().unwrap();

                let report = json!({"schema":2, "case":case, "wall_time_ms":timeline.elapsed_ms(), "admission_wait_bound_ms":2100000, "admission_timeline":timeline.snapshot(), "kind":"development", "harness_profile":if cfg!(debug_assertions) {"debug"} else {"release"}, "source_commit":output("git", &["rev-parse", "HEAD"]), "machine":{"model_identifier":output("sysctl", &["-n", "hw.model"])}, "os":{"version":output("sw_vers", &["-productVersion"]), "build":output("sw_vers", &["-buildVersion"])}, "load_1_5_15_start":load_start, "load_1_5_15_end":load_averages(), "request_count":37, "completed_count":completed, "request_errors":errors, "sample_count":samples.len(), "samples":samples, "max_resident_per_model":stats.max_resident_per_model, "max_resident_total":stats.max_resident_total, "admitted_count":ledger.admitted, "evicted_count":ledger.evicted, "compile_duration_ms_by_model":compile_duration_summary(&ledger.compile_samples),"request_wait_times":request_wait_summary(&request_waits), "shape_not_admitted_count":ledger.shape_not_admitted, "leased_evict_count":ledger.leased_evicts, "no_ane_resources":{"count":ledger.no_ane_resources}, "ane_lane_busy":{"status":"covered_by_killed_holder_test"}, "metal_ranking_check":{"status":"skipped", "reason":"Metal rerank package/lane unavailable in isolated worktree; missing2048-rung parity fixture is a parity-crate follow-up"}, "reranker_pool_check":{"finite":if pool_passed {"passed"} else {"failed"}, "byte_identical_repeats":if pool_passed {"passed"} else {"failed"}, "scores_by_rung":pool_results.iter().map(|(shape, score)|json!({"shape":shape, "score":score})).collect::<Vec<_>>()}});
                let out = std::env::var_os("ANE_STRESS_OUT")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| {
                        std::env::temp_dir()
                            .join(format!("ane-direct-stress-{}.json", std::process::id()))
                    });
                write_stress(&out, &report);
                pool_passed
                    && completed == 37
                    && errors.is_empty()
                    && stats.samples as usize == ledger.admitted + ledger.evicted
                    && ledger.leased_evicts == 0
                    && ledger.shape_not_admitted == 0
                    && ledger.no_ane_resources == 0
                    && stats.max_resident_per_model <= 4
                    && stats.max_resident_total <= 8
                    && samples.iter().all(|sample| {
                        sample
                            .iter()
                            .map(|(model, shapes)| {
                                let worker = workers
                                    .iter()
                                    .find(|worker| worker.model_ref == *model)
                                    .unwrap();
                                shapes.len() * worker.executable_count(model).unwrap()
                            })
                            .sum::<usize>()
                            <= ANE_RESIDENT_EXECUTABLES_TOTAL
                    })
            };
            drop(workers);
            drop(supervisor);
            factory.stop().await;
            std::fs::remove_dir_all(root).unwrap();
            if case == "residency_safety" {
                assert!(clean, "stress failed; see ANE_STRESS_OUTPUT above");
            }
        }
        struct StoppedWorker(Option<u32>);
        impl Drop for StoppedWorker {
            fn drop(&mut self) {
                if let Some(pid) = self.0 {
                    let _ =
                        synapse_core::without_launch_nonce(std::process::Command::new("/bin/kill"))
                            .args(["-9".to_owned(), pid.to_string()])
                            .status();
                }
            }
        }
        #[tokio::test]
        #[ignore = "subprocess entry for the real killed-holder test"]
        async fn real_lane_holder_process() {
            let root = PathBuf::from(std::env::var_os("ANE_HOLDER_ROOT").expect("ANE_HOLDER_ROOT"));
            let lock = AneDirectLaneLock::acquire(&root.join("lane.lock")).unwrap();
            let factory = Factory::new(root.clone(), &lock);
            let supervisor =
                AneResidencySupervisor::with_lane_lock(AneResidencyLimits::default(), lock);
            let channel = factory.connect("holder").await;
            let model_ref = load(&channel, SLUGS[0]).await;
            let worker: Arc<dyn AneShapeWorker> = channel.clone();
            let _lease = supervisor.lease(&worker, &model_ref, 128).await.unwrap();
            let ids = fixture_ids(SLUGS[0], "boundary-128");
            let items: Vec<_> = (0..256)
                .map(|index| WorkerTokenItem {
                    id: index.to_string(),
                    n_tokens: ids.len(),
                })
                .collect();
            let raw = encode_i32_frame(&ids.repeat(256));
            let request = WorkerRequest::EmbedBatch {
                req_id: "in-flight".into(),
                model_ref,
                pooling: WorkerPooling::Cls,
                normalize: true,
                items,
            };
            let mut guard = channel.stream.lock().await;
            let stream = &mut guard.as_mut().unwrap().stream;
            write_json(stream, &request, DEFAULT_MAX_FRAME_BYTES)
                .await
                .unwrap();
            write_raw(stream, &raw, DEFAULT_MAX_FRAME_BYTES)
                .await
                .unwrap();
            std::fs::write(root.join("ready.tmp"), serde_json::to_vec(&json!({"pids":factory.pids(), "resident":supervisor.resident_shapes(), "in_flight_sequences":256})).unwrap()).unwrap();
            std::fs::rename(root.join("ready.tmp"), root.join("ready.json")).unwrap();
            // The parent suspends the worker before killing this holder. The
            // blocked response keeps the real request and shape lease in flight.
            let _: WorkerResponse = read_json(stream, DEFAULT_MAX_FRAME_BYTES).await.unwrap();
            panic!("holder completed before parent killed it");
        }
        #[tokio::test]
        #[ignore = "requires the real worker, pinned weights and macOS ANE"]
        async fn killed_holder_with_resident_work_keeps_lane_busy_until_old_worker_exits() {
            let root = PathBuf::from(format!("../../target/ane-holder-{}", std::process::id()));
            std::fs::create_dir_all(&root).unwrap();
            let mut holder = synapse_core::without_launch_nonce_tokio(
                tokio::process::Command::new(std::env::current_exe().unwrap()),
            )
            .args([
                "--ignored",
                "--exact",
                "worker_host::ane_residency::hardware_tests::real_lane_holder_process",
                "--nocapture",
            ])
            .env("ANE_HOLDER_ROOT", &root)
            .kill_on_drop(true)
            .spawn()
            .unwrap();
            let ready = tokio::time::timeout(Duration::from_secs(120), async {
                loop {
                    if let Ok(bytes) = std::fs::read(root.join("ready.json")) {
                        break serde_json::from_slice::<Value>(&bytes).unwrap();
                    }
                    if let Some(status) = holder.try_wait().unwrap() {
                        panic!("holder exited before readiness: {status}");
                    }
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            })
            .await
            .unwrap();
            assert!(ready["resident"]
                .as_object()
                .unwrap()
                .values()
                .any(|shapes| shapes == &json!([128])));
            assert_eq!(ready["in_flight_sequences"], 256);
            let pid = ready["pids"][0].as_u64().unwrap() as u32;
            signal(pid, 17);
            let mut stopped = StoppedWorker(Some(pid));
            holder.kill().await.unwrap();
            holder.wait().await.unwrap();
            assert_eq!(
                AneDirectLaneLock::acquire(&root.join("lane.lock"))
                    .err()
                    .unwrap()
                    .code(),
                Some(ERR_ANE_LANE_BUSY)
            );
            signal(pid, 19);
            let replacement = tokio::time::timeout(Duration::from_secs(15), async {
                loop {
                    match AneDirectLaneLock::acquire(&root.join("lane.lock")) {
                        Ok(lock) => break lock,
                        Err(error) => assert_eq!(error.code(), Some(ERR_ANE_LANE_BUSY)),
                    }
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            })
            .await
            .unwrap();
            // Acquiring the inherited flock proves the old worker closed its
            // descriptor after observing supervisor EOF, not merely holder death.
            println!("KILLED_HOLDER old_worker={pid} resident_shapes=1 in_flight_sequences=256 replacement=ane_lane_busy_then_acquired");
            stopped.0 = None;
            drop(replacement);
            std::fs::remove_dir_all(root).unwrap();
        }
    }

    #[cfg(all(test, unix))]
    pub(crate) use tests::{
        module_mock_engine, module_mock_engine_admission, module_mock_engine_controlled,
        module_mock_engine_resident, ModuleMockControl,
    };

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::collections::{HashMap, HashSet};
        use std::time::Duration;
        use synapse_core::{
            decode_f32_frame, encode_f32_frame, encode_i32_frame, AneExecutable, WorkerPooling,
            WorkerTokenItem, DEFAULT_MAX_FRAME_BYTES, ERR_SHAPE_NOT_ADMITTED,
        };
        use tokio::io::DuplexStream;

        const LAYERS: u32 = 4;
        #[test]
        #[cfg(unix)]
        fn module_tests_reject_the_production_lock_path_without_opening_it() {
            let production = AneDirectLaneLock::default_path().expect("test HOME is set");
            assert!(
                std::panic::catch_unwind(|| assert_test_ane_lock_path(&production)).is_err(),
                "production lock guard must reject the machine-wide path before any file open"
            );
        }

        struct MockChannel(AneWorkerChannel<DuplexStream>);
        impl std::ops::Deref for MockChannel {
            type Target = AneWorkerChannel<DuplexStream>;
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }
        impl AneShapeWorker for MockChannel {
            fn stream_idle(&self) -> bool {
                self.0.stream_idle()
            }
            fn worker_id(&self) -> &str {
                self.0.worker_id()
            }
            fn generation(&self) -> u64 {
                self.0.generation()
            }
            fn exit_unconfirmed(&self) -> bool {
                self.0.exit_unconfirmed.load(Ordering::Acquire)
            }
            fn admit_shape_at_generation<'a>(
                &'a self,
                generation: u64,
                model_ref: &'a str,
                shape: usize,
            ) -> BoxFuture<'a, Result<AnePlacementInventory, AneResidencyError>> {
                self.0
                    .admit_shape_at_generation(generation, model_ref, shape)
            }
            fn evict_shape_at_generation<'a>(
                &'a self,
                generation: u64,
                model_ref: &'a str,
                shape: usize,
            ) -> BoxFuture<'a, Result<(), AneResidencyError>> {
                self.0
                    .evict_shape_at_generation(generation, model_ref, shape)
            }
            fn executable_count(&self, _: &str) -> Result<usize, AneResidencyError> {
                Ok(LAYERS as usize)
            }
            /// Mock compiles have no modelled cost: they queue in arrival
            /// order and are never refused early, as before the compile queue.
            fn compile_cost_seed(&self, _: &str, _: usize, _: usize) -> Duration {
                Duration::ZERO
            }
            fn admit_shape<'a>(
                &'a self,
                model_ref: &'a str,
                shape: usize,
            ) -> BoxFuture<'a, Result<AnePlacementInventory, AneResidencyError>> {
                self.0.admit_shape(model_ref, shape)
            }
            fn evict_shape<'a>(
                &'a self,
                model_ref: &'a str,
                shape: usize,
            ) -> BoxFuture<'a, Result<(), AneResidencyError>> {
                self.0.evict_shape(model_ref, shape)
            }
            fn restart(&self) -> BoxFuture<'_, Result<(), AneResidencyError>> {
                self.0.restart()
            }
            fn restart_with_exit_signal<'a>(
                &'a self,
                exited: tokio::sync::oneshot::Sender<()>,
            ) -> BoxFuture<'a, Result<(), AneResidencyError>> {
                self.0.restart_with_exit_signal(exited)
            }
        }
        type Channel = Arc<MockChannel>;

        /// What the mock workers observe, kept independently of the
        /// supervisor's own accounting so the budget checks can fail.
        #[derive(Default)]
        struct Ledger {
            loads: Vec<WorkerRequest>,
            // Read only by the Unix-only pause tests.
            #[cfg(unix)]
            confirmed_exits: u32,
            inference_gate: Option<(Arc<Notify>, Arc<Notify>)>,
            admit_gate: Option<(Arc<Notify>, Arc<Notify>)>,
            fail_inference: bool,
            invalid_inference: Option<InvalidInference>,
            evict_gate: Option<(Arc<Notify>, Arc<Notify>)>,
            /// Requests currently running on (model_ref, shape); a request
            /// counts itself here only while it holds that shape's lease.
            in_flight: HashMap<(String, usize), u32>,
            /// Shapes compiled in some live mock worker.
            resident: HashSet<(String, usize)>,
            max_per_model: usize,
            max_total: usize,
            /// `admit <model> <shape>` / `evict <model> <shape>`, in arrival order.
            events: Vec<String>,
            leased_evicts: u64,
            shape_not_admitted: u64,
            /// Admissions the mock answers with `ERR`, once each.
            fail_admit: HashSet<(String, usize)>,
            fail_evict: HashSet<(String, usize)>,
            fail_confirm_exit: bool,
            exit_gate: Option<(Arc<Notify>, Arc<Notify>)>,
            no_reply_admit: HashSet<(String, usize)>,
            no_reply_evict: HashSet<(String, usize)>,
            exhaust_admit: HashMap<(String, usize), usize>,
            admit_attempts: HashMap<(String, usize), usize>,
            admit_delay: Duration,
            connects: HashMap<String, u32>,
            /// Inference requests (embed or rerank) the mock worker received.
            inference_frames: u64,
            /// Their request ids, in arrival order.
            inference_ids: Vec<String>,
            capture_token_ids: bool,
            captured_token_ids: Vec<Vec<u32>>,
        }

        type SharedLedger = Arc<Mutex<Ledger>>;

        #[derive(Clone, Copy)]
        enum InvalidInference {
            Nonfinite,
            Dimensions,
        }

        fn record_token_ids(
            ledger: &SharedLedger,
            lengths: impl Iterator<Item = usize>,
            raw: &[u8],
        ) {
            let mut ledger = ledger.lock().unwrap();
            if !ledger.capture_token_ids {
                return;
            }
            let ids = synapse_core::decode_i32_frame(raw).unwrap();
            let mut offset = 0;
            for length in lengths {
                ledger.captured_token_ids.push(
                    ids[offset..offset + length]
                        .iter()
                        .map(|id| u32::try_from(*id).unwrap())
                        .collect(),
                );
                offset += length;
            }
            assert_eq!(offset, ids.len());
        }

        fn record_residency(ledger: &mut Ledger) {
            let mut per_model: HashMap<&str, usize> = HashMap::new();
            for (model, _) in &ledger.resident {
                *per_model.entry(model.as_str()).or_default() += 1;
            }
            let largest = per_model.values().copied().max().unwrap_or(0);
            ledger.max_per_model = ledger.max_per_model.max(largest);
            ledger.max_total = ledger.max_total.max(ledger.resident.len());
        }

        /// One mock direct-ANE worker process: compiles shapes on admit,
        /// refuses inference on a shape it has not admitted, and loses every
        /// shape when its connection closes.
        async fn serve_mock<S: AsyncRead + AsyncWrite + Unpin>(
            mut stream: S,
            ledger: SharedLedger,
        ) {
            let max = DEFAULT_MAX_FRAME_BYTES;
            let mut own: HashSet<(String, usize)> = HashSet::new();
            while let Ok(request) = read_json::<WorkerRequest, _>(&mut stream, max).await {
                let raw = if request.carries_raw_frame() {
                    match read_raw(&mut stream, max).await {
                        Ok(raw) => Some(raw),
                        Err(_) => break,
                    }
                } else {
                    None
                };
                if matches!(
                    request,
                    WorkerRequest::EmbedBatch { .. } | WorkerRequest::RerankSequences { .. }
                ) {
                    {
                        let mut ledger = ledger.lock().unwrap();
                        ledger.inference_frames += 1;
                        if let Some(id) = request.req_id() {
                            ledger.inference_ids.push(id.to_owned());
                        }
                    }
                    let gate = ledger.lock().unwrap().inference_gate.take();
                    if let Some((started, release)) = gate {
                        started.notify_one();
                        release.notified().await;
                    }
                    if std::mem::take(&mut ledger.lock().unwrap().fail_inference) {
                        break;
                    }
                }
                let (response, raw_out) = match &request {
                    WorkerRequest::Load {
                        req_id,
                        artifact_digest,
                        runtime_config,
                        ..
                    } => {
                        ledger.lock().unwrap().loads.push(request.clone());
                        let direct_profile = runtime_config
                            .get("profile")
                            .filter(|id| id.ends_with(".ane-direct-worker"));
                        (
                            WorkerResponse::Loaded {
                                req_id: req_id.clone(),
                                model_ref: direct_profile
                                    .map(|profile| {
                                        format!("ane-direct:{profile}:{artifact_digest}")
                                    })
                                    .unwrap_or_else(|| format!("mock:{artifact_digest}")),
                                dims: if direct_profile.is_some() { 1 } else { 4 },
                                cold_load_ms: 0,
                                buckets: direct_profile.map(|_| ANE_SHAPE_LADDER.to_vec()),
                            },
                            None,
                        )
                    }
                    WorkerRequest::AneAdmitShape {
                        req_id,
                        model_ref,
                        shape,
                    } => {
                        let key = (model_ref.clone(), *shape);
                        let unacknowledged = ledger.lock().unwrap().no_reply_admit.remove(&key);
                        if unacknowledged {
                            own.insert(key.clone());
                            ledger.lock().unwrap().resident.insert(key);
                            continue;
                        }
                        let gate = ledger.lock().unwrap().admit_gate.take();
                        if let Some((started, release)) = gate {
                            started.notify_one();
                            release.notified().await;
                        }
                        let delay = ledger.lock().unwrap().admit_delay;
                        tokio::time::sleep(delay).await;
                        let key = (model_ref.clone(), *shape);
                        let mut ledger = ledger.lock().unwrap();
                        *ledger.admit_attempts.entry(key.clone()).or_default() += 1;
                        let exhausted =
                            ledger.exhaust_admit.get_mut(&key).is_some_and(|remaining| {
                                if *remaining > 0 {
                                    *remaining -= 1;
                                    true
                                } else {
                                    false
                                }
                            });
                        if exhausted {
                            ledger.events.push(format!("exhaust {model_ref} {shape}"));
                            (
                                WorkerResponse::Err {
                                    req_id: Some(req_id.clone()),
                                    code: ERR_ANE_RESOURCES_EXHAUSTED.into(),
                                    msg: "injected ANE resource exhaustion".into(),
                                },
                                None,
                            )
                        } else if ledger.fail_admit.remove(&key) {
                            (
                                WorkerResponse::Err {
                                    req_id: Some(req_id.clone()),
                                    code: "compile_failed".to_string(),
                                    msg: "injected admission failure".to_string(),
                                },
                                None,
                            )
                        } else {
                            own.insert(key.clone());
                            ledger.resident.insert(key);
                            ledger.events.push(format!("admit {model_ref} {shape}"));
                            record_residency(&mut ledger);
                            let inventory = AnePlacementInventory {
                                executables: vec![AneExecutable {
                                    id: format!("{model_ref}-{shape}"),
                                    layers: (0..LAYERS).collect(),
                                }],
                                cpu_stages: vec!["token_embedding".to_string()],
                            };
                            (
                                WorkerResponse::Admitted {
                                    req_id: req_id.clone(),
                                    inventory,
                                },
                                None,
                            )
                        }
                    }
                    WorkerRequest::AneEvictShape {
                        req_id,
                        model_ref,
                        shape,
                    } => {
                        let unacknowledged = ledger
                            .lock()
                            .unwrap()
                            .no_reply_evict
                            .remove(&(model_ref.clone(), *shape));
                        if unacknowledged {
                            continue;
                        }
                        let failed = ledger
                            .lock()
                            .unwrap()
                            .fail_evict
                            .remove(&(model_ref.clone(), *shape));
                        if failed {
                            write_json(
                                &mut stream,
                                &WorkerResponse::Err {
                                    req_id: Some(req_id.clone()),
                                    code: "evict_failed".into(),
                                    msg: "injected eviction failure".into(),
                                },
                                max,
                            )
                            .await
                            .unwrap();
                            continue;
                        }
                        let gate = ledger.lock().unwrap().evict_gate.clone();
                        if let Some((started, release)) = gate {
                            started.notify_one();
                            release.notified().await;
                        }
                        let key = (model_ref.clone(), *shape);
                        let mut ledger = ledger.lock().unwrap();
                        if ledger.in_flight.get(&key).copied().unwrap_or(0) > 0 {
                            ledger.leased_evicts += 1;
                        }
                        own.remove(&key);
                        ledger.resident.remove(&key);
                        ledger.events.push(format!("evict {model_ref} {shape}"));
                        record_residency(&mut ledger);
                        (
                            WorkerResponse::Evicted {
                                req_id: req_id.clone(),
                            },
                            None,
                        )
                    }
                    WorkerRequest::EmbedBatch {
                        req_id,
                        model_ref,
                        items,
                        ..
                    } => {
                        let rungs: Vec<usize> = items
                            .iter()
                            .map(|item| ladder_rung(item.n_tokens).unwrap())
                            .collect();
                        let total: usize = items.iter().map(|item| item.n_tokens).sum();
                        assert_eq!(
                            raw.as_ref().map(Vec::len),
                            Some(total * 4),
                            "the id frame holds every item's tokens"
                        );
                        record_token_ids(
                            &ledger,
                            items.iter().map(|item| item.n_tokens),
                            raw.as_ref().unwrap(),
                        );
                        if rungs
                            .iter()
                            .all(|rung| own.contains(&(model_ref.clone(), *rung)))
                        {
                            let invalid = ledger.lock().unwrap().invalid_inference.take();
                            let dims = if matches!(invalid, Some(InvalidInference::Dimensions)) {
                                2
                            } else {
                                1
                            };
                            let values: Vec<f32> = rungs
                                .iter()
                                .flat_map(|rung| {
                                    vec![
                                        if matches!(invalid, Some(InvalidInference::Nonfinite)) {
                                            f32::NAN
                                        } else {
                                            *rung as f32
                                        };
                                        dims
                                    ]
                                })
                                .collect();
                            (
                                WorkerResponse::Vectors {
                                    req_id: req_id.clone(),
                                    dims,
                                    n: items.len(),
                                },
                                Some(encode_f32_frame(&values)),
                            )
                        } else {
                            ledger.lock().unwrap().shape_not_admitted += 1;
                            (
                                WorkerResponse::Err {
                                    req_id: Some(req_id.clone()),
                                    code: ERR_SHAPE_NOT_ADMITTED.to_string(),
                                    msg: format!("{model_ref} rungs {rungs:?} are not resident"),
                                },
                                None,
                            )
                        }
                    }
                    WorkerRequest::RerankSequences {
                        req_id,
                        model_ref,
                        sequences,
                    } => {
                        let rungs: Vec<_> = sequences
                            .iter()
                            .map(|s| ladder_rung(s.n_tokens).unwrap())
                            .collect();
                        assert_eq!(
                            raw.as_ref().map(Vec::len),
                            Some(sequences.iter().map(|s| s.n_tokens).sum::<usize>() * 4)
                        );
                        record_token_ids(
                            &ledger,
                            sequences.iter().map(|item| item.n_tokens),
                            raw.as_ref().unwrap(),
                        );
                        if rungs
                            .iter()
                            .all(|rung| own.contains(&(model_ref.clone(), *rung)))
                        {
                            (
                                WorkerResponse::Scores {
                                    req_id: req_id.clone(),
                                },
                                Some(encode_f32_frame(
                                    &rungs.iter().map(|rung| *rung as f32).collect::<Vec<_>>(),
                                )),
                            )
                        } else {
                            ledger.lock().unwrap().shape_not_admitted += 1;
                            (
                                WorkerResponse::Err {
                                    req_id: Some(req_id.clone()),
                                    code: ERR_SHAPE_NOT_ADMITTED.into(),
                                    msg: "rerank shape not admitted".into(),
                                },
                                None,
                            )
                        }
                    }
                    other => (WorkerResponse::unsupported_request(other), None),
                };
                if write_json(&mut stream, &response, max).await.is_err() {
                    break;
                }
                if let Some(raw_out) = raw_out {
                    if write_raw(&mut stream, &raw_out, max).await.is_err() {
                        break;
                    }
                }
            }
            let mut ledger = ledger.lock().unwrap();
            for key in own {
                ledger.resident.remove(&key);
            }
        }

        // This v3 peer accepts ordinary requests but has no packed request tags.
        // Short calls can finish without progress frames; longer calls still
        // use the v3 completed-work watchdog. Unknown tags fail deserialization,
        // so accidentally sending packed work closes the actual framed peer.
        #[derive(serde::Deserialize)]
        #[serde(tag = "type", rename_all = "SCREAMING_SNAKE_CASE")]
        enum V3WithoutPackedRequest {
            Load {},
            EmbedBatch {},
            Rerank {},
            RerankSequences {},
            AneAdmitShape {},
            AneEvictShape {},
            Generate {},
            Unload {},
            Ping {},
            Shutdown {},
        }

        #[derive(Default)]
        struct PackedPeer {
            requests: Vec<WorkerRequest>,
            crashed: bool,
            refuse_admission: bool,
            refuse_execution: bool,
            corrupt: bool,
            disconnect_qualification: bool,
            admission_gate: Option<(Arc<Notify>, Arc<Notify>)>,
            eviction_gate: Option<(Arc<Notify>, Arc<Notify>)>,
            inference_gates: VecDeque<(Arc<Notify>, Arc<Notify>)>,
        }

        const PACKED_MODEL: &str = "ane-direct:qwen3-embedding-0.6b.ane-direct-worker:pinned";
        const V3_WITHOUT_PACKED_HELLO: &[u8] = br#"{"v":3,"nonce":"frozen","engine":{"engine":"ane-direct-worker","version":"0.1.0","build_flags":{"kernel_revision":"ane-direct-graph-v1","precision":"f16"}},"pid":42,"max_frame":16777216,"manifest_digest":"frozen-manifest","kernel_revision":"ane-direct-graph-v1"}"#;

        async fn packed_peer(
            legacy: bool,
            peer: Arc<Mutex<PackedPeer>>,
        ) -> Arc<AneServing<DuplexStream>> {
            let connector: AneWorkerConnector<DuplexStream> = Box::new(move || {
                let peer = peer.clone();
                Box::pin(async move {
                    let (mut host, mut worker) = tokio::io::duplex(1 << 16);
                    let task = tokio::spawn(async move {
                        let max = DEFAULT_MAX_FRAME_BYTES;
                        let mut hello: synapse_core::WorkerHello =
                            serde_json::from_slice(V3_WITHOUT_PACKED_HELLO).unwrap();
                        assert_eq!(serde_json::to_vec(&hello).unwrap(), V3_WITHOUT_PACKED_HELLO);
                        if !legacy {
                            hello.packed = Some(synapse_core::PackedCapability::tight256());
                        }
                        write_json(&mut worker, &hello, max).await.unwrap();
                        let _: synapse_core::WorkerHelloAck =
                            read_json(&mut worker, max).await.unwrap();
                        let mut resident = BTreeSet::new();
                        while let Ok(bytes) = read_raw(&mut worker, max).await {
                            if legacy
                                && serde_json::from_slice::<V3WithoutPackedRequest>(&bytes).is_err()
                            {
                                peer.lock().unwrap().crashed = true;
                                break;
                            }
                            let request: WorkerRequest = serde_json::from_slice(&bytes).unwrap();
                            let raw = if request.carries_raw_frame() {
                                Some(read_raw(&mut worker, max).await.unwrap())
                            } else {
                                None
                            };
                            peer.lock().unwrap().requests.push(request.clone());
                            let program = AneProgram::Packed {
                                program_revision: synapse_core::PACKED_PROGRAM_REVISION,
                            };
                            let mut output = None;
                            let response = match &request {
                                WorkerRequest::AneAdmitShape { req_id, shape, .. } => {
                                    resident.insert((AneProgram::Single, *shape));
                                    WorkerResponse::Admitted {
                                        req_id: req_id.clone(),
                                        inventory: packed_inventory("single"),
                                    }
                                }
                                WorkerRequest::AneAdmitPacked { req_id, .. } => {
                                    let gate = peer.lock().unwrap().admission_gate.take();
                                    if let Some((started, release)) = gate {
                                        started.notify_one();
                                        release.notified().await;
                                    }
                                    if peer.lock().unwrap().refuse_admission {
                                        WorkerResponse::unsupported_request(&request)
                                    } else {
                                        resident.insert((program, 256));
                                        WorkerResponse::Admitted {
                                            req_id: req_id.clone(),
                                            inventory: packed_inventory("packed"),
                                        }
                                    }
                                }
                                WorkerRequest::AneEvictShape { req_id, shape, .. } => {
                                    resident.remove(&(AneProgram::Single, *shape));
                                    WorkerResponse::Evicted {
                                        req_id: req_id.clone(),
                                    }
                                }
                                WorkerRequest::AneEvictPacked { req_id, .. } => {
                                    let gate = peer.lock().unwrap().eviction_gate.take();
                                    if let Some((started, release)) = gate {
                                        started.notify_one();
                                        release.notified().await;
                                    }
                                    resident.remove(&(program, 256));
                                    WorkerResponse::Evicted {
                                        req_id: req_id.clone(),
                                    }
                                }
                                WorkerRequest::EmbedBatch { req_id, items, .. }
                                | WorkerRequest::EmbedPacked { req_id, items, .. } => {
                                    let gate = peer.lock().unwrap().inference_gates.pop_front();
                                    if let Some((started, release)) = gate {
                                        started.notify_one();
                                        release.notified().await;
                                    }
                                    let packed =
                                        matches!(request, WorkerRequest::EmbedPacked { .. });
                                    if packed
                                        && std::mem::take(
                                            &mut peer.lock().unwrap().disconnect_qualification,
                                        )
                                    {
                                        break;
                                    }
                                    let admitted = if packed {
                                        resident.contains(&(program, 256))
                                    } else {
                                        items.iter().all(|i| {
                                            resident.contains(&(
                                                AneProgram::Single,
                                                ladder_rung(i.n_tokens).unwrap(),
                                            ))
                                        })
                                    };
                                    if packed && peer.lock().unwrap().refuse_execution {
                                        WorkerResponse::unsupported_request(&request)
                                    } else if !admitted {
                                        WorkerResponse::Err {
                                            req_id: Some(req_id.clone()),
                                            code: "shape_not_admitted".into(),
                                            msg: "nonresident dispatch".into(),
                                        }
                                    } else {
                                        if let WorkerRequest::EmbedPacked { passes, .. } = &request
                                        {
                                            synapse_core::validate_packed_passes(
                                                &items
                                                    .iter()
                                                    .map(|i| i.n_tokens)
                                                    .collect::<Vec<_>>(),
                                                passes,
                                            )
                                            .unwrap();
                                        }
                                        let ids =
                                            synapse_core::decode_i32_frame(raw.as_ref().unwrap())
                                                .unwrap();
                                        let mut offset = 0;
                                        let values: Vec<f32> = items
                                            .iter()
                                            .map(|i| {
                                                let value = ids[offset] as f32;
                                                offset += i.n_tokens;
                                                value
                                                    + if packed && peer.lock().unwrap().corrupt {
                                                        1.0
                                                    } else {
                                                        0.0
                                                    }
                                            })
                                            .collect();
                                        assert_eq!(offset, ids.len());
                                        output = Some(encode_f32_frame(&values));
                                        WorkerResponse::Vectors {
                                            req_id: req_id.clone(),
                                            dims: 1,
                                            n: items.len(),
                                        }
                                    }
                                }
                                WorkerRequest::Ping { req_id } => WorkerResponse::Pong {
                                    req_id: req_id.clone(),
                                    rss_mb: 0,
                                    models_loaded: 1,
                                    placement_share: None,
                                    buckets: None,
                                    ane_resident_shapes: None,
                                },
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
                    let hello = synapse_core::worker_transport::handshake_on_stream_with_hello(
                        &mut host,
                        "frozen",
                        DEFAULT_MAX_FRAME_BYTES,
                        Duration::from_secs(2),
                        Some("ane-direct-worker"),
                        None,
                        None,
                    )
                    .await?;
                    Ok(AneWorkerSession {
                        pid: None,
                        stream: host,
                        packed: hello.packed,
                        confirm_exit: Box::pin(async move {
                            task.await
                                .map_err(|e| AneResidencyError::Channel(e.to_string()))?;
                            Ok(())
                        }),
                    })
                })
            });
            let channel = Arc::new(
                AneWorkerChannel::connect("packed-peer", DEFAULT_MAX_FRAME_BYTES, connector)
                    .await
                    .unwrap(),
            );
            Arc::new(
                AneServing::new(
                    channel,
                    AneResidencySupervisor::new(AneResidencyLimits::default()),
                    AneServingMetadata {
                        model_ref: PACKED_MODEL.into(),
                        buckets: ANE_SHAPE_LADDER.to_vec(),
                        dims: 1,
                        pooling: synapse_core::WorkerPooling::Last,
                        normalize: true,
                        timeout: Duration::from_secs(2),
                    },
                )
                .unwrap(),
            )
        }

        fn packed_inventory(prefix: &str) -> AnePlacementInventory {
            AnePlacementInventory {
                executables: (0..28)
                    .map(|layer| AneExecutable {
                        id: format!("{prefix}-{layer}"),
                        layers: vec![layer],
                    })
                    .collect(),
                cpu_stages: vec!["cpu_tail".into()],
            }
        }

        fn is_packed_request(request: &WorkerRequest) -> bool {
            matches!(
                request,
                WorkerRequest::EmbedPacked { .. }
                    | WorkerRequest::AneAdmitPacked { .. }
                    | WorkerRequest::AneEvictPacked { .. }
            )
        }

        async fn notified(notify: &Notify) {
            tokio::time::timeout(Duration::from_secs(3), notify.notified())
                .await
                .unwrap();
        }

        #[tokio::test]
        async fn cancelled_queued_eviction_restores_residency_without_wire_dispatch() {
            let peer = Arc::new(Mutex::new(PackedPeer::default()));
            let serving = packed_peer(false, peer.clone()).await;
            assert!(serving.prepare_packed().await.unwrap());
            let worker: Arc<dyn AneShapeWorker> = serving.channel.clone();
            let short = serving
                .supervisor
                .lease(&worker, PACKED_MODEL, 128)
                .await
                .unwrap();
            drop(
                serving
                    .supervisor
                    .lease(&worker, PACKED_MODEL, 256)
                    .await
                    .unwrap(),
            );
            let started = Arc::new(Notify::new());
            let release = Arc::new(Notify::new());
            peer.lock()
                .unwrap()
                .inference_gates
                .push_back((started.clone(), release.clone()));
            let packed = tokio::spawn({
                let serving = serving.clone();
                async move {
                    serving
                        .infer_packed(vec![vec![41; 127], vec![42; 129]])
                        .await
                }
            });
            notified(&started).await;
            let before = peer.lock().unwrap().requests.len();
            let victim = ShapeKey {
                program: AneProgram::Single,
                ..AneResidencySupervisor::packed_key(&worker, PACKED_MODEL)
            };
            let queued = tokio::spawn({
                let serving = serving.clone();
                async move { serving.infer(vec![vec![43; 257]], false).await }
            });
            tokio::time::timeout(Duration::from_secs(3), async {
                while serving.supervisor.inner.lock().slots[&victim].state != SlotState::Evicting {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            queued.abort();
            let _ = queued.await;
            tokio::time::timeout(Duration::from_secs(3), async {
                while serving.active_requests.load(Ordering::Acquire) != 1 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect(
                "an abandoned queued eviction must stop before another request releases the stream",
            );
            assert_eq!(
                serving.supervisor.inner.lock().slots[&victim].state,
                SlotState::Resident
            );
            assert_eq!(serving.supervisor.inner.lock().reserved_executables(), 84);
            assert_eq!(peer.lock().unwrap().requests.len(), before);
            release.notify_one();
            assert_eq!(packed.await.unwrap().unwrap(), vec![vec![41.0], vec![42.0]]);
            assert_eq!(
                peer.lock().unwrap().requests.len(),
                before,
                "no eviction or admission may reach the wire after its caller abandons queued work"
            );
            drop(short);
        }

        #[tokio::test]
        async fn packed_qualification_disconnect_recovers_before_next_standalone() {
            let peer = Arc::new(Mutex::new(PackedPeer {
                disconnect_qualification: true,
                ..Default::default()
            }));
            let serving = packed_peer(false, peer.clone()).await;
            let qualification =
                tokio::time::timeout(Duration::from_secs(3), serving.prepare_packed())
                    .await
                    .expect("qualification recovery must not wait on its own lease");
            assert!(matches!(qualification, Err(AneResidencyError::Channel(_))));
            let standalone = serving.infer(vec![vec![41; 1]], false).await;
            assert!(standalone.is_ok(), "qualification transport failure must recover before the next standalone request: {standalone:?}");
            assert_eq!(standalone.unwrap(), vec![vec![41.0]]);
            assert_eq!(serving.channel.generation(), 1);
            let state = serving.supervisor.inner.lock();
            assert!(state.slots.iter().all(|(key, slot)| key.program == AneProgram::Single && slot.generation == 1), "failed qualification must release its lease before recovery retires the unqualified reservation");
        }

        #[tokio::test]
        async fn packed_fast_path_yields_to_starved_admission_and_serves_singles() {
            let peer = Arc::new(Mutex::new(PackedPeer::default()));
            let serving = packed_peer(false, peer.clone()).await;
            assert!(serving.prepare_packed().await.unwrap());
            let worker: Arc<dyn AneShapeWorker> = serving.channel.clone();
            let short = serving
                .supervisor
                .lease(&worker, PACKED_MODEL, 128)
                .await
                .unwrap();
            let medium = serving
                .supervisor
                .lease(&worker, PACKED_MODEL, 256)
                .await
                .unwrap();
            let started = Arc::new(Notify::new());
            let release = Arc::new(Notify::new());
            peer.lock()
                .unwrap()
                .inference_gates
                .push_back((started.clone(), release.clone()));
            let packed = tokio::spawn({
                let serving = serving.clone();
                async move {
                    serving
                        .infer_packed(vec![vec![41; 127], vec![42; 129]])
                        .await
                }
            });
            notified(&started).await;
            let long = tokio::spawn({
                let serving = serving.clone();
                let worker = worker.clone();
                async move { serving.supervisor.lease(&worker, PACKED_MODEL, 512).await }
            });
            tokio::time::timeout(Duration::from_secs(3), async {
                while serving.supervisor.inner.lock().starved_head.is_none() {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            assert_eq!(serving.supervisor.inner.lock().reserved_executables(), 84);
            assert!(
                serving
                    .supervisor
                    .try_packed_lease(&worker, PACKED_MODEL)
                    .unwrap()
                    .is_none(),
                "new packed leases must not keep the starved head's only eviction victim busy"
            );
            let before = peer.lock().unwrap().requests.len();
            let mut arrivals = Vec::new();
            for _ in 0..2 {
                arrivals.push(tokio::spawn({
                    let serving = serving.clone();
                    async move {
                        serving
                            .infer_packed(vec![vec![51; 127], vec![52; 129]])
                            .await
                    }
                }));
            }
            tokio::time::timeout(Duration::from_secs(3), async {
                while serving.supervisor.inner.lock().waiters.len() < 3 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            let key = AneResidencySupervisor::packed_key(&worker, PACKED_MODEL);
            assert_eq!(serving.supervisor.inner.lock().slots[&key].leases, 1);
            release.notify_one();
            let long_lease = tokio::time::timeout(Duration::from_secs(3), long)
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            assert_eq!(packed.await.unwrap().unwrap(), vec![vec![41.0], vec![42.0]]);
            for arrival in arrivals {
                assert_eq!(
                    tokio::time::timeout(Duration::from_secs(3), arrival)
                        .await
                        .unwrap()
                        .unwrap()
                        .unwrap(),
                    vec![vec![51.0], vec![52.0]]
                );
            }
            assert!(
                peer.lock().unwrap().requests[before..]
                    .iter()
                    .all(|request| !matches!(request, WorkerRequest::EmbedPacked { .. })),
                "calls arriving behind a starved head must execute standalone"
            );
            drop(long_lease);
            drop(short);
            drop(medium);
        }

        #[tokio::test]
        async fn packed_preparation_skips_full_leased_budget_and_retries_idle() {
            let peer = Arc::new(Mutex::new(PackedPeer::default()));
            let mut serving = packed_peer(false, peer.clone()).await;
            let supervisor = budget_supervisor(30, Duration::from_secs(1));
            Arc::get_mut(&mut serving).unwrap().supervisor = supervisor.clone();
            let ledger = SharedLedger::default();
            let worker: Arc<dyn AneShapeWorker> = mock_channel("other-worker", &ledger).await;
            let mut held = Vec::new();
            for i in 0..7 {
                held.push(
                    supervisor
                        .lease(&worker, &format!("other-model-{i}"), 128)
                        .await
                        .unwrap(),
                );
            }
            assert!(!serving.prepare_packed().await.unwrap());
            assert!(
                peer.lock().unwrap().requests.is_empty(),
                "budget refusal must precede reference execution or packed admission"
            );
            drop(held);
            assert!(serving.prepare_packed().await.unwrap());
            assert_eq!(supervisor.inner.lock().reserved_executables(), 28);
        }

        #[tokio::test]
        async fn packed_evicted_between_plan_and_lease_replans_without_compiling() {
            let peer = Arc::new(Mutex::new(PackedPeer::default()));
            let serving = packed_peer(false, peer.clone()).await;
            assert!(serving.prepare_packed().await.unwrap());
            let worker: Arc<dyn AneShapeWorker> = serving.channel.clone();
            let key = AneResidencySupervisor::packed_key(&worker, PACKED_MODEL);
            let planned = Arc::new(Notify::new());
            let continue_lease = Arc::new(Notify::new());
            *serving.supervisor.inner.lease_decision_gate.lock().unwrap() =
                Some((planned.clone(), continue_lease.clone()));
            let infer = tokio::spawn({
                let serving = serving.clone();
                async move {
                    serving
                        .infer_packed(vec![vec![41; 127], vec![42; 129]])
                        .await
                }
            });
            notified(&planned).await;
            let id = {
                let mut state = serving.supervisor.inner.lock();
                let slot = state.slots.get_mut(&key).unwrap();
                slot.state = SlotState::Evicting;
                slot.id
            };
            assert!(serving.supervisor.evict(worker, key, id).await);
            let before = peer.lock().unwrap().requests.len();
            continue_lease.notify_one();
            assert_eq!(infer.await.unwrap().unwrap(), vec![vec![41.0], vec![42.0]]);
            assert!(
                peer.lock().unwrap().requests[before..]
                    .iter()
                    .all(|r| !is_packed_request(r)),
                "request must not re-admit an evicted packed key"
            );
        }

        /// Once the lane is closed (the compile scheduler stopped), a request
        /// whose rows are all served by the resident packed program is refused
        /// like any other (`SchedulerStopped`, which the wire reports as
        /// `engine_crashed`), and nothing reaches the worker. The slot table
        /// is left as it was, so only the packed lease's own check of the
        /// closed flag can refuse it.
        #[tokio::test]
        async fn packed_requests_are_refused_after_the_scheduler_stops() {
            let peer = Arc::new(Mutex::new(PackedPeer::default()));
            let serving = packed_peer(false, peer.clone()).await;
            assert!(serving.prepare_packed().await.unwrap());
            serving
                .supervisor
                .inner
                .closed
                .store(true, Ordering::Release);
            let before = peer.lock().unwrap().requests.len();
            let result = serving
                .infer_packed(vec![vec![41; 127], vec![42; 129]])
                .await;
            assert!(
                matches!(result, Err(AneResidencyError::SchedulerStopped)),
                "a packed request was served after the lane closed: {result:?}"
            );
            assert_eq!(
                peer.lock().unwrap().requests.len(),
                before,
                "a packed request reached the worker after the lane closed"
            );
        }

        #[tokio::test]
        async fn packed_cancelled_mixed_call_drains_with_one_lease_at_a_time() {
            let peer = Arc::new(Mutex::new(PackedPeer::default()));
            let serving = packed_peer(false, peer.clone()).await;
            assert!(serving.prepare_packed().await.unwrap());
            let before = peer.lock().unwrap().requests.len();
            let packed_started = Arc::new(Notify::new());
            let packed_release = Arc::new(Notify::new());
            peer.lock()
                .unwrap()
                .inference_gates
                .push_back((packed_started.clone(), packed_release.clone()));
            let execution = Arc::new(tokio::sync::Semaphore::new(1));
            let permits: PermitSource = Arc::new({
                let execution = execution.clone();
                move |_| {
                    let execution = execution.clone();
                    Box::pin(async move {
                        Ok(Box::new(execution.acquire_owned().await.unwrap()) as ExchangePermit)
                    })
                }
            });
            let infer = tokio::spawn({
                let serving = serving.clone();
                async move {
                    serving
                        .infer_packed_permitted(
                            vec![vec![41; 127], vec![42; 129], vec![43; 257]],
                            None,
                            (),
                            Some(permits),
                        )
                        .await
                }
            });
            notified(&packed_started).await;
            infer.abort();
            let _ = infer.await;
            assert_eq!(execution.available_permits(), 0, "the real execution permit must survive caller cancellation until the dispatched reply drains");
            {
                let state = serving.supervisor.inner.lock();
                assert_eq!(state.slots.values().map(|s| s.leases).sum::<u32>(), 1);
                assert!(state.slots.iter().any(|(key, slot)| matches!(
                    key.program,
                    AneProgram::Packed { .. }
                ) && slot.leases == 1));
            }
            packed_release.notify_one();
            tokio::time::timeout(Duration::from_secs(3), async {
                while serving.active_requests.load(Ordering::Acquire) != 0 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            assert_eq!(execution.available_permits(), 1);
            let requests = peer.lock().unwrap();
            assert_eq!(requests.requests[before..].len(), 1, "an abandoned mixed call must not admit or dispatch its undispatched standalone rungs");
            assert!(matches!(
                requests.requests[before],
                WorkerRequest::EmbedPacked { .. }
            ));
            assert_eq!(
                serving
                    .supervisor
                    .inner
                    .lock()
                    .slots
                    .values()
                    .map(|s| s.leases)
                    .sum::<u32>(),
                0
            );
            assert_eq!(serving.channel.generation(), 0);
        }

        #[tokio::test]
        async fn packed_preparation_stream_contention_keeps_shape_compiling_meaning() {
            let peer = Arc::new(Mutex::new(PackedPeer::default()));
            let serving = packed_peer(false, peer.clone()).await;
            let started = Arc::new(Notify::new());
            let release = Arc::new(Notify::new());
            peer.lock().unwrap().admission_gate = Some((started.clone(), release.clone()));
            let prepare = tokio::spawn({
                let serving = serving.clone();
                async move { serving.prepare_packed().await }
            });
            notified(&started).await;
            let before = peer.lock().unwrap().requests.len();
            let error = serving
                .infer_packed_permitted(
                    vec![vec![41; 1]],
                    Some(tokio::time::Instant::now() + Duration::from_millis(50)),
                    (),
                    None,
                )
                .await
                .unwrap_err();
            assert_eq!(error.code(), Some(ERR_ANE_SHAPE_COMPILING));
            assert_eq!(
                peer.lock().unwrap().requests.len(),
                before,
                "expired single waits for the stream, not for packed residency"
            );
            release.notify_one();
            assert!(prepare.await.unwrap().unwrap());
            assert_eq!(
                serving.infer_packed(vec![vec![41; 1]]).await.unwrap(),
                vec![vec![41.0]]
            );
        }

        #[tokio::test]
        async fn packed_nonresident_requests_are_singles_and_preserve_order() {
            let peer = Arc::new(Mutex::new(PackedPeer::default()));
            let serving = packed_peer(false, peer.clone()).await;
            let rows = vec![vec![41; 257], vec![42; 127], vec![43; 129], vec![42; 127]];
            assert_eq!(
                serving.infer_packed(rows.clone()).await.unwrap(),
                vec![vec![41.0], vec![42.0], vec![43.0], vec![42.0]]
            );
            assert!(peer
                .lock()
                .unwrap()
                .requests
                .iter()
                .all(|r| !is_packed_request(r)));
            assert!(serving.prepare_packed().await.unwrap());
            let prepared = peer.lock().unwrap().requests.len();
            assert_eq!(
                serving.infer_packed(rows).await.unwrap(),
                vec![vec![41.0], vec![42.0], vec![43.0], vec![42.0]]
            );
            assert!(
                peer.lock().unwrap().requests[prepared..]
                    .iter()
                    .any(|request| matches!(request, WorkerRequest::EmbedPacked { .. })),
                "a prepared resident packed program must actually execute packing"
            );
        }

        #[tokio::test]
        async fn packed_pending_layers_and_single_keys_are_distinct() {
            let peer = Arc::new(Mutex::new(PackedPeer::default()));
            let serving = packed_peer(false, peer.clone()).await;
            let worker: Arc<dyn AneShapeWorker> = serving.channel.clone();
            drop(
                serving
                    .supervisor
                    .lease(&worker, PACKED_MODEL, 128)
                    .await
                    .unwrap(),
            );
            drop(
                serving
                    .supervisor
                    .lease(&worker, PACKED_MODEL, 256)
                    .await
                    .unwrap(),
            );
            let started = Arc::new(Notify::new());
            let release = Arc::new(Notify::new());
            peer.lock().unwrap().admission_gate = Some((started.clone(), release.clone()));
            let prepare = tokio::spawn({
                let serving = serving.clone();
                let worker = worker.clone();
                async move {
                    serving
                        .supervisor
                        .prepare_packed_lease(&worker, PACKED_MODEL)
                        .await
                }
            });
            notified(&started).await;
            let key = AneResidencySupervisor::packed_key(&worker, PACKED_MODEL);
            let single = ShapeKey {
                program: AneProgram::Single,
                ..key.clone()
            };
            {
                let state = serving.supervisor.inner.lock();
                assert_eq!(state.slots.len(), 3);
                assert_eq!(state.slots[&key].state, SlotState::Admitting);
                assert_eq!(state.slots[&single].state, SlotState::Resident);
                assert_eq!(state.slots[&key].executables, 28);
                assert_eq!(state.reserved_executables(), 84);
                assert!(!state.is_pinned(&key));
            }
            release.notify_one();
            drop(prepare.await.unwrap().unwrap().unwrap());
        }

        #[tokio::test]
        async fn packed_budget_charge_waits_for_eviction_ack() {
            let peer = Arc::new(Mutex::new(PackedPeer::default()));
            let serving = packed_peer(false, peer.clone()).await;
            assert!(serving.prepare_packed().await.unwrap());
            let worker: Arc<dyn AneShapeWorker> = serving.channel.clone();
            let short = serving
                .supervisor
                .lease(&worker, PACKED_MODEL, 128)
                .await
                .unwrap();
            let single256 = serving
                .supervisor
                .lease(&worker, PACKED_MODEL, 256)
                .await
                .unwrap();
            let started = Arc::new(Notify::new());
            let release = Arc::new(Notify::new());
            peer.lock().unwrap().eviction_gate = Some((started.clone(), release.clone()));
            let request = tokio::spawn({
                let serving = serving.clone();
                let worker = worker.clone();
                async move { serving.supervisor.lease(&worker, PACKED_MODEL, 512).await }
            });
            notified(&started).await;
            let key = AneResidencySupervisor::packed_key(&worker, PACKED_MODEL);
            {
                let state = serving.supervisor.inner.lock();
                assert_eq!(state.slots[&key].state, SlotState::Evicting);
                assert_eq!(
                    state.reserved_executables(),
                    84,
                    "evicting programs remain charged until the peer ACK"
                );
                assert!(state.slots.keys().all(|key| key.shape != 512));
            }
            release.notify_one();
            drop(request.await.unwrap().unwrap());
            drop(short);
            drop(single256);
            assert!(!serving.supervisor.inner.lock().slots.contains_key(&key));
        }

        #[tokio::test]
        async fn packed_v3_peer_without_capability_never_receives_packed_and_stays_framed() {
            let peer = Arc::new(Mutex::new(PackedPeer::default()));
            let serving = packed_peer(true, peer.clone()).await;
            assert!(!serving.prepare_packed().await.unwrap());
            for _ in 0..2 {
                assert_eq!(
                    serving
                        .infer_packed(vec![vec![41; 127], vec![42; 129]])
                        .await
                        .unwrap(),
                    vec![vec![41.0], vec![42.0]]
                );
                let (response, _) = serving
                    .channel
                    .exchange(
                        &WorkerRequest::Ping {
                            req_id: "after-raw-frame".into(),
                        },
                        None,
                    )
                    .await
                    .unwrap();
                assert!(matches!(response, WorkerResponse::Pong { .. }));
            }
            let peer = peer.lock().unwrap();
            assert!(!peer.crashed);
            assert!(peer.requests.iter().all(|r| !is_packed_request(r)));
            assert_eq!(
                peer.requests
                    .iter()
                    .filter(|r| matches!(r, WorkerRequest::EmbedBatch { .. }))
                    .count(),
                4
            );
            assert_eq!(serving.channel.generation(), 0);
        }

        #[tokio::test]
        async fn packed_capable_unsupported_disables_and_falls_back() {
            for admission in [true, false] {
                let peer = Arc::new(Mutex::new(PackedPeer {
                    refuse_admission: admission,
                    ..Default::default()
                }));
                let serving = packed_peer(false, peer.clone()).await;
                assert_eq!(serving.prepare_packed().await.unwrap(), !admission);
                if !admission {
                    peer.lock().unwrap().refuse_execution = true;
                }
                let rows = vec![vec![41; 127], vec![42; 129], vec![43; 257]];
                assert_eq!(
                    serving.infer_packed(rows.clone()).await.unwrap(),
                    vec![vec![41.0], vec![42.0], vec![43.0]]
                );
                let packed_count = peer
                    .lock()
                    .unwrap()
                    .requests
                    .iter()
                    .filter(|r| is_packed_request(r))
                    .count();
                assert_eq!(
                    serving.infer_packed(rows).await.unwrap(),
                    vec![vec![41.0], vec![42.0], vec![43.0]]
                );
                assert_eq!(
                    peer.lock()
                        .unwrap()
                        .requests
                        .iter()
                        .filter(|r| is_packed_request(r))
                        .count(),
                    packed_count
                );
                assert!(!serving.channel.packing_available());
                assert_eq!(
                    serving.channel.generation(),
                    0,
                    "unsupported packing must not restart standalone service"
                );
            }
        }

        #[tokio::test]
        async fn packed_qualification_mismatch_disables_only_packing() {
            let peer = Arc::new(Mutex::new(PackedPeer {
                corrupt: true,
                ..Default::default()
            }));
            let serving = packed_peer(false, peer.clone()).await;
            assert!(!serving.prepare_packed().await.unwrap());
            let before = peer.lock().unwrap().requests.len();
            assert_eq!(
                serving
                    .infer_packed(vec![vec![51; 128], vec![52; 128]])
                    .await
                    .unwrap(),
                vec![vec![51.0], vec![52.0]]
            );
            assert!(peer.lock().unwrap().requests[before..]
                .iter()
                .all(|r| !is_packed_request(r)));
            assert_eq!(serving.channel.generation(), 0);
        }

        #[tokio::test]
        async fn retirement_during_exit_confirmation_never_connects_replacement() {
            let ledger = SharedLedger::default();
            let exiting = Arc::new(Notify::new());
            let allow_exit = Arc::new(Notify::new());
            ledger.lock().unwrap().exit_gate = Some((exiting.clone(), allow_exit.clone()));
            let channel = mock_channel("worker", &ledger).await;
            let restart = tokio::spawn({
                let channel = channel.clone();
                async move { channel.restart().await }
            });
            tokio::time::timeout(Duration::from_secs(2), exiting.notified())
                .await
                .unwrap();
            channel.begin_retirement();
            allow_exit.notify_one();
            let result = tokio::time::timeout(Duration::from_secs(2), restart)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(
                ledger.lock().unwrap().connects["worker"],
                1,
                "retirement must stop a replacement after owner exit"
            );
            assert!(result.is_err());
            channel.shutdown().await.unwrap();
        }

        #[tokio::test]
        async fn replacement_never_starts_before_owner_exit_confirmation() {
            let starts = Arc::new(AtomicU64::new(0));
            let old_closed = Arc::new(Notify::new());
            let confirm_allowed = Arc::new(Notify::new());
            let ledger = Arc::new(Mutex::new(Ledger::default()));
            let connector: AneWorkerConnector<DuplexStream> = Box::new({
                let starts = starts.clone();
                let old_closed = old_closed.clone();
                let confirm_allowed = confirm_allowed.clone();
                move || {
                    let epoch = starts.fetch_add(1, Ordering::SeqCst);
                    let old_closed = old_closed.clone();
                    let confirm_allowed = confirm_allowed.clone();
                    let ledger = ledger.clone();
                    Box::pin(async move {
                        let (host, worker) = tokio::io::duplex(1 << 16);
                        let task = tokio::spawn(serve_mock(worker, ledger));
                        Ok(AneWorkerSession {
                            pid: None,
                            packed: None,
                            stream: host,
                            confirm_exit: Box::pin(async move {
                                task.await.map_err(|error| {
                                    AneResidencyError::Channel(error.to_string())
                                })?;
                                if epoch == 0 {
                                    old_closed.notify_one();
                                    confirm_allowed.notified().await;
                                }
                                Ok(())
                            }),
                        })
                    })
                }
            });
            let channel = Arc::new(
                AneWorkerChannel::connect("worker", DEFAULT_MAX_FRAME_BYTES, connector)
                    .await
                    .unwrap(),
            );
            let restart = tokio::spawn({
                let channel = channel.clone();
                async move { channel.restart().await }
            });
            tokio::time::timeout(Duration::from_secs(1), old_closed.notified())
                .await
                .unwrap();
            let starts_before_confirmation = starts.load(Ordering::SeqCst);
            confirm_allowed.notify_one();
            tokio::time::timeout(Duration::from_secs(1), restart)
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            assert_eq!(
                starts_before_confirmation, 1,
                "replacement must not be spawned while old-owner exit is unconfirmed"
            );
            assert_eq!(starts.load(Ordering::SeqCst), 2);
        }

        #[tokio::test]
        async fn restart_reloads_pinned_models_before_returning_ready() {
            let ledger = Arc::new(Mutex::new(Ledger::default()));
            let channel = mock_channel("worker", &ledger).await;
            let request = WorkerRequest::Load {
                req_id: "initial-load".into(),
                artifact_path: "pinned.safetensors".into(),
                artifact_digest: "sha256:pinned".into(),
                format: "safetensors-package".into(),
                runtime_config: BTreeMap::from([(
                    "profile".into(),
                    "pinned-worker-profile".into(),
                )]),
            };
            channel.exchange(&request, None).await.unwrap();
            channel.restart().await.unwrap();
            let data = ledger.lock().unwrap();
            assert_eq!(
                data.loads.len(),
                2,
                "replacement must reload the pinned package"
            );
            match (&data.loads[0], &data.loads[1]) {
                (
                    WorkerRequest::Load {
                        req_id: old,
                        artifact_path: a,
                        artifact_digest: b,
                        format: c,
                        runtime_config: d,
                    },
                    WorkerRequest::Load {
                        req_id: new,
                        artifact_path: aa,
                        artifact_digest: bb,
                        format: cc,
                        runtime_config: dd,
                    },
                ) => {
                    assert_ne!(old, new);
                    assert_eq!((a, b, c, d), (aa, bb, cc, dd));
                }
                _ => panic!("expected load records"),
            }
            assert_eq!(data.connects["worker"], 2);
        }

        fn budget_supervisor(budget: usize, wait: Duration) -> AneResidencySupervisor {
            let mut supervisor = AneResidencySupervisor::new(AneResidencyLimits::default());
            let inner = Arc::get_mut(&mut supervisor.inner).unwrap();
            inner.executable_budget = budget;
            inner.wait_timeout = wait;
            supervisor
        }

        include!("admission_tests.rs");
        include!("compile_queue_tests.rs");
        include!("compile_queue_fault_tests.rs");

        #[tokio::test]
        async fn executable_budget_refuses_before_hardware_limit() {
            let ledger = Arc::new(Mutex::new(Ledger::default()));
            let worker: Arc<dyn AneShapeWorker> = mock_channel("worker", &ledger).await;
            let supervisor = budget_supervisor(7, Duration::from_millis(20));
            let protected = supervisor.lease(&worker, "model", 128).await.unwrap();
            let result = supervisor.lease(&worker, "model", 256).await;
            let error = match result {
                Err(error) => error,
                Ok(_) => panic!("budget sent an over-budget shape to hardware"),
            };
            assert_eq!(error.code(), Some(ERR_ANE_RESOURCES_EXHAUSTED));
            let data = ledger.lock().unwrap();
            assert_eq!(data.admit_attempts.len(), 1);
            assert!(!data.admit_attempts.contains_key(&("model".into(), 256)));
            assert_eq!(supervisor.inner.lock().reserved_executables(), 4);
            drop(data);
            drop(protected);
        }

        #[tokio::test]
        async fn pending_compiles_reserve_full_executable_budget() {
            let ledger = Arc::new(Mutex::new(Ledger::default()));
            let worker: Arc<dyn AneShapeWorker> = mock_channel("worker", &ledger).await;
            let supervisor = budget_supervisor(7, Duration::from_secs(1));
            let first = ShapeKey {
                worker_id: worker.worker_id().to_owned(),
                model_ref: "model".into(),
                shape: 128,
                program: AneProgram::Single,
            };
            let (started, release) = notify_pair();
            ledger.lock().unwrap().admit_gate = Some((started.clone(), release.clone()));
            let compiling = tokio::spawn({
                let supervisor = supervisor.clone();
                let worker = worker.clone();
                async move { supervisor.lease(&worker, "model", 128).await }
            });
            notified(&started).await;
            assert_eq!(
                supervisor.inner.lock().slots[&first].state,
                SlotState::Admitting
            );
            let second = tokio::spawn({
                let supervisor = supervisor.clone();
                let worker = worker.clone();
                async move { supervisor.lease(&worker, "model", 256).await }
            });
            until(|| supervisor.inner.lock().waiters.len() == 1).await;
            tokio::time::sleep(Duration::from_millis(50)).await;
            assert!(
                !second.is_finished(),
                "pending compile's four layers must prevent another four-layer reservation"
            );
            assert_eq!(supervisor.inner.lock().reserved_executables(), 4);
            assert!(!ledger
                .lock()
                .unwrap()
                .admit_attempts
                .contains_key(&("model".into(), 256)));
            release.notify_one();
            drop(compiling.await.unwrap().unwrap());
            second.abort();
        }

        #[tokio::test]
        async fn executable_budget_waits_then_succeeds_when_lease_releases() {
            let ledger = Arc::new(Mutex::new(Ledger::default()));
            let worker: Arc<dyn AneShapeWorker> = mock_channel("worker", &ledger).await;
            let supervisor = budget_supervisor(7, Duration::from_secs(1));
            let protected = supervisor.lease(&worker, "model", 128).await.unwrap();
            let task = tokio::spawn({
                let supervisor = supervisor.clone();
                let worker = worker.clone();
                async move { supervisor.lease(&worker, "model", 256).await }
            });
            tokio::time::timeout(Duration::from_secs(1), async {
                while supervisor.inner.lock().waiters.is_empty() {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            assert!(!task.is_finished());
            assert!(!ledger
                .lock()
                .unwrap()
                .admit_attempts
                .contains_key(&("model".into(), 256)));
            drop(protected);
            let lease = tokio::time::timeout(Duration::from_secs(1), task)
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            let data = ledger.lock().unwrap();
            assert!(data.events.contains(&"evict model 128".into()));
            assert!(data.resident.contains(&("model".into(), 256)));
            assert_eq!(data.leased_evicts, 0);
            drop(data);
            drop(lease);
        }

        #[tokio::test]
        async fn compile_rpc_measurement_observes_the_actual_worker_exchange() {
            let ledger = SharedLedger::default();
            ledger.lock().unwrap().admit_delay = Duration::from_millis(20);
            let channel = mock_channel("worker", &ledger).await;
            let elapsed = Arc::new(Mutex::new(None));
            COMPILE_RPC_MS
                .scope(elapsed.clone(), channel.admit_shape("model", 128))
                .await
                .unwrap();
            assert!(
                elapsed.lock().unwrap().unwrap() >= 20.0,
                "the compile sample must include the delayed worker response, not a proxy"
            );
        }

        #[tokio::test]
        async fn admission_wait_measurement_covers_success_and_deadline_refusal() {
            let ledger = SharedLedger::default();
            let worker: Arc<dyn AneShapeWorker> = mock_channel("worker", &ledger).await;
            let supervisor = budget_supervisor(4, Duration::from_millis(20));
            let waits = AdmissionWaits::default();
            ADMISSION_WAITS
                .scope(waits.clone(), async {
                    let held = supervisor.lease(&worker, "model", 128).await.unwrap();
                    assert!(supervisor.lease(&worker, "model", 256).await.is_err());
                    drop(held);
                })
                .await;
            let waits = waits.lock().unwrap();
            assert_eq!(waits.len(), 2);
            assert_eq!(waits[0].shape, 128);
            assert!(waits[0].admitted);
            assert_eq!(waits[1].shape, 256);
            assert!(!waits[1].admitted);
            assert!(
                waits[1].elapsed_ms >= 20.0,
                "measure the actual admission future, including its wait deadline"
            );
        }

        /// The worker cannot answer an inference while it compiles, so a
        /// request for a resident shape waits on the stream for the whole
        /// compile. That wait must not count against the exchange I/O timeout:
        /// a timeout there is a transport fault, and fault recovery restarts
        /// the worker, dropping every compiled shape.
        #[tokio::test]
        async fn inference_waiting_behind_a_compile_longer_than_its_exchange_timeout_does_not_restart_the_worker(
        ) {
            let ledger = SharedLedger::default();
            let mut serving = serving_fixture(&ledger).await;
            Arc::get_mut(&mut serving).unwrap().metadata.timeout = Duration::from_millis(100);
            serving.infer(vec![vec![1]], false).await.unwrap();
            let started = Arc::new(Notify::new());
            let release = Arc::new(Notify::new());
            ledger.lock().unwrap().admit_gate = Some((started.clone(), release.clone()));
            let compile = tokio::spawn({
                let serving = serving.clone();
                async move { serving.infer(vec![vec![1; 600]], false).await }
            });
            tokio::time::timeout(Duration::from_secs(2), started.notified())
                .await
                .unwrap();
            let waiting = tokio::spawn({
                let serving = serving.clone();
                async move {
                    serving
                        .infer_guarded(
                            vec![vec![1]],
                            false,
                            Some(tokio::time::Instant::now() + Duration::from_secs(5)),
                            (),
                        )
                        .await
                }
            });
            // Hold the compile for four exchange timeouts.
            tokio::time::sleep(Duration::from_millis(400)).await;
            release.notify_one();
            let waited = tokio::time::timeout(Duration::from_secs(5), waiting)
                .await
                .unwrap()
                .unwrap();
            let compiled = tokio::time::timeout(Duration::from_secs(5), compile)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(waited.unwrap(), vec![vec![128.0]]);
            assert_eq!(compiled.unwrap(), vec![vec![1024.0]]);
            assert_eq!(serving.supervisor.stats().restarts, 0);
            assert!(!serving.channel.faulted.load(Ordering::Relaxed));
            assert_eq!(ledger.lock().unwrap().connects["serving"], 1);
        }

        fn notify_pair() -> (Arc<Notify>, Arc<Notify>) {
            (Arc::new(Notify::new()), Arc::new(Notify::new()))
        }

        async fn until(mut condition: impl FnMut() -> bool) {
            tokio::time::timeout(Duration::from_secs(2), async {
                while !condition() {
                    tokio::time::sleep(Duration::from_millis(2)).await;
                }
            })
            .await
            .expect("condition never held");
        }

        fn gate_classification(serving: &AneServing<DuplexStream>) -> Arc<ClassifyGate> {
            let gate = ClassifyGate::new();
            *serving.channel.classify_gate.lock().unwrap() = Some(gate.clone());
            gate
        }

        async fn acknowledged(notify: &Notify) {
            tokio::time::timeout(Duration::from_secs(2), notify.notified())
                .await
                .expect("acknowledgement never arrived");
        }

        /// Both the caller and the serving task have expired and are held
        /// before classifying.
        async fn both_expired(gate: &ClassifyGate) {
            tokio::time::timeout(Duration::from_secs(2), async {
                gate.reached(GateSite::Caller, 1).await;
                gate.reached(GateSite::Inner, 1).await;
            })
            .await
            .expect("the request never expired");
        }

        fn spawn_infer(
            serving: &Arc<AneServing<DuplexStream>>,
            tokens: usize,
            deadline: Option<tokio::time::Instant>,
        ) -> tokio::task::JoinHandle<Result<Vec<Vec<f32>>, AneResidencyError>> {
            let serving = serving.clone();
            tokio::spawn(async move {
                serving
                    .infer_permitted(vec![vec![1; tokens]], false, deadline, (), None)
                    .await
            })
        }

        fn in_150_ms() -> Option<tokio::time::Instant> {
            Some(tokio::time::Instant::now() + Duration::from_millis(150))
        }

        fn embed_request(
            serving: &AneServing<DuplexStream>,
            tokens: usize,
        ) -> (WorkerRequest, Vec<u8>) {
            let request = WorkerRequest::EmbedBatch {
                req_id: serving.channel.next_req_id("embed"),
                model_ref: serving.metadata.model_ref.clone(),
                pooling: synapse_core::WorkerPooling::Mean,
                normalize: true,
                items: vec![synapse_core::WorkerTokenItem {
                    id: "0".into(),
                    n_tokens: tokens,
                }],
            };
            (request, encode_i32_frame(&vec![1; tokens]))
        }

        /// A request holding the stream while it waits for an execution
        /// permit gives up at its deadline; a queued compile then takes the
        /// stream before the caller's expiry is classified. The serving task's
        /// own classification is held back, so the caller's is the one tested.
        /// The expired wait was the permit, so the answer is deadline_exceeded.
        #[tokio::test]
        async fn expired_permit_wait_stays_deadline_exceeded_when_a_compile_starts_before_classification(
        ) {
            let ledger = SharedLedger::default();
            let serving = serving_fixture(&ledger).await;
            serving.infer(vec![vec![1]], false).await.unwrap();
            let gate = gate_classification(&serving);
            let (started, release) = notify_pair();
            ledger.lock().unwrap().admit_gate = Some((started.clone(), release.clone()));
            let never: PermitSource = Arc::new(|_| {
                Box::pin(std::future::pending::<
                    Result<ExchangePermit, AneResidencyError>,
                >())
            });
            let waiting = tokio::spawn({
                let serving = serving.clone();
                async move {
                    serving
                        .infer_permitted(vec![vec![1]], false, in_150_ms(), (), Some(never))
                        .await
                }
            });
            until(|| serving.channel.stream.try_lock().is_err()).await;
            let compile = spawn_infer(&serving, 600, None);
            both_expired(&gate).await;
            acknowledged(&started).await;
            let compiling_at_classification = serving.channel.compiling_shape();
            gate.release(GateSite::Caller);
            let answered = waiting.await.unwrap();
            gate.release(GateSite::Inner);
            release.notify_one();
            compile.await.unwrap().unwrap();
            assert_eq!(compiling_at_classification, Some(1024));
            assert!(
                matches!(answered, Err(AneResidencyError::DeadlineExceeded)),
                "{answered:?}"
            );
        }

        /// A request waiting for budget (no evictable shape) expires while an
        /// admission holds the stream; the caller classifies first. It was not
        /// waiting for that compile.
        #[tokio::test]
        async fn expired_budget_wait_stays_deadline_exceeded_while_an_admit_holds_the_stream() {
            let ledger = SharedLedger::default();
            // Room for two shapes of the fixture's 22-layer model.
            let supervisor = budget_supervisor(44, Duration::from_secs(5));
            let serving = serving_fixture_for(&ledger, "serving", supervisor.clone()).await;
            serving.infer(vec![vec![1]], false).await.unwrap();
            let worker: Arc<dyn AneShapeWorker> = serving.channel.clone();
            let held_128 = supervisor
                .lease(&worker, &serving.metadata.model_ref, 128)
                .await
                .unwrap();
            let gate = gate_classification(&serving);
            let (started, release) = notify_pair();
            ledger.lock().unwrap().admit_gate = Some((started.clone(), release.clone()));
            let compile = spawn_infer(&serving, 200, None);
            acknowledged(&started).await;
            let waiting = spawn_infer(&serving, 400, in_150_ms());
            both_expired(&gate).await;
            let compiling_at_classification = serving.channel.compiling_shape();
            gate.release(GateSite::Caller);
            let answered = waiting.await.unwrap();
            gate.release(GateSite::Inner);
            release.notify_one();
            compile.await.unwrap().unwrap();
            drop(held_128);
            assert_eq!(compiling_at_classification, Some(256));
            assert!(
                matches!(answered, Err(AneResidencyError::DeadlineExceeded)),
                "{answered:?}"
            );
        }

        /// The request's own shape is Admitting, but its admit is still queued
        /// behind an inference holding the stream when the deadline expires.
        /// The request is blocked on that compile even though no admit holds
        /// the stream yet.
        #[tokio::test]
        async fn deadline_on_an_own_admit_queued_behind_inference_is_shape_compiling() {
            let ledger = SharedLedger::default();
            let serving = serving_fixture(&ledger).await;
            serving.infer(vec![vec![1]], false).await.unwrap();
            let gate = gate_classification(&serving);
            let (started, release) = notify_pair();
            ledger.lock().unwrap().inference_gate = Some((started.clone(), release.clone()));
            let inference = spawn_infer(&serving, 1, None);
            acknowledged(&started).await;
            let waiting = spawn_infer(&serving, 600, in_150_ms());
            // Its own admit is not a deadline-bounded wait, so only the caller expires.
            tokio::time::timeout(Duration::from_secs(2), gate.reached(GateSite::Caller, 1))
                .await
                .expect("the request never expired");
            let compiling_at_classification = serving.channel.compiling_shape();
            gate.release(GateSite::Caller);
            let answered = waiting.await.unwrap();
            gate.release(GateSite::Inner);
            release.notify_one();
            inference.await.unwrap().unwrap();
            // The caller was told shape_compiling, so the compile it was told
            // to wait for still runs once the stream is free.
            let model_ref = serving.metadata.model_ref.clone();
            until(|| {
                ledger
                    .lock()
                    .unwrap()
                    .resident
                    .contains(&(model_ref.clone(), 1024))
            })
            .await;
            assert_eq!(
                compiling_at_classification, None,
                "the admit must still be queued"
            );
            assert!(
                matches!(
                    answered,
                    Err(AneResidencyError::ShapeCompiling { shape: 1024, .. })
                ),
                "{answered:?}"
            );
        }

        /// A request joins another request's compile of the same shape and its
        /// deadline expires during that compile.
        #[tokio::test]
        async fn single_flight_joiner_expiring_during_the_compile_is_shape_compiling() {
            let ledger = SharedLedger::default();
            let serving = serving_fixture(&ledger).await;
            serving.infer(vec![vec![1]], false).await.unwrap();
            let gate = gate_classification(&serving);
            let (started, release) = notify_pair();
            ledger.lock().unwrap().admit_gate = Some((started.clone(), release.clone()));
            let compile = spawn_infer(&serving, 600, None);
            acknowledged(&started).await;
            let joiner = spawn_infer(&serving, 600, in_150_ms());
            both_expired(&gate).await;
            gate.release(GateSite::Caller);
            let answered = joiner.await.unwrap();
            gate.release(GateSite::Inner);
            release.notify_one();
            compile.await.unwrap().unwrap();
            assert!(
                matches!(
                    answered,
                    Err(AneResidencyError::ShapeCompiling { shape: 1024, .. })
                ),
                "{answered:?}"
            );
            let model_ref = serving.metadata.model_ref.clone();
            assert_eq!(ledger.lock().unwrap().admit_attempts[&(model_ref, 1024)], 1);
        }

        /// A joiner whose shared admit fails waits on the Failed slot while
        /// recovery restarts the worker; its deadline expires there. That is
        /// not a compile in progress.
        #[tokio::test]
        async fn expiry_after_a_failed_admit_is_deadline_exceeded() {
            let ledger = SharedLedger::default();
            // A worker process captures the exit hold when it starts.
            let (exiting, exit) = notify_pair();
            ledger.lock().unwrap().exit_gate = Some((exiting.clone(), exit.clone()));
            let serving = serving_fixture(&ledger).await;
            serving.infer(vec![vec![1]], false).await.unwrap();
            let model_ref = serving.metadata.model_ref.clone();
            let (started, release) = notify_pair();
            {
                let mut ledger = ledger.lock().unwrap();
                ledger.fail_admit.insert((model_ref.clone(), 1024));
                ledger.admit_gate = Some((started.clone(), release.clone()));
            }
            let failing = spawn_infer(&serving, 600, None);
            acknowledged(&started).await;
            let deadline = Some(tokio::time::Instant::now() + Duration::from_millis(400));
            let joiner = spawn_infer(&serving, 600, deadline);
            until(|| !serving.supervisor.inner.lock().waiters.is_empty()).await;
            release.notify_one();
            // Recovery is confirming the old worker's exit: the failed slot is
            // kept until then (Rejected: the worker answered with an error).
            acknowledged(&exiting).await;
            let slot_failed = serving
                .supervisor
                .inner
                .lock()
                .slots
                .values()
                .any(|slot| matches!(slot.state, SlotState::Failed | SlotState::Rejected));
            let answered = tokio::time::timeout(Duration::from_secs(2), joiner)
                .await
                .unwrap()
                .unwrap();
            exit.notify_one();
            let failed = tokio::time::timeout(Duration::from_secs(5), failing)
                .await
                .unwrap()
                .unwrap();
            assert!(slot_failed);
            assert!(
                matches!(answered, Err(AneResidencyError::DeadlineExceeded)),
                "{answered:?}"
            );
            assert!(failed.is_err(), "{failed:?}");
        }

        /// (a) An inference holds the stream through the deadline; after the
        /// deadline a compile takes the stream, before classification. No
        /// compile held the stream during the wait.
        #[tokio::test]
        async fn caller_classifies_a_stream_wait_behind_inference_as_deadline_exceeded_even_if_a_compile_follows(
        ) {
            let ledger = SharedLedger::default();
            let serving = serving_fixture(&ledger).await;
            serving.infer(vec![vec![1]], false).await.unwrap();
            let gate = gate_classification(&serving);
            let (inferring, finish_inference) = notify_pair();
            ledger.lock().unwrap().inference_gate =
                Some((inferring.clone(), finish_inference.clone()));
            let inference = spawn_infer(&serving, 1, None);
            acknowledged(&inferring).await;
            let waiting = spawn_infer(&serving, 1, in_150_ms());
            both_expired(&gate).await;
            let (compiling, finish_compile) = notify_pair();
            ledger.lock().unwrap().admit_gate = Some((compiling.clone(), finish_compile.clone()));
            let compile = spawn_infer(&serving, 600, None);
            finish_inference.notify_one();
            acknowledged(&compiling).await;
            let compiling_at_classification = serving.channel.compiling_shape();
            // Only the caller classifies: the serving task's own
            // classification stays held until the caller has answered.
            gate.release(GateSite::Caller);
            let answered = waiting.await.unwrap();
            gate.release(GateSite::Inner);
            finish_compile.notify_one();
            inference.await.unwrap().unwrap();
            compile.await.unwrap().unwrap();
            assert_eq!(compiling_at_classification, Some(1024));
            assert!(
                matches!(answered, Err(AneResidencyError::DeadlineExceeded)),
                "{answered:?}"
            );
        }

        /// (b) A compile holds the stream through the deadline; after the
        /// deadline an inference takes the stream, before classification.
        #[tokio::test]
        async fn caller_classifies_a_stream_wait_behind_a_compile_as_shape_compiling_even_if_inference_follows(
        ) {
            let ledger = SharedLedger::default();
            let serving = serving_fixture(&ledger).await;
            serving.infer(vec![vec![1]], false).await.unwrap();
            let gate = gate_classification(&serving);
            let (compiling, finish_compile) = notify_pair();
            ledger.lock().unwrap().admit_gate = Some((compiling.clone(), finish_compile.clone()));
            let compile = spawn_infer(&serving, 600, None);
            acknowledged(&compiling).await;
            let waiting = spawn_infer(&serving, 1, in_150_ms());
            both_expired(&gate).await;
            let (inferring, finish_inference) = notify_pair();
            ledger.lock().unwrap().inference_gate =
                Some((inferring.clone(), finish_inference.clone()));
            finish_compile.notify_one();
            // The compiling request's own inference now holds the stream.
            acknowledged(&inferring).await;
            let compiling_at_classification = serving.channel.compiling_shape();
            // Only the caller classifies: the serving task's own
            // classification stays held until the caller has answered.
            gate.release(GateSite::Caller);
            let answered = waiting.await.unwrap();
            gate.release(GateSite::Inner);
            finish_inference.notify_one();
            compile.await.unwrap().unwrap();
            assert_eq!(compiling_at_classification, None);
            assert!(
                matches!(
                    answered,
                    Err(AneResidencyError::ShapeCompiling { shape: 1024, .. })
                ),
                "{answered:?}"
            );
        }

        /// The serving task's own stream wait (no caller involved) expires
        /// behind an inference; a compile takes the stream before the wait is
        /// classified. No compile held the stream during the wait.
        #[tokio::test]
        async fn serving_task_classifies_a_stream_wait_behind_inference_as_deadline_exceeded_even_if_a_compile_follows(
        ) {
            let ledger = SharedLedger::default();
            let serving = serving_fixture(&ledger).await;
            serving.infer(vec![vec![1]], false).await.unwrap();
            let gate = gate_classification(&serving);
            let (inferring, finish_inference) = notify_pair();
            ledger.lock().unwrap().inference_gate =
                Some((inferring.clone(), finish_inference.clone()));
            let inference = spawn_infer(&serving, 1, None);
            acknowledged(&inferring).await;
            let (request, raw) = embed_request(&serving, 1);
            let deadline = tokio::time::Instant::now() + Duration::from_millis(150);
            let ready = async { Ok(Box::new(()) as ExchangePermit) };
            let waiting = serving.channel.exchange_inference(
                &request,
                Some(&raw),
                Some(deadline),
                Duration::from_secs(1),
                ready,
            );
            let (compiling, finish_compile) = notify_pair();
            let mut compile = None;
            let switch_holder = async {
                tokio::time::timeout(Duration::from_secs(2), gate.reached(GateSite::Inner, 1))
                    .await
                    .expect("the wait never expired");
                ledger.lock().unwrap().admit_gate =
                    Some((compiling.clone(), finish_compile.clone()));
                compile = Some(spawn_infer(&serving, 600, None));
                finish_inference.notify_one();
                acknowledged(&compiling).await;
                let compiling_at_classification = serving.channel.compiling_shape();
                gate.release(GateSite::Inner);
                compiling_at_classification
            };
            let (answered, compiling_at_classification) = tokio::join!(waiting, switch_holder);
            finish_compile.notify_one();
            inference.await.unwrap().unwrap();
            compile.unwrap().await.unwrap().unwrap();
            assert_eq!(compiling_at_classification, Some(1024));
            assert!(
                matches!(answered, Err(AneResidencyError::DeadlineExceeded)),
                "{answered:?}"
            );
        }

        /// The serving task's own stream wait (no caller involved) expires
        /// behind a compile; an inference takes the stream before the wait is
        /// classified.
        #[tokio::test]
        async fn serving_task_classifies_a_stream_wait_behind_a_compile_as_shape_compiling_even_if_inference_follows(
        ) {
            let ledger = SharedLedger::default();
            let serving = serving_fixture(&ledger).await;
            serving.infer(vec![vec![1]], false).await.unwrap();
            let gate = gate_classification(&serving);
            let (compiling, finish_compile) = notify_pair();
            ledger.lock().unwrap().admit_gate = Some((compiling.clone(), finish_compile.clone()));
            let compile = spawn_infer(&serving, 600, None);
            acknowledged(&compiling).await;
            let (request, raw) = embed_request(&serving, 1);
            let deadline = tokio::time::Instant::now() + Duration::from_millis(150);
            let ready = async { Ok(Box::new(()) as ExchangePermit) };
            let waiting = serving.channel.exchange_inference(
                &request,
                Some(&raw),
                Some(deadline),
                Duration::from_secs(1),
                ready,
            );
            let (inferring, finish_inference) = notify_pair();
            let switch_holder = async {
                tokio::time::timeout(Duration::from_secs(2), gate.reached(GateSite::Inner, 1))
                    .await
                    .expect("the wait never expired");
                ledger.lock().unwrap().inference_gate =
                    Some((inferring.clone(), finish_inference.clone()));
                finish_compile.notify_one();
                // The compiling request's own inference now holds the stream.
                acknowledged(&inferring).await;
                let compiling_at_classification = serving.channel.compiling_shape();
                gate.release(GateSite::Inner);
                compiling_at_classification
            };
            let (answered, compiling_at_classification) = tokio::join!(waiting, switch_holder);
            finish_inference.notify_one();
            compile.await.unwrap().unwrap();
            assert_eq!(compiling_at_classification, None);
            assert!(
                matches!(
                    answered,
                    Err(AneResidencyError::ShapeCompiling { shape: 1024, .. })
                ),
                "{answered:?}"
            );
        }

        /// (c) A compile holds the stream through the deadline, then an
        /// inference queued ahead of the request runs, and only then is the
        /// request granted the stream (arrival order). The compile it waited
        /// behind decides, not the inference that released the stream last.
        #[tokio::test]
        async fn stream_granted_after_a_compile_and_an_inference_is_shape_compiling() {
            let ledger = SharedLedger::default();
            let serving = serving_fixture(&ledger).await;
            serving.infer(vec![vec![1]], false).await.unwrap();
            let (compiling, finish_compile) = notify_pair();
            ledger.lock().unwrap().admit_gate = Some((compiling.clone(), finish_compile.clone()));
            let compile = spawn_infer(&serving, 600, None);
            acknowledged(&compiling).await;
            let ready = || async { Ok(Box::new(()) as ExchangePermit) };
            let (inference_request, inference_raw) = embed_request(&serving, 1);
            let ahead = serving.channel.exchange_inference(
                &inference_request,
                Some(&inference_raw),
                None,
                Duration::from_secs(1),
                ready(),
            );
            tokio::pin!(ahead);
            assert!(tokio::time::timeout(Duration::ZERO, &mut ahead)
                .await
                .is_err());
            let (request, raw) = embed_request(&serving, 1);
            let deadline = tokio::time::Instant::now() + Duration::from_millis(50);
            let behind = serving.channel.exchange_inference(
                &request,
                Some(&raw),
                Some(deadline),
                Duration::from_secs(1),
                ready(),
            );
            tokio::pin!(behind);
            assert!(tokio::time::timeout(Duration::ZERO, &mut behind)
                .await
                .is_err());
            tokio::time::sleep_until(deadline + Duration::from_millis(20)).await;
            finish_compile.notify_one();
            ahead.await.unwrap();
            let answered = behind.await;
            compile.await.unwrap().unwrap();
            assert!(
                matches!(
                    answered,
                    Err(AneResidencyError::ShapeCompiling { shape: 1024, .. })
                ),
                "{answered:?}"
            );
            let received = ledger.lock().unwrap().inference_ids.clone();
            assert!(received
                .iter()
                .any(|id| Some(id.as_str()) == inference_request.req_id()));
            assert!(!received
                .iter()
                .any(|id| Some(id.as_str()) == request.req_id()));
        }

        /// An expired stream wait that began before the oldest admit still in
        /// the history cannot be classified from that history, so it counts as
        /// a compile.
        #[test]
        fn a_stream_wait_older_than_the_admit_history_counts_as_a_compile() {
            let t0 = tokio::time::Instant::now();
            let at = |ms: u64| t0 + Duration::from_millis(ms);
            let mut holder = StreamHolder::default();
            let compile = holder.begin_admit(512, at(10));
            holder.end_admit(compile, at(15));
            // A wait from 0 to 20 overlapped the 512 compile; one from 30 to
            // 40 overlapped nothing.
            assert_eq!(holder.compile_during(at(0), at(20)), Some(512));
            assert_eq!(holder.compile_during(at(30), at(40)), None);
            for index in 0..ADMIT_HISTORY as u64 {
                let later = holder.begin_admit(128, at(1_000 + index * 10));
                holder.end_admit(later, at(1_005 + index * 10));
            }
            // The 512 hold has been forgotten, yet a wait that began before it
            // ended still counts as a compile; the quiet wait stays clean.
            assert!(holder.admits.iter().all(|hold| hold.shape == 128));
            assert_eq!(holder.compile_during(at(0), at(20)), Some(512));
            assert_eq!(holder.compile_during(at(30), at(40)), None);
        }

        /// The stream is granted to a queued inference only after its deadline,
        /// in the same poll as the expired timer. It must give the stream back
        /// without even asking for an execution permit.
        #[tokio::test]
        async fn inference_granted_the_stream_after_its_deadline_never_asks_for_a_permit() {
            let ledger = SharedLedger::default();
            let serving = serving_fixture(&ledger).await;
            serving.infer(vec![vec![1]], false).await.unwrap();
            let frames_before = ledger.lock().unwrap().inference_frames;
            let (request, raw) = embed_request(&serving, 1);
            let permits_asked = Arc::new(AtomicU64::new(0));
            let held = serving.channel.stream.lock().await;
            let deadline = tokio::time::Instant::now() + Duration::from_millis(50);
            let exchange = serving.channel.exchange_inference(
                &request,
                Some(&raw),
                Some(deadline),
                Duration::from_secs(1),
                {
                    let permits_asked = permits_asked.clone();
                    async move {
                        permits_asked.fetch_add(1, Ordering::SeqCst);
                        Ok(Box::new(()) as ExchangePermit)
                    }
                },
            );
            tokio::pin!(exchange);
            // One poll queues the exchange for the stream.
            assert!(tokio::time::timeout(Duration::ZERO, &mut exchange)
                .await
                .is_err());
            tokio::time::sleep_until(deadline + Duration::from_millis(20)).await;
            // Released after the deadline: the lock goes to the queued exchange.
            drop(held);
            let answered = exchange.await;
            assert!(
                matches!(answered, Err(AneResidencyError::DeadlineExceeded)),
                "{answered:?}"
            );
            assert_eq!(permits_asked.load(Ordering::SeqCst), 0);
            assert_eq!(ledger.lock().unwrap().inference_frames, frames_before);
            assert!(!serving.channel.faulted.load(Ordering::Relaxed));
        }

        /// The stream is free, but the execution permit is granted only after
        /// the deadline, in the same poll as the expired timer. Nothing may be
        /// written.
        #[tokio::test]
        async fn inference_granted_its_permit_after_its_deadline_is_never_written() {
            let ledger = SharedLedger::default();
            let serving = serving_fixture(&ledger).await;
            serving.infer(vec![vec![1]], false).await.unwrap();
            let frames_before = ledger.lock().unwrap().inference_frames;
            let (request, raw) = embed_request(&serving, 1);
            let (grant, granted) = tokio::sync::oneshot::channel::<()>();
            let deadline = tokio::time::Instant::now() + Duration::from_millis(50);
            let exchange = serving.channel.exchange_inference(
                &request,
                Some(&raw),
                Some(deadline),
                Duration::from_secs(1),
                async move {
                    granted.await.unwrap();
                    Ok(Box::new(()) as ExchangePermit)
                },
            );
            tokio::pin!(exchange);
            // One poll takes the free stream and waits for the permit.
            assert!(tokio::time::timeout(Duration::ZERO, &mut exchange)
                .await
                .is_err());
            assert!(serving.channel.stream.try_lock().is_err());
            tokio::time::sleep_until(deadline + Duration::from_millis(20)).await;
            grant.send(()).unwrap();
            let answered = exchange.await;
            assert!(
                matches!(answered, Err(AneResidencyError::DeadlineExceeded)),
                "{answered:?}"
            );
            assert_eq!(ledger.lock().unwrap().inference_frames, frames_before);
            assert!(!serving.channel.faulted.load(Ordering::Relaxed));
        }

        /// An admit or evict that queues for the stream longer than its RPC
        /// watchdog, but whose RPC alone is short, must not fault the channel
        /// or restart the worker.
        #[tokio::test]
        async fn queued_admit_and_evict_do_not_spend_their_rpc_watchdog_waiting_for_the_stream() {
            let ledger = SharedLedger::default();
            ledger.lock().unwrap().admit_delay = Duration::from_millis(50);
            let mut channel = mock_channel("worker", &ledger).await;
            Arc::get_mut(&mut channel).unwrap().0.shape_rpc_timeout = Duration::from_millis(200);
            let worker: Arc<dyn AneShapeWorker> = channel.clone();
            let supervisor = AneResidencySupervisor::new(Default::default());
            let held = channel.0.stream.lock().await;
            let admitted = tokio::spawn({
                let supervisor = supervisor.clone();
                let worker = worker.clone();
                async move { supervisor.lease(&worker, "model", 128).await }
            });
            tokio::time::sleep(Duration::from_millis(300)).await;
            drop(held);
            let lease = tokio::time::timeout(Duration::from_secs(2), admitted)
                .await
                .unwrap()
                .unwrap();
            let admit_faulted = channel.0.faulted.load(Ordering::Relaxed);
            assert!(lease.is_ok(), "the queued admit failed");
            drop(lease);
            let held = channel.0.stream.lock().await;
            let evicted = tokio::spawn({
                let channel = channel.clone();
                async move { channel.evict_shape("model", 128).await }
            });
            tokio::time::sleep(Duration::from_millis(300)).await;
            drop(held);
            let evicted = tokio::time::timeout(Duration::from_secs(2), evicted)
                .await
                .unwrap()
                .unwrap();
            assert!(!admit_faulted);
            assert!(evicted.is_ok(), "{evicted:?}");
            assert!(!channel.0.faulted.load(Ordering::Relaxed));
            assert_eq!(supervisor.stats().restarts, 0);
            assert_eq!(ledger.lock().unwrap().connects["worker"], 1);
        }

        async fn run_rung(supervisor: &AneResidencySupervisor, worker: &Channel, tokens: usize) {
            let worker: Arc<dyn AneShapeWorker> = worker.clone();
            supervisor
                .run_by_published_rung(
                    &worker,
                    "model",
                    &[tokens],
                    &ANE_SHAPE_LADDER,
                    |_, indices| async move { Ok(vec![(); indices.len()]) },
                )
                .await
                .unwrap();
        }

        #[tokio::test]
        async fn admitting_a_larger_shape_evicts_the_smallest_rung_last() {
            let ledger = SharedLedger::default();
            let worker = mock_channel("worker", &ledger).await;
            // The mock worker charges 4 executables per shape, so a budget of
            // 12 holds three shapes.
            let supervisor = budget_supervisor(12, Duration::from_secs(1));
            for tokens in [100, 200, 400] {
                run_rung(&supervisor, &worker, tokens).await;
            }
            // Shapes 128, 256 and 512 are resident; 128 was used first, so it
            // is the least recently used.
            run_rung(&supervisor, &worker, 1000).await;
            let events = ledger.lock().unwrap().events.clone();
            assert!(events.contains(&"evict model 256".into()), "{events:?}");
            assert!(!events.contains(&"evict model 128".into()), "{events:?}");
            assert_eq!(supervisor.resident_shapes()["model"], vec![128, 512, 1024]);
        }

        #[tokio::test]
        async fn the_smallest_rung_is_evicted_when_it_is_the_only_candidate() {
            let ledger = SharedLedger::default();
            let worker = mock_channel("worker", &ledger).await;
            let shape_worker: Arc<dyn AneShapeWorker> = worker.clone();
            let supervisor = budget_supervisor(12, Duration::from_secs(1));
            for tokens in [100, 200, 400] {
                run_rung(&supervisor, &worker, tokens).await;
            }
            let held_256 = supervisor.lease(&shape_worker, "model", 256).await.unwrap();
            let held_512 = supervisor.lease(&shape_worker, "model", 512).await.unwrap();
            let lease = tokio::time::timeout(
                Duration::from_secs(2),
                supervisor.lease(&shape_worker, "model", 1024),
            )
            .await
            .unwrap()
            .unwrap();
            assert!(ledger
                .lock()
                .unwrap()
                .events
                .contains(&"evict model 128".into()));
            drop((lease, held_256, held_512));
        }

        /// A request whose shape is already resident needs no budget, so it
        /// must not queue behind a request that is waiting for another
        /// request's compile of a different shape. Exactly one compile runs
        /// for the shape both of the other requests need.
        #[tokio::test]
        async fn resident_lease_does_not_wait_behind_a_single_flight_waiter() {
            let ledger = SharedLedger::default();
            let worker: Arc<dyn AneShapeWorker> = mock_channel("worker", &ledger).await;
            let supervisor = AneResidencySupervisor::new(Default::default());
            drop(supervisor.lease(&worker, "model", 128).await.unwrap());
            let started = Arc::new(Notify::new());
            let release = Arc::new(Notify::new());
            ledger.lock().unwrap().admit_gate = Some((started.clone(), release.clone()));
            let lease_1024 = |supervisor: &AneResidencySupervisor| {
                let supervisor = supervisor.clone();
                let worker = worker.clone();
                tokio::spawn(async move { supervisor.lease(&worker, "model", 1024).await })
            };
            let compiling = lease_1024(&supervisor);
            tokio::time::timeout(Duration::from_secs(2), started.notified())
                .await
                .unwrap();
            let joining = lease_1024(&supervisor);
            tokio::time::timeout(Duration::from_secs(1), async {
                while supervisor.inner.lock().waiters.is_empty() {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            let resident = tokio::time::timeout(
                Duration::from_secs(1),
                supervisor.lease(&worker, "model", 128),
            )
            .await;
            release.notify_one();
            let compiled = compiling.await.unwrap().unwrap();
            let joined = joining.await.unwrap().unwrap();
            assert!(
                resident.is_ok_and(|lease| lease.is_ok()),
                "a resident shape must lease while another shape compiles"
            );
            assert_eq!(
                ledger.lock().unwrap().admit_attempts[&("model".into(), 1024)],
                1
            );
            drop((compiled, joined));
        }

        /// When the request at the head of the budget queue is waiting for a
        /// victim, later requests stop leasing resident shapes out of turn, so
        /// a steady stream of short leases cannot starve it.
        #[tokio::test]
        async fn a_budget_head_waiting_for_a_victim_is_not_overtaken_by_resident_leases() {
            let ledger = SharedLedger::default();
            let worker: Arc<dyn AneShapeWorker> = mock_channel("worker", &ledger).await;
            let supervisor = budget_supervisor(8, Duration::from_secs(2));
            let held_128 = supervisor.lease(&worker, "model", 128).await.unwrap();
            let held_256 = supervisor.lease(&worker, "model", 256).await.unwrap();
            let head = tokio::spawn({
                let supervisor = supervisor.clone();
                let worker = worker.clone();
                async move { supervisor.lease(&worker, "model", 512).await }
            });
            tokio::time::timeout(Duration::from_secs(1), async {
                while supervisor.inner.lock().starved_head.is_none() {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            let overtaking = tokio::spawn({
                let supervisor = supervisor.clone();
                let worker = worker.clone();
                async move { supervisor.lease(&worker, "model", 256).await }
            });
            tokio::time::sleep(Duration::from_millis(50)).await;
            assert!(
                !overtaking.is_finished(),
                "a resident lease overtook the starved budget head"
            );
            drop(held_256);
            let admitted = tokio::time::timeout(Duration::from_secs(2), head)
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            assert_eq!(admitted.shape(), 512);
            drop((admitted, held_128));
            drop(overtaking.await.unwrap().unwrap());
        }

        #[cfg(unix)]
        pub(crate) fn module_mock_engine(
            started: Arc<Notify>,
            release: Arc<Notify>,
        ) -> Arc<DirectAneEngine> {
            module_mock_engine_gate(started, release, false)
        }

        #[cfg(unix)]
        pub(crate) fn module_mock_engine_admission(
            started: Arc<Notify>,
            release: Arc<Notify>,
        ) -> Arc<DirectAneEngine> {
            module_mock_engine_gate(started, release, true)
        }

        #[cfg(unix)]
        fn module_mock_engine_gate(
            started: Arc<Notify>,
            release: Arc<Notify>,
            admission: bool,
        ) -> Arc<DirectAneEngine> {
            let ledger = SharedLedger::default();
            if admission {
                ledger.lock().unwrap().admit_gate = Some((started, release));
            } else {
                ledger.lock().unwrap().inference_gate = Some((started, release));
            }
            module_mock_engine_for(ledger)
        }

        /// Lets module tests hold the mock worker's next admission (compile)
        /// or inference open and count the compiles it performed.
        #[cfg(unix)]
        pub(crate) struct ModuleMockControl(SharedLedger);

        #[cfg(unix)]
        impl ModuleMockControl {
            #[cfg(target_os = "macos")]
            pub(crate) fn enable_token_capture(&self) {
                self.0.lock().unwrap().capture_token_ids = true;
            }

            #[cfg(target_os = "macos")]
            pub(crate) fn take_token_ids(&self) -> Vec<Vec<u32>> {
                std::mem::take(&mut self.0.lock().unwrap().captured_token_ids)
            }
            pub(crate) fn hold_next_admission(&self, started: Arc<Notify>, release: Arc<Notify>) {
                self.0.lock().unwrap().admit_gate = Some((started, release));
            }

            pub(crate) fn hold_next_inference(&self, started: Arc<Notify>, release: Arc<Notify>) {
                self.0.lock().unwrap().inference_gate = Some((started, release));
            }

            /// Compiles the mock worker has run for `shape`, on any model.
            pub(crate) fn compiles(&self, shape: usize) -> usize {
                self.0
                    .lock()
                    .unwrap()
                    .admit_attempts
                    .iter()
                    .filter(|((_, admitted), _)| *admitted == shape)
                    .map(|(_, count)| *count)
                    .sum()
            }

            /// Inference requests the mock worker has received.
            pub(crate) fn inferences(&self) -> u64 {
                self.0.lock().unwrap().inference_frames
            }

            /// Worker processes the engine has started, restarts included.
            pub(crate) fn worker_starts(&self) -> u32 {
                self.0.lock().unwrap().connects.values().sum()
            }

            pub(crate) fn loads(&self) -> usize {
                self.0.lock().unwrap().loads.len()
            }

            #[cfg(unix)]
            pub(crate) fn confirmed_exits(&self) -> u32 {
                self.0.lock().unwrap().confirmed_exits
            }

            pub(crate) fn hold_next_exit(&self, started: Arc<Notify>, release: Arc<Notify>) {
                self.0.lock().unwrap().exit_gate = Some((started, release));
            }
        }

        #[cfg(unix)]
        pub(crate) fn module_mock_engine_controlled() -> (Arc<DirectAneEngine>, ModuleMockControl) {
            let ledger = SharedLedger::default();
            (
                module_mock_engine_for(ledger.clone()),
                ModuleMockControl(ledger),
            )
        }

        #[cfg(unix)]
        fn module_mock_engine_for(ledger: SharedLedger) -> Arc<DirectAneEngine> {
            module_mock_engine_for_owner(
                ledger,
                AneResidencySupervisor::new(Default::default()),
                "module-mock",
                "gte-modernbert-base.ane-direct-worker",
            )
        }

        #[cfg(unix)]
        pub(crate) fn module_mock_engine_resident(
            supervisor: AneResidencySupervisor,
            owner: &str,
            profile: &str,
        ) -> (Arc<DirectAneEngine>, ModuleMockControl) {
            let ledger = SharedLedger::default();
            (
                module_mock_engine_for_owner(ledger.clone(), supervisor, owner, profile),
                ModuleMockControl(ledger),
            )
        }

        #[cfg(unix)]
        fn module_mock_engine_for_owner(
            ledger: SharedLedger,
            supervisor: AneResidencySupervisor,
            owner: &str,
            profile: &str,
        ) -> Arc<DirectAneEngine> {
            let runtime = Arc::new(
                tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(1)
                    .enable_all()
                    .build()
                    .unwrap(),
            );
            let owner_id = owner.to_owned();
            let connector: AneWorkerConnector<tokio::net::UnixStream> = Box::new(move || {
                let ledger = ledger.clone();
                let owner_id = owner_id.clone();
                Box::pin(async move {
                    *ledger.lock().unwrap().connects.entry(owner_id).or_default() += 1;
                    let (stream, server) = tokio::net::UnixStream::pair().unwrap();
                    let task = tokio::spawn(serve_mock(server, ledger.clone()));
                    Ok(AneWorkerSession {
                        pid: None,
                        packed: None,
                        stream,
                        confirm_exit: Box::pin(async move {
                            task.await
                                .map_err(|e| AneResidencyError::Channel(e.to_string()))?;
                            let gate = ledger.lock().unwrap().exit_gate.take();
                            if let Some((started, release)) = gate {
                                started.notify_one();
                                release.notified().await;
                            }
                            #[cfg(unix)]
                            {
                                ledger.lock().unwrap().confirmed_exits += 1;
                            }
                            Ok(())
                        }),
                    })
                })
            });
            let channel = Arc::new(
                runtime
                    .block_on(AneWorkerChannel::connect(
                        owner,
                        synapse_core::DEFAULT_MAX_FRAME_BYTES,
                        connector,
                    ))
                    .unwrap(),
            );
            let serving = Arc::new(
                AneServing::new(
                    channel,
                    supervisor,
                    AneServingMetadata {
                        model_ref: format!("ane-direct:{profile}:sha256:test"),
                        dims: 1,
                        buckets: ANE_SHAPE_LADDER.to_vec(),
                        pooling: synapse_core::WorkerPooling::Mean,
                        normalize: true,
                        timeout: Duration::from_secs(1),
                    },
                )
                .unwrap(),
            );
            Arc::new(DirectAneEngine {
                serving,
                runtime: Some(runtime),
            })
        }

        async fn serving_fixture(ledger: &SharedLedger) -> Arc<AneServing<DuplexStream>> {
            serving_fixture_for(
                ledger,
                "serving",
                AneResidencySupervisor::new(Default::default()),
            )
            .await
        }

        async fn serving_fixture_for(
            ledger: &SharedLedger,
            id: &str,
            supervisor: AneResidencySupervisor,
        ) -> Arc<AneServing<DuplexStream>> {
            let mock = mock_channel(id, ledger).await;
            let channel = Arc::new(Arc::try_unwrap(mock).ok().unwrap().0);
            let request = WorkerRequest::Load {
                req_id: channel.next_req_id("load"),
                artifact_path: "mock.safetensors".into(),
                artifact_digest: "sha256:test".into(),
                format: "safetensors-package".into(),
                runtime_config: BTreeMap::from([
                    (
                        "profile".into(),
                        "gte-modernbert-base.ane-direct-worker".into(),
                    ),
                    ("operation".into(), "embed".into()),
                ]),
            };
            let (response, _) = channel.exchange(&request, None).await.unwrap();
            let WorkerResponse::Loaded {
                model_ref,
                dims,
                buckets: Some(buckets),
                ..
            } = response
            else {
                panic!("mock LOAD failed")
            };
            Arc::new(
                AneServing::new(
                    channel,
                    supervisor,
                    AneServingMetadata {
                        model_ref,
                        dims,
                        buckets,
                        pooling: synapse_core::WorkerPooling::Mean,
                        normalize: true,
                        timeout: Duration::from_secs(1),
                    },
                )
                .unwrap(),
            )
        }

        #[tokio::test]
        async fn identical_models_in_distinct_workers_each_admit_their_own_shape() {
            let ledger = SharedLedger::default();
            let supervisor = AneResidencySupervisor::new(Default::default());
            let a = serving_fixture_for(&ledger, "A", supervisor.clone()).await;
            let b = serving_fixture_for(&ledger, "B", supervisor.clone()).await;
            assert_eq!(a.metadata.model_ref, b.metadata.model_ref);
            assert_eq!(
                a.infer(vec![vec![1]], false).await.unwrap(),
                vec![vec![128.0]]
            );
            assert_eq!(
                b.infer(vec![vec![1]], false).await.unwrap(),
                vec![vec![128.0]]
            );
            assert_eq!(
                ledger.lock().unwrap().admit_attempts[&(a.metadata.model_ref.clone(), 128)],
                2
            );
            assert_eq!(supervisor.inner.lock().reserved_executables(), 44);
            a.retire().await.unwrap();
            assert_eq!(supervisor.inner.lock().reserved_executables(), 22);
            assert_eq!(
                b.infer(vec![vec![1]], false).await.unwrap(),
                vec![vec![128.0]]
            );
            b.retire().await.unwrap();
        }

        #[tokio::test]
        async fn serving_embed_and_composed_rerank_admit_before_exchange_and_preserve_order() {
            let ledger = SharedLedger::default();
            let serving = serving_fixture(&ledger).await;
            let sequences = vec![vec![1; 300], vec![2; 1], vec![3; 200], vec![4; 8192]];
            let expected = vec![vec![512.0], vec![128.0], vec![256.0], vec![8192.0]];
            assert_eq!(
                serving.infer(sequences.clone(), false).await.unwrap(),
                expected
            );
            assert_eq!(serving.infer(sequences, true).await.unwrap(), expected);
            let before = serving.channel.request_count();
            assert!(matches!(
                serving.infer(vec![vec![1], vec![1; 8193]], false).await,
                Err(AneResidencyError::SequenceTooLong { .. })
            ));
            assert_eq!(serving.channel.request_count(), before);
            assert_eq!(ledger.lock().unwrap().shape_not_admitted, 0);
            serving.retire().await.unwrap();
        }

        #[tokio::test]
        async fn stale_serving_fault_after_recovery_does_not_restart_restored_owner() {
            let ledger = SharedLedger::default();
            let serving = serving_fixture(&ledger).await;
            let worker: Arc<dyn AneShapeWorker> = serving.channel.clone();
            let fault_generation = worker.generation();
            assert!(serving.supervisor.recover(&worker).await.is_ok());
            assert!(serving
                .supervisor
                .recover_at_generation(&worker, fault_generation)
                .await
                .is_ok());
            assert_eq!(serving.supervisor.stats().restarts, 1);
            assert_eq!(ledger.lock().unwrap().connects["serving"], 2);
            serving.retire().await.unwrap();
        }

        #[tokio::test]
        async fn serving_admission_channel_fault_restarts_exactly_once() {
            let ledger = SharedLedger::default();
            let mut serving = serving_fixture(&ledger).await;
            Arc::get_mut(&mut Arc::get_mut(&mut serving).unwrap().channel)
                .unwrap()
                .shape_rpc_timeout = Duration::from_millis(20);
            ledger
                .lock()
                .unwrap()
                .no_reply_admit
                .insert((serving.metadata.model_ref.clone(), 128));
            assert!(serving.infer(vec![vec![1]], false).await.is_err());
            assert_eq!(serving.supervisor.stats().restarts, 1);
            assert_eq!(ledger.lock().unwrap().connects["serving"], 2);
        }

        #[tokio::test]
        async fn serving_caller_cancellation_drains_reply_without_restart() {
            let ledger = SharedLedger::default();
            let serving = serving_fixture(&ledger).await;
            let started = Arc::new(Notify::new());
            let release = Arc::new(Notify::new());
            ledger.lock().unwrap().inference_gate = Some((started.clone(), release.clone()));
            let task = tokio::spawn({
                let serving = serving.clone();
                async move { serving.infer(vec![vec![1]], false).await }
            });
            tokio::time::timeout(Duration::from_secs(2), started.notified())
                .await
                .expect("inference must start before caller cancellation");
            task.abort();
            assert!(task.await.unwrap_err().is_cancelled());
            release.notify_one();
            assert_eq!(
                tokio::time::timeout(Duration::from_secs(2), serving.infer(vec![vec![2]], false))
                    .await
                    .expect("cancelled caller must not strand subsequent inference")
                    .expect("caller cancellation must drain the reply without restarting"),
                vec![vec![128.0]]
            );
            // Cancelling the caller must neither break the next request nor
            // restart the worker: the detached task drains the reply instead.
            assert_eq!(
                serving.supervisor.stats().restarts,
                0,
                "caller cancellation must drain the reply without restarting the worker"
            );
            assert_eq!(serving.supervisor.stats().admitted, 1);
            assert_eq!(ledger.lock().unwrap().connects["serving"], 1);
        }

        /// A serving caller that goes away while its compile is reserved but
        /// still waits for the worker stream leaves nothing on the wire: its
        /// serving task stops waiting, and the scheduler withdraws the compile
        /// because nobody else waits for it and nothing was promised.
        #[tokio::test]
        async fn cancelled_serving_owner_withdraws_its_undispatched_compile() {
            let ledger = SharedLedger::default();
            let serving = serving_fixture(&ledger).await;
            let busy = serving.channel.stream.lock().await;
            let cancelled = tokio::time::timeout(
                Duration::from_millis(200),
                serving.infer(vec![vec![1]], false),
            )
            .await;
            assert!(cancelled.is_err(), "the caller was meant to give up first");
            let key = ShapeKey {
                worker_id: "serving".into(),
                model_ref: serving.metadata.model_ref.clone(),
                shape: 128,
                program: AneProgram::Single,
            };
            tokio::time::timeout(Duration::from_secs(2), async {
                while serving.supervisor.inner.lock().slots.contains_key(&key) {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            })
            .await
            .expect("the cancelled owner's undispatched compile was not withdrawn");
            drop(busy);
            tokio::time::sleep(Duration::from_millis(200)).await;
            assert!(
                ledger.lock().unwrap().admit_attempts.is_empty(),
                "a compile nobody waited for reached the worker"
            );
            assert_eq!(serving.supervisor.inner.lock().reserved_executables(), 0);
        }

        #[tokio::test]
        async fn retirement_between_precheck_and_grant_leaves_no_phantom_reservation() {
            let ledger = SharedLedger::default();
            let channel = mock_channel("worker", &ledger).await;
            let worker: Arc<dyn AneShapeWorker> = channel.clone();
            let supervisor = budget_supervisor(7, Duration::from_secs(1));
            let checked = Arc::new(Notify::new());
            let continue_grant = Arc::new(Notify::new());
            *supervisor.inner.lease_decision_gate.lock().unwrap() =
                Some((checked.clone(), continue_grant.clone()));
            let pending = tokio::spawn({
                let supervisor = supervisor.clone();
                let worker = worker.clone();
                async move { supervisor.lease(&worker, "model", 128).await }
            });
            tokio::time::timeout(Duration::from_secs(2), checked.notified())
                .await
                .unwrap();
            supervisor
                .retire(&worker, channel.shutdown())
                .await
                .unwrap();
            assert_eq!(supervisor.inner.lock().reserved_executables(), 0);
            continue_grant.notify_one();
            assert!(pending.await.unwrap().is_err());
            assert_eq!(
                supervisor.inner.lock().reserved_executables(),
                0,
                "retired owner must not acquire an Admitting reservation"
            );
            assert!(supervisor.inner.lock().slots.is_empty());
            assert!(ledger.lock().unwrap().admit_attempts.is_empty());
        }

        #[tokio::test]
        async fn retirement_refuses_queued_leases_without_waiting_for_admission_deadline() {
            let ledger = SharedLedger::default();
            let serving = serving_fixture(&ledger).await;
            let worker: Arc<dyn AneShapeWorker> = serving.channel.clone();
            let mut held = vec![];
            for rung in [128, 256, 512, 1024] {
                held.push(
                    serving
                        .supervisor
                        .lease(&worker, &serving.metadata.model_ref, rung)
                        .await
                        .unwrap(),
                );
            }
            let pending = serving
                .supervisor
                .lease(&worker, &serving.metadata.model_ref, 2048);
            tokio::pin!(pending);
            assert!(
                std::future::poll_fn(|cx| std::task::Poll::Ready(pending.as_mut().poll(cx)))
                    .await
                    .is_pending()
            );
            let retire = serving.retire();
            tokio::pin!(retire);
            assert!(
                std::future::poll_fn(|cx| std::task::Poll::Ready(retire.as_mut().poll(cx)))
                    .await
                    .is_pending()
            );
            drop(held);
            retire.await.unwrap();
            assert!(tokio::time::timeout(Duration::from_millis(50), pending)
                .await
                .unwrap()
                .is_err());
            assert_eq!(serving.supervisor.stats().restarts, 0);
        }

        #[tokio::test]
        async fn serving_io_fault_recovers_and_retire_refunds_only_after_exit() {
            let ledger = SharedLedger::default();
            let serving = serving_fixture(&ledger).await;
            ledger.lock().unwrap().fail_inference = true;
            assert!(serving.infer(vec![vec![1]], false).await.is_err());
            assert_eq!(serving.supervisor.stats().restarts, 1);
            assert_eq!(ledger.lock().unwrap().connects["serving"], 2);
            assert_eq!(
                serving.infer(vec![vec![1]], false).await.unwrap(),
                vec![vec![128.0]]
            );
            serving.retire().await.unwrap();
            let ledger = SharedLedger::default();
            let started = Arc::new(Notify::new());
            let release = Arc::new(Notify::new());
            ledger.lock().unwrap().exit_gate = Some((started.clone(), release.clone()));
            let serving = serving_fixture(&ledger).await;
            serving.infer(vec![vec![1]], false).await.unwrap();
            let task = tokio::spawn({
                let serving = serving.clone();
                async move { serving.retire().await }
            });
            tokio::time::timeout(Duration::from_secs(2), started.notified())
                .await
                .unwrap();
            assert!(!serving.supervisor.resident_shapes().is_empty());
            release.notify_one();
            task.await.unwrap().unwrap();
            assert!(serving.supervisor.resident_shapes().is_empty());
        }

        async fn rejected_serving_output_is_second_fault(invalid: InvalidInference) {
            let ledger = SharedLedger::default();
            let serving = serving_fixture(&ledger).await;
            ledger.lock().unwrap().fail_inference = true;
            assert!(matches!(
                serving.infer(vec![vec![1]], false).await,
                Err(AneResidencyError::Channel(_))
            ));
            assert_eq!(serving.channel.faults.lock().unwrap().strikes, 1);
            ledger.lock().unwrap().invalid_inference = Some(invalid);
            assert!(matches!(
                serving.infer(vec![vec![1]], false).await,
                Err(AneResidencyError::Channel(_))
            ));
            assert_eq!(
                serving.channel.faults.lock().unwrap().strikes,
                2,
                "a correctly framed but rejected serving result must strike, never clear history"
            );
            assert_eq!(
                ledger.lock().unwrap().connects["serving"],
                2,
                "quarantine must not replace the second faulty generation"
            );
            assert!(matches!(
                serving.infer(vec![vec![1]], false).await,
                Err(AneResidencyError::Quarantined { .. })
            ));
            serving.retire().await.unwrap();
        }

        #[tokio::test]
        async fn ane_disconnect_then_nan_result_quarantines() {
            rejected_serving_output_is_second_fault(InvalidInference::Nonfinite).await;
        }

        #[tokio::test]
        async fn ane_disconnect_then_dimension_mismatch_quarantines() {
            rejected_serving_output_is_second_fault(InvalidInference::Dimensions).await;
        }

        #[tokio::test]
        async fn published_ladder_controls_admission_and_rejects_oversize_before_exchange() {
            let ledger = SharedLedger::default();
            let worker: Arc<dyn AneShapeWorker> = mock_channel("worker", &ledger).await;
            let supervisor = budget_supervisor(8, Duration::from_secs(1));
            let out = supervisor
                .run_by_published_rung(
                    &worker,
                    "model",
                    &[1, 300, 200],
                    &[128, 512, 8192],
                    |rung, indices| async move {
                        Ok(indices.into_iter().map(|index| (index, rung)).collect())
                    },
                )
                .await
                .unwrap();
            assert_eq!(out, vec![(0, 128), (1, 512), (2, 512)]);
            let before = ledger.lock().unwrap().events.clone();
            assert!(matches!(
                supervisor
                    .run_by_published_rung(
                        &worker,
                        "model",
                        &[8193],
                        &[128, 512, 8192],
                        |_, _| async { Ok::<Vec<usize>, AneResidencyError>(vec![]) }
                    )
                    .await,
                Err(AneResidencyError::SequenceTooLong { .. })
            ));
            assert_eq!(ledger.lock().unwrap().events, before);
        }

        #[tokio::test]
        async fn observations_off_retains_no_admitted_events() {
            let ledger = SharedLedger::default();
            let worker: Arc<dyn AneShapeWorker> = mock_channel("worker", &ledger).await;
            let supervisor = budget_supervisor(4, Duration::from_secs(1));
            drop(supervisor.lease(&worker, "model", 128).await.unwrap());
            assert_eq!(supervisor.stats().admitted, 1);
            assert!(supervisor.observations().is_none());
            assert!(supervisor.inner.lock().inventory_sink.is_none());
        }

        #[tokio::test]
        async fn observations_survive_eviction_restart_and_re_admission() {
            let ledger = SharedLedger::default();
            let channel = mock_channel("worker", &ledger).await;
            let worker: Arc<dyn AneShapeWorker> = channel.clone();
            let mut supervisor = budget_supervisor(4, Duration::from_secs(1));
            supervisor.enable_observations();
            assert_eq!(channel.request_count(), 0);
            drop(supervisor.lease(&worker, "model", 128).await.unwrap());
            assert_eq!(channel.request_count(), 1);
            drop(supervisor.lease(&worker, "model", 256).await.unwrap());
            assert_eq!(channel.request_count(), 3);
            assert!(recover_worker(&supervisor.inner, &worker).await);
            assert!(supervisor.resident_shapes().is_empty());
            drop(supervisor.lease(&worker, "model", 256).await.unwrap());
            assert_eq!(channel.request_count(), 4);
            let (inventories, admitted) = supervisor.observations().unwrap();
            assert_eq!(admitted, supervisor.stats().admitted);
            assert_eq!(admitted, 3);
            assert_eq!(inventories.len(), 3);
            assert_eq!(inventories[0].executables[0].id, "model-128");
            assert_eq!(inventories[1].executables[0].id, "model-256");
            assert_eq!(inventories[2], inventories[1]);
            assert_eq!(supervisor.stats().evicted, 1);
            assert_eq!(supervisor.stats().restarts, 1);
            assert_eq!(ledger.lock().unwrap().connects["worker"], 2);
            assert_eq!(supervisor.resident_shapes()["model"], vec![256]);
        }

        #[tokio::test]
        async fn dropping_retained_inventory_refuses_certify_placement_gate() {
            struct SnapshotRunner(synapse_certify::RunEvidence);
            impl synapse_certify::Runner for SnapshotRunner {
                fn probe_floor(&mut self, _: &str, _: &str) -> synapse_certify::Result<String> {
                    Ok("ok".into())
                }
                fn observe(
                    &mut self,
                    _: &str,
                    _: &str,
                ) -> synapse_certify::Result<synapse_certify::RunEvidence> {
                    Ok(self.0.clone())
                }
            }
            let ledger = SharedLedger::default();
            let worker: Arc<dyn AneShapeWorker> = mock_channel("worker", &ledger).await;
            let mut supervisor = budget_supervisor(4, Duration::from_secs(1));
            supervisor.enable_observations();
            drop(supervisor.lease(&worker, "model", 128).await.unwrap());
            drop(supervisor.lease(&worker, "model", 256).await.unwrap());
            let evidence = |supervisor: &AneResidencySupervisor| {
                let (inventories, admitted) = supervisor.observations().unwrap();
                serde_json::from_value(serde_json::json!({
                    "source_commit": "1".repeat(40), "machine": {}, "artifacts": [],
                    "operation": "embed", "profile_id": "test", "fingerprint": "test",
                    "fixture_set_id": "test", "expected_fixtures": 1, "tolerance_class": "test",
                    "parity": null, "admission": {
                        "tokens_8192": {"outcome":"processed", "truncated":false, "diverted":false, "worker_requests":1},
                        "tokens_8193": {"outcome":"sequence_too_long", "truncated":false, "diverted":false, "worker_requests":0}
                    }, "layer_count": LAYERS, "inventories": inventories,
                    "admitted_count": admitted
                })).unwrap()
            };
            let run = |runner: &mut SnapshotRunner| {
                synapse_certify::produce(
                    runner,
                    std::path::Path::new("."),
                    "ane-m5",
                    "gte-modernbert-base",
                )
                .unwrap_err()
                .to_string()
            };
            // All admitted inventories are present, so the placement check passes.
            // This fixture intentionally has no parity output and fails that next check.
            assert!(run(&mut SnapshotRunner(evidence(&supervisor))).contains("parity gate failed"));
            supervisor
                .inner
                .lock()
                .inventory_sink
                .as_mut()
                .unwrap()
                .pop();
            assert_eq!(supervisor.stats().admitted, 2);
            assert!(run(&mut SnapshotRunner(evidence(&supervisor)))
                .contains("placement inventory failed"));
        }

        #[tokio::test]
        async fn eviction_refunds_executables_only_after_completed_ack() {
            let ledger = SharedLedger::default();
            let worker: Arc<dyn AneShapeWorker> = mock_channel("worker", &ledger).await;
            let supervisor = budget_supervisor(7, Duration::from_secs(1));
            drop(supervisor.lease(&worker, "model", 128).await.unwrap());
            let started = Arc::new(Notify::new());
            let release = Arc::new(Notify::new());
            ledger.lock().unwrap().evict_gate = Some((started.clone(), release.clone()));
            let task = tokio::spawn({
                let supervisor = supervisor.clone();
                let worker = worker.clone();
                async move { supervisor.lease(&worker, "model", 256).await }
            });
            tokio::time::timeout(Duration::from_secs(1), started.notified())
                .await
                .unwrap();
            assert_eq!(
                supervisor.inner.lock().reserved_executables(),
                4,
                "eviction in flight must retain its reservation"
            );
            assert!(!task.is_finished());
            assert!(ledger
                .lock()
                .unwrap()
                .resident
                .contains(&("model".into(), 128)));
            release.notify_one();
            let lease = tokio::time::timeout(Duration::from_secs(2), task)
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            assert_eq!(
                supervisor.inner.lock().reserved_executables(),
                4,
                "completed eviction must permit a new four-layer shape within budget7"
            );
            assert_eq!(supervisor.resident_shapes()["model"], vec![256]);
            let data = ledger.lock().unwrap();
            assert_eq!(data.admit_attempts[&("model".into(), 256)], 1);
            assert!(!data.resident.contains(&("model".into(), 128)));
            assert_eq!(data.connects["worker"], 1);
            drop(data);
            drop(lease);
        }

        #[tokio::test]
        async fn admission_wait_deadline_is_transient_and_cleans_queue() {
            let ledger = Arc::new(Mutex::new(Ledger::default()));
            let worker: Arc<dyn AneShapeWorker> = mock_channel("worker", &ledger).await;
            let supervisor = budget_supervisor(4, Duration::from_millis(20));
            let protected = supervisor.lease(&worker, "model", 128).await.unwrap();
            let error = match supervisor.lease(&worker, "model", 256).await {
                Err(error) => error,
                Ok(_) => panic!("leased budget must time out"),
            };
            let mapped = error.to_engine_error(synapse_core::EngineErrorStage::Load);
            assert_eq!(mapped.retry_after_ms, Some(250));
            assert!(mapped.safe_to_retry_same_request);
            assert!(supervisor.inner.lock().waiters.is_empty());
            assert_eq!(supervisor.resident_shapes()["model"], vec![128]);
            assert_eq!(supervisor.stats().restarts, 0);
            drop(protected);
            drop(supervisor.lease(&worker, "model", 256).await.unwrap());
        }

        #[tokio::test]
        async fn pinned_layer_counts_charge_gte_and_qwen_before_admission() {
            let ledger = Arc::new(Mutex::new(Ledger::default()));
            let channel = mock_channel("worker", &ledger).await;
            for (slug, count) in [
                ("gte-modernbert-base", 22),
                ("gte-reranker-modernbert-base", 22),
                ("qwen3-embedding-0.6b", 28),
                ("qwen3-reranker-0.6b", 28),
            ] {
                assert_eq!(
                    channel
                        .0
                        .executable_count(&format!(
                            "ane-direct:{slug}.ane-direct-worker:sha256:placeholder"
                        ))
                        .unwrap(),
                    count
                );
            }
            assert_eq!(
                channel.0.executable_count("unknown").unwrap_err().code(),
                Some("model_unsupported")
            );
            assert!(ledger.lock().unwrap().admit_attempts.is_empty());
        }

        #[tokio::test]
        async fn resource_exhaustion_evicts_cross_worker_lru_and_retries_once_without_restart() {
            let ledger = Arc::new(Mutex::new(Ledger::default()));
            let a = mock_channel("a", &ledger).await;
            let b = mock_channel("b", &ledger).await;
            let supervisor = AneResidencySupervisor::new(AneResidencyLimits::default());
            let wa: Arc<dyn AneShapeWorker> = a;
            let wb: Arc<dyn AneShapeWorker> = b;
            drop(supervisor.lease(&wa, "old", 128).await.unwrap());
            drop(supervisor.lease(&wb, "new", 256).await.unwrap());
            ledger
                .lock()
                .unwrap()
                .exhaust_admit
                .insert(("new".into(), 512), 1);
            let lease = supervisor.lease(&wb, "new", 512).await.unwrap();
            let data = ledger.lock().unwrap();
            assert_eq!(data.admit_attempts[&("new".into(), 512)], 2);
            assert_eq!(
                data.events
                    .iter()
                    .filter(|event| event.starts_with("evict "))
                    .cloned()
                    .collect::<Vec<_>>(),
                vec!["evict old 128"]
            );
            assert_eq!(data.connects["a"], 1);
            assert_eq!(data.connects["b"], 1);
            assert_eq!(supervisor.stats().restarts, 0);
            assert!(data.max_per_model <= 4 && data.max_total <= 8);
            drop(data);
            drop(lease);
        }

        #[tokio::test]
        async fn exit_confirmation_timeout_retains_reservations_and_never_spawns_replacement() {
            let ledger = SharedLedger::default();
            ledger.lock().unwrap().exit_gate =
                Some((Arc::new(Notify::new()), Arc::new(Notify::new())));
            let mut channel = mock_channel("worker", &ledger).await;
            Arc::get_mut(&mut channel).unwrap().0.owner_exit_timeout = Duration::from_millis(20);
            let worker: Arc<dyn AneShapeWorker> = channel.clone();
            let supervisor = budget_supervisor(4, Duration::from_millis(20));
            let held = supervisor.lease(&worker, "model", 128).await.unwrap();
            assert!(!tokio::time::timeout(
                Duration::from_millis(200),
                recover_worker(&supervisor.inner, &worker)
            )
            .await
            .expect("unconfirmed exit must produce a bounded failure"));
            assert!(channel.exit_unconfirmed.load(Ordering::Relaxed));
            assert_eq!(supervisor.inner.lock().reserved_executables(), 4);
            assert_eq!(ledger.lock().unwrap().connects["worker"], 1);
            assert!(supervisor.lease(&worker, "model", 256).await.is_err());
            drop(held);
        }

        #[tokio::test]
        async fn unconfirmed_exit_keeps_worker_closed_and_budget_charged() {
            let ledger = SharedLedger::default();
            ledger.lock().unwrap().fail_confirm_exit = true;
            let channel = mock_channel("worker", &ledger).await;
            let worker: Arc<dyn AneShapeWorker> = channel.clone();
            let supervisor = budget_supervisor(8, Duration::from_millis(20));
            let held = supervisor.lease(&worker, "model", 128).await.unwrap();
            assert!(!recover_worker(&supervisor.inner, &worker).await);
            assert!(channel.exit_unconfirmed.load(Ordering::Relaxed));
            assert!(supervisor.inner.lock().recovering.contains("worker"));
            assert_eq!(supervisor.inner.lock().reserved_executables(), 4);
            assert_eq!(ledger.lock().unwrap().connects["worker"], 1);
            drop(held);
        }

        async fn unacknowledged_shape_operation(execute_eviction: bool) {
            let ledger = SharedLedger::default();
            let started = Arc::new(Notify::new());
            let confirm = Arc::new(Notify::new());
            ledger.lock().unwrap().exit_gate = Some((started.clone(), confirm.clone()));
            let mut channel = mock_channel("worker", &ledger).await;
            Arc::get_mut(&mut channel).unwrap().0.shape_rpc_timeout = Duration::from_millis(20);
            let worker: Arc<dyn AneShapeWorker> = channel.clone();
            let supervisor = budget_supervisor(4, Duration::from_secs(3));
            let shape = if execute_eviction {
                drop(supervisor.lease(&worker, "model", 128).await.unwrap());
                ledger
                    .lock()
                    .unwrap()
                    .no_reply_evict
                    .insert(("model".into(), 128));
                256
            } else {
                ledger
                    .lock()
                    .unwrap()
                    .no_reply_admit
                    .insert(("model".into(), 128));
                128
            };
            let waiting_supervisor = supervisor.clone();
            let waiting_worker = worker.clone();
            let task = tokio::spawn(async move {
                waiting_supervisor
                    .lease(&waiting_worker, "model", shape)
                    .await
            });
            tokio::time::timeout(Duration::from_millis(200), started.notified())
                .await
                .expect(
                    "unacknowledged shape RPC must time out and request owner exit confirmation",
                );
            assert!(channel.faulted.load(Ordering::Relaxed));
            assert_eq!(
                supervisor.inner.lock().reserved_executables(),
                4,
                "timeout alone must not refund capacity"
            );
            assert_eq!(ledger.lock().unwrap().connects["worker"], 1);
            confirm.notify_one();
            let result = task.await.unwrap();
            if execute_eviction {
                drop(result.expect("confirmed eviction recovery may resume the waiting admission"));
            } else {
                assert!(result.is_err());
            }
            // The failed request is answered at once; the worker's recovery
            // finishes on the scheduler.
            until(|| supervisor.stats().restarts == 1).await;
            drop(supervisor.lease(&worker, "model", shape).await.unwrap());
        }

        #[tokio::test]
        async fn unacknowledged_admission_times_out_without_refunding_before_exit() {
            unacknowledged_shape_operation(false).await;
        }
        #[tokio::test]
        async fn unacknowledged_eviction_times_out_without_refunding_before_exit() {
            unacknowledged_shape_operation(true).await;
        }

        #[tokio::test]
        async fn cancelled_exchange_faults_channel_before_a_later_request_reads_stale_reply() {
            let ledger = SharedLedger::default();
            let channel = mock_channel("worker", &ledger).await;
            let started = Arc::new(Notify::new());
            let release = Arc::new(Notify::new());
            ledger.lock().unwrap().evict_gate = Some((started.clone(), release.clone()));
            let cancelling = channel.clone();
            let task = tokio::spawn(async move { cancelling.evict_shape("model", 128).await });
            started.notified().await;
            task.abort();
            assert!(task.await.unwrap_err().is_cancelled());
            release.notify_one();
            let response = channel
                .exchange(
                    &WorkerRequest::Ping {
                        req_id: "later".into(),
                    },
                    None,
                )
                .await;
            assert!(response.is_err(),"cancelled exchange must fault the stream instead of returning the abandoned eviction response to Ping");
            assert!(channel.faulted.load(Ordering::Relaxed));
            channel.restart().await.unwrap();
            assert!(channel
                .exchange(
                    &WorkerRequest::Ping {
                        req_id: "fresh".into()
                    },
                    None
                )
                .await
                .is_ok());
        }

        #[tokio::test]
        async fn queued_eviction_from_old_owner_is_refused_after_restart() {
            let ledger = SharedLedger::default();
            let channel = mock_channel("worker", &ledger).await;
            let old_generation = channel.generation();
            let guard = channel.stream.lock().await;
            let mut restart = channel.restart();
            assert!(
                std::future::poll_fn(|cx| std::task::Poll::Ready(std::future::Future::poll(
                    restart.as_mut(),
                    cx
                )))
                .await
                .is_pending()
            );
            let mut eviction = channel.evict_shape("model", 128);
            assert!(
                std::future::poll_fn(|cx| std::task::Poll::Ready(std::future::Future::poll(
                    eviction.as_mut(),
                    cx
                )))
                .await
                .is_pending()
            );
            drop(guard);
            restart.await.unwrap();
            assert_eq!(ledger.lock().unwrap().connects["worker"], 2);
            assert!(
                eviction.await.is_err(),
                "an eviction queued for the retired owner must not reach its replacement"
            );
            assert!(!ledger
                .lock()
                .unwrap()
                .events
                .iter()
                .any(|event| event.starts_with("evict")));
            let supervisor = budget_supervisor(8, Duration::from_secs(1));
            let worker: Arc<dyn AneShapeWorker> = channel.clone();
            let held = supervisor.lease(&worker, "model", 128).await.unwrap();
            assert!(channel
                .evict_shape_at_generation(old_generation, "model", 128)
                .await
                .is_err());
            assert!(ledger
                .lock()
                .unwrap()
                .resident
                .contains(&("model".into(), 128)));
            assert_eq!(ledger.lock().unwrap().connects["worker"], 2);
            drop(held);
        }

        #[tokio::test]
        async fn drain_timeout_forces_confirmed_exit_and_reopens_worker() {
            let ledger = SharedLedger::default();
            let worker: Arc<dyn AneShapeWorker> = mock_channel("worker", &ledger).await;
            let supervisor = budget_supervisor(8, Duration::from_millis(20));
            let held = supervisor.lease(&worker, "model", 128).await.unwrap();
            assert!(tokio::time::timeout(Duration::from_millis(200),recover_worker(&supervisor.inner,&worker)).await.unwrap(),"bounded drain timeout must escalate to confirmed owner exit, not permanently close a live worker");
            assert_eq!(supervisor.stats().restarts, 1);
            assert!(supervisor.inner.lock().recovering.is_empty());
            assert!(supervisor.inner.lock().slots.is_empty());
            drop(supervisor.lease(&worker, "model", 256).await.unwrap());
            drop(held);
        }

        #[tokio::test]
        async fn failed_resource_victim_eviction_recovers_without_waiting_on_its_own_admission() {
            let ledger = SharedLedger::default();
            let worker: Arc<dyn AneShapeWorker> = mock_channel("worker", &ledger).await;
            let supervisor = budget_supervisor(8, Duration::from_secs(3));
            drop(supervisor.lease(&worker, "model", 128).await.unwrap());
            {
                let mut ledger = ledger.lock().unwrap();
                ledger.exhaust_admit.insert(("model".into(), 256), 1);
                ledger.fail_evict.insert(("model".into(), 128));
            }
            let refusal = tokio::time::timeout(
                Duration::from_millis(200),
                supervisor.lease(&worker, "model", 256),
            )
            .await
            .expect("recovery must not wait for the triggering admission's own slot");
            assert!(matches!(
                refusal,
                Err(AneResidencyError::ResourcesExhausted { .. })
            ));
            assert_eq!(supervisor.stats().restarts, 1);
            assert!(supervisor.inner.lock().recovering.is_empty());
            assert!(supervisor.inner.lock().slots.is_empty());
            drop(supervisor.lease(&worker, "model", 256).await.unwrap());
        }

        #[tokio::test]
        async fn persistent_exhaustion_after_eviction_restarts_and_refuses_transiently() {
            let ledger = SharedLedger::default();
            let channel = mock_channel("worker", &ledger).await;
            let load = WorkerRequest::Load {
                req_id: "load".into(),
                artifact_path: "pinned.safetensors".into(),
                artifact_digest: "sha256:pinned".into(),
                format: "safetensors-package".into(),
                runtime_config: BTreeMap::from([(
                    "profile".into(),
                    "pinned-worker-profile".into(),
                )]),
            };
            let (response, _) = channel.exchange(&load, None).await.unwrap();
            let WorkerResponse::Loaded { model_ref, .. } = response else {
                panic!("expected load");
            };
            let worker: Arc<dyn AneShapeWorker> = channel;
            let supervisor = budget_supervisor(100, Duration::from_secs(1));
            for rung in [128, 256] {
                drop(supervisor.lease(&worker, &model_ref, rung).await.unwrap());
            }
            ledger
                .lock()
                .unwrap()
                .exhaust_admit
                .insert((model_ref.clone(), 512), 2);
            let error = supervisor
                .lease(&worker, &model_ref, 512)
                .await
                .err()
                .unwrap();
            let refusal = error.to_engine_error(synapse_core::EngineErrorStage::Load);
            assert_eq!(refusal.retry_after_ms, Some(ANE_RESOURCES_RETRY_AFTER_MS));
            assert!(refusal.safe_to_retry_same_request);
            {
                let data = ledger.lock().unwrap();
                assert_eq!(data.admit_attempts[&(model_ref.clone(), 512)], 2);
                assert_eq!(
                    data.events
                        .iter()
                        .filter(|event| event.starts_with("evict "))
                        .count(),
                    1
                );
                assert_eq!(data.connects["worker"], 2);
                assert_eq!(
                    data.loads.len(),
                    2,
                    "persistent exhaustion must restore pinned models after confirmed owner exit"
                );
                assert!(data.resident.is_empty());
                assert!(supervisor.resident_shapes().is_empty());
                assert_eq!(supervisor.inner.lock().reserved_executables(), 0);
                assert_eq!(supervisor.stats().restarts, 1);
            }
            drop(supervisor.lease(&worker, &model_ref, 512).await.unwrap());
        }

        #[tokio::test]
        async fn persistent_exhaustion_drains_existing_leases_before_restart() {
            let ledger = SharedLedger::default();
            let worker: Arc<dyn AneShapeWorker> = mock_channel("worker", &ledger).await;
            let supervisor = budget_supervisor(100, Duration::from_secs(2));
            let held = supervisor.lease(&worker, "model", 128).await.unwrap();
            drop(supervisor.lease(&worker, "model", 256).await.unwrap());
            ledger
                .lock()
                .unwrap()
                .exhaust_admit
                .insert(("model".into(), 512), 2);
            let task = tokio::spawn({
                let worker = worker.clone();
                let supervisor = supervisor.clone();
                async move { supervisor.lease(&worker, "model", 512).await }
            });
            tokio::time::timeout(Duration::from_secs(1), async {
                loop {
                    if supervisor.inner.lock().recovering.contains("worker") {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            assert_eq!(
                ledger.lock().unwrap().connects["worker"],
                1,
                "restart must not interrupt an existing lease"
            );
            assert_eq!(
                supervisor.inner.lock().reserved_executables(),
                8,
                "retain held and failed-admission reservations during recovery"
            );
            assert!(ledger
                .lock()
                .unwrap()
                .resident
                .contains(&("model".into(), 128)));
            assert_eq!(
                supervisor.inner.lock().slots[&ShapeKey {
                    worker_id: worker.worker_id().to_owned(),
                    model_ref: "model".into(),
                    shape: 512,
                    program: AneProgram::Single,
                }]
                    .state,
                // The worker refused the retry outright: nothing is compiling,
                // but the reservation stays charged until recovery.
                SlotState::Rejected
            );
            drop(held);
            assert!(tokio::time::timeout(Duration::from_secs(1), task)
                .await
                .unwrap()
                .unwrap()
                .is_err());
            // The refused request was answered at once; the restart follows
            // once the held lease drains.
            until(|| ledger.lock().unwrap().connects["worker"] == 2).await;
            until(|| supervisor.inner.lock().reserved_executables() == 0).await;
        }

        #[tokio::test]
        async fn resource_exhaustion_never_evicts_a_leased_shape() {
            let ledger = Arc::new(Mutex::new(Ledger::default()));
            let channel = mock_channel("worker", &ledger).await;
            let worker: Arc<dyn AneShapeWorker> = channel;
            let supervisor = AneResidencySupervisor::new(AneResidencyLimits::default());
            let protected = supervisor.lease(&worker, "model", 128).await.unwrap();
            ledger
                .lock()
                .unwrap()
                .exhaust_admit
                .insert(("model".into(), 512), 1);
            let error = match supervisor.lease(&worker, "model", 512).await {
                Err(error) => error,
                Ok(_) => panic!("leased-only lane should refuse instead of evicting"),
            };
            assert_eq!(
                error
                    .to_engine_error(synapse_core::EngineErrorStage::Load)
                    .retry_after_ms,
                Some(250)
            );
            let data = ledger.lock().unwrap();
            assert_eq!(data.admit_attempts[&("model".into(), 512)], 1);
            assert!(data.resident.contains(&("model".into(), 128)));
            assert!(!data.events.iter().any(|e| e.starts_with("evict ")));
            assert_eq!(supervisor.stats().restarts, 0);
            drop(data);
            drop(protected);
        }

        #[test]
        fn typed_worker_resource_error_has_public_retry_delay() {
            let error = super::super::WorkerHostError::WorkerErr {
                code: ERR_ANE_RESOURCES_EXHAUSTED.into(),
                msg: "pressure".into(),
            };
            let mapped = error.to_engine_error(synapse_core::EngineErrorStage::Load);
            assert_eq!(mapped.retry_after_ms, Some(250));
            assert!(mapped.safe_to_retry_same_request);
            let other = super::super::WorkerHostError::WorkerErr {
                code: "compile_failed".into(),
                msg: "bad graph".into(),
            };
            assert_eq!(
                other
                    .to_engine_error(synapse_core::EngineErrorStage::Load)
                    .retry_after_ms,
                None
            );
        }

        async fn mock_channel(worker_id: &str, ledger: &SharedLedger) -> Channel {
            let connect_id = worker_id.to_string();
            let connect_ledger = ledger.clone();
            let connect: AneWorkerConnector<DuplexStream> = Box::new(move || {
                let worker_id = connect_id.clone();
                let ledger = connect_ledger.clone();
                Box::pin(async move {
                    let (host, worker) = tokio::io::duplex(1 << 16);
                    *ledger
                        .lock()
                        .unwrap()
                        .connects
                        .entry(worker_id)
                        .or_default() += 1;
                    let fail_exit = ledger.lock().unwrap().fail_confirm_exit;
                    let exit_gate = ledger.lock().unwrap().exit_gate.clone();
                    let task = tokio::spawn(serve_mock(worker, ledger));
                    Ok(AneWorkerSession {
                        pid: None,
                        packed: None,
                        stream: host,
                        confirm_exit: Box::pin(async move {
                            task.await.map_err(|error| {
                                AneResidencyError::Channel(format!("mock owner exit: {error}"))
                            })?;
                            if let Some((started, release)) = exit_gate {
                                started.notify_one();
                                release.notified().await;
                            }
                            if fail_exit {
                                Err(AneResidencyError::Channel(
                                    "mock owner exit unconfirmed".into(),
                                ))
                            } else {
                                Ok(())
                            }
                        }),
                    })
                })
            });
            Arc::new(MockChannel(
                AneWorkerChannel::connect(worker_id, DEFAULT_MAX_FRAME_BYTES, connect)
                    .await
                    .unwrap(),
            ))
        }

        /// One embed request over sequences of `lengths` tokens, run rung by
        /// rung. The mock answers each sequence with its rung, so the result
        /// also shows that outputs come back in sequence order.
        async fn embed(
            supervisor: AneResidencySupervisor,
            channel: Channel,
            model_ref: String,
            lengths: Vec<usize>,
            ledger: SharedLedger,
        ) -> Result<Vec<f32>, AneResidencyError> {
            let worker: Arc<dyn AneShapeWorker> = channel.clone();
            let lengths_ref = &lengths;
            supervisor
                .run_by_rung(&worker, &model_ref, &lengths, |rung, indices| {
                    let channel = channel.clone();
                    let ledger = ledger.clone();
                    let model_ref = model_ref.clone();
                    async move {
                        let key = (model_ref.clone(), rung);
                        *ledger
                            .lock()
                            .unwrap()
                            .in_flight
                            .entry(key.clone())
                            .or_default() += 1;
                        // Keep the lease a little while so leases overlap
                        // with other requests' admissions and evictions.
                        tokio::time::sleep(Duration::from_millis(1)).await;
                        let items: Vec<WorkerTokenItem> = indices
                            .iter()
                            .map(|index| WorkerTokenItem {
                                id: index.to_string(),
                                n_tokens: lengths_ref[*index],
                            })
                            .collect();
                        let total: usize = items.iter().map(|item| item.n_tokens).sum();
                        let request = WorkerRequest::EmbedBatch {
                            req_id: channel.next_req_id("embed"),
                            model_ref,
                            pooling: WorkerPooling::Cls,
                            normalize: true,
                            items,
                        };
                        let result = channel
                            .exchange(&request, Some(&encode_i32_frame(&vec![7; total])))
                            .await;
                        *ledger.lock().unwrap().in_flight.get_mut(&key).unwrap() -= 1;
                        match result? {
                            (WorkerResponse::Vectors { .. }, Some(raw)) => {
                                Ok(decode_f32_frame(&raw).unwrap())
                            }
                            (WorkerResponse::Err { code, msg, .. }, _) => {
                                Err(AneResidencyError::WorkerErr { code, msg })
                            }
                            (other, _) => Err(AneResidencyError::Channel(format!("{other:?}"))),
                        }
                    }
                })
                .await
        }

        fn expected_rungs(lengths: &[usize]) -> Vec<f32> {
            lengths
                .iter()
                .map(|length| ladder_rung(*length).unwrap() as f32)
                .collect()
        }

        #[test]
        fn ladder_rung_is_the_smallest_shape_that_fits() {
            assert_eq!(ladder_rung(1), Some(128));
            assert_eq!(ladder_rung(128), Some(128));
            assert_eq!(ladder_rung(129), Some(256));
            assert_eq!(ladder_rung(8192), Some(8192));
            assert_eq!(ladder_rung(8193), None);
        }

        /// Every request is submitted at once: each of the 7 shapes for each
        /// of 4 models, eight two-rung embeds (rungs 128 and 256) and one
        /// rerank-sized pool spanning rungs 128, 512 and 2048.
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn eight_concurrent_two_rung_requests_complete_within_the_budget() {
            let ledger = SharedLedger::default();
            let supervisor = AneResidencySupervisor::new(AneResidencyLimits::default());
            let models = ["gte-embed", "gte-rerank", "qwen-embed", "qwen-rerank"];
            let mut channels = HashMap::new();
            for model in models {
                channels.insert(
                    model,
                    mock_channel(&format!("worker-{model}"), &ledger).await,
                );
            }
            let mut requests: Vec<(&str, Vec<usize>)> = Vec::new();
            for model in models {
                for shape in ANE_SHAPE_LADDER {
                    requests.push((model, vec![shape]));
                }
            }
            for _ in 0..8 {
                requests.push(("gte-embed", vec![100, 200, 90, 250]));
            }
            requests.push(("gte-rerank", vec![100, 400, 1500, 120, 2000]));
            let mut tasks = Vec::new();
            for (model, lengths) in requests {
                let task = tokio::spawn(embed(
                    supervisor.clone(),
                    channels[model].clone(),
                    model.to_string(),
                    lengths.clone(),
                    ledger.clone(),
                ));
                tasks.push((task, lengths));
            }
            for (task, lengths) in tasks {
                let output = tokio::time::timeout(Duration::from_secs(60), task)
                    .await
                    .expect("no request waits forever")
                    .unwrap()
                    .expect("request completes");
                assert_eq!(output, expected_rungs(&lengths));
            }

            let ledger = ledger.lock().unwrap();
            assert_eq!(ledger.shape_not_admitted, 0);
            assert_eq!(ledger.leased_evicts, 0, "a leased shape was evicted");
            assert!(ledger.max_per_model <= ANE_RESIDENT_SHAPES_PER_MODEL);
            assert!(ledger.max_total <= ANE_RESIDENT_SHAPES_TOTAL);
            // The workload must actually press on the budget for the bounds
            // above to mean anything.
            assert_eq!(ledger.max_total, ANE_RESIDENT_SHAPES_TOTAL);
            assert!(ledger.events.iter().any(|event| event.starts_with("evict")));
            let stats = supervisor.stats();
            assert!(stats.max_resident_per_model <= ANE_RESIDENT_SHAPES_PER_MODEL);
            assert!(stats.max_resident_total <= ANE_RESIDENT_SHAPES_TOTAL);
            assert_eq!(stats.samples, stats.admitted + stats.evicted);
            assert_eq!(stats.restarts, 0);
        }

        #[tokio::test]
        async fn waiters_are_served_fifo_and_only_the_lru_unleased_shape_is_evicted() {
            let ledger = SharedLedger::default();
            let supervisor = AneResidencySupervisor::new(AneResidencyLimits {
                per_model: 8,
                total: 3,
            });
            let channel = mock_channel("worker-m", &ledger).await;
            let worker: Arc<dyn AneShapeWorker> = channel.clone();
            for shape in [128, 256, 512] {
                drop(supervisor.lease(&worker, "m", shape).await.unwrap());
            }
            // 128 is now the most recently used shape, and leased.
            let held = supervisor.lease(&worker, "m", 128).await.unwrap();
            assert_eq!(
                held.inventory().unwrap().executables[0].layers,
                vec![0, 1, 2, 3]
            );
            drop(supervisor.lease(&worker, "m", 1024).await.unwrap());
            assert_eq!(
                ledger.lock().unwrap().events.last().map(String::as_str),
                Some("admit m 1024")
            );
            assert!(ledger
                .lock()
                .unwrap()
                .events
                .contains(&"evict m 256".to_string()));
            assert!(!ledger
                .lock()
                .unwrap()
                .events
                .contains(&"evict m 128".to_string()));

            // Fill the budget with leased shapes, then queue three requests.
            let held_512 = supervisor.lease(&worker, "m", 512).await.unwrap();
            let held_1024 = supervisor.lease(&worker, "m", 1024).await.unwrap();
            let order = Arc::new(Mutex::new(Vec::new()));
            let mut waiters = Vec::new();
            for shape in [2048, 4096, 8192] {
                let supervisor = supervisor.clone();
                let worker = worker.clone();
                let order = order.clone();
                waiters.push(tokio::spawn(async move {
                    let lease = supervisor.lease(&worker, "m", shape).await.unwrap();
                    order.lock().unwrap().push(shape);
                    drop(lease);
                }));
                // Let this request reach the queue before the next one.
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            assert!(
                order.lock().unwrap().is_empty(),
                "no slot is free while all are leased"
            );
            drop(held);
            drop(held_512);
            drop(held_1024);
            for waiter in waiters {
                waiter.await.unwrap();
            }
            assert_eq!(*order.lock().unwrap(), vec![2048, 4096, 8192]);
            assert_eq!(ledger.lock().unwrap().leased_evicts, 0);
        }

        #[tokio::test]
        async fn err_ack_restarts_the_worker_drops_its_shapes_and_resumes_waiters() {
            let ledger = SharedLedger::default();
            let supervisor = AneResidencySupervisor::new(AneResidencyLimits {
                per_model: 4,
                total: 2,
            });
            let channel_a = mock_channel("worker-a", &ledger).await;
            let channel_b = mock_channel("worker-b", &ledger).await;
            let worker_a: Arc<dyn AneShapeWorker> = channel_a.clone();
            let worker_b: Arc<dyn AneShapeWorker> = channel_b.clone();
            drop(supervisor.lease(&worker_a, "a", 128).await.unwrap());
            let held_b = supervisor.lease(&worker_b, "b", 128).await.unwrap();
            {
                let mut ledger = ledger.lock().unwrap();
                ledger.fail_admit.insert(("a".to_string(), 256));
                ledger.admit_delay = Duration::from_millis(50);
            }

            let failing = {
                let supervisor = supervisor.clone();
                let worker_a = worker_a.clone();
                tokio::spawn(async move { supervisor.lease(&worker_a, "a", 256).await.map(drop) })
            };
            tokio::time::sleep(Duration::from_millis(10)).await;
            // Queued behind the failing admission with the budget full.
            let waiting = {
                let supervisor = supervisor.clone();
                let worker_b = worker_b.clone();
                tokio::spawn(async move { supervisor.lease(&worker_b, "b", 512).await.map(drop) })
            };

            let error = failing
                .await
                .unwrap()
                .expect_err("the injected ERR reaches the request");
            assert_eq!(error.code(), Some("compile_failed"));
            waiting
                .await
                .unwrap()
                .expect("the queued request resumes after the restart");

            assert_eq!(
                ledger.lock().unwrap().connects["worker-a"],
                2,
                "worker a restarted"
            );
            assert_eq!(ledger.lock().unwrap().connects["worker-b"], 1);
            let resident = supervisor.resident_shapes();
            assert!(
                !resident.contains_key("a"),
                "worker a's shapes are dropped: {resident:?}"
            );
            assert_eq!(resident["b"], vec![128, 512]);
            assert_eq!(supervisor.stats().restarts, 1);
            drop(held_b);
            // The restarted worker admits again.
            drop(supervisor.lease(&worker_a, "a", 256).await.unwrap());
        }

        #[cfg(unix)]
        #[test]
        fn second_holder_is_busy_until_the_holder_and_its_workers_exit() {
            const CHILD: &str = "SYNAPSE_ANE_LANE_LOCK_TEST_CHILD";
            const TEST: &str = "worker_host::ane_residency::tests::second_holder_is_busy_until_the_holder_and_its_workers_exit";
            if std::env::var_os(CHILD).is_none() {
                // A sibling test can fork with our locked description open. CLOEXEC
                // closes that copy only at exec, not when our worker is reaped.
                // Run the lifetime assertion without other test threads spawning.
                let output = std::process::Command::new(std::env::current_exe().unwrap())
                    .args([TEST, "--exact", "--nocapture", "--test-threads=1"])
                    .env(CHILD, "1")
                    .output()
                    .unwrap();
                let stdout = String::from_utf8_lossy(&output.stdout);
                assert!(
                    output.status.success() && stdout.contains(&format!("test {TEST} ... ok")),
                    "isolated lane-lock test did not pass: {}\n{stdout}\n{}",
                    output.status,
                    String::from_utf8_lossy(&output.stderr)
                );
                return;
            }

            let dir = std::env::temp_dir().join(format!(
                "synapse-ane-lock-{}-{}",
                std::process::id(),
                super::super::nonce_hex16()
            ));
            let path = dir.join("ane-direct.lock");
            let holder = AneDirectLaneLock::acquire(&path).unwrap();
            let mut worker = std::process::Command::new("sleep")
                .arg("30")
                .stdin(inheritable_lock_stdio(&holder.inheritable()).unwrap())
                .spawn()
                .unwrap();

            // Each acquire opens the file anew, so it contends exactly like a
            // second module process would.
            let busy = AneDirectLaneLock::acquire(&path).expect_err("holder is alive");
            assert_eq!(busy.code(), Some(ERR_ANE_LANE_BUSY));
            drop(holder);
            let busy = AneDirectLaneLock::acquire(&path).expect_err("its worker is alive");
            assert_eq!(busy.code(), Some(ERR_ANE_LANE_BUSY));
            worker.kill().unwrap();
            worker.wait().unwrap();
            let replacement = AneDirectLaneLock::acquire(&path)
                .expect("the lock is free once the holder and its workers exited");
            assert_eq!(replacement.path(), path.as_path());
            drop(replacement);
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}
