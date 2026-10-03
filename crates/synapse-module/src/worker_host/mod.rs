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
use synapse_core::{
    accept_worker_handshake_with_engine_and_protocol_version, decode_f32_frame, encode_i32_frame,
    is_owned_worker_hello_engine, prepare_listener, read_json, read_raw,
    worker_engine_names::LLAMA_WORKER_ENGINE, write_json, write_raw, EmbedEngine, EngineError,
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

#[derive(Clone, Debug)]
pub struct CrashBudget {
    pub max_crashes: usize,
    pub window: Duration,
}

impl Default for CrashBudget {
    fn default() -> Self {
        Self {
            max_crashes: 2,
            window: Duration::from_secs(60),
        }
    }
}

/// Crash/quarantine authority for a worker host.
///
/// Legacy workers use the host's per-model rolling window. Owned decode uses
/// the S3 store-backed `CrashBudget` keyed by machine/decode/runtime identity,
/// so the generic host must not maintain a second crash book for that worker.
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
    pub request_timeout: Duration,
    pub load_timeout: Duration,
    pub crash_budget: CrashBudget,
    pub crash_authority: CrashAuthority,
    pub extra_args: Vec<String>,
    pub pooling: WorkerPooling,
    pub normalize: bool,
    /// Stable catalog model identity attached to forwarded worker diagnostics.
    pub model_id: Option<String>,
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
            load_timeout: Duration::from_secs(180),
            crash_budget: CrashBudget::default(),
            crash_authority: CrashAuthority::WorkerHost,
            extra_args: Vec::new(),
            pooling: WorkerPooling::Mean,
            normalize: true,
            model_id: None,
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
    Quarantined { key: String },
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
                Self::Quarantined { .. } | Self::WorkerErr { .. } | Self::HelloRefused { .. } => {
                    None
                }
                _ => Some(250),
            },
            safe_to_retry_same_request: matches!(self, Self::EngineCrashed { .. }),
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
    crashes: HashMap<String, Vec<Instant>>,
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
            crashes: HashMap::new(),
            last_placement_share: None,
            request_counter: 0,
            model_counter: 0,
            owned_decode_stream: None,
            log_context: Arc::new(Mutex::new(WorkerLogContext::default())),
            forward_limiter,
        }
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
        if self.quarantined.contains(&crash_key) {
            return Err(WorkerHostError::Quarantined { key: crash_key });
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
        let Some(loaded) = self.loaded_models.remove(&model.model_id) else {
            return Ok(());
        };
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
        self.prune_crash_windows(Instant::now());
        let crash_count_window = self.crashes.values().map(Vec::len).sum();
        let loaded_worker_models = self
            .loaded_models
            .values()
            .filter(|loaded| loaded.worker_model_ref.is_some())
            .count();
        WorkerHostHealth {
            worker_connected: self.connection.is_some(),
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
        let loaded = self
            .loaded_models
            .get(&model.model_id)
            .cloned()
            .ok_or_else(|| {
                WorkerHostError::Protocol(format!("unknown loaded model {}", model.model_id))
            })?;
        if self.quarantined.contains(&loaded.crash_key) {
            return Err(WorkerHostError::Quarantined {
                key: loaded.crash_key,
            });
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
        if self.quarantined.contains(crash_key) {
            return Err(WorkerHostError::Quarantined {
                key: crash_key.to_string(),
            });
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
        if self.connection.is_some() {
            return Ok(());
        }
        self.start_worker().await
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
        let stream = match accept_worker_handshake_with_engine_and_protocol_version(
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
        });
        Ok(())
    }

    async fn send_request(
        &mut self,
        request: WorkerRequest,
        raw: Option<Vec<u8>>,
        expect_raw_response: bool,
    ) -> Result<(WorkerResponse, Option<Vec<u8>>), WorkerHostError> {
        self.ensure_worker().await?;
        let max_frame = self.config.max_frame;
        let request_timeout = request_timeout(&self.config, &request);
        let result = timeout(request_timeout, async {
            let connection = self
                .connection
                .as_mut()
                .expect("connection exists after ensure_worker");
            write_json(&mut connection.stream, &request, max_frame).await?;
            if let Some(raw) = raw.as_deref() {
                write_raw(&mut connection.stream, raw, max_frame).await?;
            }
            let response: WorkerResponse = read_json(&mut connection.stream, max_frame).await?;
            let raw_response =
                if expect_raw_response && !matches!(response, WorkerResponse::Err { .. }) {
                    Some(read_raw(&mut connection.stream, max_frame).await?)
                } else {
                    None
                };
            Ok::<_, WorkerHostError>((response, raw_response))
        })
        .await;

        match result {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(error @ WorkerHostError::Protocol(_)))
            | Ok(Err(error @ WorkerHostError::ProtocolVersion { .. })) => Err(error),
            Ok(Err(error)) => {
                let stderr_tail = self.kill_current().await;
                Err(WorkerHostError::EngineCrashed {
                    stage: request_stage(&request).to_string(),
                    detail: error.to_string(),
                    stderr_tail,
                })
            }
            Err(_) => {
                let stderr_tail = self.kill_current().await;
                Err(WorkerHostError::EngineCrashed {
                    stage: "timeout".to_string(),
                    detail: format!("request exceeded {} ms", request_timeout.as_millis()),
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

    fn record_crash(&mut self, key: String) -> bool {
        let now = Instant::now();
        self.prune_crash_windows(now);
        let entries = self.crashes.entry(key.clone()).or_default();
        entries.push(now);
        if entries.len() >= self.config.crash_budget.max_crashes {
            self.quarantined.insert(key);
            true
        } else {
            false
        }
    }

    fn prune_crash_windows(&mut self, now: Instant) {
        let window = self.config.crash_budget.window;
        self.crashes.retain(|_, entries| {
            entries.retain(|instant| now.duration_since(*instant) <= window);
            !entries.is_empty()
        });
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

fn request_timeout(config: &WorkerHostConfig, request: &WorkerRequest) -> Duration {
    if matches!(request, WorkerRequest::Load { .. }) {
        config.load_timeout
    } else {
        config.request_timeout
    }
}

fn request_stage(request: &WorkerRequest) -> &'static str {
    match request {
        WorkerRequest::Load { .. } => "load",
        WorkerRequest::EmbedBatch { .. } => "embed_batch",
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
    fn crash_window_health_degrades_then_recovers() {
        let mut config = WorkerHostConfig::new("/bin/false", "/tmp/synh-health");
        config.crash_budget = CrashBudget {
            max_crashes: 3,
            window: Duration::from_millis(2),
        };
        let mut host = WorkerHost::new(config);

        assert!(!host.health_snapshot().degraded);
        assert!(!host.record_crash("model-a".to_string()));
        let degraded = host.health_snapshot();
        assert!(degraded.degraded);
        assert_eq!(degraded.crash_count_window, 1);

        std::thread::sleep(Duration::from_millis(10));
        let recovered = host.health_snapshot();
        assert!(!recovered.degraded);
        assert_eq!(recovered.crash_count_window, 0);
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
    fn attach_mock_worker<F>(
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
            drop(host);

            let seen = mock.await.unwrap();
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
/// [`ANE_RESIDENT_SHAPES_TOTAL`] overall, enforced with `ANE_ADMIT_SHAPE` and
/// `ANE_EVICT_SHAPE`. A shape counts against the budget from its admit request
/// until its `EVICTED` ack. While a request runs on a shape it holds a lease,
/// and a leased shape is never evicted.
///
/// A request runs its rungs one at a time in ascending order and releases each
/// rung's lease before asking for the next one ([`AneResidencySupervisor::run_by_rung`]),
/// so no request holds a lease while it waits. Waiting requests are served
/// strictly first in, first out; when the budget is full, the request at the
/// head of the queue evicts the least recently used shape that has no lease.
///
/// Worker connections must not be held while waiting for admission: the
/// request at the head may need to evict a shape on another worker. That is why
/// workers are driven through [`AneShapeWorker`], whose connection is held only
/// for a single exchange ([`AneWorkerChannel`]).
pub mod ane_residency {
    use std::collections::{BTreeMap, VecDeque};
    use std::future::Future;
    use std::io;
    #[cfg(unix)]
    use std::path::Path;
    use std::path::PathBuf;
    use std::pin::Pin;
    use std::process::Stdio;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex, MutexGuard};

    use synapse_core::{
        read_json, read_raw, write_json, write_raw, AnePlacementInventory, TransportError,
        WorkerRequest, WorkerResponse, ERR_ANE_LANE_BUSY,
    };
    use thiserror::Error;
    use tokio::io::{AsyncRead, AsyncWrite};
    use tokio::sync::Notify;

    /// Padded sequence shapes a direct-ANE worker compiles; a sequence runs on
    /// the smallest rung at least its length.
    pub const ANE_SHAPE_LADDER: [usize; 7] = [128, 256, 512, 1024, 2048, 4096, 8192];
    pub const ANE_RESIDENT_SHAPES_PER_MODEL: usize = 4;
    pub const ANE_RESIDENT_SHAPES_TOTAL: usize = 8;
    /// Lock file, relative to the home directory, held by the one module
    /// process allowed to run direct-ANE workers.
    pub const ANE_DIRECT_LOCK_HOME_PATH: &str = "Library/Caches/ck-synapse/ane-direct.lock";

    pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

    /// The smallest ladder rung that holds `n_tokens`, or `None` above 8192.
    pub fn ladder_rung(n_tokens: usize) -> Option<usize> {
        ANE_SHAPE_LADDER
            .iter()
            .copied()
            .find(|rung| *rung >= n_tokens)
    }

    #[derive(Debug, Error)]
    pub enum AneResidencyError {
        #[error("sequence of {n_tokens} tokens exceeds the largest direct-ANE shape {max}")]
        SequenceTooLong { n_tokens: usize, max: usize },
        /// The worker answered `ERR`.
        #[error("worker returned {code}: {msg}")]
        WorkerErr { code: String, msg: String },
        #[error("direct-ANE worker channel: {0}")]
        Channel(String),
        #[error("ane_lane_busy: another live process holds {}", path.display())]
        LaneBusy { path: PathBuf },
        #[error("direct-ANE lane lock: {0}")]
        Io(#[from] io::Error),
    }

    impl AneResidencyError {
        pub fn code(&self) -> Option<&str> {
            match self {
                Self::WorkerErr { code, .. } => Some(code),
                Self::LaneBusy { .. } => Some(ERR_ANE_LANE_BUSY),
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
    pub trait AneShapeWorker: Send + Sync {
        /// Stable id of this worker; shapes are dropped per worker on restart.
        fn worker_id(&self) -> &str;
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

    #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct ShapeKey {
        model_ref: String,
        shape: usize,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum SlotState {
        Admitting,
        Resident,
        Evicting,
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
    }

    #[derive(Default)]
    struct State {
        slots: BTreeMap<ShapeKey, Slot>,
        waiters: VecDeque<u64>,
        next_ticket: u64,
        next_slot: u64,
        clock: u64,
        stats: AneResidencyStats,
    }

    impl State {
        fn tick(&mut self) -> u64 {
            self.clock += 1;
            self.clock
        }

        fn model_count(&self, model_ref: &str) -> usize {
            self.slots
                .keys()
                .filter(|key| key.model_ref == model_ref)
                .count()
        }

        fn sample(&mut self) {
            self.stats.samples += 1;
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
        state: Mutex<State>,
        changed: Notify,
        #[cfg(unix)]
        lane_lock: Option<AneDirectLaneLock>,
    }

    impl Inner {
        fn lock(&self) -> MutexGuard<'_, State> {
            self.state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
        }
    }

    /// A request's place in the admission queue; leaving the queue (served or
    /// cancelled) lets the next request re-check its turn.
    struct QueueTicket<'a> {
        inner: &'a Inner,
        id: u64,
    }

    impl<'a> QueueTicket<'a> {
        fn enqueue(inner: &'a Inner) -> Self {
            let mut state = inner.lock();
            state.next_ticket += 1;
            let id = state.next_ticket;
            state.waiters.push_back(id);
            Self { inner, id }
        }
    }

    impl Drop for QueueTicket<'_> {
        fn drop(&mut self) {
            let mut state = self.inner.lock();
            let queued = state.waiters.len();
            state.waiters.retain(|ticket| *ticket != self.id);
            let removed = state.waiters.len() != queued;
            drop(state);
            if removed {
                self.inner.changed.notify_waiters();
            }
        }
    }

    enum Step {
        Leased(AneShapeLease),
        Admit(u64),
        Evict {
            key: ShapeKey,
            slot_id: u64,
            owner: Arc<dyn AneShapeWorker>,
        },
        Wait,
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
                inner: Arc::new(Inner {
                    limits,
                    state: Mutex::new(State::default()),
                    changed: Notify::new(),
                    #[cfg(unix)]
                    lane_lock: None,
                }),
            }
        }

        /// A supervisor holding the lane lock for as long as it lives.
        #[cfg(unix)]
        pub fn with_lane_lock(limits: AneResidencyLimits, lock: AneDirectLaneLock) -> Self {
            Self {
                inner: Arc::new(Inner {
                    limits,
                    state: Mutex::new(State::default()),
                    changed: Notify::new(),
                    lane_lock: Some(lock),
                }),
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
        pub fn lane_lock(&self) -> Option<&AneDirectLaneLock> {
            self.inner.lane_lock.as_ref()
        }

        pub fn limits(&self) -> AneResidencyLimits {
            self.inner.limits
        }

        pub fn stats(&self) -> AneResidencyStats {
            self.inner.lock().stats.clone()
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
            let key = ShapeKey {
                model_ref: model_ref.to_string(),
                shape,
            };
            let ticket = QueueTicket::enqueue(&self.inner);
            loop {
                // Register for wakeups before reading the state, so a change
                // between the check and the wait is not missed.
                let notified = self.inner.changed.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                match self.next_step(ticket.id, &key, worker) {
                    Step::Leased(lease) => return Ok(lease),
                    Step::Admit(slot_id) => return self.admit(worker.clone(), key, slot_id).await,
                    Step::Evict {
                        key,
                        slot_id,
                        owner,
                    } => self.evict(owner, key, slot_id).await,
                    Step::Wait => notified.await,
                }
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
            mut run: F,
        ) -> Result<Vec<T>, AneResidencyError>
        where
            F: FnMut(usize, Vec<usize>) -> Fut,
            Fut: Future<Output = Result<Vec<T>, AneResidencyError>>,
        {
            let mut rungs: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
            for (index, &n_tokens) in lengths.iter().enumerate() {
                let rung = ladder_rung(n_tokens).ok_or(AneResidencyError::SequenceTooLong {
                    n_tokens,
                    max: ANE_SHAPE_LADDER[ANE_SHAPE_LADDER.len() - 1],
                })?;
                rungs.entry(rung).or_default().push(index);
            }
            let mut outputs: Vec<Option<T>> = (0..lengths.len()).map(|_| None).collect();
            for (rung, indices) in rungs {
                let lease = self.lease(worker, model_ref, rung).await?;
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

        fn next_step(&self, ticket: u64, key: &ShapeKey, worker: &Arc<dyn AneShapeWorker>) -> Step {
            let mut state = self.inner.lock();
            if state.waiters.front() != Some(&ticket) {
                return Step::Wait;
            }
            let now = state.tick();
            if let Some(slot) = state.slots.get_mut(key) {
                if slot.state != SlotState::Resident {
                    // Wait for the admission or eviction in flight to settle.
                    return Step::Wait;
                }
                slot.leases += 1;
                slot.last_used = now;
                let lease = AneShapeLease {
                    inner: self.inner.clone(),
                    key: key.clone(),
                    slot_id: slot.id,
                    inventory: slot.inventory.clone(),
                };
                state.waiters.pop_front();
                self.inner.changed.notify_waiters();
                return Step::Leased(lease);
            }
            let model_full = state.model_count(&key.model_ref) >= self.inner.limits.per_model;
            let total_full = state.slots.len() >= self.inner.limits.total;
            if !model_full && !total_full {
                state.next_slot += 1;
                let id = state.next_slot;
                state.slots.insert(
                    key.clone(),
                    Slot {
                        id,
                        worker: worker.clone(),
                        state: SlotState::Admitting,
                        leases: 0,
                        last_used: now,
                        inventory: None,
                    },
                );
                state.waiters.pop_front();
                self.inner.changed.notify_waiters();
                return Step::Admit(id);
            }
            // When the model is at its own limit only one of its shapes frees
            // the slot it needs; otherwise any model's shape does.
            let victim = state
                .slots
                .iter()
                .filter(|(candidate, slot)| {
                    slot.state == SlotState::Resident
                        && slot.leases == 0
                        && (!model_full || candidate.model_ref == key.model_ref)
                })
                .min_by_key(|(_, slot)| slot.last_used)
                .map(|(candidate, slot)| (candidate.clone(), slot.id, slot.worker.clone()));
            let Some((victim, slot_id, owner)) = victim else {
                return Step::Wait;
            };
            if let Some(slot) = state.slots.get_mut(&victim) {
                slot.state = SlotState::Evicting;
            }
            Step::Evict {
                key: victim,
                slot_id,
                owner,
            }
        }

        async fn admit(
            &self,
            worker: Arc<dyn AneShapeWorker>,
            key: ShapeKey,
            slot_id: u64,
        ) -> Result<AneShapeLease, AneResidencyError> {
            let inner = self.inner.clone();
            // The exchange runs on its own task so its outcome is recorded even
            // if the waiting request is dropped part way through.
            let task = tokio::spawn(async move {
                match worker.admit_shape(&key.model_ref, key.shape).await {
                    Ok(inventory) => finish_admit(&inner, &key, slot_id, inventory),
                    Err(error) => {
                        recover_worker(&inner, &worker).await;
                        Err(error)
                    }
                }
            });
            task.await.map_err(|error| {
                AneResidencyError::Channel(format!("shape admission task failed: {error}"))
            })?
        }

        async fn evict(&self, owner: Arc<dyn AneShapeWorker>, key: ShapeKey, slot_id: u64) {
            let inner = self.inner.clone();
            let task = tokio::spawn(async move {
                match owner.evict_shape(&key.model_ref, key.shape).await {
                    Ok(()) => {
                        let mut state = inner.lock();
                        if state.slots.get(&key).is_some_and(|slot| slot.id == slot_id) {
                            state.slots.remove(&key);
                            state.stats.evicted += 1;
                            state.sample();
                        }
                        drop(state);
                        inner.changed.notify_waiters();
                    }
                    Err(error) => {
                        tracing::warn!(
                            target: "worker",
                            worker_id = owner.worker_id(),
                            model_ref = %key.model_ref,
                            shape = key.shape,
                            error = %error,
                            "direct-ANE evict failed; restarting the worker"
                        );
                        recover_worker(&inner, &owner).await;
                    }
                }
            });
            if let Err(error) = task.await {
                tracing::warn!(target: "worker", error = %error, "direct-ANE eviction task failed");
            }
        }
    }

    fn finish_admit(
        inner: &Arc<Inner>,
        key: &ShapeKey,
        slot_id: u64,
        inventory: AnePlacementInventory,
    ) -> Result<AneShapeLease, AneResidencyError> {
        let mut state = inner.lock();
        let now = state.tick();
        let Some(slot) = state.slots.get_mut(key).filter(|slot| slot.id == slot_id) else {
            return Err(AneResidencyError::Channel(format!(
                "worker restarted while admitting {} shape {}",
                key.model_ref, key.shape
            )));
        };
        let inventory = Arc::new(inventory);
        slot.state = SlotState::Resident;
        slot.leases = 1;
        slot.last_used = now;
        slot.inventory = Some(inventory.clone());
        state.stats.admitted += 1;
        state.sample();
        drop(state);
        inner.changed.notify_waiters();
        Ok(AneShapeLease {
            inner: inner.clone(),
            key: key.clone(),
            slot_id,
            inventory: Some(inventory),
        })
    }

    /// Restarts a worker that answered an admit or evict with `ERR` (or lost
    /// its connection), then forgets every shape counted for it, since the new
    /// process has none, and wakes the waiting requests.
    async fn recover_worker(inner: &Inner, worker: &Arc<dyn AneShapeWorker>) {
        if let Err(error) = worker.restart().await {
            tracing::warn!(
                target: "worker",
                worker_id = worker.worker_id(),
                error = %error,
                "direct-ANE worker restart failed"
            );
        }
        let mut state = inner.lock();
        let worker_id = worker.worker_id();
        state
            .slots
            .retain(|_, slot| slot.worker.worker_id() != worker_id);
        state.stats.restarts += 1;
        drop(state);
        inner.changed.notify_waiters();
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
        }
    }

    /// Opens a fresh connection to a direct-ANE worker process (spawning it
    /// and completing its HELLO).
    pub type AneWorkerConnector<S> =
        Box<dyn Fn() -> BoxFuture<'static, Result<S, AneResidencyError>> + Send + Sync>;

    /// A direct-ANE worker connection used for one exchange at a time, so a
    /// shape admission or eviction never waits behind a whole request.
    pub struct AneWorkerChannel<S> {
        worker_id: String,
        max_frame: u32,
        stream: tokio::sync::Mutex<Option<S>>,
        connect: AneWorkerConnector<S>,
        request_counter: AtomicU64,
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
            Ok(Self {
                worker_id: worker_id.into(),
                max_frame,
                stream: tokio::sync::Mutex::new(Some(stream)),
                connect,
                request_counter: AtomicU64::new(0),
            })
        }

        pub fn next_req_id(&self, prefix: &str) -> String {
            format!(
                "{prefix}-{}",
                self.request_counter.fetch_add(1, Ordering::Relaxed)
            )
        }

        /// Sends one request (and its raw frame) and reads the response (and
        /// its raw frame). A transport failure closes the connection; the
        /// worker is unusable until [`AneShapeWorker::restart`].
        pub async fn exchange(
            &self,
            request: &WorkerRequest,
            raw: Option<&[u8]>,
        ) -> Result<(WorkerResponse, Option<Vec<u8>>), AneResidencyError> {
            let mut guard = self.stream.lock().await;
            let stream = guard.as_mut().ok_or_else(|| {
                AneResidencyError::Channel(format!("worker {} is not connected", self.worker_id))
            })?;
            let max_frame = self.max_frame;
            let result = async {
                write_json(stream, request, max_frame).await?;
                if let Some(raw) = raw {
                    write_raw(stream, raw, max_frame).await?;
                }
                let response: WorkerResponse = read_json(stream, max_frame).await?;
                let raw = if response.carries_raw_frame() {
                    Some(read_raw(stream, max_frame).await?)
                } else {
                    None
                };
                Ok::<_, TransportError>((response, raw))
            }
            .await;
            result.map_err(|error| {
                *guard = None;
                error.into()
            })
        }
    }

    impl<S> AneShapeWorker for AneWorkerChannel<S>
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        fn worker_id(&self) -> &str {
            &self.worker_id
        }

        fn admit_shape<'a>(
            &'a self,
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
                match self.exchange(&request, None).await? {
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

        fn evict_shape<'a>(
            &'a self,
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
                match self.exchange(&request, None).await? {
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
            Box::pin(async move {
                let mut guard = self.stream.lock().await;
                // Closing the connection ends the old worker process, which
                // exits when its supervisor connection closes.
                *guard = None;
                *guard = Some((self.connect)().await?);
                Ok(())
            })
        }
    }

    /// A `Stdio` for a direct-ANE worker's stdin that duplicates the locked
    /// descriptor. The duplicate shares the lock, so the lock is released only
    /// when this process and every worker it spawned have closed it.
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

    #[cfg(unix)]
    impl AneDirectLaneLock {
        /// `~/Library/Caches/ck-synapse/ane-direct.lock`.
        pub fn default_path() -> Option<PathBuf> {
            std::env::var_os("HOME").map(|home| PathBuf::from(home).join(ANE_DIRECT_LOCK_HOME_PATH))
        }

        /// Takes the lock without waiting. While any other open description of
        /// the file holds it (another module process, or a direct-ANE worker
        /// that inherited it and has not exited yet), fails with
        /// `ane_lane_busy`.
        // `File::try_lock` is a `flock` and avoids the unsafe code a direct call would need.
        pub fn acquire(path: &Path) -> Result<Self, AneResidencyError> {
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
        type Channel = Arc<AneWorkerChannel<DuplexStream>>;

        /// What the mock workers observe, kept independently of the
        /// supervisor's own accounting so the budget checks can fail.
        #[derive(Default)]
        struct Ledger {
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
            admit_delay: Duration,
            connects: HashMap<String, u32>,
        }

        type SharedLedger = Arc<Mutex<Ledger>>;

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
        async fn serve_mock(mut stream: DuplexStream, ledger: SharedLedger) {
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
                let (response, raw_out) = match &request {
                    WorkerRequest::AneAdmitShape {
                        req_id,
                        model_ref,
                        shape,
                    } => {
                        let delay = ledger.lock().unwrap().admit_delay;
                        tokio::time::sleep(delay).await;
                        let key = (model_ref.clone(), *shape);
                        let mut ledger = ledger.lock().unwrap();
                        if ledger.fail_admit.remove(&key) {
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
                        if rungs
                            .iter()
                            .all(|rung| own.contains(&(model_ref.clone(), *rung)))
                        {
                            let values: Vec<f32> = rungs.iter().map(|rung| *rung as f32).collect();
                            (
                                WorkerResponse::Vectors {
                                    req_id: req_id.clone(),
                                    dims: 1,
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
                    tokio::spawn(serve_mock(worker, ledger));
                    Ok(host)
                })
            });
            Arc::new(
                AneWorkerChannel::connect(worker_id, DEFAULT_MAX_FRAME_BYTES, connect)
                    .await
                    .unwrap(),
            )
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
