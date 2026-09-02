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
    prepare_listener, read_json, read_raw, worker_engine_names::LLAMA_WORKER_ENGINE, write_json,
    write_raw, EmbedEngine, EngineError, EngineErrorStage, EngineIdentity, EngineRiskClass,
    GenerateEngine, GenerateOutput, GenerateRequest, LoadedModel, ProgressBoundary, RerankEngine,
    RerankRequest, RerankScores, RuntimeConfig, TokenBatch, TokenIds, TransportError,
    ValidatedArtifact, Vector, Vectors, WorkerCandidate, WorkerPooling, WorkerRequest,
    WorkerResponse, WorkerTokenItem, WorkerTransportStream, DEFAULT_MAX_FRAME_BYTES,
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
    /// Optional catalog identity used by engines that share this generic host.
    pub engine_identity: Option<EngineIdentity>,
    /// Owned-CUDA has one process per stable model specification. Including
    /// the worker id in the crash key keeps equal artifacts isolated.
    pub isolate_crash_key_by_worker_id: bool,
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
            engine_identity: None,
            isolate_crash_key_by_worker_id: false,
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
                Self::Quarantined { .. } | Self::WorkerErr { .. } => None,
                _ => Some(250),
            },
            safe_to_retry_same_request: matches!(self, Self::EngineCrashed { .. }),
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

#[derive(Clone, Debug)]
struct LoadedWorkerModel {
    crash_key: String,
    artifact: ValidatedArtifact,
    runtime_config: RuntimeConfig,
    worker_model_ref: Option<String>,
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
}

impl WorkerHost {
    pub fn new(config: WorkerHostConfig) -> Self {
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

        let worker_model_ref = self
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
        let (worker_model_ref, crash_key) = self.ensure_worker_model(model).await?;
        let req_id = self.next_req_id("rerank");
        let mut ids = token_ids_to_i32(&request.query)?;
        for candidate in &request.candidates {
            ids.extend(token_ids_to_i32(candidate)?);
        }
        let worker_request = WorkerRequest::Rerank {
            req_id: req_id.clone(),
            model_ref: worker_model_ref,
            query_n_tokens: request.query.len(),
            candidates: request
                .candidates
                .iter()
                .map(|candidate| WorkerCandidate {
                    n_tokens: candidate.len(),
                })
                .collect(),
        };
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
                "RERANK returned unexpected response {other:?}"
            ))),
            Err(error @ WorkerHostError::EngineCrashed { .. }) => {
                self.record_crash_and_maybe_restart(crash_key).await;
                Err(error)
            }
            Err(error) => Err(error),
        }
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

        let worker_model_ref = self
            .load_worker_model_ref(&loaded.artifact, &loaded.runtime_config, &loaded.crash_key)
            .await?;
        if let Some(entry) = self.loaded_models.get_mut(&model.model_id) {
            entry.worker_model_ref = Some(worker_model_ref.clone());
        }
        Ok((worker_model_ref, loaded.crash_key))
    }

    async fn load_worker_model_ref(
        &mut self,
        artifact: &ValidatedArtifact,
        cfg: &RuntimeConfig,
        crash_key: &str,
    ) -> Result<String, WorkerHostError> {
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
                    ..
                },
                _,
            )) => {
                ensure_req_id(&req_id, &got)?;
                Ok(model_ref)
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
        let mut command = Command::new(&self.config.worker_bin);
        append_worker_spawn_args(&mut command, &endpoint, &nonce);
        command
            .env("SYNAPSE_WORKER_ID", &self.config.worker_id)
            .stderr(Stdio::piped())
            .stdout(Stdio::piped())
            .kill_on_drop(true);
        for arg in &self.config.extra_args {
            command.arg(arg);
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
        let mut child = command.spawn().map_err(|error| {
            WorkerHostError::Protocol(format!(
                "spawn worker {}: {error}",
                self.config.worker_bin.display()
            ))
        })?;
        let logs = LogRing::new();
        if let Some(stderr) = child.stderr.take() {
            spawn_pipe_reader("[stderr] ", stderr, logs.clone());
        }
        if let Some(stdout) = child.stdout.take() {
            spawn_pipe_reader("[stdout] ", stdout, logs.clone());
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
}

impl WorkerEngine {
    pub fn new(config: WorkerHostConfig) -> Result<Self, WorkerHostError> {
        let runtime = Runtime::new()
            .map_err(|error| WorkerHostError::Protocol(format!("create tokio runtime: {error}")))?;
        Ok(Self {
            runtime: Some(runtime),
            host: Arc::new(Mutex::new(WorkerHost::new(config))),
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

impl Drop for WorkerEngine {
    fn drop(&mut self) {
        let Some(runtime) = self.runtime.take() else {
            return;
        };
        let host = Arc::clone(&self.host);
        let teardown = move || {
            if let Ok(mut host) = host.lock() {
                let _ = runtime.block_on(host.kill_current());
            }
            drop(runtime);
        };
        if tokio::runtime::Handle::try_current().is_ok() {
            // Dropped on a thread that is driving a tokio runtime (module
            // state teardown): block_on here — or even dropping the engine
            // runtime — panics, and a panic in Drop aborts the rest of the
            // state teardown. Hand the blocking kill to a dedicated thread.
            std::thread::spawn(teardown);
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
        let mut host = self
            .lock_host()
            .map_err(|error| error.to_engine_error(EngineErrorStage::Inference))?;
        self.runtime()
            .block_on(host.embed_batch(model, batch))
            .map_err(|error| error.to_engine_error(EngineErrorStage::Inference))
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
        let mut host = self
            .lock_host()
            .map_err(|error| error.to_engine_error(EngineErrorStage::Inference))?;
        self.runtime()
            .block_on(host.rerank(model, request))
            .map_err(|error| error.to_engine_error(EngineErrorStage::Inference))
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
        let mut host = self
            .lock_host()
            .map_err(|error| error.to_engine_error(EngineErrorStage::Inference))?;
        self.runtime()
            .block_on(host.generate(model, request))
            .map_err(|error| error.to_engine_error(EngineErrorStage::Inference))
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

struct MonotonicDispatchClock {
    started: Instant,
}

impl OwnedClock for MonotonicDispatchClock {
    fn now(&self) -> u64 {
        self.started
            .elapsed()
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX)
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
    clock: MonotonicDispatchClock,
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
            clock: MonotonicDispatchClock {
                started: Instant::now(),
            },
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
        let mut engine =
            WorkerEngine::new(self.config.clone()).map_err(|_| WorkerFault::StartupFailure)?;
        let model = GenerateEngine::load(&mut engine, &self.artifact, &self.runtime_config)
            .map_err(|_| WorkerFault::StartupFailure)?;
        let worker_generation = engine
            .owned_decode_worker_generation()
            .map_err(|_| WorkerFault::StartupFailure)?;
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
        | WorkerHostError::WorkerErr { .. } => WorkerFault::Protocol,
        WorkerHostError::EngineCrashed { .. }
        | WorkerHostError::Io(_)
        | WorkerHostError::Quarantined { .. } => WorkerFault::Crash,
    }
}

fn spawn_pipe_reader<R>(prefix: &'static str, mut reader: R, ring: LogRing)
where
    R: AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let mut buffer = [0_u8; 512];
        loop {
            match reader.read(&mut buffer).await {
                Ok(0) => break,
                Ok(n) => {
                    ring.push(prefix.as_bytes());
                    ring.push(&buffer[..n]);
                }
                Err(_) => break,
            }
        }
    });
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
            },
        );

        host.forget_worker_model_refs();

        let tracked = host.loaded_models.get("stable-model").unwrap();
        assert_eq!(tracked.crash_key, "model-key");
        assert_eq!(tracked.artifact.format, "gguf");
        assert!(tracked.worker_model_ref.is_none());
    }
}
