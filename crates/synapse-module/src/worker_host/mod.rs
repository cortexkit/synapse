use std::collections::{BTreeMap, HashMap, HashSet};
use std::io;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use synapse_core::{
    accept_worker_handshake, decode_f32_frame, encode_i32_frame, prepare_listener, read_json,
    read_raw, write_json, write_raw, EmbedEngine, EngineError, EngineErrorStage, EngineIdentity,
    EngineRiskClass, GenerateEngine, GenerateOutput, GenerateRequest, LoadedModel, RerankEngine,
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

#[derive(Clone, Debug)]
pub struct WorkerHostConfig {
    pub worker_bin: PathBuf,
    pub runtime_dir: PathBuf,
    pub worker_id: String,
    pub max_frame: u32,
    pub handshake_timeout: Duration,
    pub request_timeout: Duration,
    pub crash_budget: CrashBudget,
    pub extra_args: Vec<String>,
    pub pooling: WorkerPooling,
    pub normalize: bool,
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
            extra_args: Vec::new(),
            pooling: WorkerPooling::Mean,
            normalize: true,
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
}

#[derive(Clone, Debug)]
struct LoadedWorkerModel {
    crash_key: String,
    artifact: ValidatedArtifact,
    runtime_config: RuntimeConfig,
    worker_model_ref: Option<String>,
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
        }
    }

    pub async fn load_model(
        &mut self,
        artifact: &ValidatedArtifact,
        cfg: &RuntimeConfig,
    ) -> Result<LoadedModel, WorkerHostError> {
        let artifact_path = artifact_path(cfg)?;
        let crash_key = crash_key(&artifact_path, cfg);
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
        match self
            .send_request(
                WorkerRequest::Ping {
                    req_id: req_id.clone(),
                },
                None,
                false,
            )
            .await?
        {
            (
                WorkerResponse::Pong {
                    req_id: got,
                    rss_mb,
                    models_loaded,
                    placement_share,
                },
                _,
            ) => {
                ensure_req_id(&req_id, &got)?;
                self.last_placement_share = placement_share;
                Ok(WorkerPing {
                    rss_mb,
                    models_loaded,
                    placement_share,
                })
            }
            (WorkerResponse::Err { code, msg, .. }, _) => {
                Err(WorkerHostError::WorkerErr { code, msg })
            }
            (other, _) => Err(WorkerHostError::Protocol(format!(
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

        let stream = match accept_worker_handshake(
            listener,
            &nonce,
            self.config.max_frame,
            self.config.handshake_timeout,
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
        let request_timeout = self.config.request_timeout;
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
            Ok(Err(error @ WorkerHostError::Protocol(_))) => Err(error),
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

pub struct WorkerEngine {
    runtime: Runtime,
    host: Mutex<WorkerHost>,
}

impl WorkerEngine {
    pub fn new(config: WorkerHostConfig) -> Result<Self, WorkerHostError> {
        let runtime = Runtime::new()
            .map_err(|error| WorkerHostError::Protocol(format!("create tokio runtime: {error}")))?;
        Ok(Self {
            runtime,
            host: Mutex::new(WorkerHost::new(config)),
        })
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
        self.runtime.block_on(host.ping())
    }
}

impl Drop for WorkerEngine {
    fn drop(&mut self) {
        let Ok(mut host) = self.host.lock() else {
            return;
        };
        let _ = self.runtime.block_on(host.kill_current());
    }
}

impl EmbedEngine for WorkerEngine {
    fn identity(&self) -> EngineIdentity {
        let mut build_flags = BTreeMap::new();
        build_flags.insert("risk_class".to_string(), "abort_capable".to_string());
        build_flags.insert(
            "transport".to_string(),
            worker_transport_label().to_string(),
        );
        EngineIdentity {
            engine: "llama.cpp-worker".to_string(),
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
        self.runtime
            .block_on(host.load_model(artifact, cfg))
            .map_err(|error| error.to_engine_error(EngineErrorStage::Load))
    }

    fn embed_batch(&self, model: &LoadedModel, batch: TokenBatch) -> Result<Vectors, EngineError> {
        let mut host = self
            .lock_host()
            .map_err(|error| error.to_engine_error(EngineErrorStage::Inference))?;
        self.runtime
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
            let _ = self.runtime.block_on(host.unload(model));
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
        self.runtime
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
        self.runtime
            .block_on(host.generate(model, request))
            .map_err(|error| error.to_engine_error(EngineErrorStage::Inference))
    }

    fn unload(&mut self, model: &LoadedModel) {
        <Self as EmbedEngine>::unload(self, model);
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

fn crash_key(path: &Path, cfg: &RuntimeConfig) -> String {
    let mut values = cfg.values.clone();
    values.insert(
        "artifact_path".to_string(),
        path.to_string_lossy().to_string(),
    );
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nonce_is_hex16() {
        let nonce = nonce_hex16();
        assert_eq!(nonce.len(), 16);
        assert!(nonce.chars().all(|ch| ch.is_ascii_hexdigit()));
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
        let error: WorkerHostError = accept_worker_handshake(
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
