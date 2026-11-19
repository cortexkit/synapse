use super::*;
use std::collections::{HashMap, HashSet};
use std::time::Duration;
use synapse_core::{
    encode_f32_frame, AneExecutable, DEFAULT_MAX_FRAME_BYTES, ERR_SHAPE_NOT_ADMITTED,
};
#[cfg(test)]
use tokio::io::DuplexStream;
const LAYERS: u32 = 4;
#[cfg(test)]
pub(super) struct MockChannel(pub(super) AneWorkerChannel<DuplexStream>);
#[cfg(test)]
impl std::ops::Deref for MockChannel {
    type Target = AneWorkerChannel<DuplexStream>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
#[cfg(test)]
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
    /// order and are never refused early.
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
#[cfg(test)]
pub(super) type Channel = Arc<MockChannel>;

/// What the mock workers observe, kept independently of the
/// supervisor's own accounting so the budget checks can fail.
#[derive(Default)]
pub(super) struct Ledger {
    pub(super) loads: Vec<WorkerRequest>,
    // Read only by the Unix-only pause tests.
    #[cfg(unix)]
    pub(super) confirmed_exits: u32,
    pub(super) inference_gate: Option<(Arc<Notify>, Arc<Notify>)>,
    pub(super) admit_gate: Option<(Arc<Notify>, Arc<Notify>)>,
    pub(super) fail_inference: bool,
    pub(super) invalid_inference: Option<InvalidInference>,
    pub(super) evict_gate: Option<(Arc<Notify>, Arc<Notify>)>,
    /// Requests currently running on (model_ref, shape); a request
    /// counts itself here only while it holds that shape's lease.
    pub(super) in_flight: HashMap<(String, usize), u32>,
    /// Shapes compiled in some live mock worker.
    pub(super) resident: HashSet<(String, usize)>,
    pub(super) max_per_model: usize,
    pub(super) max_total: usize,
    /// `admit <model> <shape>` / `evict <model> <shape>`, in arrival order.
    pub(super) events: Vec<String>,
    pub(super) leased_evicts: u64,
    pub(super) shape_not_admitted: u64,
    /// Admissions the mock answers with `ERR`, once each.
    pub(super) fail_admit: HashSet<(String, usize)>,
    pub(super) fail_evict: HashSet<(String, usize)>,
    #[cfg(test)]
    pub(super) fail_confirm_exit: bool,
    pub(super) exit_gate: Option<(Arc<Notify>, Arc<Notify>)>,
    pub(super) no_reply_admit: HashSet<(String, usize)>,
    pub(super) no_reply_evict: HashSet<(String, usize)>,
    pub(super) exhaust_admit: HashMap<(String, usize), usize>,
    pub(super) admit_attempts: HashMap<(String, usize), usize>,
    pub(super) admit_delay: Duration,
    pub(super) connects: HashMap<String, u32>,
    /// Inference requests (embed or rerank) the mock worker received.
    pub(super) inference_frames: u64,
    /// Their request ids, in arrival order.
    pub(super) inference_ids: Vec<String>,
    pub(super) capture_token_ids: bool,
    pub(super) captured_token_ids: Vec<Vec<u32>>,
}

pub(super) type SharedLedger = Arc<Mutex<Ledger>>;

#[derive(Clone, Copy)]
#[cfg_attr(not(test), allow(dead_code))]
pub(super) enum InvalidInference {
    Nonfinite,
    Dimensions,
}

pub(super) fn record_token_ids(
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

pub(super) fn record_residency(ledger: &mut Ledger) {
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
pub(super) async fn serve_mock<S: AsyncRead + AsyncWrite + Unpin>(
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
                            .map(|profile| format!("ane-direct:{profile}:{artifact_digest}"))
                            .unwrap_or_else(|| format!("mock:{artifact_digest}")),
                        dims: if direct_profile.is_some() { 1 } else { 4 },
                        cold_load_ms: 0,
                        buckets: direct_profile.map(|_| ANE_SHAPE_LADDER.to_vec()),
                        ane_cached_shapes: None,
                        transcribe_revision: None,
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
                let exhausted = ledger.exhaust_admit.get_mut(&key).is_some_and(|remaining| {
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
                            cache_hit_layers: None,
                            compiled_layers: None,
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
#[cfg(unix)]
pub fn module_mock_engine(started: Arc<Notify>, release: Arc<Notify>) -> Arc<DirectAneEngine> {
    module_mock_engine_gate(started, release, false)
}

#[cfg(unix)]
pub fn module_mock_engine_admission(
    started: Arc<Notify>,
    release: Arc<Notify>,
) -> Arc<DirectAneEngine> {
    module_mock_engine_gate(started, release, true)
}

#[cfg(unix)]
pub(super) fn module_mock_engine_gate(
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
pub struct ModuleMockControl(SharedLedger);

#[cfg(unix)]
impl ModuleMockControl {
    #[cfg(target_os = "macos")]
    pub fn enable_token_capture(&self) {
        self.0.lock().unwrap().capture_token_ids = true;
    }

    #[cfg(target_os = "macos")]
    pub fn take_token_ids(&self) -> Vec<Vec<u32>> {
        std::mem::take(&mut self.0.lock().unwrap().captured_token_ids)
    }
    pub fn hold_next_admission(&self, started: Arc<Notify>, release: Arc<Notify>) {
        self.0.lock().unwrap().admit_gate = Some((started, release));
    }

    pub fn hold_next_inference(&self, started: Arc<Notify>, release: Arc<Notify>) {
        self.0.lock().unwrap().inference_gate = Some((started, release));
    }

    /// Compiles the mock worker has run for `shape`, on any model.
    pub fn compiles(&self, shape: usize) -> usize {
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
    pub fn inferences(&self) -> u64 {
        self.0.lock().unwrap().inference_frames
    }

    /// Worker processes the engine has started, restarts included.
    pub fn worker_starts(&self) -> u32 {
        self.0.lock().unwrap().connects.values().sum()
    }

    pub fn loads(&self) -> usize {
        self.0.lock().unwrap().loads.len()
    }

    #[cfg(unix)]
    pub fn confirmed_exits(&self) -> u32 {
        self.0.lock().unwrap().confirmed_exits
    }

    pub fn hold_next_exit(&self, started: Arc<Notify>, release: Arc<Notify>) {
        self.0.lock().unwrap().exit_gate = Some((started, release));
    }
}

#[cfg(unix)]
pub fn module_mock_engine_controlled() -> (Arc<DirectAneEngine>, ModuleMockControl) {
    let ledger = SharedLedger::default();
    (
        module_mock_engine_for(ledger.clone()),
        ModuleMockControl(ledger),
    )
}

#[cfg(unix)]
pub(super) fn module_mock_engine_for(ledger: SharedLedger) -> Arc<DirectAneEngine> {
    module_mock_engine_for_owner(
        ledger,
        AneResidencySupervisor::new(Default::default()),
        "module-mock",
        "gte-modernbert-base.ane-direct-worker",
    )
}

#[cfg(unix)]
pub fn module_mock_engine_resident(
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
pub(super) fn module_mock_engine_for_owner(
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
                max_frame: None,
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
    // The fixture also runs as a dependency of module tests, without cfg(test).
    // Keep its instantaneous mock compiles out of the real pinned cost model.
    *channel.test_compile_seed.lock().unwrap() = Some(Duration::ZERO);
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
