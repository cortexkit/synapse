//! Temporarily refuse new Neural Engine work while lending the hardware to
//! another daemon. Pause state is held only in memory and ends on restart.
use super::*;
use tokio::sync::{OwnedRwLockReadGuard, RwLock};

const MAX_DURATION_MS: u64 = 120 * 60 * 1000;

#[derive(Default)]
pub(super) struct LanePause {
    current: Mutex<Option<Pause>>,
    gate: Arc<RwLock<()>>,
    changed: Notify,
    #[cfg(test)]
    pub(super) drain_failure: Mutex<Option<(String, String)>>,
    #[cfg(test)]
    pub(super) drain_attempts: std::sync::atomic::AtomicUsize,
}

struct Pause {
    until_ms: u64,
    deadline: tokio::time::Instant,
    reason: String,
    lanes: Vec<String>,
    phase: &'static str,
    drain_lane: Option<String>,
    drain_stage: &'static str,
    drain_error: Option<Value>,
    retry_drain: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PauseParams {
    scope: String,
    duration_ms: u64,
    reason: String,
}

fn uses_ane(engine: &str) -> bool {
    matches!(engine, "ane" | "ane-coreml-worker" | "ane-direct-worker")
}

pub(super) fn uses_ane_model(model: &EmbeddingModel) -> bool {
    uses_ane(&model.engine_identity.engine)
}

pub(super) fn admit_params(
    state: &ModuleState,
    params: &Value,
    task: ModelTask,
) -> Result<Option<OwnedRwLockReadGuard<()>>, WireOperationError> {
    admit_request(
        state,
        params
            .get("model")
            .or_else(|| params.get("model_id"))
            .and_then(Value::as_str),
        task,
        params.get("required_fingerprint").and_then(Value::as_str),
        params.get("target_fingerprint").and_then(Value::as_str),
    )
}

fn affected(runtime: &RuntimeState, lane: &str) -> bool {
    resolved_catalog_lane(runtime, lane).is_some_and(|(_, b)| uses_ane(&b.engine))
        || model_slot_snapshot(runtime, lane)
            .is_some_and(|s| uses_ane(&s.spec.engine) || uses_ane(&s.spec.engine_identity.engine))
}

fn lane_ids(runtime: &RuntimeState) -> Vec<String> {
    let mut lanes = runtime
        .release_catalog
        .models
        .iter()
        .flat_map(|entry| {
            entry
                .backends
                .iter()
                .filter(|b| uses_ane(&b.engine))
                .map(|b| catalog::lane_id(&entry.id, &b.backend))
        })
        .collect::<BTreeSet<_>>();
    lanes.extend(
        runtime
            .catalog
            .lock()
            .expect("catalog")
            .iter()
            .filter(|(_, s)| uses_ane(&s.spec.engine) || uses_ane(&s.spec.engine_identity.engine))
            .map(|(id, _)| id.clone()),
    );
    lanes.into_iter().collect()
}

fn refusal(runtime: &RuntimeState, lane: &str, pause: &Pause) -> WireOperationError {
    let remaining = pause
        .deadline
        .saturating_duration_since(tokio::time::Instant::now())
        .as_millis()
        .min(u64::MAX as u128) as u64;
    let mut details =
        json!({"lane_id":lane,"paused_until_ms":pause.until_ms,"reason":pause.reason});
    let fingerprint = resolved_catalog_lane(runtime, lane)
        .map(|(_, b)| b.fingerprint.clone())
        .or_else(|| {
            model_slot_snapshot(runtime, lane)
                .map(|s| s.spec.fingerprint.0)
                .filter(|f| !f.is_empty())
        });
    if let Some(fingerprint) = fingerprint {
        details["fingerprint"] = json!(fingerprint);
    }
    WireOperationError::from_stable(
        StableError::new(
            synapse_core::StableErrorCode::LanePaused,
            ErrorClass::Transient,
            Some(remaining.max(1000)),
            true,
        )
        .with_details(serde_json::from_value(details).expect("details object")),
        "Neural Engine lane is paused; retry later",
    )
}

/// The synchronous check and read acquisition share the pause-state mutex, so
/// no request can slip between publishing a pause and starting its drain.
pub(super) fn admit(
    runtime: &RuntimeState,
    lane: &str,
) -> Result<Option<OwnedRwLockReadGuard<()>>, WireOperationError> {
    if !affected(runtime, lane) {
        return Ok(None);
    }
    admit_ane(runtime, lane)
}

fn admit_ane(
    runtime: &RuntimeState,
    lane: &str,
) -> Result<Option<OwnedRwLockReadGuard<()>>, WireOperationError> {
    let current = runtime.lane_pause.current.lock().expect("pause state");
    if let Some(pause) = current.as_ref() {
        return Err(refusal(runtime, lane, pause));
    }
    Ok(Some(
        runtime
            .lane_pause
            .gate
            .clone()
            .try_read_owned()
            .expect("open pause gate"),
    ))
}

pub(super) fn admit_load(
    runtime: &RuntimeState,
    params: &ModelLoadParams,
) -> Result<Option<OwnedRwLockReadGuard<()>>, WireOperationError> {
    if !uses_ane(&canonical_engine_name(&params.engine)) {
        return Ok(None);
    }
    admit_ane(runtime, params.model_id.as_deref().unwrap_or("model.load"))
}

/// Resolve identity only: do not load a model, validate a self-check, or choose
/// a different backend because the requested lane is paused.
pub(super) fn admit_request(
    state: &ModuleState,
    requested: Option<&str>,
    task: ModelTask,
    required: Option<&str>,
    target: Option<&str>,
) -> Result<Option<OwnedRwLockReadGuard<()>>, WireOperationError> {
    let runtime = &state.runtime;
    if let Some(id) = requested {
        if !runtime.release_catalog.is_reserved_id(id) {
            return admit(runtime, id);
        }
    }
    let reserved = requested
        .and_then(|id| runtime.release_catalog.resolve_reserved(id))
        .or_else(|| {
            runtime
                .release_catalog
                .models
                .iter()
                .find(|e| e.task == task.as_str() && e.default_for_task)
                .map(|e| (e, None))
        });
    if let Some((entry, pinned)) = reserved {
        let backend = pinned
            .and_then(|b| entry.backend(b))
            .or_else(|| {
                required
                    .or(target)
                    .and_then(|f| entry.backends.iter().find(|b| b.fingerprint == f))
            })
            .or_else(|| {
                entry
                    .backends
                    .iter()
                    .find(|b| runtime.runnable_backends.contains(&b.backend))
            });
        if let Some(backend) = backend {
            return admit(runtime, &catalog::lane_id(&entry.id, &backend.backend));
        }
    }
    if let Ok(Some(assignment)) =
        state
            .store
            .knob_assignment(&state.machine_profile_hash, task.as_str(), runtime.knob)
    {
        return admit(runtime, &assignment.model_id);
    }
    Ok(None)
}

pub(super) fn status(runtime: &RuntimeState) -> Value {
    let current = runtime.lane_pause.current.lock().expect("pause state");
    match current.as_ref() {
        Some(p) => {
            let mut status = json!({"paused_until_ms":p.until_ms,"reason":p.reason,"lane_ids":p.lanes,
                "phase":p.phase,"hardware_free":p.phase == "paused"});
            if let Some(error) = &p.drain_error {
                status["drain_error"] = error.clone();
            }
            status
        }
        None => json!({"phase":"running","hardware_free":false,"lane_ids":[]}),
    }
}

pub(super) fn project(runtime: &RuntimeState, row: &mut Value) {
    let lane = row["model_id"].as_str().unwrap_or("");
    if !affected(runtime, lane) {
        return;
    }
    let current = runtime.lane_pause.current.lock().expect("pause state");
    if let Some(p) = current.as_ref() {
        row["paused_until_ms"] = json!(p.until_ms);
        row["pause_reason"] = json!(p.reason);
    }
}

pub(super) async fn pause(state: Arc<ModuleState>, params: Value) -> HandlerOutcome {
    let params: PauseParams = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(e) => {
            return channel_error(
                "invalid_request",
                format!("invalid lanes.pause params: {e}"),
            )
        }
    };
    if params.scope != "neural_engine"
        || params.duration_ms == 0
        || params.duration_ms > MAX_DURATION_MS
        || params.reason.trim().is_empty()
    {
        return channel_error(
            "invalid_request",
            "scope must be neural_engine; duration_ms must be 1..=7200000; reason must be nonempty",
        );
    }
    let lanes = lane_ids(&state.runtime);
    {
        let mut current = state
            .runtime
            .lane_pause
            .current
            .lock()
            .expect("pause state");
        if current.is_some() {
            return channel_error(
                "invalid_request",
                "a pause is already active; resume it first",
            );
        }
        *current = Some(Pause {
            until_ms: now_ms().saturating_add(params.duration_ms),
            deadline: tokio::time::Instant::now() + Duration::from_millis(params.duration_ms),
            reason: params.reason,
            lanes,
            phase: "draining",
            drain_lane: None,
            drain_stage: "waiting_for_admitted_requests",
            drain_error: None,
            retry_drain: true,
        });
    }
    let task_state = state.clone();
    tokio::spawn(async move {
        run_pause(task_state).await;
    });
    result_outcome(status(&state.runtime))
}

pub(super) async fn resume(state: Arc<ModuleState>) -> HandlerOutcome {
    {
        let mut current = state
            .runtime
            .lane_pause
            .current
            .lock()
            .expect("pause state");
        if let Some(p) = current.as_mut() {
            p.deadline = tokio::time::Instant::now();
            p.until_ms = now_ms();
        }
    }
    state.runtime.lane_pause.changed.notify_one();
    result_outcome(status(&state.runtime))
}

fn phase(runtime: &RuntimeState, phase: &'static str) {
    runtime
        .lane_pause
        .current
        .lock()
        .expect("pause state")
        .as_mut()
        .expect("active pause")
        .phase = phase;
}

async fn drain(state: &Arc<ModuleState>) -> Result<(), WireOperationError> {
    #[cfg(test)]
    if let Some((lane, message)) = state
        .runtime
        .lane_pause
        .drain_failure
        .lock()
        .unwrap()
        .as_ref()
    {
        drain_progress(&state.runtime, Some(lane), "injected_failure", true);
        state
            .runtime
            .lane_pause
            .drain_attempts
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        return Err(WireOperationError::from_stable(
            StableError::engine_crashed(Some(1000))
                .with_details(serde_json::from_value(json!({"lane_id":lane})).unwrap()),
            message,
        ));
    }
    // The global fence covers the interval between resolution and execution.
    // Per-lane locks additionally cover detached loads, checks and reply drains.
    for lane in lane_ids(&state.runtime) {
        drain_progress(
            &state.runtime,
            Some(&lane),
            "waiting_for_lane_lifecycle",
            true,
        );
        let lifecycle = catalog_lane_lock(&state.runtime, &lane).lock_owned().await;
        drain_progress(
            &state.runtime,
            Some(&lane),
            "waiting_for_lane_serving",
            true,
        );
        let serving = lane_serving_gate(&state.runtime, &lane).write_owned().await;
        if let Some(model) = model_slot_snapshot(&state.runtime, &lane).and_then(|s| s.loaded) {
            drain_progress(&state.runtime, Some(&lane), "unloading_worker", false);
            // A deadline can detach the blocking unload. Publish non-readiness
            // before starting it, and let its owner retain both lane guards until
            // completion, so a resumed caller never selects the retired model.
            set_model_slot_state(&state.runtime, &lane, ModelRuntimeState::Unloaded);
            tokio::task::spawn_blocking(move || {
                let _guards = (lifecycle, serving);
                unload_embedding_model_blocking(model.clone())?;
                if let EmbedBackend::Worker(engine) = &model.backend {
                    engine
                        .lock()
                        .map_err(|_| transient_model_load_error("worker mutex poisoned"))?
                        .shutdown_for_pause()
                        .map_err(|e| transient_model_load_error(e.to_string()))?;
                }
                Ok::<_, WireOperationError>(())
            })
            .await
            .map_err(|e| transient_model_load_error(format!("pause unload: {e}")))??;
        }
    }
    drain_progress(&state.runtime, None, "releasing_global_lock", true);
    #[cfg(unix)]
    if let Some(supervisor) = state
        .runtime
        .ane_supervisor
        .lock()
        .expect("ANE supervisor")
        .as_ref()
    {
        supervisor
            .release_lane()
            .map_err(ane_residency_error_to_wire)?;
    }
    Ok(())
}

fn drain_progress(runtime: &RuntimeState, lane: Option<&str>, stage: &'static str, retry: bool) {
    let mut current = runtime.lane_pause.current.lock().expect("pause state");
    let pause = current.as_mut().expect("active pause");
    pause.drain_lane = lane.map(str::to_owned);
    pause.drain_stage = stage;
    pause.retry_drain = retry;
}

fn drain_failed(runtime: &RuntimeState, error: &WireOperationError) -> bool {
    let mut current = runtime.lane_pause.current.lock().expect("pause state");
    let pause = current.as_mut().expect("active pause");
    tracing::warn!(lane_id = ?pause.drain_lane, stage = pause.drain_stage,
        code = %error.code, message = %error.message, "Neural Engine pause drain failed");
    pause.phase = "drain_failed";
    pause.drain_error = Some(json!({"lane_id":pause.drain_lane,"stage":pause.drain_stage,
        "code":error.code,"message":error.message}));
    pause.retry_drain
}

/// Keep one drain attempt alive across notifications, but stop waiting at its
/// current deadline. Resume moves that deadline to now and wakes this wait.
async fn until_pause_end<T>(
    runtime: &RuntimeState,
    work: impl std::future::Future<Output = T>,
) -> Option<T> {
    tokio::pin!(work);
    loop {
        let notified = runtime.lane_pause.changed.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        let deadline = runtime
            .lane_pause
            .current
            .lock()
            .expect("pause state")
            .as_ref()
            .expect("active pause")
            .deadline;
        if tokio::time::Instant::now() >= deadline {
            return None;
        }
        tokio::select! {
            biased;
            _ = tokio::time::sleep_until(deadline) => return None,
            _ = notified => {},
            result = &mut work => return Some(result),
        }
    }
}

async fn run_pause(state: Arc<ModuleState>) {
    let gate = until_pause_end(
        &state.runtime,
        state.runtime.lane_pause.gate.clone().write_owned(),
    )
    .await;
    let drained = if gate.is_some() {
        until_pause_end(&state.runtime, async {
            loop {
                match drain(&state).await {
                    Ok(()) => break,
                    Err(error) => {
                        if drain_failed(&state.runtime, &error) {
                            tokio::time::sleep(Duration::from_secs(1)).await;
                        } else {
                            // A failed unload may already have retired its worker.
                            // Do not mistake an emptied slot for confirmed exit on
                            // the next retry and release the hardware lock.
                            std::future::pending::<()>().await;
                        }
                    }
                }
            }
        })
        .await
        .is_some()
    } else {
        false
    };
    if drained {
        phase(&state.runtime, "paused");
        loop {
            let notified = state.runtime.lane_pause.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let deadline = state
                .runtime
                .lane_pause
                .current
                .lock()
                .expect("pause state")
                .as_ref()
                .expect("active pause")
                .deadline;
            if tokio::time::Instant::now() >= deadline {
                break;
            }
            tokio::select! { _ = tokio::time::sleep_until(deadline) => {}, _ = notified => {} }
        }
    } else {
        let current = state
            .runtime
            .lane_pause
            .current
            .lock()
            .expect("pause state");
        let pause = current.as_ref().expect("active pause");
        let error = pause.drain_error.clone().unwrap_or_else(|| json!({
            "code":"drain_interrupted", "message":"deadline or resume reached before drain completed"}));
        tracing::warn!(lane_id = ?pause.drain_lane, lane_ids = ?pause.lanes,
            stage = pause.drain_stage, %error,
            "Neural Engine pause drain ended at deadline or resume without freeing hardware");
    }
    phase(&state.runtime, "waiting_for_lock");
    loop {
        #[cfg(unix)]
        let acquired = reacquire(&state.runtime);
        #[cfg(not(unix))]
        let acquired = Ok::<_, std::io::Error>(());
        match acquired {
            Ok(()) => break,
            Err(error) => {
                tracing::warn!(%error, "Neural Engine pause waiting to reacquire global lane lock");
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }
    }
    // Release the admission write guard before clearing pause state, under the
    // same mutex as admission. A new request must never see running while the
    // guard still prevents it from entering.
    let mut current = state
        .runtime
        .lane_pause
        .current
        .lock()
        .expect("pause state");
    drop(gate);
    *current = None;
}

#[cfg(unix)]
fn reacquire(runtime: &RuntimeState) -> Result<(), worker_host::ane_residency::AneResidencyError> {
    let shared = runtime.ane_supervisor.lock().expect("ANE supervisor");
    // Even a cold module must acquire the runtime-configured lock on expiry.
    // Unit tests exercise this Unix path on Linux too, so isolation cannot hide
    // behind the macOS hardware gate.
    #[cfg(any(target_os = "macos", test))]
    let shared = {
        let mut shared = shared;
        if shared.is_none() {
            *shared = Some(acquire_runtime_ane_supervisor(runtime)?);
        }
        shared
    };
    shared.as_ref().map_or(Ok(()), |s| s.reacquire_lane())
}
