//! Local batch media jobs. SQLite owns admission and terminality; a blocking
//! lane owner drains inference even when its caller cancels or disconnects.
use super::*;
use base64::{engine::general_purpose::STANDARD, Engine as _};
use cortexkit_role_media::{
    describe::Kind,
    describe::{Engine, LatencyClass},
    jobs::SubmitRequest,
    objects,
};
use rusqlite::{params, OptionalExtension};
fn base64_encode(bytes: &[u8]) -> String {
    STANDARD.encode(bytes)
}
use std::io::{Read, Seek, SeekFrom, Write};

mod persistence;
#[cfg(test)]
mod tests;

const RETENTION_MS: u64 = 3_600_000;
const UPLOAD_MAX: u64 = 256 * 1024 * 1024;
const STORAGE_MAX: u64 = 1024 * 1024 * 1024;
const LANE_MAX: u64 = 16;
const PAGE: usize = 512 * 1024;

#[derive(Default)]
pub(super) struct Runtime {
    gate: Mutex<()>,
    lanes: Mutex<BTreeMap<String, Arc<tokio::sync::Mutex<Lane>>>>,
    #[cfg(test)]
    runner: Mutex<Option<Arc<TestRunner>>>,
    #[cfg(test)]
    clock: Mutex<Option<u64>>,
}
#[cfg(test)]
type TestRunner = dyn Fn(&str, &[u8]) -> Result<Value, Fault> + Send + Sync;

fn clock_ms(state: &ModuleState) -> u64 {
    #[cfg(test)]
    if let Some(now) = *state.runtime.media.clock.lock().unwrap() {
        return now;
    }
    let _ = state;
    now_ms()
}

#[derive(Default)]
struct Lane {
    host: Option<transcribe_host::TranscribeHost>,
    retry: Option<(Instant, Fault)>,
}
#[derive(Clone, Debug)]
struct Fault(Value);
type Answer<T> = Result<T, Fault>;
fn fault(code: &str, field: Option<&str>) -> Fault {
    let transient = matches!(
        code,
        "queue_full"
            | "model_loading"
            | "engine_crashed"
            | "lane_paused"
            | "module_restarted"
            | "deadline_exceeded"
            | "job_not_done"
            | "engine_empty_result"
    );
    let mut detail = json!({"class": if transient {"transient"} else {"permanent"}});
    if transient && code != "job_not_done" {
        detail["retry_after_ms"] = json!(1000);
    }
    if let Some(field) = field {
        detail["field"] = json!(field);
    }
    Fault(json!({"code":code,"message":code,"detail":detail}))
}
fn input_limit(limit: &str) -> Fault {
    let mut e = fault("input_too_large", Some("input.audio"));
    e.0["detail"]["limit"] = json!(limit);
    e
}
fn invalid(field: &str) -> Fault {
    fault("invalid_params", Some(field))
}
fn unsupported(field: &str) -> Fault {
    use cortexkit_role_media::errors::{Refusal, RefusalClass, RefusalDetail};
    let detail = RefusalDetail::new(RefusalClass::Permanent).with_members(BTreeMap::from([(
        "field".into(),
        Value::String(field.into()),
    )]));
    Fault(
        serde_json::to_value(Refusal::new(
            "unsupported".into(),
            "unsupported".into(),
            detail,
        ))
        .expect("media refusal serializes"),
    )
}
fn internal(error: impl std::fmt::Display) -> Fault {
    let mut e = fault("store_failure", None);
    e.0["message"] = json!(error.to_string());
    e
}
fn wire(error: WireOperationError) -> Fault {
    let mut e = fault(&error.code, None);
    e.0["detail"]["class"] = json!(error.class);
    if let Some(retry) = error.retry_after_ms {
        e.0["detail"]["retry_after_ms"] = json!(retry);
    }
    if let Some(details) = error.details {
        if let Some(map) = details.as_object() {
            for (k, v) in map {
                e.0["detail"][k] = v.clone();
            }
        }
    }
    e.0["message"] = json!(error.message.chars().take(1024).collect::<String>());
    e
}
fn outcome(answer: Answer<Value>) -> HandlerOutcome {
    match answer {
        Ok(v) => HandlerOutcome::Response(
            cortexkit_role_media::wire::reply_body(&v).expect("media reply"),
        ),
        Err(e) => HandlerOutcome::ErrorWithDetail {
            code: e.0["code"].as_str().unwrap_or("media_error").into(),
            message: e.0["message"].as_str().unwrap_or("media error").into(),
            detail: e.0["detail"].clone(),
        },
    }
}
fn parse<T: serde::de::DeserializeOwned>(v: Value, field: &str) -> Answer<T> {
    serde_json::from_value(v).map_err(|_| invalid(field))
}
fn string<'a>(v: &'a Value, field: &str) -> Answer<&'a str> {
    v[field].as_str().ok_or_else(|| invalid(field))
}

fn engine(model: &str, duration: u64, entry: &catalog::CatalogEntry) -> Engine {
    let mut engine = Engine::new(
        Kind::Transcribe,
        "local".into(),
        model.into(),
        true,
        vec!["audio/wav".into()],
        LatencyClass::Batch,
    )
    .with_max_input_bytes(UPLOAD_MAX)
    .with_max_duration_ms(duration);
    if let Some(languages) = &entry.languages {
        engine = engine.with_languages(languages.clone());
    }
    engine
}
fn engines(runtime: &RuntimeState) -> Vec<Engine> {
    let mut engines = Vec::new();
    for entry in runtime
        .release_catalog
        .models
        .iter()
        .filter(|e| e.task == "transcribe")
    {
        let runnable: Vec<_> = entry
            .backends
            .iter()
            .filter(|b| {
                catalog_lane_backend_reason(runtime, entry, b, synapse_core::Platform::current())
                    .is_none()
            })
            .collect();
        if let Some(first) = runnable.first() {
            engines.push(engine(
                &entry.id,
                first.effective_max_duration_ms().unwrap_or(0),
                entry,
            ));
        }
        for backend in runnable {
            engines.push(engine(
                &catalog::lane_id(&entry.id, &backend.backend),
                backend.effective_max_duration_ms().unwrap_or(0),
                entry,
            ));
        }
    }
    engines
}
pub(super) fn declaration(runtime: &RuntimeState) -> Value {
    json!({"groups":["jobs"],"jobs_survive_restart":false,"engines":engines(runtime)})
}
pub(super) fn start_maintenance(state: &Arc<ModuleState>) {
    let state = Arc::downgrade(state);
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(60)).await;
            let Some(state) = state.upgrade() else {
                break;
            };
            let _ = tokio::task::spawn_blocking(move || {
                let _gate = state.runtime.media.gate.lock().expect("media gate");
                if let Err(error) = persistence::sweep(&state, clock_ms(&state)) {
                    tracing::warn!(error=%error.0,"media retention sweep failed");
                }
            })
            .await;
        }
    });
}

pub(super) fn startup(state: &ModuleState) -> Result<(), ModuleError> {
    persistence::recover(state).map_err(|e| ModuleError::Config(e.0.to_string()))
}

pub(super) async fn handle(
    state: Arc<ModuleState>,
    principal: Option<&str>,
    method: &str,
    params: Value,
) -> HandlerOutcome {
    let Some(owner) = principal else {
        return outcome(Err(fault("principal_unavailable", None)));
    };
    if method == "role.describe" {
        return outcome(Ok(declaration(&state.runtime)));
    }
    let answer = if method == "media.submit" {
        submit(state.clone(), owner, params).await
    } else if method == "certify.transcribe_observe_rerun" {
        observe_rerun(state.clone(), owner, params).await
    } else if method == "certify.transcribe_measurements" {
        persistence::measurements(&state, owner, params)
    } else {
        let _gate = state.runtime.media.gate.lock().expect("media gate");
        persistence::sweep(&state, clock_ms(&state)).and_then(|_| match method {
            "media.upload_begin" => persistence::upload_begin(&state, owner, params),
            "media.upload_append" => persistence::upload_append(&state, owner, params),
            "media.upload_status" => persistence::upload_status(&state, owner, params),
            "media.upload_commit" => persistence::upload_commit(&state, owner, params),
            "media.read" => persistence::read(&state, owner, params),
            "media.status" | "media.result" | "media.cancel" => {
                persistence::job_answer(&state, owner, method, params)
            }
            "certify.transcribe_observation" => persistence::observation(&state, owner, params),
            _ => Err(fault("unknown_method", None)),
        })
    };
    outcome(answer)
}

fn select(
    state: &ModuleState,
    model: &str,
) -> Answer<(catalog::CatalogEntry, catalog::CatalogBackend)> {
    let (entry, pinned) = state
        .runtime
        .release_catalog
        .resolve_reserved(model)
        .ok_or_else(|| unsupported("model"))?;
    if entry.task != "transcribe" {
        return Err(unsupported("model"));
    }
    let backend = if let Some(pinned) = pinned {
        entry.backend(pinned)
    } else {
        entry.backends.iter().find(|b| {
            catalog_lane_backend_reason(&state.runtime, entry, b, synapse_core::Platform::current())
                .is_none()
        })
    }
    .ok_or_else(|| unsupported("model"))?;
    if catalog_lane_backend_reason(
        &state.runtime,
        entry,
        backend,
        synapse_core::Platform::current(),
    )
    .is_some()
    {
        return Err(unsupported("model"));
    }
    Ok((entry.clone(), backend.clone()))
}

async fn submit(state: Arc<ModuleState>, owner: &str, params: Value) -> Answer<Value> {
    if serde_json::to_vec(&params).map_err(internal)?.len() > cortexkit_role_media::MESSAGE_MAX {
        return Err(invalid("params"));
    }
    let client = string(&params, "client_job_id")?;
    if !(1..=128).contains(&client.chars().count()) {
        return Err(invalid("client_job_id"));
    }
    let binding = persistence::binding(&params);
    let admitted = {
        let _gate = state.runtime.media.gate.lock().expect("media gate");
        persistence::sweep(&state, clock_ms(&state))?;
        if let Some(id) = persistence::replay(&state, owner, client, &binding)? {
            return Ok(json!({"job_id":id}));
        }
        let request: SubmitRequest = parse(params.clone(), "params")?;
        if request.kind != Kind::Transcribe {
            return Err(unsupported("kind"));
        }
        let cortexkit_role_media::jobs::JobInput::Transcribe(input) = &request.input else {
            return Err(invalid("input"));
        };
        if request.model.chars().count() > 256 {
            return Err(invalid("model"));
        }
        if request
            .decision_id
            .as_ref()
            .is_some_and(|v| v.chars().count() > 256)
        {
            return Err(invalid("decision_id"));
        }
        let _disk = state.runtime.catalog_disk.lock().expect("catalog disk");
        let (entry, backend) = select(&state, &request.model)?;
        let lane = catalog::lane_id(&entry.id, &backend.backend);
        // Choose the lane before checking pause, install or quarantine. None of
        // these transient states can redirect a catalog id to a different lane.
        let pause = lane_pause::admit(&state.runtime, &lane).map_err(wire)?;
        let serving = lane_serving_gate(&state.runtime, &lane)
            .try_read_owned()
            .map_err(|_| fault("model_loading", None))?;
        let guard = (pause, serving);
        if current_catalog_install(&state, &entry, &backend.backend)
            .map_err(wire)?
            .is_none()
        {
            let mut e = fault("model_not_installed", None);
            e.0["detail"]["download_op"] = json!("models.download");
            return Err(e);
        }
        let check = catalog_self_check_projection(&state, &entry, &backend).map_err(wire)?;
        if check["state"] == "failed" {
            return Err(wire(catalog_self_check_failed(
                &entry,
                &backend,
                check["check_id"].as_str().unwrap_or(""),
                &check["reason"],
            )));
        }
        if let Ok(mut cached) = lane_cache(&state.runtime, &lane).try_lock() {
            if let Some((until, error)) = &cached.retry {
                if Instant::now() < *until {
                    return Err(error.clone());
                }
            }
            cached.retry = None;
            if let Some(retry) = cached
                .host
                .as_ref()
                .and_then(|host| host.quarantine_retry_after_ms())
            {
                return Err(quarantined(&lane, retry));
            }
        }
        let audio = input.audio.clone();
        objects::check_media(&audio).map_err(|e| Fault(serde_json::json!({"code":e.code,"message":e.code,"detail":{"class":"permanent","field":"audio"}})))?;
        if audio.bytes > UPLOAD_MAX {
            return Err(input_limit("max_input_bytes"));
        }
        persistence::check_lane_capacity(&state, &lane)?;
        let bytes = persistence::audio_bytes(&state, owner, &audio)?;
        if !cortexkit_role_media::describe::mime_matches("audio/wav", &audio.mime) {
            return Err(unsupported("input.audio.mime"));
        }
        if input.language.as_ref().is_some_and(|hint| {
            entry.languages.as_ref().is_none_or(|languages| {
                !languages.iter().any(|language| {
                    cortexkit_role_media::describe::language_matches(language, hint)
                })
            })
        }) {
            return Err(unsupported("input.language"));
        }
        if input.bias.is_some() {
            return Err(unsupported("input.bias"));
        }
        let decoded = synapse_speech::decode_wav(&bytes).map_err(|_| invalid("audio"))?;
        let profile =
            transcribe_host::TranscribeProfile::pinned(backend.profile.as_deref().unwrap_or(""))
                .map_err(internal)?;
        let samples = (decoded.samples.len() as u64 * 16_000).div_ceil(decoded.sample_rate as u64);
        let frames = synapse_speech::valid_encoder_frames(samples);
        let max_duration_ms = backend
            .effective_max_duration_ms()
            .expect("transcribe duration");
        if audio
            .duration_ms
            .is_some_and(|duration| duration > max_duration_ms)
        {
            return Err(input_limit("max_duration_ms"));
        }
        let duration_failure = if decoded.duration_ms() > max_duration_ms
            || frames > profile.max_encoder_frames as u64
        {
            let mut e = input_limit("max_duration_ms");
            e.0["detail"]["duration_ms"] = json!(decoded.duration_ms());
            e.0["detail"]["max_duration_ms"] = json!(max_duration_ms);
            e.0["detail"]["valid_encoder_frames"] = json!(frames);
            e.0["detail"]["max_encoder_frames"] = json!(profile.max_encoder_frames);
            Some(e)
        } else {
            None
        };
        let id = persistence::admit(&state, owner, &request, &binding, &lane, &audio, &bytes)?;
        (id, entry, backend, guard, duration_failure)
    };
    let (id, entry, backend, guard, duration_failure) = admitted;
    let job_id = id.clone();
    tokio::spawn(async move {
        let lane = catalog::lane_id(&entry.id, &backend.backend);
        let lock = lane_cache(&state.runtime, &lane);
        let mut owner = lock.lock_owned().await;
        let _ = tokio::task::spawn_blocking(move || {
            let cached = &mut *owner;
            let _pause_guard = guard;
            if !persistence::start(&state, &job_id).unwrap_or(false) {
                return;
            }
            let result = if let Some(error) = duration_failure {
                Err(error)
            } else {
                persistence::job_audio(&state, &job_id)
                    .and_then(|bytes| run(&state, &entry, &backend, &bytes, cached, true))
            };
            let _gate = state.runtime.media.gate.lock().expect("media gate");
            if let Err(error) = persistence::finish(&state, &job_id, result) {
                tracing::error!(error = %error.0, "media job completion failed");
            }
        })
        .await;
    });
    Ok(json!({"job_id":id}))
}

fn run(
    state: &ModuleState,
    entry: &catalog::CatalogEntry,
    backend: &catalog::CatalogBackend,
    audio: &[u8],
    cached: &mut Lane,
    observe: bool,
) -> Answer<Value> {
    #[cfg(test)]
    {
        let run = state.runtime.media.runner.lock().unwrap().clone();
        if let Some(run) = run.filter(|_| observe) {
            return checked_output(run(&catalog::lane_id(&entry.id, &backend.backend), audio)?);
        }
    }
    let lane = catalog_transcribe_lane(entry, backend).map_err(wire)?;
    if cached.host.is_none() {
        match transcribe_self_check::ensure(
            state,
            &lane,
            || load_host(state, entry, backend),
            |_| {},
        )
        .map_err(wire)
        {
            Ok(host) => {
                cached.host = Some(host);
                cached.retry = None;
            }
            Err(error) => {
                if error.0["code"] == "engine_crashed" {
                    let delay = error.0["detail"]["retry_after_ms"].as_u64().unwrap_or(1000);
                    cached.retry =
                        Some((Instant::now() + Duration::from_millis(delay), error.clone()));
                }
                return Err(error);
            }
        }
    }
    let host = cached.host.as_ref().expect("loaded speech host");
    let result = if observe {
        host.transcribe(audio, |_| {})
    } else {
        host.transcribe_self_check(audio, |_| {})
    }
    .map_err(|e| {
        if let Some(retry) = host.quarantine_retry_after_ms() {
            return quarantined(&catalog::lane_id(&entry.id, &backend.backend), retry);
        }
        match e {
            transcribe_host::TranscribeHostError::Audio(_)
            | transcribe_host::TranscribeHostError::DurationLimit(_) => invalid("audio"),
            _ => fault("engine_crashed", None),
        }
    })?;
    if !observe {
        return Ok(
            json!({"observe":false,"valid_encoder_frames":result.output.valid_encoder_frames,
            "triples":base64_encode(&result.transcript_frame)}),
        );
    }
    let words = result
        .transcript
        .words
        .iter()
        .map(|w| json!({"text":w.word,"start_ms":w.start_ms as u64,"end_ms":w.end_ms as u64}))
        .collect::<Vec<_>>();
    let mut output = json!({"text":result.transcript.text,"words":words,"no_speech":result.transcript.no_speech});
    output = checked_output(output)?;
    if state.runtime.certify_observation {
        let self_check_wall_ms = transcribe_self_check::wall_ms(state, &lane).map_err(wire)?;
        output["_measurements"] = measurement_snapshot(
            &backend.backend,
            &result,
            self_check_wall_ms,
            host.ane_shapes(),
        );
        output["_fingerprint"] = json!(backend.fingerprint);
        if let Some(frames) = result.observation {
            // Raw observation bytes are internal until completion atomically
            // stores bounded pages and publishes only their descriptors.
            output["_observation"] = json!({"lane_id":catalog::lane_id(&entry.id,&backend.backend),"fingerprint":backend.fingerprint,"valid_encoder_frames":result.output.valid_encoder_frames,"triples":base64_encode(&result.transcript_frame),"encoder_output":base64_encode(&frames.encoder_output),"decode_trace":base64_encode(&frames.decode_trace)});
        }
    }
    Ok(output)
}

fn measurement_snapshot(
    backend: &str,
    result: &transcribe_host::TranscribeResult,
    self_check_wall_ms: Option<u64>,
    ane_shapes: Vec<Value>,
) -> Value {
    json!({
        "self_check_wall_ms":self_check_wall_ms,
        "longest_progress_interval_ms":result.timing.longest_ms(),
        "ane_shapes":if backend == "ane" {ane_shapes} else {Vec::new()},
        "peak_device_memory_bytes":if matches!(backend, "cuda" | "vulkan") {result.peak_device_memory_bytes} else {None},
        "peak_storage_buffer_bytes":if backend == "vulkan" {result.peak_storage_buffer_bytes} else {None}
    })
}

async fn observe_rerun(state: Arc<ModuleState>, owner: &str, params: Value) -> Answer<Value> {
    if !state.runtime.certify_observation {
        return Err(fault("certify_observation_disabled", None));
    }
    let (entry, backend, bytes, object, guard) = {
        let _gate = state.runtime.media.gate.lock().expect("media gate");
        let input = persistence::rerun_input(&state, owner, &params)?;
        let _disk = state.runtime.catalog_disk.lock().expect("catalog disk");
        let (entry, backend) = select(&state, &input.lane)?;
        if backend.fingerprint != input.fingerprint {
            return Err(fault("object_mismatch", Some("job_id")));
        }
        let pause = lane_pause::admit(&state.runtime, &input.lane).map_err(wire)?;
        let serving = lane_serving_gate(&state.runtime, &input.lane)
            .try_read_owned()
            .map_err(|_| fault("model_loading", None))?;
        if current_catalog_install(&state, &entry, &backend.backend)
            .map_err(wire)?
            .is_none()
        {
            return Err(fault("model_not_installed", None));
        }
        persistence::check_lane_capacity(&state, &input.lane)?;
        persistence::claim_rerun(&state, &input.object)?;
        (entry, backend, input.bytes, input.object, (pause, serving))
    };
    // The detached task owns the upload and lane guards even if the caller stops waiting.
    tokio::spawn(async move {
        let lock = lane_cache(
            &state.runtime,
            &catalog::lane_id(&entry.id, &backend.backend),
        );
        let mut cached = lock.lock_owned().await;
        tokio::task::spawn_blocking(move || {
            let _guard = guard;
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run(&state, &entry, &backend, &bytes, &mut cached, false)
            }))
            .unwrap_or_else(|_| Err(fault("engine_crashed", None)));
            let _gate = state.runtime.media.gate.lock().expect("media gate");
            persistence::delete_rerun(&state, &object)?;
            result
        })
        .await
        .map_err(internal)?
    })
    .await
    .map_err(internal)?
}

fn checked_output(output: Value) -> Answer<Value> {
    let typed: cortexkit_role_media::jobs::TranscribeOutput = parse(output.clone(), "output")?;
    cortexkit_role_media::jobs::check_transcribe_output(&typed)
        .map_err(|_| fault("engine_empty_result", None))?;
    Ok(output)
}

fn load_host(
    state: &ModuleState,
    entry: &catalog::CatalogEntry,
    backend: &catalog::CatalogBackend,
) -> Result<transcribe_host::TranscribeHost, transcribe_host::TranscribeHostError> {
    use transcribe_host::{
        ObservationMode, TranscribeHost, TranscribeHostError, TranscribeProfile,
    };
    let err = |e: WireOperationError| TranscribeHostError::Profile(e.message);
    let spec = catalog_profile_lane_spec(
        state,
        entry,
        backend,
        backend.profile.as_deref().unwrap_or(""),
        true,
    )
    .map_err(err)?;
    let path = locator_path(&spec.model_locator, &state.model_cache).map_err(err)?;
    let config = model_runtime_config(
        &spec,
        &path.path,
        state.model_cache.root(),
        state.runtime.microllm_max_tokens,
    );
    let artifact = ValidatedArtifact {
        digest: spec.artifact_digest.clone(),
        format: spec.artifact_format.clone(),
    };
    let profile = TranscribeProfile::pinned(backend.profile.as_deref().unwrap_or(""))?;
    let mode = if state.runtime.certify_observation {
        ObservationMode::Certification
    } else {
        ObservationMode::Serving
    };
    if backend.backend == "metal" {
        return TranscribeHost::load_metal(profile, &artifact, &config, mode);
    }
    let binary = spec
        .worker_bin
        .clone()
        .or_else(|| env::var_os(worker_binary_env_var(&spec.engine)).map(PathBuf::from))
        .or_else(|| resolve_worker_binary_sibling(&spec.engine))
        .ok_or_else(|| TranscribeHostError::Profile("worker binary unavailable".into()))?;
    let directory = persistence::directory(state)
        .map_err(|e| TranscribeHostError::Profile(e.0.to_string()))?
        .join("workers");
    let mut worker = worker_host::WorkerHostConfig::new(binary, directory);
    worker.worker_id = format!("synapse-{}-{}", spec.engine, spec.model_id);
    worker.model_id = Some(spec.model_id.clone());
    worker.catalog_lane = Some(spec.model_id.clone());
    worker.engine_identity = Some(spec.engine_identity.clone());
    worker.worker_forward_lines_per_sec = state.runtime.log.worker_forward_lines_per_sec;
    #[cfg(unix)]
    if backend.backend == "ane" {
        let supervisor = direct_ane_supervisor(&state.runtime, &spec.engine)
            .map_err(err)?
            .ok_or_else(|| TranscribeHostError::Profile("ANE supervisor unavailable".into()))?;
        return TranscribeHost::load_ane(profile, worker, supervisor, &artifact, &config, mode);
    }
    TranscribeHost::load_worker(profile, worker, &artifact, &config, mode)
}

fn lane_cache(runtime: &RuntimeState, lane: &str) -> Arc<tokio::sync::Mutex<Lane>> {
    runtime
        .media
        .lanes
        .lock()
        .expect("media lanes")
        .entry(lane.into())
        .or_default()
        .clone()
}
fn quarantined(lane: &str, retry: u64) -> Fault {
    let mut e = fault("engine_crashed", None);
    e.0["detail"]["lane_id"] = json!(lane);
    e.0["detail"]["quarantined"] = json!(true);
    e.0["detail"]["retry_after_ms"] = json!(retry);
    e
}
pub(super) async fn unload(
    runtime: &RuntimeState,
    model: Option<&str>,
    neural_only: bool,
) -> Result<(), WireOperationError> {
    let lanes = runtime
        .media
        .lanes
        .lock()
        .expect("media lanes")
        .iter()
        .filter(|(id, _)| {
            let matches = model.is_none_or(|model| {
                model == id.as_str()
                    || runtime
                        .release_catalog
                        .resolve_reserved(model)
                        .is_some_and(|(e, b)| b.is_none() && id.starts_with(&format!("{}-", e.id)))
            });
            matches && (!neural_only || id.ends_with("-ane"))
        })
        .map(|(_, lane)| lane.clone())
        .collect::<Vec<_>>();
    for lane in lanes {
        let mut lane = lane.lock().await;
        if let Some(mut host) = lane.host.take() {
            tokio::task::spawn_blocking(move || host.unload())
                .await
                .map_err(|e| transient_model_load_error(e.to_string()))?
                .map_err(|e| transient_model_load_error(e.to_string()))?;
        }
        lane.retry = None;
    }
    Ok(())
}

pub(super) fn holders(
    state: &ModuleState,
    entry: &catalog::CatalogEntry,
) -> Result<Vec<Value>, WireOperationError> {
    let prefix = format!("{}-", entry.id);
    let mut holders = Vec::new();
    for (id, cache) in state
        .runtime
        .media
        .lanes
        .lock()
        .expect("media lanes")
        .iter()
        .filter(|(id, _)| id.starts_with(&prefix))
    {
        match cache.try_lock() {
            Ok(lane) if lane.host.is_none() => {}
            _ => holders.push(json!({"kind":"media_lane","lane_id":id})),
        }
    }
    let pending = state
        .store
        .sqlite()
        .with_conn(|c| {
            let mut q = c.prepare(
                "SELECT id FROM media_jobs WHERE lane LIKE ?1 AND state IN ('queued','running')",
            )?;
            let jobs = q
                .query_map([format!("{prefix}%")], |r| r.get::<_, String>(0))?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            Ok(jobs)
        })
        .map_err(catalog_store_error)?;
    holders.extend(
        pending
            .into_iter()
            .map(|id| json!({"kind":"media_job","job_id":id})),
    );
    Ok(holders)
}
