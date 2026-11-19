#[cfg(unix)]
use super::worker_host::ane_residency::AneShapeWorker;
use super::{channel_error, media_jobs, result_outcome, EmbedBackend, ModuleState, RuntimeState};
use crate::transcribe_host::CertifyCounter;
use serde_json::{json, Value};
use std::{collections::BTreeMap, sync::Arc};
use subc_client_rs::HandlerOutcome;
use synapse_core::EmbedEngine;

pub(super) async fn handle(state: Arc<ModuleState>) -> HandlerOutcome {
    if !state.runtime.certify_observation {
        return channel_error(
            "certify_observation_disabled",
            "certification observation mode is disabled",
        );
    }
    result_outcome(snapshot(&state.runtime).await)
}

pub(super) async fn snapshot(runtime: &RuntimeState) -> Value {
    let transcribe = media_jobs::certify_counters(runtime).await;
    let models = runtime.loaded_models();
    // A lane that isn't loaded has done no engine work, so it reports nothing
    // and the snapshot stays complete. Only a loaded backend that can't report
    // its counter makes the snapshot unavailable. Certification reads this
    // before a fresh store has loaded anything, to prove that a refusal
    // happened without loading the engine.
    let mut available = true;
    let mut requests = BTreeMap::new();
    let mut in_process = BTreeMap::new();
    for (lane, counter) in transcribe {
        match counter {
            Some(CertifyCounter::WorkerRequests(id, count)) => {
                requests.insert(id, count);
            }
            Some(CertifyCounter::InProcessEngineInvocations(count)) => {
                in_process.insert(lane, count);
            }
            None => available = false,
        }
    }
    for model in models {
        match &model.backend {
            EmbedBackend::Owned(_) => {}
            #[cfg(unix)]
            EmbedBackend::DirectAne(engine) => {
                requests.insert(
                    engine.serving.channel.worker_id().to_owned(),
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
            #[cfg(feature = "test-support")]
            EmbedBackend::TestDeterministic(_) => available = false,
        }
    }
    // Direct ANE placement evidence comes from the shared residency supervisor;
    // transport counters alone cannot certify where its admitted layers execute.
    let observations = runtime
        .ane_supervisor
        .lock()
        .ok()
        .and_then(|s| s.as_ref().and_then(|s| s.observations()));
    let (inventories, admitted_count) = observations.unwrap_or_default();
    json!({
        "inventories": inventories,
        "worker_requests": requests,
        "in_process_transcribe_engine_invocations": in_process,
        "admitted_count": admitted_count,
        "available": available,
    })
}
