use super::*;
use cortexkit_role_media_conformance::{
    subject::{
        Capabilities, MediaSubject, Request, Script, ScriptStep, SubjectError, SubjectResult,
    },
    Outcome,
};
use std::{collections::VecDeque, sync::Condvar};

#[derive(Default)]
struct Control {
    steps: VecDeque<ScriptStep>,
    inputs: Vec<Value>,
    released: bool,
}

struct LocalSubject {
    state: Arc<ModuleState>,
    control: Arc<(Mutex<Control>, Condvar)>,
    held: Mutex<Option<tokio::sync::OwnedMutexGuard<Lane>>>,
}

impl LocalSubject {
    fn new(state: Arc<ModuleState>) -> Self {
        *state.runtime.media.clock.lock().unwrap() = Some(now_ms());
        Self {
            state,
            control: Arc::default(),
            held: Mutex::default(),
        }
    }

    fn release(&self) {
        self.held.lock().unwrap().take();
        let (lock, ready) = &*self.control;
        lock.lock().unwrap().released = true;
        ready.notify_all();
    }
}

impl Drop for LocalSubject {
    fn drop(&mut self) {
        self.release();
    }
}

#[async_trait]
impl MediaSubject for LocalSubject {
    fn capabilities(&self) -> Capabilities {
        Capabilities::new()
            .with_local_provider(true)
            .with_has_speech_check(true)
    }

    async fn configure_engines(&self, _: Vec<Engine>) -> SubjectResult<()> {
        // Synapse's catalog, not the kit's synthetic multi-kind registry, owns
        // declarations. Never manufacture engine facts to satisfy a kit case.
        Ok(())
    }

    async fn engine_declaration(&self) -> SubjectResult<Vec<Engine>> {
        serde_json::from_value(declaration(&self.state.runtime)["engines"].clone())
            .map_err(|e| SubjectError::new(e.to_string()))
    }

    async fn configure_peers(
        &self,
        _: cortexkit_role_media_conformance::peers::RecordingRoutingPeer,
        _: cortexkit_role_media_conformance::peers::RecordingVendor,
    ) -> SubjectResult<()> {
        Ok(())
    }

    async fn configure_principals(
        &self,
        _: cortexkit_role_media_conformance::subject::Principals,
    ) -> SubjectResult<()> {
        Ok(())
    }

    async fn call_as(&self, principal: &str, request: Request) -> SubjectResult<Value> {
        let mut params = request.params;
        // The only fixture-model translation is to the real local catalog id.
        if request.method == "media.submit" && params["model"] == "transcribe" {
            params["model"] = json!(MODEL);
        }
        match handle(self.state.clone(), Some(principal), &request.method, params).await {
            HandlerOutcome::Response(body) => {
                serde_json::from_slice(&body).map_err(|e| SubjectError::new(e.to_string()))
            }
            HandlerOutcome::ErrorWithDetail {
                code,
                message,
                detail,
            } => Err(SubjectError::refused(
                json!({"code":code,"message":message,"detail":detail}),
            )),
            other => Err(SubjectError::new(format!("unexpected {other:?}"))),
        }
    }

    async fn script_engine(&self, model: &str, script: Script) -> SubjectResult<()> {
        if model != "transcribe" {
            return Err(SubjectError::unsupported(
                "script_engine: model not declared",
            ));
        }
        if script.decoded_duration_ms.is_some() {
            return Err(SubjectError::unsupported(
                "script_engine: duration belongs to WAV decoder",
            ));
        }
        let control = self.control.clone();
        {
            let mut c = control.0.lock().unwrap();
            c.steps = script.steps.into();
            c.released = false;
        }
        *self.state.runtime.media.runner.lock().unwrap() = Some(Arc::new(move |_, audio| {
            let (lock, ready) = &*control;
            let mut c = lock.lock().unwrap();
            c.inputs.push(json!({"audio":media(audio)}));
            while let Some(step) = c.steps.pop_front() {
                match step {
                    ScriptStep::Output(output) => return Ok(output),
                    ScriptStep::Failure(f) => {
                        let mut detail = f["detail"].as_object().cloned().unwrap_or_default();
                        detail.insert("class".into(), f["class"].clone());
                        if let Some(retry) = f.get("retry_after_ms") {
                            detail.insert("retry_after_ms".into(), retry.clone());
                        }
                        return Err(Fault(
                            json!({"code":f["code"],"message":f["message"],"detail":detail}),
                        ));
                    }
                    ScriptStep::Pause => {
                        while !c.released {
                            c = ready.wait(c).unwrap();
                        }
                    }
                    _ => return Err(fault("unsupported_script_step", None)),
                }
            }
            Err(fault("script_exhausted", None))
        }));
        Ok(())
    }

    async fn engine_inputs(&self, _: &str) -> SubjectResult<Vec<Value>> {
        Ok(self.control.0.lock().unwrap().inputs.clone())
    }

    async fn hold_engine(&self, _: &str) -> SubjectResult<()> {
        let lane = catalog::lane_id(MODEL, &first(&self.state));
        let guard = lane_cache(&self.state.runtime, &lane).lock_owned().await;
        *self.held.lock().unwrap() = Some(guard);
        Ok(())
    }

    async fn release_engine(&self, _: &str) -> SubjectResult<()> {
        self.release();
        Ok(())
    }

    async fn restart(&self) -> SubjectResult<()> {
        let result = startup(&self.state).map_err(|e| SubjectError::new(e.to_string()));
        self.release();
        result
    }

    async fn advance_clock(&self, ms: u64) -> SubjectResult<()> {
        let now = {
            let mut clock = self.state.runtime.media.clock.lock().unwrap();
            let now = clock.unwrap().saturating_add(ms);
            *clock = Some(now);
            now
        };
        let _gate = self.state.runtime.media.gate.lock().unwrap();
        persistence::sweep(&self.state, now).map_err(|e| SubjectError::new(e.0.to_string()))
    }
}

#[tokio::test]
async fn media_published_transcribe_conformance_report() {
    use cortexkit_role_media_conformance::checks::{CASES, STREAM_REALTIME_CASES};
    let mut counts = [0; 3];
    for &name in CASES
        .iter()
        .filter(|name| !STREAM_REALTIME_CASES.contains(name))
    {
        let (root, state) = fixture(name, true);
        let subject = LocalSubject::new(state.clone());
        let outcome = cortexkit_role_media_conformance::run_case(&subject, name).await;
        println!("MEDIA_CONFORMANCE {name}: {outcome:?}");
        if let Outcome::Failed(reason) = &outcome {
            // Known conformance gaps are reported, not passing results. Any new
            // failure must be reviewed, including provider contract mismatches.
            let expected = match name {
                "wire_extension" => "describe omitted the engine with unknown enum values",
                "submit_refuses_outside_declared_facts" => {
                    "expected field input.format: {\"code\":\"unsupported\",\"detail\":{\"class\":\"permanent\",\"field\":\"kind\"}"
                }
                "input_too_large" => "media.submit accepted instead of input_too_large",
                "describe_matches_engine_declaration" | "describe_checker_codes" => {
                    "describe/declaration facts differ"
                }
                "empty_output_is_a_failure" | "chunked_result_checksum" => {
                    "subject hook script_engine: model not declared is not implemented"
                }
                "usage_feed_oversize_entry_acks" => {
                    "subject hook seed_usage_entry is not implemented"
                }
                _ => "usage.read: wire refusal; Some(Object {\"code\": String(\"unknown_method\")",
            };
            assert!(
                reason.contains(expected),
                "unexpected kit failure: {reason}"
            );
        }
        counts[match outcome {
            Outcome::Passed => 0,
            Outcome::NotApplicable(_) => 1,
            Outcome::Failed(_) => 2,
            _ => panic!("unknown conformance outcome"),
        }] += 1;
        drop(subject);
        cleanup(root, state).await;
    }
    println!(
        "MEDIA_CONFORMANCE totals: passed={}, not_applicable={}, failed={}",
        counts[0], counts[1], counts[2]
    );
    assert_eq!(counts, [0, 19, 27], "pinned kit applicability changed");
}
