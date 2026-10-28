//! Bounded diagnoses from existing certification, self-check and crash-fence evidence.

use super::*;

const DEFAULT_WINDOW_MS: u64 = 21_600_000;
const MAX_WINDOW_MS: u64 = 604_800_000;
const MAX_SAMPLES: usize = 5;
const MAX_REPLY_BYTES: usize = 65_536;

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Params {
    window_ms: Option<u64>,
}

fn parse_window(params: Value) -> Result<u64, ()> {
    let params = if params.is_null() {
        Params::default()
    } else {
        if !params.is_object() {
            return Err(());
        }
        serde_json::from_value::<Params>(params).map_err(|_| ())?
    };
    let window = params.window_ms.unwrap_or(DEFAULT_WINDOW_MS);
    (1..=MAX_WINDOW_MS)
        .contains(&window)
        .then_some(window)
        .ok_or(())
}

pub(super) fn handle(state: &ModuleState, params: Value) -> HandlerOutcome {
    let window = match parse_window(params) {
        Ok(window) => window,
        Err(()) => {
            return channel_error(
                "invalid_request",
                "diagnostics.findings params must be an object or null; window_ms must be an integer in 1..=604800000 or null; unknown params are refused",
            );
        }
    };
    match reply(state, window, now_ms()) {
        Ok(result) => result_outcome(result),
        Err(()) => channel_error(
            "store_failure",
            "diagnostics.findings could not read health state",
        ),
    }
}

fn finding(
    check: &str,
    severity: &str,
    evidence: Vec<(u64, Value)>,
    generated_at_ms: u64,
    summary: &str,
    hint: &str,
) -> Option<Value> {
    if evidence.is_empty() {
        return None;
    }
    let first_seen_ms = evidence.iter().map(|(at, _)| *at).min().unwrap();
    Some(json!({
        "check": check,
        "key": check,
        "severity": severity,
        "count": evidence.len(),
        "first_seen_ms": first_seen_ms,
        "last_seen_ms": generated_at_ms,
        "summary": format!("{} {summary}", evidence.len()),
        "samples": evidence.into_iter().take(MAX_SAMPLES).map(|(_, sample)| sample).collect::<Vec<_>>(),
        "hint": hint,
    }))
}

fn reply(state: &ModuleState, window_ms: u64, generated_at_ms: u64) -> Result<Value, ()> {
    // Unlike a diagnosis, an unreadable store is a failed query, not healthy evidence.
    state.store.storage_health_inputs().map_err(|_| ())?;
    let health = module_health(state);
    let stale_since_ms = health.certification.stale_since_ms;
    let first_stale_evidence = stale_since_ms
        .or(health.last_rotation_at_ms)
        .unwrap_or(generated_at_ms);
    let stale = lanes_refusing_for_stale_certification(state)
        .into_iter()
        .map(|lane_id| {
            (
                first_stale_evidence,
                json!({"lane_id": lane_id, "stale_since_ms": stale_since_ms}),
            )
        })
        .collect();
    let mut failed_checks = Vec::new();
    let mut quarantined = Vec::new();
    let mut lanes = catalog_measurement_summary(state).lanes;
    lanes.sort_by(|a, b| a.slot.spec.model_id.cmp(&b.slot.spec.model_id));
    for lane in lanes {
        let slot = lane.slot;
        let lane_id = &slot.spec.model_id;
        if let Some((entry, backend)) = resolved_catalog_lane(&state.runtime, lane_id) {
            let check = catalog_self_check_projection(state, entry, backend).map_err(|_| ())?;
            if check["state"] == "failed" {
                let checked_at_ms = check["checked_at_ms"].as_u64();
                failed_checks.push((
                    checked_at_ms.unwrap_or(generated_at_ms),
                    json!({"lane_id": lane_id, "checked_at_ms": checked_at_ms}),
                ));
            }
        }
        let active = if let Some(model) = &slot.loaded {
            if let EmbedBackend::Worker(engine) = &model.backend {
                engine
                    .lock()
                    .map_err(|_| ())?
                    .model_quarantined(&model.loaded_model)
                    .map_err(|_| ())?
            } else {
                false
            }
        } else {
            false
        };
        // Cold load failures retain a quarantine refusal but no fence expiry.
        // Label that cached evidence instead of claiming the fence is still active.
        let cached = matches!(&slot.state, ModelRuntimeState::Failed(error)
            if error.message.contains("quarantined")
                || error.details.as_ref().is_some_and(|detail| detail["quarantined"] == true));
        if active || cached {
            quarantined.push((
                generated_at_ms,
                json!({"lane_id": lane_id, "cached": !active}),
            ));
        }
    }
    let mut findings = Vec::new();
    findings.extend(finding(
        "certification_stale",
        "error",
        stale,
        generated_at_ms,
        "certification-required lanes refuse requests because certification is stale.",
        "Re-certify with: ck-synapse-opctl probe run",
    ));
    findings.extend(finding(
        "catalog_self_check_failed", "warn", failed_checks, generated_at_ms,
        "catalog lanes have a failed on-device self-check.",
        "Inspect probe.report and models.catalog for the failed lanes before retrying their self-checks.",
    ));
    findings.extend(finding(
        "worker_quarantined",
        "warn",
        quarantined,
        generated_at_ms,
        "lanes have worker-crash quarantine evidence (cached samples may outlive the fence).",
        "Inspect probe.report for the affected lanes and worker health before retrying requests.",
    ));
    let mut body = json!({"result": {"generated_at_ms": generated_at_ms, "window_ms": window_ms, "findings": findings}});
    // Keep aggregate counts when large identifiers force evidence trimming.
    while serde_json::to_vec(&body).map_err(|_| ())?.len() >= MAX_REPLY_BYTES {
        let findings = body["result"]["findings"].as_array_mut().ok_or(())?;
        if let Some(samples) = findings.iter_mut().find_map(|finding| {
            finding["samples"]
                .as_array_mut()
                .filter(|samples| !samples.is_empty())
        }) {
            samples.pop();
        } else if findings.pop().is_none() {
            return Err(());
        }
    }
    Ok(body["result"].take())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{
        response_result, test_machine_profile, test_module_state, test_storage_descriptor,
    };

    #[test]
    fn diagnostics_findings_request_samples_and_strict_params() {
        for (sample, expected) in [
            (
                include_str!("../tests/fixtures/diagnostics-findings-v1/request-empty.json"),
                21_600_000,
            ),
            (
                include_str!("../tests/fixtures/diagnostics-findings-v1/request-window.json"),
                60_000,
            ),
            (
                include_str!("../tests/fixtures/diagnostics-findings-v1/request-max-window.json"),
                604_800_000,
            ),
        ] {
            let request: MethodEnvelope = serde_json::from_str(sample).unwrap();
            assert_eq!(request.method, "diagnostics.findings");
            assert_eq!(parse_window(request.params), Ok(expected));
        }
        for request in [
            json!({"method": "diagnostics.findings"}),
            json!({"method": "diagnostics.findings", "params": null}),
            json!({"method": "diagnostics.findings", "params": {"window_ms": null}}),
        ] {
            let request: MethodEnvelope = serde_json::from_value(request).unwrap();
            assert_eq!(parse_window(request.params), Ok(21_600_000));
        }
        assert_eq!(parse_window(json!({"window_ms": 1})), Ok(1));
        for params in [
            json!(false),
            json!([]),
            json!(7),
            json!("60000"),
            json!({"window_ms": 0}),
            json!({"window_ms": -1}),
            json!({"window_ms": 604_800_001u64}),
            json!({"window_ms": 1.5}),
            json!({"window_ms": "60000"}),
            json!({"window_ms": true}),
            json!({"window_ms": []}),
            json!({"extra": null}),
            json!({"window_ms": 60000, "flow_id": "flow-forged"}),
        ] {
            assert_eq!(parse_window(params.clone()), Err(()), "{params}");
        }
    }

    #[tokio::test]
    async fn diagnostics_findings_stale_certification_and_healthy_reply() {
        let (root, descriptor) = test_storage_descriptor("diagnostics-findings");
        let store = Arc::new(SynapseStore::open(&descriptor).unwrap());
        let profile_a = test_machine_profile("test-os-a");
        store.observe_profile(&profile_a, 10, 1).unwrap();
        store
            .store_class_scoped_cert_row(&ClassScopedCertificationRow {
                certification_class: CertificationClass::Embedding,
                assurance_class: AssuranceClass::Measured,
                status: CertificationStatus::Certified,
                key_hash: profile_a.hash(),
                machine_profile_hash: Some(profile_a.hash()),
                remote_profile_hash: None,
                identity_revision: None,
                numeric_profile_id: Some(NumericProfileId("test-profile".into())),
                fingerprint: Fingerprint("test-fingerprint".into()),
                certified_at_ms: 11,
                os_build: profile_a.os_build.clone(),
                module_generation: 1,
                evidence: json!({"private": "not exposed"}),
            })
            .unwrap();
        let healthy = test_module_state(Arc::clone(&store), profile_a);
        assert_eq!(
            json!({"result": reply(&healthy, 21_600_000, 10_000_000).unwrap()}),
            json!({"result": {"findings": [], "generated_at_ms": 10_000_000, "window_ms": 21_600_000}}),
        );
        let profile_b = test_machine_profile("test-os-b");
        store.observe_profile(&profile_b, 12, 1).unwrap();
        store
            .observe_certification_stale(&profile_b.revisioned_hash(), 13)
            .unwrap();
        let stale = test_module_state(Arc::clone(&store), profile_b);
        let result = response_result(
            dispatch_request(
                Arc::clone(&stale),
                MethodEnvelope {
                    method: "diagnostics.findings".into(),
                    params: json!({"window_ms": 1}),
                },
                None,
            )
            .await,
            "diagnostics.findings",
        );
        assert_eq!(result["window_ms"], 1);
        assert!(result["generated_at_ms"].as_u64().is_some());
        assert_eq!(
            result["findings"],
            json!([{
                "check": "certification_stale", "key": "certification_stale", "severity": "error",
                "count": 1, "first_seen_ms": 13, "last_seen_ms": result["generated_at_ms"],
                "summary": "1 certification-required lanes refuse requests because certification is stale.",
                "samples": [{"lane_id": "stuck-model", "stale_since_ms": 13}],
                "hint": "Re-certify with: ck-synapse-opctl probe run",
            }])
        );
        println!(
            "STALE_DIAGNOSTICS_REPLY={}",
            serde_json::to_string(&json!({"result": result})).unwrap()
        );
        assert_eq!(
            store
                .storage_health_inputs()
                .unwrap()
                .certification_stale_since_ms,
            Some(13)
        );
        assert_eq!(
            stale.runtime.loaded_model_count(),
            0,
            "diagnosis must not load models"
        );
        let base = stale.runtime.catalog.lock().unwrap()["stuck-model"]
            .spec
            .clone();
        let mut profile_spec = base.clone();
        profile_spec.model_id = "profile-lane".into();
        profile_spec
            .engine_identity
            .build_flags
            .insert("profile".into(), "self-checked".into());
        stale.runtime.catalog.lock().unwrap().insert(
            profile_spec.model_id.clone(),
            ModelSlot {
                spec: profile_spec,
                loaded: None,
                state: ModelRuntimeState::Unloaded,
                last_cold_load_ms: None,
                notify: Arc::new(Notify::new()),
            },
        );
        assert_eq!(
            reply(&stale, 1, 100).unwrap()["findings"][0]["count"],
            1,
            "profile self-check lanes must not count as stale legacy certification"
        );
        for index in 0..6 {
            let mut spec = base.clone();
            spec.model_id = format!("lane-{index}");
            stale.runtime.catalog.lock().unwrap().insert(
                spec.model_id.clone(),
                ModelSlot {
                    spec,
                    loaded: None,
                    state: ModelRuntimeState::Unloaded,
                    last_cold_load_ms: None,
                    notify: Arc::new(Notify::new()),
                },
            );
        }
        let capped = reply(&stale, 1, 100).unwrap();
        assert_eq!(capped["findings"][0]["count"], 7);
        assert_eq!(
            capped["findings"][0]["samples"].as_array().unwrap().len(),
            5
        );
        let mut oversized = base;
        oversized.model_id = "a".repeat(70_000);
        stale.runtime.catalog.lock().unwrap().insert(
            oversized.model_id.clone(),
            ModelSlot {
                spec: oversized,
                loaded: None,
                state: ModelRuntimeState::Unloaded,
                last_cold_load_ms: None,
                notify: Arc::new(Notify::new()),
            },
        );
        let bounded = json!({"result": reply(&stale, 1, 100).unwrap()});
        assert_eq!(bounded["result"]["findings"][0]["count"], 8);
        assert!(serde_json::to_vec(&bounded).unwrap().len() < 65_536);
        drop(stale);
        drop(healthy);
        drop(store);
        std::fs::remove_dir_all(root).unwrap();
    }
}
