use std::collections::BTreeMap;
use std::path::PathBuf;
use synapse_certify::*;

struct Assets(PathBuf);
impl Assets {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "synapse-certify-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        for file in ["ck-synapse", "worker", "candidate.zip"] {
            std::fs::write(path.join(file), file.as_bytes()).unwrap();
        }
        Self(path)
    }
}
impl Drop for Assets {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}
fn parity(model: &str) -> Parity {
    Parity {
        model: model.into(),
        operation: if model.contains("reranker") {
            "rerank"
        } else {
            "embed"
        }
        .into(),
        profile_id: format!("{model}.owned-metal"),
        fingerprint: "f".repeat(64),
        fixture_set_id: "fixture-set".into(),
        completed_fixtures: 10,
        expected_fixtures: 10,
        tolerance_class: "fp16".into(),
        metrics: serde_json::json!({"min_cosine": 1.0}),
        gates: GATES.iter().map(|name| ((*name).into(), true)).collect(),
        failures: vec![],
    }
}
fn admission() -> Admission {
    Admission {
        tokens_8192: Outcome {
            outcome: "processed".into(),
            truncated: false,
            diverted: false,
            worker_requests: 1,
        },
        tokens_8193: Outcome {
            outcome: "sequence_too_long".into(),
            truncated: false,
            diverted: false,
            worker_requests: 0,
        },
    }
}
fn evidence(model: &str) -> RunEvidence {
    let p = parity(model);
    RunEvidence {
        source_commit: "a".repeat(40),
        machine: serde_json::json!({"model_identifier": "Mac17,6", "platform_uuid": "machine"}),
        artifacts: vec![
            ArtifactFile {
                role: "ck-synapse".into(),
                file: "ck-synapse".into(),
            },
            ArtifactFile {
                role: "ck-synapse-worker-ane-direct".into(),
                file: "worker".into(),
            },
            ArtifactFile {
                role: "ck-synapse-worker-cuda".into(),
                file: "worker".into(),
            },
            ArtifactFile {
                role: "ck-synapse-worker-vulkan".into(),
                file: "worker".into(),
            },
        ],
        operation: p.operation.clone(),
        profile_id: p.profile_id.clone(),
        fingerprint: p.fingerprint.clone(),
        fixture_set_id: p.fixture_set_id.clone(),
        expected_fixtures: 10,
        tolerance_class: "fp16".into(),
        parity: Some(p),
        admission: admission(),
        layer_count: 2,
        inventories: vec![Inventory {
            executables: vec![Executable {
                id: "graph".into(),
                layers: vec![0, 1],
            }],
            cpu_stages: vec!["token_embedding".into()],
        }],
        raw_series: None,
    }
}
struct Mock {
    evidence: RunEvidence,
    floor: String,
    probes: usize,
}
impl Runner for Mock {
    fn probe_floor(&mut self, _: &str, _: &str) -> Result<String> {
        self.probes += 1;
        Ok(self.floor.clone())
    }
    fn observe(&mut self, _: &str, _: &str) -> Result<RunEvidence> {
        Ok(self.evidence.clone())
    }
}
fn runner(model: &str) -> Mock {
    Mock {
        evidence: evidence(model),
        floor: "ok".into(),
        probes: 0,
    }
}
fn matrix(assets: &Assets) -> Vec<Record> {
    ROWS.into_iter()
        .flat_map(|row| MODELS.into_iter().map(move |model| (row, model)))
        .map(|(row, model)| produce(&mut runner(model), &assets.0, row, model).unwrap())
        .collect()
}
fn qwen(records: &mut [Record]) -> &mut Record {
    records
        .iter_mut()
        .find(|r| r.row_id == "ane-m5" && r.model == "qwen3-embedding-0.6b")
        .unwrap()
}
fn series(ratio: f64) -> RawSeries {
    let mut ane = vec![999.0; 3];
    ane.extend(vec![ratio; 20]);
    let mut metal = vec![0.01; 3];
    metal.extend(vec![1.0; 20]);
    RawSeries {
        session_id: "same-session".into(),
        ane,
        metal,
    }
}

#[test]
fn metal_never_probes_and_schema_has_only_r6_fields() {
    let assets = Assets::new();
    let mut mock = runner(MODELS[0]);
    mock.floor = "refused".into();
    let record = produce(&mut mock, &assets.0, "metal-m5", MODELS[0]).unwrap();
    assert_eq!(mock.probes, 0);
    assert!(record.eligible());
    let value = serde_json::to_value(&record).unwrap();
    let keys = value
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect::<Vec<_>>();
    assert_eq!(
        keys,
        vec![
            "admission",
            "executed_artifacts",
            "fingerprint",
            "machine",
            "model",
            "operation",
            "parity",
            "profile_id",
            "row_id",
            "schema",
            "source_commit",
            "status"
        ]
    );
    let path = write_record(&record, &assets.0).unwrap();
    assert!(path.ends_with(format!(
        "docs/evidence/certification/{}/metal-m5/{}.json",
        "a".repeat(40),
        MODELS[0]
    )));
}

#[test]
fn producer_refuses_each_failed_gate_without_writing() {
    let assets = Assets::new();
    for row in ROWS {
        if row == "metal-m5" {
            continue;
        }
        let mut mock = runner(MODELS[0]);
        mock.floor = "refused".into();
        assert!(
            produce(&mut mock, &assets.0, row, MODELS[0]).is_err(),
            "{row}"
        );
    }
    for gate in GATES {
        for row in ROWS {
            let mut mock = runner(MODELS[0]);
            mock.evidence
                .parity
                .as_mut()
                .unwrap()
                .gates
                .insert(gate.into(), false);
            assert!(
                produce(&mut mock, &assets.0, row, MODELS[0]).is_err(),
                "{row}/{gate}"
            );
        }
    }
    for row in ROWS {
        for mutate in [
            (|e: &mut RunEvidence| e.admission.tokens_8192.truncated = true)
                as fn(&mut RunEvidence),
            |e| e.admission.tokens_8192.diverted = true,
            |e| e.admission.tokens_8192.outcome = "queued".into(),
            |e| e.admission.tokens_8193.outcome = "processed".into(),
            |e| e.admission.tokens_8193.worker_requests = 1,
        ] {
            let mut mock = runner(MODELS[0]);
            mutate(&mut mock.evidence);
            assert!(
                produce(&mut mock, &assets.0, row, MODELS[0]).is_err(),
                "{row}"
            );
        }
    }
    let mutations: Vec<fn(&mut RunEvidence)> = vec![
        |e| {
            e.parity
                .as_mut()
                .unwrap()
                .gates
                .insert("score".into(), false)
                .map(|_| ())
                .unwrap()
        },
        |e| e.admission.tokens_8192.truncated = true,
        |e| e.admission.tokens_8192.diverted = true,
        |e| e.admission.tokens_8192.outcome = "queued".into(),
        |e| e.admission.tokens_8193.outcome = "processed".into(),
        |e| e.admission.tokens_8193.worker_requests = 1,
        |e| {
            e.inventories[0].executables[0]
                .layers
                .pop()
                .map(|_| ())
                .unwrap()
        },
        |e| e.inventories[0].cpu_stages.push("transformer_layer".into()),
        |e| e.inventories[0].executables[0].layers.push(1),
        |e| e.inventories.clear(),
    ];
    for mutate in mutations {
        let mut mock = runner(MODELS[0]);
        mutate(&mut mock.evidence);
        assert!(produce(&mut mock, &assets.0, "ane-m5", MODELS[0]).is_err());
    }
    assert!(!assets.0.join("docs").exists());
}

#[test]
fn missing_parity_drops_only_qwen_ane_and_parity_precedes_latency() {
    let assets = Assets::new();
    for model in MODELS {
        let mut mock = runner(model);
        mock.evidence.parity = None;
        mock.evidence.inventories.clear();
        mock.evidence.raw_series = Some(series(4.0));
        let result = produce(&mut mock, &assets.0, "ane-m5", model);
        if model.starts_with("qwen3-") {
            let record = result.unwrap();
            assert_eq!(record.status, "dropped");
            assert_eq!(record.drop_cause.as_deref(), Some("parity"));
            assert_eq!(record.parity.completed_fixtures, 0);
            assert_eq!(record.parity.gates.len(), GATES.len());
            assert!(record.parity.gates.values().all(|passed| !passed));
            assert!(!record.eligible());
        } else {
            assert!(result.is_err());
        }
        assert!(produce(&mut mock, &assets.0, "metal-m5", model).is_err());
    }
    let mut mock = runner(MODELS[2]);
    mock.evidence.parity = None;
    mock.evidence.inventories[0]
        .cpu_stages
        .push("unlisted".into());
    assert!(produce(&mut mock, &assets.0, "ane-m5", MODELS[2]).is_err());
}

#[test]
fn matrix_requires_all_combinations_and_valid_statuses() {
    let assets = Assets::new();
    let records = matrix(&assets);
    assert_eq!(validate(&records, &assets.0).unwrap().len(), 32);
    assert!(validate(&records[..31], &assets.0).is_err());
    let mut refused = records.clone();
    refused[0].status = "refused".into();
    assert!(validate(&refused, &assets.0).is_err());
    let mut dropped = records.clone();
    dropped[0].status = "dropped".into();
    dropped[0].parity.gates.insert("score".into(), false);
    assert!(validate(&dropped, &assets.0).is_err());
    let mut duplicate = records;
    duplicate[0] = duplicate[1].clone();
    assert!(validate(&duplicate, &assets.0).is_err());
}

#[test]
fn drop_recomputation_is_strict_and_cause_agnostic() {
    let assets = Assets::new();
    let mut records = matrix(&assets);
    let record = qwen(&mut records);
    record.status = "dropped".into();
    record.drop_cause = Some("parity".into());
    record.raw_series = Some(series(3.0));
    assert!(validate(&records, &assets.0).is_err());
    qwen(&mut records).raw_series = Some(series(3.000_001));
    assert_eq!(validate(&records, &assets.0).unwrap().len(), 31);
    let record = qwen(&mut records);
    record.raw_series = Some(series(3.0));
    record.drop_cause = Some("latency".into());
    record.parity.gates.insert("score".into(), false);
    assert_eq!(validate(&records, &assets.0).unwrap().len(), 31);
    qwen(&mut records).raw_series = None;
    assert!(validate(&records, &assets.0).is_ok());
}

#[test]
fn median_discards_warmups_and_averages_middle_pair() {
    let mut samples = vec![10000.0; 3];
    samples.extend((1..=20).rev().map(f64::from));
    assert_eq!(median(&samples).unwrap(), 10.5);
    assert!(median(&samples[..22]).is_err());
    samples[12] = f64::NAN;
    assert!(median(&samples).is_err());
}

#[test]
fn executed_digest_is_extracted_file_not_zip_sidecar() {
    let assets = Assets::new();
    let mut records = matrix(&assets);
    records[0].executed_artifacts[0].sha256 = digest(&assets.0, "candidate.zip").unwrap();
    assert!(validate(&records, &assets.0).is_err());
    records[0].executed_artifacts[0].sha256 = "0".repeat(64);
    assert!(validate(&records, &assets.0).is_err());
}

#[test]
fn parity_identity_must_match_all_four_fields() {
    let assets = Assets::new();
    let records = matrix(&assets);
    for field in ["model", "operation", "profile_id", "fingerprint"] {
        let mut changed = records.clone();
        let p = &mut qwen(&mut changed).parity;
        match field {
            "model" => p.model = MODELS[0].into(),
            "operation" => p.operation = "rerank".into(),
            "profile_id" => p.profile_id = "other-profile".into(),
            _ => p.fingerprint = "other-fingerprint".into(),
        }
        assert!(validate(&changed, &assets.0).is_err(), "{field}");
    }
}

#[test]
fn canonical_checkout_round_trip_and_unknown_fields_rejected() {
    let assets = Assets::new();
    let records = matrix(&assets);
    for record in &records {
        write_record(record, &assets.0).unwrap();
    }
    assert_eq!(
        validate_checkout(&assets.0, &assets.0, &"a".repeat(40))
            .unwrap()
            .len(),
        32
    );
    let mut value = serde_json::to_value(&records[0]).unwrap();
    value["reuse_certified"] = true.into();
    assert!(serde_json::from_value::<Record>(value).is_err());
    let mut invalid = records[0].clone();
    invalid.source_commit = "short".into();
    assert!(write_record(&invalid, &assets.0).is_err());
}

#[test]
fn validator_does_not_add_producer_gates() {
    let assets = Assets::new();
    let mut records = matrix(&assets);
    records[0].admission.tokens_8192.truncated = true;
    records[0].parity.gates = BTreeMap::from([("score".into(), false)]);
    assert!(validate(&records, &assets.0).is_ok());
}

#[test]
fn validate_dispatch_uses_supplied_clean_candidate_source() {
    let assets = Assets::new();
    for record in matrix(&assets) {
        write_record(&record, &assets.0).unwrap();
    }
    let source = "a".repeat(40);
    let report = synapse_certify::command::dispatch(
        synapse_certify::command::Command::Validate {
            assets: assets.0.clone(),
            checkout: assets.0.clone(),
        },
        Some(&source),
    )
    .unwrap();
    assert_eq!(report["source_commit"], source);
    assert_eq!(report["eligible"].as_array().unwrap().len(), 32);
}

#[test]
fn validate_dispatch_without_clean_source_refuses_and_writes_nothing() {
    let assets = Assets::new();
    let checkout = assets.0.join("absent-checkout");
    let error = synapse_certify::command::dispatch(
        synapse_certify::command::Command::Validate {
            assets: assets.0.join("absent-assets"),
            checkout: checkout.clone(),
        },
        None,
    )
    .unwrap_err();
    assert_eq!(error.to_string(), "certification_refused: candidate was built from a dirty tree or without git; evidence cannot be bound to a commit");
    assert!(!checkout.exists());
    assert_eq!(std::fs::read_dir(&assets.0).unwrap().count(), 3);
}

// The producer and the validator each apply the latency drop rule, so each
// needs its own boundary test: a slowdown of exactly 3x keeps a Qwen3 ANE cell
// in the release, and only a slowdown strictly above 3x may drop it.
#[test]
fn producer_latency_drop_is_strictly_above_three() {
    let assets = Assets::new();
    for model in MODELS.iter().filter(|m| m.starts_with("qwen3-")) {
        let mut mock = runner(model);
        mock.evidence.raw_series = Some(series(3.0));
        let record = produce(&mut mock, &assets.0, "ane-m5", model).unwrap();
        assert_eq!(record.status, "passed", "{model} at exactly 3x");
        let mut mock = runner(model);
        mock.evidence.raw_series = Some(series(3.000_001));
        let record = produce(&mut mock, &assets.0, "ane-m5", model).unwrap();
        assert_eq!(record.status, "dropped", "{model} above 3x");
        assert_eq!(record.drop_cause.as_deref(), Some("latency"));
    }
}
