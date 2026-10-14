use std::path::PathBuf;
use synapse_certify::*;
use synapse_parity::evaluator::{fixture_set_id, FixtureSet};
use synapse_parity::manifest::{Manifest, ToleranceClass};

const TEST_MODELS: [&str; 4] = [
    "gte-modernbert-base",
    "gte-reranker-modernbert-base",
    "qwen3-embedding-0.6b",
    "qwen3-reranker-0.6b",
];
fn rows() -> impl Iterator<Item = &'static str> {
    ROW_BACKENDS.iter().map(|(row, _)| *row)
}
fn manifest() -> Manifest {
    Manifest::from_slice(include_bytes!("../../../bench/parity/models.json")).unwrap()
}
fn reference_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../bench/parity")
}
fn engine(row: &str) -> &str {
    match row_backend(row).unwrap() {
        "metal" => "owned-metal",
        "ane" => "ane-direct-worker",
        "cuda" => "owned-cuda",
        _ => "owned-vulkan",
    }
}
fn catalog() -> Catalog {
    let lanes = TEST_MODELS
        .into_iter()
        .flat_map(|model| {
            ["metal", "ane", "cuda", "vulkan"]
                .into_iter()
                .filter_map(move |backend| {
                    if (model == "qwen3-reranker-0.6b" && backend == "metal")
                        || (model == "gte-reranker-modernbert-base" && backend == "ane")
                    {
                        return None;
                    }
                    let engine = match backend {
                        "metal" => "owned-metal",
                        "ane" => "ane-direct-worker",
                        "cuda" => "owned-cuda",
                        _ => "owned-vulkan",
                    };
                    Some((model.into(), backend.into(), engine.into(), "f".repeat(64)))
                })
        })
        .collect();
    Catalog {
        lanes,
        tasks: TEST_MODELS
            .into_iter()
            .map(|model| {
                (
                    model.into(),
                    if model.contains("reranker") {
                        "rerank"
                    } else {
                        "embed"
                    }
                    .into(),
                )
            })
            .collect(),
    }
}

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
        let index: serde_json::Value =
            serde_json::from_slice(include_bytes!("../../../bench/parity/fixtures/index.json"))
                .unwrap();
        for entry in index.as_object().unwrap().values() {
            let file = entry["path"].as_str().unwrap();
            let destination = path.join("bench/parity").join(file);
            std::fs::create_dir_all(destination.parent().unwrap()).unwrap();
            std::fs::copy(reference_root().join(file), destination).unwrap();
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
    let manifest = manifest();
    let count = FixtureSet::load(&reference_root(), &manifest, model)
        .unwrap()
        .cases()
        .len();
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
        fixture_set_id: fixture_set_id(&manifest, model),
        completed_fixtures: count,
        expected_fixtures: count,
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
        // Filled per row by `Mock::observe`; a test that sets it keeps its value.
        machine: serde_json::Value::Null,
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
        expected_fixtures: p.expected_fixtures,
        tolerance_class: "fp16".into(),
        parity: Some(p),
        admission: admission(),
        layer_count: manifest()
            .model(model)
            .unwrap()
            .architecture
            .int("num_hidden_layers")
            .unwrap() as usize,
        admitted_count: 1,
        inventories: vec![Inventory {
            executables: vec![Executable {
                id: "graph".into(),
                layers: (0..manifest()
                    .model(model)
                    .unwrap()
                    .architecture
                    .int("num_hidden_layers")
                    .unwrap() as usize)
                    .collect(),
            }],
            cpu_stages: vec!["token_embedding".into()],
        }],
    }
}
/// The machine block the live collector writes for `row`, built from fixed
/// fixture UUIDs.
fn machine(row: &str) -> serde_json::Value {
    if matches!(row, "metal-m5" | "ane-m5") {
        serde_json::json!({
            "model_identifier": "Mac17,6",
            "platform_uuid_sha256": platform_uuid_sha256(APPLE_UUID),
        })
    } else {
        serde_json::json!({
            "system_product_name": "fixture",
            "gpu": {"name": "fixture", "uuid_sha256": gpu_uuid_sha256(GPU_UUID)},
        })
    }
}
const APPLE_UUID: &str = "4F3A2B1C-0D9E-4A7B-8C6D-5E4F3A2B1C0D";
const GPU_UUID: &str = "GPU-7D1C9E2A-3B4F-4C5D-8E6F-0A1B2C3D4E5F";
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
    fn observe(&mut self, row: &str, model: &str) -> Result<RunEvidence> {
        let mut evidence = self.evidence.clone();
        if evidence.machine.is_null() {
            evidence.machine = machine(row);
        }
        evidence.profile_id = format!("{model}.{}", engine(row));
        evidence.tolerance_class =
            match manifest().profiles[&evidence.profile_id].rerank_tolerance_class {
                ToleranceClass::Fp32 => "fp32",
                ToleranceClass::Fp16 => "fp16",
            }
            .into();
        if let Some(parity) = &mut evidence.parity {
            parity.profile_id = evidence.profile_id.clone();
            parity.tolerance_class = evidence.tolerance_class.clone();
        }
        Ok(evidence)
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
    derive_cells(&catalog().lanes)
        .into_iter()
        .map(|cell| {
            produce(
                &mut runner(&cell.model),
                &assets.0,
                &cell.row_id,
                &cell.model,
            )
            .unwrap()
        })
        .collect()
}
fn qwen(records: &mut [Record]) -> &mut Record {
    records
        .iter_mut()
        .find(|r| r.row_id == "ane-m5" && r.model == "qwen3-embedding-0.6b")
        .unwrap()
}

#[test]
fn metal_never_probes_and_schema_three_has_only_record_fields() {
    let assets = Assets::new();
    let mut mock = runner(TEST_MODELS[0]);
    mock.floor = "refused".into();
    let record = produce(&mut mock, &assets.0, "metal-m5", TEST_MODELS[0]).unwrap();
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
        TEST_MODELS[0]
    )));
}

#[test]
fn producer_refuses_each_failed_gate_without_writing() {
    let assets = Assets::new();
    for row in rows() {
        if row == "metal-m5" {
            continue;
        }
        let mut mock = runner(TEST_MODELS[0]);
        mock.floor = "refused".into();
        assert!(
            produce(&mut mock, &assets.0, row, TEST_MODELS[0]).is_err(),
            "{row}"
        );
    }
    for gate in GATES {
        for row in rows() {
            let mut mock = runner(TEST_MODELS[0]);
            mock.evidence
                .parity
                .as_mut()
                .unwrap()
                .gates
                .insert(gate.into(), false);
            assert!(
                produce(&mut mock, &assets.0, row, TEST_MODELS[0]).is_err(),
                "{row}/{gate}"
            );
        }
    }
    for row in rows() {
        for mutate in [
            (|e: &mut RunEvidence| e.admission.tokens_8192.truncated = true)
                as fn(&mut RunEvidence),
            |e| e.admission.tokens_8192.diverted = true,
            |e| e.admission.tokens_8192.outcome = "queued".into(),
            |e| e.admission.tokens_8193.outcome = "processed".into(),
            |e| e.admission.tokens_8193.worker_requests = 1,
        ] {
            let mut mock = runner(TEST_MODELS[0]);
            mutate(&mut mock.evidence);
            assert!(
                produce(&mut mock, &assets.0, row, TEST_MODELS[0]).is_err(),
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
        let mut mock = runner(TEST_MODELS[0]);
        mutate(&mut mock.evidence);
        assert!(produce(&mut mock, &assets.0, "ane-m5", TEST_MODELS[0]).is_err());
    }
    assert!(!assets.0.join("docs").exists());
}

#[test]
fn matrix_requires_all_combinations_and_valid_statuses() {
    let assets = Assets::new();
    let records = matrix(&assets);
    assert_eq!(
        validate(&records, &assets.0, &assets.0, &catalog())
            .unwrap()
            .len(),
        30
    );
    assert!(validate(&records[..29], &assets.0, &assets.0, &catalog()).is_err());
    let mut refused = records.clone();
    refused[0].status = "refused".into();
    assert!(validate(&refused, &assets.0, &assets.0, &catalog()).is_err());
    let mut dropped = records.clone();
    dropped[0].status = "dropped".into();
    dropped[0].parity.gates.insert("score".into(), false);
    assert!(validate(&dropped, &assets.0, &assets.0, &catalog()).is_err());
    let mut duplicate = records;
    duplicate[0] = duplicate[1].clone();
    assert!(validate(&duplicate, &assets.0, &assets.0, &catalog()).is_err());
}

#[test]
fn executed_digest_is_extracted_file_not_zip_sidecar() {
    let assets = Assets::new();
    let mut records = matrix(&assets);
    records[0].executed_artifacts[0].sha256 = digest(&assets.0, "candidate.zip").unwrap();
    assert!(validate(&records, &assets.0, &assets.0, &catalog()).is_err());
    records[0].executed_artifacts[0].sha256 = "0".repeat(64);
    assert!(validate(&records, &assets.0, &assets.0, &catalog()).is_err());
}

#[test]
fn parity_identity_must_match_all_four_fields() {
    let assets = Assets::new();
    let records = matrix(&assets);
    for field in ["model", "operation", "profile_id", "fingerprint"] {
        let mut changed = records.clone();
        let p = &mut qwen(&mut changed).parity;
        match field {
            "model" => p.model = TEST_MODELS[0].into(),
            "operation" => p.operation = "rerank".into(),
            "profile_id" => p.profile_id = "other-profile".into(),
            _ => p.fingerprint = "other-fingerprint".into(),
        }
        assert!(
            validate(&changed, &assets.0, &assets.0, &catalog()).is_err(),
            "{field}"
        );
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
        validate_checkout(&assets.0, &assets.0, &"a".repeat(40), &catalog())
            .unwrap()
            .len(),
        30
    );
    let mut value = serde_json::to_value(&records[0]).unwrap();
    value["reuse_certified"] = true.into();
    assert!(serde_json::from_value::<Record>(value).is_err());
    let mut invalid = records[0].clone();
    invalid.source_commit = "short".into();
    assert!(write_record(&invalid, &assets.0).is_err());
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
        &catalog(),
    )
    .unwrap();
    assert_eq!(report["source_commit"], source);
    assert_eq!(report["eligible"].as_array().unwrap().len(), 30);
}

#[test]
fn shipped_validate_entry_point_accepts_a_complete_synthetic_matrix() {
    let assets = Assets::new();
    for record in matrix(&assets) {
        write_record(&record, &assets.0).unwrap();
    }
    let arguments = vec![
        "certify".into(),
        "validate".into(),
        "--assets".into(),
        assets.0.clone().into_os_string(),
        assets.0.clone().into_os_string(),
    ];
    let command = synapse_certify::command::parse(&arguments)
        .unwrap()
        .unwrap();
    let source = "a".repeat(40);
    let report = synapse_certify::command::dispatch(command, Some(&source), &catalog()).unwrap();
    assert_eq!(report["source_commit"], source);
    assert_eq!(
        report["eligible"].as_array().unwrap().len(),
        derive_cells(&catalog().lanes).len()
    );
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
        &catalog(),
    )
    .unwrap_err();
    assert_eq!(error.to_string(), "certification_refused: candidate was built from a dirty tree or without git; evidence cannot be bound to a commit");
    assert!(!checkout.exists());
    assert_eq!(std::fs::read_dir(&assets.0).unwrap().count(), 4);
}

#[test]
fn serialized_inventory_omission_fails_placement_gate() {
    let assets = Assets::new();
    let mut mock = runner(TEST_MODELS[0]);
    mock.evidence
        .inventories
        .push(mock.evidence.inventories[0].clone());
    mock.evidence.admitted_count = 2;
    assert!(produce(&mut mock, &assets.0, "ane-m5", TEST_MODELS[0]).is_ok());
    let mut serialized = serde_json::to_value(&mock.evidence).unwrap();
    serialized["inventories"].as_array_mut().unwrap().pop();
    mock.evidence = serde_json::from_value(serialized).unwrap();
    let error = produce(&mut mock, &assets.0, "ane-m5", TEST_MODELS[0]).unwrap_err();
    assert!(
        error.to_string().contains("placement inventory failed"),
        "{error}"
    );
}

// The expected digests were computed outside this crate, with
// `printf 'synapse-certify/machine/v1\0%s' <uuid> | shasum -a 256` and the
// `gpu/v1` equivalent, so a change to either prefix or to how the bytes are
// joined fails here rather than silently starting a new, unlinkable series.
#[test]
fn machine_uuid_digests_are_pinned_and_stable() {
    assert_eq!(
        platform_uuid_sha256(APPLE_UUID),
        "2909c561b627ba9c561fdc2a46e81e8003bba789f4af2569b3f198326f2010ef"
    );
    assert_eq!(
        gpu_uuid_sha256(GPU_UUID),
        "86b9be077c7820aa8554ac2f0e65ad7e227781fd74f768f3feee9fbf59523030"
    );
    let first = platform_uuid_sha256(APPLE_UUID);
    let again = platform_uuid_sha256(APPLE_UUID);
    assert_eq!(first, again);
    assert_ne!(
        platform_uuid_sha256(APPLE_UUID),
        platform_uuid_sha256("4F3A2B1C-0D9E-4A7B-8C6D-5E4F3A2B1C0E")
    );
    assert_ne!(platform_uuid_sha256(GPU_UUID), gpu_uuid_sha256(GPU_UUID));
}

#[test]
fn records_with_a_raw_hardware_uuid_or_no_digest_are_refused() {
    let assets = Assets::new();
    let records = matrix(&assets);
    assert!(records.iter().all(|record| record.schema == RECORD_SCHEMA));
    assert!(validate(&records, &assets.0, &assets.0, &catalog()).is_ok());
    let apple = records
        .iter()
        .position(|record| record.row_id == "metal-m5")
        .unwrap();
    let gpu = records
        .iter()
        .position(|record| record.row_id == "cuda-linux-nvidia")
        .unwrap();
    type MachineEdit = (usize, fn(&mut serde_json::Value));
    let edits: [MachineEdit; 5] = [
        (apple, |machine| {
            machine["platform_uuid"] = APPLE_UUID.into();
        }),
        (apple, |machine| {
            machine
                .as_object_mut()
                .unwrap()
                .remove("platform_uuid_sha256");
        }),
        (apple, |machine| {
            machine["platform_uuid_sha256"] = APPLE_UUID.into();
        }),
        (gpu, |machine| {
            machine["gpu"]["uuid"] = GPU_UUID.into();
        }),
        (gpu, |machine| {
            machine["gpu"]
                .as_object_mut()
                .unwrap()
                .remove("uuid_sha256");
        }),
    ];
    for (index, edit) in edits {
        let mut changed = records.clone();
        edit(&mut changed[index].machine);
        assert!(
            validate(&changed, &assets.0, &assets.0, &catalog()).is_err(),
            "{}",
            changed[index].machine
        );
    }
    let mut old_schema = records.clone();
    old_schema[0].schema = 1;
    assert!(validate(&old_schema, &assets.0, &assets.0, &catalog()).is_err());

    let mut mock = runner(TEST_MODELS[0]);
    mock.evidence.machine = serde_json::json!({
        "model_identifier": "Mac17,6",
        "platform_uuid": APPLE_UUID,
        "platform_uuid_sha256": platform_uuid_sha256(APPLE_UUID),
    });
    assert!(produce(&mut mock, &assets.0, "metal-m5", TEST_MODELS[0]).is_err());
    assert!(!assets.0.join("docs").exists());
}

fn assert_record_mutation(
    row: &str,
    model: &str,
    reason: &str,
    edit: impl FnOnce(&mut serde_json::Value),
) {
    let assets = Assets::new();
    let record = produce(&mut runner(model), &assets.0, row, model).unwrap();
    validate_record(&record, &assets.0, &assets.0, &catalog()).unwrap();
    let mut value = serde_json::to_value(record).unwrap();
    edit(&mut value);
    let changed: Record = serde_json::from_value(value).unwrap();
    let error = validate_record(&changed, &assets.0, &assets.0, &catalog()).unwrap_err();
    assert!(error.to_string().contains(reason), "{error}");
}

#[test]
fn validator_refuses_failed_or_missing_parity_gates() {
    for gate in GATES {
        for missing in [false, true] {
            assert_record_mutation("metal-m5", TEST_MODELS[0], "parity_gate_failed", |value| {
                if missing {
                    value["parity"]["gates"]
                        .as_object_mut()
                        .unwrap()
                        .remove(gate);
                } else {
                    value["parity"]["gates"][gate] = false.into();
                }
            });
        }
    }
}

#[test]
fn validator_refuses_failed_admission() {
    for (tokens, field, value) in [
        ("tokens_8192", "outcome", serde_json::json!("queued")),
        ("tokens_8192", "truncated", serde_json::json!(true)),
        ("tokens_8192", "diverted", serde_json::json!(true)),
        ("tokens_8193", "outcome", serde_json::json!("processed")),
        ("tokens_8193", "worker_requests", serde_json::json!(1)),
        ("tokens_8193", "truncated", serde_json::json!(true)),
        ("tokens_8193", "diverted", serde_json::json!(true)),
    ] {
        assert_record_mutation(
            "metal-m5",
            TEST_MODELS[0],
            "admission_gate_failed",
            |record| record["admission"][tokens][field] = value,
        );
    }
}

#[test]
fn validator_refuses_wrong_fixture_set_id() {
    assert_record_mutation(
        "metal-m5",
        TEST_MODELS[0],
        "fixture_set_id_mismatch",
        |value| value["parity"]["fixture_set_id"] = "wrong".into(),
    );
}

#[test]
fn validator_refuses_self_consistently_lowered_fixture_counts() {
    for fields in [
        vec!["completed_fixtures"],
        vec!["expected_fixtures"],
        vec!["completed_fixtures", "expected_fixtures"],
    ] {
        assert_record_mutation(
            "metal-m5",
            TEST_MODELS[0],
            "fixture_count_mismatch",
            |value| {
                let lowered = value["parity"]["expected_fixtures"].as_u64().unwrap() - 1;
                for field in fields {
                    value["parity"][field] = lowered.into();
                }
            },
        );
    }
}

#[test]
fn validator_refuses_non_catalog_fingerprint() {
    assert_record_mutation(
        "metal-m5",
        TEST_MODELS[0],
        "catalog_fingerprint_mismatch",
        |value| {
            value["fingerprint"] = "0".repeat(64).into();
            value["parity"]["fingerprint"] = "0".repeat(64).into();
        },
    );
}

#[test]
fn validator_refuses_cuda_profile_on_metal_reranker() {
    for fields in [vec!["record"], vec!["parity"], vec!["record", "parity"]] {
        assert_record_mutation(
            "metal-m5",
            TEST_MODELS[1],
            "catalog_profile_mismatch",
            |value| {
                for field in fields {
                    let profile = "gte-reranker-modernbert-base.owned-cuda".into();
                    if field == "record" {
                        value["profile_id"] = profile;
                    } else {
                        value["parity"]["profile_id"] = profile;
                    }
                }
            },
        );
    }
}

#[test]
fn validator_refuses_changed_tolerance_class() {
    assert_record_mutation(
        "metal-m5",
        TEST_MODELS[1],
        "tolerance_class_mismatch",
        |value| value["parity"]["tolerance_class"] = "fp16".into(),
    );
}

#[test]
fn validator_refuses_self_consistently_changed_operations() {
    for fields in [vec!["record"], vec!["parity"], vec!["record", "parity"]] {
        assert_record_mutation(
            "metal-m5",
            TEST_MODELS[0],
            "catalog_operation_mismatch",
            |value| {
                for field in fields {
                    if field == "record" {
                        value["operation"] = "rerank".into();
                    } else {
                        value["parity"]["operation"] = "rerank".into();
                    }
                }
            },
        );
    }
}

#[test]
fn validator_refuses_missing_placement() {
    assert_record_mutation("ane-m5", TEST_MODELS[0], "placement_missing", |value| {
        value.as_object_mut().unwrap().remove("placement");
    });
}

#[test]
fn validator_refuses_zero_placement_admissions() {
    assert_record_mutation(
        "ane-m5",
        TEST_MODELS[0],
        "placement_admission_missing",
        |value| {
            value["placement"]["admitted_count"] = 0.into();
            value["placement"]["inventories"] = serde_json::json!([]);
        },
    );
}

#[test]
fn validator_refuses_placement_inventory_count_mismatch() {
    assert_record_mutation(
        "ane-m5",
        TEST_MODELS[0],
        "placement_inventory_count_mismatch",
        |value| value["placement"]["admitted_count"] = 2.into(),
    );
}

#[test]
fn validator_refuses_wrong_placement_layer_count() {
    assert_record_mutation(
        "ane-m5",
        TEST_MODELS[0],
        "placement_layer_count_mismatch",
        |value| {
            let layers = value["placement"]["layer_count"].as_u64().unwrap();
            value["placement"]["layer_count"] = (layers + 1).into();
        },
    );
}

#[test]
fn validator_refuses_failed_placement_inventory() {
    for fault in ["missing_layer", "duplicate_layer", "cpu_stage"] {
        assert_record_mutation(
            "ane-m5",
            TEST_MODELS[0],
            "placement_inventory_failed",
            |value| {
                let inventory = &mut value["placement"]["inventories"][0];
                match fault {
                    "missing_layer" => {
                        inventory["executables"][0]["layers"]
                            .as_array_mut()
                            .unwrap()
                            .pop();
                    }
                    "duplicate_layer" => inventory["executables"][0]["layers"]
                        .as_array_mut()
                        .unwrap()
                        .push(0.into()),
                    _ => inventory["cpu_stages"]
                        .as_array_mut()
                        .unwrap()
                        .push("transformer_layer".into()),
                }
            },
        );
    }
}

#[test]
fn validator_refuses_schema_two() {
    assert_record_mutation(
        "metal-m5",
        TEST_MODELS[0],
        "record_schema_mismatch",
        |value| value["schema"] = 2.into(),
    );
}

#[test]
fn historical_schema_two_records_keep_the_same_refusal_without_rental_ids() {
    let checkout = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let sources = [
        "dd7412b0409eb3c97d026c1bdd38797669813752",
        "ff24e6715e1bb58ad6ba0c120b889ef7431ca6c3",
    ];
    let mut checked = 0;
    for source in sources {
        let directory = checkout.join("docs/evidence/certification").join(source);
        let mut rows = std::fs::read_dir(directory)
            .unwrap()
            .map(|entry| entry.unwrap())
            .collect::<Vec<_>>();
        rows.sort_by_key(|row| row.file_name());
        for row in rows {
            let mut files = std::fs::read_dir(row.path())
                .unwrap()
                .map(|entry| entry.unwrap())
                .collect::<Vec<_>>();
            files.sort_by_key(|file| file.file_name());
            for file in files {
                let original: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(file.path()).unwrap()).unwrap();
                assert_eq!(original["schema"], 2);
                let assets = Assets::new();
                let relative = file.path().strip_prefix(&checkout).unwrap().to_path_buf();
                let destination = assets.0.join(&relative);
                std::fs::create_dir_all(destination.parent().unwrap()).unwrap();
                let mut redacted = original.clone();
                redacted["machine"]
                    .as_object_mut()
                    .unwrap()
                    .remove("instance_id");
                // Historical records must remain historical: removing rental metadata
                // must neither admit an obsolete schema nor mask its schema refusal.
                for (label, value) in [("original", original), ("redacted", redacted)] {
                    std::fs::write(&destination, serde_json::to_vec(&value).unwrap()).unwrap();
                    let refusal =
                        validate_checkout(&assets.0, &assets.0, source, &catalog()).unwrap_err();
                    assert_eq!(
                        refusal.to_string(),
                        "certification_refused: record_schema_mismatch"
                    );
                    println!(
                        "{} {label} instance_id={} -> {refusal}",
                        relative.display(),
                        value["machine"].get("instance_id").is_some()
                    );
                }
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 8);
}

#[test]
fn public_certification_records_omit_rental_instance_ids() {
    let directory =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/evidence/certification");
    let mut checked = 0;
    for source in std::fs::read_dir(directory).unwrap() {
        let source = source.unwrap();
        if !source.file_type().unwrap().is_dir() {
            continue;
        }
        for row in std::fs::read_dir(source.path()).unwrap() {
            let row = row.unwrap();
            if !row.file_type().unwrap().is_dir() {
                continue;
            }
            for file in std::fs::read_dir(row.path()).unwrap() {
                let path = file.unwrap().path();
                if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
                    continue;
                }
                let record: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
                assert!(
                    record["machine"].get("instance_id").is_none(),
                    "public certification record exposes a rental instance_id: {}",
                    path.display()
                );
                checked += 1;
            }
        }
    }
    assert!(
        checked >= 10,
        "public certification records were not inspected"
    );
}

#[test]
fn validator_refuses_fixture_and_checkout_index_changed_together() {
    let assets = Assets::new();
    let record = produce(
        &mut runner(TEST_MODELS[0]),
        &assets.0,
        "metal-m5",
        TEST_MODELS[0],
    )
    .unwrap();
    validate_record(&record, &assets.0, &assets.0, &catalog()).unwrap();
    let mut index: serde_json::Value =
        serde_json::from_slice(include_bytes!("../../../bench/parity/fixtures/index.json"))
            .unwrap();
    let entry = &mut index[&record.parity.fixture_set_id];
    let fixture_path = assets
        .0
        .join("bench/parity")
        .join(entry["path"].as_str().unwrap());
    let mut fixture: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&fixture_path).unwrap()).unwrap();
    fixture["cases"][0]["output"][0] = 0.5.into();
    let bytes = serde_json::to_vec(&fixture).unwrap();
    std::fs::write(&fixture_path, &bytes).unwrap();
    entry["sha256"] = synapse_parity::canonical::sha256_hex(&bytes).into();
    std::fs::write(
        assets.0.join("bench/parity/fixtures/index.json"),
        serde_json::to_vec(&index).unwrap(),
    )
    .unwrap();
    std::fs::write(
        assets.0.join("bench/parity/models.json"),
        b"not the embedded manifest",
    )
    .unwrap();
    let error = validate_record(&record, &assets.0, &assets.0, &catalog()).unwrap_err();
    assert!(
        error.to_string().contains("fixture_digest_mismatch"),
        "{error}"
    );
}

#[test]
fn validator_matrix_refuses_missing_declared_cell() {
    let assets = Assets::new();
    let mut records = matrix(&assets);
    let absent = records.pop().unwrap();
    for record in &records {
        write_record(record, &assets.0).unwrap();
    }
    let error = validate_checkout(&assets.0, &assets.0, &"a".repeat(40), &catalog()).unwrap_err();
    assert!(
        error.to_string().contains(&format!(
            "missing_certification_cell: {}/{}",
            absent.row_id, absent.model
        )),
        "{error}"
    );
}

#[test]
fn validator_matrix_refuses_undeclared_record_file() {
    let assets = Assets::new();
    let records = matrix(&assets);
    for record in &records {
        write_record(record, &assets.0).unwrap();
    }
    let mut reduced = catalog();
    reduced
        .lanes
        .retain(|lane| !(lane.0 == TEST_MODELS[0] && lane.1 == "cuda"));
    let error = validate_checkout(&assets.0, &assets.0, &"a".repeat(40), &reduced).unwrap_err();
    assert!(
        error.to_string().contains("undeclared_certification_cell"),
        "{error}"
    );
    for record in records
        .iter()
        .filter(|record| record.model == TEST_MODELS[0] && record.row_id.starts_with("cuda-"))
    {
        std::fs::remove_file(record.path(&assets.0)).unwrap();
    }
    assert_eq!(
        validate_checkout(&assets.0, &assets.0, &"a".repeat(40), &reduced)
            .unwrap()
            .len(),
        28
    );
}

#[test]
fn validator_accepts_embedded_references_without_checkout_metadata_or_git() {
    let assets = Assets::new();
    let records = matrix(&assets);
    for record in &records {
        write_record(record, &assets.0).unwrap();
    }
    assert!(!assets.0.join(".git").exists());
    assert!(!assets.0.join("bench/parity/fixtures/index.json").exists());
    assert!(!assets.0.join("bench/parity/models.json").exists());
    assert_eq!(
        validate_checkout(&assets.0, &assets.0, &"a".repeat(40), &catalog())
            .unwrap()
            .len(),
        30
    );
}

#[test]
fn windows_cuda_extraction_paths_round_trip_including_runtime_dlls() {
    let assets = Assets::new();
    std::fs::create_dir(assets.0.join("windows-x64")).unwrap();
    let mut mock = runner(TEST_MODELS[0]);
    mock.evidence.artifacts.clear();
    for (role, file) in [
        ("ck-synapse", "ck-synapse.exe"),
        ("ck-synapse-worker-cuda", "ck-synapse-worker-cuda.exe"),
        ("cuda-runtime:cudart64_12.dll", "cudart64_12.dll"),
        ("cuda-runtime:cublas64_12.dll", "cublas64_12.dll"),
    ] {
        let path = format!("windows-x64/{file}");
        std::fs::write(assets.0.join(&path), file.as_bytes()).unwrap();
        mock.evidence.artifacts.push(ArtifactFile {
            role: role.into(),
            file: path,
        });
    }
    let record = produce(&mut mock, &assets.0, "cuda-windows-nvidia", TEST_MODELS[0]).unwrap();
    assert!(record
        .executed_artifacts
        .iter()
        .all(|artifact| artifact.file.starts_with("windows-x64/")));
    write_record(&record, &assets.0).unwrap();
    let mut records = matrix(&assets);
    let index = records
        .iter()
        .position(|candidate| candidate.row_id == record.row_id && candidate.model == record.model)
        .unwrap();
    records[index] = record;
    for record in &records {
        write_record(record, &assets.0).unwrap();
    }
    assert_eq!(
        validate_checkout(&assets.0, &assets.0, &"a".repeat(40), &catalog())
            .unwrap()
            .len(),
        30
    );
}

#[test]
fn missing_parity_always_refuses_and_writes_nothing() {
    let assets = Assets::new();
    for model in TEST_MODELS {
        let mut mock = runner(model);
        mock.evidence.parity = None;
        for row in ["ane-m5", "metal-m5"] {
            assert!(produce(&mut mock, &assets.0, row, model).is_err());
        }
    }
    assert!(!assets.0.join("docs").exists());
}

#[test]
fn validator_refuses_non_passed_status() {
    for status in ["refused", "dropped"] {
        assert_record_mutation(
            "metal-m5",
            TEST_MODELS[0],
            "invalid_certification_status",
            |value| value["status"] = status.into(),
        );
    }
}
