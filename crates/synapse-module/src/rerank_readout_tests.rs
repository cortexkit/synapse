use super::*;

const CASE_IDS: [&str; 4] = ["pool100-000", "pool100-007", "pool100-042", "pool100-063"];
// Raw scores captured at source revision f1149e55 from the native ModernBERT
// head for pool100-000/007/042/063 in the first 64-candidate parity fixture request.
// Native catalog and legacy loads share that raw head. The replay checks that
// the module preserves those exact bytes when serving the legacy lane.
const MASTER_LOGITS: [f32; 4] = [3.311_804_3, 3.711_782, -0.079_588_935, 0.104_514_48];
const MASTER_LOGIT_BYTES: [u8; 16] = [
    154, 244, 83, 64, 214, 141, 109, 64, 134, 255, 162, 189, 176, 11, 214, 61,
];

fn fixture_pairs_and_references() -> (Vec<Vec<u32>>, Vec<f64>) {
    let fixture: Value = serde_json::from_slice(include_bytes!(
        "../../../bench/parity/fixtures/gte-reranker-modernbert-base/gte-reranker-modernbert-base.ref-v1.transformers-5.16.1.seed-0.json"
    )).unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    CASE_IDS
        .iter()
        .map(|id| {
            let case = cases.iter().find(|case| case["id"] == *id).unwrap();
            (
                serde_json::from_value(case["input_ids"].clone()).unwrap(),
                case["output"].as_f64().unwrap(),
            )
        })
        .unzip()
}

fn replay_master_logits(pairs: &[Vec<u32>]) -> synapse_core::RerankScores {
    let (fixed_pairs, _) = fixture_pairs_and_references();
    assert_eq!(pairs, fixed_pairs);
    synapse_core::RerankScores {
        scores: MASTER_LOGITS.to_vec(),
    }
}

#[cfg(unix)]
fn replay_worker(
    request: &synapse_core::WorkerRequest,
    raw: Option<&[u8]>,
) -> (synapse_core::WorkerResponse, Option<Vec<u8>>) {
    use synapse_core::{WorkerRequest, WorkerResponse};
    let req_id = match request {
        WorkerRequest::Rerank { req_id, .. } | WorkerRequest::RerankSequences { req_id, .. } => {
            req_id
        }
        other => return (WorkerResponse::unsupported_request(other), None),
    };
    let (pairs, _) = fixture_pairs_and_references();
    let expected: Vec<i32> = pairs.iter().flatten().map(|id| *id as i32).collect();
    assert_eq!(
        synapse_core::decode_i32_frame(raw.unwrap()).unwrap(),
        expected
    );
    (
        WorkerResponse::Scores {
            req_id: req_id.clone(),
        },
        Some(MASTER_LOGIT_BYTES.to_vec()),
    )
}

#[cfg(unix)]
fn replay_probability_worker(
    request: &synapse_core::WorkerRequest,
    raw: Option<&[u8]>,
) -> (synapse_core::WorkerResponse, Option<Vec<u8>>) {
    let (response, _) = replay_worker(request, raw);
    let (_, references) = fixture_pairs_and_references();
    (
        response,
        Some(synapse_core::encode_f32_frame(
            &references
                .iter()
                .map(|score| *score as f32)
                .collect::<Vec<_>>(),
        )),
    )
}

#[cfg(unix)]
fn replay_model(root: &Path, spec: &StoredModelConfig, probabilities: bool) -> Arc<EmbeddingModel> {
    let mut seed = spec.clone();
    seed.engine_identity.build_flags.insert(
        "profile".into(),
        "gte-reranker-modernbert-base.owned-metal".into(),
    );
    let mut model = tests::catalog_test_model(root, &seed);
    let inner = Arc::get_mut(&mut model).unwrap();
    inner.engine_identity = spec.engine_identity.clone();
    let EmbedBackend::Worker(engine) = &inner.backend else {
        unreachable!()
    };
    engine
        .lock()
        .unwrap()
        .attach_mock_worker_for_test(if probabilities {
            replay_probability_worker
        } else {
            replay_worker
        });
    model
}

async fn serve_replay(
    state: &ModuleState,
    root: &Path,
    spec: &StoredModelConfig,
    pairs: Vec<Vec<u32>>,
) -> synapse_core::RerankScores {
    #[cfg(unix)]
    {
        let root = root.to_path_buf();
        let spec = spec.clone();
        let model = tokio::task::spawn_blocking(move || {
            replay_model(
                &root,
                &spec,
                spec.engine_identity.build_flags.contains_key("profile"),
            )
        })
        .await
        .unwrap();
        let result = execute_rerank(
            &state.runtime,
            &model,
            RerankRequest {
                query: Vec::new(),
                candidates: pairs.clone(),
            },
            Some(pairs),
            None,
            None,
        )
        .await
        .unwrap();
        tokio::task::spawn_blocking(move || drop(model))
            .await
            .unwrap();
        result
    }
    #[cfg(not(unix))]
    {
        let _ = root;
        let owned = resolved_catalog_lane(&state.runtime, &spec.model_id).is_some();
        let engine_scores = if spec.engine_identity.build_flags.contains_key("profile") {
            synapse_core::RerankScores {
                scores: fixture_pairs_and_references()
                    .1
                    .iter()
                    .map(|score| *score as f32)
                    .collect(),
            }
        } else {
            replay_master_logits(&pairs)
        };
        RerankReadout::select(owned, &spec.engine_identity).apply(engine_scores)
    }
}

fn legacy_spec(root: &Path) -> StoredModelConfig {
    let fixture: Value = serde_json::from_slice(include_bytes!(
        "../../../bench/parity/preload/gte-reranker-modernbert-base-f32.json"
    ))
    .unwrap();
    fs::create_dir_all(root).unwrap();
    fs::write(root.join("config.json"), br#"{"model_type":"modernbert"}"#).unwrap();
    let owned = owned_catalog_config(
        root,
        Some("gte-modernbert"),
        Some("f32"),
        Some("explicit"),
        None,
        None,
        Vec::new(),
    )
    .unwrap();
    build_stored_model_config(
        fixture["model_id"].as_str().unwrap().into(),
        "owned-metal",
        ModelTask::Rerank,
        fixture["artifact_digest"].as_str().unwrap().into(),
        "safetensors".into(),
        fixture["sanitized_tokenizer_digest"]
            .as_str()
            .unwrap()
            .into(),
        ModelAssetLocator::LocalPath {
            path: "unused".into(),
        },
        ModelAssetLocator::LocalPath {
            path: "unused".into(),
        },
        String::new(),
        String::new(),
        parse_pooling(fixture["pooling"].as_str().unwrap()).unwrap(),
        fixture["normalize"].as_bool().unwrap(),
        fixture["max_tokens"].as_u64().unwrap() as usize,
        fixture["quant"].as_str().unwrap().into(),
        false,
        None,
        None,
        Vec::new(),
        Some(owned),
        &serde_json::from_value(fixture["inline"].clone()).unwrap(),
        &serde_json::from_value(fixture["jobs"].clone()).unwrap(),
    )
    .unwrap()
}

fn test_state(name: &str) -> (PathBuf, Arc<ModuleState>) {
    let (root, descriptor) = tests::test_storage_descriptor(name);
    let store = Arc::new(SynapseStore::open(&descriptor).unwrap());
    let profile = tests::test_machine_profile("readout");
    store.observe_profile(&profile, 1, 1).unwrap();
    (root, tests::test_module_state(store, profile))
}

#[tokio::test]
async fn legacy_rerank_readout_keeps_master_bytes_and_fingerprint() {
    let (root, state) = test_state("legacy-rerank-readout");
    let spec = legacy_spec(&root);
    assert_eq!(spec.model_id, "gte-reranker-modernbert-base-f32");
    assert_eq!(
        spec.fingerprint.0,
        "2fa5f24c0208f30c6db4bf18eb66bc0b2c46f765882cb744fcc58cd080b2b92d"
    );
    let catalog_owned = resolved_catalog_lane(&state.runtime, &spec.model_id).is_some();
    assert!(!catalog_owned);
    let (pairs, _) = fixture_pairs_and_references();
    let scores = serve_replay(&state, &root, &spec, pairs).await;
    let bytes: Vec<u8> = scores
        .scores
        .iter()
        .flat_map(|score| score.to_le_bytes())
        .collect();
    assert_eq!(
        bytes, MASTER_LOGIT_BYTES,
        "legacy rerank output bytes changed"
    );
    drop(state);
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn every_catalog_rerank_backend_selects_probability_readout_by_ownership() {
    let (root, state) = test_state("catalog-rerank-readout");
    let (pairs, reference) = fixture_pairs_and_references();
    let mut seen = BTreeSet::new();
    for entry in state
        .runtime
        .release_catalog
        .models
        .iter()
        .filter(|entry| entry.task == "rerank")
    {
        for backend in &entry.backends {
            let spec = catalog_lane_spec(&state, entry, backend, false).unwrap();
            let owned = resolved_catalog_lane(&state.runtime, &spec.model_id).is_some();
            assert!(owned, "{}", spec.model_id);
            let readout = RerankReadout::select(owned, &spec.engine_identity);
            assert_ne!(readout, RerankReadout::Legacy, "{}", spec.model_id);
            let engine_scores = if backend.profile.is_none() {
                assert_eq!(readout, RerankReadout::CatalogLogits);
                assert_eq!(
                    spec.engine_identity.build_flags["rerank_readout"],
                    "sigmoid_classifier_logit"
                );
                replay_master_logits(&pairs)
            } else {
                assert_eq!(readout, RerankReadout::CatalogProbability);
                synapse_core::RerankScores {
                    scores: reference.iter().map(|score| *score as f32).collect(),
                }
            };
            let expected_scores = readout.apply(engine_scores);
            let scores =
                serve_replay(&state, &root.join(&spec.model_id), &spec, pairs.clone()).await;
            assert_eq!(scores.scores, expected_scores.scores);
            for (actual, expected) in scores.scores.iter().zip(&reference) {
                assert!((0.0..=1.0).contains(actual));
                assert!(
                    (f64::from(*actual) - expected).abs() < 0.001,
                    "{}",
                    spec.model_id
                );
            }
            // An ordinary preload is not a reserved catalog lane: even with a
            // profile flag, it must select the unchanged legacy readout.
            assert_eq!(
                RerankReadout::select(false, &spec.engine_identity),
                RerankReadout::Legacy
            );
            seen.insert((entry.id.clone(), backend.backend.clone()));
        }
    }
    for backend in ["metal", "ane", "cuda", "vulkan"] {
        assert!(
            seen.iter().any(|(_, name)| name == backend),
            "missing {backend}"
        );
    }
    assert_eq!(seen.len(), 6);
    drop(state);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn native_catalog_pins_match_real_lane_construction() {
    let (root, state) = test_state("catalog-readout-pins");
    let tokenizers: BTreeMap<String, String> = serde_json::from_slice(include_bytes!(
        "../tests/fixtures/catalog-tokenizer-digests.json"
    ))
    .unwrap();
    let mut checked = 0;
    for entry in &state.runtime.release_catalog.models {
        for backend in entry
            .backends
            .iter()
            .filter(|backend| backend.profile.is_none())
        {
            let spec =
                catalog_native_lane_spec(&state, entry, backend, tokenizers[&entry.id].clone())
                    .unwrap();
            println!("{}: {}", spec.model_id, spec.fingerprint.0);
            assert_eq!(spec.fingerprint.0, backend.fingerprint, "{}", spec.model_id);
            checked += 1;
        }
    }
    assert_eq!(checked, 3);
    drop(state);
    fs::remove_dir_all(root).unwrap();
}

// Replay the captured catalog checkpoint's raw head through the ordinary legacy
// loader. The frozen preload separately pins the production legacy identity;
// its checkpoint is not needed or read from the production store here.
// CKDEV_RERANK_MODEL=<snapshot>/model.safetensors cargo test --release --locked
// -p synapse-module --lib legacy_metal_engine_keeps_master_logit_bytes -- --ignored --nocapture
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires original pinned ModernBERT reranker weights and Metal"]
fn legacy_metal_engine_keeps_master_logit_bytes() {
    let model_path =
        PathBuf::from(std::env::var_os("CKDEV_RERANK_MODEL").expect("CKDEV_RERANK_MODEL"));
    let (root, _) = tests::test_storage_descriptor("legacy-metal-logit-bytes");
    let spec = legacy_spec(&root);
    let catalog = catalog::compiled_catalog().unwrap();
    let entry = catalog.entry("gte-reranker-modernbert-base").unwrap();
    assert_eq!(
        sha256_file(&model_path).unwrap(),
        entry.backend_files("metal")["model"].sha256
    );
    let fixture: Value = serde_json::from_slice(include_bytes!(
        "../../../bench/parity/fixtures/gte-reranker-modernbert-base/gte-reranker-modernbert-base.ref-v1.transformers-5.16.1.seed-0.json"
    )).unwrap();
    let pairs: Vec<Vec<u32>> = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|case| case["pool"] == "pool-100")
        .take(64)
        .map(|case| serde_json::from_value(case["input_ids"].clone()).unwrap())
        .collect();
    let mut config = model_runtime_config(&spec, &model_path, &[], &root.join("cache"), 8192, None);
    // The frozen identity omits the deployment's attention budget. Supply enough
    // to admit its 8192-token context; this replay still uses only short inputs.
    config
        .values
        .insert("attention_units".into(), (8192 * 8192).to_string());
    let mut engine = OwnedMetalEmbedEngine::new(OwnedFamily::GteModernBert, OwnedDType::F32);
    let model = EmbedEngine::load(
        &mut engine,
        &synapse_core::ValidatedArtifact {
            digest: spec.artifact_digest.clone(),
            format: "safetensors-package".into(),
        },
        &config,
    )
    .unwrap();
    let scores = engine.rerank_pairs(&model, pairs).unwrap();
    let bytes: Vec<u8> = [0, 7, 42, 63]
        .into_iter()
        .flat_map(|index| scores.scores[index].to_le_bytes())
        .collect();
    assert_eq!(
        bytes, MASTER_LOGIT_BYTES,
        "legacy Metal engine logit bytes changed"
    );
    assert_eq!(
        spec.fingerprint.0,
        "2fa5f24c0208f30c6db4bf18eb66bc0b2c46f765882cb744fcc58cd080b2b92d"
    );
    EmbedEngine::unload(&mut engine, &model);
    fs::remove_dir_all(root).unwrap();
}
