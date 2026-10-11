use super::*;
use store::{CatalogInstallMember, CatalogInstallRecord, DownloadJobRequest};
use worker_host::ane_residency::{module_mock_engine_controlled, ModuleMockControl};

const MASTER: &str = "027e1d30b11feac82dd4ac4a343c2de9e023bad3";
const LANES: [&str; 6] = [
    "gte-modernbert-base-metal",
    "gte-modernbert-base-ane",
    "gte-reranker-modernbert-base-metal",
    "qwen3-embedding-0.6b-metal",
    "qwen3-embedding-0.6b-ane",
    "qwen3-reranker-0.6b-ane",
];

enum Capture {
    Metal(Arc<Mutex<Vec<Vec<u32>>>>),
    Ane(ModuleMockControl),
}

impl Capture {
    fn take(&self) -> Vec<Vec<u32>> {
        match self {
            Self::Metal(ids) => std::mem::take(&mut *ids.lock().unwrap()),
            Self::Ane(control) => control.take_token_ids(),
        }
    }
}

fn fixture(lane: &str, assets: &Path) -> (PathBuf, Arc<ModuleState>, Capture) {
    let (root, descriptor) = tests::test_storage_descriptor("master-token-identity");
    let store = Arc::new(SynapseStore::open(&descriptor).unwrap());
    let profile = tests::test_machine_profile("catalog-token-identity-os");
    store.observe_profile(&profile, 1, 1).unwrap();
    let mut state = tests::test_module_state(store, profile);
    let inner = Arc::get_mut(&mut state).unwrap();
    inner.model_cache = Arc::new(ModelCache::new(root.join("cache")));
    let runtime = Arc::get_mut(&mut inner.runtime).unwrap();
    runtime.catalog.lock().unwrap().clear();
    runtime.runnable_backends = BTreeSet::from([lane.rsplit_once('-').unwrap().1.into()]);
    let (entry, backend) = state
        .runtime
        .release_catalog
        .resolve_reserved(lane)
        .unwrap();
    let backend = entry.backend(backend.unwrap()).unwrap();
    let files = entry.backend_files(&backend.backend);
    let tokenizer_path = assets.join(&files["tokenizer"].sha256);
    assert_eq!(
        sha256_file(&tokenizer_path).unwrap(),
        files["tokenizer"].sha256
    );
    let spec = catalog_lane_spec(&state, entry, backend, false).unwrap();
    let owned = stored_owned_profile(&spec).unwrap();
    let tokenizer = SanitizedTokenizer::from_file(
        &tokenizer_path,
        TokenizerConfig {
            max_tokens: if spec.engine_identity.build_flags.contains_key("profile") {
                usize::MAX
            } else {
                owned_tokenizer_max_tokens(spec.max_tokens, owned.as_ref())
            },
        },
    )
    .unwrap();
    let (engine_backend, loaded_model, policy, capture) = if backend.backend == "metal" {
        let config_path = assets.join(&files["config"].sha256);
        assert_eq!(sha256_file(&config_path).unwrap(), files["config"].sha256);
        let config: Value = serde_json::from_slice(&fs::read(config_path).unwrap()).unwrap();
        let owned = owned.as_ref().unwrap();
        let policy = OwnedTokenizerPolicy {
            add_special_tokens: true,
            pad_token_id: config["pad_token_id"].as_u64().unwrap_or(0) as u32,
            terminal_token_id: (owned.family == OwnedFamily::Qwen3)
                .then(|| config["eos_token_id"].as_u64().unwrap() as u32),
        };
        let captured = Arc::new(Mutex::new(Vec::new()));
        let mut engine = OwnedMetalEmbedEngine::new(owned.family, owned.dtype);
        let loaded = engine.insert_token_capture_test_model(lane.into(), policy, captured.clone());
        let policy = engine.tokenizer_policy(&loaded).unwrap();
        (
            EmbedBackend::Owned(Arc::new(Mutex::new(engine))),
            loaded,
            Some(policy),
            Capture::Metal(captured),
        )
    } else {
        let (engine, control) = module_mock_engine_controlled();
        control.enable_token_capture();
        (
            EmbedBackend::DirectAne(engine),
            LoadedModel {
                model_id: lane.into(),
            },
            None,
            Capture::Ane(control),
        )
    };
    let model = Arc::new(EmbeddingModel {
        model_id: spec.model_id.clone(),
        task: if entry.task == "embed" {
            ModelTask::Embed
        } else {
            ModelTask::Rerank
        },
        loaded_model,
        backend: engine_backend,
        tokenizer,
        numeric_profile_id: spec.numeric_profile_id.clone(),
        fingerprint: spec.fingerprint.clone(),
        certification_fingerprint: spec.fingerprint.clone(),
        engine_identity: spec.engine_identity.clone(),
        owned_tokenizer_policy: policy,
        owned_decode_resolution_refusal: None,
    });
    let job = state
        .store
        .admit_download_job(&DownloadJobRequest {
            request_key: "token-identity-fixture",
            request_digest: "token-identity-fixture",
            module_generation: 1,
            params_json: &json!({}),
            now_ms: 1,
            result_retention_ttl_ms: 60000,
            entry_complete: false,
        })
        .unwrap();
    state
        .store
        .commit_download_job(
            &job.record().job_id,
            &[CatalogInstallRecord {
                catalog_id: entry.id.clone(),
                manifest_digest: entry.manifest_digest(),
                backend: backend.backend.clone(),
                members: files
                    .values()
                    .map(|file| CatalogInstallMember {
                        path: file.path.clone(),
                        digest: file.sha256.clone(),
                    })
                    .collect(),
            }],
            2,
        )
        .unwrap();
    let (check_id, key) = catalog_self_check_key(&state, entry, backend).unwrap();
    let generation = catalog_check_generation(&state, &check_id, &key).unwrap();
    assert!(catalog_complete_check(&state, &check_id, generation, "passed", None).unwrap());
    state.runtime.catalog.lock().unwrap().insert(
        spec.model_id.clone(),
        ModelSlot {
            spec,
            loaded: Some(model),
            state: ModelRuntimeState::Ready,
            notify: Arc::new(Notify::new()),
            last_cold_load_ms: None,
        },
    );
    (root, state, capture)
}

fn cases(lane: &str) -> Vec<(String, &'static str, Value)> {
    let mut cases = Vec::new();
    for (name, text) in [
        (
            "short",
            "The quick brown fox jumps over the lazy dog.".to_string(),
        ),
        (
            "approximately-512",
            vec!["The quick brown fox jumps over the lazy dog."; 50].join(" "),
        ),
    ] {
        if lane.contains("rerank") {
            cases.push((name.into(), "rerank.score", json!({"model":lane,"query":"What is a vector database?","candidates":[text],"deadline_ms":30000})));
        } else {
            cases.push((
                name.into(),
                "embed.query",
                json!({"model":lane,"text":text,"deadline_ms":30000}),
            ));
            cases.push((
                name.into(),
                "embed.batch",
                json!({"model":lane,"items":[{"id":"row","text":text}],"deadline_ms":30000}),
            ));
        }
    }
    cases
}

fn assets() -> PathBuf {
    PathBuf::from(env::var_os("SYNAPSE_TOKEN_IDENTITY_ASSETS").expect(
        "set SYNAPSE_TOKEN_IDENTITY_ASSETS to the checksum-named pinned tokenizer/config files",
    ))
}

fn capture_cases(assets: &Path) -> Vec<Value> {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let mut captured = Vec::new();
    for lane in LANES {
        let (root, state, capture) = fixture(lane, assets);
        for (case, method, params) in cases(lane) {
            let response = tests::response_result(
                runtime.block_on(dispatch_request(
                    state.clone(),
                    MethodEnvelope {
                        method: method.into(),
                        params: params.clone(),
                    },
                    None,
                )),
                method,
            );
            assert!(
                response.get("error").is_none(),
                "{lane}/{method}/{case}: {response}"
            );
            let ids = capture.take();
            assert_eq!(
                ids.len(),
                1,
                "exactly one boundary sequence for {lane}/{method}/{case}"
            );
            if case == "approximately-512" {
                assert!(
                    (490..=600).contains(&ids[0].len()),
                    "long control must actually be near 512 tokens: {lane} {}",
                    ids[0].len()
                );
            }
            captured.push(
                json!({"lane_id":lane,"case":case,"method":method,"params":params,"token_ids":ids}),
            );
        }
        drop(capture);
        drop(state);
        fs::remove_dir_all(root).unwrap();
    }
    captured
}

// The literals are recorded at native model-family calls and direct-ANE IPC
// receivers by this same harness on the reference commit. Full upstream
// tokenizer bytes are checksum-verified; expected ids are not computed by the
// request path under test.
#[test]
#[ignore = "requires checksum-named pinned tokenizer/config files; see docs/evidence/catalog-token-identity.md"]
fn catalog_boundary_token_ids_match_master() {
    let golden: Value = serde_json::from_str(include_str!(
        "../tests/fixtures/catalog-master-token-ids.json"
    ))
    .unwrap();
    assert_eq!(golden["master_commit"], MASTER);
    let actual = capture_cases(&assets());
    let expected = golden["cases"].as_array().unwrap();
    assert_eq!(actual.len(), 20);
    assert_eq!(expected.len(), actual.len());
    for (expected, actual) in expected.iter().zip(actual) {
        assert_eq!(actual["lane_id"], expected["lane_id"]);
        assert_eq!(actual["method"], expected["method"]);
        assert_eq!(actual["case"], expected["case"]);
        assert_eq!(actual["params"], expected["params"]);
        assert_eq!(
            actual["token_ids"], expected["token_ids"],
            "master token identity changed for {}/{}/{}",
            actual["lane_id"], actual["method"], actual["case"]
        );
    }
}

#[test]
#[ignore = "capture tool: run only on the reference source snapshot, never to update expectations from the candidate"]
fn capture_master_boundary_token_ids() {
    let output = PathBuf::from(
        env::var_os("SYNAPSE_TOKEN_IDENTITY_CAPTURE").expect("set the master capture output path"),
    );
    let cases = capture_cases(&assets());
    fs::write(
        &output,
        serde_json::to_vec_pretty(&json!({"master_commit":MASTER,"cases":cases})).unwrap(),
    )
    .unwrap();
    eprintln!(
        "captured {} literal boundary sequences from {MASTER} into {}",
        cases.len(),
        output.display()
    );
}
