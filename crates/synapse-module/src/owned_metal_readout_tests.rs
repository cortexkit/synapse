use super::*;

fn fixture(label: &str) -> (PathBuf, Arc<ModuleState>, PreloadModelConfig) {
    let (root, descriptor) = test_storage_descriptor(label);
    let store = Arc::new(SynapseStore::open(&descriptor).unwrap());
    let profile = test_machine_profile(label);
    store.activate_profile(&profile, 1, 1000).unwrap();
    let mut state = test_module_state(store, profile);
    Arc::get_mut(&mut state).unwrap().model_cache = Arc::new(ModelCache::new(root.join("cache")));
    make_test_tokenizer(&root, 512);
    fs::write(root.join("config.json"), br#"{"model_type":"modernbert"}"#).unwrap();
    let header = br#"{"embedding.weight":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#;
    let mut weights = (header.len() as u64).to_le_bytes().to_vec();
    weights.extend_from_slice(header);
    weights.extend_from_slice(&[0; 4]);
    fs::write(root.join("model.safetensors"), weights).unwrap();
    let preload = serde_json::from_value(json!({
        "model_id": "legacy-gte-embed", "engine": "owned-metal", "task": "embed",
        "family": "gte-modernbert", "dtype": "f16", "pooling": "mean",
        "model_path": root.join("model.safetensors"),
        "tokenizer_path": root.join("tokenizer.json")
    }))
    .unwrap();
    (root, state, preload)
}

#[test]
fn readout_table_is_operation_specific() {
    for (family, task, expected) in [
        (
            OwnedFamily::GteModernBert,
            ModelTask::Embed,
            WorkerPooling::Cls,
        ),
        (
            OwnedFamily::GteModernBert,
            ModelTask::Rerank,
            WorkerPooling::Mean,
        ),
        (OwnedFamily::Qwen3, ModelTask::Embed, WorkerPooling::Last),
        (OwnedFamily::Qwen3, ModelTask::Rerank, WorkerPooling::Last),
    ] {
        assert_eq!(owned_metal_readout_pooling(family, task), Some(expected));
    }
}

#[tokio::test]
async fn new_owned_metal_gte_embed_load_refuses_mean_and_omitted_pooling() {
    let (root, state, _) = fixture("metal-readout-load");
    for pooling in [Some("mean"), None] {
        let mut params = json!({
            "source": "file", "path": root, "engine": "owned-metal",
            "family": "gte-modernbert", "task": "embed",
            "model_id": "new-gte-embed",
            "files": {"model": "model.safetensors", "tokenizer": "tokenizer.json", "config": "config.json"},
            "request_key": format!("readout-{pooling:?}")
        });
        if let Some(pooling) = pooling {
            params["pooling"] = json!(pooling);
        }
        let accepted = response_result(model_load(state.clone(), params).await, "model.load");
        let job_id = accepted["job_id"].as_str().expect("load job admitted");
        let record = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                let record = state.store.get_job(job_id).unwrap().unwrap();
                if record.state == JOB_STATE_FAILED_PERMANENT
                    || record.state == JOB_STATE_FAILED_TRANSIENT
                    || record.state == JOB_STATE_DONE
                {
                    break record;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("load job completed");
        let error = record.error_json.unwrap_or(Value::Null);
        assert_eq!(error["code"], "invalid_request", "{error}");
        assert_eq!(
            error["message"],
            "owned-metal gte-modernbert embedding computes cls; declare \"pooling\": \"cls\""
        );
        assert!(state.store.catalog_models().unwrap().is_empty());
        assert!(model_slot_snapshot(&state.runtime, "new-gte-embed").is_none());
    }
    drop(state);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn new_owned_metal_preload_refuses_mean_but_accepts_cls() {
    let (root, state, mut preload) = fixture("metal-readout-preload");
    for pooling in [Some("mean"), None] {
        preload.pooling = pooling.map(str::to_string);
        let config = ModuleConfig {
            preload_models: vec![preload.clone()],
            ..ModuleConfig::default()
        };
        let error = sync_and_load_catalog_models(
            &state.store,
            &config,
            &state.runtime.release_catalog,
            &state.model_cache,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("computes cls; declare \"pooling\": \"cls\""));
        assert!(state.store.catalog_models().unwrap().is_empty());
    }
    preload.pooling = Some("cls".into());
    let config = ModuleConfig {
        preload_models: vec![preload],
        ..ModuleConfig::default()
    };
    let loaded = sync_and_load_catalog_models(
        &state.store,
        &config,
        &state.runtime.release_catalog,
        &state.model_cache,
    )
    .unwrap();
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].pooling, "cls");
    drop(state);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn persisted_owned_metal_mean_survives_startup_with_and_without_preload() {
    let (root, state, preload) = fixture("metal-readout-persisted");
    // Capture the legacy declaration through the unchanged builder, before
    // simulating a restart with a row that already exists in the store.
    let legacy = build_preload_catalog_model(
        0,
        preload.clone(),
        &state.runtime.inline,
        &state.runtime.jobs,
    )
    .unwrap();
    state.store.upsert_model(&legacy, 1).unwrap();
    let legacy_bytes = serde_json::to_vec(&legacy).unwrap();
    for preloads in [Vec::new(), vec![preload]] {
        let config = ModuleConfig {
            preload_models: preloads,
            ..ModuleConfig::default()
        };
        let loaded = sync_and_load_catalog_models(
            &state.store,
            &config,
            &state.runtime.release_catalog,
            &state.model_cache,
        )
        .unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].pooling, "mean");
        assert_eq!(loaded[0].fingerprint, legacy.fingerprint);
        assert_eq!(serde_json::to_vec(&loaded[0]).unwrap(), legacy_bytes);
        assert_eq!(state.store.catalog_models().unwrap(), vec![legacy.clone()]);
        let runtime = RuntimeState::from_catalog(config, loaded).unwrap();
        let slot = model_slot_snapshot(&runtime, &legacy.model_id).unwrap();
        assert_eq!(slot.spec, legacy);
    }
    drop(state);
    fs::remove_dir_all(root).unwrap();
}
