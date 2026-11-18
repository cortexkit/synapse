use super::*;

#[test]
fn populated_v18_store_migrates_and_registers_only_kept_models() {
    let (root, descriptor) = test_storage_descriptor("retired-lanes-v18");
    let config = ModuleConfig::default();
    let catalog = runtime_release_catalog().unwrap();
    let cache = ModelCache::new(root.join("cache"));
    let mut metal = catalog_fixture_config("gte-modernbert-base.owned-metal");
    metal.model_id = "kept-metal".into();
    let metal = normalize_catalog_model(metal, &config.inline, &config.jobs).unwrap();
    let mut llama = stuck_model_spec();
    llama.model_id = "kept-llama".into();
    llama.engine = LLAMA_ENGINE.into();
    llama.task = "generate".into();
    let llama = normalize_catalog_model(llama, &config.inline, &config.jobs).unwrap();
    let mut models = vec![llama.clone(), metal.clone()];
    for engine in ["ane", "ane-coreml-worker", "coreml", "neural_engine"] {
        let mut retired = metal.clone();
        retired.model_id = format!("retired-{engine}");
        retired.engine = engine.into();
        models.push(retired);
    }
    for task in ["embed", "rerank"] {
        let mut retired = metal.clone();
        retired.model_id = format!("free-form-{task}");
        retired.task = task.into();
        retired.engine_identity.build_flags.remove("profile");
        models.push(retired);
    }
    let mut decode = llama.clone();
    decode.model_id = "qwen-decode".into();
    decode.engine = "owned-metal-decode".into();
    models.push(decode);
    let profile = test_machine_profile("v18-os");
    SynapseStore::seed_v18_retirement_fixture(&descriptor, &models, &profile).unwrap();

    let store = SynapseStore::open(&descriptor).unwrap();
    assert_eq!(
        store.catalog_models().unwrap(),
        [llama.clone(), metal.clone()]
    );
    let startup = sync_and_load_catalog_models(&store, &config, &catalog, &cache).unwrap();
    assert_eq!(startup, [llama, metal]);
    let runtime = RuntimeState::from_catalog(config, startup).unwrap();
    assert_eq!(runtime.catalog.lock().unwrap().len(), 2);
    assert_eq!(store.profile_state().unwrap().snapshot, Some(profile));
    assert_eq!(
        store
            .knob_assignment("machine", "embed", PerfKnob::Balanced)
            .unwrap()
            .unwrap()
            .model_id,
        "kept-metal"
    );
    assert!(store
        .knob_assignment("machine", "generate", PerfKnob::Balanced)
        .unwrap()
        .is_none());
    store.pub_connection_for_test(|conn| {
        for retired in [
            "alias_rows", "approvals", "approval_digest_corruption_events", "approval_migration_markers",
            "cert_rows", "cert_row_rebuild_events", "perf_rows", "profile_rotation_certification_outcomes",
            "serving_artifacts", "serving_certification_records", "serving_approvals", "serving_sessions",
            "serving_retained_states",
        ] {
            let count: u32 = conn.query_row("SELECT count(*) FROM sqlite_schema WHERE name=?1", [retired], |row| row.get(0))?;
            assert_eq!(count, 0, "{retired} was not dropped");
        }
        let stale: u32 = conn.query_row("SELECT count(*) FROM pragma_table_info('profile_state') WHERE name='certification_stale_since_ms'", [], |row| row.get(0))?;
        assert_eq!(stale, 0);
        for (table, key) in [("catalog_self_checks", "check_id = 'check'"), ("jobs", "job_id = 'kept-job'"), ("profile_rotation_events", "event_id = 'rotation'")] {
            let count: u32 = conn.query_row(&format!("SELECT count(*) FROM {table} WHERE {key}"), [], |row| row.get(0))?;
            assert_eq!(count, 1, "{table} lost a kept row");
        }
        Ok(())
    }).unwrap();
    drop(store);
    std::fs::remove_dir_all(root).unwrap();
}
