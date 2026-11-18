use super::*;

#[derive(Clone)]
struct Writer(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for Writer {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn startup_skips_retired_coreml_rows_with_one_named_warning() {
    let (root, descriptor) = test_storage_descriptor("retired-engine-startup");
    let store = SynapseStore::open(&descriptor).unwrap();
    let config = ModuleConfig::default();
    let catalog = runtime_release_catalog().unwrap();
    let cache = ModelCache::new(root.join("cache"));
    let mut kept = catalog_fixture_config("gte-modernbert-base.owned-metal");
    kept.model_id = "kept-metal".into();
    let kept = normalize_catalog_model(kept, &config.inline, &config.jobs).unwrap();
    store.upsert_model(&kept, 1).unwrap();

    // Both persisted spellings must be ignored before normalization tries to
    // construct an identity for an engine that no longer exists.
    for engine in ["ane", "ane-coreml-worker"] {
        let mut retired = stuck_model_spec();
        retired.model_id = format!("retired-{engine}");
        retired.engine = engine.into();
        retired.engine_identity.engine = "ane-coreml-worker".into();
        store.upsert_model(&retired, 2).unwrap();
    }

    let bytes = Arc::new(Mutex::new(Vec::new()));
    let writer = Writer(bytes.clone());
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .without_time()
        .with_writer(move || writer.clone())
        .finish();
    let restored = tracing::subscriber::with_default(subscriber, || {
        sync_and_load_catalog_models(&store, &config, &catalog, &cache).unwrap()
    });
    assert_eq!(restored, vec![kept]);
    assert_eq!(
        store.catalog_models().unwrap().len(),
        3,
        "migration owns deletion"
    );
    let logs = String::from_utf8(bytes.lock().unwrap().clone()).unwrap();
    let warnings: Vec<_> = logs.lines().filter(|line| line.contains("WARN")).collect();
    assert_eq!(warnings.len(), 2, "{logs}");
    for id in ["retired-ane", "retired-ane-coreml-worker"] {
        assert_eq!(
            warnings
                .iter()
                .filter(|line| line.contains(&format!("model_id={id} ")))
                .count(),
            1,
            "{logs}"
        );
    }
    assert!(warnings
        .iter()
        .all(|line| line.contains("skipping persisted registration with retired engine")));
    drop(store);
    std::fs::remove_dir_all(root).unwrap();
}
