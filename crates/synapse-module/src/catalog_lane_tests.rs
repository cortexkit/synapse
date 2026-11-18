use super::*;
use catalog_runtime_tests::isolated_catalog_state;
use tests::response_result;

const SPEECH: &str = "parakeet-tdt-0.6b-v3";

fn install_record(state: &ModuleState, entry: &catalog::CatalogEntry, backend: &str) {
    state
        .store
        .sqlite()
        .with_conn_fenced(|tx| {
            tx.execute(
                "INSERT INTO catalog_installs VALUES (?1,?2,?3)",
                rusqlite::params![entry.id, entry.manifest_digest(), backend],
            )?;
            Ok(())
        })
        .unwrap();
}

#[test]
fn catalog_transcribe_config_never_uses_an_embed_owned_family() {
    let (root, state) = isolated_catalog_state("speech-config", false);
    let entry = state.runtime.release_catalog.entry(SPEECH).unwrap();
    for backend in &entry.backends {
        let profile = CatalogProfile::load(backend.profile.as_deref().unwrap()).unwrap();
        let error = profile.owned_config(None, None).unwrap_err().to_string();
        assert!(
            error.contains(SPEECH) && error.contains("transcribe lane has no embed owned config")
        );
        let config = profile.transcribe_config().unwrap();
        assert_eq!(config.family, "parakeet-tdt");
        assert_eq!(config.max_encoder_frames, 7500);
        assert_eq!(config.dtype.as_str(), backend.dtype.as_deref().unwrap());
        assert_eq!(config.execution, backend.execution.as_deref().unwrap());
        let numeric = profile.transcribe_numeric_profile(&config);
        assert_eq!(numeric.engine, config.identity);
        assert_eq!(numeric.fingerprint().0, backend.fingerprint);
        assert_eq!(
            numeric.manifest_profile_digest.as_deref(),
            Some(profile.typed.digests.profiles[&profile.id].as_str())
        );
        assert_eq!(
            profile.typed.profile_digest(&profile.id).unwrap(),
            profile.typed.digests.profiles[&profile.id]
        );
        assert_eq!(
            numeric.sanitized_tokenizer_digest,
            profile.model()["tokenizer_digest"]
        );
        assert!(numeric.input_grammar.is_none() && numeric.prompt_template.is_none());
        assert_eq!(numeric.certified_shape.max_sequences, 1);
        assert_eq!(numeric.certified_shape.max_context_tokens, 7500);
        assert_eq!(numeric.certified_shape.max_batch_tokens, 7500);
        assert_eq!(numeric.certified_shape.max_micro_batch_tokens, 7500);
        let spec = catalog_lane_spec(&state, entry, backend, false).unwrap();
        assert_eq!(spec.task, "transcribe");
        assert_eq!(spec.engine_identity, config.identity);
        assert_eq!(spec.fingerprint.0, backend.fingerprint);
        assert!(spec.owned_attention_units.is_none());
    }
    for id in [
        "gte-modernbert-base.owned-cuda",
        "gte-reranker-modernbert-base.owned-vulkan",
    ] {
        assert!(CatalogProfile::load(id)
            .unwrap()
            .transcribe_config()
            .unwrap_err()
            .to_string()
            .contains("has no transcribe config"));
    }
    drop(state);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn catalog_certified_duration_does_not_change_lane_identity_or_installed_bytes() {
    let (root, state) = isolated_catalog_state("speech-cap-identities", false);
    let mut lanes = 0;
    for entry in &state.runtime.release_catalog.models {
        let mut uncapped = entry.clone();
        for backend in &mut uncapped.backends {
            backend.certified_max_duration_ms = None;
        }
        assert_eq!(entry.manifest_digest(), uncapped.manifest_digest());
        for (backend, original) in entry.backends.iter().zip(&uncapped.backends) {
            let capped = catalog_lane_spec(&state, entry, backend, false).unwrap();
            let original = catalog_lane_spec(&state, &uncapped, original, false).unwrap();
            assert_eq!(capped.numeric_profile_id, original.numeric_profile_id);
            assert_eq!(capped.fingerprint, original.fingerprint);
            lanes += 1;
        }
    }
    assert_eq!(lanes, 18);
    drop(state);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn catalog_transcribe_lane_refuses_embed_and_rerank_before_token_length_checks() {
    let (root, state) = isolated_catalog_state("speech-task-refusal", false);
    let oversized = "a ".repeat(9000);
    for model in [
        SPEECH.to_string(),
        format!("{SPEECH}-metal"),
        format!("{SPEECH}-cuda"),
    ] {
        for query in [None, Some("query")] {
            let error =
                preflight_catalog_sequences(&state, Some(&model), None, None, query, &[&oversized])
                    .unwrap_err();
            assert_eq!(error.code, "invalid_request");
            assert_eq!(
                error.details.as_ref().unwrap()["catalog_task"],
                "transcribe"
            );
            assert!(!error.message.contains("sequence_too_long"));
        }
    }
    drop(state);
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn catalog_transcribe_list_and_catalog_publish_duration_not_token_advice() {
    let (root, mut state) = isolated_catalog_state("speech-audio-rows", true);
    Arc::get_mut(&mut Arc::get_mut(&mut state).unwrap().runtime)
        .unwrap()
        .runnable_backends
        .insert("ane".into());
    let entry = state.runtime.release_catalog.entry(SPEECH).unwrap();
    install_record(&state, entry, "metal");
    install_record(&state, entry, "ane");
    let listed = models_list_payload(&state);
    let row = listed["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["model_id"] == format!("{SPEECH}-metal"))
        .unwrap();
    assert_eq!(row["max_duration_ms"], 600000);
    assert_eq!(row["self_check"]["state"], "pending");
    let ane = listed["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["model_id"] == format!("{SPEECH}-ane"))
        .unwrap();
    assert_eq!(ane["max_duration_ms"], 81_850);
    for field in [
        "max_tokens",
        "max_tokens_source",
        "recommended_batch",
        "bucket_ladder",
        "dims",
        "pooling",
        "normalize",
        "attention_units",
    ] {
        assert!(row.get(field).is_none(), "{field}: {row}");
    }
    let catalog = response_result(
        models_catalog(state.clone(), json!({"task":"transcribe"})).await,
        "models.catalog",
    );
    let rows = catalog["models"].as_array().unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(catalog["catalog_revision"], 2);
    for row in rows[0]["backends"].as_array().unwrap() {
        assert_eq!(
            row["max_duration_ms"],
            if row["backend"] == "ane" {
                81_850
            } else {
                600_000
            }
        );
        for field in [
            "max_tokens",
            "attention_units",
            "pooling",
            "normalize",
            "dims",
            "recommended_batch",
        ] {
            assert!(row.get(field).is_none(), "{field}");
        }
    }
    drop(state);
    fs::remove_dir_all(root).unwrap();
}

#[cfg(not(target_os = "macos"))]
#[tokio::test]
async fn catalog_transcribe_fingerprint_mismatch_refuses_download_before_engine_invocation() {
    let (root, mut state) = isolated_catalog_state("speech-pin-refusal", false);
    let runtime = Arc::get_mut(&mut Arc::get_mut(&mut state).unwrap().runtime).unwrap();
    runtime.runnable_backends = BTreeSet::from(["cuda".into()]);
    runtime.backend_floor_observed.insert(
        "cuda".into(),
        json!({"driver_api":13020,"compute_capability":{"major":7,"minor":5}}),
    );
    let entry = runtime
        .release_catalog
        .models
        .iter_mut()
        .find(|entry| entry.id == SPEECH)
        .unwrap();
    entry
        .backends
        .iter_mut()
        .find(|b| b.backend == "cuda")
        .unwrap()
        .fingerprint = "0".repeat(64);
    let result = response_result(
        models_download(state.clone(), json!({"catalog_id":SPEECH})).await,
        "models.download",
    );
    assert_eq!(result["error"]["code"], "artifact_invalid");
    assert_eq!(
        result["error"]["details"]["expected_fingerprint"],
        "0".repeat(64)
    );
    assert_ne!(
        result["error"]["details"]["actual_fingerprint"],
        "0".repeat(64)
    );
    assert!(state.runtime.catalog.lock().unwrap().is_empty());
    assert!(state
        .store
        .active_download_job(&catalog_request_digest(
            state.runtime.release_catalog.entry(SPEECH).unwrap()
        ))
        .unwrap()
        .is_none());
    drop(state);
    fs::remove_dir_all(root).unwrap();
}

#[cfg(not(target_os = "macos"))]
#[tokio::test]
async fn catalog_transcribe_vulkan_floor_refuses_download_and_hides_installed_lane() {
    let (root, mut state) = isolated_catalog_state("speech-vulkan-floor", false);
    let runtime = Arc::get_mut(&mut Arc::get_mut(&mut state).unwrap().runtime).unwrap();
    runtime.runnable_backends = BTreeSet::from(["vulkan".into()]);
    runtime.backend_floor_observed.insert(
        "vulkan".into(),
        json!({"device_local_heaps":[3276015103u64],"max_storage_buffer_range":134217728}),
    );
    let entry = state.runtime.release_catalog.entry(SPEECH).unwrap();
    install_record(&state, entry, "vulkan");
    let result = response_result(
        models_download(state.clone(), json!({"catalog_id":SPEECH})).await,
        "models.download",
    );
    assert_eq!(result["error"]["code"], "backend_unavailable");
    let reasons = result["error"]["details"]["backends"].as_array().unwrap();
    let vulkan = reasons.iter().find(|r| r["backend"] == "vulkan").unwrap();
    assert_eq!(vulkan["reason"], "device_below_floor");
    assert!(vulkan["reason_details"]
        .as_str()
        .unwrap()
        .contains("3276015103"));
    let catalog = response_result(
        models_catalog(state.clone(), json!({"task":"transcribe"})).await,
        "models.catalog",
    );
    let backend = catalog["models"][0]["backends"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["backend"] == "vulkan")
        .unwrap();
    assert_eq!(backend["runnable"], false);
    assert_eq!(backend["reason"], "device_below_floor");
    let listed = models_list_payload(&state);
    assert!(listed["models"]
        .as_array()
        .unwrap()
        .iter()
        .all(|r| r["model_id"] != format!("{SPEECH}-vulkan")));
    let embed = state
        .runtime
        .release_catalog
        .entry("gte-modernbert-base")
        .unwrap();
    assert_eq!(
        catalog_lane_backend_reason(
            &state.runtime,
            embed,
            embed.backend("vulkan").unwrap(),
            synapse_core::Platform::current()
        )
        .map(|r| r.code),
        catalog_backend_reason(&state.runtime, "vulkan", synapse_core::Platform::current())
    );
    drop(state);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn catalog_transcribe_floors_use_manifest_values_and_require_observations() {
    for lane in ["owned-cuda", "owned-vulkan"] {
        let profile = CatalogProfile::load(&format!("{SPEECH}.{lane}")).unwrap();
        let missing = transcribe_profile_floor_reason(profile.profile(), &Value::Null).unwrap();
        assert_eq!(missing.code, "probe_failed");
        assert!(missing.details.contains("missing"));
    }
    let cuda = CatalogProfile::load(&format!("{SPEECH}.owned-cuda")).unwrap();
    for observed in [
        json!({"driver_api":13019,"compute_capability":{"major":7,"minor":5}}),
        json!({"driver_api":13020,"compute_capability":{"major":7,"minor":4}}),
    ] {
        assert_eq!(
            transcribe_profile_floor_reason(cuda.profile(), &observed)
                .unwrap()
                .code,
            "device_below_floor"
        );
    }
    assert!(transcribe_profile_floor_reason(
        cuda.profile(),
        &json!({"driver_api":13020,"compute_capability":{"major":8,"minor":0}})
    )
    .is_none());
    let vulkan = CatalogProfile::load(&format!("{SPEECH}.owned-vulkan")).unwrap();
    let refusal = transcribe_profile_floor_reason(
        vulkan.profile(),
        &json!({"device_local_heaps":[3276015104u64],"max_storage_buffer_range":134217727}),
    )
    .unwrap();
    assert_eq!(refusal.code, "device_below_floor");
    assert!(refusal.details.contains("storage buffer range"));
}

#[tokio::test]
async fn catalog_transcribe_models_download_installs_checksum_pinned_files_without_inference() {
    let (root, mut state) = isolated_catalog_state("speech-download", true);
    let cache = state.model_cache.clone();
    let runtime = Arc::get_mut(&mut Arc::get_mut(&mut state).unwrap().runtime).unwrap();
    let entry = runtime
        .release_catalog
        .models
        .iter_mut()
        .find(|entry| entry.id == SPEECH)
        .unwrap();
    // Tiny checksum-pinned cache fixtures exercise download publication, not model execution.
    for file in &mut entry.files {
        let bytes = file.path.as_bytes();
        file.sha256 = sha256_hex(bytes);
        file.size_bytes = bytes.len() as u64;
        let path = cache.blob_path(&file.sha256);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, bytes).unwrap();
    }
    let result = response_result(
        models_download(state.clone(), json!({"catalog_id": SPEECH})).await,
        "models.download",
    );
    let job = result["job_id"].as_str().unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let record = state.store.get_job(job).unwrap().unwrap();
            assert!(
                !matches!(record.state.as_str(), "failed" | "cancelled"),
                "{record:?}"
            );
            if record.state == "committed" && Arc::strong_count(&state) == 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let entry = state.runtime.release_catalog.entry(SPEECH).unwrap();
    assert!(current_catalog_install(&state, entry, "metal")
        .unwrap()
        .is_some());
    let slot = model_slot_snapshot(&state.runtime, &format!("{SPEECH}-metal")).unwrap();
    assert_eq!(slot.spec.task, "transcribe");
    assert!(slot.loaded.is_none());
    assert_eq!(
        slot.spec.fingerprint.0,
        entry.backend("metal").unwrap().fingerprint
    );
    drop(state);
    fs::remove_dir_all(root).unwrap();
}
