use super::*;

fn wav(samples: usize) -> Vec<u8> {
    let mut bytes = b"RIFF".to_vec();
    bytes.extend_from_slice(&(36 + samples as u32 * 2).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt \x10\0\0\0\x01\0\x01\0");
    bytes.extend_from_slice(&16_000u32.to_le_bytes());
    bytes.extend_from_slice(&32_000u32.to_le_bytes());
    bytes.extend_from_slice(b"\x02\0\x10\0data");
    bytes.extend_from_slice(&(samples as u32 * 2).to_le_bytes());
    bytes.resize(44 + samples * 2, 0);
    bytes
}

fn selected_profile(state: &ModuleState) -> transcribe_host::TranscribeProfile {
    transcribe_host::TranscribeProfile::pinned(
        state
            .runtime
            .release_catalog
            .entry(MODEL)
            .unwrap()
            .backend(&first(state))
            .unwrap()
            .profile
            .as_deref()
            .unwrap(),
    )
    .unwrap()
}

fn piece_id(piece: &str) -> i32 {
    let tokenizer = synapse_speech::Tokenizer::pinned();
    (0..synapse_speech::BLANK)
        .find(|&id| tokenizer.piece(id).unwrap() == piece)
        .unwrap() as i32
}

fn triples(values: &[[i32; 3]]) -> Vec<synapse_core::TranscriptRecord> {
    values
        .iter()
        .map(
            |&[token_id, frame_index, duration]| synapse_core::TranscriptRecord {
                token_id,
                frame_index,
                duration,
            },
        )
        .collect()
}

#[cfg(unix)]
fn certification_mock(
    profile: transcribe_host::TranscribeProfile,
    seen: Arc<std::sync::Mutex<Vec<bool>>>,
) -> transcribe_host::TranscribeHost {
    use synapse_core::{WorkerRequest, WorkerResponse};
    let engine =
        worker_host::WorkerEngine::new(worker_host::WorkerHostConfig::new("unused", "unused"))
            .unwrap();
    let model = engine.insert_loaded_model_for_test("speech".into(), 1024, 0, None);
    engine.attach_mock_worker_frames_for_test(move |request, _| {
        let WorkerRequest::Transcribe {
            req_id,
            observe,
            valid_mel_frames,
            ..
        } = request
        else {
            panic!("unexpected request")
        };
        let frames = valid_mel_frames.div_ceil(8);
        let mut seen = seen.lock().unwrap();
        let peak = if seen.is_empty() { 123456 } else { 999999 };
        seen.push(*observe);
        std::thread::sleep(Duration::from_millis(23));
        let mut replies = Vec::new();
        if *observe {
            replies.push((
                WorkerResponse::EncoderOutput {
                    req_id: req_id.clone(),
                    frames,
                },
                Some(vec![0; frames * 4096]),
            ));
            replies.push((
                WorkerResponse::DecodeTrace {
                    req_id: req_id.clone(),
                    steps: 0,
                },
                Some(vec![]),
            ));
        }
        replies.push((
            WorkerResponse::Transcript {
                req_id: req_id.clone(),
                valid_encoder_frames: frames,
                count: 1,
                peak_device_memory_bytes: Some(peak),
                peak_storage_buffer_bytes: Some(65432),
            },
            Some(synapse_core::encode_transcript_frame(&triples(&[[
                piece_id("▁the"),
                0,
                1,
            ]]))),
        ));
        replies
    });
    transcribe_host::TranscribeHost::worker_for_test(profile, engine, model)
}

#[cfg(unix)]
#[tokio::test]
async fn transcribe_certification_snapshot_units_and_backend_nullability_from_mock() {
    let result = tokio::task::spawn_blocking(|| {
        certification_mock(
            transcribe_host::TranscribeProfile::pinned("parakeet-tdt-0.6b-v3.owned-cuda").unwrap(),
            Default::default(),
        )
        .transcribe_self_check(&wav(4008), |_| {})
        .unwrap()
    })
    .await
    .unwrap();
    let shapes = json!([{"encoder_frames":128,"executables":(0..24).map(|layer|json!({"id":format!("layer-{layer}"),"layers":[layer]})).collect::<Vec<_>>(),"cpu_stages":["frontend"]}]);
    for backend in ["cuda", "vulkan", "metal", "ane"] {
        let snapshot = measurement_snapshot(
            backend,
            &result,
            Some(37),
            shapes.as_array().unwrap().clone(),
        );
        assert_eq!(snapshot.as_object().unwrap().len(), 5);
        assert_eq!(snapshot["self_check_wall_ms"], 37);
        assert!(snapshot["longest_progress_interval_ms"].as_u64().unwrap() >= 23);
        assert_eq!(
            snapshot["peak_device_memory_bytes"],
            if matches!(backend, "cuda" | "vulkan") {
                json!(123456)
            } else {
                Value::Null
            }
        );
        assert_eq!(
            snapshot["peak_storage_buffer_bytes"],
            if backend == "vulkan" {
                json!(65432)
            } else {
                Value::Null
            }
        );
        assert_eq!(
            snapshot["ane_shapes"],
            if backend == "ane" {
                shapes.clone()
            } else {
                json!([])
            }
        );
    }
}

#[cfg(unix)]
async fn fresh_upload(state: &Arc<ModuleState>, owner: &str, bytes: &[u8]) -> Value {
    let mut facts = media(bytes);
    facts.as_object_mut().unwrap().remove("data");
    let upload = call(state, owner, "media.upload_begin", facts.clone())
        .await
        .unwrap()["upload_id"]
        .clone();
    call(
        state,
        owner,
        "media.upload_append",
        json!({"upload_id":upload,"offset":0,"data":base64_encode(bytes)}),
    )
    .await
    .unwrap();
    call(
        state,
        owner,
        "media.upload_commit",
        json!({"upload_id":upload}),
    )
    .await
    .unwrap();
    facts["object_id"] = upload;
    facts
}

#[cfg(unix)]
#[tokio::test]
async fn transcribe_certification_measurements_freeze_and_observe_rerun_is_private_identical_and_deletes_upload(
) {
    let (root, mut state) = fixture("transcribe-certification-rerun", true);
    Arc::get_mut(&mut Arc::get_mut(&mut state).unwrap().runtime)
        .unwrap()
        .certify_observation = true;
    *state.runtime.media.runner.lock().unwrap() = None;
    let backend = first(&state);
    let lane = catalog::lane_id(MODEL, &backend);
    let entry = state.runtime.release_catalog.entry(MODEL).unwrap();
    let check_lane = catalog_transcribe_lane(entry, entry.backend(&backend).unwrap()).unwrap();
    let (check_id, key) = transcribe_self_check::key(&state, &check_lane).unwrap();
    let generation = catalog_check_generation(&state, &check_id, &key).unwrap();
    catalog_complete_check(&state, &check_id, generation, "passed", None).unwrap();
    state
        .store
        .sqlite()
        .with_conn_fenced(|tx| {
            tx.execute(
                "UPDATE catalog_self_checks SET self_check_wall_ms=37 WHERE check_id=?1",
                [&check_id],
            )?;
            Ok(())
        })
        .unwrap();
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let profile = selected_profile(&state);
    let recorded = seen.clone();
    let host = tokio::task::spawn_blocking(move || certification_mock(profile, recorded))
        .await
        .unwrap();
    lane_cache(&state.runtime, &lane).lock().await.host = Some(host);
    let bytes = wav(4008);
    let mut submit = request("certification-original");
    submit["input"]["audio"] = media(&bytes);
    let id = call(&state, "alice", "media.submit", submit).await.unwrap()["job_id"].clone();
    assert_eq!(done(&state, id.as_str().unwrap()).await["status"], "done");
    let original = call(&state, "alice", "media.result", json!({"job_id":id}))
        .await
        .unwrap();
    let measurements = call(
        &state,
        "alice",
        "certify.transcribe_measurements",
        json!({"job_id":id}),
    )
    .await
    .unwrap();
    assert_eq!(measurements["self_check_wall_ms"], 37);
    assert!(
        measurements["longest_progress_interval_ms"]
            .as_u64()
            .unwrap()
            >= 23
    );
    assert_eq!(measurements.as_object().unwrap().len(), 5);
    assert!(!original
        .as_object()
        .unwrap()
        .keys()
        .any(|key| key.starts_with('_')));
    let original_object: String = state
        .store
        .sqlite()
        .with_conn_fenced(|tx| {
            let object = tx.query_row(
                "SELECT object_id FROM media_jobs WHERE id=?1",
                [id.as_str().unwrap()],
                |row| row.get::<_, String>(0),
            )?;
            tx.execute(
                "UPDATE media_objects SET state='gone' WHERE id=?1",
                [&object],
            )?;
            Ok(object)
        })
        .unwrap();
    fs::remove_file(
        persistence::directory(&state)
            .unwrap()
            .join(original_object),
    )
    .unwrap();

    for method in [
        "certify.transcribe_measurements",
        "certify.transcribe_observe_rerun",
    ] {
        assert_eq!(
            call(&state, "bob", method, json!({"job_id":id}))
                .await
                .unwrap_err()
                .0["code"],
            "unknown_job"
        );
    }
    let different = fresh_upload(&state, "alice", &wav(4010)).await;
    let before = state
        .store
        .sqlite()
        .with_conn(|c| Ok(c.total_changes()))
        .unwrap();
    assert_eq!(
        call(
            &state,
            "alice",
            "certify.transcribe_observe_rerun",
            json!({"job_id":id,"audio":different})
        )
        .await
        .unwrap_err()
        .0["detail"]["field"],
        "audio.sha256"
    );
    assert_eq!(
        state
            .store
            .sqlite()
            .with_conn(|c| Ok(c.total_changes()))
            .unwrap(),
        before,
        "SHA refusal writes nothing"
    );
    assert_eq!(*seen.lock().unwrap(), [true]);

    let upload = fresh_upload(&state, "alice", &bytes).await;
    for (field, value) in [("bytes", json!(1)), ("mime", json!("audio/mp3"))] {
        let mut wrong = upload.clone();
        wrong[field] = value;
        assert!(call(
            &state,
            "alice",
            "certify.transcribe_observe_rerun",
            json!({"job_id":id,"audio":wrong})
        )
        .await
        .is_err());
    }
    let foreign = fresh_upload(&state, "bob", &bytes).await;
    assert_eq!(
        call(
            &state,
            "alice",
            "certify.transcribe_observe_rerun",
            json!({"job_id":id,"audio":foreign})
        )
        .await
        .unwrap_err()
        .0["code"],
        "unknown_object"
    );
    let held = lane_cache(&state.runtime, &lane).lock_owned().await;
    let rerun_state = state.clone();
    let rerun_params = json!({"job_id":id,"audio":upload});
    let pending = tokio::spawn(async move {
        call(
            &rerun_state,
            "alice",
            "certify.transcribe_observe_rerun",
            rerun_params,
        )
        .await
    });
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert_eq!(
        *seen.lock().unwrap(),
        [true],
        "rerun must wait for the normal lane lock"
    );
    drop(held);
    let rerun = pending.await.unwrap().unwrap();
    assert_eq!(
        rerun,
        json!({"observe":false,"valid_encoder_frames":original["observation"]["valid_encoder_frames"],"triples":original["observation"]["triples"]})
    );
    assert_eq!(*seen.lock().unwrap(), [true, false]);
    assert_eq!(
        call(
            &state,
            "alice",
            "media.upload_status",
            json!({"upload_id":upload["object_id"]})
        )
        .await
        .unwrap()["state"],
        "gone"
    );
    assert!(!persistence::directory(&state)
        .unwrap()
        .join(upload["object_id"].as_str().unwrap())
        .exists());
    assert_eq!(
        call(
            &state,
            "alice",
            "certify.transcribe_measurements",
            json!({"job_id":id})
        )
        .await
        .unwrap(),
        measurements
    );
    assert_eq!(
        call(&state, "alice", "media.result", json!({"job_id":id}))
            .await
            .unwrap(),
        original
    );
    assert_eq!(
        call(
            &state,
            "alice",
            "certify.transcribe_observe_rerun",
            json!({"job_id":id,"audio":upload})
        )
        .await
        .unwrap_err()
        .0["code"],
        "unknown_object"
    );
    state
        .store
        .sqlite()
        .with_conn_fenced(|tx| {
            tx.execute(
                "UPDATE catalog_self_checks SET self_check_wall_ms=777 WHERE check_id=?1",
                [&check_id],
            )?;
            Ok(())
        })
        .unwrap();
    let mut later = request("certification-later");
    later["input"]["audio"] = media(&bytes);
    let later_id = call(&state, "alice", "media.submit", later).await.unwrap()["job_id"].clone();
    assert_eq!(
        done(&state, later_id.as_str().unwrap()).await["status"],
        "done"
    );
    let later_measurements = call(
        &state,
        "alice",
        "certify.transcribe_measurements",
        json!({"job_id":later_id}),
    )
    .await
    .unwrap();
    assert_eq!(later_measurements["self_check_wall_ms"], 777);
    if matches!(backend.as_str(), "cuda" | "vulkan") {
        assert_eq!(later_measurements["peak_device_memory_bytes"], 999999);
    }
    assert_eq!(
        call(
            &state,
            "alice",
            "certify.transcribe_measurements",
            json!({"job_id":id})
        )
        .await
        .unwrap(),
        measurements
    );
    cleanup(root, state).await;
}

#[tokio::test]
async fn transcribe_certification_operations_refuse_without_observation_mode() {
    let (root, state) = fixture("transcribe-certification-disabled", true);
    for method in [
        "certify.transcribe_measurements",
        "certify.transcribe_observe_rerun",
    ] {
        let before = state
            .store
            .sqlite()
            .with_conn(|c| Ok(c.total_changes()))
            .unwrap();
        assert_eq!(
            call(&state, "alice", method, json!({"job_id":"unknown"}))
                .await
                .unwrap_err()
                .0["code"],
            "certify_observation_disabled"
        );
        assert_eq!(
            state
                .store
                .sqlite()
                .with_conn(|c| Ok(c.total_changes()))
                .unwrap(),
            before
        );
    }
    cleanup(root, state).await;
}

#[tokio::test]
async fn media_results_use_shared_words_silence_zero_durations_and_final_clamp() {
    let the = piece_id("▁the");
    let s = piece_id("s");
    for (name, samples, emitted, expected) in [
        (
            "silence-1s",
            16_000,
            vec![],
            json!({"text":"","words":[],"no_speech":true}),
        ),
        (
            "silence-30s",
            480_000,
            vec![],
            json!({"text":"","words":[],"no_speech":true}),
        ),
        (
            "leading-silence",
            16_000,
            vec![[s, 2, 1], [the, 4, 2]],
            json!({
                "text":"s the", "words":[{"text":"s","start_ms":160,"end_ms":240},
                    {"text":"the","start_ms":320,"end_ms":480}],"no_speech":false
            }),
        ),
        (
            "zero-durations",
            16_000,
            vec![[the, 0, 0], [s, 0, 0], [the, 0, 0]],
            json!({
                "text":"thes the", "words":[{"text":"thes","start_ms":0,"end_ms":80},
                    {"text":"the","start_ms":0,"end_ms":80}],"no_speech":false
            }),
        ),
        (
            "final-clamp",
            4_008,
            vec![[the, 3, 4]],
            json!({
                "text":"the", "words":[{"text":"the","start_ms":240,"end_ms":250}],"no_speech":false
            }),
        ),
    ] {
        let (root, state) = fixture(name, true);
        *state.runtime.media.runner.lock().unwrap() = None;
        let lane = catalog::lane_id(MODEL, &first(&state));
        let emitted = triples(&emitted);
        lane_cache(&state.runtime, &lane).lock().await.host =
            Some(transcribe_host::TranscribeHost::in_process_for_test(
                selected_profile(&state),
                move |_| {
                    Ok(synapse_core::TranscribeOutput {
                        valid_encoder_frames: synapse_speech::valid_encoder_frames(samples as u64)
                            as usize,
                        triples: emitted.clone(),
                        observation: None,
                    })
                },
            ));
        let mut p = request(name);
        p["input"]["audio"] = media(&wav(samples));
        let id = call(&state, "alice", "media.submit", p).await.unwrap()["job_id"].clone();
        assert_eq!(
            done(&state, id.as_str().unwrap()).await["status"],
            "done",
            "{name}"
        );
        let output = call(&state, "alice", "media.result", json!({"job_id":id}))
            .await
            .unwrap();
        assert_eq!(output, expected, "{name}");
        assert_eq!(
            call(
                &state,
                "alice",
                "certify.transcribe_observation",
                json!({"job_id":id,"part":"encoder_output","page":0})
            )
            .await
            .unwrap_err()
            .0["code"],
            "certify_observation_disabled"
        );
        cleanup(root, state).await;
    }
}

#[cfg(unix)]
#[tokio::test]
async fn media_results_mock_worker_observation_pages_are_exact_private_and_expire() {
    use synapse_core::{DecodeTraceRecord, WorkerRequest, WorkerResponse};
    let (root, mut state) = fixture("media-framed-observation", true);
    Arc::get_mut(&mut Arc::get_mut(&mut state).unwrap().runtime)
        .unwrap()
        .certify_observation = true;
    *state.runtime.media.runner.lock().unwrap() = None;
    let backend = first(&state);
    let lane = catalog::lane_id(MODEL, &backend);
    let fingerprint = state
        .runtime
        .release_catalog
        .entry(MODEL)
        .unwrap()
        .backend(&backend)
        .unwrap()
        .fingerprint
        .clone();
    let encoder = synapse_core::encode_f32_frame(
        &(0..129 * 1024).map(|n| (n % 31) as f32).collect::<Vec<_>>(),
    );
    let emitted = triples(&[[piece_id("▁the"), 128, 4]]);
    let transcript = synapse_core::encode_transcript_frame(&emitted);
    let trace_records: Vec<_> = (0..=128)
        .step_by(4)
        .map(|frame_index| DecodeTraceRecord {
            frame_index,
            token_top1: if frame_index == 128 {
                emitted[0].token_id
            } else {
                8192
            },
            token_top2: if frame_index == 128 {
                8192
            } else {
                emitted[0].token_id
            },
            duration_top1: 4,
            duration_top2: 1,
            token_logit1: 2.0,
            token_logit2: 1.0,
            duration_logit1: 3.0,
            duration_logit2: 1.0,
            flags: u32::from(frame_index == 128),
        })
        .collect();
    let trace = synapse_core::encode_decode_trace_frame(&trace_records);
    let profile = selected_profile(&state);
    let sent_encoder = encoder.clone();
    let sent_trace = trace.clone();
    let sent_transcript = transcript.clone();
    let host = tokio::task::spawn_blocking(move || {
        let engine =
            worker_host::WorkerEngine::new(worker_host::WorkerHostConfig::new("unused", "unused"))
                .unwrap();
        let model = engine.insert_loaded_model_for_test("speech".into(), 1024, 0, None);
        engine.attach_mock_worker_frames_for_test(move |request, raw| {
            let WorkerRequest::Transcribe {
                req_id,
                observe,
                mel_frames,
                valid_mel_frames,
                ..
            } = request
            else {
                panic!("unexpected worker request: {request:?}")
            };
            assert!(*observe);
            assert_eq!((*mel_frames, *valid_mel_frames), (1033, 1032));
            assert_eq!(raw.unwrap().len(), 1033 * 128 * 4);
            vec![
                (
                    WorkerResponse::EncoderOutput {
                        req_id: req_id.clone(),
                        frames: 129,
                    },
                    Some(sent_encoder.clone()),
                ),
                (
                    WorkerResponse::DecodeTrace {
                        req_id: req_id.clone(),
                        steps: 33,
                    },
                    Some(sent_trace.clone()),
                ),
                (
                    WorkerResponse::Transcript {
                        req_id: req_id.clone(),
                        valid_encoder_frames: 129,
                        count: 1,
                        peak_device_memory_bytes: None,
                        peak_storage_buffer_bytes: None,
                    },
                    Some(sent_transcript.clone()),
                ),
            ]
        });
        transcribe_host::TranscribeHost::worker_for_test(profile, engine, model)
    })
    .await
    .unwrap();
    lane_cache(&state.runtime, &lane).lock().await.host = Some(host);
    let mut p = request("framed-observation");
    p["input"]["audio"] = media(&wav(129 * 1280));
    let id = call(&state, "alice", "media.submit", p).await.unwrap()["job_id"].clone();
    assert_eq!(done(&state, id.as_str().unwrap()).await["status"], "done");
    let output = call(&state, "alice", "media.result", json!({"job_id":id}))
        .await
        .unwrap();
    assert_eq!(output["text"], "the");
    assert!(output.get("language").is_none());
    assert!(output.get("_observation").is_none());
    let observation = &output["observation"];
    assert_eq!(observation["lane_id"], lane);
    assert_eq!(observation["fingerprint"], fingerprint);
    assert_eq!(observation["valid_encoder_frames"], 129);
    assert_eq!(
        STANDARD
            .decode(observation["triples"].as_str().unwrap())
            .unwrap(),
        transcript
    );
    assert_eq!(observation["encoder_output"]["pages"], 2);
    for (part, expected) in [("encoder_output", encoder), ("decode_trace", trace)] {
        let descriptor = &observation[part];
        assert_eq!(descriptor["bytes"], expected.len());
        assert_eq!(
            descriptor["sha256"],
            format!("{:x}", Sha256::digest(&expected))
        );
        let mut reassembled = Vec::new();
        let pages = descriptor["pages"].as_u64().unwrap();
        for page in 0..pages {
            let reply = call(
                &state,
                "alice",
                "certify.transcribe_observation",
                json!({"job_id":id,"part":part,"page":page}),
            )
            .await
            .unwrap();
            let bytes = STANDARD.decode(reply["data"].as_str().unwrap()).unwrap();
            assert!(!bytes.is_empty() && bytes.len() <= PAGE);
            assert_eq!(
                bytes.len(),
                (expected.len() - page as usize * PAGE).min(PAGE)
            );
            reassembled.extend(bytes);
        }
        assert_eq!(reassembled, expected);
        assert_eq!(
            reassembled.len(),
            descriptor["bytes"].as_u64().unwrap() as usize
        );
        assert_eq!(
            format!("{:x}", Sha256::digest(&reassembled)),
            descriptor["sha256"]
        );
        assert_eq!(
            call(
                &state,
                "alice",
                "certify.transcribe_observation",
                json!({"job_id":id,"part":part,"page":pages})
            )
            .await
            .unwrap_err()
            .0["detail"]["field"],
            "page"
        );
    }
    assert_eq!(
        call(
            &state,
            "bob",
            "certify.transcribe_observation",
            json!({"job_id":id,"part":"encoder_output","page":0})
        )
        .await
        .unwrap_err()
        .0["code"],
        "unknown_job"
    );
    for (part, page, field) in [
        ("triples", json!(0), "part"),
        ("decode_trace", json!(-1), "page"),
    ] {
        assert_eq!(
            call(
                &state,
                "alice",
                "certify.transcribe_observation",
                json!({"job_id":id,"part":part,"page":page})
            )
            .await
            .unwrap_err()
            .0["detail"]["field"],
            field
        );
    }
    persistence::sweep(&state, now_ms() + RETENTION_MS + 1).unwrap();
    let pages: u64 = state
        .store
        .sqlite()
        .with_conn(|c| {
            c.query_row("SELECT count(*) FROM media_observation_pages", [], |r| {
                r.get(0)
            })
        })
        .unwrap();
    assert_eq!(pages, 0);
    assert_eq!(
        call(
            &state,
            "alice",
            "certify.transcribe_observation",
            json!({"job_id":id,"part":"encoder_output","page":0})
        )
        .await
        .unwrap_err()
        .0["code"],
        "unknown_job"
    );
    cleanup(root, state).await;
}
