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
        .store
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
