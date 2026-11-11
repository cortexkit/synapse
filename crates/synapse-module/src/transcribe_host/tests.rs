use super::*;

fn profile() -> TranscribeProfile {
    TranscribeProfile::pinned("parakeet-tdt-0.6b-v3.owned-metal").unwrap()
}

fn wav(samples: usize, rate: u32) -> Vec<u8> {
    let mut bytes = b"RIFF".to_vec();
    bytes.extend_from_slice(&(36 + samples as u32 * 2).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt \x10\0\0\0\x01\0\x01\0");
    bytes.extend_from_slice(&rate.to_le_bytes());
    bytes.extend_from_slice(&(rate * 2).to_le_bytes());
    bytes.extend_from_slice(b"\x02\0\x10\0data");
    bytes.extend_from_slice(&(samples as u32 * 2).to_le_bytes());
    bytes.resize(44 + samples * 2, 0);
    bytes
}

#[test]
fn transcribe_profiles_have_encoder_envelopes_and_task_specific_revisions() {
    for (lane, kernel) in [
        ("owned-metal", METAL_TRANSCRIBE_KERNEL_REVISION),
        ("owned-cuda", CUDA_TRANSCRIBE_KERNEL_REVISION),
        ("owned-vulkan", VULKAN_TRANSCRIBE_KERNEL_REVISION),
        ("ane-direct-worker", ANE_DIRECT_TRANSCRIBE_KERNEL_REVISION),
    ] {
        let p = TranscribeProfile::pinned(&format!("parakeet-tdt-0.6b-v3.{lane}")).unwrap();
        assert_eq!(p.max_duration_ms, 600_000);
        assert_eq!(p.max_encoder_frames, 7500);
        assert_eq!(p.kernel_revision, kernel);
        assert_eq!(p.decode_rule, DECODE_RULE_REVISION);
        let cfg = p.runtime_config(&RuntimeConfig::default());
        assert_eq!(cfg.values["family"], "parakeet-tdt");
        assert_eq!(cfg.values["max_encoder_frames"], "7500");
        assert!(!cfg.values.contains_key("max_tokens"));
        assert_eq!(
            cfg.values["execution"],
            if lane == "owned-metal" {
                "explicit"
            } else {
                "supervised"
            }
        );
    }
}

#[test]
fn transcribe_preparation_is_shared_unpadded_mel_and_observation_is_certification_only() {
    let wav = wav(1600, 16_000);
    let direct = Frontend::new().process_wav(&wav).unwrap();
    for mode in [ObservationMode::Serving, ObservationMode::Certification] {
        let prepared = prepare_wav(&profile(), &wav, mode).unwrap();
        assert_eq!(prepared.request.mel, direct.values);
        assert_eq!(prepared.request.mel_frames, 11);
        assert_eq!(prepared.request.valid_mel_frames, 10);
        assert_eq!(prepared.valid_encoder_frames, 2);
        assert_eq!(
            prepared.request.observe,
            mode == ObservationMode::Certification
        );
    }
    assert!(matches!(
        prepare_wav(&profile(), b"not audio", ObservationMode::Serving),
        Err(TranscribeHostError::Audio(_))
    ));
}

#[test]
fn transcribe_duration_and_resampled_frame_limits_refuse_before_features() {
    let mut p = profile();
    p.max_duration_ms = 100;
    p.max_encoder_frames = 2;
    let error = prepare_wav(&p, &wav(4411, 44_100), ObservationMode::Serving)
        .err()
        .unwrap();
    let TranscribeHostError::DurationLimit(details) = error else {
        panic!("expected duration refusal")
    };
    assert_eq!(
        serde_json::to_value(details).unwrap(),
        serde_json::json!({
            "duration_ms": 101, "max_duration_ms": 100,
            "valid_encoder_frames": 2, "max_encoder_frames": 2,
        })
    );
    p.max_encoder_frames = 1;
    let error = prepare_wav(&p, &wav(4410, 44_100), ObservationMode::Serving)
        .err()
        .unwrap();
    let TranscribeHostError::DurationLimit(details) = error else {
        panic!("expected frame refusal")
    };
    assert_eq!(
        details,
        DurationLimit {
            duration_ms: 100,
            max_duration_ms: 100,
            valid_encoder_frames: 2,
            max_encoder_frames: 1
        }
    );
}

#[test]
fn transcribe_largest_padded_mel_frame_must_fit_negotiated_limit() {
    let p = profile();
    let bytes = (8192 * 8 + 1) * 128 * 4;
    p.validate_max_frame(bytes).unwrap();
    assert_eq!(
        p.validate_max_frame(bytes - 1).unwrap_err().code(),
        Some("transcribe_frame_limit")
    );
}

#[test]
fn transcribe_loaded_revision_is_task_specific_and_mandatory() {
    let p = TranscribeProfile::pinned("parakeet-tdt-0.6b-v3.owned-cuda").unwrap();
    let cfg = p.runtime_config(&RuntimeConfig::default());
    let revision = TranscribeRevision {
        kernel_revision: p.kernel_revision,
        decode_rule: p.decode_rule,
    };
    validate_worker_load(&cfg, 64 * 1024 * 1024, Some(&revision)).unwrap();
    for stale in [
        None,
        Some(TranscribeRevision {
            kernel_revision: "old-transcribe-v0".into(),
            ..revision.clone()
        }),
        Some(TranscribeRevision {
            decode_rule: "old-rule".into(),
            ..revision.clone()
        }),
    ] {
        assert_eq!(
            validate_worker_load(&cfg, 64 * 1024 * 1024, stale.as_ref())
                .unwrap_err()
                .code(),
            Some("transcribe_revision_mismatch")
        );
    }
    validate_worker_load(&RuntimeConfig::default(), 1, None).unwrap();
}

#[test]
fn transcribe_result_requires_shared_frame_count_and_clamps_to_original_duration() {
    let output = TranscribeOutput {
        valid_encoder_frames: 1,
        triples: vec![synapse_core::TranscriptRecord {
            token_id: 5,
            frame_index: 0,
            duration: 4,
        }],
        observation: None,
    };
    assert!(finish(output.clone(), None, None, vec![], 2, 23.5).is_err());
    let result = finish(output, None, None, vec![], 1, 23.5).unwrap();
    assert_eq!(result.transcript.words[0].end_ms, 23.5);
    let silence = finish(
        TranscribeOutput {
            valid_encoder_frames: 1,
            ..Default::default()
        },
        None,
        None,
        vec![],
        1,
        80.0,
    )
    .unwrap();
    assert!(silence.transcript.no_speech);
    assert!(silence.transcript.words.is_empty());
    assert_eq!(silence.transcript.text, "");
}

#[test]
fn transcribe_workers_never_reuse_other_lane_ids_or_hello_bindings() {
    let existing = WorkerHostConfig::new("worker", "runtime");
    let mut a = existing.clone();
    let mut b = existing.clone();
    isolate_worker(&mut a);
    isolate_worker(&mut b);
    assert_ne!(a.worker_id, existing.worker_id);
    assert_ne!(a.worker_id, b.worker_id);
    assert!(a.isolate_crash_key_by_worker_id);
    assert!(a.expected_hello_binding.is_none());
    assert_eq!(existing.worker_id, "default");
    assert!(!existing.isolate_crash_key_by_worker_id);
}

#[cfg(unix)]
#[test]
fn transcribe_host_self_check_keeps_observe_off_and_refuses_audio_before_worker_work() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    let audio = wav(1600, 16_000);
    let expected =
        synapse_core::encode_f32_frame(&Frontend::new().process_wav(&audio).unwrap().values);
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = calls.clone();
    let engine = WorkerEngine::new(WorkerHostConfig::new("/bin/false", "unused")).unwrap();
    let model = engine.insert_loaded_model_for_test("speech".into(), 1024, 0, None);
    engine.attach_mock_worker_for_test(move |request, raw| {
        let synapse_core::WorkerRequest::Transcribe {
            req_id,
            mel_frames,
            valid_mel_frames,
            observe,
            ..
        } = request
        else {
            panic!("only TRANSCRIBE is allowed");
        };
        assert!(
            !observe,
            "self-checks never request certification observations"
        );
        assert_eq!((*mel_frames, *valid_mel_frames), (11, 10));
        assert_eq!(raw.unwrap(), expected);
        seen.fetch_add(1, Ordering::Relaxed);
        (
            WorkerResponse::Transcript {
                req_id: req_id.clone(),
                valid_encoder_frames: 2,
                count: 0,
            },
            Some(vec![]),
        )
    });
    let mut profile = TranscribeProfile::pinned("parakeet-tdt-0.6b-v3.owned-cuda").unwrap();
    profile.max_duration_ms = 100;
    let host = TranscribeHost {
        profile,
        mode: ObservationMode::Certification,
        backend: Backend::Worker(engine, model),
    };
    let result = host
        .transcribe_self_check(&audio, |_| panic!("not a Metal lane"))
        .unwrap();
    assert!(result.transcript.no_speech);
    assert!(result.observation.is_none());
    assert!(result.transcript_frame.is_empty());
    assert!(matches!(
        host.transcribe(b"not a WAV", |_| {}),
        Err(TranscribeHostError::Audio(_))
    ));
    assert!(matches!(
        host.transcribe(&wav(1601, 16_000), |_| {}),
        Err(TranscribeHostError::DurationLimit(_))
    ));
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}
