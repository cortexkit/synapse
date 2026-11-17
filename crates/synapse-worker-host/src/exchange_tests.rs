use super::*;

fn profile() -> TranscribeProfile {
    TranscribeProfile::pinned("parakeet-tdt-0.6b-v3.owned-metal").unwrap()
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
