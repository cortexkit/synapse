#![forbid(unsafe_code)]

use synapse_core::error_contract::StableErrorCode;

const DOC: &str = include_str!("../../../docs/wire-contract-v1.md");

// An exhaustive match deliberately makes an added enum variant require documentation review,
// even if the variant was accidentally omitted from StableErrorCode::ALL.
fn stable_name(code: StableErrorCode) -> &'static str {
    use StableErrorCode::*;
    match code {
        QueueFull => "queue_full",
        SequenceTooLong => "sequence_too_long",
        DeadlineExceeded => "deadline_exceeded",
        ModelLoading => "model_loading",
        LanePaused => "lane_paused",
        NotCertified => "not_certified",
        SubstitutionRejected => "substitution_rejected",
        ArtifactInvalid => "artifact_invalid",
        OwnedCudaUnsupported => "owned_cuda_unsupported",
        EngineCrashed => "engine_crashed",
        ProbeRequired => "probe_required",
        MigrationRequired => "migration_required",
        ModuleRestarted => "module_restarted",
        InvalidRequest => "invalid_request",
        IdempotencyConflict => "idempotency_conflict",
        ModelNotInstalled => "model_not_installed",
        UnknownModel => "unknown_model",
        SelfCheckFailed => "self_check_failed",
        BackendUnavailable => "backend_unavailable",
        ModelInUse => "model_in_use",
        DownloadFailed => "download_failed",
    }
}

#[test]
fn stable_errors_are_exhaustively_documented_and_keep_wire_names() {
    for code in StableErrorCode::ALL {
        let name = stable_name(code);
        assert_eq!(serde_json::to_value(code).unwrap(), name);
        assert!(
            DOC.contains(&format!("`{name}`")),
            "undocumented stable error {name}"
        );
    }
}
