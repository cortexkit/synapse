#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use synapse_core::error_contract::StableErrorCode;

const DOC: &str = include_str!("../../../docs/wire-contract-v1.md");
const SOURCE: &str = include_str!("../src/lib.rs");

fn documented_ops(doc: &str) -> BTreeSet<&str> {
    let ops = doc
        .split("The management registry in this snapshot is ")
        .nth(1)
        .expect("registry inventory")
        .split("\n\n")
        .next()
        .unwrap();
    ops.split('`')
        .enumerate()
        .filter_map(|(i, s)| (i % 2 == 1).then_some(s))
        .collect()
}

fn registered_ops(source: &str) -> BTreeSet<&str> {
    source
        .split("fn management_operations()")
        .nth(1)
        .expect("operation registry")
        .split("fn manifest(")
        .next()
        .unwrap()
        .lines()
        .filter_map(|line| {
            line.trim()
                .strip_prefix("op(\"")
                .and_then(|s| s.split('"').next())
        })
        .collect()
}

#[test]
fn management_registry_matches_wire_contract() {
    assert_eq!(registered_ops(SOURCE), documented_ops(DOC));
}

// An exhaustive match deliberately makes an added enum variant require documentation review,
// even if the variant was accidentally omitted from StableErrorCode::ALL.
fn stable_name(code: StableErrorCode) -> &'static str {
    use StableErrorCode::*;
    match code {
        QueueFull => "queue_full",
        DeadlineExceeded => "deadline_exceeded",
        ModelLoading => "model_loading",
        NotCertified => "not_certified",
        SubstitutionRejected => "substitution_rejected",
        ArtifactInvalid => "artifact_invalid",
        OwnedCudaUnsupported => "owned_cuda_unsupported",
        EngineCrashed => "engine_crashed",
        ProbeRequired => "probe_required",
        MigrationRequired => "migration_required",
        ModuleRestarted => "module_restarted",
        InvalidRequest => "invalid_request",
        DeclaredIdentityNotAccepted => "declared_identity_not_accepted",
        RemoteIdentityDrift => "remote_identity_drift",
        ProviderUnavailable => "provider_unavailable",
        ProviderProtocolViolation => "provider_protocol_violation",
        IdempotencyConflict => "idempotency_conflict",
        NeedsReauth => "needs_reauth",
        NeedsReauthExpired => "needs_reauth_expired",
        RemoteDeploymentChanged => "remote_deployment_changed",
        CredentialConfigInvalid => "credential_config_invalid",
        OpNotSupportedForRemote => "op_not_supported_for_remote",
        SentinelCalibrationRefused => "sentinel_calibration_refused",
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

#[test]
fn drift_scanner_detects_added_ops_and_missing_documentation() {
    let source = SOURCE.replacen(
        "op(\"models.catalog\", Query)",
        "op(\"models.catalog\", Query),\n        op(\"models.undocumented\", Query)",
        1,
    );
    assert!(registered_ops(&source).contains("models.catalog"));
    assert!(registered_ops(&source).contains("models.undocumented"));
    assert_ne!(registered_ops(&source), documented_ops(DOC));
    let doc = DOC.replace("`models.download.cancel`", "`models.cancel.retired`");
    assert_ne!(registered_ops(SOURCE), documented_ops(&doc));
}
