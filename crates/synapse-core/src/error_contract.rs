use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorClass {
    Transient,
    Permanent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StableErrorCode {
    QueueFull,
    SequenceTooLong,
    DeadlineExceeded,
    ModelLoading,
    NotCertified,
    SubstitutionRejected,
    ArtifactInvalid,
    OwnedCudaUnsupported,
    EngineCrashed,
    ProbeRequired,
    MigrationRequired,
    ModuleRestarted,
    InvalidRequest,
    DeclaredIdentityNotAccepted,
    RemoteIdentityDrift,
    ProviderUnavailable,
    ProviderProtocolViolation,
    IdempotencyConflict,
    NeedsReauth,
    NeedsReauthExpired,
    RemoteDeploymentChanged,
    CredentialConfigInvalid,
    OpNotSupportedForRemote,
    SentinelCalibrationRefused,
    ModelNotInstalled,
    UnknownModel,
    SelfCheckFailed,
    BackendUnavailable,
    ModelInUse,
    DownloadFailed,
}

impl StableErrorCode {
    /// Every stable error code in declaration order.
    pub const ALL: [Self; 30] = [
        Self::QueueFull,
        Self::SequenceTooLong,
        Self::DeadlineExceeded,
        Self::ModelLoading,
        Self::NotCertified,
        Self::SubstitutionRejected,
        Self::ArtifactInvalid,
        Self::OwnedCudaUnsupported,
        Self::EngineCrashed,
        Self::ProbeRequired,
        Self::MigrationRequired,
        Self::ModuleRestarted,
        Self::InvalidRequest,
        Self::DeclaredIdentityNotAccepted,
        Self::RemoteIdentityDrift,
        Self::ProviderUnavailable,
        Self::ProviderProtocolViolation,
        Self::IdempotencyConflict,
        Self::NeedsReauth,
        Self::NeedsReauthExpired,
        Self::RemoteDeploymentChanged,
        Self::CredentialConfigInvalid,
        Self::OpNotSupportedForRemote,
        Self::SentinelCalibrationRefused,
        Self::ModelNotInstalled,
        Self::UnknownModel,
        Self::SelfCheckFailed,
        Self::BackendUnavailable,
        Self::ModelInUse,
        Self::DownloadFailed,
    ];
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StableError {
    pub code: StableErrorCode,
    pub class: ErrorClass,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_after_ms: Option<u64>,
    pub safe_to_retry_same_request: bool,
    /// Structured, code-specific context for the refusal (for example the
    /// catalog id a `model_not_installed` error refers to). It is always a JSON
    /// object when present and is omitted from the wire when absent, so errors
    /// without details serialize exactly as they did before the field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Map<String, Value>>,
}

impl StableError {
    pub const fn new(
        code: StableErrorCode,
        class: ErrorClass,
        retry_after_ms: Option<u64>,
        safe_to_retry_same_request: bool,
    ) -> Self {
        Self {
            code,
            class,
            retry_after_ms,
            safe_to_retry_same_request,
            details: None,
        }
    }

    /// Returns this error with `details` replaced by the given object.
    pub fn with_details(mut self, details: Map<String, Value>) -> Self {
        self.details = Some(details);
        self
    }

    pub fn sequence_too_long(tokens: usize, max_tokens: usize, item_id: Option<&str>) -> Self {
        let mut details = Map::new();
        details.insert("tokens".into(), Value::from(tokens));
        details.insert("max_tokens".into(), Value::from(max_tokens));
        if let Some(id) = item_id {
            details.insert("item_id".into(), Value::from(id));
        }
        Self::new(
            StableErrorCode::SequenceTooLong,
            ErrorClass::Permanent,
            None,
            false,
        )
        .with_details(details)
    }

    pub const fn queue_full(retry_after_ms: Option<u64>) -> Self {
        Self::new(
            StableErrorCode::QueueFull,
            ErrorClass::Transient,
            retry_after_ms,
            true,
        )
    }

    pub const fn deadline_exceeded() -> Self {
        Self::new(
            StableErrorCode::DeadlineExceeded,
            ErrorClass::Transient,
            None,
            false,
        )
    }

    pub const fn model_loading(retry_after_ms: Option<u64>) -> Self {
        Self::new(
            StableErrorCode::ModelLoading,
            ErrorClass::Transient,
            retry_after_ms,
            true,
        )
    }

    pub const fn not_certified() -> Self {
        Self::new(
            StableErrorCode::NotCertified,
            ErrorClass::Permanent,
            None,
            false,
        )
    }

    pub const fn substitution_rejected() -> Self {
        Self::new(
            StableErrorCode::SubstitutionRejected,
            ErrorClass::Permanent,
            None,
            false,
        )
    }

    pub const fn artifact_invalid() -> Self {
        Self::new(
            StableErrorCode::ArtifactInvalid,
            ErrorClass::Permanent,
            None,
            false,
        )
    }

    pub const fn owned_cuda_unsupported() -> Self {
        Self::new(
            StableErrorCode::OwnedCudaUnsupported,
            ErrorClass::Permanent,
            None,
            false,
        )
    }

    pub const fn engine_crashed(retry_after_ms: Option<u64>) -> Self {
        Self::new(
            StableErrorCode::EngineCrashed,
            ErrorClass::Transient,
            retry_after_ms,
            true,
        )
    }

    pub const fn probe_required() -> Self {
        Self::new(
            StableErrorCode::ProbeRequired,
            ErrorClass::Permanent,
            None,
            false,
        )
    }

    pub const fn migration_required() -> Self {
        Self::new(
            StableErrorCode::MigrationRequired,
            ErrorClass::Permanent,
            None,
            false,
        )
    }

    pub const fn module_restarted() -> Self {
        Self::new(
            StableErrorCode::ModuleRestarted,
            ErrorClass::Transient,
            None,
            true,
        )
    }

    pub const fn invalid_request() -> Self {
        Self::new(
            StableErrorCode::InvalidRequest,
            ErrorClass::Permanent,
            None,
            false,
        )
    }

    pub const fn declared_identity_not_accepted() -> Self {
        Self::new(
            StableErrorCode::DeclaredIdentityNotAccepted,
            ErrorClass::Permanent,
            None,
            false,
        )
    }

    pub const fn remote_identity_drift() -> Self {
        Self::new(
            StableErrorCode::RemoteIdentityDrift,
            ErrorClass::Permanent,
            None,
            false,
        )
    }

    pub const fn provider_unavailable(retry_after_ms: Option<u64>) -> Self {
        Self::new(
            StableErrorCode::ProviderUnavailable,
            ErrorClass::Transient,
            retry_after_ms,
            true,
        )
    }

    pub const fn provider_protocol_violation() -> Self {
        Self::new(
            StableErrorCode::ProviderProtocolViolation,
            ErrorClass::Permanent,
            None,
            false,
        )
    }

    pub const fn idempotency_conflict() -> Self {
        Self::new(
            StableErrorCode::IdempotencyConflict,
            ErrorClass::Permanent,
            None,
            false,
        )
    }

    pub const fn needs_reauth() -> Self {
        Self::new(
            StableErrorCode::NeedsReauth,
            ErrorClass::Permanent,
            None,
            false,
        )
    }

    pub const fn needs_reauth_expired() -> Self {
        Self::new(
            StableErrorCode::NeedsReauthExpired,
            ErrorClass::Permanent,
            None,
            false,
        )
    }

    pub const fn remote_deployment_changed() -> Self {
        Self::new(
            StableErrorCode::RemoteDeploymentChanged,
            ErrorClass::Permanent,
            None,
            false,
        )
    }

    pub const fn credential_config_invalid() -> Self {
        Self::new(
            StableErrorCode::CredentialConfigInvalid,
            ErrorClass::Permanent,
            None,
            false,
        )
    }

    pub const fn op_not_supported_for_remote() -> Self {
        Self::new(
            StableErrorCode::OpNotSupportedForRemote,
            ErrorClass::Permanent,
            None,
            false,
        )
    }

    pub const fn sentinel_calibration_refused() -> Self {
        Self::new(
            StableErrorCode::SentinelCalibrationRefused,
            ErrorClass::Permanent,
            None,
            false,
        )
    }

    pub const fn model_not_installed() -> Self {
        Self::new(
            StableErrorCode::ModelNotInstalled,
            ErrorClass::Permanent,
            None,
            false,
        )
    }

    pub const fn unknown_model() -> Self {
        Self::new(
            StableErrorCode::UnknownModel,
            ErrorClass::Permanent,
            None,
            false,
        )
    }

    pub const fn self_check_failed() -> Self {
        Self::new(
            StableErrorCode::SelfCheckFailed,
            ErrorClass::Permanent,
            None,
            false,
        )
    }

    pub const fn backend_unavailable() -> Self {
        Self::new(
            StableErrorCode::BackendUnavailable,
            ErrorClass::Permanent,
            None,
            false,
        )
    }

    pub const fn model_in_use() -> Self {
        Self::new(
            StableErrorCode::ModelInUse,
            ErrorClass::Permanent,
            None,
            false,
        )
    }

    /// Delay a client should wait before retrying a failed model download.
    pub const DOWNLOAD_FAILED_RETRY_AFTER_MS: u64 = 1_000;

    /// A model download failed for a reason that may clear on its own (network
    /// drop, HTTP error status, full disk), so retrying the same request is safe.
    pub const fn download_failed() -> Self {
        Self::new(
            StableErrorCode::DownloadFailed,
            ErrorClass::Transient,
            Some(Self::DOWNLOAD_FAILED_RETRY_AFTER_MS),
            true,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn stable_error_code_all_covers_every_variant() {
        const VARIANT_COUNT: usize = 30;

        // This match is intentionally exhaustive so enum additions update the
        // enumeration and its expected cardinality together.
        fn assert_exhaustive(error_code: StableErrorCode) {
            match error_code {
                StableErrorCode::QueueFull
                | StableErrorCode::SequenceTooLong
                | StableErrorCode::DeadlineExceeded
                | StableErrorCode::ModelLoading
                | StableErrorCode::NotCertified
                | StableErrorCode::SubstitutionRejected
                | StableErrorCode::ArtifactInvalid
                | StableErrorCode::OwnedCudaUnsupported
                | StableErrorCode::EngineCrashed
                | StableErrorCode::ProbeRequired
                | StableErrorCode::MigrationRequired
                | StableErrorCode::ModuleRestarted
                | StableErrorCode::InvalidRequest
                | StableErrorCode::DeclaredIdentityNotAccepted
                | StableErrorCode::RemoteIdentityDrift
                | StableErrorCode::ProviderUnavailable
                | StableErrorCode::ProviderProtocolViolation
                | StableErrorCode::IdempotencyConflict
                | StableErrorCode::NeedsReauth
                | StableErrorCode::NeedsReauthExpired
                | StableErrorCode::RemoteDeploymentChanged
                | StableErrorCode::CredentialConfigInvalid
                | StableErrorCode::OpNotSupportedForRemote
                | StableErrorCode::SentinelCalibrationRefused
                | StableErrorCode::ModelNotInstalled
                | StableErrorCode::UnknownModel
                | StableErrorCode::SelfCheckFailed
                | StableErrorCode::BackendUnavailable
                | StableErrorCode::ModelInUse
                | StableErrorCode::DownloadFailed => {}
            }
        }

        assert_exhaustive(StableErrorCode::QueueFull);
        assert_eq!(StableErrorCode::ALL.len(), VARIANT_COUNT);
    }

    #[test]
    fn stable_error_contract_round_trips_through_json() {
        let errors = [
            StableError::queue_full(Some(25)),
            StableError::sequence_too_long(8193, 8192, Some("row")),
            StableError::deadline_exceeded(),
            StableError::model_loading(Some(150)),
            StableError::not_certified(),
            StableError::substitution_rejected(),
            StableError::artifact_invalid(),
            StableError::engine_crashed(Some(500)),
            StableError::probe_required(),
            StableError::migration_required(),
            StableError::module_restarted(),
            StableError::invalid_request(),
            StableError::declared_identity_not_accepted(),
            StableError::remote_identity_drift(),
            StableError::provider_unavailable(Some(1_000)),
            StableError::provider_protocol_violation(),
            StableError::idempotency_conflict(),
            StableError::needs_reauth(),
            StableError::needs_reauth_expired(),
            StableError::remote_deployment_changed(),
            StableError::credential_config_invalid(),
            StableError::op_not_supported_for_remote(),
            StableError::sentinel_calibration_refused(),
            StableError::model_not_installed(),
            StableError::unknown_model(),
            StableError::self_check_failed(),
            StableError::backend_unavailable(),
            StableError::model_in_use(),
            StableError::download_failed(),
            StableError::unknown_model()
                .with_details(details(json!({"model_id": "no-such-model"}))),
            StableError::download_failed().with_details(details(json!({
                "file": "model.safetensors",
                "reason": "http_status",
                "http_status": 500,
            }))),
        ];

        let json = serde_json::to_string(&errors).expect("serialize stable errors");
        let decoded: Vec<StableError> =
            serde_json::from_str(&json).expect("deserialize stable errors");
        assert_eq!(decoded, errors);
    }

    #[test]
    fn new_stable_error_codes_round_trip_with_their_frozen_wire_shape() {
        // (constructor output, wire code, class, retry_after_ms, safe to retry)
        let cases = [
            (
                StableError::model_not_installed(),
                "model_not_installed",
                "permanent",
                None,
                false,
            ),
            (
                StableError::unknown_model(),
                "unknown_model",
                "permanent",
                None,
                false,
            ),
            (
                StableError::self_check_failed(),
                "self_check_failed",
                "permanent",
                None,
                false,
            ),
            (
                StableError::backend_unavailable(),
                "backend_unavailable",
                "permanent",
                None,
                false,
            ),
            (
                StableError::model_in_use(),
                "model_in_use",
                "permanent",
                None,
                false,
            ),
            (
                StableError::download_failed(),
                "download_failed",
                "transient",
                Some(1_000),
                true,
            ),
        ];

        for (error, code, class, retry_after_ms, safe) in cases {
            let mut expected = json!({
                "code": code,
                "class": class,
                "safe_to_retry_same_request": safe,
            });
            if let Some(retry_after_ms) = retry_after_ms {
                expected["retry_after_ms"] = json!(retry_after_ms);
            }
            let wire = serde_json::to_value(&error).expect("serialize stable error");
            assert_eq!(wire, expected, "wire shape for {code}");
            let decoded: StableError =
                serde_json::from_value(expected).expect("deserialize stable error");
            assert_eq!(decoded, error, "round trip for {code}");
        }
    }

    #[test]
    fn new_stable_error_code_classes_are_frozen() {
        for error in [
            StableError::model_not_installed(),
            StableError::unknown_model(),
            StableError::self_check_failed(),
            StableError::backend_unavailable(),
            StableError::model_in_use(),
        ] {
            assert_eq!(error.class, ErrorClass::Permanent, "{:?}", error.code);
            assert_eq!(error.retry_after_ms, None, "{:?}", error.code);
            assert!(!error.safe_to_retry_same_request, "{:?}", error.code);
            assert_eq!(error.details, None, "{:?}", error.code);
        }

        let download = StableError::download_failed();
        assert_eq!(download.code, StableErrorCode::DownloadFailed);
        assert_eq!(download.class, ErrorClass::Transient);
        assert_eq!(download.retry_after_ms, Some(1_000));
        assert!(download.safe_to_retry_same_request);
    }

    #[test]
    fn details_are_serialized_only_when_present() {
        let without = serde_json::to_value(StableError::unknown_model()).expect("serialize");
        assert!(
            without.get("details").is_none(),
            "absent details must be omitted from the wire: {without}"
        );

        let holders = json!({
            "catalog_id": "gte-modernbert-base",
            "holders": [{"kind": "job", "job_id": "job-1"}],
        });
        let with = serde_json::to_value(
            StableError::model_in_use().with_details(details(holders.clone())),
        )
        .expect("serialize");
        assert_eq!(with["details"], holders);

        // Errors written before the field existed carry no `details` key and
        // must still decode.
        let legacy: StableError = serde_json::from_value(json!({
            "code": "queue_full",
            "class": "transient",
            "retry_after_ms": 25,
            "safe_to_retry_same_request": true,
        }))
        .expect("deserialize legacy error");
        assert_eq!(legacy, StableError::queue_full(Some(25)));

        // `details` is an object on the wire; any other JSON type is refused.
        let not_object = serde_json::from_value::<StableError>(json!({
            "code": "unknown_model",
            "class": "permanent",
            "safe_to_retry_same_request": false,
            "details": "no-such-model",
        }));
        assert!(not_object.is_err());
    }

    fn details(value: Value) -> Map<String, Value> {
        match value {
            Value::Object(map) => map,
            other => panic!("details must be a JSON object, got {other}"),
        }
    }

    #[test]
    fn remote_error_classes_are_frozen() {
        for error in [
            StableError::invalid_request(),
            StableError::declared_identity_not_accepted(),
            StableError::remote_identity_drift(),
            StableError::provider_protocol_violation(),
            StableError::idempotency_conflict(),
            StableError::needs_reauth(),
            StableError::needs_reauth_expired(),
            StableError::remote_deployment_changed(),
            StableError::credential_config_invalid(),
            StableError::op_not_supported_for_remote(),
            StableError::sentinel_calibration_refused(),
        ] {
            assert_eq!(error.class, ErrorClass::Permanent);
            assert!(!error.safe_to_retry_same_request);
        }
        let unavailable = StableError::provider_unavailable(Some(250));
        assert_eq!(unavailable.class, ErrorClass::Transient);
        assert!(unavailable.safe_to_retry_same_request);
        assert_eq!(unavailable.retry_after_ms, Some(250));
    }
}
