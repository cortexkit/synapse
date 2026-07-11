use serde::{Deserialize, Serialize};

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
    DeadlineExceeded,
    ModelLoading,
    NotCertified,
    SubstitutionRejected,
    ArtifactInvalid,
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
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StableError {
    pub code: StableErrorCode,
    pub class: ErrorClass,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_after_ms: Option<u64>,
    pub safe_to_retry_same_request: bool,
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
        }
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_error_contract_round_trips_through_json() {
        let errors = [
            StableError::queue_full(Some(25)),
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
        ];

        let json = serde_json::to_string(&errors).expect("serialize stable errors");
        let decoded: Vec<StableError> =
            serde_json::from_str(&json).expect("deserialize stable errors");
        assert_eq!(decoded, errors);
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
