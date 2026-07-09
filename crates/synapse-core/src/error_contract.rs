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
        ];

        let json = serde_json::to_string(&errors).expect("serialize stable errors");
        let decoded: Vec<StableError> =
            serde_json::from_str(&json).expect("deserialize stable errors");
        assert_eq!(decoded, errors);
    }
}
