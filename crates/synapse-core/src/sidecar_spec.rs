//! Shared contracts for request-scoped semantic sidecars.
//!
//! These types cross the module/worker seam without granting a sidecar commit
//! authority. The bank's view payload contains target-vocabulary token IDs;
//! schema/layout identities and build metadata accompany those views. The target
//! decode loop remains responsible for deciding whether any continuation is legal
//! and can be committed.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Configuration for the dormant semantic-sidecar feature.
///
/// The feature is disabled by default. Enabling it is intentionally separate from
/// routing: callers must still establish request eligibility before launching a
/// sidecar job.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SidecarSpec {
    #[serde(default)]
    pub enabled: bool,
    pub model_id: String,
    #[serde(default)]
    pub strategy: SidecarStrategy,
    #[serde(default = "default_max_new_tokens")]
    pub max_new_tokens: u32,
    #[serde(default)]
    pub placement: SidecarPlacement,
}

impl Default for SidecarSpec {
    fn default() -> Self {
        Self {
            enabled: false,
            model_id: "qwen3-0.6b-sidecar".to_string(),
            strategy: SidecarStrategy::WholeObject,
            max_new_tokens: default_max_new_tokens(),
            placement: SidecarPlacement::Metal,
        }
    }
}

impl SidecarSpec {
    /// Reject configurations that could otherwise start an unbounded or unnamed
    /// sidecar job when the dormant feature is explicitly enabled.
    pub fn validate(&self) -> Result<(), SidecarSpecError> {
        if self.enabled && self.model_id.trim().is_empty() {
            return Err(SidecarSpecError::MissingModelId);
        }
        if self.enabled && self.max_new_tokens == 0 {
            return Err(SidecarSpecError::ZeroOutputLimit);
        }
        Ok(())
    }
}

const fn default_max_new_tokens() -> u32 {
    256
}

/// The semantic prediction layout used by a sidecar job.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SidecarStrategy {
    #[default]
    WholeObject,
    PerField,
}

/// The placement selected for a sidecar job.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SidecarPlacement {
    #[default]
    Metal,
    Ane,
}

/// A configuration error caught before any job can be launched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SidecarSpecError {
    MissingModelId,
    ZeroOutputLimit,
}

impl std::fmt::Display for SidecarSpecError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingModelId => {
                formatter.write_str("enabled sidecar config requires a model_id")
            }
            Self::ZeroOutputLimit => formatter
                .write_str("enabled sidecar config requires max_new_tokens greater than zero"),
        }
    }
}

impl std::error::Error for SidecarSpecError {}

/// A request-scoped bank of target-tokenizer views.
///
/// Each inner vector is one complete rendered view. View boundaries are explicit,
/// so a worker never needs to treat an EOS separator as a speculative token. The
/// `built_at` value is observational metadata and is deliberately excluded from
/// [`Self::content_digest`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SidecarHintBank {
    pub views: Vec<Vec<u32>>,
    pub schema_identity: String,
    pub render_policy_digest: String,
    pub built_at: u64,
}

impl SidecarHintBank {
    /// Return a stable identity for bank content.
    ///
    /// Completion timing does not enter this digest: identical frozen views must
    /// retain the same identity regardless of when a sidecar finished.
    #[must_use]
    pub fn content_digest(&self) -> String {
        let mut hasher = Sha256::new();
        hash_bytes(&mut hasher, self.schema_identity.as_bytes());
        hash_bytes(&mut hasher, self.render_policy_digest.as_bytes());
        hash_u64(&mut hasher, self.views.len() as u64);
        for view in &self.views {
            hash_u64(&mut hasher, view.len() as u64);
            for token in view {
                hasher.update(token.to_le_bytes());
            }
        }
        hex::encode(hasher.finalize())
    }
}

fn hash_bytes(hasher: &mut Sha256, bytes: &[u8]) {
    hash_u64(hasher, bytes.len() as u64);
    hasher.update(bytes);
}

fn hash_u64(hasher: &mut Sha256, value: u64) {
    hasher.update(value.to_le_bytes());
}

/// Whether a usable bank ever supplied a verification-accepted continuation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SidecarBankEffect {
    Used,
    Unused,
}

/// Terminal outcome of one launched sidecar job.
///
/// Unsupported placement is rejected before launch, produces no `SidecarOutcome`,
/// and must be excluded from launched-job outcome metrics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SidecarOutcome {
    Cancelled,
    Failed,
    CompletedLate,
    CompletedInvalid,
    CompletedUsable { bank_effect: SidecarBankEffect },
}

impl SidecarOutcome {
    /// Apply the terminal-outcome precedence for one launched job.
    #[must_use]
    pub const fn classify(events: SidecarOutcomeEvents) -> Self {
        if events.cancelled {
            Self::Cancelled
        } else if events.failed {
            Self::Failed
        } else if events.completed_after_target {
            Self::CompletedLate
        } else if !events.completed_valid {
            Self::CompletedInvalid
        } else {
            Self::CompletedUsable {
                bank_effect: if events.bank_used {
                    SidecarBankEffect::Used
                } else {
                    SidecarBankEffect::Unused
                },
            }
        }
    }

    /// A usable bank is a successful sidecar completion whether or not it won a
    /// suffix match during target decoding.
    #[must_use]
    pub const fn is_usable(self) -> bool {
        matches!(self, Self::CompletedUsable { .. })
    }
}

/// Monotonic terminal facts used to assign an outcome.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SidecarOutcomeEvents {
    pub cancelled: bool,
    pub failed: bool,
    pub completed_after_target: bool,
    pub completed_valid: bool,
    pub bank_used: bool,
}

/// Attribution class for a token accepted through sidecar verification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpanClass {
    Structural,
    Value,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_dormant_and_denies_unknown_fields() {
        let config = SidecarSpec::default();
        assert!(!config.enabled);
        assert!(config.validate().is_ok());
        assert!(serde_json::from_str::<SidecarSpec>(
            r#"{"enabled":false,"model_id":"m","strategy":"whole_object","max_new_tokens":1,"placement":"metal","typo":true}"#
        )
        .is_err());
    }

    #[test]
    fn bank_digest_ignores_observational_completion_time() {
        let bank = SidecarHintBank {
            views: vec![vec![1, 2], vec![3]],
            schema_identity: "schema-v1".to_string(),
            render_policy_digest: "layout-v1".to_string(),
            built_at: 1,
        };
        let later = SidecarHintBank {
            built_at: 9_999,
            ..bank.clone()
        };

        assert_eq!(bank.content_digest(), later.content_digest());
    }

    #[test]
    fn outcome_precedence_is_exhaustive_for_launched_jobs() {
        assert_eq!(
            SidecarOutcome::classify(SidecarOutcomeEvents {
                cancelled: true,
                failed: true,
                completed_after_target: true,
                completed_valid: true,
                bank_used: true,
            }),
            SidecarOutcome::Cancelled
        );
        assert_eq!(
            SidecarOutcome::classify(SidecarOutcomeEvents {
                failed: true,
                ..SidecarOutcomeEvents::default()
            }),
            SidecarOutcome::Failed
        );
        assert_eq!(
            SidecarOutcome::classify(SidecarOutcomeEvents {
                completed_after_target: true,
                ..SidecarOutcomeEvents::default()
            }),
            SidecarOutcome::CompletedLate
        );
        assert_eq!(
            SidecarOutcome::classify(SidecarOutcomeEvents::default()),
            SidecarOutcome::CompletedInvalid
        );
        assert_eq!(
            SidecarOutcome::classify(SidecarOutcomeEvents {
                completed_valid: true,
                bank_used: true,
                ..SidecarOutcomeEvents::default()
            }),
            SidecarOutcome::CompletedUsable {
                bank_effect: SidecarBankEffect::Used
            }
        );
    }
}
