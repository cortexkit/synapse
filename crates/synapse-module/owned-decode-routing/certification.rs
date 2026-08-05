//! Certification-row access and structural-band fork validation.
//!
//! Certification is machine-profile-local. Unconstrained rows key on
//! `(machine_profile_hash, decode_fingerprint)`; constrained rows key on
//! `(machine_profile_hash, decode_fingerprint, constraint_runtime_identity)`.
//! A request's per-request `constraint_fingerprint` is an exact substitution
//! check, NOT a certification key, so it is deliberately absent here.
//!
//! Structural-band rules come from `structural-band-v1`: the first f16
//! certification records a fork signature permitting at most two top-2 swaps,
//! Q8 permits zero, and recertification requires the stored signature exactly.
//! No cross-profile f16 equality is promised.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use synapse_core::Fingerprint;

use crate::owned_decode_contracts::StructuralBandManifest;
use crate::owned_decode_routing::error::OwnedDecodeError;

/// Unconstrained certification key.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnconstrainedCertKey {
    pub machine_profile_hash: String,
    pub decode_fingerprint: Fingerprint,
}

/// Constrained certification key. The third component is the constraint
/// runtime identity digest (shared across requests), not the per-request
/// constraint fingerprint.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConstrainedCertKey {
    pub machine_profile_hash: String,
    pub decode_fingerprint: Fingerprint,
    pub constraint_runtime_identity: String,
}

/// Read access to certification rows. Routing depends on this trait so the
/// authoritative store can live elsewhere while routing stays testable.
pub trait CertificationAccess {
    /// Whether an unconstrained row certifies this fingerprint on this profile.
    fn is_unconstrained_certified(&self, key: &UnconstrainedCertKey) -> bool;
    /// Whether a constrained row certifies this runtime identity on this profile.
    fn is_constrained_certified(&self, key: &ConstrainedCertKey) -> bool;
}

/// In-memory certification store. Production wiring backs this trait with the
/// authoritative persistent store; routing only requires read access plus the
/// ability to record rows during certification.
#[derive(Clone, Debug, Default)]
pub struct CertificationStore {
    unconstrained: BTreeSet<UnconstrainedCertKey>,
    constrained: BTreeSet<ConstrainedCertKey>,
    /// Stored fork signatures keyed by `(machine_profile_hash, decode_fingerprint)`.
    fork_signatures: BTreeMap<(String, String), String>,
}

impl CertificationStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an unconstrained certification row.
    pub fn certify_unconstrained(
        &mut self,
        machine_profile_hash: &str,
        decode_fingerprint: Fingerprint,
    ) {
        self.unconstrained.insert(UnconstrainedCertKey {
            machine_profile_hash: machine_profile_hash.to_string(),
            decode_fingerprint,
        });
    }

    /// Record a constrained certification row.
    pub fn certify_constrained(
        &mut self,
        machine_profile_hash: &str,
        decode_fingerprint: Fingerprint,
        constraint_runtime_identity: &str,
    ) {
        self.constrained.insert(ConstrainedCertKey {
            machine_profile_hash: machine_profile_hash.to_string(),
            decode_fingerprint,
            constraint_runtime_identity: constraint_runtime_identity.to_string(),
        });
    }

    /// Remove certification rows for a fingerprint (cutover invalidation).
    pub fn invalidate(&mut self, machine_profile_hash: &str, decode_fingerprint: &Fingerprint) {
        self.unconstrained.retain(|key| {
            !(key.machine_profile_hash == machine_profile_hash
                && key.decode_fingerprint == *decode_fingerprint)
        });
        self.constrained.retain(|key| {
            !(key.machine_profile_hash == machine_profile_hash
                && key.decode_fingerprint == *decode_fingerprint)
        });
        self.fork_signatures.remove(&(
            machine_profile_hash.to_string(),
            decode_fingerprint.0.clone(),
        ));
    }

    /// The stored fork signature for a profile/fingerprint, if one was recorded.
    pub fn stored_fork_signature(
        &self,
        machine_profile_hash: &str,
        decode_fingerprint: &Fingerprint,
    ) -> Option<&str> {
        self.fork_signatures
            .get(&(
                machine_profile_hash.to_string(),
                decode_fingerprint.0.clone(),
            ))
            .map(String::as_str)
    }

    /// Store a fork signature after a successful first certification.
    pub fn store_fork_signature(
        &mut self,
        machine_profile_hash: &str,
        decode_fingerprint: &Fingerprint,
        signature: &str,
    ) {
        self.fork_signatures.insert(
            (
                machine_profile_hash.to_string(),
                decode_fingerprint.0.clone(),
            ),
            signature.to_string(),
        );
    }
}

impl CertificationAccess for CertificationStore {
    fn is_unconstrained_certified(&self, key: &UnconstrainedCertKey) -> bool {
        self.unconstrained.contains(key)
    }

    fn is_constrained_certified(&self, key: &ConstrainedCertKey) -> bool {
        self.constrained.contains(key)
    }
}

/// Structural-band fork-signature checker built from `structural-band-v1`.
#[derive(Clone, Debug, Default)]
pub struct StructuralBandChecker {
    /// `(family, weight_quant)` -> maximum permitted top-2 swaps.
    max_swaps: BTreeMap<(String, String), u32>,
}

/// Result of a successful structural-band check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ForkCheckOutcome {
    /// First certification: the observed signature was within band and should be
    /// stored.
    Recorded(String),
    /// Recertification: the observed signature matched the stored one exactly.
    Matched,
}

impl StructuralBandChecker {
    /// Build a checker from the structural-band manifest rules.
    pub fn from_manifest(manifest: &StructuralBandManifest) -> Self {
        let mut max_swaps = BTreeMap::new();
        for rule in &manifest.rules {
            max_swaps.insert(
                (rule.family.clone(), rule.weight_quant.clone()),
                rule.max_top2_swaps,
            );
        }
        Self { max_swaps }
    }

    /// The maximum permitted top-2 swaps for a family/format, if the manifest
    /// has a rule for it.
    pub fn max_top2_swaps(&self, family: &str, weight_quant: &str) -> Option<u32> {
        self.max_swaps
            .get(&(family.to_string(), weight_quant.to_string()))
            .copied()
    }

    /// Validate an observed fork signature against the band rules.
    ///
    /// `stored` is the previously recorded signature for this profile/fingerprint.
    /// On first certification (`stored` is `None`) the observed swap count must
    /// be within the band and the observed signature is recorded. On
    /// recertification the observed signature must match the stored one exactly.
    pub fn check(
        &self,
        family: &str,
        weight_quant: &str,
        observed_top2_swaps: u32,
        observed_signature: &str,
        stored: Option<&str>,
    ) -> Result<ForkCheckOutcome, OwnedDecodeError> {
        let max = self
            .max_top2_swaps(family, weight_quant)
            .ok_or(OwnedDecodeError::Unsupported)?;

        match stored {
            Some(stored) => {
                if stored == observed_signature {
                    Ok(ForkCheckOutcome::Matched)
                } else {
                    Err(OwnedDecodeError::CertificationFailed)
                }
            }
            None => {
                if observed_top2_swaps <= max {
                    Ok(ForkCheckOutcome::Recorded(observed_signature.to_string()))
                } else {
                    Err(OwnedDecodeError::CertificationFailed)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::owned_decode_contracts::StructuralBandRule;

    fn fp(s: &str) -> Fingerprint {
        Fingerprint(s.to_string())
    }

    fn manifest() -> StructuralBandManifest {
        StructuralBandManifest {
            manifest_revision: "structural-band-v1".to_string(),
            schema_revision: "owned-decode-contracts-v1".to_string(),
            rules: vec![
                StructuralBandRule {
                    family: "qwen3-0.6b".to_string(),
                    weight_quant: "f16".to_string(),
                    max_top2_swaps: 2,
                    fork_signature_recertification: "exact".to_string(),
                },
                StructuralBandRule {
                    family: "qwen3-0.6b".to_string(),
                    weight_quant: "q8_0".to_string(),
                    max_top2_swaps: 0,
                    fork_signature_recertification: "exact".to_string(),
                },
            ],
        }
    }

    #[test]
    fn unconstrained_and_constrained_keys_are_distinct() {
        let mut store = CertificationStore::new();
        store.certify_unconstrained("profile-a", fp("decode-1"));
        store.certify_constrained("profile-a", fp("decode-1"), "cri-1");

        assert!(store.is_unconstrained_certified(&UnconstrainedCertKey {
            machine_profile_hash: "profile-a".to_string(),
            decode_fingerprint: fp("decode-1"),
        }));
        // A different profile is not certified (machine-profile-local).
        assert!(!store.is_unconstrained_certified(&UnconstrainedCertKey {
            machine_profile_hash: "profile-b".to_string(),
            decode_fingerprint: fp("decode-1"),
        }));
        // Constrained lookup requires the exact runtime identity.
        assert!(store.is_constrained_certified(&ConstrainedCertKey {
            machine_profile_hash: "profile-a".to_string(),
            decode_fingerprint: fp("decode-1"),
            constraint_runtime_identity: "cri-1".to_string(),
        }));
        assert!(!store.is_constrained_certified(&ConstrainedCertKey {
            machine_profile_hash: "profile-a".to_string(),
            decode_fingerprint: fp("decode-1"),
            constraint_runtime_identity: "cri-other".to_string(),
        }));
    }

    #[test]
    fn invalidation_removes_rows_and_fork_signature() {
        let mut store = CertificationStore::new();
        store.certify_unconstrained("profile-a", fp("decode-1"));
        store.certify_constrained("profile-a", fp("decode-1"), "cri-1");
        store.store_fork_signature("profile-a", &fp("decode-1"), "sig-1");

        store.invalidate("profile-a", &fp("decode-1"));
        assert!(!store.is_unconstrained_certified(&UnconstrainedCertKey {
            machine_profile_hash: "profile-a".to_string(),
            decode_fingerprint: fp("decode-1"),
        }));
        assert!(store
            .stored_fork_signature("profile-a", &fp("decode-1"))
            .is_none());
    }

    #[test]
    fn first_f16_certification_within_band_records_signature() {
        let checker = StructuralBandChecker::from_manifest(&manifest());
        let outcome = checker
            .check("qwen3-0.6b", "f16", 2, "sig-abc", None)
            .expect("within band");
        assert_eq!(outcome, ForkCheckOutcome::Recorded("sig-abc".to_string()));
    }

    #[test]
    fn first_f16_certification_over_band_fails() {
        let checker = StructuralBandChecker::from_manifest(&manifest());
        assert_eq!(
            checker.check("qwen3-0.6b", "f16", 3, "sig-abc", None),
            Err(OwnedDecodeError::CertificationFailed)
        );
    }

    #[test]
    fn q8_requires_zero_forks() {
        let checker = StructuralBandChecker::from_manifest(&manifest());
        assert_eq!(
            checker.check("qwen3-0.6b", "q8_0", 0, "sig-q8", None),
            Ok(ForkCheckOutcome::Recorded("sig-q8".to_string()))
        );
        assert_eq!(
            checker.check("qwen3-0.6b", "q8_0", 1, "sig-q8", None),
            Err(OwnedDecodeError::CertificationFailed)
        );
    }

    #[test]
    fn recertification_requires_stored_signature_exactly() {
        let checker = StructuralBandChecker::from_manifest(&manifest());
        assert_eq!(
            checker.check("qwen3-0.6b", "f16", 0, "sig-abc", Some("sig-abc")),
            Ok(ForkCheckOutcome::Matched)
        );
        assert_eq!(
            checker.check("qwen3-0.6b", "f16", 0, "sig-different", Some("sig-abc")),
            Err(OwnedDecodeError::CertificationFailed)
        );
    }

    #[test]
    fn unknown_family_format_is_unsupported() {
        let checker = StructuralBandChecker::from_manifest(&manifest());
        assert_eq!(
            checker.check("llama-3", "f16", 0, "sig", None),
            Err(OwnedDecodeError::Unsupported)
        );
    }

    #[test]
    fn certification_keys_reject_unknown_fields() {
        // fail-closed posture: an unknown field in a certification key is
        // rejected at parse time rather than silently dropped.
        let bad_unconstrained = serde_json::json!({
            "machine_profile_hash": "profile-a",
            "decode_fingerprint": "fp",
            "unknown": "x",
        });
        assert!(serde_json::from_value::<UnconstrainedCertKey>(bad_unconstrained).is_err());

        let bad_constrained = serde_json::json!({
            "machine_profile_hash": "profile-a",
            "decode_fingerprint": "fp",
            "constraint_runtime_identity": "cri",
            "unknown": "x",
        });
        assert!(serde_json::from_value::<ConstrainedCertKey>(bad_constrained).is_err());
    }
}
