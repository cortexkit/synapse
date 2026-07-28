//! Certification probes: run the parity fixture battery through a decode seam,
//! compare against the independent oracle, apply the structural-band fork
//! rules, and record certification rows.
//!
//! Certification is machine-profile-local and fails closed. Unconstrained rows
//! key on `(machine_profile_hash, decode_fingerprint)`; constrained rows key on
//! `(machine_profile_hash, decode_fingerprint, constraint_runtime_identity)`.
//! The first f16 certification records a structural-band-compliant fork
//! signature (at most two top-2 swaps); Q8 requires zero forks; recertification
//! requires the stored signature exactly.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use synapse_core::Fingerprint;

use crate::owned_decode_certification::fixtures::{
    battery_digest, OracleStore, ParityFixture, PARITY_GROUP,
};
use crate::owned_decode_routing::certification::{CertificationStore, StructuralBandChecker};
use crate::owned_decode_routing::error::OwnedDecodeError;

/// The decode-execution seam a probe generates through. Production wiring
/// backs this with the supervised worker lane; hardware-independent tests back
/// it with a deterministic double; the mandatory `macos-metal` lane backs it
/// with the real Metal step engines.
pub trait DecodeProbe {
    /// Generate the greedy-top-1 token stream for one fixture prompt.
    fn generate(&mut self, fixture: &ParityFixture, prompt_index: u32) -> Vec<u32>;
}

/// A test double that reproduces the registered oracle bytes exactly. Stands
/// in for a fully parity-verified worker in hardware-independent tests; the
/// oracle is the authority, never the other way around.
pub struct OracleReproducingProbe<'a> {
    oracle: &'a OracleStore,
}

impl<'a> OracleReproducingProbe<'a> {
    pub fn new(oracle: &'a OracleStore) -> Self {
        Self { oracle }
    }
}

impl DecodeProbe for OracleReproducingProbe<'_> {
    fn generate(&mut self, fixture: &ParityFixture, prompt_index: u32) -> Vec<u32> {
        self.oracle
            .stream(&fixture.id, prompt_index)
            .unwrap_or_else(|| panic!("oracle missing {}:{}", fixture.id, prompt_index))
            .to_vec()
    }
}

/// One position where the produced stream diverged from the oracle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForkDivergence {
    pub prompt_index: u32,
    pub step: u32,
    pub produced: u32,
    pub oracle: u32,
}

/// Aggregate fork statistics for one certification run. Each divergent position
/// is modeled as a top-2 swap (a conservative count: every divergence charges
/// against the structural band), and the signature is a deterministic digest of
/// the ordered divergence list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForkSummary {
    pub top2_swaps: u32,
    pub signature: String,
    pub divergences: Vec<ForkDivergence>,
}

/// Compare one produced stream against its oracle. A length mismatch counts as
/// a divergence at the shorter stream's end.
pub fn compare_streams(produced: &[u32], oracle: &[u32], prompt_index: u32) -> Vec<ForkDivergence> {
    let mut divergences = Vec::new();
    let min_len = produced.len().min(oracle.len());
    for step in 0..min_len {
        if produced[step] != oracle[step] {
            divergences.push(ForkDivergence {
                prompt_index,
                step: step as u32,
                produced: produced[step],
                oracle: oracle[step],
            });
        }
    }
    if produced.len() != oracle.len() {
        divergences.push(ForkDivergence {
            prompt_index,
            step: min_len as u32,
            produced: produced.len() as u32,
            oracle: oracle.len() as u32,
        });
    }
    divergences
}

/// Compute the fork summary over an ordered divergence list.
pub fn fork_summary(divergences: Vec<ForkDivergence>) -> ForkSummary {
    let canonical = serde_json::to_vec(&divergences).expect("divergences serialize");
    ForkSummary {
        top2_swaps: divergences.len() as u32,
        signature: hex::encode(Sha256::digest(canonical)),
        divergences,
    }
}

/// Evidence recorded by one successful certification run. Certification and CI
/// evidence record the registry revision, the fixture group, and every executed
/// fixture ID.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CertificationEvidence {
    pub machine_profile_hash: String,
    pub decode_fingerprint: Fingerprint,
    /// Present exactly for constrained certification runs.
    pub constraint_runtime_identity: Option<String>,
    pub fixture_registry_revision: String,
    pub group: String,
    pub executed_fixture_ids: Vec<String>,
    /// Fixture ID -> digest of the produced token-stream battery.
    pub stream_digests: BTreeMap<String, String>,
    pub fork_signature: String,
    pub top2_swaps: u32,
}

impl CertificationEvidence {
    /// Stable evidence ID used by cutover records and release evidence.
    pub fn evidence_id(&self) -> String {
        let kind = match &self.constraint_runtime_identity {
            Some(cri) => format!("constrained:{cri}"),
            None => "unconstrained".to_string(),
        };
        format!(
            "cert-evidence:{}:{}:{kind}",
            self.machine_profile_hash, self.decode_fingerprint.0
        )
    }
}

/// The certification probe for one machine profile. Holds the independent
/// oracle and the structural-band checker; records rows into a certification
/// store on success and records nothing on failure (fail closed).
pub struct CertificationProbe<'a> {
    machine_profile_hash: String,
    fixture_registry_revision: String,
    oracle: &'a OracleStore,
    band_checker: StructuralBandChecker,
}

impl<'a> CertificationProbe<'a> {
    pub fn new(
        machine_profile_hash: impl Into<String>,
        fixture_registry_revision: impl Into<String>,
        oracle: &'a OracleStore,
        band_checker: StructuralBandChecker,
    ) -> Self {
        Self {
            machine_profile_hash: machine_profile_hash.into(),
            fixture_registry_revision: fixture_registry_revision.into(),
            oracle,
            band_checker,
        }
    }

    /// Certify one unconstrained lane: run the fixture through the probe,
    /// compare every prompt against the oracle, apply the structural band, and
    /// record the certification row and fork signature on success.
    pub fn certify_unconstrained_lane(
        &self,
        probe: &mut dyn DecodeProbe,
        fixture: &ParityFixture,
        decode_fingerprint: Fingerprint,
        store: &mut CertificationStore,
    ) -> Result<CertificationEvidence, OwnedDecodeError> {
        let evidence = self.run_battery(probe, fixture, decode_fingerprint, None, store)?;
        store.certify_unconstrained(
            &self.machine_profile_hash,
            evidence.decode_fingerprint.clone(),
        );
        store.store_fork_signature(
            &self.machine_profile_hash,
            &evidence.decode_fingerprint,
            &evidence.fork_signature,
        );
        Ok(evidence)
    }

    /// Certify one constrained lane, keying the row on the constraint runtime
    /// identity (shared across requests), not the per-request fingerprint.
    pub fn certify_constrained_lane(
        &self,
        probe: &mut dyn DecodeProbe,
        fixture: &ParityFixture,
        decode_fingerprint: Fingerprint,
        constraint_runtime_identity: &str,
        store: &mut CertificationStore,
    ) -> Result<CertificationEvidence, OwnedDecodeError> {
        let evidence = self.run_battery(
            probe,
            fixture,
            decode_fingerprint,
            Some(constraint_runtime_identity.to_string()),
            store,
        )?;
        store.certify_constrained(
            &self.machine_profile_hash,
            evidence.decode_fingerprint.clone(),
            constraint_runtime_identity,
        );
        store.store_fork_signature(
            &self.machine_profile_hash,
            &evidence.decode_fingerprint,
            &evidence.fork_signature,
        );
        Ok(evidence)
    }

    /// Run the battery, compare against the oracle, and enforce the structural
    /// band against the signature stored for this profile and fingerprint (if
    /// any). Records nothing; the caller records rows on success.
    fn run_battery(
        &self,
        probe: &mut dyn DecodeProbe,
        fixture: &ParityFixture,
        decode_fingerprint: Fingerprint,
        constraint_runtime_identity: Option<String>,
        store: &CertificationStore,
    ) -> Result<CertificationEvidence, OwnedDecodeError> {
        let mut divergences = Vec::new();
        let mut produced_streams = Vec::with_capacity(fixture.prompt_count as usize);
        for prompt_index in 0..fixture.prompt_count {
            let oracle = self
                .oracle
                .stream(&fixture.id, prompt_index)
                .ok_or(OwnedDecodeError::CertificationFailed)?;
            let produced = probe.generate(fixture, prompt_index);
            divergences.extend(compare_streams(&produced, oracle, prompt_index));
            produced_streams.push(produced);
        }

        let summary = fork_summary(divergences);
        let stored = store
            .stored_fork_signature(&self.machine_profile_hash, &decode_fingerprint)
            .map(str::to_string);
        self.band_checker
            .check(
                fixture.family.as_str(),
                fixture.weight_quant.as_str(),
                summary.top2_swaps,
                &summary.signature,
                stored.as_deref(),
            )
            .map_err(|_| OwnedDecodeError::CertificationFailed)?;

        let mut stream_digests = BTreeMap::new();
        stream_digests.insert(fixture.id.clone(), battery_digest(&produced_streams));

        Ok(CertificationEvidence {
            machine_profile_hash: self.machine_profile_hash.clone(),
            decode_fingerprint,
            constraint_runtime_identity,
            fixture_registry_revision: self.fixture_registry_revision.clone(),
            group: PARITY_GROUP.to_string(),
            executed_fixture_ids: vec![fixture.id.clone()],
            stream_digests,
            fork_signature: summary.signature,
            top2_swaps: summary.top2_swaps,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::owned_decode_certification::fixtures::parity_battery;
    use crate::owned_decode_contracts::StructuralBandManifest;
    use crate::owned_decode_routing::certification::CertificationAccess;
    use crate::owned_decode_routing::family::Family;
    use crate::owned_decode_routing::identity::{
        ActivationDType, DecodeIdentityInputs, WeightQuant,
    };

    fn band_checker() -> StructuralBandChecker {
        StructuralBandChecker::from_manifest(&structural_band_manifest())
    }

    fn structural_band_manifest() -> StructuralBandManifest {
        serde_json::from_str(
            r#"{
              "manifest_revision": "structural-band-v1",
              "schema_revision": "owned-decode-contracts-v1",
              "rules": [
                { "family": "qwen3-0.6b", "weight_quant": "f16", "max_top2_swaps": 2,
                  "fork_signature_recertification": "r" },
                { "family": "qwen3-0.6b", "weight_quant": "q8_0", "max_top2_swaps": 0,
                  "fork_signature_recertification": "r" },
                { "family": "lfm2-1.2b", "weight_quant": "f16", "max_top2_swaps": 2,
                  "fork_signature_recertification": "r" },
                { "family": "lfm2-1.2b", "weight_quant": "q8_0", "max_top2_swaps": 0,
                  "fork_signature_recertification": "r" }
              ]
            }"#,
        )
        .expect("structural band manifest parses")
    }

    fn decode_fingerprint(seed: &str) -> Fingerprint {
        DecodeIdentityInputs {
            family: Family::Qwen3_0_6b,
            activation_dtype: ActivationDType::F16,
            weight_quant: WeightQuant::F16,
            artifact_source_digest: format!("{seed}-source"),
            arithmetic_identity_revision: format!("{seed}-arithmetic"),
            q8: None,
        }
        .decode_fingerprint()
        .expect("valid identity inputs")
    }

    /// A probe that reproduces the oracle except for a configurable number of
    /// single-token flips on prompt zero, modeling near-tie top-2 forks.
    struct ForkingProbe<'a> {
        oracle: &'a OracleStore,
        forks: u32,
    }

    impl DecodeProbe for ForkingProbe<'_> {
        fn generate(&mut self, fixture: &ParityFixture, prompt_index: u32) -> Vec<u32> {
            let mut tokens = self
                .oracle
                .stream(&fixture.id, prompt_index)
                .expect("oracle registered")
                .to_vec();
            if prompt_index == 0 {
                for step in 0..(self.forks as usize).min(tokens.len()) {
                    tokens[step] = tokens[step].wrapping_add(1);
                }
            }
            tokens
        }
    }

    #[test]
    fn compare_streams_reports_positions_and_length_mismatch() {
        assert!(compare_streams(&[1, 2, 3], &[1, 2, 3], 0).is_empty());
        let divergences = compare_streams(&[1, 9, 3], &[1, 2, 3], 4);
        assert_eq!(divergences.len(), 1);
        assert_eq!(divergences[0].step, 1);
        assert_eq!(divergences[0].prompt_index, 4);
        let length = compare_streams(&[1, 2], &[1, 2, 3], 0);
        assert_eq!(length.len(), 1);
        assert_eq!(length[0].step, 2);
    }

    #[test]
    fn fork_summary_is_deterministic_and_counts_swaps() {
        let a = fork_summary(vec![ForkDivergence {
            prompt_index: 0,
            step: 1,
            produced: 9,
            oracle: 2,
        }]);
        let b = fork_summary(vec![ForkDivergence {
            prompt_index: 0,
            step: 1,
            produced: 9,
            oracle: 2,
        }]);
        assert_eq!(a.signature, b.signature);
        assert_eq!(a.top2_swaps, 1);
        let zero = fork_summary(Vec::new());
        assert_eq!(zero.top2_swaps, 0);
        assert_ne!(zero.signature, a.signature);
    }

    #[test]
    fn matching_probe_certifies_and_records_evidence() {
        let battery = parity_battery();
        let fixture = &battery[0];
        let mut oracle = OracleStore::new();
        oracle.register_synthetic_battery(&battery);
        let mut reproducing = OracleReproducingProbe::new(&oracle);
        let probe = CertificationProbe::new(
            "profile-m5",
            "decode-fixture-registry-v1",
            &oracle,
            band_checker(),
        );
        let mut store = CertificationStore::new();
        let fp = decode_fingerprint("qwen3-0.6b");

        let evidence = probe
            .certify_unconstrained_lane(&mut reproducing, fixture, fp.clone(), &mut store)
            .expect("byte-identical run certifies");

        assert_eq!(evidence.top2_swaps, 0);
        assert_eq!(evidence.group, PARITY_GROUP);
        assert_eq!(evidence.executed_fixture_ids, vec![fixture.id.clone()]);
        assert_eq!(
            evidence.fixture_registry_revision,
            "decode-fixture-registry-v1"
        );
        assert!(evidence.constraint_runtime_identity.is_none());
        assert!(evidence
            .evidence_id()
            .starts_with("cert-evidence:profile-m5:"));
        assert!(store.is_unconstrained_certified(
            &crate::owned_decode_routing::certification::UnconstrainedCertKey {
                machine_profile_hash: "profile-m5".to_string(),
                decode_fingerprint: fp.clone(),
            }
        ));
        assert_eq!(
            store.stored_fork_signature("profile-m5", &fp),
            Some(evidence.fork_signature.as_str())
        );
    }

    #[test]
    fn f16_within_band_certifies_and_q8_any_fork_fails_closed() {
        let battery = parity_battery();
        let mut oracle = OracleStore::new();
        oracle.register_synthetic_battery(&battery);

        // f16: one fork is within the two-swap band and certifies.
        let f16 = &battery[0];
        let mut forking = ForkingProbe {
            oracle: &oracle,
            forks: 1,
        };
        let probe = CertificationProbe::new(
            "profile-m5",
            "decode-fixture-registry-v1",
            &oracle,
            band_checker(),
        );
        let mut store = CertificationStore::new();
        let fp = decode_fingerprint("qwen3-0.6b");
        let evidence = probe
            .certify_unconstrained_lane(&mut forking, f16, fp.clone(), &mut store)
            .expect("one fork is within the f16 band");
        assert_eq!(evidence.top2_swaps, 1);

        // Q8: a single fork violates the zero-fork band and records nothing.
        let q8 = &battery[1];
        let mut forking = ForkingProbe {
            oracle: &oracle,
            forks: 1,
        };
        let q8_fp = decode_fingerprint("qwen3-0.6b-q8");
        let error = probe
            .certify_unconstrained_lane(&mut forking, q8, q8_fp.clone(), &mut store)
            .expect_err("Q8 requires zero forks");
        assert_eq!(error, OwnedDecodeError::CertificationFailed);
        assert!(
            !store.is_unconstrained_certified(
                &crate::owned_decode_routing::certification::UnconstrainedCertKey {
                    machine_profile_hash: "profile-m5".to_string(),
                    decode_fingerprint: q8_fp,
                }
            ),
            "failed certification records no row"
        );
        // The f16 fork over the band also fails: three forks exceed the ceiling.
        let mut forking = ForkingProbe {
            oracle: &oracle,
            forks: 3,
        };
        let error = probe
            .certify_unconstrained_lane(&mut forking, f16, fp, &mut store)
            .expect_err("three forks exceed the f16 band");
        assert_eq!(error, OwnedDecodeError::CertificationFailed);
    }

    #[test]
    fn recertification_requires_the_stored_signature_exactly() {
        let battery = parity_battery();
        let fixture = &battery[0];
        let mut oracle = OracleStore::new();
        oracle.register_synthetic_battery(&battery);
        let probe = CertificationProbe::new(
            "profile-m5",
            "decode-fixture-registry-v1",
            &oracle,
            band_checker(),
        );
        let mut store = CertificationStore::new();
        let fp = decode_fingerprint("qwen3-0.6b");

        // First certification with one fork records the signature.
        let mut forking = ForkingProbe {
            oracle: &oracle,
            forks: 1,
        };
        probe
            .certify_unconstrained_lane(&mut forking, fixture, fp.clone(), &mut store)
            .expect("first certification within band");

        // Recertification with the identical fork set matches the stored signature.
        let mut forking = ForkingProbe {
            oracle: &oracle,
            forks: 1,
        };
        probe
            .certify_unconstrained_lane(&mut forking, fixture, fp.clone(), &mut store)
            .expect("identical fork set recertifies");

        // Recertification with a different fork set fails: the stored signature
        // must match exactly.
        let mut forking = ForkingProbe {
            oracle: &oracle,
            forks: 2,
        };
        let error = probe
            .certify_unconstrained_lane(&mut forking, fixture, fp, &mut store)
            .expect_err("changed fork set fails recertification");
        assert_eq!(error, OwnedDecodeError::CertificationFailed);
    }

    #[test]
    fn missing_oracle_fails_closed() {
        let battery = parity_battery();
        let fixture = &battery[0];
        let oracle = OracleStore::new(); // nothing registered
        let probe = CertificationProbe::new(
            "profile-m5",
            "decode-fixture-registry-v1",
            &oracle,
            band_checker(),
        );
        let mut store = CertificationStore::new();
        // The probe machinery must fail closed with a typed error before the
        // seam is ever asked to generate.
        struct EmptyProbe;
        impl DecodeProbe for EmptyProbe {
            fn generate(&mut self, fixture: &ParityFixture, _prompt_index: u32) -> Vec<u32> {
                let _ = fixture;
                panic!("generate must not be called when the oracle is missing")
            }
        }
        let mut empty = EmptyProbe;
        let error = probe
            .certify_unconstrained_lane(
                &mut empty,
                fixture,
                decode_fingerprint("qwen3-0.6b"),
                &mut store,
            )
            .expect_err("missing oracle fails closed");
        assert_eq!(error, OwnedDecodeError::CertificationFailed);
    }

    #[test]
    fn constrained_certification_keys_on_the_runtime_identity() {
        let battery = parity_battery();
        let fixture = &battery[0];
        let mut oracle = OracleStore::new();
        oracle.register_synthetic_battery(&battery);
        let mut reproducing = OracleReproducingProbe::new(&oracle);
        let probe = CertificationProbe::new(
            "profile-m5",
            "decode-fixture-registry-v1",
            &oracle,
            band_checker(),
        );
        let mut store = CertificationStore::new();
        let fp = decode_fingerprint("qwen3-0.6b");

        let evidence = probe
            .certify_constrained_lane(
                &mut reproducing,
                fixture,
                fp.clone(),
                "cri-digest-1",
                &mut store,
            )
            .expect("constrained certification succeeds");
        assert_eq!(
            evidence.constraint_runtime_identity.as_deref(),
            Some("cri-digest-1")
        );
        assert!(evidence.evidence_id().contains("constrained:cri-digest-1"));

        use crate::owned_decode_routing::certification::{
            ConstrainedCertKey, UnconstrainedCertKey,
        };
        assert!(store.is_constrained_certified(&ConstrainedCertKey {
            machine_profile_hash: "profile-m5".to_string(),
            decode_fingerprint: fp.clone(),
            constraint_runtime_identity: "cri-digest-1".to_string(),
        }));
        assert!(
            !store.is_constrained_certified(&ConstrainedCertKey {
                machine_profile_hash: "profile-m5".to_string(),
                decode_fingerprint: fp.clone(),
                constraint_runtime_identity: "other-cri".to_string(),
            }),
            "a different runtime identity is not certified"
        );
        assert!(
            !store.is_unconstrained_certified(&UnconstrainedCertKey {
                machine_profile_hash: "profile-m5".to_string(),
                decode_fingerprint: fp,
            }),
            "constrained certification does not imply unconstrained"
        );
    }
}
