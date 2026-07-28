//! Immutable parity fixtures and the independent oracle store.
//!
//! The `decode-parity` group of `decode-fixture-registry-v1` fixes the 20-prompt
//! by 64-token batteries for both production families (`qwen3-0.6b`, `lfm2-1.2b`)
//! and both weight formats (`f16`, `q8_0`). Each [`ParityFixture`] carries the
//! identity metadata a certification probe needs (family, formats, source and
//! Q8 digests, arithmetic revision, stop IDs, selector) plus the expected
//! token-stream digest of its battery.
//!
//! Oracle bytes live in [`OracleStore`] and change only through explicit oracle
//! review: re-registering different bytes for an existing fixture prompt is a
//! hard error, and [`OracleProvenance`] has no production-output variant because
//! oracle bytes are never produced from production-worker output. The synthetic
//! [`spike_reference_stream`] stands in for reviewed spike-harness bytes in
//! hardware-independent tests; the mandatory `macos-metal` lane registers the
//! real spike reference streams instead.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::owned_decode_routing::family::Family;
use crate::owned_decode_routing::identity::{ActivationDType, WeightQuant};

/// Prompts per parity battery, per the fixture registry.
pub const PARITY_PROMPT_COUNT: u32 = 20;
/// Generated tokens per parity prompt, per the fixture registry.
pub const PARITY_MAX_TOKENS: u32 = 64;
/// The only selector the certified baseline accepts.
pub const GREEDY_TOP1: &str = "greedy_top1";
/// Fixture-registry group the parity batteries belong to.
pub const PARITY_GROUP: &str = "decode-parity";

/// Provenance of oracle bytes. Oracle bytes change only through explicit oracle
/// review and are never produced from production-worker output, so there is
/// deliberately no production-output variant: a deserialization attempt such as
/// `"production_output"` fails.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OracleProvenance {
    /// Bytes captured from the spike harness under the pinned fixture protocol.
    SpikeHarness,
    /// Bytes accepted through an explicit independent oracle review.
    IndependentReview,
}

/// One immutable parity fixture: the registry-defined identity and protocol for
/// a single lane (family + weight format) of the 20x64 battery.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParityFixture {
    /// Stable registry entry ID (e.g. `parity-qwen3-0.6b-f16-20x64`).
    pub id: String,
    pub family: Family,
    pub activation_dtype: ActivationDType,
    pub weight_quant: WeightQuant,
    /// Digest of the source weight artifact bytes.
    pub source_digest: String,
    /// Q8 derived digest; present exactly when `weight_quant` is `q8_0`.
    pub q8_derived_digest: Option<String>,
    /// Q8 quantizer revision; present exactly when `weight_quant` is `q8_0`.
    pub q8_quantizer_revision: Option<String>,
    /// Token-function identity (distinct from the metallib deployment revision).
    pub arithmetic_identity_revision: String,
    /// Digest identifying the pinned 20-prompt fixture set.
    pub prompt_fixture_digest: String,
    /// Stop token IDs treated as non-committed controls.
    pub stop_ids: Vec<u32>,
    pub max_tokens: u32,
    pub prompt_count: u32,
    /// Deterministic selector; exactly `greedy_top1` in the certified baseline.
    pub selector: String,
    /// Digest of the complete expected oracle token-stream battery.
    pub expected_stream_digest: String,
    pub oracle_provenance: OracleProvenance,
}

/// SHA-256 hex over a token stream's canonical little-endian encoding.
pub fn token_stream_digest(tokens: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    hex::encode(hasher.finalize())
}

/// Digest of a complete battery: the newline-joined per-prompt stream digests,
/// hashed. Order is part of the identity.
pub fn battery_digest(streams: &[Vec<u32>]) -> String {
    let joined = streams
        .iter()
        .map(|stream| token_stream_digest(stream))
        .collect::<Vec<_>>()
        .join("\n");
    hex::encode(Sha256::digest(joined.as_bytes()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Seed of the synthetic reference stream for one fixture prompt. Shared by
/// the single-pass oracle generator and the chunked production double so both
/// compute the identical hash chain through different control flow.
pub fn reference_stream_seed(fixture_id: &str, prompt_index: u32) -> [u8; 32] {
    Sha256::digest(format!("{fixture_id}:prompt-{prompt_index}").as_bytes()).into()
}

/// One hash-chain step of the reference stream.
pub fn reference_chain_step(state: [u8; 32]) -> ([u8; 32], u32) {
    let mut hasher = Sha256::new();
    hasher.update(b"spike-step");
    hasher.update(state);
    let next: [u8; 32] = hasher.finalize().into();
    let token = u32::from_le_bytes(next[0..4].try_into().expect("4-byte prefix"));
    (next, token)
}

/// Synthetic stand-in for the reviewed spike-harness reference stream.
///
/// Hardware-independent tests use these bytes as the oracle: they are a
/// deterministic hash chain fixed by the fixture ID and prompt index, computed
/// here independently of any production code path. The mandatory `macos-metal`
/// lane instead registers the real spike reference streams (LFM2 step reference
/// JSONL and Qwen3 spike CLI output) into the same [`OracleStore`] shape.
pub fn spike_reference_stream(fixture: &ParityFixture, prompt_index: u32) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(fixture.max_tokens as usize);
    let mut state = reference_stream_seed(&fixture.id, prompt_index);
    for _ in 0..fixture.max_tokens {
        let (next, token) = reference_chain_step(state);
        state = next;
        tokens.push(token);
    }
    tokens
}

/// One battery lane definition: registry ID, family, weight format, and the
/// Q8 quantizer revision and derived digest when the format is `q8_0`.
type BatteryLane = (
    &'static str,
    Family,
    WeightQuant,
    Option<(&'static str, &'static str)>,
);

/// The complete four-lane parity battery with the stable registry IDs.
///
/// Covers both families and both formats: `qwen3-0.6b` and `lfm2-1.2b`, each in
/// `f16` and `q8_0`. Expected stream digests are computed over the synthetic
/// reference streams and are immutable for this revision of the battery.
pub fn parity_battery() -> Vec<ParityFixture> {
    let lanes: [BatteryLane; 4] = [
        (
            "parity-qwen3-0.6b-f16-20x64",
            Family::Qwen3_0_6b,
            WeightQuant::F16,
            None,
        ),
        (
            "parity-qwen3-0.6b-q8_0-20x64",
            Family::Qwen3_0_6b,
            WeightQuant::Q8_0,
            Some(("qwen3-quantizer-v1", "qwen3-q8-derived-v1")),
        ),
        (
            "parity-lfm2-1.2b-f16-20x64",
            Family::Lfm2_1_2b,
            WeightQuant::F16,
            None,
        ),
        (
            "parity-lfm2-1.2b-q8_0-20x64",
            Family::Lfm2_1_2b,
            WeightQuant::Q8_0,
            Some(("lfm2-quantizer-v1", "lfm2-q8-derived-v1")),
        ),
    ];

    lanes
        .into_iter()
        .map(|(id, family, weight_quant, q8)| {
            let mut fixture = ParityFixture {
                id: id.to_string(),
                family,
                activation_dtype: ActivationDType::F16,
                weight_quant,
                source_digest: format!("{}-source-v1", id.trim_start_matches("parity-")),
                q8_derived_digest: q8.map(|(_, derived)| derived.to_string()),
                q8_quantizer_revision: q8.map(|(revision, _)| revision.to_string()),
                arithmetic_identity_revision: format!("{}-arithmetic-v1", family.as_str()),
                prompt_fixture_digest: sha256_hex(
                    format!("decode-prompts.jsonl:{PARITY_PROMPT_COUNT}").as_bytes(),
                ),
                stop_ids: match family {
                    Family::Qwen3_0_6b => vec![151645],
                    Family::Lfm2_1_2b => vec![2],
                },
                max_tokens: PARITY_MAX_TOKENS,
                prompt_count: PARITY_PROMPT_COUNT,
                selector: GREEDY_TOP1.to_string(),
                // Filled below once the synthetic reference battery is computed.
                expected_stream_digest: String::new(),
                oracle_provenance: OracleProvenance::SpikeHarness,
            };
            let streams: Vec<Vec<u32>> = (0..PARITY_PROMPT_COUNT)
                .map(|prompt| spike_reference_stream(&fixture, prompt))
                .collect();
            fixture.expected_stream_digest = battery_digest(&streams);
            fixture
        })
        .collect()
}

/// Failure modes for oracle registration and verification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixtureError {
    /// Different bytes were registered for an already-fixed fixture prompt.
    OracleMutated {
        fixture_id: String,
        prompt_index: u32,
    },
    /// A fixture prompt was registered under a different provenance.
    ProvenanceConflict { fixture_id: String },
    /// No oracle bytes are registered for this fixture prompt.
    OracleMissing {
        fixture_id: String,
        prompt_index: u32,
    },
    /// The registered oracle battery does not match the fixture's expected digest.
    DigestMismatch {
        fixture_id: String,
        expected: String,
        actual: String,
    },
}

/// The independent oracle store. Oracle bytes are immutable once registered:
/// re-registering identical bytes is an idempotent no-op, while re-registering
/// different bytes (or a different provenance) is a hard error.
#[derive(Clone, Debug, Default)]
pub struct OracleStore {
    streams: BTreeMap<(String, u32), Vec<u32>>,
    provenance: BTreeMap<String, OracleProvenance>,
}

impl OracleStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register oracle bytes for one fixture prompt. Idempotent for identical
    /// bytes and provenance; any mutation is rejected.
    pub fn register(
        &mut self,
        fixture_id: &str,
        prompt_index: u32,
        provenance: OracleProvenance,
        tokens: Vec<u32>,
    ) -> Result<(), FixtureError> {
        if let Some(existing_provenance) = self.provenance.get(fixture_id) {
            if *existing_provenance != provenance {
                return Err(FixtureError::ProvenanceConflict {
                    fixture_id: fixture_id.to_string(),
                });
            }
        }
        let key = (fixture_id.to_string(), prompt_index);
        if let Some(existing) = self.streams.get(&key) {
            if existing != &tokens {
                return Err(FixtureError::OracleMutated {
                    fixture_id: fixture_id.to_string(),
                    prompt_index,
                });
            }
            return Ok(());
        }
        self.provenance.insert(fixture_id.to_string(), provenance);
        self.streams.insert(key, tokens);
        Ok(())
    }

    /// Register the synthetic reference battery for every fixture and prompt.
    pub fn register_synthetic_battery(&mut self, battery: &[ParityFixture]) {
        for fixture in battery {
            for prompt_index in 0..fixture.prompt_count {
                self.register(
                    &fixture.id,
                    prompt_index,
                    fixture.oracle_provenance,
                    spike_reference_stream(fixture, prompt_index),
                )
                .expect("synthetic battery registration never conflicts");
            }
        }
    }

    /// The registered oracle stream for one fixture prompt, if any.
    pub fn stream(&self, fixture_id: &str, prompt_index: u32) -> Option<&[u32]> {
        self.streams
            .get(&(fixture_id.to_string(), prompt_index))
            .map(Vec::as_slice)
    }

    /// The registered provenance for a fixture, if any.
    pub fn provenance(&self, fixture_id: &str) -> Option<OracleProvenance> {
        self.provenance.get(fixture_id).copied()
    }

    /// Verify that the registered oracle battery for `fixture` matches its
    /// checked-in expected stream digest.
    pub fn verify_expected_digest(&self, fixture: &ParityFixture) -> Result<(), FixtureError> {
        let mut streams = Vec::with_capacity(fixture.prompt_count as usize);
        for prompt_index in 0..fixture.prompt_count {
            match self.stream(&fixture.id, prompt_index) {
                Some(stream) => streams.push(stream.to_vec()),
                None => {
                    return Err(FixtureError::OracleMissing {
                        fixture_id: fixture.id.clone(),
                        prompt_index,
                    })
                }
            }
        }
        let actual = battery_digest(&streams);
        if actual != fixture.expected_stream_digest {
            return Err(FixtureError::DigestMismatch {
                fixture_id: fixture.id.clone(),
                expected: fixture.expected_stream_digest.clone(),
                actual,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn battery_covers_both_families_and_both_formats() {
        let battery = parity_battery();
        assert_eq!(battery.len(), 4);
        let ids: Vec<&str> = battery.iter().map(|f| f.id.as_str()).collect();
        assert_eq!(
            ids,
            vec![
                "parity-qwen3-0.6b-f16-20x64",
                "parity-qwen3-0.6b-q8_0-20x64",
                "parity-lfm2-1.2b-f16-20x64",
                "parity-lfm2-1.2b-q8_0-20x64",
            ]
        );
        for fixture in &battery {
            assert_eq!(fixture.prompt_count, PARITY_PROMPT_COUNT);
            assert_eq!(fixture.max_tokens, PARITY_MAX_TOKENS);
            assert_eq!(fixture.selector, GREEDY_TOP1);
            assert_eq!(fixture.activation_dtype, ActivationDType::F16);
            let has_q8 = fixture.weight_quant == WeightQuant::Q8_0;
            assert_eq!(fixture.q8_derived_digest.is_some(), has_q8);
            assert_eq!(fixture.q8_quantizer_revision.is_some(), has_q8);
            assert!(!fixture.expected_stream_digest.is_empty());
        }
    }

    #[test]
    fn oracle_rejects_mutated_bytes_and_is_idempotent_for_identical_bytes() {
        let battery = parity_battery();
        let fixture = &battery[0];
        let mut store = OracleStore::new();
        let tokens = spike_reference_stream(fixture, 0);
        store
            .register(
                &fixture.id,
                0,
                OracleProvenance::SpikeHarness,
                tokens.clone(),
            )
            .expect("first registration succeeds");
        store
            .register(
                &fixture.id,
                0,
                OracleProvenance::SpikeHarness,
                tokens.clone(),
            )
            .expect("identical re-registration is idempotent");

        let mut mutated = tokens;
        mutated[0] = mutated[0].wrapping_add(1);
        let error = store
            .register(&fixture.id, 0, OracleProvenance::SpikeHarness, mutated)
            .expect_err("mutated oracle bytes are rejected");
        assert_eq!(
            error,
            FixtureError::OracleMutated {
                fixture_id: fixture.id.clone(),
                prompt_index: 0
            }
        );
    }

    #[test]
    fn oracle_rejects_provenance_conflict() {
        let battery = parity_battery();
        let fixture = &battery[0];
        let mut store = OracleStore::new();
        store
            .register(
                &fixture.id,
                0,
                OracleProvenance::SpikeHarness,
                spike_reference_stream(fixture, 0),
            )
            .expect("first registration succeeds");
        let error = store
            .register(
                &fixture.id,
                1,
                OracleProvenance::IndependentReview,
                spike_reference_stream(fixture, 1),
            )
            .expect_err("mixed provenance for one fixture is rejected");
        assert_eq!(
            error,
            FixtureError::ProvenanceConflict {
                fixture_id: fixture.id.clone()
            }
        );
    }

    #[test]
    fn oracle_provenance_has_no_production_output_variant() {
        // Oracle bytes are never produced from production-worker output, so the
        // provenance enum must not deserialize a production variant.
        let error = serde_json::from_str::<OracleProvenance>("\"production_output\"")
            .expect_err("production provenance is not representable");
        assert!(error.to_string().contains("unknown variant"));
        assert_eq!(
            serde_json::from_str::<OracleProvenance>("\"spike_harness\"").expect("spike parses"),
            OracleProvenance::SpikeHarness
        );
    }

    #[test]
    fn expected_digest_verifies_the_synthetic_battery_and_detects_drift() {
        let battery = parity_battery();
        let mut store = OracleStore::new();
        store.register_synthetic_battery(&battery);
        for fixture in &battery {
            store
                .verify_expected_digest(fixture)
                .expect("synthetic oracle matches the checked-in digest");
        }

        // A fresh store holding different bytes for one prompt fails the digest
        // check, modeling an oracle that drifted from the checked-in fixture.
        let fixture = &battery[0];
        let mut drifted = OracleStore::new();
        for prompt_index in 0..fixture.prompt_count {
            let mut tokens = spike_reference_stream(fixture, prompt_index);
            if prompt_index == 3 {
                tokens[10] = tokens[10].wrapping_add(1);
            }
            drifted
                .register(
                    &fixture.id,
                    prompt_index,
                    OracleProvenance::SpikeHarness,
                    tokens,
                )
                .expect("fresh store accepts first registration");
        }
        let error = drifted
            .verify_expected_digest(fixture)
            .expect_err("drifted oracle fails digest verification");
        match error {
            FixtureError::DigestMismatch { fixture_id, .. } => {
                assert_eq!(fixture_id, fixture.id)
            }
            other => panic!("expected digest mismatch, got {other:?}"),
        }
    }

    #[test]
    fn stream_digest_is_order_and_content_sensitive() {
        let a = token_stream_digest(&[1, 2, 3]);
        let b = token_stream_digest(&[3, 2, 1]);
        let c = token_stream_digest(&[1, 2, 4]);
        assert_ne!(a, b, "order is part of the digest");
        assert_ne!(a, c, "content is part of the digest");
        assert_eq!(a, token_stream_digest(&[1, 2, 3]));
    }
}
