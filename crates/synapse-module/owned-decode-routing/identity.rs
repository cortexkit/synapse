//! Identity computation for the owned-metal-decode lane.
//!
//! The specification keeps several identities deliberately separate so cache
//! invalidation, certification, substitution, and wire provenance can use the
//! narrowest correct identity:
//!
//! - `decode_fingerprint` identifies the canonical-token-ID-to-generated-token-ID
//!   function. It rotates on changes to artifact bytes, activation dtype, weight
//!   quantization, engine family, engine name, or arithmetic identity revision
//!   (and, for Q8, quantizer revision / derived digest). It never depends on
//!   scheduler settings or on `metallib_revision`.
//! - `processing_fingerprint` additionally covers the module-owned processing
//!   assets: tokenizer sanitized digest, prompt-template revision, special-token
//!   policy revision, stop-token policy revision, and detokenizer revision.
//! - `runtime_config_digest` is deployment/runtime identity: worker and protocol
//!   revisions, metallib revision, chain-K and batched-verification settings,
//!   resident limit, reservation inputs, context-manifest revision, crash policy,
//!   quarantine duration, and exactly the five runtime-effective scheduler fields.
//!   The cancellation-latency bound and other workload/evidence values never
//!   enter it.
//! - Constraint identities follow `worker_protocol_contract`: a runtime identity
//!   shared across requests, and a per-request constraint fingerprint that
//!   additionally covers schema, initial-state, and automaton digests.
//!
//! All digests are SHA-256 over a canonical serde encoding of a fixed field set,
//! so adding or reordering a field changes every derived identity.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use synapse_core::{worker_engine_names::DECODE_WORKER_ENGINE, Fingerprint};

use crate::owned_decode_contracts::SchedulerRuntimeRecord;
use crate::owned_decode_routing::error::OwnedDecodeError;
use crate::owned_decode_routing::family::Family;

/// Canonical engine name for the production owned-decode lane.
pub const OWNED_DECODE_ENGINE: &str = DECODE_WORKER_ENGINE;

/// Supported activation dtype. The specification ships exactly `f16`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationDType {
    #[serde(rename = "f16")]
    F16,
}

impl ActivationDType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F16 => "f16",
        }
    }

    pub fn parse(value: &str) -> Result<Self, OwnedDecodeError> {
        match value {
            "f16" => Ok(Self::F16),
            _ => Err(OwnedDecodeError::Unsupported),
        }
    }
}

/// Supported weight quantization: full-precision `f16` or block-quantized `q8_0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeightQuant {
    #[serde(rename = "f16")]
    F16,
    #[serde(rename = "q8_0")]
    Q8_0,
}

impl WeightQuant {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Q8_0 => "q8_0",
        }
    }

    pub fn parse(value: &str) -> Result<Self, OwnedDecodeError> {
        match value {
            "f16" => Ok(Self::F16),
            "q8_0" => Ok(Self::Q8_0),
            _ => Err(OwnedDecodeError::Unsupported),
        }
    }

    pub const fn is_q8(self) -> bool {
        matches!(self, Self::Q8_0)
    }
}

/// Q8-only identity fields. Rotating `quantizer_revision` creates a distinct
/// artifact, decode fingerprint, and transaction key, and never overwrites the
/// prior object.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Q8Identity {
    pub quantizer_revision: String,
    pub derived_digest: String,
}

/// Inputs to `decode_fingerprint`. `q8` must be present exactly when
/// `weight_quant` is `q8_0`; any other combination is rejected.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecodeIdentityInputs {
    pub family: Family,
    pub activation_dtype: ActivationDType,
    pub weight_quant: WeightQuant,
    /// Digest of the source weight artifact bytes.
    pub artifact_source_digest: String,
    /// Token-function identity. Arithmetic changes rotate this and the decode
    /// fingerprint. Distinct from `metallib_revision`, which is deployment
    /// identity and lives in `runtime_config_digest`.
    pub arithmetic_identity_revision: String,
    /// Present exactly when `weight_quant` is `q8_0`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub q8: Option<Q8Identity>,
}

/// Canonical encoding fed to the decode fingerprint hash. Field order is part
/// of the identity; do not reorder.
#[derive(Serialize)]
struct DecodeFingerprintPayload<'a> {
    engine: &'static str,
    family: &'a str,
    activation_dtype: &'a str,
    weight_quant: &'a str,
    artifact_source_digest: &'a str,
    arithmetic_identity_revision: &'a str,
    quantizer_revision: Option<&'a str>,
    derived_digest: Option<&'a str>,
}

impl DecodeIdentityInputs {
    /// Validate the format combination: activation dtype must be supported and
    /// the Q8 block must be present exactly for `q8_0`.
    pub fn validate(&self) -> Result<(), OwnedDecodeError> {
        // ActivationDType/WeightQuant are enums, so unsupported strings are
        // already rejected at parse time; here we enforce the q8-block invariant.
        match self.weight_quant {
            WeightQuant::Q8_0 if self.q8.is_none() => Err(OwnedDecodeError::Unsupported),
            WeightQuant::F16 if self.q8.is_some() => Err(OwnedDecodeError::Unsupported),
            _ => Ok(()),
        }
    }

    /// Compute the decode fingerprint. Rejects malformed format combinations
    /// before hashing so an invalid entry never yields a servable identity.
    pub fn decode_fingerprint(&self) -> Result<Fingerprint, OwnedDecodeError> {
        self.validate()?;
        let payload = DecodeFingerprintPayload {
            engine: OWNED_DECODE_ENGINE,
            family: self.family.as_str(),
            activation_dtype: self.activation_dtype.as_str(),
            weight_quant: self.weight_quant.as_str(),
            artifact_source_digest: &self.artifact_source_digest,
            arithmetic_identity_revision: &self.arithmetic_identity_revision,
            quantizer_revision: self.q8.as_ref().map(|q| q.quantizer_revision.as_str()),
            derived_digest: self.q8.as_ref().map(|q| q.derived_digest.as_str()),
        };
        Ok(Fingerprint(sha256_hex(&canonical_bytes(&payload))))
    }
}

/// Inputs to `processing_fingerprint`: the decode fingerprint plus the
/// module-owned processing-asset revisions.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessingIdentityInputs {
    pub decode_fingerprint: Fingerprint,
    pub tokenizer_sanitized_digest: String,
    pub prompt_template_revision: String,
    pub special_token_policy_revision: String,
    pub stop_token_policy_revision: String,
    pub detokenizer_revision: String,
}

#[derive(Serialize)]
struct ProcessingFingerprintPayload<'a> {
    decode_fingerprint: &'a str,
    tokenizer_sanitized_digest: &'a str,
    prompt_template_revision: &'a str,
    special_token_policy_revision: &'a str,
    stop_token_policy_revision: &'a str,
    detokenizer_revision: &'a str,
}

impl ProcessingIdentityInputs {
    /// Compute the processing fingerprint over the decode fingerprint and the
    /// five processing-asset revisions.
    pub fn processing_fingerprint(&self) -> Fingerprint {
        let payload = ProcessingFingerprintPayload {
            decode_fingerprint: &self.decode_fingerprint.0,
            tokenizer_sanitized_digest: &self.tokenizer_sanitized_digest,
            prompt_template_revision: &self.prompt_template_revision,
            special_token_policy_revision: &self.special_token_policy_revision,
            stop_token_policy_revision: &self.stop_token_policy_revision,
            detokenizer_revision: &self.detokenizer_revision,
        };
        Fingerprint(sha256_hex(&canonical_bytes(&payload)))
    }
}

/// The canonical worker runtime manifest whose digest is `runtime_config_digest`.
///
/// The scheduler portion is exactly the five runtime-effective fields
/// (`SchedulerRuntimeRecord`); the cancellation-latency bound and all workload
/// or evidence fields are deliberately absent so they cannot enter runtime
/// identity unless a later contract makes them runtime-effective.
///
/// `SchedulerRuntimeRecord` does not derive `PartialEq`/`Eq`, so this manifest
/// intentionally omits those derives too; identity comparisons use `digest()`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeConfigManifest {
    pub worker_revision: String,
    pub protocol_revision: String,
    pub metallib_revision: String,
    /// Baseline chain-K is one.
    pub chain_k: u32,
    /// Baseline batched verification is disabled.
    pub batched_verification: bool,
    /// Version 1 permits exactly one resident generation.
    pub resident_limit: u32,
    /// Attention-KV reservation input (units of the context bucket).
    pub attention_kv_reservation_units: u64,
    /// LFM2 convolution-cache reservation input in bytes; zero for families
    /// without a convolution cache.
    pub lfm2_conv_cache_reservation_bytes: u64,
    /// Revision of the shippable context manifest (`decode-context-buckets-v1`).
    pub context_manifest_revision: String,
    pub crash_policy_revision: String,
    pub quarantine_duration_ms: u64,
    /// Exactly the five runtime-effective scheduler fields.
    pub scheduler: SchedulerRuntimeRecord,
}

impl RuntimeConfigManifest {
    /// Compute `runtime_config_digest` over the canonical runtime manifest.
    pub fn digest(&self) -> String {
        sha256_hex(&canonical_bytes(self))
    }
}

/// Constraint runtime identity shared across requests for one certified
/// constrained lane. Covers exactly the fields `worker_protocol_contract` names:
/// base decode fingerprint, representation revision, grammar-subset revision,
/// grammar-compiler revision, tokenizer-vocabulary digest, limits-manifest ID,
/// and worker constraint-runtime revision.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConstraintRuntimeIdentity {
    pub base_decode_fingerprint: Fingerprint,
    pub representation_revision: String,
    pub grammar_subset_revision: String,
    pub grammar_compiler_revision: String,
    pub tokenizer_vocabulary_digest: String,
    pub limits_manifest_id: String,
    pub worker_constraint_runtime_revision: String,
}

impl ConstraintRuntimeIdentity {
    /// Stable digest of the runtime identity. This is the constrained
    /// certification key component, not the per-request fingerprint.
    pub fn digest(&self) -> String {
        sha256_hex(&canonical_bytes(self))
    }
}

/// Per-request constraint fingerprint inputs. Additionally covers the canonical
/// schema digest, initial-state encoding and digest, and compiled automaton
/// digest. This is an exact substitution check, not a certification key.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConstraintFingerprintInputs {
    pub runtime_identity_digest: String,
    pub canonical_schema_digest: String,
    pub initial_state_encoding: String,
    pub initial_state_digest: String,
    pub compiled_automaton_digest: String,
}

impl ConstraintFingerprintInputs {
    /// Compute the per-request constraint fingerprint.
    pub fn fingerprint(&self) -> Fingerprint {
        Fingerprint(sha256_hex(&canonical_bytes(self)))
    }
}

/// Serialize a value to canonical bytes. Struct field order is fixed by the
/// type definition, so the encoding is deterministic for a given input.
fn canonical_bytes<T: Serialize>(value: &T) -> Vec<u8> {
    serde_json::to_vec(value).expect("identity payload serializes")
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16_inputs() -> DecodeIdentityInputs {
        DecodeIdentityInputs {
            family: Family::Qwen3_0_6b,
            activation_dtype: ActivationDType::F16,
            weight_quant: WeightQuant::F16,
            artifact_source_digest: "sha256:weights".to_string(),
            arithmetic_identity_revision: "arith-v1".to_string(),
            q8: None,
        }
    }

    fn q8_inputs() -> DecodeIdentityInputs {
        DecodeIdentityInputs {
            family: Family::Qwen3_0_6b,
            activation_dtype: ActivationDType::F16,
            weight_quant: WeightQuant::Q8_0,
            artifact_source_digest: "sha256:weights".to_string(),
            arithmetic_identity_revision: "arith-v1".to_string(),
            q8: Some(Q8Identity {
                quantizer_revision: "quant-v1".to_string(),
                derived_digest: "sha256:q8-derived".to_string(),
            }),
        }
    }

    #[test]
    fn decode_fingerprint_is_stable_for_identical_inputs() {
        let a = f16_inputs().decode_fingerprint().unwrap();
        let b = f16_inputs().decode_fingerprint().unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn decode_fingerprint_rotates_on_each_identity_input() {
        let base = f16_inputs().decode_fingerprint().unwrap();

        let mut changed = f16_inputs();
        changed.family = Family::Lfm2_1_2b;
        assert_ne!(changed.decode_fingerprint().unwrap(), base, "family");

        let mut changed = f16_inputs();
        changed.artifact_source_digest = "sha256:other".to_string();
        assert_ne!(
            changed.decode_fingerprint().unwrap(),
            base,
            "artifact bytes"
        );

        let mut changed = f16_inputs();
        changed.arithmetic_identity_revision = "arith-v2".to_string();
        assert_ne!(
            changed.decode_fingerprint().unwrap(),
            base,
            "arithmetic revision"
        );

        // f16 -> q8_0 rotates the fingerprint.
        assert_ne!(
            q8_inputs().decode_fingerprint().unwrap(),
            base,
            "weight quant"
        );
    }

    #[test]
    fn q8_quantizer_revision_rotates_decode_fingerprint() {
        let a = q8_inputs().decode_fingerprint().unwrap();
        let mut rotated = q8_inputs();
        rotated.q8.as_mut().unwrap().quantizer_revision = "quant-v2".to_string();
        assert_ne!(rotated.decode_fingerprint().unwrap(), a);
    }

    #[test]
    fn q8_block_must_match_weight_quant() {
        // q8_0 without the q8 block is rejected.
        let missing = DecodeIdentityInputs {
            q8: None,
            ..q8_inputs()
        };
        assert_eq!(
            missing.decode_fingerprint(),
            Err(OwnedDecodeError::Unsupported)
        );

        // f16 with a stray q8 block is rejected.
        let stray = DecodeIdentityInputs {
            q8: Some(Q8Identity {
                quantizer_revision: "quant-v1".to_string(),
                derived_digest: "sha256:x".to_string(),
            }),
            ..f16_inputs()
        };
        assert_eq!(
            stray.decode_fingerprint(),
            Err(OwnedDecodeError::Unsupported)
        );
    }

    #[test]
    fn processing_fingerprint_rotates_on_processing_assets_only() {
        let decode = f16_inputs().decode_fingerprint().unwrap();
        let base = ProcessingIdentityInputs {
            decode_fingerprint: decode.clone(),
            tokenizer_sanitized_digest: "sha256:tok".to_string(),
            prompt_template_revision: "tmpl-v1".to_string(),
            special_token_policy_revision: "special-v1".to_string(),
            stop_token_policy_revision: "stop-v1".to_string(),
            detokenizer_revision: "detok-v1".to_string(),
        }
        .processing_fingerprint();

        // A processing-asset change rotates the processing fingerprint...
        let mut changed = ProcessingIdentityInputs {
            decode_fingerprint: decode.clone(),
            tokenizer_sanitized_digest: "sha256:tok".to_string(),
            prompt_template_revision: "tmpl-v2".to_string(),
            special_token_policy_revision: "special-v1".to_string(),
            stop_token_policy_revision: "stop-v1".to_string(),
            detokenizer_revision: "detok-v1".to_string(),
        };
        assert_ne!(changed.processing_fingerprint(), base);

        // ...while identical processing assets over the same decode fingerprint
        // reproduce it.
        changed.prompt_template_revision = "tmpl-v1".to_string();
        assert_eq!(changed.processing_fingerprint(), base);
    }

    fn runtime_manifest() -> RuntimeConfigManifest {
        RuntimeConfigManifest {
            worker_revision: "worker-v1".to_string(),
            protocol_revision: "owned-metal-decode-worker-v1".to_string(),
            metallib_revision: "metallib-v1".to_string(),
            chain_k: 1,
            batched_verification: false,
            resident_limit: 1,
            attention_kv_reservation_units: 2048,
            lfm2_conv_cache_reservation_bytes: 0,
            context_manifest_revision: "decode-context-buckets-v1".to_string(),
            crash_policy_revision: "crash-v1".to_string(),
            quarantine_duration_ms: 60_000,
            scheduler: SchedulerRuntimeRecord {
                production_n: 16,
                yield_policy_revision: "yield-on-contention-v1".to_string(),
                decode_weight: 4,
                decode_aging_window_ms: 250,
                progress_protocol_revision: "generate-progress-v1".to_string(),
            },
        }
    }

    #[test]
    fn runtime_digest_rotates_on_runtime_and_scheduler_fields() {
        let base = runtime_manifest().digest();

        let mut changed = runtime_manifest();
        changed.metallib_revision = "metallib-v2".to_string();
        assert_ne!(
            changed.digest(),
            base,
            "metallib revision is runtime identity"
        );

        let mut changed = runtime_manifest();
        changed.scheduler.production_n = 32;
        assert_ne!(changed.digest(), base, "production N enters runtime digest");

        let mut changed = runtime_manifest();
        changed.scheduler.decode_weight = 8;
        assert_ne!(
            changed.digest(),
            base,
            "decode weight enters runtime digest"
        );
    }

    #[test]
    fn constraint_identities_are_field_sensitive() {
        let decode = f16_inputs().decode_fingerprint().unwrap();
        let runtime = ConstraintRuntimeIdentity {
            base_decode_fingerprint: decode,
            representation_revision: "token-id-json-constraint-v1".to_string(),
            grammar_subset_revision: "synapse-json-schema-v1".to_string(),
            grammar_compiler_revision: "compiler-v1".to_string(),
            tokenizer_vocabulary_digest: "sha256:vocab".to_string(),
            limits_manifest_id: "limits-v1".to_string(),
            worker_constraint_runtime_revision: "runtime-v1".to_string(),
        };
        let runtime_digest = runtime.digest();

        let request = ConstraintFingerprintInputs {
            runtime_identity_digest: runtime_digest.clone(),
            canonical_schema_digest: "sha256:schema".to_string(),
            initial_state_encoding: "json".to_string(),
            initial_state_digest: "sha256:initial".to_string(),
            compiled_automaton_digest: "sha256:automaton".to_string(),
        };
        let fp = request.fingerprint();

        // A schema change rotates the request fingerprint but not the runtime id.
        let mut changed = request.clone();
        changed.canonical_schema_digest = "sha256:schema2".to_string();
        assert_ne!(changed.fingerprint(), fp);

        // A compiler revision change rotates the runtime identity.
        let mut changed_runtime = runtime.clone();
        changed_runtime.grammar_compiler_revision = "compiler-v2".to_string();
        assert_ne!(changed_runtime.digest(), runtime_digest);
    }

    #[test]
    fn identity_structs_reject_unknown_fields() {
        // fail-closed posture: every deserializable identity-bearing struct
        // rejects an unknown field at parse time. One representative per
        // struct family: DecodeIdentityInputs, ProcessingIdentityInputs,
        // RuntimeConfigManifest, ConstraintRuntimeIdentity,
        // ConstraintFingerprintInputs, and Q8Identity.
        let bad_decode = serde_json::json!({
            "family": "qwen3-0.6b",
            "activation_dtype": "f16",
            "weight_quant": "f16",
            "artifact_source_digest": "sha256:w",
            "arithmetic_identity_revision": "arith-v1",
            "unknown": "x",
        });
        assert!(serde_json::from_value::<DecodeIdentityInputs>(bad_decode).is_err());

        let bad_proc = serde_json::json!({
            "decode_fingerprint": "fp",
            "tokenizer_sanitized_digest": "d",
            "prompt_template_revision": "r",
            "special_token_policy_revision": "r",
            "stop_token_policy_revision": "r",
            "detokenizer_revision": "r",
            "unknown": "x",
        });
        assert!(serde_json::from_value::<ProcessingIdentityInputs>(bad_proc).is_err());

        let bad_runtime = serde_json::json!({
            "worker_revision": "w",
            "protocol_revision": "p",
            "metallib_revision": "m",
            "chain_k": 1,
            "batched_verification": false,
            "resident_limit": 1,
            "attention_kv_reservation_units": 2048,
            "lfm2_conv_cache_reservation_bytes": 0,
            "context_manifest_revision": "c",
            "crash_policy_revision": "cr",
            "quarantine_duration_ms": 60000,
            "scheduler": {
                "production_n": 16,
                "yield_policy_revision": "y",
                "decode_weight": 4,
                "decode_aging_window_ms": 250,
                "progress_protocol_revision": "p"
            },
            "unknown": "x",
        });
        assert!(serde_json::from_value::<RuntimeConfigManifest>(bad_runtime).is_err());

        let bad_constraint_runtime = serde_json::json!({
            "base_decode_fingerprint": "fp",
            "representation_revision": "r",
            "grammar_subset_revision": "r",
            "grammar_compiler_revision": "r",
            "tokenizer_vocabulary_digest": "d",
            "limits_manifest_id": "l",
            "worker_constraint_runtime_revision": "r",
            "unknown": "x",
        });
        assert!(
            serde_json::from_value::<ConstraintRuntimeIdentity>(bad_constraint_runtime).is_err()
        );

        let bad_constraint_fp = serde_json::json!({
            "runtime_identity_digest": "d",
            "canonical_schema_digest": "d",
            "initial_state_encoding": "e",
            "initial_state_digest": "d",
            "compiled_automaton_digest": "d",
            "unknown": "x",
        });
        assert!(serde_json::from_value::<ConstraintFingerprintInputs>(bad_constraint_fp).is_err());

        let bad_q8 = serde_json::json!({
            "quantizer_revision": "q",
            "derived_digest": "d",
            "unknown": "x",
        });
        assert!(serde_json::from_value::<Q8Identity>(bad_q8).is_err());
    }
}
