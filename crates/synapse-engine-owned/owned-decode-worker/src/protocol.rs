//! Worker protocol frames for `owned-metal-decode-worker-v1`.
//!
//! One `microllm.oneshot` creates one logical `generation_id`. Each transport
//! session binds to one immutable `worker_generation`. The module sends
//! [`GenerateStart`], then drives a progress/continuation loop: after each
//! non-final quantum the worker emits [`GenerateProgress`] and waits for
//! [`GenerateContinue`] or [`GenerateCancel`]. A successful generation ends with
//! [`FinalResponse`], which carries complete generated IDs but no authoritative
//! text (the module owns detokenization).
//!
//! Frames are wrapped in a [`FrameEnvelope`] tagged with the protocol ID so a
//! legacy llama frame (or any foreign frame) is rejected as a protocol mismatch
//! before its body is interpreted.
//!
//! ## Stop-token selection obligation
//!
//! The grammar contract requires the worker to compute the permitted content
//! tokens from the compiled automaton, treat the configured stop IDs as
//! non-committed control candidates, run greedy selection over the union, and
//! end with `grammar_stop_before_completion` when a stop candidate wins while
//! the automaton is incomplete. The real Metal worker owns this production
//! selection; the S5 grammar-scheduler module's `greedy_generate`
//! (`owned-decode-grammar-scheduler`) is the reference semantics its fixtures
//! must match. The scripted worker in this crate models the observable
//! outcomes of that selection (stop omission from generated IDs, the
//! `stop_token` finish) but not production logits.

use serde::{Deserialize, Serialize};

use crate::error::DecodeError;
use crate::identity::{CONSTRAINT_ENCODING_ID, WORKER_PROTOCOL_ID};

/// The only sampling mode version 1 accepts.
pub const GREEDY_TOP1: &str = "greedy_top1";

/// External finish reasons. Exactly these four appear on a successful final
/// response; stop controls are omitted from generated IDs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    StopToken,
    MaxTokens,
    GrammarComplete,
    Cancelled,
}

impl FinishReason {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StopToken => "stop_token",
            Self::MaxTokens => "max_tokens",
            Self::GrammarComplete => "grammar_complete",
            Self::Cancelled => "cancelled",
        }
    }

    /// Whether this reason is a natural terminal completion. A natural
    /// completion takes precedence over a pending cancellation or deadline at
    /// the same boundary (resolution r2 #4).
    #[must_use]
    pub const fn is_terminal_completion(self) -> bool {
        matches!(
            self,
            Self::StopToken | Self::MaxTokens | Self::GrammarComplete
        )
    }
}

/// The compiled `token-id-json-constraint-v1` carried over the boundary. Raw
/// schema or grammar never crosses; only this compiled representation does.
///
/// Every field participates in constraint-identity validation. Runtime identity
/// covers the base decode fingerprint, representation revision, subset revision,
/// compiler revision, vocabulary digest, limits ID, and worker constraint-runtime
/// revision; the request fingerprint additionally covers schema, initial-state,
/// and automaton digests. Any field mismatch returns
/// `owned_decode_constraint_version_mismatch` before the first token commit.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenIdJsonConstraint {
    pub encoding_id: String,
    pub constraint_runtime_identity: String,
    pub constraint_fingerprint: String,
    pub grammar_subset_revision: String,
    pub grammar_compiler_revision: String,
    pub tokenizer_vocabulary_digest: String,
    pub limits_manifest_id: String,
    pub worker_constraint_runtime_revision: String,
    pub canonical_schema_digest: String,
    pub initial_state_encoding: String,
    pub initial_state_digest: String,
    pub compiled_automaton_digest: String,
    /// Opaque compiled automaton bytes. The worker applies the automaton before
    /// every content-token commit.
    #[serde(with = "serde_bytes_b64")]
    pub automaton_bytes: Vec<u8>,
}

impl TokenIdJsonConstraint {
    /// The ordered runtime-identity fields. Two constraints share a runtime
    /// identity (and thus a persistent certification key) when all of these
    /// agree; request-specific fields (schema, initial-state, automaton digests)
    /// are checked separately for exact substitution.
    #[must_use]
    pub fn runtime_identity_fields(&self) -> Vec<(&'static str, String)> {
        vec![
            (
                "base_decode_fingerprint",
                self.constraint_runtime_identity.clone(),
            ),
            ("representation_revision", self.encoding_id.clone()),
            ("subset_revision", self.grammar_subset_revision.clone()),
            ("compiler_revision", self.grammar_compiler_revision.clone()),
            (
                "vocabulary_digest",
                self.tokenizer_vocabulary_digest.clone(),
            ),
            ("limits_id", self.limits_manifest_id.clone()),
            (
                "worker_constraint_runtime_revision",
                self.worker_constraint_runtime_revision.clone(),
            ),
        ]
    }

    /// The ordered request-fingerprint fields: runtime identity plus the
    /// schema, initial-state, and automaton digests.
    #[must_use]
    pub fn request_fingerprint_fields(&self) -> Vec<(&'static str, String)> {
        let mut fields = self.runtime_identity_fields();
        fields.push((
            "canonical_schema_digest",
            self.canonical_schema_digest.clone(),
        ));
        fields.push(("initial_state_digest", self.initial_state_digest.clone()));
        fields.push((
            "compiled_automaton_digest",
            self.compiled_automaton_digest.clone(),
        ));
        fields
    }

    /// Compare every field against `expected`, returning the name of the first
    /// mismatching field. Used to produce a precise
    /// `owned_decode_constraint_version_mismatch`.
    #[must_use]
    pub fn first_mismatched_field(&self, expected: &Self) -> Option<&'static str> {
        let pairs: &[(&str, &str, &str)] = &[
            ("encoding_id", &self.encoding_id, &expected.encoding_id),
            (
                "constraint_runtime_identity",
                &self.constraint_runtime_identity,
                &expected.constraint_runtime_identity,
            ),
            (
                "constraint_fingerprint",
                &self.constraint_fingerprint,
                &expected.constraint_fingerprint,
            ),
            (
                "grammar_subset_revision",
                &self.grammar_subset_revision,
                &expected.grammar_subset_revision,
            ),
            (
                "grammar_compiler_revision",
                &self.grammar_compiler_revision,
                &expected.grammar_compiler_revision,
            ),
            (
                "tokenizer_vocabulary_digest",
                &self.tokenizer_vocabulary_digest,
                &expected.tokenizer_vocabulary_digest,
            ),
            (
                "limits_manifest_id",
                &self.limits_manifest_id,
                &expected.limits_manifest_id,
            ),
            (
                "worker_constraint_runtime_revision",
                &self.worker_constraint_runtime_revision,
                &expected.worker_constraint_runtime_revision,
            ),
            (
                "canonical_schema_digest",
                &self.canonical_schema_digest,
                &expected.canonical_schema_digest,
            ),
            (
                "initial_state_encoding",
                &self.initial_state_encoding,
                &expected.initial_state_encoding,
            ),
            (
                "initial_state_digest",
                &self.initial_state_digest,
                &expected.initial_state_digest,
            ),
            (
                "compiled_automaton_digest",
                &self.compiled_automaton_digest,
                &expected.compiled_automaton_digest,
            ),
        ];
        for (name, actual, expected_value) in pairs {
            if actual != expected_value {
                return Some(name);
            }
        }
        if self.automaton_bytes != expected.automaton_bytes {
            return Some("automaton_bytes");
        }
        None
    }
}

/// Serialize `Vec<u8>` as a base64 string so frames stay valid JSON text.
mod serde_bytes_b64 {
    use serde::{Deserialize, Deserializer, Serializer};

    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    fn to_b64(bytes: &[u8]) -> String {
        let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
        for chunk in bytes.chunks(3) {
            let b0 = u32::from(chunk[0]);
            let b1 = chunk.get(1).copied().map_or(0, u32::from);
            let b2 = chunk.get(2).copied().map_or(0, u32::from);
            let n = (b0 << 16) | (b1 << 8) | b2;
            out.push(TABLE[((n >> 18) & 63) as usize] as char);
            out.push(TABLE[((n >> 12) & 63) as usize] as char);
            out.push(if chunk.len() > 1 {
                TABLE[((n >> 6) & 63) as usize] as char
            } else {
                '='
            });
            out.push(if chunk.len() > 2 {
                TABLE[(n & 63) as usize] as char
            } else {
                '='
            });
        }
        out
    }

    fn from_b64(text: &str) -> Result<Vec<u8>, String> {
        fn value(byte: u8) -> Result<u32, String> {
            match byte {
                b'A'..=b'Z' => Ok(u32::from(byte - b'A')),
                b'a'..=b'z' => Ok(u32::from(byte - b'a' + 26)),
                b'0'..=b'9' => Ok(u32::from(byte - b'0' + 52)),
                b'+' => Ok(62),
                b'/' => Ok(63),
                other => Err(format!("invalid base64 byte {other}")),
            }
        }
        let bytes = text.as_bytes();
        let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
        for chunk in bytes.chunks(4) {
            if chunk.len() != 4 {
                return Err("base64 length not a multiple of 4".into());
            }
            let pad = chunk.iter().rev().take_while(|&&b| b == b'=').count();
            let mut n = 0u32;
            for (i, &byte) in chunk.iter().enumerate() {
                let v = if byte == b'=' { 0 } else { value(byte)? };
                n |= v << (18 - 6 * i);
            }
            out.push(((n >> 16) & 0xff) as u8);
            if pad < 2 {
                out.push(((n >> 8) & 0xff) as u8);
            }
            if pad < 1 {
                out.push((n & 0xff) as u8);
            }
        }
        Ok(out)
    }

    pub fn serialize<S: Serializer>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&to_b64(bytes))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
        let text = String::deserialize(deserializer)?;
        from_b64(&text).map_err(serde::de::Error::custom)
    }
}

/// Sampling selection. Version 1 accepts only [`GREEDY_TOP1`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sampling {
    pub mode: String,
    /// Reserved for future stochastic modes; empty for greedy-top-1.
    #[serde(default)]
    pub params: serde_json::Value,
}

impl Sampling {
    #[must_use]
    pub fn greedy_top1() -> Self {
        Self {
            mode: GREEDY_TOP1.to_string(),
            params: serde_json::Value::Null,
        }
    }
}

/// The start frame. Authorizes `min(production_n, max_tokens)` tokens for the
/// first quantum. Before committing a token the worker validates loaded-model
/// reference, decode fingerprint, and runtime digest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerateStart {
    pub generation_id: String,
    pub loaded_model_ref: String,
    pub decode_fingerprint: String,
    pub runtime_config_digest: String,
    pub prompt_ids: Vec<u32>,
    pub stop_ids: Vec<u32>,
    pub max_tokens: u32,
    pub sampling: Sampling,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constraint: Option<TokenIdJsonConstraint>,
}

/// Progress emitted after each non-final quantum. Carries no token IDs or text.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerateProgress {
    pub generation_id: String,
    /// First sequence is one; later sequences increment by one.
    pub quantum_sequence: u32,
    /// Attempt-local cumulative committed-token count.
    pub committed_token_count: u32,
}

/// Continuation authorizing the next quantum.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerateContinue {
    pub generation_id: String,
    /// The next expected quantum sequence (the one after the last progress).
    pub next_expected_sequence: u32,
    /// Greater than zero and no greater than N or the remaining request budget.
    pub next_token_budget: u32,
}

/// Cancellation. The worker destroys resident state and acknowledges.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerateCancel {
    pub generation_id: String,
}

/// A successful final response. Contains complete generated IDs for the
/// successful attempt and accounting, but no authoritative text.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FinalResponse {
    pub generation_id: String,
    pub generated_ids: Vec<u32>,
    pub committed_token_count: u32,
    pub decode_fingerprint: String,
    pub runtime_config_digest: String,
    pub worker_generation: u64,
    pub finish_reason: FinishReason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constraint_identity: Option<String>,
    #[serde(default)]
    pub constraint_complete: bool,
    pub last_completed_sequence: u32,
}

/// A frame emitted by the worker over a transport session.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, tag = "kind", rename_all = "snake_case")]
pub enum WorkerFrame {
    Progress(GenerateProgress),
    Final(FinalResponse),
    /// A typed error the worker surfaced as a frame (e.g. a grammar error or a
    /// worker-start mismatch). Protocol-fatal frames charge the crash budget.
    Error {
        id: String,
    },
}

/// The wire envelope. Every frame is tagged with the protocol ID; a frame whose
/// `protocol` is not [`WORKER_PROTOCOL_ID`] is a protocol mismatch before its
/// body is read. Legacy llama frames carrying raw grammar never parse here.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameEnvelope {
    pub protocol: String,
    #[serde(flatten)]
    pub frame: WorkerFrame,
}

impl FrameEnvelope {
    #[must_use]
    pub fn new(frame: WorkerFrame) -> Self {
        Self {
            protocol: WORKER_PROTOCOL_ID.to_string(),
            frame,
        }
    }

    /// Serialize to the wire form (one JSON object per frame).
    pub fn to_wire(&self) -> String {
        serde_json::to_string(self).expect("frame envelope serializes")
    }

    /// Parse a wire frame, enforcing protocol-ID and frame-structure validity.
    /// Any failure maps to `owned_decode_protocol_mismatch` — the dedicated ID
    /// for protocol-ID/frame-structure incompatibility (resolution r2 #7).
    pub fn from_wire(bytes: &str) -> Result<Self, DecodeError> {
        let envelope: FrameEnvelope =
            serde_json::from_str(bytes).map_err(|_| DecodeError::ProtocolMismatch)?;
        if envelope.protocol != WORKER_PROTOCOL_ID {
            return Err(DecodeError::ProtocolMismatch);
        }
        Ok(envelope)
    }
}

/// Owned-decode commands carried inside the standard length-prefixed worker
/// transport after the synapse-core nonce handshake. LOAD, PING, UNLOAD, and
/// SHUTDOWN retain their standard synapse-core shapes; these commands add only
/// the resident-generation protocol.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, tag = "type", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DecodeTransportRequest {
    GenerateStart {
        req_id: String,
        start: Box<GenerateStart>,
    },
    GenerateContinue {
        req_id: String,
        continuation: GenerateContinue,
    },
    GenerateCancel {
        req_id: String,
        cancellation: GenerateCancel,
    },
}

/// Response to an owned-decode transport command.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, tag = "type", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DecodeTransportResponse {
    Frame {
        req_id: String,
        envelope: FrameEnvelope,
    },
    Cancelled {
        req_id: String,
        generation_id: String,
        committed_token_count: u32,
    },
}

/// Validate a decoded envelope's structural invariants that are independent of
/// any loaded model: the protocol ID must match and a final/progress frame must
/// carry a generation id. Returns the dedicated protocol-mismatch ID on failure.
pub fn validate_frame_structure(envelope: &FrameEnvelope) -> Result<(), DecodeError> {
    if envelope.protocol != WORKER_PROTOCOL_ID {
        return Err(DecodeError::ProtocolMismatch);
    }
    match &envelope.frame {
        WorkerFrame::Progress(progress) if progress.generation_id.is_empty() => {
            Err(DecodeError::ProtocolMismatch)
        }
        WorkerFrame::Final(final_response) if final_response.generation_id.is_empty() => {
            Err(DecodeError::ProtocolMismatch)
        }
        _ => Ok(()),
    }
}

/// A helper for building a constraint with every field populated. Fixtures use
/// it and then perturb one field at a time to prove each mismatch maps to
/// `owned_decode_constraint_version_mismatch`.
#[must_use]
pub fn sample_constraint() -> TokenIdJsonConstraint {
    TokenIdJsonConstraint {
        encoding_id: CONSTRAINT_ENCODING_ID.to_string(),
        constraint_runtime_identity: "runtime-identity-1".into(),
        constraint_fingerprint: "constraint-fp-1".into(),
        grammar_subset_revision: "synapse-json-schema-v1".into(),
        grammar_compiler_revision: "compiler-r1".into(),
        tokenizer_vocabulary_digest: "vocab-digest-1".into(),
        limits_manifest_id: "limits-v1".into(),
        worker_constraint_runtime_revision: "worker-constraint-r1".into(),
        canonical_schema_digest: "schema-digest-1".into(),
        initial_state_encoding: "initial-state-encoding-v1".into(),
        initial_state_digest: "initial-state-digest-1".into(),
        compiled_automaton_digest: "automaton-digest-1".into(),
        automaton_bytes: vec![1, 2, 3, 4],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finish_reasons_are_the_documented_four() {
        assert_eq!(FinishReason::StopToken.as_str(), "stop_token");
        assert_eq!(FinishReason::MaxTokens.as_str(), "max_tokens");
        assert_eq!(FinishReason::GrammarComplete.as_str(), "grammar_complete");
        assert_eq!(FinishReason::Cancelled.as_str(), "cancelled");
        assert!(FinishReason::StopToken.is_terminal_completion());
        assert!(FinishReason::MaxTokens.is_terminal_completion());
        assert!(FinishReason::GrammarComplete.is_terminal_completion());
        assert!(!FinishReason::Cancelled.is_terminal_completion());
    }

    #[test]
    fn envelope_round_trips_and_rejects_foreign_protocol() {
        let envelope = FrameEnvelope::new(WorkerFrame::Progress(GenerateProgress {
            generation_id: "g1".into(),
            quantum_sequence: 1,
            committed_token_count: 8,
        }));
        let wire = envelope.to_wire();
        let parsed = FrameEnvelope::from_wire(&wire).expect("parse");
        assert_eq!(parsed, envelope);

        // A foreign protocol ID is a protocol mismatch.
        let foreign = wire.replace(WORKER_PROTOCOL_ID, "llama-generate-v0");
        assert_eq!(
            FrameEnvelope::from_wire(&foreign),
            Err(DecodeError::ProtocolMismatch)
        );

        // Malformed JSON is a protocol mismatch.
        assert_eq!(
            FrameEnvelope::from_wire("{not json"),
            Err(DecodeError::ProtocolMismatch)
        );
    }

    #[test]
    fn constraint_bytes_round_trip_through_base64() {
        let constraint = sample_constraint();
        let json = serde_json::to_string(&constraint).expect("serialize");
        let parsed: TokenIdJsonConstraint = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.automaton_bytes, vec![1, 2, 3, 4]);
        assert_eq!(parsed, constraint);
    }

    #[test]
    fn owned_transport_round_trips_and_rejects_unknown_fields() {
        let request = DecodeTransportRequest::GenerateCancel {
            req_id: "r1".into(),
            cancellation: GenerateCancel {
                generation_id: "g1".into(),
            },
        };
        let wire = serde_json::to_value(&request).unwrap();
        assert_eq!(
            serde_json::from_value::<DecodeTransportRequest>(wire).unwrap(),
            request
        );
        let unknown = serde_json::json!({
            "type": "GENERATE_CANCEL",
            "req_id": "r1",
            "cancellation": { "generation_id": "g1" },
            "grammar": "raw schemas are forbidden"
        });
        assert!(serde_json::from_value::<DecodeTransportRequest>(unknown).is_err());
    }

    #[test]
    fn start_and_constraint_reject_unknown_wire_fields() {
        let mut start = serde_json::to_value(GenerateStart {
            generation_id: "g1".into(),
            loaded_model_ref: "m1".into(),
            decode_fingerprint: "d1".into(),
            runtime_config_digest: "r1".into(),
            prompt_ids: vec![1],
            stop_ids: vec![2],
            max_tokens: 1,
            sampling: Sampling::greedy_top1(),
            constraint: Some(sample_constraint()),
        })
        .unwrap();
        start["raw_schema"] = serde_json::json!({});
        assert!(serde_json::from_value::<GenerateStart>(start).is_err());

        let mut constraint = serde_json::to_value(sample_constraint()).unwrap();
        constraint["unknown_revision"] = serde_json::json!("v1");
        assert!(serde_json::from_value::<TokenIdJsonConstraint>(constraint).is_err());
    }

    #[test]
    fn constraint_first_mismatched_field_is_precise() {
        let base = sample_constraint();
        let mut perturbed = sample_constraint();
        perturbed.grammar_compiler_revision = "compiler-r2".into();
        assert_eq!(
            perturbed.first_mismatched_field(&base),
            Some("grammar_compiler_revision")
        );
        assert_eq!(base.first_mismatched_field(&base), None);
    }
}
