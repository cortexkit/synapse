//! Grammar compilation into the `token-id-json-constraint-v1` representation.
//!
//! The module exclusively owns grammar compilation. A request's `grammar` field
//! (a JSON schema in the `synapse-json-schema-v1` subset) is parsed, validated
//! against the checked-in limits, and compiled into a versioned token-ID
//! constraint that is the *only* thing that crosses the worker boundary — raw
//! schema or grammar never does. The worker applies the compiled automaton before
//! every content-token commit.
//!
//! This module also fixes the two constraint identities from
//! `worker_protocol_contract`:
//! - [`ConstraintRuntimeIdentity`]: shared across requests for one certified
//!   constrained lane (base decode fingerprint, representation revision,
//!   grammar-subset revision, compiler revision, vocabulary digest, limits ID,
//!   worker constraint-runtime revision). It is the constrained certification key
//!   component.
//! - The per-request [`ConstraintFingerprintInputs`]: additionally covers the
//!   canonical schema digest, initial-state encoding and digest, and compiled
//!   automaton digest. It is an exact substitution check, not a certification key.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::owned_decode_grammar_scheduler::grammar_automaton::Automaton;
use crate::owned_decode_grammar_scheduler::grammar_limits::{
    GrammarLimits, GrammarSubsetManifest, REPRESENTATION_REVISION,
};
use crate::owned_decode_grammar_scheduler::grammar_schema::{parse_schema, Schema, SchemaError};
use crate::owned_decode_routing::error::OwnedDecodeError;
use crate::owned_decode_routing::identity::{
    ConstraintFingerprintInputs, ConstraintRuntimeIdentity,
};
use synapse_core::Fingerprint;

/// The encoding label for the compiled automaton bytes. The worker uses this to
/// select the constraint runtime that interprets `automaton_bytes`.
pub const AUTOMATON_ENCODING: &str = "json-pushdown-automaton-v1";

/// The encoding label for the initial constraint state handed to the worker.
pub const INITIAL_STATE_ENCODING: &str = "json-automaton-initial-state-v1";

/// The compiled, versioned token-ID constraint that crosses the worker boundary.
///
/// Field set matches `worker_protocol_contract`'s `token-id-json-constraint-v1`
/// exactly. Any field mismatch at worker start returns
/// `owned_decode_constraint_version_mismatch` before the first token commit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenIdJsonConstraintV1 {
    /// Always `token-id-json-constraint-v1`.
    pub representation_revision: String,
    /// Shared constrained-lane identity (the certification key component).
    pub constraint_runtime_identity: ConstraintRuntimeIdentity,
    /// Per-request exact-substitution fingerprint.
    pub constraint_fingerprint: Fingerprint,
    /// Digest of the tokenizer vocabulary the constraint was compiled against.
    pub tokenizer_vocabulary_digest: String,
    /// Checked-in limits-manifest identity enforced during compilation.
    pub limits_manifest_id: String,
    /// Digest of the canonical schema (key-sorted JSON of the validated schema).
    pub canonical_schema_digest: String,
    /// Label naming how the initial state is encoded.
    pub initial_state_encoding: String,
    /// Digest of the encoded initial constraint state.
    pub initial_state_digest: String,
    /// Digest of the compiled automaton bytes.
    pub compiled_automaton_digest: String,
    /// The serialized compiled automaton the worker interprets.
    pub automaton_bytes: Vec<u8>,
}

/// The serialized compiled automaton. This is what `automaton_bytes` encodes:
/// everything the worker needs to run the constraint, and nothing else (no raw
/// schema text, no grammar source).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledAutomaton {
    /// The validated schema arena (the compiled grammar).
    pub schema: Schema,
    /// Encoding label for these bytes.
    pub encoding: String,
    /// Compiler revision that produced this automaton.
    pub grammar_compiler_revision: String,
    /// Grammar-subset revision the schema conforms to.
    pub grammar_subset_revision: String,
    /// Limits enforced at compile time.
    pub limits: GrammarLimits,
}

/// A compiled constraint plus the live automaton, returned together so callers
/// can both ship the wire representation and drive generation in-process.
#[derive(Clone, Debug)]
pub struct CompiledConstraint {
    /// The wire representation.
    pub constraint: TokenIdJsonConstraintV1,
    /// The runnable automaton (rebuilt from the same schema).
    pub automaton: Automaton,
}

/// Inputs needed to compile a grammar that are not part of the grammar text
/// itself: the base decode fingerprint of the selected lane and the digest of the
/// tokenizer vocabulary the constraint is compiled against.
#[derive(Clone, Debug)]
pub struct CompileContext {
    pub base_decode_fingerprint: Fingerprint,
    pub tokenizer_vocabulary_digest: String,
}

/// Compile a raw grammar string into a [`CompiledConstraint`].
///
/// Returns the grammar parse/feature error if the schema is malformed or outside
/// the subset or limits. The `manifest` supplies the versioned revisions and
/// limits that enter constraint-runtime identity.
pub fn compile_grammar(
    raw_grammar: &str,
    context: &CompileContext,
    manifest: &GrammarSubsetManifest,
) -> Result<CompiledConstraint, SchemaError> {
    let limits = manifest.limits;
    let schema = parse_schema(raw_grammar, &limits)?;

    // Compiled-state limit, enforced pre-dispatch like the other checked-in
    // limits. The v1 compiler emits a push-down automaton whose control state
    // is exactly the schema arena (one schema node per addressable state), so
    // the schema node count is the honest v1 compiled-state measure; a future
    // compiler that expands states must compare its real state count here.
    if schema.node_count() > limits.max_compiled_state_count {
        return Err(SchemaError::feature(format!(
            "compiled automaton state count {} exceeds the {} limit",
            schema.node_count(),
            limits.max_compiled_state_count
        )));
    }

    let canonical_schema_digest = sha256_hex(&canonical_schema_bytes(&schema));

    let compiled = CompiledAutomaton {
        schema: schema.clone(),
        encoding: AUTOMATON_ENCODING.to_string(),
        grammar_compiler_revision: manifest.grammar_compiler_revision.clone(),
        grammar_subset_revision: manifest.grammar_subset_revision.clone(),
        limits,
    };
    let automaton_bytes = serde_json::to_vec(&compiled).expect("compiled automaton serializes");
    let compiled_automaton_digest = sha256_hex(&automaton_bytes);

    let initial_state_digest = sha256_hex(&initial_state_bytes(&schema));

    let runtime_identity = ConstraintRuntimeIdentity {
        base_decode_fingerprint: context.base_decode_fingerprint.clone(),
        representation_revision: REPRESENTATION_REVISION.to_string(),
        grammar_subset_revision: manifest.grammar_subset_revision.clone(),
        grammar_compiler_revision: manifest.grammar_compiler_revision.clone(),
        tokenizer_vocabulary_digest: context.tokenizer_vocabulary_digest.clone(),
        limits_manifest_id: manifest.limits_manifest_id.clone(),
        worker_constraint_runtime_revision: manifest.worker_constraint_runtime_revision.clone(),
    };

    let fingerprint_inputs = ConstraintFingerprintInputs {
        runtime_identity_digest: runtime_identity.digest(),
        canonical_schema_digest: canonical_schema_digest.clone(),
        initial_state_encoding: INITIAL_STATE_ENCODING.to_string(),
        initial_state_digest: initial_state_digest.clone(),
        compiled_automaton_digest: compiled_automaton_digest.clone(),
    };

    let constraint = TokenIdJsonConstraintV1 {
        representation_revision: REPRESENTATION_REVISION.to_string(),
        constraint_runtime_identity: runtime_identity,
        constraint_fingerprint: fingerprint_inputs.fingerprint(),
        tokenizer_vocabulary_digest: context.tokenizer_vocabulary_digest.clone(),
        limits_manifest_id: manifest.limits_manifest_id.clone(),
        canonical_schema_digest,
        initial_state_encoding: INITIAL_STATE_ENCODING.to_string(),
        initial_state_digest,
        compiled_automaton_digest,
        automaton_bytes,
    };

    Ok(CompiledConstraint {
        constraint,
        automaton: Automaton::new(schema),
    })
}

/// Rebuild a runnable automaton from a shipped constraint's `automaton_bytes`.
/// This is the worker-side load path in model form: it verifies the encoding and
/// revisions before returning the automaton, mirroring the worker's
/// constraint-field checks that return `owned_decode_constraint_version_mismatch`.
pub fn load_automaton(
    constraint: &TokenIdJsonConstraintV1,
    manifest: &GrammarSubsetManifest,
) -> Result<Automaton, OwnedDecodeError> {
    let compiled: CompiledAutomaton = serde_json::from_slice(&constraint.automaton_bytes)
        .map_err(|_| OwnedDecodeError::ConstraintVersionMismatch)?;
    if compiled.encoding != AUTOMATON_ENCODING
        || compiled.grammar_compiler_revision != manifest.grammar_compiler_revision
        || compiled.grammar_subset_revision != manifest.grammar_subset_revision
        || compiled.limits != manifest.limits
    {
        return Err(OwnedDecodeError::ConstraintVersionMismatch);
    }
    if sha256_hex(&constraint.automaton_bytes) != constraint.compiled_automaton_digest {
        return Err(OwnedDecodeError::ConstraintVersionMismatch);
    }
    Ok(Automaton::new(compiled.schema))
}

/// Canonical schema bytes: the validated schema arena serialized to JSON. The
/// arena's node order is fixed by construction, so this is deterministic for a
/// given parsed schema.
fn canonical_schema_bytes(schema: &Schema) -> Vec<u8> {
    serde_json::to_vec(schema).expect("schema serializes")
}

/// The encoded initial constraint state. It names the encoding and the root
/// type so the digest is tied to the schema rather than being a global constant.
fn initial_state_bytes(schema: &Schema) -> Vec<u8> {
    let root_type = format!("{:?}", schema.root().ty);
    serde_json::to_vec(&serde_json::json!({
        "encoding": INITIAL_STATE_ENCODING,
        "root_type": root_type,
        "stack_depth": 0,
        "complete": false,
    }))
    .expect("initial state serializes")
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

/// A convenience digest over a tokenizer vocabulary, so callers can derive the
/// `tokenizer_vocabulary_digest` identity input from the vocabulary itself.
pub fn vocabulary_digest(vocabulary: &[String]) -> String {
    let mut hasher = Sha256::new();
    for token in vocabulary {
        hasher.update(token.as_bytes());
        hasher.update([0u8]); // frame separator so ["ab","c"] != ["a","bc"]
    }
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::owned_decode_grammar_scheduler::grammar_limits::LIMITS_MANIFEST_ID;

    fn context() -> CompileContext {
        CompileContext {
            base_decode_fingerprint: Fingerprint("base-decode-fp".to_string()),
            tokenizer_vocabulary_digest: "vocab-digest".to_string(),
        }
    }

    const SCHEMA: &str = r#"{
        "type": "object",
        "properties": { "name": { "type": "string" } },
        "required": ["name"],
        "additionalProperties": false
    }"#;

    #[test]
    fn compile_produces_all_identity_fields() {
        let manifest = GrammarSubsetManifest::default();
        let compiled = compile_grammar(SCHEMA, &context(), &manifest).expect("compiles");
        let constraint = &compiled.constraint;
        assert_eq!(
            constraint.representation_revision,
            "token-id-json-constraint-v1"
        );
        assert_eq!(constraint.limits_manifest_id, LIMITS_MANIFEST_ID);
        assert_eq!(
            constraint
                .constraint_runtime_identity
                .grammar_subset_revision,
            "synapse-json-schema-v1"
        );
        assert!(!constraint.canonical_schema_digest.is_empty());
        assert!(!constraint.initial_state_digest.is_empty());
        assert!(!constraint.compiled_automaton_digest.is_empty());
        assert!(!constraint.automaton_bytes.is_empty());
        assert!(!constraint.constraint_fingerprint.0.is_empty());
    }

    #[test]
    fn compile_is_deterministic() {
        let manifest = GrammarSubsetManifest::default();
        let a = compile_grammar(SCHEMA, &context(), &manifest).expect("compiles");
        let b = compile_grammar(SCHEMA, &context(), &manifest).expect("compiles");
        assert_eq!(a.constraint, b.constraint);
    }

    #[test]
    fn schema_change_rotates_fingerprint_but_not_runtime_identity() {
        let manifest = GrammarSubsetManifest::default();
        let other = r#"{
            "type": "object",
            "properties": { "age": { "type": "integer" } },
            "required": ["age"],
            "additionalProperties": false
        }"#;
        let a = compile_grammar(SCHEMA, &context(), &manifest).expect("compiles");
        let b = compile_grammar(other, &context(), &manifest).expect("compiles");
        // Different schema: per-request fingerprint differs...
        assert_ne!(
            a.constraint.constraint_fingerprint,
            b.constraint.constraint_fingerprint
        );
        // ...but the shared runtime identity is unchanged (same lane, compiler,
        // vocabulary, limits), so constrained certification is unaffected.
        assert_eq!(
            a.constraint.constraint_runtime_identity,
            b.constraint.constraint_runtime_identity
        );
    }

    #[test]
    fn vocabulary_change_rotates_runtime_identity() {
        let manifest = GrammarSubsetManifest::default();
        let ctx_a = context();
        let ctx_b = CompileContext {
            tokenizer_vocabulary_digest: "different-vocab".to_string(),
            ..context()
        };
        let a = compile_grammar(SCHEMA, &ctx_a, &manifest).expect("compiles");
        let b = compile_grammar(SCHEMA, &ctx_b, &manifest).expect("compiles");
        assert_ne!(
            a.constraint.constraint_runtime_identity,
            b.constraint.constraint_runtime_identity
        );
    }

    #[test]
    fn load_round_trips_the_automaton() {
        let manifest = GrammarSubsetManifest::default();
        let compiled = compile_grammar(SCHEMA, &context(), &manifest).expect("compiles");
        let automaton = load_automaton(&compiled.constraint, &manifest).expect("loads");
        // The reloaded automaton accepts a valid document.
        let state = crate::owned_decode_grammar_scheduler::grammar_automaton::drain_bytes(
            &automaton,
            br#"{"name":"ada"}"#,
        )
        .expect("document accepted");
        assert!(automaton.has_complete_value(&state));
    }

    #[test]
    fn load_rejects_revision_mismatch() {
        let manifest = GrammarSubsetManifest::default();
        let compiled = compile_grammar(SCHEMA, &context(), &manifest).expect("compiles");
        let mut rotated = manifest.clone();
        rotated.grammar_compiler_revision = "grammar-compiler-v2".to_string();
        let error = load_automaton(&compiled.constraint, &rotated).expect_err("mismatch rejected");
        assert_eq!(error, OwnedDecodeError::ConstraintVersionMismatch);
    }

    #[test]
    fn compile_rejects_unsupported_schema() {
        let manifest = GrammarSubsetManifest::default();
        let raw = r#"{ "type": "string", "pattern": "^a" }"#;
        let error = compile_grammar(raw, &context(), &manifest).expect_err("rejected");
        assert_eq!(
            error.wire_error(),
            OwnedDecodeError::GrammarFeatureUnsupported
        );
    }

    #[test]
    fn compile_enforces_compiled_state_count_limit() {
        // The test schema compiles to two nodes (object root + one property).
        // Lowering the compiled-state limit below that count rejects the
        // schema pre-dispatch with the typed feature error.
        let mut manifest = GrammarSubsetManifest::default();
        manifest.limits.max_compiled_state_count = 1;
        let error = compile_grammar(SCHEMA, &context(), &manifest)
            .expect_err("over-state-limit schema rejected");
        assert_eq!(
            error.wire_error(),
            OwnedDecodeError::GrammarFeatureUnsupported
        );
        assert!(
            error.message.contains("state count"),
            "reason should name the state-count limit: {}",
            error.message
        );

        // The same schema compiles when the limit admits its state count.
        manifest.limits.max_compiled_state_count = 2;
        compile_grammar(SCHEMA, &context(), &manifest).expect("within limit compiles");
    }

    #[test]
    fn vocabulary_digest_is_order_and_framing_sensitive() {
        let a = vocabulary_digest(&["ab".to_string(), "c".to_string()]);
        let b = vocabulary_digest(&["a".to_string(), "bc".to_string()]);
        assert_ne!(a, b);
    }
}
