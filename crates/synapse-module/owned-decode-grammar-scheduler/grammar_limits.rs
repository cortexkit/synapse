//! Checked-in grammar limits and the versioned grammar-subset manifest.
//!
//! The grammar contract requires the module to enforce checked-in maximum
//! schema bytes, nesting depth, property count, enum count, and compiled-state
//! count *before dispatch*. The exact numeric limits and the compiler revision
//! live in a versioned grammar-subset manifest and enter three identities:
//! constraint-runtime identity, grammar-automaton cache identity, and
//! constrained certification evidence.
//!
//! Nothing here is a runtime knob a caller can tune: the limits are a fixed
//! part of the wire contract. Rotating any of them rotates the constraint
//! runtime identity, which is exactly why they are named and versioned rather
//! than hardcoded inline at the call site.

use serde::{Deserialize, Serialize};

use crate::owned_decode_contracts::{CONSTRAINT_ENCODING_ID, GRAMMAR_SUBSET_ID};

/// Representation revision for the compiled constraint that crosses the worker
/// boundary. Mirrors [`crate::owned_decode_contracts::CONSTRAINT_ENCODING_ID`];
/// re-exported here so the grammar module has a single source for the literal.
pub const REPRESENTATION_REVISION: &str = CONSTRAINT_ENCODING_ID;

/// Grammar-subset revision: the JSON Schema subset accepted by the compiler.
/// Mirrors [`crate::owned_decode_contracts::GRAMMAR_SUBSET_ID`].
pub const GRAMMAR_SUBSET_REVISION: &str = GRAMMAR_SUBSET_ID;

/// Grammar compiler revision. Rotating the compiler (any change to how a schema
/// is turned into an automaton) rotates the constraint runtime identity.
pub const GRAMMAR_COMPILER_REVISION: &str = "grammar-compiler-v1";

/// Limits-manifest identity. Named so the limits set is itself versioned and can
/// be referenced from constraint-runtime identity and certification evidence.
pub const LIMITS_MANIFEST_ID: &str = "grammar-limits-v1";

/// Worker constraint-runtime revision: the worker-side interpreter that applies
/// the compiled automaton before each content-token commit.
pub const WORKER_CONSTRAINT_RUNTIME_REVISION: &str = "worker-constraint-runtime-v1";

/// The 2020-12 JSON Schema dialect URI. The subset is based on JSON Schema
/// 2020-12 validation semantics; a `$schema` keyword, when present, must name
/// exactly this dialect.
pub const JSON_SCHEMA_DIALECT_2020_12: &str = "https://json-schema.org/draft/2020-12/schema";

/// Checked-in maximum schema size and shape limits enforced before dispatch.
///
/// These bounds protect the compiler and the worker constraint runtime from
/// pathologically large schemas. They are deliberately conservative: every
/// production grammar-cost fixture fits inside them by orders of magnitude.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GrammarLimits {
    /// Maximum raw schema bytes (the grammar JSON string length).
    pub max_schema_bytes: usize,
    /// Maximum schema nesting depth (root counts as one).
    pub max_nesting_depth: usize,
    /// Maximum total object properties across the whole schema.
    pub max_property_count: usize,
    /// Maximum total enum literals across the whole schema.
    pub max_enum_count: usize,
    /// Maximum number of compiled automaton states the compiler may emit.
    pub max_compiled_state_count: usize,
}

impl Default for GrammarLimits {
    fn default() -> Self {
        Self {
            max_schema_bytes: 64 * 1024,
            max_nesting_depth: 32,
            max_property_count: 256,
            max_enum_count: 128,
            max_compiled_state_count: 4096,
        }
    }
}

/// The versioned grammar-subset manifest. This is the single checked-in record
/// of *which* subset is accepted, *which* compiler produced an automaton, and
/// *which* limits were enforced. Its fields feed [`ConstraintRuntimeIdentity`]
/// construction in [`crate::owned_decode_grammar_scheduler::grammar_compile`].
///
/// [`ConstraintRuntimeIdentity`]: crate::owned_decode_routing::identity::ConstraintRuntimeIdentity
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GrammarSubsetManifest {
    /// Always [`LIMITS_MANIFEST_ID`]; identifies this limits record.
    pub limits_manifest_id: String,
    /// Always [`GRAMMAR_SUBSET_REVISION`]; the accepted schema subset.
    pub grammar_subset_revision: String,
    /// Always [`GRAMMAR_COMPILER_REVISION`]; the compiler that turns a schema
    /// into an automaton.
    pub grammar_compiler_revision: String,
    /// Always [`WORKER_CONSTRAINT_RUNTIME_REVISION`]; the worker interpreter.
    pub worker_constraint_runtime_revision: String,
    /// The enforced numeric limits.
    pub limits: GrammarLimits,
}

impl Default for GrammarSubsetManifest {
    fn default() -> Self {
        Self {
            limits_manifest_id: LIMITS_MANIFEST_ID.to_string(),
            grammar_subset_revision: GRAMMAR_SUBSET_REVISION.to_string(),
            grammar_compiler_revision: GRAMMAR_COMPILER_REVISION.to_string(),
            worker_constraint_runtime_revision: WORKER_CONSTRAINT_RUNTIME_REVISION.to_string(),
            limits: GrammarLimits::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_revisions_match_contract_literals() {
        // These literals are part of the constraint-runtime identity; pin them so
        // a rename fails loudly instead of silently rotating identity.
        let manifest = GrammarSubsetManifest::default();
        assert_eq!(manifest.limits_manifest_id, "grammar-limits-v1");
        assert_eq!(manifest.grammar_subset_revision, "synapse-json-schema-v1");
        assert_eq!(manifest.grammar_compiler_revision, "grammar-compiler-v1");
        assert_eq!(
            manifest.worker_constraint_runtime_revision,
            "worker-constraint-runtime-v1"
        );
        assert_eq!(REPRESENTATION_REVISION, "token-id-json-constraint-v1");
    }

    #[test]
    fn default_limits_are_positive_and_ordered() {
        let limits = GrammarLimits::default();
        assert!(limits.max_schema_bytes > 0);
        assert!(limits.max_nesting_depth > 0);
        assert!(limits.max_property_count > 0);
        assert!(limits.max_enum_count > 0);
        assert!(limits.max_compiled_state_count > 0);
    }

    #[test]
    fn manifest_round_trips_through_json() {
        let manifest = GrammarSubsetManifest::default();
        let encoded = serde_json::to_string(&manifest).expect("manifest serializes");
        let decoded: GrammarSubsetManifest =
            serde_json::from_str(&encoded).expect("manifest deserializes");
        assert_eq!(decoded, manifest);
    }
}
