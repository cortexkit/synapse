//! Production model-family registry for the owned-metal-decode lane.
//!
//! The module exclusively owns tokenization, prompt templates, special and stop
//! tokens, and detokenization; the worker receives only canonical token IDs and
//! returns only generated token IDs. Each supported family therefore registers
//! the identity revisions of those module-owned processing assets here. The
//! registered revisions feed `processing_fingerprint` (see `identity.rs`) and
//! the per-family stop/special token ID sets are what lane selection and the
//! worker protocol treat as non-committed control candidates.
//!
//! The specification recognizes exactly two production families, `qwen3-0.6b`
//! and `lfm2-1.2b`. Family parsing and serialization recognize both; any other
//! family string is unsupported and fails closed.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::owned_decode_routing::error::OwnedDecodeError;

/// The two production model families. Serialized as their canonical kebab-case
/// catalog strings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Family {
    #[serde(rename = "qwen3-0.6b")]
    Qwen3_0_6b,
    #[serde(rename = "lfm2-1.2b")]
    Lfm2_1_2b,
}

impl Family {
    /// The canonical catalog string for this family.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Qwen3_0_6b => "qwen3-0.6b",
            Self::Lfm2_1_2b => "lfm2-1.2b",
        }
    }

    /// Parse a family from its canonical catalog string. Returns
    /// [`OwnedDecodeError::Unsupported`] for any family the production catalog
    /// does not recognize.
    pub fn parse(value: &str) -> Result<Self, OwnedDecodeError> {
        match value {
            "qwen3-0.6b" => Ok(Self::Qwen3_0_6b),
            "lfm2-1.2b" => Ok(Self::Lfm2_1_2b),
            _ => Err(OwnedDecodeError::Unsupported),
        }
    }

    /// Both production families, in canonical order.
    pub const fn all() -> [Family; 2] {
        [Self::Qwen3_0_6b, Self::Lfm2_1_2b]
    }
}

/// Module-owned processing-asset identity for one family.
///
/// Every field is a revision (or a sanitized digest) of an asset the module
/// owns. Together with the decode fingerprint these determine the
/// `processing_fingerprint`; rotating any of them rotates the processing
/// fingerprint without touching the decode fingerprint.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FamilyRegistration {
    pub family: Family,
    /// SHA-256 digest of the sanitized tokenizer asset.
    pub tokenizer_sanitized_digest: String,
    /// Revision of the prompt-template applied before tokenization.
    pub prompt_template_revision: String,
    /// Revision of the special-token policy (which special IDs are injected).
    pub special_token_policy_revision: String,
    /// Revision of the stop-token policy (which IDs terminate generation).
    pub stop_token_policy_revision: String,
    /// Revision of the detokenizer that renders generated IDs into text.
    pub detokenizer_revision: String,
    /// Special token IDs owned by this family's policy. These are injected by
    /// the template, never emitted as content.
    pub special_token_ids: Vec<u32>,
    /// Stop token IDs treated as non-committed control candidates. Generation
    /// ends when one is selected; it is omitted from generated IDs.
    pub stop_token_ids: Vec<u32>,
}

/// Registry of per-family processing-asset identities.
///
/// Production routing constructs the registry once from [`FamilyRegistry::production`]
/// and treats it as immutable: a family with no registration is unsupported.
#[derive(Clone, Debug, Default)]
pub struct FamilyRegistry {
    registrations: BTreeMap<Family, FamilyRegistration>,
}

impl FamilyRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// The production registry with both recognized families registered.
    pub fn production() -> Self {
        let mut registry = Self::new();
        registry.register(qwen3_registration());
        registry.register(lfm2_registration());
        registry
    }

    /// Register (or replace) a family's processing-asset identity.
    pub fn register(&mut self, registration: FamilyRegistration) {
        self.registrations.insert(registration.family, registration);
    }

    /// Look up a family's registration. Absence means the family is not part of
    /// the production set and routing fails closed with `Unsupported`.
    pub fn get(&self, family: Family) -> Result<&FamilyRegistration, OwnedDecodeError> {
        self.registrations
            .get(&family)
            .ok_or(OwnedDecodeError::Unsupported)
    }

    /// Whether a family has a registration.
    pub fn contains(&self, family: Family) -> bool {
        self.registrations.contains_key(&family)
    }

    /// All registered families, in canonical order.
    pub fn families(&self) -> Vec<Family> {
        self.registrations.keys().copied().collect()
    }
}

/// Qwen3-0.6B production processing-asset identity.
fn qwen3_registration() -> FamilyRegistration {
    FamilyRegistration {
        family: Family::Qwen3_0_6b,
        tokenizer_sanitized_digest: "sha256:qwen3-0.6b-tokenizer-v1".to_string(),
        prompt_template_revision: "qwen3-chat-template-v1".to_string(),
        special_token_policy_revision: "qwen3-special-tokens-v1".to_string(),
        stop_token_policy_revision: "qwen3-stop-tokens-v1".to_string(),
        detokenizer_revision: "qwen3-detokenizer-v1".to_string(),
        // Representative Qwen3 control IDs: im_start / im_end / end-of-text.
        special_token_ids: vec![151643, 151644, 151645],
        stop_token_ids: vec![151645],
    }
}

/// LFM2-1.2B production processing-asset identity.
fn lfm2_registration() -> FamilyRegistration {
    FamilyRegistration {
        family: Family::Lfm2_1_2b,
        tokenizer_sanitized_digest: "sha256:lfm2-1.2b-tokenizer-v1".to_string(),
        prompt_template_revision: "lfm2-chat-template-v1".to_string(),
        special_token_policy_revision: "lfm2-special-tokens-v1".to_string(),
        stop_token_policy_revision: "lfm2-stop-tokens-v1".to_string(),
        detokenizer_revision: "lfm2-detokenizer-v1".to_string(),
        // Representative LFM2 control IDs: bos / eos / pad.
        special_token_ids: vec![1, 2, 0],
        stop_token_ids: vec![2],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_families_round_trip_through_canonical_strings() {
        for family in Family::all() {
            assert_eq!(Family::parse(family.as_str()), Ok(family));
        }
        assert_eq!(Family::Qwen3_0_6b.as_str(), "qwen3-0.6b");
        assert_eq!(Family::Lfm2_1_2b.as_str(), "lfm2-1.2b");
    }

    #[test]
    fn unknown_family_is_unsupported() {
        assert_eq!(Family::parse("llama-3"), Err(OwnedDecodeError::Unsupported));
        assert_eq!(Family::parse(""), Err(OwnedDecodeError::Unsupported));
    }

    #[test]
    fn production_registry_registers_both_families_with_assets() {
        let registry = FamilyRegistry::production();
        for family in Family::all() {
            let registration = registry.get(family).expect("family registered");
            assert_eq!(registration.family, family);
            assert!(!registration.tokenizer_sanitized_digest.is_empty());
            assert!(!registration.prompt_template_revision.is_empty());
            assert!(!registration.special_token_policy_revision.is_empty());
            assert!(!registration.stop_token_policy_revision.is_empty());
            assert!(!registration.detokenizer_revision.is_empty());
            assert!(
                !registration.stop_token_ids.is_empty(),
                "family needs stop tokens"
            );
        }
    }

    #[test]
    fn families_have_distinct_processing_assets() {
        let registry = FamilyRegistry::production();
        let qwen3 = registry.get(Family::Qwen3_0_6b).unwrap();
        let lfm2 = registry.get(Family::Lfm2_1_2b).unwrap();
        // The two families must not share tokenizer or template identity, or a
        // processing fingerprint could not distinguish them.
        assert_ne!(
            qwen3.tokenizer_sanitized_digest,
            lfm2.tokenizer_sanitized_digest
        );
        assert_ne!(
            qwen3.prompt_template_revision,
            lfm2.prompt_template_revision
        );
        assert_ne!(qwen3.detokenizer_revision, lfm2.detokenizer_revision);
    }

    #[test]
    fn unregistered_family_lookup_fails_closed() {
        let registry = FamilyRegistry::new();
        assert_eq!(
            registry.get(Family::Lfm2_1_2b),
            Err(OwnedDecodeError::Unsupported)
        );
        assert!(!registry.contains(Family::Lfm2_1_2b));
    }
}
