//! Q8 first-load derivation (ingest) orchestration.
//!
//! For `weight_quant=q8_0` the first load lacking a valid cached artifact runs
//! one atomic ingest transaction keyed by `(source_manifest_digest,
//! quantizer_revision)`. The transaction quantizes into a private temporary
//! object, computes its derived digest, writes lineage metadata, and atomically
//! publishes object plus metadata. Cancellation or failure before publication
//! leaves no loadable entry.
//!
//! Trust follows the artifact identity contract:
//! - With a registered expected digest, ingest verifies equality before marking
//!   the object trusted; a mismatch marks it `artifact_poisoned`.
//! - Without a registered expected digest, ingest publishes only an untrusted
//!   object; loads fail closed with `owned_decode_not_certified` until the
//!   expected digest is registered and the object is successfully verified.
//!   Registration never retroactively trusts bytes without verification.
//!
//! Later loads reuse a trusted object without requantization. Eviction removes
//! the loadable object while retaining lineage so rederivation reproduces the
//! registered digest. Rotating `quantizer_revision` creates a distinct key and
//! never overwrites the prior object.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::owned_decode_routing::error::OwnedDecodeError;

/// Transaction key: `(source_manifest_digest, quantizer_revision)`. Rotating
/// either component creates a distinct key, object, and digest.
pub type IngestKey = (String, String);

/// Trust state of a published Q8 artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrustState {
    /// Verified against a registered expected digest; may be certified/served.
    Trusted,
    /// Published without a registered expected digest; cannot be certified or
    /// served until verified.
    Untrusted,
    /// Digest mismatch or post-publication corruption; can never be certified
    /// or served.
    Poisoned,
}

/// A published Q8 artifact entry with its lineage and trust metadata.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Q8ArtifactEntry {
    pub source_manifest_digest: String,
    pub quantizer_revision: String,
    pub derived_digest: String,
    pub format: String,
    pub lineage: String,
    pub trust_state: TrustState,
    pub reproducible: bool,
    pub derivable: bool,
    pub evictable: bool,
    /// Whether the loadable object bytes are present. Eviction clears this while
    /// retaining the lineage needed to rederive.
    pub loadable: bool,
}

/// Whether a caller started the transaction or joined one already in flight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransactionRole {
    /// This caller owns the transaction and must publish or abort it.
    Owner,
    /// A transaction for this key is already in flight; this caller joins/waits
    /// and must not publish a competing object.
    Joined,
}

/// Outcome of a successful load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadOutcome {
    pub entry: Q8ArtifactEntry,
    /// True when an existing trusted object was reused without requantization.
    pub reused: bool,
}

/// Registry of Q8 artifacts, expected digests, and in-flight transactions.
#[derive(Clone, Debug, Default)]
pub struct Q8IngestRegistry {
    entries: BTreeMap<IngestKey, Q8ArtifactEntry>,
    expected_digests: BTreeMap<IngestKey, String>,
    in_flight: BTreeSet<IngestKey>,
}

impl Q8IngestRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    fn key(source_manifest_digest: &str, quantizer_revision: &str) -> IngestKey {
        (
            source_manifest_digest.to_string(),
            quantizer_revision.to_string(),
        )
    }

    /// Register the expected derived digest for an exact source manifest and
    /// quantizer revision. Registration alone never trusts existing bytes; the
    /// object must still be verified against this digest.
    pub fn register_expected_digest(
        &mut self,
        source_manifest_digest: &str,
        quantizer_revision: &str,
        expected_digest: &str,
    ) {
        self.expected_digests.insert(
            Self::key(source_manifest_digest, quantizer_revision),
            expected_digest.to_string(),
        );
    }

    /// Look up a published entry without loading or ingesting.
    pub fn entry(
        &self,
        source_manifest_digest: &str,
        quantizer_revision: &str,
    ) -> Option<&Q8ArtifactEntry> {
        self.entries
            .get(&Self::key(source_manifest_digest, quantizer_revision))
    }

    /// Whether a transaction is currently in flight for this key.
    pub fn is_in_flight(&self, source_manifest_digest: &str, quantizer_revision: &str) -> bool {
        self.in_flight
            .contains(&Self::key(source_manifest_digest, quantizer_revision))
    }

    /// Begin an ingest transaction. Returns [`TransactionRole::Owner`] for the
    /// first caller (marking the key in flight) and [`TransactionRole::Joined`]
    /// for concurrent callers, who then wait on the owner's publication rather
    /// than publishing a competing object.
    pub fn begin_ingest(
        &mut self,
        source_manifest_digest: &str,
        quantizer_revision: &str,
    ) -> TransactionRole {
        let key = Self::key(source_manifest_digest, quantizer_revision);
        if self.in_flight.contains(&key) {
            TransactionRole::Joined
        } else {
            self.in_flight.insert(key);
            TransactionRole::Owner
        }
    }

    /// Atomically publish a derived object and its metadata, clearing the
    /// in-flight marker. Object and metadata become visible together.
    pub fn publish(&mut self, entry: Q8ArtifactEntry) {
        let key = Self::key(&entry.source_manifest_digest, &entry.quantizer_revision);
        self.in_flight.remove(&key);
        self.entries.insert(key, entry);
    }

    /// Abort an in-flight transaction before publication. Leaves no loadable
    /// entry, satisfying "cancellation or process failure before publication
    /// leaves no loadable entry".
    pub fn abort(&mut self, source_manifest_digest: &str, quantizer_revision: &str) {
        self.in_flight
            .remove(&Self::key(source_manifest_digest, quantizer_revision));
    }

    /// Settle a derived digest against the registered expected digest: trusted
    /// on a match, poisoned on a mismatch, untrusted when none is registered.
    fn settle(&self, key: &IngestKey, derived_digest: &str) -> TrustState {
        match self.expected_digests.get(key) {
            Some(expected) if expected == derived_digest => TrustState::Trusted,
            Some(_) => TrustState::Poisoned,
            None => TrustState::Untrusted,
        }
    }

    /// Load an existing entry or run one ingest transaction.
    ///
    /// `quantize` derives the Q8 digest from the source bytes (the private
    /// temporary object's content digest). Reuse of an existing trusted object
    /// never calls it.
    pub fn load_or_ingest<F>(
        &mut self,
        source_manifest_digest: &str,
        quantizer_revision: &str,
        format: &str,
        source_bytes: &[u8],
        quantize: F,
    ) -> Result<LoadOutcome, OwnedDecodeError>
    where
        F: FnOnce(&[u8]) -> String,
    {
        let key = Self::key(source_manifest_digest, quantizer_revision);

        // Existing entry path: reuse, verify-on-demand, or fail closed.
        if let Some(existing) = self.entries.get(&key).cloned() {
            return match existing.trust_state {
                TrustState::Trusted if existing.loadable => Ok(LoadOutcome {
                    entry: existing,
                    reused: true,
                }),
                TrustState::Poisoned => Err(OwnedDecodeError::ArtifactPoisoned),
                // Untrusted (or trusted-but-evicted): verify against a now-registered
                // expected digest if one exists; otherwise fail closed.
                _ => self.verify_existing(&key, existing),
            };
        }

        // No entry: run one ingest transaction as the owner.
        debug_assert_eq!(
            self.begin_ingest(source_manifest_digest, quantizer_revision),
            TransactionRole::Owner
        );
        let derived_digest = quantize(source_bytes);
        let trust_state = self.settle(&key, &derived_digest);
        let entry = Q8ArtifactEntry {
            source_manifest_digest: source_manifest_digest.to_string(),
            quantizer_revision: quantizer_revision.to_string(),
            derived_digest,
            format: format.to_string(),
            lineage: format!("derived from {source_manifest_digest} via {quantizer_revision}"),
            trust_state,
            reproducible: true,
            derivable: true,
            evictable: true,
            // Poisoned objects are published as metadata but never loadable.
            loadable: trust_state != TrustState::Poisoned,
        };
        self.publish(entry.clone());

        match trust_state {
            TrustState::Trusted => Ok(LoadOutcome {
                entry,
                reused: false,
            }),
            TrustState::Poisoned => Err(OwnedDecodeError::ArtifactPoisoned),
            // Untrusted publishes but cannot be served until verified.
            TrustState::Untrusted => Err(OwnedDecodeError::NotCertified),
        }
    }

    /// Verify an already-published (untrusted or evicted) entry against a
    /// registered expected digest, promoting to trusted on a match or poisoning
    /// on a mismatch. With no registered digest the load fails closed.
    fn verify_existing(
        &mut self,
        key: &IngestKey,
        mut entry: Q8ArtifactEntry,
    ) -> Result<LoadOutcome, OwnedDecodeError> {
        match self.expected_digests.get(key) {
            Some(expected) if expected == &entry.derived_digest => {
                entry.trust_state = TrustState::Trusted;
                entry.loadable = true;
                self.entries.insert(key.clone(), entry.clone());
                Ok(LoadOutcome {
                    entry,
                    reused: true,
                })
            }
            Some(_) => {
                entry.trust_state = TrustState::Poisoned;
                entry.loadable = false;
                self.entries.insert(key.clone(), entry.clone());
                Err(OwnedDecodeError::ArtifactPoisoned)
            }
            None => Err(OwnedDecodeError::NotCertified),
        }
    }

    /// Evict the loadable object while retaining lineage so it can be rederived.
    pub fn evict(&mut self, source_manifest_digest: &str, quantizer_revision: &str) {
        let key = Self::key(source_manifest_digest, quantizer_revision);
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.loadable = false;
        }
    }

    /// Rederive an evicted (or absent) object and republish it. The rederived
    /// digest must reproduce the registered expected digest; a mismatch poisons
    /// the entry. Returns the republished entry.
    pub fn rederive<F>(
        &mut self,
        source_manifest_digest: &str,
        quantizer_revision: &str,
        source_bytes: &[u8],
        quantize: F,
    ) -> Result<Q8ArtifactEntry, OwnedDecodeError>
    where
        F: FnOnce(&[u8]) -> String,
    {
        let key = Self::key(source_manifest_digest, quantizer_revision);
        let derived_digest = quantize(source_bytes);

        let mut entry = self
            .entries
            .get(&key)
            .cloned()
            .unwrap_or_else(|| Q8ArtifactEntry {
                source_manifest_digest: source_manifest_digest.to_string(),
                quantizer_revision: quantizer_revision.to_string(),
                derived_digest: derived_digest.clone(),
                format: "q8_0".to_string(),
                lineage: format!(
                    "rederived from {source_manifest_digest} via {quantizer_revision}"
                ),
                trust_state: TrustState::Untrusted,
                reproducible: true,
                derivable: true,
                evictable: true,
                loadable: false,
            });

        // Rederivation must reproduce the registered digest.
        if let Some(expected) = self.expected_digests.get(&key) {
            if expected != &derived_digest {
                entry.trust_state = TrustState::Poisoned;
                entry.loadable = false;
                self.entries.insert(key, entry.clone());
                return Err(OwnedDecodeError::ArtifactPoisoned);
            }
            entry.trust_state = TrustState::Trusted;
        }
        entry.derived_digest = derived_digest;
        entry.loadable = entry.trust_state == TrustState::Trusted;
        self.entries.insert(key, entry.clone());
        Ok(entry)
    }

    /// Record post-publication corruption or digest mismatch, marking the entry
    /// poisoned and unloadable.
    pub fn mark_corrupted(&mut self, source_manifest_digest: &str, quantizer_revision: &str) {
        let key = Self::key(source_manifest_digest, quantizer_revision);
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.trust_state = TrustState::Poisoned;
            entry.loadable = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = "sha256:source-manifest";
    const QUANT: &str = "quant-v1";
    const DERIVED: &str = "sha256:q8-derived";

    fn quantizer(_bytes: &[u8]) -> String {
        DERIVED.to_string()
    }

    #[test]
    fn ingest_with_matching_expected_digest_is_trusted_and_served() {
        let mut registry = Q8IngestRegistry::new();
        registry.register_expected_digest(SOURCE, QUANT, DERIVED);

        let outcome = registry
            .load_or_ingest(SOURCE, QUANT, "q8_0", b"weights", quantizer)
            .expect("trusted ingest");
        assert_eq!(outcome.entry.trust_state, TrustState::Trusted);
        assert!(outcome.entry.loadable);
        assert!(!outcome.reused);
    }

    #[test]
    fn ingest_without_expected_digest_is_untrusted_and_fails_closed() {
        let mut registry = Q8IngestRegistry::new();
        let err = registry
            .load_or_ingest(SOURCE, QUANT, "q8_0", b"weights", quantizer)
            .expect_err("untrusted cannot be served");
        assert_eq!(err, OwnedDecodeError::NotCertified);
        // The object is published as untrusted metadata but is not loadable for service.
        let entry = registry.entry(SOURCE, QUANT).expect("published");
        assert_eq!(entry.trust_state, TrustState::Untrusted);
    }

    #[test]
    fn registering_expected_digest_later_verifies_existing_untrusted_bytes() {
        let mut registry = Q8IngestRegistry::new();
        // First ingest publishes untrusted and fails closed.
        assert_eq!(
            registry.load_or_ingest(SOURCE, QUANT, "q8_0", b"weights", quantizer),
            Err(OwnedDecodeError::NotCertified)
        );
        // Registering the expected digest and reloading verifies the SAME bytes.
        registry.register_expected_digest(SOURCE, QUANT, DERIVED);
        let outcome = registry
            .load_or_ingest(SOURCE, QUANT, "q8_0", b"weights", quantizer)
            .expect("verified on demand");
        assert_eq!(outcome.entry.trust_state, TrustState::Trusted);
        assert!(outcome.reused, "no requantization on verification");
    }

    #[test]
    fn digest_mismatch_poisons_and_cannot_be_served() {
        let mut registry = Q8IngestRegistry::new();
        registry.register_expected_digest(SOURCE, QUANT, "sha256:other-expected");
        let err = registry
            .load_or_ingest(SOURCE, QUANT, "q8_0", b"weights", quantizer)
            .expect_err("mismatch poisons");
        assert_eq!(err, OwnedDecodeError::ArtifactPoisoned);
        let entry = registry.entry(SOURCE, QUANT).unwrap();
        assert_eq!(entry.trust_state, TrustState::Poisoned);
        assert!(!entry.loadable);
        // Subsequent loads keep returning poisoned.
        assert_eq!(
            registry.load_or_ingest(SOURCE, QUANT, "q8_0", b"weights", quantizer),
            Err(OwnedDecodeError::ArtifactPoisoned)
        );
    }

    #[test]
    fn trusted_object_is_reused_without_requantization() {
        let mut registry = Q8IngestRegistry::new();
        registry.register_expected_digest(SOURCE, QUANT, DERIVED);
        registry
            .load_or_ingest(SOURCE, QUANT, "q8_0", b"weights", quantizer)
            .unwrap();

        // Second load reuses; the quantizer closure panics if invoked.
        let outcome = registry
            .load_or_ingest(SOURCE, QUANT, "q8_0", b"weights", |_b| {
                panic!("must not requantize on reuse")
            })
            .expect("reuse");
        assert!(outcome.reused);
        assert_eq!(outcome.entry.trust_state, TrustState::Trusted);
    }

    #[test]
    fn concurrent_loads_join_and_publish_one_object() {
        let mut registry = Q8IngestRegistry::new();
        registry.register_expected_digest(SOURCE, QUANT, DERIVED);

        // First caller owns the transaction.
        assert_eq!(registry.begin_ingest(SOURCE, QUANT), TransactionRole::Owner);
        assert!(registry.is_in_flight(SOURCE, QUANT));
        // Concurrent callers join rather than publish a competing object.
        assert_eq!(
            registry.begin_ingest(SOURCE, QUANT),
            TransactionRole::Joined
        );
        assert_eq!(
            registry.begin_ingest(SOURCE, QUANT),
            TransactionRole::Joined
        );

        // Only the owner publishes, exactly once.
        registry.publish(Q8ArtifactEntry {
            source_manifest_digest: SOURCE.to_string(),
            quantizer_revision: QUANT.to_string(),
            derived_digest: DERIVED.to_string(),
            format: "q8_0".to_string(),
            lineage: "test".to_string(),
            trust_state: TrustState::Trusted,
            reproducible: true,
            derivable: true,
            evictable: true,
            loadable: true,
        });
        assert!(!registry.is_in_flight(SOURCE, QUANT));
        assert_eq!(
            registry.entry(SOURCE, QUANT).unwrap().trust_state,
            TrustState::Trusted
        );
    }

    #[test]
    fn abort_before_publication_leaves_no_loadable_entry() {
        let mut registry = Q8IngestRegistry::new();
        assert_eq!(registry.begin_ingest(SOURCE, QUANT), TransactionRole::Owner);
        registry.abort(SOURCE, QUANT);
        assert!(!registry.is_in_flight(SOURCE, QUANT));
        assert!(registry.entry(SOURCE, QUANT).is_none());
    }

    #[test]
    fn eviction_retains_lineage_and_rederivation_reproduces_digest() {
        let mut registry = Q8IngestRegistry::new();
        registry.register_expected_digest(SOURCE, QUANT, DERIVED);
        registry
            .load_or_ingest(SOURCE, QUANT, "q8_0", b"weights", quantizer)
            .unwrap();

        registry.evict(SOURCE, QUANT);
        assert!(!registry.entry(SOURCE, QUANT).unwrap().loadable);

        let rederived = registry
            .rederive(SOURCE, QUANT, b"weights", quantizer)
            .expect("rederive");
        assert_eq!(rederived.derived_digest, DERIVED);
        assert!(rederived.loadable);
        assert_eq!(rederived.trust_state, TrustState::Trusted);
    }

    #[test]
    fn rederivation_mismatch_poisons() {
        let mut registry = Q8IngestRegistry::new();
        registry.register_expected_digest(SOURCE, QUANT, DERIVED);
        registry
            .load_or_ingest(SOURCE, QUANT, "q8_0", b"weights", quantizer)
            .unwrap();
        registry.evict(SOURCE, QUANT);

        let err = registry
            .rederive(SOURCE, QUANT, b"weights", |_| "sha256:drifted".to_string())
            .expect_err("rederive mismatch poisons");
        assert_eq!(err, OwnedDecodeError::ArtifactPoisoned);
    }

    #[test]
    fn quantizer_rotation_creates_a_distinct_key() {
        let mut registry = Q8IngestRegistry::new();
        registry.register_expected_digest(SOURCE, QUANT, DERIVED);
        registry.register_expected_digest(SOURCE, "quant-v2", "sha256:q8-derived-v2");

        registry
            .load_or_ingest(SOURCE, QUANT, "q8_0", b"weights", quantizer)
            .unwrap();
        let rotated = registry
            .load_or_ingest(SOURCE, "quant-v2", "q8_0", b"weights", |_| {
                "sha256:q8-derived-v2".to_string()
            })
            .expect("rotated ingest");

        // Both objects coexist; rotation never overwrites the prior object.
        assert_eq!(
            registry.entry(SOURCE, QUANT).unwrap().derived_digest,
            DERIVED
        );
        assert_eq!(rotated.entry.derived_digest, "sha256:q8-derived-v2");
        assert_eq!(
            registry
                .entry(SOURCE, "quant-v2")
                .unwrap()
                .quantizer_revision,
            "quant-v2"
        );
    }

    #[test]
    fn post_publication_corruption_poisons() {
        let mut registry = Q8IngestRegistry::new();
        registry.register_expected_digest(SOURCE, QUANT, DERIVED);
        registry
            .load_or_ingest(SOURCE, QUANT, "q8_0", b"weights", quantizer)
            .unwrap();

        registry.mark_corrupted(SOURCE, QUANT);
        assert_eq!(
            registry.load_or_ingest(SOURCE, QUANT, "q8_0", b"weights", quantizer),
            Err(OwnedDecodeError::ArtifactPoisoned)
        );
    }

    #[test]
    fn q8_artifact_entry_rejects_unknown_field() {
        // fail-closed posture: an unknown field in a Q8 artifact entry is
        // rejected at parse time rather than silently dropped.
        let json = serde_json::json!({
            "source_manifest_digest": "sha256:src",
            "quantizer_revision": "quant-v1",
            "derived_digest": "sha256:derived",
            "format": "q8_0",
            "lineage": "line",
            "trust_state": "trusted",
            "reproducible": true,
            "derivable": true,
            "evictable": false,
            "loadable": true,
            "unknown_field": "should be rejected",
        });
        assert!(serde_json::from_value::<Q8ArtifactEntry>(json).is_err());
    }
}
