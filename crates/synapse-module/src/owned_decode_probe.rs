//! Store-backed probe snapshots and terminal certification writes.

pub use synapse_owned_decode::owned_decode_certification::probe::*;

use crate::owned_decode_routing::error::OwnedDecodeError;
use crate::store::{
    OwnedDecodeCertificationRow, OwnedDecodeMatchInputs, ProbeWriteOutcome, SynapseStore,
    SynapseStoreError,
};
use serde::{Deserialize, Serialize};

/// The complete runtime identity captured once when a probe begins. A terminal
/// write must compare this exact tuple with a fresh resolution before evidence
/// can become matchable.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProbeSnapshot {
    revisioned_machine_profile_hash: String,
    profile_activation_epoch: u64,
    model_id: String,
    decode_fingerprint: String,
    processing_fingerprint: String,
    runtime_config_digest: String,
    constraint_runtime_identities: Vec<String>,
    worker_path_evidence: serde_json::Value,
    evidence_schema_revision: String,
    g_dec_manifest_revision: String,
}

impl ProbeSnapshot {
    pub fn capture(inputs: &OwnedDecodeMatchInputs) -> Result<Self, OwnedDecodeError> {
        if inputs.profile_activation_epoch == 0 {
            return Err(OwnedDecodeError::NotCertified);
        }
        let mut identities = inputs.constraint_runtime_identities.clone();
        identities.sort();
        if identities.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(OwnedDecodeError::ConstraintVersionMismatch);
        }
        Ok(Self {
            revisioned_machine_profile_hash: inputs.revisioned_machine_profile_hash.clone(),
            profile_activation_epoch: inputs.profile_activation_epoch,
            model_id: inputs.model_id.clone(),
            decode_fingerprint: inputs.decode_fingerprint.clone(),
            processing_fingerprint: inputs.processing_fingerprint.clone(),
            runtime_config_digest: inputs.runtime_config_digest.clone(),
            constraint_runtime_identities: identities,
            worker_path_evidence: inputs.worker_path_evidence.clone(),
            evidence_schema_revision: inputs.evidence_schema_revision.clone(),
            g_dec_manifest_revision: inputs.g_dec_manifest_revision.clone(),
        })
    }

    #[must_use]
    pub fn to_match_inputs(&self) -> OwnedDecodeMatchInputs {
        OwnedDecodeMatchInputs {
            revisioned_machine_profile_hash: self.revisioned_machine_profile_hash.clone(),
            profile_activation_epoch: self.profile_activation_epoch,
            model_id: self.model_id.clone(),
            decode_fingerprint: self.decode_fingerprint.clone(),
            processing_fingerprint: self.processing_fingerprint.clone(),
            runtime_config_digest: self.runtime_config_digest.clone(),
            constraint_runtime_identities: self.constraint_runtime_identities.clone(),
            worker_path_evidence: self.worker_path_evidence.clone(),
            evidence_schema_revision: self.evidence_schema_revision.clone(),
            g_dec_manifest_revision: self.g_dec_manifest_revision.clone(),
        }
    }

    #[must_use]
    pub fn profile_activation_epoch(&self) -> u64 {
        self.profile_activation_epoch
    }

    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    #[must_use]
    pub fn decode_fingerprint(&self) -> &str {
        &self.decode_fingerprint
    }
}

/// Perform the fenced terminal probe write. The store compares the immutable
/// start snapshot with the terminal resolution and persisted profile activation epoch; a
/// changed member returns `ProbeStale` and writes no certification row.
pub fn commit_terminal_probe(
    store: &SynapseStore,
    snapshot: &ProbeSnapshot,
    terminal: &ProbeSnapshot,
    row: &OwnedDecodeCertificationRow,
    observed_at_ms: u64,
) -> Result<ProbeWriteOutcome, SynapseStoreError> {
    store.store_owned_decode_cert_row_if_current(
        &snapshot.to_match_inputs(),
        &terminal.to_match_inputs(),
        row,
        observed_at_ms,
    )
}
