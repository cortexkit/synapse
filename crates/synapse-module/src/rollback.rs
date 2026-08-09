//! Approval rollback operations.
//!
//! Rollback is deliberately identity-based: an individual action names both
//! the canonical model ID and its decode fingerprint. Emergency rollback uses
//! one fenced transaction to disable every approval.

use crate::store::{ApprovalRow, SynapseStore, SynapseStoreError};

pub(crate) fn disable_exact_approval(
    store: &SynapseStore,
    model_id: &str,
    decode_fingerprint: &str,
    reason: &str,
    updated_at_ms: u64,
) -> Result<ApprovalRow, SynapseStoreError> {
    store.disable_approval(model_id, decode_fingerprint, reason, updated_at_ms)
}

pub(crate) fn disable_all_approvals(
    store: &SynapseStore,
    reason: &str,
    updated_at_ms: u64,
) -> Result<usize, SynapseStoreError> {
    store.emergency_rollback(reason, updated_at_ms)
}
