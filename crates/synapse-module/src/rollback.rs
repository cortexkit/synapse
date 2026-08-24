//! Approval rollback operations.
//!
//! Existing rollback identifies approvals by model ID and decode fingerprint. The
//! serving catalog APIs below instead identify a catalog fingerprint and distinguish
//! ordinary disablement from emergency revocation. Revocation requests active sessions
//! to terminate at their next committed boundary.

use crate::store::{ApprovalRow, ServingControlOutcome, SynapseStore, SynapseStoreError};

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

/// Ordinary disable fences new catalog admissions and continuations while active
/// sessions retain their reservation until natural completion.
#[allow(dead_code)]
pub(crate) fn disable_serving_catalog(
    store: &SynapseStore,
    catalog_fingerprint: &str,
    reason: &str,
    updated_at_ms: u64,
) -> Result<ServingControlOutcome, SynapseStoreError> {
    store.disable_serving_catalog(catalog_fingerprint, reason, updated_at_ms)
}

/// Emergency revoke fences admissions, invalidates retained states, and requests
/// terminal accounting from active sessions at their next committed boundary.
#[allow(dead_code)]
pub(crate) fn revoke_serving_catalog(
    store: &SynapseStore,
    catalog_fingerprint: &str,
    reason: &str,
    updated_at_ms: u64,
) -> Result<ServingControlOutcome, SynapseStoreError> {
    store.revoke_serving_catalog(catalog_fingerprint, reason, updated_at_ms)
}
