//! One-shot approval migration boundaries.
//!
//! The migration seed is embedded only in the store's administrative migration
//! operation. Serving and certification use SQLite approvals and measured
//! evidence, so neither path can reach the retired source records.

use crate::owned_decode_contracts::WireErrorBindingsManifest;

/// The symbolic wire IDs in the contract are documentation placeholders, not
/// values that may be used by the serving predicate.
pub fn wire_bindings_are_literal(bindings: &WireErrorBindingsManifest) -> bool {
    const SYMBOLIC_DEADLINE: &str = "existing_deadline_error";
    const SYMBOLIC_CANCELLATION: &str = "existing_cancellation_error";
    let literal =
        |id: &str| !id.is_empty() && id != SYMBOLIC_DEADLINE && id != SYMBOLIC_CANCELLATION;
    literal(&bindings.deadline_error_id) && literal(&bindings.cancellation_error_id)
}
