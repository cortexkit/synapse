//! Wire error bindings for the owned-decode worker protocol.
//!
//! The authoritative binding manifest is `owned-decode-wire-error-bindings-v1`,
//! owned and checked in by the module crate (`crates/synapse-module/`, the
//! deployed surface). This module mirrors the two literal IDs that manifest
//! binds so the worker-supervision fixtures can assert against the exact wire
//! literals without depending on the module crate (which is outside this
//! slice's file fence).
//!
//! The specification is explicit that the symbolic names `existing_deadline_error`
//! and `existing_cancellation_error` are placeholders only and must never appear
//! in emitted responses or passing evidence. [`assert_no_symbolic_placeholders`]
//! is a fixture guard for that rule.

/// Binding-manifest revision mirrored here. Changing either literal requires a
/// new manifest revision and reruns the deadline/cancellation fixture groups.
pub const BINDING_REVISION: &str = "owned-decode-wire-error-bindings-v1";

/// The literal deadline wire ID. This is the existing stable wire literal
/// (`StableErrorCode::DeadlineExceeded` serializes as `deadline_exceeded`).
pub const DEADLINE_ERROR_ID: &str = "deadline_exceeded";

/// The literal cancellation wire ID. It matches the external
/// `finish_reason=cancelled` normalization so a cancelled operation surfaces a
/// single consistent literal on the wire.
pub const CANCELLATION_ERROR_ID: &str = "cancelled";

/// Guard: neither bound literal may be a symbolic placeholder. Fixtures call
/// this to prove no released evidence carries an unresolved symbolic name.
///
/// # Panics
/// Panics if either literal is a known symbolic placeholder.
pub fn assert_no_symbolic_placeholders() {
    const PLACEHOLDERS: &[&str] = &["existing_deadline_error", "existing_cancellation_error"];
    for literal in [DEADLINE_ERROR_ID, CANCELLATION_ERROR_ID] {
        assert!(
            !PLACEHOLDERS.contains(&literal),
            "wire binding '{literal}' is a symbolic placeholder, not a literal wire ID"
        );
        assert!(
            !literal.is_empty(),
            "wire binding literal must not be empty"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bindings_are_literal_not_symbolic() {
        assert_no_symbolic_placeholders();
        assert_eq!(DEADLINE_ERROR_ID, "deadline_exceeded");
        assert_eq!(CANCELLATION_ERROR_ID, "cancelled");
        assert_eq!(BINDING_REVISION, "owned-decode-wire-error-bindings-v1");
    }
}
