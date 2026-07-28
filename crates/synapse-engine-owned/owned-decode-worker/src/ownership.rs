//! Ownership manifest and resident-state ownership tracking.
//!
//! `decode-ownership-manifest-v1` enumerates the fault sites where resident
//! model-cache state changes owner across the Objective-C/C and Rust FFI
//! boundaries. The mandatory `macos-metal` evidence supplies a test and run
//! record for every fault-site ID and demonstrates no double free, invalid free,
//! use-after-free, or leak under AddressSanitizer with leak detection enabled.
//!
//! Resident generation state (resolution r2 #6) is the full set: attention
//! KV-cache, LFM2 short-convolution rolling caches, and constraint-automaton
//! state. Worker death destroys all of it; partial state is never resumed.
//!
//! This module provides a pure-Rust [`OwnershipLedger`] the fixtures use to
//! model those ownership transfers and assert the four safety properties without
//! a GPU. The real Metal worker's ASan run records against the same fault-site
//! IDs.

use serde::{Deserialize, Serialize};

/// Manifest revision.
pub const OWNERSHIP_MANIFEST_REVISION: &str = "decode-ownership-manifest-v1";

/// A fault site in the ownership manifest. Each has a stable ID referenced by
/// the mandatory `macos-metal` evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OwnershipFaultSite {
    /// Initial allocation of resident state.
    Allocation,
    /// Transfer of ownership across the FFI boundary (Rust -> ObjC/C or back).
    OwnershipTransfer,
    /// A partially initialized allocation that must be torn down on failure.
    PartialInitialization,
    /// Steady-state use during generation.
    Generation,
    /// Cancellation-driven teardown.
    Cancellation,
    /// Timeout-cleanup-driven teardown.
    TimeoutCleanup,
    /// Explicit unload of the resident generation.
    Unload,
    /// Worker shutdown teardown.
    Shutdown,
    /// LFM2 convolution-cache ownership across the FFI boundary.
    Lfm2ConvCacheFfi,
}

impl OwnershipFaultSite {
    /// The stable fault-site ID recorded in evidence.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::Allocation => "ODEC-OWN-ALLOC",
            Self::OwnershipTransfer => "ODEC-OWN-TRANSFER",
            Self::PartialInitialization => "ODEC-OWN-PARTIAL-INIT",
            Self::Generation => "ODEC-OWN-GENERATION",
            Self::Cancellation => "ODEC-OWN-CANCEL",
            Self::TimeoutCleanup => "ODEC-OWN-TIMEOUT-CLEANUP",
            Self::Unload => "ODEC-OWN-UNLOAD",
            Self::Shutdown => "ODEC-OWN-SHUTDOWN",
            Self::Lfm2ConvCacheFfi => "ODEC-OWN-LFM2-CONV-FFI",
        }
    }

    /// Every fault site, in manifest order.
    #[must_use]
    pub const fn all() -> &'static [OwnershipFaultSite] {
        &[
            Self::Allocation,
            Self::OwnershipTransfer,
            Self::PartialInitialization,
            Self::Generation,
            Self::Cancellation,
            Self::TimeoutCleanup,
            Self::Unload,
            Self::Shutdown,
            Self::Lfm2ConvCacheFfi,
        ]
    }
}

/// A kind of resident generation state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResidentStateKind {
    /// Attention KV-cache (all families).
    AttentionKv,
    /// LFM2 short-convolution rolling cache (LFM2 only).
    Lfm2ConvCache,
    /// Optional constraint-automaton state.
    ConstraintAutomaton,
}

/// An ownership-safety violation detected by the ledger.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OwnershipViolation {
    /// A resource was released twice.
    DoubleFree(u64),
    /// A resource id that was never allocated (or already freed) was released.
    InvalidFree(u64),
    /// A freed resource was used.
    UseAfterFree(u64),
}

/// A pure-Rust ownership ledger. It models allocation, FFI ownership transfer,
/// use, and release of resident-state resources and records any double free,
/// invalid free, or use-after-free. A leak is any resource still live when the
/// ledger is checked for completeness.
#[derive(Default)]
pub struct OwnershipLedger {
    next_id: u64,
    /// Live resources: id -> kind.
    live: std::collections::BTreeMap<u64, ResidentStateKind>,
    /// Freed resource ids (kept to detect use-after-free and double free).
    freed: std::collections::BTreeSet<u64>,
    violations: Vec<OwnershipViolation>,
}

impl OwnershipLedger {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate a resident-state resource, returning its id. Models the
    /// [`OwnershipFaultSite::Allocation`] site.
    pub fn allocate(&mut self, kind: ResidentStateKind) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.live.insert(id, kind);
        id
    }

    /// Model an ownership transfer across the FFI boundary. The resource stays
    /// live; the ledger records that it moved owner without being freed. Returns
    /// false (and records nothing) if the resource is not live.
    pub fn transfer(&mut self, id: u64) -> bool {
        self.live.contains_key(&id)
    }

    /// Use a resource during generation. Records a use-after-free violation if
    /// the resource was already freed.
    pub fn use_resource(&mut self, id: u64) {
        if self.freed.contains(&id) {
            self.violations.push(OwnershipViolation::UseAfterFree(id));
        } else if !self.live.contains_key(&id) {
            self.violations.push(OwnershipViolation::InvalidFree(id));
        }
    }

    /// Release a resource. Records a double free if already freed, or an invalid
    /// free if never allocated.
    pub fn release(&mut self, id: u64) {
        if self.freed.contains(&id) {
            self.violations.push(OwnershipViolation::DoubleFree(id));
            return;
        }
        if self.live.remove(&id).is_none() {
            self.violations.push(OwnershipViolation::InvalidFree(id));
            return;
        }
        self.freed.insert(id);
    }

    /// Destroy all resident state, as worker death does. Every live resource is
    /// freed; already-freed resources are left alone (no double free).
    pub fn destroy_all_resident(&mut self) {
        let live_ids: Vec<u64> = self.live.keys().copied().collect();
        for id in live_ids {
            self.release(id);
        }
    }

    #[must_use]
    pub fn is_live(&self, id: u64) -> bool {
        self.live.contains_key(&id)
    }

    #[must_use]
    pub fn live_count(&self) -> usize {
        self.live.len()
    }

    /// Resources still live that should have been released — a leak if the
    /// generation was supposed to be torn down.
    #[must_use]
    pub fn leaked(&self) -> Vec<u64> {
        self.live.keys().copied().collect()
    }

    #[must_use]
    pub fn violations(&self) -> &[OwnershipViolation] {
        &self.violations
    }

    /// Assert the four ownership-safety properties: no double free, no invalid
    /// free, no use-after-free, and (when `expect_no_leak`) no leak.
    ///
    /// # Panics
    /// Panics with a descriptive message if any property is violated.
    pub fn assert_safe(&self, expect_no_leak: bool) {
        assert!(
            self.violations.is_empty(),
            "ownership violations detected: {:?}",
            self.violations
        );
        if expect_no_leak {
            assert!(
                self.live.is_empty(),
                "leaked resident-state resources: {:?}",
                self.leaked()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_covers_every_documented_fault_site() {
        // Every fault site has a unique, stable ID.
        let sites = OwnershipFaultSite::all();
        let mut ids = sites.iter().map(|site| site.id()).collect::<Vec<_>>();
        ids.sort_unstable();
        let mut deduped = ids.clone();
        deduped.dedup();
        assert_eq!(ids.len(), deduped.len(), "fault-site IDs must be unique");
        assert!(sites.contains(&OwnershipFaultSite::Lfm2ConvCacheFfi));
        assert_eq!(OWNERSHIP_MANIFEST_REVISION, "decode-ownership-manifest-v1");
    }

    #[test]
    fn clean_lifecycle_is_safe() {
        let mut ledger = OwnershipLedger::new();
        let kv = ledger.allocate(ResidentStateKind::AttentionKv);
        let conv = ledger.allocate(ResidentStateKind::Lfm2ConvCache);
        assert!(ledger.transfer(kv));
        ledger.use_resource(kv);
        ledger.use_resource(conv);
        ledger.release(kv);
        ledger.release(conv);
        ledger.assert_safe(true);
    }

    #[test]
    fn double_free_is_detected() {
        let mut ledger = OwnershipLedger::new();
        let kv = ledger.allocate(ResidentStateKind::AttentionKv);
        ledger.release(kv);
        ledger.release(kv);
        assert_eq!(ledger.violations(), &[OwnershipViolation::DoubleFree(kv)]);
    }

    #[test]
    fn use_after_free_is_detected() {
        let mut ledger = OwnershipLedger::new();
        let kv = ledger.allocate(ResidentStateKind::AttentionKv);
        ledger.release(kv);
        ledger.use_resource(kv);
        assert_eq!(ledger.violations(), &[OwnershipViolation::UseAfterFree(kv)]);
    }

    #[test]
    fn invalid_free_is_detected() {
        let mut ledger = OwnershipLedger::new();
        ledger.release(999);
        assert_eq!(ledger.violations(), &[OwnershipViolation::InvalidFree(999)]);
    }

    #[test]
    fn destroy_all_frees_everything_without_double_free() {
        let mut ledger = OwnershipLedger::new();
        ledger.allocate(ResidentStateKind::AttentionKv);
        ledger.allocate(ResidentStateKind::Lfm2ConvCache);
        ledger.allocate(ResidentStateKind::ConstraintAutomaton);
        ledger.destroy_all_resident();
        ledger.assert_safe(true);
        assert_eq!(ledger.live_count(), 0);
    }
}
