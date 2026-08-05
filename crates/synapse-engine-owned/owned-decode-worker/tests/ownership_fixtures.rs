//! Ownership-safety fixture battery for `decode-ownership-manifest-v1`.
//!
//! The mandatory `macos-metal` lane runs the real Metal worker under
//! AddressSanitizer with leak detection and records a test and run record for
//! every fault-site ID in [`OwnershipFaultSite`]. These fixtures are the
//! hardware-independent counterpart: they model resident generation state
//! (attention KV-cache, LFM2 convolution rolling cache, constraint-automaton
//! state) in the pure-Rust [`OwnershipLedger`] and prove the four safety
//! properties — no double free, no invalid free, no use-after-free, no leak —
//! at every manifest fault site.

use owned_decode_worker::{
    OwnershipFaultSite, OwnershipLedger, OwnershipViolation, ResidentStateKind,
    OWNERSHIP_MANIFEST_REVISION,
};

/// Allocate the full resident set for one generation.
fn allocate_resident_set(ledger: &mut OwnershipLedger) -> Vec<u64> {
    vec![
        ledger.allocate(ResidentStateKind::AttentionKv),
        ledger.allocate(ResidentStateKind::Lfm2ConvCache),
        ledger.allocate(ResidentStateKind::ConstraintAutomaton),
    ]
}

#[test]
fn fixture_manifest_revision_and_fault_site_coverage() {
    assert_eq!(OWNERSHIP_MANIFEST_REVISION, "decode-ownership-manifest-v1");
    let sites = OwnershipFaultSite::all();
    // Every documented fault site is present with a unique stable ID.
    let ids: Vec<&str> = sites.iter().map(|site| site.id()).collect();
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(ids.len(), sorted.len(), "fault-site IDs must be unique");
    // The manifest covers the FFI-boundary sites explicitly.
    assert!(sites.contains(&OwnershipFaultSite::OwnershipTransfer));
    assert!(sites.contains(&OwnershipFaultSite::Lfm2ConvCacheFfi));
    assert!(sites.contains(&OwnershipFaultSite::PartialInitialization));
}

#[test]
fn fixture_allocation_site_is_safe() {
    let mut ledger = OwnershipLedger::new();
    let resources = allocate_resident_set(&mut ledger);
    assert_eq!(ledger.live_count(), 3);
    for id in resources {
        ledger.release(id);
    }
    ledger.assert_safe(true);
}

#[test]
fn fixture_ownership_transfer_across_ffi_is_safe() {
    let mut ledger = OwnershipLedger::new();
    let kv = ledger.allocate(ResidentStateKind::AttentionKv);
    // Transfer Rust -> ObjC/C and back; the resource stays live, never freed.
    assert!(ledger.transfer(kv), "live resource transfers cleanly");
    ledger.use_resource(kv);
    ledger.release(kv);
    // Transferring a freed resource reports no live owner.
    assert!(!ledger.transfer(kv));
    ledger.assert_safe(true);
}

#[test]
fn fixture_partial_initialization_teardown_is_safe() {
    // A partial initialization where the third allocation fails: teardown must
    // free exactly the two resources that were actually allocated. The failed
    // allocation created no ledger entry, so there is no id to free for it —
    // the never-allocated resource is neither leaked nor freed.
    let mut ledger = OwnershipLedger::new();
    let kv = ledger
        .attempt_allocate(ResidentStateKind::AttentionKv, true)
        .expect("first allocation succeeds");
    let conv = ledger
        .attempt_allocate(ResidentStateKind::Lfm2ConvCache, true)
        .expect("second allocation succeeds");
    // The constraint-automaton allocation fails: no id, nothing to free.
    let automaton = ledger.attempt_allocate(ResidentStateKind::ConstraintAutomaton, false);
    assert!(automaton.is_none(), "the third allocation fails");
    assert_eq!(ledger.failed_allocation_attempts(), 1);
    assert_eq!(
        ledger.live_count(),
        2,
        "only two resources exist to tear down"
    );

    // Teardown frees exactly what was allocated.
    ledger.release(kv);
    ledger.release(conv);
    ledger.assert_safe(true);

    // Negative control: the never-allocated third resource has no id; tearing
    // it down anyway is an invalid free, which the ledger catches. A teardown
    // that wrongly "frees" the failed allocation would not be safe.
    ledger.release(999);
    assert_eq!(ledger.violations(), &[OwnershipViolation::InvalidFree(999)]);
}

#[test]
fn fixture_generation_site_is_safe() {
    let mut ledger = OwnershipLedger::new();
    let resources = allocate_resident_set(&mut ledger);
    // Steady-state use during generation.
    for _ in 0..64 {
        for &id in &resources {
            ledger.use_resource(id);
        }
    }
    for id in resources {
        ledger.release(id);
    }
    ledger.assert_safe(true);
}

#[test]
fn fixture_cancellation_teardown_is_safe() {
    let mut ledger = OwnershipLedger::new();
    let _ = allocate_resident_set(&mut ledger);
    // Cancellation destroys all resident state.
    ledger.destroy_all_resident();
    ledger.assert_safe(true);
}

#[test]
fn fixture_timeout_cleanup_teardown_is_safe() {
    let mut ledger = OwnershipLedger::new();
    let resources = allocate_resident_set(&mut ledger);
    for &id in &resources {
        ledger.use_resource(id);
    }
    // Timeout cleanup destroys all resident state.
    ledger.destroy_all_resident();
    ledger.assert_safe(true);
}

#[test]
fn fixture_unload_site_is_safe() {
    let mut ledger = OwnershipLedger::new();
    let resources = allocate_resident_set(&mut ledger);
    for id in resources {
        ledger.release(id);
    }
    ledger.assert_safe(true);
}

#[test]
fn fixture_shutdown_site_is_safe() {
    let mut ledger = OwnershipLedger::new();
    // Two resident sets (e.g. across a redispatch) both torn down at shutdown.
    let _ = allocate_resident_set(&mut ledger);
    ledger.destroy_all_resident();
    let _ = allocate_resident_set(&mut ledger);
    ledger.destroy_all_resident();
    ledger.assert_safe(true);
}

#[test]
fn fixture_lfm2_conv_cache_ffi_site_is_safe() {
    let mut ledger = OwnershipLedger::new();
    let conv = ledger.allocate(ResidentStateKind::Lfm2ConvCache);
    assert!(ledger.transfer(conv));
    ledger.use_resource(conv);
    ledger.release(conv);
    ledger.assert_safe(true);
}

#[test]
fn fixture_worker_death_destroys_all_resident_state() {
    // Worker death destroys the full resident set; nothing is resumed, nothing
    // leaks, and a later use of the freed state is caught as use-after-free.
    let mut ledger = OwnershipLedger::new();
    let resources = allocate_resident_set(&mut ledger);
    ledger.destroy_all_resident();
    assert_eq!(ledger.live_count(), 0);
    // Negative control: touching freed resident state is a use-after-free.
    ledger.use_resource(resources[0]);
    assert_eq!(ledger.violations().len(), 1);
}

#[test]
fn fixture_every_fault_site_has_a_clean_lifecycle() {
    // For every fault site in the manifest, a correctly handled resident-state
    // lifecycle satisfies all four ownership-safety properties.
    for site in OwnershipFaultSite::all() {
        let mut ledger = OwnershipLedger::new();
        let resources = if *site == OwnershipFaultSite::PartialInitialization {
            // The third allocation fails in this partial scenario, so only two
            // resources ever exist; teardown frees exactly those two.
            let kv = ledger
                .attempt_allocate(ResidentStateKind::AttentionKv, true)
                .expect("first allocation succeeds");
            let conv = ledger
                .attempt_allocate(ResidentStateKind::Lfm2ConvCache, true)
                .expect("second allocation succeeds");
            assert!(
                ledger
                    .attempt_allocate(ResidentStateKind::ConstraintAutomaton, false)
                    .is_none(),
                "the third allocation never happens"
            );
            vec![kv, conv]
        } else {
            allocate_resident_set(&mut ledger)
        };
        match site {
            OwnershipFaultSite::OwnershipTransfer | OwnershipFaultSite::Lfm2ConvCacheFfi => {
                for &id in &resources {
                    assert!(ledger.transfer(id), "{site:?} transfer of live state");
                }
            }
            _ => {}
        }
        // Use during generation, then tear everything down.
        for &id in &resources {
            if ledger.is_live(id) {
                ledger.use_resource(id);
            }
        }
        ledger.destroy_all_resident();
        ledger.assert_safe(true);
    }
}
