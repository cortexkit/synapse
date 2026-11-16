//! Owned-decode contracts, sidecar normalization, routing, scheduling and pure certification.

#[path = "../owned-decode-certification/mod.rs"]
pub mod owned_decode_certification;
#[path = "../owned-decode-contracts/mod.rs"]
pub mod owned_decode_contracts;
#[path = "../owned-decode-grammar-scheduler/mod.rs"]
pub mod owned_decode_grammar_scheduler;
#[path = "../owned-decode-routing/mod.rs"]
pub mod owned_decode_routing;
#[path = "../owned-decode-sidecar/mod.rs"]
pub mod owned_decode_sidecar;

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}
