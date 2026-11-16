//! Pure certification APIs plus store-backed probe snapshot and write helpers.

pub use synapse_owned_decode::owned_decode_certification::*;

#[path = "owned_decode_probe.rs"]
pub mod probe;
