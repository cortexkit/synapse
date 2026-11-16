//! Re-exports of the shared launched-job outcome taxonomy.
//!
//! The taxonomy lives in synapse-core because the module publishes outcomes and
//! the worker consumes their additive provenance without a routing dependency.

pub use synapse_core::{SidecarBankEffect, SidecarOutcome, SidecarOutcomeEvents, SpanClass};
