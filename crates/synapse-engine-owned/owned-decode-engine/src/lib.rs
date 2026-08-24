//! Production-owned Metal decode engines for Qwen3-0.6B and LFM2-1.2B.
//!
//! This module ports the proven Metal step-decode implementations from
//! `bench/spikes/unified-rt/` into production-owned code under
//! `crates/synapse-engine-owned/owned-decode-engine/`. The spike tree is
//! read-only reference material; this is the production-owned copy.
//!
//! ## Scope
//! - Qwen3-0.6B and LFM2-1.2B in f16 and Q8_0 weight formats
//! - Causal prefill and Metal token stepping (no MPSGraph decode)
//! - K=1 constrained stepping via `token-id-json-constraint-v1`
//! - Q8 execution consumes only the complete ingest-published tensor inventory
//! - No production path loads `bench/spikes/`
//!
//! ## Byte-identity
//! The Metal kernels (`.metal`), Objective-C drivers (`.m`), and FFI bindings
//! are byte-identical to the spike so the pinned fixture batteries reproduce
//! exactly. Direct M5 spike-harness comparisons produce byte-identical token
//! streams for all four lanes (Qwen3 f16, Qwen3 Q8_0, LFM2 f16, LFM2 Q8_0).

#![cfg(target_os = "macos")]
// The decode engines are not yet wired into the production serving path (that
// is the D-009 cutover slice). Until then, the engine types and FFI bindings
// are exercised by the macos-metal CI lane and the parity fixtures, not by
// the embedding-only production code. Allow dead code at the module level so
// clippy does not flag the ported-but-not-yet-consumed engine surface.
#![allow(dead_code)]

mod decode_kernel;
mod json_constraint;
mod lfm2_decode_metal_step;
mod lfm2_decode_model;
mod quant;
mod qwen3_decode_metal_step;
mod qwen3_decode_model;
mod session;

pub use decode_kernel::{top_logits, DecodeKernel, DecodeRuntime, TopLogit};
pub use json_constraint::{DecodeConstraint, JsonConstraint, TokenMask, TokenVocabulary};
pub use lfm2_decode_metal_step::{Lfm2HybridStepCache, Lfm2HybridStepEngine};
pub use lfm2_decode_model::Model as Lfm2DecodeModel;
pub use quant::{Q8_0Tensor, WeightQuantization};

/// Re-export of the runtime precision enum for decode engine consumers.
pub use crate::Precision;
pub use qwen3_decode_metal_step::{
    MetalStepDecoder, MetalStepKvCache, MetalStepKvCache as Qwen3StepCache,
};
pub use qwen3_decode_model::Model as Qwen3DecodeModel;
pub use session::{
    required_kv_evaluation_matrix, select_kv_configuration, Active, ActiveKvSession, Closed,
    ClosedKvSession, DecodeOutput, DepthController, DepthControllerMeasurement, DepthDecision,
    DepthDecisionTelemetry, DepthObservation, DepthRequest, DraftProposal, DraftSource,
    DraftSourceKind, FinishReason, FixedDepthController, KvAllocator, KvAllocatorAccounting,
    KvBlockLease, KvBlockSize, KvConfiguration, KvMatrixCoordinate, KvMatrixMeasurement,
    KvSessionId, NativeMtpDepthController, NativeMtpExecutor, NativeMtpHead, NativeMtpHeadPin,
    NativeMtpRound, NoopTokenTap, OwnedDecodeError, OwnedDecodeResult, OwnedDecodeSession,
    PrefillTelemetry, ProposalExecution, Retained, RetainedKvSession,
    SelectedKvConfigurationTelemetry, SpeculativeDecodeOutput, SpeculativeDecodeTelemetry,
    TokenTap, TokenTapEvent, KV_BLOCK_SIZES, KV_REUSE_BUCKETS,
};

/// Decode lane identity for the owned-metal-decode engine.
pub const DECODE_LANE: &str = "owned-metal-decode";

/// Worker protocol ID for the owned-metal-decode worker.
pub const WORKER_PROTOCOL_ID: &str = "owned-metal-decode-worker-v1";

/// Constraint encoding carried over the worker boundary.
pub const CONSTRAINT_ENCODING_ID: &str = "token-id-json-constraint-v1";

/// Supported decode context buckets. The shippable context manifest
/// (`decode-context-buckets-v1`) starts with `{512,1024,2048}` per family.
pub const SUPPORTED_BUCKETS: &[usize] = &[512, 1024, 2048];

/// Greedy-top-1 selector name. The worker protocol accepts only this mode.
pub const GREEDY_TOP1: &str = "greedy_top1";
