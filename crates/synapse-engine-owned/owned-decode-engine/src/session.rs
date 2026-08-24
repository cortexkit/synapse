//! Owned serial and speculative decode with per-session KV ownership.
//!
//! The policy layer is independent of a model family: `DecodeKernel` owns the
//! target-model cache, while `DraftSource` only proposes tokens. Serial decode
//! and native MTP proposals therefore share one greedy acceptance loop without
//! requiring a separate draft model.

use std::collections::BTreeSet;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result as AnyResult;
use thiserror::Error;

use super::decode_kernel::{top_logits, DecodeKernel};

/// The registered block sizes permitted for per-session KV reuse.
pub const KV_BLOCK_SIZES: [KvBlockSize; 3] = [
    KvBlockSize::Tokens256,
    KvBlockSize::Tokens512,
    KvBlockSize::Tokens1024,
];

/// The reused-prefix buckets every KV-selection run must measure.
pub const KV_REUSE_BUCKETS: [usize; 3] = [4096, 8192, 16_384];

/// Errors whose callers need to distinguish from a generic decode failure.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum OwnedDecodeError {
    #[error("invalid KV configuration: {0}")]
    InvalidKvConfiguration(String),
    #[error(
        "invalid KV alignment at token position {position}; required alignment is {required_alignment}"
    )]
    InvalidKvAlignment {
        position: usize,
        required_alignment: usize,
    },
    #[error("KV allocator is exhausted: requested {requested_blocks} blocks, {available_blocks} available")]
    KvAllocatorExhausted {
        requested_blocks: usize,
        available_blocks: usize,
    },
    #[error("KV block {block_id} was released more than once")]
    KvDoubleFree { block_id: u64 },
    #[error("KV session {session_id} was used after close")]
    KvSessionUseAfterClose { session_id: u64 },
    #[error("KV allocator lock was poisoned")]
    KvAllocatorPoisoned,
    #[error("invalid fixed KV evaluation matrix: {0}")]
    InvalidKvEvaluationMatrix(String),
    #[error("no alignment-valid KV candidate met the retained-memory overhead limit")]
    NoEligibleKvConfiguration,
    #[error("depth-controller measurement is missing")]
    MissingDepthControllerMeasurement,
    #[error(
        "depth-controller measurement does not match the resident artifact or machine (expected {expected_machine}/{expected_artifact}, got {actual_machine}/{actual_artifact})"
    )]
    MismatchedDepthControllerMeasurement {
        expected_machine: String,
        expected_artifact: String,
        actual_machine: String,
        actual_artifact: String,
    },
    #[error("wave-1 speculative decode requires the pinned native MTP head")]
    UnsupportedDraftSource,
    #[error("depth controller selected invalid depth {selected}; maximum is {maximum}")]
    InvalidDepthSelection { selected: usize, maximum: usize },
    #[error("draft source returned no tokens for a positive depth")]
    EmptyDraftProposal,
    #[error("draft source returned {actual} tokens for a depth-{requested} request")]
    OversizedDraftProposal { actual: usize, requested: usize },
    #[error(
        "native MTP work must be chained with its backbone step without a host synchronization"
    )]
    NativeMtpCommandBufferNotChained,
    #[error("decode session is already stopped at token {stop_token}")]
    DecodeSessionStopped { stop_token: u32 },
    #[error("decode kernel: {0}")]
    Kernel(String),
}

/// Result type used by this engine-owned policy layer.
pub type OwnedDecodeResult<T> = Result<T, OwnedDecodeError>;

fn kernel_result<T>(result: AnyResult<T>) -> OwnedDecodeResult<T> {
    result.map_err(|error| OwnedDecodeError::Kernel(error.to_string()))
}

/// A supported coarse KV block grain.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum KvBlockSize {
    Tokens256,
    Tokens512,
    Tokens1024,
}

impl KvBlockSize {
    #[must_use]
    pub const fn tokens(self) -> usize {
        match self {
            Self::Tokens256 => 256,
            Self::Tokens512 => 512,
            Self::Tokens1024 => 1024,
        }
    }

    /// Converts a manifest block size while rejecting unregistered values.
    pub fn try_from_tokens(tokens: usize) -> OwnedDecodeResult<Self> {
        match tokens {
            256 => Ok(Self::Tokens256),
            512 => Ok(Self::Tokens512),
            1024 => Ok(Self::Tokens1024),
            _ => Err(OwnedDecodeError::InvalidKvConfiguration(format!(
                "block size {tokens} is not one of {{256,512,1024}}"
            ))),
        }
    }
}

/// KV parameters shared by allocation, snapshot, and continuation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvConfiguration {
    pub block_size: KvBlockSize,
    /// Model-defined retained-state grain. Snapshot boundaries must satisfy the
    /// least common multiple of this value and the selected block size.
    pub recurrent_state_grain: usize,
}

impl KvConfiguration {
    pub fn new(block_size: KvBlockSize, recurrent_state_grain: usize) -> OwnedDecodeResult<Self> {
        if recurrent_state_grain == 0 {
            return Err(OwnedDecodeError::InvalidKvConfiguration(
                "recurrent-state grain must be non-zero".to_string(),
            ));
        }
        let configuration = Self {
            block_size,
            recurrent_state_grain,
        };
        let _ = configuration.alignment()?;
        Ok(configuration)
    }

    /// The one boundary at which both block and recurrent state are valid.
    pub fn alignment(self) -> OwnedDecodeResult<usize> {
        lcm(self.block_size.tokens(), self.recurrent_state_grain).ok_or_else(|| {
            OwnedDecodeError::InvalidKvConfiguration(format!(
                "LCM of {} and {} overflows usize",
                self.block_size.tokens(),
                self.recurrent_state_grain
            ))
        })
    }

    /// Rejects rather than truncating, copying, or re-prefilling an unaligned
    /// snapshot/reuse boundary.
    pub fn validate_boundary(self, position: usize) -> OwnedDecodeResult<()> {
        let alignment = self.alignment()?;
        if position % alignment != 0 {
            return Err(OwnedDecodeError::InvalidKvAlignment {
                position,
                required_alignment: alignment,
            });
        }
        Ok(())
    }
}

fn lcm(left: usize, right: usize) -> Option<usize> {
    let divisor = gcd(left, right);
    left.checked_div(divisor)?.checked_mul(right)
}

const fn gcd(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

#[derive(Debug)]
struct KvAllocatorState {
    capacity: usize,
    next_block_id: u64,
    next_session_id: u64,
    free_block_ids: BTreeSet<u64>,
    live_block_ids: BTreeSet<u64>,
}

/// A cloneable handle to the session-local KV block allocator.
#[derive(Clone, Debug)]
pub struct KvAllocator {
    state: Arc<Mutex<KvAllocatorState>>,
}

/// Observable allocator accounting used by close/reclamation checks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvAllocatorAccounting {
    pub capacity_blocks: usize,
    pub allocated_blocks: usize,
    pub available_blocks: usize,
}

impl KvAllocator {
    #[must_use]
    pub fn new(capacity_blocks: usize) -> Self {
        Self {
            state: Arc::new(Mutex::new(KvAllocatorState {
                capacity: capacity_blocks,
                next_block_id: 0,
                next_session_id: 0,
                free_block_ids: BTreeSet::new(),
                live_block_ids: BTreeSet::new(),
            })),
        }
    }

    pub fn accounting(&self) -> OwnedDecodeResult<KvAllocatorAccounting> {
        let state = self
            .state
            .lock()
            .map_err(|_| OwnedDecodeError::KvAllocatorPoisoned)?;
        Ok(KvAllocatorAccounting {
            capacity_blocks: state.capacity,
            allocated_blocks: state.live_block_ids.len(),
            available_blocks: state.capacity - state.live_block_ids.len(),
        })
    }

    /// Acquires one non-cloneable lease. Releasing a lease twice returns the
    /// typed runtime fault instead of corrupting allocator accounting.
    pub fn acquire_block(&self) -> OwnedDecodeResult<KvBlockLease> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| OwnedDecodeError::KvAllocatorPoisoned)?;
        let available = state.capacity - state.live_block_ids.len();
        if available == 0 {
            return Err(OwnedDecodeError::KvAllocatorExhausted {
                requested_blocks: 1,
                available_blocks: 0,
            });
        }
        let block_id = state.free_block_ids.pop_first().unwrap_or_else(|| {
            let block_id = state.next_block_id;
            state.next_block_id += 1;
            block_id
        });
        let inserted = state.live_block_ids.insert(block_id);
        debug_assert!(inserted, "free and live KV block sets must not overlap");
        Ok(KvBlockLease {
            allocator: self.clone(),
            block_id,
            released: false,
        })
    }

    fn release_block(&self, block_id: u64) -> OwnedDecodeResult<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| OwnedDecodeError::KvAllocatorPoisoned)?;
        if !state.live_block_ids.remove(&block_id) {
            return Err(OwnedDecodeError::KvDoubleFree { block_id });
        }
        let inserted = state.free_block_ids.insert(block_id);
        debug_assert!(inserted, "live KV block cannot already be free");
        Ok(())
    }

    /// Opens a typestate-owned session table. The allocator remains shared only
    /// for accounting and block acquisition; no other session can reference the
    /// returned table's mutable leases.
    pub fn open_session(
        &self,
        configuration: KvConfiguration,
        context_ceiling: usize,
    ) -> OwnedDecodeResult<ActiveKvSession> {
        if context_ceiling == 0 {
            return Err(OwnedDecodeError::InvalidKvConfiguration(
                "context ceiling must be positive".to_string(),
            ));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| OwnedDecodeError::KvAllocatorPoisoned)?;
        let session_id = KvSessionId(state.next_session_id);
        state.next_session_id += 1;
        drop(state);
        Ok(KvBlockTable {
            session_id,
            configuration,
            context_ceiling,
            position: 0,
            reusable_prefix_tokens: 0,
            blocks: Vec::new(),
            allocator: self.clone(),
            state: PhantomData,
        })
    }
}

/// An owned block reservation. It is intentionally not `Clone`, so a table
/// transition can move block ownership but cannot duplicate it.
///
/// ```compile_fail
/// use synapse_engine_owned::owned_decode_engine::KvAllocator;
///
/// let allocator = KvAllocator::new(1);
/// let block = allocator.acquire_block().unwrap();
/// let duplicate_owner = block;
/// let _still_owned_here = block.block_id();
/// # drop(duplicate_owner);
/// ```
#[derive(Debug)]
pub struct KvBlockLease {
    allocator: KvAllocator,
    block_id: u64,
    released: bool,
}

impl KvBlockLease {
    #[must_use]
    pub const fn block_id(&self) -> u64 {
        self.block_id
    }

    pub fn release(&mut self) -> OwnedDecodeResult<()> {
        if self.released {
            return Err(OwnedDecodeError::KvDoubleFree {
                block_id: self.block_id,
            });
        }
        self.allocator.release_block(self.block_id)?;
        self.released = true;
        Ok(())
    }
}

impl Drop for KvBlockLease {
    fn drop(&mut self) {
        if !self.released {
            // Explicit close reports errors to callers. Drop is the safety net
            // for an interrupted request, where preserving allocator integrity
            // is more important than reporting a secondary cleanup failure.
            let _ = self.release();
        }
    }
}

/// Stable identity for allocator-owned KV session tables.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct KvSessionId(pub u64);

/// Active session typestate marker.
#[derive(Debug)]
pub struct Active;
/// Retained snapshot typestate marker.
#[derive(Debug)]
pub struct Retained;
/// Closed session typestate marker.
#[derive(Debug)]
pub struct Closed;

/// A per-session KV block table whose state owns all currently-live leases.
///
/// Successful lifecycle transitions consume `self`, moving the table's leases
/// into the successor. A closed table has no operation that can successfully
/// prefill, snapshot, or continue, which prevents use-after-close and
/// use-after-move in ordinary Rust code.
///
/// ```compile_fail
/// use synapse_engine_owned::owned_decode_engine::{
///     KvAllocator, KvBlockSize, KvConfiguration,
/// };
///
/// let allocator = KvAllocator::new(2);
/// let config = KvConfiguration::new(KvBlockSize::Tokens256, 1).unwrap();
/// let mut active = allocator.open_session(config, 512).unwrap();
/// active.cold_prefill_to(256, 1).unwrap();
/// let retained = active.snapshot().unwrap();
/// let _use_after_move = active.position();
/// # drop(retained);
/// ```
#[derive(Debug)]
pub struct KvBlockTable<State> {
    session_id: KvSessionId,
    configuration: KvConfiguration,
    context_ceiling: usize,
    position: usize,
    reusable_prefix_tokens: usize,
    blocks: Vec<KvBlockLease>,
    allocator: KvAllocator,
    state: PhantomData<State>,
}

/// Active per-session KV state.
pub type ActiveKvSession = KvBlockTable<Active>;
/// Retained, aligned KV state suitable only for continuation or close.
pub type RetainedKvSession = KvBlockTable<Retained>;
/// Closed KV state. It holds no live allocator block.
pub type ClosedKvSession = KvBlockTable<Closed>;

impl<State> KvBlockTable<State> {
    #[must_use]
    pub const fn session_id(&self) -> KvSessionId {
        self.session_id
    }

    #[must_use]
    pub const fn configuration(&self) -> KvConfiguration {
        self.configuration
    }

    #[must_use]
    pub const fn position(&self) -> usize {
        self.position
    }

    #[must_use]
    pub fn allocated_blocks(&self) -> usize {
        self.blocks.len()
    }

    fn into_state<Next>(self) -> KvBlockTable<Next> {
        let Self {
            session_id,
            configuration,
            context_ceiling,
            position,
            reusable_prefix_tokens,
            blocks,
            allocator,
            state: _,
        } = self;
        KvBlockTable {
            session_id,
            configuration,
            context_ceiling,
            position,
            reusable_prefix_tokens,
            blocks,
            allocator,
            state: PhantomData,
        }
    }

    fn release_all(&mut self) -> OwnedDecodeResult<()> {
        for lease in &mut self.blocks {
            lease.release()?;
        }
        self.blocks.clear();
        Ok(())
    }

    /// Releases every owned block. The returned closed typestate has no method
    /// that can allocate, prefill, snapshot, or continue.
    pub fn close(mut self) -> OwnedDecodeResult<ClosedKvSession> {
        self.release_all()?;
        Ok(self.into_state())
    }
}

impl KvBlockTable<Closed> {
    /// Rejects a dynamic continuation attempt after close. Typestate prevents
    /// normal callers from reaching this state with an active table; this guard
    /// makes a supervisor-side stale handle a typed runtime fault as well.
    pub fn continue_session(self) -> OwnedDecodeResult<ActiveKvSession> {
        Err(OwnedDecodeError::KvSessionUseAfterClose {
            session_id: self.session_id.0,
        })
    }
}

impl KvBlockTable<Active> {
    /// Records a cold prefill. Only newly covered KV blocks are acquired.
    pub fn cold_prefill_to(
        &mut self,
        position: usize,
        prefill_kernel_dispatches: usize,
    ) -> OwnedDecodeResult<PrefillTelemetry> {
        self.ensure_position(position)?;
        self.reusable_prefix_tokens = 0;
        Ok(PrefillTelemetry {
            reused: false,
            reused_tokens: 0,
            reused_blocks: 0,
            prefill_kernel_dispatches,
            reused_prefill_kernel_dispatches: 0,
        })
    }

    /// Converts an aligned active table into a retained continuation boundary.
    /// The move preserves exclusive ownership of the exact same leases.
    pub fn snapshot(self) -> OwnedDecodeResult<RetainedKvSession> {
        self.configuration.validate_boundary(self.position)?;
        Ok(self.into_state())
    }

    fn ensure_position(&mut self, position: usize) -> OwnedDecodeResult<()> {
        if position > self.context_ceiling {
            return Err(OwnedDecodeError::InvalidKvConfiguration(format!(
                "position {position} exceeds context ceiling {}",
                self.context_ceiling
            )));
        }
        let required_blocks = position.div_ceil(self.configuration.block_size.tokens());
        let additional_blocks = required_blocks.saturating_sub(self.blocks.len());
        if additional_blocks > 0 {
            let accounting = self.allocator.accounting()?;
            if additional_blocks > accounting.available_blocks {
                return Err(OwnedDecodeError::KvAllocatorExhausted {
                    requested_blocks: additional_blocks,
                    available_blocks: accounting.available_blocks,
                });
            }
            for _ in 0..additional_blocks {
                self.blocks.push(self.allocator.acquire_block()?);
            }
        }
        self.position = position;
        Ok(())
    }
}

impl KvBlockTable<Retained> {
    /// Moves retained blocks back into an active session. Reuse is strictly
    /// within this session because the snapshot, table, and block leases move
    /// together; this API never accepts another session's snapshot.
    #[must_use]
    pub fn continue_session(mut self) -> ActiveKvSession {
        self.reusable_prefix_tokens = self.position;
        self.into_state()
    }
}

impl KvBlockTable<Active> {
    /// Extends a continued session. The retained prefix is counted as reused
    /// work and therefore has exactly zero prefill kernel dispatches.
    pub fn warm_prefill_to(
        &mut self,
        position: usize,
        new_prefill_kernel_dispatches: usize,
    ) -> OwnedDecodeResult<PrefillTelemetry> {
        let reused_tokens = self.reusable_prefix_tokens;
        if reused_tokens == 0 {
            return Err(OwnedDecodeError::InvalidKvConfiguration(
                "warm prefill requires an aligned retained session".to_string(),
            ));
        }
        if position < self.position {
            return Err(OwnedDecodeError::InvalidKvConfiguration(format!(
                "warm prefill position {position} precedes retained position {}",
                self.position
            )));
        }
        if position == self.position && new_prefill_kernel_dispatches != 0 {
            return Err(OwnedDecodeError::InvalidKvConfiguration(
                "a fully reused prefill range cannot dispatch a prefill kernel".to_string(),
            ));
        }
        self.ensure_position(position)?;
        let reused_blocks = reused_tokens / self.configuration.block_size.tokens();
        self.reusable_prefix_tokens = 0;
        Ok(PrefillTelemetry {
            reused: true,
            reused_tokens,
            reused_blocks,
            prefill_kernel_dispatches: new_prefill_kernel_dispatches,
            reused_prefill_kernel_dispatches: 0,
        })
    }
}

/// Work counters and reuse facts emitted by cold/warm prefill.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefillTelemetry {
    pub reused: bool,
    pub reused_tokens: usize,
    pub reused_blocks: usize,
    /// Dispatches for newly processed tokens, never for the reused range.
    pub prefill_kernel_dispatches: usize,
    /// Stored explicitly so callers can verify that reused tokens incurred no
    /// prefill dispatches instead of inferring that fact from another counter.
    pub reused_prefill_kernel_dispatches: usize,
}

/// One mandatory point in the fixed KV evaluation matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvMatrixCoordinate {
    pub block_size: KvBlockSize,
    pub reused_prefix_tokens: usize,
}

/// Returns all nine pre-registered KV evaluation coordinates.
#[must_use]
pub fn required_kv_evaluation_matrix() -> Vec<KvMatrixCoordinate> {
    let mut coordinates = Vec::with_capacity(KV_BLOCK_SIZES.len() * KV_REUSE_BUCKETS.len());
    for block_size in KV_BLOCK_SIZES {
        for reused_prefix_tokens in KV_REUSE_BUCKETS {
            coordinates.push(KvMatrixCoordinate {
                block_size,
                reused_prefix_tokens,
            });
        }
    }
    coordinates
}

/// Measured facts for one KV-matrix candidate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvMatrixMeasurement {
    pub coordinate: KvMatrixCoordinate,
    pub recurrent_state_grain: usize,
    pub theoretical_minimum_retained_bytes: u64,
    pub retained_bytes: u64,
    pub warm_ttft: Duration,
}

/// The selected configuration and the facts needed for serving telemetry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SelectedKvConfigurationTelemetry {
    pub block_size: KvBlockSize,
    pub reused_prefix_bucket: usize,
    pub recurrent_state_grain: usize,
    pub warm_ttft: Duration,
    pub retained_bytes: u64,
    pub theoretical_minimum_retained_bytes: u64,
}

/// Executes the fixed matrix selection rule.
///
/// Each registered coordinate must appear exactly once. Alignment-invalid and
/// over-budget records are retained as measurement evidence but discarded from
/// selection. Among eligible records the smallest warm TTFT wins; an exact tie
/// chooses the larger KV block.
pub fn select_kv_configuration(
    measurements: &[KvMatrixMeasurement],
) -> OwnedDecodeResult<SelectedKvConfigurationTelemetry> {
    let expected = required_kv_evaluation_matrix();
    if measurements.len() != expected.len() {
        return Err(OwnedDecodeError::InvalidKvEvaluationMatrix(format!(
            "expected {} measurements, got {}",
            expected.len(),
            measurements.len()
        )));
    }
    let mut actual = BTreeSet::new();
    for measurement in measurements {
        actual.insert((
            measurement.coordinate.block_size,
            measurement.coordinate.reused_prefix_tokens,
        ));
    }
    let required = expected
        .iter()
        .map(|coordinate| (coordinate.block_size, coordinate.reused_prefix_tokens))
        .collect::<BTreeSet<_>>();
    if actual != required {
        return Err(OwnedDecodeError::InvalidKvEvaluationMatrix(
            "measurements must contain each registered block-size/prefix-bucket pair exactly once"
                .to_string(),
        ));
    }

    let mut selected: Option<&KvMatrixMeasurement> = None;
    for measurement in measurements {
        let Ok(configuration) = KvConfiguration::new(
            measurement.coordinate.block_size,
            measurement.recurrent_state_grain,
        ) else {
            continue;
        };
        if configuration
            .validate_boundary(measurement.coordinate.reused_prefix_tokens)
            .is_err()
            || measurement.theoretical_minimum_retained_bytes == 0
            || !within_retained_memory_overhead(measurement)
        {
            continue;
        }
        let replace = selected.is_none_or(|current| {
            measurement.warm_ttft < current.warm_ttft
                || (measurement.warm_ttft == current.warm_ttft
                    && measurement.coordinate.block_size > current.coordinate.block_size)
        });
        if replace {
            selected = Some(measurement);
        }
    }
    let selected = selected.ok_or(OwnedDecodeError::NoEligibleKvConfiguration)?;
    Ok(SelectedKvConfigurationTelemetry {
        block_size: selected.coordinate.block_size,
        reused_prefix_bucket: selected.coordinate.reused_prefix_tokens,
        recurrent_state_grain: selected.recurrent_state_grain,
        warm_ttft: selected.warm_ttft,
        retained_bytes: selected.retained_bytes,
        theoretical_minimum_retained_bytes: selected.theoretical_minimum_retained_bytes,
    })
}

fn within_retained_memory_overhead(measurement: &KvMatrixMeasurement) -> bool {
    // Multiply in u128 to avoid allowing an overflowed 10% boundary.
    u128::from(measurement.retained_bytes) * 10
        <= u128::from(measurement.theoretical_minimum_retained_bytes) * 11
}

/// Identifies whether a draft source is permitted in the wave-1 serving path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DraftSourceKind {
    PinnedNativeMtp,
    Experimental,
}

/// How proposal work was encoded. The native MTP implementation is required to
/// report a single chained Metal command buffer and zero added host waits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProposalExecution {
    NoHeadWork,
    MetalCommandBufferChained { extra_host_synchronizations: usize },
}

/// A draft span and its execution evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DraftProposal {
    pub tokens: Vec<u32>,
    pub execution: ProposalExecution,
}

/// Drafter-agnostic seam. The wave-1 decode path admits only
/// `DraftSourceKind::PinnedNativeMtp`; an experimental source can exist behind
/// this seam but is rejected before it can consume serving resources.
pub trait DraftSource {
    fn kind(&self) -> DraftSourceKind;
    fn propose(&mut self, context: &[u32], depth: usize) -> OwnedDecodeResult<DraftProposal>;
}

/// The artifact-bound identity of a native MTP head.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeMtpHeadPin {
    pub catalog_fingerprint: String,
    pub head_revision: String,
}

impl NativeMtpHeadPin {
    #[must_use]
    pub fn new(catalog_fingerprint: impl Into<String>, head_revision: impl Into<String>) -> Self {
        Self {
            catalog_fingerprint: catalog_fingerprint.into(),
            head_revision: head_revision.into(),
        }
    }
}

/// Execution evidence returned by one native MTP round.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeMtpRound {
    pub tokens: Vec<u32>,
    pub execution: ProposalExecution,
}

impl NativeMtpRound {
    /// Builds the only valid wave-1 execution shape: head forward and backbone
    /// verification chained in one command buffer with no intermediate host wait.
    #[must_use]
    pub fn chained(tokens: Vec<u32>) -> Self {
        Self {
            tokens,
            execution: ProposalExecution::MetalCommandBufferChained {
                extra_host_synchronizations: 0,
            },
        }
    }

    /// Preserves executor evidence for validation by the wave-1 acceptance
    /// loop. A non-chained value is rejected before a proposal is verified.
    #[must_use]
    pub fn with_execution(tokens: Vec<u32>, execution: ProposalExecution) -> Self {
        Self { tokens, execution }
    }
}

/// Native bridge used by the pinned MTP source.
///
/// Implementations encode head forward and the matching backbone verification
/// in one Metal command buffer. It returns after the command buffer's regular
/// proposal readback, not after a second host synchronization between stages.
pub trait NativeMtpExecutor {
    fn encode_chained_round(&mut self, context: &[u32], depth: usize) -> AnyResult<NativeMtpRound>;
}

impl<F> NativeMtpExecutor for F
where
    F: FnMut(&[u32], usize) -> AnyResult<NativeMtpRound>,
{
    fn encode_chained_round(&mut self, context: &[u32], depth: usize) -> AnyResult<NativeMtpRound> {
        self(context, depth)
    }
}

/// The concrete `DraftSource` for the initial serving policy: a native MTP
/// head associated with the resident base model. Other sources are rejected
/// before speculative serving work begins.
#[derive(Debug)]
pub struct NativeMtpHead<E> {
    pin: NativeMtpHeadPin,
    executor: E,
}

impl<E> NativeMtpHead<E> {
    #[must_use]
    pub fn new(pin: NativeMtpHeadPin, executor: E) -> Self {
        Self { pin, executor }
    }

    #[must_use]
    pub fn pin(&self) -> &NativeMtpHeadPin {
        &self.pin
    }
}

impl<E: NativeMtpExecutor> DraftSource for NativeMtpHead<E> {
    fn kind(&self) -> DraftSourceKind {
        DraftSourceKind::PinnedNativeMtp
    }

    fn propose(&mut self, context: &[u32], depth: usize) -> OwnedDecodeResult<DraftProposal> {
        if depth == 0 {
            return Ok(DraftProposal {
                tokens: Vec::new(),
                execution: ProposalExecution::NoHeadWork,
            });
        }
        let round = kernel_result(self.executor.encode_chained_round(context, depth))?;
        Ok(DraftProposal {
            tokens: round.tokens,
            execution: round.execution,
        })
    }
}

/// Inputs available to a depth controller before a speculative round.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DepthRequest {
    pub context_tokens: usize,
    pub remaining_tokens: usize,
    pub maximum_depth: usize,
}

/// Controller decision recorded before each round.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthDecision {
    pub proposed_depth: usize,
    pub reason: &'static str,
    pub expected_committed_tokens: f64,
    pub estimated_round_cost: Duration,
}

/// Facts fed back after every serial or speculative round.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DepthObservation {
    pub proposed_depth: usize,
    pub accepted_depth: usize,
    pub verified_tokens: usize,
}

/// Per-round policy seam. It makes a depth decision observable and lets tests
/// prove that a zero-depth decision does not invoke MTP-head work.
pub trait DepthController {
    fn decide(&mut self, request: DepthRequest) -> DepthDecision;
    fn observe(&mut self, observation: DepthObservation);
}

/// Deterministic controller useful for configured fixed-depth service tests.
#[derive(Clone, Copy, Debug)]
pub struct FixedDepthController {
    depth: usize,
}

impl FixedDepthController {
    #[must_use]
    pub const fn new(depth: usize) -> Self {
        Self { depth }
    }
}

impl DepthController for FixedDepthController {
    fn decide(&mut self, request: DepthRequest) -> DepthDecision {
        let proposed_depth = self
            .depth
            .min(request.maximum_depth)
            .min(request.remaining_tokens);
        DepthDecision {
            proposed_depth,
            reason: "fixed_depth",
            expected_committed_tokens: 1.0,
            estimated_round_cost: Duration::ZERO,
        }
    }

    fn observe(&mut self, _observation: DepthObservation) {}
}

/// A native-MTP cost measurement for the target M5 machine and complete
/// resident model artifact. Controllers use this measured command-buffer cost
/// rather than estimates from documentation or another host.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DepthControllerMeasurement {
    pub machine_profile_hash: String,
    pub catalog_fingerprint: String,
    pub native_mtp_head_forward: Duration,
    pub backbone_step: Duration,
}

/// Adaptive controller using the registered matching measurement as its initial
/// cost model and token-domain acceptance observations as it serves rounds.
#[derive(Clone, Debug)]
pub struct NativeMtpDepthController {
    maximum_depth: usize,
    measurement: DepthControllerMeasurement,
    acceptance_probability: Vec<f64>,
}

impl NativeMtpDepthController {
    pub fn new(
        maximum_depth: usize,
        expected_machine_profile_hash: impl Into<String>,
        expected_catalog_fingerprint: impl Into<String>,
        measurement: Option<DepthControllerMeasurement>,
    ) -> OwnedDecodeResult<Self> {
        if maximum_depth == 0 {
            return Err(OwnedDecodeError::InvalidKvConfiguration(
                "native MTP controller maximum depth must be positive".to_string(),
            ));
        }
        let expected_machine_profile_hash = expected_machine_profile_hash.into();
        let expected_catalog_fingerprint = expected_catalog_fingerprint.into();
        let measurement = measurement.ok_or(OwnedDecodeError::MissingDepthControllerMeasurement)?;
        if measurement.machine_profile_hash != expected_machine_profile_hash
            || measurement.catalog_fingerprint != expected_catalog_fingerprint
        {
            return Err(OwnedDecodeError::MismatchedDepthControllerMeasurement {
                expected_machine: expected_machine_profile_hash,
                expected_artifact: expected_catalog_fingerprint,
                actual_machine: measurement.machine_profile_hash,
                actual_artifact: measurement.catalog_fingerprint,
            });
        }
        Ok(Self {
            maximum_depth,
            measurement,
            // Optimistic but bounded initial prior. Real accepted/rejected rows
            // overwrite it with the same per-token EMA used by the controller.
            acceptance_probability: vec![0.6; maximum_depth],
        })
    }

    fn expected_committed_tokens(&self, depth: usize) -> f64 {
        let mut expected_committed = 1.0;
        let mut accepted_run = 1.0;
        for probability in self.acceptance_probability.iter().take(depth) {
            accepted_run *= *probability;
            expected_committed += accepted_run;
        }
        expected_committed
    }

    fn score(&self, depth: usize) -> (f64, Duration) {
        if depth == 0 {
            return (1.0, self.measurement.backbone_step);
        }
        let expected_committed = self.expected_committed_tokens(depth);
        let head_cost = self
            .measurement
            .native_mtp_head_forward
            .saturating_mul(depth as u32);
        let cost = self.measurement.backbone_step.saturating_add(head_cost);
        (
            expected_committed / cost.as_secs_f64().max(f64::MIN_POSITIVE),
            cost,
        )
    }
}

impl DepthController for NativeMtpDepthController {
    fn decide(&mut self, request: DepthRequest) -> DepthDecision {
        let maximum_depth = self
            .maximum_depth
            .min(request.maximum_depth)
            .min(request.remaining_tokens);
        let mut selected = 0;
        let mut selected_score = self.score(0).0;
        for depth in 1..=maximum_depth {
            let score = self.score(depth).0;
            if score > selected_score {
                selected = depth;
                selected_score = score;
            }
        }
        let (_, estimated_round_cost) = self.score(selected);
        DepthDecision {
            proposed_depth: selected,
            reason: "registered_m5_cost_model",
            expected_committed_tokens: self.expected_committed_tokens(selected),
            estimated_round_cost,
        }
    }

    fn observe(&mut self, observation: DepthObservation) {
        let observed = observation
            .proposed_depth
            .min(self.acceptance_probability.len());
        for (index, probability) in self
            .acceptance_probability
            .iter_mut()
            .take(observed)
            .enumerate()
        {
            let accepted = f64::from((index < observation.accepted_depth) as u8);
            *probability = *probability * 0.92 + accepted * 0.08;
            if index >= observation.accepted_depth {
                break;
            }
        }
    }
}

/// Read-only record delivered after acceptance and before token emission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenTapEvent {
    pub generated_index: usize,
    pub token_id: u32,
}

/// Read-only post-acceptance/pre-emission observer. It receives token data by
/// value and no cache or output collection reference, so it cannot pause,
/// splice, rewrite, or address KV through this API.
pub trait TokenTap {
    fn observe(&mut self, event: TokenTapEvent);
}

impl<F> TokenTap for F
where
    F: FnMut(TokenTapEvent),
{
    fn observe(&mut self, event: TokenTapEvent) {
        self(event);
    }
}

/// A tap that preserves the exact untapped path.
#[derive(Default)]
pub struct NoopTokenTap;

impl TokenTap for NoopTokenTap {
    fn observe(&mut self, _event: TokenTapEvent) {}
}

/// Completion status for one generated output window.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReason {
    Length,
    StopToken(u32),
}

/// Generated token IDs and their stable byte representation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodeOutput {
    pub tokens: Vec<u32>,
    /// Little-endian token-ID bytes. Worker framing/tokenization may encode
    /// these IDs differently, but a tap cannot affect this engine-level output.
    pub emitted_token_bytes: Vec<u8>,
    pub finish_reason: FinishReason,
}

impl DecodeOutput {
    fn from_tokens(tokens: Vec<u32>, finish_reason: FinishReason) -> Self {
        let emitted_token_bytes = tokens
            .iter()
            .flat_map(|token| token.to_le_bytes())
            .collect();
        Self {
            tokens,
            emitted_token_bytes,
            finish_reason,
        }
    }
}

/// Decision-level telemetry required to diagnose native MTP rounds.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DepthDecisionTelemetry {
    pub proposed_depth: usize,
    pub accepted_depth: usize,
    pub verification_work: usize,
    pub draft_source_invoked: bool,
    pub command_buffer_chained: bool,
    pub extra_host_synchronizations: usize,
    pub controller_reason: &'static str,
}

/// Aggregate speculative work and controller-decision telemetry.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SpeculativeDecodeTelemetry {
    pub proposal_calls: usize,
    pub proposed_tokens: usize,
    pub accepted_tokens: usize,
    pub verified_tokens: usize,
    pub rejection_count: usize,
    pub controller_decisions: Vec<DepthDecisionTelemetry>,
}

impl SpeculativeDecodeTelemetry {
    #[must_use]
    pub fn acceptance_rate(&self) -> f64 {
        self.accepted_tokens as f64 / self.proposed_tokens.max(1) as f64
    }
}

/// Result of speculative decode, keeping output and telemetry inseparable.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeculativeDecodeOutput {
    pub output: DecodeOutput,
    pub telemetry: SpeculativeDecodeTelemetry,
}

/// An owned target-model session. The kernel cache and the pending exact greedy
/// token are private, so serial and speculative paths share one fidelity source.
pub struct OwnedDecodeSession<'a, K: DecodeKernel> {
    kernel: &'a mut K,
    cache: K::Cache,
    sequence: Vec<u32>,
    generated: Vec<u32>,
    pending_greedy: u32,
    stopped_at: Option<u32>,
}

impl<'a, K: DecodeKernel> OwnedDecodeSession<'a, K> {
    /// Causal prefill against the owned target model.
    pub fn prefill(kernel: &'a mut K, prompt: &[u32]) -> OwnedDecodeResult<Self> {
        if prompt.is_empty() {
            return Err(OwnedDecodeError::Kernel(
                "decode prompt must contain at least one token".to_string(),
            ));
        }
        let (cache, pending_greedy) = kernel_result(kernel.prefill(prompt))?;
        if kernel.cache_position(&cache) != prompt.len() {
            return Err(OwnedDecodeError::Kernel(
                "prefill cache position does not match prompt length".to_string(),
            ));
        }
        Ok(Self {
            kernel,
            cache,
            sequence: prompt.to_vec(),
            generated: Vec::new(),
            pending_greedy,
            stopped_at: None,
        })
    }

    #[must_use]
    pub fn sequence(&self) -> &[u32] {
        &self.sequence
    }

    #[must_use]
    pub fn generated(&self) -> &[u32] {
        &self.generated
    }

    #[must_use]
    pub fn cache_position(&self) -> usize {
        self.kernel.cache_position(&self.cache)
    }

    /// Certified owned serial greedy decode. It is intentionally the same
    /// pending-greedy/cache state that speculative acceptance references.
    pub fn decode_serial(
        &mut self,
        max_tokens: usize,
        stop_tokens: &BTreeSet<u32>,
        tap: &mut dyn TokenTap,
    ) -> OwnedDecodeResult<DecodeOutput> {
        self.ensure_not_stopped()?;
        let start = self.generated.len();
        let mut finish_reason = FinishReason::Length;
        for _ in 0..max_tokens {
            let token = self.pending_greedy;
            self.advance_and_commit(token, tap)?;
            if stop_tokens.contains(&token) {
                self.stopped_at = Some(token);
                finish_reason = FinishReason::StopToken(token);
                break;
            }
        }
        Ok(DecodeOutput::from_tokens(
            self.generated[start..].to_vec(),
            finish_reason,
        ))
    }

    /// Greedy speculative acceptance. Positive depths call only the pinned
    /// native MTP source; depth zero is an owned serial round and performs no
    /// head work. At a rejection the target argmax is committed after a logical
    /// cache rewind, preserving token-for-token serial equivalence.
    pub fn decode_speculative<D: DraftSource, C: DepthController>(
        &mut self,
        draft_source: &mut D,
        controller: &mut C,
        max_tokens: usize,
        maximum_depth: usize,
        stop_tokens: &BTreeSet<u32>,
        tap: &mut dyn TokenTap,
    ) -> OwnedDecodeResult<SpeculativeDecodeOutput> {
        self.ensure_not_stopped()?;
        if draft_source.kind() != DraftSourceKind::PinnedNativeMtp {
            return Err(OwnedDecodeError::UnsupportedDraftSource);
        }
        let start = self.generated.len();
        let mut produced = 0;
        let mut telemetry = SpeculativeDecodeTelemetry::default();
        let mut finish_reason = FinishReason::Length;

        while produced < max_tokens {
            let capacity_left = self.kernel.capacity().saturating_sub(self.cache_position());
            if capacity_left == 0 {
                return Err(OwnedDecodeError::Kernel(
                    "decode cache capacity exhausted".to_string(),
                ));
            }
            let decision = controller.decide(DepthRequest {
                context_tokens: self.sequence.len(),
                remaining_tokens: max_tokens - produced,
                maximum_depth: maximum_depth.min(capacity_left),
            });
            let max_allowed = maximum_depth.min(capacity_left).min(max_tokens - produced);
            if decision.proposed_depth > max_allowed {
                return Err(OwnedDecodeError::InvalidDepthSelection {
                    selected: decision.proposed_depth,
                    maximum: max_allowed,
                });
            }

            if decision.proposed_depth == 0 {
                let token = self.pending_greedy;
                self.advance_and_commit(token, tap)?;
                produced += 1;
                controller.observe(DepthObservation {
                    proposed_depth: 0,
                    accepted_depth: 0,
                    verified_tokens: 0,
                });
                telemetry.controller_decisions.push(DepthDecisionTelemetry {
                    proposed_depth: 0,
                    accepted_depth: 0,
                    verification_work: 0,
                    draft_source_invoked: false,
                    command_buffer_chained: false,
                    extra_host_synchronizations: 0,
                    controller_reason: decision.reason,
                });
                if stop_tokens.contains(&token) {
                    self.stopped_at = Some(token);
                    finish_reason = FinishReason::StopToken(token);
                    break;
                }
                continue;
            }

            let proposal = draft_source.propose(&self.sequence, decision.proposed_depth)?;
            if proposal.tokens.is_empty() {
                return Err(OwnedDecodeError::EmptyDraftProposal);
            }
            if proposal.tokens.len() > decision.proposed_depth {
                return Err(OwnedDecodeError::OversizedDraftProposal {
                    actual: proposal.tokens.len(),
                    requested: decision.proposed_depth,
                });
            }
            let extra_host_synchronizations = match proposal.execution {
                ProposalExecution::MetalCommandBufferChained {
                    extra_host_synchronizations,
                } if extra_host_synchronizations == 0 => extra_host_synchronizations,
                _ => return Err(OwnedDecodeError::NativeMtpCommandBufferNotChained),
            };

            telemetry.proposal_calls += 1;
            telemetry.proposed_tokens += proposal.tokens.len();
            let start_position = self.cache_position();
            let post_token_argmaxes =
                kernel_result(self.kernel.verify_tokens(&mut self.cache, &proposal.tokens))?;
            if post_token_argmaxes.len() != proposal.tokens.len() {
                return Err(OwnedDecodeError::Kernel(format!(
                    "verifier returned {} argmaxes for {} proposals",
                    post_token_argmaxes.len(),
                    proposal.tokens.len()
                )));
            }
            telemetry.verified_tokens += proposal.tokens.len();

            let mut accepted_depth = 0;
            let mut mismatch = None;
            let mut stopped = false;
            for (index, proposed) in proposal.tokens.iter().copied().enumerate() {
                let expected = if index == 0 {
                    self.pending_greedy
                } else {
                    post_token_argmaxes[index - 1]
                };
                if proposed != expected {
                    mismatch = Some(expected);
                    break;
                }
                accepted_depth += 1;
                if stop_tokens.contains(&proposed) {
                    stopped = true;
                    break;
                }
            }

            telemetry.accepted_tokens += accepted_depth;
            if stopped || mismatch.is_some() {
                kernel_result(
                    self.kernel
                        .rewind(&mut self.cache, start_position + accepted_depth),
                )?;
            }
            for token in proposal.tokens.iter().copied().take(accepted_depth) {
                self.commit(token, tap);
                produced += 1;
            }

            if stopped {
                self.pending_greedy = post_token_argmaxes[accepted_depth - 1];
                self.stopped_at = proposal.tokens.get(accepted_depth - 1).copied();
                finish_reason =
                    FinishReason::StopToken(self.stopped_at.expect("accepted stop token exists"));
            } else if let Some(correct_token) = mismatch {
                telemetry.rejection_count += 1;
                // `correct_token` is the verifier argmax at exactly the retained
                // prefix. Advancing it replaces the rejected proposal's KV slot.
                self.advance_and_commit(correct_token, tap)?;
                produced += 1;
                if stop_tokens.contains(&correct_token) {
                    self.stopped_at = Some(correct_token);
                    finish_reason = FinishReason::StopToken(correct_token);
                }
            } else {
                self.pending_greedy = *post_token_argmaxes
                    .last()
                    .expect("non-empty proposal returned an argmax");
            }

            controller.observe(DepthObservation {
                proposed_depth: decision.proposed_depth,
                accepted_depth,
                verified_tokens: proposal.tokens.len(),
            });
            telemetry.controller_decisions.push(DepthDecisionTelemetry {
                proposed_depth: decision.proposed_depth,
                accepted_depth,
                verification_work: proposal.tokens.len(),
                draft_source_invoked: true,
                command_buffer_chained: true,
                extra_host_synchronizations,
                controller_reason: decision.reason,
            });
            if stopped || self.stopped_at.is_some() {
                break;
            }
        }

        Ok(SpeculativeDecodeOutput {
            output: DecodeOutput::from_tokens(self.generated[start..].to_vec(), finish_reason),
            telemetry,
        })
    }

    fn ensure_not_stopped(&self) -> OwnedDecodeResult<()> {
        if let Some(stop_token) = self.stopped_at {
            return Err(OwnedDecodeError::DecodeSessionStopped { stop_token });
        }
        Ok(())
    }

    fn advance_and_commit(&mut self, token: u32, tap: &mut dyn TokenTap) -> OwnedDecodeResult<()> {
        let logits = kernel_result(self.kernel.advance(&mut self.cache, token))?;
        self.pending_greedy = top_logits(&logits, 1)
            .first()
            .ok_or_else(|| OwnedDecodeError::Kernel("decoder produced empty logits".to_string()))?
            .token_id;
        self.commit(token, tap);
        Ok(())
    }

    fn commit(&mut self, token: u32, tap: &mut dyn TokenTap) {
        // The tap happens after target acceptance but before this token becomes
        // visible in the generated output. It cannot mutate either collection.
        tap.observe(TokenTapEvent {
            generated_index: self.generated.len(),
            token_id: token,
        });
        self.sequence.push(token);
        self.generated.push(token);
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::VecDeque;
    use std::rc::Rc;

    use anyhow::Result;

    use super::*;

    #[derive(Debug)]
    struct TestCache {
        position: usize,
    }

    struct TestKernel {
        target: Vec<u32>,
        prompt_len: usize,
        capacity: usize,
    }

    impl TestKernel {
        fn new(target: &[u32]) -> Self {
            Self {
                target: target.to_vec(),
                prompt_len: 0,
                capacity: 64,
            }
        }

        fn logits(token: u32) -> Vec<f32> {
            let mut logits = vec![f32::NEG_INFINITY; 128];
            logits[token as usize] = 1.0;
            logits
        }
    }

    impl DecodeKernel for TestKernel {
        type Cache = TestCache;

        fn capacity(&self) -> usize {
            self.capacity
        }

        fn prefill(&mut self, tokens: &[u32]) -> Result<(Self::Cache, u32)> {
            self.prompt_len = tokens.len();
            Ok((
                TestCache {
                    position: tokens.len(),
                },
                self.target[0],
            ))
        }

        fn advance(&mut self, cache: &mut Self::Cache, _token: u32) -> Result<Vec<f32>> {
            cache.position += 1;
            let target_index = cache.position - self.prompt_len;
            let next = self.target.get(target_index).copied().unwrap_or(0);
            Ok(Self::logits(next))
        }

        fn cache_position(&self, cache: &Self::Cache) -> usize {
            cache.position
        }

        fn inspect_cache_layer(&self, _cache: &Self::Cache, _layer: usize) -> Result<Vec<f32>> {
            Ok(Vec::new())
        }

        fn rewind(&mut self, cache: &mut Self::Cache, position: usize) -> Result<()> {
            cache.position = position;
            Ok(())
        }
    }

    fn native_source(proposals: Vec<Vec<u32>>) -> NativeMtpHead<impl NativeMtpExecutor> {
        let mut proposals = VecDeque::from(proposals);
        NativeMtpHead::new(
            NativeMtpHeadPin::new("catalog", "pinned-native-mtp-v1"),
            move |_context: &[u32], _depth: usize| {
                Ok(NativeMtpRound::chained(
                    proposals
                        .pop_front()
                        .expect("test drafted enough proposals"),
                ))
            },
        )
    }

    #[test]
    fn leviathan_rejection_matches_owned_serial_and_reports_work() {
        let target = [7, 8, 9];
        let mut serial_kernel = TestKernel::new(&target);
        let mut speculative_kernel = TestKernel::new(&target);
        let stop_tokens = BTreeSet::new();

        let serial = OwnedDecodeSession::prefill(&mut serial_kernel, &[1])
            .unwrap()
            .decode_serial(3, &stop_tokens, &mut NoopTokenTap)
            .unwrap();
        let mut speculative = OwnedDecodeSession::prefill(&mut speculative_kernel, &[1]).unwrap();
        let mut source = native_source(vec![vec![7, 99], vec![9]]);
        let mut controller = FixedDepthController::new(2);
        let output = speculative
            .decode_speculative(
                &mut source,
                &mut controller,
                3,
                2,
                &stop_tokens,
                &mut NoopTokenTap,
            )
            .unwrap();

        assert_eq!(output.output, serial);
        assert_eq!(output.telemetry.proposed_tokens, 3);
        assert_eq!(output.telemetry.accepted_tokens, 2);
        assert_eq!(output.telemetry.verified_tokens, 3);
        assert_eq!(output.telemetry.rejection_count, 1);
        assert_eq!(output.telemetry.acceptance_rate(), 2.0 / 3.0);
    }

    #[test]
    fn depth_zero_performs_no_head_work() {
        let mut kernel = TestKernel::new(&[7, 8]);
        let calls = Rc::new(Cell::new(0));
        let executor_calls = Rc::clone(&calls);
        let executor = move |_context: &[u32], _depth: usize| {
            executor_calls.set(executor_calls.get() + 1);
            Ok(NativeMtpRound::chained(vec![7]))
        };
        let mut source = NativeMtpHead::new(NativeMtpHeadPin::new("catalog", "head"), executor);
        let mut session = OwnedDecodeSession::prefill(&mut kernel, &[1]).unwrap();
        let mut controller = FixedDepthController::new(0);
        let output = session
            .decode_speculative(
                &mut source,
                &mut controller,
                2,
                2,
                &BTreeSet::new(),
                &mut NoopTokenTap,
            )
            .unwrap();

        assert_eq!(output.output.tokens, vec![7, 8]);
        assert_eq!(calls.get(), 0);
        assert!(output
            .telemetry
            .controller_decisions
            .iter()
            .all(|decision| !decision.draft_source_invoked));
    }

    #[test]
    fn native_mtp_reports_chained_command_buffer_without_extra_host_sync() {
        let mut kernel = TestKernel::new(&[7, 8]);
        let mut session = OwnedDecodeSession::prefill(&mut kernel, &[1]).unwrap();
        let mut source = native_source(vec![vec![7, 8]]);
        let mut controller = FixedDepthController::new(2);
        let output = session
            .decode_speculative(
                &mut source,
                &mut controller,
                2,
                2,
                &BTreeSet::new(),
                &mut NoopTokenTap,
            )
            .unwrap();

        assert_eq!(output.output.tokens, vec![7, 8]);
        assert_eq!(output.telemetry.controller_decisions.len(), 1);
        let decision = &output.telemetry.controller_decisions[0];
        assert!(decision.command_buffer_chained);
        assert_eq!(decision.extra_host_synchronizations, 0);
    }

    #[test]
    fn unchained_native_mtp_round_is_rejected_before_verification() {
        let mut kernel = TestKernel::new(&[7]);
        let mut session = OwnedDecodeSession::prefill(&mut kernel, &[1]).unwrap();
        let mut source = NativeMtpHead::new(
            NativeMtpHeadPin::new("catalog", "head"),
            |_context: &[u32], _depth: usize| {
                Ok(NativeMtpRound::with_execution(
                    vec![7],
                    ProposalExecution::NoHeadWork,
                ))
            },
        );
        let mut controller = FixedDepthController::new(1);

        assert_eq!(
            session
                .decode_speculative(
                    &mut source,
                    &mut controller,
                    1,
                    1,
                    &BTreeSet::new(),
                    &mut NoopTokenTap,
                )
                .unwrap_err(),
            OwnedDecodeError::NativeMtpCommandBufferNotChained
        );
    }

    #[test]
    fn post_acceptance_tap_cannot_change_emitted_bytes() {
        let target = [7, 8, 9];
        let mut untapped_kernel = TestKernel::new(&target);
        let mut tapped_kernel = TestKernel::new(&target);
        let mut untapped = OwnedDecodeSession::prefill(&mut untapped_kernel, &[1]).unwrap();
        let mut tapped = OwnedDecodeSession::prefill(&mut tapped_kernel, &[1]).unwrap();
        let mut source_one = native_source(vec![vec![7, 8], vec![9]]);
        let mut source_two = native_source(vec![vec![7, 8], vec![9]]);
        let mut controller_one = FixedDepthController::new(2);
        let mut controller_two = FixedDepthController::new(2);
        let mut observed = Vec::new();

        let untapped_output = untapped
            .decode_speculative(
                &mut source_one,
                &mut controller_one,
                3,
                2,
                &BTreeSet::new(),
                &mut NoopTokenTap,
            )
            .unwrap()
            .output;
        let tapped_output = tapped
            .decode_speculative(
                &mut source_two,
                &mut controller_two,
                3,
                2,
                &BTreeSet::new(),
                &mut |event: TokenTapEvent| observed.push(event.token_id),
            )
            .unwrap()
            .output;

        assert_eq!(tapped_output.tokens, untapped_output.tokens);
        assert_eq!(
            tapped_output.emitted_token_bytes,
            untapped_output.emitted_token_bytes
        );
        assert_eq!(observed, untapped_output.tokens);
    }

    #[test]
    fn kv_snapshot_continuation_reuses_no_prefill_dispatches_and_close_reclaims() {
        let allocator = KvAllocator::new(8);
        let configuration = KvConfiguration::new(KvBlockSize::Tokens256, 128).unwrap();
        let mut session = allocator.open_session(configuration, 2048).unwrap();
        let cold = session.cold_prefill_to(512, 2).unwrap();
        assert_eq!(cold.prefill_kernel_dispatches, 2);
        let retained = session.snapshot().unwrap();
        let mut continued = retained.continue_session();
        assert!(matches!(
            continued.warm_prefill_to(512, 1),
            Err(OwnedDecodeError::InvalidKvConfiguration(_))
        ));
        let warm = continued.warm_prefill_to(1024, 2).unwrap();
        assert!(warm.reused);
        assert_eq!(warm.reused_tokens, 512);
        assert_eq!(warm.reused_blocks, 2);
        assert_eq!(warm.reused_prefill_kernel_dispatches, 0);
        assert_eq!(allocator.accounting().unwrap().allocated_blocks, 4);
        let closed = continued.close().unwrap();
        assert_eq!(closed.allocated_blocks(), 0);
        assert_eq!(allocator.accounting().unwrap().allocated_blocks, 0);
    }

    #[test]
    fn misaligned_snapshot_is_a_typed_hard_error() {
        let allocator = KvAllocator::new(2);
        let configuration = KvConfiguration::new(KvBlockSize::Tokens256, 128).unwrap();
        let mut session = allocator.open_session(configuration, 1024).unwrap();
        session.cold_prefill_to(257, 1).unwrap();

        let error = session.snapshot().unwrap_err();
        assert_eq!(
            error,
            OwnedDecodeError::InvalidKvAlignment {
                position: 257,
                required_alignment: 256,
            }
        );
        // A failed transition drops the still-exclusive table and its leases,
        // so an invalid snapshot cannot leak allocator accounting.
        assert_eq!(allocator.accounting().unwrap().allocated_blocks, 0);
    }

    #[test]
    fn allocator_reports_runtime_double_free() {
        let allocator = KvAllocator::new(1);
        let mut block = allocator.acquire_block().unwrap();
        let id = block.block_id();
        block.release().unwrap();
        assert_eq!(
            block.release().unwrap_err(),
            OwnedDecodeError::KvDoubleFree { block_id: id }
        );
    }

    #[test]
    fn closed_session_rejects_runtime_continuation() {
        let allocator = KvAllocator::new(1);
        let configuration = KvConfiguration::new(KvBlockSize::Tokens256, 1).unwrap();
        let session = allocator.open_session(configuration, 256).unwrap();
        let session_id = session.session_id().0;
        let closed = session.close().unwrap();

        assert_eq!(
            closed.continue_session().unwrap_err(),
            OwnedDecodeError::KvSessionUseAfterClose { session_id }
        );
    }

    #[test]
    fn kv_matrix_requires_every_coordinate_and_breaks_ties_to_larger_blocks() {
        let measurements = required_kv_evaluation_matrix()
            .into_iter()
            .map(|coordinate| KvMatrixMeasurement {
                coordinate,
                recurrent_state_grain: 128,
                theoretical_minimum_retained_bytes: 100,
                retained_bytes: 110,
                warm_ttft: if coordinate.block_size == KvBlockSize::Tokens256 {
                    Duration::from_millis(2)
                } else {
                    Duration::from_millis(1)
                },
            })
            .collect::<Vec<_>>();
        let selected = select_kv_configuration(&measurements).unwrap();
        assert_eq!(selected.block_size, KvBlockSize::Tokens1024);

        let mut over_budget = measurements.clone();
        for measurement in &mut over_budget {
            if measurement.coordinate.block_size == KvBlockSize::Tokens1024 {
                measurement.retained_bytes = 111;
            }
        }
        assert_eq!(
            select_kv_configuration(&over_budget).unwrap().block_size,
            KvBlockSize::Tokens512
        );

        let mut incomplete = measurements;
        incomplete.pop();
        assert!(matches!(
            select_kv_configuration(&incomplete),
            Err(OwnedDecodeError::InvalidKvEvaluationMatrix(_))
        ));
    }

    #[test]
    fn depth_controller_refuses_missing_or_mismatched_measurements() {
        assert_eq!(
            NativeMtpDepthController::new(2, "machine", "artifact", None).unwrap_err(),
            OwnedDecodeError::MissingDepthControllerMeasurement
        );
        let measurement = DepthControllerMeasurement {
            machine_profile_hash: "other-machine".to_string(),
            catalog_fingerprint: "artifact".to_string(),
            native_mtp_head_forward: Duration::from_millis(1),
            backbone_step: Duration::from_millis(3),
        };
        assert!(matches!(
            NativeMtpDepthController::new(2, "machine", "artifact", Some(measurement)),
            Err(OwnedDecodeError::MismatchedDepthControllerMeasurement { .. })
        ));
    }

    #[test]
    fn wave_one_refuses_non_native_draft_sources() {
        struct Experimental;
        impl DraftSource for Experimental {
            fn kind(&self) -> DraftSourceKind {
                DraftSourceKind::Experimental
            }

            fn propose(
                &mut self,
                _context: &[u32],
                _depth: usize,
            ) -> OwnedDecodeResult<DraftProposal> {
                unreachable!("wave-1 gate rejects before proposal")
            }
        }

        let mut kernel = TestKernel::new(&[7]);
        let mut session = OwnedDecodeSession::prefill(&mut kernel, &[1]).unwrap();
        let mut source = Experimental;
        let mut controller = FixedDepthController::new(1);
        assert_eq!(
            session
                .decode_speculative(
                    &mut source,
                    &mut controller,
                    1,
                    1,
                    &BTreeSet::new(),
                    &mut NoopTokenTap,
                )
                .unwrap_err(),
            OwnedDecodeError::UnsupportedDraftSource
        );
    }
}
