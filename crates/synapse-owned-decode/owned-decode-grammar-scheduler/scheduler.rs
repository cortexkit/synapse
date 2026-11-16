//! The four-queue scheduler and quantum-sequencing protocol for owned decode.
//!
//! The scheduler is intentionally hardware-independent: it makes queue precedence,
//! weighted fairness, committed boundaries, and service-isolation telemetry explicit
//! before work reaches a Metal worker. Control is the only priority queue. Interactive,
//! Bulk, and Decode always take part in the same weighted fair cycle.

use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::owned_decode_routing::error::OwnedDecodeError;

/// The scheduler's normative queue identifiers.
///
/// The wire spelling remains snake_case for compatibility. Telemetry and test hooks
/// use [`Self::identifier`], whose values are the normative `Control`, `Interactive`,
/// `Bulk`, and `Decode` names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueueClass {
    Control,
    Interactive,
    Bulk,
    Decode,
}

impl QueueClass {
    /// The normative identifier emitted by scheduler telemetry.
    pub const fn identifier(self) -> &'static str {
        match self {
            Self::Control => "Control",
            Self::Interactive => "Interactive",
            Self::Bulk => "Bulk",
            Self::Decode => "Decode",
        }
    }

    /// The backward-compatible wire serialization.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Control => "control",
            Self::Interactive => "interactive",
            Self::Bulk => "bulk",
            Self::Decode => "decode",
        }
    }

    /// Parse a wire serialization back into a class.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "control" => Ok(Self::Control),
            "interactive" => Ok(Self::Interactive),
            "bulk" => Ok(Self::Bulk),
            "decode" => Ok(Self::Decode),
            other => Err(format!("unknown queue class '{other}'")),
        }
    }
}

/// Work that must stop at a committed scheduler boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantumWorkKind {
    Prefill,
    Decode,
    Mtp,
}

impl QuantumWorkKind {
    pub const fn identifier(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Mtp => "mtp",
        }
    }
}

/// Scheduler runtime configuration used by the shipped `embed-load-v1` profile.
///
/// `production_n` bounds every prefill, Decode, and MTP quantum. The queue weights
/// affect only the fair cycle; Control remains outside that cycle at strict precedence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeSchedulerConfig {
    /// Exactly one production N from `{8,16,32}`; N=1 is prohibited.
    pub production_n: u32,
    /// Decode weight in the Interactive/Bulk/Decode fair cycle.
    pub decode_weight: u32,
    /// Interactive (embed and rerank) weight in the fair cycle.
    pub interactive_weight: u32,
    /// Bulk weight in the fair cycle.
    pub bulk_weight: u32,
    /// Recorded wait threshold kept in scheduler manifests for backward compatibility.
    /// It does not bypass the weighted fair scheduling decision.
    pub decode_aging_window_ms: u64,
}

/// The production scheduler profile used for `embed-load-v1`, not a test fixture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShippedSchedulerConfiguration {
    pub profile_id: &'static str,
    pub scheduler: DecodeSchedulerConfig,
    pub decode_context_ceiling_tokens: u32,
    pub embed_p95_slo_ms: u64,
}

pub const EMBED_LOAD_V1_PROFILE: &str = "embed-load-v1";
pub const EMBED_LOAD_V1_CONTEXT_CEILING_TOKENS: u32 = 32_768;
pub const EMBED_LOAD_V1_EMBED_P95_SLO_MS: u64 = 150;
pub const SHIPPED_EMBED_LOAD_V1: ShippedSchedulerConfiguration = ShippedSchedulerConfiguration {
    profile_id: EMBED_LOAD_V1_PROFILE,
    scheduler: DecodeSchedulerConfig {
        production_n: 16,
        decode_weight: 4,
        interactive_weight: 1,
        bulk_weight: 1,
        decode_aging_window_ms: 250,
    },
    decode_context_ceiling_tokens: EMBED_LOAD_V1_CONTEXT_CEILING_TOKENS,
    embed_p95_slo_ms: EMBED_LOAD_V1_EMBED_P95_SLO_MS,
};

impl Default for DecodeSchedulerConfig {
    fn default() -> Self {
        SHIPPED_EMBED_LOAD_V1.scheduler
    }
}

impl DecodeSchedulerConfig {
    /// Return the shipped, production configuration for `embed-load-v1`.
    pub const fn shipped_embed_load_v1() -> Self {
        SHIPPED_EMBED_LOAD_V1.scheduler
    }

    const fn weight(&self, class: QueueClass) -> u32 {
        match class {
            QueueClass::Decode => self.decode_weight,
            QueueClass::Interactive => self.interactive_weight,
            QueueClass::Bulk => self.bulk_weight,
            QueueClass::Control => 0,
        }
    }

    const fn quantum_tokens(&self, kind: QuantumWorkKind) -> u32 {
        match kind {
            QuantumWorkKind::Prefill | QuantumWorkKind::Decode | QuantumWorkKind::Mtp => {
                self.production_n
            }
        }
    }
}

/// A scheduler operation tracked across queued, resident, and boundary states.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodeOp {
    pub op_id: String,
    pub generation_id: String,
    /// Admission time; the aging anchor before the first committed token.
    pub admitted_at_ms: u64,
    /// The aging anchor: admission time before the first committed token, and the
    /// most recent committed-token time thereafter.
    pub anchor_ms: u64,
    pub committed_tokens: u32,
    pub max_tokens: u32,
    /// True once the operation has dispatched and holds resident continuation state.
    pub resident: bool,
    /// When an abort request was recorded for evaluation at a committed boundary.
    pub cancelled_at_ms: Option<u64>,
    /// Absolute deadline; expiry is evaluated at boundaries and while queued.
    pub deadline_at_ms: Option<u64>,
}

/// Rejection from the quantum-boundary guard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantumCommitError {
    UnknownOperation,
    NotActiveQuantum,
    NonMonotonicCommit,
    ExceedsQuantumBudget,
    ExceedsMaximumTokens,
}

/// The outcome of a committed boundary evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryOutcome {
    /// An emergency revoke wins at the next committed boundary so the caller can
    /// emit terminal `artifact_revoked` accounting with the proven token count.
    Revoked,
    /// The quantum completed normally; pending abort or deadline state does not
    /// retroactively fail a completion that happened before those controls.
    Completed(FinishReason),
    /// An abort recorded before or at this non-terminal boundary.
    Cancelled,
    /// A deadline expired before or at this non-terminal boundary.
    DeadlineExceeded,
    /// Neither control applies; the operation continues with another quantum.
    Continue,
}

/// External successful finish reasons (the wire contract's four values).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    StopToken,
    MaxTokens,
    GrammarComplete,
    Cancelled,
}

/// The kind of boundary observed by the module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryKind {
    /// A non-final `generate_progress` boundary.
    Progress,
    /// A successful final response with its finish reason.
    Final(FinishReason),
}

/// A permit lifecycle event, recorded for scheduler measurements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PermitEvent {
    /// The module acquired the Decode permit for a Decode dispatch.
    Acquired,
    /// Decode won again while already holding the permit (no release overhead).
    Retained,
    /// A non-Decode class won the fair cycle, so the Decode permit was released.
    Released,
    /// The arbitration did not touch the Decode permit.
    Unchanged,
}

/// The result of one boundary arbitration.
#[derive(Clone, Debug, PartialEq)]
pub struct Arbitration {
    /// The class selected for the next dispatch opportunity.
    pub selected: QueueClass,
    /// The selected operation. Decode selection uses oldest-anchor order.
    pub op_id: Option<String>,
    /// What happened to the module-held Decode permit.
    pub permit_event: PermitEvent,
}

/// A queue/execution measurement tagged with the normative queue identifier.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueueExecutionTelemetry {
    pub queue: QueueClass,
    pub queue_identifier: &'static str,
    pub work_kind: Option<QuantumWorkKind>,
    pub op_id: String,
    pub queued_ms: u64,
    pub dispatched_at_ms: u64,
    pub queue_wait_ms: u64,
    pub execution_ms: Option<u64>,
    pub ttft_ms: Option<u64>,
}

/// Outcome of a production embed or rerank operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbedServiceOutcome {
    Completed { latency_ms: u64 },
    Failed,
    TimedOut,
}

/// Aggregate scheduler measurements and production-load test hooks.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Measurements {
    pub queue_depth_samples: Vec<u32>,
    pub per_op_waiting_ms: Vec<(String, u64)>,
    pub permit_events: Vec<PermitEvent>,
    pub continuation_count: u32,
    pub sequence_traces: Vec<String>,
    pub cancellation_latency_ms: Vec<u64>,
    pub deadline_latency_ms: Vec<u64>,
    pub queue_execution: Vec<QueueExecutionTelemetry>,
    pub embed_service_outcomes: Vec<EmbedServiceOutcome>,
}

impl Measurements {
    /// The nearest-rank p95 over completed embeds. Failures are deliberately not
    /// dropped: [`Self::embed_load_v1_slo_met`] rejects a run containing any.
    pub fn completed_embed_p95_ms(&self) -> Option<u64> {
        let mut latencies: Vec<u64> = self
            .embed_service_outcomes
            .iter()
            .filter_map(|outcome| match outcome {
                EmbedServiceOutcome::Completed { latency_ms } => Some(*latency_ms),
                EmbedServiceOutcome::Failed | EmbedServiceOutcome::TimedOut => None,
            })
            .collect();
        if latencies.is_empty() {
            return None;
        }
        latencies.sort_unstable();
        let rank = (latencies.len() * 95).div_ceil(100).max(1);
        latencies.get(rank - 1).copied()
    }

    /// Check the shipped `embed-load-v1` latency target. A failed or timed-out
    /// embed is a breach even when the completed samples meet nearest-rank p95.
    pub fn embed_load_v1_slo_met(&self) -> bool {
        self.embed_service_outcomes
            .iter()
            .all(|outcome| matches!(outcome, EmbedServiceOutcome::Completed { .. }))
            && self
                .completed_embed_p95_ms()
                .is_some_and(|p95| p95 <= EMBED_LOAD_V1_EMBED_P95_SLO_MS)
    }
}

/// The scheduler holds one queue per normative class, the operation table, the
/// Decode permit, weighted fair-cycle credits, and production telemetry.
#[derive(Clone, Debug)]
pub struct DecodeScheduler {
    config: DecodeSchedulerConfig,
    control: VecDeque<String>,
    interactive: VecDeque<String>,
    bulk: VecDeque<String>,
    decode: VecDeque<String>,
    ops: BTreeMap<String, DecodeOp>,
    quantum_work: BTreeMap<String, QuantumWorkKind>,
    revoked_at_ms: BTreeMap<String, u64>,
    active_quantum: Option<String>,
    permit_held: bool,
    /// Smooth weighted round-robin credits keyed by workload class.
    credits: BTreeMap<QueueClass, i64>,
    measurements: Measurements,
}

impl DecodeScheduler {
    pub fn new(config: DecodeSchedulerConfig) -> Self {
        assert!(
            matches!(config.production_n, 8 | 16 | 32),
            "production N must be one of 8, 16, or 32"
        );
        assert!(
            config.decode_weight > 0 && config.interactive_weight > 0 && config.bulk_weight > 0,
            "every weighted workload class must have a positive weight"
        );
        Self {
            config,
            control: VecDeque::new(),
            interactive: VecDeque::new(),
            bulk: VecDeque::new(),
            decode: VecDeque::new(),
            ops: BTreeMap::new(),
            quantum_work: BTreeMap::new(),
            revoked_at_ms: BTreeMap::new(),
            active_quantum: None,
            permit_held: false,
            credits: BTreeMap::new(),
            measurements: Measurements::default(),
        }
    }

    /// Build the shipped scheduler rather than a test-only tuned configuration.
    pub fn shipped_embed_load_v1() -> Self {
        Self::new(DecodeSchedulerConfig::shipped_embed_load_v1())
    }

    pub fn config(&self) -> &DecodeSchedulerConfig {
        &self.config
    }

    pub fn measurements(&self) -> &Measurements {
        &self.measurements
    }

    pub fn permit_held(&self) -> bool {
        self.permit_held
    }

    pub fn active_quantum(&self) -> Option<&str> {
        self.active_quantum.as_deref()
    }

    pub fn op(&self, op_id: &str) -> Option<&DecodeOp> {
        self.ops.get(op_id)
    }

    /// Return a scheduler-visible queue depth for production telemetry and tests.
    pub fn queued_count_for(&self, class: QueueClass) -> usize {
        self.queue(class).len()
    }

    /// Number of queued operations across all normative classes.
    pub fn queued_count(&self) -> usize {
        self.control.len() + self.interactive.len() + self.bulk.len() + self.decode.len()
    }

    /// Return the kind of bounded work associated with an operation.
    pub fn quantum_work_kind(&self, op_id: &str) -> Option<QuantumWorkKind> {
        self.quantum_work.get(op_id).copied()
    }

    /// Return the largest legal next committed span for bounded work.
    pub fn quantum_budget(&self, op_id: &str) -> Option<u32> {
        let kind = self.quantum_work_kind(op_id)?;
        let op = self.ops.get(op_id)?;
        Some(
            self.config
                .quantum_tokens(kind)
                .min(op.max_tokens.saturating_sub(op.committed_tokens)),
        )
    }

    fn queue_mut(&mut self, class: QueueClass) -> &mut VecDeque<String> {
        match class {
            QueueClass::Control => &mut self.control,
            QueueClass::Interactive => &mut self.interactive,
            QueueClass::Bulk => &mut self.bulk,
            QueueClass::Decode => &mut self.decode,
        }
    }

    fn queue(&self, class: QueueClass) -> &VecDeque<String> {
        match class {
            QueueClass::Control => &self.control,
            QueueClass::Interactive => &self.interactive,
            QueueClass::Bulk => &self.bulk,
            QueueClass::Decode => &self.decode,
        }
    }

    fn is_queued(&self, op_id: &str) -> bool {
        [
            QueueClass::Control,
            QueueClass::Interactive,
            QueueClass::Bulk,
            QueueClass::Decode,
        ]
        .into_iter()
        .any(|class| self.queue(class).iter().any(|id| id == op_id))
    }

    /// Admit Decode work. Decode spans are bounded by the shipped production N.
    pub fn admit_decode(&mut self, op: DecodeOp) {
        self.admit_quantum_work(QueueClass::Decode, QuantumWorkKind::Decode, op);
    }

    /// Admit native MTP work. It shares the Decode queue and the same N-token
    /// committed boundary as ordinary Decode work.
    pub fn admit_mtp(&mut self, op: DecodeOp) {
        self.admit_quantum_work(QueueClass::Decode, QuantumWorkKind::Mtp, op);
    }

    /// Admit prefill work to an Interactive or Bulk workload queue. Prefill uses
    /// the production N-token quantum so it can yield to embeds and controls.
    pub fn admit_prefill(
        &mut self,
        class: QueueClass,
        op_id: impl Into<String>,
        admitted_at_ms: u64,
        max_tokens: u32,
    ) {
        assert!(
            matches!(class, QueueClass::Interactive | QueueClass::Bulk),
            "prefill belongs to Interactive or Bulk"
        );
        let op_id = op_id.into();
        self.admit_quantum_work(
            class,
            QuantumWorkKind::Prefill,
            DecodeOp {
                op_id,
                generation_id: String::new(),
                admitted_at_ms,
                anchor_ms: admitted_at_ms,
                committed_tokens: 0,
                max_tokens,
                resident: false,
                cancelled_at_ms: None,
                deadline_at_ms: None,
            },
        );
    }

    fn admit_quantum_work(&mut self, class: QueueClass, kind: QuantumWorkKind, op: DecodeOp) {
        assert_eq!(
            op.anchor_ms, op.admitted_at_ms,
            "anchor starts at admission"
        );
        assert!(
            op.max_tokens > 0,
            "bounded work needs a positive token budget"
        );
        assert!(
            matches!(
                (class, kind),
                (
                    QueueClass::Decode,
                    QuantumWorkKind::Decode | QuantumWorkKind::Mtp
                ) | (
                    QueueClass::Interactive | QueueClass::Bulk,
                    QuantumWorkKind::Prefill
                )
            ),
            "work kind must use its normative scheduler queue"
        );
        let op_id = op.op_id.clone();
        self.queue_mut(class).push_back(op_id.clone());
        self.quantum_work.insert(op_id.clone(), kind);
        self.ops.insert(op_id, op);
        self.sample_depth();
    }

    /// Admit ordinary unbounded service work to Control, Interactive, or Bulk.
    pub fn admit_other(&mut self, class: QueueClass, op_id: impl Into<String>) {
        self.admit_other_at(class, op_id, 0);
    }

    /// Admit ordinary service work with its actual queue timestamp for telemetry.
    pub fn admit_other_at(
        &mut self,
        class: QueueClass,
        op_id: impl Into<String>,
        admitted_at_ms: u64,
    ) {
        assert_ne!(
            class,
            QueueClass::Decode,
            "use Decode, MTP, or prefill admission"
        );
        let op_id = op_id.into();
        self.queue_mut(class).push_back(op_id.clone());
        self.ops.insert(
            op_id.clone(),
            DecodeOp {
                op_id,
                generation_id: String::new(),
                admitted_at_ms,
                anchor_ms: admitted_at_ms,
                committed_tokens: 0,
                max_tokens: 0,
                resident: false,
                cancelled_at_ms: None,
                deadline_at_ms: None,
            },
        );
        self.sample_depth();
    }

    /// Embed traffic is permanently admitted to the Interactive lane.
    pub fn admit_embed(&mut self, op_id: impl Into<String>, admitted_at_ms: u64) {
        self.admit_other_at(QueueClass::Interactive, op_id, admitted_at_ms);
    }

    /// Rerank traffic shares the permanently available Interactive service lane.
    pub fn admit_rerank(&mut self, op_id: impl Into<String>, admitted_at_ms: u64) {
        self.admit_other_at(QueueClass::Interactive, op_id, admitted_at_ms);
    }

    fn sample_depth(&mut self) {
        self.measurements
            .queue_depth_samples
            .push(self.queued_count() as u32);
    }

    /// The oldest waiting Decode operation, with admission order as the tie-breaker.
    fn select_decode(&self) -> Option<String> {
        self.decode
            .iter()
            .filter_map(|op_id| self.ops.get(op_id))
            .min_by(|a, b| {
                a.anchor_ms
                    .cmp(&b.anchor_ms)
                    .then_with(|| a.admitted_at_ms.cmp(&b.admitted_at_ms))
            })
            .map(|op| op.op_id.clone())
    }

    fn take_decode(&mut self) -> Option<String> {
        let op_id = self.select_decode()?;
        let position = self.decode.iter().position(|id| id == &op_id)?;
        self.decode.remove(position)
    }

    fn runnable_classes(&self) -> Vec<QueueClass> {
        [
            QueueClass::Interactive,
            QueueClass::Bulk,
            QueueClass::Decode,
        ]
        .into_iter()
        .filter(|class| !self.queue(*class).is_empty())
        .collect()
    }

    /// Smooth weighted round-robin selection over the runnable workload classes.
    fn fair_cycle_pick(&mut self, runnable: &[QueueClass]) -> QueueClass {
        self.credits.retain(|class, _| runnable.contains(class));
        let total: i64 = runnable
            .iter()
            .map(|class| self.config.weight(*class) as i64)
            .sum();
        for class in runnable {
            let credit = self.credits.entry(*class).or_insert(0);
            *credit += self.config.weight(*class) as i64;
        }
        let picked = *runnable
            .iter()
            .max_by_key(|class| self.credits.get(class).copied().unwrap_or(0))
            .expect("runnable is non-empty");
        *self.credits.entry(picked).or_default() -= total;
        picked
    }

    fn record_dispatch(&mut self, class: QueueClass, op_id: &str, now_ms: u64) {
        let work_kind = self.quantum_work.get(op_id).copied();
        let Some(op) = self.ops.get_mut(op_id) else {
            return;
        };
        let queued_ms = op.admitted_at_ms;
        let queue_wait_ms = now_ms.saturating_sub(queued_ms);
        if work_kind.is_some() {
            self.active_quantum = Some(op_id.to_string());
            op.resident = true;
        }
        self.measurements
            .per_op_waiting_ms
            .push((op_id.to_string(), queue_wait_ms));
        self.measurements
            .queue_execution
            .push(QueueExecutionTelemetry {
                queue: class,
                queue_identifier: class.identifier(),
                work_kind,
                op_id: op_id.to_string(),
                queued_ms,
                dispatched_at_ms: now_ms,
                queue_wait_ms,
                execution_ms: None,
                ttft_ms: None,
            });
    }

    fn arbitration(
        &mut self,
        selected: QueueClass,
        op_id: Option<String>,
        permit_event: PermitEvent,
        now_ms: u64,
    ) -> Arbitration {
        if let Some(op_id) = op_id.as_deref() {
            self.record_dispatch(selected, op_id, now_ms);
        }
        self.sample_depth();
        Arbitration {
            selected,
            op_id,
            permit_event,
        }
    }

    /// Arbitrate exactly one committed-boundary dispatch opportunity.
    ///
    /// Control has strict precedence. Once Control is empty, Interactive, Bulk, and
    /// Decode are selected only by the weighted fair cycle. Decode never obtains an
    /// aging-based priority bypass because that would remove it from the fair cycle.
    pub fn arbitrate(&mut self, now_ms: u64) -> Option<Arbitration> {
        if self.active_quantum.is_some() {
            return None;
        }
        self.sample_depth();

        if !self.control.is_empty() {
            let op_id = self.control.pop_front();
            let permit_event = self.set_permit(false);
            return Some(self.arbitration(QueueClass::Control, op_id, permit_event, now_ms));
        }

        let runnable = self.runnable_classes();
        if runnable.is_empty() {
            return None;
        }
        let decode_runnable = runnable.contains(&QueueClass::Decode);
        let picked = if runnable == [QueueClass::Decode] {
            QueueClass::Decode
        } else {
            self.fair_cycle_pick(&runnable)
        };
        let op_id = if picked == QueueClass::Decode {
            self.take_decode()
        } else {
            self.queue_mut(picked).pop_front()
        };
        let permit_event = if picked == QueueClass::Decode {
            self.set_permit(true)
        } else {
            // A non-Decode winner yields the permit when resident Decode work was
            // competing. This leaves the Interactive lane serviceable under decode.
            self.set_permit(!decode_runnable)
        };
        Some(self.arbitration(picked, op_id, permit_event, now_ms))
    }

    /// Set the module-held Decode permit, returning the lifecycle event.
    fn set_permit(&mut self, held: bool) -> PermitEvent {
        let event = match (self.permit_held, held) {
            (false, true) => PermitEvent::Acquired,
            (true, true) => PermitEvent::Retained,
            (true, false) => PermitEvent::Released,
            (false, false) => PermitEvent::Unchanged,
        };
        self.permit_held = held;
        self.measurements.permit_events.push(event);
        event
    }

    /// Begin a Decode or MTP quantum when resuming after a worker handoff interrupted
    /// normal arbitration. Ordinary dispatch uses [`Self::arbitrate`], which begins
    /// bounded work as it selects it.
    pub fn begin_decode_quantum(&mut self, op_id: &str, now_ms: u64) {
        if self.active_quantum.as_deref() == Some(op_id) {
            return;
        }
        if self.active_quantum.is_some() || !self.quantum_work.contains_key(op_id) {
            return;
        }
        if let Some(position) = self.decode.iter().position(|id| id == op_id) {
            self.decode.remove(position);
        }
        self.record_dispatch(QueueClass::Decode, op_id, now_ms);
        self.sample_depth();
    }

    /// Record a bounded operation's execution time and optional time-to-first-token.
    pub fn record_execution(&mut self, op_id: &str, finished_at_ms: u64, ttft_ms: Option<u64>) {
        if let Some(sample) = self
            .measurements
            .queue_execution
            .iter_mut()
            .rev()
            .find(|sample| sample.op_id == op_id && sample.execution_ms.is_none())
        {
            sample.execution_ms = Some(finished_at_ms.saturating_sub(sample.dispatched_at_ms));
            sample.ttft_ms = ttft_ms;
        }
    }

    /// Record an embed or rerank outcome for the shipped `embed-load-v1` SLO.
    pub fn record_embed_service_outcome(&mut self, outcome: EmbedServiceOutcome) {
        self.measurements.embed_service_outcomes.push(outcome);
    }

    /// Re-enqueue a Decode or MTP continuation after a committed boundary.
    pub fn requeue_continuation(&mut self, op_id: &str) {
        let can_continue = self
            .ops
            .get(op_id)
            .is_some_and(|op| op.cancelled_at_ms.is_none())
            && !self.revoked_at_ms.contains_key(op_id)
            && matches!(
                self.quantum_work.get(op_id),
                Some(QuantumWorkKind::Decode | QuantumWorkKind::Mtp)
            )
            && self.active_quantum.is_none();
        if can_continue && !self.decode.iter().any(|id| id == op_id) {
            self.decode.push_back(op_id.to_string());
            self.measurements.continuation_count += 1;
            self.sample_depth();
        }
    }

    /// Commit a bounded quantum, enforcing the production-N limit.
    pub fn try_commit_quantum(
        &mut self,
        op_id: &str,
        committed_tokens: u32,
        now_ms: u64,
    ) -> Result<(), QuantumCommitError> {
        let kind = self
            .quantum_work
            .get(op_id)
            .copied()
            .ok_or(QuantumCommitError::UnknownOperation)?;
        if self.active_quantum.as_deref() != Some(op_id) {
            return Err(QuantumCommitError::NotActiveQuantum);
        }
        let quantum_budget = self.config.quantum_tokens(kind);
        let op = self
            .ops
            .get_mut(op_id)
            .ok_or(QuantumCommitError::UnknownOperation)?;
        if committed_tokens <= op.committed_tokens {
            return Err(QuantumCommitError::NonMonotonicCommit);
        }
        if committed_tokens > op.max_tokens {
            return Err(QuantumCommitError::ExceedsMaximumTokens);
        }
        if committed_tokens - op.committed_tokens > quantum_budget {
            return Err(QuantumCommitError::ExceedsQuantumBudget);
        }
        op.committed_tokens = committed_tokens;
        op.anchor_ms = now_ms;
        self.active_quantum = None;
        Ok(())
    }

    /// Compatibility wrapper for existing callers. Invalid commits are ignored;
    /// production callers that need a reason use [`Self::try_commit_quantum`].
    pub fn commit_quantum(&mut self, op_id: &str, committed_tokens: u32, now_ms: u64) {
        let _ = self.try_commit_quantum(op_id, committed_tokens, now_ms);
    }

    /// Remove a completed or terminated operation from the scheduler entirely.
    pub fn remove_op(&mut self, op_id: &str) {
        self.ops.remove(op_id);
        self.quantum_work.remove(op_id);
        self.revoked_at_ms.remove(op_id);
        if self.active_quantum.as_deref() == Some(op_id) {
            self.active_quantum = None;
        }
        for queue in [
            &mut self.control,
            &mut self.interactive,
            &mut self.bulk,
            &mut self.decode,
        ] {
            queue.retain(|id| id != op_id);
        }
        self.sample_depth();
    }

    fn request_boundary_control(
        &mut self,
        op_id: &str,
        now_ms: u64,
        is_revoke: bool,
    ) -> CancelResult {
        if !self.ops.contains_key(op_id) {
            return CancelResult::NotFound;
        }
        if self.is_queued(op_id) && self.active_quantum.as_deref() != Some(op_id) {
            self.remove_op(op_id);
            if !is_revoke {
                self.measurements.cancellation_latency_ms.push(0);
            }
            return CancelResult::RemovedQueued;
        }
        if self.active_quantum.as_deref() == Some(op_id) {
            if let Some(op) = self.ops.get_mut(op_id) {
                if !is_revoke {
                    op.cancelled_at_ms = Some(now_ms);
                }
            }
            if is_revoke {
                self.revoked_at_ms.insert(op_id.to_string(), now_ms);
            }
            return CancelResult::DeferredToBoundary;
        }
        CancelResult::NotFound
    }

    /// Abort an operation. An active quantum observes this only after it commits.
    pub fn request_cancel(&mut self, op_id: &str, now_ms: u64) -> CancelResult {
        self.request_boundary_control(op_id, now_ms, false)
    }

    /// Emergency revoke an operation. An active quantum emits `artifact_revoked`
    /// accounting at its next committed boundary rather than exposing uncommitted work.
    pub fn request_revoke(&mut self, op_id: &str, now_ms: u64) -> CancelResult {
        self.request_boundary_control(op_id, now_ms, true)
    }

    /// Remove queued operations whose deadline has expired. Deadline expiry while
    /// queued is a clean removal and never dispatches work.
    pub fn remove_expired_deadlines(&mut self, now_ms: u64) -> Vec<String> {
        let expired: Vec<String> = self
            .ops
            .values()
            .filter(|op| {
                op.deadline_at_ms.is_some_and(|deadline| deadline <= now_ms)
                    && self.is_queued(&op.op_id)
            })
            .map(|op| op.op_id.clone())
            .collect();
        for op_id in &expired {
            if let Some(op) = self.ops.get(op_id) {
                let latency = now_ms.saturating_sub(op.deadline_at_ms.unwrap_or(now_ms));
                self.measurements.deadline_latency_ms.push(latency);
            }
            self.remove_op(op_id);
        }
        expired
    }

    /// Evaluate a committed boundary. Revocation is checked first; abort and
    /// deadlines apply only to non-terminal progress boundaries.
    pub fn evaluate_boundary(
        &self,
        op_id: &str,
        boundary: BoundaryKind,
        now_ms: u64,
    ) -> BoundaryOutcome {
        let Some(op) = self.ops.get(op_id) else {
            return BoundaryOutcome::Continue;
        };
        if self
            .revoked_at_ms
            .get(op_id)
            .is_some_and(|revoked_at| *revoked_at <= now_ms)
        {
            return BoundaryOutcome::Revoked;
        }
        match boundary {
            BoundaryKind::Final(reason) => BoundaryOutcome::Completed(reason),
            BoundaryKind::Progress => {
                if op
                    .cancelled_at_ms
                    .is_some_and(|cancelled_at| cancelled_at <= now_ms)
                {
                    return BoundaryOutcome::Cancelled;
                }
                if op.deadline_at_ms.is_some_and(|deadline| deadline <= now_ms) {
                    return BoundaryOutcome::DeadlineExceeded;
                }
                BoundaryOutcome::Continue
            }
        }
    }
}

/// The result of an abort or revoke request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CancelResult {
    /// The operation was queued and removed immediately at a committed boundary.
    RemovedQueued,
    /// The operation is active, so the control is observed at the next boundary.
    DeferredToBoundary,
    /// The operation is no longer scheduler-owned.
    NotFound,
}

/// The module-side validator for one logical generation's quantum-sequencing
/// protocol (`owned-metal-decode-worker-v1`). It authorizes N-token spans and
/// validates progress and continuation frames, returning
/// [`OwnedDecodeError::ProtocolMismatch`] on any sequence, budget, generation, or
/// session violation.
#[derive(Clone, Debug)]
pub struct GenerationProtocol {
    generation_id: String,
    production_n: u32,
    max_tokens: u32,
    /// The next expected `generate_progress` quantum sequence (first is one).
    expected_sequence: u32,
    /// Attempt-local cumulative committed tokens.
    committed_tokens: u32,
    /// The last completed quantum sequence (zero before any progress).
    last_completed_sequence: u32,
    /// The bound worker generation; frames from a superseded generation are stale.
    worker_generation: u64,
    started: bool,
    closed: bool,
}

/// A `generate_continue` frame the module sends to authorize the next span.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContinueFrame {
    pub generation_id: String,
    pub expected_sequence: u32,
    pub next_token_budget: u32,
}

impl GenerationProtocol {
    pub fn new(generation_id: impl Into<String>, production_n: u32, max_tokens: u32) -> Self {
        assert!(
            matches!(production_n, 8 | 16 | 32),
            "production N must be one of 8, 16, or 32"
        );
        Self {
            generation_id: generation_id.into(),
            production_n,
            max_tokens,
            expected_sequence: 1,
            committed_tokens: 0,
            last_completed_sequence: 0,
            worker_generation: 0,
            started: false,
            closed: false,
        }
    }

    pub fn generation_id(&self) -> &str {
        &self.generation_id
    }

    pub fn committed_tokens(&self) -> u32 {
        self.committed_tokens
    }

    pub fn last_completed_sequence(&self) -> u32 {
        self.last_completed_sequence
    }

    /// Authorize the first span (`generate_start`): `min(N, max_tokens)` tokens.
    /// Binds the worker generation for stale-session detection.
    pub fn authorize_start(&mut self, worker_generation: u64) -> Result<u32, OwnedDecodeError> {
        if self.started {
            return Err(OwnedDecodeError::ProtocolMismatch);
        }
        self.started = true;
        self.worker_generation = worker_generation;
        Ok(self.first_span_budget())
    }

    fn first_span_budget(&self) -> u32 {
        self.production_n.min(self.max_tokens)
    }

    /// Validate and apply a `generate_progress` frame. The sequence must equal the
    /// next expected sequence (repeated or skipped sequences are a mismatch) and
    /// the cumulative committed count must advance without exceeding `max_tokens`.
    pub fn receive_progress(
        &mut self,
        worker_generation: u64,
        quantum_sequence: u32,
        committed_token_count: u32,
    ) -> Result<(), OwnedDecodeError> {
        if self.closed || !self.started {
            return Err(OwnedDecodeError::ProtocolMismatch);
        }
        if worker_generation != self.worker_generation {
            // A frame from a superseded worker generation is stale.
            return Err(OwnedDecodeError::ProtocolMismatch);
        }
        if quantum_sequence != self.expected_sequence {
            return Err(OwnedDecodeError::ProtocolMismatch);
        }
        if committed_token_count <= self.committed_tokens || committed_token_count > self.max_tokens
        {
            return Err(OwnedDecodeError::ProtocolMismatch);
        }
        self.committed_tokens = committed_token_count;
        self.last_completed_sequence = quantum_sequence;
        self.expected_sequence += 1;
        Ok(())
    }

    /// Build the next `generate_continue` frame, or `None` when the request budget
    /// is exhausted (the next boundary is final). The budget is greater than zero
    /// and no greater than N or the remaining request budget.
    pub fn next_continue(&self) -> Option<ContinueFrame> {
        let remaining = self.max_tokens.saturating_sub(self.committed_tokens);
        if remaining == 0 {
            return None;
        }
        Some(ContinueFrame {
            generation_id: self.generation_id.clone(),
            expected_sequence: self.expected_sequence,
            next_token_budget: self.production_n.min(remaining),
        })
    }

    /// Validate an externally constructed continue frame against the protocol
    /// state (used by tests and the worker-side check). Invalid budgets or a wrong
    /// expected sequence are a protocol mismatch.
    pub fn validate_continue(&self, frame: &ContinueFrame) -> Result<(), OwnedDecodeError> {
        let expected = self
            .next_continue()
            .ok_or(OwnedDecodeError::ProtocolMismatch)?;
        if frame.generation_id != self.generation_id
            || frame.expected_sequence != expected.expected_sequence
            || frame.next_token_budget == 0
            || frame.next_token_budget != expected.next_token_budget
        {
            return Err(OwnedDecodeError::ProtocolMismatch);
        }
        Ok(())
    }

    /// Mark the generation closed (after a final response or cleanup). Further
    /// frames are a protocol mismatch (continuation after cleanup).
    pub fn close(&mut self) {
        self.closed = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op(id: &str, admitted_at_ms: u64, max_tokens: u32) -> DecodeOp {
        DecodeOp {
            op_id: id.to_string(),
            generation_id: format!("gen-{id}"),
            admitted_at_ms,
            anchor_ms: admitted_at_ms,
            committed_tokens: 0,
            max_tokens,
            resident: false,
            cancelled_at_ms: None,
            deadline_at_ms: None,
        }
    }

    #[test]
    fn normative_queue_identifiers_and_wire_spellings_are_stable() {
        for (class, identifier, wire) in [
            (QueueClass::Control, "Control", "control"),
            (QueueClass::Interactive, "Interactive", "interactive"),
            (QueueClass::Bulk, "Bulk", "bulk"),
            (QueueClass::Decode, "Decode", "decode"),
        ] {
            assert_eq!(class.identifier(), identifier);
            assert_eq!(class.as_str(), wire);
            assert_eq!(QueueClass::parse(wire), Ok(class));
            assert_eq!(
                serde_json::to_string(&class).expect("serializes"),
                format!("\"{wire}\"")
            );
        }
    }

    #[test]
    fn control_has_strict_precedence_over_every_workload_class() {
        let mut scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());
        scheduler.admit_decode(op("d1", 0, 64));
        scheduler.admit_other(QueueClass::Interactive, "i1");
        scheduler.admit_other(QueueClass::Bulk, "b1");
        scheduler.admit_other(QueueClass::Control, "c1");
        let arbitration = scheduler.arbitrate(10).expect("arbitrates");
        assert_eq!(arbitration.selected, QueueClass::Control);
        assert_eq!(arbitration.op_id.as_deref(), Some("c1"));
        let telemetry = scheduler
            .measurements()
            .queue_execution
            .last()
            .expect("control dispatch telemetry");
        assert_eq!(telemetry.queue_identifier, "Control");
        assert_eq!(scheduler.queued_count_for(QueueClass::Interactive), 1);
        assert_eq!(scheduler.queued_count_for(QueueClass::Bulk), 1);
        assert_eq!(scheduler.queued_count_for(QueueClass::Decode), 1);
    }

    #[test]
    fn decode_is_sole_runnable_executes_without_release() {
        let mut scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());
        scheduler.admit_decode(op("d1", 0, 64));
        let first = scheduler.arbitrate(0).expect("arbitrates");
        assert_eq!(first.selected, QueueClass::Decode);
        assert_eq!(first.permit_event, PermitEvent::Acquired);
        // Re-queue the continuation; with DECODE still sole runnable, the permit is
        // retained (no artificial release/reacquire).
        scheduler.commit_quantum("d1", 16, 1);
        scheduler.requeue_continuation("d1");
        let second = scheduler.arbitrate(1).expect("arbitrates");
        assert_eq!(second.selected, QueueClass::Decode);
        assert_eq!(second.permit_event, PermitEvent::Retained);
        assert!(scheduler.permit_held());
    }

    #[test]
    fn non_decode_win_releases_the_permit() {
        let mut scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());
        scheduler.admit_decode(op("d1", 0, 1_024));
        scheduler.admit_other(QueueClass::Interactive, "e1");
        scheduler.admit_other(QueueClass::Interactive, "e2");
        scheduler.admit_other(QueueClass::Interactive, "e3");
        scheduler.admit_other(QueueClass::Interactive, "e4");
        scheduler.admit_other(QueueClass::Interactive, "e5");

        // Drive enough boundaries that the fair cycle selects Interactive at least
        // once while DECODE is runnable; that must release the decode permit.
        let mut saw_release = false;
        for tick in 0..20 {
            if let Some(arbitration) = scheduler.arbitrate(tick) {
                if arbitration.selected == QueueClass::Interactive {
                    // Re-admit embedding work so Interactive stays runnable.
                    scheduler.admit_other(QueueClass::Interactive, format!("e{}", 10 + tick));
                }
                if arbitration.permit_event == PermitEvent::Released {
                    saw_release = true;
                }
                if arbitration.selected == QueueClass::Decode {
                    let committed =
                        scheduler.op("d1").expect("decode exists").committed_tokens + 16;
                    scheduler
                        .try_commit_quantum("d1", committed, tick)
                        .expect("commits at the boundary");
                    scheduler.requeue_continuation("d1");
                }
            }
        }
        assert!(
            saw_release,
            "a non-DECODE win must release the decode permit"
        );
    }

    #[test]
    fn workload_classes_receive_weighted_fair_turns_without_aging_bypass() {
        let config = DecodeSchedulerConfig {
            decode_weight: 4,
            interactive_weight: 2,
            bulk_weight: 1,
            ..DecodeSchedulerConfig::default()
        };
        let mut scheduler = DecodeScheduler::new(config);
        scheduler.admit_decode(op("d1", 0, 20_000));
        scheduler.admit_other(QueueClass::Interactive, "i0");
        scheduler.admit_other(QueueClass::Bulk, "b0");
        let mut interactive_wins = 0u32;
        let mut bulk_wins = 0u32;
        let mut decode_wins = 0u32;

        for tick in 0..700u64 {
            let arbitration = scheduler
                .arbitrate(tick)
                .expect("all classes remain runnable");
            match arbitration.selected {
                QueueClass::Interactive => {
                    interactive_wins += 1;
                    scheduler.admit_other(QueueClass::Interactive, format!("i{tick}"));
                }
                QueueClass::Bulk => {
                    bulk_wins += 1;
                    scheduler.admit_other(QueueClass::Bulk, format!("b{tick}"));
                }
                QueueClass::Decode => {
                    decode_wins += 1;
                    let committed =
                        scheduler.op("d1").expect("decode exists").committed_tokens + 16;
                    scheduler
                        .try_commit_quantum("d1", committed, tick)
                        .expect("within the Decode quantum");
                    scheduler.requeue_continuation("d1");
                }
                QueueClass::Control => panic!("no control work was admitted"),
            }
        }

        // The stable weights are 4:2:1, so every workload class participates and
        // converges to its declared share rather than an aging-priority bypass.
        assert_eq!((decode_wins, interactive_wins, bulk_wins), (400, 200, 100));
        let telemetry_ids: std::collections::BTreeSet<_> = scheduler
            .measurements()
            .queue_execution
            .iter()
            .map(|sample| sample.queue_identifier)
            .collect();
        assert_eq!(telemetry_ids, ["Bulk", "Decode", "Interactive"].into());
    }

    #[test]
    fn within_decode_selects_oldest_anchor_fifo() {
        let mut scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());
        scheduler.admit_decode(op("older", 0, 64));
        scheduler.admit_decode(op("newer", 10, 64));
        let arbitration = scheduler.arbitrate(10).expect("arbitrates");
        assert_eq!(arbitration.op_id.as_deref(), Some("older"));
    }

    #[test]
    fn committing_quantum_advances_anchor_behind_older_op() {
        let mut scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());
        scheduler.admit_decode(op("a", 0, 128));
        scheduler.admit_decode(op("b", 5, 128));
        // 'a' runs and commits at t=20, advancing its anchor to 20; 'b' (anchor 5)
        // is now older and must be selected next.
        scheduler.begin_decode_quantum("a", 0);
        scheduler.commit_quantum("a", 16, 20);
        scheduler.requeue_continuation("a");
        let arbitration = scheduler.arbitrate(20).expect("arbitrates");
        assert_eq!(arbitration.op_id.as_deref(), Some("b"));
    }

    #[test]
    fn prefill_decode_and_mtp_work_are_bounded_by_production_n() {
        let mut scheduler = DecodeScheduler::shipped_embed_load_v1();
        scheduler.admit_prefill(QueueClass::Interactive, "p1", 0, 64);
        let prefill = scheduler.arbitrate(0).expect("prefill dispatches");
        assert_eq!(prefill.selected, QueueClass::Interactive);
        assert_eq!(scheduler.quantum_budget("p1"), Some(16));
        assert_eq!(
            scheduler.try_commit_quantum("p1", 17, 1),
            Err(QuantumCommitError::ExceedsQuantumBudget)
        );
        scheduler
            .try_commit_quantum("p1", 16, 1)
            .expect("prefill commits at the boundary");

        scheduler.admit_mtp(op("m1", 2, 64));
        let mtp = scheduler.arbitrate(2).expect("MTP dispatches");
        assert_eq!(mtp.selected, QueueClass::Decode);
        assert_eq!(
            scheduler.quantum_work_kind("m1"),
            Some(QuantumWorkKind::Mtp)
        );
        assert_eq!(
            scheduler.try_commit_quantum("m1", 17, 3),
            Err(QuantumCommitError::ExceedsQuantumBudget)
        );
        scheduler
            .try_commit_quantum("m1", 16, 3)
            .expect("MTP commits at the boundary");

        scheduler.admit_decode(op("d1", 4, 64));
        scheduler.arbitrate(4).expect("Decode dispatches");
        assert_eq!(
            scheduler.try_commit_quantum("d1", 17, 5),
            Err(QuantumCommitError::ExceedsQuantumBudget)
        );
    }

    #[test]
    fn revoke_stops_active_work_at_the_next_committed_boundary() {
        let mut scheduler = DecodeScheduler::shipped_embed_load_v1();
        scheduler.admit_decode(op("d1", 0, 64));
        scheduler.arbitrate(0).expect("Decode dispatches");
        assert_eq!(
            scheduler.request_revoke("d1", 1),
            CancelResult::DeferredToBoundary
        );
        scheduler
            .try_commit_quantum("d1", 16, 2)
            .expect("the already-running quantum can commit");
        assert_eq!(
            scheduler.evaluate_boundary("d1", BoundaryKind::Final(FinishReason::StopToken), 2),
            BoundaryOutcome::Revoked
        );
        scheduler.requeue_continuation("d1");
        assert_eq!(scheduler.queued_count_for(QueueClass::Decode), 0);
    }

    #[test]
    fn interactive_embed_and_rerank_service_remain_available_while_decode_is_resident() {
        let mut scheduler = DecodeScheduler::shipped_embed_load_v1();
        scheduler.admit_decode(op("d1", 0, 256));
        scheduler.arbitrate(0).expect("initial Decode dispatch");
        scheduler
            .try_commit_quantum("d1", 16, 1)
            .expect("first Decode boundary");
        scheduler.requeue_continuation("d1");
        scheduler.admit_embed("embed-1", 1);
        scheduler.admit_rerank("rerank-1", 1);

        let mut interactive_dispatches = 0;
        for tick in 2..8 {
            let arbitration = scheduler.arbitrate(tick).expect("work remains runnable");
            match arbitration.selected {
                QueueClass::Decode => {
                    let committed =
                        scheduler.op("d1").expect("Decode exists").committed_tokens + 16;
                    scheduler
                        .try_commit_quantum("d1", committed, tick)
                        .expect("Decode quantum is bounded");
                    scheduler.requeue_continuation("d1");
                }
                QueueClass::Interactive => {
                    interactive_dispatches += 1;
                    let op_id = arbitration.op_id.expect("service operation");
                    scheduler.record_execution(&op_id, tick + 3, Some(1));
                }
                QueueClass::Bulk | QueueClass::Control => {
                    panic!("no bulk or control work admitted")
                }
            }
        }
        assert!(
            interactive_dispatches > 0,
            "interactive service must receive fair turns"
        );
        assert!(scheduler
            .measurements()
            .queue_execution
            .iter()
            .any(|sample| {
                sample.queue_identifier == "Interactive" && sample.execution_ms.is_some()
            }));
    }

    #[test]
    fn shipped_embed_load_configuration_exposes_a_failing_slo_for_errors() {
        assert_eq!(SHIPPED_EMBED_LOAD_V1.profile_id, "embed-load-v1");
        assert_eq!(SHIPPED_EMBED_LOAD_V1.decode_context_ceiling_tokens, 32_768);
        assert_eq!(SHIPPED_EMBED_LOAD_V1.embed_p95_slo_ms, 150);
        let mut scheduler = DecodeScheduler::shipped_embed_load_v1();
        for _ in 0..20 {
            scheduler
                .record_embed_service_outcome(EmbedServiceOutcome::Completed { latency_ms: 150 });
        }
        assert_eq!(scheduler.measurements().completed_embed_p95_ms(), Some(150));
        assert!(scheduler.measurements().embed_load_v1_slo_met());
        scheduler.record_embed_service_outcome(EmbedServiceOutcome::TimedOut);
        assert!(!scheduler.measurements().embed_load_v1_slo_met());
    }

    #[test]
    fn queued_cancellation_removes_immediately_with_zero_latency() {
        let mut scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());
        scheduler.admit_decode(op("d1", 0, 64));
        let result = scheduler.request_cancel("d1", 5);
        assert_eq!(result, CancelResult::RemovedQueued);
        assert!(scheduler.op("d1").is_none());
        assert_eq!(scheduler.measurements().cancellation_latency_ms, vec![0]);
    }

    #[test]
    fn resident_cancellation_defers_to_boundary() {
        let mut scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());
        scheduler.admit_decode(op("d1", 0, 64));
        scheduler.begin_decode_quantum("d1", 0);
        let result = scheduler.request_cancel("d1", 5);
        assert_eq!(result, CancelResult::DeferredToBoundary);
        // At the next progress boundary the cancellation is observed.
        let outcome = scheduler.evaluate_boundary("d1", BoundaryKind::Progress, 6);
        assert_eq!(outcome, BoundaryOutcome::Cancelled);
    }

    #[test]
    fn deadline_removal_while_queued_is_clean() {
        let mut scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());
        let mut d1 = op("d1", 0, 64);
        d1.deadline_at_ms = Some(50);
        scheduler.admit_decode(d1);
        let removed = scheduler.remove_expired_deadlines(60);
        assert_eq!(removed, vec!["d1".to_string()]);
        assert!(scheduler.op("d1").is_none());
        assert_eq!(scheduler.measurements().deadline_latency_ms, vec![10]);
    }

    #[test]
    fn boundary_precedence_terminal_beats_cancellation_and_deadline() {
        let mut scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());
        let mut d1 = op("d1", 0, 64);
        d1.deadline_at_ms = Some(5);
        scheduler.admit_decode(d1);
        scheduler.begin_decode_quantum("d1", 0);
        scheduler.request_cancel("d1", 4);
        // A terminal completion wins over both a pending cancellation and an expired
        // deadline; the completed result is not retroactively failed.
        let outcome =
            scheduler.evaluate_boundary("d1", BoundaryKind::Final(FinishReason::StopToken), 10);
        assert_eq!(outcome, BoundaryOutcome::Completed(FinishReason::StopToken));
    }

    #[test]
    fn boundary_precedence_cancellation_beats_deadline_at_progress() {
        let mut scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());
        let mut d1 = op("d1", 0, 64);
        d1.deadline_at_ms = Some(5);
        scheduler.admit_decode(d1);
        scheduler.begin_decode_quantum("d1", 0);
        scheduler.request_cancel("d1", 4);
        // At a non-terminal boundary the binding order is cancellation before
        // deadline, so cancellation is observed first.
        let outcome = scheduler.evaluate_boundary("d1", BoundaryKind::Progress, 10);
        assert_eq!(outcome, BoundaryOutcome::Cancelled);
    }

    #[test]
    fn embedding_gets_fair_share_under_mixed_load() {
        // With decode_weight=4 and interactive_weight=1, embedding (Interactive)
        // should receive roughly 1/5 of fair-cycle dispatches when both classes are
        // continuously runnable and nothing ages. This bounds embedding regression.
        let config = DecodeSchedulerConfig {
            decode_aging_window_ms: 1_000_000, // never age; force the fair cycle
            ..DecodeSchedulerConfig::default()
        };
        let mut scheduler = DecodeScheduler::new(config);
        scheduler.admit_decode(op("d1", 0, 1_000_000));
        for i in 0..100 {
            scheduler.admit_other(QueueClass::Interactive, format!("e{i}"));
        }
        let mut interactive_wins = 0u32;
        let mut decode_wins = 0u32;
        for tick in 0..500u64 {
            if let Some(arbitration) = scheduler.arbitrate(tick) {
                match arbitration.selected {
                    QueueClass::Interactive => {
                        interactive_wins += 1;
                        scheduler.admit_other(QueueClass::Interactive, format!("r{tick}"));
                    }
                    QueueClass::Decode => {
                        decode_wins += 1;
                        scheduler.commit_quantum("d1", decode_wins, tick);
                        scheduler.requeue_continuation("d1");
                    }
                    _ => {}
                }
            }
        }
        let total = interactive_wins + decode_wins;
        let interactive_share = interactive_wins as f64 / total as f64;
        // Expected ~0.20; assert it is meaningfully present and bounded so embedding
        // is neither starved nor favored beyond its weight.
        assert!(
            (0.10..=0.35).contains(&interactive_share),
            "interactive share {interactive_share} outside fair bounds"
        );
    }

    // -- quantum sequencing protocol --

    #[test]
    fn start_authorizes_min_n_max_tokens() {
        let mut protocol = GenerationProtocol::new("gen-1", 16, 40);
        assert_eq!(protocol.authorize_start(1).expect("starts"), 16);
        let short = GenerationProtocol::new("gen-2", 16, 10);
        let mut short = short;
        assert_eq!(short.authorize_start(1).expect("starts"), 10);
    }

    #[test]
    fn progress_sequence_must_increment_by_one() {
        let mut protocol = GenerationProtocol::new("gen-1", 16, 64);
        protocol.authorize_start(1).expect("starts");
        protocol.receive_progress(1, 1, 16).expect("first progress");
        // Skipping sequence 2 is a mismatch.
        assert_eq!(
            protocol.receive_progress(1, 3, 48),
            Err(OwnedDecodeError::ProtocolMismatch)
        );
        // Repeating sequence 1 is a mismatch.
        assert_eq!(
            protocol.receive_progress(1, 1, 32),
            Err(OwnedDecodeError::ProtocolMismatch)
        );
        protocol
            .receive_progress(1, 2, 32)
            .expect("second progress");
        assert_eq!(protocol.last_completed_sequence(), 2);
    }

    #[test]
    fn committed_count_must_advance_and_not_exceed_max() {
        let mut protocol = GenerationProtocol::new("gen-1", 16, 32);
        protocol.authorize_start(1).expect("starts");
        protocol.receive_progress(1, 1, 16).expect("first");
        // Non-advancing count is a mismatch.
        assert_eq!(
            protocol.receive_progress(1, 2, 16),
            Err(OwnedDecodeError::ProtocolMismatch)
        );
        // Exceeding max_tokens is a mismatch.
        assert_eq!(
            protocol.receive_progress(1, 2, 48),
            Err(OwnedDecodeError::ProtocolMismatch)
        );
    }

    #[test]
    fn continue_budget_is_bounded_by_n_and_remaining() {
        let mut protocol = GenerationProtocol::new("gen-1", 16, 40);
        protocol.authorize_start(1).expect("starts");
        protocol.receive_progress(1, 1, 16).expect("first");
        let frame = protocol.next_continue().expect("continues");
        assert_eq!(frame.next_token_budget, 16);
        assert_eq!(frame.expected_sequence, 2);
        protocol.receive_progress(1, 2, 32).expect("second");
        // Remaining is 8, below N, so the budget truncates to the remainder.
        let frame = protocol.next_continue().expect("continues");
        assert_eq!(frame.next_token_budget, 8);
        protocol.receive_progress(1, 3, 40).expect("third");
        // Budget exhausted: no further continuation.
        assert!(protocol.next_continue().is_none());
    }

    #[test]
    fn chain_k_16_still_yields_at_the_16_token_quantum_boundary() {
        // A fused K=16 submission is one scheduler quantum, not two hidden
        // quanta. The supervisor therefore sees exactly 16 committed tokens and
        // authorizes the next quantum with the normal N=16 budget.
        let mut protocol = GenerationProtocol::new("gen-chain-16", 16, 32);
        protocol.authorize_start(1).expect("starts");
        protocol
            .receive_progress(1, 1, 16)
            .expect("K=16 progress accounts for one quantum");
        let continuation = protocol.next_continue().expect("second quantum");
        assert_eq!(continuation.next_token_budget, 16);
        assert_eq!(continuation.expected_sequence, 2);
    }

    #[test]
    fn stale_worker_generation_is_a_mismatch() {
        let mut protocol = GenerationProtocol::new("gen-1", 16, 64);
        protocol.authorize_start(7).expect("starts");
        assert_eq!(
            protocol.receive_progress(6, 1, 16),
            Err(OwnedDecodeError::ProtocolMismatch)
        );
        protocol
            .receive_progress(7, 1, 16)
            .expect("current generation accepted");
    }

    #[test]
    fn continuation_after_cleanup_is_a_mismatch() {
        let mut protocol = GenerationProtocol::new("gen-1", 16, 64);
        protocol.authorize_start(1).expect("starts");
        protocol.close();
        assert_eq!(
            protocol.receive_progress(1, 1, 16),
            Err(OwnedDecodeError::ProtocolMismatch)
        );
    }

    #[test]
    fn validate_continue_rejects_wrong_budget() {
        let mut protocol = GenerationProtocol::new("gen-1", 16, 64);
        protocol.authorize_start(1).expect("starts");
        protocol.receive_progress(1, 1, 16).expect("first");
        let good = protocol.next_continue().expect("continues");
        protocol.validate_continue(&good).expect("valid");
        let bad = ContinueFrame {
            generation_id: "gen-1".to_string(),
            expected_sequence: 2,
            next_token_budget: 4,
        };
        assert_eq!(
            protocol.validate_continue(&bad),
            Err(OwnedDecodeError::ProtocolMismatch)
        );
    }
}
