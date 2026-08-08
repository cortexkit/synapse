//! The dedicated DECODE scheduler class and its quantum-sequencing protocol.
//!
//! Owned generation does not reuse the Interactive or Bulk classes: it has a
//! dedicated [`QueueClass::Decode`] with its own admission, fair-cycle weight,
//! and aging window. This module models that scheduler in hardware-independent
//! form so its mechanism — Control precedence, weighted boundary arbitration,
//! oldest-anchor DECODE ordering, module-held permit release and reacquisition,
//! FIFO continuation, queued cancellation and deadline removal, and N-token
//! quantum sequencing — can be exercised and measured without Metal.
//!
//! Two binding decisions from the specification's resolutions shape the model:
//! - *Yield-on-contention*: at a quantum boundary the module releases the decode
//!   permit whenever weighted fair-cycle arbitration selects queued work from any
//!   other class; the permit is retained only when DECODE is the sole runnable
//!   class.
//! - *Aggregate aging guarantee*: within DECODE, ordering is FIFO by admission
//!   and the testable guarantee is that in every aging window in which at least
//!   one DECODE operation is continuously runnable, at least one DECODE quantum
//!   commits (rather than a per-operation one-window bound).

use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::owned_decode_routing::error::OwnedDecodeError;

/// The scheduler queue classes. Owned generation uses [`QueueClass::Decode`];
/// llama fallback keeps its existing Interactive/Bulk class and consumes no
/// DECODE accounting. Serialized in snake_case, so `Decode` is `"decode"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueueClass {
    Control,
    Interactive,
    Decode,
    Bulk,
}

impl QueueClass {
    /// The wire serialization. `Decode` serializes as `"decode"`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Control => "control",
            Self::Interactive => "interactive",
            Self::Decode => "decode",
            Self::Bulk => "bulk",
        }
    }

    /// Parse a wire serialization back into a class.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "control" => Ok(Self::Control),
            "interactive" => Ok(Self::Interactive),
            "decode" => Ok(Self::Decode),
            "bulk" => Ok(Self::Bulk),
            other => Err(format!("unknown queue class '{other}'")),
        }
    }
}

/// Scheduler runtime configuration. The five runtime-effective scheduler fields
/// from `decode-sched-manifest-v1` are `production_n`, `decode_weight`,
/// `decode_aging_window_ms`, plus the yield-policy and progress-protocol
/// revisions (carried as strings for identity). Interactive and Bulk weights are
/// local modeling inputs and do not enter runtime identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeSchedulerConfig {
    /// Exactly one production N from `{8,16,32}`; the maximum committed tokens per
    /// quantum. N=1 is prohibited.
    pub production_n: u32,
    /// DECODE weight in the weighted fair cycle.
    pub decode_weight: u32,
    /// Interactive (embedding) weight in the weighted fair cycle.
    pub interactive_weight: u32,
    /// Bulk weight in the weighted fair cycle.
    pub bulk_weight: u32,
    /// The DECODE aging window in milliseconds.
    pub decode_aging_window_ms: u64,
}

impl Default for DecodeSchedulerConfig {
    fn default() -> Self {
        Self {
            production_n: 16,
            decode_weight: 4,
            interactive_weight: 1,
            bulk_weight: 1,
            decode_aging_window_ms: 250,
        }
    }
}

impl DecodeSchedulerConfig {
    fn weight(&self, class: QueueClass) -> u32 {
        match class {
            QueueClass::Decode => self.decode_weight,
            QueueClass::Interactive => self.interactive_weight,
            QueueClass::Bulk => self.bulk_weight,
            QueueClass::Control => 0,
        }
    }
}

/// A DECODE operation tracked by the scheduler.
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
    /// True once the operation has dispatched at least once and holds resident
    /// continuation state awaiting its next quantum.
    pub resident: bool,
    /// When a cancellation was recorded (evaluated at the next boundary).
    pub cancelled_at_ms: Option<u64>,
    /// Absolute deadline; expiry is evaluated at boundaries and while queued.
    pub deadline_at_ms: Option<u64>,
}

/// The outcome of a boundary evaluation, applying the binding precedence order
/// terminal completion > cancellation > deadline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryOutcome {
    /// The quantum completed normally; any pending cancellation or deadline is a
    /// no-op and does not retroactively fail the result.
    Completed(FinishReason),
    /// A cancellation recorded before or at this (non-terminal) boundary.
    Cancelled,
    /// A deadline expired before or at this (non-terminal) boundary.
    DeadlineExceeded,
    /// Neither control applies; the operation continues with another quantum.
    Continue,
}

/// External finish reasons (exactly the four the wire contract allows).
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
    /// The module acquired the decode permit for a DECODE dispatch.
    Acquired,
    /// DECODE won again while already holding the permit (no release overhead).
    Retained,
    /// A non-DECODE class won the fair cycle; the decode permit was released.
    Released,
    /// The arbitration did not touch the decode permit.
    Unchanged,
}

/// The result of one boundary arbitration.
#[derive(Clone, Debug, PartialEq)]
pub struct Arbitration {
    /// The class selected for the next dispatch opportunity.
    pub selected: QueueClass,
    /// For a DECODE selection, the selected operation (oldest aging anchor).
    pub op_id: Option<String>,
    /// What happened to the module-held decode permit.
    pub permit_event: PermitEvent,
}

/// Aggregate scheduler measurements, mirroring the `decode-sched-manifest-v1`
/// evidence record fields.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Measurements {
    pub queue_depth_samples: Vec<u32>,
    pub per_op_waiting_ms: Vec<(String, u64)>,
    pub permit_events: Vec<PermitEvent>,
    pub continuation_count: u32,
    pub sequence_traces: Vec<String>,
    pub cancellation_latency_ms: Vec<u64>,
    pub deadline_latency_ms: Vec<u64>,
}

/// The DECODE scheduler. Holds one queue per class, the operation table, the
/// module-held decode permit, the weighted fair-cycle credits, and measurements.
#[derive(Clone, Debug)]
pub struct DecodeScheduler {
    config: DecodeSchedulerConfig,
    control: VecDeque<String>,
    interactive: VecDeque<String>,
    bulk: VecDeque<String>,
    decode: VecDeque<String>,
    ops: BTreeMap<String, DecodeOp>,
    permit_held: bool,
    /// Smooth weighted round-robin credits keyed by class.
    credits: BTreeMap<QueueClass, i64>,
    measurements: Measurements,
}

impl DecodeScheduler {
    pub fn new(config: DecodeSchedulerConfig) -> Self {
        assert!(config.production_n > 1, "N=1 is prohibited");
        Self {
            config,
            control: VecDeque::new(),
            interactive: VecDeque::new(),
            bulk: VecDeque::new(),
            decode: VecDeque::new(),
            ops: BTreeMap::new(),
            permit_held: false,
            credits: BTreeMap::new(),
            measurements: Measurements::default(),
        }
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

    pub fn op(&self, op_id: &str) -> Option<&DecodeOp> {
        self.ops.get(op_id)
    }

    /// Number of queued operations across all classes.
    pub fn queued_count(&self) -> usize {
        self.control.len() + self.interactive.len() + self.bulk.len() + self.decode.len()
    }

    fn queue_mut(&mut self, class: QueueClass) -> &mut VecDeque<String> {
        match class {
            QueueClass::Control => &mut self.control,
            QueueClass::Interactive => &mut self.interactive,
            QueueClass::Decode => &mut self.decode,
            QueueClass::Bulk => &mut self.bulk,
        }
    }

    fn queue(&self, class: QueueClass) -> &VecDeque<String> {
        match class {
            QueueClass::Control => &self.control,
            QueueClass::Interactive => &self.interactive,
            QueueClass::Decode => &self.decode,
            QueueClass::Bulk => &self.bulk,
        }
    }

    /// Admit a new DECODE operation. Its aging anchor starts at admission time.
    pub fn admit_decode(&mut self, op: DecodeOp) {
        assert_eq!(
            op.anchor_ms, op.admitted_at_ms,
            "anchor starts at admission"
        );
        self.decode.push_back(op.op_id.clone());
        self.ops.insert(op.op_id.clone(), op);
        self.sample_depth();
    }

    /// Admit a non-DECODE operation (Control, Interactive, or Bulk).
    pub fn admit_other(&mut self, class: QueueClass, op_id: impl Into<String>) {
        assert_ne!(class, QueueClass::Decode, "use admit_decode for DECODE");
        let op_id = op_id.into();
        self.queue_mut(class).push_back(op_id.clone());
        self.ops.insert(
            op_id.clone(),
            DecodeOp {
                op_id,
                generation_id: String::new(),
                admitted_at_ms: 0,
                anchor_ms: 0,
                committed_tokens: 0,
                max_tokens: 0,
                resident: false,
                cancelled_at_ms: None,
                deadline_at_ms: None,
            },
        );
        self.sample_depth();
    }

    fn sample_depth(&mut self) {
        self.measurements
            .queue_depth_samples
            .push(self.queued_count() as u32);
    }

    /// The oldest continuously runnable DECODE operation: the one with the
    /// smallest aging anchor, ties broken by admission order (FIFO).
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

    /// The age in milliseconds of the oldest-anchor DECODE operation.
    fn oldest_decode_age_ms(&self, now_ms: u64) -> Option<u64> {
        self.select_decode()
            .and_then(|op_id| self.ops.get(&op_id))
            .map(|op| now_ms.saturating_sub(op.anchor_ms))
    }

    fn runnable_classes(&self) -> Vec<QueueClass> {
        [
            QueueClass::Interactive,
            QueueClass::Decode,
            QueueClass::Bulk,
        ]
        .into_iter()
        .filter(|class| !self.queue(*class).is_empty())
        .collect()
    }

    /// Smooth weighted round-robin selection over the runnable classes. Returns
    /// the class that wins the next fair-cycle opportunity.
    fn fair_cycle_pick(&mut self, runnable: &[QueueClass]) -> QueueClass {
        // Add each runnable class's weight to its credit, then pick the class with
        // the largest credit and subtract the total runnable weight from it. Over a
        // stable runnable set this distributes selections in proportion to weights.
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
        if let Some(credit) = self.credits.get_mut(&picked) {
            *credit -= total;
        }
        picked
    }

    /// Arbitrate the next dispatch opportunity at a quantum boundary.
    ///
    /// Precedence: runnable Control first; else an aged DECODE operation; else the
    /// weighted fair cycle over runnable Interactive/DECODE/Bulk. The decode permit
    /// is released whenever a non-DECODE class wins while DECODE is runnable, and
    /// retained when DECODE is the sole runnable class.
    pub fn arbitrate(&mut self, now_ms: u64) -> Option<Arbitration> {
        self.sample_depth();

        // 1. Control has strict precedence.
        if !self.control.is_empty() {
            let op_id = self.control.pop_front();
            return Some(Arbitration {
                selected: QueueClass::Control,
                op_id,
                permit_event: self.set_permit(false),
            });
        }

        let runnable = self.runnable_classes();
        if runnable.is_empty() {
            return None;
        }

        // 2. An aged DECODE operation receives the next dispatch opportunity.
        let decode_runnable = runnable.contains(&QueueClass::Decode);
        if decode_runnable {
            if let Some(age) = self.oldest_decode_age_ms(now_ms) {
                if age >= self.config.decode_aging_window_ms {
                    let op_id = self.select_decode();
                    return Some(Arbitration {
                        selected: QueueClass::Decode,
                        op_id,
                        permit_event: self.set_permit(true),
                    });
                }
            }
        }

        // 3. DECODE is the sole runnable class: consecutive quanta without release.
        if runnable == [QueueClass::Decode] {
            let op_id = self.select_decode();
            return Some(Arbitration {
                selected: QueueClass::Decode,
                op_id,
                permit_event: self.set_permit(true),
            });
        }

        // 4. Weighted fair cycle over the runnable classes.
        let picked = self.fair_cycle_pick(&runnable);
        if picked == QueueClass::Decode {
            let op_id = self.select_decode();
            Some(Arbitration {
                selected: QueueClass::Decode,
                op_id,
                permit_event: self.set_permit(true),
            })
        } else {
            let op_id = self.queue_mut(picked).pop_front();
            Some(Arbitration {
                selected: picked,
                op_id,
                // Yield-on-contention: a non-DECODE win releases the decode permit
                // whenever DECODE was runnable.
                permit_event: self.set_permit(!decode_runnable),
            })
        }
    }

    /// Set the module-held decode permit, returning the lifecycle event.
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

    /// Record that the selected DECODE operation began a quantum: pop it from the
    /// head of the DECODE queue if present (a resident continuation may not be at
    /// the head), mark it resident, and record per-operation waiting time.
    pub fn begin_decode_quantum(&mut self, op_id: &str, now_ms: u64) {
        if let Some(position) = self.decode.iter().position(|id| id == op_id) {
            self.decode.remove(position);
        }
        if let Some(op) = self.ops.get_mut(op_id) {
            let waiting = now_ms.saturating_sub(op.anchor_ms);
            self.measurements
                .per_op_waiting_ms
                .push((op_id.to_string(), waiting));
            op.resident = true;
        }
        self.sample_depth();
    }

    /// Re-enqueue a resident continuation after a quantum boundary. It competes
    /// through the same DECODE queue, fair-cycle, and aging machinery as a new
    /// admission. The aging anchor was already advanced by [`Self::commit_quantum`].
    pub fn requeue_continuation(&mut self, op_id: &str) {
        if !self.decode.iter().any(|id| id == op_id) {
            self.decode.push_back(op_id.to_string());
        }
        self.measurements.continuation_count += 1;
        self.sample_depth();
    }

    /// Record that a DECODE operation committed a quantum: advance the aging
    /// anchor to the most recent committed-token time and update the committed
    /// count. Updating the anchor prevents the operation from retaining aged
    /// priority over an older operation.
    pub fn commit_quantum(&mut self, op_id: &str, committed_tokens: u32, now_ms: u64) {
        if let Some(op) = self.ops.get_mut(op_id) {
            op.committed_tokens = committed_tokens;
            op.anchor_ms = now_ms;
        }
    }

    /// Remove a completed or terminated operation from the scheduler entirely.
    pub fn remove_op(&mut self, op_id: &str) {
        self.ops.remove(op_id);
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

    /// Request cancellation of an operation. If it is still queued (not resident
    /// in an active quantum), it is removed immediately with zero cancellation
    /// latency and consumes no crash budget. Otherwise the cancellation is recorded
    /// for evaluation at the next boundary.
    pub fn request_cancel(&mut self, op_id: &str, now_ms: u64) -> CancelResult {
        let queued = self.decode.iter().any(|id| id == op_id)
            || self.interactive.iter().any(|id| id == op_id)
            || self.bulk.iter().any(|id| id == op_id)
            || self.control.iter().any(|id| id == op_id);
        let is_resident = self.ops.get(op_id).map(|op| op.resident).unwrap_or(false);
        if queued && !is_resident {
            self.remove_op(op_id);
            self.measurements.cancellation_latency_ms.push(0);
            CancelResult::RemovedQueued
        } else {
            if let Some(op) = self.ops.get_mut(op_id) {
                op.cancelled_at_ms = Some(now_ms);
            }
            CancelResult::DeferredToBoundary
        }
    }

    /// Remove queued operations whose deadline has expired. Returns the removed op
    /// IDs. Deadline expiry while queued is a clean removal: no crash-budget
    /// consumption and no dispatch.
    pub fn remove_expired_deadlines(&mut self, now_ms: u64) -> Vec<String> {
        let expired: Vec<String> = self
            .ops
            .values()
            .filter(|op| {
                op.deadline_at_ms
                    .map(|deadline| deadline <= now_ms)
                    .unwrap_or(false)
            })
            .filter(|op| {
                // Only remove operations still waiting in a queue.
                self.decode.iter().any(|id| *id == op.op_id)
                    || self.interactive.iter().any(|id| *id == op.op_id)
                    || self.bulk.iter().any(|id| *id == op.op_id)
                    || self.control.iter().any(|id| *id == op.op_id)
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

    /// Evaluate a boundary for an operation, applying the binding precedence order
    /// terminal completion > cancellation > deadline.
    pub fn evaluate_boundary(
        &self,
        op_id: &str,
        boundary: BoundaryKind,
        now_ms: u64,
    ) -> BoundaryOutcome {
        let op = match self.ops.get(op_id) {
            Some(op) => op,
            None => return BoundaryOutcome::Continue,
        };
        match boundary {
            // A terminal completion always wins; pending cancellation is a no-op and
            // a deadline that expired during the final quantum does not retroactively
            // fail the completed result.
            //
            // A worker-reported `Final(Cancelled)` is recorded here as
            // `Completed(Cancelled)` for measurement. Intentional cross-crate
            // divergence: the S3 supervisor classifies the same frame as its
            // cancellation decision with payload suppression. The observable
            // outcome is identical in both (finish reason `cancelled`, no
            // payload); only the internal classification differs.
            BoundaryKind::Final(reason) => BoundaryOutcome::Completed(reason),
            BoundaryKind::Progress => {
                if let Some(cancelled_at) = op.cancelled_at_ms {
                    if cancelled_at <= now_ms {
                        return BoundaryOutcome::Cancelled;
                    }
                }
                if let Some(deadline) = op.deadline_at_ms {
                    if deadline <= now_ms {
                        return BoundaryOutcome::DeadlineExceeded;
                    }
                }
                BoundaryOutcome::Continue
            }
        }
    }
}

/// The result of a cancellation request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CancelResult {
    /// The operation was queued and removed immediately (zero latency, no budget).
    RemovedQueued,
    /// The operation is resident; cancellation is evaluated at the next boundary.
    DeferredToBoundary,
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
        assert!(production_n > 1, "N=1 is prohibited");
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
    fn decode_serializes_as_decode() {
        assert_eq!(QueueClass::Decode.as_str(), "decode");
        assert_eq!(QueueClass::parse("decode"), Ok(QueueClass::Decode));
        let encoded = serde_json::to_string(&QueueClass::Decode).expect("serializes");
        assert_eq!(encoded, "\"decode\"");
    }

    #[test]
    fn control_has_strict_precedence() {
        let mut scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());
        scheduler.admit_decode(op("d1", 0, 64));
        scheduler.admit_other(QueueClass::Control, "c1");
        let arbitration = scheduler.arbitrate(0).expect("arbitrates");
        assert_eq!(arbitration.selected, QueueClass::Control);
        assert_eq!(arbitration.op_id.as_deref(), Some("c1"));
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
        scheduler.admit_decode(op("d1", 0, 64));
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
                    scheduler.commit_quantum("d1", 16, tick);
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
    fn aged_decode_preempts_fair_cycle() {
        let config = DecodeSchedulerConfig {
            decode_aging_window_ms: 100,
            ..DecodeSchedulerConfig::default()
        };
        let mut scheduler = DecodeScheduler::new(config);
        scheduler.admit_decode(op("d1", 0, 64));
        // Keep Interactive runnable so the fair cycle would otherwise compete.
        for i in 0..10 {
            scheduler.admit_other(QueueClass::Interactive, format!("e{i}"));
        }
        // Before the aging window elapses, arbitration may pick either class.
        // After it elapses, the aged DECODE operation must be selected.
        let arbitration = scheduler.arbitrate(150).expect("arbitrates");
        assert_eq!(arbitration.selected, QueueClass::Decode);
        assert_eq!(arbitration.op_id.as_deref(), Some("d1"));
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
    fn aggregate_aging_guarantee_one_commit_per_window() {
        // In every aging window where a DECODE op is continuously runnable and
        // Control does not occupy the scheduler, at least one DECODE quantum commits.
        let config = DecodeSchedulerConfig {
            decode_aging_window_ms: 100,
            ..DecodeSchedulerConfig::default()
        };
        let window = config.decode_aging_window_ms;
        let mut scheduler = DecodeScheduler::new(config);
        scheduler.admit_decode(op("d1", 0, 1024));
        for i in 0..50 {
            scheduler.admit_other(QueueClass::Interactive, format!("e{i}"));
        }
        let mut commit_times: Vec<u64> = Vec::new();
        let mut committed = 0u32;
        for tick in 0..1000u64 {
            scheduler.remove_expired_deadlines(tick);
            if let Some(arbitration) = scheduler.arbitrate(tick) {
                match arbitration.selected {
                    QueueClass::Decode => {
                        let op_id = arbitration.op_id.clone().expect("decode op");
                        committed += 16;
                        scheduler.commit_quantum(&op_id, committed, tick);
                        commit_times.push(tick);
                        scheduler.requeue_continuation(&op_id);
                    }
                    QueueClass::Interactive => {
                        scheduler.admit_other(QueueClass::Interactive, format!("refill{tick}"));
                    }
                    _ => {}
                }
            }
        }
        // The gap between consecutive DECODE commits must never exceed the aging
        // window (allowing the first commit to occur within the first window).
        assert!(!commit_times.is_empty(), "decode committed at least once");
        assert!(
            commit_times[0] <= window,
            "first commit within the first window"
        );
        for window_pair in commit_times.windows(2) {
            let gap = window_pair[1] - window_pair[0];
            assert!(gap <= window, "gap {gap} exceeded the aging window");
        }
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
