//! Terminal-control boundary evaluation.
//!
//! Every progress response and every successful final response is a
//! terminal-control evaluation boundary. The module compares the recorded
//! cancellation time, the request deadline, and the boundary-observation time,
//! and decides whether to accept the payload, cancel, or return the deadline
//! error.
//!
//! Precedence (resolution r2 #4 reconciled with `error_contract` and
//! `worker_protocol_contract`): terminal completion > cancellation > deadline.
//! 1. A natural terminal completion (`stop_token`, `max_tokens`,
//!    `grammar_complete`) wins outright. A pending cancellation is acknowledged
//!    as a no-op and a deadline that expired during that final quantum does not
//!    retroactively fail it.
//! 2. At a boundary with no terminal completion, cancellation is evaluated
//!    first: if it was recorded before or at the boundary, the payload is
//!    suppressed and `cancelled` is returned after state cleanup. Cancellation
//!    therefore wins when both cancellation and deadline are pending at the
//!    same boundary — the deadline error is reserved for operations that ran
//!    out of time without the caller abandoning them.
//! 3. Otherwise, if the deadline expired before or at the boundary, the
//!    payload is suppressed and the bound deadline error is returned.
//! 4. Otherwise the payload is accepted.
//!
//! Timestamps are monotonic milliseconds; fixtures drive them deterministically.

use crate::protocol::FinishReason;

/// A monotonic timestamp in milliseconds.
pub type Timestamp = u64;

/// The terminal-control inputs observed at a boundary.
#[derive(Clone, Copy, Debug, Default)]
pub struct BoundaryInputs {
    /// The finish reason if the quantum naturally completed; `None` for a
    /// non-final progress boundary.
    pub completion: Option<FinishReason>,
    /// When cancellation was recorded, if it was.
    pub cancel_recorded_at: Option<Timestamp>,
    /// The absolute request deadline, if any.
    pub deadline_at: Option<Timestamp>,
    /// When the module observed this boundary.
    pub observed_at: Timestamp,
}

/// The outcome of evaluating a terminal-control boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryDecision {
    /// Accept a natural terminal completion and surface its finish reason.
    AcceptCompletion(FinishReason),
    /// Accept a non-final progress frame; the supervisor should continue.
    AcceptProgress,
    /// Suppress the payload and return the bound deadline error.
    DeadlineExceeded,
    /// Suppress the payload and return `cancelled` after state cleanup.
    Cancelled,
}

impl BoundaryDecision {
    /// Whether the successful payload is suppressed (no generated IDs or text
    /// are exposed to the caller).
    #[must_use]
    pub const fn suppresses_payload(self) -> bool {
        matches!(self, Self::DeadlineExceeded | Self::Cancelled)
    }
}

/// Evaluate a terminal-control boundary.
#[must_use]
pub fn evaluate_boundary(inputs: BoundaryInputs) -> BoundaryDecision {
    // 1. A natural terminal completion wins over a pending cancellation or a
    //    deadline that expired during the final quantum.
    if let Some(reason) = inputs.completion {
        if reason.is_terminal_completion() {
            return BoundaryDecision::AcceptCompletion(reason);
        }
        // A worker-reported cancelled finish is a cancellation outcome.
        // Intentional cross-crate divergence: the S5 module-side model records
        // the same boundary as `Completed(Cancelled)` for measurement, while
        // this supervisor classifies it as the cancellation decision with
        // payload suppression. The observable outcome is identical in both
        // (finish reason `cancelled`, no payload exposed); only the internal
        // classification differs.
        if reason == FinishReason::Cancelled {
            return BoundaryDecision::Cancelled;
        }
    }

    // 2. Cancellation recorded before or at the boundary. The binding
    //    precedence order is terminal completion > cancellation > deadline
    //    (spec resolutions round 2, #4): a caller who already asked to abandon
    //    the operation receives `cancelled` even when the deadline also expired
    //    during the same quantum — the deadline error is reserved for
    //    operations that ran out of time without the caller abandoning them.
    if let Some(cancelled_at) = inputs.cancel_recorded_at {
        if cancelled_at <= inputs.observed_at {
            return BoundaryDecision::Cancelled;
        }
    }

    // 3. Deadline expiry at a non-terminal boundary with no pending cancellation.
    if let Some(deadline) = inputs.deadline_at {
        if deadline <= inputs.observed_at {
            return BoundaryDecision::DeadlineExceeded;
        }
    }

    // 4. Accept the progress payload.
    BoundaryDecision::AcceptProgress
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs() -> BoundaryInputs {
        BoundaryInputs {
            completion: None,
            cancel_recorded_at: None,
            deadline_at: None,
            observed_at: 100,
        }
    }

    #[test]
    fn progress_with_no_controls_is_accepted() {
        assert_eq!(
            evaluate_boundary(inputs()),
            BoundaryDecision::AcceptProgress
        );
    }

    #[test]
    fn terminal_completion_wins_over_cancel_and_deadline() {
        let mut i = inputs();
        i.completion = Some(FinishReason::StopToken);
        i.cancel_recorded_at = Some(50);
        i.deadline_at = Some(80); // expired before the boundary at 100
        assert_eq!(
            evaluate_boundary(i),
            BoundaryDecision::AcceptCompletion(FinishReason::StopToken)
        );
    }

    #[test]
    fn cancellation_wins_over_deadline_when_both_pending() {
        let mut i = inputs();
        i.cancel_recorded_at = Some(60);
        i.deadline_at = Some(90); // both before observed_at=100
        assert_eq!(evaluate_boundary(i), BoundaryDecision::Cancelled);
    }

    #[test]
    fn cancellation_applies_when_no_deadline_expired() {
        let mut i = inputs();
        i.cancel_recorded_at = Some(60);
        i.deadline_at = Some(200); // not yet expired
        assert_eq!(evaluate_boundary(i), BoundaryDecision::Cancelled);
    }

    #[test]
    fn deadline_at_exact_boundary_is_expired() {
        let mut i = inputs();
        i.deadline_at = Some(100); // == observed_at
        assert_eq!(evaluate_boundary(i), BoundaryDecision::DeadlineExceeded);
    }

    #[test]
    fn cancellation_recorded_after_boundary_is_ignored() {
        let mut i = inputs();
        i.cancel_recorded_at = Some(150); // after observed_at=100
        assert_eq!(evaluate_boundary(i), BoundaryDecision::AcceptProgress);
    }

    #[test]
    fn worker_reported_cancelled_finish_is_cancellation() {
        let mut i = inputs();
        i.completion = Some(FinishReason::Cancelled);
        assert_eq!(evaluate_boundary(i), BoundaryDecision::Cancelled);
        assert!(evaluate_boundary(i).suppresses_payload());
    }
}
