use serde::{Deserialize, Serialize};

use crate::StableError;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueueClass {
    Interactive,
    Bulk,
    Control,
}

pub trait QueueClassMarker {
    const CLASS: QueueClass;
}

pub struct InteractiveQueue;
pub struct BulkQueue;
pub struct ControlQueue;

impl QueueClassMarker for InteractiveQueue {
    const CLASS: QueueClass = QueueClass::Interactive;
}

impl QueueClassMarker for BulkQueue {
    const CLASS: QueueClass = QueueClass::Bulk;
}

impl QueueClassMarker for ControlQueue {
    const CLASS: QueueClass = QueueClass::Control;
}

pub trait Clock {
    fn now_ms(&self) -> u64;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdmissionRequest {
    pub queue_class: QueueClass,
    pub deadline_ms: Option<u64>,
    pub max_queue_ms: u64,
    pub request_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaneBudgetSnapshot {
    pub queued_bytes: u64,
    pub in_flight_bytes: u64,
    pub byte_budget: u64,
    pub predicted_start_delay_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "decision")]
pub enum AdmissionDecision {
    Accept(AcceptedAdmission),
    Reject(RejectedAdmission),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcceptedAdmission {
    pub queue_class: QueueClass,
    pub admitted_at_ms: u64,
    pub predicted_start_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RejectedAdmission {
    pub error: StableError,
    pub reason: String,
}

pub fn decide_admission<C: Clock>(
    clock: &C,
    request: &AdmissionRequest,
    lane: &LaneBudgetSnapshot,
) -> AdmissionDecision {
    let admitted_at_ms = clock.now_ms();
    let predicted_start_ms = admitted_at_ms.saturating_add(lane.predicted_start_delay_ms);
    let total_bytes = lane
        .queued_bytes
        .saturating_add(lane.in_flight_bytes)
        .saturating_add(request.request_bytes);

    if total_bytes > lane.byte_budget {
        return AdmissionDecision::Reject(RejectedAdmission {
            error: StableError::queue_full(None),
            reason: format!(
                "request would use {total_bytes} bytes against lane budget {}",
                lane.byte_budget
            ),
        });
    }

    if lane.predicted_start_delay_ms > request.max_queue_ms {
        return AdmissionDecision::Reject(RejectedAdmission {
            error: StableError::deadline_exceeded(),
            reason: format!(
                "predicted start delay {}ms exceeds max_queue_ms {}ms",
                lane.predicted_start_delay_ms, request.max_queue_ms
            ),
        });
    }

    if let Some(deadline_ms) = request.deadline_ms {
        if predicted_start_ms > deadline_ms {
            return AdmissionDecision::Reject(RejectedAdmission {
                error: StableError::deadline_exceeded(),
                reason: format!(
                    "predicted start {}ms exceeds deadline {}ms",
                    predicted_start_ms, deadline_ms
                ),
            });
        }
    }

    AdmissionDecision::Accept(AcceptedAdmission {
        queue_class: request.queue_class,
        admitted_at_ms,
        predicted_start_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeClock {
        now_ms: u64,
    }

    impl Clock for FakeClock {
        fn now_ms(&self) -> u64 {
            self.now_ms
        }
    }

    #[test]
    fn admission_accepts_when_budget_and_deadline_are_meetable() {
        let clock = FakeClock { now_ms: 1_000 };
        let request = AdmissionRequest {
            queue_class: QueueClass::Interactive,
            deadline_ms: Some(1_250),
            max_queue_ms: 300,
            request_bytes: 128,
        };
        let lane = LaneBudgetSnapshot {
            queued_bytes: 256,
            in_flight_bytes: 256,
            byte_budget: 1_024,
            predicted_start_delay_ms: 200,
        };

        assert_eq!(
            decide_admission(&clock, &request, &lane),
            AdmissionDecision::Accept(AcceptedAdmission {
                queue_class: QueueClass::Interactive,
                admitted_at_ms: 1_000,
                predicted_start_ms: 1_200,
            })
        );
    }

    #[test]
    fn admission_rejects_when_byte_budget_is_exhausted() {
        let clock = FakeClock { now_ms: 10 };
        let request = AdmissionRequest {
            queue_class: QueueClass::Bulk,
            deadline_ms: None,
            max_queue_ms: 500,
            request_bytes: 300,
        };
        let lane = LaneBudgetSnapshot {
            queued_bytes: 500,
            in_flight_bytes: 300,
            byte_budget: 1_000,
            predicted_start_delay_ms: 50,
        };

        let AdmissionDecision::Reject(rejection) = decide_admission(&clock, &request, &lane) else {
            panic!("expected queue_full rejection");
        };
        assert_eq!(rejection.error.code, crate::StableErrorCode::QueueFull);
        assert!(rejection.reason.contains("bytes"));
    }

    #[test]
    fn admission_rejects_when_predicted_wait_exceeds_max_queue() {
        let clock = FakeClock { now_ms: 10 };
        let request = AdmissionRequest {
            queue_class: QueueClass::Control,
            deadline_ms: Some(500),
            max_queue_ms: 100,
            request_bytes: 1,
        };
        let lane = LaneBudgetSnapshot {
            queued_bytes: 0,
            in_flight_bytes: 0,
            byte_budget: 10,
            predicted_start_delay_ms: 101,
        };

        let AdmissionDecision::Reject(rejection) = decide_admission(&clock, &request, &lane) else {
            panic!("expected deadline rejection");
        };
        assert_eq!(rejection.error.code, crate::StableErrorCode::DeadlineExceeded);
        assert!(rejection.reason.contains("max_queue_ms"));
    }

    #[test]
    fn admission_rejects_when_deadline_arrives_before_predicted_start() {
        let clock = FakeClock { now_ms: 1_000 };
        let request = AdmissionRequest {
            queue_class: QueueClass::Interactive,
            deadline_ms: Some(1_050),
            max_queue_ms: 500,
            request_bytes: 1,
        };
        let lane = LaneBudgetSnapshot {
            queued_bytes: 0,
            in_flight_bytes: 0,
            byte_budget: 10,
            predicted_start_delay_ms: 60,
        };

        let AdmissionDecision::Reject(rejection) = decide_admission(&clock, &request, &lane) else {
            panic!("expected deadline rejection");
        };
        assert_eq!(rejection.error.code, crate::StableErrorCode::DeadlineExceeded);
        assert!(rejection.reason.contains("deadline"));
    }
}
