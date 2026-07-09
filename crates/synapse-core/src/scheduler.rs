use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::StableError;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueueClass {
    Interactive,
    Bulk,
    Control,
}

pub trait Clock {
    fn now_ms(&self) -> u64;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdmissionRequest {
    pub queue_class: QueueClass,
    /// Absolute deadline in the same millisecond epoch as the injected clock.
    pub deadline_ms: Option<u64>,
    pub max_queue_ms: u64,
    pub request_bytes: u64,
    #[serde(default)]
    pub estimated_execution_ms: u64,
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
    pub predicted_finish_ms: u64,
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
    let predicted_finish_ms = predicted_start_ms.saturating_add(request.estimated_execution_ms);
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
        if predicted_finish_ms > deadline_ms {
            return AdmissionDecision::Reject(RejectedAdmission {
                error: StableError::deadline_exceeded(),
                reason: format!(
                    "predicted finish {}ms exceeds deadline {}ms",
                    predicted_finish_ms, deadline_ms
                ),
            });
        }
    }

    AdmissionDecision::Accept(AcceptedAdmission {
        queue_class: request.queue_class,
        admitted_at_ms,
        predicted_start_ms,
        predicted_finish_ms,
    })
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchedulerConfig {
    pub byte_budget: u64,
    pub bulk_quantum_tokens: u64,
    pub interactive_weight: u32,
    pub bulk_weight: u32,
    pub bulk_aging_ms: u64,
    pub max_concurrent_workers: usize,
    pub default_execution_ms: u64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            byte_budget: 64 * 1024 * 1024,
            bulk_quantum_tokens: 2_048,
            interactive_weight: 3,
            bulk_weight: 1,
            bulk_aging_ms: 250,
            max_concurrent_workers: 1,
            default_execution_ms: 25,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkRequest<T> {
    pub queue_class: QueueClass,
    pub deadline_ms: Option<u64>,
    pub max_queue_ms: u64,
    pub request_bytes: u64,
    pub token_cost: u64,
    #[serde(default)]
    pub estimated_execution_ms: u64,
    pub payload: T,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScheduledDispatch<T> {
    pub request_id: u64,
    pub queue_class: QueueClass,
    pub payload: T,
    pub quantum_tokens: u64,
    pub final_quantum: bool,
}

#[derive(Clone, Debug)]
struct QueuedWork<T> {
    id: u64,
    request: WorkRequest<T>,
    admitted_at_ms: u64,
    remaining_tokens: u64,
}

#[derive(Clone, Debug)]
struct RunningQuantum {
    queue_class: QueueClass,
    until_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchedulerStateSnapshot {
    pub queued_bytes: u64,
    pub in_flight_bytes: u64,
    pub byte_budget: u64,
    pub interactive_depth: usize,
    pub bulk_depth: usize,
    pub control_depth: usize,
    pub in_flight_workers: usize,
}

#[derive(Clone, Debug)]
pub struct LaneScheduler<T> {
    config: SchedulerConfig,
    interactive: VecDeque<QueuedWork<T>>,
    bulk: VecDeque<QueuedWork<T>>,
    control: VecDeque<QueuedWork<T>>,
    queued_bytes: u64,
    in_flight_bytes: u64,
    next_request_id: u64,
    fair_cycle: Vec<QueueClass>,
    fair_cursor: usize,
    running_quantum: Option<RunningQuantum>,
    in_flight_workers: usize,
    in_flight_requests: Vec<(u64, u64)>,
}

impl<T> LaneScheduler<T> {
    pub fn new(config: SchedulerConfig) -> Self {
        let mut fair_cycle = Vec::new();
        fair_cycle.extend(std::iter::repeat_n(
            QueueClass::Interactive,
            config.interactive_weight.max(1) as usize,
        ));
        fair_cycle.extend(std::iter::repeat_n(
            QueueClass::Bulk,
            config.bulk_weight.max(1) as usize,
        ));
        Self {
            config,
            interactive: VecDeque::new(),
            bulk: VecDeque::new(),
            control: VecDeque::new(),
            queued_bytes: 0,
            in_flight_bytes: 0,
            next_request_id: 1,
            fair_cycle,
            fair_cursor: 0,
            running_quantum: None,
            in_flight_workers: 0,
            in_flight_requests: Vec::new(),
        }
    }

    pub fn snapshot(&self) -> SchedulerStateSnapshot {
        SchedulerStateSnapshot {
            queued_bytes: self.queued_bytes,
            in_flight_bytes: self.in_flight_bytes,
            byte_budget: self.config.byte_budget,
            interactive_depth: self.interactive.len(),
            bulk_depth: self.bulk.len(),
            control_depth: self.control.len(),
            in_flight_workers: self.in_flight_workers,
        }
    }

    pub fn admit<C: Clock>(
        &mut self,
        clock: &C,
        mut request: WorkRequest<T>,
    ) -> Result<u64, RejectedAdmission> {
        if request.estimated_execution_ms == 0 {
            request.estimated_execution_ms = self.config.default_execution_ms;
        }
        let lane = LaneBudgetSnapshot {
            queued_bytes: self.queued_bytes,
            in_flight_bytes: self.in_flight_bytes,
            byte_budget: self.config.byte_budget,
            predicted_start_delay_ms: self
                .predicted_start_delay_ms(clock.now_ms(), request.queue_class),
        };
        match decide_admission(
            clock,
            &AdmissionRequest {
                queue_class: request.queue_class,
                deadline_ms: request.deadline_ms,
                max_queue_ms: request.max_queue_ms,
                request_bytes: request.request_bytes,
                estimated_execution_ms: request.estimated_execution_ms,
            },
            &lane,
        ) {
            AdmissionDecision::Accept(_) => {
                let id = self.next_request_id;
                self.next_request_id = self.next_request_id.saturating_add(1);
                self.queued_bytes = self.queued_bytes.saturating_add(request.request_bytes);
                let remaining_tokens = request.token_cost.max(1);
                let queued = QueuedWork {
                    id,
                    request,
                    admitted_at_ms: clock.now_ms(),
                    remaining_tokens,
                };
                match queued.request.queue_class {
                    QueueClass::Interactive => self.interactive.push_back(queued),
                    QueueClass::Bulk => self.bulk.push_back(queued),
                    QueueClass::Control => self.control.push_back(queued),
                }
                Ok(id)
            }
            AdmissionDecision::Reject(rejection) => Err(rejection),
        }
    }

    pub fn next_dispatch<C: Clock>(&mut self, clock: &C) -> Option<ScheduledDispatch<T>>
    where
        T: Clone,
    {
        let now_ms = clock.now_ms();
        if self.quantum_boundary_delay_ms(now_ms) > 0 {
            return None;
        }
        if self.in_flight_workers >= self.config.max_concurrent_workers.max(1) {
            return None;
        }
        let class = self.choose_class(now_ms)?;
        self.pop_dispatch(class)
    }

    pub fn complete_dispatch(&mut self, dispatch: &ScheduledDispatch<T>) {
        self.in_flight_workers = self.in_flight_workers.saturating_sub(1);
        if dispatch.final_quantum {
            let bytes = self.take_in_flight_bytes(dispatch.request_id);
            self.in_flight_bytes = self.in_flight_bytes.saturating_sub(bytes);
        }
        if self
            .running_quantum
            .as_ref()
            .is_some_and(|running| running.queue_class == dispatch.queue_class)
        {
            self.running_quantum = None;
        }
    }

    pub fn start_bulk_quantum_for_test(&mut self, until_ms: u64) {
        self.running_quantum = Some(RunningQuantum {
            queue_class: QueueClass::Bulk,
            until_ms,
        });
    }

    fn predicted_start_delay_ms(&self, now_ms: u64, queue_class: QueueClass) -> u64 {
        let boundary_delay = self.quantum_boundary_delay_ms(now_ms);
        match queue_class {
            QueueClass::Interactive => boundary_delay,
            QueueClass::Bulk => boundary_delay
                .saturating_add(self.bulk.len() as u64 * self.config.default_execution_ms),
            QueueClass::Control => boundary_delay,
        }
    }

    fn quantum_boundary_delay_ms(&self, now_ms: u64) -> u64 {
        self.running_quantum
            .as_ref()
            .map(|running| running.until_ms.saturating_sub(now_ms))
            .unwrap_or(0)
    }

    fn choose_class(&mut self, now_ms: u64) -> Option<QueueClass> {
        if !self.control.is_empty() {
            return Some(QueueClass::Control);
        }
        if !self.bulk.is_empty() && self.bulk_has_aged(now_ms) {
            return Some(QueueClass::Bulk);
        }
        if self.interactive.is_empty() {
            return (!self.bulk.is_empty()).then_some(QueueClass::Bulk);
        }
        if self.bulk.is_empty() {
            return Some(QueueClass::Interactive);
        }

        for _ in 0..self.fair_cycle.len() {
            let class = self.fair_cycle[self.fair_cursor];
            self.fair_cursor = (self.fair_cursor + 1) % self.fair_cycle.len();
            match class {
                QueueClass::Interactive if !self.interactive.is_empty() => return Some(class),
                QueueClass::Bulk if !self.bulk.is_empty() => return Some(class),
                QueueClass::Control => {}
                _ => {}
            }
        }
        None
    }

    fn bulk_has_aged(&self, now_ms: u64) -> bool {
        self.bulk.front().is_some_and(|bulk| {
            now_ms.saturating_sub(bulk.admitted_at_ms) >= self.config.bulk_aging_ms
        })
    }

    fn pop_dispatch(&mut self, class: QueueClass) -> Option<ScheduledDispatch<T>>
    where
        T: Clone,
    {
        let mut queued = match class {
            QueueClass::Interactive => self.interactive.pop_front()?,
            QueueClass::Bulk => self.bulk.pop_front()?,
            QueueClass::Control => self.control.pop_front()?,
        };

        let (quantum_tokens, final_quantum) = if class == QueueClass::Bulk {
            let quantum = queued
                .remaining_tokens
                .min(self.config.bulk_quantum_tokens.max(1));
            queued.remaining_tokens = queued.remaining_tokens.saturating_sub(quantum);
            (quantum, queued.remaining_tokens == 0)
        } else {
            (queued.remaining_tokens, true)
        };

        if final_quantum {
            self.queued_bytes = self
                .queued_bytes
                .saturating_sub(queued.request.request_bytes);
            self.in_flight_bytes = self
                .in_flight_bytes
                .saturating_add(queued.request.request_bytes);
            self.in_flight_requests
                .push((queued.id, queued.request.request_bytes));
        } else {
            self.bulk.push_back(queued.clone());
        }

        self.in_flight_workers = self.in_flight_workers.saturating_add(1);
        Some(ScheduledDispatch {
            request_id: queued.id,
            queue_class: class,
            payload: queued.request.payload,
            quantum_tokens,
            final_quantum,
        })
    }

    fn take_in_flight_bytes(&mut self, request_id: u64) -> u64 {
        if let Some(index) = self
            .in_flight_requests
            .iter()
            .position(|(id, _)| *id == request_id)
        {
            self.in_flight_requests.swap_remove(index).1
        } else {
            0
        }
    }
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
            estimated_execution_ms: 25,
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
                predicted_finish_ms: 1_225,
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
            estimated_execution_ms: 25,
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
            estimated_execution_ms: 1,
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
        assert_eq!(
            rejection.error.code,
            crate::StableErrorCode::DeadlineExceeded
        );
        assert!(rejection.reason.contains("max_queue_ms"));
    }

    #[test]
    fn admission_rejects_when_deadline_arrives_before_predicted_finish() {
        let clock = FakeClock { now_ms: 1_000 };
        let request = AdmissionRequest {
            queue_class: QueueClass::Interactive,
            deadline_ms: Some(1_050),
            max_queue_ms: 500,
            request_bytes: 1,
            estimated_execution_ms: 10,
        };
        let lane = LaneBudgetSnapshot {
            queued_bytes: 0,
            in_flight_bytes: 0,
            byte_budget: 10,
            predicted_start_delay_ms: 45,
        };

        let AdmissionDecision::Reject(rejection) = decide_admission(&clock, &request, &lane) else {
            panic!("expected deadline rejection");
        };
        assert_eq!(
            rejection.error.code,
            crate::StableErrorCode::DeadlineExceeded
        );
        assert!(rejection.reason.contains("finish"));
    }

    #[test]
    fn interactive_fast_fails_while_bulk_quantum_is_mid_flight() {
        let clock = FakeClock { now_ms: 10 };
        let mut scheduler = LaneScheduler::<&'static str>::new(SchedulerConfig {
            byte_budget: 1_024,
            default_execution_ms: 5,
            ..SchedulerConfig::default()
        });
        scheduler.start_bulk_quantum_for_test(200);

        let rejection = scheduler
            .admit(
                &clock,
                WorkRequest {
                    queue_class: QueueClass::Interactive,
                    deadline_ms: Some(100),
                    max_queue_ms: 50,
                    request_bytes: 1,
                    token_cost: 1,
                    estimated_execution_ms: 5,
                    payload: "query",
                },
            )
            .expect_err("query must fail instead of waiting behind the bulk quantum");
        assert_eq!(
            rejection.error.code,
            crate::StableErrorCode::DeadlineExceeded
        );
        assert!(rejection.reason.contains("max_queue_ms") || rejection.reason.contains("deadline"));
    }

    #[test]
    fn bulk_aging_prevents_starvation_under_interactive_pressure() {
        let mut scheduler = LaneScheduler::new(SchedulerConfig {
            byte_budget: 1_024,
            interactive_weight: 32,
            bulk_weight: 1,
            bulk_aging_ms: 10,
            default_execution_ms: 1,
            ..SchedulerConfig::default()
        });
        scheduler
            .admit(
                &FakeClock { now_ms: 0 },
                WorkRequest {
                    queue_class: QueueClass::Bulk,
                    deadline_ms: Some(1_000),
                    max_queue_ms: 1_000,
                    request_bytes: 10,
                    token_cost: 100,
                    estimated_execution_ms: 1,
                    payload: "bulk",
                },
            )
            .unwrap();
        for index in 0..8 {
            scheduler
                .admit(
                    &FakeClock { now_ms: 11 },
                    WorkRequest {
                        queue_class: QueueClass::Interactive,
                        deadline_ms: Some(1_000),
                        max_queue_ms: 1_000,
                        request_bytes: 1,
                        token_cost: 1,
                        estimated_execution_ms: 1,
                        payload: if index == 0 { "query" } else { "query-more" },
                    },
                )
                .unwrap();
        }

        let dispatch = scheduler
            .next_dispatch(&FakeClock { now_ms: 11 })
            .expect("aged bulk work should receive the next quantum");
        assert_eq!(dispatch.queue_class, QueueClass::Bulk);
        assert_eq!(dispatch.payload, "bulk");
    }

    #[test]
    fn control_loads_wait_for_quantum_boundaries() {
        let mut scheduler = LaneScheduler::new(SchedulerConfig {
            byte_budget: 1_024,
            default_execution_ms: 1,
            ..SchedulerConfig::default()
        });
        scheduler.start_bulk_quantum_for_test(100);
        scheduler
            .admit(
                &FakeClock { now_ms: 20 },
                WorkRequest {
                    queue_class: QueueClass::Control,
                    deadline_ms: Some(150),
                    max_queue_ms: 100,
                    request_bytes: 10,
                    token_cost: 1,
                    estimated_execution_ms: 1,
                    payload: "load",
                },
            )
            .unwrap();

        assert!(scheduler.next_dispatch(&FakeClock { now_ms: 99 }).is_none());
        let dispatch = scheduler
            .next_dispatch(&FakeClock { now_ms: 100 })
            .expect("control work should start at the quantum boundary");
        assert_eq!(dispatch.queue_class, QueueClass::Control);
        assert_eq!(dispatch.payload, "load");
    }

    #[test]
    fn scheduler_admission_rejects_memory_budget_overflow() {
        let mut scheduler = LaneScheduler::new(SchedulerConfig {
            byte_budget: 16,
            default_execution_ms: 1,
            ..SchedulerConfig::default()
        });
        scheduler
            .admit(
                &FakeClock { now_ms: 0 },
                WorkRequest {
                    queue_class: QueueClass::Interactive,
                    deadline_ms: None,
                    max_queue_ms: 10,
                    request_bytes: 12,
                    token_cost: 1,
                    estimated_execution_ms: 1,
                    payload: "first",
                },
            )
            .unwrap();

        let rejection = scheduler
            .admit(
                &FakeClock { now_ms: 0 },
                WorkRequest {
                    queue_class: QueueClass::Bulk,
                    deadline_ms: None,
                    max_queue_ms: 10,
                    request_bytes: 8,
                    token_cost: 1,
                    estimated_execution_ms: 1,
                    payload: "second",
                },
            )
            .expect_err("second request should exceed the lane byte budget");
        assert_eq!(rejection.error.code, crate::StableErrorCode::QueueFull);
    }
}
