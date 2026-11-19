use std::time::Duration;

pub(super) const SPEECH_BATCH_CAPACITY: usize = 128;
const MAX_BATCHED_SPEECH_RUNG: usize = 1024;

// Leave headroom below Windows' usual two-second GPU timeout, including when
// memory contention suddenly slows down the next submission. Grouping changes
// timing only: dispatch order and arithmetic stay the same, so output bits
// and the kernel revision used in model fingerprints should not change.
const SUBMISSION_BUDGET: Duration = Duration::from_millis(250);
const FAST_BATCHES_TO_GROW: u8 = 4;

#[derive(Clone, Copy)]
struct Limit {
    dispatches: usize,
    fast_batches: u8,
}

pub(super) struct SpeechBatchPolicy {
    rungs: [Limit; 4],
    active: Option<usize>,
}

impl Default for SpeechBatchPolicy {
    fn default() -> Self {
        Self {
            rungs: [Limit {
                dispatches: 1,
                fast_batches: 0,
            }; 4],
            active: None,
        }
    }
}

impl SpeechBatchPolicy {
    pub(super) fn select_rung(&mut self, rung: usize) {
        self.active = match rung {
            128 => Some(0),
            256 => Some(1),
            512 => Some(2),
            MAX_BATCHED_SPEECH_RUNG => Some(3),
            // Keep long rungs at one dispatch per submission even if they are fast.
            _ => None,
        };
    }

    pub(super) fn dispatch_limit(&self) -> usize {
        self.active.map_or(1, |index| self.rungs[index].dispatches)
    }

    pub(super) fn completed(&mut self, dispatches: usize, elapsed: Duration) {
        let Some(index) = self.active else { return };
        let limit = &mut self.rungs[index];
        if elapsed > SUBMISSION_BUDGET {
            limit.dispatches = (limit.dispatches / 2).max(1);
            limit.fast_batches = 0;
        } else if elapsed >= SUBMISSION_BUDGET / 2 {
            limit.fast_batches = 0;
        } else if dispatches == limit.dispatches {
            // A short tail does not tell us that a full submission has headroom.
            limit.fast_batches += 1;
            if limit.fast_batches == FAST_BATCHES_TO_GROW {
                limit.dispatches = (limit.dispatches + 1).min(SPEECH_BATCH_CAPACITY);
                limit.fast_batches = 0;
            }
        }
    }
}

#[cfg(test)]
#[derive(Default, serde::Serialize)]
pub(crate) struct SubmissionStats {
    submissions: usize,
    dispatches: usize,
    max_dispatches: usize,
    max_elapsed_ms: f64,
    over_budget: usize,
}

#[cfg(test)]
impl SubmissionStats {
    pub(super) fn record(&mut self, dispatches: usize, elapsed: Duration) {
        self.submissions += 1;
        self.dispatches += dispatches;
        self.max_dispatches = self.max_dispatches.max(dispatches);
        self.max_elapsed_ms = self.max_elapsed_ms.max(elapsed.as_secs_f64() * 1000.0);
        self.over_budget += usize::from(elapsed > SUBMISSION_BUDGET);
    }
}

#[test]
fn speech_batch_policy_backs_off_and_learns_each_rung_independently() {
    let mut policy = SpeechBatchPolicy::default();
    let fast = Duration::from_millis(100);
    let warm = Duration::from_millis(125);
    let slow = Duration::from_millis(251);
    policy.select_rung(1024);
    assert_eq!(policy.dispatch_limit(), 1);
    policy.completed(1, slow);
    assert_eq!(policy.dispatch_limit(), 1);
    for _ in 0..3 {
        policy.completed(1, fast);
    }
    assert_eq!(policy.dispatch_limit(), 1);
    policy.completed(1, warm);
    for _ in 0..3 {
        policy.completed(1, fast);
    }
    assert_eq!(policy.dispatch_limit(), 1);
    policy.completed(1, fast);
    assert_eq!(policy.dispatch_limit(), 2);
    for _ in 0..8 {
        policy.completed(1, fast);
    }
    assert_eq!(policy.dispatch_limit(), 2);
    for _ in 0..4 {
        policy.completed(2, fast);
    }
    for _ in 0..4 {
        policy.completed(3, fast);
    }
    assert_eq!(policy.dispatch_limit(), 4);
    policy.completed(4, slow);
    assert_eq!(policy.dispatch_limit(), 2);

    for rung in [128, 256, 512, 2048, 4096, 8192] {
        policy.select_rung(rung);
        assert_eq!(policy.dispatch_limit(), 1, "rung {rung}");
        if rung > 1024 {
            for _ in 0..8 {
                policy.completed(1, fast);
            }
            assert_eq!(policy.dispatch_limit(), 1, "rung {rung}");
        }
    }
    policy.select_rung(1024);
    assert_eq!(policy.dispatch_limit(), 2);
    for _ in 0..(4 * SPEECH_BATCH_CAPACITY) {
        policy.completed(policy.dispatch_limit(), fast);
    }
    assert_eq!(policy.dispatch_limit(), SPEECH_BATCH_CAPACITY);
}
