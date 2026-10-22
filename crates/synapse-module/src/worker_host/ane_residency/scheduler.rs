//! The compile-queue scheduler: the one task that makes every decision about
//! cold compiles, slot reservations, evictions, failures and worker
//! recovery for all direct-ANE workers of a module.
//!
//! Requests never drive the queue. A request that cannot lease a resident
//! shape on the fast path (`AneResidencySupervisor::try_lease`) sends a
//! [`Want`] and waits for one [`Answer`]. Each pass of the scheduler takes one
//! instant, decides everything at that instant, spawns the resulting compile,
//! evict and recovery operations into a `JoinSet` it awaits itself (so every
//! panic is observed), and arms one wake-up timer computed from the same
//! instant. The queue state lives in the shared `State` behind the
//! supervisor's mutex so the lease fast path, statistics and tests can read
//! it; only this task changes queue entries, waiters, promises and slot
//! lifecycles. Requests only add and drop leases.
use super::*;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinSet;

/// How long the scheduler waits, after starting a worker's restart, for the
/// old process's exit to be confirmed before it stops holding the compile
/// gate for that worker. The worker channel waits 2 s for a voluntary exit,
/// then sends SIGKILL and allows 30 s in total (its owner exit timeout). A
/// process still not reaped after SIGKILL is stuck in the kernel, typically
/// inside the Neural Engine driver, and waiting longer does not help, while
/// every other worker's cold compiles would stall behind it. The extra 5 s
/// lets the channel's own timeout report first. Giving up risks only that a
/// compile overlaps the stuck process and is refused by the hardware
/// (`ane_resources_exhausted`), a refusal rather than wrong output; the
/// worker itself stays closed with its slots charged.
pub(super) const EXIT_CONFIRMATION_GIVE_UP: std::time::Duration =
    std::time::Duration::from_secs(35);

/// How many queued messages the scheduler handles between two passes, on top
/// of the one that woke it.
const MESSAGES_PER_PASS: usize = 64;

/// A request's admission trace context in tests; empty otherwise.
#[derive(Clone, Default)]
pub(super) struct Trace {
    #[cfg(test)]
    pub(super) context: Option<admission_trace::Context>,
}

impl Trace {
    /// The trace of the current task's request.
    pub(super) fn current() -> Self {
        Self {
            #[cfg(test)]
            context: admission_trace::current(),
        }
    }
}

/// What packed preparation is answered: the prepared program's lease, or
/// `None` when it cannot be prepared now.
pub(super) type Prepared = Result<Option<AneShapeLease>, AneResidencyError>;

/// How many recent answers are kept so a serving caller whose own timer
/// fires just after the scheduler answered gets the same answer.
const RECENT_ANSWERS: usize = 1024;

pub(super) enum Msg {
    Want(Want),
    /// Something requests do not report in detail changed: a request went
    /// away, or a lease the scheduler watches was dropped.
    Nudge,
    Dispatched {
        slot_id: u64,
        at: tokio::time::Instant,
    },
    ExitConfirmed {
        worker_id: String,
        generation: u64,
    },
    /// The serving caller's own timer fired while its request waited on the
    /// scheduler; answer it now, once.
    Expired {
        id: u64,
        reply: oneshot::Sender<AnswerKind>,
    },
    Recover {
        worker: Arc<dyn AneShapeWorker>,
        generation: u64,
        reply: RecoverReply,
    },
    Retire {
        worker: Arc<dyn AneShapeWorker>,
        reply: oneshot::Sender<()>,
    },
    RetireDone {
        worker_id: String,
        exit_confirmed: bool,
        reply: oneshot::Sender<()>,
    },
    #[cfg(test)]
    Evict {
        key: ShapeKey,
        slot_id: u64,
        reply: oneshot::Sender<bool>,
    },
}

pub(super) struct Want {
    pub(super) id: u64,
    pub(super) key: ShapeKey,
    pub(super) worker: Arc<dyn AneShapeWorker>,
    pub(super) executables: usize,
    /// The latest moment the request can use its shape; `None` for packed
    /// preparation, which never waits.
    pub(super) bound: Option<tokio::time::Instant>,
    pub(super) deadline: Option<tokio::time::Instant>,
    pub(super) reply: Reply,
    pub(super) trace: Trace,
}

pub(super) enum Reply {
    Lease(oneshot::Sender<Answer>),
    Prepare(oneshot::Sender<Prepared>),
}

impl Reply {
    fn is_closed(&self) -> bool {
        match self {
            Self::Lease(sender) => sender.is_closed(),
            Self::Prepare(sender) => sender.is_closed(),
        }
    }
}

pub(super) enum Answer {
    /// A lease on the resident shape. `compiled` says the request waited for
    /// the shape's compile.
    Lease {
        lease: AneShapeLease,
        compiled: bool,
    },
    /// The shape this request waited for has just been compiled; it may lease
    /// it ahead of a starved head during the 250 ms handoff.
    Ready,
    Failed(AneResidencyError),
}

/// An answer the scheduler gave, kept so a later caller-side expiry of the
/// same request is given the same one.
#[derive(Clone, Debug)]
pub(super) enum AnswerKind {
    ShapeCompiling { shape: usize, retry_after_ms: u64 },
    DeadlineExceeded,
    Resources(String),
    Channel(String),
    WorkerClosed(String),
}

impl AnswerKind {
    pub(super) fn into_error(self) -> AneResidencyError {
        match self {
            Self::ShapeCompiling {
                shape,
                retry_after_ms,
            } => AneResidencyError::ShapeCompiling {
                shape,
                retry_after_ms,
            },
            Self::DeadlineExceeded => AneResidencyError::DeadlineExceeded,
            Self::Resources(msg) => AneResidencyError::ResourcesExhausted {
                msg,
                retry_after_ms: ANE_RESOURCES_RETRY_AFTER_MS,
            },
            Self::Channel(msg) => AneResidencyError::Channel(msg),
            Self::WorkerClosed(worker_id) => worker_closed(&worker_id),
        }
    }

    fn of(error: &AneResidencyError) -> Self {
        match error {
            AneResidencyError::ShapeCompiling {
                shape,
                retry_after_ms,
            } => Self::ShapeCompiling {
                shape: *shape,
                retry_after_ms: *retry_after_ms,
            },
            AneResidencyError::DeadlineExceeded => Self::DeadlineExceeded,
            AneResidencyError::ResourcesExhausted { msg, .. } => Self::Resources(msg.clone()),
            AneResidencyError::WorkerClosed { worker_id } => Self::WorkerClosed(worker_id.clone()),
            other => Self::Channel(other.to_string()),
        }
    }
}

/// Where a compile operation reports the moment its admission reached the
/// worker (set by the worker channel, or by a test worker).
pub(super) struct DispatchCell {
    at: Mutex<Option<tokio::time::Instant>>,
    slot_id: u64,
    report: mpsc::UnboundedSender<Msg>,
}

impl DispatchCell {
    pub(super) fn mark(&self) {
        let now = tokio::time::Instant::now();
        *self
            .at
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(now);
        let _ = self.report.send(Msg::Dispatched {
            slot_id: self.slot_id,
            at: now,
        });
    }

    fn at(&self) -> Option<tokio::time::Instant> {
        *self
            .at
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// A shape compile counts as dispatched once its admission reached the
/// worker. Workers other than the channel (tests) call this themselves.
pub(super) fn mark_compile_dispatched() {
    let _ = COMPILE_DISPATCHED.try_with(|cell| cell.mark());
}

struct Waiter {
    key: ShapeKey,
    worker: Arc<dyn AneShapeWorker>,
    executables: usize,
    bound: Option<tokio::time::Instant>,
    deadline: Option<tokio::time::Instant>,
    reply: Reply,
    /// The request waited for this shape's compile at some point.
    on_compile: bool,
    /// The request waits on a compile right now (queued for it, or joined to
    /// it), as opposed to waiting for budget or for a worker's recovery.
    waiting_on_compile: bool,
    trace: Trace,
}

impl Waiter {
    fn prepare(&self) -> bool {
        matches!(self.reply, Reply::Prepare(_))
    }

    /// The latest moment request `id` can still use its shape. Once its
    /// compile is running, the admission wait backstop no longer applies (it
    /// bounds queue waits, not a compile in flight): a request that joined
    /// the compile is answered at its own deadline, and the request whose
    /// turn started the compile waits for it to finish, as its serving task
    /// holds the lane until the compile's reply has drained (its caller is
    /// answered at the deadline through `Msg::Expired`).
    fn bound(&self, id: u64, state: &State) -> Option<tokio::time::Instant> {
        match state.slots.get(&self.key) {
            Some(slot) if slot.state == SlotState::Admitting => {
                (slot.owner != Some(id)).then_some(self.deadline).flatten()
            }
            _ => self.bound,
        }
    }
}

enum OpKind {
    Compile {
        key: ShapeKey,
        slot_id: u64,
        generation: u64,
        retry: bool,
    },
    Evict {
        key: ShapeKey,
        slot_id: u64,
        generation: u64,
        owner: Arc<dyn AneShapeWorker>,
        /// The queued compile this eviction makes room for.
        for_head: Option<ShapeKey>,
        /// The admission retried after this eviction, with the refusal that
        /// caused it.
        for_retry: Option<(ShapeKey, u64, String)>,
        /// A test waiting for this eviction's outcome.
        reply: Option<oneshot::Sender<bool>>,
    },
    Recover {
        worker_id: String,
    },
}

enum OpDone {
    Compile {
        result: Result<AnePlacementInventory, AneResidencyError>,
        dispatched: Option<tokio::time::Instant>,
        finished: tokio::time::Instant,
    },
    Evict(Result<(), AneResidencyError>),
    /// `exited`: the old process's exit was confirmed during the restart. It
    /// is also reported early by `Msg::ExitConfirmed`, but the completion
    /// carries it too, so it does not depend on which one is handled first.
    Recover {
        result: Result<(), AneResidencyError>,
        exited: bool,
    },
}

pub(super) type RecoverReply = oneshot::Sender<Result<(), AneResidencyError>>;

struct Recovery {
    worker: Arc<dyn AneShapeWorker>,
    generation: u64,
    replies: Vec<RecoverReply>,
    drain_deadline: tokio::time::Instant,
    started: bool,
    exit_deadline: Option<tokio::time::Instant>,
    exit_confirmed: bool,
    gave_up: bool,
    /// The restart operation has ended without confirming the old process's
    /// exit. The entry stays only so that the compile gate is still held
    /// until `exit_deadline`, as for a restart that is still running.
    finished: bool,
}

struct Retirement {
    reply: Option<oneshot::Sender<()>>,
    deadline: tokio::time::Instant,
}

/// Replies are sent after the state lock is released: a lease dropped by a
/// closed receiver locks the state.
#[derive(Default)]
struct Effects {
    leases: Vec<(oneshot::Sender<Answer>, Answer)>,
    prepares: Vec<(oneshot::Sender<Prepared>, Prepared)>,
    bools: Vec<(oneshot::Sender<bool>, bool)>,
    recoveries: Vec<(RecoverReply, Result<(), AneResidencyError>)>,
    units: Vec<oneshot::Sender<()>>,
    progress: bool,
}

pub(super) struct Scheduler {
    inner: std::sync::Weak<Inner>,
    rx: mpsc::UnboundedReceiver<Msg>,
    tx: mpsc::WeakUnboundedSender<Msg>,
    waiters: BTreeMap<u64, Waiter>,
    ops: JoinSet<OpDone>,
    op_kinds: BTreeMap<tokio::task::Id, OpKind>,
    /// Dropping an operation's guard withdraws it if it has not reached the
    /// worker yet (its stream wait is cancellable).
    op_guards: BTreeMap<tokio::task::Id, RequestAbandonment>,
    recoveries: BTreeMap<String, Recovery>,
    retirements: BTreeMap<String, Retirement>,
    /// Workers closed until the module restarts: a restart ended, or the
    /// scheduler gave up waiting, without the old process's exit confirmed,
    /// so nobody knows whether it still owns the Neural Engine. Requests for
    /// them are refused at once.
    closed: BTreeSet<String>,
    answered: VecDeque<(u64, AnswerKind)>,
}

/// Marks the supervisor closed when the scheduler's loop ends for any
/// reason other than the supervisor itself going away, so every waiting and
/// later request gets `SchedulerStopped` instead of hanging.
struct DeathGuard(std::sync::Weak<Inner>);

impl Drop for DeathGuard {
    fn drop(&mut self) {
        if let Some(inner) = self.0.upgrade() {
            inner.closed.store(true, Ordering::Release);
            tracing::error!(
                target: "worker",
                "direct-ANE compile scheduler stopped; direct-ANE lanes are closed until the module restarts"
            );
        }
    }
}

impl Scheduler {
    pub(super) fn new(inner: &Arc<Inner>) -> (Self, mpsc::UnboundedSender<Msg>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (
            Self {
                inner: Arc::downgrade(inner),
                rx,
                tx: tx.downgrade(),
                waiters: BTreeMap::new(),
                ops: JoinSet::new(),
                op_kinds: BTreeMap::new(),
                op_guards: BTreeMap::new(),
                recoveries: BTreeMap::new(),
                retirements: BTreeMap::new(),
                closed: BTreeSet::new(),
                answered: VecDeque::new(),
            },
            tx,
        )
    }

    pub(super) async fn run(mut self) {
        let _guard = DeathGuard(self.inner.clone());
        loop {
            let now = tokio::time::Instant::now();
            let mut wake = None;
            // Decide until nothing changes at this instant.
            for _ in 0..64 {
                let (next, progress) = self.pass(now);
                wake = next;
                if !progress {
                    break;
                }
            }
            #[cfg(test)]
            if wake.is_some() {
                self.hold_after_decision().await;
            }
            tokio::select! {
                biased;
                message = self.rx.recv() => match message {
                    Some(message) => self.handle(message, tokio::time::Instant::now()),
                    None => return,
                },
                Some(done) = self.ops.join_next_with_id(), if !self.ops.is_empty() => {
                    self.finish(done, tokio::time::Instant::now());
                }
                _ = sleep_until_option(wake) => {}
            }
            // At most a bounded number of further messages per pass, and every
            // operation that has already finished, so that a busy mailbox can
            // starve neither completions nor the timers the next pass checks.
            let mut drained = false;
            for _ in 0..MESSAGES_PER_PASS {
                let Ok(message) = self.rx.try_recv() else {
                    drained = true;
                    break;
                };
                self.handle(message, tokio::time::Instant::now());
            }
            while let Some(done) = self.ops.try_join_next_with_id() {
                self.finish(done, tokio::time::Instant::now());
            }
            if !drained {
                // With the mailbox still full this task would never wait, so
                // the operations it spawned on the same thread (compiles,
                // evictions, restarts) would never run. Let them.
                tokio::task::yield_now().await;
            }
        }
    }

    /// Lets a test move the paused clock between a pass's decisions and the
    /// timer it arms.
    #[cfg(test)]
    async fn hold_after_decision(&self) {
        let gate = self
            .inner
            .upgrade()
            .and_then(|inner| inner.wait_decision_gate.lock().unwrap().take());
        if let Some((decided, resume)) = gate {
            decided.notify_one();
            resume.notified().await;
        }
    }

    fn handle(&mut self, message: Msg, now: tokio::time::Instant) {
        let Some(inner) = self.inner.upgrade() else {
            return;
        };
        #[cfg(test)]
        if matches!(message, Msg::Nudge) {
            // Outside the state lock, so only the scheduler is slowed.
            let cost = *inner.slow_nudges.lock().unwrap();
            if let Some(cost) = cost {
                std::thread::sleep(cost);
            }
        }
        let mut effects = Effects::default();
        {
            let mut state = inner.lock();
            match message {
                Msg::Want(want) => self.admit_want(&inner, &mut state, want, now, &mut effects),
                Msg::Nudge => {}
                Msg::Dispatched { slot_id, at } => {
                    if let Some(slot) = state.slots.values_mut().find(|slot| slot.id == slot_id) {
                        slot.dispatched_at = Some(at);
                    }
                }
                Msg::ExitConfirmed {
                    worker_id,
                    generation,
                } => self.exit_confirmed(&mut state, &worker_id, generation),
                Msg::Expired { id, reply } => {
                    let owns_compile = self.waiters.get(&id).is_some_and(|waiter| {
                        state.slots.get(&waiter.key).is_some_and(|slot| {
                            slot.state == SlotState::Admitting && slot.owner == Some(id)
                        })
                    });
                    let kind = match self.waiters.contains_key(&id) {
                        // The caller is answered, but the request that started
                        // the compile keeps waiting for it (see `Waiter::bound`).
                        true if owns_compile => {
                            let error = self.expiry_answer(&mut state, id, now, true);
                            let kind = AnswerKind::of(&error);
                            self.answered.push_back((id, kind.clone()));
                            Some(kind)
                        }
                        true => {
                            let error = self.expiry_answer(&mut state, id, now, true);
                            self.answer(&mut state, id, Answer::Failed(error), &mut effects);
                            self.recent(id)
                        }
                        false => self.recent(id),
                    };
                    let _ = reply.send(kind.unwrap_or(AnswerKind::DeadlineExceeded));
                }
                Msg::Recover {
                    worker,
                    generation,
                    reply,
                } => self.start_recovery(
                    &mut state,
                    &worker,
                    generation,
                    Some(reply),
                    now,
                    &mut effects,
                ),
                Msg::Retire { worker, reply } => {
                    let id = worker.worker_id().to_owned();
                    state.retired.insert(id.clone());
                    state.recovering.insert(id.clone());
                    // A retired worker compiles nothing more, so its queued
                    // compiles are dropped at once, promised or not.
                    state.entries.retain(|key, _| key.worker_id != id);
                    self.retirements.insert(
                        id,
                        Retirement {
                            reply: Some(reply),
                            deadline: now + inner.wait_timeout,
                        },
                    );
                }
                Msg::RetireDone {
                    worker_id,
                    exit_confirmed,
                    reply,
                } => {
                    self.retirements.remove(&worker_id);
                    if exit_confirmed {
                        // Reservations are summed from these slots. Refund only
                        // this owner, and only after its exit is confirmed; the
                        // next worker instance must not inherit its charge.
                        state
                            .slots
                            .retain(|_, slot| slot.worker.worker_id() != worker_id);
                        state.recovering.remove(&worker_id);
                        self.closed.remove(&worker_id);
                        // A recovery that ended without confirming the exit
                        // only held the gate; the retirement confirmed it.
                        if self
                            .recoveries
                            .get(&worker_id)
                            .is_some_and(|recovery| recovery.finished)
                        {
                            self.recoveries.remove(&worker_id);
                        } else if let Some(recovery) = self.recoveries.get_mut(&worker_id) {
                            recovery.exit_confirmed = true;
                        }
                    } else {
                        give_up_gate(&mut state, &worker_id, None);
                    }
                    effects.units.push(reply);
                }
                #[cfg(test)]
                Msg::Evict {
                    key,
                    slot_id,
                    reply,
                } => {
                    let owner = state
                        .slots
                        .get(&key)
                        .filter(|slot| slot.id == slot_id)
                        .map(|slot| (slot.worker.clone(), slot.generation));
                    match owner {
                        Some((owner, generation)) => self.spawn_evict(
                            &mut state,
                            key,
                            slot_id,
                            generation,
                            owner,
                            None,
                            None,
                            Some(reply),
                            Default::default(),
                        ),
                        None => effects.bools.push((reply, true)),
                    }
                }
            }
        }
        self.apply(effects);
    }

    fn admit_want(
        &mut self,
        inner: &Arc<Inner>,
        state: &mut State,
        want: Want,
        _now: tokio::time::Instant,
        effects: &mut Effects,
    ) {
        if let Reply::Prepare(sender) = &want.reply {
            // Preparation does not wait for budget behind live requests or
            // for the worker's own in-flight work.
            let busy = !state.waiters.is_empty()
                || !state.entries.is_empty()
                || state.slots.values().any(|slot| {
                    slot.worker.worker_id() == want.worker.worker_id()
                        && (slot.leases != 0 || slot.state == SlotState::Admitting)
                });
            if busy {
                if let Reply::Prepare(sender) = want.reply {
                    effects.prepares.push((sender, Ok(None)));
                }
                return;
            }
            let _ = sender;
        }
        let _ = inner;
        state.waiters.push_back(want.id);
        state.waiter_keys.insert(want.id, want.key.clone());
        #[cfg(test)]
        admission_trace::event_in(
            &want.trace.context,
            "queue_entry",
            &want.key.model_ref,
            want.key.shape,
            Some(want.id),
        );
        self.waiters.insert(
            want.id,
            Waiter {
                key: want.key,
                worker: want.worker,
                executables: want.executables,
                bound: want.bound,
                deadline: want.deadline,
                reply: want.reply,
                on_compile: false,
                waiting_on_compile: false,
                trace: want.trace,
            },
        );
    }

    /// One pass at the instant `now`. Returns the next wake-up and whether
    /// anything changed (then the pass runs again at the same instant).
    fn pass(&mut self, now: tokio::time::Instant) -> (Option<tokio::time::Instant>, bool) {
        let Some(inner) = self.inner.upgrade() else {
            return (None, false);
        };
        let mut effects = Effects::default();
        let wake;
        {
            let mut state = inner.lock();
            self.drop_departed(&mut state, now, &mut effects);
            self.withdraw_unwanted_compiles(&state, now);
            self.expire_bounds(&mut state, now, &mut effects);
            end_handoffs(&mut state, now);
            self.drive_recoveries(&mut state, now, &mut effects);
            self.drive_retirements(&mut state, now, &mut effects);
            self.decide(&inner, &mut state, now, &mut effects);
            self.expire_entries(&mut state, now);
            wake = self.next_wake(&state, now);
            let watch = state.starved_head.is_some()
                || self.recoveries.values().any(|recovery| !recovery.started)
                || self
                    .retirements
                    .values()
                    .any(|retirement| retirement.reply.is_some())
                || state.slots.values().any(|slot| slot.handing_off(now));
            inner.watch_leases.store(watch, Ordering::Release);
        }
        let progress = effects.progress;
        self.apply(effects);
        (wake, progress)
    }

    fn apply(&mut self, effects: Effects) {
        for (sender, answer) in effects.leases {
            let _ = sender.send(answer);
        }
        for (sender, answer) in effects.prepares {
            let _ = sender.send(answer);
        }
        for (sender, value) in effects.bools {
            let _ = sender.send(value);
        }
        for (sender, outcome) in effects.recoveries {
            let _ = sender.send(outcome);
        }
        for sender in effects.units {
            let _ = sender.send(());
        }
    }

    fn recent(&self, id: u64) -> Option<AnswerKind> {
        self.answered
            .iter()
            .rev()
            .find(|(answered, _)| *answered == id)
            .map(|(_, kind)| kind.clone())
    }

    /// Removes waiter `id` and queues its answer.
    fn answer(&mut self, state: &mut State, id: u64, answer: Answer, effects: &mut Effects) {
        let Some(waiter) = self.waiters.remove(&id) else {
            return;
        };
        state.leave_queue(id);
        effects.progress = true;
        if let Answer::Failed(error) = &answer {
            self.answered.push_back((id, AnswerKind::of(error)));
            if self.answered.len() > RECENT_ANSWERS {
                self.answered.pop_front();
            }
        }
        match waiter.reply {
            Reply::Lease(sender) => effects.leases.push((sender, answer)),
            Reply::Prepare(sender) => effects.prepares.push((
                sender,
                match answer {
                    Answer::Lease { lease, .. } => Ok(Some(lease)),
                    Answer::Ready => Ok(None),
                    Answer::Failed(error) => Err(error),
                },
            )),
        }
    }

    /// Requests whose caller went away are dropped. Work started only for
    /// them, and not yet on the wire, is withdrawn: an eviction for a queued
    /// compile, or a compile, that nobody waits for and nobody was promised.
    fn drop_departed(
        &mut self,
        state: &mut State,
        now: tokio::time::Instant,
        effects: &mut Effects,
    ) {
        let departed: Vec<u64> = self
            .waiters
            .iter()
            .filter(|(_, waiter)| waiter.reply.is_closed())
            .map(|(id, _)| *id)
            .collect();
        for id in departed {
            let waiter = self.waiters.remove(&id).expect("listed above");
            state.leave_queue(id);
            effects.progress = true;
            let key = waiter.key;
            if self.waiters.values().any(|other| other.key == key) {
                continue;
            }
            let entry_promised = state
                .entries
                .get(&key)
                .and_then(|entry| entry.promise)
                .is_some_and(|promise| promise_live(promise, now));
            let slot = state
                .slots
                .get(&key)
                .filter(|slot| slot.state == SlotState::Admitting)
                .map(|slot| {
                    (
                        slot.id,
                        slot.promise
                            .is_some_and(|promise| promise_live(promise, now)),
                    )
                });
            let withdraw: Vec<tokio::task::Id> = self
                .op_kinds
                .iter()
                .filter(|(_, kind)| match kind {
                    OpKind::Evict {
                        for_head: Some(head),
                        ..
                    } => *head == key && !entry_promised,
                    OpKind::Compile {
                        slot_id,
                        retry: false,
                        ..
                    } => slot.is_some_and(|(id, promised)| id == *slot_id && !promised),
                    _ => false,
                })
                .map(|(task, _)| *task)
                .collect();
            for task in withdraw {
                self.op_guards.remove(&task);
            }
        }
    }

    /// A compile that has not reached its worker yet is withdrawn once nobody
    /// waits for it and the last promise made for it has expired, so a
    /// compile kept only for a promise does not reach the wire after that
    /// promise lapsed. (A waiter leaving is handled by `drop_departed`.)
    fn withdraw_unwanted_compiles(&mut self, state: &State, now: tokio::time::Instant) {
        let withdraw: Vec<tokio::task::Id> = self
            .op_kinds
            .iter()
            .filter(|(task, kind)| {
                let OpKind::Compile {
                    key,
                    slot_id,
                    retry: false,
                    ..
                } = kind
                else {
                    return false;
                };
                self.op_guards.contains_key(task)
                    && !self.waiters.values().any(|waiter| waiter.key == *key)
                    && state.slots.get(key).is_some_and(|slot| {
                        slot.id == *slot_id
                            && slot.state == SlotState::Admitting
                            && slot.dispatched_at.is_none()
                            && slot
                                .promise
                                .is_some_and(|promise| !promise_live(promise, now))
                    })
            })
            .map(|(task, _)| *task)
            .collect();
        for task in withdraw {
            self.op_guards.remove(&task);
        }
    }

    /// Whether `key` is waiting on a compile: queued, or being compiled.
    fn compiling(state: &State, key: &ShapeKey) -> bool {
        state.entries.contains_key(key)
            || state
                .slots
                .get(key)
                .is_some_and(|slot| slot.state == SlotState::Admitting)
    }

    /// Records `told` as the promise of `key`'s compile, queued or running.
    fn promise(
        state: &mut State,
        key: &ShapeKey,
        told: (tokio::time::Instant, std::time::Duration),
    ) {
        if let Some(entry) = state.entries.get_mut(key) {
            entry.promise = Some(told);
        } else if let Some(slot) = state.slots.get_mut(key) {
            slot.promise = Some(told);
        }
    }

    /// `shape_compiling` with the prediction at `now`, recorded as the
    /// promise of the compile.
    fn refusal(
        state: &mut State,
        key: &ShapeKey,
        ready: Option<tokio::time::Instant>,
        now: tokio::time::Instant,
    ) -> AneResidencyError {
        let left = ready.map_or(std::time::Duration::ZERO, |ready| {
            ready.saturating_duration_since(now)
        });
        let retry_after_ms = u64::try_from(left.as_millis())
            .unwrap_or(u64::MAX)
            .max(ANE_SHAPE_COMPILING_RETRY_AFTER_MS);
        Self::promise(
            state,
            key,
            (now, std::time::Duration::from_millis(retry_after_ms)),
        );
        tracing::debug!(
            target: "worker",
            model_ref = %key.model_ref,
            shape = key.shape,
            worker_id = %key.worker_id,
            retry_after_ms,
            "direct-ANE compile queue: refused, shape not ready in time"
        );
        AneResidencyError::ShapeCompiling {
            shape: key.shape,
            retry_after_ms,
        }
    }

    /// The answer for waiter `id` whose bound (or, from the caller,
    /// deadline) has passed.
    fn expiry_answer(
        &mut self,
        state: &mut State,
        id: u64,
        now: tokio::time::Instant,
        caller_deadline: bool,
    ) -> AneResidencyError {
        let waiter = &self.waiters[&id];
        let key = waiter.key.clone();
        if waiter.waiting_on_compile && Self::compiling(state, &key) {
            let ready = state.predicted_ready(&key, now);
            return Self::refusal(state, &key, ready, now);
        }
        if caller_deadline || waiter.deadline.is_some_and(|deadline| now >= deadline) {
            return AneResidencyError::DeadlineExceeded;
        }
        resource_refusal(AneResidencyError::Channel(
            "admission budget wait deadline exceeded".into(),
        ))
    }

    fn expire_bounds(
        &mut self,
        state: &mut State,
        now: tokio::time::Instant,
        effects: &mut Effects,
    ) {
        let due: Vec<u64> = self
            .waiters
            .iter()
            .filter(|(id, waiter)| waiter.bound(**id, state).is_some_and(|bound| now >= bound))
            .map(|(id, _)| *id)
            .collect();
        for id in due {
            let error = self.expiry_answer(state, id, now, false);
            self.answer(state, id, Answer::Failed(error), effects);
        }
    }

    fn decide(
        &mut self,
        inner: &Arc<Inner>,
        state: &mut State,
        now: tokio::time::Instant,
        effects: &mut Effects,
    ) {
        // The starved-head barrier is recomputed by every pass; requests
        // behind the head starved in the previous pass keep waiting in order.
        let previous_starved = state.starved_head.take();
        let ids: Vec<u64> = self.waiters.keys().copied().collect();
        for id in ids {
            let Some(waiter) = self.waiters.get_mut(&id) else {
                continue;
            };
            waiter.waiting_on_compile = false;
            let key = waiter.key.clone();
            let worker = waiter.worker.clone();
            let executables = waiter.executables;
            let prepare = waiter.prepare();
            if state.retired.contains(worker.worker_id()) {
                self.answer(
                    state,
                    id,
                    Answer::Failed(AneResidencyError::Channel("worker is retired".into())),
                    effects,
                );
                continue;
            }
            if self.closed.contains(worker.worker_id()) {
                // Closed until the module restarts: waiting cannot help.
                let error = worker_closed(worker.worker_id());
                self.answer(state, id, Answer::Failed(error), effects);
                continue;
            }
            if state.recovering.contains(worker.worker_id()) {
                if prepare {
                    self.answer(state, id, Answer::Ready, effects);
                }
                continue;
            }
            if let Some(slot_state) = state.slots.get(&key).map(|slot| slot.state) {
                match slot_state {
                    SlotState::Admitting => {
                        state.slots.get_mut(&key).unwrap().joined.insert(id);
                        let waiter = self.waiters.get_mut(&id).unwrap();
                        waiter.on_compile = true;
                        waiter.waiting_on_compile = true;
                        let bound = waiter.bound(id, state);
                        self.check_bound(state, id, &key, bound, now, effects);
                    }
                    SlotState::Resident => {
                        let slot = &state.slots[&key];
                        let cohort = slot.handing_off(now) && slot.joined.contains(&id);
                        if !cohort && !prepare && previous_starved.is_some_and(|head| id > head) {
                            continue;
                        }
                        let tick = state.tick();
                        let slot = state.slots.get_mut(&key).unwrap();
                        slot.leases += 1;
                        slot.last_used = tick;
                        let lease = AneShapeLease {
                            inner: inner.clone(),
                            key: key.clone(),
                            slot_id: slot.id,
                            inventory: slot.inventory.clone(),
                        };
                        let compiled = self.waiters[&id].on_compile;
                        #[cfg(test)]
                        admission_trace::event_in(
                            &self.waiters[&id].trace.context,
                            "lease_grant",
                            &key.model_ref,
                            key.shape,
                            Some(id),
                        );
                        self.answer(state, id, Answer::Lease { lease, compiled }, effects);
                    }
                    _ => {
                        if prepare {
                            self.answer(state, id, Answer::Ready, effects);
                        }
                    }
                }
                continue;
            }
            // A cold compile: queued once per shape.
            if !state.entries.contains_key(&key) {
                state.create_entry(&key, &worker, executables, now);
                effects.progress = true;
            }
        }
        self.decide_queue(inner, state, now, effects);
    }

    /// Elects the queued compile that starts next and acts for it. The head
    /// is the queued compile with the smallest start key whose worker can
    /// compile and that somebody still wants: a waiting request, or a live
    /// promise given to one that left. Every other queued request waits on
    /// its compile and is refused at once if its shape cannot be ready by its
    /// bound.
    fn decide_queue(
        &mut self,
        inner: &Arc<Inner>,
        state: &mut State,
        now: tokio::time::Instant,
        effects: &mut Effects,
    ) {
        let head = state
            .entries
            .iter()
            .filter(|(key, entry)| {
                !state.worker_unavailable(&key.worker_id)
                    && (state.requested(key)
                        || entry
                            .promise
                            .is_some_and(|promise| promise_live(promise, now)))
            })
            .min_by_key(|(_, entry)| entry.order())
            .map(|(key, _)| key.clone());
        let queued: Vec<u64> = self
            .waiters
            .iter()
            .filter(|(_, waiter)| {
                state.entries.contains_key(&waiter.key)
                    && !state.worker_unavailable(waiter.worker.worker_id())
            })
            .map(|(id, _)| *id)
            .collect();
        let mut head_waiters = Vec::new();
        for id in queued {
            let waiter = self.waiters.get_mut(&id).expect("listed above");
            if Some(&waiter.key) == head.as_ref() {
                head_waiters.push(id);
                continue;
            }
            if waiter.prepare() {
                // Preparation never waits behind other compiles.
                self.answer(state, id, Answer::Ready, effects);
                continue;
            }
            waiter.on_compile = true;
            waiter.waiting_on_compile = true;
            let (key, bound) = (waiter.key.clone(), waiter.bound(id, state));
            self.check_bound(state, id, &key, bound, now, effects);
        }
        let Some(head) = head else {
            return;
        };
        // The head's requests are refused before anything is dispatched or
        // evicted for them when the shape cannot be ready by their bound.
        let ready = state.predicted_ready(&head, now);
        let mut keep = Vec::new();
        for id in head_waiters {
            let waiter = self.waiters.get_mut(&id).expect("listed above");
            if waiter.prepare() {
                keep.push(id);
                continue;
            }
            waiter.on_compile = true;
            waiter.waiting_on_compile = true;
            let bound = waiter.bound(id, state);
            match (bound, ready) {
                (Some(bound), Some(ready)) if ready > bound => {
                    let error = Self::refusal(state, &head, Some(ready), now);
                    self.answer(state, id, Answer::Failed(error), effects);
                }
                (Some(bound), _) => {
                    state.waiter_bounds.insert(id, bound);
                    keep.push(id);
                }
                (None, _) => keep.push(id),
            }
        }
        let head_waiters = keep;
        let promised = state
            .entries
            .get(&head)
            .and_then(|entry| entry.promise)
            .is_some_and(|promise| promise_live(promise, now));
        if head_waiters.is_empty() && !promised {
            return;
        }
        if self
            .op_kinds
            .values()
            .any(|kind| matches!(kind, OpKind::Evict { for_head: Some(key), .. } if *key == head))
        {
            return;
        }
        let only_prepare =
            !head_waiters.is_empty() && head_waiters.iter().all(|id| self.waiters[id].prepare());
        let entry = &state.entries[&head];
        let (worker, executables) = (entry.worker.clone(), entry.executables);
        let model_full = state.model_count(&head.model_ref) >= inner.limits.per_model;
        let total_full = state.slots.len() >= inner.limits.total;
        let executable_full =
            state.reserved_executables().saturating_add(executables) > inner.executable_budget;
        if !model_full && !total_full && !executable_full {
            // One compile at a time; the head starts once the gate is free and
            // its own gathering window has closed. Packed programs are prepared
            // only while nothing is queued, so they do not gather.
            let gathered =
                head.program != AneProgram::Single || state.entries[&head].gathered_at() <= now;
            if state.compile_running() || !gathered {
                if only_prepare {
                    for id in head_waiters {
                        self.answer(state, id, Answer::Ready, effects);
                    }
                }
                return;
            }
            let owner = head_waiters.first().copied();
            self.reserve(inner, state, owner, &head, &worker, executables);
            effects.progress = true;
            return;
        }
        // A compile kept only for a promise (nobody waits for it now) does not
        // evict ahead of its turn: it might expire before it could start, and
        // the shape it evicts would be lost for nothing.
        if head_waiters.is_empty() && state.compile_running() {
            return;
        }
        // When the model is at its own limit only one of its shapes frees the
        // slot it needs; otherwise any model's shape does. An idle stream
        // refunds promptly; queued demand and then the smallest rung break
        // ties before LRU. Busy streams remain a last resort.
        let victim = state
            .slots
            .iter()
            .filter(|(candidate, slot)| {
                slot.state == SlotState::Resident
                    && slot.leases == 0
                    && !slot.handing_off(now)
                    && !state.recovering.contains(slot.worker.worker_id())
                    && (!model_full || candidate.model_ref == head.model_ref)
            })
            .min_by_key(|(candidate, slot)| state.eviction_rank(candidate, slot))
            .map(|(candidate, slot)| {
                (
                    candidate.clone(),
                    slot.id,
                    slot.worker.clone(),
                    slot.generation,
                )
            });
        let Some((victim, slot_id, owner, generation)) = victim else {
            if only_prepare {
                for id in head_waiters {
                    self.answer(state, id, Answer::Ready, effects);
                }
                return;
            }
            for id in &head_waiters {
                self.waiters
                    .get_mut(id)
                    .expect("listed above")
                    .waiting_on_compile = false;
            }
            state.starved_head = Some(
                head_waiters
                    .first()
                    .copied()
                    .unwrap_or_else(|| inner.next_waiter.load(Ordering::Relaxed)),
            );
            return;
        };
        state.slots.get_mut(&victim).expect("selected above").state = SlotState::Evicting;
        tracing::debug!(
            target: "worker",
            model_ref = %victim.model_ref,
            shape = victim.shape,
            worker_id = %victim.worker_id,
            for_shape = head.shape,
            "direct-ANE compile queue: evicting to make room"
        );
        let trace = head_waiters
            .first()
            .map(|id| self.waiters[id].trace.clone())
            .unwrap_or_default();
        self.spawn_evict(
            state,
            victim,
            slot_id,
            generation,
            owner,
            Some(head),
            None,
            None,
            trace,
        );
        effects.progress = true;
    }

    /// A request waiting on a compile is refused at once when the queue
    /// predicts its shape will be ready only after its bound; otherwise it is
    /// accepted, and newer compiles will not be ordered ahead of it if that
    /// would break the bound.
    fn check_bound(
        &mut self,
        state: &mut State,
        id: u64,
        key: &ShapeKey,
        bound: Option<tokio::time::Instant>,
        now: tokio::time::Instant,
        effects: &mut Effects,
    ) {
        let Some(bound) = bound else {
            return;
        };
        let ready = state.predicted_ready(key, now);
        match ready {
            Some(ready) if ready > bound => {
                let error = Self::refusal(state, key, Some(ready), now);
                self.answer(state, id, Answer::Failed(error), effects);
            }
            Some(_) if state.entries.contains_key(key) => {
                state.waiter_bounds.insert(id, bound);
            }
            _ => {}
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn reserve(
        &mut self,
        inner: &Arc<Inner>,
        state: &mut State,
        owner: Option<u64>,
        key: &ShapeKey,
        worker: &Arc<dyn AneShapeWorker>,
        executables: usize,
    ) {
        let queued = state.entries.remove(key);
        let compile_cost = queued
            .as_ref()
            .map_or(Default::default(), |entry| state.estimate(key, entry.seed));
        let tick = state.tick();
        state.next_slot += 1;
        let slot_id = state.next_slot;
        let generation = worker.generation();
        // Every request for the shape now waits on the slot.
        let joined: BTreeSet<u64> = self
            .waiters
            .iter()
            .filter(|(_, waiter)| waiter.key == *key)
            .map(|(waiter, _)| *waiter)
            .collect();
        state.slots.insert(
            key.clone(),
            Slot {
                id: slot_id,
                worker: worker.clone(),
                state: SlotState::Admitting,
                leases: 0,
                last_used: tick,
                inventory: None,
                executables,
                generation,
                packed_qualified: false,
                joined,
                handoff_until: None,
                compile_cost,
                compile_seed: queued
                    .as_ref()
                    .map_or(Default::default(), |entry| entry.seed),
                queued_as: queued.as_ref().map(QueuedCompile::place),
                owner,
                promise: queued.as_ref().and_then(|entry| entry.promise),
                dispatched_at: None,
            },
        );
        tracing::debug!(
            target: "worker",
            model_ref = %key.model_ref,
            shape = key.shape,
            worker_id = %key.worker_id,
            predicted_ms = u64::try_from(compile_cost.as_millis()).unwrap_or(u64::MAX),
            "direct-ANE compile queue: admitting"
        );
        let trace = owner
            .and_then(|id| self.waiters.get(&id))
            .map(|waiter| waiter.trace.clone())
            .unwrap_or_default();
        #[cfg(test)]
        admission_trace::event_in(
            &trace.context,
            "reservation",
            &key.model_ref,
            key.shape,
            owner,
        );
        self.spawn_compile(
            inner,
            key.clone(),
            slot_id,
            generation,
            worker.clone(),
            false,
            trace,
        );
    }

    fn report_sender(&self) -> Option<mpsc::UnboundedSender<Msg>> {
        self.tx.upgrade()
    }

    #[allow(clippy::too_many_arguments)]
    fn spawn_compile(
        &mut self,
        inner: &Arc<Inner>,
        key: ShapeKey,
        slot_id: u64,
        generation: u64,
        worker: Arc<dyn AneShapeWorker>,
        retry: bool,
        trace: Trace,
    ) {
        // Only tests carry a trace context.
        #[cfg(not(test))]
        let _ = trace;
        let Some(report) = self.report_sender() else {
            return;
        };
        let waits = Arc::new(RequestWaits::new());
        let guard = RequestAbandonment(waits.clone());
        let cell = Arc::new(DispatchCell {
            at: Mutex::new(None),
            slot_id,
            report,
        });
        #[cfg(test)]
        let gate = inner.admission_gate.clone();
        #[cfg(not(test))]
        let _ = inner;
        let model_ref = key.model_ref.clone();
        let (shape, program) = (key.shape, key.program);
        let operation = async move {
            #[cfg(test)]
            let _serial_admission = match &gate {
                Some(gate) => Some(gate.lock().await),
                None => None,
            };
            let result = REQUEST_WAITS
                .scope(
                    waits,
                    COMPILE_DISPATCHED.scope(
                        cell.clone(),
                        worker.admit_program(generation, &model_ref, shape, program),
                    ),
                )
                .await;
            OpDone::Compile {
                result,
                dispatched: cell.at(),
                finished: tokio::time::Instant::now(),
            }
        };
        #[cfg(test)]
        let operation = admission_trace::scope(trace.context, operation);
        let task = self.ops.spawn(operation).id();
        self.op_kinds.insert(
            task,
            OpKind::Compile {
                key,
                slot_id,
                generation,
                retry,
            },
        );
        // Dropping the guard would withdraw the operation; it is kept until the
        // operation finishes or the scheduler withdraws it on purpose.
        self.op_guards.insert(task, guard);
    }

    #[allow(clippy::too_many_arguments)]
    fn spawn_evict(
        &mut self,
        state: &mut State,
        key: ShapeKey,
        slot_id: u64,
        generation: u64,
        owner: Arc<dyn AneShapeWorker>,
        for_head: Option<ShapeKey>,
        for_retry: Option<(ShapeKey, u64, String)>,
        reply: Option<oneshot::Sender<bool>>,
        trace: Trace,
    ) {
        // Only tests carry a trace context.
        #[cfg(not(test))]
        let _ = trace;
        let _ = state;
        let waits = Arc::new(RequestWaits::new());
        let guard = RequestAbandonment(waits.clone());
        let worker = owner.clone();
        let model_ref = key.model_ref.clone();
        let (shape, program) = (key.shape, key.program);
        let operation = async move {
            #[cfg(test)]
            admission_trace::event("eviction_dispatch", &model_ref, shape);
            let result = REQUEST_WAITS
                .scope(
                    waits,
                    worker.evict_program(generation, &model_ref, shape, program),
                )
                .await;
            #[cfg(test)]
            if result.is_ok() {
                admission_trace::event("eviction_ack", &model_ref, shape);
            }
            OpDone::Evict(result)
        };
        #[cfg(test)]
        let operation = admission_trace::scope(trace.context, operation);
        let task = self.ops.spawn(operation).id();
        self.op_kinds.insert(
            task,
            OpKind::Evict {
                key,
                slot_id,
                generation,
                owner,
                for_head,
                for_retry,
                reply,
            },
        );
        // Dropping the guard would withdraw the operation; it is kept until the
        // operation finishes or the scheduler withdraws it on purpose.
        self.op_guards.insert(task, guard);
    }

    fn start_recovery(
        &mut self,
        state: &mut State,
        worker: &Arc<dyn AneShapeWorker>,
        generation: u64,
        reply: Option<RecoverReply>,
        now: tokio::time::Instant,
        effects: &mut Effects,
    ) {
        let id = worker.worker_id().to_owned();
        if state.retired.contains(&id) {
            if let Some(reply) = reply {
                effects.recoveries.push((
                    reply,
                    Err(AneResidencyError::Channel("worker is retired".into())),
                ));
            }
            return;
        }
        if worker.generation() != generation {
            if let Some(reply) = reply {
                effects.recoveries.push((reply, Ok(())));
            }
            return;
        }
        // A closed worker cannot be restarted: its old process may still own
        // the Neural Engine. A recovery that ended without confirming the exit
        // but still holds the gate counts as closed too.
        let (finished, gave_up) = self.recoveries.get(&id).map_or((false, false), |recovery| {
            (recovery.finished, recovery.gave_up)
        });
        if self.closed.contains(&id) || finished {
            if let Some(reply) = reply {
                effects.recoveries.push((reply, Err(worker_closed(&id))));
            }
            return;
        }
        if gave_up {
            if let Some(reply) = reply {
                effects.recoveries.push((reply, Err(exit_not_confirmed())));
            }
            return;
        }
        let Some(inner) = self.inner.upgrade() else {
            return;
        };
        state.recovering.insert(id.clone());
        effects.progress = true;
        let recovery = self.recoveries.entry(id).or_insert_with(|| Recovery {
            worker: worker.clone(),
            generation,
            replies: Vec::new(),
            drain_deadline: now + inner.wait_timeout,
            started: false,
            exit_deadline: None,
            exit_confirmed: false,
            gave_up: false,
            finished: false,
        });
        if let Some(reply) = reply {
            recovery.replies.push(reply);
        }
    }

    /// Recovery stops new leases and waits for already dispatched work to
    /// finish (bounded by the admission wait timeout, after which active I/O
    /// is interrupted), then restarts the worker.
    fn drive_recoveries(
        &mut self,
        state: &mut State,
        now: tokio::time::Instant,
        effects: &mut Effects,
    ) {
        let ids: Vec<String> = self.recoveries.keys().cloned().collect();
        for id in ids {
            let recovery = self.recoveries.get_mut(&id).expect("listed above");
            if !recovery.started {
                let drained = state
                    .slots
                    .values()
                    .filter(|slot| slot.worker.worker_id() == id)
                    .all(|slot| slot.leases == 0 && slot.state != SlotState::Admitting);
                if !drained && now < recovery.drain_deadline {
                    continue;
                }
                if !drained {
                    tracing::warn!(target: "worker", worker_id = %id, "worker recovery drain deadline exceeded; forcing confirmed-exit restart");
                }
                if recovery.worker.generation() != recovery.generation {
                    let recovery = self.recoveries.remove(&id).expect("listed above");
                    state.rejoin(&id, now);
                    state.recovering.remove(&id);
                    for reply in recovery.replies {
                        effects.recoveries.push((reply, Ok(())));
                    }
                    effects.progress = true;
                    continue;
                }
                recovery.started = true;
                recovery.exit_deadline = Some(now + EXIT_CONFIRMATION_GIVE_UP);
                let worker = recovery.worker.clone();
                let generation = recovery.generation;
                self.spawn_recover(worker, generation);
                effects.progress = true;
            } else if !recovery.exit_confirmed
                && !recovery.gave_up
                && recovery
                    .exit_deadline
                    .is_some_and(|deadline| now >= deadline)
            {
                recovery.gave_up = true;
                let pid = recovery.worker.process_id();
                give_up_gate(state, &id, pid);
                // Whoever waits for this recovery is answered now rather than
                // when (or if) the restart operation returns. The worker stays
                // closed while that operation runs; it reopens only if the
                // operation still confirms the old process's exit.
                for reply in recovery.replies.drain(..) {
                    effects.recoveries.push((reply, Err(exit_not_confirmed())));
                }
                if recovery.finished {
                    self.recoveries.remove(&id);
                }
                effects.progress = true;
            }
        }
    }

    fn spawn_recover(&mut self, worker: Arc<dyn AneShapeWorker>, generation: u64) {
        let Some(report) = self.report_sender() else {
            return;
        };
        let worker_id = worker.worker_id().to_owned();
        let reported = worker_id.clone();
        let operation = async move {
            let (exited, exit) = oneshot::channel();
            let forward = async move {
                let exited = exit.await.is_ok();
                if exited {
                    let _ = report.send(Msg::ExitConfirmed {
                        worker_id: reported,
                        generation,
                    });
                }
                exited
            };
            // The restart runs as its own task so that a panic in it (for
            // example while the replacement starts) cannot drop the
            // forwarder above before it has reported an exit that was already
            // confirmed. Its guard aborts it if this operation is aborted.
            let restart = AbortOnDrop(tokio::spawn(async move {
                worker.restart_with_exit_signal(exited).await
            }));
            let (result, exited) = tokio::join!(restart.join(), forward);
            OpDone::Recover {
                result: result.unwrap_or_else(|error| {
                    Err(AneResidencyError::Panicked(format!(
                        "worker recovery task failed: {error}"
                    )))
                }),
                exited,
            }
        };
        let task = self.ops.spawn(operation).id();
        self.op_kinds.insert(task, OpKind::Recover { worker_id });
    }

    /// The old process has exited: every slot of that worker, whatever its
    /// state, belonged to it and is removed and refunded, so a failed
    /// replacement cannot leave charged or leasable ghost slots, and a
    /// failed compile's hold on the gate ends before the replacement starts.
    fn exit_confirmed(&mut self, state: &mut State, worker_id: &str, generation: u64) {
        if let Some(recovery) = self.recoveries.get_mut(worker_id) {
            if recovery.generation == generation {
                recovery.exit_confirmed = true;
            }
        }
        release_exited_worker(state, worker_id);
        tracing::debug!(target: "worker", worker_id, "direct-ANE worker exit confirmed; its slots are released");
    }

    fn drive_retirements(
        &mut self,
        state: &mut State,
        now: tokio::time::Instant,
        effects: &mut Effects,
    ) {
        for (id, retirement) in &mut self.retirements {
            if retirement.reply.is_none() {
                continue;
            }
            let drained = state
                .slots
                .values()
                .filter(|slot| slot.worker.worker_id() == id)
                .all(|slot| slot.leases == 0 && slot.state != SlotState::Admitting);
            if drained || now >= retirement.deadline {
                effects
                    .units
                    .push(retirement.reply.take().expect("checked above"));
            }
        }
    }

    /// Entries nobody waits for are kept until their promise expires.
    fn expire_entries(&mut self, state: &mut State, now: tokio::time::Instant) {
        let expired: Vec<ShapeKey> = state
            .entries
            .iter()
            .filter(|(key, entry)| {
                !state.requested(key)
                    && !entry
                        .promise
                        .is_some_and(|promise| promise_live(promise, now))
            })
            .map(|(key, _)| key.clone())
            .collect();
        for key in expired {
            state.entries.remove(&key);
            tracing::debug!(
                target: "worker",
                model_ref = %key.model_ref,
                shape = key.shape,
                worker_id = %key.worker_id,
                "direct-ANE compile queue: queued compile expired, nobody came back for it"
            );
        }
    }

    fn next_wake(&self, state: &State, now: tokio::time::Instant) -> Option<tokio::time::Instant> {
        let bounds = self
            .waiters
            .iter()
            .filter_map(|(id, waiter)| waiter.bound(*id, state));
        let entries = state.entries.iter().flat_map(|(key, entry)| {
            let expiry = (!state.requested(key))
                .then(|| entry.orphan_expiry())
                .flatten();
            [Some(entry.gathered_at()), expiry]
        });
        let handoffs = state
            .slots
            .values()
            .filter(|slot| slot.handing_off(now))
            .filter_map(|slot| slot.handoff_until);
        let recoveries = self.recoveries.values().filter_map(|recovery| {
            if !recovery.started {
                Some(recovery.drain_deadline)
            } else if !recovery.exit_confirmed && !recovery.gave_up {
                recovery.exit_deadline
            } else {
                None
            }
        });
        let retirements = self
            .retirements
            .values()
            .filter(|retirement| retirement.reply.is_some())
            .map(|retirement| retirement.deadline);
        // An undispatched compile is withdrawn when its promise lapses.
        let lapses = state
            .slots
            .values()
            .filter(|slot| slot.state == SlotState::Admitting && slot.dispatched_at.is_none())
            .filter_map(|slot| slot.promise)
            .map(|(at, delay)| at + delay * ORPHAN_EXPIRY_FACTOR);
        bounds
            .chain(entries.flatten())
            .chain(handoffs)
            .chain(recoveries)
            .chain(retirements)
            .chain(lapses)
            .filter(|at| *at > now)
            .min()
    }

    fn finish(
        &mut self,
        done: Result<(tokio::task::Id, OpDone), tokio::task::JoinError>,
        now: tokio::time::Instant,
    ) {
        let (task, done) = match done {
            Ok((task, done)) => (task, Ok(done)),
            Err(error) => (error.id(), Err(error)),
        };
        self.op_guards.remove(&task);
        let Some(kind) = self.op_kinds.remove(&task) else {
            return;
        };
        let Some(inner) = self.inner.upgrade() else {
            return;
        };
        let mut effects = Effects::default();
        {
            let mut state = inner.lock();
            match kind {
                OpKind::Compile {
                    key,
                    slot_id,
                    generation,
                    retry,
                } => {
                    let (result, dispatched, finished) = match done {
                        Ok(OpDone::Compile {
                            result,
                            dispatched,
                            finished,
                        }) => (result, dispatched, finished),
                        Ok(_) => return,
                        Err(error) => (
                            Err(AneResidencyError::Panicked(format!(
                                "shape admission task failed: {error}"
                            ))),
                            None,
                            now,
                        ),
                    };
                    self.compiled(
                        &inner,
                        &mut state,
                        key,
                        slot_id,
                        generation,
                        retry,
                        result,
                        dispatched,
                        finished,
                        now,
                        &mut effects,
                    );
                }
                OpKind::Evict {
                    key,
                    slot_id,
                    generation,
                    owner,
                    for_retry,
                    reply,
                    ..
                } => {
                    let result = match done {
                        Ok(OpDone::Evict(result)) => result,
                        Ok(_) => return,
                        Err(error) => Err(AneResidencyError::Panicked(format!(
                            "shape eviction task failed: {error}"
                        ))),
                    };
                    let evicted = result.is_ok();
                    self.evicted(
                        &inner,
                        &mut state,
                        key,
                        slot_id,
                        generation,
                        owner,
                        for_retry,
                        result,
                        now,
                        &mut effects,
                    );
                    if let Some(reply) = reply {
                        effects.bools.push((reply, evicted));
                    }
                }
                OpKind::Recover { worker_id } => {
                    let (result, exited) = match done {
                        Ok(OpDone::Recover { result, exited }) => (result, exited),
                        Ok(_) => return,
                        Err(error) => (
                            Err(AneResidencyError::Panicked(format!(
                                "worker recovery task failed: {error}"
                            ))),
                            false,
                        ),
                    };
                    if exited {
                        if let Some(recovery) = self.recoveries.get_mut(&worker_id) {
                            if !recovery.exit_confirmed {
                                recovery.exit_confirmed = true;
                                release_exited_worker(&mut state, &worker_id);
                            }
                        }
                    }
                    self.recovered(&mut state, &worker_id, result, now, &mut effects);
                }
            }
            effects.progress = true;
        }
        self.apply(effects);
    }

    /// Answers the request that reserved a slot, if it still waits.
    fn answer_owner(
        &mut self,
        state: &mut State,
        key: &ShapeKey,
        owner: Option<u64>,
        error: AneResidencyError,
        effects: &mut Effects,
    ) {
        if let Some(owner) = owner.filter(|owner| self.waiters.contains_key(owner)) {
            let _ = key;
            self.answer(state, owner, Answer::Failed(error), effects);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn compiled(
        &mut self,
        inner: &Arc<Inner>,
        state: &mut State,
        key: ShapeKey,
        slot_id: u64,
        generation: u64,
        retry: bool,
        result: Result<AnePlacementInventory, AneResidencyError>,
        dispatched: Option<tokio::time::Instant>,
        finished: tokio::time::Instant,
        now: tokio::time::Instant,
        effects: &mut Effects,
    ) {
        let Some(slot) = state
            .slots
            .get(&key)
            .filter(|slot| slot.id == slot_id && slot.state == SlotState::Admitting)
        else {
            // The slot went with its worker's confirmed exit: late result.
            return;
        };
        let owner = slot.owner;
        let worker = slot.worker.clone();
        match result {
            Ok(inventory) => {
                let tick = state.tick();
                let slot = state.slots.get_mut(&key).expect("checked above");
                let inventory = Arc::new(inventory);
                let seed = slot.compile_seed;
                slot.state = SlotState::Resident;
                slot.last_used = tick;
                slot.inventory = Some(inventory.clone());
                slot.promise = None;
                if let Some(sink) = state.inventory_sink.as_mut() {
                    sink.push(Arc::clone(&inventory));
                }
                state.stats.admitted += 1;
                // Learn only from a standalone compile, measured from the
                // moment it reached the worker.
                if let (false, Some(dispatched)) = (retry, dispatched) {
                    state.learn(&key, seed, finished.saturating_duration_since(dispatched));
                }
                state.sample();
                tracing::debug!(
                    target: "worker",
                    model_ref = %key.model_ref,
                    shape = key.shape,
                    worker_id = %key.worker_id,
                    "direct-ANE compile queue: compiled"
                );
                // Every request waiting for the shape is told it is ready and
                // may lease it ahead of a starved head for the 250 ms handoff.
                let cohort: Vec<u64> = self
                    .waiters
                    .iter()
                    .filter(|(_, waiter)| waiter.key == key)
                    .map(|(id, _)| *id)
                    .collect();
                let slot = state.slots.get_mut(&key).expect("checked above");
                slot.joined.clear();
                slot.handoff_until = Some(now + ADMISSION_HANDOFF);
                for id in cohort {
                    if self.waiters[&id].prepare() {
                        let tick = state.tick();
                        let slot = state.slots.get_mut(&key).expect("checked above");
                        slot.leases += 1;
                        slot.last_used = tick;
                        let lease = AneShapeLease {
                            inner: inner.clone(),
                            key: key.clone(),
                            slot_id,
                            inventory: slot.inventory.clone(),
                        };
                        self.answer(
                            state,
                            id,
                            Answer::Lease {
                                lease,
                                compiled: false,
                            },
                            effects,
                        );
                        continue;
                    }
                    // Answering removes the request from every cohort, so it
                    // joins this one afterwards.
                    self.answer(state, id, Answer::Ready, effects);
                    state
                        .slots
                        .get_mut(&key)
                        .expect("checked above")
                        .joined
                        .insert(id);
                }
            }
            Err(AneResidencyError::Abandoned) => {
                // Withdrawn before it reached the worker: nothing compiled.
                // Requeue it in its old place if anyone still waits for it or
                // was promised it.
                let slot = state.slots.remove(&key).expect("checked above");
                let wanted = self.waiters.values().any(|waiter| waiter.key == key)
                    || slot
                        .promise
                        .is_some_and(|promise| promise_live(promise, now));
                if let (true, Some(place)) = (wanted, slot.queued_as) {
                    state.entries.insert(
                        key.clone(),
                        QueuedCompile {
                            seq: place.seq,
                            enqueued_at: place.enqueued_at,
                            start_key: place.start_key,
                            seed: slot.compile_seed,
                            worker: slot.worker.clone(),
                            executables: slot.executables,
                            promise: slot.promise,
                        },
                    );
                }
            }
            Err(error)
                if dispatched.is_none() && !matches!(error, AneResidencyError::Panicked(_)) =>
            {
                // Refused before the admission reached the worker (the
                // channel is faulted or quarantined, not connected, or of
                // another generation): nothing is compiling and the worker
                // holds nothing for it. So the slot goes at once, without
                // holding the gate, and the worker is not restarted for it; a
                // fault that needs a restart is recovered by the request that
                // met it.
                state.slots.remove(&key);
                self.answer_owner(state, &key, owner, error, effects);
            }
            Err(error) if worker.generation() != generation => {
                // A recovery restarted the worker under this compile; its exit
                // confirmation removes the slot.
                state.slots.get_mut(&key).expect("checked above").state = SlotState::Failed;
                self.answer_owner(state, &key, owner, resource_refusal(error), effects);
            }
            Err(error) if error.code() == Some(ERR_ANE_RESOURCES_EXHAUSTED) && !retry => {
                // Keep this shape's reservation (and the gate) while freeing
                // one physical resident, then retry once. The worker owns
                // partial rollback.
                let victim = state.resource_victim(now);
                match victim {
                    None => {
                        state.slots.remove(&key);
                        self.answer_owner(state, &key, owner, resource_refusal(error), effects);
                    }
                    Some((victim, victim_id, victim_owner, victim_generation)) => {
                        state.slots.get_mut(&victim).expect("selected above").state =
                            SlotState::Evicting;
                        self.spawn_evict(
                            state,
                            victim,
                            victim_id,
                            victim_generation,
                            victim_owner,
                            None,
                            Some((key, slot_id, error.to_string())),
                            None,
                            Default::default(),
                        );
                    }
                }
            }
            Err(error) if error.code() == Some(ERR_ANE_RESOURCES_EXHAUSTED) => {
                // Refused again after an eviction: one retry only. The worker
                // is restarted (after confirmed exit) rather than evicting
                // further; the request gets a transient refusal.
                state.slots.get_mut(&key).expect("checked above").state = SlotState::Rejected;
                self.start_recovery(state, &worker, generation, None, now, effects);
                self.answer_owner(state, &key, owner, resource_refusal(error), effects);
            }
            Err(error)
                if key.program != AneProgram::Single
                    && error.code() == Some("unsupported_request") =>
            {
                state.slots.remove(&key);
                self.answer_owner(state, &key, owner, error, effects);
            }
            Err(error) => {
                // A definite refusal from the worker means nothing can still
                // be compiling: release the gate at once (the reservation stays
                // charged until recovery). Any other failure leaves the outcome
                // unknown: hold the gate until the old process's exit is
                // confirmed.
                let definite = matches!(error, AneResidencyError::WorkerErr { .. });
                state.slots.get_mut(&key).expect("checked above").state = if definite {
                    SlotState::Rejected
                } else {
                    SlotState::Failed
                };
                tracing::debug!(
                    target: "worker",
                    model_ref = %key.model_ref,
                    shape = key.shape,
                    worker_id = %key.worker_id,
                    definite,
                    error = %error,
                    "direct-ANE compile queue: compile failed; recovering the worker"
                );
                self.start_recovery(state, &worker, generation, None, now, effects);
                self.answer_owner(state, &key, owner, error, effects);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn evicted(
        &mut self,
        inner: &Arc<Inner>,
        state: &mut State,
        key: ShapeKey,
        slot_id: u64,
        generation: u64,
        owner: Arc<dyn AneShapeWorker>,
        for_retry: Option<(ShapeKey, u64, String)>,
        result: Result<(), AneResidencyError>,
        now: tokio::time::Instant,
        effects: &mut Effects,
    ) {
        match result {
            Ok(()) => {
                if state.slots.get(&key).is_some_and(|slot| slot.id == slot_id) {
                    state.slots.remove(&key);
                    state.stats.evicted += 1;
                    state.sample();
                }
                if let Some((retry_key, retry_slot, _)) = for_retry {
                    self.retry_compile(inner, state, retry_key, retry_slot);
                }
            }
            Err(AneResidencyError::Abandoned) => {
                // No eviction reached the wire, so the worker still owns the
                // resident program and its executable charge is unchanged.
                if let Some(slot) = state
                    .slots
                    .get_mut(&key)
                    .filter(|slot| slot.id == slot_id && slot.state == SlotState::Evicting)
                {
                    slot.state = SlotState::Resident;
                }
            }
            Err(_) if owner.generation() != generation => {
                // The owner was restarted under this eviction; its exit
                // confirmation removes the slot.
                if let Some((retry_key, retry_slot, _)) = for_retry {
                    self.retry_compile(inner, state, retry_key, retry_slot);
                }
            }
            Err(error) => {
                tracing::warn!(
                    target: "worker",
                    worker_id = owner.worker_id(),
                    model_ref = %key.model_ref,
                    shape = key.shape,
                    error = %error,
                    "direct-ANE evict failed; restarting the worker"
                );
                self.start_recovery(state, &owner, generation, None, now, effects);
                if let Some((retry_key, retry_slot, refusal)) = for_retry {
                    let retry_owner = state
                        .slots
                        .get(&retry_key)
                        .filter(|slot| slot.id == retry_slot)
                        .map(|slot| (slot.owner, slot.worker.worker_id() == owner.worker_id()));
                    if let Some((retry_owner, same_worker)) = retry_owner {
                        if same_worker {
                            state
                                .slots
                                .get_mut(&retry_key)
                                .expect("checked above")
                                .state = SlotState::Failed;
                        } else {
                            state.slots.remove(&retry_key);
                        }
                        self.answer_owner(
                            state,
                            &retry_key,
                            retry_owner,
                            resource_refusal(AneResidencyError::Channel(refusal)),
                            effects,
                        );
                    }
                }
            }
        }
    }

    fn retry_compile(
        &mut self,
        inner: &Arc<Inner>,
        state: &mut State,
        key: ShapeKey,
        slot_id: u64,
    ) {
        let Some(slot) = state
            .slots
            .get(&key)
            .filter(|slot| slot.id == slot_id && slot.state == SlotState::Admitting)
        else {
            return;
        };
        let (worker, generation) = (slot.worker.clone(), slot.generation);
        let trace = slot
            .owner
            .and_then(|owner| self.waiters.get(&owner))
            .map(|waiter| waiter.trace.clone())
            .unwrap_or_default();
        self.spawn_compile(inner, key, slot_id, generation, worker, true, trace);
    }

    fn recovered(
        &mut self,
        state: &mut State,
        worker_id: &str,
        result: Result<(), AneResidencyError>,
        now: tokio::time::Instant,
        effects: &mut Effects,
    ) {
        let Some(mut recovery) = self.recoveries.remove(worker_id) else {
            return;
        };
        let outcome = match result {
            Ok(()) => {
                state
                    .slots
                    .retain(|_, slot| slot.worker.worker_id() != worker_id);
                state.rejoin(worker_id, now);
                state.recovering.remove(worker_id);
                self.closed.remove(worker_id);
                state.stats.restarts += 1;
                Ok(())
            }
            // Only the old process's confirmed exit lets its slots go. The
            // error alone says nothing about it: a restart refused because the
            // worker is being retired, or one that panicked, may never have
            // stopped the old process.
            Err(error) if recovery.exit_confirmed => {
                tracing::warn!(target: "worker", worker_id, error = %error, "direct-ANE worker restart failed after the old process exited");
                // Only the replacement failed. The old process's slots are
                // gone with it, and a later request can recover the worker.
                release_exited_worker(state, worker_id);
                state.rejoin(worker_id, now);
                state.recovering.remove(worker_id);
                self.closed.remove(worker_id);
                Err(error)
            }
            Err(error) => {
                tracing::warn!(target: "worker", worker_id, error = %error, "direct-ANE worker restart failed without confirming the old process's exit");
                // The old process's hardware ownership is unknown: the worker
                // stays closed with its slots charged, and its queued compiles
                // (which can never run) are dropped. Its compiling or failed
                // slots keep holding the gate until `exit_deadline`, exactly
                // as if the restart were still running, so an early error (or
                // a panic) cannot release the gate sooner than that rule.
                state.entries.retain(|key, _| key.worker_id != worker_id);
                self.closed.insert(worker_id.to_owned());
                if !recovery.gave_up {
                    recovery.finished = true;
                    let replies = std::mem::take(&mut recovery.replies);
                    self.recoveries.insert(worker_id.to_owned(), recovery);
                    for reply in replies {
                        effects
                            .recoveries
                            .push((reply, Err(worker_closed(worker_id))));
                    }
                    return;
                }
                Err(worker_closed(worker_id))
            }
        };
        for reply in recovery.replies {
            let outcome = match &outcome {
                Ok(()) => Ok(()),
                Err(error) => Err(reissue(error)),
            };
            effects.recoveries.push((reply, outcome));
        }
    }
}

/// A copy of a recovery's error for each caller waiting on it (the error
/// type is not `Clone`, because of its I/O variant).
fn reissue(error: &AneResidencyError) -> AneResidencyError {
    match error {
        AneResidencyError::WorkerClosed { worker_id } => worker_closed(worker_id),
        AneResidencyError::Panicked(message) => AneResidencyError::Panicked(message.clone()),
        AneResidencyError::WorkerErr { code, msg } => AneResidencyError::WorkerErr {
            code: code.clone(),
            msg: msg.clone(),
        },
        other => AneResidencyError::Channel(other.to_string()),
    }
}

/// The refusal for a worker closed until the module restarts.
fn worker_closed(worker_id: &str) -> AneResidencyError {
    AneResidencyError::WorkerClosed {
        worker_id: worker_id.to_owned(),
    }
}

/// The answer to a recovery whose old process's exit was not confirmed within
/// `EXIT_CONFIRMATION_GIVE_UP` while its restart is still running.
fn exit_not_confirmed() -> AneResidencyError {
    AneResidencyError::Channel(format!(
        "the old worker process's exit was not confirmed within {} s; the worker stays closed while its restart continues",
        EXIT_CONFIRMATION_GIVE_UP.as_secs()
    ))
}

/// Aborts a spawned task when dropped, so a task spawned by an operation does
/// not outlive that operation being aborted.
struct AbortOnDrop<T>(tokio::task::JoinHandle<T>);

impl<T> AbortOnDrop<T> {
    async fn join(mut self) -> Result<T, tokio::task::JoinError> {
        (&mut self.0).await
    }
}

impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Every slot of a worker whose old process has exited is removed and its
/// reservation refunded, whatever its state: nothing of that process can be
/// compiling, resident or leased any more.
fn release_exited_worker(state: &mut State, exited: &str) {
    state
        .slots
        .retain(|_, slot| slot.worker.worker_id() != exited);
}

/// Stops a worker whose process's exit cannot be confirmed from holding the
/// compile gate: its compiling or failed slots stay charged but no longer
/// block other workers' compiles.
fn give_up_gate(state: &mut State, worker_id: &str, pid: Option<u32>) {
    let mut held = false;
    for slot in state.slots.values_mut() {
        if slot.worker.worker_id() == worker_id
            && matches!(slot.state, SlotState::Admitting | SlotState::Failed)
        {
            slot.state = SlotState::Abandoned;
            held = true;
        }
    }
    tracing::warn!(
        target: "worker",
        worker_id,
        pid = pid.map_or_else(|| "unknown".to_owned(), |pid| pid.to_string()),
        held_compile_gate = held,
        "direct-ANE worker exit could not be confirmed after SIGKILL and the grace period; \
         the worker stays closed with its reservations charged, and the compile gate is released"
    );
}

fn end_handoffs(state: &mut State, now: tokio::time::Instant) {
    for slot in state.slots.values_mut() {
        if slot.handoff_until.is_some()
            && slot.state == SlotState::Resident
            && !slot.handing_off(now)
        {
            slot.handoff_until = None;
            slot.joined.clear();
        }
    }
}

fn promise_live(
    promise: (tokio::time::Instant, std::time::Duration),
    now: tokio::time::Instant,
) -> bool {
    now < promise.0 + promise.1 * ORPHAN_EXPIRY_FACTOR
}

async fn sleep_until_option(wake: Option<tokio::time::Instant>) {
    match wake {
        Some(at) => tokio::time::sleep_until(at).await,
        None => std::future::pending::<()>().await,
    }
}
