// These tests control the worker reply and hold requests unpolled at chosen
// points, so scheduler wakeup order cannot hide a lost single-flight handoff.

/// Sends a request for `shape` straight to the scheduler and returns its id
/// and answer receiver without polling it, as a request whose task is not
/// running would leave it.
fn raw_want(
    supervisor: &AneResidencySupervisor,
    worker: &Arc<dyn AneShapeWorker>,
    model_ref: &str,
    shape: usize,
) -> (u64, tokio::sync::oneshot::Receiver<scheduler::Answer>) {
    let id = supervisor.inner.next_waiter.fetch_add(1, Ordering::Relaxed) + 1;
    let (reply, answer) = tokio::sync::oneshot::channel();
    supervisor
        .inner
        .send(scheduler::Msg::Want(scheduler::Want {
            id,
            key: ShapeKey {
                worker_id: worker.worker_id().into(),
                model_ref: model_ref.into(),
                shape,
                program: AneProgram::Single,
            },
            worker: worker.clone(),
            executables: worker.executable_count(model_ref).unwrap(),
            bound: Some(tokio::time::Instant::now() + supervisor.inner.wait_timeout),
            deadline: None,
            reply: scheduler::Reply::Lease(reply),
            trace: scheduler::Trace::default(),
        }))
        .unwrap();
    (id, answer)
}

#[tokio::test]
async fn joined_admission_hands_off_before_a_starved_head_and_excludes_newcomers() {
    let ledger = SharedLedger::default();
    let worker: Arc<dyn AneShapeWorker> = mock_channel("joined", &ledger).await;
    let supervisor = budget_supervisor(4, Duration::from_secs(2));
    let key = |shape| ShapeKey {
        worker_id: "joined".into(),
        model_ref: "model".into(),
        shape,
        program: AneProgram::Single,
    };
    let (started, release) = notify_pair();
    ledger.lock().unwrap().admit_gate = Some((started.clone(), release.clone()));
    let (original, original_answer) = raw_want(&supervisor, &worker, "model", 128);
    notified(&started).await;
    let head = tokio::spawn({
        let supervisor = supervisor.clone();
        let worker = worker.clone();
        async move { supervisor.lease(&worker, "model", 256).await }
    });
    until(|| supervisor.inner.lock().starved_head.is_some()).await;
    let joiners: Vec<_> = (0..8)
        .map(|_| raw_want(&supervisor, &worker, "model", 128))
        .collect();
    until(|| supervisor.inner.lock().slots[&key(128)].joined.len() == 9).await;
    release.notify_one();
    until(|| {
        supervisor
            .inner
            .lock()
            .slots
            .get(&key(128))
            .is_none_or(|slot| slot.state != SlotState::Admitting)
    })
    .await;
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(
        supervisor
            .inner
            .lock()
            .resource_victim(tokio::time::Instant::now())
            .is_none(),
        "resource-refusal recovery must also protect the pending handoff"
    );
    assert!(
        !head.is_finished() && !ledger.lock().unwrap().events.contains(&"evict model 128".into()),
        "newly ready shape was selected before its eight joiners leased it"
    );
    assert!(
        supervisor.try_lease(&key(128), &worker, None).unwrap().is_none(),
        "post-completion traffic must not extend the handoff"
    );
    // The handoff's own timer; the head must not have to wait for it.
    let handoff_until = supervisor.inner.lock().slots[&key(128)]
        .handoff_until
        .expect("the completed shape opened a handoff");
    let mut members = std::iter::once((original, original_answer))
        .chain(joiners)
        .peekable();
    while let Some((id, answer)) = members.next() {
        let scheduler::Answer::Joined(membership) = answer.await.unwrap() else {
            panic!("a pre-completion joiner was not handed the shape");
        };
        let Some(lease) = supervisor.try_lease(&key(128), &worker, Some(id)).unwrap() else {
            panic!("a pre-completion joiner lost its handoff to the budget head");
        };
        // Leasing through the cohort already left it.
        drop(membership);
        if members.peek().is_some() {
            drop(lease);
            // Let the wakeup from this lease drop be handled first.
            tokio::time::sleep(Duration::from_millis(5)).await;
            continue;
        }
        // The last member still holds its lease, so no lease drop wakes the
        // scheduler: only the last-member nudge can end the handoff before its
        // timer.
        tokio::time::timeout(Duration::from_millis(100), async {
            while supervisor.inner.lock().slots[&key(128)]
                .handoff_until
                .is_some()
            {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("the handoff outlived its last member's lease until its timer");
        drop(lease);
    }
    drop(
        tokio::time::timeout(Duration::from_secs(2), head)
            .await
            .expect("the finite handoff must stop protecting the shape after its cohort drains")
            .unwrap()
            .unwrap(),
    );
    assert!(
        tokio::time::Instant::now() < handoff_until,
        "the head waited for the handoff timer although the last cohort member had leased"
    );
    assert!(ledger.lock().unwrap().events.contains(&"evict model 128".into()));
}

/// Eviction still picks the idle stream while another worker's compile runs,
/// so the head's budget is ready when its turn comes. Its own compile, though,
/// waits for the running one to end: this test used to assert the opposite
/// (that the head admitted before the long compile ended), which is exactly
/// the side-by-side compiling the compile queue removes, because Apple's
/// compiler serves one layer at a time machine-wide.
#[tokio::test]
async fn eviction_prefers_an_idle_stream_and_evicts_before_the_long_compile_ends() {
    let busy_ledger = SharedLedger::default();
    let idle_ledger = SharedLedger::default();
    let busy: Arc<dyn AneShapeWorker> = mock_channel("busy", &busy_ledger).await;
    let idle: Arc<dyn AneShapeWorker> = mock_channel("idle", &idle_ledger).await;
    let target: Arc<dyn AneShapeWorker> = mock_channel("target", &SharedLedger::default()).await;
    let supervisor = budget_supervisor(12, Duration::from_secs(2));
    let cheap = supervisor.lease(&busy, "busy-model", 256).await.unwrap();
    let pinned = supervisor.lease(&idle, "idle-model", 128).await.unwrap();
    supervisor
        .inner
        .lock()
        .smallest_rungs
        .insert("idle-model".into(), 128);
    let (started, release) = notify_pair();
    busy_ledger.lock().unwrap().admit_gate = Some((started.clone(), release.clone()));
    let compile = tokio::spawn({
        let supervisor = supervisor.clone();
        let busy = busy.clone();
        async move { supervisor.lease(&busy, "busy-model", 512).await }
    });
    notified(&started).await;
    drop((cheap, pinned));
    let mut head = tokio::spawn({
        let supervisor = supervisor.clone();
        async move { supervisor.lease(&target, "target-model", 128).await }
    });
    let before_compile_end = tokio::time::timeout(Duration::from_millis(250), &mut head).await;
    let idle_chosen = idle_ledger
        .lock()
        .unwrap()
        .events
        .contains(&"evict idle-model 128".into());
    let still_compiling = !compile.is_finished();
    release.notify_one();
    drop(compile.await.unwrap().unwrap());
    assert!(
        idle_chosen && still_compiling,
        "budget head queued eviction behind the long compile instead of choosing the idle stream"
    );
    assert!(
        before_compile_end.is_err(),
        "a second compile was admitted while another was still running"
    );
    drop(
        tokio::time::timeout(Duration::from_secs(2), head)
            .await
            .expect("the head's compile starts once the long compile ends")
            .unwrap()
            .unwrap(),
    );
}

#[tokio::test]
async fn eviction_uses_a_busy_stream_when_it_is_the_only_option() {
    let ledger = SharedLedger::default();
    let busy: Arc<dyn AneShapeWorker> = mock_channel("busy", &ledger).await;
    let target: Arc<dyn AneShapeWorker> = mock_channel("target", &SharedLedger::default()).await;
    let supervisor = budget_supervisor(8, Duration::from_secs(2));
    let cheap = supervisor.lease(&busy, "busy-model", 128).await.unwrap();
    let (started, release) = notify_pair();
    ledger.lock().unwrap().admit_gate = Some((started.clone(), release.clone()));
    let compile = tokio::spawn({
        let supervisor = supervisor.clone();
        let busy = busy.clone();
        async move { supervisor.lease(&busy, "busy-model", 256).await }
    });
    notified(&started).await;
    drop(cheap);
    let head = tokio::spawn({
        let supervisor = supervisor.clone();
        async move { supervisor.lease(&target, "target-model", 128).await }
    });
    let selected = tokio::time::timeout(Duration::from_millis(250), async {
        loop {
            if supervisor
                .inner
                .lock()
                .slots
                .values()
                .any(|slot| slot.state == SlotState::Evicting)
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await;
    assert!(
        !head.is_finished(),
        "refund must await the eviction acknowledgement"
    );
    assert_eq!(supervisor.inner.lock().reserved_executables(), 8);
    release.notify_one();
    drop(compile.await.unwrap().unwrap());
    assert!(
        selected.is_ok(),
        "the only busy-stream victim was never selected"
    );
    drop(
        tokio::time::timeout(Duration::from_secs(2), head)
            .await
            .unwrap()
            .unwrap()
            .unwrap(),
    );
    assert!(ledger
        .lock()
        .unwrap()
        .events
        .contains(&"evict busy-model 128".into()));
}

#[tokio::test]
async fn suspended_joiner_handoff_expires_without_an_external_wakeup() {
    let ledger = SharedLedger::default();
    let worker: Arc<dyn AneShapeWorker> = mock_channel("worker", &ledger).await;
    let supervisor = budget_supervisor(4, Duration::from_secs(2));
    let (started, release) = notify_pair();
    ledger.lock().unwrap().admit_gate = Some((started.clone(), release.clone()));
    let owner = tokio::spawn({
        let supervisor = supervisor.clone();
        let worker = worker.clone();
        async move { supervisor.lease(&worker, "model", 128).await }
    });
    notified(&started).await;
    let head = tokio::spawn({
        let supervisor = supervisor.clone();
        let worker = worker.clone();
        async move { supervisor.lease(&worker, "model", 256).await }
    });
    until(|| supervisor.inner.lock().starved_head.is_some()).await;
    // A joiner whose task never runs again: it holds its "ready" answer
    // without leasing.
    let (_suspended, _answer) = raw_want(&supervisor, &worker, "model", 128);
    release.notify_one();
    drop(owner.await.unwrap().unwrap());
    // No joined request is polled or cancelled, so only the handoff timer can
    // wake the head. Its normal admission backstop is deliberately much later.
    drop(
        tokio::time::timeout(Duration::from_secs(1), head)
            .await
            .expect("250 ms handoff did not expire autonomously")
            .unwrap()
            .unwrap(),
    );
    assert_eq!(ADMISSION_HANDOFF, Duration::from_millis(250));
}

#[tokio::test]
async fn eviction_prefers_no_queued_demand_before_smallest_rung_and_lru() {
    let ledger = SharedLedger::default();
    let worker: Arc<dyn AneShapeWorker> = mock_channel("worker", &ledger).await;
    let supervisor = budget_supervisor(8, Duration::from_secs(2));
    let demanded = supervisor.lease(&worker, "model", 256).await.unwrap();
    let other = supervisor.lease(&worker, "model", 128).await.unwrap();
    supervisor
        .inner
        .lock()
        .smallest_rungs
        .insert("model".into(), 128);
    let head = tokio::spawn({
        let supervisor = supervisor.clone();
        let worker = worker.clone();
        async move { supervisor.lease(&worker, "model", 512).await }
    });
    until(|| supervisor.inner.lock().starved_head.is_some()).await;
    // Queued behind the starved head, a request for the resident 256 waits.
    let waiter = tokio::spawn({
        let supervisor = supervisor.clone();
        let worker = worker.clone();
        async move { supervisor.lease(&worker, "model", 256).await }
    });
    until(|| supervisor.inner.lock().waiters.len() == 2).await;
    drop((demanded, other));
    drop(
        tokio::time::timeout(Duration::from_secs(2), head)
            .await
            .unwrap()
            .unwrap()
            .unwrap(),
    );
    let evictions: Vec<String> = ledger
        .lock()
        .unwrap()
        .events
        .iter()
        .filter(|event| event.starts_with("evict "))
        .cloned()
        .collect();
    assert_eq!(
        evictions.first().map(String::as_str),
        Some("evict model 128"),
        "queued demand should outweigh the smallest-rung preference"
    );
    drop(waiter.await.unwrap().unwrap());
}

#[tokio::test]
async fn admission_timeline_records_correlated_queue_stream_rpc_eviction_and_lease_events() {
    let ledger = SharedLedger::default();
    let worker: Arc<dyn AneShapeWorker> = mock_channel("worker", &ledger).await;
    let supervisor = budget_supervisor(4, Duration::from_secs(2));
    let timeline = admission_trace::Timeline::new();
    let context = admission_trace::Context::new(timeline.clone(), "request".into());
    admission_trace::scope(Some(context), async {
        drop(supervisor.lease(&worker, "model", 128).await.unwrap());
        drop(supervisor.lease(&worker, "model", 256).await.unwrap());
    })
    .await;
    let events = timeline.snapshot();
    let names: Vec<_> = events
        .iter()
        .map(|event| event["event"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        [
            "queue_entry",
            "reservation",
            "stream_grant",
            "rpc_start",
            "rpc_end",
            "lease_grant",
            "queue_entry",
            "eviction_dispatch",
            "stream_grant",
            "rpc_start",
            "rpc_end",
            "eviction_ack",
            "reservation",
            "stream_grant",
            "rpc_start",
            "rpc_end",
            "lease_grant"
        ]
    );
    assert!(events
        .windows(2)
        .all(|pair| pair[0]["at_ms"].as_f64().unwrap() <= pair[1]["at_ms"].as_f64().unwrap()));
    for (index, event) in events.iter().enumerate() {
        assert_eq!(event["request_id"], "request");
        assert_eq!(event["model"], "model");
        assert_eq!(event["ticket"], if index < 6 { 1 } else { 2 });
        assert_eq!(
            event["shape"],
            if index < 6 || (7..12).contains(&index) {
                128
            } else {
                256
            }
        );
        assert_eq!(
            event.as_object().unwrap().len(),
            6,
            "timeline contains only scheduling metadata"
        );
    }
}
