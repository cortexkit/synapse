// These tests control the worker reply and the supervisor's decision order
// separately, so scheduler wakeup order cannot hide a lost single-flight handoff.
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
    let original = QueueTicket::enqueue(&supervisor.inner);
    let Step::Admit(id) = supervisor.next_step(original.id, &key(128), &worker, 4) else {
        panic!("initial shape must reserve budget");
    };
    let head = QueueTicket::enqueue(&supervisor.inner);
    assert!(matches!(
        supervisor.next_step(head.id, &key(256), &worker, 4),
        Step::Wait(_)
    ));
    let joiners: Vec<_> = (0..8)
        .map(|_| QueueTicket::enqueue(&supervisor.inner))
        .collect();
    for joiner in &joiners {
        assert!(matches!(
            supervisor.next_step(joiner.id, &key(128), &worker, 4),
            Step::Wait(Some(128))
        ));
    }
    let (started, release) = notify_pair();
    ledger.lock().unwrap().admit_gate = Some((started.clone(), release.clone()));
    let admitting = tokio::spawn({
        let supervisor = supervisor.clone();
        let worker = worker.clone();
        let key = key(128);
        async move { supervisor.admit(worker, key, id).await }
    });
    notified(&started).await;
    release.notify_one();
    drop(admitting.await.unwrap().unwrap());
    assert!(
        resource_victim(&supervisor.inner).is_none(),
        "resource-refusal recovery must also protect the pending handoff"
    );
    assert!(
        matches!(
            supervisor.next_step(head.id, &key(256), &worker, 4),
            Step::Wait(_)
        ),
        "newly ready shape was selected before its eight joiners leased it"
    );
    let newcomer = QueueTicket::enqueue(&supervisor.inner);
    assert!(
        matches!(
            supervisor.next_step(newcomer.id, &key(128), &worker, 4),
            Step::Wait(_)
        ),
        "post-completion traffic must not extend the handoff"
    );
    for joiner in &joiners {
        let Step::Leased(lease) = supervisor.next_step(joiner.id, &key(128), &worker, 4) else {
            panic!("a pre-completion joiner lost its handoff to the budget head");
        };
        drop(lease);
    }
    assert!(
        matches!(
            supervisor.next_step(head.id, &key(256), &worker, 4),
            Step::Evict { .. }
        ),
        "the finite handoff must stop protecting the shape after its cohort drains"
    );
}

#[tokio::test]
async fn eviction_prefers_an_idle_stream_and_admits_before_the_long_compile_ends() {
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
        before_compile_end.is_ok() && idle_chosen && still_compiling,
        "budget head queued eviction behind the long compile instead of choosing the idle stream"
    );
    drop(before_compile_end.unwrap().unwrap().unwrap());
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
    let key = ShapeKey {
        worker_id: "worker".into(),
        model_ref: "model".into(),
        shape: 128,
        program: AneProgram::Single,
    };
    let owner = QueueTicket::enqueue(&supervisor.inner);
    let Step::Admit(id) = supervisor.next_step(owner.id, &key, &worker, 4) else {
        panic!("reservation");
    };
    let head = tokio::spawn({
        let supervisor = supervisor.clone();
        let worker = worker.clone();
        async move { supervisor.lease(&worker, "model", 256).await }
    });
    until(|| supervisor.inner.lock().starved_head.is_some()).await;
    let suspended = QueueTicket::enqueue(&supervisor.inner);
    assert!(matches!(
        supervisor.next_step(suspended.id, &key, &worker, 4),
        Step::Wait(Some(128))
    ));
    drop(supervisor.admit(worker.clone(), key, id).await.unwrap());
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
    let key = |shape| ShapeKey {
        worker_id: "worker".into(),
        model_ref: "model".into(),
        shape,
        program: AneProgram::Single,
    };
    let head = QueueTicket::enqueue(&supervisor.inner);
    assert!(matches!(
        supervisor.next_step(head.id, &key(512), &worker, 4),
        Step::Wait(_)
    ));
    let waiter = QueueTicket::enqueue(&supervisor.inner);
    assert!(matches!(
        supervisor.next_step(waiter.id, &key(256), &worker, 4),
        Step::Wait(_)
    ));
    drop((demanded, other));
    let Step::Evict { key: victim, .. } = supervisor.next_step(head.id, &key(512), &worker, 4)
    else {
        panic!("victim");
    };
    assert_eq!(
        victim.shape, 128,
        "queued demand should outweigh the smallest-rung preference"
    );
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
