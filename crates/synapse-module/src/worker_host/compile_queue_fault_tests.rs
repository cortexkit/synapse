// Fault and lifecycle tests for the direct-ANE compile queue: panics, orphan
// drivers, cancelled admissions, learning, repricing, the 600 s bound, early
// refusal before dispatch or eviction, the gathering-window wakeup, recovery
// and failed admissions. They use the shared-compiler fake from
// compile_queue_tests.rs on a paused clock, or the mock-channel serving
// fixture where the real channel's stream and classification code matters.

fn queue_key(worker: &str, model_ref: &str, shape: usize) -> ShapeKey {
    ShapeKey {
        worker_id: worker.into(),
        model_ref: model_ref.into(),
        shape,
        program: AneProgram::Single,
    }
}

fn queued_keys(supervisor: &AneResidencySupervisor) -> Vec<ShapeKey> {
    supervisor.inner.lock().entries.keys().cloned().collect()
}

fn gate_held(supervisor: &AneResidencySupervisor) -> bool {
    supervisor.inner.lock().compile_running()
}

/// A panic in an orphan-driven admission must not strand the
/// module-wide compile gate. The gate and the slot's budget are released only
/// after the worker's restart (its confirmed exit), and then another worker
/// can compile.
#[tokio::test(start_paused = true)]
async fn panicking_orphan_admission_releases_the_gate_after_confirmed_restart() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let wide = fake_worker(&compiler, "wide");
    let small = fake_handle(&compiler, "small");
    small.panic_next_admit.store(true, Ordering::SeqCst);
    let (restarting, release_restart) = (Arc::new(Notify::new()), Arc::new(Notify::new()));
    *small.restart_gate.lock().unwrap() = Some((restarting.clone(), release_restart.clone()));
    let small_dyn: Arc<dyn AneShapeWorker> = small.clone();
    let wide_answer = ask_at(&supervisor, &compiler, &wide, QUEUE_QWEN, 4096, 0.0, None);
    let refused = ask_at(&supervisor, &compiler, &small_dyn, QUEUE_GTE, 128, 1.0, Some(10.0))
        .await
        .unwrap();
    assert_eq!(refused.code.as_deref(), Some(ERR_ANE_SHAPE_COMPILING));
    assert!(wide_answer.await.unwrap().ok());
    // The orphan driver now admits GTE@128 and the admission panics. Its
    // worker must be restarted; until that restart has confirmed the old
    // process gone, the gate stays held.
    // The panic happens when the wide compile ends, at about 84.6 s; the
    // recovery must start then, not after a drain timeout.
    tokio::time::timeout_at(
        compiler.epoch + Duration::from_secs(100),
        restarting.notified(),
    )
    .await
    .expect("the panicked admission's worker was not restarted promptly");
    assert!(gate_held(&supervisor), "gate released before the restart confirmed exit");
    let other = fake_worker(&compiler, "other");
    let other_answer = ask_at(&supervisor, &compiler, &other, QUEUE_GTE, 512, 100.0, None);
    tokio::time::sleep_until(compiler.epoch + Duration::from_secs(120)).await;
    assert!(compiler.compiles_of("other").is_empty(), "a compile overlapped an unconfirmed one");
    release_restart.notify_one();
    let other_answer = tokio::time::timeout(Duration::from_secs(60), other_answer)
        .await
        .expect("another worker could not compile after cleanup")
        .unwrap();
    assert!(other_answer.ok(), "{other_answer:?}");
    assert_eq!(small.restarts.load(Ordering::SeqCst), 1);
    let state = supervisor.inner.lock();
    assert!(!state.slots.keys().any(|key| key.worker_id == "small"));
    assert!(!state.compile_running());
}

/// Two refusals for the same shape before the runtime yields must not
/// start two orphan drivers that keep each other alive past the expiry.
#[tokio::test(start_paused = true)]
async fn two_refusals_before_a_yield_still_expire_the_orphan() {
    let compiler = SharedCompiler::new(3.0);
    let supervisor = queue_supervisor();
    let wide = fake_worker(&compiler, "wide");
    let small = fake_worker(&compiler, "small");
    let wide_answer = ask_at(&supervisor, &compiler, &wide, QUEUE_QWEN, 4096, 0.0, None);
    tokio::time::sleep_until(compiler.epoch + Duration::from_secs(1)).await;
    let deadline = || Some(tokio::time::Instant::now() + Duration::from_secs(10));
    let first = supervisor.lease_until(&small, QUEUE_GTE, 128, deadline()).await;
    let second = supervisor.lease_until(&small, QUEUE_GTE, 128, deadline()).await;
    let promised = [first, second].map(|answer| match answer {
        Err(error) => {
            assert_eq!(error.code(), Some(ERR_ANE_SHAPE_COMPILING));
            error
                .to_engine_error(synapse_core::EngineErrorStage::Inference)
                .retry_after_ms
                .unwrap()
        }
        Ok(_) => panic!("both requests must be refused at once"),
    });
    let expiry = 1.0 + 2.0 * promised[1] as f64 / 1000.0;
    tokio::time::sleep_until(compiler.epoch + Duration::from_secs_f64(expiry + 1.0)).await;
    assert!(
        queued_keys(&supervisor).is_empty(),
        "the orphan entry outlived twice its promise: {:?}",
        queued_keys(&supervisor)
    );
    assert!(wide_answer.await.unwrap().ok());
    tokio::time::sleep(Duration::from_secs(60)).await;
    assert!(
        compiler.compiles_of("small").is_empty(),
        "an expired orphan compiled: {:?}",
        compiler.compiles_of("small")
    );
}

/// Learning starts at compile dispatch, not at reservation, so a
/// stream held by inference does not count as compile time.
#[tokio::test]
async fn learning_ignores_the_stream_wait_before_dispatch() {
    let ledger = SharedLedger::default();
    ledger.lock().unwrap().admit_delay = Duration::from_millis(100);
    let serving = serving_fixture(&ledger).await;
    *serving.channel.test_compile_seed.lock().unwrap() = Some(Duration::from_millis(100));
    serving.infer(vec![vec![1]], false).await.unwrap();
    let (started, release) = notify_pair();
    ledger.lock().unwrap().inference_gate = Some((started.clone(), release.clone()));
    let inference = spawn_infer(&serving, 1, None);
    acknowledged(&started).await;
    let compile = spawn_infer(&serving, 600, None);
    tokio::time::sleep(Duration::from_millis(700)).await;
    release.notify_one();
    inference.await.unwrap().unwrap();
    compile.await.unwrap().unwrap();
    let key = (serving.metadata.model_ref.clone(), 1024, AneProgram::Single);
    let state = serving.supervisor.inner.lock();
    let measured = state.measured_costs.get(&key).copied();
    assert!(
        measured.is_some_and(|took| took < Duration::from_millis(400)),
        "learned {measured:?} for a 100 ms compile that waited 0.7 s for the stream"
    );
    assert!(
        state.machine_factor.is_some_and(|factor| factor < 3.0),
        "machine factor {:?}",
        state.machine_factor
    );
}

/// A compile retried after the worker ran out of Neural Engine
/// resources includes an eviction and a second compile, so it is not a
/// standalone sample.
#[tokio::test]
async fn learning_skips_a_compile_retried_after_resource_exhaustion() {
    let ledger = SharedLedger::default();
    let serving = serving_fixture(&ledger).await;
    *serving.channel.test_compile_seed.lock().unwrap() = Some(Duration::from_millis(100));
    serving.infer(vec![vec![1]], false).await.unwrap();
    let model_ref = serving.metadata.model_ref.clone();
    ledger
        .lock()
        .unwrap()
        .exhaust_admit
        .insert((model_ref.clone(), 1024), 1);
    serving.infer(vec![vec![1; 600]], false).await.unwrap();
    let state = serving.supervisor.inner.lock();
    assert!(
        !state
            .measured_costs
            .contains_key(&(model_ref, 1024, AneProgram::Single)),
        "a retried compile was learned as a standalone sample"
    );
}

/// A packed program is not learned against the single-shape seed.
#[tokio::test]
async fn learning_skips_packed_programs() {
    let peer = Arc::new(Mutex::new(PackedPeer::default()));
    let serving = packed_peer(false, peer.clone()).await;
    *serving.channel.test_compile_seed.lock().unwrap() = Some(Duration::from_millis(100));
    assert!(serving.prepare_packed().await.unwrap());
    let state = serving.supervisor.inner.lock();
    if !state
        .measured_costs
        .keys()
        .any(|(_, _, program)| *program == AneProgram::Single)
    {
        assert_eq!(state.machine_factor, None, "the factor was learned from a packed program");
    }
    assert!(
        !state
            .measured_costs
            .keys()
            .any(|(_, _, program)| *program != AneProgram::Single),
        "{:?}",
        state.measured_costs.keys().collect::<Vec<_>>()
    );
}

/// Entries queued before the first measured compile are repriced
/// once the machine factor is known, for queued and running compiles alike.
#[tokio::test(start_paused = true)]
async fn queued_entries_are_repriced_after_learning() {
    // Twice as slow as the seeds.
    let compiler = SharedCompiler::new(2.0);
    let supervisor = queue_supervisor();
    let probe = fake_worker(&compiler, "probe");
    let small = fake_worker(&compiler, "small");
    let wide = fake_worker(&compiler, "wide");
    let probe_answer = ask_at(&supervisor, &compiler, &probe, QUEUE_GTE, 512, 0.0, None);
    let small_answer = ask_at(&supervisor, &compiler, &small, QUEUE_QWEN, 128, 1.0, None);
    let wide_answer = ask_at(&supervisor, &compiler, &wide, QUEUE_QWEN, 2048, 1.0, None);
    // The probe ends at about 16.3 s and teaches the factor 2. Qwen3@128 then
    // runs until about 37.6 s, and Qwen3@2048 is still queued behind it.
    let joined = ask_at(&supervisor, &compiler, &wide, QUEUE_QWEN, 2048, 20.0, Some(60.0))
        .await
        .unwrap();
    assert!(probe_answer.await.unwrap().ok());
    assert!(small_answer.await.unwrap().ok());
    let small_end = compiler.compiles_of("small")[0].end.unwrap();
    let expected = small_end + 2.0 * standalone(QUEUE_QWEN, 2048) - joined.asked_at;
    assert_eq!(joined.code.as_deref(), Some(ERR_ANE_SHAPE_COMPILING), "{joined:?}");
    assert!(joined.waited() < 0.01, "accepted on a stale prediction: {joined:?}");
    assert_close(
        joined.retry_after_ms.unwrap() as f64 / 1000.0,
        expected,
        0.05,
        "retry_after_ms",
    );
    assert!(wide_answer.await.unwrap().ok());
}

/// A request without a deadline that was accepted on a prediction
/// that later overran reaches the 600 s bound while waiting only on the
/// compile queue. It is answered shape_compiling with a fresh prediction, and
/// its compile stays queued for the retry.
#[tokio::test(start_paused = true)]
async fn accepted_wait_past_the_600s_bound_is_shape_compiling_and_keeps_its_compile() {
    let compiler = SharedCompiler::new(8.0);
    let supervisor = queue_supervisor();
    let wide = fake_worker(&compiler, "wide");
    let small = fake_worker(&compiler, "small");
    let wide_answer = ask_at(&supervisor, &compiler, &wide, QUEUE_QWEN, 4096, 0.0, None);
    let answered = ask_at(&supervisor, &compiler, &small, QUEUE_GTE, 128, 1.0, None)
        .await
        .unwrap();
    assert_eq!(
        answered.code.as_deref(),
        Some(ERR_ANE_SHAPE_COMPILING),
        "{answered:?}"
    );
    assert!((answered.waited() - 600.0).abs() < 0.1, "{answered:?}");
    assert!(queued_keys(&supervisor).contains(&queue_key("small", QUEUE_GTE, 128)));
    assert!(supervisor.inner.lock().entries[&queue_key("small", QUEUE_GTE, 128)]
        .promise
        .is_some());
    let retry = answered.retry_after_ms.unwrap() as f64 / 1000.0;
    let retried = ask_at(
        &supervisor,
        &compiler,
        &small,
        QUEUE_GTE,
        128,
        answered.at + retry,
        None,
    )
    .await
    .unwrap();
    assert!(retried.ok(), "{retried:?}");
    assert_eq!(compiler.compiles_of("small").len(), 1);
    assert!(wide_answer.await.unwrap().ok());
}

/// When a queued head becomes dispatchable after the running compile
/// overran, its bound is checked before dispatch, not after the compile.
#[tokio::test(start_paused = true)]
async fn head_is_refused_before_dispatch_when_its_turn_comes_too_late() {
    let compiler = SharedCompiler::new(1.35);
    let supervisor = queue_supervisor();
    let first = fake_worker(&compiler, "first");
    let small = fake_worker(&compiler, "small");
    let first_answer = ask_at(&supervisor, &compiler, &first, QUEUE_QWEN, 2048, 0.0, None);
    // Predicted ready at about 46 s, inside the bound at 60 s; the running
    // compile actually ends at about 56.4 s, too late for GTE@128.
    let answered = ask_at(&supervisor, &compiler, &small, QUEUE_GTE, 128, 1.0, Some(59.0))
        .await
        .unwrap();
    let first_end = compiler.compiles_of("first")[0].end.unwrap();
    assert_eq!(answered.code.as_deref(), Some(ERR_ANE_SHAPE_COMPILING), "{answered:?}");
    assert!(
        answered.at < first_end + 0.1,
        "answered at {:.2} s, after dispatching a compile it could not use (turn came at {first_end:.2} s)",
        answered.at
    );
    assert!(first_answer.await.unwrap().ok());
    // The refused request's compile still runs, for the retry it was told
    // to make.
    tokio::time::sleep(Duration::from_secs(30)).await;
    assert_eq!(compiler.compiles_of("small").len(), 1);
    assert!(compiler.compiles_of("small")[0].start >= first_end);
    assert!(queued_keys(&supervisor).is_empty());
}

/// A head that is about to be refused never evicts first, even when
/// its only victim sits on the stream of the worker that is compiling.
#[tokio::test(start_paused = true)]
async fn head_is_refused_before_evicting_on_a_busy_stream() {
    let compiler = SharedCompiler::new(1.0);
    let mut supervisor = queue_supervisor();
    Arc::get_mut(&mut supervisor.inner).unwrap().executable_budget = 90;
    let busy = fake_handle(&compiler, "busy");
    let busy_dyn: Arc<dyn AneShapeWorker> = busy.clone();
    let leased = fake_worker(&compiler, "leased");
    let late = fake_worker(&compiler, "late");
    drop(supervisor.lease(&busy_dyn, QUEUE_QWEN, 128).await.unwrap());
    let held = supervisor.lease(&leased, QUEUE_GTE, 128).await.unwrap();
    // Reserved: busy Qwen3@128 (28) + leased GTE@128 (22) + the running
    // Qwen3@4096 (28) = 78; GTE@512 needs 22 more than the 90 allowed.
    let wide_answer = ask_at(
        &supervisor,
        &compiler,
        &busy_dyn,
        QUEUE_QWEN,
        4096,
        compiler.now(),
        None,
    );
    tokio::time::sleep(Duration::from_secs(1)).await;
    let answered = ask_at(
        &supervisor,
        &compiler,
        &late,
        QUEUE_GTE,
        512,
        compiler.now(),
        Some(30.0),
    )
    .await
    .unwrap();
    assert!(answered.waited() < 0.01, "refused only after {:.2} s: {answered:?}", answered.waited());
    assert_eq!(answered.code.as_deref(), Some(ERR_ANE_SHAPE_COMPILING), "{answered:?}");
    assert!(busy.evicted().is_empty(), "evicted {:?} for a refused head", busy.evicted());
    assert_eq!(
        supervisor.inner.lock().slots[&queue_key("busy", QUEUE_QWEN, 128)].state,
        SlotState::Resident
    );
    drop(held);
    assert!(wide_answer.await.unwrap().ok());
}

/// The gathering window's end must wake the head even when time
/// passes between its decision (not gathered yet) and arming its timer. The
/// paused clock is advanced at exactly that point by a test hook.
#[tokio::test(start_paused = true)]
async fn gathering_window_end_wakes_the_head_after_a_late_timer() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let (decided, resume) = (Arc::new(Notify::new()), Arc::new(Notify::new()));
    *supervisor.inner.wait_decision_gate.lock().unwrap() = Some((decided.clone(), resume.clone()));
    let small = fake_worker(&compiler, "small");
    let answer = ask_at(&supervisor, &compiler, &small, QUEUE_GTE, 128, 0.0, None);
    decided.notified().await;
    tokio::time::advance(Duration::from_millis(60)).await;
    resume.notify_one();
    let answer = tokio::time::timeout(Duration::from_secs(30), answer)
        .await
        .expect("the head slept past the end of its gathering window")
        .unwrap();
    assert!(answer.ok(), "{answer:?}");
    assert!(compiler.compiles_of("small")[0].start < 0.1);
}

/// A head whose worker is recovering must not stop other workers'
/// compiles.
#[tokio::test(start_paused = true)]
async fn recovering_head_does_not_block_other_workers() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let first = fake_worker(&compiler, "first");
    let recovering = fake_handle(&compiler, "recovering");
    let (restarting, release_restart) = (Arc::new(Notify::new()), Arc::new(Notify::new()));
    *recovering.restart_gate.lock().unwrap() = Some((restarting.clone(), release_restart.clone()));
    let recovering_dyn: Arc<dyn AneShapeWorker> = recovering.clone();
    let other = fake_worker(&compiler, "other");
    let first_answer = ask_at(&supervisor, &compiler, &first, QUEUE_QWEN, 2048, 0.0, None);
    // GTE@128's start key (1 + 4.62 s) is earlier than GTE@512's (2 + 8.14 s).
    let head_answer = ask_at(&supervisor, &compiler, &recovering_dyn, QUEUE_GTE, 128, 1.0, None);
    let other_answer = ask_at(&supervisor, &compiler, &other, QUEUE_GTE, 512, 2.0, None);
    tokio::time::sleep_until(compiler.epoch + Duration::from_secs(3)).await;
    let recovery = tokio::spawn({
        let supervisor = supervisor.clone();
        let worker = recovering_dyn.clone();
        async move { supervisor.recover(&worker).await }
    });
    restarting.notified().await;
    let other_answer = tokio::time::timeout(Duration::from_secs(120), other_answer)
        .await
        .expect("a recovering head blocked another worker's compile")
        .unwrap();
    assert!(other_answer.ok(), "{other_answer:?}");
    release_restart.notify_one();
    assert!(recovery.await.unwrap());
    assert!(head_answer.await.unwrap().ok());
    assert!(first_answer.await.unwrap().ok());
}

/// A failed restart must not leave the worker marked recovering for
/// ever; a later recovery can lead and succeed.
#[tokio::test(start_paused = true)]
async fn failed_restart_does_not_leave_the_worker_recovering() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let worker = fake_handle(&compiler, "worker");
    worker.fail_next_restart.store(true, Ordering::SeqCst);
    let worker_dyn: Arc<dyn AneShapeWorker> = worker.clone();
    assert!(!supervisor.recover(&worker_dyn).await);
    assert!(
        !supervisor.inner.lock().recovering.contains("worker"),
        "a failed restart left the worker recovering for ever"
    );
    assert!(
        tokio::time::timeout(Duration::from_secs(5), supervisor.recover(&worker_dyn))
            .await
            .expect("a second recovery could not lead")
    );
    assert_eq!(worker.restarts.load(Ordering::SeqCst), 1);
}

/// A driver first polled after its entry's expiry must
/// still remove the entry.
#[tokio::test(start_paused = true)]
async fn orphan_driver_first_polled_after_expiry_removes_its_entry() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let wide = fake_worker(&compiler, "wide");
    let small = fake_worker(&compiler, "small");
    let wide_answer = ask_at(&supervisor, &compiler, &wide, QUEUE_QWEN, 4096, 0.0, None);
    tokio::time::sleep_until(compiler.epoch + Duration::from_secs(1)).await;
    let deadline = Some(tokio::time::Instant::now() + Duration::from_secs(10));
    let refused = supervisor.lease_until(&small, QUEUE_GTE, 128, deadline).await;
    assert!(refused.is_err());
    // The driver has been spawned but not polled; jump past the expiry
    // (1 + 2 * 88 s) before it first runs.
    tokio::time::advance(Duration::from_secs(400)).await;
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(
        queued_keys(&supervisor).is_empty(),
        "a driver polled after expiry leaked {:?}",
        queued_keys(&supervisor)
    );
    assert!(wide_answer.await.unwrap().ok());
}

/// A retry that has enqueued but not yet decided
/// counts as attached, so the orphan entry and its start key survive.
#[tokio::test(start_paused = true)]
async fn retry_that_has_not_decided_yet_keeps_the_orphan_entry() {
    let compiler = SharedCompiler::new(3.0);
    let supervisor = queue_supervisor();
    let wide = fake_worker(&compiler, "wide");
    let small = fake_worker(&compiler, "small");
    let wide_answer = ask_at(&supervisor, &compiler, &wide, QUEUE_QWEN, 4096, 0.0, None);
    tokio::time::sleep_until(compiler.epoch + Duration::from_secs(1)).await;
    let deadline = Some(tokio::time::Instant::now() + Duration::from_secs(10));
    let refused = supervisor
        .lease_until(&small, QUEUE_GTE, 128, deadline)
        .await
        .err()
        .expect("refused at once");
    let promised = refused
        .to_engine_error(synapse_core::EngineErrorStage::Inference)
        .retry_after_ms
        .unwrap() as f64
        / 1000.0;
    tokio::time::sleep(Duration::from_millis(10)).await;
    let key = queue_key("small", QUEUE_GTE, 128);
    let seq = supervisor.inner.lock().entries[&key].seq;
    // The retry enqueues just before the expiry and is held before its first
    // decision while the expiry passes.
    let expiry = 1.0 + 2.0 * promised;
    tokio::time::sleep_until(compiler.epoch + Duration::from_secs_f64(expiry - 1.0)).await;
    let (checked, continue_grant) = (Arc::new(Notify::new()), Arc::new(Notify::new()));
    *supervisor.inner.lease_decision_gate.lock().unwrap() =
        Some((checked.clone(), continue_grant.clone()));
    let retry = tokio::spawn({
        let supervisor = supervisor.clone();
        let small = small.clone();
        async move { supervisor.lease_until(&small, QUEUE_GTE, 128, None).await }
    });
    checked.notified().await;
    tokio::time::sleep_until(compiler.epoch + Duration::from_secs_f64(expiry + 1.0)).await;
    continue_grant.notify_one();
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert_eq!(
        supervisor.inner.lock().entries.get(&key).map(|entry| entry.seq),
        Some(seq),
        "the orphan entry expired under a retry that had already enqueued"
    );
    assert!(wide_answer.await.unwrap().ok());
    drop(retry.await.unwrap().unwrap());
}

/// An orphan driver must not evict a resident shape
/// and then expire without compiling.
#[tokio::test(start_paused = true)]
async fn orphan_driver_does_not_evict_then_expire() {
    let compiler = SharedCompiler::new(3.0);
    let mut supervisor = queue_supervisor();
    // Victim GTE@128 (22) + running Qwen3@4096 (28) = 50; GTE@128 on another
    // worker needs 22 more than the 60 allowed.
    Arc::get_mut(&mut supervisor.inner).unwrap().executable_budget = 60;
    let victim = fake_handle(&compiler, "victim");
    let victim_dyn: Arc<dyn AneShapeWorker> = victim.clone();
    let wide = fake_worker(&compiler, "wide");
    let small = fake_worker(&compiler, "small");
    drop(supervisor.lease(&victim_dyn, QUEUE_GTE, 128).await.unwrap());
    // Forget what that compile taught, so the next predictions are the
    // optimistic seeds and the orphan's promise expires before its turn.
    {
        let mut state = supervisor.inner.lock();
        state.measured_costs.clear();
        state.machine_factor = None;
    }
    let start = compiler.now();
    let wide_answer = ask_at(&supervisor, &compiler, &wide, QUEUE_QWEN, 4096, start, None);
    let refused = ask_at(&supervisor, &compiler, &small, QUEUE_GTE, 128, start + 1.0, Some(10.0))
        .await
        .unwrap();
    assert_eq!(refused.code.as_deref(), Some(ERR_ANE_SHAPE_COMPILING), "{refused:?}");
    assert!(wide_answer.await.unwrap().ok());
    tokio::time::sleep(Duration::from_secs(60)).await;
    let compiled = !compiler.compiles_of("small").is_empty();
    assert!(
        compiled || victim.evicted().is_empty(),
        "the orphan evicted {:?} and expired without compiling",
        victim.evicted()
    );
    assert!(queued_keys(&supervisor).is_empty());
}

/// When the serving caller's own timer answers first, its
/// retry_after_ms is the queue's current prediction, the same one the
/// orphan entry is kept for.
#[tokio::test]
async fn caller_expiry_reports_a_fresh_prediction() {
    let ledger = SharedLedger::default();
    let serving = serving_fixture(&ledger).await;
    // Compile 128 first with an unknown cost, so nothing is learned from the
    // instant mock compile.
    serving.infer(vec![vec![1]], false).await.unwrap();
    *serving.channel.test_compile_seed.lock().unwrap() = Some(Duration::from_millis(1500));
    let gate = gate_classification(&serving);
    let (started, release) = notify_pair();
    ledger.lock().unwrap().admit_gate = Some((started.clone(), release.clone()));
    let compile = spawn_infer(&serving, 600, None);
    acknowledged(&started).await;
    // Queued behind the held 1024 compile: predicted ready in about 3 s,
    // inside its 3.3 s deadline, so it waits.
    let deadline = Some(tokio::time::Instant::now() + Duration::from_millis(3300));
    let waiting = spawn_infer(&serving, 200, deadline);
    tokio::time::timeout(Duration::from_secs(6), async {
        gate.reached(GateSite::Caller, 1).await;
        gate.reached(GateSite::Inner, 1).await;
    })
    .await
    .expect("the request never expired");
    gate.release(GateSite::Caller);
    let answered = waiting.await.unwrap();
    let retry = match &answered {
        Err(error) => error
            .to_engine_error(synapse_core::EngineErrorStage::Inference)
            .retry_after_ms
            .unwrap(),
        Ok(_) => panic!("expected shape_compiling"),
    };
    gate.release(GateSite::Inner);
    // The held compile has overrun: one layer's share of it, then the 256
    // compile itself.
    assert!(
        retry >= 1500,
        "the caller was told {retry} ms, a stale prediction"
    );
    let key = queue_key("serving", &serving.metadata.model_ref, 256);
    until(|| {
        serving
            .supervisor
            .inner
            .lock()
            .entries
            .get(&key)
            .is_some_and(|entry| entry.promise.is_some())
    })
    .await;
    let promise = serving.supervisor.inner.lock().entries[&key].promise.unwrap().1;
    assert!(
        promise.as_millis().abs_diff(u128::from(retry)) <= 20,
        "caller told {retry} ms, entry kept for {promise:?}"
    );
    release.notify_one();
    compile.await.unwrap().unwrap();
}

/// A failed admission holds the compile gate until its worker's exit
/// is confirmed, so another worker's compile cannot overlap one that may
/// still be running.
#[tokio::test]
async fn failed_admission_holds_the_gate_until_exit_is_confirmed() {
    let supervisor = AneResidencySupervisor::new(Default::default());
    let failing_ledger = SharedLedger::default();
    let (exiting, exit) = notify_pair();
    failing_ledger.lock().unwrap().exit_gate = Some((exiting.clone(), exit.clone()));
    let failing = serving_fixture_for(&failing_ledger, "failing", supervisor.clone()).await;
    failing.infer(vec![vec![1]], false).await.unwrap();
    failing_ledger
        .lock()
        .unwrap()
        .fail_admit
        .insert((failing.metadata.model_ref.clone(), 1024));
    let other_ledger = SharedLedger::default();
    let other = serving_fixture_for(&other_ledger, "other", supervisor.clone()).await;
    other.infer(vec![vec![1]], false).await.unwrap();
    let failed = spawn_infer(&failing, 600, None);
    acknowledged(&exiting).await;
    let compiled = spawn_infer(&other, 600, None);
    tokio::time::sleep(Duration::from_millis(300)).await;
    let overlapped = other_ledger
        .lock()
        .unwrap()
        .admit_attempts
        .contains_key(&(other.metadata.model_ref.clone(), 1024));
    exit.notify_one();
    let _ = failed.await.unwrap();
    compiled.await.unwrap().unwrap();
    assert!(
        !overlapped,
        "another worker compiled before the failed one's exit was confirmed"
    );
}
