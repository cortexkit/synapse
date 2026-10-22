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

/// Waits on the paused clock until `condition` holds, for up to `seconds`.
async fn until_within(seconds: u64, mut condition: impl FnMut() -> bool) {
    tokio::time::timeout(Duration::from_secs(seconds), async {
        while !condition() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("condition never held");
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
    // Still inside the exit-confirmation grace (the restart began at about
    // 84.6 s).
    tokio::time::sleep_until(compiler.epoch + Duration::from_secs(110)).await;
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
    // The wide compile has overrun its prediction (84.56 s): it is given one
    // more layer (3.02 s), then GTE@128 takes 4.62 s.
    let expected = 28.0f64.recip() * standalone(QUEUE_QWEN, 4096) + standalone(QUEUE_GTE, 128);
    assert_close(
        answered.retry_after_ms.unwrap() as f64 / 1000.0,
        expected,
        0.02,
        "retry_after_ms at the backstop",
    );
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
        async move { supervisor.recover(&worker).await.is_ok() }
    });
    restarting.notified().await;
    let other_answer = tokio::time::timeout(Duration::from_secs(120), other_answer)
        .await
        .expect("a recovering head blocked another worker's compile")
        .unwrap();
    assert!(other_answer.ok(), "{other_answer:?}");
    release_restart.notify_one();
    // The restart was held past the scheduler's give-up on the old process's
    // exit, so its caller was answered with an error then; the restart still
    // completes and the worker rejoins.
    assert!(!recovery.await.unwrap());
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
    assert!(supervisor.recover(&worker_dyn).await.is_err());
    assert!(
        !supervisor.inner.lock().recovering.contains("worker"),
        "a failed restart left the worker recovering for ever"
    );
    assert!(
        tokio::time::timeout(Duration::from_secs(5), supervisor.recover(&worker_dyn))
            .await
            .expect("a second recovery could not lead")
            .is_ok()
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

/// A retry the scheduler has registered before the expiry counts as
/// attached, so the kept entry and its start key survive the expiry even
/// while the retry's own task has not run again.
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
    // The retry registers just before the expiry; its task then does not
    // run again while the expiry passes.
    let expiry = 1.0 + 2.0 * promised;
    tokio::time::sleep_until(compiler.epoch + Duration::from_secs_f64(expiry - 1.0)).await;
    let (_id, retry) = raw_want(&supervisor, &small, QUEUE_GTE, 128);
    tokio::time::sleep_until(compiler.epoch + Duration::from_secs_f64(expiry + 1.0)).await;
    assert_eq!(
        supervisor.inner.lock().entries.get(&key).map(|entry| entry.seq),
        Some(seq),
        "the orphan entry expired under a retry that had already enqueued"
    );
    assert!(wide_answer.await.unwrap().ok());
    assert!(matches!(
        retry.await.unwrap(),
        scheduler::Answer::Ready | scheduler::Answer::Joined(_)
    ));
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

/// A compile that failed with an unknown outcome (the connection broke)
/// holds the compile gate until its worker's exit is confirmed, so another
/// worker's compile cannot overlap one that may still be running.
#[tokio::test(start_paused = true)]
async fn failed_admission_holds_the_gate_until_exit_is_confirmed() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let failing = fake_handle(&compiler, "failing");
    failing.transport_fail_next_admit.store(true, Ordering::SeqCst);
    let (exiting, exit) = (Arc::new(Notify::new()), Arc::new(Notify::new()));
    *failing.restart_gate.lock().unwrap() = Some((exiting.clone(), exit.clone()));
    let failing_dyn: Arc<dyn AneShapeWorker> = failing.clone();
    let failed = ask_at(&supervisor, &compiler, &failing_dyn, QUEUE_GTE, 512, 0.0, None);
    exiting.notified().await;
    let other = fake_worker(&compiler, "other");
    let compiled = ask_at(&supervisor, &compiler, &other, QUEUE_GTE, 128, compiler.now(), None);
    tokio::time::sleep(Duration::from_secs(30)).await;
    let overlapped = !compiler.compiles_of("other").is_empty();
    assert!(gate_held(&supervisor));
    exit.notify_one();
    assert!(!failed.await.unwrap().ok());
    assert!(compiled.await.unwrap().ok());
    assert!(
        !overlapped,
        "another worker compiled before the failed one's exit was confirmed"
    );
    assert_queue_idle(&supervisor);
}

/// A worker that answers an admission with an error is not compiling
/// anything, so the compile gate is free at once, even while that worker's
/// recovery waits for one of its leases to drain.
#[tokio::test(start_paused = true)]
async fn definite_worker_error_releases_the_gate_at_once() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let failing = fake_handle(&compiler, "failing");
    let failing_dyn: Arc<dyn AneShapeWorker> = failing.clone();
    let held = supervisor.lease(&failing_dyn, QUEUE_GTE, 128).await.unwrap();
    failing.fail_next_admit.store(true, Ordering::SeqCst);
    let start = compiler.now();
    let failed = ask_at(&supervisor, &compiler, &failing_dyn, QUEUE_GTE, 512, start, None)
        .await
        .unwrap();
    assert_eq!(failed.code.as_deref(), Some("compile_failed"), "{failed:?}");
    let other = fake_worker(&compiler, "other");
    let compiled = ask_at(&supervisor, &compiler, &other, QUEUE_GTE, 2048, compiler.now(), None);
    let compiled = tokio::time::timeout(Duration::from_secs(30), compiled)
        .await
        .expect("a definite worker error held the compile gate")
        .unwrap();
    assert!(compiled.ok(), "{compiled:?}");
    assert!(supervisor.inner.lock().recovering.contains("failing"));
    drop(held);
}

/// Once the old process's exit is confirmed, a compile that failed in it
/// cannot still be running, so its slot releases the compile gate and its
/// budget even when the restart then fails. One broken worker must not stall
/// every other worker's cold compiles.
#[tokio::test(start_paused = true)]
async fn failed_compile_releases_the_gate_once_exit_is_confirmed_even_if_restart_fails() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let broken = fake_handle(&compiler, "broken");
    broken.fail_next_admit.store(true, Ordering::SeqCst);
    broken.fail_next_restart.store(true, Ordering::SeqCst);
    let broken_dyn: Arc<dyn AneShapeWorker> = broken.clone();
    let failed = ask_at(&supervisor, &compiler, &broken_dyn, QUEUE_GTE, 128, 0.0, None)
        .await
        .unwrap();
    assert!(!failed.ok(), "{failed:?}");
    let other = fake_worker(&compiler, "other");
    let answer = ask_at(&supervisor, &compiler, &other, QUEUE_GTE, 512, 1.0, None);
    let answer = tokio::time::timeout(Duration::from_secs(30), answer)
        .await
        .expect("a failed compile with a confirmed exit still held the compile gate")
        .unwrap();
    assert!(answer.ok(), "{answer:?}");
    let state = supervisor.inner.lock();
    assert!(!state.slots.keys().any(|key| key.worker_id == "broken"));
    assert!(!state.compile_running());
    assert_eq!(state.reserved_executables(), 22);
}

/// While the old process's exit is unconfirmed, a compile that failed in it
/// may still be running, so its slot holds the compile gate; but only for the
/// fixed grace after the restart began. Then the gate is released (another
/// worker compiles), and the worker stays closed with its slot charged.
#[tokio::test(start_paused = true)]
async fn exit_never_confirmed_releases_the_gate_after_the_grace_and_keeps_the_worker_closed() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let broken = fake_handle(&compiler, "broken");
    broken.transport_fail_next_admit.store(true, Ordering::SeqCst);
    broken.exit_never_confirmed.store(true, Ordering::SeqCst);
    let broken_dyn: Arc<dyn AneShapeWorker> = broken.clone();
    let failed = ask_at(&supervisor, &compiler, &broken_dyn, QUEUE_GTE, 128, 0.0, None)
        .await
        .unwrap();
    assert!(!failed.ok(), "{failed:?}");
    let failed_at = failed.at;
    let other = fake_worker(&compiler, "other");
    let answer = ask_at(&supervisor, &compiler, &other, QUEUE_GTE, 512, failed_at + 1.0, None);
    tokio::time::sleep_until(compiler.epoch + Duration::from_secs_f64(failed_at + 30.0)).await;
    assert!(
        compiler.compiles_of("other").is_empty(),
        "a compile started beside one whose process may still be running"
    );
    assert!(gate_held(&supervisor));
    let answer = tokio::time::timeout(Duration::from_secs(60), answer)
        .await
        .expect("an exit that is never confirmed held the compile gate past the grace")
        .unwrap();
    assert!(answer.ok(), "{answer:?}");
    let state = supervisor.inner.lock();
    assert!(state.recovering.contains("broken"), "the worker must stay closed");
    assert!(state
        .slots
        .iter()
        .any(|(key, slot)| key.worker_id == "broken" && slot.state == SlotState::Abandoned));
    assert_eq!(state.reserved_executables(), 44, "its reservation stays charged");
}

/// After a worker's old process exits, every slot it had is gone, whatever
/// its state, even when the replacement then fails to start: nothing stays
/// charged or leasable for a process that no longer exists.
#[tokio::test(start_paused = true)]
async fn confirmed_exit_with_failed_replacement_leaves_no_ghost_slots() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let worker = fake_handle(&compiler, "worker");
    let worker_dyn: Arc<dyn AneShapeWorker> = worker.clone();
    drop(supervisor.lease(&worker_dyn, QUEUE_GTE, 128).await.unwrap());
    drop(supervisor.lease(&worker_dyn, QUEUE_GTE, 512).await.unwrap());
    assert_eq!(supervisor.inner.lock().reserved_executables(), 44);
    // One slot is being evicted when the process exits: it goes too.
    supervisor
        .inner
        .lock()
        .slots
        .get_mut(&queue_key("worker", QUEUE_GTE, 512))
        .unwrap()
        .state = SlotState::Evicting;
    worker.fail_next_restart.store(true, Ordering::SeqCst);
    assert!(supervisor.recover(&worker_dyn).await.is_err());
    let state = supervisor.inner.lock();
    assert!(
        !state.slots.keys().any(|key| key.worker_id == "worker"),
        "ghost slots after a confirmed exit: {:?}",
        state.slots.keys().collect::<Vec<_>>()
    );
    assert_eq!(state.reserved_executables(), 0);
    drop(state);
    assert!(supervisor
        .try_lease(&queue_key("worker", QUEUE_GTE, 128), &worker_dyn, None)
        .unwrap()
        .is_none());
}

/// A request told shape_compiling while the shape's compile was reserved but
/// still waiting for its worker keeps that compile alive when the request that
/// reserved it is cancelled: the compile runs once the worker is free.
#[tokio::test(start_paused = true)]
async fn promise_given_during_a_reserved_compile_survives_the_owner_cancelling() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let worker = fake_handle(&compiler, "worker");
    let worker_dyn: Arc<dyn AneShapeWorker> = worker.clone();
    // An inference holds the worker's stream.
    let inference = worker.stream.lock().await;
    let owner = tokio::spawn({
        let supervisor = supervisor.clone();
        let worker = worker_dyn.clone();
        async move { supervisor.lease(&worker, QUEUE_QWEN, 512).await }
    });
    until(|| {
        supervisor
            .inner
            .lock()
            .slots
            .get(&queue_key("worker", QUEUE_QWEN, 512))
            .is_some_and(|slot| slot.state == SlotState::Admitting)
    })
    .await;
    let deadline = Some(tokio::time::Instant::now() + Duration::from_secs(5));
    let told = supervisor
        .lease_until(&worker_dyn, QUEUE_QWEN, 512, deadline)
        .await
        .err()
        .expect("the joiner cannot be served in 5 s");
    assert_eq!(told.code(), Some(ERR_ANE_SHAPE_COMPILING));
    owner.abort();
    let _ = owner.await;
    tokio::time::sleep(Duration::from_secs(1)).await;
    drop(inference);
    until_within(120, || {
        supervisor
            .inner
            .lock()
            .slots
            .get(&queue_key("worker", QUEUE_QWEN, 512))
            .is_some_and(|slot| slot.state == SlotState::Resident)
    })
    .await;
    assert_eq!(compiler.compiles_of("worker").len(), 1);
    assert_queue_idle(&supervisor);
}

/// The same reserved compile, kept only for its promise, is withdrawn once
/// that promise lapses while it still waits for the worker: nobody came back
/// for it, so it never reaches the wire.
#[tokio::test(start_paused = true)]
async fn reserved_compile_is_withdrawn_when_its_promise_lapses_before_dispatch() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let worker = fake_handle(&compiler, "worker");
    let worker_dyn: Arc<dyn AneShapeWorker> = worker.clone();
    let key = queue_key("worker", QUEUE_QWEN, 512);
    let inference = worker.stream.lock().await;
    let owner = tokio::spawn({
        let supervisor = supervisor.clone();
        let worker = worker_dyn.clone();
        async move { supervisor.lease(&worker, QUEUE_QWEN, 512).await }
    });
    until(|| {
        supervisor
            .inner
            .lock()
            .slots
            .get(&key)
            .is_some_and(|slot| slot.state == SlotState::Admitting)
    })
    .await;
    let deadline = Some(tokio::time::Instant::now() + Duration::from_secs(5));
    let told = supervisor
        .lease_until(&worker_dyn, QUEUE_QWEN, 512, deadline)
        .await
        .err()
        .expect("the joiner cannot be served in 5 s");
    let ShapeCompilingRetry(retry_after_ms) = ShapeCompilingRetry::of(&told);
    owner.abort();
    let _ = owner.await;
    // Past twice the promised delay, with the worker still busy.
    tokio::time::sleep(Duration::from_millis(retry_after_ms * 2 + 1_000)).await;
    until_within(5, || !supervisor.inner.lock().slots.contains_key(&key)).await;
    drop(inference);
    tokio::time::sleep(Duration::from_secs(60)).await;
    assert!(
        compiler.compiles_of("worker").is_empty(),
        "a compile kept only for a lapsed promise reached the worker"
    );
    assert_queue_idle(&supervisor);
}

struct ShapeCompilingRetry(u64);

impl ShapeCompilingRetry {
    fn of(error: &AneResidencyError) -> Self {
        match error {
            AneResidencyError::ShapeCompiling { retry_after_ms, .. } => Self(*retry_after_ms),
            other => panic!("expected shape_compiling, got {other:?}"),
        }
    }
}

/// A caller cancelled while its admission's resource-refusal retry evicts a
/// victim does not lose the retry: the compile still completes.
#[tokio::test(start_paused = true)]
async fn exhaustion_retry_survives_the_caller_cancelling_during_its_eviction() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let victim = fake_handle(&compiler, "victim");
    let victim_dyn: Arc<dyn AneShapeWorker> = victim.clone();
    drop(supervisor.lease(&victim_dyn, QUEUE_GTE, 128).await.unwrap());
    let worker = fake_handle(&compiler, "worker");
    worker.exhaust_next_admit.store(true, Ordering::SeqCst);
    let worker_dyn: Arc<dyn AneShapeWorker> = worker.clone();
    // The victim's stream is busy, so the retry's eviction waits for it.
    let busy = victim.stream.lock().await;
    let caller = tokio::spawn({
        let supervisor = supervisor.clone();
        let worker = worker_dyn.clone();
        async move { supervisor.lease(&worker, QUEUE_GTE, 512).await }
    });
    until(|| {
        supervisor
            .inner
            .lock()
            .slots
            .values()
            .any(|slot| slot.state == SlotState::Evicting)
    })
    .await;
    caller.abort();
    let _ = caller.await;
    drop(busy);
    until_within(120, || {
        supervisor
            .inner
            .lock()
            .slots
            .get(&queue_key("worker", QUEUE_GTE, 512))
            .is_some_and(|slot| slot.state == SlotState::Resident)
    })
    .await;
    assert_eq!(victim.evicted(), vec![(QUEUE_GTE.to_owned(), 128)]);
}

/// A compile that failed with an unknown outcome holds the gate only until
/// the old process's exit is confirmed, not while the replacement starts.
#[tokio::test(start_paused = true)]
async fn failed_compile_releases_the_gate_before_the_replacement_starts() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let failing = fake_handle(&compiler, "failing");
    failing.transport_fail_next_admit.store(true, Ordering::SeqCst);
    let (starting, start) = (Arc::new(Notify::new()), Arc::new(Notify::new()));
    *failing.replacement_gate.lock().unwrap() = Some((starting.clone(), start.clone()));
    let failing_dyn: Arc<dyn AneShapeWorker> = failing.clone();
    let failed = ask_at(&supervisor, &compiler, &failing_dyn, QUEUE_GTE, 512, 0.0, None);
    starting.notified().await;
    let other = fake_worker(&compiler, "other");
    let compiled = ask_at(&supervisor, &compiler, &other, QUEUE_GTE, 128, compiler.now(), None);
    let compiled = tokio::time::timeout(Duration::from_secs(30), compiled)
        .await
        .expect("the gate was held while the replacement started")
        .unwrap();
    assert!(compiled.ok(), "{compiled:?}");
    start.notify_one();
    assert!(!failed.await.unwrap().ok());
}

/// A head whose only victim fails to evict (and whose owner then stays
/// closed) does not keep a smaller request that fits from compiling.
#[tokio::test(start_paused = true)]
async fn failed_eviction_does_not_block_a_smaller_request_that_fits() {
    let compiler = SharedCompiler::new(1.0);
    let mut supervisor = queue_supervisor();
    Arc::get_mut(&mut supervisor.inner).unwrap().executable_budget = 66;
    let victim = fake_handle(&compiler, "victim");
    let victim_dyn: Arc<dyn AneShapeWorker> = victim.clone();
    let leased = fake_worker(&compiler, "leased");
    drop(supervisor.lease(&victim_dyn, QUEUE_GTE, 128).await.unwrap());
    let held = supervisor.lease(&leased, QUEUE_GTE, 128).await.unwrap();
    victim.fail_next_evict.store(true, Ordering::SeqCst);
    victim.exit_unconfirmed.store(true, Ordering::SeqCst);
    // Reserved 44; Qwen3@128 needs 28 (72 > 66), so it evicts the victim,
    // which fails; the victim's worker then cannot be confirmed gone.
    let wide = fake_worker(&compiler, "wide");
    let start = compiler.now();
    let blocked = ask_at(&supervisor, &compiler, &wide, QUEUE_QWEN, 128, start, Some(20.0));
    let small = fake_worker(&compiler, "small");
    let fits = ask_at(&supervisor, &compiler, &small, QUEUE_GTE, 512, start + 1.0, None);
    let blocked = blocked.await.unwrap();
    assert!(!blocked.ok(), "{blocked:?}");
    let fits = tokio::time::timeout(Duration::from_secs(60), fits)
        .await
        .expect("a smaller request that fits stayed blocked")
        .unwrap();
    assert!(fits.ok(), "{fits:?}");
    assert!(queued_keys(&supervisor).is_empty());
    drop(held);
}

/// The 250 ms handoff's end wakes the queue head even when time passes
/// between the scheduler's decision and arming its timer (held by a test
/// hook while the paused clock moves past the handoff's end).
#[tokio::test(start_paused = true)]
async fn handoff_end_wakes_the_head_after_a_late_timer() {
    let compiler = SharedCompiler::new(1.0);
    let mut supervisor = queue_supervisor();
    Arc::get_mut(&mut supervisor.inner).unwrap().executable_budget = 22;
    let shared = fake_worker(&compiler, "shared");
    let (_joiner, _held_answer) = raw_want(&supervisor, &shared, QUEUE_GTE, 128);
    let owner = supervisor.lease(&shared, QUEUE_GTE, 128).await.unwrap();
    drop(owner);
    let head_worker = fake_worker(&compiler, "head");
    let head = ask_at(&supervisor, &compiler, &head_worker, QUEUE_GTE, 512, compiler.now(), None);
    until(|| supervisor.inner.lock().starved_head.is_some()).await;
    let (decided, resume) = (Arc::new(Notify::new()), Arc::new(Notify::new()));
    *supervisor.inner.wait_decision_gate.lock().unwrap() = Some((decided.clone(), resume.clone()));
    supervisor.inner.nudge();
    decided.notified().await;
    tokio::time::advance(Duration::from_millis(400)).await;
    resume.notify_one();
    let head = tokio::time::timeout(Duration::from_secs(30), head)
        .await
        .expect("the head slept past the end of the handoff")
        .unwrap();
    assert!(head.ok(), "{head:?}");
}

/// A compile whose requester was cancelled after it reached the worker, and
/// which then panics, is still cleaned up: the scheduler observes the panic
/// itself, recovers the worker, and another worker compiles afterwards.
#[tokio::test(start_paused = true)]
async fn panic_after_the_direct_caller_is_cancelled_is_still_cleaned_up() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let worker = fake_handle(&compiler, "worker");
    let (dispatched, release) = (Arc::new(Notify::new()), Arc::new(Notify::new()));
    *worker.admit_gate.lock().unwrap() = Some((dispatched.clone(), release.clone()));
    worker.panic_next_admit.store(true, Ordering::SeqCst);
    let worker_dyn: Arc<dyn AneShapeWorker> = worker.clone();
    let caller = tokio::spawn({
        let supervisor = supervisor.clone();
        let worker = worker_dyn.clone();
        async move { supervisor.lease(&worker, QUEUE_GTE, 128).await }
    });
    dispatched.notified().await;
    caller.abort();
    let _ = caller.await;
    release.notify_one();
    let other = fake_worker(&compiler, "other");
    let compiled = ask_at(&supervisor, &compiler, &other, QUEUE_GTE, 512, compiler.now(), None);
    let compiled = tokio::time::timeout(Duration::from_secs(60), compiled)
        .await
        .expect("a panicked compile with a cancelled caller stranded the gate")
        .unwrap();
    assert!(compiled.ok(), "{compiled:?}");
    assert_eq!(worker.restarts.load(Ordering::SeqCst), 1);
    assert!(!supervisor
        .inner
        .lock()
        .slots
        .keys()
        .any(|key| key.worker_id == "worker"));
}

/// When a worker's recovery begins, the next queued compile on another
/// worker is elected at once, with no other event to wake the queue.
#[tokio::test(start_paused = true)]
async fn recovery_start_elects_the_next_head_at_once() {
    let compiler = SharedCompiler::new(1.0);
    let mut supervisor = AneResidencySupervisor::new(AneResidencyLimits {
        per_model: 1,
        total: 8,
    });
    Arc::get_mut(&mut supervisor.inner).unwrap().wait_timeout = Duration::from_secs(600);
    let blocked = fake_worker(&compiler, "blocked");
    let held = supervisor.lease(&blocked, QUEUE_GTE, 512).await.unwrap();
    // GTE@128 is the head (its start key is earliest) but its model is at its
    // limit and the only GTE shape is leased.
    let start = compiler.now();
    let head = ask_at(&supervisor, &compiler, &blocked, QUEUE_GTE, 128, start, None);
    let other = fake_worker(&compiler, "other");
    let next = ask_at(&supervisor, &compiler, &other, QUEUE_QWEN, 128, start + 0.001, None);
    until(|| supervisor.inner.lock().starved_head.is_some()).await;
    tokio::time::sleep(Duration::from_secs(5)).await;
    assert!(compiler.compiles_of("other").is_empty());
    let recovery = tokio::spawn({
        let supervisor = supervisor.clone();
        let blocked = blocked.clone();
        async move { supervisor.recover(&blocked).await.is_ok() }
    });
    let next = tokio::time::timeout(Duration::from_secs(30), next)
        .await
        .expect("recovery began but the next head was not elected")
        .unwrap();
    assert!(next.ok(), "{next:?}");
    drop(held);
    assert!(recovery.await.unwrap());
    let _ = head.await;
}

/// A recovering worker's queued compile rejoins the queue without breaking
/// a bound the queue accepted while it was away.
#[tokio::test(start_paused = true)]
async fn recovered_entries_rejoin_behind_accepted_bounds() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let running = fake_worker(&compiler, "running");
    let recovering = fake_handle(&compiler, "recovering");
    let (restarting, finish) = (Arc::new(Notify::new()), Arc::new(Notify::new()));
    *recovering.restart_gate.lock().unwrap() = Some((restarting.clone(), finish.clone()));
    let recovering_dyn: Arc<dyn AneShapeWorker> = recovering.clone();
    let accepted = fake_worker(&compiler, "accepted");
    let running_answer = ask_at(&supervisor, &compiler, &running, QUEUE_QWEN, 2048, 0.0, None);
    // GTE@128 (start key 5.62 s) queues first, then its worker recovers.
    let early = ask_at(&supervisor, &compiler, &recovering_dyn, QUEUE_GTE, 128, 1.0, None);
    tokio::time::sleep_until(compiler.epoch + Duration::from_secs(2)).await;
    let recovery = tokio::spawn({
        let supervisor = supervisor.clone();
        let worker = recovering_dyn.clone();
        async move { supervisor.recover(&worker).await.is_ok() }
    });
    restarting.notified().await;
    // Ready in about 49.9 s without GTE@128 ahead, 54.5 s with it; its bound
    // is 53 s.
    let bound_respected = ask_at(&supervisor, &compiler, &accepted, QUEUE_GTE, 512, 3.0, Some(50.0));
    tokio::time::sleep_until(compiler.epoch + Duration::from_secs(10)).await;
    finish.notify_one();
    assert!(recovery.await.unwrap());
    let bound_respected = bound_respected.await.unwrap();
    assert!(bound_respected.ok(), "an accepted bound was broken: {bound_respected:?}");
    assert!(
        compiler.compiles_of("accepted")[0].start < compiler.compiles_of("recovering")[0].start,
        "the rejoining compile went ahead of an accepted request"
    );
    assert!(early.await.unwrap().ok());
    assert!(running_answer.await.unwrap().ok());
}

/// Two compiles of a recovering worker that each alone would fit ahead of an
/// accepted bound, but not together, rejoin so that the bound still holds:
/// the second goes behind the accepted request.
#[tokio::test(start_paused = true)]
async fn recovered_entries_rejoin_cumulatively_behind_accepted_bounds() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let running = fake_worker(&compiler, "running");
    let recovering = fake_handle(&compiler, "recovering");
    let (restarting, finish) = (Arc::new(Notify::new()), Arc::new(Notify::new()));
    *recovering.restart_gate.lock().unwrap() = Some((restarting.clone(), finish.clone()));
    let recovering_dyn: Arc<dyn AneShapeWorker> = recovering.clone();
    let accepted = fake_worker(&compiler, "accepted");
    // Qwen3@2048 runs until 41.72 s.
    let running_answer = ask_at(&supervisor, &compiler, &running, QUEUE_QWEN, 2048, 0.0, None);
    // GTE@128 (4.62 s, start key 5.62 s) and GTE@512 (8.14 s, start key
    // 9.64 s) queue, then their worker recovers.
    let small = ask_at(&supervisor, &compiler, &recovering_dyn, QUEUE_GTE, 128, 1.0, None);
    let medium = ask_at(&supervisor, &compiler, &recovering_dyn, QUEUE_GTE, 512, 1.5, None);
    tokio::time::sleep_until(compiler.epoch + Duration::from_secs(2)).await;
    let recovery = tokio::spawn({
        let supervisor = supervisor.clone();
        let worker = recovering_dyn.clone();
        async move { supervisor.recover(&worker).await.is_ok() }
    });
    restarting.notified().await;
    // GTE@2048 (18.92 s, start key 21.92 s) is ready at 60.64 s with neither
    // rejoining compile ahead, 65.26 s or 68.78 s with one, and 73.4 s with
    // both. Its bound is 70 s.
    let bound_respected =
        ask_at(&supervisor, &compiler, &accepted, QUEUE_GTE, 2048, 3.0, Some(67.0));
    tokio::time::sleep_until(compiler.epoch + Duration::from_secs(10)).await;
    finish.notify_one();
    assert!(recovery.await.unwrap());
    let bound_respected = bound_respected.await.unwrap();
    assert!(
        bound_respected.ok(),
        "two rejoining compiles together broke an accepted bound: {bound_respected:?}"
    );
    assert!(small.await.unwrap().ok());
    assert!(medium.await.unwrap().ok());
    assert!(running_answer.await.unwrap().ok());
}

/// The running compile is predicted from the moment its admission reached
/// the worker, not from its reservation, so a long wait for the worker's
/// stream does not make it look almost finished.
#[tokio::test(start_paused = true)]
async fn running_prediction_counts_from_dispatch() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let worker = fake_handle(&compiler, "worker");
    let worker_dyn: Arc<dyn AneShapeWorker> = worker.clone();
    let inference = worker.stream.lock().await;
    let running = ask_at(&supervisor, &compiler, &worker_dyn, QUEUE_QWEN, 2048, 0.0, None);
    tokio::time::sleep_until(compiler.epoch + Duration::from_secs(20)).await;
    drop(inference);
    tokio::time::sleep_until(compiler.epoch + Duration::from_secs(21)).await;
    let other = fake_worker(&compiler, "other");
    let refused = ask_at(&supervisor, &compiler, &other, QUEUE_GTE, 128, 21.0, Some(30.0))
        .await
        .unwrap();
    assert_eq!(refused.code.as_deref(), Some(ERR_ANE_SHAPE_COMPILING), "{refused:?}");
    let dispatched = compiler.compiles_of("worker")[0].start;
    let expected = dispatched + standalone(QUEUE_QWEN, 2048) + standalone(QUEUE_GTE, 128) - 21.0;
    assert_close(
        refused.retry_after_ms.unwrap() as f64 / 1000.0,
        expected,
        0.05,
        "retry_after_ms",
    );
    assert!(running.await.unwrap().ok());
}

/// If the scheduler's task ends, a waiting request and a new request both
/// get a typed error promptly, and the lane stays closed.
#[tokio::test(start_paused = true)]
async fn stopped_scheduler_fails_waiting_and_new_requests_promptly() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let worker = fake_handle(&compiler, "worker");
    let worker_dyn: Arc<dyn AneShapeWorker> = worker.clone();
    // A resident shape: leasing it needs no message to the scheduler, so
    // only the lane's closed flag can refuse it.
    drop(supervisor.lease(&worker_dyn, QUEUE_GTE, 2048).await.unwrap());
    let inference = worker.stream.lock().await;
    let waiting = tokio::spawn({
        let supervisor = supervisor.clone();
        let worker = worker_dyn.clone();
        async move { supervisor.lease(&worker, QUEUE_GTE, 128).await }
    });
    until(|| {
        supervisor
            .inner
            .lock()
            .slots
            .contains_key(&queue_key("worker", QUEUE_GTE, 128))
    })
    .await;
    supervisor
        .inner
        .scheduler_task
        .lock()
        .unwrap()
        .take()
        .expect("the scheduler was started")
        .abort();
    let waited = tokio::time::timeout(Duration::from_secs(1), waiting)
        .await
        .expect("a waiting request hung after the scheduler stopped")
        .unwrap();
    assert!(
        matches!(waited, Err(AneResidencyError::SchedulerStopped)),
        "{:?}",
        waited.err()
    );
    let resident = supervisor.lease(&worker_dyn, QUEUE_GTE, 2048).await;
    assert!(
        matches!(resident, Err(AneResidencyError::SchedulerStopped)),
        "a resident shape was leased after the scheduler stopped"
    );
    let fresh = tokio::time::timeout(
        Duration::from_secs(1),
        supervisor.lease(&worker_dyn, QUEUE_GTE, 512),
    )
    .await
    .expect("a new request hung after the scheduler stopped");
    assert!(
        matches!(fresh, Err(AneResidencyError::SchedulerStopped)),
        "{:?}",
        fresh.err()
    );
    drop(inference);
}

/// A recovery whose restart is refused because the worker is being retired
/// has not stopped the old process, so nothing may be refunded yet: the
/// worker's slots stay charged until the retirement confirms the exit.
#[tokio::test(start_paused = true)]
async fn retirement_refusing_a_recovery_keeps_the_live_process_charged() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let worker = fake_handle(&compiler, "worker");
    let worker_dyn: Arc<dyn AneShapeWorker> = worker.clone();
    drop(supervisor.lease(&worker_dyn, QUEUE_GTE, 128).await.unwrap());
    drop(supervisor.lease(&worker_dyn, QUEUE_GTE, 512).await.unwrap());
    worker.retired.store(true, Ordering::SeqCst);
    let refused = supervisor.recover(&worker_dyn).await;
    assert!(refused.is_err(), "{refused:?}");
    assert_eq!(
        supervisor.inner.lock().reserved_executables(),
        44,
        "a refused restart refunded a process whose exit nobody confirmed"
    );
    supervisor
        .retire(&worker_dyn, async { Ok(()) })
        .await
        .unwrap();
    let state = supervisor.inner.lock();
    assert!(!state.slots.keys().any(|key| key.worker_id == "worker"));
    assert_eq!(state.reserved_executables(), 0, "the confirmed retirement refunds");
}

/// A restart that panics after the old process's exit was confirmed (while
/// the replacement starts) still delivers that confirmation: the old
/// process's slots are released instead of staying charged as ghosts.
#[tokio::test(start_paused = true)]
async fn replacement_panic_does_not_lose_a_confirmed_exit() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let worker = fake_handle(&compiler, "worker");
    let worker_dyn: Arc<dyn AneShapeWorker> = worker.clone();
    drop(supervisor.lease(&worker_dyn, QUEUE_GTE, 128).await.unwrap());
    worker.panic_restart_after_exit.store(true, Ordering::SeqCst);
    let outcome = supervisor.recover(&worker_dyn).await;
    assert!(
        matches!(outcome, Err(AneResidencyError::Panicked(_))),
        "{outcome:?}"
    );
    let state = supervisor.inner.lock();
    assert!(
        !state.slots.keys().any(|key| key.worker_id == "worker"),
        "a confirmed exit was lost with the restart's panic"
    );
    assert_eq!(state.reserved_executables(), 0);
    assert!(!state.recovering.contains("worker"), "the worker can be recovered again");
}

/// A restart that panics before the old process's exit is confirmed leaves
/// that process's ownership unknown, so its failed compile keeps holding the
/// gate for the same grace as a restart that never returns, not less.
#[tokio::test(start_paused = true)]
async fn recovery_panic_before_exit_keeps_the_gate_for_the_grace() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let broken = fake_handle(&compiler, "broken");
    broken.transport_fail_next_admit.store(true, Ordering::SeqCst);
    broken.panic_restart_before_exit.store(true, Ordering::SeqCst);
    let broken_dyn: Arc<dyn AneShapeWorker> = broken.clone();
    let failed = ask_at(&supervisor, &compiler, &broken_dyn, QUEUE_GTE, 128, 0.0, None)
        .await
        .unwrap();
    assert!(!failed.ok(), "{failed:?}");
    let failed_at = failed.at;
    let other = fake_worker(&compiler, "other");
    let answer = ask_at(&supervisor, &compiler, &other, QUEUE_GTE, 512, failed_at + 1.0, None);
    tokio::time::sleep_until(compiler.epoch + Duration::from_secs_f64(failed_at + 30.0)).await;
    assert!(
        compiler.compiles_of("other").is_empty(),
        "a panicked recovery released the gate before the grace ended"
    );
    let answer = tokio::time::timeout(Duration::from_secs(60), answer)
        .await
        .expect("the gate was held past the grace")
        .unwrap();
    assert!(answer.ok(), "{answer:?}");
    assert!(supervisor.inner.lock().recovering.contains("broken"));
}

/// When the old process's exit is never confirmed, a caller waiting on the
/// worker's recovery is answered when the scheduler gives up, not left
/// waiting for a restart that may never return.
#[tokio::test(start_paused = true)]
async fn recover_caller_is_answered_when_the_scheduler_gives_up() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let broken = fake_handle(&compiler, "broken");
    let broken_dyn: Arc<dyn AneShapeWorker> = broken.clone();
    drop(supervisor.lease(&broken_dyn, QUEUE_GTE, 128).await.unwrap());
    broken.exit_never_confirmed.store(true, Ordering::SeqCst);
    let outcome = tokio::time::timeout(
        scheduler::EXIT_CONFIRMATION_GIVE_UP + Duration::from_secs(5),
        supervisor.recover(&broken_dyn),
    )
    .await
    .expect("a recover() caller was not answered when the scheduler gave up");
    assert!(
        matches!(&outcome, Err(AneResidencyError::Channel(message)) if message.contains("not confirmed within 35 s")),
        "{outcome:?}"
    );
    let again = tokio::time::timeout(Duration::from_secs(1), supervisor.recover(&broken_dyn))
        .await
        .expect("a later recover() caller waited on the abandoned restart");
    assert!(again.is_err(), "{again:?}");
}

/// A restart that ends without confirming the old process's exit closes the
/// worker until the module restarts: its recovery caller and later requests
/// are refused at once with `WorkerClosed` instead of waiting out the
/// admission bound, and it is not restarted again.
#[tokio::test(start_paused = true)]
async fn closed_worker_refuses_recovery_and_requests_at_once() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let broken = fake_handle(&compiler, "broken");
    let broken_dyn: Arc<dyn AneShapeWorker> = broken.clone();
    drop(supervisor.lease(&broken_dyn, QUEUE_GTE, 128).await.unwrap());
    broken.exit_unconfirmed.store(true, Ordering::SeqCst);
    let outcome = tokio::time::timeout(Duration::from_secs(1), supervisor.recover(&broken_dyn))
        .await
        .expect("a recover() caller waited after the restart ended");
    assert!(
        matches!(&outcome, Err(AneResidencyError::WorkerClosed { worker_id }) if worker_id == "broken"),
        "{outcome:?}"
    );
    let refused = tokio::time::timeout(
        Duration::from_secs(1),
        supervisor.lease(&broken_dyn, QUEUE_GTE, 512),
    )
    .await
    .expect("a request for a closed worker waited");
    assert!(
        matches!(refused, Err(AneResidencyError::WorkerClosed { .. })),
        "{:?}",
        refused.err()
    );
    let generation = broken.generation();
    assert!(matches!(
        supervisor.recover(&broken_dyn).await,
        Err(AneResidencyError::WorkerClosed { .. })
    ));
    assert_eq!(broken.generation(), generation, "a closed worker was restarted again");
    assert_eq!(supervisor.inner.lock().reserved_executables(), 22, "its slot stays charged");
}

/// A mailbox kept busier than the scheduler can drain does not stop it from
/// deciding and from taking finished operations: a cold compile still starts,
/// finishes and is answered while the flood goes on.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn busy_mailbox_does_not_starve_decisions_or_completions() {
    let compiler = SharedCompiler::new(0.01);
    let supervisor = queue_supervisor();
    let worker = fake_worker(&compiler, "worker");
    // Each nudge costs the scheduler 1 ms; the flood sends about three per ms.
    *supervisor.inner.slow_nudges.lock().unwrap() = Some(Duration::from_millis(1));
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flooder = std::thread::spawn({
        let inner = supervisor.inner.clone();
        let stop = stop.clone();
        move || {
            while !stop.load(Ordering::SeqCst) {
                for _ in 0..3 {
                    inner.nudge();
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    });
    let leased = tokio::time::timeout(
        Duration::from_secs(5),
        supervisor.lease(&worker, QUEUE_GTE, 128),
    )
    .await;
    stop.store(true, Ordering::SeqCst);
    flooder.join().unwrap();
    *supervisor.inner.slow_nudges.lock().unwrap() = None;
    assert!(
        matches!(leased, Ok(Ok(_))),
        "a busy mailbox starved the scheduler: {:?}",
        leased.map(|lease| lease.err())
    );
}

/// An admission refused before it reached the worker (here: the channel is
/// faulted) is a definite outcome: nothing is compiling, so the gate is not
/// held, the slot is not kept, and the worker is not restarted for it.
#[tokio::test(start_paused = true)]
async fn refusal_before_the_wire_is_definite() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let refusing = fake_handle(&compiler, "refusing");
    refusing
        .refuse_next_admit_before_wire
        .store(true, Ordering::SeqCst);
    let refusing_dyn: Arc<dyn AneShapeWorker> = refusing.clone();
    let generation = refusing.generation();
    let refused = ask_at(&supervisor, &compiler, &refusing_dyn, QUEUE_GTE, 512, 0.0, None)
        .await
        .unwrap();
    assert!(!refused.ok(), "{refused:?}");
    let other = fake_worker(&compiler, "other");
    let compiled = tokio::time::timeout(
        Duration::from_secs(30),
        ask_at(&supervisor, &compiler, &other, QUEUE_GTE, 2048, compiler.now(), None),
    )
    .await
    .expect("a refusal before the wire held the compile gate")
    .unwrap();
    assert!(compiled.ok(), "{compiled:?}");
    assert_eq!(refusing.generation(), generation, "the worker was restarted");
    let state = supervisor.inner.lock();
    assert!(!state.recovering.contains("refusing"));
    assert!(!state.slots.keys().any(|key| key.worker_id == "refusing"));
}

/// The same as the definite-error case, after a transport fault (an unknown
/// outcome): the failed compile holds the gate only until the old process's
/// exit is confirmed, even when the replacement then fails. Only the exit
/// confirmation can release it here; a failed restart alone releases nothing.
#[tokio::test(start_paused = true)]
async fn transport_fault_releases_the_gate_once_exit_is_confirmed_even_if_restart_fails() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let broken = fake_handle(&compiler, "broken");
    broken.transport_fail_next_admit.store(true, Ordering::SeqCst);
    broken.fail_next_restart.store(true, Ordering::SeqCst);
    let broken_dyn: Arc<dyn AneShapeWorker> = broken.clone();
    let failed = ask_at(&supervisor, &compiler, &broken_dyn, QUEUE_GTE, 128, 0.0, None)
        .await
        .unwrap();
    assert!(!failed.ok(), "{failed:?}");
    let other = fake_worker(&compiler, "other");
    let answer = ask_at(&supervisor, &compiler, &other, QUEUE_GTE, 512, 1.0, None);
    let answer = tokio::time::timeout(Duration::from_secs(30), answer)
        .await
        .expect("a transport-faulted compile with a confirmed exit still held the gate")
        .unwrap();
    assert!(answer.ok(), "{answer:?}");
    let state = supervisor.inner.lock();
    assert!(!state.slots.keys().any(|key| key.worker_id == "broken"));
    assert!(!state.recovering.contains("broken"), "the worker can be recovered again");
    assert_eq!(state.reserved_executables(), 22);
}

/// A request for a resident shape held behind a starved head is granted in
/// the same decision round once that barrier is gone, not left waiting for
/// an unrelated event or its own deadline. The test leaves a barrier behind
/// with no request under it, as a head answered between two passes does.
#[tokio::test(start_paused = true)]
async fn resident_request_behind_a_lapsed_starvation_barrier_is_granted() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let worker = fake_worker(&compiler, "worker");
    drop(supervisor.lease(&worker, QUEUE_GTE, 128).await.unwrap());
    supervisor.inner.lock().starved_head = Some(0);
    let leased = tokio::time::timeout(
        Duration::from_secs(1),
        supervisor.lease(&worker, QUEUE_GTE, 128),
    )
    .await
    .expect("a resident request stayed behind a starvation barrier that had lapsed");
    assert!(leased.is_ok(), "{:?}", leased.err());
}

/// A caller expiry that reaches the scheduler before its request was ever
/// decided is answered after that request is classified: a request queued
/// behind a running compile is told `shape_compiling`, not a bare deadline.
#[tokio::test(start_paused = true)]
async fn expiry_before_the_first_decision_is_answered_after_it() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let running = fake_worker(&compiler, "running");
    let running_answer = ask_at(&supervisor, &compiler, &running, QUEUE_QWEN, 2048, 0.0, None);
    until(|| supervisor.inner.lock().compile_running()).await;
    let other = fake_worker(&compiler, "other");
    let now = tokio::time::Instant::now();
    let (reply, answer) = tokio::sync::oneshot::channel();
    let (expired, expiry) = tokio::sync::oneshot::channel();
    // Both messages are in the mailbox before the scheduler runs again. As
    // `lease_until` sends it, the bound is the request's deadline, which has
    // already passed.
    supervisor
        .inner
        .send(scheduler::Msg::Want(scheduler::Want {
            id: 9_000,
            key: queue_key("other", QUEUE_GTE, 128),
            worker: other,
            executables: 22,
            bound: Some(now),
            deadline: Some(now),
            reply: scheduler::Reply::Lease(reply),
            trace: scheduler::Trace::current(),
        }))
        .unwrap();
    supervisor
        .inner
        .send(scheduler::Msg::Expired {
            id: 9_000,
            reply: expired,
        })
        .unwrap();
    let kind = expiry.await.unwrap();
    assert!(
        matches!(kind, scheduler::AnswerKind::ShapeCompiling { .. }),
        "an expiry decided before its request was classified: {kind:?}"
    );
    assert!(
        matches!(
            answer.await.unwrap(),
            scheduler::Answer::Failed(AneResidencyError::ShapeCompiling { .. })
        ),
        "a request whose bound had passed was expired before it was decided"
    );
    assert!(running_answer.await.unwrap().ok());
}

/// Sends a request straight to the scheduler, as `lease_until` does, and
/// returns its answer channel.
fn want_with_id(
    supervisor: &AneResidencySupervisor,
    worker: &Arc<dyn AneShapeWorker>,
    id: u64,
    shape: usize,
    deadline: Option<tokio::time::Instant>,
) -> tokio::sync::oneshot::Receiver<scheduler::Answer> {
    let (reply, answer) = tokio::sync::oneshot::channel();
    let bound = tokio::time::Instant::now() + Duration::from_secs(600);
    supervisor
        .inner
        .send(scheduler::Msg::Want(scheduler::Want {
            id,
            key: queue_key(worker.worker_id(), QUEUE_GTE, shape),
            worker: worker.clone(),
            executables: 22,
            bound: Some(deadline.map_or(bound, |deadline| deadline.min(bound))),
            deadline,
            reply: scheduler::Reply::Lease(reply),
            trace: scheduler::Trace::current(),
        }))
        .unwrap();
    answer
}

/// Holds the scheduler after its next pass, runs `meanwhile`, then lets it
/// go on and gives it a few polls without moving the clock.
async fn while_scheduler_held(supervisor: &AneResidencySupervisor, meanwhile: impl Future) {
    let (decided, resume) = (Arc::new(Notify::new()), Arc::new(Notify::new()));
    *supervisor.inner.wait_decision_gate.lock().unwrap() =
        Some((decided.clone(), resume.clone()));
    supervisor.inner.nudge();
    decided.notified().await;
    meanwhile.await;
    resume.notify_one();
    for _ in 0..20 {
        tokio::task::yield_now().await;
    }
}

/// A joiner whose bound passed while a compile's completion was waiting to
/// be handled is answered by its bound, not handed the shape late.
#[tokio::test(start_paused = true)]
async fn completion_does_not_outrun_an_overdue_joiner_bound() {
    // The machine compiles 1.5 times slower than predicted: GTE@128 is
    // predicted ready at 4.67 s but finishes at about 7 s.
    let compiler = SharedCompiler::new(1.5);
    let supervisor = queue_supervisor();
    let worker = fake_worker(&compiler, "worker");
    let owner = ask_at(&supervisor, &compiler, &worker, QUEUE_GTE, 128, 0.0, None);
    tokio::time::sleep_until(compiler.epoch + Duration::from_secs(1)).await;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    let mut joined = want_with_id(&supervisor, &worker, 9_002, 128, Some(deadline));
    // Its 6 s bound (accepted: the prediction is earlier) and the compile's
    // end both pass while the scheduler is
    // held, so it sees the completion before the bound's timer.
    while_scheduler_held(
        &supervisor,
        tokio::time::sleep_until(compiler.epoch + Duration::from_secs(8)),
    )
    .await;
    assert!(
        matches!(joined.try_recv(), Ok(scheduler::Answer::Failed(_))),
        "a joiner was handed the shape after its bound had passed"
    );
    assert!(owner.await.unwrap().ok());
}

/// A joiner that went away before its compile's completion was handled is
/// dropped then, instead of staying in the handoff cohort as a member nobody
/// will lease for.
#[tokio::test(start_paused = true)]
async fn closed_joiner_does_not_stay_in_the_handoff_cohort() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let worker = fake_worker(&compiler, "worker");
    let key = queue_key("worker", QUEUE_GTE, 128);
    let owner = ask_at(&supervisor, &compiler, &worker, QUEUE_GTE, 128, 0.0, None);
    tokio::time::sleep_until(compiler.epoch + Duration::from_secs(1)).await;
    // A deadline gives the scheduler a timer to hold on; it is accepted.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(100);
    let joiner = want_with_id(&supervisor, &worker, 9_001, 128, Some(deadline));
    until(|| {
        supervisor
            .inner
            .lock()
            .slots
            .get(&key)
            .is_some_and(|slot| slot.joined.contains(&9_001))
    })
    .await;
    // The joiner goes away and the compile finishes while the scheduler is
    // held, so the completion is handled before the departure is noticed.
    while_scheduler_held(&supervisor, async {
        drop(joiner);
        tokio::time::sleep_until(compiler.epoch + Duration::from_secs(5)).await;
    })
    .await;
    let joined = supervisor.inner.lock().slots[&key].joined.clone();
    assert!(
        !joined.contains(&9_001),
        "a joiner that had gone away stayed in the handoff cohort: {joined:?}"
    );
    assert!(owner.await.unwrap().ok());
}

/// After the scheduler stops, a supervisor whose worker holds it (as the
/// worker channel's connector does) is still freed once its users drop it:
/// the stopped scheduler's slots no longer keep the worker, and so the
/// supervisor, alive.
#[tokio::test(start_paused = true)]
async fn supervisor_drops_cleanly_after_the_scheduler_stops() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let worker = fake_handle(&compiler, "worker");
    let worker_dyn: Arc<dyn AneShapeWorker> = worker.clone();
    drop(supervisor.lease(&worker_dyn, QUEUE_GTE, 128).await.unwrap());
    *worker.held_supervisor.lock().unwrap() = Some(supervisor.clone());
    let inner = Arc::downgrade(&supervisor.inner);
    supervisor
        .inner
        .scheduler_task
        .lock()
        .unwrap()
        .take()
        .expect("the scheduler was started")
        .abort();
    until(|| supervisor.inner.closed.load(Ordering::Acquire)).await;
    drop((supervisor, worker, worker_dyn));
    assert!(
        inner.upgrade().is_none(),
        "the supervisor leaked through its stopped scheduler's slots"
    );
}

/// A scheduler on its own thread ends when the supervisor is dropped, even
/// while one of its operations never finishes: operations do not keep its
/// mailbox open.
#[test]
fn dedicated_scheduler_ends_when_the_supervisor_drops() {
    let supervisor = queue_supervisor();
    let (signal, exited) = std::sync::mpsc::channel();
    *supervisor.inner.scheduler_exit_signal.lock().unwrap() = Some(signal);
    let compiler = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async { SharedCompiler::new(1.0) });
    let broken = fake_handle(&compiler, "broken");
    broken.exit_never_confirmed.store(true, Ordering::SeqCst);
    let broken_dyn: Arc<dyn AneShapeWorker> = broken.clone();
    // No Tokio runtime on this thread, so the scheduler starts on its own.
    let (reply, _answer) = tokio::sync::oneshot::channel();
    supervisor
        .inner
        .send(scheduler::Msg::Recover {
            worker: broken_dyn.clone(),
            generation: 0,
            reply,
        })
        .unwrap();
    // The restart has begun (it bumps the generation) and never returns.
    let started = std::time::Instant::now();
    while broken.generation() == 0 {
        assert!(started.elapsed() < Duration::from_secs(5), "the restart never began");
        std::thread::sleep(Duration::from_millis(5));
    }
    drop((supervisor, broken_dyn));
    assert_eq!(
        exited.recv_timeout(Duration::from_secs(5)),
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected),
        "the scheduler's thread outlived its supervisor"
    );
}

/// A confirmation of the old process's exit handled after the restart's
/// completion has already reopened the worker (the completion can be taken
/// before queued messages) releases only the old process's slots: a compile
/// the replacement started meanwhile keeps its slot and the gate.
#[tokio::test(start_paused = true)]
async fn late_exit_confirmation_does_not_release_the_replacement_slots() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let worker = fake_handle(&compiler, "worker");
    let worker_dyn: Arc<dyn AneShapeWorker> = worker.clone();
    let old_generation = worker.generation();
    assert!(supervisor.recover(&worker_dyn).await.is_ok());
    assert!(worker.generation() > old_generation);
    // The replacement starts a compile, held once it reached the worker.
    let (reached, release) = (Arc::new(Notify::new()), Arc::new(Notify::new()));
    *worker.admit_gate.lock().unwrap() = Some((reached.clone(), release.clone()));
    let compile = ask_at(&supervisor, &compiler, &worker_dyn, QUEUE_GTE, 128, compiler.now(), None);
    reached.notified().await;
    // The old process's confirmation arrives only now.
    supervisor
        .inner
        .send(scheduler::Msg::ExitConfirmed {
            worker_id: "worker".into(),
            generation: old_generation,
        })
        .unwrap();
    supervisor.inner.nudge();
    for _ in 0..20 {
        tokio::task::yield_now().await;
    }
    {
        let state = supervisor.inner.lock();
        assert!(
            state
                .slots
                .get(&queue_key("worker", QUEUE_GTE, 128))
                .is_some_and(|slot| slot.state == SlotState::Admitting),
            "a late exit confirmation removed the replacement's compiling slot"
        );
        assert!(state.compile_running(), "the gate opened mid-compile");
        assert_eq!(state.reserved_executables(), 22);
    }
    release.notify_one();
    assert!(compile.await.unwrap().ok());
}

/// A cohort member whose answer is never read (its request went away after
/// the answer was sent) leaves the handoff cohort when that unread answer is
/// dropped, instead of holding the handoff open for its full 250 ms.
#[tokio::test(start_paused = true)]
async fn unread_cohort_answer_leaves_the_handoff_at_once() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let worker = fake_worker(&compiler, "worker");
    let key = queue_key("worker", QUEUE_GTE, 128);
    // GTE@128 compiles from about 0.05 s to 4.67 s; its handoff then lasts
    // until about 4.92 s.
    let owner = ask_at(&supervisor, &compiler, &worker, QUEUE_GTE, 128, 0.0, None);
    tokio::time::sleep_until(compiler.epoch + Duration::from_secs(1)).await;
    let joiner = want_with_id(&supervisor, &worker, 9_003, 128, None);
    tokio::time::sleep_until(compiler.epoch + Duration::from_secs_f64(4.75)).await;
    assert!(
        supervisor.inner.lock().slots[&key].joined.contains(&9_003),
        "the joiner was meant to be in the open handoff"
    );
    // The request goes away without reading its answer.
    drop(joiner);
    for _ in 0..20 {
        tokio::task::yield_now().await;
    }
    {
        let state = supervisor.inner.lock();
        assert!(
            !state.slots[&key].joined.contains(&9_003),
            "an unread cohort answer kept its member in the handoff"
        );
        assert!(!state.slots[&key].handing_off(tokio::time::Instant::now()));
    }
    assert!(owner.await.unwrap().ok());
}

/// A compile's dispatch is visible in the shared state the moment its
/// admission reaches the worker, not only once the scheduler handles the
/// report: a cancelled owner deciding whether its compile is running must
/// never read it as undispatched while it is on the wire.
#[tokio::test(start_paused = true)]
async fn dispatch_is_visible_the_moment_the_admission_reaches_the_worker() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let worker = fake_handle(&compiler, "worker");
    *worker.held_supervisor.lock().unwrap() = Some(supervisor.clone());
    let worker_dyn: Arc<dyn AneShapeWorker> = worker.clone();
    let compiled = ask_at(&supervisor, &compiler, &worker_dyn, QUEUE_GTE, 128, 0.0, None);
    assert!(compiled.await.unwrap().ok());
    let seen = worker.dispatch_seen_at_wire.lock().unwrap().take();
    *worker.held_supervisor.lock().unwrap() = None;
    assert_eq!(
        seen,
        Some(true),
        "the slot read as undispatched after its admission reached the worker"
    );
}

/// A retirement whose shutdown fails without confirming the old process's
/// exit keeps that worker's failed compile holding the gate for the same
/// grace as a restart would, counted from the shutdown, instead of giving it
/// up at once.
#[tokio::test(start_paused = true)]
async fn failed_retirement_shutdown_keeps_the_gate_for_the_grace() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let failing = fake_handle(&compiler, "failing");
    failing.transport_fail_next_admit.store(true, Ordering::SeqCst);
    // Its recovery stops before the old process's exit is confirmed.
    let (restarting, _never) = (Arc::new(Notify::new()), Arc::new(Notify::new()));
    *failing.restart_gate.lock().unwrap() = Some((restarting.clone(), _never.clone()));
    let failing_dyn: Arc<dyn AneShapeWorker> = failing.clone();
    let failed = ask_at(&supervisor, &compiler, &failing_dyn, QUEUE_GTE, 512, 0.0, None)
        .await
        .unwrap();
    assert!(!failed.ok(), "{failed:?}");
    restarting.notified().await;
    let retired_at = compiler.now();
    let shutdown = supervisor
        .retire(&failing_dyn, async {
            Err(AneResidencyError::Channel("injected shutdown failure".into()))
        })
        .await;
    assert!(shutdown.is_err());
    let other = fake_worker(&compiler, "other");
    let answer = ask_at(&supervisor, &compiler, &other, QUEUE_GTE, 128, compiler.now(), None);
    tokio::time::sleep_until(compiler.epoch + Duration::from_secs_f64(retired_at + 30.0)).await;
    assert!(
        compiler.compiles_of("other").is_empty(),
        "a failed retirement shutdown released the gate before the grace ended"
    );
    let answer = tokio::time::timeout(Duration::from_secs(60), answer)
        .await
        .expect("the gate was held past the grace")
        .unwrap();
    assert!(answer.ok(), "{answer:?}");
}
