// Tests for compile predictions that know which shapes aned already holds
// compiled, and for the background prewarm of a ready lane's short shapes.
// They reuse the fake worker and shared compiler of the compile-queue tests;
// the fake's `aned_cache` stands in for aned's compiled-program cache, and
// `reports_origins` for a worker that sends the `ADMITTED` cache fields.

/// The rungs a direct-ANE worker reports on in `LOADED`.
const CACHE_LADDER: [usize; 7] = ANE_SHAPE_LADDER;

/// A fake worker with an aned cache holding `cached` shapes of `model_ref`,
/// that reports where each admitted layer came from.
fn caching_worker(
    compiler: &Arc<SharedCompiler>,
    id: &str,
    model_ref: &str,
    cached: &[usize],
) -> Arc<FakeCompileWorker> {
    let worker = fake_handle(compiler, id);
    *worker.aned_cache.lock().unwrap() = Some(
        cached
            .iter()
            .map(|shape| (model_ref.to_owned(), *shape))
            .collect(),
    );
    worker.reports_origins.store(true, Ordering::SeqCst);
    worker
}

/// Lets the scheduler handle every message sent so far.
async fn settle() {
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }
}

#[tokio::test(start_paused = true)]
async fn cached_shape_is_predicted_at_load_cost_and_not_refused_early() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let handle = caching_worker(&compiler, "qwen", QUEUE_QWEN, &[4096]);
    let worker: Arc<dyn AneShapeWorker> = handle.clone();
    supervisor
        .record_compile_cache(QUEUE_QWEN, &CACHE_LADDER, &[4096])
        .unwrap();
    // Predicted as a compile, Qwen3@4096 takes about 84.6 s and a 30 s
    // deadline is refused at once. aned holds it, so it loads in about 1.7 s.
    let answer = ask_at(&supervisor, &compiler, &worker, QUEUE_QWEN, 4096, 0.0, Some(30.0))
        .await
        .unwrap();
    assert!(answer.ok(), "a cached shape was refused: {answer:?}");
    assert_close(answer.waited(), 28.0 * FAKE_CACHED_LOAD_PER_LAYER + 0.05, 0.1, "cached load");
    assert!(compiler.records().is_empty(), "nothing may compile");
    assert_eq!(handle.cached_loads.lock().unwrap().len(), 1);
    let state = supervisor.inner.lock();
    assert!(
        state.measured_loads.contains_key(&(QUEUE_QWEN.to_owned(), 4096)),
        "the load was not learned as a load"
    );
    assert!(state.measured_costs.is_empty(), "a load was learned as a compile");
    assert_eq!(state.machine_factor, None);
}

#[tokio::test(start_paused = true)]
async fn cached_load_that_compiles_is_repriced_as_a_compile() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    // LOAD reported 4096 cached, but aned dropped it since: the admission
    // compiles for 84.6 s.
    let wide = caching_worker(&compiler, "wide", QUEUE_QWEN, &[]);
    let wide: Arc<dyn AneShapeWorker> = wide;
    let small = fake_worker(&compiler, "small");
    supervisor
        .record_compile_cache(QUEUE_QWEN, &CACHE_LADDER, &[4096])
        .unwrap();
    let wide_answer = ask_at(&supervisor, &compiler, &wide, QUEUE_QWEN, 4096, 0.0, None);
    // Asked while the load prediction (4.2 s) still holds: accepted, then
    // refused as soon as the admission runs past twice that plus 1 s.
    let early = ask_at(&supervisor, &compiler, &small, QUEUE_GTE, 128, 1.0, Some(30.0));
    // Asked after that moment: refused at once with the compile's remaining
    // time plus its own compile.
    let late = ask_at(&supervisor, &compiler, &small, QUEUE_GTE, 128, 20.0, Some(30.0));
    let early = early.await.unwrap();
    let late = late.await.unwrap();
    let gte = standalone(QUEUE_GTE, 128);
    let wide_compile = standalone(QUEUE_QWEN, 4096);
    let load_seed = 28.0 * 0.15;
    let missed_at = 2.0 * load_seed + 1.0 + 0.05;
    assert_eq!(early.code.as_deref(), Some(ERR_ANE_SHAPE_COMPILING), "{early:?}");
    assert_close(early.at, missed_at, 0.02, "the accepted request was re-checked at the miss");
    assert_close(
        early.retry_after_ms.unwrap() as f64 / 1000.0,
        wide_compile - missed_at + gte,
        0.05,
        "retry after the repriced compile",
    );
    assert_eq!(late.code.as_deref(), Some(ERR_ANE_SHAPE_COMPILING), "{late:?}");
    assert!(late.waited() < 0.1, "refused late: {late:?}");
    assert_close(
        late.retry_after_ms.unwrap() as f64 / 1000.0,
        wide_compile - 20.0 + gte,
        0.05,
        "retry after the repriced compile",
    );
    assert!(wide_answer.await.unwrap().ok());
    let state = supervisor.inner.lock();
    // A full compile is learned as one, not as the shape's load time.
    let learned = state.measured_costs[&(QUEUE_QWEN.to_owned(), 4096, AneProgram::Single)];
    assert_close(learned.as_secs_f64(), wide_compile, 0.02, "learned compile");
    assert!(!state.measured_loads.contains_key(&(QUEUE_QWEN.to_owned(), 4096)));
    assert_eq!(state.cached.get(&(QUEUE_QWEN.to_owned(), 4096)), Some(&true));
}

#[tokio::test(start_paused = true)]
async fn cached_loads_do_not_change_the_machine_factor() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let worker: Arc<dyn AneShapeWorker> =
        caching_worker(&compiler, "qwen", QUEUE_QWEN, &[128, 256, 512]);
    supervisor
        .record_compile_cache(QUEUE_QWEN, &CACHE_LADDER, &[128, 256, 512])
        .unwrap();
    for (index, shape) in [128, 256, 512].into_iter().enumerate() {
        let answer = ask_at(
            &supervisor,
            &compiler,
            &worker,
            QUEUE_QWEN,
            shape,
            index as f64 * 5.0,
            None,
        );
        assert!(answer.await.unwrap().ok());
    }
    assert_eq!(supervisor.inner.lock().machine_factor, None);
    // Qwen3@2048 is not cached: its seed (41.7 s) still applies, so a 30 s
    // deadline is refused at once rather than accepted on a factor learned
    // from 1.7 s loads.
    let answer = ask_at(&supervisor, &compiler, &worker, QUEUE_QWEN, 2048, 20.0, Some(30.0))
        .await
        .unwrap();
    assert_eq!(answer.code.as_deref(), Some(ERR_ANE_SHAPE_COMPILING), "{answer:?}");
    assert!(answer.waited() < 0.1, "{answer:?}");
    assert_close(
        answer.retry_after_ms.unwrap() as f64 / 1000.0,
        28.0 * 1.49,
        0.02,
        "retry after the seed compile",
    );
}

#[tokio::test(start_paused = true)]
async fn older_worker_without_cache_fields_keeps_full_compile_predictions() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    // aned holds the shape, but the worker neither reports it in LOADED nor
    // says where admitted layers came from.
    let handle = caching_worker(&compiler, "qwen", QUEUE_QWEN, &[128, 4096]);
    handle.reports_origins.store(false, Ordering::SeqCst);
    let worker: Arc<dyn AneShapeWorker> = handle.clone();
    let answer = ask_at(&supervisor, &compiler, &worker, QUEUE_QWEN, 4096, 0.0, Some(30.0))
        .await
        .unwrap();
    assert_eq!(answer.code.as_deref(), Some(ERR_ANE_SHAPE_COMPILING), "{answer:?}");
    assert!(answer.waited() < 0.1, "{answer:?}");
    assert_close(
        answer.retry_after_ms.unwrap() as f64 / 1000.0,
        28.0 * 3.02,
        0.02,
        "retry after the seed compile",
    );
    // Its admissions are learned as compiles, as before.
    assert!(ask_at(&supervisor, &compiler, &worker, QUEUE_QWEN, 128, 100.0, None)
        .await
        .unwrap()
        .ok());
    let state = supervisor.inner.lock();
    assert!(state.machine_factor.is_some());
    assert!(state.measured_loads.is_empty());
    assert!(state.cached.is_empty());
}

#[tokio::test(start_paused = true)]
async fn prewarm_never_delays_a_queued_request() {
    let compiler = SharedCompiler::new(1.0);
    // Room for every shape here, so only the order is under test.
    let supervisor = budget_supervisor(200, ANE_ADMISSION_WAIT_TIMEOUT);
    let lane = fake_worker(&compiler, "lane");
    let other = fake_worker(&compiler, "other");
    let first = ask_at(&supervisor, &compiler, &lane, QUEUE_QWEN, 2048, 0.0, None);
    tokio::time::sleep_until(compiler.epoch + Duration::from_secs(1)).await;
    supervisor
        .prewarm(&lane, QUEUE_QWEN, "qwen-lane", &[128, 256, 512])
        .unwrap();
    // A second readiness signal for the same worker starts nothing more.
    supervisor
        .prewarm(&lane, QUEUE_QWEN, "qwen-lane", &[128, 256, 512])
        .unwrap();
    let request = ask_at(&supervisor, &compiler, &other, QUEUE_GTE, 512, 2.0, None);
    assert!(first.await.unwrap().ok());
    let request = request.await.unwrap();
    assert!(request.ok());
    let qwen_2048 = standalone(QUEUE_QWEN, 2048);
    assert_close(
        request.waited(),
        qwen_2048 - 2.0 + standalone(QUEUE_GTE, 512),
        0.02,
        "the request waited only for the compile already running",
    );
    tokio::time::sleep(Duration::from_secs(120)).await;
    let mut records = compiler.records();
    records.sort_by(|left, right| left.start.total_cmp(&right.start));
    let order: Vec<_> = records
        .iter()
        .map(|record| (record.worker.as_str(), record.shape))
        .collect();
    assert_eq!(
        order,
        [
            ("lane", 2048),
            ("other", 512),
            ("lane", 128),
            ("lane", 256),
            ("lane", 512)
        ]
    );
    assert_eq!(compiler.peak.load(Ordering::SeqCst), 1);
    assert_queue_idle(&supervisor);
}

#[tokio::test(start_paused = true)]
async fn prewarm_waits_for_a_request_in_its_gathering_window() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let lane = fake_worker(&compiler, "lane");
    let other = fake_worker(&compiler, "other");
    // The request waits 50 ms before its compile may start, so compiles
    // asked for together start cheapest first. A prewarm signalled inside
    // that window must not take the compiler first.
    let request = ask_at(&supervisor, &compiler, &other, QUEUE_GTE, 512, 0.0, None);
    tokio::time::sleep_until(compiler.epoch + Duration::from_millis(10)).await;
    supervisor
        .prewarm(&lane, QUEUE_QWEN, "qwen-lane", &[128])
        .unwrap();
    let request = request.await.unwrap();
    assert!(request.ok());
    assert_close(
        request.waited(),
        standalone(QUEUE_GTE, 512) + 0.05,
        0.02,
        "the request started before the prewarm",
    );
    tokio::time::sleep(Duration::from_secs(30)).await;
    assert_eq!(compiler.compiles_of("lane").len(), 1, "prewarm never ran");
}

#[tokio::test(start_paused = true)]
async fn prewarm_skips_what_the_free_budget_cannot_hold_and_evicts_nothing() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let gte_handle = fake_handle(&compiler, "gte");
    let gte: Arc<dyn AneShapeWorker> = gte_handle.clone();
    let qwen = fake_worker(&compiler, "qwen");
    // 44 of the 100 executables in use, one of the shapes leased.
    let leased = supervisor.lease_until(&gte, QUEUE_GTE, 128, None).await.unwrap();
    drop(supervisor.lease_until(&gte, QUEUE_GTE, 512, None).await.unwrap());
    supervisor
        .prewarm(&qwen, QUEUE_QWEN, "qwen-lane", &[128, 256, 512])
        .unwrap();
    tokio::time::sleep(Duration::from_secs(120)).await;
    let prewarmed: Vec<_> = compiler
        .compiles_of("qwen")
        .iter()
        .map(|record| record.shape)
        .collect();
    // 128 and 256 take 56 more; 512 would need 28 beyond the 100.
    assert_eq!(prewarmed, [128, 256]);
    assert!(gte_handle.evicted().is_empty(), "prewarm evicted a shape");
    let state = supervisor.inner.lock();
    assert_eq!(state.reserved_executables(), 100);
    assert!(state.slots.keys().any(|key| key.model_ref == QUEUE_GTE && key.shape == 512));
    drop(state);
    drop(leased);
}

#[tokio::test(start_paused = true)]
async fn prewarm_refused_for_resources_evicts_nothing() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let gte_handle = fake_handle(&compiler, "gte");
    let gte: Arc<dyn AneShapeWorker> = gte_handle.clone();
    let qwen_handle = fake_handle(&compiler, "qwen");
    let qwen: Arc<dyn AneShapeWorker> = qwen_handle.clone();
    drop(supervisor.lease_until(&gte, QUEUE_GTE, 512, None).await.unwrap());
    qwen_handle.exhaust_next_admit.store(true, Ordering::SeqCst);
    supervisor
        .prewarm(&qwen, QUEUE_QWEN, "qwen-lane", &[128, 256, 512])
        .unwrap();
    tokio::time::sleep(Duration::from_secs(120)).await;
    assert!(
        gte_handle.evicted().is_empty(),
        "a refused prewarm evicted a shape to retry"
    );
    assert!(compiler.compiles_of("qwen").is_empty(), "prewarm went on after the refusal");
    let state = supervisor.inner.lock();
    assert!(state.slots.keys().all(|key| key.worker_id == "gte"));
    assert!(!state.compile_running());
}

#[tokio::test(start_paused = true)]
async fn prewarm_stops_when_the_lane_retires() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let lane = fake_worker(&compiler, "lane");
    supervisor
        .prewarm(&lane, QUEUE_QWEN, "qwen-lane", &[128, 256, 512])
        .unwrap();
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert_eq!(compiler.compiles_of("lane").len(), 1, "prewarm did not start");
    // Unloading the lane (or pausing the Neural Engine, which unloads it)
    // retires its worker. The admission already running finishes; nothing
    // else starts.
    supervisor
        .retire(&lane, async { Ok::<(), AneResidencyError>(()) })
        .await
        .unwrap();
    settle().await;
    tokio::time::sleep(Duration::from_secs(120)).await;
    let prewarmed: Vec<_> = compiler
        .compiles_of("lane")
        .iter()
        .map(|record| record.shape)
        .collect();
    assert_eq!(prewarmed, [128]);
    assert!(supervisor.inner.lock().slots.is_empty());
    // A retired worker is not prewarmed again.
    supervisor
        .prewarm(&lane, QUEUE_QWEN, "qwen-lane", &[128, 256, 512])
        .unwrap();
    tokio::time::sleep(Duration::from_secs(120)).await;
    assert_eq!(compiler.compiles_of("lane").len(), 1);
}

#[tokio::test(start_paused = true)]
async fn prewarm_loads_cached_shapes_without_compiling() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let handle = caching_worker(&compiler, "lane", QUEUE_QWEN, &[128, 256, 512]);
    let lane: Arc<dyn AneShapeWorker> = handle.clone();
    supervisor
        .record_compile_cache(QUEUE_QWEN, &CACHE_LADDER, &[128, 256, 512])
        .unwrap();
    supervisor
        .prewarm(&lane, QUEUE_QWEN, "qwen-lane", &[128, 256, 512])
        .unwrap();
    tokio::time::sleep(Duration::from_secs(10)).await;
    assert!(compiler.records().is_empty());
    let loads: Vec<_> = handle
        .cached_loads
        .lock()
        .unwrap()
        .iter()
        .map(|(_, shape, _)| *shape)
        .collect();
    assert_eq!(loads, [128, 256, 512]);
    let (key, slot_id) = {
        let mut state = supervisor.inner.lock();
        assert_eq!(
            state
                .slots
                .values()
                .filter(|slot| slot.state == SlotState::Resident)
                .count(),
            3
        );
        let (key, slot) = state
            .slots
            .iter_mut()
            .find(|(key, _)| key.shape == 128)
            .unwrap();
        slot.state = SlotState::Evicting;
        (key.clone(), slot.id)
    };
    assert!(supervisor.evict(lane.clone(), key, slot_id).await);
    // The lane was prewarmed once; another readiness signal for the same
    // worker loads nothing again.
    supervisor
        .prewarm(&lane, QUEUE_QWEN, "qwen-lane", &[128, 256, 512])
        .unwrap();
    tokio::time::sleep(Duration::from_secs(10)).await;
    assert_eq!(handle.cached_loads.lock().unwrap().len(), 3);
}

#[tokio::test(start_paused = true)]
async fn prewarm_admission_stays_when_a_joined_request_is_cancelled() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let handle = fake_handle(&compiler, "lane");
    let lane: Arc<dyn AneShapeWorker> = handle.clone();
    // A long inference holds the stream, so the prewarm admission has not
    // reached the worker yet when a request joins it and is cancelled.
    let stream = handle.stream.lock().await;
    supervisor.prewarm(&lane, QUEUE_QWEN, "qwen-lane", &[128]).unwrap();
    settle().await;
    let joined = {
        let supervisor = supervisor.clone();
        let lane = lane.clone();
        tokio::spawn(async move { supervisor.lease_until(&lane, QUEUE_QWEN, 128, None).await })
    };
    tokio::time::sleep(Duration::from_secs(1)).await;
    joined.abort();
    let _ = joined.await;
    // Make sure the scheduler runs a pass while the admission still waits.
    let _ = supervisor.inner.send(scheduler::Msg::Nudge);
    tokio::time::sleep(Duration::from_secs(1)).await;
    drop(stream);
    tokio::time::sleep(Duration::from_secs(30)).await;
    assert_eq!(
        compiler.compiles_of("lane").len(),
        1,
        "the prewarm admission was withdrawn with the request that joined it"
    );
}

#[tokio::test(start_paused = true)]
async fn prewarm_admission_stays_after_a_joined_request_promise_lapses() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let handle = fake_handle(&compiler, "lane");
    let lane: Arc<dyn AneShapeWorker> = handle.clone();
    let stream = handle.stream.lock().await;
    supervisor.prewarm(&lane, QUEUE_QWEN, "qwen-lane", &[128]).unwrap();
    settle().await;
    // Joins the waiting prewarm admission and is answered shape_compiling
    // at its 1 s deadline, which records a promise on the admission.
    let answer = ask_at(&supervisor, &compiler, &lane, QUEUE_QWEN, 128, 0.5, Some(1.0))
        .await
        .unwrap();
    assert_eq!(answer.code.as_deref(), Some(ERR_ANE_SHAPE_COMPILING), "{answer:?}");
    // Long after twice that promise, the stream frees.
    tokio::time::sleep(Duration::from_secs(60)).await;
    drop(stream);
    tokio::time::sleep(Duration::from_secs(30)).await;
    assert_eq!(
        compiler.compiles_of("lane").len(),
        1,
        "the prewarm admission was withdrawn when the promise lapsed"
    );
}
