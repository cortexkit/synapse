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
        .record_compile_cache(QUEUE_QWEN, &CACHE_LADDER, &[4096], tokio::time::Instant::now())
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
        .record_compile_cache(QUEUE_QWEN, &CACHE_LADDER, &[4096], tokio::time::Instant::now())
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
        .record_compile_cache(QUEUE_QWEN, &CACHE_LADDER, &[128, 256, 512], tokio::time::Instant::now())
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
        .prewarm(&lane, QUEUE_QWEN, "qwen-lane", &[128, 256, 512], None)
        .unwrap();
    // A second readiness signal for the same worker starts nothing more.
    supervisor
        .prewarm(&lane, QUEUE_QWEN, "qwen-lane", &[128, 256, 512], None)
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
        .prewarm(&lane, QUEUE_QWEN, "qwen-lane", &[128], None)
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
        .prewarm(&qwen, QUEUE_QWEN, "qwen-lane", &[128, 256, 512], None)
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
        .prewarm(&qwen, QUEUE_QWEN, "qwen-lane", &[128, 256, 512], None)
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
        .prewarm(&lane, QUEUE_QWEN, "qwen-lane", &[128, 256, 512], None)
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
        .prewarm(&lane, QUEUE_QWEN, "qwen-lane", &[128, 256, 512], None)
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
        .record_compile_cache(QUEUE_QWEN, &CACHE_LADDER, &[128, 256, 512], tokio::time::Instant::now())
        .unwrap();
    supervisor
        .prewarm(&lane, QUEUE_QWEN, "qwen-lane", &[128, 256, 512], None)
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
        .prewarm(&lane, QUEUE_QWEN, "qwen-lane", &[128, 256, 512], None)
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
    // The prewarm admission is held before it reaches the worker, so it
    // could still be withdrawn when a request joins it and is cancelled.
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    *handle.pre_wire_gate.lock().unwrap() = Some((started.clone(), release.clone()));
    supervisor.prewarm(&lane, QUEUE_QWEN, "qwen-lane", &[128], None).unwrap();
    started.notified().await;
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
    release.notify_one();
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
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    *handle.pre_wire_gate.lock().unwrap() = Some((started.clone(), release.clone()));
    supervisor.prewarm(&lane, QUEUE_QWEN, "qwen-lane", &[128], None).unwrap();
    started.notified().await;
    // Joins the waiting prewarm admission and is answered shape_compiling
    // at its 1 s deadline, which records a promise on the admission.
    let answer = ask_at(&supervisor, &compiler, &lane, QUEUE_QWEN, 128, 0.5, Some(1.0))
        .await
        .unwrap();
    assert_eq!(answer.code.as_deref(), Some(ERR_ANE_SHAPE_COMPILING), "{answer:?}");
    // Long after twice that promise, the admission goes on to the worker.
    tokio::time::sleep(Duration::from_secs(60)).await;
    release.notify_one();
    tokio::time::sleep(Duration::from_secs(30)).await;
    assert_eq!(
        compiler.compiles_of("lane").len(),
        1,
        "the prewarm admission was withdrawn when the promise lapsed"
    );
}

#[tokio::test(start_paused = true)]
async fn packed_preparation_waits_for_a_prewarm_admission_instead_of_giving_up() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let handle = fake_handle(&compiler, "lane");
    let lane: Arc<dyn AneShapeWorker> = handle.clone();
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    *handle.admit_gate.lock().unwrap() = Some((started.clone(), release.clone()));
    supervisor.prewarm(&lane, QUEUE_QWEN, "qwen-lane", &[128], None).unwrap();
    started.notified().await;
    let preparation = {
        let supervisor = supervisor.clone();
        let lane = lane.clone();
        tokio::spawn(async move { supervisor.prepare_packed_lease(&lane, QUEUE_QWEN).await })
    };
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert!(
        !preparation.is_finished(),
        "packed preparation gave up behind a prewarm admission"
    );
    release.notify_one();
    // Once the prewarm admission ends, the packed program is admitted; the
    // fake worker has no packed program, so that admission is refused.
    let prepared = preparation.await.unwrap();
    assert_eq!(
        prepared.err().and_then(|error| error.code().map(str::to_owned)).as_deref(),
        Some("unsupported_request")
    );
    assert_eq!(compiler.compiles_of("lane").len(), 1);
}

/// Holds `handle`'s next admission once it reaches the worker; returns the
/// notifications that it started and that releases it.
fn hold_admission(handle: &FakeCompileWorker) -> (Arc<Notify>, Arc<Notify>) {
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    *handle.admit_gate.lock().unwrap() = Some((started.clone(), release.clone()));
    (started, release)
}

#[tokio::test(start_paused = true)]
async fn joined_request_runs_its_inference_before_the_next_prewarm() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let handle = fake_handle(&compiler, "lane");
    let lane: Arc<dyn AneShapeWorker> = handle.clone();
    supervisor
        .prewarm(&lane, QUEUE_QWEN, "qwen-lane", &[128, 256], None)
        .unwrap();
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert_eq!(compiler.compiles_of("lane").len(), 1, "prewarm 128 did not start");
    // R asks for 128 while it is being prewarmed, and joins that admission.
    let request = {
        let supervisor = supervisor.clone();
        let lane = lane.clone();
        tokio::spawn(async move { supervisor.lease_until(&lane, QUEUE_QWEN, 128, None).await })
    };
    tokio::time::sleep(Duration::from_secs(1)).await;
    // Hold R right after its handoff answer, before it leases, while the
    // scheduler runs well past the handoff.
    let checked = Arc::new(Notify::new());
    let resume = Arc::new(Notify::new());
    *supervisor.inner.lease_decision_gate.lock().unwrap() = Some((checked.clone(), resume.clone()));
    checked.notified().await;
    tokio::time::sleep(Duration::from_secs(30)).await;
    assert_eq!(
        compiler.compiles_of("lane").len(),
        1,
        "a second prewarm admission started while a request was on its way to its lease"
    );
    resume.notify_one();
    let lease = request.await.unwrap().unwrap();
    // R's inference takes the worker stream at once: no prewarm holds it.
    let inference = handle
        .stream
        .try_lock()
        .expect("a prewarm admission holds the stream R's inference needs");
    tokio::time::sleep(Duration::from_secs(5)).await;
    assert_eq!(compiler.compiles_of("lane").len(), 1);
    drop(inference);
    drop(lease);
    tokio::time::sleep(Duration::from_secs(60)).await;
    let shapes: Vec<_> = compiler
        .compiles_of("lane")
        .iter()
        .map(|record| record.shape)
        .collect();
    assert_eq!(shapes, [128, 256], "prewarm did not resume once R finished");
}

#[tokio::test(start_paused = true)]
async fn resident_shape_inference_holds_off_prewarm() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let handle = fake_handle(&compiler, "lane");
    let lane: Arc<dyn AneShapeWorker> = handle.clone();
    let lease = supervisor
        .lease_until(&lane, QUEUE_QWEN, 2048, None)
        .await
        .unwrap();
    supervisor
        .prewarm(&lane, QUEUE_QWEN, "qwen-lane", &[128], None)
        .unwrap();
    tokio::time::sleep(Duration::from_secs(60)).await;
    assert_eq!(
        compiler.compiles_of("lane").len(),
        1,
        "prewarm started while a resident shape of the lane was leased"
    );
    // Stream use without a lease (an exchange still draining) also holds it
    // off; nothing reports its end, so prewarm looks again by itself.
    let stream = handle.stream.lock().await;
    drop(lease);
    tokio::time::sleep(Duration::from_secs(5)).await;
    assert_eq!(
        compiler.compiles_of("lane").len(),
        1,
        "prewarm started while the worker stream was in use"
    );
    drop(stream);
    tokio::time::sleep(Duration::from_secs(20)).await;
    let shapes: Vec<_> = compiler
        .compiles_of("lane")
        .iter()
        .map(|record| record.shape)
        .collect();
    assert_eq!(shapes, [2048, 128], "prewarm did not start once the stream was free");
}

#[tokio::test(start_paused = true)]
async fn prewarm_resumes_after_a_recovery_releases_its_admission() {
    let compiler = SharedCompiler::new(1.0);
    // A short drain bound, so the recovery restarts the worker while the
    // prewarm admission is still held.
    let supervisor = budget_supervisor(ANE_RESIDENT_EXECUTABLES_TOTAL, Duration::from_secs(5));
    let handle = fake_handle(&compiler, "lane");
    let lane: Arc<dyn AneShapeWorker> = handle.clone();
    let (started, release) = hold_admission(&handle);
    supervisor
        .prewarm(&lane, QUEUE_QWEN, "qwen-lane", &[128, 256], None)
        .unwrap();
    started.notified().await;
    let recovery = {
        let supervisor = supervisor.clone();
        let lane = lane.clone();
        tokio::spawn(async move { supervisor.recover(&lane).await })
    };
    recovery.await.unwrap().unwrap();
    assert_eq!(handle.restarts.load(Ordering::SeqCst), 1);
    assert!(
        supervisor.inner.lock().slots.is_empty(),
        "the confirmed exit did not release the prewarm admission's slot"
    );
    // The old process's admission ends only now, after its slot is gone.
    release.notify_one();
    tokio::time::sleep(Duration::from_secs(120)).await;
    let shapes: Vec<_> = compiler
        .compiles_of("lane")
        .iter()
        .map(|record| record.shape)
        .collect();
    assert_eq!(
        shapes,
        [128, 128, 256],
        "the prewarm did not load its interrupted shape again and go on"
    );
}

#[tokio::test(start_paused = true)]
async fn definite_prewarm_failure_ends_the_prewarm_without_a_restart() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let handle = fake_handle(&compiler, "lane");
    let lane: Arc<dyn AneShapeWorker> = handle.clone();
    handle.fail_next_admit.store(true, Ordering::SeqCst);
    supervisor
        .prewarm(&lane, QUEUE_QWEN, "qwen-lane", &[128, 256], None)
        .unwrap();
    tokio::time::sleep(Duration::from_secs(60)).await;
    assert_eq!(
        handle.restarts.load(Ordering::SeqCst),
        0,
        "a refused prewarm admission restarted the lane's worker"
    );
    assert!(compiler.compiles_of("lane").is_empty(), "the prewarm went on");
    {
        let state = supervisor.inner.lock();
        assert!(state.slots.is_empty(), "the refused admission stayed charged");
        assert!(!state.recovering.contains("lane"));
    }
    // An unknown outcome still restarts the worker: the stream's state is
    // unknown, as for any admission.
    let other = fake_handle(&compiler, "other");
    let other_dyn: Arc<dyn AneShapeWorker> = other.clone();
    other.transport_fail_next_admit.store(true, Ordering::SeqCst);
    supervisor
        .prewarm(&other_dyn, QUEUE_QWEN, "other-lane", &[128], None)
        .unwrap();
    tokio::time::sleep(Duration::from_secs(60)).await;
    assert_eq!(other.restarts.load(Ordering::SeqCst), 1);
}

#[tokio::test(start_paused = true)]
async fn retirement_withdraws_an_undispatched_prewarm_admission() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let handle = fake_handle(&compiler, "lane");
    let lane: Arc<dyn AneShapeWorker> = handle.clone();
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    *handle.pre_wire_gate.lock().unwrap() = Some((started.clone(), release.clone()));
    supervisor
        .prewarm(&lane, QUEUE_QWEN, "qwen-lane", &[128], None)
        .unwrap();
    started.notified().await;
    let retired = tokio::time::timeout(
        Duration::from_secs(5),
        supervisor.retire(&lane, async { Ok::<(), AneResidencyError>(()) }),
    )
    .await;
    assert!(
        retired.is_ok(),
        "retirement waited for a prewarm admission that had not reached the worker"
    );
    retired.unwrap().unwrap();
    release.notify_one();
    tokio::time::sleep(Duration::from_secs(30)).await;
    assert!(compiler.compiles_of("lane").is_empty());
    assert!(supervisor.inner.lock().slots.is_empty());
}

#[tokio::test(start_paused = true)]
async fn packed_preparation_waits_for_budget_a_prewarm_admission_holds() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let gte = fake_worker(&compiler, "gte");
    let handle = fake_handle(&compiler, "lane");
    let lane: Arc<dyn AneShapeWorker> = handle.clone();
    // Three leased 22-executable shapes elsewhere: 66 of 100.
    let mut leases = Vec::new();
    for shape in [128, 512, 2048] {
        leases.push(supervisor.lease_until(&gte, QUEUE_GTE, shape, None).await.unwrap());
    }
    let notified = Arc::new(AtomicU64::new(0));
    let notify: PrewarmNotify = {
        let notified = notified.clone();
        Arc::new(move || {
            notified.fetch_add(1, Ordering::SeqCst);
        })
    };
    let (started, release) = hold_admission(&handle);
    supervisor
        .prewarm(&lane, QUEUE_QWEN, "qwen-lane", &[128], Some(notify))
        .unwrap();
    started.notified().await;
    // 94 of 100 reserved: a packed program (28) fits once the prewarmed
    // shape is resident and can be evicted.
    assert_eq!(supervisor.inner.lock().reserved_executables(), 94);
    assert!(
        supervisor.preparation_possible(&lane),
        "preparation gave up although the prewarm's budget is about to be reclaimable"
    );
    let preparation = {
        let supervisor = supervisor.clone();
        let lane = lane.clone();
        tokio::spawn(async move { supervisor.prepare_packed_lease(&lane, QUEUE_QWEN).await })
    };
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert!(
        !preparation.is_finished(),
        "packed preparation gave up behind budget a prewarm admission holds"
    );
    assert_eq!(notified.load(Ordering::SeqCst), 0);
    release.notify_one();
    let prepared = preparation.await.unwrap();
    // The prewarmed shape was evicted for the packed program, which the
    // fake worker does not have.
    assert_eq!(
        prepared
            .err()
            .and_then(|error| error.code().map(str::to_owned))
            .as_deref(),
        Some("unsupported_request")
    );
    assert_eq!(handle.evicted(), [(QUEUE_QWEN.to_owned(), 128)]);
    // Notified when the prewarm admission ended and when the prewarm ended.
    assert_eq!(notified.load(Ordering::SeqCst), 2);
    drop(leases);
}

#[tokio::test(start_paused = true)]
async fn late_cache_check_does_not_overwrite_a_newer_admission() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let worker: Arc<dyn AneShapeWorker> = caching_worker(&compiler, "qwen", QUEUE_QWEN, &[]);
    let key = (QUEUE_QWEN.to_owned(), 128);
    // A LOAD-time check begins, then an admission of 128 finishes, then the
    // check's report (128 not cached) arrives.
    let check_began = tokio::time::Instant::now();
    tokio::time::sleep(Duration::from_millis(10)).await;
    drop(supervisor.lease_until(&worker, QUEUE_QWEN, 128, None).await.unwrap());
    supervisor
        .record_compile_cache(QUEUE_QWEN, &CACHE_LADDER, &[], check_began)
        .unwrap();
    settle().await;
    assert_eq!(
        supervisor.inner.lock().cached.get(&key),
        Some(&true),
        "an older cache check overwrote what a later admission saw"
    );
    // A check that began after that admission still says what aned holds.
    supervisor
        .record_compile_cache(QUEUE_QWEN, &CACHE_LADDER, &[], tokio::time::Instant::now())
        .unwrap();
    settle().await;
    assert_eq!(supervisor.inner.lock().cached.get(&key), Some(&false));
}

#[tokio::test(start_paused = true)]
async fn mixed_origin_admission_is_learned_as_neither() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let handle = caching_worker(&compiler, "qwen", QUEUE_QWEN, &[]);
    handle.mixed_origins.store(true, Ordering::SeqCst);
    let worker: Arc<dyn AneShapeWorker> = handle.clone();
    drop(supervisor.lease_until(&worker, QUEUE_QWEN, 128, None).await.unwrap());
    let state = supervisor.inner.lock();
    assert!(state.measured_costs.is_empty(), "a mixed admission was learned as a compile");
    assert!(state.measured_loads.is_empty(), "a mixed admission was learned as a load");
    assert_eq!(state.machine_factor, None);
    assert_eq!(state.cached.get(&(QUEUE_QWEN.to_owned(), 128)), Some(&true));
}
