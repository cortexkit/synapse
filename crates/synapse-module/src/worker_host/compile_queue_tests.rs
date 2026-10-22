// Tests for the direct-ANE cold-compile queue, run against a stand-in for
// Apple's Neural Engine compiler. Measurements on an M4 mini showed that the
// compiler serves one layer at a time for the whole machine: each layer of a
// shape compile holds one shared step, and compiles running side by side
// alternate layer by layer. So a small shape compiled beside a wide one waits
// one wide layer per layer of its own (GTE@128 took 5.2 s alone and 66.7 s
// beside Qwen3@4096). The fake below reproduces that with a fair FIFO lock
// held for one layer's time, on Tokio's paused clock, so every duration in
// these tests is exact virtual time.

const QUEUE_GTE: &str = "ane-direct:gte-modernbert-base.ane-direct-worker:test";
const QUEUE_QWEN: &str = "ane-direct:qwen3-embedding-0.6b.ane-direct-worker:test";

/// Layer count and median per-layer compile seconds measured on the M4 mini.
/// Kept apart from the supervisor's own seed table on purpose: a wrong seed
/// must show up as a wrong prediction here, not cancel out.
fn fake_compile_profile(model_ref: &str, shape: usize) -> (usize, f64) {
    let gte = model_ref == QUEUE_GTE;
    let per_layer = match (gte, shape) {
        (true, 128) => 0.21,
        (true, 512) => 0.37,
        (true, 2048) => 0.86,
        (false, 128) => 0.38,
        (false, 512) => 0.76,
        (false, 2048) => 1.49,
        (false, 4096) => 3.02,
        other => panic!("no measured layer time for {other:?}"),
    };
    (if gte { 22 } else { 28 }, per_layer)
}

/// Standalone compile seconds of one shape on the fake, before any slowdown.
fn standalone(model_ref: &str, shape: usize) -> f64 {
    let (layers, per_layer) = fake_compile_profile(model_ref, shape);
    layers as f64 * per_layer
}

#[derive(Clone, Debug)]
struct CompileRecord {
    worker: String,
    model_ref: String,
    shape: usize,
    start: f64,
    end: Option<f64>,
}

/// The machine-wide compiler every fake worker shares.
struct SharedCompiler {
    /// One layer at a time, in arrival order (Tokio's mutex is fair).
    step: tokio::sync::Mutex<()>,
    /// How much slower than the measured table this machine compiles.
    slowdown: f64,
    epoch: tokio::time::Instant,
    log: Mutex<Vec<CompileRecord>>,
    running: std::sync::atomic::AtomicUsize,
    peak: std::sync::atomic::AtomicUsize,
}

impl SharedCompiler {
    fn new(slowdown: f64) -> Arc<Self> {
        Arc::new(Self {
            step: tokio::sync::Mutex::new(()),
            slowdown,
            epoch: tokio::time::Instant::now(),
            log: Mutex::new(Vec::new()),
            running: Default::default(),
            peak: Default::default(),
        })
    }

    fn now(&self) -> f64 {
        self.epoch.elapsed().as_secs_f64()
    }

    async fn compile(&self, worker: &str, model_ref: &str, shape: usize) {
        let (layers, per_layer) = fake_compile_profile(model_ref, shape);
        let index = {
            let mut log = self.log.lock().unwrap();
            log.push(CompileRecord {
                worker: worker.into(),
                model_ref: model_ref.into(),
                shape,
                start: self.now(),
                end: None,
            });
            log.len() - 1
        };
        let running = self.running.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(running, Ordering::SeqCst);
        for _ in 0..layers {
            let _step = self.step.lock().await;
            tokio::time::sleep(Duration::from_secs_f64(per_layer * self.slowdown)).await;
        }
        self.running.fetch_sub(1, Ordering::SeqCst);
        self.log.lock().unwrap()[index].end = Some(self.now());
    }

    fn records(&self) -> Vec<CompileRecord> {
        self.log.lock().unwrap().clone()
    }

    fn compiles_of(&self, worker: &str) -> Vec<CompileRecord> {
        self.records()
            .into_iter()
            .filter(|record| record.worker == worker)
            .collect()
    }
}

struct FakeCompileWorker {
    id: String,
    compiler: Arc<SharedCompiler>,
}

impl AneShapeWorker for FakeCompileWorker {
    fn worker_id(&self) -> &str {
        &self.id
    }
    fn stream_idle(&self) -> bool {
        true
    }
    fn admit_shape<'a>(
        &'a self,
        model_ref: &'a str,
        shape: usize,
    ) -> BoxFuture<'a, Result<AnePlacementInventory, AneResidencyError>> {
        Box::pin(async move {
            self.compiler.compile(&self.id, model_ref, shape).await;
            Ok(AnePlacementInventory::default())
        })
    }
    fn evict_shape<'a>(
        &'a self,
        _: &'a str,
        _: usize,
    ) -> BoxFuture<'a, Result<(), AneResidencyError>> {
        Box::pin(async { Ok(()) })
    }
    fn restart(&self) -> BoxFuture<'_, Result<(), AneResidencyError>> {
        Box::pin(async { Ok(()) })
    }
}

fn fake_worker(compiler: &Arc<SharedCompiler>, id: &str) -> Arc<dyn AneShapeWorker> {
    Arc::new(FakeCompileWorker {
        id: id.into(),
        compiler: compiler.clone(),
    })
}

/// What a request got, and when (seconds since the compiler's epoch).
#[derive(Debug)]
struct QueueAnswer {
    at: f64,
    asked_at: f64,
    code: Option<String>,
    retry_after_ms: Option<u64>,
}

impl QueueAnswer {
    fn ok(&self) -> bool {
        self.code.is_none()
    }
    fn waited(&self) -> f64 {
        self.at - self.asked_at
    }
}

/// Asks for `shape` at `at` seconds after the epoch, with an optional
/// deadline that many seconds after asking, and drops the lease at once.
fn ask_at(
    supervisor: &AneResidencySupervisor,
    compiler: &Arc<SharedCompiler>,
    worker: &Arc<dyn AneShapeWorker>,
    model_ref: &'static str,
    shape: usize,
    at: f64,
    deadline: Option<f64>,
) -> tokio::task::JoinHandle<QueueAnswer> {
    let supervisor = supervisor.clone();
    let compiler = compiler.clone();
    let worker = worker.clone();
    tokio::spawn(async move {
        tokio::time::sleep_until(compiler.epoch + Duration::from_secs_f64(at)).await;
        let asked_at = compiler.now();
        let request_deadline = deadline
            .map(|seconds| tokio::time::Instant::now() + Duration::from_secs_f64(seconds));
        let result = supervisor
            .lease_until(&worker, model_ref, shape, request_deadline)
            .await;
        let at = compiler.now();
        match result {
            Ok(lease) => {
                drop(lease);
                QueueAnswer {
                    at,
                    asked_at,
                    code: None,
                    retry_after_ms: None,
                }
            }
            Err(error) => QueueAnswer {
                at,
                asked_at,
                code: Some(error.code().unwrap_or("other").to_owned()),
                retry_after_ms: error
                    .to_engine_error(synapse_core::EngineErrorStage::Inference)
                    .retry_after_ms,
            },
        }
    })
}

fn queue_supervisor() -> AneResidencySupervisor {
    AneResidencySupervisor::new(AneResidencyLimits::default())
}

fn assert_close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= expected * tolerance,
        "{what}: {actual:.3} is not within {:.0}% of {expected:.3}",
        tolerance * 100.0
    );
}

#[tokio::test(start_paused = true)]
async fn small_compile_beside_wide_finishes_in_its_standalone_time() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let wide = fake_worker(&compiler, "wide");
    let small = fake_worker(&compiler, "small");
    let wide_answer = ask_at(&supervisor, &compiler, &wide, QUEUE_QWEN, 4096, 0.0, None);
    let small_answer = ask_at(&supervisor, &compiler, &small, QUEUE_GTE, 128, 0.001, None);
    let small_answer = small_answer.await.unwrap();
    let wide_answer = wide_answer.await.unwrap();
    assert!(small_answer.ok() && wide_answer.ok());
    let small_alone = standalone(QUEUE_GTE, 128);
    // Today the small compile alternates with the wide one layer by layer
    // and takes about 22 * (3.02 + 0.21) = 71 s instead of 4.6 s.
    assert!(
        small_answer.waited() <= 1.1 * small_alone + 0.1,
        "GTE@128 beside Qwen3@4096 took {:.1} s; alone it takes {small_alone:.1} s",
        small_answer.waited()
    );
    let wide_limit = standalone(QUEUE_QWEN, 4096) + small_alone + 0.2;
    assert!(
        wide_answer.waited() <= wide_limit,
        "Qwen3@4096 took {:.1} s, more than its own and the small compile's time ({wide_limit:.1} s)",
        wide_answer.waited()
    );
}

#[tokio::test(start_paused = true)]
async fn at_most_one_cold_compile_runs_across_workers() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let mut answers = Vec::new();
    for (index, model_ref) in [QUEUE_GTE, QUEUE_GTE, QUEUE_QWEN, QUEUE_QWEN]
        .into_iter()
        .enumerate()
    {
        let worker = fake_worker(&compiler, &format!("w{index}"));
        for shape in [128, 512, 2048] {
            answers.push(ask_at(&supervisor, &compiler, &worker, model_ref, shape, 0.0, None));
        }
    }
    for answer in answers {
        let answer = answer.await.unwrap();
        assert!(answer.ok(), "request failed: {answer:?}");
    }
    assert_eq!(compiler.records().len(), 12);
    let mut keys: Vec<_> = compiler
        .records()
        .into_iter()
        .map(|record| (record.worker, record.model_ref, record.shape))
        .collect();
    keys.sort();
    keys.dedup();
    assert_eq!(keys.len(), 12, "a shape was compiled twice");
    assert_eq!(
        compiler.peak.load(Ordering::SeqCst),
        1,
        "more than one shape compile ran at once: {:?}",
        compiler.records()
    );
}

#[tokio::test(start_paused = true)]
async fn queued_compiles_start_cheapest_first() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let requests = [
        ("first", QUEUE_QWEN, 2048, 0.0),
        ("wide", QUEUE_QWEN, 4096, 1.0),
        ("gte", QUEUE_GTE, 512, 2.0),
        ("qwen", QUEUE_QWEN, 128, 3.0),
    ];
    let mut answers = Vec::new();
    for (id, model_ref, shape, at) in requests {
        let worker = fake_worker(&compiler, id);
        answers.push(ask_at(&supervisor, &compiler, &worker, model_ref, shape, at, None));
    }
    for answer in answers {
        assert!(answer.await.unwrap().ok());
    }
    let mut records = compiler.records();
    records.sort_by(|left, right| left.start.total_cmp(&right.start));
    let order: Vec<_> = records.iter().map(|record| record.worker.as_str()).collect();
    // While the first compile runs, the three queued ones start cheapest
    // first (GTE@512 8.1 s, Qwen3@128 10.6 s, Qwen3@4096 84.6 s), not in
    // arrival order.
    assert_eq!(order, ["first", "gte", "qwen", "wide"]);
}

#[tokio::test(start_paused = true)]
async fn wide_compile_starts_after_its_aging_window() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let first = fake_worker(&compiler, "first");
    let wide = fake_worker(&compiler, "wide");
    let mut answers = vec![
        ask_at(&supervisor, &compiler, &first, QUEUE_QWEN, 2048, 0.0, None),
        ask_at(&supervisor, &compiler, &wide, QUEUE_QWEN, 4096, 1.0, None),
    ];
    // A steady stream of small compiles, one every 4 s on fresh workers, each
    // a little longer (4.62 s) than the gap between them, so the small
    // backlog never drains while the stream lasts.
    let arrivals: Vec<f64> = (0..30).map(|index| 2.0 + 4.0 * index as f64).collect();
    for (index, at) in arrivals.iter().enumerate() {
        let worker = fake_worker(&compiler, &format!("small{index}"));
        answers.push(ask_at(&supervisor, &compiler, &worker, QUEUE_GTE, 128, *at, None));
    }
    for answer in answers {
        let answer = answer.await.unwrap();
        assert!(answer.ok(), "request failed: {answer:?}");
    }
    let records = compiler.records();
    let start_of = |worker: &str| {
        records
            .iter()
            .find(|record| record.worker == worker)
            .map(|record| record.start)
            .unwrap()
    };
    let wide_start = start_of("wide");
    // The wide compile's start key is its arrival plus its cost:
    // 1 + 84.56 = 85.56 s. Small compiles whose own key (arrival + 4.62 s) is
    // earlier go first; later ones wait behind it, however cheap they are.
    let wide_key = 1.0 + standalone(QUEUE_QWEN, 4096);
    for (index, at) in arrivals.iter().enumerate() {
        let small_start = start_of(&format!("small{index}"));
        if at + standalone(QUEUE_GTE, 128) < wide_key {
            assert!(
                small_start < wide_start,
                "small compile asked at {at} s started at {small_start:.2} s, after the wide one ({wide_start:.2} s)"
            );
        } else {
            assert!(
                small_start > wide_start,
                "small compile asked at {at} s started at {small_start:.2} s, ahead of the wide one ({wide_start:.2} s) that had aged past it"
            );
        }
    }
    assert_eq!(compiler.peak.load(Ordering::SeqCst), 1);
}

#[tokio::test(start_paused = true)]
async fn queued_compile_holds_no_executable_budget() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let requests = [
        ("first", QUEUE_QWEN, 2048, 0.0),
        ("a", QUEUE_QWEN, 128, 0.1),
        ("b", QUEUE_GTE, 128, 0.1),
        ("c", QUEUE_GTE, 512, 0.1),
    ];
    let mut answers = Vec::new();
    for (id, model_ref, shape, at) in requests {
        let worker = fake_worker(&compiler, id);
        answers.push(ask_at(&supervisor, &compiler, &worker, model_ref, shape, at, None));
    }
    tokio::time::sleep_until(compiler.epoch + Duration::from_secs(5)).await;
    // Only the running Qwen3@2048 compile (28 layers) is charged; the three
    // queued compiles reserve nothing until their turn.
    assert_eq!(supervisor.inner.lock().reserved_executables(), 28);
    for answer in answers {
        assert!(answer.await.unwrap().ok());
    }
}

#[tokio::test(start_paused = true)]
async fn predicted_overrun_refuses_early_with_honest_retry() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let wide = fake_worker(&compiler, "wide");
    let small = fake_worker(&compiler, "small");
    let wide_answer = ask_at(&supervisor, &compiler, &wide, QUEUE_QWEN, 4096, 0.0, None);
    let refused = ask_at(&supervisor, &compiler, &small, QUEUE_GTE, 128, 1.0, Some(30.0))
        .await
        .unwrap();
    assert_eq!(refused.code.as_deref(), Some(ERR_ANE_SHAPE_COMPILING));
    assert!(
        refused.waited() < 0.01,
        "a request that cannot be served by its deadline waited {:.2} s for it",
        refused.waited()
    );
    // The wide compile ends at about 84.6 s, then GTE@128 takes 4.62 s.
    let expected = 0.05 + standalone(QUEUE_QWEN, 4096) + standalone(QUEUE_GTE, 128) - 1.0;
    let retry = refused.retry_after_ms.unwrap() as f64 / 1000.0;
    assert_close(retry, expected, 0.1, "retry_after_ms");
    // The compile stays queued without the request, so the retry it was
    // told to make finds the shape ready (or finishing) instead of
    // compiling it again.
    let retried = ask_at(
        &supervisor,
        &compiler,
        &small,
        QUEUE_GTE,
        128,
        refused.at + retry,
        Some(30.0),
    )
    .await
    .unwrap();
    assert!(retried.ok(), "retry failed: {retried:?}");
    assert!(retried.waited() < 0.05, "retry waited {:.2} s", retried.waited());
    assert_eq!(compiler.compiles_of("small").len(), 1);
    assert!(wide_answer.await.unwrap().ok());
}

#[tokio::test(start_paused = true)]
async fn queue_beyond_600s_refuses_early_without_a_deadline() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let mut answers = Vec::new();
    for index in 0..8 {
        let worker = fake_worker(&compiler, &format!("w{index}"));
        let at = 0.001 * index as f64;
        answers.push(ask_at(&supervisor, &compiler, &worker, QUEUE_QWEN, 4096, at, None));
    }
    let mut results = Vec::new();
    for answer in answers {
        results.push(answer.await.unwrap());
    }
    for answer in &results[..7] {
        assert!(answer.ok(), "a request predicted within 600 s failed: {answer:?}");
    }
    // The eighth Qwen3@4096 compile is predicted at 8 * 84.56 = 676 s, beyond
    // the 600 s admission bound: it is told so at once, with the prediction,
    // instead of failing with ane_resources_exhausted after 600 s.
    let last = &results[7];
    assert_eq!(last.code.as_deref(), Some(ERR_ANE_SHAPE_COMPILING), "{last:?}");
    assert!(last.waited() < 0.01, "refused after {:.2} s", last.waited());
    let expected = 0.05 + 8.0 * standalone(QUEUE_QWEN, 4096) - last.asked_at;
    assert_close(
        last.retry_after_ms.unwrap() as f64 / 1000.0,
        expected,
        0.1,
        "retry_after_ms",
    );
}

#[tokio::test(start_paused = true)]
async fn deadline_while_queued_for_compile_answers_shape_compiling() {
    // This machine compiles twice as slowly as predicted, and nothing has
    // been measured yet, so the queue's first prediction is wrong.
    let compiler = SharedCompiler::new(2.0);
    let supervisor = queue_supervisor();
    let first = fake_worker(&compiler, "first");
    let small = fake_worker(&compiler, "small");
    let first_answer = ask_at(&supervisor, &compiler, &first, QUEUE_QWEN, 2048, 0.0, None);
    // Predicted ready at about 46 s, inside the 60 s deadline, so the request
    // waits; the running compile actually ends at about 83 s.
    let expired = ask_at(&supervisor, &compiler, &small, QUEUE_GTE, 128, 1.0, Some(60.0))
        .await
        .unwrap();
    assert_eq!(
        expired.code.as_deref(),
        Some(ERR_ANE_SHAPE_COMPILING),
        "a request queued for its compile must not answer deadline_exceeded: {expired:?}"
    );
    assert!(
        (expired.waited() - 60.0).abs() < 0.01,
        "answered after {:.2} s, not at its 60 s deadline",
        expired.waited()
    );
    assert!(expired.retry_after_ms.unwrap() >= ANE_SHAPE_COMPILING_RETRY_AFTER_MS);
    // The queued compile survives the request; a retry within the promised
    // delay attaches to it, and the shape is compiled once, after the
    // running compile and never beside it.
    let retry = expired.retry_after_ms.unwrap() as f64 / 1000.0;
    let retried = ask_at(
        &supervisor,
        &compiler,
        &small,
        QUEUE_GTE,
        128,
        expired.at + retry,
        Some(60.0),
    )
    .await
    .unwrap();
    assert!(retried.ok(), "retry failed: {retried:?}");
    let first_end = compiler.compiles_of("first")[0].end.unwrap();
    let small_compiles = compiler.compiles_of("small");
    assert_eq!(small_compiles.len(), 1, "{small_compiles:?}");
    assert!(
        small_compiles[0].start >= first_end,
        "GTE@128 started at {:.2} s beside the running compile (ends {first_end:.2} s)",
        small_compiles[0].start
    );
    assert!(first_answer.await.unwrap().ok());
}

#[tokio::test(start_paused = true)]
async fn machine_factor_corrects_unseen_shape_predictions() {
    // This machine compiles 1.3 times slower than the seed table, like the
    // M5 Max against the M4 mini.
    let compiler = SharedCompiler::new(1.3);
    let supervisor = queue_supervisor();
    let probe = fake_worker(&compiler, "probe");
    assert!(ask_at(&supervisor, &compiler, &probe, QUEUE_GTE, 512, 0.0, None)
        .await
        .unwrap()
        .ok());
    let running = fake_worker(&compiler, "running");
    let small = fake_worker(&compiler, "small");
    let running_answer = ask_at(&supervisor, &compiler, &running, QUEUE_QWEN, 2048, 11.0, None);
    // Unscaled, the queue would predict GTE@128 ready 45.4 s after asking,
    // inside the 50 s deadline. Scaled by the measured 1.3, it predicts
    // 59.3 s, so the request is refused at once.
    let refused = ask_at(&supervisor, &compiler, &small, QUEUE_GTE, 128, 12.0, Some(50.0))
        .await
        .unwrap();
    assert_eq!(refused.code.as_deref(), Some(ERR_ANE_SHAPE_COMPILING), "{refused:?}");
    assert!(refused.waited() < 0.01, "refused after {:.2} s", refused.waited());
    let running_start = compiler.compiles_of("running")[0].start;
    let expected = running_start + 1.3 * standalone(QUEUE_QWEN, 2048)
        + 1.3 * standalone(QUEUE_GTE, 128)
        - refused.asked_at;
    assert_close(
        refused.retry_after_ms.unwrap() as f64 / 1000.0,
        expected,
        0.05,
        "retry_after_ms",
    );
    assert!(running_answer.await.unwrap().ok());
}

#[tokio::test(start_paused = true)]
async fn orphaned_entry_expires_after_twice_its_promise() {
    // Three times slower than predicted: the running compile overruns its
    // prediction by far, so the refused request's promise cannot be kept.
    let compiler = SharedCompiler::new(3.0);
    let supervisor = queue_supervisor();
    let wide = fake_worker(&compiler, "wide");
    let small = fake_worker(&compiler, "small");
    let wide_answer = ask_at(&supervisor, &compiler, &wide, QUEUE_QWEN, 4096, 0.0, None);
    let refused = ask_at(&supervisor, &compiler, &small, QUEUE_GTE, 128, 1.0, Some(10.0))
        .await
        .unwrap();
    assert_eq!(refused.code.as_deref(), Some(ERR_ANE_SHAPE_COMPILING), "{refused:?}");
    assert!(refused.waited() < 0.01, "refused after {:.2} s", refused.waited());
    let promised = refused.retry_after_ms.unwrap() as f64 / 1000.0;
    // Nobody comes back. The wide compile ends at about 254 s, after the
    // entry's expiry at 1 + 2 * 88 = 177 s, so the abandoned GTE@128 compile
    // must never run.
    assert!(wide_answer.await.unwrap().ok());
    let wide_end = compiler.compiles_of("wide")[0].end.unwrap();
    assert!(wide_end > refused.at + 2.0 * promised);
    tokio::time::sleep_until(compiler.epoch + Duration::from_secs(400)).await;
    assert!(
        compiler.compiles_of("small").is_empty(),
        "an entry nobody came back for kept compiling: {:?}",
        compiler.compiles_of("small")
    );
}

#[tokio::test(start_paused = true)]
async fn cheaper_newcomer_does_not_break_an_accepted_deadline() {
    let compiler = SharedCompiler::new(1.0);
    let supervisor = queue_supervisor();
    let first = fake_worker(&compiler, "first");
    let accepted = fake_worker(&compiler, "accepted");
    let newcomer = fake_worker(&compiler, "newcomer");
    let first_answer = ask_at(&supervisor, &compiler, &first, QUEUE_QWEN, 2048, 0.0, None);
    // Qwen3@128 is predicted ready at about 52.4 s, inside its 56 s bound,
    // so it is accepted and waits.
    let accepted_answer = ask_at(
        &supervisor,
        &compiler,
        &accepted,
        QUEUE_QWEN,
        128,
        1.0,
        Some(55.0),
    );
    // GTE@512 is cheaper and its start key (2 + 8.14 s) is earlier, but going
    // first would push the accepted request to about 60.6 s, past its bound.
    let newcomer_answer = ask_at(&supervisor, &compiler, &newcomer, QUEUE_GTE, 512, 2.0, None);
    let accepted_answer = accepted_answer.await.unwrap();
    assert!(
        accepted_answer.ok(),
        "a request told its compile would be ready in time missed its deadline: {accepted_answer:?}"
    );
    assert!(newcomer_answer.await.unwrap().ok());
    assert!(first_answer.await.unwrap().ok());
    let accepted_start = compiler.compiles_of("accepted")[0].start;
    let newcomer_start = compiler.compiles_of("newcomer")[0].start;
    assert!(
        accepted_start < newcomer_start,
        "the cheaper newcomer ({newcomer_start:.2} s) went ahead of the accepted request ({accepted_start:.2} s)"
    );
}
