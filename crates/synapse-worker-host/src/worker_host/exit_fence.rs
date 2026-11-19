//! Bounded confirmation that a killed worker process has exited.
//!
//! A worker blocked in uninterruptible I/O (a page fault on a hung storage
//! mount, for example) may not exit promptly even after SIGKILL. Waiting for
//! it without a bound would keep the caller's request, and every lane and
//! engine guard it holds, stuck behind that one process. Instead the host waits
//! a fixed time, then hands the child to an independent reaper task and fences
//! the host: no replacement worker starts until the reaper confirms the exit,
//! and requests in the meantime get a transient `engine_crashed` refusal.

use std::sync::Arc;
use std::time::Duration;

use tokio::process::Child;
use tokio::task::JoinHandle;

use super::{WorkerHost, WorkerHostError};

/// How long the host waits for a killed worker process to exit before treating
/// the exit as unconfirmed. The direct-ANE channel uses the same bound when it
/// retires or restarts its worker, including on Neural Engine pause.
pub(crate) const OWNER_EXIT_TIMEOUT: Duration = Duration::from_secs(30);

/// Killed worker processes whose exit has not been confirmed yet.
#[derive(Default)]
pub(super) struct ExitFence {
    reapers: Vec<JoinHandle<()>>,
    /// Replaces [`OWNER_EXIT_TIMEOUT`] in tests.
    #[cfg(test)]
    pub(super) test_grace: Option<Duration>,
    /// Holds every exit wait until the test closes the semaphore, to stand in
    /// for a process that does not exit when killed.
    #[cfg(test)]
    pub(super) test_hold: Option<Arc<tokio::sync::Semaphore>>,
}

impl ExitFence {
    fn grace(&self) -> Duration {
        #[cfg(test)]
        if let Some(grace) = self.test_grace {
            return grace;
        }
        OWNER_EXIT_TIMEOUT
    }

    /// Whether every killed process has been confirmed exited.
    pub(super) fn clear(&mut self) -> bool {
        self.reapers.retain(|reaper| !reaper.is_finished());
        self.reapers.is_empty()
    }

    /// Takes the reapers still waiting, for an owner that must keep waiting
    /// after this host is gone.
    pub(super) fn take_pending(&mut self) -> Vec<JoinHandle<()>> {
        self.clear();
        std::mem::take(&mut self.reapers)
    }

    /// Kills `child` and waits up to `grace` for its exit. Past that, an
    /// independent task keeps waiting and reaps it; `lock_ref` (the inherited
    /// Neural Engine lane lock, if any) stays alive inside that task until the
    /// exit is confirmed, so an unconfirmed exit never releases it.
    pub(super) async fn kill_and_confirm(
        &mut self,
        mut child: Child,
        grace: Option<Duration>,
        lock_ref: Option<Arc<std::fs::File>>,
    ) {
        // An already-exited child refuses the signal; the wait below still
        // collects its status.
        let _ = child.start_kill();
        #[cfg(test)]
        let hold = self.test_hold.clone();
        let wait = async move {
            #[cfg(test)]
            if let Some(hold) = hold {
                let _ = hold.acquire().await;
            }
            // A wait error means the process was already reaped elsewhere.
            let _ = child.wait().await;
        };
        let mut wait = Box::pin(wait);
        let grace = grace.unwrap_or_else(|| self.grace());
        if tokio::time::timeout(grace, &mut wait).await.is_ok() {
            return;
        }
        tracing::warn!(
            target: "synapse.worker",
            grace_ms = grace.as_millis() as u64,
            "killed worker process has not exited; reaping it in the background and starting no replacement until it does"
        );
        self.reapers.push(tokio::spawn(async move {
            wait.await;
            drop(lock_ref);
        }));
    }
}

impl WorkerHost {
    /// Refuses work while a killed worker's exit is unconfirmed. The refusal
    /// uses the crash error class, so callers get a transient, retryable
    /// `engine_crashed` with the usual retry delay.
    pub(super) fn exit_fence_check(&mut self) -> Result<(), WorkerHostError> {
        if self.exit_fence.clear() {
            return Ok(());
        }
        Err(WorkerHostError::EngineCrashed {
            stage: "restart".into(),
            detail: "previous worker process exit is not yet confirmed; no replacement starts until it is".into(),
            stderr_tail: String::new(),
        })
    }

    /// Kills and reaps `child`, bounded as [`ExitFence::kill_and_confirm`]
    /// describes.
    pub(super) async fn reap_child(&mut self, child: Child, grace: Option<Duration>) {
        let lock_ref = self.config.inherited_lane_lock.clone();
        self.exit_fence
            .kill_and_confirm(child, grace, lock_ref)
            .await;
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::super::test_support::progress::live_host_with_script;
    use super::super::*;

    fn batch() -> TokenBatch {
        TokenBatch {
            items: vec![vec![7]],
        }
    }

    /// A worker that stalls past the no-progress window and then does not exit
    /// when killed must not hold the request: the exchange fails as a crash,
    /// later requests are refused without spawning a replacement, and the
    /// fence lifts once the old process is confirmed gone.
    #[tokio::test]
    async fn unconfirmed_exit_fails_the_exchange_and_fences_replacement() {
        let (host, model, peer) = live_host_with_script(1, Duration::ZERO, true).await;
        let mut host = host.lock().await;
        host.test_idle_window = Some(Duration::from_millis(100));
        host.config.handshake_timeout = Duration::from_millis(100);
        host.config.worker_id = format!("exit-fence-{}", nonce_hex16());
        let hold = Arc::new(tokio::sync::Semaphore::new(0));
        host.exit_fence.test_hold = Some(hold.clone());
        host.exit_fence.test_grace = Some(Duration::from_millis(50));

        let error =
            tokio::time::timeout(Duration::from_secs(10), host.embed_batch(&model, batch()))
                .await
                .expect("an unconfirmed exit must not hold the exchange")
                .unwrap_err();
        assert!(
            matches!(error, WorkerHostError::EngineCrashed { .. }),
            "{error:?}"
        );
        assert!(host.connection.is_none());
        assert_eq!(host.health_snapshot().crash_count_window, 1);

        // Refused as a transient crash, without charging another strike and
        // without spawning a replacement (the configured binary is /bin/false,
        // so a spawn would fail its handshake with a different error).
        let refused = host.embed_batch(&model, batch()).await.unwrap_err();
        assert!(
            matches!(&refused, WorkerHostError::EngineCrashed { detail, .. } if detail.contains("not yet confirmed")),
            "{refused:?}"
        );
        assert!(
            refused
                .to_engine_error(EngineErrorStage::Inference)
                .safe_to_retry_same_request
        );
        assert_eq!(host.health_snapshot().crash_count_window, 1);
        assert!(host.connection.is_none());

        hold.close();
        tokio::time::timeout(Duration::from_secs(10), async {
            while !host.exit_fence.clear() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("the reaper confirms the exit once the process is gone");
        let after = host.embed_batch(&model, batch()).await.unwrap_err();
        assert!(
            !matches!(&after, WorkerHostError::EngineCrashed { detail, .. } if detail.contains("not yet confirmed")),
            "the fence must lift after a confirmed exit: {after:?}"
        );
        peer.await.unwrap();
    }

    /// Pause must not release the inherited Neural Engine lock while the
    /// worker's exit is unconfirmed, and must succeed once it is confirmed.
    #[test]
    fn pause_keeps_the_inherited_lock_until_exit_is_confirmed() {
        let engine =
            WorkerEngine::new(WorkerHostConfig::new("/bin/false", std::env::temp_dir())).unwrap();
        engine.attach_mock_worker_for_test(|_, _| unreachable!("no request is sent"));
        let lock = Arc::new(tempfile_lock());
        let hold = Arc::new(tokio::sync::Semaphore::new(0));
        {
            let mut host = engine.lock_host().unwrap();
            host.config.inherited_lane_lock = Some(lock.clone());
            host.exit_fence.test_hold = Some(hold.clone());
            host.exit_fence.test_grace = Some(Duration::from_millis(50));
        }
        assert!(engine.shutdown_for_pause().is_err());
        assert!(engine
            .lock_host()
            .unwrap()
            .config
            .inherited_lane_lock
            .is_some());
        // The reaper holds its own reference until the exit is confirmed.
        assert_eq!(Arc::strong_count(&lock), 3);

        hold.close();
        let started = std::time::Instant::now();
        while engine.shutdown_for_pause().is_err() {
            assert!(started.elapsed() < Duration::from_secs(10));
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(engine
            .lock_host()
            .unwrap()
            .config
            .inherited_lane_lock
            .is_none());
        assert_eq!(Arc::strong_count(&lock), 1);
    }

    /// Dropping an engine outside a Tokio context waits at most its teardown
    /// budget, and the lane lock outlives the drop until the exit is confirmed.
    #[test]
    fn drop_outside_tokio_is_bounded_and_keeps_the_lock_until_exit() {
        let mut engine =
            WorkerEngine::new(WorkerHostConfig::new("/bin/false", std::env::temp_dir())).unwrap();
        engine.attach_mock_worker_for_test(|_, _| unreachable!("no request is sent"));
        engine.teardown_budget = Duration::from_millis(50);
        let lock = Arc::new(tempfile_lock());
        let hold = Arc::new(tokio::sync::Semaphore::new(0));
        {
            let mut host = engine.lock_host().unwrap();
            host.config.inherited_lane_lock = Some(lock.clone());
            host.exit_fence.test_hold = Some(hold.clone());
        }
        let (dropped_tx, dropped) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            drop(engine);
            let _ = dropped_tx.send(());
        });
        dropped
            .recv_timeout(Duration::from_secs(10))
            .expect("drop must not wait for an unconfirmed exit");
        assert!(Arc::strong_count(&lock) > 1, "lock released before exit");

        hold.close();
        let started = std::time::Instant::now();
        while Arc::strong_count(&lock) > 1 {
            assert!(started.elapsed() < Duration::from_secs(10));
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn tempfile_lock() -> std::fs::File {
        let path = std::env::temp_dir().join(format!("synapse-exit-fence-{}", nonce_hex16()));
        let file = std::fs::File::create(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        file
    }
}
