//! Worker lane health for module health reports, read without waiting.
//!
//! A worker lane's engine mutex is held for the whole of a request, which can
//! last minutes (a slow embedding, a cold reload, or reaping a stalled
//! worker). Health must still answer promptly and report every lane, so it
//! never waits for that mutex: when the engine is busy it reports the last
//! snapshot it managed to read.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, Weak};

use synapse_worker_host::worker_host::{WorkerEngine, WorkerHostHealth};

type SharedEngine = Arc<Mutex<WorkerEngine>>;
type Snapshots = HashMap<usize, (Weak<Mutex<WorkerEngine>>, WorkerHostHealth)>;

/// Last snapshot read from each engine, keyed by the engine's allocation. The
/// stored weak reference keeps that allocation reserved, so the key cannot be
/// reused by a different engine while its entry exists; entries of dropped
/// engines are pruned on every successful read.
static LAST_SNAPSHOT: LazyLock<Mutex<Snapshots>> = LazyLock::new(Default::default);

/// The engine's current health, or the last one read while it is busy.
/// `None` when no snapshot was ever read or its mutex is poisoned.
pub(crate) fn worker_health(engine: &SharedEngine) -> Option<WorkerHostHealth> {
    let key = Arc::as_ptr(engine) as usize;
    let fresh = match engine.try_lock() {
        Ok(engine) => engine.try_health_snapshot(),
        Err(std::sync::TryLockError::WouldBlock) => None,
        Err(std::sync::TryLockError::Poisoned(_)) => return None,
    };
    let mut cache = LAST_SNAPSHOT
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    match fresh {
        Some(health) => {
            cache.retain(|_, (weak, _)| weak.strong_count() > 0);
            cache.insert(key, (Arc::downgrade(engine), health.clone()));
            Some(health)
        }
        None => cache.get(&key).map(|(_, health)| health.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use synapse_worker_host::worker_host::WorkerHostConfig;

    #[test]
    fn busy_engine_reports_its_last_snapshot_without_waiting() {
        let engine: SharedEngine = Arc::new(Mutex::new(
            WorkerEngine::new(WorkerHostConfig::new("unused-worker", std::env::temp_dir()))
                .unwrap(),
        ));
        let first = worker_health(&engine).expect("an idle engine reports its health");

        let held = engine.lock().unwrap();
        let (sent, received) = std::sync::mpsc::channel();
        let reader = engine.clone();
        std::thread::spawn(move || {
            let _ = sent.send(worker_health(&reader));
        });
        let busy = received
            .recv_timeout(Duration::from_secs(10))
            .expect("health must not wait for a busy engine");
        drop(held);
        assert_eq!(busy, Some(first));
    }
}
