//! Fences around durable `embed.batch` job execution.
//!
//! Result pages and item checkpoints belong to a request digest, not to one
//! job, so every job admitted for the same digest (for example under two
//! request keys) reads and appends one shared page sequence. These helpers
//! keep that sequence single-writer and single-fingerprint, and give the job
//! one execution deadline taken from its admitted record.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use synapse_core::StableError;
use synapse_store::{JobAdmission, JobRecord, SynapseStore, SynapseStoreError};

use crate::WireOperationError;

/// The tokio deadline matching the record's wall-clock execution TTL, if any.
pub(crate) fn execution_deadline(record: &JobRecord) -> Option<tokio::time::Instant> {
    let expires_ms = record.execution_expires_ms?;
    let remaining = expires_ms.saturating_sub(crate::now_ms());
    Some(tokio::time::Instant::now() + Duration::from_millis(remaining))
}

/// Reported when a job's attempt finishes after the job had already ended,
/// normally because its execution TTL passed. The store keeps the recorded
/// outcome; this only closes the attempt's own accounting.
pub(crate) fn ended_error() -> WireOperationError {
    WireOperationError::from_stable(
        StableError::deadline_exceeded(),
        "durable job ended before its attempt finished",
    )
}

/// Refusal for retained pages computed under another fingerprint. The request
/// key and content are unchanged, but the work they named can no longer be
/// attached to or resumed by the model now serving that id.
pub(crate) fn fingerprint_conflict(retained: &str, current: &str) -> WireOperationError {
    WireOperationError::from_stable(
        StableError::idempotency_conflict(),
        format!(
            "retained results for this request were computed under fingerprint '{retained}', \
             but the model now resolves to '{current}'; resubmit with a new request_key"
        ),
    )
}

/// Checks the fingerprint recorded in the job's first retained page, if it has
/// one. Pages without a fingerprint field give no evidence and pass.
pub(crate) fn check_retained_fingerprint(
    store: &SynapseStore,
    record: &JobRecord,
    current: &str,
) -> Result<(), WireOperationError> {
    if record.page_count == 0 {
        return Ok(());
    }
    let page = store.get_job_page(&record.job_id, 0).map_err(store_error)?;
    let retained = page
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|page| page["fingerprint"].as_str().map(str::to_owned));
    match retained {
        Some(retained) if retained != current => Err(fingerprint_conflict(&retained, current)),
        _ => Ok(()),
    }
}

/// Refuses to attach to or resume pages retained under another fingerprint.
/// A job minted for that refusal ends at once with the same error.
pub(crate) fn refuse_foreign_retained_pages(
    state: &crate::ModuleState,
    admission: &JobAdmission,
    fingerprint: &str,
) -> Result<(), WireOperationError> {
    let record = admission.record();
    check_retained_fingerprint(&state.store, record, fingerprint).inspect_err(|error| {
        if matches!(admission, JobAdmission::Admitted(_)) {
            // Count the job as minted too, so the open-job count stays balanced.
            state.runtime.admission_telemetry.record_job_minted();
            let transient = error.class == synapse_core::ErrorClass::Transient;
            crate::fail_job_with_wire_error(state, &record.job_id, transient, error.clone());
        }
    })
}

fn store_error(error: SynapseStoreError) -> WireOperationError {
    WireOperationError::from_stable(
        StableError::engine_crashed(Some(100)),
        format!("read retained job page: {error}"),
    )
}

/// One writer at a time per request digest. Entries exist only while a job
/// holds or awaits one, so the map stays as small as the active digests.
#[derive(Default)]
pub(crate) struct DigestWriters(Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>);

/// Exclusive right to append to one digest's page sequence.
pub(crate) struct DigestWriter<'a> {
    writers: &'a DigestWriters,
    digest: String,
    guard: Option<tokio::sync::OwnedMutexGuard<()>>,
}

impl DigestWriters {
    pub(crate) async fn acquire(&self, digest: &str) -> DigestWriter<'_> {
        let lock = Arc::clone(
            self.0
                .lock()
                .expect("digest writer registry")
                .entry(digest.to_owned())
                .or_default(),
        );
        let guard = lock.lock_owned().await;
        DigestWriter {
            writers: self,
            digest: digest.to_owned(),
            guard: Some(guard),
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.0.lock().unwrap().len()
    }
}

impl Drop for DigestWriter<'_> {
    fn drop(&mut self) {
        // Release first, so only the registry's own reference can remain
        // when nobody else holds or awaits this digest.
        drop(self.guard.take());
        let mut writers = self.writers.0.lock().expect("digest writer registry");
        if writers
            .get(&self.digest)
            .is_some_and(|lock| Arc::strong_count(lock) == 1)
        {
            writers.remove(&self.digest);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_second_writer_waits_and_the_registry_empties_after_both() {
        let writers = DigestWriters::default();
        let first = writers.acquire("digest").await;
        let second = writers.acquire("digest");
        tokio::pin!(second);
        assert!(
            poll_once(second.as_mut()).await.is_none(),
            "two writers held one digest"
        );
        drop(first);
        let second = second.await;
        assert_eq!(writers.len(), 1);
        drop(second);
        assert_eq!(writers.len(), 0);
    }

    /// Pages retained under one fingerprint must not be attached to or
    /// resumed by a model that now resolves to another.
    #[test]
    fn retained_pages_from_another_fingerprint_are_refused() {
        let (root, descriptor) = crate::tests::test_storage_descriptor("retained-fingerprint");
        let store = SynapseStore::open(&descriptor).unwrap();
        let generation = store.next_module_generation().unwrap();
        let now = crate::now_ms();
        let admission = store
            .admit_job(
                "key",
                "digest",
                "embed.batch",
                generation,
                &serde_json::json!({}),
                now,
                60_000,
                60_000,
            )
            .unwrap();
        let job_id = admission.record().job_id.clone();
        assert!(store.mark_job_running(&job_id, generation, now).unwrap());
        let item = synapse_store::CheckpointItem {
            item_id: "a".into(),
            result: vec![1],
        };
        store
            .commit_job_page(&job_id, 0, br#"{"fingerprint":"old"}"#, &[item], now)
            .unwrap();
        let record = store.get_job(&job_id).unwrap().unwrap();
        let refused = check_retained_fingerprint(&store, &record, "new");
        let same = check_retained_fingerprint(&store, &record, "old");
        drop(store);
        let _ = std::fs::remove_dir_all(root);
        assert_eq!(refused.unwrap_err().code, "idempotency_conflict");
        assert!(same.is_ok());
    }

    async fn poll_once<F: std::future::Future + Unpin>(future: F) -> Option<F::Output> {
        let mut future = future;
        std::future::poll_fn(|cx| {
            std::task::Poll::Ready(match std::pin::Pin::new(&mut future).poll(cx) {
                std::task::Poll::Ready(value) => Some(value),
                std::task::Poll::Pending => None,
            })
        })
        .await
    }
}
