use std::sync::atomic::{AtomicU64, Ordering};

use cortexkit_store::{open_sqlite, Migration, SqliteStore, StoreError};
use cortexkit_store_types::StorageDescriptor;
use rusqlite::{params, OptionalExtension, Row};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use synapse_core::{AliasRow, Fingerprint};
use thiserror::Error;

const NAMESPACE: &str = "synapse_module";
pub const JOB_STATE_QUEUED: &str = "queued";
pub const JOB_STATE_RUNNING: &str = "running";
pub const JOB_STATE_DONE: &str = "done";
pub const JOB_STATE_FAILED_TRANSIENT: &str = "failed_transient";
pub const JOB_STATE_FAILED_PERMANENT: &str = "failed_permanent";

const MIGRATIONS: &[Migration] = &[
    Migration {
        version: 1,
        statements: "CREATE TABLE module_meta (\
                     id INTEGER PRIMARY KEY CHECK (id = 0),\
                     module_generation INTEGER NOT NULL DEFAULT 0,\
                     table_epoch INTEGER NOT NULL DEFAULT 0\
                 );\
                 INSERT OR IGNORE INTO module_meta (id, module_generation, table_epoch) VALUES (0, 0, 0);\
                 CREATE TABLE jobs (\
                     job_id TEXT PRIMARY KEY,\
                     request_key TEXT NOT NULL UNIQUE,\
                     module_generation INTEGER NOT NULL,\
                     state TEXT NOT NULL,\
                     created_ms INTEGER NOT NULL\
                 );\
                 CREATE TABLE alias_rows (\
                     left_fingerprint TEXT NOT NULL,\
                     right_fingerprint TEXT NOT NULL,\
                     valid_from_epoch INTEGER NOT NULL,\
                     valid_to_epoch_exclusive INTEGER,\
                     PRIMARY KEY (left_fingerprint, right_fingerprint, valid_from_epoch)\
                 );\
                 CREATE TABLE cert_rows (\
                     fingerprint TEXT NOT NULL,\
                     machine_profile_hash TEXT NOT NULL,\
                     certified_shape_json TEXT NOT NULL,\
                     created_ms INTEGER NOT NULL,\
                     PRIMARY KEY (fingerprint, machine_profile_hash)\
                 );",
    },
    Migration {
        version: 2,
        statements: r#"
                 ALTER TABLE jobs RENAME TO jobs_v1;
                 CREATE TABLE jobs (
                     job_id TEXT PRIMARY KEY,
                     request_key TEXT NOT NULL UNIQUE,
                     kind TEXT NOT NULL,
                     module_generation INTEGER NOT NULL,
                     state TEXT NOT NULL,
                     created_ms INTEGER NOT NULL,
                     updated_ms INTEGER NOT NULL,
                     expires_ms INTEGER NOT NULL,
                     params_json BLOB NOT NULL,
                     result_json BLOB,
                     error_json BLOB,
                     page_count INTEGER NOT NULL DEFAULT 0
                 );
                 INSERT INTO jobs (
                     job_id, request_key, kind, module_generation, state, created_ms,
                     updated_ms, expires_ms, params_json, result_json, error_json, page_count
                 )
                 SELECT job_id, request_key, 'unknown', module_generation, state, created_ms,
                        created_ms, created_ms + 86400000, '{}', NULL, NULL, 0
                 FROM jobs_v1;
                 DROP TABLE jobs_v1;
                 CREATE TABLE job_pages (
                     job_id TEXT NOT NULL,
                     page_index INTEGER NOT NULL,
                     page_json BLOB NOT NULL,
                     PRIMARY KEY (job_id, page_index),
                     FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
                 );
                 CREATE INDEX jobs_state_generation_idx ON jobs(state, module_generation);
                 CREATE INDEX jobs_expires_idx ON jobs(expires_ms);
        "#,
    },
];

static JOB_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Error)]
pub enum SynapseStoreError {
    #[error("synapse store: {0}")]
    Store(#[from] StoreError),
    #[error("synapse store json: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ModelCatalogEntry {
    pub model_id: String,
    pub state: String,
    pub fingerprints: Vec<Fingerprint>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CatalogSnapshot {
    pub table_epoch: u64,
    pub models: Vec<ModelCatalogEntry>,
    pub alias_rows: Vec<AliasRow>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct JobRecord {
    pub job_id: String,
    pub request_key: String,
    pub kind: String,
    pub module_generation: u64,
    pub state: String,
    pub created_ms: u64,
    pub updated_ms: u64,
    pub expires_ms: u64,
    pub page_count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_json: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_json: Option<Value>,
}

impl JobRecord {
    #[must_use]
    pub fn is_terminal_failed(&self) -> bool {
        self.state == JOB_STATE_FAILED_TRANSIENT || self.state == JOB_STATE_FAILED_PERMANENT
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JobAdmission {
    Existing(JobRecord),
    Admitted(JobRecord),
}

impl JobAdmission {
    #[must_use]
    pub fn record(&self) -> &JobRecord {
        match self {
            JobAdmission::Existing(record) | JobAdmission::Admitted(record) => record,
        }
    }
}

pub struct SynapseStore {
    store: SqliteStore,
}

impl SynapseStore {
    pub fn open(descriptor: &StorageDescriptor) -> Result<Self, SynapseStoreError> {
        let store = open_sqlite(descriptor)?;
        store.migrate(NAMESPACE, MIGRATIONS)?;
        Ok(Self { store })
    }

    pub fn next_module_generation(&self) -> Result<u64, SynapseStoreError> {
        let next = self.store.with_conn_fenced(|tx| {
            let current: i64 = tx.query_row(
                "SELECT module_generation FROM module_meta WHERE id = 0",
                [],
                |row| row.get(0),
            )?;
            let next = current + 1;
            tx.execute(
                "UPDATE module_meta SET module_generation = ?1 WHERE id = 0",
                params![next],
            )?;
            Ok(next as u64)
        })?;
        Ok(next)
    }

    pub fn catalog_snapshot(&self) -> Result<CatalogSnapshot, SynapseStoreError> {
        let snapshot = self.store.with_conn(|conn| {
            let table_epoch: i64 = conn.query_row(
                "SELECT table_epoch FROM module_meta WHERE id = 0",
                [],
                |row| row.get(0),
            )?;
            let mut stmt = conn.prepare(
                "SELECT left_fingerprint, right_fingerprint, valid_from_epoch, valid_to_epoch_exclusive \
                 FROM alias_rows \
                 ORDER BY left_fingerprint, right_fingerprint, valid_from_epoch",
            )?;
            let alias_rows = stmt
                .query_map([], |row| {
                    Ok(AliasRow::new(
                        Fingerprint(row.get(0)?),
                        Fingerprint(row.get(1)?),
                        row.get::<_, i64>(2)? as u64,
                        row.get::<_, Option<i64>>(3)?.map(|value| value as u64),
                    ))
                })?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(CatalogSnapshot {
                table_epoch: table_epoch as u64,
                models: Vec::new(),
                alias_rows,
            })
        })?;
        Ok(snapshot)
    }

    pub fn admit_job(
        &self,
        request_key: &str,
        kind: &str,
        module_generation: u64,
        params_json: &Value,
        now_ms: u64,
        ttl_ms: u64,
    ) -> Result<JobAdmission, SynapseStoreError> {
        let params_bytes = serde_json::to_vec(params_json)?;
        let request_key = request_key.to_string();
        let kind = kind.to_string();
        let admission = self.store.with_conn_fenced(|tx| {
            purge_expired_jobs_tx(tx, now_ms as i64)?;
            if let Some(existing) = job_by_request_key_tx(tx, &request_key)? {
                if !existing.is_terminal_failed() {
                    return Ok(JobAdmission::Existing(existing));
                }
                let new_job_id = new_job_id(&request_key, module_generation, now_ms);
                tx.execute(
                    "DELETE FROM job_pages WHERE job_id = ?1",
                    params![existing.job_id],
                )?;
                tx.execute(
                    "UPDATE jobs SET job_id = ?1, kind = ?2, module_generation = ?3, \
                         state = ?4, created_ms = ?5, updated_ms = ?5, expires_ms = ?6, \
                         params_json = ?7, result_json = NULL, error_json = NULL, page_count = 0 \
                     WHERE request_key = ?8",
                    params![
                        new_job_id,
                        kind,
                        module_generation as i64,
                        JOB_STATE_QUEUED,
                        now_ms as i64,
                        now_ms.saturating_add(ttl_ms) as i64,
                        params_bytes,
                        request_key
                    ],
                )?;
                return Ok(JobAdmission::Admitted(
                    job_by_id_tx(tx, &new_job_id)?.expect("updated job is readable"),
                ));
            }

            let job_id = new_job_id(&request_key, module_generation, now_ms);
            tx.execute(
                "INSERT INTO jobs (\
                     job_id, request_key, kind, module_generation, state, created_ms, updated_ms,\
                     expires_ms, params_json, result_json, error_json, page_count\
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6, ?7, ?8, NULL, NULL, 0)",
                params![
                    job_id,
                    request_key,
                    kind,
                    module_generation as i64,
                    JOB_STATE_QUEUED,
                    now_ms as i64,
                    now_ms.saturating_add(ttl_ms) as i64,
                    params_bytes
                ],
            )?;
            Ok(JobAdmission::Admitted(
                job_by_id_tx(tx, &job_id)?.expect("inserted job is readable"),
            ))
        })?;
        Ok(admission)
    }

    pub fn mark_job_running(
        &self,
        job_id: &str,
        module_generation: u64,
        now_ms: u64,
    ) -> Result<bool, SynapseStoreError> {
        let changed = self.store.with_conn_fenced(|tx| {
            tx.execute(
                "UPDATE jobs SET state = ?1, updated_ms = ?2 \
                 WHERE job_id = ?3 AND module_generation = ?4 AND state = ?5",
                params![
                    JOB_STATE_RUNNING,
                    now_ms as i64,
                    job_id,
                    module_generation as i64,
                    JOB_STATE_QUEUED
                ],
            )
        })?;
        Ok(changed > 0)
    }

    pub fn complete_job(
        &self,
        job_id: &str,
        result_json: &Value,
        pages: &[Vec<u8>],
        now_ms: u64,
    ) -> Result<(), SynapseStoreError> {
        let result_bytes = serde_json::to_vec(result_json)?;
        self.store.with_conn_fenced(|tx| {
            tx.execute("DELETE FROM job_pages WHERE job_id = ?1", params![job_id])?;
            for (index, page) in pages.iter().enumerate() {
                tx.execute(
                    "INSERT INTO job_pages (job_id, page_index, page_json) VALUES (?1, ?2, ?3)",
                    params![job_id, index as i64, page],
                )?;
            }
            tx.execute(
                "UPDATE jobs SET state = ?1, updated_ms = ?2, result_json = ?3, \
                     error_json = NULL, page_count = ?4 \
                 WHERE job_id = ?5",
                params![
                    JOB_STATE_DONE,
                    now_ms as i64,
                    result_bytes,
                    pages.len() as i64,
                    job_id
                ],
            )?;
            Ok(())
        })?;
        Ok(())
    }

    pub fn fail_job(
        &self,
        job_id: &str,
        transient: bool,
        error_json: &Value,
        now_ms: u64,
    ) -> Result<(), SynapseStoreError> {
        let error_bytes = serde_json::to_vec(error_json)?;
        let state = if transient {
            JOB_STATE_FAILED_TRANSIENT
        } else {
            JOB_STATE_FAILED_PERMANENT
        };
        self.store.with_conn_fenced(|tx| {
            tx.execute("DELETE FROM job_pages WHERE job_id = ?1", params![job_id])?;
            tx.execute(
                "UPDATE jobs SET state = ?1, updated_ms = ?2, error_json = ?3, \
                     result_json = NULL, page_count = 0 \
                 WHERE job_id = ?4",
                params![state, now_ms as i64, error_bytes, job_id],
            )?;
            Ok(())
        })?;
        Ok(())
    }

    pub fn fail_prior_generation_incomplete_jobs(
        &self,
        current_generation: u64,
        error_json: &Value,
        now_ms: u64,
    ) -> Result<usize, SynapseStoreError> {
        let error_bytes = serde_json::to_vec(error_json)?;
        let changed = self.store.with_conn_fenced(|tx| {
            tx.execute(
                "UPDATE jobs SET state = ?1, updated_ms = ?2, error_json = ?3, \
                     result_json = NULL, page_count = 0 \
                 WHERE module_generation < ?4 AND state IN (?5, ?6)",
                params![
                    JOB_STATE_FAILED_TRANSIENT,
                    now_ms as i64,
                    error_bytes,
                    current_generation as i64,
                    JOB_STATE_QUEUED,
                    JOB_STATE_RUNNING
                ],
            )
        })?;
        Ok(changed)
    }

    pub fn purge_expired_jobs(&self, now_ms: u64) -> Result<usize, SynapseStoreError> {
        let purged = self
            .store
            .with_conn_fenced(|tx| purge_expired_jobs_tx(tx, now_ms as i64))?;
        Ok(purged)
    }

    pub fn get_job(&self, job_id: &str) -> Result<Option<JobRecord>, SynapseStoreError> {
        let record = self.store.with_conn(|conn| job_by_id_conn(conn, job_id))?;
        Ok(record)
    }

    pub fn get_job_page(
        &self,
        job_id: &str,
        page_index: u32,
    ) -> Result<Option<Vec<u8>>, SynapseStoreError> {
        let page = self.store.with_conn(|conn| {
            conn.query_row(
                "SELECT page_json FROM job_pages WHERE job_id = ?1 AND page_index = ?2",
                params![job_id, page_index as i64],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .optional()
        })?;
        Ok(page)
    }
}

fn purge_expired_jobs_tx(tx: &rusqlite::Transaction<'_>, now_ms: i64) -> rusqlite::Result<usize> {
    tx.execute(
        "DELETE FROM jobs WHERE expires_ms <= ?1 AND state IN (?2, ?3, ?4)",
        params![
            now_ms,
            JOB_STATE_DONE,
            JOB_STATE_FAILED_TRANSIENT,
            JOB_STATE_FAILED_PERMANENT
        ],
    )
}

fn job_by_request_key_tx(
    tx: &rusqlite::Transaction<'_>,
    request_key: &str,
) -> rusqlite::Result<Option<JobRecord>> {
    let sql = format!("{JOB_SELECT_SQL} WHERE request_key = ?1");
    tx.query_row(&sql, params![request_key], row_to_job)
        .optional()
}

fn job_by_id_tx(
    tx: &rusqlite::Transaction<'_>,
    job_id: &str,
) -> rusqlite::Result<Option<JobRecord>> {
    let sql = format!("{JOB_SELECT_SQL} WHERE job_id = ?1");
    tx.query_row(&sql, params![job_id], row_to_job).optional()
}

fn job_by_id_conn(
    conn: &rusqlite::Connection,
    job_id: &str,
) -> rusqlite::Result<Option<JobRecord>> {
    let sql = format!("{JOB_SELECT_SQL} WHERE job_id = ?1");
    conn.query_row(&sql, params![job_id], row_to_job).optional()
}

const JOB_SELECT_SQL: &str = "SELECT job_id, request_key, kind, module_generation, state,\
        created_ms, updated_ms, expires_ms, page_count, result_json, error_json FROM jobs";

fn row_to_job(row: &Row<'_>) -> rusqlite::Result<JobRecord> {
    Ok(JobRecord {
        job_id: row.get(0)?,
        request_key: row.get(1)?,
        kind: row.get(2)?,
        module_generation: row.get::<_, i64>(3)? as u64,
        state: row.get(4)?,
        created_ms: row.get::<_, i64>(5)? as u64,
        updated_ms: row.get::<_, i64>(6)? as u64,
        expires_ms: row.get::<_, i64>(7)? as u64,
        page_count: row.get::<_, i64>(8)? as u32,
        result_json: decode_optional_json(row.get(9)?, 9)?,
        error_json: decode_optional_json(row.get(10)?, 10)?,
    })
}

fn decode_optional_json(bytes: Option<Vec<u8>>, index: usize) -> rusqlite::Result<Option<Value>> {
    bytes
        .map(|bytes| {
            serde_json::from_slice(&bytes).map_err(|error| {
                rusqlite::Error::FromSqlConversionFailure(
                    index,
                    rusqlite::types::Type::Blob,
                    Box::new(error),
                )
            })
        })
        .transpose()
}

fn new_job_id(request_key: &str, generation: u64, now_ms: u64) -> String {
    let counter = JOB_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut hasher = Sha256::new();
    hasher.update(request_key.as_bytes());
    hasher.update(generation.to_le_bytes());
    hasher.update(now_ms.to_le_bytes());
    hasher.update(counter.to_le_bytes());
    hasher.update(std::process::id().to_le_bytes());
    format!("job_{}", hex::encode(&hasher.finalize()[..16]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cortexkit_store_types::{Isolation, StorageBackend};

    #[test]
    fn restart_marks_prior_generation_queued_and_running_jobs_failed_transient() {
        let (root, descriptor) = temp_descriptor("restart-jobs");
        let (queued_job, running_job) = {
            let store = SynapseStore::open(&descriptor).unwrap();
            let generation = store.next_module_generation().unwrap();
            let queued = store
                .admit_job(
                    "queued-key",
                    "embed.batch",
                    generation,
                    &serde_json::json!({"items": 2}),
                    10,
                    1_000,
                )
                .unwrap()
                .record()
                .job_id
                .clone();
            let running = store
                .admit_job(
                    "running-key",
                    "embed.batch",
                    generation,
                    &serde_json::json!({"items": 3}),
                    11,
                    1_000,
                )
                .unwrap()
                .record()
                .job_id
                .clone();
            assert!(store.mark_job_running(&running, generation, 12).unwrap());
            (queued, running)
        };

        let store = SynapseStore::open(&descriptor).unwrap();
        let generation = store.next_module_generation().unwrap();
        assert_eq!(generation, 2);
        let failed = store
            .fail_prior_generation_incomplete_jobs(
                generation,
                &serde_json::json!({"code": "module_restarted"}),
                20,
            )
            .unwrap();
        assert_eq!(failed, 2);
        for job_id in [queued_job, running_job] {
            let job = store
                .get_job(&job_id)
                .unwrap()
                .expect("job survives restart");
            assert_eq!(job.state, JOB_STATE_FAILED_TRANSIENT);
            assert_eq!(job.error_json.unwrap()["code"], "module_restarted");
        }
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn request_key_is_idempotent_until_terminal_failure_then_admits_fresh() {
        let (root, descriptor) = temp_descriptor("request-key");
        let store = SynapseStore::open(&descriptor).unwrap();
        let generation = store.next_module_generation().unwrap();
        let first = store
            .admit_job(
                "same-key",
                "embed.batch",
                generation,
                &serde_json::json!({"batch": 1}),
                100,
                1_000,
            )
            .unwrap();
        assert!(matches!(first, JobAdmission::Admitted(_)));
        let first_id = first.record().job_id.clone();
        let second = store
            .admit_job(
                "same-key",
                "embed.batch",
                generation,
                &serde_json::json!({"batch": 1}),
                101,
                1_000,
            )
            .unwrap();
        assert!(matches!(second, JobAdmission::Existing(_)));
        assert_eq!(second.record().job_id, first_id);

        store
            .fail_job(
                &first_id,
                true,
                &serde_json::json!({"code": "module_restarted"}),
                102,
            )
            .unwrap();
        let fresh = store
            .admit_job(
                "same-key",
                "embed.batch",
                generation,
                &serde_json::json!({"batch": 1}),
                103,
                1_000,
            )
            .unwrap();
        assert!(matches!(fresh, JobAdmission::Admitted(_)));
        assert_ne!(fresh.record().job_id, first_id);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn result_pages_survive_reopen_until_ttl_expires() {
        let (root, descriptor) = temp_descriptor("result-pages");
        let job_id = {
            let store = SynapseStore::open(&descriptor).unwrap();
            let generation = store.next_module_generation().unwrap();
            let job_id = store
                .admit_job(
                    "page-key",
                    "embed.batch",
                    generation,
                    &serde_json::json!({"batch": 2}),
                    1_000,
                    100,
                )
                .unwrap()
                .record()
                .job_id
                .clone();
            store
                .complete_job(
                    &job_id,
                    &serde_json::json!({"page_count": 2}),
                    &[
                        serde_json::to_vec(&serde_json::json!({"page": 0})).unwrap(),
                        serde_json::to_vec(&serde_json::json!({"page": 1})).unwrap(),
                    ],
                    1_010,
                )
                .unwrap();
            job_id
        };

        let store = SynapseStore::open(&descriptor).unwrap();
        let page = store
            .get_job_page(&job_id, 1)
            .unwrap()
            .expect("page survives");
        let value: Value = serde_json::from_slice(&page).unwrap();
        assert_eq!(value["page"], 1);
        assert_eq!(store.purge_expired_jobs(1_099).unwrap(), 0);
        assert!(store.get_job(&job_id).unwrap().is_some());
        assert_eq!(store.purge_expired_jobs(1_100).unwrap(), 1);
        assert!(store.get_job(&job_id).unwrap().is_none());
        let _ = std::fs::remove_dir_all(root);
    }

    fn temp_descriptor(label: &str) -> (std::path::PathBuf, StorageDescriptor) {
        let root = std::env::temp_dir().join(format!(
            "synapse-store-{label}-{}-{}",
            std::process::id(),
            JOB_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let descriptor = StorageDescriptor {
            module_id: "synapse-test".to_string(),
            storage_namespace: "default".to_string(),
            isolation: Isolation::Module,
            backend: StorageBackend::Sqlite {
                path: root.join("store.db").to_string_lossy().to_string(),
            },
        };
        (root, descriptor)
    }
}
