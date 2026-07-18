use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

use cortexkit_store::{open_sqlite, Migration, SqliteStore, StoreError};
use cortexkit_store_types::StorageDescriptor;
use rusqlite::{params, OptionalExtension, Row};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use synapse_core::{AliasRow, AliasTable, EngineIdentity, Fingerprint, NumericProfileId};
use thiserror::Error;

use crate::PerfKnob;

const NAMESPACE: &str = "synapse_module";
pub const JOB_STATE_QUEUED: &str = "queued";
pub const JOB_STATE_RUNNING: &str = "running";
pub const JOB_STATE_PAUSED_NEEDS_REAUTH: &str = "paused_needs_reauth";
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
    Migration {
        version: 3,
        statements: r#"
                 ALTER TABLE alias_rows RENAME TO alias_rows_v2;
                 CREATE TABLE alias_rows (
                     fingerprint_a TEXT NOT NULL,
                     fingerprint_b TEXT NOT NULL,
                     valid_from_ms INTEGER NOT NULL,
                     valid_to_ms INTEGER,
                     evidence_json TEXT NOT NULL DEFAULT '{}',
                     PRIMARY KEY (fingerprint_a, fingerprint_b, valid_from_ms)
                 );
                 INSERT INTO alias_rows (
                     fingerprint_a, fingerprint_b, valid_from_ms, valid_to_ms, evidence_json
                 )
                 SELECT left_fingerprint, right_fingerprint, valid_from_epoch,
                        valid_to_epoch_exclusive, '{}'
                 FROM alias_rows_v2;
                 DROP TABLE alias_rows_v2;
 
                 ALTER TABLE cert_rows RENAME TO cert_rows_v1;
                 CREATE TABLE cert_rows (
                     machine_profile_hash TEXT NOT NULL,
                     numeric_profile_id TEXT NOT NULL,
                     fingerprint TEXT NOT NULL,
                     certified_at_ms INTEGER NOT NULL,
                     evidence_json TEXT NOT NULL,
                     PRIMARY KEY (machine_profile_hash, fingerprint)
                 );
                 INSERT OR IGNORE INTO cert_rows (
                     machine_profile_hash, numeric_profile_id, fingerprint, certified_at_ms, evidence_json
                 )
                 SELECT machine_profile_hash, '', fingerprint, created_ms,
                        json_object('certified_shape_json', certified_shape_json)
                 FROM cert_rows_v1;
                 DROP TABLE cert_rows_v1;
                 CREATE INDEX cert_rows_fingerprint_idx ON cert_rows(fingerprint);
         "#,
    },
    Migration {
        version: 4,
        statements: r#"
                 CREATE TABLE models (
                     model_id TEXT PRIMARY KEY,
                     engine TEXT NOT NULL,
                     task TEXT NOT NULL,
                     fingerprint TEXT NOT NULL,
                     config_json BLOB NOT NULL,
                     created_ms INTEGER NOT NULL,
                     updated_ms INTEGER NOT NULL
                 );
                 CREATE INDEX models_fingerprint_idx ON models(fingerprint);
        "#,
    },
    Migration {
        version: 5,
        statements: r#"
                 DROP INDEX IF EXISTS cert_rows_fingerprint_idx;
                 ALTER TABLE cert_rows RENAME TO cert_rows_v4;
                 CREATE TABLE cert_rows (
                     machine_profile_hash TEXT NOT NULL,
                     numeric_profile_id TEXT NOT NULL,
                     fingerprint TEXT NOT NULL,
                     certified_at_ms INTEGER NOT NULL,
                     os_build TEXT NOT NULL,
                     module_generation INTEGER NOT NULL,
                     evidence_json TEXT NOT NULL,
                     PRIMARY KEY (machine_profile_hash, fingerprint)
                 );
                 INSERT INTO cert_rows (
                     machine_profile_hash, numeric_profile_id, fingerprint, certified_at_ms,
                     os_build, module_generation, evidence_json
                 )
                 SELECT machine_profile_hash, numeric_profile_id, fingerprint, certified_at_ms,
                        '', 0, evidence_json
                 FROM cert_rows_v4;
                 DROP TABLE cert_rows_v4;
                 CREATE INDEX cert_rows_fingerprint_idx ON cert_rows(fingerprint);

                 CREATE TABLE perf_rows (
                     machine_profile_hash TEXT NOT NULL,
                     model_id TEXT NOT NULL,
                     workload TEXT NOT NULL,
                     numeric_profile_id TEXT NOT NULL,
                     fingerprint TEXT NOT NULL,
                     engine TEXT NOT NULL,
                     measured_at_ms INTEGER NOT NULL,
                     os_build TEXT NOT NULL,
                     module_generation INTEGER NOT NULL,
                     throughput_tok_s REAL NOT NULL,
                     cold_load_ms REAL NOT NULL,
                     single_item_latency_p50_ms REAL NOT NULL,
                     details_json TEXT NOT NULL,
                     PRIMARY KEY (machine_profile_hash, fingerprint)
                 );
                 CREATE INDEX perf_rows_fingerprint_idx ON perf_rows(fingerprint);
                 CREATE INDEX perf_rows_workload_idx ON perf_rows(machine_profile_hash, workload);

                 CREATE TABLE knob_assignments (
                     machine_profile_hash TEXT NOT NULL,
                     workload TEXT NOT NULL,
                     knob TEXT NOT NULL,
                     model_id TEXT NOT NULL,
                     numeric_profile_id TEXT NOT NULL,
                     fingerprint TEXT NOT NULL,
                     engine TEXT NOT NULL,
                     measured_at_ms INTEGER NOT NULL,
                     os_build TEXT NOT NULL,
                     module_generation INTEGER NOT NULL,
                     throughput_tok_s REAL NOT NULL,
                     single_item_latency_p50_ms REAL NOT NULL,
                     PRIMARY KEY (machine_profile_hash, workload, knob)
                 );
                 CREATE INDEX knob_assignments_lookup_idx
                     ON knob_assignments(machine_profile_hash, knob, workload);
        "#,
    },
    Migration {
        version: 6,
        statements: r#"
                 DROP INDEX IF EXISTS cert_rows_fingerprint_idx;
                 ALTER TABLE cert_rows RENAME TO cert_rows_v5;
                 CREATE TABLE cert_rows (
                     assurance_class TEXT NOT NULL CHECK (assurance_class IN ('measured', 'declared')),
                     key_hash TEXT NOT NULL,
                     machine_profile_hash TEXT,
                     remote_profile_hash TEXT,
                     numeric_profile_id TEXT NOT NULL,
                     fingerprint TEXT NOT NULL,
                     certified_at_ms INTEGER NOT NULL,
                     os_build TEXT NOT NULL,
                     module_generation INTEGER NOT NULL,
                     evidence_json TEXT NOT NULL,
                     PRIMARY KEY (assurance_class, key_hash, fingerprint),
                     CHECK (
                         (assurance_class = 'measured' AND machine_profile_hash = key_hash AND remote_profile_hash IS NULL)
                         OR
                         (assurance_class = 'declared' AND remote_profile_hash = key_hash AND machine_profile_hash IS NULL)
                     )
                 );
                 INSERT INTO cert_rows (
                     assurance_class, key_hash, machine_profile_hash, remote_profile_hash,
                     numeric_profile_id, fingerprint, certified_at_ms, os_build,
                     module_generation, evidence_json
                 )
                 SELECT 'measured', machine_profile_hash, machine_profile_hash, NULL,
                        numeric_profile_id, fingerprint, certified_at_ms, os_build,
                        module_generation, evidence_json
                 FROM cert_rows_v5;
                 DROP TABLE cert_rows_v5;
                 CREATE INDEX cert_rows_fingerprint_idx ON cert_rows(fingerprint);

                 DROP INDEX IF EXISTS jobs_state_generation_idx;
                 DROP INDEX IF EXISTS jobs_expires_idx;
                 ALTER TABLE jobs RENAME TO jobs_v5;
                 ALTER TABLE job_pages RENAME TO job_pages_v5;
                 CREATE TABLE jobs (
                     job_id TEXT PRIMARY KEY,
                     request_key TEXT NOT NULL,
                     request_digest TEXT NOT NULL,
                     kind TEXT NOT NULL,
                     module_generation INTEGER NOT NULL,
                     state TEXT NOT NULL,
                     created_ms INTEGER NOT NULL,
                     updated_ms INTEGER NOT NULL,
                     execution_expires_ms INTEGER,
                     result_retention_ttl_ms INTEGER NOT NULL,
                     terminal_at_ms INTEGER,
                     active_attempt_id TEXT,
                     logical_handle TEXT,
                     paused_at_ms INTEGER,
                     resume_deadline_ms INTEGER,
                     params_json BLOB NOT NULL,
                     result_json BLOB,
                     error_json BLOB,
                     page_count INTEGER NOT NULL DEFAULT 0
                 );
                 INSERT INTO jobs (
                     job_id, request_key, request_digest, kind, module_generation, state,
                     created_ms, updated_ms, execution_expires_ms, result_retention_ttl_ms,
                     terminal_at_ms, active_attempt_id, logical_handle, paused_at_ms,
                     resume_deadline_ms, params_json, result_json, error_json, page_count
                 )
                 SELECT job_id, request_key, 'legacy:' || job_id, kind, module_generation, state,
                        created_ms, updated_ms,
                        CASE WHEN state IN ('queued', 'running') THEN expires_ms ELSE NULL END,
                        CASE WHEN expires_ms > updated_ms THEN expires_ms - updated_ms ELSE 0 END,
                        CASE WHEN state IN ('done', 'failed_transient', 'failed_permanent')
                             THEN updated_ms ELSE NULL END,
                        NULL, NULL, NULL, NULL, CAST(params_json AS BLOB),
                        CAST(result_json AS BLOB), CAST(error_json AS BLOB), page_count
                 FROM jobs_v5;
                 CREATE INDEX jobs_request_key_idx ON jobs(request_key, created_ms DESC);
                 CREATE INDEX jobs_request_digest_idx ON jobs(request_digest, created_ms DESC);
                 CREATE INDEX jobs_state_generation_idx ON jobs(state, module_generation);
                 CREATE INDEX jobs_execution_expires_idx ON jobs(execution_expires_ms);
                 CREATE INDEX jobs_terminal_retention_idx ON jobs(terminal_at_ms);
                 CREATE INDEX jobs_resume_deadline_idx ON jobs(resume_deadline_ms);

                 CREATE TABLE result_pages (
                     request_digest TEXT NOT NULL,
                     page_no INTEGER NOT NULL,
                     page_json BLOB NOT NULL,
                     provider_request_ids_json BLOB,
                     committed_at INTEGER NOT NULL,
                     PRIMARY KEY (request_digest, page_no)
                 );
                 INSERT INTO result_pages (
                     request_digest, page_no, page_json, provider_request_ids_json, committed_at
                 )
                 SELECT 'legacy:' || job_id, page_index, CAST(page_json AS BLOB), NULL, 0
                 FROM job_pages_v5;
                 DROP TABLE job_pages_v5;
                 DROP TABLE jobs_v5;

                 CREATE TABLE remote_checkpoints (
                     request_digest TEXT NOT NULL,
                     item_id TEXT NOT NULL,
                     result BLOB NOT NULL,
                     page_no INTEGER NOT NULL,
                     provider_request_id TEXT,
                     committed_at INTEGER NOT NULL,
                     PRIMARY KEY (request_digest, item_id),
                     FOREIGN KEY (request_digest, page_no)
                         REFERENCES result_pages(request_digest, page_no)
                 );
                 CREATE INDEX remote_checkpoints_page_idx
                     ON remote_checkpoints(request_digest, page_no);
                 CREATE TRIGGER remote_checkpoint_immutable
                 BEFORE UPDATE ON remote_checkpoints
                 BEGIN
                     SELECT RAISE(ABORT, 'remote checkpoint is immutable');
                 END;

                 CREATE TABLE remote_url_bindings (
                     remote_profile_hash TEXT PRIMARY KEY,
                     last_base_url TEXT NOT NULL,
                     missing_since_ms INTEGER
                 );
                 CREATE TRIGGER remote_url_binding_identity_immutable
                 BEFORE UPDATE OF remote_profile_hash, last_base_url ON remote_url_bindings
                 BEGIN
                     SELECT RAISE(ABORT, 'remote URL binding is immutable');
                 END;
        "#,
    },
    Migration {
        version: 7,
        statements: r#"
                 DROP INDEX IF EXISTS cert_rows_fingerprint_idx;
                 ALTER TABLE cert_rows RENAME TO cert_rows_v6;
                 CREATE TABLE cert_rows (
                     assurance_class TEXT NOT NULL CHECK (assurance_class IN ('measured', 'declared')),
                     key_hash TEXT NOT NULL,
                     machine_profile_hash TEXT NOT NULL,
                     remote_profile_hash TEXT,
                     identity_revision TEXT,
                     numeric_profile_id TEXT NOT NULL,
                     fingerprint TEXT NOT NULL,
                     certified_at_ms INTEGER NOT NULL,
                     os_build TEXT NOT NULL,
                     module_generation INTEGER NOT NULL,
                     evidence_json TEXT NOT NULL,
                     PRIMARY KEY (assurance_class, key_hash, fingerprint),
                     CHECK (
                         (assurance_class = 'measured' AND machine_profile_hash = key_hash
                          AND remote_profile_hash IS NULL AND identity_revision IS NULL)
                         OR
                         (assurance_class = 'declared' AND remote_profile_hash IS NOT NULL
                          AND identity_revision IS NOT NULL)
                     )
                 );
                 INSERT INTO cert_rows (
                     assurance_class, key_hash, machine_profile_hash, remote_profile_hash,
                     identity_revision, numeric_profile_id, fingerprint, certified_at_ms,
                     os_build, module_generation, evidence_json
                 )
                 SELECT assurance_class, key_hash, machine_profile_hash, NULL, NULL,
                        numeric_profile_id, fingerprint, certified_at_ms, os_build,
                        module_generation, evidence_json
                 FROM cert_rows_v6
                 WHERE assurance_class = 'measured';
                 DROP TABLE cert_rows_v6;
                 CREATE INDEX cert_rows_fingerprint_idx ON cert_rows(fingerprint);
        "#,
    },
    Migration {
        version: 8,
        statements: r#"
                 ALTER TABLE cert_rows ADD COLUMN status TEXT NOT NULL DEFAULT 'certified'
                     CHECK (status IN ('certified', 'uncertified'));
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
    #[error("synapse store decode: {0}")]
    Decode(String),
    #[error("request key '{request_key}' is already bound to a different digest")]
    IdempotencyConflict {
        request_key: String,
        existing_digest: String,
        requested_digest: String,
    },
    #[error("remote profile '{remote_profile_hash}' changed base URL from '{previous_url}' to '{requested_url}'")]
    RemoteDeploymentChanged {
        remote_profile_hash: String,
        previous_url: String,
        requested_url: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ModelCatalogEntry {
    pub model_id: String,
    pub state: String,
    pub fingerprints: Vec<Fingerprint>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ModelAssetLocator {
    LocalPath { path: PathBuf },
    CacheDigest { digest: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredModelConfig {
    pub model_id: String,
    pub engine: String,
    pub task: String,
    pub artifact_digest: String,
    pub artifact_format: String,
    pub tokenizer_sanitized_digest: String,
    pub model_locator: ModelAssetLocator,
    pub tokenizer_locator: ModelAssetLocator,
    pub model_source_url: String,
    pub tokenizer_source_url: String,
    pub pooling: String,
    pub normalize: bool,
    pub max_tokens: usize,
    pub quant: String,
    pub pin: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owned_family: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owned_dtype: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owned_execution: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owned_attention_units: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_locator: Option<ModelAssetLocator>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extra_locators: Vec<ModelAssetLocator>,
    pub engine_identity: EngineIdentity,
    pub numeric_profile_id: NumericProfileId,
    pub fingerprint: Fingerprint,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker_bin: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker_runtime_dir: Option<PathBuf>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CatalogSnapshot {
    pub table_epoch: u64,
    pub models: Vec<ModelCatalogEntry>,
    pub alias_rows: Vec<AliasRow>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssuranceClass {
    Measured,
    Declared,
}

impl AssuranceClass {
    fn as_str(self) -> &'static str {
        match self {
            Self::Measured => "measured",
            Self::Declared => "declared",
        }
    }

    fn parse(value: &str) -> Result<Self, SynapseStoreError> {
        match value {
            "measured" => Ok(Self::Measured),
            "declared" => Ok(Self::Declared),
            other => Err(SynapseStoreError::Decode(format!(
                "unknown assurance class '{other}'"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CertificationStatus {
    Certified,
    Uncertified,
}

impl CertificationStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Certified => "certified",
            Self::Uncertified => "uncertified",
        }
    }

    fn parse(value: &str) -> Result<Self, SynapseStoreError> {
        match value {
            "certified" => Ok(Self::Certified),
            "uncertified" => Ok(Self::Uncertified),
            other => Err(SynapseStoreError::Decode(format!(
                "unknown certification status '{other}'"
            ))),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CertificationKey {
    Measured {
        machine_profile_hash: String,
    },
    Declared {
        machine_profile_hash: String,
        remote_profile_hash: String,
        identity_revision: String,
    },
}

impl CertificationKey {
    fn key_hash(&self) -> String {
        match self {
            Self::Measured {
                machine_profile_hash,
            } => machine_profile_hash.clone(),
            Self::Declared {
                machine_profile_hash,
                remote_profile_hash,
                identity_revision,
            } => hex::encode(Sha256::digest(
                serde_json::to_vec(&(machine_profile_hash, remote_profile_hash, identity_revision))
                    .expect("declared certification key serializes"),
            )),
        }
    }

    fn assurance_class(&self) -> AssuranceClass {
        match self {
            Self::Measured { .. } => AssuranceClass::Measured,
            Self::Declared { .. } => AssuranceClass::Declared,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CertificationRow {
    pub assurance_class: AssuranceClass,
    pub status: CertificationStatus,
    #[serde(flatten)]
    pub key: CertificationKey,
    pub numeric_profile_id: NumericProfileId,
    pub fingerprint: Fingerprint,
    pub certified_at_ms: u64,
    pub os_build: String,
    pub module_generation: u64,
    pub evidence: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PerfRow {
    pub machine_profile_hash: String,
    pub model_id: String,
    pub workload: String,
    pub numeric_profile_id: NumericProfileId,
    pub fingerprint: Fingerprint,
    pub engine: String,
    pub measured_at_ms: u64,
    pub os_build: String,
    pub module_generation: u64,
    pub throughput_tok_s: f64,
    pub cold_load_ms: f64,
    pub single_item_latency_p50_ms: f64,
    pub details: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct KnobAssignmentRow {
    pub machine_profile_hash: String,
    pub workload: String,
    pub knob: PerfKnob,
    pub model_id: String,
    pub numeric_profile_id: NumericProfileId,
    pub fingerprint: Fingerprint,
    pub engine: String,
    pub measured_at_ms: u64,
    pub os_build: String,
    pub module_generation: u64,
    pub throughput_tok_s: f64,
    pub single_item_latency_p50_ms: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct JobRecord {
    pub job_id: String,
    pub request_key: String,
    pub request_digest: String,
    pub kind: String,
    pub module_generation: u64,
    pub state: String,
    pub created_ms: u64,
    pub updated_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_expires_ms: Option<u64>,
    pub result_retention_ttl_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal_at_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_attempt_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logical_handle: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paused_at_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resume_deadline_ms: Option<u64>,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JobAttemptClaim {
    Claimed(JobRecord),
    Attached {
        record: JobRecord,
        active_attempt_id: String,
    },
    NotClaimable(JobRecord),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckpointItem {
    pub item_id: String,
    pub result: Vec<u8>,
    pub provider_request_id: Option<String>,
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
        let alias_table = self.alias_table()?;
        let models = self
            .catalog_models()?
            .into_iter()
            .map(|model| ModelCatalogEntry {
                model_id: model.model_id,
                state: "unloaded".to_string(),
                fingerprints: vec![model.fingerprint],
            })
            .collect();
        Ok(CatalogSnapshot {
            table_epoch: alias_table.table_epoch,
            models,
            alias_rows: alias_table.rows,
        })
    }

    pub fn catalog_models(&self) -> Result<Vec<StoredModelConfig>, SynapseStoreError> {
        let models = self.store.with_conn(|conn| {
            let mut stmt = conn
                .prepare("SELECT config_json FROM models ORDER BY created_ms ASC, model_id ASC")?;
            let rows = stmt
                .query_map([], |row| decode_model_config(row.get::<_, Vec<u8>>(0)?))?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(rows)
        })?;
        Ok(models)
    }

    pub fn upsert_model(
        &self,
        config: &StoredModelConfig,
        now_ms: u64,
    ) -> Result<(), SynapseStoreError> {
        let config_json = serde_json::to_vec(config)?;
        self.store.with_conn_fenced(|tx| {
            tx.execute(
                "INSERT INTO models (
                     model_id, engine, task, fingerprint, config_json, created_ms, updated_ms
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6)
                 ON CONFLICT(model_id) DO UPDATE SET
                     engine = excluded.engine,
                     task = excluded.task,
                     fingerprint = excluded.fingerprint,
                     config_json = excluded.config_json,
                     updated_ms = excluded.updated_ms",
                params![
                    &config.model_id,
                    &config.engine,
                    &config.task,
                    &config.fingerprint.0,
                    config_json,
                    now_ms as i64,
                ],
            )?;
            Ok(())
        })?;
        Ok(())
    }

    pub fn alias_table(&self) -> Result<AliasTable, SynapseStoreError> {
        let rows = self.store.with_conn(|conn| {
            let table_epoch: i64 = conn.query_row(
                "SELECT table_epoch FROM module_meta WHERE id = 0",
                [],
                |row| row.get(0),
            )?;
            let mut stmt = conn.prepare(
                "SELECT fingerprint_a, fingerprint_b, valid_from_ms, valid_to_ms, evidence_json \
                 FROM alias_rows \
                 ORDER BY fingerprint_a, fingerprint_b, valid_from_ms",
            )?;
            let rows = stmt
                .query_map([], |row| {
                    let evidence_json: String = row.get(4)?;
                    let evidence = serde_json::from_str(&evidence_json).map_err(|error| {
                        rusqlite::Error::FromSqlConversionFailure(
                            4,
                            rusqlite::types::Type::Text,
                            Box::new(error),
                        )
                    })?;
                    Ok(AliasRow::with_evidence(
                        Fingerprint(row.get(0)?),
                        Fingerprint(row.get(1)?),
                        row.get::<_, i64>(2)? as u64,
                        row.get::<_, Option<i64>>(3)?.map(|value| value as u64),
                        evidence,
                    ))
                })?
                .collect::<Result<Vec<_>, _>>()?;
            Ok((table_epoch as u64, rows))
        })?;
        Ok(AliasTable {
            table_epoch: rows.0,
            rows: rows.1,
        })
    }

    pub fn store_cert_row(&self, row: &CertificationRow) -> Result<(), SynapseStoreError> {
        if row.assurance_class != row.key.assurance_class() {
            return Err(SynapseStoreError::Decode(
                "certification assurance class does not match its key variant".to_string(),
            ));
        }
        if row.assurance_class == AssuranceClass::Declared
            && row.status != CertificationStatus::Certified
        {
            return Err(SynapseStoreError::Decode(
                "declared certification rows must be certified".to_string(),
            ));
        }
        let evidence_json = serde_json::to_string(&row.evidence)?;
        let (machine_profile_hash, remote_profile_hash, identity_revision) = match &row.key {
            CertificationKey::Measured {
                machine_profile_hash,
            } => (machine_profile_hash.as_str(), None, None),
            CertificationKey::Declared {
                machine_profile_hash,
                remote_profile_hash,
                identity_revision,
            } => (
                machine_profile_hash.as_str(),
                Some(remote_profile_hash.as_str()),
                Some(identity_revision.as_str()),
            ),
        };
        self.store.with_conn_fenced(|tx| {
            tx.execute(
                "INSERT INTO cert_rows (
                     assurance_class, status, key_hash, machine_profile_hash, remote_profile_hash,
                     identity_revision, numeric_profile_id, fingerprint, certified_at_ms,
                     os_build, module_generation, evidence_json
                  ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
                  ON CONFLICT(assurance_class, key_hash, fingerprint) DO UPDATE SET
                     status = excluded.status,
                     numeric_profile_id = excluded.numeric_profile_id,
                     certified_at_ms = excluded.certified_at_ms,
                     os_build = excluded.os_build,
                     module_generation = excluded.module_generation,
                     evidence_json = excluded.evidence_json",
                params![
                    row.assurance_class.as_str(),
                    row.status.as_str(),
                    row.key.key_hash(),
                    machine_profile_hash,
                    remote_profile_hash,
                    identity_revision,
                    &row.numeric_profile_id.0,
                    &row.fingerprint.0,
                    row.certified_at_ms as i64,
                    &row.os_build,
                    row.module_generation as i64,
                    evidence_json,
                ],
            )
        })?;
        Ok(())
    }

    pub fn get_cert_row(
        &self,
        machine_profile_hash: &str,
        fingerprint: &Fingerprint,
    ) -> Result<Option<CertificationRow>, SynapseStoreError> {
        let raw = self.store.with_conn(|conn| {
            conn.query_row(
                &format!(
                    "{CERT_SELECT_SQL} WHERE assurance_class = 'measured' AND status = 'certified' AND key_hash = ?1 AND fingerprint = ?2"
                ),
                params![machine_profile_hash, &fingerprint.0],
                cert_row_from_row,
            )
            .optional()
        })?;
        raw.map(decode_cert_row).transpose()
    }

    pub fn latest_cert_row(
        &self,
        fingerprint: &Fingerprint,
    ) -> Result<Option<CertificationRow>, SynapseStoreError> {
        let raw = self.store.with_conn(|conn| {
            conn.query_row(
                &format!(
                    "{CERT_SELECT_SQL} WHERE status = 'certified' AND fingerprint = ?1 ORDER BY certified_at_ms DESC LIMIT 1"
                ),
                params![&fingerprint.0],
                cert_row_from_row,
            )
            .optional()
        })?;
        raw.map(decode_cert_row).transpose()
    }

    pub fn get_probe_row(
        &self,
        machine_profile_hash: &str,
        fingerprint: &Fingerprint,
    ) -> Result<Option<CertificationRow>, SynapseStoreError> {
        let raw = self.store.with_conn(|conn| {
            conn.query_row(
                &format!(
                    "{CERT_SELECT_SQL} WHERE assurance_class = 'measured' AND key_hash = ?1 AND fingerprint = ?2"
                ),
                params![machine_profile_hash, &fingerprint.0],
                cert_row_from_row,
            )
            .optional()
        })?;
        raw.map(decode_cert_row).transpose()
    }

    pub fn latest_probe_row(
        &self,
        fingerprint: &Fingerprint,
    ) -> Result<Option<CertificationRow>, SynapseStoreError> {
        let raw = self.store.with_conn(|conn| {
            conn.query_row(
                &format!(
                    "{CERT_SELECT_SQL} WHERE assurance_class = 'measured' AND fingerprint = ?1 ORDER BY certified_at_ms DESC LIMIT 1"
                ),
                params![&fingerprint.0],
                cert_row_from_row,
            )
            .optional()
        })?;
        raw.map(decode_cert_row).transpose()
    }

    pub fn has_stale_cert_row(
        &self,
        machine_profile_hash: &str,
        fingerprint: &Fingerprint,
    ) -> Result<bool, SynapseStoreError> {
        let count = self.store.with_conn(|conn| {
            conn.query_row(
                "SELECT COUNT(1) FROM cert_rows
                 WHERE assurance_class = 'measured' AND status = 'certified'
                   AND fingerprint = ?1 AND key_hash <> ?2",
                params![&fingerprint.0, machine_profile_hash],
                |row| row.get::<_, i64>(0),
            )
        })?;
        Ok(count > 0)
    }

    #[allow(dead_code)]
    pub fn get_declared_cert_row(
        &self,
        machine_profile_hash: &str,
        remote_profile_hash: &str,
        identity_revision: &str,
        fingerprint: &Fingerprint,
    ) -> Result<Option<CertificationRow>, SynapseStoreError> {
        let key = CertificationKey::Declared {
            machine_profile_hash: machine_profile_hash.to_string(),
            remote_profile_hash: remote_profile_hash.to_string(),
            identity_revision: identity_revision.to_string(),
        };
        let raw = self.store.with_conn(|conn| {
            conn.query_row(
                &format!(
                    "{CERT_SELECT_SQL} WHERE assurance_class = 'declared' AND key_hash = ?1 AND fingerprint = ?2"
                ),
                params![key.key_hash(), &fingerprint.0],
                cert_row_from_row,
            )
            .optional()
        })?;
        raw.map(decode_cert_row).transpose()
    }

    pub fn declared_cert_row_for_fingerprint(
        &self,
        fingerprint: &Fingerprint,
    ) -> Result<Option<CertificationRow>, SynapseStoreError> {
        let raw = self.store.with_conn(|conn| {
            conn.query_row(
                &format!(
                    "{CERT_SELECT_SQL} WHERE assurance_class = 'declared' AND fingerprint = ?1 ORDER BY certified_at_ms DESC LIMIT 1"
                ),
                params![&fingerprint.0],
                cert_row_from_row,
            )
            .optional()
        })?;
        raw.map(decode_cert_row).transpose()
    }

    #[allow(dead_code)]
    pub fn bind_remote_profile_url(
        &self,
        remote_profile_hash: &str,
        normalized_base_url: &str,
    ) -> Result<(), SynapseStoreError> {
        let profile = remote_profile_hash.to_string();
        let requested_url = normalized_base_url.to_string();
        let previous = self.store.with_conn_fenced(|tx| {
            let previous = tx
                .query_row(
                    "SELECT last_base_url FROM remote_url_bindings WHERE remote_profile_hash = ?1",
                    params![profile],
                    |row| row.get::<_, String>(0),
                )
                .optional()?;
            if previous.is_none() {
                tx.execute(
                    "INSERT INTO remote_url_bindings (
                         remote_profile_hash, last_base_url, missing_since_ms
                     ) VALUES (?1, ?2, NULL)",
                    params![profile, requested_url],
                )?;
            } else if previous.as_deref() == Some(requested_url.as_str()) {
                tx.execute(
                    "UPDATE remote_url_bindings SET missing_since_ms = NULL
                     WHERE remote_profile_hash = ?1",
                    params![profile],
                )?;
            }
            Ok(previous)
        })?;
        if let Some(previous_url) = previous {
            if previous_url != requested_url {
                return Err(SynapseStoreError::RemoteDeploymentChanged {
                    remote_profile_hash: remote_profile_hash.to_string(),
                    previous_url,
                    requested_url,
                });
            }
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub fn sweep_remote_url_bindings(
        &self,
        active_remote_profile_hashes: &[String],
        now_ms: u64,
        removal_grace_ms: u64,
    ) -> Result<usize, SynapseStoreError> {
        let active = active_remote_profile_hashes
            .iter()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        let removed = self.store.with_conn_fenced(|tx| {
            let mut stmt = tx
                .prepare("SELECT remote_profile_hash, missing_since_ms FROM remote_url_bindings")?;
            let rows = stmt
                .query_map([], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, Option<i64>>(1)?))
                })?
                .collect::<Result<Vec<_>, _>>()?;
            drop(stmt);
            let mut removed = 0;
            for (profile, missing_since_ms) in rows {
                if active.contains(profile.as_str()) {
                    tx.execute(
                        "UPDATE remote_url_bindings SET missing_since_ms = NULL
                         WHERE remote_profile_hash = ?1",
                        params![profile],
                    )?;
                    continue;
                }
                match missing_since_ms {
                    None => {
                        tx.execute(
                            "UPDATE remote_url_bindings SET missing_since_ms = ?2
                             WHERE remote_profile_hash = ?1",
                            params![profile, now_ms as i64],
                        )?;
                    }
                    Some(missing_since)
                        if now_ms >= (missing_since as u64).saturating_add(removal_grace_ms) =>
                    {
                        removed += tx.execute(
                            "DELETE FROM remote_url_bindings WHERE remote_profile_hash = ?1",
                            params![profile],
                        )?;
                    }
                    Some(_) => {}
                }
            }
            Ok(removed)
        })?;
        Ok(removed)
    }

    pub fn store_perf_row(&self, row: &PerfRow) -> Result<(), SynapseStoreError> {
        let details_json = serde_json::to_string(&row.details)?;
        self.store.with_conn_fenced(|tx| {
            tx.execute(
                "INSERT INTO perf_rows (
                     machine_profile_hash, model_id, workload, numeric_profile_id, fingerprint,
                     engine, measured_at_ms, os_build, module_generation, throughput_tok_s,
                     cold_load_ms, single_item_latency_p50_ms, details_json
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
                 ON CONFLICT(machine_profile_hash, fingerprint) DO UPDATE SET
                     model_id = excluded.model_id,
                     workload = excluded.workload,
                     numeric_profile_id = excluded.numeric_profile_id,
                     engine = excluded.engine,
                     measured_at_ms = excluded.measured_at_ms,
                     os_build = excluded.os_build,
                     module_generation = excluded.module_generation,
                     throughput_tok_s = excluded.throughput_tok_s,
                     cold_load_ms = excluded.cold_load_ms,
                     single_item_latency_p50_ms = excluded.single_item_latency_p50_ms,
                     details_json = excluded.details_json",
                params![
                    &row.machine_profile_hash,
                    &row.model_id,
                    &row.workload,
                    &row.numeric_profile_id.0,
                    &row.fingerprint.0,
                    &row.engine,
                    row.measured_at_ms as i64,
                    &row.os_build,
                    row.module_generation as i64,
                    row.throughput_tok_s,
                    row.cold_load_ms,
                    row.single_item_latency_p50_ms,
                    details_json,
                ],
            )
        })?;
        Ok(())
    }

    pub fn get_perf_row(
        &self,
        machine_profile_hash: &str,
        fingerprint: &Fingerprint,
    ) -> Result<Option<PerfRow>, SynapseStoreError> {
        let raw = self.store.with_conn(|conn| {
            conn.query_row(
                "SELECT machine_profile_hash, model_id, workload, numeric_profile_id, fingerprint,
                        engine, measured_at_ms, os_build, module_generation, throughput_tok_s,
                        cold_load_ms, single_item_latency_p50_ms, details_json
                 FROM perf_rows WHERE machine_profile_hash = ?1 AND fingerprint = ?2",
                params![machine_profile_hash, &fingerprint.0],
                perf_row_from_row,
            )
            .optional()
        })?;
        raw.map(decode_perf_row).transpose()
    }

    pub fn latest_perf_row(
        &self,
        fingerprint: &Fingerprint,
    ) -> Result<Option<PerfRow>, SynapseStoreError> {
        let raw = self.store.with_conn(|conn| {
            conn.query_row(
                "SELECT machine_profile_hash, model_id, workload, numeric_profile_id, fingerprint,
                        engine, measured_at_ms, os_build, module_generation, throughput_tok_s,
                        cold_load_ms, single_item_latency_p50_ms, details_json
                 FROM perf_rows
                 WHERE fingerprint = ?1
                 ORDER BY measured_at_ms DESC
                 LIMIT 1",
                params![&fingerprint.0],
                perf_row_from_row,
            )
            .optional()
        })?;
        raw.map(decode_perf_row).transpose()
    }

    pub fn current_perf_rows(
        &self,
        machine_profile_hash: &str,
    ) -> Result<Vec<PerfRow>, SynapseStoreError> {
        let rows = self.store.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT machine_profile_hash, model_id, workload, numeric_profile_id, fingerprint,
                        engine, measured_at_ms, os_build, module_generation, throughput_tok_s,
                        cold_load_ms, single_item_latency_p50_ms, details_json
                 FROM perf_rows
                 WHERE machine_profile_hash = ?1
                 ORDER BY workload ASC, throughput_tok_s DESC, model_id ASC",
            )?;
            let rows = stmt
                .query_map(params![machine_profile_hash], perf_row_from_row)?
                .map(|row| row.and_then(|raw| decode_perf_row(raw).map_err(to_sql_error)))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(rows)
        })?;
        Ok(rows)
    }

    pub fn replace_knob_assignments(
        &self,
        machine_profile_hash: &str,
        rows: &[KnobAssignmentRow],
    ) -> Result<(), SynapseStoreError> {
        self.store.with_conn_fenced(|tx| {
            tx.execute(
                "DELETE FROM knob_assignments WHERE machine_profile_hash = ?1",
                params![machine_profile_hash],
            )?;
            for row in rows {
                tx.execute(
                    "INSERT INTO knob_assignments (
                         machine_profile_hash, workload, knob, model_id, numeric_profile_id,
                         fingerprint, engine, measured_at_ms, os_build, module_generation,
                         throughput_tok_s, single_item_latency_p50_ms
                     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                    params![
                        &row.machine_profile_hash,
                        &row.workload,
                        row.knob.as_str(),
                        &row.model_id,
                        &row.numeric_profile_id.0,
                        &row.fingerprint.0,
                        &row.engine,
                        row.measured_at_ms as i64,
                        &row.os_build,
                        row.module_generation as i64,
                        row.throughput_tok_s,
                        row.single_item_latency_p50_ms,
                    ],
                )?;
            }
            Ok(())
        })?;
        Ok(())
    }

    pub fn knob_assignment(
        &self,
        machine_profile_hash: &str,
        workload: &str,
        knob: PerfKnob,
    ) -> Result<Option<KnobAssignmentRow>, SynapseStoreError> {
        let raw = self.store.with_conn(|conn| {
            conn.query_row(
                "SELECT machine_profile_hash, workload, knob, model_id, numeric_profile_id,
                        fingerprint, engine, measured_at_ms, os_build, module_generation,
                        throughput_tok_s, single_item_latency_p50_ms
                 FROM knob_assignments
                 WHERE machine_profile_hash = ?1 AND workload = ?2 AND knob = ?3",
                params![machine_profile_hash, workload, knob.as_str()],
                knob_assignment_from_row,
            )
            .optional()
        })?;
        raw.map(decode_knob_assignment_row).transpose()
    }

    pub fn knob_assignments(
        &self,
        machine_profile_hash: &str,
    ) -> Result<Vec<KnobAssignmentRow>, SynapseStoreError> {
        let rows = self.store.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT machine_profile_hash, workload, knob, model_id, numeric_profile_id,
                        fingerprint, engine, measured_at_ms, os_build, module_generation,
                        throughput_tok_s, single_item_latency_p50_ms
                 FROM knob_assignments
                 WHERE machine_profile_hash = ?1
                 ORDER BY workload ASC, knob ASC",
            )?;
            let rows = stmt
                .query_map(params![machine_profile_hash], knob_assignment_from_row)?
                .map(|row| {
                    row.and_then(|raw| decode_knob_assignment_row(raw).map_err(to_sql_error))
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(rows)
        })?;
        Ok(rows)
    }

    pub fn declare_alias_pair(
        &self,
        fingerprint_a: &Fingerprint,
        fingerprint_b: &Fingerprint,
        evidence: &Value,
        now_ms: u64,
    ) -> Result<(bool, u64), SynapseStoreError> {
        let row = AliasRow::with_evidence(
            fingerprint_a.clone(),
            fingerprint_b.clone(),
            now_ms,
            None,
            evidence.clone(),
        );
        let evidence_json = serde_json::to_string(&row.evidence)?;
        let outcome = self.store.with_conn_fenced(|tx| {
            let current_epoch = table_epoch_tx(tx)?;
            let existing: Option<i64> = tx
                .query_row(
                    "SELECT rowid FROM alias_rows \
                     WHERE fingerprint_a = ?1 AND fingerprint_b = ?2 AND valid_to_ms IS NULL",
                    params![&row.fingerprint_a.0, &row.fingerprint_b.0],
                    |row| row.get(0),
                )
                .optional()?;
            if existing.is_some() {
                return Ok((false, current_epoch as u64));
            }
            tx.execute(
                "INSERT INTO alias_rows (fingerprint_a, fingerprint_b, valid_from_ms, valid_to_ms, evidence_json) \
                 VALUES (?1, ?2, ?3, NULL, ?4)",
                params![
                    &row.fingerprint_a.0,
                    &row.fingerprint_b.0,
                    row.valid_from_ms as i64,
                    evidence_json,
                ],
            )?;
            let next_epoch = bump_table_epoch_tx(tx)?;
            Ok((true, next_epoch as u64))
        })?;
        Ok(outcome)
    }

    pub fn retract_alias_pair(
        &self,
        fingerprint_a: &Fingerprint,
        fingerprint_b: &Fingerprint,
        evidence: &Value,
        now_ms: u64,
    ) -> Result<(bool, u64), SynapseStoreError> {
        let row = AliasRow::with_evidence(
            fingerprint_a.clone(),
            fingerprint_b.clone(),
            now_ms,
            Some(now_ms),
            evidence.clone(),
        );
        let evidence_json = serde_json::to_string(&row.evidence)?;
        let outcome = self.store.with_conn_fenced(|tx| {
            let changed = tx.execute(
                "UPDATE alias_rows SET valid_to_ms = ?3, evidence_json = ?4 \
                 WHERE fingerprint_a = ?1 AND fingerprint_b = ?2 AND valid_to_ms IS NULL",
                params![
                    &row.fingerprint_a.0,
                    &row.fingerprint_b.0,
                    now_ms as i64,
                    evidence_json,
                ],
            )?;
            if changed == 0 {
                return Ok((false, table_epoch_tx(tx)? as u64));
            }
            let next_epoch = bump_table_epoch_tx(tx)?;
            Ok((true, next_epoch as u64))
        })?;
        Ok(outcome)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn admit_job(
        &self,
        request_key: &str,
        request_digest: &str,
        kind: &str,
        module_generation: u64,
        logical_handle: Option<&str>,
        params_json: &Value,
        now_ms: u64,
        execution_ttl_ms: u64,
        result_retention_ttl_ms: u64,
    ) -> Result<JobAdmission, SynapseStoreError> {
        enum TxAdmission {
            Ready(Box<JobAdmission>),
            Conflict(String),
        }

        let params_bytes = serde_json::to_vec(params_json)?;
        let request_key = request_key.to_string();
        let request_digest = request_digest.to_string();
        let kind = kind.to_string();
        let logical_handle = logical_handle.map(str::to_string);
        let admission = self.store.with_conn_fenced(|tx| {
            expire_jobs_tx(tx, now_ms as i64)?;
            purge_retained_jobs_tx(tx, now_ms as i64)?;
            if let Some(existing) = job_by_request_key_tx(tx, &request_key)? {
                if existing.request_digest != request_digest {
                    return Ok(TxAdmission::Conflict(existing.request_digest));
                }
                if existing.state == JOB_STATE_PAUSED_NEEDS_REAUTH {
                    tx.execute(
                        "UPDATE jobs SET state = ?1, module_generation = ?2, updated_ms = ?3,
                             execution_expires_ms = ?4, active_attempt_id = NULL,
                             paused_at_ms = NULL, resume_deadline_ms = NULL, params_json = ?5
                         WHERE job_id = ?6 AND state = ?7",
                        params![
                            JOB_STATE_QUEUED,
                            module_generation as i64,
                            now_ms as i64,
                            now_ms.saturating_add(execution_ttl_ms) as i64,
                            params_bytes,
                            existing.job_id,
                            JOB_STATE_PAUSED_NEEDS_REAUTH,
                        ],
                    )?;
                    return Ok(TxAdmission::Ready(Box::new(JobAdmission::Admitted(
                        job_by_id_tx(tx, &existing.job_id)?.expect("resumed job is readable"),
                    ))));
                }
                if !existing.is_terminal_failed() {
                    return Ok(TxAdmission::Ready(Box::new(JobAdmission::Existing(
                        existing,
                    ))));
                }
            }

            let job_id = new_job_id(&request_key, module_generation, now_ms);
            let page_count = tx.query_row(
                "SELECT COUNT(1) FROM result_pages WHERE request_digest = ?1",
                params![request_digest],
                |row| row.get::<_, i64>(0),
            )?;
            tx.execute(
                "INSERT INTO jobs (
                     job_id, request_key, request_digest, kind, module_generation, state,
                     created_ms, updated_ms, execution_expires_ms,
                     result_retention_ttl_ms, terminal_at_ms, active_attempt_id,
                     logical_handle, paused_at_ms, resume_deadline_ms, params_json,
                     result_json, error_json, page_count
                 ) VALUES (
                     ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7, ?8, ?9, NULL, NULL,
                     ?10, NULL, NULL, ?11, NULL, NULL, ?12
                 )",
                params![
                    job_id,
                    request_key,
                    request_digest,
                    kind,
                    module_generation as i64,
                    JOB_STATE_QUEUED,
                    now_ms as i64,
                    now_ms.saturating_add(execution_ttl_ms) as i64,
                    result_retention_ttl_ms as i64,
                    logical_handle,
                    params_bytes,
                    page_count,
                ],
            )?;
            Ok(TxAdmission::Ready(Box::new(JobAdmission::Admitted(
                job_by_id_tx(tx, &job_id)?.expect("inserted job is readable"),
            ))))
        })?;
        match admission {
            TxAdmission::Ready(admission) => Ok(*admission),
            TxAdmission::Conflict(existing_digest) => Err(SynapseStoreError::IdempotencyConflict {
                request_key,
                existing_digest,
                requested_digest: request_digest,
            }),
        }
    }

    pub fn mark_job_running(
        &self,
        job_id: &str,
        module_generation: u64,
        now_ms: u64,
    ) -> Result<bool, SynapseStoreError> {
        Ok(matches!(
            self.claim_job_attempt(job_id, module_generation, now_ms)?,
            JobAttemptClaim::Claimed(_)
        ))
    }

    pub fn claim_job_attempt(
        &self,
        job_id: &str,
        module_generation: u64,
        now_ms: u64,
    ) -> Result<JobAttemptClaim, SynapseStoreError> {
        let claim = self.store.with_conn_fenced(|tx| {
            expire_jobs_tx(tx, now_ms as i64)?;
            let active_attempt_id = new_job_id(job_id, module_generation, now_ms);
            let changed = tx.execute(
                "UPDATE jobs SET state = ?1, updated_ms = ?2, active_attempt_id = ?3
                 WHERE job_id = ?4 AND module_generation = ?5 AND state = ?6
                   AND active_attempt_id IS NULL AND execution_expires_ms > ?2",
                params![
                    JOB_STATE_RUNNING,
                    now_ms as i64,
                    active_attempt_id,
                    job_id,
                    module_generation as i64,
                    JOB_STATE_QUEUED,
                ],
            )?;
            let record = job_by_id_tx(tx, job_id)?.ok_or(rusqlite::Error::QueryReturnedNoRows)?;
            if changed > 0 {
                return Ok(JobAttemptClaim::Claimed(record));
            }
            if let Some(active_attempt_id) = record.active_attempt_id.clone() {
                return Ok(JobAttemptClaim::Attached {
                    record,
                    active_attempt_id,
                });
            }
            Ok(JobAttemptClaim::NotClaimable(record))
        })?;
        Ok(claim)
    }

    pub fn complete_job(
        &self,
        job_id: &str,
        result_json: &Value,
        pages: &[Vec<u8>],
        now_ms: u64,
    ) -> Result<(), SynapseStoreError> {
        let initial_page = self
            .get_job(job_id)?
            .ok_or_else(|| SynapseStoreError::Decode(format!("unknown job '{job_id}'")))?
            .page_count;
        for (offset, page) in pages.iter().enumerate() {
            self.commit_job_page(
                job_id,
                initial_page.saturating_add(offset as u32),
                page,
                &[],
                now_ms,
            )?;
        }
        self.finish_job(job_id, result_json, now_ms)
    }

    pub fn commit_job_page(
        &self,
        job_id: &str,
        page_no: u32,
        page_json: &[u8],
        items: &[CheckpointItem],
        now_ms: u64,
    ) -> Result<(), SynapseStoreError> {
        if items.is_empty() {
            return Err(SynapseStoreError::Decode(
                "page commit must contain at least one checkpoint item".to_string(),
            ));
        }
        let mut provider_request_ids = Vec::new();
        for item in items {
            if let Some(provider_request_id) = item.provider_request_id.as_deref() {
                if provider_request_id.is_empty() || provider_request_id.len() > 128 {
                    return Err(SynapseStoreError::Decode(
                        "provider request ids must contain 1..=128 bytes".to_string(),
                    ));
                }
                if !provider_request_ids.contains(&provider_request_id) {
                    provider_request_ids.push(provider_request_id);
                }
            }
        }
        if provider_request_ids.len() > 16 {
            return Err(SynapseStoreError::Decode(
                "a committed page may contain at most 16 provider request ids".to_string(),
            ));
        }
        let provider_request_ids_json = if provider_request_ids.is_empty() {
            None
        } else {
            Some(serde_json::to_vec(&provider_request_ids)?)
        };
        self.store.with_conn_fenced(|tx| {
            let (request_digest, visible_page_count, state): (String, i64, String) = tx.query_row(
                "SELECT request_digest, page_count, state FROM jobs WHERE job_id = ?1",
                params![job_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )?;
            if visible_page_count != i64::from(page_no) {
                return Err(rusqlite::Error::InvalidParameterName(format!(
                    "page {page_no} cannot follow visible page count {visible_page_count}"
                )));
            }
            if state != JOB_STATE_RUNNING && state != JOB_STATE_QUEUED {
                return Err(rusqlite::Error::InvalidParameterName(format!(
                    "cannot append a page while job state is {state}"
                )));
            }
            tx.execute(
                "INSERT INTO result_pages (
                     request_digest, page_no, page_json, provider_request_ids_json, committed_at
                 ) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    request_digest,
                    page_no as i64,
                    page_json,
                    provider_request_ids_json,
                    now_ms as i64,
                ],
            )?;
            for item in items {
                tx.execute(
                    "INSERT INTO remote_checkpoints (
                         request_digest, item_id, result, page_no,
                         provider_request_id, committed_at
                     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![
                        request_digest,
                        item.item_id,
                        item.result,
                        page_no as i64,
                        item.provider_request_id,
                        now_ms as i64,
                    ],
                )?;
            }
            let changed = tx.execute(
                "UPDATE jobs SET page_count = page_count + 1, updated_ms = ?1
                 WHERE job_id = ?2 AND page_count = ?3",
                params![now_ms as i64, job_id, page_no as i64],
            )?;
            if changed != 1 {
                return Err(rusqlite::Error::ExecuteReturnedResults);
            }
            Ok(())
        })?;
        Ok(())
    }

    pub fn finish_job(
        &self,
        job_id: &str,
        result_json: &Value,
        now_ms: u64,
    ) -> Result<(), SynapseStoreError> {
        let result_bytes = serde_json::to_vec(result_json)?;
        self.store.with_conn_fenced(|tx| {
            tx.execute(
                "UPDATE jobs SET state = ?1, updated_ms = ?2, terminal_at_ms = ?2,
                     execution_expires_ms = NULL, active_attempt_id = NULL,
                     result_json = ?3, error_json = NULL
                 WHERE job_id = ?4",
                params![JOB_STATE_DONE, now_ms as i64, result_bytes, job_id],
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
            tx.execute(
                "UPDATE jobs SET state = ?1, updated_ms = ?2, terminal_at_ms = ?2,
                     execution_expires_ms = NULL, active_attempt_id = NULL,
                     error_json = ?3, result_json = NULL
                 WHERE job_id = ?4",
                params![state, now_ms as i64, error_bytes, job_id],
            )?;
            Ok(())
        })?;
        Ok(())
    }

    pub fn quarantine_job_for_continuity(
        &self,
        job_id: &str,
        message: &str,
        now_ms: u64,
    ) -> Result<(), SynapseStoreError> {
        self.fail_job(
            job_id,
            false,
            &serde_json::json!({
                "code": "remote_identity_drift",
                "class": "permanent",
                "safe_to_retry_same_request": false,
                "message": message,
            }),
            now_ms,
        )
    }

    #[allow(dead_code)]
    pub fn pause_job_needs_reauth(
        &self,
        job_id: &str,
        logical_handle: &str,
        now_ms: u64,
        resume_window_ms: u64,
    ) -> Result<bool, SynapseStoreError> {
        let changed = self.store.with_conn_fenced(|tx| {
            tx.execute(
                "UPDATE jobs SET state = ?1, updated_ms = ?2, execution_expires_ms = NULL,
                     active_attempt_id = NULL, logical_handle = ?3, paused_at_ms = ?2,
                     resume_deadline_ms = ?4
                 WHERE job_id = ?5 AND state IN (?6, ?7)",
                params![
                    JOB_STATE_PAUSED_NEEDS_REAUTH,
                    now_ms as i64,
                    logical_handle,
                    now_ms.saturating_add(resume_window_ms) as i64,
                    job_id,
                    JOB_STATE_QUEUED,
                    JOB_STATE_RUNNING,
                ],
            )
        })?;
        Ok(changed > 0)
    }

    pub fn resume_paused_job(
        &self,
        job_id: &str,
        module_generation: u64,
        now_ms: u64,
        execution_ttl_ms: u64,
    ) -> Result<bool, SynapseStoreError> {
        let changed = self.store.with_conn_fenced(|tx| {
            expire_jobs_tx(tx, now_ms as i64)?;
            tx.execute(
                "UPDATE jobs SET state = ?1, module_generation = ?2, updated_ms = ?3,
                     execution_expires_ms = ?4, active_attempt_id = NULL,
                     paused_at_ms = NULL, resume_deadline_ms = NULL, error_json = NULL,
                     terminal_at_ms = NULL
                 WHERE job_id = ?5 AND state = ?6 AND resume_deadline_ms > ?3",
                params![
                    JOB_STATE_QUEUED,
                    module_generation as i64,
                    now_ms as i64,
                    now_ms.saturating_add(execution_ttl_ms) as i64,
                    job_id,
                    JOB_STATE_PAUSED_NEEDS_REAUTH,
                ],
            )
        })?;
        Ok(changed > 0)
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
                "UPDATE jobs SET state = ?1, updated_ms = ?2, terminal_at_ms = ?2,
                     execution_expires_ms = NULL, active_attempt_id = NULL,
                     error_json = ?3, result_json = NULL
                 WHERE module_generation < ?4 AND state IN (?5, ?6, ?7)",
                params![
                    JOB_STATE_FAILED_TRANSIENT,
                    now_ms as i64,
                    error_bytes,
                    current_generation as i64,
                    JOB_STATE_QUEUED,
                    JOB_STATE_RUNNING,
                    JOB_STATE_PAUSED_NEEDS_REAUTH,
                ],
            )
        })?;
        Ok(changed)
    }

    pub fn purge_expired_jobs(&self, now_ms: u64) -> Result<usize, SynapseStoreError> {
        let purged = self.store.with_conn_fenced(|tx| {
            expire_jobs_tx(tx, now_ms as i64)?;
            purge_retained_jobs_tx(tx, now_ms as i64)
        })?;
        Ok(purged)
    }

    pub fn get_job(&self, job_id: &str) -> Result<Option<JobRecord>, SynapseStoreError> {
        let record = self.store.with_conn_fenced(|tx| {
            let now = unix_now_ms() as i64;
            expire_job_tx(tx, job_id, now)?;
            job_by_id_tx(tx, job_id)
        })?;
        Ok(record)
    }

    pub fn get_job_page(
        &self,
        job_id: &str,
        page_index: u32,
    ) -> Result<Option<Vec<u8>>, SynapseStoreError> {
        let page = self.store.with_conn(|conn| {
            conn.query_row(
                "SELECT p.page_json
                 FROM jobs j
                 JOIN result_pages p ON p.request_digest = j.request_digest
                 WHERE j.job_id = ?1 AND p.page_no = ?2",
                params![job_id, page_index as i64],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .optional()
        })?;
        Ok(page)
    }

    pub fn checkpoint_count(&self, request_digest: &str) -> Result<u64, SynapseStoreError> {
        let count = self.store.with_conn(|conn| {
            conn.query_row(
                "SELECT COUNT(1) FROM remote_checkpoints WHERE request_digest = ?1",
                params![request_digest],
                |row| row.get::<_, i64>(0),
            )
        })?;
        Ok(count as u64)
    }

    pub fn committed_item_ids(
        &self,
        request_digest: &str,
    ) -> Result<std::collections::BTreeSet<String>, SynapseStoreError> {
        let ids = self.store.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT item_id FROM remote_checkpoints
                 WHERE request_digest = ?1 ORDER BY page_no, item_id",
            )?;
            let ids = stmt
                .query_map(params![request_digest], |row| row.get::<_, String>(0))?
                .collect::<Result<std::collections::BTreeSet<_>, _>>()?;
            Ok(ids)
        })?;
        Ok(ids)
    }

    #[cfg(test)]
    fn get_job_at(
        &self,
        job_id: &str,
        now_ms: u64,
    ) -> Result<Option<JobRecord>, SynapseStoreError> {
        let record = self.store.with_conn_fenced(|tx| {
            expire_job_tx(tx, job_id, now_ms as i64)?;
            job_by_id_tx(tx, job_id)
        })?;
        Ok(record)
    }
}

fn expire_jobs_tx(tx: &rusqlite::Transaction<'_>, now_ms: i64) -> rusqlite::Result<usize> {
    let execution_error = serde_json::to_vec(&serde_json::json!({
        "code": "deadline_exceeded",
        "class": "transient",
        "safe_to_retry_same_request": false,
        "message": "durable job execution TTL expired",
    }))
    .expect("execution expiry error serializes");
    let reauth_error = serde_json::to_vec(&serde_json::json!({
        "code": "needs_reauth_expired",
        "class": "permanent",
        "safe_to_retry_same_request": false,
        "message": "credential reauthentication deadline expired",
    }))
    .expect("reauth expiry error serializes");
    let execution_expired = tx.execute(
        "UPDATE jobs SET state = ?1, updated_ms = ?2, terminal_at_ms = ?2,
             execution_expires_ms = NULL, active_attempt_id = NULL, error_json = ?3
         WHERE state IN (?4, ?5) AND execution_expires_ms <= ?2",
        params![
            JOB_STATE_FAILED_TRANSIENT,
            now_ms,
            execution_error,
            JOB_STATE_QUEUED,
            JOB_STATE_RUNNING,
        ],
    )?;
    let reauth_expired = tx.execute(
        "UPDATE jobs SET state = ?1, updated_ms = ?2, terminal_at_ms = ?2,
             active_attempt_id = NULL, error_json = ?3
         WHERE state = ?4 AND resume_deadline_ms <= ?2",
        params![
            JOB_STATE_FAILED_PERMANENT,
            now_ms,
            reauth_error,
            JOB_STATE_PAUSED_NEEDS_REAUTH,
        ],
    )?;
    Ok(execution_expired + reauth_expired)
}

fn expire_job_tx(
    tx: &rusqlite::Transaction<'_>,
    job_id: &str,
    now_ms: i64,
) -> rusqlite::Result<()> {
    let execution_error = serde_json::to_vec(&serde_json::json!({
        "code": "deadline_exceeded",
        "class": "transient",
        "safe_to_retry_same_request": false,
        "message": "durable job execution TTL expired",
    }))
    .expect("execution expiry error serializes");
    let reauth_error = serde_json::to_vec(&serde_json::json!({
        "code": "needs_reauth_expired",
        "class": "permanent",
        "safe_to_retry_same_request": false,
        "message": "credential reauthentication deadline expired",
    }))
    .expect("reauth expiry error serializes");
    tx.execute(
        "UPDATE jobs SET state = ?1, updated_ms = ?2, terminal_at_ms = ?2,
             execution_expires_ms = NULL, active_attempt_id = NULL, error_json = ?3
         WHERE job_id = ?4 AND state IN (?5, ?6) AND execution_expires_ms <= ?2",
        params![
            JOB_STATE_FAILED_TRANSIENT,
            now_ms,
            execution_error,
            job_id,
            JOB_STATE_QUEUED,
            JOB_STATE_RUNNING,
        ],
    )?;
    tx.execute(
        "UPDATE jobs SET state = ?1, updated_ms = ?2, terminal_at_ms = ?2,
             active_attempt_id = NULL, error_json = ?3
         WHERE job_id = ?4 AND state = ?5 AND resume_deadline_ms <= ?2",
        params![
            JOB_STATE_FAILED_PERMANENT,
            now_ms,
            reauth_error,
            job_id,
            JOB_STATE_PAUSED_NEEDS_REAUTH,
        ],
    )?;
    Ok(())
}

fn purge_retained_jobs_tx(tx: &rusqlite::Transaction<'_>, now_ms: i64) -> rusqlite::Result<usize> {
    let mut stmt = tx.prepare(
        "SELECT job_id, request_digest FROM jobs
         WHERE terminal_at_ms IS NOT NULL
           AND terminal_at_ms + result_retention_ttl_ms <= ?1",
    )?;
    let expired = stmt
        .query_map(params![now_ms], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    drop(stmt);
    for (job_id, _) in &expired {
        tx.execute("DELETE FROM jobs WHERE job_id = ?1", params![job_id])?;
    }
    let digests = expired
        .iter()
        .map(|(_, digest)| digest)
        .collect::<std::collections::BTreeSet<_>>();
    for digest in digests {
        let remaining: i64 = tx.query_row(
            "SELECT COUNT(1) FROM jobs WHERE request_digest = ?1",
            params![digest],
            |row| row.get(0),
        )?;
        if remaining == 0 {
            tx.execute(
                "DELETE FROM remote_checkpoints WHERE request_digest = ?1",
                params![digest],
            )?;
            tx.execute(
                "DELETE FROM result_pages WHERE request_digest = ?1",
                params![digest],
            )?;
        }
    }
    Ok(expired.len())
}

fn job_by_request_key_tx(
    tx: &rusqlite::Transaction<'_>,
    request_key: &str,
) -> rusqlite::Result<Option<JobRecord>> {
    let sql = format!(
        "{JOB_SELECT_SQL} WHERE request_key = ?1 ORDER BY created_ms DESC, rowid DESC LIMIT 1"
    );
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

const JOB_SELECT_SQL: &str = "SELECT job_id, request_key, request_digest, kind,
        module_generation, state, created_ms, updated_ms, execution_expires_ms,
        result_retention_ttl_ms, terminal_at_ms, active_attempt_id, logical_handle,
        paused_at_ms, resume_deadline_ms, page_count, result_json, error_json FROM jobs";

fn unix_now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

const CERT_SELECT_SQL: &str = "SELECT assurance_class, status, machine_profile_hash,
        remote_profile_hash, identity_revision, numeric_profile_id, fingerprint,
        certified_at_ms, os_build, module_generation, evidence_json FROM cert_rows";

struct RawCertificationRow {
    assurance_class: String,
    status: String,
    machine_profile_hash: Option<String>,
    remote_profile_hash: Option<String>,
    identity_revision: Option<String>,
    numeric_profile_id: String,
    fingerprint: String,
    certified_at_ms: u64,
    os_build: String,
    module_generation: u64,
    evidence_json: String,
}

struct RawPerfRow {
    machine_profile_hash: String,
    model_id: String,
    workload: String,
    numeric_profile_id: String,
    fingerprint: String,
    engine: String,
    measured_at_ms: u64,
    os_build: String,
    module_generation: u64,
    throughput_tok_s: f64,
    cold_load_ms: f64,
    single_item_latency_p50_ms: f64,
    details_json: String,
}

struct RawKnobAssignmentRow {
    machine_profile_hash: String,
    workload: String,
    knob: String,
    model_id: String,
    numeric_profile_id: String,
    fingerprint: String,
    engine: String,
    measured_at_ms: u64,
    os_build: String,
    module_generation: u64,
    throughput_tok_s: f64,
    single_item_latency_p50_ms: f64,
}

fn cert_row_from_row(row: &Row<'_>) -> rusqlite::Result<RawCertificationRow> {
    Ok(RawCertificationRow {
        assurance_class: row.get(0)?,
        status: row.get(1)?,
        machine_profile_hash: row.get(2)?,
        remote_profile_hash: row.get(3)?,
        identity_revision: row.get(4)?,
        numeric_profile_id: row.get(5)?,
        fingerprint: row.get(6)?,
        certified_at_ms: row.get::<_, i64>(7)? as u64,
        os_build: row.get(8)?,
        module_generation: row.get::<_, i64>(9)? as u64,
        evidence_json: row.get(10)?,
    })
}

fn perf_row_from_row(row: &Row<'_>) -> rusqlite::Result<RawPerfRow> {
    Ok(RawPerfRow {
        machine_profile_hash: row.get(0)?,
        model_id: row.get(1)?,
        workload: row.get(2)?,
        numeric_profile_id: row.get(3)?,
        fingerprint: row.get(4)?,
        engine: row.get(5)?,
        measured_at_ms: row.get::<_, i64>(6)? as u64,
        os_build: row.get(7)?,
        module_generation: row.get::<_, i64>(8)? as u64,
        throughput_tok_s: row.get(9)?,
        cold_load_ms: row.get(10)?,
        single_item_latency_p50_ms: row.get(11)?,
        details_json: row.get(12)?,
    })
}

fn knob_assignment_from_row(row: &Row<'_>) -> rusqlite::Result<RawKnobAssignmentRow> {
    Ok(RawKnobAssignmentRow {
        machine_profile_hash: row.get(0)?,
        workload: row.get(1)?,
        knob: row.get(2)?,
        model_id: row.get(3)?,
        numeric_profile_id: row.get(4)?,
        fingerprint: row.get(5)?,
        engine: row.get(6)?,
        measured_at_ms: row.get::<_, i64>(7)? as u64,
        os_build: row.get(8)?,
        module_generation: row.get::<_, i64>(9)? as u64,
        throughput_tok_s: row.get(10)?,
        single_item_latency_p50_ms: row.get(11)?,
    })
}

fn decode_cert_row(row: RawCertificationRow) -> Result<CertificationRow, SynapseStoreError> {
    let assurance_class = AssuranceClass::parse(&row.assurance_class)?;
    let key = match assurance_class {
        AssuranceClass::Measured => CertificationKey::Measured {
            machine_profile_hash: row.machine_profile_hash.ok_or_else(|| {
                SynapseStoreError::Decode(
                    "measured certification row has no machine profile key".to_string(),
                )
            })?,
        },
        AssuranceClass::Declared => CertificationKey::Declared {
            machine_profile_hash: row.machine_profile_hash.ok_or_else(|| {
                SynapseStoreError::Decode(
                    "declared certification row has no machine profile key".to_string(),
                )
            })?,
            remote_profile_hash: row.remote_profile_hash.ok_or_else(|| {
                SynapseStoreError::Decode(
                    "declared certification row has no remote profile key".to_string(),
                )
            })?,
            identity_revision: row.identity_revision.ok_or_else(|| {
                SynapseStoreError::Decode(
                    "declared certification row has no identity revision".to_string(),
                )
            })?,
        },
    };
    Ok(CertificationRow {
        assurance_class,
        status: CertificationStatus::parse(&row.status)?,
        key,
        numeric_profile_id: NumericProfileId(row.numeric_profile_id),
        fingerprint: Fingerprint(row.fingerprint),
        certified_at_ms: row.certified_at_ms,
        os_build: row.os_build,
        module_generation: row.module_generation,
        evidence: serde_json::from_str(&row.evidence_json)?,
    })
}

fn decode_perf_row(row: RawPerfRow) -> Result<PerfRow, SynapseStoreError> {
    Ok(PerfRow {
        machine_profile_hash: row.machine_profile_hash,
        model_id: row.model_id,
        workload: row.workload,
        numeric_profile_id: NumericProfileId(row.numeric_profile_id),
        fingerprint: Fingerprint(row.fingerprint),
        engine: row.engine,
        measured_at_ms: row.measured_at_ms,
        os_build: row.os_build,
        module_generation: row.module_generation,
        throughput_tok_s: row.throughput_tok_s,
        cold_load_ms: row.cold_load_ms,
        single_item_latency_p50_ms: row.single_item_latency_p50_ms,
        details: serde_json::from_str(&row.details_json)?,
    })
}

fn decode_knob_assignment_row(
    row: RawKnobAssignmentRow,
) -> Result<KnobAssignmentRow, SynapseStoreError> {
    Ok(KnobAssignmentRow {
        machine_profile_hash: row.machine_profile_hash,
        workload: row.workload,
        knob: PerfKnob::parse(&row.knob).map_err(SynapseStoreError::Decode)?,
        model_id: row.model_id,
        numeric_profile_id: NumericProfileId(row.numeric_profile_id),
        fingerprint: Fingerprint(row.fingerprint),
        engine: row.engine,
        measured_at_ms: row.measured_at_ms,
        os_build: row.os_build,
        module_generation: row.module_generation,
        throughput_tok_s: row.throughput_tok_s,
        single_item_latency_p50_ms: row.single_item_latency_p50_ms,
    })
}

fn to_sql_error(error: SynapseStoreError) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(error))
}

fn table_epoch_tx(tx: &rusqlite::Transaction<'_>) -> rusqlite::Result<i64> {
    tx.query_row(
        "SELECT table_epoch FROM module_meta WHERE id = 0",
        [],
        |row| row.get(0),
    )
}

fn bump_table_epoch_tx(tx: &rusqlite::Transaction<'_>) -> rusqlite::Result<i64> {
    let next = table_epoch_tx(tx)? + 1;
    tx.execute(
        "UPDATE module_meta SET table_epoch = ?1 WHERE id = 0",
        params![next],
    )?;
    Ok(next)
}

fn row_to_job(row: &Row<'_>) -> rusqlite::Result<JobRecord> {
    Ok(JobRecord {
        job_id: row.get(0)?,
        request_key: row.get(1)?,
        request_digest: row.get(2)?,
        kind: row.get(3)?,
        module_generation: row.get::<_, i64>(4)? as u64,
        state: row.get(5)?,
        created_ms: row.get::<_, i64>(6)? as u64,
        updated_ms: row.get::<_, i64>(7)? as u64,
        execution_expires_ms: row.get::<_, Option<i64>>(8)?.map(|value| value as u64),
        result_retention_ttl_ms: row.get::<_, i64>(9)? as u64,
        terminal_at_ms: row.get::<_, Option<i64>>(10)?.map(|value| value as u64),
        active_attempt_id: row.get(11)?,
        logical_handle: row.get(12)?,
        paused_at_ms: row.get::<_, Option<i64>>(13)?.map(|value| value as u64),
        resume_deadline_ms: row.get::<_, Option<i64>>(14)?.map(|value| value as u64),
        page_count: row.get::<_, i64>(15)? as u32,
        result_json: decode_optional_json(row.get(16)?, 16)?,
        error_json: decode_optional_json(row.get(17)?, 17)?,
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

fn decode_model_config(bytes: Vec<u8>) -> rusqlite::Result<StoredModelConfig> {
    serde_json::from_slice(&bytes).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Blob, Box::new(error))
    })
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
    use std::collections::BTreeMap;

    use super::*;
    use cortexkit_store_types::{Isolation, StorageBackend};

    fn admit(
        store: &SynapseStore,
        request_key: &str,
        request_digest: &str,
        generation: u64,
        now_ms: u64,
        execution_ttl_ms: u64,
        retention_ttl_ms: u64,
    ) -> JobAdmission {
        admit_kind(
            store,
            request_key,
            request_digest,
            "embed.batch",
            generation,
            now_ms,
            execution_ttl_ms,
            retention_ttl_ms,
        )
    }

    fn admit_kind(
        store: &SynapseStore,
        request_key: &str,
        request_digest: &str,
        kind: &str,
        generation: u64,
        now_ms: u64,
        execution_ttl_ms: u64,
        retention_ttl_ms: u64,
    ) -> JobAdmission {
        store
            .admit_job(
                request_key,
                request_digest,
                kind,
                generation,
                None,
                &serde_json::json!({"items": 2}),
                now_ms,
                execution_ttl_ms,
                retention_ttl_ms,
            )
            .unwrap()
    }

    #[test]
    fn restart_marks_queued_running_and_paused_attempts_failed_without_losing_pages() {
        let (root, descriptor) = temp_descriptor("restart-jobs");
        let store = SynapseStore::open(&descriptor).unwrap();
        let generation = store.next_module_generation().unwrap();
        let queued = admit(
            &store,
            "queued-key",
            "queued-digest",
            generation,
            10,
            1_000,
            500,
        )
        .record()
        .job_id
        .clone();
        let running = admit(
            &store,
            "running-key",
            "running-digest",
            generation,
            11,
            1_000,
            500,
        )
        .record()
        .job_id
        .clone();
        assert!(store.mark_job_running(&running, generation, 12).unwrap());
        store
            .commit_job_page(
                &running,
                0,
                br#"{"page":0}"#,
                &[CheckpointItem {
                    item_id: "item-1".to_string(),
                    result: br#"{"id":"item-1"}"#.to_vec(),
                    provider_request_id: None,
                }],
                13,
            )
            .unwrap();
        let paused = admit(
            &store,
            "paused-key",
            "paused-digest",
            generation,
            14,
            1_000,
            500,
        )
        .record()
        .job_id
        .clone();
        assert!(store
            .pause_job_needs_reauth(&paused, "vault/provider", 15, 1_000)
            .unwrap());
        let probe = admit_kind(
            &store,
            "probe-key",
            "probe-digest",
            "probe",
            generation,
            16,
            1_000,
            500,
        )
        .record()
        .job_id
        .clone();
        assert!(store.mark_job_running(&probe, generation, 17).unwrap());

        let failed = store
            .fail_prior_generation_incomplete_jobs(
                generation + 1,
                &serde_json::json!({"code": "module_restarted"}),
                20,
            )
            .unwrap();
        assert_eq!(failed, 4);
        for job_id in [queued, running.clone(), paused, probe] {
            let job = store
                .get_job_at(&job_id, 21)
                .unwrap()
                .expect("job survives restart");
            assert_eq!(job.state, JOB_STATE_FAILED_TRANSIENT);
            assert_eq!(job.error_json.unwrap()["code"], "module_restarted");
        }
        assert!(store.get_job_page(&running, 0).unwrap().is_some());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn request_key_rejects_a_different_digest_and_same_digest_creates_a_fresh_attempt() {
        let (root, descriptor) = temp_descriptor("request-key");
        let store = SynapseStore::open(&descriptor).unwrap();
        let generation = store.next_module_generation().unwrap();
        let first = admit(&store, "same-key", "digest-a", generation, 100, 1_000, 500);
        let first_id = first.record().job_id.clone();
        let same = admit(&store, "same-key", "digest-a", generation, 101, 1_000, 500);
        assert!(matches!(same, JobAdmission::Existing(_)));
        assert_eq!(same.record().job_id, first_id);

        let conflict = store
            .admit_job(
                "same-key",
                "digest-b",
                "embed.batch",
                generation,
                None,
                &serde_json::json!({"items": 2}),
                102,
                1_000,
                500,
            )
            .unwrap_err();
        assert!(matches!(
            conflict,
            SynapseStoreError::IdempotencyConflict { .. }
        ));

        store
            .fail_job(
                &first_id,
                true,
                &serde_json::json!({"code": "module_restarted"}),
                103,
            )
            .unwrap();
        let fresh = admit(&store, "same-key", "digest-a", generation, 104, 1_000, 500);
        assert!(matches!(fresh, JobAdmission::Admitted(_)));
        assert_ne!(fresh.record().job_id, first_id);
        assert!(store.get_job_at(&first_id, 105).unwrap().is_some());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn page_commit_rolls_back_as_a_unit_and_resume_skips_committed_ids() {
        let (root, descriptor) = temp_descriptor("atomic-page");
        let store = SynapseStore::open(&descriptor).unwrap();
        let generation = store.next_module_generation().unwrap();
        let job_id = admit(
            &store,
            "page-key",
            "page-digest",
            generation,
            1_000,
            1_000,
            100,
        )
        .record()
        .job_id
        .clone();
        assert!(store.mark_job_running(&job_id, generation, 1_001).unwrap());
        let active_attempt_id = store
            .get_job_at(&job_id, 1_001)
            .unwrap()
            .unwrap()
            .active_attempt_id
            .unwrap();
        assert_ne!(active_attempt_id, job_id);
        assert!(matches!(
            store.claim_job_attempt(&job_id, generation, 1_001).unwrap(),
            JobAttemptClaim::Attached {
                active_attempt_id: attached,
                ..
            } if attached == active_attempt_id
        ));
        assert!(store
            .commit_job_page(&job_id, 0, br#"{"page":0}"#, &[], 1_002)
            .is_err());
        assert!(store
            .commit_job_page(
                &job_id,
                0,
                br#"{"page":0}"#,
                &[CheckpointItem {
                    item_id: "invalid-provider-id".to_string(),
                    result: vec![0],
                    provider_request_id: Some(String::new()),
                }],
                1_002,
            )
            .is_err());
        let duplicate_items = [
            CheckpointItem {
                item_id: "item-1".to_string(),
                result: vec![1],
                provider_request_id: None,
            },
            CheckpointItem {
                item_id: "item-1".to_string(),
                result: vec![2],
                provider_request_id: None,
            },
        ];
        assert!(store
            .commit_job_page(&job_id, 0, br#"{"page":0}"#, &duplicate_items, 1_002)
            .is_err());
        assert_eq!(store.checkpoint_count("page-digest").unwrap(), 0);
        assert_eq!(
            store
                .get_job_at(&job_id, 1_003)
                .unwrap()
                .unwrap()
                .page_count,
            0
        );
        assert!(store.get_job_page(&job_id, 0).unwrap().is_none());

        store
            .commit_job_page(
                &job_id,
                0,
                br#"{"page":0}"#,
                &[CheckpointItem {
                    item_id: "item-1".to_string(),
                    result: vec![1],
                    provider_request_id: None,
                }],
                1_004,
            )
            .unwrap();
        store
            .fail_job(
                &job_id,
                true,
                &serde_json::json!({"code": "module_restarted"}),
                1_005,
            )
            .unwrap();
        let resumed = admit(
            &store,
            "page-key",
            "page-digest",
            generation,
            1_006,
            1_000,
            100,
        );
        assert_eq!(resumed.record().page_count, 1);
        assert_eq!(
            store.committed_item_ids("page-digest").unwrap(),
            ["item-1".to_string()].into_iter().collect()
        );
        assert!(store
            .get_job_page(&resumed.record().job_id, 0)
            .unwrap()
            .is_some());
        let _ = std::fs::remove_dir_all(root);
    }

    struct FakeContinuityCheck {
        calls: std::sync::atomic::AtomicUsize,
        fail: bool,
    }

    #[subc_client_rs::async_trait]
    impl crate::remote::ContinuityCheck for FakeContinuityCheck {
        async fn check(
            &self,
            _request_digest: &str,
            _synapse_model_id: &str,
            _logical_handle: Option<&str>,
        ) -> Result<(), crate::remote::ContinuityError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail {
                Err(crate::remote::ContinuityError::new("sentinel mismatch"))
            } else {
                Ok(())
            }
        }
    }

    #[tokio::test]
    async fn restart_resubmission_runs_continuity_and_failure_preserves_prior_pages() {
        let (root, descriptor) = temp_descriptor("continuity");
        let first_job = {
            let store = SynapseStore::open(&descriptor).unwrap();
            let generation = store.next_module_generation().unwrap();
            let job_id = admit(
                &store,
                "continuity-key",
                "continuity-digest",
                generation,
                100,
                1_000,
                500,
            )
            .record()
            .job_id
            .clone();
            assert!(store.mark_job_running(&job_id, generation, 101).unwrap());
            store
                .commit_job_page(
                    &job_id,
                    0,
                    br#"{"page":0}"#,
                    &[CheckpointItem {
                        item_id: "item-1".to_string(),
                        result: vec![1],
                        provider_request_id: None,
                    }],
                    102,
                )
                .unwrap();
            job_id
        };

        let store = SynapseStore::open(&descriptor).unwrap();
        let generation = store.next_module_generation().unwrap();
        store
            .fail_prior_generation_incomplete_jobs(
                generation,
                &serde_json::json!({"code": "module_restarted"}),
                110,
            )
            .unwrap();
        let resumed = admit(
            &store,
            "continuity-key",
            "continuity-digest",
            generation,
            111,
            1_000,
            500,
        );
        let resumed_id = resumed.record().job_id.clone();
        assert_ne!(resumed_id, first_job);
        assert_eq!(resumed.record().page_count, 1);
        assert!(matches!(
            store
                .claim_job_attempt(&resumed_id, generation, 112)
                .unwrap(),
            JobAttemptClaim::Claimed(_)
        ));
        let check = FakeContinuityCheck {
            calls: std::sync::atomic::AtomicUsize::new(0),
            fail: true,
        };
        assert!(!crate::apply_checkpoint_continuity(
            &store,
            &check,
            &resumed_id,
            "continuity-digest",
            "remote-model",
            Some("vault/provider"),
            113,
        )
        .await
        .unwrap());
        assert_eq!(check.calls.load(Ordering::SeqCst), 1);
        let quarantined = store.get_job_at(&resumed_id, 114).unwrap().unwrap();
        assert_eq!(quarantined.state, JOB_STATE_FAILED_PERMANENT);
        assert_eq!(
            quarantined.error_json.unwrap()["code"],
            "remote_identity_drift"
        );
        assert!(store.get_job_page(&resumed_id, 0).unwrap().is_some());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn pause_suspends_execution_ttl_and_deadline_expires_lazily_and_in_sweep() {
        let (root, descriptor) = temp_descriptor("pause-ttl");
        let store = SynapseStore::open(&descriptor).unwrap();
        let generation = store.next_module_generation().unwrap();
        let job_id = admit(&store, "pause-key", "pause-digest", generation, 100, 20, 50)
            .record()
            .job_id
            .clone();
        assert!(store
            .pause_job_needs_reauth(&job_id, "vault/provider", 110, 100)
            .unwrap());
        assert_eq!(store.purge_expired_jobs(150).unwrap(), 0);
        let paused = store.get_job_at(&job_id, 150).unwrap().unwrap();
        assert_eq!(paused.state, JOB_STATE_PAUSED_NEEDS_REAUTH);
        assert_eq!(paused.execution_expires_ms, None);
        assert_eq!(paused.resume_deadline_ms, Some(210));

        assert_eq!(store.purge_expired_jobs(210).unwrap(), 0);
        let expired = store.get_job_at(&job_id, 210).unwrap().unwrap();
        assert_eq!(expired.state, JOB_STATE_FAILED_PERMANENT);
        assert_eq!(expired.error_json.unwrap()["code"], "needs_reauth_expired");
        assert_eq!(store.purge_expired_jobs(259).unwrap(), 0);
        assert_eq!(store.purge_expired_jobs(260).unwrap(), 1);
        assert!(store.get_job_at(&job_id, 260).unwrap().is_none());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn migration_from_v5_preserves_jobs_pages_and_marks_certifications_measured() {
        let (root, descriptor) = temp_descriptor("migration-v5");
        {
            let legacy = open_sqlite(&descriptor).unwrap();
            legacy.migrate(NAMESPACE, &MIGRATIONS[..5]).unwrap();
            legacy
                .with_conn_fenced(|tx| {
                    tx.execute(
                        "INSERT INTO cert_rows (
                             machine_profile_hash, numeric_profile_id, fingerprint,
                             certified_at_ms, os_build, module_generation, evidence_json
                         ) VALUES ('machine', 'np', 'fp', 5, 'os', 1, '{}')",
                        [],
                    )?;
                    tx.execute(
                        "INSERT INTO jobs (
                             job_id, request_key, kind, module_generation, state, created_ms,
                             updated_ms, expires_ms, params_json, result_json, error_json, page_count
                         ) VALUES (
                             'legacy-job', 'legacy-key', 'embed.batch', 1, 'done', 10,
                             20, 120, '{}', '{\"page_count\":1}', NULL, 1
                         )",
                        [],
                    )?;
                    tx.execute(
                        "INSERT INTO job_pages (job_id, page_index, page_json)
                         VALUES ('legacy-job', 0, '{\"page\":0}')",
                        [],
                    )?;
                    Ok(())
                })
                .unwrap();
        }

        let migrated = SynapseStore::open(&descriptor).unwrap();
        let cert = migrated
            .get_cert_row("machine", &Fingerprint("fp".to_string()))
            .unwrap()
            .unwrap();
        assert_eq!(cert.assurance_class, AssuranceClass::Measured);
        assert_eq!(cert.status, CertificationStatus::Certified);
        assert_eq!(
            cert.key,
            CertificationKey::Measured {
                machine_profile_hash: "machine".to_string()
            }
        );
        let job = migrated.get_job_at("legacy-job", 21).unwrap().unwrap();
        assert_eq!(job.request_digest, "legacy:legacy-job");
        assert_eq!(job.result_retention_ttl_ms, 100);
        assert!(migrated.get_job_page("legacy-job", 0).unwrap().is_some());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn uncertified_reprobe_demotes_the_machine_fingerprint_pair() {
        let (root, descriptor) = temp_descriptor("uncertified-reprobe");
        let store = SynapseStore::open(&descriptor).unwrap();
        let fingerprint = Fingerprint("decode-fp".to_string());
        let mut row = CertificationRow {
            assurance_class: AssuranceClass::Measured,
            status: CertificationStatus::Certified,
            key: CertificationKey::Measured {
                machine_profile_hash: "machine-a".to_string(),
            },
            numeric_profile_id: NumericProfileId("decode-np".to_string()),
            fingerprint: fingerprint.clone(),
            certified_at_ms: 10,
            os_build: "os-a".to_string(),
            module_generation: 1,
            evidence: serde_json::json!({"gate": "token_exact"}),
        };
        store.store_cert_row(&row).unwrap();
        assert!(store
            .get_cert_row("machine-a", &fingerprint)
            .unwrap()
            .is_some());

        row.status = CertificationStatus::Uncertified;
        row.certified_at_ms = 20;
        row.evidence = serde_json::json!({
            "blocking_reason": "token_mismatch",
            "mismatches": [{"prompt": "corrupted fixture"}],
        });
        store.store_cert_row(&row).unwrap();

        assert!(store
            .get_cert_row("machine-a", &fingerprint)
            .unwrap()
            .is_none());
        assert_eq!(
            store.get_probe_row("machine-a", &fingerprint).unwrap(),
            Some(row)
        );
        let refused = crate::ensure_fingerprint_certified(
            &store,
            "machine-a",
            &fingerprint,
            "owned-qwen3",
            false,
        )
        .expect_err("an uncertified decode outcome must refuse future owned requests");
        assert_eq!(refused.code, "not_certified");
        assert!(refused.message.contains("token_mismatch"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn declared_certifications_use_remote_profile_keys() {
        let (root, descriptor) = temp_descriptor("declared-cert");
        let store = SynapseStore::open(&descriptor).unwrap();
        let fingerprint = Fingerprint("declared-fp".to_string());
        let row = CertificationRow {
            assurance_class: AssuranceClass::Declared,
            status: CertificationStatus::Certified,
            key: CertificationKey::Declared {
                machine_profile_hash: "machine-a".to_string(),
                remote_profile_hash: "remote-profile".to_string(),
                identity_revision: "r1".to_string(),
            },
            numeric_profile_id: NumericProfileId("np".to_string()),
            fingerprint: fingerprint.clone(),
            certified_at_ms: 10,
            os_build: String::new(),
            module_generation: 1,
            evidence: serde_json::json!({"sentinels": 5}),
        };
        store.store_cert_row(&row).unwrap();
        assert_eq!(
            store
                .get_declared_cert_row("machine-a", "remote-profile", "r1", &fingerprint)
                .unwrap(),
            Some(row)
        );
        assert!(store
            .get_cert_row("remote-profile", &fingerprint)
            .unwrap()
            .is_none());
        assert!(store
            .get_declared_cert_row("machine-b", "remote-profile", "r1", &fingerprint)
            .unwrap()
            .is_none());
        let rejected = crate::declared_certification_for_request(
            &store,
            &fingerprint,
            "declared-model",
            false,
        )
        .unwrap_err();
        assert_eq!(rejected.code, "declared_identity_not_accepted");
        assert!(crate::declared_certification_for_request(
            &store,
            &fingerprint,
            "declared-model",
            true,
        )
        .unwrap()
        .is_some());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn url_bindings_reject_changes_and_remove_missing_profiles_after_grace() {
        let (root, descriptor) = temp_descriptor("url-binding");
        let store = SynapseStore::open(&descriptor).unwrap();
        store
            .bind_remote_profile_url("profile", "https://provider.example/v1")
            .unwrap();
        store
            .bind_remote_profile_url("profile", "https://provider.example/v1")
            .unwrap();
        assert!(matches!(
            store.bind_remote_profile_url("profile", "https://moved.example/v1"),
            Err(SynapseStoreError::RemoteDeploymentChanged { .. })
        ));
        assert_eq!(store.sweep_remote_url_bindings(&[], 100, 50).unwrap(), 0);
        assert_eq!(store.sweep_remote_url_bindings(&[], 149, 50).unwrap(), 0);
        assert_eq!(store.sweep_remote_url_bindings(&[], 150, 50).unwrap(), 1);
        store
            .bind_remote_profile_url("profile", "https://moved.example/v1")
            .unwrap();
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn catalog_models_survive_reopen_and_appear_in_snapshot() {
        let (root, descriptor) = temp_descriptor("catalog-models");
        let config = StoredModelConfig {
            model_id: "minilm".to_string(),
            engine: "ort".to_string(),
            task: "embed".to_string(),
            artifact_digest: format!("sha256:{}", "1".repeat(64)),
            artifact_format: "onnx".to_string(),
            tokenizer_sanitized_digest: format!("sha256:{}", "2".repeat(64)),
            model_locator: ModelAssetLocator::CacheDigest {
                digest: format!("sha256:{}", "1".repeat(64)),
            },
            tokenizer_locator: ModelAssetLocator::CacheDigest {
                digest: format!("sha256:{}", "3".repeat(64)),
            },
            model_source_url: "file:///tmp/model.onnx".to_string(),
            tokenizer_source_url: "file:///tmp/tokenizer.json".to_string(),
            pooling: "mean".to_string(),
            normalize: true,
            max_tokens: 512,
            quant: "fp32".to_string(),
            pin: true,
            owned_family: None,
            owned_dtype: None,
            owned_execution: None,
            owned_attention_units: None,
            config_locator: None,
            extra_locators: Vec::new(),
            engine_identity: EngineIdentity {
                engine: "ort".to_string(),
                version: "test".to_string(),
                build_flags: BTreeMap::new(),
            },
            numeric_profile_id: NumericProfileId("np-test".to_string()),
            fingerprint: Fingerprint("fp-test".to_string()),
            worker_bin: None,
            worker_runtime_dir: None,
        };

        {
            let store = SynapseStore::open(&descriptor).unwrap();
            store.upsert_model(&config, 10).unwrap();
        }

        let reopened = SynapseStore::open(&descriptor).unwrap();
        let models = reopened.catalog_models().unwrap();
        assert_eq!(models, vec![config.clone()]);
        let snapshot = reopened.catalog_snapshot().unwrap();
        assert_eq!(snapshot.models.len(), 1);
        assert_eq!(snapshot.models[0].model_id, config.model_id);
        assert_eq!(snapshot.models[0].state, "unloaded");
        assert_eq!(snapshot.models[0].fingerprints, vec![config.fingerprint]);
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
