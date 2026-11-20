use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use cortexkit_store::{open_sqlite, Migration, SqliteStore, StoreError};
use cortexkit_store_types::{Isolation, StorageBackend, StorageDescriptor};
use rusqlite::{params, OptionalExtension, Row, TransactionBehavior};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use synapse_core::{EngineIdentity, Fingerprint, MachineProfile, NumericProfileId};
use thiserror::Error;

#[cfg(test)]
mod job_tests;
mod registration_reconciliation;

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PerfKnob {
    Performance,
    #[default]
    Balanced,
    Quiet,
}

impl PerfKnob {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Performance => "performance",
            Self::Balanced => "balanced",
            Self::Quiet => "quiet",
        }
    }

    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "performance" => Ok(Self::Performance),
            "balanced" => Ok(Self::Balanced),
            "quiet" => Ok(Self::Quiet),
            other => Err(format!("unknown performance knob '{other}'")),
        }
    }
}

const NAMESPACE: &str = "synapse_module";
pub const JOB_STATE_QUEUED: &str = "queued";
pub const JOB_STATE_RUNNING: &str = "running";
pub const JOB_STATE_DONE: &str = "done";
pub const JOB_STATE_FAILED_TRANSIENT: &str = "failed_transient";
pub const JOB_STATE_FAILED_PERMANENT: &str = "failed_permanent";

/// Minimum dead-space volume in bytes required before triggering a VACUUM.
///
/// A small store with proportionally many free pages (for example, several
/// megabytes of free space in an otherwise tiny database) is not worth the latency
/// and exclusive I/O of a full SQLite rewrite. A 350 MB VACUUM requires roughly
/// one second of exclusive I/O. Reclaiming free pages is worthwhile only when
/// dead space is substantial (at least 64 MiB), meaningfully cutting disk usage
/// and reducing the size of nightly online backups.
pub const RECLAIM_FREELIST_MIN_BYTES: u64 = 64 * 1024 * 1024;

/// Divisor for the minimum ratio of free pages to total pages required before
/// triggering a VACUUM (specifically, `freelist_count >= page_count / 2`).
///
/// When free pages make up less than half of the database, routine application
/// writes will naturally reuse those free pages without increasing the database
/// file size. VACUUM is reserved for stores where at least 50% of allocated space
/// consists of empty freelist pages.
pub const RECLAIM_FREELIST_PAGE_RATIO_DIVISOR: u64 = 2;

#[cfg(test)]
thread_local! {
    /// Lets a test interrupt a restore import right after the named table is
    /// copied, to prove the import rolls back as one transaction. Release
    /// builds have no such hook.
    static FAIL_RESTORE_AFTER_TABLE: std::cell::Cell<Option<&'static str>> =
        const { std::cell::Cell::new(None) };
}

/// Newest schema version this binary can migrate a store to.
///
/// Derived from the migration list rather than restated as a literal: a literal
/// keeps reporting the old number the first time a migration is appended, and a
/// daemon comparing this against a store's actual version would then read a
/// stale binary as current. `max` rather than `last` so it does not depend on
/// the list staying sorted.
pub fn newest_schema_version() -> u32 {
    MIGRATIONS
        .iter()
        .map(|migration| migration.version)
        .max()
        .unwrap_or(0)
}

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
    Migration {
        version: 9,
        statements: r#"
                  DROP INDEX IF EXISTS cert_rows_fingerprint_idx;
                  ALTER TABLE cert_rows RENAME TO cert_rows_v8;
                  CREATE TABLE cert_rows_rebuilt (
                      row_id INTEGER PRIMARY KEY,
                      certification_class TEXT NOT NULL,
                      assurance_class TEXT NOT NULL,
                      status TEXT NOT NULL CHECK (status IN ('certified', 'uncertified')),
                      key_hash TEXT NOT NULL,
                      machine_profile_hash TEXT,
                      revisioned_machine_profile_hash TEXT,
                      profile_activation_epoch INTEGER,
                      model_id TEXT,
                      decode_fingerprint TEXT,
                      remote_profile_hash TEXT,
                      identity_revision TEXT,
                      numeric_profile_id TEXT,
                      fingerprint TEXT NOT NULL,
                      certified_at_ms INTEGER NOT NULL,
                      os_build TEXT NOT NULL,
                      module_generation INTEGER NOT NULL,
                      evidence_schema_revision TEXT,
                      processing_fingerprint TEXT,
                      runtime_config_digest TEXT,
                      constraint_runtime_identities_json TEXT,
                      worker_path_evidence_json TEXT,
                      g_dec_manifest_revision TEXT,
                      evidence_json TEXT NOT NULL,
                      CHECK (certification_class IN ('measured_owned_decode', 'declared', 'remote', 'embedding', 'rerank', 'legacy_owned_decode')),
                      CHECK (certification_class <> 'measured_owned_decode' OR (
                          revisioned_machine_profile_hash IS NOT NULL
                          AND profile_activation_epoch > 0
                          AND model_id IS NOT NULL
                          AND decode_fingerprint IS NOT NULL
                          AND evidence_schema_revision IS NOT NULL
                      )),
                      CHECK (certification_class <> 'legacy_owned_decode' OR status <> 'certified')
                  );
                  INSERT INTO cert_rows_rebuilt (
                      certification_class, assurance_class, status, key_hash,
                      machine_profile_hash, remote_profile_hash, identity_revision,
                      numeric_profile_id, fingerprint, certified_at_ms, os_build,
                      module_generation, evidence_json
                  )
                  SELECT CASE WHEN assurance_class = 'declared' THEN 'declared' ELSE 'legacy_owned_decode' END,
                         assurance_class,
                         CASE WHEN assurance_class = 'declared' THEN status ELSE 'uncertified' END,
                         key_hash,
                         machine_profile_hash,
                         remote_profile_hash,
                         identity_revision,
                         numeric_profile_id,
                         fingerprint,
                         certified_at_ms,
                         os_build,
                         module_generation,
                         evidence_json
                  FROM cert_rows_v8;
                  DROP TABLE cert_rows_v8;
                  ALTER TABLE cert_rows_rebuilt RENAME TO cert_rows;
                  CREATE UNIQUE INDEX cert_rows_owned_decode_identity_v1
                      ON cert_rows (revisioned_machine_profile_hash, profile_activation_epoch,
                                    model_id, decode_fingerprint, evidence_schema_revision)
                      WHERE certification_class = 'measured_owned_decode';
                  CREATE UNIQUE INDEX cert_rows_declared_identity_v1
                      ON cert_rows (key_hash, fingerprint)
                      WHERE certification_class = 'declared';
                  CREATE UNIQUE INDEX cert_rows_remote_identity_v1
                      ON cert_rows (key_hash, fingerprint)
                      WHERE certification_class = 'remote';
                  CREATE UNIQUE INDEX cert_rows_embedding_identity_v1
                      ON cert_rows (key_hash, fingerprint)
                      WHERE certification_class = 'embedding';
                  CREATE UNIQUE INDEX cert_rows_rerank_identity_v1
                      ON cert_rows (key_hash, fingerprint)
                      WHERE certification_class = 'rerank';
                  CREATE INDEX cert_rows_fingerprint_idx ON cert_rows(fingerprint);

                  CREATE TABLE approvals (
                      row_id INTEGER PRIMARY KEY,
                      schema_revision TEXT NOT NULL,
                      model_id TEXT NOT NULL,
                      decode_fingerprint TEXT NOT NULL,
                      enabled INTEGER NOT NULL CHECK (enabled IN (0, 1)),
                      grammar_enabled INTEGER NOT NULL CHECK (grammar_enabled IN (0, 1)),
                      disabled_reason TEXT,
                      approved_by TEXT,
                      approved_at_ms INTEGER,
                      updated_at_ms INTEGER NOT NULL,
                      evidence_requirements_revision TEXT NOT NULL,
                      semantic_digest TEXT NOT NULL,
                      generation INTEGER NOT NULL DEFAULT 0,
                      fencing_metadata TEXT NOT NULL DEFAULT '{}',
                      UNIQUE (model_id, decode_fingerprint)
                  );
                  CREATE INDEX approvals_enabled_idx ON approvals(enabled, model_id, decode_fingerprint);
                  CREATE TABLE approval_digest_corruption_events (
                      event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                      model_id TEXT NOT NULL,
                      decode_fingerprint TEXT NOT NULL,
                      observed_digest TEXT NOT NULL,
                      recomputed_digest TEXT NOT NULL,
                      observed_at_ms INTEGER NOT NULL
                  );
                  CREATE TABLE approval_migration_markers (
                      seed_revision TEXT PRIMARY KEY,
                      schema_revision TEXT NOT NULL,
                      seed_digest TEXT NOT NULL,
                      applied_at_ms INTEGER NOT NULL,
                      row_count INTEGER NOT NULL
                  );

                  CREATE TABLE profile_state (
                      id INTEGER PRIMARY KEY CHECK (id = 0),
                      snapshot_json TEXT,
                      revisioned_machine_profile_hash TEXT,
                      profile_activation_epoch INTEGER,
                      previous_revisioned_machine_profile_hash TEXT,
                      last_rotation_reason TEXT,
                      last_rotation_at_ms INTEGER
                  );
                  INSERT INTO profile_state (id) VALUES (0);
                  CREATE TABLE profile_rotation_events (
                      event_id TEXT PRIMARY KEY,
                      old_revisioned_machine_profile_hash TEXT,
                      new_revisioned_machine_profile_hash TEXT NOT NULL,
                      old_profile_activation_epoch INTEGER,
                      new_profile_activation_epoch INTEGER NOT NULL CHECK (new_profile_activation_epoch > 0),
                      changed_fields_json TEXT NOT NULL,
                      previous_snapshot_json TEXT,
                      current_snapshot_json TEXT NOT NULL,
                      observed_at_ms INTEGER NOT NULL,
                      module_generation INTEGER NOT NULL,
                      created_at_ms INTEGER NOT NULL,
                      UNIQUE (new_revisioned_machine_profile_hash, new_profile_activation_epoch)
                  );
                  CREATE TABLE profile_rotation_certification_outcomes (
                      event_id TEXT NOT NULL REFERENCES profile_rotation_events(event_id) ON DELETE CASCADE,
                      model_id TEXT NOT NULL,
                      decode_fingerprint TEXT NOT NULL,
                      outcome_state TEXT NOT NULL CHECK (outcome_state IN ('not_required', 'required', 'in_progress', 'passed', 'failed')),
                      certified_at_ms INTEGER,
                      failure_reason TEXT,
                      PRIMARY KEY (event_id, model_id, decode_fingerprint)
                  );
                  CREATE INDEX profile_rotation_outcomes_order_idx
                      ON profile_rotation_certification_outcomes(event_id, model_id, decode_fingerprint);
                  CREATE TABLE cert_row_rebuild_events (
                      event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                      outcome TEXT NOT NULL,
                      detail TEXT NOT NULL,
                      occurred_at_ms INTEGER NOT NULL
                  );
         "#,
    },
    Migration {
        version: 10,
        statements: r#"
                   DROP INDEX IF EXISTS cert_rows_owned_decode_identity_v1;
                   ALTER TABLE cert_rows
                       ADD COLUMN constraint_runtime_identities_digest TEXT;
                   CREATE UNIQUE INDEX cert_rows_owned_decode_identity_v2
                       ON cert_rows (revisioned_machine_profile_hash, profile_activation_epoch,
                                     model_id, decode_fingerprint, evidence_schema_revision,
                                     constraint_runtime_identities_digest)
                       WHERE certification_class = 'measured_owned_decode';
         "#,
    },
    Migration {
        version: 11,
        statements: r#"
                   CREATE TABLE serving_artifacts (
                       artifact_id TEXT PRIMARY KEY,
                       model_id TEXT NOT NULL,
                       source_format TEXT NOT NULL CHECK (source_format = 'gguf'),
                       source_quantization TEXT NOT NULL CHECK (source_quantization = 'q4_k_m'),
                       source_digest TEXT NOT NULL,
                       derived_digest TEXT,
                       derivation_contract TEXT,
                       deterministic_inputs_digest TEXT,
                       verified_derived_digest TEXT,
                       artifact_json TEXT NOT NULL,
                       gc_pinned INTEGER NOT NULL DEFAULT 0 CHECK (gc_pinned IN (0, 1)),
                       ingested_at_ms INTEGER NOT NULL,
                       CHECK (
                           (derived_digest IS NULL
                            AND derivation_contract IS NULL
                            AND deterministic_inputs_digest IS NULL
                            AND verified_derived_digest IS NULL)
                           OR
                           (derived_digest IS NOT NULL
                            AND derivation_contract = 'q8-ingest-v1'
                            AND deterministic_inputs_digest IS NOT NULL
                            AND verified_derived_digest = derived_digest)
                       )
                   );
                   CREATE UNIQUE INDEX serving_artifacts_source_identity_idx
                       ON serving_artifacts(model_id, source_digest, derived_digest);

                   CREATE TABLE serving_certification_records (
                       certification_record_id TEXT PRIMARY KEY,
                       catalog_fingerprint TEXT NOT NULL,
                       artifact_id TEXT NOT NULL REFERENCES serving_artifacts(artifact_id),
                       record_json TEXT NOT NULL,
                       recorded_at_ms INTEGER NOT NULL
                   );
                   CREATE INDEX serving_certification_catalog_idx
                       ON serving_certification_records(catalog_fingerprint, artifact_id);

                   CREATE TABLE serving_approvals (
                       catalog_fingerprint TEXT PRIMARY KEY,
                       certification_record_id TEXT NOT NULL
                           REFERENCES serving_certification_records(certification_record_id),
                       artifact_id TEXT NOT NULL REFERENCES serving_artifacts(artifact_id),
                       state TEXT NOT NULL CHECK (state IN ('enabled', 'disabled', 'revoked')),
                       reason TEXT,
                       approved_by TEXT NOT NULL,
                       approved_at_ms INTEGER NOT NULL,
                       updated_at_ms INTEGER NOT NULL,
                       generation INTEGER NOT NULL,
                       semantic_digest TEXT NOT NULL,
                       CHECK (
                           (state = 'enabled' AND reason IS NULL)
                           OR
                           (state IN ('disabled', 'revoked') AND reason IS NOT NULL AND length(trim(reason)) > 0)
                       )
                   );
                   CREATE INDEX serving_approvals_certification_idx
                       ON serving_approvals(certification_record_id, artifact_id);

                   CREATE TABLE serving_sessions (
                       session_id TEXT PRIMARY KEY,
                       catalog_fingerprint TEXT NOT NULL,
                       approval_generation INTEGER NOT NULL,
                       state TEXT NOT NULL CHECK (
                           state IN ('active', 'termination_requested', 'finished', 'terminated')
                       ),
                       committed_token_count INTEGER NOT NULL DEFAULT 0,
                       terminal_reason TEXT,
                       created_at_ms INTEGER NOT NULL,
                       updated_at_ms INTEGER NOT NULL,
                       CHECK (
                           (state IN ('active', 'termination_requested') AND terminal_reason IS NULL)
                           OR
                           (state = 'finished' AND terminal_reason = 'completed')
                           OR
                           (state = 'terminated' AND terminal_reason = 'artifact_revoked')
                       )
                   );
                   CREATE INDEX serving_sessions_active_artifact_idx
                       ON serving_sessions(catalog_fingerprint, state);

                   CREATE TABLE serving_retained_states (
                       state_id TEXT PRIMARY KEY,
                       catalog_fingerprint TEXT NOT NULL,
                       valid INTEGER NOT NULL CHECK (valid IN (0, 1)),
                       invalidation_reason TEXT,
                       created_at_ms INTEGER NOT NULL,
                       updated_at_ms INTEGER NOT NULL,
                       CHECK (
                           (valid = 1 AND invalidation_reason IS NULL)
                           OR
                           (valid = 0 AND invalidation_reason IN ('artifact_disabled', 'artifact_revoked'))
                       )
                   );
                   CREATE INDEX serving_retained_states_artifact_idx
                       ON serving_retained_states(catalog_fingerprint, valid);
         "#,
    },
    Migration {
        version: 12,
        statements: r#"
                   ALTER TABLE profile_state
                       ADD COLUMN certification_stale_since_ms INTEGER;
        "#,
    },
    // Catalog installs, catalog download jobs and catalog-lane self-checks.
    // Additive only: no existing table is touched.
    Migration {
        version: 13,
        statements: r#"
                   CREATE TABLE catalog_installs (
                       catalog_id TEXT NOT NULL,
                       manifest_digest TEXT NOT NULL,
                       backend TEXT NOT NULL,
                       PRIMARY KEY (catalog_id, manifest_digest, backend)
                   );
                   CREATE TABLE catalog_install_members (
                       catalog_id TEXT NOT NULL,
                       manifest_digest TEXT NOT NULL,
                       backend TEXT NOT NULL,
                       path TEXT NOT NULL,
                       digest TEXT NOT NULL,
                       PRIMARY KEY (catalog_id, manifest_digest, backend, path)
                   );
                   CREATE INDEX catalog_install_members_digest_idx
                       ON catalog_install_members(digest);

                   CREATE TABLE download_acquisitions (
                       job_id TEXT NOT NULL,
                       digest TEXT NOT NULL,
                       newly_published INTEGER NOT NULL CHECK (newly_published IN (0, 1)),
                       PRIMARY KEY (job_id, digest)
                   );
                   CREATE INDEX download_acquisitions_digest_idx
                       ON download_acquisitions(digest);

                   CREATE TABLE download_key_bindings (
                       request_key TEXT PRIMARY KEY,
                       request_digest TEXT NOT NULL,
                       job_id TEXT NOT NULL
                   );
                   CREATE INDEX download_key_bindings_job_idx
                       ON download_key_bindings(job_id);

                   CREATE TABLE catalog_self_checks (
                       check_id TEXT PRIMARY KEY,
                       catalog_id TEXT NOT NULL,
                       manifest_digest TEXT NOT NULL,
                       backend TEXT NOT NULL,
                       fingerprint TEXT NOT NULL,
                       engine_identity_json TEXT NOT NULL,
                       os_build TEXT NOT NULL,
                       fixture_revision TEXT NOT NULL,
                       state TEXT NOT NULL
                           CHECK (state IN ('pending', 'running', 'passed', 'failed')),
                       generation INTEGER NOT NULL,
                       checked_at_ms INTEGER,
                       reason TEXT
                   );
                   CREATE INDEX catalog_self_checks_lane_idx
                       ON catalog_self_checks(catalog_id, backend);

                   CREATE TABLE self_check_run_seq (
                       id INTEGER PRIMARY KEY CHECK (id = 0),
                       value INTEGER NOT NULL
                   );
                   INSERT INTO self_check_run_seq (id, value) VALUES (0, 0);
        "#,
    },
    Migration {
        version: 14,
        statements: r#"
            CREATE TABLE catalog_file_verifications (
                catalog_id TEXT NOT NULL,
                manifest_digest TEXT NOT NULL,
                backend TEXT NOT NULL,
                path TEXT NOT NULL,
                expected_sha256 TEXT NOT NULL,
                device TEXT NOT NULL,
                inode TEXT NOT NULL,
                size TEXT NOT NULL,
                mtime_ns TEXT NOT NULL,
                ctime_ns TEXT NOT NULL,
                PRIMARY KEY (catalog_id, manifest_digest, backend, path)
            );
            CREATE TRIGGER catalog_member_verification_cleanup
            AFTER DELETE ON catalog_install_members BEGIN
                DELETE FROM catalog_file_verifications
                WHERE catalog_id = OLD.catalog_id AND manifest_digest = OLD.manifest_digest
                  AND backend = OLD.backend AND path = OLD.path;
            END;
        "#,
    },
    Migration {
        version: 15,
        statements: r#"
            DROP TABLE remote_url_bindings;
            DELETE FROM cert_rows
                WHERE assurance_class = 'declared' OR certification_class IN ('declared', 'remote');
            DROP INDEX cert_rows_declared_identity_v1;
            DROP INDEX cert_rows_remote_identity_v1;
            UPDATE jobs SET state = 'failed_permanent', active_attempt_id = NULL,
                terminal_at_ms = updated_ms,
                error_json = CAST('{"code":"invalid_request","class":"permanent","safe_to_retry_same_request":false,"message":"remote providers were removed; modules should call providers directly"}' AS BLOB)
                WHERE state = 'paused_needs_reauth';
            DROP INDEX jobs_resume_deadline_idx;
            ALTER TABLE jobs DROP COLUMN logical_handle;
            ALTER TABLE jobs DROP COLUMN paused_at_ms;
            ALTER TABLE jobs DROP COLUMN resume_deadline_ms;
            ALTER TABLE remote_checkpoints RENAME TO checkpoints;
            ALTER TABLE checkpoints DROP COLUMN provider_request_id;
            ALTER TABLE result_pages DROP COLUMN provider_request_ids_json;
            DROP TRIGGER remote_checkpoint_immutable;
            DROP INDEX remote_checkpoints_page_idx;
            CREATE INDEX checkpoints_page_idx ON checkpoints(request_digest, page_no);
            CREATE TRIGGER checkpoint_immutable BEFORE UPDATE ON checkpoints
            BEGIN
                SELECT RAISE(ABORT, 'checkpoint is immutable');
            END;
        "#,
    },
    Migration {
        version: 16,
        statements: r#"
            CREATE TABLE media_jobs (
                id TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                client_id TEXT NOT NULL,
                binding TEXT NOT NULL,
                lane TEXT NOT NULL,
                object_id TEXT NOT NULL,
                state TEXT NOT NULL,
                accepted_ms INTEGER NOT NULL,
                deadline_ms INTEGER,
                retained_ms INTEGER NOT NULL,
                result TEXT,
                failure TEXT,
                expired INTEGER NOT NULL DEFAULT 0,
                UNIQUE(owner, client_id)
            );
            CREATE INDEX media_jobs_lane ON media_jobs(lane, state);
            CREATE TABLE media_objects (
                id TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                mime TEXT NOT NULL,
                bytes INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                state TEXT NOT NULL,
                received INTEGER NOT NULL DEFAULT 0,
                retained_ms INTEGER NOT NULL,
                output BLOB
            );
            CREATE TABLE media_observation_pages (
                job_id TEXT NOT NULL REFERENCES media_jobs(id) ON DELETE CASCADE,
                part TEXT NOT NULL,
                page INTEGER NOT NULL,
                data BLOB NOT NULL,
                PRIMARY KEY(job_id, part, page)
            );
        "#,
    },
    Migration {
        version: 17,
        statements: r#"
            ALTER TABLE catalog_self_checks ADD COLUMN self_check_wall_ms INTEGER;
            ALTER TABLE media_jobs ADD COLUMN transcribe_measurements TEXT;
            ALTER TABLE media_jobs ADD COLUMN transcribe_fingerprint TEXT;
        "#,
    },
    Migration {
        version: 18,
        statements: crate::event_outbox::MIGRATION,
    },
    Migration {
        version: 19,
        statements: include_str!("store/retire_legacy_lanes.sql"),
    },
];

static JOB_COUNTER: AtomicU64 = AtomicU64::new(0);

// This independent inventory makes omission from the delete/import policy a
// refusal instead of silently narrowing the restore contract.
const RESTORE_APPLICATION_TABLES: &[&str] = &[
    "catalog_file_verifications",
    "catalog_install_members",
    "catalog_installs",
    "catalog_self_checks",
    "download_acquisitions",
    "download_key_bindings",
    "jobs",
    "media_jobs",
    "media_objects",
    "media_observation_pages",
    "knob_assignments",
    "models",
    "profile_rotation_events",
    "profile_state",
    "checkpoints",
    "result_pages",
];

const RESTORE_KEEP_TABLES: &[&str] = &["knob_assignments"];

// All jobs, including terminal rows, are discarded so restored request keys
// cannot point at result pages or checkpoints from an older execution world.
// Result/model/artifact/URL caches are also cleared rather than imported; every
// consumer already treats their absence as a cache miss. Catalog install
// records and self-check rows are cleared the same way: the blobs stay in the
// content-addressed cache, so the next `models.download` reuses them without
// network I/O and the self-check reruns on first use. `module_meta` is the
// one cache kept from the live store because its generation and table epoch
// fence concurrent writers and must never regress to captured values.
const RESTORE_CLEAR_TABLES: &[&str] = &[
    "media_observation_pages",
    "media_jobs",
    "media_objects",
    "checkpoints",
    "result_pages",
    "download_key_bindings",
    "download_acquisitions",
    "jobs",
    "catalog_file_verifications",
    "catalog_install_members",
    "catalog_installs",
    "catalog_self_checks",
    "profile_rotation_events",
    "profile_state",
    "models",
    "knob_assignments",
];

// `self_check_run_seq` stays with the live store for the same reason as
// `module_meta`. Each self-check run takes the next value as its generation,
// and a completion is written only if its row still carries that generation.
// Moving the counter back to a captured value would hand out generations
// already used, so a stale completion could match a newer run's row.
const RESTORE_LIVE_ONLY_TABLES: &[&str] = &[
    // Publication history and diagnostic generations belong to the live store,
    // not a captured database snapshot. Startup compares findings after restore.
    "event_outbox",
    "findings_state",
    "module_meta",
    "self_check_run_seq",
    "cortexkit_fence",
    "cortexkit_schema_version",
];

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RestoreImportReport {
    pub imported_rows: BTreeMap<String, u64>,
}

#[derive(Debug, Error)]
pub enum SynapseStoreError {
    #[error("synapse store: {0}")]
    Store(#[from] StoreError),
    #[error("synapse store json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("synapse store decode: {0}")]
    Decode(String),
    #[error(
        "synapse store schema is ahead of this binary: store recorded version {recorded}, binary chain ends at {chain_max}; \
         this schema chain rebuilds and drops tables, so an older binary cannot serve it"
    )]
    SchemaAheadOfBinary { recorded: u32, chain_max: u32 },
    #[error("request key '{request_key}' is already bound to a different digest")]
    IdempotencyConflict {
        request_key: String,
        existing_digest: String,
        requested_digest: String,
    },
    #[error("cert_rows_rebuild_failed: {0}")]
    CertRowsRebuildFailed(String),
    #[error("profile state corruption: {0}")]
    ProfileStateCorrupt(String),
    #[error("restore import refused: {0}")]
    RestoreRefused(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecommendedBatch {
    pub rows: usize,
    pub token_budget: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelCatalogEntry {
    pub model_id: String,
    pub state: String,
    pub fingerprints: Vec<Fingerprint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recommended_batch: Option<RecommendedBatch>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens_source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bucket_ladder: Option<Vec<usize>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dims: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_class: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub certified: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warm_load_cost_hint_ms: Option<f64>,
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
    pub models: Vec<ModelCatalogEntry>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ProfileState {
    pub snapshot: Option<MachineProfile>,
    pub revisioned_machine_profile_hash: Option<String>,
    pub profile_activation_epoch: Option<u64>,
    pub previous_revisioned_machine_profile_hash: Option<String>,
    pub last_rotation_reason: Option<String>,
    pub last_rotation_at_ms: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RotationLedgerEvent {
    pub event_id: String,
    pub old_revisioned_machine_profile_hash: Option<String>,
    pub new_revisioned_machine_profile_hash: String,
    pub old_profile_activation_epoch: Option<u64>,
    pub new_profile_activation_epoch: u64,
    pub changed_fields: Vec<String>,
    pub previous_snapshot: Option<MachineProfile>,
    pub current_snapshot: MachineProfile,
    pub observed_at_ms: u64,
    pub module_generation: u64,
    pub created_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ProfileActivation {
    pub state: ProfileState,
    pub rotated: bool,
    pub event: Option<RotationLedgerEvent>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StorageHealthInputs {
    pub previous_revisioned_machine_profile_hash: Option<String>,
    pub current_revisioned_machine_profile_hash: Option<String>,
    pub profile_activation_epoch: Option<u64>,
    pub last_rotation_at_ms: Option<u64>,
    pub last_rotation_reason: Option<String>,
    pub rotation_event_count: u64,
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
    pub page_count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_json: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_json: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params_json: Option<Value>,
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
    pub(crate) store: SqliteStore,
}

fn sqlite_path(descriptor: &StorageDescriptor) -> Result<PathBuf, SynapseStoreError> {
    match &descriptor.backend {
        StorageBackend::Sqlite { path } => Ok(PathBuf::from(path)),
        _ => Err(SynapseStoreError::RestoreRefused(
            "restore import requires a SQLite live store".to_string(),
        )),
    }
}

fn validate_restore_table_set(
    connection: &rusqlite::Connection,
    schema: &str,
) -> rusqlite::Result<()> {
    let expected = RESTORE_APPLICATION_TABLES
        .iter()
        .copied()
        .chain(
            RESTORE_LIVE_ONLY_TABLES
                .iter()
                .copied()
                .filter(|table| *table != "cortexkit_fence"),
        )
        .collect::<BTreeSet<_>>();

    // `schema` is selected only by this module, never from CLI input.
    let mut statement = connection.prepare(&format!(
        "SELECT name FROM {schema}.sqlite_master \
         WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ))?;
    let actual = statement
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<Result<BTreeSet<_>, _>>()?;
    let expected = expected
        .into_iter()
        .map(str::to_string)
        .collect::<BTreeSet<_>>();
    let mut expected_with_fence = expected.clone();
    expected_with_fence.insert("cortexkit_fence".to_string());
    if actual != expected && actual != expected_with_fence {
        return Err(rusqlite::Error::InvalidParameterName(format!(
            "restore table contract mismatch in {schema}: expected {expected:?} \
             with optional cortexkit_fence, found {actual:?}"
        )));
    }
    Ok(())
}

impl SynapseStore {
    #[cfg(feature = "test-support")]
    pub fn seed_v18_retirement_fixture(
        descriptor: &StorageDescriptor,
        models: &[StoredModelConfig],
        profile: &MachineProfile,
    ) -> Result<(), SynapseStoreError> {
        let store = open_sqlite(descriptor)?;
        store.migrate(NAMESPACE, &MIGRATIONS[..18])?;
        let store = Self { store };
        for model in models {
            store.upsert_model(model, 1)?;
        }
        store.observe_profile(profile, 1, 1)?;
        store.store.with_conn_fenced(|tx| {
            tx.execute_batch(include_str!("store/legacy_v18_fixture.sql"))
        })?;
        Ok(())
    }

    /// Modules that own their tables' SQL beside the feature, such as media jobs, access SQLite through this handle.
    pub fn sqlite(&self) -> &SqliteStore {
        &self.store
    }

    /// Import only durable operator intent from an Engram scratch database.
    ///
    /// Opening the capture first applies this binary's normal forward migrations
    /// and preserves the typed refusal for captures written by a newer schema.
    /// The target must be unserved; opening its store takes the same lease used
    /// by the module before any live row is changed.
    pub fn restore_from_capture(
        live_descriptor: &StorageDescriptor,
        capture_path: &Path,
    ) -> Result<RestoreImportReport, SynapseStoreError> {
        let metadata = std::fs::metadata(capture_path).map_err(|error| {
            SynapseStoreError::RestoreRefused(format!(
                "capture '{}' is not readable: {error}",
                capture_path.display()
            ))
        })?;
        if !metadata.is_file() {
            return Err(SynapseStoreError::RestoreRefused(format!(
                "capture '{}' is not a regular file",
                capture_path.display()
            )));
        }

        let live_path = sqlite_path(live_descriptor)?;
        if live_path.exists()
            && std::fs::canonicalize(capture_path).ok() == std::fs::canonicalize(&live_path).ok()
        {
            return Err(SynapseStoreError::RestoreRefused(
                "capture and live store resolve to the same file".to_string(),
            ));
        }

        let capture_descriptor = StorageDescriptor {
            module_id: "synapse-restore-capture".to_string(),
            storage_namespace: "restore-import".to_string(),
            isolation: Isolation::Module,
            backend: StorageBackend::Sqlite {
                path: capture_path.to_string_lossy().into_owned(),
            },
        };
        let migrated_capture = Self::open(&capture_descriptor)?;
        drop(migrated_capture);

        let live = Self::open(live_descriptor).map_err(|error| match error {
            SynapseStoreError::Store(StoreError::Lease(lease)) => {
                SynapseStoreError::RestoreRefused(format!(
                    "target store lease for '{}' is held; stop the module before restore: {lease}",
                    live_path.display()
                ))
            }
            other => other,
        })?;
        live.import_migrated_capture(capture_path)
    }

    fn import_migrated_capture(
        &self,
        capture_path: &Path,
    ) -> Result<RestoreImportReport, SynapseStoreError> {
        let capture_path = capture_path.to_string_lossy().into_owned();
        let report = self.store.with_conn(|connection| {
            connection.execute(
                "ATTACH DATABASE ?1 AS restore_capture",
                params![capture_path],
            )?;
            let foreign_keys_enabled: bool =
                connection.query_row("PRAGMA foreign_keys", [], |row| {
                    row.get::<_, i64>(0).map(|value| value != 0)
                })?;
            if foreign_keys_enabled {
                connection.pragma_update(None, "foreign_keys", false)?;
            }

            let operation = (|| {
                validate_restore_table_set(connection, "main")?;
                validate_restore_table_set(connection, "restore_capture")?;
                let transaction = rusqlite::Transaction::new_unchecked(
                    connection,
                    TransactionBehavior::Immediate,
                )?;

                for table in RESTORE_CLEAR_TABLES {
                    transaction.execute(&format!("DELETE FROM \"{table}\""), [])?;
                }

                let mut imported_rows = BTreeMap::new();
                for table in RESTORE_KEEP_TABLES {
                    let rows = transaction.execute(
                        &format!(
                            "INSERT INTO main.\"{table}\" SELECT * FROM restore_capture.\"{table}\""
                        ),
                        [],
                    )?;
                    imported_rows.insert((*table).to_string(), rows as u64);
                    #[cfg(test)]
                    if FAIL_RESTORE_AFTER_TABLE.with(std::cell::Cell::get) == Some(*table) {
                        return Err(rusqlite::Error::InvalidParameterName(format!(
                            "simulated restore interruption after {table}"
                        )));
                    }
                }

                transaction.commit()?;
                Ok(RestoreImportReport { imported_rows })
            })();

            let detach = connection.execute_batch("DETACH DATABASE restore_capture");
            let restore_foreign_keys = if foreign_keys_enabled {
                connection.pragma_update(None, "foreign_keys", true)
            } else {
                Ok(())
            };
            let cleanup = detach.and(restore_foreign_keys);
            match (operation, cleanup) {
                (Ok(report), Ok(())) => Ok(report),
                (Err(error), _) | (_, Err(error)) => Err(error),
            }
        })?;
        Ok(report)
    }

    pub fn open(descriptor: &StorageDescriptor) -> Result<Self, SynapseStoreError> {
        let store = open_sqlite(descriptor)?;
        let outcome = store.migrate(NAMESPACE, MIGRATIONS).map_err(|error| {
            let message = error.to_string();
            if message.contains("cert_rows") || message.contains("cert_rows_rebuilt") {
                SynapseStoreError::CertRowsRebuildFailed(message)
            } else {
                SynapseStoreError::Store(error)
            }
        })?;
        // The migrator deliberately does not refuse a store written by a longer
        // chain (that would brick binary rollbacks for additive-only schemas).
        // Ours is not additive-only: migrations have rebuilt cert_rows and
        // dropped the cutover tables, so an older binary would start serving
        // against tables it cannot query. Fail closed instead of serving.
        if outcome.store_ahead() {
            return Err(SynapseStoreError::SchemaAheadOfBinary {
                recorded: outcome.recorded,
                chain_max: outcome.chain_max,
            });
        }
        let synapse_store = Self { store };
        // Reclaim is housekeeping, never a boot precondition: VACUUM needs an
        // exclusive lock, and a backup reader holding a snapshot at this moment
        // makes it SQLITE_BUSY. Failing open on that would turn a nightly
        // capture into a module that cannot start; the next open or GC sweep
        // retries.
        if let Err(error) = synapse_store.reclaim_freelist_if_needed() {
            tracing::warn!(
                target: "maintenance",
                error = %error,
                "freelist reclaim skipped at open"
            );
        }
        Ok(synapse_store)
    }

    /// Return the current SQLite freelist page count.
    pub fn freelist_count(&self) -> Result<u64, SynapseStoreError> {
        let count = self.store.with_conn(|conn| {
            let count: i64 = conn.query_row("PRAGMA freelist_count", [], |row| row.get(0))?;
            Ok(count.max(0) as u64)
        })?;
        Ok(count)
    }

    /// Return the current SQLite total page count.
    pub fn page_count(&self) -> Result<u64, SynapseStoreError> {
        let count = self.store.with_conn(|conn| {
            let count: i64 = conn.query_row("PRAGMA page_count", [], |row| row.get(0))?;
            Ok(count.max(0) as u64)
        })?;
        Ok(count)
    }

    /// Return the current SQLite database page size in bytes.
    pub fn page_size(&self) -> Result<u64, SynapseStoreError> {
        let size = self.store.with_conn(|conn| {
            let size: i64 = conn.query_row("PRAGMA page_size", [], |row| row.get(0))?;
            Ok(size.max(0) as u64)
        })?;
        Ok(size)
    }

    #[doc(hidden)]
    pub fn pub_connection_for_test<F, T>(&self, op: F) -> Result<T, SynapseStoreError>
    where
        F: FnOnce(&rusqlite::Connection) -> Result<T, rusqlite::Error>,
    {
        let result = self.store.with_conn(|conn| op(conn))?;
        Ok(result)
    }

    /// Reclaim unused database pages by executing VACUUM when free page volume exceeds thresholds.
    ///
    /// The SQLite online backup API copies every allocated page in the database file,
    /// including free pages on the freelist. When schema migrations or large deletions
    /// leave tens of thousands of free pages behind, every subsequent backup generation
    /// wastes time and storage copying unallocated space.
    ///
    /// To avoid repeated exclusive I/O penalties on an active or data-only store, VACUUM
    /// is executed only when both of the following criteria are satisfied:
    /// 1. `freelist_count * page_size >= RECLAIM_FREELIST_MIN_BYTES` (at least 64 MiB of free pages).
    /// 2. `freelist_count >= page_count / RECLAIM_FREELIST_PAGE_RATIO_DIVISOR` (at least half of all pages are free).
    ///
    /// Concurrency and safety:
    /// - Serialized writes: This method executes through `self.store.with_conn`, acquiring
    ///   the store's internal connection mutex. This ensures no serving sessions, writes, or
    ///   other maintenance routines run concurrently.
    /// - Outside transactions: SQLite rejects `VACUUM` inside open transactions. `with_conn`
    ///   provides direct connection access outside of any transaction block.
    /// - WAL mode compatibility: In WAL mode, SQLite performs VACUUM safely into a temporary file
    ///   and replaces the database file. It cannot and does not change the journal mode.
    pub fn reclaim_freelist_if_needed(&self) -> Result<bool, SynapseStoreError> {
        let start = std::time::Instant::now();
        let vacuum_ran = self.store.with_conn(|conn| {
            let page_size: i64 = conn.query_row("PRAGMA page_size", [], |row| row.get(0))?;
            let page_count: i64 = conn.query_row("PRAGMA page_count", [], |row| row.get(0))?;
            let freelist_count: i64 =
                conn.query_row("PRAGMA freelist_count", [], |row| row.get(0))?;

            let page_size_bytes = page_size.max(0) as u64;
            let pages = page_count.max(0) as u64;
            let free_pages = freelist_count.max(0) as u64;
            let free_bytes = free_pages.saturating_mul(page_size_bytes);

            let byte_threshold_met = free_bytes >= RECLAIM_FREELIST_MIN_BYTES;
            let ratio_threshold_met = free_pages >= pages / RECLAIM_FREELIST_PAGE_RATIO_DIVISOR;

            if byte_threshold_met && ratio_threshold_met {
                conn.execute_batch("VACUUM; PRAGMA wal_checkpoint(TRUNCATE);")?;
                let after_page_count: i64 =
                    conn.query_row("PRAGMA page_count", [], |row| row.get(0))?;
                let after_freelist_count: i64 =
                    conn.query_row("PRAGMA freelist_count", [], |row| row.get(0))?;
                Ok(Some((
                    pages,
                    after_page_count.max(0) as u64,
                    free_pages,
                    after_freelist_count.max(0) as u64,
                )))
            } else {
                Ok(None)
            }
        })?;

        if let Some((before_pages, after_pages, before_free, after_free)) = vacuum_ran {
            let wall_ms = start.elapsed().as_millis() as u64;
            tracing::info!(
                target: "maintenance",
                before_page_count = before_pages,
                after_page_count = after_pages,
                before_freelist_count = before_free,
                after_freelist_count = after_free,
                wall_ms,
                "vacuum reclaimed free pages"
            );
            Ok(true)
        } else {
            Ok(false)
        }
    }

    // Staged storage API consumed by the epic's runtime slice; remove this allow there.
    #[allow(dead_code)]
    pub fn open_with_profile(
        descriptor: &StorageDescriptor,
        profile: &MachineProfile,
        observed_at_ms: u64,
        module_generation: u64,
    ) -> Result<Self, SynapseStoreError> {
        let store = Self::open(descriptor)?;
        store.observe_profile(profile, observed_at_ms, module_generation)?;
        Ok(store)
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
        let models = self
            .catalog_models()?
            .into_iter()
            .map(|model| {
                let dtype = model
                    .owned_dtype
                    .clone()
                    .or_else(|| match model.engine.as_str() {
                        "owned-cuda" => Some("f16".to_string()),
                        "ort" => Some("f32".to_string()),
                        _ => None,
                    });
                let device_class = match model.engine.as_str() {
                    "owned-metal" => Some("metal".to_string()),
                    "owned-cuda" => Some("cuda".to_string()),
                    "ort" | "llama" => Some("cpu".to_string()),
                    _ => None,
                };
                let certified = if model.engine == "llama" && model.task == "generate" {
                    None
                } else {
                    Some(false)
                };
                ModelCatalogEntry {
                    model_id: model.model_id,
                    state: "unloaded".to_string(),
                    fingerprints: vec![model.fingerprint],
                    recommended_batch: None,
                    max_tokens: Some(model.max_tokens),
                    max_tokens_source: Some("catalog_unloaded".to_string()),
                    bucket_ladder: None,
                    dims: None,
                    dtype,
                    device_class,
                    certified,
                    warm_load_cost_hint_ms: None,
                }
            })
            .collect();
        Ok(CatalogSnapshot { models })
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

    pub fn profile_state(&self) -> Result<ProfileState, SynapseStoreError> {
        let state = self.store.with_conn(|conn| {
            conn.query_row(
                "SELECT snapshot_json, revisioned_machine_profile_hash,
                         profile_activation_epoch, previous_revisioned_machine_profile_hash,
                          last_rotation_reason, last_rotation_at_ms
                  FROM profile_state WHERE id = 0",
                [],
                profile_state_from_row,
            )
        })?;
        validate_profile_state(&state)?;
        Ok(state)
    }

    /// Observe and durably activate a collected profile. The first observation
    /// creates epoch 1; a changed revisioned hash performs the snapshot, epoch,
    /// and rotation event writes in one fenced transaction.
    pub fn observe_profile(
        &self,
        profile: &MachineProfile,
        observed_at_ms: u64,
        module_generation: u64,
    ) -> Result<ProfileActivation, SynapseStoreError> {
        let current_hash = profile.revisioned_hash();
        let current_snapshot_json = serde_json::to_string(profile)?;
        let activation = self.store.with_conn_fenced(|tx| {
            // Restore import clears the machine profile. Recreate the singleton
            // on first observation rather than use the captured machine identity.
            tx.execute("INSERT OR IGNORE INTO profile_state (id) VALUES (0)", [])?;
            let persisted = tx.query_row(
                "SELECT snapshot_json, revisioned_machine_profile_hash,
                         profile_activation_epoch, previous_revisioned_machine_profile_hash,
                          last_rotation_reason, last_rotation_at_ms
                  FROM profile_state WHERE id = 0",
                [],
                profile_state_from_row,
            )?;
            validate_profile_state(&persisted).map_err(to_sql_error)?;
            let Some(old_hash) = persisted.revisioned_machine_profile_hash.clone() else {
                if persisted.snapshot.is_some() || persisted.profile_activation_epoch.is_some() {
                    return Err(rusqlite::Error::InvalidQuery);
                }
                tx.execute(
                    "UPDATE profile_state SET snapshot_json = ?1,
                         revisioned_machine_profile_hash = ?2,
                         profile_activation_epoch = 1
                     WHERE id = 0 AND revisioned_machine_profile_hash IS NULL
                       AND profile_activation_epoch IS NULL",
                    params![&current_snapshot_json, &current_hash],
                )?;
                let state = ProfileState {
                    snapshot: Some(profile.clone()),
                    revisioned_machine_profile_hash: Some(current_hash.clone()),
                    profile_activation_epoch: Some(1),
                    previous_revisioned_machine_profile_hash: None,
                    last_rotation_reason: None,
                    last_rotation_at_ms: None,
                };
                return Ok(ProfileActivation {
                    state,
                    rotated: false,
                    event: None,
                });
            };
            let old_epoch = persisted
                .profile_activation_epoch
                .ok_or_else(|| rusqlite::Error::InvalidQuery)?;
            if old_epoch == 0 || !is_digest(&old_hash) {
                return Err(rusqlite::Error::InvalidQuery);
            }
            if old_hash == current_hash {
                return Ok(ProfileActivation {
                    state: persisted,
                    rotated: false,
                    event: None,
                });
            }

            let previous_snapshot = persisted.snapshot.clone();
            let changed_fields = previous_snapshot
                .as_ref()
                .map(|previous| changed_profile_fields(previous, profile))
                .unwrap_or_else(|| vec!["unknown_previous_snapshot".to_string()]);
            let reason = if previous_snapshot.is_some() {
                changed_fields.join(",")
            } else {
                "unknown_previous_snapshot".to_string()
            };
            let next_epoch = old_epoch
                .checked_add(1)
                .ok_or_else(|| rusqlite::Error::InvalidQuery)?;
            let update_count = tx.execute(
                "UPDATE profile_state SET snapshot_json = ?1,
                     revisioned_machine_profile_hash = ?2,
                     profile_activation_epoch = ?3,
                     previous_revisioned_machine_profile_hash = ?4,
                      last_rotation_reason = ?5,
                       last_rotation_at_ms = ?6
                  WHERE id = 0 AND revisioned_machine_profile_hash = ?7
                   AND profile_activation_epoch = ?8",
                params![
                    &current_snapshot_json,
                    &current_hash,
                    next_epoch as i64,
                    &old_hash,
                    &reason,
                    observed_at_ms as i64,
                    &old_hash,
                    old_epoch as i64,
                ],
            )?;
            if update_count != 1 {
                let adopted = tx.query_row(
                    "SELECT snapshot_json, revisioned_machine_profile_hash,
                         profile_activation_epoch, previous_revisioned_machine_profile_hash,
                          last_rotation_reason, last_rotation_at_ms
                  FROM profile_state WHERE id = 0",
                    [],
                    profile_state_from_row,
                )?;
                if adopted.revisioned_machine_profile_hash.as_deref() == Some(current_hash.as_str())
                {
                    return Ok(ProfileActivation {
                        state: adopted,
                        rotated: false,
                        event: None,
                    });
                }
                return Err(rusqlite::Error::InvalidQuery);
            }
            let event_id = rotation_event_id(&current_hash, next_epoch);
            let previous_snapshot_json = previous_snapshot
                .as_ref()
                .map(serde_json::to_string)
                .transpose()
                .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;
            let changed_fields_json = serde_json::to_string(&changed_fields)
                .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;
            tx.execute(
                "INSERT INTO profile_rotation_events (
                     event_id, old_revisioned_machine_profile_hash,
                     new_revisioned_machine_profile_hash, old_profile_activation_epoch,
                     new_profile_activation_epoch, changed_fields_json,
                     previous_snapshot_json, current_snapshot_json, observed_at_ms,
                     module_generation, created_at_ms
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?9)",
                params![
                    &event_id,
                    &old_hash,
                    &current_hash,
                    old_epoch as i64,
                    next_epoch as i64,
                    changed_fields_json,
                    previous_snapshot_json,
                    &current_snapshot_json,
                    observed_at_ms as i64,
                    module_generation as i64,
                ],
            )?;
            let event = rotation_event_tx(tx, &event_id)?;
            let state = ProfileState {
                snapshot: Some(profile.clone()),
                revisioned_machine_profile_hash: Some(current_hash.clone()),
                profile_activation_epoch: Some(next_epoch),
                previous_revisioned_machine_profile_hash: Some(old_hash),
                last_rotation_reason: Some(reason),
                last_rotation_at_ms: Some(observed_at_ms),
            };
            Ok(ProfileActivation {
                state,
                rotated: true,
                event: Some(event),
            })
        })?;
        if activation.rotated {
            if let Some(event) = activation.event.as_ref() {
                if event.changed_fields.iter().any(|field| field == "os_build") {
                    tracing::warn!(
                        target: "cert",
                        epoch = event.new_profile_activation_epoch,
                        changed_fields = %event.changed_fields.join(","),
                        "machine profile rotation"
                    );
                } else {
                    tracing::info!(
                        target: "cert",
                        epoch = event.new_profile_activation_epoch,
                        changed_fields = %event.changed_fields.join(","),
                        "machine profile rotation"
                    );
                }
            }
        }
        Ok(activation)
    }

    pub fn activate_profile(
        &self,
        profile: &MachineProfile,
        observed_at_ms: u64,
        module_generation: u64,
    ) -> Result<ProfileActivation, SynapseStoreError> {
        self.observe_profile(profile, observed_at_ms, module_generation)
    }

    pub fn rotation_events(&self) -> Result<Vec<RotationLedgerEvent>, SynapseStoreError> {
        let ids = self.store.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT event_id FROM profile_rotation_events
                 ORDER BY new_profile_activation_epoch ASC, event_id ASC",
            )?;
            let ids = stmt
                .query_map([], |row| row.get::<_, String>(0))?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(ids)
        })?;
        ids.iter()
            .map(|event_id| {
                self.rotation_event(event_id)?.ok_or_else(|| {
                    SynapseStoreError::Decode(format!("missing rotation event {event_id}"))
                })
            })
            .collect()
    }

    pub fn rotation_event(
        &self,
        event_id: &str,
    ) -> Result<Option<RotationLedgerEvent>, SynapseStoreError> {
        Ok(self
            .store
            .with_conn(|conn| rotation_event_conn(conn, event_id))?)
    }

    /// Return persisted values used to add storage information to health responses,
    /// without changing the existing machine-profile hash calculation or its
    /// compatibility behavior.
    pub fn storage_health_inputs(&self) -> Result<StorageHealthInputs, SynapseStoreError> {
        let state = self.profile_state()?;
        let rotation_event_count = self.store.with_conn(|conn| {
            conn.query_row("SELECT COUNT(*) FROM profile_rotation_events", [], |row| {
                row.get::<_, i64>(0)
            })
        })? as u64;
        Ok(StorageHealthInputs {
            previous_revisioned_machine_profile_hash: state
                .previous_revisioned_machine_profile_hash,
            current_revisioned_machine_profile_hash: state.revisioned_machine_profile_hash,
            profile_activation_epoch: state.profile_activation_epoch,
            last_rotation_at_ms: state.last_rotation_at_ms,
            last_rotation_reason: state.last_rotation_reason,
            rotation_event_count,
        })
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

    #[allow(clippy::too_many_arguments)]
    pub fn admit_job(
        &self,
        request_key: &str,
        request_digest: &str,
        kind: &str,
        module_generation: u64,
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
        let admission = self.store.with_conn_fenced(|tx| {
            expire_jobs_tx(tx, now_ms as i64)?;
            purge_retained_jobs_tx(tx, now_ms as i64)?;
            if let Some(existing) = job_by_request_key_tx(tx, &request_key)? {
                if existing.request_digest != request_digest {
                    return Ok(TxAdmission::Conflict(existing.request_digest));
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
                     params_json,
                     result_json, error_json, page_count
                 ) VALUES (
                     ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7, ?8, ?9, NULL, NULL,
                     ?10, NULL, NULL, ?11
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
            // Pages belong to the digest, so an attempt starts after every page
            // another job for the same digest committed since this one was admitted.
            let changed = tx.execute(
                "UPDATE jobs SET state = ?1, updated_ms = ?2, active_attempt_id = ?3,
                     page_count = (SELECT COUNT(1) FROM result_pages p
                                   WHERE p.request_digest = jobs.request_digest)
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
        self.finish_job(job_id, result_json, now_ms).map(drop)
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
        self.store.with_conn_fenced(|tx| {
            // A job past its execution TTL fails here even if no poll or sweep
            // has expired it yet, so a late attempt cannot publish pages.
            expire_job_tx(tx, job_id, now_ms as i64)?;
            let (request_digest, visible_page_count, state, kind): (String, i64, String, String) =
                tx.query_row(
                    "SELECT request_digest, page_count, state, kind FROM jobs WHERE job_id = ?1",
                    params![job_id],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
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
                     request_digest, page_no, page_json, committed_at
                 ) VALUES (?1, ?2, ?3, ?4)",
                params![request_digest, page_no as i64, page_json, now_ms as i64,],
            )?;
            if kind == "embed.batch" {
                // Page rows carry no unique job id, so enqueue here rather than in a page trigger.
                crate::event_outbox::enqueue_job_page(tx, job_id, page_no, &state, now_ms)?;
            }
            for item in items {
                tx.execute(
                    "INSERT INTO checkpoints (
                         request_digest, item_id, result, page_no,
                         committed_at
                     ) VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![
                        request_digest,
                        item.item_id,
                        item.result,
                        page_no as i64,
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

    /// Marks an active job done. Returns `false`, changing nothing, when the
    /// job already ended, including by passing its execution TTL: a recorded
    /// terminal outcome is never replaced.
    pub fn finish_job(
        &self,
        job_id: &str,
        result_json: &Value,
        now_ms: u64,
    ) -> Result<bool, SynapseStoreError> {
        let result_bytes = serde_json::to_vec(result_json)?;
        let finished = self.store.with_conn_fenced(|tx| {
            expire_job_tx(tx, job_id, now_ms as i64)?;
            let changed = tx.execute(
                "UPDATE jobs SET state = ?1, updated_ms = ?2, terminal_at_ms = ?2,
                     execution_expires_ms = NULL, active_attempt_id = NULL,
                     result_json = ?3, error_json = NULL
                 WHERE job_id = ?4 AND state IN (?5, ?6)",
                params![
                    JOB_STATE_DONE,
                    now_ms as i64,
                    result_bytes,
                    job_id,
                    JOB_STATE_QUEUED,
                    JOB_STATE_RUNNING,
                ],
            )?;
            Ok(changed > 0)
        })?;
        Ok(finished)
    }

    /// Records the failure of an active job. A job that already ended,
    /// including by passing its execution TTL, keeps its recorded outcome.
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
            expire_job_tx(tx, job_id, now_ms as i64)?;
            tx.execute(
                "UPDATE jobs SET state = ?1, updated_ms = ?2, terminal_at_ms = ?2,
                     execution_expires_ms = NULL, active_attempt_id = NULL,
                     error_json = ?3, result_json = NULL
                 WHERE job_id = ?4 AND state IN (?5, ?6)",
                params![
                    state,
                    now_ms as i64,
                    error_bytes,
                    job_id,
                    JOB_STATE_QUEUED,
                    JOB_STATE_RUNNING,
                ],
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
                "UPDATE jobs SET state = ?1, updated_ms = ?2, terminal_at_ms = ?2,
                     execution_expires_ms = NULL, active_attempt_id = NULL,
                     error_json = ?3, result_json = NULL
                 WHERE module_generation < ?4 AND state IN (?5, ?6) AND kind <> ?7",
                params![
                    JOB_STATE_FAILED_TRANSIENT,
                    now_ms as i64,
                    error_bytes,
                    current_generation as i64,
                    JOB_STATE_QUEUED,
                    JOB_STATE_RUNNING,
                    // Download jobs fail at startup through
                    // fail_interrupted_download_jobs, which also runs their
                    // staging and blob cleanup.
                    DOWNLOAD_JOB_KIND,
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
                "SELECT COUNT(1) FROM checkpoints WHERE request_digest = ?1",
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
                "SELECT item_id FROM checkpoints
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

// ---------------------------------------------------------------------------
// Catalog installs and catalog download jobs
// ---------------------------------------------------------------------------

/// `jobs.kind` of a catalog `models.download` job.
pub const DOWNLOAD_JOB_KIND: &str = "models.download";
/// Download job states, stored in `jobs.state`. The first three are
/// non-terminal; a job leaves them exactly once, through the commit, cancel or
/// failure transaction.
#[cfg_attr(not(test), allow(dead_code))]
pub const DOWNLOAD_STATE_QUEUED: &str = "queued";
#[cfg_attr(not(test), allow(dead_code))]
pub const DOWNLOAD_STATE_DOWNLOADING: &str = "downloading";
#[cfg_attr(not(test), allow(dead_code))]
pub const DOWNLOAD_STATE_VERIFYING: &str = "verifying";
#[cfg_attr(not(test), allow(dead_code))]
pub const DOWNLOAD_STATE_COMMITTED: &str = "committed";
#[cfg_attr(not(test), allow(dead_code))]
pub const DOWNLOAD_STATE_FAILED: &str = "failed";
#[cfg_attr(not(test), allow(dead_code))]
pub const DOWNLOAD_STATE_CANCELLED: &str = "cancelled";
/// The non-terminal download states as an SQL list, for `state IN ...`.
#[cfg_attr(not(test), allow(dead_code))]
const DOWNLOAD_NON_TERMINAL_SQL: &str = "('queued', 'downloading', 'verifying')";

/// Engines a free-form embed or rerank registration may name: the owned
/// accelerated engines, none of which has a CPU execution mode. Embedding and
/// reranking serve only on owned accelerated lanes, so every other engine,
/// including the CPU-bound `ort` and `llama`, is refused for those tasks.
/// The direct-ANE worker is named through its shared constant so this list
/// can't drift from the identity the worker reports.
#[cfg_attr(not(test), allow(dead_code))]
pub const OWNED_EMBED_RERANK_ENGINES: &[&str] = &[
    "owned-metal",
    synapse_core::ANE_DIRECT_WORKER_ENGINE,
    "owned-cuda",
    "owned-vulkan",
    #[cfg(feature = "test-support")]
    "test-deterministic",
];

// Normal builds must never admit the inference double as a serving engine.
#[cfg(not(feature = "test-support"))]
const _: () = assert!(
    !list_contains(OWNED_EMBED_RERANK_ENGINES, "test-deterministic"),
    "production embed/rerank allow-list contains test-deterministic"
);

#[cfg(not(feature = "test-support"))]
const fn list_contains(list: &[&str], needle: &str) -> bool {
    let mut i = 0;
    while i < list.len() {
        let left = list[i].as_bytes();
        let right = needle.as_bytes();
        if left.len() == right.len() {
            let mut j = 0;
            while j < left.len() && left[j] == right[j] {
                j += 1;
            }
            if j == left.len() {
                return true;
            }
        }
        i += 1;
    }
    false
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn is_download_non_terminal(state: &str) -> bool {
    matches!(
        state,
        DOWNLOAD_STATE_QUEUED | DOWNLOAD_STATE_DOWNLOADING | DOWNLOAD_STATE_VERIFYING
    )
}

/// The content-addressed model cache as the catalog transactions see it.
///
/// Catalog install members and download acquisitions root blobs that live on
/// disk, outside SQLite, so the cleanup and quarantine transactions touch both.
/// The store calls these methods while its transaction is open, and an error
/// rolls the transaction back. File deletions cannot be rolled back, so every
/// method must be safe to repeat: removing something already gone succeeds.
///
/// Digests are 64 lowercase hex characters with no `sha256:` prefix.
pub trait CatalogBlobCache {
    /// Size in bytes of the published blob, or `None` when it is absent.
    fn blob_size(&self, digest: &str) -> std::io::Result<Option<u64>>;
    /// Whether any `cache.pin` holds the blob.
    fn is_pinned(&self, digest: &str) -> std::io::Result<bool>;
    /// Drop every `cache.pin` on the blob.
    fn remove_pins(&self, digest: &str) -> std::io::Result<()>;
    /// Unlink the blob and its cache metadata; returns the bytes unlinked.
    fn delete_blob(&self, digest: &str) -> std::io::Result<u64>;
    /// Delete the job's staging directory and everything in it.
    fn delete_staging(&self, job_id: &str) -> std::io::Result<()>;
}

/// One file of an installed catalog backend: the path the catalog gives it and
/// the digest of the cached blob holding its bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogInstallMember {
    pub path: String,
    pub digest: String,
}

/// The install record of one catalog backend for one manifest, with its member
/// rows. Member rows are the only source for which blobs an install references.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogInstallRecord {
    pub catalog_id: String,
    pub manifest_digest: String,
    pub backend: String,
    pub members: Vec<CatalogInstallMember>,
}

/// A blob a download job published or reused and has not yet committed.
/// `newly_published` is true when the job (or, after a hand-off, another job
/// whose cleanup left it here) created the blob, so the job's cleanup may
/// delete it; a pre-existing blob is never deleted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DownloadAcquisition {
    pub job_id: String,
    pub digest: String,
    pub newly_published: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DownloadKeyBinding {
    pub request_key: String,
    pub request_digest: String,
    pub job_id: String,
}

/// Inputs to [`SynapseStore::admit_download_job`].
#[derive(Clone, Debug)]
pub struct DownloadJobRequest<'a> {
    /// The caller's request key, already defaulted when it was blank.
    pub request_key: &'a str,
    /// The digest of `{catalog_id, manifest_digest}`. All downloads of one
    /// entry manifest share it, which is how they share one job.
    pub request_digest: &'a str,
    pub module_generation: u64,
    pub params_json: &'a Value,
    pub now_ms: u64,
    pub result_retention_ttl_ms: u64,
    /// Whether the entry is complete right now. A key bound to a committed job
    /// returns that job only while this holds; otherwise it binds to a fresh
    /// job that repairs or finishes the install.
    pub entry_complete: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DownloadAdmission {
    /// A new job was minted and the key bound to it.
    Created(JobRecord),
    /// The key now refers to an existing job.
    Attached(JobRecord),
}

impl DownloadAdmission {
    #[must_use]
    pub fn record(&self) -> &JobRecord {
        match self {
            DownloadAdmission::Created(record) | DownloadAdmission::Attached(record) => record,
        }
    }
}

/// Result of [`SynapseStore::record_download_publication`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DownloadPublication {
    /// The acquisition row is written; the caller may publish the blob.
    Proceed { newly_published: bool },
    /// The job is already terminal; nothing was written and the caller must
    /// not publish.
    Stopped(Box<JobRecord>),
}

/// Result of [`SynapseStore::root_reused_download_blob`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DownloadReuse {
    /// The blob is present with its size and now rooted by this job.
    Rooted,
    /// The blob is absent or has the wrong size; nothing was written and the
    /// caller must fetch the file.
    Absent,
    /// The job is already terminal; nothing was written.
    Stopped(Box<JobRecord>),
}

/// How a cancel or failure ends a download job.
#[derive(Clone, Copy, Debug)]
pub enum DownloadEnd<'a> {
    Cancelled,
    Failed { error_json: &'a Value },
}

/// What a cancel or failure cleanup did with the job's acquired blobs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DownloadCleanup {
    /// Newly published blobs nothing else referenced, now unlinked.
    pub deleted_blobs: Vec<String>,
    /// Bytes those unlinks freed.
    pub freed_bytes: u64,
    /// Newly published blobs another job had also acquired; that job's
    /// acquisition row now carries `newly_published`, so its own cleanup
    /// deletes the blob if it too ends without committing.
    pub handed_off: Vec<String>,
    /// Newly published blobs kept because a member row, a `cache.pin` or a
    /// free-form registration references them.
    pub kept: Vec<String>,
}

/// Outcome of a compare-and-swap that ends a download job.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DownloadTransition {
    /// This call moved the job to its terminal state.
    Applied {
        record: JobRecord,
        cleanup: DownloadCleanup,
    },
    /// The job was already terminal, so the call changed nothing. Carries the
    /// job as it stands, or `None` when no download job has that id.
    Unchanged(Option<JobRecord>),
}

/// The key of one catalog install record.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CatalogInstallKey {
    pub catalog_id: String,
    pub manifest_digest: String,
    pub backend: String,
}

/// What [`SynapseStore::quarantine_catalog_blob`] removed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct QuarantineReport {
    /// Install records deleted because one of their members named the blob.
    pub removed_installs: Vec<CatalogInstallKey>,
    /// Non-terminal download jobs failed because they had acquired the blob.
    pub failed_jobs: Vec<String>,
    /// Bytes unlinked: the quarantined blob plus the failed jobs' cleanup.
    pub freed_bytes: u64,
}

/// What [`SynapseStore::reconcile_persisted_registrations`] decided for the
/// persisted `models` rows at startup.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RegistrationReconciliation {
    /// Rows the runtime may register, in `catalog_models` order.
    pub registrable: Vec<StoredModelConfig>,
    /// `(model_id, engine)` of embed/rerank rows deleted because their engine
    /// is outside [`OWNED_EMBED_RERANK_ENGINES`].
    pub deleted: Vec<(String, String)>,
    /// Blob digests whose `cache.pin` was dropped with a deleted row.
    pub unpinned_digests: Vec<String>,
    /// Rows kept in the table but not registered because their `model_id` is
    /// catalog-reserved.
    pub skipped_reserved: Vec<String>,
}

// The module's download, serving and startup paths call these; until they are
// wired in, a non-test build sees some of them unused.
#[cfg_attr(not(test), allow(dead_code))]
impl SynapseStore {
    /// Bind a `models.download` request key to a job, minting the job when
    /// needed. At most one non-terminal download job exists per request digest
    /// (that is, per catalog entry manifest), and every key for that digest
    /// attaches to it. A key keeps returning its committed job only while the
    /// entry is complete. A key bound to a different digest is an
    /// [`SynapseStoreError::IdempotencyConflict`].
    ///
    /// The job row gets the internal request key `models.download:<job_id>`,
    /// because `jobs.request_key` lookups belong to `admit_job`, and no
    /// execution deadline: download jobs end only through their own
    /// transactions.
    pub fn admit_download_job(
        &self,
        request: &DownloadJobRequest<'_>,
    ) -> Result<DownloadAdmission, SynapseStoreError> {
        enum TxAdmission {
            Ready(Box<DownloadAdmission>),
            Conflict(String),
        }

        let params_bytes = serde_json::to_vec(request.params_json)?;
        let admission = self.store.with_conn_fenced(|tx| {
            let now_ms = request.now_ms as i64;
            expire_jobs_tx(tx, now_ms)?;
            purge_retained_jobs_tx(tx, now_ms)?;
            if let Some(binding) = download_key_binding_tx(tx, request.request_key)? {
                if binding.request_digest != request.request_digest {
                    return Ok(TxAdmission::Conflict(binding.request_digest));
                }
                if let Some(job) = job_by_id_tx(tx, &binding.job_id)? {
                    let reusable = is_download_non_terminal(&job.state)
                        || (job.state == DOWNLOAD_STATE_COMMITTED && request.entry_complete);
                    if reusable {
                        return Ok(TxAdmission::Ready(Box::new(DownloadAdmission::Attached(
                            job,
                        ))));
                    }
                }
            }

            let admission = match active_download_job_tx(tx, request.request_digest)? {
                Some(job) => DownloadAdmission::Attached(job),
                None => {
                    let job_id = new_job_id(
                        request.request_key,
                        request.module_generation,
                        request.now_ms,
                    );
                    tx.execute(
                        "INSERT INTO jobs (
                             job_id, request_key, request_digest, kind, module_generation,
                             state, created_ms, updated_ms, execution_expires_ms,
                             result_retention_ttl_ms, terminal_at_ms, active_attempt_id,
                             params_json,
                             result_json, error_json, page_count
                         ) VALUES (
                             ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7, NULL, ?8, NULL, NULL,
                             ?9, NULL, NULL, 0
                         )",
                        params![
                            job_id,
                            format!("{DOWNLOAD_JOB_KIND}:{job_id}"),
                            request.request_digest,
                            DOWNLOAD_JOB_KIND,
                            request.module_generation as i64,
                            DOWNLOAD_STATE_QUEUED,
                            now_ms,
                            request.result_retention_ttl_ms as i64,
                            &params_bytes,
                        ],
                    )?;
                    DownloadAdmission::Created(
                        job_by_id_tx(tx, &job_id)?.expect("inserted download job is readable"),
                    )
                }
            };
            tx.execute(
                "INSERT INTO download_key_bindings (request_key, request_digest, job_id)
                 VALUES (?1, ?2, ?3)
                 ON CONFLICT(request_key) DO UPDATE SET
                     request_digest = excluded.request_digest,
                     job_id = excluded.job_id",
                params![
                    request.request_key,
                    request.request_digest,
                    admission.record().job_id,
                ],
            )?;
            Ok(TxAdmission::Ready(Box::new(admission)))
        })?;
        match admission {
            TxAdmission::Ready(admission) => Ok(*admission),
            TxAdmission::Conflict(existing_digest) => Err(SynapseStoreError::IdempotencyConflict {
                request_key: request.request_key.to_string(),
                existing_digest,
                requested_digest: request.request_digest.to_string(),
            }),
        }
    }

    /// Move a non-terminal download job to `downloading` or `verifying`.
    /// Returns false when the job is already terminal.
    pub fn advance_download_job(
        &self,
        job_id: &str,
        state: &str,
        now_ms: u64,
    ) -> Result<bool, SynapseStoreError> {
        if state != DOWNLOAD_STATE_DOWNLOADING && state != DOWNLOAD_STATE_VERIFYING {
            return Err(SynapseStoreError::Decode(format!(
                "download job progress cannot move to state '{state}'"
            )));
        }
        let changed = self.store.with_conn_fenced(|tx| {
            tx.execute(
                &format!(
                    "UPDATE jobs SET state = ?1, updated_ms = ?2
                     WHERE job_id = ?3 AND kind = ?4 AND state IN {DOWNLOAD_NON_TERMINAL_SQL}"
                ),
                params![state, now_ms as i64, job_id, DOWNLOAD_JOB_KIND],
            )
        })?;
        Ok(changed > 0)
    }

    /// Write the acquisition row for a blob the job is about to publish.
    ///
    /// The caller holds the job's publish lock, which the commit, cancel and
    /// failure paths also take, and publishes only on `Proceed`. So no blob is
    /// published after the job is terminal, and every published blob has a row
    /// its cleanup will see. `newly_published` is true only when the blob is
    /// absent now; a row this job already holds keeps `newly_published` if
    /// either write set it.
    pub fn record_download_publication(
        &self,
        job_id: &str,
        digest: &str,
        cache: &dyn CatalogBlobCache,
    ) -> Result<DownloadPublication, SynapseStoreError> {
        let digest = normalize_blob_digest(digest)?;
        let publication = self.store.with_conn_fenced(|tx| {
            if let Some(stopped) = stopped_download_job_tx(tx, job_id)? {
                return Ok(DownloadPublication::Stopped(Box::new(stopped)));
            }
            let newly_published = cache
                .blob_size(&digest)
                .map_err(|error| cache_error("stat blob", &digest, error))?
                .is_none();
            tx.execute(
                "INSERT INTO download_acquisitions (job_id, digest, newly_published)
                 VALUES (?1, ?2, ?3)
                 ON CONFLICT(job_id, digest) DO UPDATE SET
                     newly_published = MAX(newly_published, excluded.newly_published)",
                params![job_id, digest, newly_published],
            )?;
            Ok(DownloadPublication::Proceed { newly_published })
        })?;
        Ok(publication)
    }

    /// Root an already-cached blob for this job instead of fetching it.
    ///
    /// In one transaction the job confirms the blob is present with
    /// `size_bytes` and writes its acquisition row with `newly_published`
    /// false, so another job's cleanup cannot delete the blob before this job
    /// commits or ends, and this job's own cleanup never deletes it.
    pub fn root_reused_download_blob(
        &self,
        job_id: &str,
        digest: &str,
        size_bytes: u64,
        cache: &dyn CatalogBlobCache,
    ) -> Result<DownloadReuse, SynapseStoreError> {
        let digest = normalize_blob_digest(digest)?;
        let reuse = self.store.with_conn_fenced(|tx| {
            if let Some(stopped) = stopped_download_job_tx(tx, job_id)? {
                return Ok(DownloadReuse::Stopped(Box::new(stopped)));
            }
            let present = cache
                .blob_size(&digest)
                .map_err(|error| cache_error("stat blob", &digest, error))?;
            if present != Some(size_bytes) {
                return Ok(DownloadReuse::Absent);
            }
            tx.execute(
                "INSERT INTO download_acquisitions (job_id, digest, newly_published)
                 VALUES (?1, ?2, 0)
                 ON CONFLICT(job_id, digest) DO NOTHING",
                params![job_id, digest],
            )?;
            Ok(DownloadReuse::Rooted)
        })?;
        Ok(reuse)
    }

    /// Commit a download job: compare-and-swap from a non-terminal state to
    /// `committed`, in one transaction that writes an install record with its
    /// member rows for every backend in `installs` (the job's target set,
    /// whether or not any byte was fetched), replacing any earlier member set
    /// for the same record, and deletes the job's acquisition rows, whose
    /// blobs the member rows now root. Commit never runs cleanup. If the job
    /// is already terminal nothing changes.
    pub fn commit_download_job(
        &self,
        job_id: &str,
        installs: &[CatalogInstallRecord],
        now_ms: u64,
    ) -> Result<DownloadTransition, SynapseStoreError> {
        let installs = installs
            .iter()
            .map(normalize_install_record)
            .collect::<Result<Vec<_>, _>>()?;
        let transition = self.store.with_conn_fenced(|tx| {
            let changed = tx.execute(
                &format!(
                    "UPDATE jobs SET state = ?1, updated_ms = ?2, terminal_at_ms = ?2,
                         execution_expires_ms = NULL, active_attempt_id = NULL,
                         error_json = NULL
                     WHERE job_id = ?3 AND kind = ?4 AND state IN {DOWNLOAD_NON_TERMINAL_SQL}"
                ),
                params![
                    DOWNLOAD_STATE_COMMITTED,
                    now_ms as i64,
                    job_id,
                    DOWNLOAD_JOB_KIND
                ],
            )?;
            if changed == 0 {
                return Ok(DownloadTransition::Unchanged(download_job_tx(tx, job_id)?));
            }
            for install in &installs {
                tx.execute(
                    "DELETE FROM catalog_install_members
                     WHERE catalog_id = ?1 AND manifest_digest = ?2 AND backend = ?3",
                    params![install.catalog_id, install.manifest_digest, install.backend],
                )?;
                tx.execute(
                    "INSERT OR IGNORE INTO catalog_installs (catalog_id, manifest_digest, backend)
                     VALUES (?1, ?2, ?3)",
                    params![install.catalog_id, install.manifest_digest, install.backend],
                )?;
                for member in &install.members {
                    tx.execute(
                        "INSERT INTO catalog_install_members (
                             catalog_id, manifest_digest, backend, path, digest
                         ) VALUES (?1, ?2, ?3, ?4, ?5)",
                        params![
                            install.catalog_id,
                            install.manifest_digest,
                            install.backend,
                            member.path,
                            member.digest,
                        ],
                    )?;
                }
            }
            tx.execute(
                "DELETE FROM download_acquisitions WHERE job_id = ?1",
                params![job_id],
            )?;
            Ok(DownloadTransition::Applied {
                record: job_by_id_tx(tx, job_id)?.expect("committed download job is readable"),
                cleanup: DownloadCleanup::default(),
            })
        })?;
        Ok(transition)
    }

    /// Cancel or fail a download job: compare-and-swap from a non-terminal
    /// state, and in the same transaction run the job's cleanup (see
    /// [`DownloadCleanup`]). A job that already committed, or already ended,
    /// is left unchanged.
    pub fn end_download_job(
        &self,
        job_id: &str,
        end: DownloadEnd<'_>,
        cache: &dyn CatalogBlobCache,
        now_ms: u64,
    ) -> Result<DownloadTransition, SynapseStoreError> {
        let (state, error_bytes) = download_end_columns(end)?;
        let transition = self.store.with_conn_fenced(|tx| {
            let Some(cleanup) = end_download_job_tx(
                tx,
                job_id,
                state,
                error_bytes.as_deref(),
                now_ms,
                cache,
                None,
            )?
            else {
                return Ok(DownloadTransition::Unchanged(download_job_tx(tx, job_id)?));
            };
            Ok(DownloadTransition::Applied {
                record: job_by_id_tx(tx, job_id)?.expect("ended download job is readable"),
                cleanup,
            })
        })?;
        Ok(transition)
    }

    /// Fail, with cleanup, every non-terminal download job minted by an
    /// earlier module generation. Runs at startup, before any new download
    /// job exists; a job whose process died has no one left to finish it.
    /// Returns the ids of the failed jobs.
    pub fn fail_interrupted_download_jobs(
        &self,
        current_generation: u64,
        error_json: &Value,
        cache: &dyn CatalogBlobCache,
        now_ms: u64,
    ) -> Result<Vec<String>, SynapseStoreError> {
        let error_bytes = serde_json::to_vec(error_json)?;
        let failed = self.store.with_conn_fenced(|tx| {
            let mut stmt = tx.prepare(&format!(
                "SELECT job_id FROM jobs
                 WHERE kind = ?1 AND module_generation < ?2
                   AND state IN {DOWNLOAD_NON_TERMINAL_SQL}
                 ORDER BY created_ms, job_id"
            ))?;
            let job_ids = stmt
                .query_map(
                    params![DOWNLOAD_JOB_KIND, current_generation as i64],
                    |row| row.get::<_, String>(0),
                )?
                .collect::<Result<Vec<_>, _>>()?;
            drop(stmt);
            let mut failed = Vec::new();
            for job_id in job_ids {
                if end_download_job_tx(
                    tx,
                    &job_id,
                    DOWNLOAD_STATE_FAILED,
                    Some(&error_bytes),
                    now_ms,
                    cache,
                    None,
                )?
                .is_some()
                {
                    failed.push(job_id);
                }
            }
            Ok(failed)
        })?;
        Ok(failed)
    }

    /// Quarantine a cached blob whose bytes no longer match their digest.
    ///
    /// One transaction fails every non-terminal download job holding an
    /// acquisition row for the blob (running that job's cleanup), deletes
    /// every install record with a member naming the blob and the member rows
    /// naming it, drops the blob's `cache.pin`s and unlinks the blob. Other
    /// blobs are untouched: an affected install's other member rows stay, so
    /// those blobs remain rooted and the next download reuses them and fetches
    /// only the quarantined file. The commit that reinstalls the backend
    /// replaces its member set. A free-form registration naming the blob stays
    /// registered; its next load finds the blob missing.
    pub fn quarantine_catalog_blob(
        &self,
        digest: &str,
        job_error_json: &Value,
        cache: &dyn CatalogBlobCache,
        now_ms: u64,
    ) -> Result<QuarantineReport, SynapseStoreError> {
        let digest = normalize_blob_digest(digest)?;
        let error_bytes = serde_json::to_vec(job_error_json)?;
        let report = self.store.with_conn_fenced(|tx| {
            let mut report = QuarantineReport::default();
            let mut stmt = tx.prepare(&format!(
                "SELECT DISTINCT a.job_id FROM download_acquisitions a
                 JOIN jobs j ON j.job_id = a.job_id
                 WHERE a.digest = ?1 AND j.kind = ?2
                   AND j.state IN {DOWNLOAD_NON_TERMINAL_SQL}
                 ORDER BY a.job_id"
            ))?;
            let holders = stmt
                .query_map(params![digest, DOWNLOAD_JOB_KIND], |row| {
                    row.get::<_, String>(0)
                })?
                .collect::<Result<Vec<_>, _>>()?;
            drop(stmt);
            for job_id in holders {
                if let Some(cleanup) = end_download_job_tx(
                    tx,
                    &job_id,
                    DOWNLOAD_STATE_FAILED,
                    Some(&error_bytes),
                    now_ms,
                    cache,
                    Some(&digest),
                )? {
                    report.freed_bytes = report.freed_bytes.saturating_add(cleanup.freed_bytes);
                    report.failed_jobs.push(job_id);
                }
            }

            let mut stmt = tx.prepare(
                "SELECT DISTINCT catalog_id, manifest_digest, backend
                 FROM catalog_install_members WHERE digest = ?1
                 ORDER BY catalog_id, manifest_digest, backend",
            )?;
            let installs = stmt
                .query_map(params![digest], |row| {
                    Ok(CatalogInstallKey {
                        catalog_id: row.get(0)?,
                        manifest_digest: row.get(1)?,
                        backend: row.get(2)?,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?;
            drop(stmt);
            for install in &installs {
                tx.execute(
                    "DELETE FROM catalog_installs
                     WHERE catalog_id = ?1 AND manifest_digest = ?2 AND backend = ?3",
                    params![install.catalog_id, install.manifest_digest, install.backend],
                )?;
            }
            report.removed_installs = installs;
            tx.execute(
                "DELETE FROM catalog_install_members WHERE digest = ?1",
                params![digest],
            )?;

            cache
                .remove_pins(&digest)
                .map_err(|error| cache_error("remove pins of", &digest, error))?;
            let freed = cache
                .delete_blob(&digest)
                .map_err(|error| cache_error("delete blob", &digest, error))?;
            report.freed_bytes = report.freed_bytes.saturating_add(freed);
            Ok(report)
        })?;
        Ok(report)
    }

    /// Match derived verification against the expected digest and current file identity.
    pub fn catalog_file_verified(
        &self,
        install: &CatalogInstallRecord,
        path: &str,
        expected: &str,
        stamp: &[String; 5],
    ) -> Result<bool, SynapseStoreError> {
        Ok(self.store.with_conn(|conn| {
            conn.query_row(
                "SELECT EXISTS(SELECT 1 FROM catalog_file_verifications
                 WHERE catalog_id=?1 AND manifest_digest=?2 AND backend=?3 AND path=?4
                   AND expected_sha256=?5 AND device=?6 AND inode=?7 AND size=?8
                   AND mtime_ns=?9 AND ctime_ns=?10)",
                params![
                    install.catalog_id,
                    install.manifest_digest,
                    install.backend,
                    path,
                    expected,
                    stamp[0],
                    stamp[1],
                    stamp[2],
                    stamp[3],
                    stamp[4]
                ],
                |row| row.get(0),
            )
        })?)
    }

    pub fn record_catalog_file_verification(
        &self,
        install: &CatalogInstallRecord,
        path: &str,
        expected: &str,
        stamp: &[String; 5],
    ) -> Result<(), SynapseStoreError> {
        self.store.with_conn_fenced(|tx| {
            tx.execute(
                "INSERT OR REPLACE INTO catalog_file_verifications
                 SELECT catalog_id, manifest_digest, backend, path, ?5, ?6, ?7, ?8, ?9, ?10
                 FROM catalog_install_members WHERE catalog_id=?1 AND manifest_digest=?2
                 AND backend=?3 AND path=?4 AND digest=?5",
                params![
                    install.catalog_id,
                    install.manifest_digest,
                    install.backend,
                    path,
                    expected,
                    stamp[0],
                    stamp[1],
                    stamp[2],
                    stamp[3],
                    stamp[4]
                ],
            )?;
            Ok(())
        })?;
        Ok(())
    }

    /// Every install record of a catalog entry, current and stale manifests
    /// alike, with its member rows.
    pub fn catalog_installs(
        &self,
        catalog_id: &str,
    ) -> Result<Vec<CatalogInstallRecord>, SynapseStoreError> {
        let installs = self.store.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT manifest_digest, backend FROM catalog_installs
                 WHERE catalog_id = ?1 ORDER BY manifest_digest, backend",
            )?;
            let keys = stmt
                .query_map(params![catalog_id], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })?
                .collect::<Result<Vec<_>, _>>()?;
            let mut members = conn.prepare(
                "SELECT path, digest FROM catalog_install_members
                 WHERE catalog_id = ?1 AND manifest_digest = ?2 AND backend = ?3
                 ORDER BY path",
            )?;
            let mut installs = Vec::with_capacity(keys.len());
            for (manifest_digest, backend) in keys {
                let rows = members
                    .query_map(params![catalog_id, manifest_digest, backend], |row| {
                        Ok(CatalogInstallMember {
                            path: row.get(0)?,
                            digest: row.get(1)?,
                        })
                    })?
                    .collect::<Result<Vec<_>, _>>()?;
                installs.push(CatalogInstallRecord {
                    catalog_id: catalog_id.to_string(),
                    manifest_digest,
                    backend,
                    members: rows,
                });
            }
            Ok(installs)
        })?;
        Ok(installs)
    }

    /// The acquisition rows a download job holds.
    pub fn download_acquisitions(
        &self,
        job_id: &str,
    ) -> Result<Vec<DownloadAcquisition>, SynapseStoreError> {
        let rows = self.store.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT job_id, digest, newly_published FROM download_acquisitions
                 WHERE job_id = ?1 ORDER BY digest",
            )?;
            let rows = stmt
                .query_map(params![job_id], |row| {
                    Ok(DownloadAcquisition {
                        job_id: row.get(0)?,
                        digest: row.get(1)?,
                        newly_published: row.get(2)?,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(rows)
        })?;
        Ok(rows)
    }

    pub fn download_key_binding(
        &self,
        request_key: &str,
    ) -> Result<Option<DownloadKeyBinding>, SynapseStoreError> {
        let binding = self
            .store
            .with_conn_fenced(|tx| download_key_binding_tx(tx, request_key))?;
        Ok(binding)
    }

    /// The non-terminal download job for a request digest, if one exists.
    pub fn active_download_job(
        &self,
        request_digest: &str,
    ) -> Result<Option<JobRecord>, SynapseStoreError> {
        let job = self
            .store
            .with_conn_fenced(|tx| active_download_job_tx(tx, request_digest))?;
        Ok(job)
    }
}

fn download_end_columns(
    end: DownloadEnd<'_>,
) -> Result<(&'static str, Option<Vec<u8>>), SynapseStoreError> {
    Ok(match end {
        DownloadEnd::Cancelled => (DOWNLOAD_STATE_CANCELLED, None),
        DownloadEnd::Failed { error_json } => {
            (DOWNLOAD_STATE_FAILED, Some(serde_json::to_vec(error_json)?))
        }
    })
}

/// Compare-and-swap a download job into `state` (cancelled or failed) and run
/// its cleanup. Returns `None`, changing nothing, when the job is not a
/// non-terminal download job. `quarantined` names a blob the caller deletes
/// itself, which the cleanup therefore leaves alone.
#[cfg_attr(not(test), allow(dead_code))]
fn end_download_job_tx(
    tx: &rusqlite::Transaction<'_>,
    job_id: &str,
    state: &str,
    error_bytes: Option<&[u8]>,
    now_ms: u64,
    cache: &dyn CatalogBlobCache,
    quarantined: Option<&str>,
) -> rusqlite::Result<Option<DownloadCleanup>> {
    let changed = tx.execute(
        &format!(
            "UPDATE jobs SET state = ?1, updated_ms = ?2, terminal_at_ms = ?2,
                 execution_expires_ms = NULL, active_attempt_id = NULL,
                 error_json = ?3, result_json = NULL
             WHERE job_id = ?4 AND kind = ?5 AND state IN {DOWNLOAD_NON_TERMINAL_SQL}"
        ),
        params![state, now_ms as i64, error_bytes, job_id, DOWNLOAD_JOB_KIND],
    )?;
    if changed == 0 {
        return Ok(None);
    }
    cleanup_download_job_tx(tx, job_id, cache, quarantined).map(Some)
}

/// Cleanup for a download job that ends without committing. It runs only in
/// the transaction that moves the job to `cancelled` or `failed`.
///
/// The staging directory is deleted. Each newly published blob is then
/// handed off when another job also acquired it (that job's row takes over
/// `newly_published`), kept when a member row, a `cache.pin` or a free-form
/// registration references it, and otherwise unlinked. Blobs the job only
/// reused are pre-existing and never deleted. Finally the job's acquisition
/// rows go.
#[cfg_attr(not(test), allow(dead_code))]
fn cleanup_download_job_tx(
    tx: &rusqlite::Transaction<'_>,
    job_id: &str,
    cache: &dyn CatalogBlobCache,
    quarantined: Option<&str>,
) -> rusqlite::Result<DownloadCleanup> {
    cache
        .delete_staging(job_id)
        .map_err(|error| cache_error("delete staging of job", job_id, error))?;
    let mut stmt = tx.prepare(
        "SELECT digest FROM download_acquisitions
         WHERE job_id = ?1 AND newly_published = 1 ORDER BY digest",
    )?;
    let published = stmt
        .query_map(params![job_id], |row| row.get::<_, String>(0))?
        .collect::<Result<Vec<_>, _>>()?;
    drop(stmt);

    let mut cleanup = DownloadCleanup::default();
    for digest in published {
        if quarantined == Some(digest.as_str()) {
            continue;
        }
        let handed_off = tx.execute(
            "UPDATE download_acquisitions SET newly_published = 1
             WHERE digest = ?1 AND job_id <> ?2",
            params![digest, job_id],
        )?;
        if handed_off > 0 {
            cleanup.handed_off.push(digest);
            continue;
        }
        let member: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM catalog_install_members WHERE digest = ?1)",
            params![digest],
            |row| row.get(0),
        )?;
        let pinned = cache
            .is_pinned(&digest)
            .map_err(|error| cache_error("read pins of", &digest, error))?;
        if member || pinned || free_form_references_digest_tx(tx, &digest)? {
            cleanup.kept.push(digest);
            continue;
        }
        let freed = cache
            .delete_blob(&digest)
            .map_err(|error| cache_error("delete blob", &digest, error))?;
        cleanup.freed_bytes = cleanup.freed_bytes.saturating_add(freed);
        cleanup.deleted_blobs.push(digest);
    }
    tx.execute(
        "DELETE FROM download_acquisitions WHERE job_id = ?1",
        params![job_id],
    )?;
    Ok(cleanup)
}

/// Whether a persisted free-form registration names the blob. The check is a
/// substring match over the stored config JSON, so it errs towards keeping a
/// blob: any locator, artifact digest or other field carrying the digest
/// counts, in either case and with or without the `sha256:` prefix.
#[cfg_attr(not(test), allow(dead_code))]
fn free_form_references_digest_tx(
    tx: &rusqlite::Transaction<'_>,
    digest: &str,
) -> rusqlite::Result<bool> {
    tx.query_row(
        "SELECT EXISTS(
             SELECT 1 FROM models WHERE instr(lower(CAST(config_json AS TEXT)), ?1) > 0
         )",
        params![digest.to_ascii_lowercase()],
        |row| row.get(0),
    )
}

/// The blob digests a stored registration names: every `digest` or
/// `artifact_digest` string field anywhere in its config that is a sha256.
/// Read from raw JSON so a row this binary can no longer decode still
/// releases its pins.
#[cfg_attr(not(test), allow(dead_code))]
fn registration_blob_digests(config_json: &[u8]) -> BTreeSet<String> {
    fn collect(value: &Value, digests: &mut BTreeSet<String>) {
        match value {
            Value::Object(fields) => {
                for (key, field) in fields {
                    if key == "digest" || key == "artifact_digest" {
                        if let Some(digest) = field
                            .as_str()
                            .and_then(|text| normalize_blob_digest(text).ok())
                        {
                            digests.insert(digest);
                        }
                    }
                    collect(field, digests);
                }
            }
            Value::Array(items) => {
                for item in items {
                    collect(item, digests);
                }
            }
            _ => {}
        }
    }

    let mut digests = BTreeSet::new();
    if let Ok(value) = serde_json::from_slice::<Value>(config_json) {
        collect(&value, &mut digests);
    }
    digests
}

/// The job, when it is a download job that can no longer acquire blobs
/// because it is terminal. `None` means it is still running. An id that names
/// no download job is an error: only a running job's own worker calls this.
#[cfg_attr(not(test), allow(dead_code))]
fn stopped_download_job_tx(
    tx: &rusqlite::Transaction<'_>,
    job_id: &str,
) -> rusqlite::Result<Option<JobRecord>> {
    let job = download_job_tx(tx, job_id)?.ok_or_else(|| {
        rusqlite::Error::InvalidParameterName(format!("no download job '{job_id}'"))
    })?;
    if is_download_non_terminal(&job.state) {
        Ok(None)
    } else {
        Ok(Some(job))
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn download_job_tx(
    tx: &rusqlite::Transaction<'_>,
    job_id: &str,
) -> rusqlite::Result<Option<JobRecord>> {
    Ok(job_by_id_tx(tx, job_id)?.filter(|job| job.kind == DOWNLOAD_JOB_KIND))
}

#[cfg_attr(not(test), allow(dead_code))]
fn active_download_job_tx(
    tx: &rusqlite::Transaction<'_>,
    request_digest: &str,
) -> rusqlite::Result<Option<JobRecord>> {
    let sql = format!(
        "{JOB_SELECT_SQL} WHERE kind = ?1 AND request_digest = ?2
           AND state IN {DOWNLOAD_NON_TERMINAL_SQL}
         ORDER BY created_ms DESC, rowid DESC LIMIT 1"
    );
    tx.query_row(&sql, params![DOWNLOAD_JOB_KIND, request_digest], row_to_job)
        .optional()
}

#[cfg_attr(not(test), allow(dead_code))]
fn download_key_binding_tx(
    tx: &rusqlite::Transaction<'_>,
    request_key: &str,
) -> rusqlite::Result<Option<DownloadKeyBinding>> {
    tx.query_row(
        "SELECT request_key, request_digest, job_id FROM download_key_bindings
         WHERE request_key = ?1",
        params![request_key],
        |row| {
            Ok(DownloadKeyBinding {
                request_key: row.get(0)?,
                request_digest: row.get(1)?,
                job_id: row.get(2)?,
            })
        },
    )
    .optional()
}

/// A blob digest in the form the catalog tables store: 64 lowercase hex
/// characters, with any `sha256:` prefix removed.
#[cfg_attr(not(test), allow(dead_code))]
fn normalize_blob_digest(value: &str) -> Result<String, SynapseStoreError> {
    let raw = value.strip_prefix("sha256:").unwrap_or(value);
    if raw.len() != 64 || !raw.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(SynapseStoreError::Decode(format!(
            "blob digest must be 64 hex characters, got '{value}'"
        )));
    }
    Ok(raw.to_ascii_lowercase())
}

#[cfg_attr(not(test), allow(dead_code))]
fn normalize_install_record(
    record: &CatalogInstallRecord,
) -> Result<CatalogInstallRecord, SynapseStoreError> {
    if record.catalog_id.is_empty()
        || record.manifest_digest.is_empty()
        || record.backend.is_empty()
    {
        return Err(SynapseStoreError::Decode(
            "catalog install records need a catalog id, manifest digest and backend".to_string(),
        ));
    }
    let members = record
        .members
        .iter()
        .map(|member| {
            if member.path.is_empty() {
                return Err(SynapseStoreError::Decode(format!(
                    "catalog install member of {} has an empty path",
                    record.catalog_id
                )));
            }
            Ok(CatalogInstallMember {
                path: member.path.clone(),
                digest: normalize_blob_digest(&member.digest)?,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(CatalogInstallRecord {
        members,
        ..record.clone()
    })
}

#[cfg_attr(not(test), allow(dead_code))]
fn cache_error(action: &str, subject: &str, error: std::io::Error) -> rusqlite::Error {
    rusqlite::Error::ToSqlConversionFailure(Box::new(std::io::Error::new(
        error.kind(),
        format!("model cache: {action} {subject}: {error}"),
    )))
}
fn expire_jobs_tx(tx: &rusqlite::Transaction<'_>, now_ms: i64) -> rusqlite::Result<usize> {
    expire_jobs_where_tx(tx, None, now_ms)
}

fn expire_job_tx(
    tx: &rusqlite::Transaction<'_>,
    job_id: &str,
    now_ms: i64,
) -> rusqlite::Result<()> {
    expire_jobs_where_tx(tx, Some(job_id), now_ms).map(drop)
}

/// Fails every active job past its execution TTL, or only `job_id` when given.
fn expire_jobs_where_tx(
    tx: &rusqlite::Transaction<'_>,
    job_id: Option<&str>,
    now_ms: i64,
) -> rusqlite::Result<usize> {
    let execution_error = serde_json::to_vec(&serde_json::json!({
        "code": "deadline_exceeded",
        "class": "transient",
        "safe_to_retry_same_request": false,
        "message": "durable job execution TTL expired",
    }))
    .expect("execution expiry error serializes");
    // Download jobs have no execution TTL: they are admitted with a NULL
    // execution_expires_ms, and the kind filter keeps them exempt even if a
    // deadline were ever written. They end only through their own commit,
    // cancel or failure transactions, which also clean up their blobs.
    // A separate predicate per scope keeps the single-job form on the primary key.
    let scope = if job_id.is_some() {
        "job_id = ?7"
    } else {
        "?7 IS NULL"
    };
    tx.execute(
        &format!(
            "UPDATE jobs SET state = ?1, updated_ms = ?2, terminal_at_ms = ?2,
                 execution_expires_ms = NULL, active_attempt_id = NULL, error_json = ?3
             WHERE {scope} AND state IN (?4, ?5) AND execution_expires_ms <= ?2
               AND kind <> ?6"
        ),
        params![
            JOB_STATE_FAILED_TRANSIENT,
            now_ms,
            execution_error,
            JOB_STATE_QUEUED,
            JOB_STATE_RUNNING,
            DOWNLOAD_JOB_KIND,
            job_id,
        ],
    )
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
        // A download request key stays bound to its job only as long as the
        // job row exists; a binding to a purged job would point at nothing.
        tx.execute(
            "DELETE FROM download_key_bindings WHERE job_id = ?1",
            params![job_id],
        )?;
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
                "DELETE FROM checkpoints WHERE request_digest = ?1",
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
        result_retention_ttl_ms, terminal_at_ms, active_attempt_id,
        page_count, result_json, error_json, params_json FROM jobs";

fn unix_now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
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

#[allow(dead_code)]
fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_hexdigit())
        && value.bytes().all(|byte| !byte.is_ascii_uppercase())
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

fn validate_profile_state(state: &ProfileState) -> Result<(), SynapseStoreError> {
    match (
        state.snapshot.as_ref(),
        state.revisioned_machine_profile_hash.as_deref(),
        state.profile_activation_epoch,
    ) {
        (None, None, None) => {}
        (Some(snapshot), Some(hash), Some(epoch)) if epoch > 0 && is_digest(hash) => {
            if snapshot.revisioned_hash() != hash {
                return Err(SynapseStoreError::ProfileStateCorrupt(
                    "snapshot does not match revisioned profile hash".to_string(),
                ));
            }
        }
        (None, Some(hash), Some(epoch)) if epoch > 0 && is_digest(hash) => {}
        _ => {
            return Err(SynapseStoreError::ProfileStateCorrupt(
                "profile snapshot, revisioned hash, and positive epoch are inconsistent"
                    .to_string(),
            ));
        }
    }
    if let Some(previous_hash) = state.previous_revisioned_machine_profile_hash.as_deref() {
        if !is_digest(previous_hash) {
            return Err(SynapseStoreError::ProfileStateCorrupt(
                "previous revisioned profile hash is malformed".to_string(),
            ));
        }
    }
    Ok(())
}

fn profile_state_from_row(row: &Row<'_>) -> rusqlite::Result<ProfileState> {
    let snapshot_json: Option<String> = row.get(0)?;
    let snapshot = snapshot_json
        .as_deref()
        .map(serde_json::from_str)
        .transpose()
        .map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(
                0,
                rusqlite::types::Type::Text,
                Box::new(error),
            )
        })?;
    let epoch_value = row.get::<_, Option<i64>>(2)?;
    let last_rotation_value = row.get::<_, Option<i64>>(5)?;
    if epoch_value.is_some_and(|value| value < 0)
        || last_rotation_value.is_some_and(|value| value < 0)
    {
        return Err(rusqlite::Error::InvalidQuery);
    }
    Ok(ProfileState {
        snapshot,
        revisioned_machine_profile_hash: row.get(1)?,
        profile_activation_epoch: epoch_value.map(|value| value as u64),
        previous_revisioned_machine_profile_hash: row.get(3)?,
        last_rotation_reason: row.get(4)?,
        last_rotation_at_ms: last_rotation_value.map(|value| value as u64),
    })
}

fn changed_profile_fields(previous: &MachineProfile, current: &MachineProfile) -> Vec<String> {
    let mut changed = Vec::new();
    if previous.os_build != current.os_build {
        changed.push("os_build".to_string());
    }
    if previous.arch != current.arch {
        changed.push("arch".to_string());
    }
    if previous.chip_model != current.chip_model {
        changed.push("chip_model".to_string());
    }
    if previous.ram_class != current.ram_class {
        changed.push("ram_class".to_string());
    }
    if previous.ane_subtype != current.ane_subtype {
        changed.push("ane_subtype".to_string());
    }
    if previous.engine_identities != current.engine_identities {
        changed.push("engine_identities".to_string());
    }
    changed.sort_unstable();
    changed
}

fn rotation_event_id(new_hash: &str, epoch: u64) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"rotation-ledger-v1\0");
    hasher.update(new_hash.as_bytes());
    hasher.update(epoch.to_be_bytes());
    hex::encode(hasher.finalize())
}

fn rotation_event_conn(
    conn: &rusqlite::Connection,
    event_id: &str,
) -> rusqlite::Result<Option<RotationLedgerEvent>> {
    let raw = conn
        .query_row(
            "SELECT event_id, old_revisioned_machine_profile_hash,
                    new_revisioned_machine_profile_hash, old_profile_activation_epoch,
                    new_profile_activation_epoch, changed_fields_json,
                    previous_snapshot_json, current_snapshot_json, observed_at_ms,
                    module_generation, created_at_ms
             FROM profile_rotation_events WHERE event_id = ?1",
            params![event_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<i64>>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, Option<String>>(6)?,
                    row.get::<_, String>(7)?,
                    row.get::<_, i64>(8)?,
                    row.get::<_, i64>(9)?,
                    row.get::<_, i64>(10)?,
                ))
            },
        )
        .optional()?;
    let Some((
        event_id,
        old_hash,
        new_hash,
        old_epoch,
        new_epoch,
        changed_fields_json,
        previous_snapshot_json,
        current_snapshot_json,
        observed_at_ms,
        module_generation,
        created_at_ms,
    )) = raw
    else {
        return Ok(None);
    };
    let changed_fields: Vec<String> =
        serde_json::from_str(&changed_fields_json).map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(
                5,
                rusqlite::types::Type::Text,
                Box::new(error),
            )
        })?;
    let previous_snapshot = previous_snapshot_json
        .as_deref()
        .map(serde_json::from_str)
        .transpose()
        .map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(
                6,
                rusqlite::types::Type::Text,
                Box::new(error),
            )
        })?;
    let current_snapshot: MachineProfile =
        serde_json::from_str(&current_snapshot_json).map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(
                7,
                rusqlite::types::Type::Text,
                Box::new(error),
            )
        })?;
    Ok(Some(RotationLedgerEvent {
        event_id,
        old_revisioned_machine_profile_hash: old_hash,
        new_revisioned_machine_profile_hash: new_hash,
        old_profile_activation_epoch: old_epoch.map(|value| value as u64),
        new_profile_activation_epoch: new_epoch as u64,
        changed_fields,
        previous_snapshot,
        current_snapshot,
        observed_at_ms: observed_at_ms as u64,
        module_generation: module_generation as u64,
        created_at_ms: created_at_ms as u64,
    }))
}

fn rotation_event_tx(
    tx: &rusqlite::Transaction<'_>,
    event_id: &str,
) -> rusqlite::Result<RotationLedgerEvent> {
    rotation_event_conn(tx, event_id)?.ok_or(rusqlite::Error::QueryReturnedNoRows)
}

fn to_sql_error(error: SynapseStoreError) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(error))
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
        page_count: row.get::<_, i64>(12)? as u32,
        result_json: decode_optional_json(row.get(13)?, 13)?,
        error_json: decode_optional_json(row.get(14)?, 14)?,
        params_json: decode_optional_json(row.get(15)?, 15)?,
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
    use synapse_core::MachineProfile;

    fn registration_fixture(model_id: &str) -> StoredModelConfig {
        StoredModelConfig {
            model_id: model_id.to_string(),
            engine: "llama".to_string(),
            task: "generate".to_string(),
            artifact_digest: format!("sha256:{}", "1".repeat(64)),
            artifact_format: "gguf".to_string(),
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
                engine: "llama".to_string(),
                version: "test".to_string(),
                build_flags: BTreeMap::new(),
            },
            numeric_profile_id: NumericProfileId("np-test".to_string()),
            fingerprint: Fingerprint("fp-test".to_string()),
            worker_bin: None,
            worker_runtime_dir: None,
        }
    }

    pub(super) fn admit(
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

    // The admission tuple is inherently wide (idempotency key, digest, kind,
    // generation, clock, TTL pair); bundling them into a one-off struct would
    // only rename the width.
    #[allow(clippy::too_many_arguments)]
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
                &serde_json::json!({"items": 2}),
                now_ms,
                execution_ttl_ms,
                retention_ttl_ms,
            )
            .unwrap()
    }

    #[test]
    fn restart_marks_queued_and_running_attempts_failed_without_losing_pages() {
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
                }],
                13,
            )
            .unwrap();
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
        assert_eq!(failed, 3);
        for job_id in [queued, running.clone(), probe] {
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
    fn newest_schema_version_is_the_highest_migration_this_binary_carries() {
        // Asserts the PROPERTY the derivation must satisfy, not the derivation
        // itself. Restating `MIGRATIONS.iter().map(..).max()` here would make
        // this agree with the code by construction and pass whatever the
        // function did. A regressed derivation that still returns a positive
        // number would pass any check that looks only at its type. These two bounds pin it from both sides: `.min()`
        // or a hardcoded low literal breaks the first, and a literal above the
        // list breaks the second.
        //
        // `.last()` is NOT caught while the list stays sorted, because it then
        // returns the same value as `.max()` -- an equivalent mutant, not a
        // gap. It becomes a real regression only once the list is unsorted,
        // and the first bound catches it exactly then.
        let newest = newest_schema_version();
        assert!(
            MIGRATIONS
                .iter()
                .all(|migration| migration.version <= newest),
            "newest_schema_version() reported {newest}, which is below a migration this binary carries"
        );
        assert!(
            MIGRATIONS
                .iter()
                .any(|migration| migration.version == newest),
            "newest_schema_version() reported {newest}, which is not a migration this binary carries"
        );
    }

    #[test]
    fn remote_removal_migrates_v14_and_preserves_finished_jobs_and_pages() {
        let (root, descriptor) = temp_descriptor("remote-removal-v14");
        let now = unix_now_ms() as i64;
        {
            let legacy = open_sqlite(&descriptor).unwrap();
            legacy.migrate(NAMESPACE, &MIGRATIONS[..14]).unwrap();
            legacy
                .with_conn_fenced(|tx| {
                    tx.execute(
                        "INSERT INTO jobs (job_id, request_key, request_digest, kind,
                        module_generation, state, created_ms, updated_ms,
                        result_retention_ttl_ms, terminal_at_ms, params_json, result_json,
                        page_count, logical_handle)
                     VALUES ('finished', 'key', 'digest', 'embed.batch', 1, 'done', ?1, ?1,
                        86400000, ?1, X'7B7D', X'7B7D', 1, 'vault/provider')",
                        params![now],
                    )?;
                    tx.execute(
                        "INSERT INTO jobs (job_id, request_key, request_digest, kind,
                        module_generation, state, created_ms, updated_ms,
                        result_retention_ttl_ms, params_json, logical_handle,
                        paused_at_ms, resume_deadline_ms)
                     VALUES ('paused', 'paused-key', 'paused-digest', 'embed.batch', 1,
                        'paused_needs_reauth', ?1, ?1, 86400000, X'7B7D',
                        'vault/provider', ?1, ?1 + 86400000)",
                        params![now],
                    )?;
                    tx.execute_batch(
                        r#"
                    INSERT INTO result_pages (request_digest, page_no, page_json,
                        provider_request_ids_json, committed_at)
                    VALUES ('digest', 0, X'7B7D', X'5B22726571225D', 1);
                    INSERT INTO remote_checkpoints (request_digest, item_id, result,
                        page_no, provider_request_id, committed_at)
                    VALUES ('digest', 'item', X'7B7D', 0, 'req', 1);
                    INSERT INTO remote_url_bindings (remote_profile_hash, last_base_url)
                    VALUES ('remote', 'https://example.test');
                    INSERT INTO cert_rows (certification_class, assurance_class, status,
                        key_hash, remote_profile_hash, identity_revision, fingerprint,
                        certified_at_ms, os_build, module_generation, evidence_json)
                    VALUES ('declared', 'declared', 'certified', 'remote', 'remote', 'r1',
                        'remote', 1, 'os', 1, '{"remote_sentinel":{}}');
                    INSERT INTO cert_rows (certification_class, assurance_class, status,
                        key_hash, machine_profile_hash, fingerprint,
                        certified_at_ms, os_build, module_generation, evidence_json)
                    VALUES ('embedding', 'measured', 'certified', 'local', 'local',
                        'local', 1, 'os', 1, '{}');
                "#,
                    )
                })
                .unwrap();
        }
        let store = SynapseStore::open(&descriptor).unwrap();
        assert_eq!(newest_schema_version(), 19);
        let finished = store.get_job("finished").unwrap().unwrap();
        assert_eq!(finished.state, JOB_STATE_DONE);
        assert_eq!(finished.request_digest, "digest");
        assert_eq!(finished.terminal_at_ms, Some(now as u64));
        assert_eq!(finished.result_retention_ttl_ms, 86_400_000);
        assert_eq!(finished.page_count, 1);
        assert_eq!(finished.result_json, Some(serde_json::json!({})));
        assert_eq!(
            store.get_job_page("finished", 0).unwrap(),
            Some(b"{}".to_vec())
        );
        assert_eq!(
            store.committed_item_ids("digest").unwrap(),
            ["item".to_string()].into()
        );
        let paused = store.get_job("paused").unwrap().unwrap();
        assert_eq!(paused.state, JOB_STATE_FAILED_PERMANENT);
        assert_eq!(paused.error_json.unwrap()["code"], "invalid_request");
        assert_eq!(paused.terminal_at_ms, Some(now as u64));
        store
            .store
            .with_conn(|conn| {
                for retired in ["remote_url_bindings", "remote_checkpoints", "cert_rows"] {
                    let count: i64 = conn.query_row(
                        "SELECT COUNT(*) FROM sqlite_master WHERE name = ?1",
                        [retired],
                        |row| row.get(0),
                    )?;
                    assert_eq!(count, 0, "retired table {retired}");
                }
                let mut columns = conn.prepare("SELECT name FROM pragma_table_info('jobs')")?;
                let columns = columns
                    .query_map([], |row| row.get::<_, String>(0))?
                    .collect::<Result<Vec<_>, _>>()?;
                for retired in ["logical_handle", "paused_at_ms", "resume_deadline_ms"] {
                    assert!(
                        !columns.iter().any(|column| column == retired),
                        "retired column {retired}"
                    );
                }
                Ok(())
            })
            .unwrap();
        drop(store);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn schema_keeps_the_known_guard_triggers() {
        // Ordinary store operations can still succeed after a guard is lost.
        // Pin the surviving guards by name to catch an accidental schema deletion.
        let (root, descriptor) = temp_descriptor("schema-trigger-set");
        let store = SynapseStore::open(&descriptor).unwrap();
        let triggers = store
            .store
            .with_conn(|conn| {
                let mut stmt =
                    conn.prepare("SELECT name FROM sqlite_master WHERE type = 'trigger'")?;
                let names = stmt
                    .query_map([], |row| row.get::<_, String>(0))?
                    .collect::<Result<BTreeSet<_>, _>>()?;
                Ok(names)
            })
            .unwrap();
        for guard in [
            "catalog_member_verification_cleanup",
            "checkpoint_immutable",
        ] {
            assert!(
                triggers.contains(guard),
                "schema lost guard trigger {guard}; present: {triggers:?}"
            );
        }
        drop(store);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn open_refuses_a_store_written_by_a_longer_schema_chain() {
        let (root, descriptor) = temp_descriptor("schema-ahead");
        let chain_max = MIGRATIONS.last().unwrap().version;
        {
            // A future binary's chain: everything this binary knows plus one
            // additive step. Only the recorded version matters to the refusal.
            let mut longer: Vec<Migration> = MIGRATIONS.to_vec();
            longer.push(Migration {
                version: chain_max + 1,
                statements: "CREATE TABLE future_only (id INTEGER PRIMARY KEY)",
            });
            let future = open_sqlite(&descriptor).unwrap();
            let outcome = future.migrate(NAMESPACE, &longer).unwrap();
            assert_eq!(outcome.recorded, chain_max + 1);
            assert!(!outcome.store_ahead());
        }

        let Err(error) = SynapseStore::open(&descriptor) else {
            panic!("a store ahead of the binary's chain must not open");
        };
        let rendered = error.to_string();
        match error {
            SynapseStoreError::SchemaAheadOfBinary {
                recorded,
                chain_max: seen,
            } => {
                assert_eq!(recorded, chain_max + 1);
                assert_eq!(seen, chain_max);
            }
            other => panic!("expected SchemaAheadOfBinary, got {other:?}"),
        }
        assert!(rendered.contains(&format!("recorded version {}", chain_max + 1)));
        assert!(rendered.contains(&format!("chain ends at {chain_max}")));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn migration_from_v5_preserves_jobs_and_pages() {
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
        let job = migrated.get_job_at("legacy-job", 21).unwrap().unwrap();
        assert_eq!(job.request_digest, "legacy:legacy-job");
        assert_eq!(job.result_retention_ttl_ms, 100);
        assert!(migrated.get_job_page("legacy-job", 0).unwrap().is_some());
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

    #[test]
    fn profile_activation_is_epoch_one_then_single_parent_per_rotation() {
        let (root, descriptor) = temp_descriptor("profile-rotation");
        let store = SynapseStore::open(&descriptor).unwrap();
        let profile_a = MachineProfile {
            os_build: "os-a".to_string(),
            arch: "aarch64".to_string(),
            chip_model: "chip-a".to_string(),
            ram_class: "le_32_gib".to_string(),
            ane_subtype: None,
            engine_identities: Vec::new(),
        };
        let mut profile_b = profile_a.clone();
        profile_b.engine_identities.push(EngineIdentity {
            engine: "owned-metal-decode".to_string(),
            version: "worker-v2".to_string(),
            build_flags: BTreeMap::new(),
        });
        let first = store.observe_profile(&profile_a, 10, 1).unwrap();
        assert_eq!(first.state.profile_activation_epoch, Some(1));
        assert!(!first.rotated);
        let rotated = store.observe_profile(&profile_b, 20, 1).unwrap();
        assert_eq!(rotated.state.profile_activation_epoch, Some(2));
        assert_eq!(
            rotated.state.last_rotation_reason.as_deref(),
            Some("engine_identities")
        );
        assert!(rotated.event.is_some());
        assert_eq!(
            store
                .observe_profile(&profile_b, 21, 1)
                .unwrap()
                .state
                .profile_activation_epoch,
            Some(2)
        );
        let returned = store.observe_profile(&profile_a, 30, 1).unwrap();
        assert_eq!(returned.state.profile_activation_epoch, Some(3));
        assert_eq!(store.rotation_events().unwrap().len(), 2);
        assert_eq!(
            store.profile_state().unwrap().profile_activation_epoch,
            Some(3)
        );
        let _ = std::fs::remove_dir_all(root);
    }

    fn populate_every_restore_table(store: &SynapseStore, seed: &str) {
        // Differs between the capture and live seeds, so the restore tests can
        // tell a counter copied from the capture from the live one.
        let run_seq = 70 + seed.len();
        store.next_module_generation().unwrap();
        store
            .store
            .with_conn(|connection| {
                connection.execute_batch(&format!(
                    r#"
                    UPDATE module_meta
                    SET module_generation = 71, table_epoch = 72
                    WHERE id = 0;
                    INSERT INTO jobs (
                        job_id, request_key, request_digest, kind, module_generation,
                        state, created_ms, updated_ms, result_retention_ttl_ms,
                        terminal_at_ms, params_json, page_count
                    ) VALUES ('job-{seed}', 'key-{seed}', 'digest-{seed}', 'embed.batch',
                              1, 'done', 1, 2, 100, 2, X'7B7D', 1);
                    INSERT INTO media_objects (
                        id, owner, mime, bytes, sha256, state, received, retained_ms
                    ) VALUES ('object-{seed}', 'owner-{seed}', 'audio/wav', 100,
                              'sha-{seed}', 'committed', 100, 3600000);
                    INSERT INTO media_jobs (
                        id, owner, client_id, binding, lane, object_id, state,
                        accepted_ms, retained_ms, result
                    ) VALUES ('media-{seed}', 'owner-{seed}', 'client-{seed}', '{{}}',
                              'speech-cuda', 'object-{seed}', 'done', 1, 3600000, '{{}}');
                    INSERT INTO media_observation_pages (job_id, part, page, data)
                        VALUES ('media-{seed}', 'encoder_output', 0, X'00000000');
                    INSERT INTO result_pages (
                        request_digest, page_no, page_json, committed_at
                    ) VALUES ('digest-{seed}', 0, X'7B7D', 2);
                    INSERT INTO checkpoints (
                        request_digest, item_id, result, page_no, committed_at
                    ) VALUES ('digest-{seed}', 'item-{seed}', X'7B7D', 0, 2);
                    INSERT INTO models (
                        model_id, engine, task, fingerprint, config_json, created_ms, updated_ms
                    ) VALUES ('model-{seed}', 'llama', 'generate', 'fingerprint-{seed}', X'7B7D', 1, 2);
                    INSERT INTO knob_assignments (
                        machine_profile_hash, workload, knob, model_id, numeric_profile_id,
                        fingerprint, engine, measured_at_ms, os_build, module_generation,
                        throughput_tok_s, single_item_latency_p50_ms
                    ) VALUES ('machine-{seed}', 'embed.batch', 'balanced', 'model-{seed}',
                              'numeric-{seed}', 'knob-{seed}', 'engine', 1, 'os', 1, 3.0, 5.0);
                    UPDATE profile_state
                    SET snapshot_json = '{{}}', revisioned_machine_profile_hash = 'profile-{seed}',
                        profile_activation_epoch = 1
                    WHERE id = 0;
                    INSERT INTO profile_rotation_events (
                        event_id, new_revisioned_machine_profile_hash,
                        new_profile_activation_epoch, changed_fields_json,
                        current_snapshot_json, observed_at_ms, module_generation, created_at_ms
                    ) VALUES ('rotation-{seed}', 'profile-{seed}', 1, '[]', '{{}}', 1, 1, 1);
                    INSERT INTO catalog_installs (catalog_id, manifest_digest, backend)
                    VALUES ('catalog-{seed}', 'manifest-{seed}', 'metal');
                    INSERT INTO catalog_install_members (
                        catalog_id, manifest_digest, backend, path, digest
                    ) VALUES ('catalog-{seed}', 'manifest-{seed}', 'metal', 'model.safetensors',
                              'blob-{seed}');
                    INSERT INTO catalog_file_verifications VALUES (
                        'catalog-{seed}', 'manifest-{seed}', 'metal', 'model.safetensors',
                        'blob-{seed}', '1', '2', '3', '4', '5');
                    INSERT INTO download_acquisitions (job_id, digest, newly_published)
                    VALUES ('download-{seed}', 'blob-{seed}', 1);
                    INSERT INTO download_key_bindings (request_key, request_digest, job_id)
                    VALUES ('download-key-{seed}', 'download-digest-{seed}', 'download-{seed}');
                    INSERT INTO catalog_self_checks (
                        check_id, catalog_id, manifest_digest, backend, fingerprint,
                        engine_identity_json, os_build, fixture_revision, state, generation
                    ) VALUES ('check-{seed}', 'catalog-{seed}', 'manifest-{seed}', 'metal',
                              'fingerprint-{seed}', '{{}}', 'os', '1', 'passed', 1);
                    UPDATE self_check_run_seq SET value = {run_seq} WHERE id = 0;
                    "#
                ))?;
                Ok(())
            })
            .unwrap();
    }

    fn restore_application_tables() -> Vec<&'static str> {
        RESTORE_APPLICATION_TABLES.to_vec()
    }

    fn table_rows(store: &SynapseStore, table: &str) -> Vec<Vec<rusqlite::types::Value>> {
        store
            .store
            .with_conn(|connection| {
                let mut statement =
                    connection.prepare(&format!("SELECT * FROM \"{table}\" ORDER BY rowid"))?;
                let columns = statement.column_count();
                let rows = statement
                    .query_map([], |row| {
                        (0..columns)
                            .map(|index| row.get(index))
                            .collect::<rusqlite::Result<Vec<_>>>()
                    })?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                Ok(rows)
            })
            .unwrap()
    }

    fn table_count(store: &SynapseStore, table: &str) -> i64 {
        store
            .store
            .with_conn(|connection| {
                connection.query_row(&format!("SELECT COUNT(*) FROM \"{table}\""), [], |row| {
                    row.get(0)
                })
            })
            .unwrap()
    }

    #[test]
    fn restore_policy_classifies_every_application_table() {
        let application = RESTORE_APPLICATION_TABLES
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let cleared = RESTORE_CLEAR_TABLES
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let kept = RESTORE_KEEP_TABLES.iter().copied().collect::<BTreeSet<_>>();
        assert_eq!(cleared, application);
        assert!(kept.is_subset(&cleared));
    }

    #[test]
    fn restore_keeps_live_outbox_and_forward_only_findings_generation() {
        let (capture_root, capture_descriptor) = temp_descriptor("restore-event-capture");
        let capture = SynapseStore::open(&capture_descriptor).unwrap();
        populate_every_restore_table(&capture, "capture");
        capture.store.with_conn_fenced(|tx| tx.execute_batch(
            "UPDATE findings_state SET digest='captured',findings_generation=4; \
             INSERT INTO event_outbox(event_id,subject,body,created_at) VALUES('capture-event','event.synapse.findings_changed.v1','{}',1);"
        )).unwrap();
        let (live_root, live_descriptor) = temp_descriptor("restore-event-live");
        let live = SynapseStore::open(&live_descriptor).unwrap();
        populate_every_restore_table(&live, "live");
        live.store.with_conn_fenced(|tx| tx.execute_batch(
            "UPDATE findings_state SET digest='live',findings_generation=9; \
             INSERT INTO event_outbox(event_id,subject,body,created_at) VALUES('live-event','event.synapse.findings_changed.v1','{}',1);"
        )).unwrap();
        drop(live);
        SynapseStore::restore_from_capture(
            &live_descriptor,
            &sqlite_path(&capture_descriptor).unwrap(),
        )
        .unwrap();
        let restored = SynapseStore::open(&live_descriptor).unwrap();
        restored
            .store
            .with_conn(|c| {
                let generation: u64 =
                    c.query_row("SELECT findings_generation FROM findings_state", [], |r| {
                        r.get(0)
                    })?;
                assert_eq!(generation, 9);
                let digest: String =
                    c.query_row("SELECT digest FROM findings_state", [], |r| r.get(0))?;
                assert_eq!(digest, "live");
                let captured: u64 = c.query_row(
                    "SELECT count(*) FROM event_outbox WHERE event_id='capture-event'",
                    [],
                    |r| r.get(0),
                )?;
                let live: u64 = c.query_row(
                    "SELECT count(*) FROM event_outbox WHERE event_id='live-event'",
                    [],
                    |r| r.get(0),
                )?;
                assert_eq!((captured, live), (0, 1));
                Ok(())
            })
            .unwrap();
        restored
            .record_findings_digest("post-restore", "ok", 2)
            .unwrap();
        let generation: u64 = restored
            .store
            .with_conn(|c| {
                c.query_row("SELECT findings_generation FROM findings_state", [], |r| {
                    r.get(0)
                })
            })
            .unwrap();
        assert_eq!(generation, 10);
        drop(restored);
        drop(capture);
        std::fs::remove_dir_all(live_root).unwrap();
        std::fs::remove_dir_all(capture_root).unwrap();
    }

    #[test]
    fn restore_import_preserves_knob_assignments_and_clears_rebuildable_state() {
        let (capture_root, capture_descriptor) = temp_descriptor("restore-capture-all");
        let capture = SynapseStore::open(&capture_descriptor).unwrap();
        populate_every_restore_table(&capture, "capture");
        let keep_rows = RESTORE_KEEP_TABLES
            .iter()
            .map(|table| (*table, table_rows(&capture, table)))
            .collect::<BTreeMap<_, _>>();
        for table in restore_application_tables() {
            assert!(
                table_count(&capture, table) > 0,
                "capture table {table} is empty"
            );
        }
        let capture_path = sqlite_path(&capture_descriptor).unwrap();
        drop(capture);

        let (live_root, live_descriptor) = temp_descriptor("restore-live-all");
        let live = SynapseStore::open(&live_descriptor).unwrap();
        populate_every_restore_table(&live, "live");
        let live_only_before = RESTORE_LIVE_ONLY_TABLES
            .iter()
            .map(|table| (*table, table_rows(&live, table)))
            .chain(std::iter::once((
                "sqlite_sequence",
                table_rows(&live, "sqlite_sequence"),
            )))
            .collect::<BTreeMap<_, _>>();
        drop(live);

        let report = SynapseStore::restore_from_capture(&live_descriptor, &capture_path).unwrap();
        let restored = SynapseStore::open(&live_descriptor).unwrap();
        for table in RESTORE_KEEP_TABLES {
            assert_eq!(
                report.imported_rows.get(*table).copied(),
                Some(keep_rows[*table].len() as u64)
            );
            assert_eq!(table_rows(&restored, table), keep_rows[*table]);
        }
        for table in restore_application_tables() {
            if !RESTORE_KEEP_TABLES.contains(&table) {
                assert_eq!(
                    table_count(&restored, table),
                    0,
                    "restore resurrected table {table}"
                );
            }
        }
        for (table, rows) in live_only_before {
            assert_eq!(
                table_rows(&restored, table),
                rows,
                "live framework table {table} changed during restore"
            );
        }
        drop(restored);
        let _ = std::fs::remove_dir_all(capture_root);
        let _ = std::fs::remove_dir_all(live_root);
    }

    #[test]
    fn restore_import_rolls_back_the_whole_trust_set_after_mid_import_failure() {
        let (capture_root, capture_descriptor) = temp_descriptor("restore-capture-rollback");
        let capture = SynapseStore::open(&capture_descriptor).unwrap();
        populate_every_restore_table(&capture, "capture");
        let capture_path = sqlite_path(&capture_descriptor).unwrap();
        drop(capture);

        let (live_root, live_descriptor) = temp_descriptor("restore-live-rollback");
        let live = SynapseStore::open(&live_descriptor).unwrap();
        populate_every_restore_table(&live, "live");
        let mut tables = restore_application_tables();
        tables.extend_from_slice(RESTORE_LIVE_ONLY_TABLES);
        tables.push("sqlite_sequence");
        let before = tables
            .iter()
            .map(|table| (*table, table_rows(&live, table)))
            .collect::<BTreeMap<_, _>>();

        FAIL_RESTORE_AFTER_TABLE.with(|table| table.set(Some("knob_assignments")));
        let error = live.import_migrated_capture(&capture_path).unwrap_err();
        FAIL_RESTORE_AFTER_TABLE.with(|table| table.set(None));
        assert!(error.to_string().contains("simulated restore interruption"));
        for (table, rows) in before {
            assert_eq!(
                table_rows(&live, table),
                rows,
                "mid-import failure split the trust set at table {table}"
            );
        }
        drop(live);
        let _ = std::fs::remove_dir_all(capture_root);
        let _ = std::fs::remove_dir_all(live_root);
    }

    #[test]
    fn restore_import_refuses_a_held_target_store_lease_without_changes() {
        let (capture_root, capture_descriptor) = temp_descriptor("restore-capture-lease");
        let capture = SynapseStore::open(&capture_descriptor).unwrap();
        populate_every_restore_table(&capture, "capture");
        let capture_path = sqlite_path(&capture_descriptor).unwrap();
        drop(capture);

        let (live_root, live_descriptor) = temp_descriptor("restore-live-lease");
        let live = SynapseStore::open(&live_descriptor).unwrap();
        populate_every_restore_table(&live, "live");
        let before = table_rows(&live, "knob_assignments");
        let error =
            SynapseStore::restore_from_capture(&live_descriptor, &capture_path).unwrap_err();
        let rendered = error.to_string();
        assert!(rendered.contains("target store lease"), "{rendered}");
        assert!(rendered.contains("store.db"), "{rendered}");
        assert_eq!(table_rows(&live, "knob_assignments"), before);
        drop(live);
        let _ = std::fs::remove_dir_all(capture_root);
        let _ = std::fs::remove_dir_all(live_root);
    }

    #[test]
    fn restore_import_refuses_a_schema_ahead_capture_by_typed_arm() {
        let (capture_root, capture_descriptor) = temp_descriptor("restore-capture-ahead");
        let chain_max = newest_schema_version();
        let mut longer = MIGRATIONS.to_vec();
        longer.push(Migration {
            version: chain_max + 1,
            statements: "CREATE TABLE future_only (id INTEGER PRIMARY KEY)",
        });
        let capture_store = open_sqlite(&capture_descriptor).unwrap();
        capture_store.migrate(NAMESPACE, &longer).unwrap();
        drop(capture_store);

        let (live_root, live_descriptor) = temp_descriptor("restore-live-ahead");
        let capture_path = sqlite_path(&capture_descriptor).unwrap();
        let error =
            SynapseStore::restore_from_capture(&live_descriptor, &capture_path).unwrap_err();
        assert!(matches!(
            error,
            SynapseStoreError::SchemaAheadOfBinary {
                recorded,
                chain_max: seen,
            } if recorded == chain_max + 1 && seen == chain_max
        ));
        assert!(!sqlite_path(&live_descriptor).unwrap().exists());
        let _ = std::fs::remove_dir_all(capture_root);
        let _ = std::fs::remove_dir_all(live_root);
    }

    #[test]
    fn restore_import_migrates_a_behind_capture_before_copying_trust() {
        let (capture_root, capture_descriptor) = temp_descriptor("restore-capture-behind");
        let capture_store = open_sqlite(&capture_descriptor).unwrap();
        capture_store.migrate(NAMESPACE, &MIGRATIONS[..18]).unwrap();
        capture_store
            .with_conn(|connection| {
                connection.execute(
                    "INSERT INTO knob_assignments VALUES ('machine', 'generate', 'balanced', 'llama', 'numeric', 'fp', 'llama', 1, 'os', 1, 3.0, 5.0)",
                    [],
                )?;
                Ok(())
            })
            .unwrap();
        drop(capture_store);

        let (live_root, live_descriptor) = temp_descriptor("restore-live-behind");
        let capture_path = sqlite_path(&capture_descriptor).unwrap();
        SynapseStore::restore_from_capture(&live_descriptor, &capture_path).unwrap();
        let restored = SynapseStore::open(&live_descriptor).unwrap();
        assert_eq!(table_count(&restored, "knob_assignments"), 1);
        drop(restored);
        let migrated_capture = SynapseStore::open(&capture_descriptor).unwrap();
        assert_eq!(table_count(&migrated_capture, "knob_assignments"), 1);
        let _ = std::fs::remove_dir_all(capture_root);
        let _ = std::fs::remove_dir_all(live_root);
    }

    #[test]
    fn profile_observation_recreates_the_empty_row_after_restore() {
        let (root, descriptor) = temp_descriptor("restore-empty-profile");
        let store = SynapseStore::open(&descriptor).unwrap();
        store
            .store
            .with_conn(|connection| {
                connection.execute("DELETE FROM profile_state", [])?;
                Ok(())
            })
            .unwrap();
        let profile = MachineProfile {
            os_build: "os".to_string(),
            arch: "arch".to_string(),
            chip_model: "chip".to_string(),
            ram_class: "ram".to_string(),
            ane_subtype: None,
            engine_identities: Vec::new(),
        };
        let activation = store.observe_profile(&profile, 1, 1).unwrap();
        assert_eq!(activation.state.profile_activation_epoch, Some(1));
        assert_eq!(table_count(&store, "profile_state"), 1);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn reopen_above_threshold_reclaims_freelist_and_shrinks_file() {
        let (root, descriptor) = temp_descriptor("reopen-above-threshold");
        let db_path = match &descriptor.backend {
            StorageBackend::Sqlite { path } => PathBuf::from(path),
            _ => unreachable!(),
        };
        let store = SynapseStore::open(&descriptor).unwrap();
        // Insert enough data to exceed RECLAIM_FREELIST_MIN_BYTES (64 MiB = 16,384 pages at 4 KiB).
        // 66 chunks of 1 MiB each = 66 MiB.
        store
            .store
            .with_conn(|conn| {
                conn.execute(
                    "CREATE TABLE bloat_seed (id INTEGER PRIMARY KEY, data BLOB)",
                    [],
                )?;
                let chunk = vec![0xfeu8; 1024 * 1024];
                let mut stmt = conn.prepare("INSERT INTO bloat_seed (data) VALUES (?1)")?;
                for _ in 0..66 {
                    stmt.execute(params![&chunk])?;
                }
                drop(stmt);
                conn.execute("DELETE FROM bloat_seed", [])?;
                Ok(())
            })
            .unwrap();

        let freelist_before = store.freelist_count().unwrap();
        let page_count_before = store.page_count().unwrap();
        let page_size = store.page_size().unwrap();
        assert!(freelist_before * page_size >= RECLAIM_FREELIST_MIN_BYTES);
        assert!(freelist_before >= page_count_before / RECLAIM_FREELIST_PAGE_RATIO_DIVISOR);
        drop(store);

        let file_size_before = std::fs::metadata(&db_path).unwrap().len();
        assert!(file_size_before >= 64 * 1024 * 1024);

        let reopened = SynapseStore::open(&descriptor).unwrap();
        let freelist_after = reopened.freelist_count().unwrap();
        let file_size_after = std::fs::metadata(&db_path).unwrap().len();

        assert_eq!(freelist_after, 0);
        assert!(
            file_size_after < file_size_before / 10,
            "expected file to shrink: before={file_size_before}, after={file_size_after}"
        );

        drop(reopened);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn reopen_below_threshold_does_not_vacuum() {
        let (root, descriptor) = temp_descriptor("reopen-below-threshold");
        let store = SynapseStore::open(&descriptor).unwrap();
        // Insert a small amount below 64 MiB: 512 KiB (128 pages).
        store
            .store
            .with_conn(|conn| {
                conn.execute(
                    "CREATE TABLE small_bloat (id INTEGER PRIMARY KEY, data BLOB)",
                    [],
                )?;
                let chunk = vec![0x42u8; 1024];
                let mut stmt = conn.prepare("INSERT INTO small_bloat (data) VALUES (?1)")?;
                for _ in 0..512 {
                    stmt.execute(params![&chunk])?;
                }
                drop(stmt);
                conn.execute("DELETE FROM small_bloat", [])?;
                Ok(())
            })
            .unwrap();

        let freelist_before = store.freelist_count().unwrap();
        let page_count_before = store.page_count().unwrap();
        let page_size = store.page_size().unwrap();
        // Free pages must satisfy the ratio threshold (>= page_count / 2) so that
        // the byte threshold alone protects against vacuuming.
        assert!(freelist_before >= 100);
        assert!(
            freelist_before >= page_count_before / RECLAIM_FREELIST_PAGE_RATIO_DIVISOR,
            "freelist={freelist_before}, page_count={page_count_before}"
        );
        assert!(freelist_before * page_size < RECLAIM_FREELIST_MIN_BYTES);
        drop(store);

        let reopened = SynapseStore::open(&descriptor).unwrap();
        let freelist_after = reopened.freelist_count().unwrap();

        // Freelist count must remain unchanged because no VACUUM ran.
        assert_eq!(freelist_after, freelist_before);

        drop(reopened);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    #[ignore]
    fn drill_bloat_reclaim_measured() {
        let (root, descriptor) = temp_descriptor("drill-bloat");
        let db_path = match &descriptor.backend {
            StorageBackend::Sqlite { path } => PathBuf::from(path),
            _ => unreachable!(),
        };
        let store = SynapseStore::open(&descriptor).expect("scratch store opens");
        // Bloat the scratch store past the threshold: 100 MB of blob rows
        store
            .store
            .with_conn(|conn| {
                conn.execute(
                    "CREATE TABLE drill_bloat (id INTEGER PRIMARY KEY, payload BLOB)",
                    [],
                )?;
                let chunk = vec![0xabu8; 1024 * 1024];
                let mut stmt = conn.prepare("INSERT INTO drill_bloat (payload) VALUES (?1)")?;
                for _ in 0..100 {
                    stmt.execute(params![&chunk])?;
                }
                drop(stmt);
                conn.execute("DELETE FROM drill_bloat", [])?;
                Ok(())
            })
            .expect("bloat inserted and deleted");

        let page_size = store.page_size().unwrap();
        let page_count_before = store.page_count().unwrap();
        let freelist_count_before = store.freelist_count().unwrap();
        drop(store);

        let file_size_before = std::fs::metadata(&db_path).unwrap().len();

        println!("\n=== MEASURED DRILL: BEFORE RECLAIM ===");
        println!("Database path: {}", db_path.display());
        println!("Page size: {page_size} bytes");
        println!("Page count before: {page_count_before}");
        println!("Freelist count before: {freelist_count_before}");
        println!(
            "Freelist bytes before: {} bytes ({:.2} MiB)",
            freelist_count_before * page_size,
            (freelist_count_before * page_size) as f64 / (1024.0 * 1024.0)
        );
        println!(
            "File size before: {file_size_before} bytes ({:.2} MiB)",
            file_size_before as f64 / (1024.0 * 1024.0)
        );

        let reopened = SynapseStore::open(&descriptor).expect("reopen triggers guarded vacuum");

        let page_count_after = reopened.page_count().unwrap();
        let freelist_count_after = reopened.freelist_count().unwrap();
        let file_size_after = std::fs::metadata(&db_path).unwrap().len();

        println!("=== MEASURED DRILL: AFTER RECLAIM ===");
        println!("Page count after: {page_count_after}");
        println!("Freelist count after: {freelist_count_after}");
        println!(
            "Freelist bytes after: {} bytes",
            freelist_count_after * page_size
        );
        println!(
            "File size after: {file_size_after} bytes ({:.2} MiB)",
            file_size_after as f64 / (1024.0 * 1024.0)
        );
        println!(
            "Freelist reduction: {} -> {}",
            freelist_count_before, freelist_count_after
        );
        println!(
            "File size reduction: {} -> {} bytes ({:.2} MiB -> {:.2} MiB)",
            file_size_before,
            file_size_after,
            file_size_before as f64 / (1024.0 * 1024.0),
            file_size_after as f64 / (1024.0 * 1024.0)
        );

        assert_eq!(freelist_count_after, 0);
        assert!(file_size_after < file_size_before / 10);
        let _ = std::fs::remove_dir_all(root);
    }

    /// In-memory stand-in for the model cache: blob sizes by digest, pinned
    /// digests and job ids with a staging directory.
    #[derive(Default)]
    struct FakeBlobCache {
        blobs: std::cell::RefCell<BTreeMap<String, u64>>,
        pins: std::cell::RefCell<BTreeSet<String>>,
        staging: std::cell::RefCell<BTreeSet<String>>,
    }

    impl FakeBlobCache {
        fn publish(&self, digest: &str, size: u64) {
            self.blobs.borrow_mut().insert(digest.to_string(), size);
        }

        fn has_blob(&self, digest: &str) -> bool {
            self.blobs.borrow().contains_key(digest)
        }

        fn pin(&self, digest: &str) {
            self.pins.borrow_mut().insert(digest.to_string());
        }

        fn pinned(&self, digest: &str) -> bool {
            self.pins.borrow().contains(digest)
        }

        fn stage(&self, job_id: &str) {
            self.staging.borrow_mut().insert(job_id.to_string());
        }

        fn staged(&self, job_id: &str) -> bool {
            self.staging.borrow().contains(job_id)
        }
    }

    impl CatalogBlobCache for FakeBlobCache {
        fn blob_size(&self, digest: &str) -> std::io::Result<Option<u64>> {
            Ok(self.blobs.borrow().get(digest).copied())
        }

        fn is_pinned(&self, digest: &str) -> std::io::Result<bool> {
            Ok(self.pinned(digest))
        }

        fn remove_pins(&self, digest: &str) -> std::io::Result<()> {
            self.pins.borrow_mut().remove(digest);
            Ok(())
        }

        fn delete_blob(&self, digest: &str) -> std::io::Result<u64> {
            Ok(self.blobs.borrow_mut().remove(digest).unwrap_or(0))
        }

        fn delete_staging(&self, job_id: &str) -> std::io::Result<()> {
            self.staging.borrow_mut().remove(job_id);
            Ok(())
        }
    }

    /// A blob digest made of one repeated hex character.
    fn blob(seed: char) -> String {
        seed.to_string().repeat(64)
    }

    fn download_request<'a>(
        request_key: &'a str,
        request_digest: &'a str,
        params_json: &'a Value,
        now_ms: u64,
        entry_complete: bool,
    ) -> DownloadJobRequest<'a> {
        DownloadJobRequest {
            request_key,
            request_digest,
            module_generation: 1,
            params_json,
            now_ms,
            result_retention_ttl_ms: 1_000,
            entry_complete,
        }
    }

    fn admit_download(
        store: &SynapseStore,
        request_key: &str,
        request_digest: &str,
        now_ms: u64,
    ) -> JobRecord {
        let params = serde_json::json!({ "request_digest": request_digest });
        store
            .admit_download_job(&download_request(
                request_key,
                request_digest,
                &params,
                now_ms,
                false,
            ))
            .unwrap()
            .record()
            .clone()
    }

    fn install(catalog_id: &str, backend: &str, members: &[(&str, &str)]) -> CatalogInstallRecord {
        CatalogInstallRecord {
            catalog_id: catalog_id.to_string(),
            manifest_digest: format!("manifest-{catalog_id}"),
            backend: backend.to_string(),
            members: members
                .iter()
                .map(|(path, digest)| CatalogInstallMember {
                    path: (*path).to_string(),
                    digest: (*digest).to_string(),
                })
                .collect(),
        }
    }

    fn acquisitions(store: &SynapseStore, job_id: &str) -> Vec<(String, bool)> {
        store
            .download_acquisitions(job_id)
            .unwrap()
            .into_iter()
            .map(|row| (row.digest, row.newly_published))
            .collect()
    }

    fn job_state(store: &SynapseStore, job_id: &str) -> String {
        store.get_job_at(job_id, 0).unwrap().unwrap().state
    }

    fn applied(transition: DownloadTransition) -> (JobRecord, DownloadCleanup) {
        match transition {
            DownloadTransition::Applied { record, cleanup } => (record, cleanup),
            other => panic!("expected the transition to apply, got {other:?}"),
        }
    }

    #[test]
    fn download_admission_shares_one_job_per_entry_manifest_and_binds_keys() {
        let (root, descriptor) = temp_descriptor("download-admission");
        let store = SynapseStore::open(&descriptor).unwrap();
        let params = serde_json::json!({ "catalog_id": "gte-modernbert-base" });

        let DownloadAdmission::Created(job) = store
            .admit_download_job(&download_request("key-a", "digest-1", &params, 10, false))
            .unwrap()
        else {
            panic!("the first download of a manifest mints a job");
        };
        assert_eq!(job.kind, DOWNLOAD_JOB_KIND);
        assert_eq!(job.state, DOWNLOAD_STATE_QUEUED);
        assert_eq!(job.request_key, format!("models.download:{}", job.job_id));
        assert_eq!(job.execution_expires_ms, None);
        assert_eq!(job.params_json.as_ref(), Some(&params));

        // A second key for the same manifest attaches to the same job.
        assert_eq!(
            store
                .admit_download_job(&download_request("key-b", "digest-1", &params, 11, false))
                .unwrap(),
            DownloadAdmission::Attached(job.clone())
        );
        // Reusing a key for another manifest is refused.
        match store.admit_download_job(&download_request("key-a", "digest-2", &params, 12, false)) {
            Err(SynapseStoreError::IdempotencyConflict {
                existing_digest,
                requested_digest,
                ..
            }) => {
                assert_eq!(existing_digest, "digest-1");
                assert_eq!(requested_digest, "digest-2");
            }
            other => panic!("expected an idempotency conflict, got {other:?}"),
        }
        // Another manifest gets its own job.
        let other = admit_download(&store, "key-c", "digest-2", 13);
        assert_ne!(other.job_id, job.job_id);

        applied(store.commit_download_job(&job.job_id, &[], 20).unwrap());
        // While the entry is complete the key keeps returning its committed job.
        let DownloadAdmission::Attached(committed) = store
            .admit_download_job(&download_request("key-a", "digest-1", &params, 21, true))
            .unwrap()
        else {
            panic!("a complete entry returns the committed job");
        };
        assert_eq!(committed.job_id, job.job_id);
        assert_eq!(committed.state, DOWNLOAD_STATE_COMMITTED);

        // Once the entry is incomplete the key binds to a fresh job, and a key
        // still bound to the committed job attaches to that fresh job.
        let DownloadAdmission::Created(repair) = store
            .admit_download_job(&download_request("key-a", "digest-1", &params, 22, false))
            .unwrap()
        else {
            panic!("an incomplete entry mints a new job");
        };
        assert_ne!(repair.job_id, job.job_id);
        assert_eq!(
            store.download_key_binding("key-a").unwrap().unwrap().job_id,
            repair.job_id
        );
        assert_eq!(
            store
                .admit_download_job(&download_request("key-b", "digest-1", &params, 23, false))
                .unwrap(),
            DownloadAdmission::Attached(repair.clone())
        );
        assert_eq!(
            store
                .active_download_job("digest-1")
                .unwrap()
                .unwrap()
                .job_id,
            repair.job_id
        );
        drop(store);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn download_jobs_are_exempt_from_execution_ttl_and_purge_drops_their_bindings() {
        let (root, descriptor) = temp_descriptor("download-ttl");
        let store = SynapseStore::open(&descriptor).unwrap();
        let cache = FakeBlobCache::default();
        let download = admit_download(&store, "key-a", "digest-1", 10);
        let running = admit_download(&store, "key-b", "digest-2", 10);
        let load = store
            .admit_job(
                "load-key",
                "load-digest",
                "model.load",
                1,
                &serde_json::json!({}),
                10,
                100,
                1_000,
            )
            .unwrap()
            .record()
            .clone();

        // Far past any execution TTL: the load job expires, downloads do not.
        store.purge_expired_jobs(1_000_000).unwrap();
        assert_eq!(job_state(&store, &load.job_id), JOB_STATE_FAILED_TRANSIENT);
        assert_eq!(
            store
                .get_job_at(&download.job_id, 1_000_000)
                .unwrap()
                .unwrap()
                .state,
            DOWNLOAD_STATE_QUEUED
        );
        // The generic restart sweep leaves downloads to their own startup path.
        store
            .fail_prior_generation_incomplete_jobs(5, &serde_json::json!({}), 1_000_001)
            .unwrap();
        assert_eq!(job_state(&store, &download.job_id), DOWNLOAD_STATE_QUEUED);

        applied(
            store
                .end_download_job(&download.job_id, DownloadEnd::Cancelled, &cache, 2_000_000)
                .unwrap(),
        );
        store.purge_expired_jobs(2_000_999).unwrap();
        assert!(store.get_job_at(&download.job_id, 0).unwrap().is_some());
        assert!(store.download_key_binding("key-a").unwrap().is_some());

        // Retention elapsed: the terminal job and its binding go together; the
        // non-terminal job and its binding stay.
        store.purge_expired_jobs(2_001_000).unwrap();
        assert!(store.get_job_at(&download.job_id, 0).unwrap().is_none());
        assert!(store.download_key_binding("key-a").unwrap().is_none());
        assert_eq!(job_state(&store, &running.job_id), DOWNLOAD_STATE_QUEUED);
        assert_eq!(
            store.download_key_binding("key-b").unwrap().unwrap().job_id,
            running.job_id
        );
        drop(store);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn publication_marks_only_absent_blobs_newly_published_and_stops_once_terminal() {
        let (root, descriptor) = temp_descriptor("download-publication");
        let store = SynapseStore::open(&descriptor).unwrap();
        let cache = FakeBlobCache::default();
        cache.publish(&blob('a'), 5);
        let job = admit_download(&store, "key-a", "digest-1", 10);
        assert!(store
            .advance_download_job(&job.job_id, DOWNLOAD_STATE_DOWNLOADING, 11)
            .unwrap());

        assert_eq!(
            store
                .record_download_publication(&job.job_id, &blob('b'), &cache)
                .unwrap(),
            DownloadPublication::Proceed {
                newly_published: true
            }
        );
        assert_eq!(
            store
                .record_download_publication(&job.job_id, &blob('a'), &cache)
                .unwrap(),
            DownloadPublication::Proceed {
                newly_published: false
            }
        );
        // A prefixed, upper-case digest names the same row.
        let prefixed = format!("sha256:{}", blob('b').to_ascii_uppercase());
        store
            .record_download_publication(&job.job_id, &prefixed, &cache)
            .unwrap();
        assert_eq!(
            acquisitions(&store, &job.job_id),
            vec![(blob('a'), false), (blob('b'), true)]
        );

        applied(
            store
                .end_download_job(&job.job_id, DownloadEnd::Cancelled, &cache, 12)
                .unwrap(),
        );
        let DownloadPublication::Stopped(stopped) = store
            .record_download_publication(&job.job_id, &blob('c'), &cache)
            .unwrap()
        else {
            panic!("a terminal job must not publish");
        };
        assert_eq!(stopped.state, DOWNLOAD_STATE_CANCELLED);
        assert!(acquisitions(&store, &job.job_id).is_empty());
        assert!(!store
            .advance_download_job(&job.job_id, DOWNLOAD_STATE_VERIFYING, 13)
            .unwrap());
        drop(store);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn reuse_root_writes_a_not_newly_published_row_only_for_a_present_blob_of_its_size() {
        let (root, descriptor) = temp_descriptor("download-reuse");
        let store = SynapseStore::open(&descriptor).unwrap();
        let cache = FakeBlobCache::default();
        cache.publish(&blob('a'), 5);
        let job = admit_download(&store, "key-a", "digest-1", 10);

        assert_eq!(
            store
                .root_reused_download_blob(&job.job_id, &blob('b'), 5, &cache)
                .unwrap(),
            DownloadReuse::Absent
        );
        assert_eq!(
            store
                .root_reused_download_blob(&job.job_id, &blob('a'), 6, &cache)
                .unwrap(),
            DownloadReuse::Absent
        );
        assert!(acquisitions(&store, &job.job_id).is_empty());
        assert_eq!(
            store
                .root_reused_download_blob(&job.job_id, &blob('a'), 5, &cache)
                .unwrap(),
            DownloadReuse::Rooted
        );
        assert_eq!(acquisitions(&store, &job.job_id), vec![(blob('a'), false)]);

        // Rooting a blob this job itself published keeps it newly published.
        store
            .record_download_publication(&job.job_id, &blob('b'), &cache)
            .unwrap();
        cache.publish(&blob('b'), 3);
        assert_eq!(
            store
                .root_reused_download_blob(&job.job_id, &blob('b'), 3, &cache)
                .unwrap(),
            DownloadReuse::Rooted
        );
        assert_eq!(
            acquisitions(&store, &job.job_id),
            vec![(blob('a'), false), (blob('b'), true)]
        );

        // Cleanup deletes what the job published and never what it reused.
        let (_, cleanup) = applied(
            store
                .end_download_job(&job.job_id, DownloadEnd::Cancelled, &cache, 11)
                .unwrap(),
        );
        assert_eq!(cleanup.deleted_blobs, vec![blob('b')]);
        assert!(cache.has_blob(&blob('a')));
        assert!(!cache.has_blob(&blob('b')));
        assert!(matches!(
            store
                .root_reused_download_blob(&job.job_id, &blob('a'), 5, &cache)
                .unwrap(),
            DownloadReuse::Stopped(_)
        ));
        assert!(acquisitions(&store, &job.job_id).is_empty());
        drop(store);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn commit_writes_installs_drops_acquisitions_and_wins_exactly_once() {
        let (root, descriptor) = temp_descriptor("download-commit");
        let store = SynapseStore::open(&descriptor).unwrap();
        let cache = FakeBlobCache::default();
        cache.publish(&blob('b'), 2);
        let job = admit_download(&store, "key-a", "digest-1", 10);
        store
            .record_download_publication(&job.job_id, &blob('a'), &cache)
            .unwrap();
        cache.publish(&blob('a'), 1);
        store
            .root_reused_download_blob(&job.job_id, &blob('b'), 2, &cache)
            .unwrap();

        let installs = vec![
            install(
                "gte",
                "metal",
                &[
                    ("model.safetensors", &blob('a')),
                    ("tokenizer.json", &blob('b')),
                ],
            ),
            install("gte", "ane", &[("tokenizer.json", &blob('b'))]),
        ];
        let (record, cleanup) = applied(
            store
                .commit_download_job(&job.job_id, &installs, 20)
                .unwrap(),
        );
        assert_eq!(record.state, DOWNLOAD_STATE_COMMITTED);
        assert_eq!(record.terminal_at_ms, Some(20));
        assert_eq!(cleanup, DownloadCleanup::default());
        let mut expected = installs.clone();
        expected.sort_by(|left, right| left.backend.cmp(&right.backend));
        assert_eq!(store.catalog_installs("gte").unwrap(), expected);
        assert!(acquisitions(&store, &job.job_id).is_empty());

        // The commit won; a second commit, a cancel and a failure all lose and
        // change nothing.
        assert_eq!(
            store.commit_download_job(&job.job_id, &[], 21).unwrap(),
            DownloadTransition::Unchanged(Some(record.clone()))
        );
        assert_eq!(
            store
                .end_download_job(&job.job_id, DownloadEnd::Cancelled, &cache, 22)
                .unwrap(),
            DownloadTransition::Unchanged(Some(record.clone()))
        );
        let error = serde_json::json!({ "code": "download_failed" });
        assert_eq!(
            store
                .end_download_job(
                    &job.job_id,
                    DownloadEnd::Failed { error_json: &error },
                    &cache,
                    23
                )
                .unwrap(),
            DownloadTransition::Unchanged(Some(record))
        );
        assert!(cache.has_blob(&blob('a')) && cache.has_blob(&blob('b')));
        assert_eq!(store.catalog_installs("gte").unwrap(), expected);

        // A later repair commit replaces the member set of the same record.
        let repair = admit_download(&store, "key-a", "digest-1", 30);
        assert_ne!(repair.job_id, job.job_id);
        let replacement = install(
            "gte",
            "metal",
            &[
                ("model.safetensors", &blob('c')),
                ("tokenizer.json", &blob('b')),
            ],
        );
        applied(
            store
                .commit_download_job(&repair.job_id, std::slice::from_ref(&replacement), 31)
                .unwrap(),
        );
        assert_eq!(
            store.catalog_installs("gte").unwrap(),
            vec![expected[0].clone(), replacement]
        );
        assert_eq!(
            store.commit_download_job("job_missing", &[], 32).unwrap(),
            DownloadTransition::Unchanged(None)
        );
        drop(store);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn cancel_cleanup_deletes_only_unreferenced_blobs_the_job_published() {
        let (root, descriptor) = temp_descriptor("download-cancel");
        let store = SynapseStore::open(&descriptor).unwrap();
        let cache = FakeBlobCache::default();
        // e: pre-existing and reused, so never deleted.
        cache.publish(&blob('e'), 50);
        let job = admit_download(&store, "key-a", "digest-1", 10);
        for (seed, size) in [('a', 1), ('b', 2), ('c', 3), ('d', 4)] {
            assert_eq!(
                store
                    .record_download_publication(&job.job_id, &blob(seed), &cache)
                    .unwrap(),
                DownloadPublication::Proceed {
                    newly_published: true
                }
            );
            cache.publish(&blob(seed), size);
        }
        store
            .root_reused_download_blob(&job.job_id, &blob('e'), 50, &cache)
            .unwrap();
        cache.stage(&job.job_id);

        // b: a member row of another entry's install references it.
        let other = admit_download(&store, "key-o", "digest-o", 11);
        applied(
            store
                .commit_download_job(
                    &other.job_id,
                    &[install("other", "metal", &[("model", &blob('b'))])],
                    12,
                )
                .unwrap(),
        );
        // c: a cache.pin holds it.
        cache.pin(&blob('c'));
        // d: a free-form registration names it.
        let mut free_form = registration_fixture("llama-fixture");
        free_form.model_id = "free-form".to_string();
        free_form.model_locator = ModelAssetLocator::CacheDigest {
            digest: format!("sha256:{}", blob('d')),
        };
        store.upsert_model(&free_form, 13).unwrap();

        let (record, cleanup) = applied(
            store
                .end_download_job(&job.job_id, DownloadEnd::Cancelled, &cache, 14)
                .unwrap(),
        );
        assert_eq!(record.state, DOWNLOAD_STATE_CANCELLED);
        assert_eq!(record.error_json, None);
        assert_eq!(
            cleanup,
            DownloadCleanup {
                deleted_blobs: vec![blob('a')],
                freed_bytes: 1,
                handed_off: Vec::new(),
                kept: vec![blob('b'), blob('c'), blob('d')],
            }
        );
        assert!(!cache.has_blob(&blob('a')));
        for seed in ['b', 'c', 'd', 'e'] {
            assert!(cache.has_blob(&blob(seed)), "blob {seed} was deleted");
        }
        assert!(!cache.staged(&job.job_id));
        assert!(acquisitions(&store, &job.job_id).is_empty());

        // The cancel won: a late commit writes nothing.
        assert!(matches!(
            store
                .commit_download_job(
                    &job.job_id,
                    &[install("gte", "metal", &[("model", &blob('a'))])],
                    15,
                )
                .unwrap(),
            DownloadTransition::Unchanged(Some(_))
        ));
        assert!(store.catalog_installs("gte").unwrap().is_empty());
        drop(store);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn cleanup_hands_a_shared_newly_published_blob_to_the_job_that_reused_it() {
        let (root, descriptor) = temp_descriptor("download-hand-off");
        let store = SynapseStore::open(&descriptor).unwrap();
        let cache = FakeBlobCache::default();

        // Job a publishes the blob, job b reuses it, and a is cancelled
        // before commit.
        let job_a = admit_download(&store, "key-a", "digest-a", 10);
        let job_b = admit_download(&store, "key-b", "digest-b", 10);
        store
            .record_download_publication(&job_a.job_id, &blob('a'), &cache)
            .unwrap();
        cache.publish(&blob('a'), 7);
        assert_eq!(
            store
                .root_reused_download_blob(&job_b.job_id, &blob('a'), 7, &cache)
                .unwrap(),
            DownloadReuse::Rooted
        );
        let (_, cleanup) = applied(
            store
                .end_download_job(&job_a.job_id, DownloadEnd::Cancelled, &cache, 11)
                .unwrap(),
        );
        assert_eq!(cleanup.handed_off, vec![blob('a')]);
        assert!(cleanup.deleted_blobs.is_empty());
        assert!(cache.has_blob(&blob('a')));
        assert_eq!(acquisitions(&store, &job_b.job_id), vec![(blob('a'), true)]);
        // b commits with every member present and nothing refetched.
        applied(
            store
                .commit_download_job(
                    &job_b.job_id,
                    &[install("shared", "metal", &[("model", &blob('a'))])],
                    12,
                )
                .unwrap(),
        );
        assert!(cache.has_blob(&blob('a')));

        // Same hand-off, but the job that took the blob over then fails: its
        // cleanup now deletes the blob, and the failure keeps its error.
        let job_c = admit_download(&store, "key-c", "digest-c", 20);
        let job_d = admit_download(&store, "key-d", "digest-d", 20);
        store
            .record_download_publication(&job_c.job_id, &blob('c'), &cache)
            .unwrap();
        cache.publish(&blob('c'), 9);
        store
            .root_reused_download_blob(&job_d.job_id, &blob('c'), 9, &cache)
            .unwrap();
        cache.stage(&job_d.job_id);
        applied(
            store
                .end_download_job(&job_c.job_id, DownloadEnd::Cancelled, &cache, 21)
                .unwrap(),
        );
        let error = serde_json::json!({
            "code": "download_failed",
            "details": { "reason": "network" },
        });
        let (record, cleanup) = applied(
            store
                .end_download_job(
                    &job_d.job_id,
                    DownloadEnd::Failed { error_json: &error },
                    &cache,
                    22,
                )
                .unwrap(),
        );
        assert_eq!(record.state, DOWNLOAD_STATE_FAILED);
        assert_eq!(record.error_json, Some(error));
        assert_eq!(cleanup.deleted_blobs, vec![blob('c')]);
        assert_eq!(cleanup.freed_bytes, 9);
        assert!(!cache.has_blob(&blob('c')));
        assert!(!cache.staged(&job_d.job_id));
        drop(store);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn quarantine_removes_the_blob_its_installs_members_and_pins_and_fails_holding_jobs() {
        let (root, descriptor) = temp_descriptor("download-quarantine");
        let store = SynapseStore::open(&descriptor).unwrap();
        let cache = FakeBlobCache::default();
        let (bad, shared, other) = (blob('a'), blob('b'), blob('c'));
        cache.publish(&bad, 10);
        cache.publish(&shared, 20);
        cache.publish(&other, 30);
        cache.pin(&bad);
        cache.pin(&shared);
        let setup = admit_download(&store, "key-setup", "digest-setup", 10);
        applied(
            store
                .commit_download_job(
                    &setup.job_id,
                    &[
                        install("gte", "metal", &[("model", &bad), ("tokenizer", &shared)]),
                        install("qwen", "metal", &[("model", &bad), ("tokenizer", &other)]),
                        install(
                            "rerank",
                            "metal",
                            &[("model", &other), ("tokenizer", &shared)],
                        ),
                    ],
                    11,
                )
                .unwrap(),
        );
        let mut free_form = registration_fixture("llama-fixture");
        free_form.model_id = "free-form".to_string();
        free_form.model_locator = ModelAssetLocator::CacheDigest {
            digest: format!("sha256:{bad}"),
        };
        store.upsert_model(&free_form, 12).unwrap();

        // A running job that reused the bad blob and published another one,
        // and a running job that never touched it.
        let holder = admit_download(&store, "key-h", "digest-h", 13);
        store
            .root_reused_download_blob(&holder.job_id, &bad, 10, &cache)
            .unwrap();
        store
            .record_download_publication(&holder.job_id, &blob('d'), &cache)
            .unwrap();
        cache.publish(&blob('d'), 4);
        cache.stage(&holder.job_id);
        let bystander = admit_download(&store, "key-x", "digest-x", 13);
        store
            .record_download_publication(&bystander.job_id, &blob('e'), &cache)
            .unwrap();
        cache.publish(&blob('e'), 5);

        store
            .store
            .with_conn_fenced(|tx| {
                tx.execute(
                    "INSERT INTO catalog_file_verifications
                SELECT catalog_id, manifest_digest, backend, path, digest, '1','2','3','4','5'
                FROM catalog_install_members",
                    [],
                )?;
                Ok(())
            })
            .unwrap();
        let verified_before = table_count(&store, "catalog_file_verifications");
        let error = serde_json::json!({ "code": "artifact_invalid" });
        let report = store
            .quarantine_catalog_blob(&format!("sha256:{bad}"), &error, &cache, 20)
            .unwrap();
        assert_eq!(
            report,
            QuarantineReport {
                removed_installs: vec![
                    CatalogInstallKey {
                        catalog_id: "gte".to_string(),
                        manifest_digest: "manifest-gte".to_string(),
                        backend: "metal".to_string(),
                    },
                    CatalogInstallKey {
                        catalog_id: "qwen".to_string(),
                        manifest_digest: "manifest-qwen".to_string(),
                        backend: "metal".to_string(),
                    },
                ],
                failed_jobs: vec![holder.job_id.clone()],
                freed_bytes: 10 + 4,
            }
        );
        assert_eq!(
            table_count(&store, "catalog_file_verifications"),
            verified_before - 2
        );
        assert!(!cache.has_blob(&bad));
        assert!(!cache.pinned(&bad));
        assert!(cache.pinned(&shared));
        for digest in [&shared, &other, &blob('e')] {
            assert!(cache.has_blob(digest), "blob {digest} was touched");
        }
        assert!(!cache.has_blob(&blob('d')), "the failed job's cleanup ran");
        assert!(!cache.staged(&holder.job_id));
        assert!(store.catalog_installs("gte").unwrap().is_empty());
        assert!(store.catalog_installs("qwen").unwrap().is_empty());
        assert_eq!(store.catalog_installs("rerank").unwrap().len(), 1);
        // Only member rows naming the bad blob go; the others keep their blobs
        // rooted so the repair download fetches only the quarantined file.
        let members = store
            .store
            .with_conn(|conn| {
                let mut stmt = conn.prepare(
                    "SELECT catalog_id, digest FROM catalog_install_members
                     ORDER BY catalog_id, path",
                )?;
                let rows = stmt
                    .query_map([], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                    })?
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .unwrap();
        assert_eq!(
            members,
            vec![
                ("gte".to_string(), shared.clone()),
                ("qwen".to_string(), other.clone()),
                ("rerank".to_string(), other.clone()),
                ("rerank".to_string(), shared.clone()),
            ]
        );
        let failed = store.get_job_at(&holder.job_id, 0).unwrap().unwrap();
        assert_eq!(failed.state, DOWNLOAD_STATE_FAILED);
        assert_eq!(failed.error_json, Some(error));
        assert!(acquisitions(&store, &holder.job_id).is_empty());
        assert_eq!(job_state(&store, &bystander.job_id), DOWNLOAD_STATE_QUEUED);
        assert_eq!(
            acquisitions(&store, &bystander.job_id),
            vec![(blob('e'), true)]
        );
        assert!(store
            .catalog_models()
            .unwrap()
            .iter()
            .any(|model| model.model_id == "free-form"));
        drop(store);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn startup_fails_interrupted_download_jobs_with_cleanup() {
        let (root, descriptor) = temp_descriptor("download-startup");
        let store = SynapseStore::open(&descriptor).unwrap();
        let cache = FakeBlobCache::default();
        let queued = admit_download(&store, "key-q", "digest-q", 10);
        store
            .record_download_publication(&queued.job_id, &blob('a'), &cache)
            .unwrap();
        cache.publish(&blob('a'), 3);
        cache.stage(&queued.job_id);
        let verifying = admit_download(&store, "key-v", "digest-v", 11);
        store
            .advance_download_job(&verifying.job_id, DOWNLOAD_STATE_VERIFYING, 12)
            .unwrap();
        let committed = admit_download(&store, "key-c", "digest-c", 12);
        applied(
            store
                .commit_download_job(&committed.job_id, &[], 13)
                .unwrap(),
        );
        let params = serde_json::json!({});
        let mut current = download_request("key-n", "digest-n", &params, 14, false);
        current.module_generation = 2;
        let current = store.admit_download_job(&current).unwrap().record().clone();

        let error = serde_json::json!({ "code": "module_restarted" });
        let failed = store
            .fail_interrupted_download_jobs(2, &error, &cache, 20)
            .unwrap();
        assert_eq!(
            failed,
            vec![queued.job_id.clone(), verifying.job_id.clone()]
        );
        for job_id in &failed {
            let record = store.get_job_at(job_id, 0).unwrap().unwrap();
            assert_eq!(record.state, DOWNLOAD_STATE_FAILED);
            assert_eq!(record.error_json.as_ref(), Some(&error));
            assert!(acquisitions(&store, job_id).is_empty());
        }
        assert!(!cache.has_blob(&blob('a')));
        assert!(!cache.staged(&queued.job_id));
        assert_eq!(
            job_state(&store, &committed.job_id),
            DOWNLOAD_STATE_COMMITTED
        );
        assert_eq!(job_state(&store, &current.job_id), DOWNLOAD_STATE_QUEUED);
        drop(store);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn startup_reconciliation_deletes_disallowed_embed_rerank_rows_and_skips_reserved_ids() {
        let (root, descriptor) = temp_descriptor("registration-reconcile");
        let store = SynapseStore::open(&descriptor).unwrap();
        let cache = FakeBlobCache::default();
        let registration = |model_id: &str, engine: &str, task: &str, digests: &[String]| {
            let mut config = registration_fixture("llama-fixture");
            config.model_id = model_id.to_string();
            config.engine = engine.to_string();
            config.task = task.to_string();
            if matches!(model_id, "owned-embed" | "direct-ane-embed") {
                config
                    .engine_identity
                    .build_flags
                    .insert("profile".into(), format!("{model_id}-profile"));
            }
            config.artifact_digest = format!("sha256:{}", digests[0]);
            config.model_locator = ModelAssetLocator::CacheDigest {
                digest: format!("sha256:{}", digests[0]),
            };
            config.tokenizer_locator = ModelAssetLocator::CacheDigest {
                digest: format!("sha256:{}", digests[digests.len() - 1]),
            };
            config
        };
        let rows = [
            registration("minilm", "ort", "embed", &[blob('a'), blob('b')]),
            registration("bge-rerank", "llama", "rerank", &[blob('c')]),
            registration("gte-modernbert-base", "owned-metal", "embed", &[blob('d')]),
            registration("owned-embed", "owned-metal", "embed", &[blob('c')]),
            registration(
                "direct-ane-embed",
                synapse_core::ANE_DIRECT_WORKER_ENGINE,
                "embed",
                &[blob('c')],
            ),
            registration("decoder", "llama", "generate", &[blob('e')]),
        ];
        for (index, row) in rows.iter().enumerate() {
            store.upsert_model(row, 10 + index as u64).unwrap();
        }
        for seed in ['a', 'b', 'c', 'd', 'e'] {
            cache.pin(&blob(seed));
        }

        let reconciliation = store
            .reconcile_persisted_registrations(&|id| id == "gte-modernbert-base", &cache)
            .unwrap();
        assert_eq!(
            reconciliation.deleted,
            vec![
                ("minilm".to_string(), "ort".to_string()),
                ("bge-rerank".to_string(), "llama".to_string()),
            ]
        );
        assert_eq!(
            reconciliation
                .registrable
                .iter()
                .map(|model| model.model_id.as_str())
                .collect::<Vec<_>>(),
            vec!["owned-embed", "direct-ane-embed", "decoder"]
        );
        assert_eq!(
            reconciliation.skipped_reserved,
            vec!["gte-modernbert-base".to_string()]
        );
        // c stays pinned: the kept owned-metal row still names it.
        assert_eq!(reconciliation.unpinned_digests, vec![blob('a'), blob('b')]);
        assert!(!cache.pinned(&blob('a')) && !cache.pinned(&blob('b')));
        for seed in ['c', 'd', 'e'] {
            assert!(cache.pinned(&blob(seed)), "pin on {seed} was dropped");
        }
        // The reserved row stays in the table; only the disallowed rows go.
        assert_eq!(
            store
                .catalog_models()
                .unwrap()
                .iter()
                .map(|model| model.model_id.as_str())
                .collect::<Vec<_>>(),
            vec![
                "gte-modernbert-base",
                "owned-embed",
                "direct-ane-embed",
                "decoder"
            ]
        );
        drop(store);
        let _ = std::fs::remove_dir_all(root);
    }

    pub(super) fn temp_descriptor(label: &str) -> (std::path::PathBuf, StorageDescriptor) {
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
