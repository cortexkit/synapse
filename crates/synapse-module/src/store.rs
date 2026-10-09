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
use synapse_core::{
    AliasRow, AliasTable, EngineIdentity, Fingerprint, MachineProfile, NumericProfileId,
};
use thiserror::Error;

use crate::{owned_decode_certification::CertificationRecord, PerfKnob};

const NAMESPACE: &str = "synapse_module";
const APPROVAL_MIGRATION_SEED_DIGEST: &str =
    "a799f7e694991b0fd47902f9959c168990d5b0a3ec6ce2e7d6d3ac184bf80103";
pub const JOB_STATE_QUEUED: &str = "queued";
pub const JOB_STATE_RUNNING: &str = "running";
pub const JOB_STATE_PAUSED_NEEDS_REAUTH: &str = "paused_needs_reauth";
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

static MAINTENANCE_VACUUM_COUNT: AtomicU64 = AtomicU64::new(0);

/// Return the total number of guarded freelist VACUUM operations executed in this process.
///
/// Intended for telemetry and test assertions verifying whether maintenance reclamation ran.
#[doc(hidden)]
pub fn maintenance_vacuum_count() -> u64 {
    MAINTENANCE_VACUUM_COUNT.load(Ordering::Relaxed)
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
];

static JOB_COUNTER: AtomicU64 = AtomicU64::new(0);

// This independent inventory makes omission from the delete/import policy a
// refusal instead of silently narrowing the restore contract.
const RESTORE_APPLICATION_TABLES: &[&str] = &[
    "alias_rows",
    "approval_digest_corruption_events",
    "approval_migration_markers",
    "approvals",
    "cert_row_rebuild_events",
    "catalog_file_verifications",
    "catalog_install_members",
    "catalog_installs",
    "catalog_self_checks",
    "cert_rows",
    "download_acquisitions",
    "download_key_bindings",
    "jobs",
    "knob_assignments",
    "models",
    "perf_rows",
    "profile_rotation_certification_outcomes",
    "profile_rotation_events",
    "profile_state",
    "remote_checkpoints",
    "remote_url_bindings",
    "result_pages",
    "serving_approvals",
    "serving_artifacts",
    "serving_certification_records",
    "serving_retained_states",
    "serving_sessions",
];

const RESTORE_KEEP_TABLES: &[&str] = &[
    "approvals",
    "serving_approvals",
    "approval_migration_markers",
    "alias_rows",
    "knob_assignments",
];

// All jobs, including terminal rows, are discarded so restored request keys
// cannot point at result pages or checkpoints from an older execution world.
// Result/model/artifact/URL caches are also cleared rather than imported; every
// consumer already treats their absence as a cache miss. Catalog install
// records and self-check rows are cleared the same way: the blobs stay in the
// content-addressed cache, so the next `models.download` reuses them without
// network I/O and the self-check reruns on first use. `module_meta` is the
// one cache kept from the live store because its generation and table epoch
// fence concurrent writers and must never regress to captured values.
//
// Deletion order follows the live foreign-key graph even though restored
// serving approvals intentionally outlive discarded certifications and
// artifacts. Foreign keys are disabled only for this transaction: admission
// reloads both missing rows and refuses with CertificationMismatch until a
// probe re-derives them.
const RESTORE_CLEAR_TABLES: &[&str] = &[
    "serving_sessions",
    "serving_retained_states",
    "serving_approvals",
    "serving_certification_records",
    "serving_artifacts",
    "remote_checkpoints",
    "result_pages",
    "download_key_bindings",
    "download_acquisitions",
    "jobs",
    "catalog_file_verifications",
    "catalog_install_members",
    "catalog_installs",
    "catalog_self_checks",
    "profile_rotation_certification_outcomes",
    "profile_rotation_events",
    "profile_state",
    "cert_rows",
    "perf_rows",
    "approval_digest_corruption_events",
    "cert_row_rebuild_events",
    "remote_url_bindings",
    "models",
    "approvals",
    "approval_migration_markers",
    "alias_rows",
    "knob_assignments",
];

// `self_check_run_seq` stays with the live store for the same reason as
// `module_meta`. Each self-check run takes the next value as its generation,
// and a completion is written only if its row still carries that generation.
// Moving the counter back to a captured value would hand out generations
// already used, so a stale completion could match a newer run's row.
const RESTORE_LIVE_ONLY_TABLES: &[&str] = &[
    "module_meta",
    "self_check_run_seq",
    "cortexkit_fence",
    "cortexkit_schema_version",
];

/// The reason recorded on approvals and serving approvals forced disabled during
/// a restore operation. Restored approvals remain disabled until an operator
/// explicitly inspects and re-enables them.
pub const RESTORE_DISABLED_REASON: &str = "restored approval requires operator re-enable";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RestoreImportReport {
    pub imported_rows: BTreeMap<String, u64>,
    pub disabled_on_import: BTreeMap<String, u64>,
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
    #[error("remote profile '{remote_profile_hash}' changed base URL from '{previous_url}' to '{requested_url}'")]
    RemoteDeploymentChanged {
        remote_profile_hash: String,
        previous_url: String,
        requested_url: String,
    },
    #[error("cert_rows_rebuild_failed: {0}")]
    CertRowsRebuildFailed(String),
    #[error("approval digest mismatch for ({model_id}, {decode_fingerprint}): observed {observed}, recomputed {recomputed}")]
    ApprovalDigestMismatch {
        model_id: String,
        decode_fingerprint: String,
        observed: String,
        recomputed: String,
    },
    #[error("profile state corruption: {0}")]
    ProfileStateCorrupt(String),
    #[error("profile activation compare-and-set lost to another observer")]
    ProfileActivationLost,
    #[error("approval migration state corrupt: {0}")]
    ApprovalMigrationStateCorrupt(String),
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
    /// Whether this lane is approved to take traffic: `"enabled"` or
    /// `"disabled"`, omitted for lane classes with no approval concept.
    ///
    /// Separate from `certified` because they are separate facts and a lane can
    /// hold one without the other. `certified` says evidence exists for this
    /// machine; this says an operator approved serving. A lane that is certified
    /// and unapproved refuses, which is why a consumer choosing where to send
    /// traffic wants this field rather than `certified`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub serving_admission: Option<String>,
    /// Why `serving_admission` is `"disabled"`; absent when it is enabled or
    /// when the lane has no approval concept.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub serving_admission_reason: Option<String>,
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

    // Staged storage API consumed by the epic's runtime slice; remove this allow there.
    #[allow(dead_code)]
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

/// Explicit discriminator for every class that shares `cert_rows`.
///
/// In particular, `LegacyOwnedDecode` is deliberately not an alias for
/// `MeasuredOwnedDecode`: legacy rows have no complete runtime binding and are
/// retained for audit only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CertificationClass {
    MeasuredOwnedDecode,
    Declared,
    Remote,
    Embedding,
    Rerank,
    LegacyOwnedDecode,
}

impl CertificationClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MeasuredOwnedDecode => "measured_owned_decode",
            Self::Declared => "declared",
            Self::Remote => "remote",
            Self::Embedding => "embedding",
            Self::Rerank => "rerank",
            Self::LegacyOwnedDecode => "legacy_owned_decode",
        }
    }

    fn parse(value: &str) -> Result<Self, SynapseStoreError> {
        match value {
            "measured_owned_decode" => Ok(Self::MeasuredOwnedDecode),
            "declared" => Ok(Self::Declared),
            "remote" => Ok(Self::Remote),
            "embedding" => Ok(Self::Embedding),
            "rerank" => Ok(Self::Rerank),
            "legacy_owned_decode" => Ok(Self::LegacyOwnedDecode),
            other => Err(SynapseStoreError::Decode(format!(
                "unknown certification class '{other}'"
            ))),
        }
    }
}

// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApprovalIdentity {
    pub model_id: String,
    pub decode_fingerprint: String,
}

// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
impl ApprovalIdentity {
    pub fn new(model_id: impl Into<String>, decode_fingerprint: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            decode_fingerprint: decode_fingerprint.into(),
        }
    }
}

// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApprovalRow {
    pub schema_revision: String,
    pub model_id: String,
    pub decode_fingerprint: String,
    pub enabled: bool,
    pub grammar_enabled: bool,
    pub disabled_reason: Option<String>,
    pub approved_by: Option<String>,
    pub approved_at_ms: Option<u64>,
    pub updated_at_ms: u64,
    pub evidence_requirements_revision: String,
    pub semantic_digest: String,
    pub row_id: u64,
    pub generation: u64,
    pub fencing_metadata: Value,
}

// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
impl ApprovalRow {
    pub fn expected_digest(&self) -> Result<String, SynapseStoreError> {
        let separator = if self.schema_revision == APPROVAL_SCHEMA_REVISION {
            APPROVAL_DIGEST_DOMAIN.to_string()
        } else {
            format!("owned-decode-approval-row-{}\0", self.schema_revision)
        };
        let mut hasher = Sha256::new();
        hasher.update(separator.as_bytes());
        digest_text(&mut hasher, &self.schema_revision);
        digest_text(&mut hasher, &self.model_id);
        digest_text(&mut hasher, &self.decode_fingerprint);
        digest_text(&mut hasher, if self.enabled { "true" } else { "false" });
        digest_text(
            &mut hasher,
            if self.grammar_enabled {
                "true"
            } else {
                "false"
            },
        );
        digest_optional_text(&mut hasher, self.disabled_reason.as_deref());
        digest_optional_text(&mut hasher, self.approved_by.as_deref());
        digest_optional_u64(&mut hasher, self.approved_at_ms);
        digest_text(&mut hasher, &self.evidence_requirements_revision);
        Ok(hex::encode(hasher.finalize()))
    }
}

// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ApprovalDigestMismatch {
    pub model_id: String,
    pub decode_fingerprint: String,
    pub observed_digest: String,
    pub recomputed_digest: String,
    pub observed_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
pub struct ApprovalMigrationResult {
    pub outcome: String,
    pub seed_revision: String,
    pub rows: usize,
    pub marker: String,
}

impl ApprovalMigrationResult {
    /// Render the stable operator-facing migration result without making the
    /// marker column carry protocol syntax.
    pub fn rendering(&self) -> String {
        match self.outcome.as_str() {
            "applied" => format!(
                "applied: seed_revision={} rows={} marker={}",
                self.seed_revision, self.rows, self.marker
            ),
            "already_applied" => format!(
                "already_applied: seed_revision={} rows={} marker={}",
                self.seed_revision, self.rows, self.marker
            ),
            "invalid_seed" => format!(
                "invalid_seed: reason={} marker={}",
                self.marker, "unchanged"
            ),
            "unmappable_identity" => {
                format!("unmappable_identity: {} marker=unchanged", self.marker)
            }
            "duplicate_identity" => format!("duplicate_identity: {} marker=unchanged", self.marker),
            "transaction_failed" => format!("transaction_failed: {} marker=unchanged", self.marker),
            _ => format!("{}: marker=unchanged", self.outcome),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
pub struct ApprovalMigrationSeedEntry {
    pub source_catalog_entry_id: String,
    pub model_id: String,
    pub decode_fingerprint: String,
    pub enabled: bool,
    pub grammar_enabled: bool,
    pub disabled_reason: Option<String>,
}

// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
pub const APPROVAL_SCHEMA_REVISION: &str = "runtime-bound-records-contracts-v1";
// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
pub const APPROVAL_EVIDENCE_REQUIREMENTS_REVISION: &str = "owned-decode-evidence-requirements-v1";
// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
pub const APPROVAL_DIGEST_DOMAIN: &str = "owned-decode-approval-row-v1\0";
/// Schema of the `evidence` JSON on a `measured_owned_decode` certification
/// row.
///
/// v2 records the one gate the generate probe measures (see
/// [`PROBE_GATE_ID`]) under `evidence.gates`, passed or failed. v1 rows
/// listed G-DEC-01 to G-DEC-12 as passed whenever that single measurement
/// passed, which claimed eleven gates nobody ran. Every certification lookup
/// filters on this revision, so v1 rows no longer certify anything and a
/// lane certified under v1 must be probed again.
// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
pub const CERT_EVIDENCE_SCHEMA_REVISION: &str = "owned-decode-cert-evidence-v2";
/// Revision stamped on a certification row (column `g_dec_manifest_revision`)
/// and on each gate entry in its evidence, naming the gate set the probe
/// records.
///
/// The column and constant names predate v2 evidence and are kept so the
/// stored schema does not change; the value no longer refers to the G-DEC
/// acceptance gates. v1 rows carried `decode-fixture-registry-v1` here.
// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
pub const G_DEC_MANIFEST_REVISION: &str = "owned-decode-probe-gates-v1";
/// Identifier of the one gate the generate probe measures and records.
///
/// The probe passes it when every fixture prompt reproduces its expected
/// tokens exactly or within the fixture's structural band, the configured
/// chain shape reproduces the k=1 output, and the constrained worker path
/// emits the JSON literal `null`. Admission requires exactly this gate,
/// passed; the G-DEC acceptance gates are evaluated by
/// `owned_decode_certification::gates::GateRunner`, which the serving path
/// does not run, so a certification row makes no claim about them.
pub const PROBE_GATE_ID: &str = "structural_band";

/// The only derivation identifier accepted for an ingest-time Q8 repack.
pub const Q8_INGEST_DERIVATION_CONTRACT: &str = "q8-ingest-v1";
/// The schema revision for approval records that control agentic serving.
pub const SERVING_APPROVAL_SCHEMA_REVISION: &str = "serving-approval-record-v1";

/// A source artifact offered to the serving ingest boundary.
///
/// Serving accepts exactly GGUF Q4_K_M source weights. MLX group quantization is
/// named explicitly so it is rejected rather than silently treated as a compatible
/// source format.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactIngestRequest {
    pub model_id: String,
    pub source_format: String,
    pub source_quantization: String,
    pub source_digest: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub q8_derivation: Option<Q8IngestDerivation>,
}

/// The verified output of the only supported deterministic repack operation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Q8IngestDerivation {
    pub derivation_contract: String,
    pub deterministic_inputs_digest: String,
    pub derived_quantization: String,
    pub derived_digest: String,
    pub verified_derived_digest: String,
}

/// Immutable serving identity for a source artifact and its optional ingest-time
/// Q8 derivative. Machine and certification data intentionally do not appear here.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelArtifactId {
    pub artifact_id: String,
    pub model_id: String,
    pub source_format: String,
    pub source_quantization: String,
    pub source_digest: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derived_digest: Option<String>,
}

impl ModelArtifactId {
    fn from_ingest_request(request: &ArtifactIngestRequest) -> Result<Self, SynapseStoreError> {
        validate_artifact_ingest(request)?;
        let derived_digest = request
            .q8_derivation
            .as_ref()
            .map(|derivation| derivation.derived_digest.clone());
        let mut artifact = Self {
            artifact_id: String::new(),
            model_id: request.model_id.clone(),
            source_format: request.source_format.clone(),
            source_quantization: request.source_quantization.clone(),
            source_digest: request.source_digest.clone(),
            derived_digest,
        };
        artifact.artifact_id = artifact.expected_artifact_id();
        Ok(artifact)
    }

    /// Recompute the deterministic identity used as the durable artifact key.
    #[must_use]
    pub fn expected_artifact_id(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"serving-artifact-id-v1\0");
        digest_text(&mut hasher, &self.model_id);
        digest_text(&mut hasher, &self.source_format);
        digest_text(&mut hasher, &self.source_quantization);
        digest_text(&mut hasher, &self.source_digest);
        digest_optional_text(&mut hasher, self.derived_digest.as_deref());
        hex::encode(hasher.finalize())
    }
}

/// An immutable complete certification record associated with its ingested source.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StoredServingCertification {
    pub artifact: ModelArtifactId,
    pub record: CertificationRecord,
    pub recorded_at_ms: u64,
}

/// The approval state is the serving control point for one catalog fingerprint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServingApprovalState {
    Enabled,
    Disabled,
    Revoked,
}

impl ServingApprovalState {
    fn as_str(self) -> &'static str {
        match self {
            Self::Enabled => "enabled",
            Self::Disabled => "disabled",
            Self::Revoked => "revoked",
        }
    }

    fn parse(value: &str) -> Result<Self, SynapseStoreError> {
        match value {
            "enabled" => Ok(Self::Enabled),
            "disabled" => Ok(Self::Disabled),
            "revoked" => Ok(Self::Revoked),
            other => Err(SynapseStoreError::Decode(format!(
                "unknown serving approval state '{other}'"
            ))),
        }
    }
}

/// A durable approval row that binds exactly one catalog fingerprint to one
/// immutable certification record and its source artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServingApprovalRecord {
    pub schema_revision: String,
    pub catalog_fingerprint: String,
    pub certification_record_id: String,
    pub artifact_id: String,
    pub state: ServingApprovalState,
    pub reason: Option<String>,
    pub approved_by: String,
    pub approved_at_ms: u64,
    pub updated_at_ms: u64,
    pub generation: u64,
    pub semantic_digest: String,
}

impl ServingApprovalRecord {
    fn expected_digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"serving-approval-record-v1\0");
        digest_text(&mut hasher, &self.schema_revision);
        digest_text(&mut hasher, &self.catalog_fingerprint);
        digest_text(&mut hasher, &self.certification_record_id);
        digest_text(&mut hasher, &self.artifact_id);
        digest_text(&mut hasher, self.state.as_str());
        digest_optional_text(&mut hasher, self.reason.as_deref());
        digest_text(&mut hasher, &self.approved_by);
        digest_text(&mut hasher, &self.approved_at_ms.to_string());
        hex::encode(hasher.finalize())
    }
}

/// A typed refusal produced by serving admission or retained-state continuation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServingRefusal {
    ArtifactUnapproved,
    ArtifactDisabled,
    ArtifactRevoked,
    CertificationMismatch,
    RetainedStateInvalidated,
}

impl ServingRefusal {
    /// Stable wire vocabulary used by session admission and continuation callers.
    #[must_use]
    pub const fn wire_id(self) -> &'static str {
        match self {
            Self::ArtifactUnapproved => "artifact_unapproved",
            Self::ArtifactDisabled => "artifact_disabled",
            Self::ArtifactRevoked => "artifact_revoked",
            Self::CertificationMismatch => "artifact_not_certified",
            Self::RetainedStateInvalidated => "retained_state_invalidated",
        }
    }
}

/// The result of attempting to create a new active serving session.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "outcome")]
pub enum ServingSessionAdmission {
    Admitted {
        session_id: String,
        catalog_fingerprint: String,
        approval_generation: u64,
    },
    Refused {
        reason: ServingRefusal,
    },
}

/// The result of attempting to continue from a retained state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "outcome")]
pub enum ServingContinuationAdmission {
    Admitted {
        state_id: String,
        catalog_fingerprint: String,
        approval_generation: u64,
    },
    Refused {
        reason: ServingRefusal,
    },
}

/// The state of a durable active-session ledger entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServingSessionState {
    Active,
    TerminationRequested,
    Finished,
    Terminated,
}

impl ServingSessionState {
    fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::TerminationRequested => "termination_requested",
            Self::Finished => "finished",
            Self::Terminated => "terminated",
        }
    }

    fn parse(value: &str) -> Result<Self, SynapseStoreError> {
        match value {
            "active" => Ok(Self::Active),
            "termination_requested" => Ok(Self::TerminationRequested),
            "finished" => Ok(Self::Finished),
            "terminated" => Ok(Self::Terminated),
            other => Err(SynapseStoreError::Decode(format!(
                "unknown serving session state '{other}'"
            ))),
        }
    }

    const fn is_active(self) -> bool {
        matches!(self, Self::Active | Self::TerminationRequested)
    }
}

/// A session row exposed to the engine's boundary and completion paths.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServingSessionRecord {
    pub session_id: String,
    pub catalog_fingerprint: String,
    pub approval_generation: u64,
    pub state: ServingSessionState,
    pub committed_token_count: u64,
    pub terminal_reason: Option<String>,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

/// A retained state is separate from active execution so control operations can
/// invalidate continuations without terminating a normally disabled session.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetainedServingState {
    pub state_id: String,
    pub catalog_fingerprint: String,
    pub valid: bool,
    pub invalidation_reason: Option<String>,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

/// The control transaction's side effects and whether the caller may unload now.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ServingControlOutcome {
    pub approval: ServingApprovalRecord,
    pub invalidated_retained_states: u64,
    pub active_sessions: u64,
    pub termination_requested_sessions: u64,
    pub unload_artifact: bool,
}

/// A commit at a proven token boundary either continues normally or delivers the
/// pending revoke terminal accounting with the committed token count.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum ServingBoundaryOutcome {
    Continue {
        committed_token_count: u64,
    },
    Terminated {
        terminal_reason: String,
        tokens_emitted: u64,
        unload_artifact: bool,
    },
}

/// Completion of a natural active session and its deferred unload decision.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ServingSessionCompletion {
    pub session: ServingSessionRecord,
    pub unload_artifact: bool,
}

/// GC observes artifact pins, while serving control deliberately does not.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactGcDisposition {
    Collectable,
    Pinned,
}

// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OwnedDecodeCertificationRow {
    pub status: CertificationStatus,
    pub revisioned_machine_profile_hash: String,
    pub profile_activation_epoch: u64,
    pub model_id: String,
    pub decode_fingerprint: String,
    pub processing_fingerprint: String,
    pub runtime_config_digest: String,
    pub constraint_runtime_identities: Vec<String>,
    pub worker_path_evidence: Value,
    pub evidence_schema_revision: String,
    pub g_dec_manifest_revision: String,
    pub numeric_profile_id: Option<NumericProfileId>,
    pub fingerprint: Fingerprint,
    pub certified_at_ms: u64,
    pub os_build: String,
    pub module_generation: u64,
    pub evidence: Value,
}

// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
impl OwnedDecodeCertificationRow {
    pub fn identity(&self) -> (&str, u64, &str, &str, &str) {
        (
            &self.revisioned_machine_profile_hash,
            self.profile_activation_epoch,
            &self.model_id,
            &self.decode_fingerprint,
            &self.evidence_schema_revision,
        )
    }
}

// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OwnedDecodeMatchInputs {
    pub revisioned_machine_profile_hash: String,
    pub profile_activation_epoch: u64,
    pub model_id: String,
    pub decode_fingerprint: String,
    pub processing_fingerprint: String,
    pub runtime_config_digest: String,
    pub constraint_runtime_identities: Vec<String>,
    pub worker_path_evidence: Value,
    pub evidence_schema_revision: String,
    pub g_dec_manifest_revision: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
pub struct ClassScopedCertificationRow {
    pub certification_class: CertificationClass,
    pub assurance_class: AssuranceClass,
    pub status: CertificationStatus,
    pub key_hash: String,
    pub machine_profile_hash: Option<String>,
    pub remote_profile_hash: Option<String>,
    pub identity_revision: Option<String>,
    pub numeric_profile_id: Option<NumericProfileId>,
    pub fingerprint: Fingerprint,
    pub certified_at_ms: u64,
    pub os_build: String,
    pub module_generation: u64,
    pub evidence: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
pub struct ProfileState {
    pub snapshot: Option<MachineProfile>,
    pub revisioned_machine_profile_hash: Option<String>,
    pub profile_activation_epoch: Option<u64>,
    pub previous_revisioned_machine_profile_hash: Option<String>,
    pub last_rotation_reason: Option<String>,
    pub last_rotation_at_ms: Option<u64>,
    pub certification_stale_since_ms: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
pub struct RotationCertificationOutcome {
    pub event_id: String,
    pub model_id: String,
    pub decode_fingerprint: String,
    pub outcome_state: String,
    pub certified_at_ms: Option<u64>,
    pub failure_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
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
    pub outcomes: Vec<RotationCertificationOutcome>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
pub struct ProfileActivation {
    pub state: ProfileState,
    pub rotated: bool,
    pub event: Option<RotationLedgerEvent>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
pub struct EvidenceRequirementsDivergence {
    pub model_id: String,
    pub decode_fingerprint: String,
    pub result: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
pub struct ApprovalCertificationHealth {
    pub model_id: String,
    pub decode_fingerprint: String,
    pub enabled: bool,
    pub admission: String,
    pub state: String,
    pub certified_at_ms: Option<u64>,
    pub failure_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
pub struct StorageHealthInputs {
    pub previous_revisioned_machine_profile_hash: Option<String>,
    pub current_revisioned_machine_profile_hash: Option<String>,
    pub profile_activation_epoch: Option<u64>,
    pub certification_stale_since_ms: Option<u64>,
    pub last_rotation_at_ms: Option<u64>,
    pub last_rotation_reason: Option<String>,
    pub rotation_event_count: u64,
    pub re_certification_state: String,
    pub evidence_requirements_divergence: Vec<EvidenceRequirementsDivergence>,
    pub approval_certification_outcomes: Vec<ApprovalCertificationHealth>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
pub struct OwnedDecodeAdmission {
    pub approval: ApprovalRow,
    pub certification: OwnedDecodeCertificationRow,
    pub revisioned_machine_profile_hash: String,
    pub profile_activation_epoch: u64,
}

/// The failed arm from a fenced owned-decode admission read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OwnedDecodeAdmissionRefusal {
    /// No operator approval exists for the exact model/fingerprint identity.
    ApprovalAbsent,
    /// The operator explicitly disabled approval and supplied this reason.
    ApprovalDisabled { disabled_reason: String },
    /// Profile, identity, or certification evidence is missing or stale.
    NotCertified,
}

/// The approval row and complete certification result from one fenced read.
/// Keeping a refused approval row lets routing report an explicit disable without
/// re-reading mutable state outside the admission fence.
#[derive(Clone, Debug, PartialEq)]
pub enum OwnedDecodeAdmissionEvaluation {
    Admitted(Box<OwnedDecodeAdmission>),
    Refused {
        approval: Box<Option<ApprovalRow>>,
        refusal: OwnedDecodeAdmissionRefusal,
    },
}

impl OwnedDecodeAdmissionEvaluation {
    pub fn admission(&self) -> Option<&OwnedDecodeAdmission> {
        match self {
            Self::Admitted(admission) => Some(admission.as_ref()),
            Self::Refused { .. } => None,
        }
    }

    pub fn approval(&self) -> Option<&ApprovalRow> {
        match self {
            Self::Admitted(admission) => Some(&admission.approval),
            Self::Refused { approval, .. } => approval.as_ref().as_ref(),
        }
    }

    pub fn refusal(&self) -> Option<&OwnedDecodeAdmissionRefusal> {
        match self {
            Self::Admitted(_) => None,
            Self::Refused { refusal, .. } => Some(refusal),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbeWriteOutcome {
    Certified,
    ProbeStale,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaleProbeEvent {
    pub outcome: String,
    pub model_id: String,
    pub decode_fingerprint: String,
    pub observed_profile_activation_epoch: u64,
    pub current_profile_activation_epoch: Option<u64>,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
pub enum RotationOutcomeState {
    NotRequired,
    Required,
    InProgress,
    Passed,
    Failed,
}

// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
impl RotationOutcomeState {
    fn encode(self) -> &'static str {
        match self {
            Self::NotRequired => "not_required",
            Self::Required => "required",
            Self::InProgress => "in_progress",
            Self::Passed => "passed",
            Self::Failed => "failed",
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
    pub(crate) knob: PerfKnob,
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
    /// Import only durable operator intent from an Engram scratch database.
    ///
    /// Opening the capture first applies this binary's normal forward migrations
    /// and preserves the typed refusal for captures written by a newer schema.
    /// The target must be unserved; opening its store takes the same lease used
    /// by the module before any live row is changed. Restored approvals are
    /// imported disabled and require operator re-enable before serving resumes.
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
        live.import_migrated_capture(capture_path, None)
    }

    fn import_migrated_capture(
        &self,
        capture_path: &Path,
        fail_after_table: Option<&str>,
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
                    if fail_after_table == Some(*table) {
                        return Err(rusqlite::Error::InvalidParameterName(format!(
                            "simulated restore interruption after {table}"
                        )));
                    }
                }

                // Restored approvals must come back disabled. An emergency rollback
                // or revocation flips approval states in place, so a snapshot taken
                // prior to that rollback carries approvals as enabled. Importing them
                // as enabled would resurrect revoked approvals as soon as a probe
                // re-derives certification. To remain fail-closed without an external
                // journal, we force every imported approval to disabled inside this
                // same transaction, requiring an operator to review and re-enable.
                let now = unix_now_ms();
                let mut disabled_on_import = BTreeMap::new();

                let mut approval_stmt = transaction.prepare(
                    "SELECT row_id, schema_revision, model_id, decode_fingerprint, enabled,
                            grammar_enabled, disabled_reason, approved_by, approved_at_ms,
                            updated_at_ms, evidence_requirements_revision, semantic_digest,
                            generation, fencing_metadata
                     FROM main.approvals ORDER BY row_id",
                )?;
                let approval_rows = approval_stmt
                    .query_map([], approval_from_row)?
                    .collect::<Result<Vec<_>, _>>()?;
                drop(approval_stmt);

                let mut disabled_approvals = 0u64;
                for mut row in approval_rows {
                    row.enabled = false;
                    row.grammar_enabled = false;
                    row.disabled_reason = Some(RESTORE_DISABLED_REASON.to_string());
                    row.updated_at_ms = now;
                    row.generation = row.generation.saturating_add(1);
                    row.semantic_digest = row.expected_digest().map_err(to_sql_error)?;
                    transaction.execute(
                        "UPDATE main.approvals
                         SET enabled = 0, grammar_enabled = 0, disabled_reason = ?1,
                             updated_at_ms = ?2, generation = ?3, semantic_digest = ?4
                         WHERE row_id = ?5",
                        params![
                            row.disabled_reason.as_deref(),
                            row.updated_at_ms as i64,
                            row.generation as i64,
                            &row.semantic_digest,
                            row.row_id as i64,
                        ],
                    )?;
                    disabled_approvals += 1;
                }
                disabled_on_import.insert("approvals".to_string(), disabled_approvals);

                let mut serving_stmt = transaction.prepare(
                    "SELECT catalog_fingerprint, certification_record_id, artifact_id, state,
                            reason, approved_by, approved_at_ms, updated_at_ms, generation,
                            semantic_digest
                     FROM main.serving_approvals ORDER BY rowid",
                )?;
                let serving_rows = serving_stmt
                    .query_map([], |row| {
                        Ok(ServingApprovalRecord {
                            schema_revision: SERVING_APPROVAL_SCHEMA_REVISION.to_string(),
                            catalog_fingerprint: row.get(0)?,
                            certification_record_id: row.get(1)?,
                            artifact_id: row.get(2)?,
                            state: ServingApprovalState::parse(&row.get::<_, String>(3)?)
                                .map_err(to_sql_error)?,
                            reason: row.get(4)?,
                            approved_by: row.get(5)?,
                            approved_at_ms: row.get::<_, i64>(6)? as u64,
                            updated_at_ms: row.get::<_, i64>(7)? as u64,
                            generation: row.get::<_, i64>(8)? as u64,
                            semantic_digest: row.get(9)?,
                        })
                    })?
                    .collect::<Result<Vec<_>, _>>()?;
                drop(serving_stmt);

                let mut disabled_serving = 0u64;
                for mut row in serving_rows {
                    row.state = ServingApprovalState::Disabled;
                    row.reason = Some(RESTORE_DISABLED_REASON.to_string());
                    row.updated_at_ms = now;
                    row.generation = row.generation.saturating_add(1);
                    row.semantic_digest = row.expected_digest();
                    transaction.execute(
                        "UPDATE main.serving_approvals
                         SET state = ?1, reason = ?2, updated_at_ms = ?3, generation = ?4,
                             semantic_digest = ?5
                         WHERE catalog_fingerprint = ?6",
                        params![
                            row.state.as_str(),
                            row.reason.as_deref(),
                            row.updated_at_ms as i64,
                            row.generation as i64,
                            &row.semantic_digest,
                            &row.catalog_fingerprint,
                        ],
                    )?;
                    disabled_serving += 1;
                }
                disabled_on_import.insert("serving_approvals".to_string(), disabled_serving);

                transaction.commit()?;
                Ok(RestoreImportReport {
                    imported_rows,
                    disabled_on_import,
                })
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
            MAINTENANCE_VACUUM_COUNT.fetch_add(1, Ordering::Relaxed);
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
        let alias_table = self.alias_table()?;
        let models = self
            .catalog_models()?
            .into_iter()
            .map(|model| {
                let dtype = model
                    .owned_dtype
                    .clone()
                    .or_else(|| match model.engine.as_str() {
                        "ane" => Some("f16".to_string()),
                        "owned-cuda" => Some("f16".to_string()),
                        "ort" => Some("f32".to_string()),
                        _ => None,
                    });
                let device_class = match model.engine.as_str() {
                    "owned-metal" | "owned-metal-decode" => Some("metal".to_string()),
                    "ane" => Some("ane".to_string()),
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
                    // This store-side listing has no approval reader; the
                    // module's catalog path supplies serving_admission where it
                    // is meaningful. Omitted rather than defaulted, so a
                    // consumer cannot read absence as "enabled".
                    serving_admission: None,
                    serving_admission_reason: None,
                    warm_load_cost_hint_ms: None,
                }
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

    /// Insert an approval row only through this explicit administrative path,
    /// after validation; serving and certification code never call it.
    // Staged storage API consumed by the epic's runtime slice; remove this allow there.
    #[allow(dead_code)]
    pub fn insert_approval(&self, row: &ApprovalRow) -> Result<ApprovalRow, SynapseStoreError> {
        validate_approval(row, false)?;
        let catalog_matches: i64 = self.store.with_conn(|conn| {
            conn.query_row(
                "SELECT COUNT(*) FROM models WHERE model_id = ?1",
                params![&row.model_id],
                |query_row| query_row.get(0),
            )
        })?;
        if catalog_matches != 1 {
            return Err(SynapseStoreError::Decode(format!(
                "unmappable_identity: model_id={} matches={catalog_matches}",
                row.model_id
            )));
        }
        let inserted = self.store.with_conn_fenced(|tx| {
            let mut row = row.clone();
            row.semantic_digest = row.expected_digest().map_err(to_sql_error)?;
            validate_approval(&row, true).map_err(to_sql_error)?;
            let fencing_metadata = serde_json::to_string(&row.fencing_metadata)
                .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;
            tx.execute(
                "INSERT INTO approvals (
                     schema_revision, model_id, decode_fingerprint, enabled, grammar_enabled,
                     disabled_reason, approved_by, approved_at_ms, updated_at_ms,
                     evidence_requirements_revision, semantic_digest, generation, fencing_metadata
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
                params![
                    &row.schema_revision,
                    &row.model_id,
                    &row.decode_fingerprint,
                    row.enabled as i64,
                    row.grammar_enabled as i64,
                    row.disabled_reason.as_deref(),
                    row.approved_by.as_deref(),
                    row.approved_at_ms.map(|value| value as i64),
                    row.updated_at_ms as i64,
                    &row.evidence_requirements_revision,
                    &row.semantic_digest,
                    row.generation as i64,
                    fencing_metadata,
                ],
            )?;
            let row_id = tx.last_insert_rowid();
            load_approval_tx(tx, row_id)?.ok_or_else(|| rusqlite::Error::QueryReturnedNoRows)
        })?;
        Ok(inserted)
    }

    // Staged storage API consumed by the epic's runtime slice; remove this allow there.
    #[allow(dead_code)]
    pub fn create_approval(
        &self,
        model_id: &str,
        decode_fingerprint: &str,
        grammar_enabled: bool,
        approved_by: &str,
        approved_at_ms: u64,
        updated_at_ms: u64,
    ) -> Result<ApprovalRow, SynapseStoreError> {
        let mut row = ApprovalRow {
            schema_revision: APPROVAL_SCHEMA_REVISION.to_string(),
            model_id: model_id.to_string(),
            decode_fingerprint: decode_fingerprint.to_string(),
            enabled: true,
            grammar_enabled,
            disabled_reason: None,
            approved_by: Some(approved_by.to_string()),
            approved_at_ms: Some(approved_at_ms),
            updated_at_ms,
            evidence_requirements_revision: APPROVAL_EVIDENCE_REQUIREMENTS_REVISION.to_string(),
            semantic_digest: String::new(),
            row_id: 0,
            generation: 0,
            fencing_metadata: Value::Object(Default::default()),
        };
        row.semantic_digest = row.expected_digest()?;
        self.insert_approval(&row)
    }

    /// Create or explicitly re-enable one approval identity. The approval stamp,
    /// grammar switch, semantic digest, and fencing generation commit together.
    pub fn enable_or_create_approval(
        &self,
        model_id: &str,
        decode_fingerprint: &str,
        grammar_enabled: bool,
        approved_by: &str,
        approved_at_ms: u64,
    ) -> Result<ApprovalRow, SynapseStoreError> {
        let mut candidate = ApprovalRow {
            schema_revision: APPROVAL_SCHEMA_REVISION.to_string(),
            model_id: model_id.to_string(),
            decode_fingerprint: decode_fingerprint.to_string(),
            enabled: true,
            grammar_enabled,
            disabled_reason: None,
            approved_by: Some(approved_by.to_string()),
            approved_at_ms: Some(approved_at_ms),
            updated_at_ms: approved_at_ms,
            evidence_requirements_revision: APPROVAL_EVIDENCE_REQUIREMENTS_REVISION.to_string(),
            semantic_digest: String::new(),
            row_id: 0,
            generation: 0,
            fencing_metadata: serde_json::json!({
                "operation": "approval_enable",
            }),
        };
        candidate.semantic_digest = candidate.expected_digest()?;
        validate_approval(&candidate, true)?;

        let row = self.store.with_conn_fenced(|tx| {
            let catalog_matches: i64 = tx.query_row(
                "SELECT COUNT(*) FROM models
                 WHERE model_id = ?1 AND engine = 'owned-metal-decode' AND task = 'generate'",
                params![model_id],
                |row| row.get(0),
            )?;
            if catalog_matches != 1 {
                return Err(to_sql_error(SynapseStoreError::Decode(format!(
                    "unmappable_identity: model_id={model_id} matches={catalog_matches}"
                ))));
            }

            if let Some(mut row) = load_approval_tx_by_identity(tx, model_id, decode_fingerprint)? {
                validate_approval(&row, true).map_err(to_sql_error)?;
                row.enabled = true;
                row.grammar_enabled = grammar_enabled;
                row.disabled_reason = None;
                row.approved_by = Some(approved_by.to_string());
                row.approved_at_ms = Some(approved_at_ms);
                row.updated_at_ms = approved_at_ms;
                row.generation = row.generation.saturating_add(1);
                row.fencing_metadata = serde_json::json!({
                    "operation": "approval_enable",
                });
                row.semantic_digest = row.expected_digest().map_err(to_sql_error)?;
                validate_approval(&row, true).map_err(to_sql_error)?;
                let fencing_metadata = serde_json::to_string(&row.fencing_metadata)
                    .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;
                tx.execute(
                    "UPDATE approvals SET enabled = 1, grammar_enabled = ?1,
                         disabled_reason = NULL, approved_by = ?2, approved_at_ms = ?3,
                         updated_at_ms = ?4, semantic_digest = ?5, generation = ?6,
                         fencing_metadata = ?7 WHERE row_id = ?8",
                    params![
                        grammar_enabled as i64,
                        approved_by,
                        approved_at_ms as i64,
                        approved_at_ms as i64,
                        &row.semantic_digest,
                        row.generation as i64,
                        fencing_metadata,
                        row.row_id as i64,
                    ],
                )?;
                return Ok(row);
            }

            let fencing_metadata = serde_json::to_string(&candidate.fencing_metadata)
                .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;
            tx.execute(
                "INSERT INTO approvals (
                     schema_revision, model_id, decode_fingerprint, enabled, grammar_enabled,
                     disabled_reason, approved_by, approved_at_ms, updated_at_ms,
                     evidence_requirements_revision, semantic_digest, generation, fencing_metadata
                 ) VALUES (?1, ?2, ?3, 1, ?4, NULL, ?5, ?6, ?7, ?8, ?9, 0, ?10)",
                params![
                    &candidate.schema_revision,
                    model_id,
                    decode_fingerprint,
                    grammar_enabled as i64,
                    approved_by,
                    approved_at_ms as i64,
                    approved_at_ms as i64,
                    &candidate.evidence_requirements_revision,
                    &candidate.semantic_digest,
                    fencing_metadata,
                ],
            )?;
            let row_id = tx.last_insert_rowid();
            load_approval_tx(tx, row_id)?.ok_or(rusqlite::Error::QueryReturnedNoRows)
        })?;
        Ok(row)
    }

    // Staged storage API consumed by the epic's runtime slice; remove this allow there.
    #[allow(dead_code)]
    pub fn migrate_owned_decode_approvals(
        &self,
        seed_revision: &str,
        schema_revision: &str,
    ) -> Result<ApprovalMigrationResult, SynapseStoreError> {
        const SEED_SOURCE: &str =
            include_str!("../owned-decode-manifests/migration-seed-manifest-v1.json");
        self.migrate_owned_decode_approvals_from_seed(SEED_SOURCE, seed_revision, schema_revision)
    }

    fn migrate_owned_decode_approvals_from_seed(
        &self,
        seed_source: &str,
        seed_revision: &str,
        schema_revision: &str,
    ) -> Result<ApprovalMigrationResult, SynapseStoreError> {
        let seed_digest = hex::encode(Sha256::digest(seed_source.as_bytes()));
        if seed_digest != APPROVAL_MIGRATION_SEED_DIGEST {
            return Ok(ApprovalMigrationResult {
                outcome: "invalid_seed".to_string(),
                seed_revision: seed_revision.to_string(),
                rows: 0,
                marker: "seed_digest_mismatch".to_string(),
            });
        }
        if seed_revision != "owned-decode-approval-migration-v1"
            || schema_revision != APPROVAL_SCHEMA_REVISION
        {
            return Ok(ApprovalMigrationResult {
                outcome: "invalid_seed".to_string(),
                seed_revision: seed_revision.to_string(),
                rows: 0,
                marker: "unchanged".to_string(),
            });
        }
        let seed: Value = match serde_json::from_str(seed_source) {
            Ok(seed) => seed,
            Err(error) => {
                return Ok(ApprovalMigrationResult {
                    outcome: "invalid_seed".to_string(),
                    seed_revision: seed_revision.to_string(),
                    rows: 0,
                    marker: format!("parse_error:{error}"),
                })
            }
        };
        let Some(seed_object) = seed.as_object() else {
            return Ok(ApprovalMigrationResult {
                outcome: "invalid_seed".to_string(),
                seed_revision: seed_revision.to_string(),
                rows: 0,
                marker: "unchanged".to_string(),
            });
        };
        if !has_only_keys(
            seed_object,
            &[
                "manifest_revision",
                "schema_revision",
                "artifact_kind",
                "source",
                "expansion",
                "seed_revision",
                "initial_evidence_requirements_revision",
                "entries",
                "mechanical_proof",
                "reachability",
            ],
        ) || seed_object.get("seed_revision").and_then(Value::as_str) != Some(seed_revision)
            || seed_object.get("schema_revision").and_then(Value::as_str)
                != Some("runtime-bound-records-contracts-v1")
            || seed_object
                .get("initial_evidence_requirements_revision")
                .and_then(Value::as_str)
                != Some(APPROVAL_EVIDENCE_REQUIREMENTS_REVISION)
        {
            return Ok(ApprovalMigrationResult {
                outcome: "invalid_seed".to_string(),
                seed_revision: seed_revision.to_string(),
                rows: 0,
                marker: "unchanged".to_string(),
            });
        }
        let Some(seed_entries) = seed_object.get("entries").and_then(Value::as_array) else {
            return Ok(ApprovalMigrationResult {
                outcome: "invalid_seed".to_string(),
                seed_revision: seed_revision.to_string(),
                rows: 0,
                marker: "unchanged".to_string(),
            });
        };
        let mut entries = Vec::new();
        for value in seed_entries {
            let Some(entry) = value.as_object() else {
                return Ok(ApprovalMigrationResult {
                    outcome: "invalid_seed".to_string(),
                    seed_revision: seed_revision.to_string(),
                    rows: 0,
                    marker: "unchanged".to_string(),
                });
            };
            if !has_only_keys(
                entry,
                &[
                    "source_catalog_entry_id",
                    "model_id",
                    "decode_fingerprint",
                    "enabled",
                    "grammar_enabled",
                    "disabled_reason",
                    "d009_provenance",
                ],
            ) {
                return Ok(ApprovalMigrationResult {
                    outcome: "invalid_seed".to_string(),
                    seed_revision: seed_revision.to_string(),
                    rows: 0,
                    marker: "unknown_fields".to_string(),
                });
            }
            let source_catalog_entry_id = entry
                .get("source_catalog_entry_id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let model_id = entry
                .get("model_id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let decode_fingerprint = entry
                .get("decode_fingerprint")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let enabled = entry
                .get("enabled")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let grammar_enabled = entry
                .get("grammar_enabled")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let disabled_reason = entry
                .get("disabled_reason")
                .and_then(|value| {
                    if value.is_null() {
                        None
                    } else {
                        value.as_str()
                    }
                })
                .map(str::to_string);
            let provenance_valid = entry
                .get("d009_provenance")
                .and_then(Value::as_object)
                .map(|provenance| {
                    has_only_keys(
                        provenance,
                        &["source_manifest_revision", "source_record_index"],
                    ) && provenance
                        .get("source_manifest_revision")
                        .and_then(Value::as_str)
                        == Some("d009-cutover-records-v1")
                        && provenance
                            .get("source_record_index")
                            .and_then(Value::as_u64)
                            .is_some()
                })
                .unwrap_or(false);
            if source_catalog_entry_id != model_id
                || !is_catalog_model_id(model_id)
                || !is_digest(decode_fingerprint)
                || !provenance_valid
                || (enabled && disabled_reason.is_some())
                || (!enabled
                    && disabled_reason
                        .as_deref()
                        .map(str::is_empty)
                        .unwrap_or(true))
            {
                return Ok(ApprovalMigrationResult {
                    outcome: "invalid_seed".to_string(),
                    seed_revision: seed_revision.to_string(),
                    rows: 0,
                    marker: "unchanged".to_string(),
                });
            }
            entries.push(ApprovalMigrationSeedEntry {
                source_catalog_entry_id: source_catalog_entry_id.to_string(),
                model_id: model_id.to_string(),
                decode_fingerprint: decode_fingerprint.to_string(),
                enabled,
                grammar_enabled,
                disabled_reason,
            });
        }
        if entries.len() != 4 {
            return Ok(ApprovalMigrationResult {
                outcome: "invalid_seed".to_string(),
                seed_revision: seed_revision.to_string(),
                rows: entries.len(),
                marker: "entry_count_must_equal_four".to_string(),
            });
        }
        let mut identities = BTreeSet::new();
        for entry in &entries {
            if !identities.insert((entry.model_id.clone(), entry.decode_fingerprint.clone())) {
                return Ok(ApprovalMigrationResult {
                    outcome: "duplicate_identity".to_string(),
                    seed_revision: seed_revision.to_string(),
                    rows: 0,
                    marker: format!(
                        "model_id={} decode_fingerprint={}",
                        entry.model_id, entry.decode_fingerprint
                    ),
                });
            }
        }
        let context_buckets: crate::owned_decode_contracts::ContextBucketsManifest =
            serde_json::from_str(include_str!(
                "../owned-decode-manifests/decode-context-buckets-v1.json"
            ))
            .expect("checked-in decode context buckets parse");
        let unmappable = self.store.with_conn(|conn| {
            for entry in &entries {
                let stored = conn
                    .query_row(
                        "SELECT engine, task, config_json FROM models WHERE model_id = ?1",
                        params![&entry.source_catalog_entry_id],
                        |row| {
                            Ok((
                                row.get::<_, String>(0)?,
                                row.get::<_, String>(1)?,
                                row.get::<_, rusqlite::types::Value>(2)?,
                            ))
                        },
                    )
                    .optional()?;
                let Some((engine, task, config_value)) = stored else {
                    return Ok(Some((entry.source_catalog_entry_id.clone(), 0)));
                };
                let config_json = match config_value {
                    rusqlite::types::Value::Blob(bytes) => bytes,
                    rusqlite::types::Value::Text(text) => text.into_bytes(),
                    _ => return Ok(Some((entry.source_catalog_entry_id.clone(), 0))),
                };
                if engine != "owned-metal-decode" || task != "generate" {
                    return Ok(Some((entry.source_catalog_entry_id.clone(), 0)));
                }
                let Ok(config) = serde_json::from_slice::<StoredModelConfig>(&config_json) else {
                    return Ok(Some((entry.source_catalog_entry_id.clone(), 0)));
                };
                if config.engine != "owned-metal-decode" || config.task != "generate" {
                    return Ok(Some((entry.source_catalog_entry_id.clone(), 0)));
                }
                let Ok(catalog_entry) = crate::owned_decode_catalog_entry(&config) else {
                    return Ok(Some((entry.source_catalog_entry_id.clone(), 0)));
                };
                let decode_fingerprint = catalog_entry
                    .decode_identity_inputs()
                    .decode_fingerprint()
                    .ok();
                if catalog_entry.entry_id != entry.source_catalog_entry_id
                    || catalog_entry.validate(&context_buckets).is_err()
                    || decode_fingerprint.as_ref().map(|value| value.0.as_str())
                        != Some(entry.decode_fingerprint.as_str())
                {
                    return Ok(Some((entry.source_catalog_entry_id.clone(), 0)));
                }
            }
            Ok(None)
        })?;
        if let Some((source_catalog_entry_id, matches)) = unmappable {
            return Ok(ApprovalMigrationResult {
                outcome: "unmappable_identity".to_string(),
                seed_revision: seed_revision.to_string(),
                rows: 0,
                marker: format!(
                    "source_catalog_entry_id={source_catalog_entry_id} matches={matches}"
                ),
            });
        }
        self.validate_migration_state(seed_revision, schema_revision, &seed_digest, &entries)?;
        let now_ms = unix_now_ms();
        let prepared_rows = match entries
            .iter()
            .map(|entry| {
                let mut row = ApprovalRow {
                    schema_revision: schema_revision.to_string(),
                    model_id: entry.model_id.clone(),
                    decode_fingerprint: entry.decode_fingerprint.clone(),
                    enabled: entry.enabled,
                    grammar_enabled: entry.grammar_enabled,
                    disabled_reason: entry.disabled_reason.clone(),
                    approved_by: entry.enabled.then(|| format!("migration:{seed_revision}")),
                    approved_at_ms: entry.enabled.then_some(now_ms),
                    updated_at_ms: now_ms,
                    evidence_requirements_revision: APPROVAL_EVIDENCE_REQUIREMENTS_REVISION
                        .to_string(),
                    semantic_digest: String::new(),
                    row_id: 0,
                    generation: 0,
                    fencing_metadata: serde_json::json!({
                        "operation": "approval_migration",
                        "seed_revision": seed_revision
                    }),
                };
                row.semantic_digest = row.expected_digest()?;
                validate_approval(&row, true)?;
                Ok(row)
            })
            .collect::<Result<Vec<_>, SynapseStoreError>>()
        {
            Ok(rows) => rows,
            Err(error) => {
                return Ok(ApprovalMigrationResult {
                    outcome: "invalid_seed".to_string(),
                    seed_revision: seed_revision.to_string(),
                    rows: 0,
                    marker: format!("digest_verification:{error}"),
                })
            }
        };
        let result = self.store.with_conn_fenced(|tx| {
            let marker = tx
                .query_row(
                    "SELECT 1 FROM approval_migration_markers WHERE seed_revision = ?1",
                    params![seed_revision],
                    |row| row.get::<_, i64>(0),
                )
                .optional()?;
            if marker.is_some() {
                return Ok(ApprovalMigrationResult {
                    outcome: "already_applied".to_string(),
                    seed_revision: seed_revision.to_string(),
                    rows: entries.len(),
                    marker: "unchanged".to_string(),
                });
            }
            let existing = tx
                .prepare(
                    "SELECT model_id, decode_fingerprint FROM approvals
                     ORDER BY model_id ASC, decode_fingerprint ASC LIMIT 1",
                )?
                .query_row([], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })
                .optional()?;
            if let Some((model_id, decode_fingerprint)) = existing {
                return Ok(ApprovalMigrationResult {
                    outcome: "duplicate_identity".to_string(),
                    seed_revision: seed_revision.to_string(),
                    rows: 0,
                    marker: format!("model_id={model_id} decode_fingerprint={decode_fingerprint}"),
                });
            }
            for row in &prepared_rows {
                let fencing_metadata = serde_json::to_string(&row.fencing_metadata)
                    .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;
                tx.execute(
                    "INSERT INTO approvals (
                         schema_revision, model_id, decode_fingerprint, enabled,
                         grammar_enabled, disabled_reason, approved_by, approved_at_ms,
                         updated_at_ms, evidence_requirements_revision, semantic_digest,
                         generation, fencing_metadata
                     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
                    params![
                        &row.schema_revision,
                        &row.model_id,
                        &row.decode_fingerprint,
                        row.enabled as i64,
                        row.grammar_enabled as i64,
                        row.disabled_reason.as_deref(),
                        row.approved_by.as_deref(),
                        row.approved_at_ms.map(|value| value as i64),
                        row.updated_at_ms as i64,
                        &row.evidence_requirements_revision,
                        &row.semantic_digest,
                        row.generation as i64,
                        fencing_metadata,
                    ],
                )?;
            }
            tx.execute(
                "INSERT INTO approval_migration_markers (
                     seed_revision, schema_revision, seed_digest, applied_at_ms, row_count
                 ) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    seed_revision,
                    schema_revision,
                    seed_digest,
                    now_ms as i64,
                    entries.len() as i64,
                ],
            )?;
            Ok(ApprovalMigrationResult {
                outcome: "applied".to_string(),
                seed_revision: seed_revision.to_string(),
                rows: entries.len(),
                marker: "committed".to_string(),
            })
        });
        match result {
            Ok(result) => Ok(result),
            Err(_error) => Ok(ApprovalMigrationResult {
                outcome: "transaction_failed".to_string(),
                seed_revision: seed_revision.to_string(),
                rows: 0,
                marker: "phase=commit".to_string(),
            }),
        }
    }

    fn validate_migration_state(
        &self,
        seed_revision: &str,
        schema_revision: &str,
        seed_digest: &str,
        entries: &[ApprovalMigrationSeedEntry],
    ) -> Result<(), SynapseStoreError> {
        let marker = self.store.with_conn(|conn| {
            conn.query_row(
                "SELECT schema_revision, seed_digest, row_count
                 FROM approval_migration_markers WHERE seed_revision = ?1",
                params![seed_revision],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, i64>(2)?,
                    ))
                },
            )
            .optional()
        })?;
        let Some((recorded_schema, recorded_digest, row_count)) = marker else {
            return Ok(());
        };
        if recorded_schema != schema_revision
            || recorded_digest != seed_digest
            || row_count != entries.len() as i64
            || row_count <= 0
        {
            return Err(SynapseStoreError::ApprovalMigrationStateCorrupt(
                format!(
                    "seed_revision={seed_revision} schema_revision={recorded_schema} row_count={row_count}"
                ),
            ));
        }
        let rows = self.store.with_conn(|conn| {
            let mut statement = conn.prepare(
                "SELECT row_id, schema_revision, model_id, decode_fingerprint, enabled,
                        grammar_enabled, disabled_reason, approved_by, approved_at_ms,
                        updated_at_ms, evidence_requirements_revision, semantic_digest,
                        generation, fencing_metadata
                 FROM approvals ORDER BY model_id ASC, decode_fingerprint ASC",
            )?;
            let rows = statement
                .query_map([], approval_from_row)?
                .collect::<Result<Vec<_>, _>>();
            rows
        })?;
        if rows.len() < entries.len()
            || entries.iter().any(|entry| {
                !rows.iter().any(|row| {
                    row.model_id == entry.model_id
                        && row.decode_fingerprint == entry.decode_fingerprint
                })
            })
        {
            return Err(SynapseStoreError::ApprovalMigrationStateCorrupt(
                "migration marker acknowledges rows that are not present".to_string(),
            ));
        }
        for row in rows {
            validate_approval(&row, true).map_err(|error| {
                SynapseStoreError::ApprovalMigrationStateCorrupt(error.to_string())
            })?;
        }
        Ok(())
    }

    /// Validate the durable acknowledgement before any serving lookup. A
    /// marker is optional for stores that have not run migration; once present,
    /// malformed marker metadata or missing acknowledged rows fail closed.
    fn validate_persisted_migration_state(&self) -> Result<(), SynapseStoreError> {
        let marker = self.store.with_conn(|conn| {
            conn.query_row(
                "SELECT schema_revision, seed_digest, row_count
                 FROM approval_migration_markers
                 WHERE seed_revision = 'owned-decode-approval-migration-v1'",
                [],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, i64>(2)?,
                    ))
                },
            )
            .optional()
        })?;
        let Some((schema_revision, seed_digest, row_count)) = marker else {
            return Ok(());
        };
        let expected_digest = APPROVAL_MIGRATION_SEED_DIGEST;
        if schema_revision != APPROVAL_SCHEMA_REVISION
            || seed_digest != expected_digest
            || row_count != 4
        {
            return Err(SynapseStoreError::ApprovalMigrationStateCorrupt(
                "migration acknowledgement metadata is invalid".to_string(),
            ));
        }
        let rows = self.approvals_without_migration_check()?;
        if rows.len() < row_count as usize {
            return Err(SynapseStoreError::ApprovalMigrationStateCorrupt(
                "migration acknowledgement rows are missing".to_string(),
            ));
        }
        Ok(())
    }

    fn approvals_without_migration_check(&self) -> Result<Vec<ApprovalRow>, SynapseStoreError> {
        Ok(self.store.with_conn(|conn| {
            let mut statement = conn.prepare(
                "SELECT row_id, schema_revision, model_id, decode_fingerprint, enabled,
                        grammar_enabled, disabled_reason, approved_by, approved_at_ms,
                        updated_at_ms, evidence_requirements_revision, semantic_digest,
                        generation, fencing_metadata
                 FROM approvals ORDER BY model_id ASC, decode_fingerprint ASC",
            )?;
            let rows = statement
                .query_map([], approval_from_row)?
                .collect::<Result<Vec<_>, _>>();
            rows
        })?)
    }

    // Staged storage API consumed by the epic's runtime slice; remove this allow there.
    #[allow(dead_code)]
    pub fn get_approval(
        &self,
        model_id: &str,
        decode_fingerprint: &str,
    ) -> Result<Option<ApprovalRow>, SynapseStoreError> {
        self.validate_persisted_migration_state()?;
        let row = self.store.with_conn(|conn| {
            conn.query_row(
                "SELECT row_id, schema_revision, model_id, decode_fingerprint, enabled,
                        grammar_enabled, disabled_reason, approved_by, approved_at_ms,
                        updated_at_ms, evidence_requirements_revision, semantic_digest,
                        generation, fencing_metadata
                 FROM approvals WHERE model_id = ?1 AND decode_fingerprint = ?2",
                params![model_id, decode_fingerprint],
                approval_from_row,
            )
            .optional()
        })?;
        let Some(row) = row else { return Ok(None) };
        match validate_approval(&row, true) {
            Ok(()) => Ok(Some(row)),
            Err(SynapseStoreError::ApprovalDigestMismatch {
                model_id,
                decode_fingerprint,
                observed,
                recomputed,
            }) => {
                let observed_at_ms = unix_now_ms();
                self.store.with_conn_fenced(|tx| {
                    tx.execute(
                        "INSERT INTO approval_digest_corruption_events (
                             model_id, decode_fingerprint, observed_digest,
                             recomputed_digest, observed_at_ms
                         ) VALUES (?1, ?2, ?3, ?4, ?5)",
                        params![
                            &model_id,
                            &decode_fingerprint,
                            &observed,
                            &recomputed,
                            observed_at_ms as i64,
                        ],
                    )?;
                    Ok(())
                })?;
                tracing::warn!(
                    target: "cert",
                    model_id,
                    decode_fingerprint,
                    "approval_digest_mismatch"
                );
                Err(SynapseStoreError::ApprovalDigestMismatch {
                    model_id,
                    decode_fingerprint,
                    observed,
                    recomputed,
                })
            }
            Err(error) => Err(error),
        }
    }

    // Staged storage API consumed by the epic's runtime slice; remove this allow there.
    #[allow(dead_code)]
    pub fn approvals(&self) -> Result<Vec<ApprovalRow>, SynapseStoreError> {
        self.validate_persisted_migration_state()?;
        let rows = self.store.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT row_id, schema_revision, model_id, decode_fingerprint, enabled,
                        grammar_enabled, disabled_reason, approved_by, approved_at_ms,
                        updated_at_ms, evidence_requirements_revision, semantic_digest,
                        generation, fencing_metadata
                 FROM approvals ORDER BY model_id ASC, decode_fingerprint ASC",
            )?;
            let rows = stmt
                .query_map([], approval_from_row)?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(rows)
        })?;
        rows.into_iter()
            .map(|row| {
                validate_approval(&row, true)?;
                Ok(row)
            })
            .collect()
    }

    // Staged storage API consumed by the epic's runtime slice; remove this allow there.
    #[allow(dead_code)]
    pub fn enable_approval(
        &self,
        model_id: &str,
        decode_fingerprint: &str,
        approved_by: &str,
        approved_at_ms: u64,
        updated_at_ms: u64,
    ) -> Result<ApprovalRow, SynapseStoreError> {
        self.mutate_approval(model_id, decode_fingerprint, updated_at_ms, |row| {
            row.enabled = true;
            row.disabled_reason = None;
            row.approved_by = Some(approved_by.to_string());
            row.approved_at_ms = Some(approved_at_ms);
        })
    }

    // Staged storage API consumed by the epic's runtime slice; remove this allow there.
    #[allow(dead_code)]
    pub fn disable_approval(
        &self,
        model_id: &str,
        decode_fingerprint: &str,
        reason: &str,
        updated_at_ms: u64,
    ) -> Result<ApprovalRow, SynapseStoreError> {
        if reason.trim().is_empty() {
            return Err(SynapseStoreError::Decode(
                "disabled approval requires a non-empty reason".to_string(),
            ));
        }
        self.mutate_approval(model_id, decode_fingerprint, updated_at_ms, |row| {
            row.enabled = false;
            row.disabled_reason = Some(reason.to_string());
        })
    }

    // Staged storage API consumed by the epic's runtime slice; remove this allow there.
    #[allow(dead_code)]
    pub fn set_approval_grammar_enabled(
        &self,
        model_id: &str,
        decode_fingerprint: &str,
        grammar_enabled: bool,
        updated_at_ms: u64,
    ) -> Result<ApprovalRow, SynapseStoreError> {
        self.mutate_approval(model_id, decode_fingerprint, updated_at_ms, |row| {
            row.grammar_enabled = grammar_enabled;
        })
    }

    // Staged storage API consumed by the epic's runtime slice; remove this allow there.
    #[allow(dead_code)]
    pub fn emergency_rollback(
        &self,
        reason: &str,
        updated_at_ms: u64,
    ) -> Result<usize, SynapseStoreError> {
        if reason.trim().is_empty() {
            return Err(SynapseStoreError::Decode(
                "emergency rollback requires a non-empty reason".to_string(),
            ));
        }
        let changed = self.store.with_conn_fenced(|tx| {
            let mut stmt = tx.prepare(
                "SELECT row_id, schema_revision, model_id, decode_fingerprint, enabled,
                        grammar_enabled, disabled_reason, approved_by, approved_at_ms,
                        updated_at_ms, evidence_requirements_revision, semantic_digest,
                        generation, fencing_metadata
                 FROM approvals ORDER BY row_id",
            )?;
            let rows = stmt
                .query_map([], approval_from_row)?
                .collect::<Result<Vec<_>, _>>()?;
            drop(stmt);
            let mut changed = 0;
            for mut row in rows {
                if row.enabled || row.disabled_reason.as_deref() != Some(reason) {
                    row.enabled = false;
                    row.disabled_reason = Some(reason.to_string());
                    row.updated_at_ms = updated_at_ms;
                    row.generation = row.generation.saturating_add(1);
                    row.semantic_digest = row.expected_digest().map_err(to_sql_error)?;
                    tx.execute(
                        "UPDATE approvals SET enabled = 0, disabled_reason = ?1,
                             updated_at_ms = ?2, generation = ?3, semantic_digest = ?4
                         WHERE row_id = ?5",
                        params![
                            row.disabled_reason.as_deref(),
                            updated_at_ms as i64,
                            row.generation as i64,
                            &row.semantic_digest,
                            row.row_id as i64,
                        ],
                    )?;
                    changed += 1;
                }
            }
            Ok(changed)
        })?;
        Ok(changed)
    }

    // This storage helper is temporarily unused. Remove the dead-code allowance
    // when a runtime caller is added.
    #[allow(dead_code)]
    fn mutate_approval<F>(
        &self,
        model_id: &str,
        decode_fingerprint: &str,
        updated_at_ms: u64,
        mutate: F,
    ) -> Result<ApprovalRow, SynapseStoreError>
    where
        F: FnOnce(&mut ApprovalRow),
    {
        let row = self.store.with_conn_fenced(|tx| {
            let mut row = load_approval_tx_by_identity(tx, model_id, decode_fingerprint)?
                .ok_or_else(|| rusqlite::Error::QueryReturnedNoRows)?;
            validate_approval(&row, true).map_err(to_sql_error)?;
            mutate(&mut row);
            row.updated_at_ms = updated_at_ms;
            row.generation = row.generation.saturating_add(1);
            row.semantic_digest = row.expected_digest().map_err(to_sql_error)?;
            let fencing_metadata = serde_json::to_string(&row.fencing_metadata)
                .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;
            tx.execute(
                "UPDATE approvals SET enabled = ?1, grammar_enabled = ?2,
                     disabled_reason = ?3, approved_by = ?4, approved_at_ms = ?5,
                     updated_at_ms = ?6, semantic_digest = ?7, generation = ?8,
                     fencing_metadata = ?9 WHERE row_id = ?10",
                params![
                    row.enabled as i64,
                    row.grammar_enabled as i64,
                    row.disabled_reason.as_deref(),
                    row.approved_by.as_deref(),
                    row.approved_at_ms.map(|value| value as i64),
                    updated_at_ms as i64,
                    &row.semantic_digest,
                    row.generation as i64,
                    fencing_metadata,
                    row.row_id as i64,
                ],
            )?;
            Ok(row)
        })?;
        Ok(row)
    }

    /// Validate, assign, and durably record an immutable serving artifact. A Q8
    /// repack is accepted only when its independently verified digest matches the
    /// deterministic derivation result supplied at ingest.
    pub fn ingest_serving_artifact(
        &self,
        request: &ArtifactIngestRequest,
        ingested_at_ms: u64,
    ) -> Result<ModelArtifactId, SynapseStoreError> {
        let artifact = ModelArtifactId::from_ingest_request(request)?;
        let artifact_json = serde_json::to_string(&artifact)?;
        let derivation = request.q8_derivation.as_ref();
        self.store
            .with_conn_fenced(|tx| {
                let existing = load_serving_artifact_tx(tx, &artifact.artifact_id)?;
                if let Some(existing) = existing {
                    if existing.artifact == artifact && existing.derivation == request.q8_derivation
                    {
                        return Ok(existing.artifact);
                    }
                    return Err(to_sql_error(SynapseStoreError::Decode(format!(
                        "serving artifact identity collision for '{}'",
                        artifact.artifact_id
                    ))));
                }
                tx.execute(
                    "INSERT INTO serving_artifacts (
                     artifact_id, model_id, source_format, source_quantization,
                     source_digest, derived_digest, derivation_contract,
                     deterministic_inputs_digest, verified_derived_digest, artifact_json,
                     gc_pinned, ingested_at_ms
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, 0, ?11)",
                    params![
                        &artifact.artifact_id,
                        &artifact.model_id,
                        &artifact.source_format,
                        &artifact.source_quantization,
                        &artifact.source_digest,
                        artifact.derived_digest.as_deref(),
                        derivation.map(|value| value.derivation_contract.as_str()),
                        derivation.map(|value| value.deterministic_inputs_digest.as_str()),
                        derivation.map(|value| value.verified_derived_digest.as_str()),
                        artifact_json,
                        ingested_at_ms as i64,
                    ],
                )?;
                Ok(artifact.clone())
            })
            .map_err(Into::into)
    }

    /// Set a cache-retention pin. Pins are consulted only by GC disposition and
    /// never by admission, disablement, or revocation control transactions.
    pub fn set_serving_artifact_gc_pin(
        &self,
        artifact_id: &str,
        pinned: bool,
    ) -> Result<(), SynapseStoreError> {
        let changed = self.store.with_conn_fenced(|tx| {
            tx.execute(
                "UPDATE serving_artifacts SET gc_pinned = ?1 WHERE artifact_id = ?2",
                params![pinned as i64, artifact_id],
            )
        })?;
        if changed != 1 {
            return Err(SynapseStoreError::Decode(format!(
                "unknown serving artifact '{artifact_id}'"
            )));
        }
        Ok(())
    }

    /// Return the GC-only disposition for an ingested artifact.
    pub fn serving_artifact_gc_disposition(
        &self,
        artifact_id: &str,
    ) -> Result<ArtifactGcDisposition, SynapseStoreError> {
        let pinned = self.store.with_conn(|conn| {
            conn.query_row(
                "SELECT gc_pinned FROM serving_artifacts WHERE artifact_id = ?1",
                params![artifact_id],
                |row| row.get::<_, i64>(0),
            )
            .optional()
        })?;
        match pinned {
            Some(0) => Ok(ArtifactGcDisposition::Collectable),
            Some(1) => Ok(ArtifactGcDisposition::Pinned),
            Some(_) => Err(SynapseStoreError::Decode(
                "serving artifact has invalid GC pin state".to_string(),
            )),
            None => Err(SynapseStoreError::Decode(format!(
                "unknown serving artifact '{artifact_id}'"
            ))),
        }
    }

    /// Persist a complete immutable certification record only when it matches an
    /// already ingested source artifact and its optional verified derivation.
    pub fn store_serving_certification(
        &self,
        record: &CertificationRecord,
        recorded_at_ms: u64,
    ) -> Result<StoredServingCertification, SynapseStoreError> {
        validate_serving_certification(record)?;
        let record_json = serde_json::to_string(record)?;
        let stored = self.store.with_conn_fenced(|tx| {
            let artifact_id = &record.artifact_lineage.artifact_id;
            let artifact = load_serving_artifact_tx(tx, artifact_id)?.ok_or_else(|| {
                to_sql_error(SynapseStoreError::Decode(format!(
                    "certification references unknown artifact '{artifact_id}'"
                )))
            })?;
            validate_certification_artifact_match(record, &artifact).map_err(to_sql_error)?;
            let existing = load_serving_certification_tx(tx, &record.record_id)?;
            if let Some(existing) = existing {
                if existing.record == *record && existing.artifact == artifact.artifact {
                    return Ok(existing);
                }
                return Err(to_sql_error(SynapseStoreError::Decode(format!(
                    "certification record '{}' is immutable",
                    record.record_id
                ))));
            }
            tx.execute(
                "INSERT INTO serving_certification_records (
                     certification_record_id, catalog_fingerprint, artifact_id,
                     record_json, recorded_at_ms
                 ) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    &record.record_id,
                    &record.unit.catalog_fingerprint,
                    artifact_id,
                    record_json,
                    recorded_at_ms as i64,
                ],
            )?;
            Ok(StoredServingCertification {
                artifact: artifact.artifact,
                record: record.clone(),
                recorded_at_ms,
            })
        })?;
        Ok(stored)
    }

    /// Load and revalidate a serving certification record before use.
    pub fn serving_certification(
        &self,
        certification_record_id: &str,
    ) -> Result<Option<StoredServingCertification>, SynapseStoreError> {
        let stored = self
            .store
            .with_conn(|conn| load_serving_certification_conn(conn, certification_record_id))?;
        if let Some(stored) = &stored {
            validate_serving_certification(&stored.record)?;
            let artifact = self
                .store
                .with_conn(|conn| load_serving_artifact_conn(conn, &stored.artifact.artifact_id))?;
            let artifact = artifact.ok_or_else(|| {
                SynapseStoreError::Decode(format!(
                    "certification references missing artifact '{}'",
                    stored.artifact.artifact_id
                ))
            })?;
            validate_certification_artifact_match(&stored.record, &artifact)?;
        }
        Ok(stored)
    }

    /// Atomically bind an enabled approval to the exact catalog fingerprint and
    /// complete certification record. A different fingerprint or source artifact
    /// cannot be approved through this transaction.
    pub fn approve_serving_catalog(
        &self,
        catalog_fingerprint: &str,
        certification_record_id: &str,
        approved_by: &str,
        approved_at_ms: u64,
    ) -> Result<ServingApprovalRecord, SynapseStoreError> {
        if !is_digest(catalog_fingerprint) {
            return Err(SynapseStoreError::Decode(
                "catalog_fingerprint must be a lowercase SHA-256 digest".to_string(),
            ));
        }
        if approved_by.trim().is_empty() {
            return Err(SynapseStoreError::Decode(
                "serving approval requires a non-empty approver".to_string(),
            ));
        }
        let approval = self.store.with_conn_fenced(|tx| {
            let certification = load_serving_certification_tx(tx, certification_record_id)?
                .ok_or_else(|| {
                    to_sql_error(SynapseStoreError::Decode(format!(
                        "unknown serving certification '{certification_record_id}'"
                    )))
                })?;
            validate_serving_certification(&certification.record).map_err(to_sql_error)?;
            if certification.record.unit.catalog_fingerprint != catalog_fingerprint {
                return Err(to_sql_error(SynapseStoreError::Decode(
                    "approval catalog fingerprint does not match certification".to_string(),
                )));
            }
            let artifact = load_serving_artifact_tx(tx, &certification.artifact.artifact_id)?
                .ok_or_else(|| {
                    to_sql_error(SynapseStoreError::Decode(
                        "serving certification references a missing artifact".to_string(),
                    ))
                })?;
            validate_certification_artifact_match(&certification.record, &artifact)
                .map_err(to_sql_error)?;

            let generation = load_serving_approval_tx(tx, catalog_fingerprint)?
                .map(|current| current.generation.saturating_add(1))
                .unwrap_or(0);
            let mut approval = ServingApprovalRecord {
                schema_revision: SERVING_APPROVAL_SCHEMA_REVISION.to_string(),
                catalog_fingerprint: catalog_fingerprint.to_string(),
                certification_record_id: certification_record_id.to_string(),
                artifact_id: certification.artifact.artifact_id,
                state: ServingApprovalState::Enabled,
                reason: None,
                approved_by: approved_by.to_string(),
                approved_at_ms,
                updated_at_ms: approved_at_ms,
                generation,
                semantic_digest: String::new(),
            };
            approval.semantic_digest = approval.expected_digest();
            validate_serving_approval(&approval, true).map_err(to_sql_error)?;
            tx.execute(
                "INSERT INTO serving_approvals (
                     catalog_fingerprint, certification_record_id, artifact_id, state,
                     reason, approved_by, approved_at_ms, updated_at_ms, generation,
                     semantic_digest
                 ) VALUES (?1, ?2, ?3, ?4, NULL, ?5, ?6, ?6, ?7, ?8)
                 ON CONFLICT(catalog_fingerprint) DO UPDATE SET
                     certification_record_id = excluded.certification_record_id,
                     artifact_id = excluded.artifact_id,
                     state = excluded.state,
                     reason = NULL,
                     approved_by = excluded.approved_by,
                     approved_at_ms = excluded.approved_at_ms,
                     updated_at_ms = excluded.updated_at_ms,
                     generation = excluded.generation,
                     semantic_digest = excluded.semantic_digest",
                params![
                    &approval.catalog_fingerprint,
                    &approval.certification_record_id,
                    &approval.artifact_id,
                    approval.state.as_str(),
                    &approval.approved_by,
                    approval.approved_at_ms as i64,
                    approval.generation as i64,
                    &approval.semantic_digest,
                ],
            )?;
            Ok(approval)
        })?;
        Ok(approval)
    }

    /// Return one serving approval after verifying its tamper-evident fields.
    pub fn serving_approval(
        &self,
        catalog_fingerprint: &str,
    ) -> Result<Option<ServingApprovalRecord>, SynapseStoreError> {
        let approval = self
            .store
            .with_conn(|conn| load_serving_approval_conn(conn, catalog_fingerprint))?;
        if let Some(approval) = &approval {
            validate_serving_approval(approval, true)?;
        }
        Ok(approval)
    }

    /// Disable one catalog fingerprint. The transaction rejects new admissions and
    /// invalidates retained states, but deliberately leaves active sessions running.
    pub fn disable_serving_catalog(
        &self,
        catalog_fingerprint: &str,
        reason: &str,
        updated_at_ms: u64,
    ) -> Result<ServingControlOutcome, SynapseStoreError> {
        self.control_serving_catalog(
            catalog_fingerprint,
            ServingApprovalState::Disabled,
            reason,
            updated_at_ms,
        )
    }

    /// Revoke one catalog fingerprint as an emergency control action. The
    /// transaction invalidates retained states and marks active sessions for
    /// terminal `artifact_revoked` accounting at their next committed boundary.
    pub fn revoke_serving_catalog(
        &self,
        catalog_fingerprint: &str,
        reason: &str,
        updated_at_ms: u64,
    ) -> Result<ServingControlOutcome, SynapseStoreError> {
        self.control_serving_catalog(
            catalog_fingerprint,
            ServingApprovalState::Revoked,
            reason,
            updated_at_ms,
        )
    }

    /// Admit a new active session only through the current approval and matching
    /// certification record. A refusal writes no session row.
    pub fn admit_serving_session(
        &self,
        session_id: &str,
        catalog_fingerprint: &str,
        created_at_ms: u64,
    ) -> Result<ServingSessionAdmission, SynapseStoreError> {
        validate_session_id(session_id)?;
        self.store
            .with_conn_fenced(|tx| {
                if load_serving_session_tx(tx, session_id)?.is_some() {
                    return Err(to_sql_error(SynapseStoreError::Decode(format!(
                        "serving session '{session_id}' already exists"
                    ))));
                }
                let Some(approval) = load_serving_approval_tx(tx, catalog_fingerprint)? else {
                    return Ok(ServingSessionAdmission::Refused {
                        reason: ServingRefusal::ArtifactUnapproved,
                    });
                };
                validate_serving_approval(&approval, true).map_err(to_sql_error)?;
                let refusal = serving_approval_refusal_tx(tx, &approval)?;
                if let Some(reason) = refusal {
                    return Ok(ServingSessionAdmission::Refused { reason });
                }
                tx.execute(
                    "INSERT INTO serving_sessions (
                     session_id, catalog_fingerprint, approval_generation, state,
                     committed_token_count, terminal_reason, created_at_ms, updated_at_ms
                 ) VALUES (?1, ?2, ?3, 'active', 0, NULL, ?4, ?4)",
                    params![
                        session_id,
                        catalog_fingerprint,
                        approval.generation as i64,
                        created_at_ms as i64,
                    ],
                )?;
                Ok(ServingSessionAdmission::Admitted {
                    session_id: session_id.to_string(),
                    catalog_fingerprint: catalog_fingerprint.to_string(),
                    approval_generation: approval.generation,
                })
            })
            .map_err(Into::into)
    }

    /// Retain a continuation state while the catalog remains admitted. Later
    /// disablement or revocation changes this row atomically with its approval.
    pub fn retain_serving_state(
        &self,
        state_id: &str,
        catalog_fingerprint: &str,
        created_at_ms: u64,
    ) -> Result<ServingContinuationAdmission, SynapseStoreError> {
        validate_session_id(state_id)?;
        self.store
            .with_conn_fenced(|tx| {
                if load_retained_serving_state_tx(tx, state_id)?.is_some() {
                    return Err(to_sql_error(SynapseStoreError::Decode(format!(
                        "retained serving state '{state_id}' already exists"
                    ))));
                }
                let Some(approval) = load_serving_approval_tx(tx, catalog_fingerprint)? else {
                    return Ok(ServingContinuationAdmission::Refused {
                        reason: ServingRefusal::ArtifactUnapproved,
                    });
                };
                validate_serving_approval(&approval, true).map_err(to_sql_error)?;
                if let Some(reason) = serving_approval_refusal_tx(tx, &approval)? {
                    return Ok(ServingContinuationAdmission::Refused { reason });
                }
                tx.execute(
                    "INSERT INTO serving_retained_states (
                     state_id, catalog_fingerprint, valid, invalidation_reason,
                     created_at_ms, updated_at_ms
                 ) VALUES (?1, ?2, 1, NULL, ?3, ?3)",
                    params![state_id, catalog_fingerprint, created_at_ms as i64],
                )?;
                Ok(ServingContinuationAdmission::Admitted {
                    state_id: state_id.to_string(),
                    catalog_fingerprint: catalog_fingerprint.to_string(),
                    approval_generation: approval.generation,
                })
            })
            .map_err(Into::into)
    }

    /// Check a retained continuation against both its invalidation state and the
    /// current approval. Disabled and revoked continuations retain those precise
    /// typed refusals instead of collapsing into a generic stale-state result.
    pub fn admit_serving_continuation(
        &self,
        state_id: &str,
    ) -> Result<ServingContinuationAdmission, SynapseStoreError> {
        self.store
            .with_conn_fenced(|tx| {
                let Some(state) = load_retained_serving_state_tx(tx, state_id)? else {
                    return Ok(ServingContinuationAdmission::Refused {
                        reason: ServingRefusal::RetainedStateInvalidated,
                    });
                };
                if !state.valid {
                    return Ok(ServingContinuationAdmission::Refused {
                        reason: retained_state_refusal(&state),
                    });
                }
                let Some(approval) = load_serving_approval_tx(tx, &state.catalog_fingerprint)?
                else {
                    return Ok(ServingContinuationAdmission::Refused {
                        reason: ServingRefusal::ArtifactUnapproved,
                    });
                };
                validate_serving_approval(&approval, true).map_err(to_sql_error)?;
                if let Some(reason) = serving_approval_refusal_tx(tx, &approval)? {
                    return Ok(ServingContinuationAdmission::Refused { reason });
                }
                Ok(ServingContinuationAdmission::Admitted {
                    state_id: state.state_id,
                    catalog_fingerprint: state.catalog_fingerprint,
                    approval_generation: approval.generation,
                })
            })
            .map_err(Into::into)
    }

    /// Commit an absolute token count at a session boundary. A pending emergency
    /// revoke turns this exact boundary into the terminal accounting record.
    pub fn commit_serving_session_boundary(
        &self,
        session_id: &str,
        committed_token_count: u64,
        updated_at_ms: u64,
    ) -> Result<ServingBoundaryOutcome, SynapseStoreError> {
        self.store
            .with_conn_fenced(|tx| {
                let mut session = load_serving_session_tx(tx, session_id)?.ok_or_else(|| {
                    to_sql_error(SynapseStoreError::Decode(format!(
                        "unknown serving session '{session_id}'"
                    )))
                })?;
                if !session.state.is_active() {
                    return Err(to_sql_error(SynapseStoreError::Decode(format!(
                        "serving session '{session_id}' is already terminal"
                    ))));
                }
                if committed_token_count < session.committed_token_count {
                    return Err(to_sql_error(SynapseStoreError::Decode(
                        "committed token count cannot move backward".to_string(),
                    )));
                }
                let approval = load_serving_approval_tx(tx, &session.catalog_fingerprint)?;
                let termination_requested = session.state
                    == ServingSessionState::TerminationRequested
                    || approval
                        .as_ref()
                        .is_some_and(|approval| approval.state == ServingApprovalState::Revoked);
                if termination_requested {
                    session.state = ServingSessionState::Terminated;
                    session.terminal_reason = Some("artifact_revoked".to_string());
                }
                session.committed_token_count = committed_token_count;
                session.updated_at_ms = updated_at_ms;
                tx.execute(
                    "UPDATE serving_sessions
                 SET state = ?1, committed_token_count = ?2, terminal_reason = ?3,
                     updated_at_ms = ?4
                 WHERE session_id = ?5",
                    params![
                        session.state.as_str(),
                        committed_token_count as i64,
                        session.terminal_reason.as_deref(),
                        updated_at_ms as i64,
                        session_id,
                    ],
                )?;
                if session.state == ServingSessionState::Terminated {
                    let unload_artifact =
                        serving_artifact_unload_ready_tx(tx, &session.catalog_fingerprint)?;
                    return Ok(ServingBoundaryOutcome::Terminated {
                        terminal_reason: "artifact_revoked".to_string(),
                        tokens_emitted: committed_token_count,
                        unload_artifact,
                    });
                }
                Ok(ServingBoundaryOutcome::Continue {
                    committed_token_count,
                })
            })
            .map_err(Into::into)
    }

    /// Finish an active session naturally. A disabled approval reaches unload only
    /// after this final active session leaves the ledger.
    pub fn complete_serving_session(
        &self,
        session_id: &str,
        updated_at_ms: u64,
    ) -> Result<ServingSessionCompletion, SynapseStoreError> {
        let completion = self.store.with_conn_fenced(|tx| {
            let mut session = load_serving_session_tx(tx, session_id)?.ok_or_else(|| {
                to_sql_error(SynapseStoreError::Decode(format!(
                    "unknown serving session '{session_id}'"
                )))
            })?;
            if session.state != ServingSessionState::Active {
                return Err(to_sql_error(SynapseStoreError::Decode(
                    "only an active serving session can finish naturally".to_string(),
                )));
            }
            session.state = ServingSessionState::Finished;
            session.terminal_reason = Some("completed".to_string());
            session.updated_at_ms = updated_at_ms;
            tx.execute(
                "UPDATE serving_sessions
                 SET state = 'finished', terminal_reason = 'completed', updated_at_ms = ?1
                 WHERE session_id = ?2",
                params![updated_at_ms as i64, session_id],
            )?;
            let unload_artifact =
                serving_artifact_unload_ready_tx(tx, &session.catalog_fingerprint)?;
            Ok(ServingSessionCompletion {
                session,
                unload_artifact,
            })
        })?;
        Ok(completion)
    }

    /// Read a session ledger entry for recovery and terminal accounting.
    pub fn serving_session(
        &self,
        session_id: &str,
    ) -> Result<Option<ServingSessionRecord>, SynapseStoreError> {
        Ok(self
            .store
            .with_conn(|conn| load_serving_session_conn(conn, session_id))?)
    }

    /// Read a retained-state ledger entry for continuation recovery.
    pub fn retained_serving_state(
        &self,
        state_id: &str,
    ) -> Result<Option<RetainedServingState>, SynapseStoreError> {
        Ok(self
            .store
            .with_conn(|conn| load_retained_serving_state_conn(conn, state_id))?)
    }

    fn control_serving_catalog(
        &self,
        catalog_fingerprint: &str,
        requested_state: ServingApprovalState,
        reason: &str,
        updated_at_ms: u64,
    ) -> Result<ServingControlOutcome, SynapseStoreError> {
        if requested_state == ServingApprovalState::Enabled || reason.trim().is_empty() {
            return Err(SynapseStoreError::Decode(
                "serving control requires a disabling state and non-empty reason".to_string(),
            ));
        }
        let outcome = self.store.with_conn_fenced(|tx| {
            let mut approval =
                load_serving_approval_tx(tx, catalog_fingerprint)?.ok_or_else(|| {
                    to_sql_error(SynapseStoreError::Decode(format!(
                        "no serving approval for catalog fingerprint '{catalog_fingerprint}'"
                    )))
                })?;
            validate_serving_approval(&approval, true).map_err(to_sql_error)?;
            let effective_state = if approval.state == ServingApprovalState::Revoked {
                ServingApprovalState::Revoked
            } else {
                requested_state
            };
            let effective_reason = if effective_state == ServingApprovalState::Revoked
                && approval.state == ServingApprovalState::Revoked
            {
                approval
                    .reason
                    .clone()
                    .expect("validated revoked approval has a reason")
            } else {
                reason.to_string()
            };
            if approval.state != effective_state
                || approval.reason.as_deref() != Some(effective_reason.as_str())
            {
                approval.state = effective_state;
                approval.reason = Some(effective_reason);
                approval.updated_at_ms = updated_at_ms;
                approval.generation = approval.generation.saturating_add(1);
                approval.semantic_digest = approval.expected_digest();
                validate_serving_approval(&approval, true).map_err(to_sql_error)?;
                tx.execute(
                    "UPDATE serving_approvals
                     SET state = ?1, reason = ?2, updated_at_ms = ?3, generation = ?4,
                         semantic_digest = ?5
                     WHERE catalog_fingerprint = ?6",
                    params![
                        approval.state.as_str(),
                        approval.reason.as_deref(),
                        updated_at_ms as i64,
                        approval.generation as i64,
                        &approval.semantic_digest,
                        catalog_fingerprint,
                    ],
                )?;
            }
            let invalidated_retained_states = match effective_state {
                ServingApprovalState::Disabled => tx.execute(
                    "UPDATE serving_retained_states
                     SET valid = 0, invalidation_reason = 'artifact_disabled', updated_at_ms = ?1
                     WHERE catalog_fingerprint = ?2 AND valid = 1",
                    params![updated_at_ms as i64, catalog_fingerprint],
                )? as u64,
                ServingApprovalState::Revoked => tx.execute(
                    "UPDATE serving_retained_states
                     SET valid = 0, invalidation_reason = 'artifact_revoked', updated_at_ms = ?1
                     WHERE catalog_fingerprint = ?2
                       AND (valid != 0 OR invalidation_reason != 'artifact_revoked')",
                    params![updated_at_ms as i64, catalog_fingerprint],
                )? as u64,
                ServingApprovalState::Enabled => unreachable!("control never enables an approval"),
            };
            let termination_requested_sessions = if effective_state == ServingApprovalState::Revoked
            {
                tx.execute(
                    "UPDATE serving_sessions
                     SET state = 'termination_requested', updated_at_ms = ?1
                     WHERE catalog_fingerprint = ?2 AND state = 'active'",
                    params![updated_at_ms as i64, catalog_fingerprint],
                )? as u64
            } else {
                0
            };
            let active_sessions = active_serving_session_count_tx(tx, catalog_fingerprint)?;
            let unload_artifact = active_sessions == 0;
            Ok(ServingControlOutcome {
                approval,
                invalidated_retained_states,
                active_sessions,
                termination_requested_sessions,
                unload_artifact,
            })
        })?;
        Ok(outcome)
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
        let (
            certification_class,
            status,
            machine_profile_hash,
            remote_profile_hash,
            identity_revision,
        ) = match &row.key {
            CertificationKey::Measured {
                machine_profile_hash,
            } => (
                CertificationClass::LegacyOwnedDecode,
                CertificationStatus::Uncertified,
                Some(machine_profile_hash.as_str()),
                None,
                None,
            ),
            CertificationKey::Declared {
                machine_profile_hash,
                remote_profile_hash,
                identity_revision,
            } => (
                CertificationClass::Declared,
                row.status,
                Some(machine_profile_hash.as_str()),
                Some(remote_profile_hash.as_str()),
                Some(identity_revision.as_str()),
            ),
        };
        self.store.with_conn_fenced(|tx| {
            tx.execute(
                "INSERT INTO cert_rows (
                     certification_class, assurance_class, status, key_hash,
                     machine_profile_hash, remote_profile_hash, identity_revision,
                     numeric_profile_id, fingerprint, certified_at_ms, os_build,
                     module_generation, evidence_json
                  ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
                  ON CONFLICT (key_hash, fingerprint)
                  WHERE certification_class = 'declared' DO UPDATE SET
                     status = excluded.status,
                     machine_profile_hash = excluded.machine_profile_hash,
                     remote_profile_hash = excluded.remote_profile_hash,
                     identity_revision = excluded.identity_revision,
                     numeric_profile_id = excluded.numeric_profile_id,
                     certified_at_ms = excluded.certified_at_ms,
                     os_build = excluded.os_build,
                     module_generation = excluded.module_generation,
                     evidence_json = excluded.evidence_json",
                params![
                    certification_class.as_str(),
                    row.assurance_class.as_str(),
                    status.as_str(),
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

    /// Store a complete measured owned-decode row under its class-scoped key.
    /// This is the only certification write that can create a matchable owned row.
    // Staged storage API consumed by the epic's runtime slice; remove this allow there.
    /// Store a complete measured owned-decode row under its shape-scoped key.
    /// This is the only certification write that can create a matchable owned row.
    // This direct writer is retained for storage tests; production probes use
    // the fenced multi-row terminal write below.
    #[allow(dead_code)]
    pub fn store_owned_decode_cert_row(
        &self,
        row: &OwnedDecodeCertificationRow,
    ) -> Result<(), SynapseStoreError> {
        validate_owned_decode_cert_row(row)?;
        self.store
            .with_conn_fenced(|tx| upsert_owned_decode_cert_row_tx(tx, row))?;
        Ok(())
    }

    /// Atomically revalidate a probe snapshot and write terminal evidence. A
    /// changed tuple records `stale_probe` and commits no cert row.
    pub fn store_owned_decode_cert_row_if_current(
        &self,
        snapshot: &OwnedDecodeMatchInputs,
        terminal: &OwnedDecodeMatchInputs,
        row: &OwnedDecodeCertificationRow,
        observed_at_ms: u64,
    ) -> Result<ProbeWriteOutcome, SynapseStoreError> {
        self.store_owned_decode_cert_rows_if_current(
            snapshot,
            terminal,
            std::slice::from_ref(row),
            observed_at_ms,
        )
    }

    /// Commit one row for each request shape exercised by a terminal probe.
    pub fn store_owned_decode_cert_rows_if_current(
        &self,
        snapshot: &OwnedDecodeMatchInputs,
        terminal: &OwnedDecodeMatchInputs,
        rows: &[OwnedDecodeCertificationRow],
        observed_at_ms: u64,
    ) -> Result<ProbeWriteOutcome, SynapseStoreError> {
        if rows.is_empty() {
            return Err(SynapseStoreError::Decode(
                "terminal probe must write at least one certification row".to_string(),
            ));
        }
        let mut shape_identities = BTreeSet::new();
        for row in rows {
            validate_owned_decode_cert_row(row)?;
            if !shape_identities.insert(row.constraint_runtime_identities.clone()) {
                return Err(SynapseStoreError::Decode(
                    "terminal probe contains duplicate request-shape evidence".to_string(),
                ));
            }
            let shape_matches = row.constraint_runtime_identities.is_empty()
                || row.constraint_runtime_identities == terminal.constraint_runtime_identities;
            let row_matches_terminal = row.revisioned_machine_profile_hash
                == terminal.revisioned_machine_profile_hash
                && row.profile_activation_epoch == terminal.profile_activation_epoch
                && row.model_id == terminal.model_id
                && row.decode_fingerprint == terminal.decode_fingerprint
                && row.processing_fingerprint == terminal.processing_fingerprint
                && row.runtime_config_digest == terminal.runtime_config_digest
                && shape_matches
                && row.worker_path_evidence == terminal.worker_path_evidence
                && row.evidence_schema_revision == terminal.evidence_schema_revision
                && row.g_dec_manifest_revision == terminal.g_dec_manifest_revision;
            if row.revisioned_machine_profile_hash != snapshot.revisioned_machine_profile_hash
                || row.profile_activation_epoch != snapshot.profile_activation_epoch
                || row.model_id != snapshot.model_id
                || row.decode_fingerprint != snapshot.decode_fingerprint
                || !row_matches_terminal
            {
                return self.record_stale_probe(
                    snapshot,
                    terminal,
                    "probe_tuple_changed",
                    observed_at_ms,
                );
            }
        }
        if !owned_decode_match_inputs_equal(snapshot, terminal) {
            return self.record_stale_probe(
                snapshot,
                terminal,
                "probe_tuple_changed",
                observed_at_ms,
            );
        }
        let outcome = self.store.with_conn_fenced(|tx| {
            let (current_hash, current_epoch): (Option<String>, Option<i64>) = tx.query_row(
                "SELECT revisioned_machine_profile_hash, profile_activation_epoch
                 FROM profile_state WHERE id = 0",
                [],
                |query_row| Ok((query_row.get(0)?, query_row.get(1)?)),
            )?;
            if current_hash.as_deref() != Some(snapshot.revisioned_machine_profile_hash.as_str())
                || current_epoch != Some(snapshot.profile_activation_epoch as i64)
            {
                let detail = serde_json::to_string(&StaleProbeEvent {
                    outcome: "stale_probe".to_string(),
                    model_id: snapshot.model_id.clone(),
                    decode_fingerprint: snapshot.decode_fingerprint.clone(),
                    observed_profile_activation_epoch: snapshot.profile_activation_epoch,
                    current_profile_activation_epoch: current_epoch.map(|value| value as u64),
                    reason: "profile_activation_changed".to_string(),
                })
                .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;
                tx.execute(
                    "INSERT INTO cert_row_rebuild_events (outcome, detail, occurred_at_ms)
                     VALUES (?1, ?2, ?3)",
                    params!["stale_probe", detail, observed_at_ms as i64],
                )?;
                return Ok(ProbeWriteOutcome::ProbeStale);
            }
            for row in rows {
                upsert_owned_decode_cert_row_tx(tx, row)?;
            }
            Ok(ProbeWriteOutcome::Certified)
        })?;
        Ok(outcome)
    }

    fn record_stale_probe(
        &self,
        snapshot: &OwnedDecodeMatchInputs,
        terminal: &OwnedDecodeMatchInputs,
        reason: &str,
        observed_at_ms: u64,
    ) -> Result<ProbeWriteOutcome, SynapseStoreError> {
        let current_epoch = self.current_profile_activation_epoch()?;
        let detail = serde_json::to_string(&StaleProbeEvent {
            outcome: "stale_probe".to_string(),
            model_id: snapshot.model_id.clone(),
            decode_fingerprint: snapshot.decode_fingerprint.clone(),
            observed_profile_activation_epoch: snapshot.profile_activation_epoch,
            current_profile_activation_epoch: current_epoch,
            reason: if snapshot.profile_activation_epoch != terminal.profile_activation_epoch {
                "profile_activation_changed".to_string()
            } else {
                reason.to_string()
            },
        })?;
        self.store.with_conn_fenced(|tx| {
            tx.execute(
                "INSERT INTO cert_row_rebuild_events (outcome, detail, occurred_at_ms)
                 VALUES (?1, ?2, ?3)",
                params!["stale_probe", detail, observed_at_ms as i64],
            )?;
            Ok(())
        })?;
        Ok(ProbeWriteOutcome::ProbeStale)
    }

    pub fn get_owned_decode_cert_row(
        &self,
        revisioned_machine_profile_hash: &str,
        profile_activation_epoch: u64,
        model_id: &str,
        decode_fingerprint: &str,
        evidence_schema_revision: &str,
        constraint_runtime_identities: &[String],
    ) -> Result<Option<OwnedDecodeCertificationRow>, SynapseStoreError> {
        let identities_digest =
            constraint_runtime_identities_digest(constraint_runtime_identities)?;
        let raw = self.store.with_conn(|conn| {
            conn.query_row(
                &format!(
                    "{OWNED_CERT_SELECT_SQL}
                     WHERE certification_class = 'measured_owned_decode'
                       AND status = 'certified'
                       AND revisioned_machine_profile_hash = ?1
                       AND profile_activation_epoch = ?2
                       AND model_id = ?3
                       AND decode_fingerprint = ?4
                       AND evidence_schema_revision = ?5
                       AND constraint_runtime_identities_digest = ?6"
                ),
                params![
                    revisioned_machine_profile_hash,
                    profile_activation_epoch as i64,
                    model_id,
                    decode_fingerprint,
                    evidence_schema_revision,
                    identities_digest,
                ],
                owned_decode_cert_row_from_row,
            )
            .optional()
        })?;
        raw.map(decode_owned_decode_cert_row).transpose()
    }

    pub fn get_owned_decode_measurement_row(
        &self,
        revisioned_machine_profile_hash: &str,
        profile_activation_epoch: u64,
        model_id: &str,
        decode_fingerprint: &str,
        evidence_schema_revision: &str,
        constraint_runtime_identities: &[String],
    ) -> Result<Option<OwnedDecodeCertificationRow>, SynapseStoreError> {
        let identities_digest =
            constraint_runtime_identities_digest(constraint_runtime_identities)?;
        let raw = self.store.with_conn(|conn| {
            conn.query_row(
                &format!(
                    "{OWNED_CERT_SELECT_SQL}
                     WHERE certification_class = 'measured_owned_decode'
                       AND revisioned_machine_profile_hash = ?1
                       AND profile_activation_epoch = ?2
                       AND model_id = ?3
                       AND decode_fingerprint = ?4
                       AND evidence_schema_revision = ?5
                       AND constraint_runtime_identities_digest = ?6
                     ORDER BY certified_at_ms DESC LIMIT 1"
                ),
                params![
                    revisioned_machine_profile_hash,
                    profile_activation_epoch as i64,
                    model_id,
                    decode_fingerprint,
                    evidence_schema_revision,
                    identities_digest,
                ],
                owned_decode_cert_row_from_row,
            )
            .optional()
        })?;
        raw.map(decode_owned_decode_cert_row).transpose()
    }

    /// The newest certified decode row for this lane under ANY machine profile.
    ///
    /// The profile-scoped lookup above answers "is this lane certified HERE".
    /// This answers "was it ever certified", which is the other half an operator
    /// needs: together they separate a lane that was certified and then had its
    /// profile rotate from one that was never probed. Without it the decode
    /// branch has only one fact and both states render identically -- see
    /// `docs/audits/meaning-serving-gates.md`.
    pub fn latest_owned_decode_measurement_row(
        &self,
        model_id: &str,
        decode_fingerprint: &str,
        evidence_schema_revision: &str,
        constraint_runtime_identities: &[String],
    ) -> Result<Option<OwnedDecodeCertificationRow>, SynapseStoreError> {
        let identities_digest =
            constraint_runtime_identities_digest(constraint_runtime_identities)?;
        let raw = self.store.with_conn(|conn| {
            conn.query_row(
                &format!(
                    "{OWNED_CERT_SELECT_SQL}
                     WHERE certification_class = 'measured_owned_decode'
                       AND status = 'certified'
                       AND model_id = ?1
                       AND decode_fingerprint = ?2
                       AND evidence_schema_revision = ?3
                       AND constraint_runtime_identities_digest = ?4
                     ORDER BY certified_at_ms DESC LIMIT 1"
                ),
                params![
                    model_id,
                    decode_fingerprint,
                    evidence_schema_revision,
                    identities_digest,
                ],
                owned_decode_cert_row_from_row,
            )
            .optional()
        })?;
        raw.map(decode_owned_decode_cert_row).transpose()
    }

    fn current_owned_decode_cert_rows(
        &self,
        revisioned_machine_profile_hash: &str,
        profile_activation_epoch: u64,
        model_id: &str,
        decode_fingerprint: &str,
    ) -> Result<Vec<OwnedDecodeCertificationRow>, SynapseStoreError> {
        let raw = self.store.with_conn(|conn| {
            let mut statement = conn.prepare(&format!(
                "{OWNED_CERT_SELECT_SQL}
                 WHERE certification_class = 'measured_owned_decode'
                   AND status = 'certified'
                   AND revisioned_machine_profile_hash = ?1
                   AND profile_activation_epoch = ?2
                   AND model_id = ?3
                   AND decode_fingerprint = ?4
                   AND evidence_schema_revision = ?5
                 ORDER BY certified_at_ms DESC, row_id DESC"
            ))?;
            let rows = statement
                .query_map(
                    params![
                        revisioned_machine_profile_hash,
                        profile_activation_epoch as i64,
                        model_id,
                        decode_fingerprint,
                        CERT_EVIDENCE_SCHEMA_REVISION,
                    ],
                    owned_decode_cert_row_from_row,
                )?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(rows)
        })?;
        raw.into_iter().map(decode_owned_decode_cert_row).collect()
    }

    // Staged storage API consumed by the epic's runtime slice; remove this allow there.
    #[allow(dead_code)]
    pub fn get_owned_decode_cert_row_matching(
        &self,
        inputs: &OwnedDecodeMatchInputs,
    ) -> Result<Option<OwnedDecodeCertificationRow>, SynapseStoreError> {
        let Some(row) = self.get_owned_decode_cert_row(
            &inputs.revisioned_machine_profile_hash,
            inputs.profile_activation_epoch,
            &inputs.model_id,
            &inputs.decode_fingerprint,
            &inputs.evidence_schema_revision,
            &inputs.constraint_runtime_identities,
        )?
        else {
            return Ok(None);
        };
        let mut required_identities = inputs.constraint_runtime_identities.clone();
        required_identities.sort();
        required_identities.dedup();
        if required_identities != inputs.constraint_runtime_identities
            || row.processing_fingerprint != inputs.processing_fingerprint
            || row.runtime_config_digest != inputs.runtime_config_digest
            || row.constraint_runtime_identities != inputs.constraint_runtime_identities
            || row.worker_path_evidence != inputs.worker_path_evidence
            || row.g_dec_manifest_revision != inputs.g_dec_manifest_revision
            || !probe_gate_evidence_passed(&row.evidence, &inputs.g_dec_manifest_revision)
        {
            return Ok(None);
        }
        Ok(Some(row))
    }

    /// Compare the approval stamp and profile epoch immediately before dispatch.
    /// The join keeps approval mutation and epoch rotation in one fenced read.
    pub fn owned_decode_dispatch_admission_matches(
        &self,
        profile_activation_epoch: u64,
        model_id: &str,
        decode_fingerprint: &str,
        approval_semantic_digest: &str,
        approval_generation: u64,
    ) -> Result<bool, SynapseStoreError> {
        if profile_activation_epoch == 0 {
            return Ok(false);
        }
        self.validate_persisted_migration_state()?;
        let matches = self.store.with_conn_fenced(|tx| {
            let current = tx
                .query_row(
                    "SELECT profile_state.profile_activation_epoch, approvals.enabled,
                            approvals.semantic_digest, approvals.generation
                     FROM profile_state
                     JOIN approvals ON approvals.model_id = ?1
                         AND approvals.decode_fingerprint = ?2
                     WHERE profile_state.id = 0",
                    params![model_id, decode_fingerprint],
                    |row| {
                        Ok((
                            row.get::<_, Option<i64>>(0)?,
                            row.get::<_, i64>(1)? != 0,
                            row.get::<_, String>(2)?,
                            row.get::<_, i64>(3)? as u64,
                        ))
                    },
                )
                .optional()?;
            Ok(matches!(
                current,
                Some((Some(epoch), true, digest, generation))
                    if epoch == profile_activation_epoch as i64
                        && digest == approval_semantic_digest
                        && generation == approval_generation
            ))
        })?;
        Ok(matches)
    }

    /// Read the two independent values that authorize owned-decode admission—the
    /// persisted revisioned profile hash and its positive activation epoch—in one
    /// consistent database read.
    // Staged storage API consumed by the epic's runtime slice; remove this allow there.
    #[allow(dead_code)]
    pub fn owned_decode_admission(
        &self,
        model_id: &str,
        decode_fingerprint: &str,
        revisioned_machine_profile_hash: &str,
        profile_activation_epoch: u64,
    ) -> Result<Option<OwnedDecodeAdmission>, SynapseStoreError> {
        self.validate_persisted_migration_state()?;
        if profile_activation_epoch == 0 {
            return Ok(None);
        }
        let identities_digest = constraint_runtime_identities_digest(&[])?;
        let admission = self.store.with_conn_fenced(|tx| {
            let (persisted_hash, persisted_epoch): (Option<String>, Option<i64>) = tx.query_row(
                "SELECT revisioned_machine_profile_hash, profile_activation_epoch
                 FROM profile_state WHERE id = 0",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )?;
            if persisted_hash.as_deref() != Some(revisioned_machine_profile_hash)
                || persisted_epoch != Some(profile_activation_epoch as i64)
            {
                return Ok(None);
            }
            let Some(approval) = load_approval_tx_by_identity(tx, model_id, decode_fingerprint)?
            else {
                return Ok(None);
            };
            validate_approval(&approval, true).map_err(to_sql_error)?;
            if !approval.enabled {
                return Ok(None);
            }
            let raw = tx
                .query_row(
                    &format!(
                        "{OWNED_CERT_SELECT_SQL}
                         WHERE certification_class = 'measured_owned_decode'
                           AND status = 'certified'
                           AND revisioned_machine_profile_hash = ?1
                           AND profile_activation_epoch = ?2
                           AND model_id = ?3
                           AND decode_fingerprint = ?4
                           AND evidence_schema_revision = ?5
                           AND constraint_runtime_identities_digest = ?6"
                    ),
                    params![
                        revisioned_machine_profile_hash,
                        profile_activation_epoch as i64,
                        model_id,
                        decode_fingerprint,
                        CERT_EVIDENCE_SCHEMA_REVISION,
                        &identities_digest,
                    ],
                    owned_decode_cert_row_from_row,
                )
                .optional()?;
            let Some(raw) = raw else { return Ok(None) };
            let certification = decode_owned_decode_cert_row(raw).map_err(to_sql_error)?;
            validate_owned_decode_cert_row(&certification).map_err(to_sql_error)?;
            if !probe_gate_evidence_passed(&certification.evidence, G_DEC_MANIFEST_REVISION) {
                return Ok(None);
            }
            Ok(Some(OwnedDecodeAdmission {
                approval,
                certification,
                revisioned_machine_profile_hash: revisioned_machine_profile_hash.to_string(),
                profile_activation_epoch,
            }))
        })?;
        Ok(admission)
    }

    /// Resolve the complete owned-decode match tuple in one fenced read. The
    /// compatibility wrapper preserves the old admission-only API for health and
    /// reporting callers that do not expose a refusal to a request caller.
    pub fn owned_decode_admission_matching(
        &self,
        inputs: &OwnedDecodeMatchInputs,
    ) -> Result<Option<OwnedDecodeAdmission>, SynapseStoreError> {
        Ok(self
            .owned_decode_admission_evaluation(inputs)?
            .admission()
            .cloned())
    }

    /// Resolve the complete owned-decode match tuple in one fenced read while
    /// retaining the failed predicate arm for routing. Only a missing or disabled
    /// approval is a cutover refusal; all later mismatches mean current evidence
    /// cannot certify serving.
    pub fn owned_decode_admission_evaluation(
        &self,
        inputs: &OwnedDecodeMatchInputs,
    ) -> Result<OwnedDecodeAdmissionEvaluation, SynapseStoreError> {
        self.validate_persisted_migration_state()?;
        if inputs.profile_activation_epoch == 0
            || inputs.evidence_schema_revision != CERT_EVIDENCE_SCHEMA_REVISION
            || inputs.g_dec_manifest_revision != G_DEC_MANIFEST_REVISION
        {
            return Ok(OwnedDecodeAdmissionEvaluation::Refused {
                approval: Box::new(None),
                refusal: OwnedDecodeAdmissionRefusal::NotCertified,
            });
        }
        let mut expected_constraints = inputs.constraint_runtime_identities.clone();
        expected_constraints.sort();
        expected_constraints.dedup();
        if expected_constraints != inputs.constraint_runtime_identities {
            return Ok(OwnedDecodeAdmissionEvaluation::Refused {
                approval: Box::new(None),
                refusal: OwnedDecodeAdmissionRefusal::NotCertified,
            });
        }
        let identities_digest = constraint_runtime_identities_digest(&expected_constraints)?;
        Ok(self.store.with_conn_fenced(|tx| {
            let (persisted_hash, persisted_epoch): (Option<String>, Option<i64>) = tx.query_row(
                "SELECT revisioned_machine_profile_hash, profile_activation_epoch
                 FROM profile_state WHERE id = 0",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )?;
            if persisted_hash.as_deref() != Some(inputs.revisioned_machine_profile_hash.as_str())
                || persisted_epoch != Some(inputs.profile_activation_epoch as i64)
            {
                return Ok(OwnedDecodeAdmissionEvaluation::Refused {
                    approval: Box::new(None),
                    refusal: OwnedDecodeAdmissionRefusal::NotCertified,
                });
            }
            let catalog_matches: i64 = tx.query_row(
                "SELECT COUNT(*) FROM models WHERE model_id = ?1",
                params![&inputs.model_id],
                |row| row.get(0),
            )?;
            if catalog_matches != 1 || !is_catalog_model_id(&inputs.model_id) {
                return Ok(OwnedDecodeAdmissionEvaluation::Refused {
                    approval: Box::new(None),
                    refusal: OwnedDecodeAdmissionRefusal::NotCertified,
                });
            }
            let Some(approval) =
                load_approval_tx_by_identity(tx, &inputs.model_id, &inputs.decode_fingerprint)?
            else {
                return Ok(OwnedDecodeAdmissionEvaluation::Refused {
                    approval: Box::new(None),
                    refusal: OwnedDecodeAdmissionRefusal::ApprovalAbsent,
                });
            };
            validate_approval(&approval, true).map_err(to_sql_error)?;
            if !approval.enabled {
                return Ok(OwnedDecodeAdmissionEvaluation::Refused {
                    approval: Box::new(Some(approval.clone())),
                    refusal: OwnedDecodeAdmissionRefusal::ApprovalDisabled {
                        disabled_reason: approval
                            .disabled_reason
                            .clone()
                            .expect("validated disabled approval has a reason"),
                    },
                });
            }
            let raw = tx
                .query_row(
                    &format!(
                        "{OWNED_CERT_SELECT_SQL}
                         WHERE certification_class = 'measured_owned_decode'
                           AND status = 'certified'
                           AND revisioned_machine_profile_hash = ?1
                           AND profile_activation_epoch = ?2
                           AND model_id = ?3
                           AND decode_fingerprint = ?4
                           AND evidence_schema_revision = ?5
                           AND constraint_runtime_identities_digest = ?6"
                    ),
                    params![
                        &inputs.revisioned_machine_profile_hash,
                        inputs.profile_activation_epoch as i64,
                        &inputs.model_id,
                        &inputs.decode_fingerprint,
                        &inputs.evidence_schema_revision,
                        &identities_digest,
                    ],
                    owned_decode_cert_row_from_row,
                )
                .optional()?;
            let Some(raw) = raw else {
                return Ok(OwnedDecodeAdmissionEvaluation::Refused {
                    approval: Box::new(Some(approval)),
                    refusal: OwnedDecodeAdmissionRefusal::NotCertified,
                });
            };
            let certification = decode_owned_decode_cert_row(raw).map_err(to_sql_error)?;
            validate_owned_decode_cert_row(&certification).map_err(to_sql_error)?;
            if certification.revisioned_machine_profile_hash
                != inputs.revisioned_machine_profile_hash
                || certification.profile_activation_epoch != inputs.profile_activation_epoch
                || certification.model_id != inputs.model_id
                || certification.decode_fingerprint != inputs.decode_fingerprint
                || certification.processing_fingerprint != inputs.processing_fingerprint
                || certification.runtime_config_digest != inputs.runtime_config_digest
                || certification.constraint_runtime_identities != expected_constraints
                || certification.worker_path_evidence != inputs.worker_path_evidence
                || certification.evidence_schema_revision != inputs.evidence_schema_revision
                || certification.g_dec_manifest_revision != inputs.g_dec_manifest_revision
                || !probe_gate_evidence_passed(
                    &certification.evidence,
                    &inputs.g_dec_manifest_revision,
                )
            {
                return Ok(OwnedDecodeAdmissionEvaluation::Refused {
                    approval: Box::new(Some(approval)),
                    refusal: OwnedDecodeAdmissionRefusal::NotCertified,
                });
            }
            Ok(OwnedDecodeAdmissionEvaluation::Admitted(Box::new(
                OwnedDecodeAdmission {
                    approval,
                    certification,
                    revisioned_machine_profile_hash: inputs.revisioned_machine_profile_hash.clone(),
                    profile_activation_epoch: inputs.profile_activation_epoch,
                },
            )))
        })?)
    }

    // Staged storage API consumed by the epic's runtime slice; remove this allow there.
    #[allow(dead_code)]
    pub fn store_class_scoped_cert_row(
        &self,
        row: &ClassScopedCertificationRow,
    ) -> Result<(), SynapseStoreError> {
        if matches!(
            row.certification_class,
            CertificationClass::MeasuredOwnedDecode | CertificationClass::LegacyOwnedDecode
        ) {
            return Err(SynapseStoreError::Decode(
                "owned classes require their dedicated row API".to_string(),
            ));
        }
        if row.key_hash.trim().is_empty() || row.fingerprint.0.trim().is_empty() {
            return Err(SynapseStoreError::Decode(
                "class-scoped certification identity must be non-empty".to_string(),
            ));
        }
        if matches!(
            row.certification_class,
            CertificationClass::Declared | CertificationClass::Remote
        ) && (row
            .remote_profile_hash
            .as_deref()
            .map(str::is_empty)
            .unwrap_or(true)
            || row
                .identity_revision
                .as_deref()
                .map(str::is_empty)
                .unwrap_or(true))
        {
            return Err(SynapseStoreError::Decode(
                "declared and remote rows require remote profile and identity revision".to_string(),
            ));
        }
        let evidence_json = serde_json::to_string(&row.evidence)?;
        let class = row.certification_class.as_str();
        let sql = format!(
            "INSERT INTO cert_rows (
                 certification_class, assurance_class, status, key_hash,
                 machine_profile_hash, remote_profile_hash, identity_revision,
                 numeric_profile_id, fingerprint, certified_at_ms, os_build,
                 module_generation, evidence_json
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
             ON CONFLICT (key_hash, fingerprint)
             WHERE certification_class = '{class}' DO UPDATE SET
                 status = excluded.status,
                 machine_profile_hash = excluded.machine_profile_hash,
                 remote_profile_hash = excluded.remote_profile_hash,
                 identity_revision = excluded.identity_revision,
                 numeric_profile_id = excluded.numeric_profile_id,
                 certified_at_ms = excluded.certified_at_ms,
                 os_build = excluded.os_build,
                 module_generation = excluded.module_generation,
                 evidence_json = excluded.evidence_json"
        );
        self.store.with_conn_fenced(|tx| {
            tx.execute(
                &sql,
                params![
                    class,
                    row.assurance_class.as_str(),
                    row.status.as_str(),
                    &row.key_hash,
                    row.machine_profile_hash.as_deref(),
                    row.remote_profile_hash.as_deref(),
                    row.identity_revision.as_deref(),
                    row.numeric_profile_id.as_ref().map(|id| &id.0),
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

    // Staged storage API consumed by the epic's runtime slice; remove this allow there.
    #[allow(dead_code)]
    pub fn get_class_scoped_cert_row(
        &self,
        certification_class: CertificationClass,
        key_hash: &str,
        fingerprint: &Fingerprint,
    ) -> Result<Option<ClassScopedCertificationRow>, SynapseStoreError> {
        let raw = self.store.with_conn(|conn| {
            conn.query_row(
                "SELECT certification_class, assurance_class, status, key_hash,
                        machine_profile_hash, remote_profile_hash, identity_revision,
                        numeric_profile_id, fingerprint, certified_at_ms, os_build,
                        module_generation, evidence_json
                 FROM cert_rows
                 WHERE certification_class = ?1 AND key_hash = ?2 AND fingerprint = ?3
                 ORDER BY certified_at_ms DESC LIMIT 1",
                params![certification_class.as_str(), key_hash, &fingerprint.0],
                class_scoped_cert_row_from_row,
            )
            .optional()
        })?;
        raw.map(decode_class_scoped_cert_row).transpose()
    }

    pub fn get_cert_row(
        &self,
        certification_class: CertificationClass,
        machine_profile_hash: &str,
        fingerprint: &Fingerprint,
    ) -> Result<Option<CertificationRow>, SynapseStoreError> {
        if !matches!(
            certification_class,
            CertificationClass::Embedding | CertificationClass::Rerank
        ) {
            return Err(SynapseStoreError::Decode(
                "measured certification reads require embedding or rerank class".to_string(),
            ));
        }
        let raw = self.store.with_conn(|conn| {
            conn.query_row(
                &format!(
                    "{CERT_SELECT_SQL} WHERE assurance_class = 'measured'
                     AND certification_class = ?1 AND status = 'certified'
                     AND key_hash = ?2 AND fingerprint = ?3"
                ),
                params![
                    certification_class.as_str(),
                    machine_profile_hash,
                    &fingerprint.0
                ],
                cert_row_from_row,
            )
            .optional()
        })?;
        raw.map(decode_cert_row).transpose()
    }

    pub fn latest_cert_row(
        &self,
        certification_class: CertificationClass,
        fingerprint: &Fingerprint,
    ) -> Result<Option<CertificationRow>, SynapseStoreError> {
        let raw = self.store.with_conn(|conn| {
            conn.query_row(
                &format!(
                    "{CERT_SELECT_SQL} WHERE certification_class = ?1
                     AND status = 'certified' AND fingerprint = ?2
                     ORDER BY certified_at_ms DESC LIMIT 1"
                ),
                params![certification_class.as_str(), &fingerprint.0],
                cert_row_from_row,
            )
            .optional()
        })?;
        raw.map(decode_cert_row).transpose()
    }

    pub fn get_probe_row(
        &self,
        certification_class: CertificationClass,
        machine_profile_hash: &str,
        fingerprint: &Fingerprint,
    ) -> Result<Option<CertificationRow>, SynapseStoreError> {
        let raw = self.store.with_conn(|conn| {
            conn.query_row(
                &format!(
                    "{CERT_SELECT_SQL} WHERE assurance_class = 'measured'
                     AND certification_class = ?1 AND key_hash = ?2
                     AND fingerprint = ?3 ORDER BY certified_at_ms DESC LIMIT 1"
                ),
                params![
                    certification_class.as_str(),
                    machine_profile_hash,
                    &fingerprint.0
                ],
                cert_row_from_row,
            )
            .optional()
        })?;
        raw.map(decode_cert_row).transpose()
    }

    pub fn latest_probe_row(
        &self,
        certification_class: CertificationClass,
        fingerprint: &Fingerprint,
    ) -> Result<Option<CertificationRow>, SynapseStoreError> {
        let raw = self.store.with_conn(|conn| {
            conn.query_row(
                &format!(
                    "{CERT_SELECT_SQL} WHERE assurance_class = 'measured'
                     AND certification_class = ?1 AND fingerprint = ?2
                     ORDER BY certified_at_ms DESC LIMIT 1"
                ),
                params![certification_class.as_str(), &fingerprint.0],
                cert_row_from_row,
            )
            .optional()
        })?;
        raw.map(decode_cert_row).transpose()
    }

    pub fn has_stale_cert_row(
        &self,
        certification_class: CertificationClass,
        machine_profile_hash: &str,
        fingerprint: &Fingerprint,
    ) -> Result<bool, SynapseStoreError> {
        let count = self.store.with_conn(|conn| {
            conn.query_row(
                "SELECT COUNT(1) FROM cert_rows
                 WHERE assurance_class = 'measured' AND certification_class = ?1
                   AND (status = 'certified' OR (
                       status = 'uncertified'
                       AND json_extract(evidence_json, '$.blocking_reason') IS NULL
                   ))
                   AND fingerprint = ?2 AND key_hash <> ?3",
                params![
                    certification_class.as_str(),
                    &fingerprint.0,
                    machine_profile_hash
                ],
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

    pub fn current_profile_activation_epoch(&self) -> Result<Option<u64>, SynapseStoreError> {
        let epoch = self.store.with_conn(|conn| {
            conn.query_row(
                "SELECT profile_activation_epoch FROM profile_state WHERE id = 0",
                [],
                |row| row.get::<_, Option<i64>>(0),
            )
        })?;
        Ok(epoch.map(|value| value as u64))
    }

    pub fn profile_state(&self) -> Result<ProfileState, SynapseStoreError> {
        let state = self.store.with_conn(|conn| {
            conn.query_row(
                "SELECT snapshot_json, revisioned_machine_profile_hash,
                         profile_activation_epoch, previous_revisioned_machine_profile_hash,
                         last_rotation_reason, last_rotation_at_ms,
                         certification_stale_since_ms
                  FROM profile_state WHERE id = 0",
                [],
                profile_state_from_row,
            )
        })?;
        validate_profile_state(&state)?;
        Ok(state)
    }

    /// Persist the first certification mismatch observed for the current profile.
    /// A hash guard prevents a request that raced a profile rotation from marking
    /// the newly activated profile stale.
    pub fn observe_certification_stale(
        &self,
        revisioned_machine_profile_hash: &str,
        observed_at_ms: u64,
    ) -> Result<Option<u64>, SynapseStoreError> {
        let observed_at_ms = observed_at_ms.min(i64::MAX as u64) as i64;
        let stale_since = self.store.with_conn_fenced(|tx| {
            tx.execute(
                "UPDATE profile_state
                 SET certification_stale_since_ms = COALESCE(certification_stale_since_ms, ?1)
                 WHERE id = 0 AND revisioned_machine_profile_hash = ?2",
                params![observed_at_ms, revisioned_machine_profile_hash],
            )?;
            tx.query_row(
                "SELECT certification_stale_since_ms
                 FROM profile_state
                 WHERE id = 0 AND revisioned_machine_profile_hash = ?1",
                params![revisioned_machine_profile_hash],
                |row| row.get::<_, Option<i64>>(0),
            )
            .optional()
        })?;
        Ok(stale_since.flatten().map(|value| value as u64))
    }

    /// Observe and durably activate a collected profile. The first observation
    /// creates epoch 1; a changed revisioned hash performs the snapshot, epoch,
    /// parent event, and child outcome writes in one fenced transaction.
    pub fn observe_profile(
        &self,
        profile: &MachineProfile,
        observed_at_ms: u64,
        module_generation: u64,
    ) -> Result<ProfileActivation, SynapseStoreError> {
        let current_hash = profile.revisioned_hash();
        let current_snapshot_json = serde_json::to_string(profile)?;
        let activation = self.store.with_conn_fenced(|tx| {
            // A trust-only restore removes the captured machine profile. Recreate
            // the empty singleton on first observation so no stale machine fact
            // is consulted while the replacement profile is being derived.
            tx.execute("INSERT OR IGNORE INTO profile_state (id) VALUES (0)", [])?;
            let persisted = tx.query_row(
                "SELECT snapshot_json, revisioned_machine_profile_hash,
                         profile_activation_epoch, previous_revisioned_machine_profile_hash,
                         last_rotation_reason, last_rotation_at_ms,
                         certification_stale_since_ms
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
                    certification_stale_since_ms: None,
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
                      last_rotation_at_ms = ?6,
                      certification_stale_since_ms = NULL
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
                         last_rotation_reason, last_rotation_at_ms,
                         certification_stale_since_ms
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
            let mut stmt = tx.prepare(
                "SELECT model_id, decode_fingerprint, enabled
                 FROM approvals ORDER BY model_id ASC, decode_fingerprint ASC",
            )?;
            let approvals = stmt
                .query_map([], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, i64>(2)? != 0,
                    ))
                })?
                .collect::<Result<Vec<_>, _>>()?;
            drop(stmt);
            for (model_id, decode_fingerprint, enabled) in approvals {
                tx.execute(
                    "INSERT INTO profile_rotation_certification_outcomes (
                         event_id, model_id, decode_fingerprint, outcome_state
                     ) VALUES (?1, ?2, ?3, ?4)",
                    params![
                        &event_id,
                        model_id,
                        decode_fingerprint,
                        if enabled { "required" } else { "not_required" },
                    ],
                )?;
            }
            let event = rotation_event_tx(tx, &event_id)?;
            let state = ProfileState {
                snapshot: Some(profile.clone()),
                revisioned_machine_profile_hash: Some(current_hash.clone()),
                profile_activation_epoch: Some(next_epoch),
                previous_revisioned_machine_profile_hash: Some(old_hash),
                last_rotation_reason: Some(reason),
                last_rotation_at_ms: Some(observed_at_ms),
                certification_stale_since_ms: None,
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

    // Staged storage API consumed by the epic's runtime slice; remove this allow there.
    #[allow(dead_code)]
    pub fn activate_profile(
        &self,
        profile: &MachineProfile,
        observed_at_ms: u64,
        module_generation: u64,
    ) -> Result<ProfileActivation, SynapseStoreError> {
        self.observe_profile(profile, observed_at_ms, module_generation)
    }

    // Staged storage API consumed by the epic's runtime slice; remove this allow there.
    #[allow(dead_code)]
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

    // Staged storage API consumed by the epic's runtime slice; remove this allow there.
    #[allow(dead_code)]
    pub fn rotation_event(
        &self,
        event_id: &str,
    ) -> Result<Option<RotationLedgerEvent>, SynapseStoreError> {
        Ok(self
            .store
            .with_conn(|conn| rotation_event_conn(conn, event_id))?)
    }

    // Staged storage API consumed by the epic's runtime slice; remove this allow there.
    #[allow(dead_code)]
    pub fn set_rotation_certification_outcome(
        &self,
        event_id: &str,
        model_id: &str,
        decode_fingerprint: &str,
        outcome_state: RotationOutcomeState,
        certified_at_ms: Option<u64>,
        failure_reason: Option<&str>,
    ) -> Result<(), SynapseStoreError> {
        match outcome_state {
            RotationOutcomeState::Passed if certified_at_ms.is_none() => {
                return Err(SynapseStoreError::Decode(
                    "passed rotation outcome requires certified_at_ms".to_string(),
                ));
            }
            RotationOutcomeState::Failed if failure_reason.map(str::is_empty).unwrap_or(true) => {
                return Err(SynapseStoreError::Decode(
                    "failed rotation outcome requires a failure reason".to_string(),
                ));
            }
            _ => {}
        }
        self.store.with_conn_fenced(|tx| {
            let changed = tx.execute(
                "UPDATE profile_rotation_certification_outcomes
                 SET outcome_state = ?1, certified_at_ms = ?2, failure_reason = ?3
                 WHERE event_id = ?4 AND model_id = ?5 AND decode_fingerprint = ?6",
                params![
                    outcome_state.encode(),
                    certified_at_ms.map(|value| value as i64),
                    failure_reason,
                    event_id,
                    model_id,
                    decode_fingerprint,
                ],
            )?;
            if changed != 1 {
                return Err(rusqlite::Error::QueryReturnedNoRows);
            }
            Ok(())
        })?;
        Ok(())
    }

    pub fn rotation_outcomes(
        &self,
        event_id: &str,
    ) -> Result<Vec<RotationCertificationOutcome>, SynapseStoreError> {
        Ok(self.store.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT event_id, model_id, decode_fingerprint, outcome_state,
                        certified_at_ms, failure_reason
                 FROM profile_rotation_certification_outcomes
                 WHERE event_id = ?1 ORDER BY model_id ASC, decode_fingerprint ASC",
            )?;
            let rows = stmt
                .query_map(params![event_id], rotation_outcome_from_row)?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(rows)
        })?)
    }

    /// Return persisted values used to add storage information to health responses,
    /// without changing the existing machine-profile hash calculation or its
    /// compatibility behavior.
    // Staged storage API consumed by the epic's runtime slice; remove this allow there.
    #[allow(dead_code)]
    pub fn storage_health_inputs(&self) -> Result<StorageHealthInputs, SynapseStoreError> {
        let state = self.profile_state()?;
        let latest_event = self.store.with_conn(|conn| {
            conn.query_row(
                "SELECT event_id FROM profile_rotation_events
                     ORDER BY new_profile_activation_epoch DESC LIMIT 1",
                [],
                |row| row.get::<_, String>(0),
            )
            .optional()
        })?;
        let ledger_outcomes = match latest_event.as_deref() {
            Some(event_id) => self.rotation_outcomes(event_id)?,
            None => Vec::new(),
        };
        let outcome_map = ledger_outcomes
            .into_iter()
            .map(|outcome| {
                (
                    (outcome.model_id.clone(), outcome.decode_fingerprint.clone()),
                    outcome,
                )
            })
            .collect::<BTreeMap<_, _>>();
        let approvals = self.store.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT row_id, schema_revision, model_id, decode_fingerprint, enabled,
                        grammar_enabled, disabled_reason, approved_by, approved_at_ms,
                        updated_at_ms, evidence_requirements_revision, semantic_digest,
                        generation, fencing_metadata
                 FROM approvals ORDER BY model_id ASC, decode_fingerprint ASC",
            )?;
            let rows = stmt
                .query_map([], approval_from_row)?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(rows)
        })?;
        let mut evidence_requirements_divergence = Vec::new();
        let mut approval_certification_outcomes = Vec::with_capacity(approvals.len());
        for approval in approvals {
            let key = (
                approval.model_id.clone(),
                approval.decode_fingerprint.clone(),
            );
            let ledger = outcome_map.get(&key);
            let validation = validate_approval(&approval, true);
            let (admission, failure_reason) = match validation {
                Err(SynapseStoreError::Decode(message))
                    if message.contains("unsupported approval schema revision")
                        || message
                            .contains("unsupported approval evidence requirements revision") =>
                {
                    evidence_requirements_divergence.push(EvidenceRequirementsDivergence {
                        model_id: approval.model_id.clone(),
                        decode_fingerprint: approval.decode_fingerprint.clone(),
                        result: "approval_revision_unsupported".to_string(),
                    });
                    ("disabled".to_string(), Some(message))
                }
                Err(SynapseStoreError::ApprovalDigestMismatch { .. }) => (
                    "digest_mismatch".to_string(),
                    Some("approval_digest_mismatch".to_string()),
                ),
                Err(error) => ("disabled".to_string(), Some(error.to_string())),
                Ok(()) if !approval.enabled => {
                    ("disabled".to_string(), approval.disabled_reason.clone())
                }
                Ok(()) => {
                    let stored_revisions = match (
                        state.revisioned_machine_profile_hash.as_deref(),
                        state.profile_activation_epoch,
                    ) {
                        (Some(hash), Some(epoch)) => self.store.with_conn(|conn| {
                            conn.query_row(
                                "SELECT evidence_schema_revision, g_dec_manifest_revision
                                 FROM cert_rows
                                 WHERE certification_class = 'measured_owned_decode'
                                   AND status = 'certified'
                                   AND revisioned_machine_profile_hash = ?1
                                   AND profile_activation_epoch = ?2
                                   AND model_id = ?3
                                   AND decode_fingerprint = ?4
                                 ORDER BY certified_at_ms DESC LIMIT 1",
                                params![
                                    hash,
                                    epoch as i64,
                                    &approval.model_id,
                                    &approval.decode_fingerprint,
                                ],
                                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
                            )
                            .optional()
                        })?,
                        _ => None,
                    };
                    let divergence = match stored_revisions {
                        Some((schema_revision, _))
                            if schema_revision != CERT_EVIDENCE_SCHEMA_REVISION =>
                        {
                            "evidence_schema_incompatible"
                        }
                        Some((_, manifest_revision))
                            if manifest_revision != G_DEC_MANIFEST_REVISION =>
                        {
                            "manifest_revision_incompatible"
                        }
                        _ => "compatible",
                    };
                    evidence_requirements_divergence.push(EvidenceRequirementsDivergence {
                        model_id: approval.model_id.clone(),
                        decode_fingerprint: approval.decode_fingerprint.clone(),
                        result: divergence.to_string(),
                    });
                    let current = match (
                        state.revisioned_machine_profile_hash.as_deref(),
                        state.profile_activation_epoch,
                    ) {
                        (Some(hash), Some(epoch)) => self
                            .current_owned_decode_cert_rows(
                                hash,
                                epoch,
                                &approval.model_id,
                                &approval.decode_fingerprint,
                            )?
                            .into_iter()
                            .any(|row| {
                                let inputs = OwnedDecodeMatchInputs {
                                    revisioned_machine_profile_hash: row
                                        .revisioned_machine_profile_hash,
                                    profile_activation_epoch: row.profile_activation_epoch,
                                    model_id: row.model_id,
                                    decode_fingerprint: row.decode_fingerprint,
                                    processing_fingerprint: row.processing_fingerprint,
                                    runtime_config_digest: row.runtime_config_digest,
                                    constraint_runtime_identities: row
                                        .constraint_runtime_identities,
                                    worker_path_evidence: row.worker_path_evidence,
                                    evidence_schema_revision: row.evidence_schema_revision,
                                    g_dec_manifest_revision: row.g_dec_manifest_revision,
                                };
                                self.owned_decode_admission_matching(&inputs)
                                    .ok()
                                    .flatten()
                                    .is_some()
                            }),
                        _ => false,
                    };
                    if current {
                        ("serving".to_string(), None)
                    } else {
                        (
                            "no_current_evidence".to_string(),
                            Some("no_current_evidence".to_string()),
                        )
                    }
                }
            };
            approval_certification_outcomes.push(ApprovalCertificationHealth {
                model_id: approval.model_id,
                decode_fingerprint: approval.decode_fingerprint,
                enabled: approval.enabled,
                admission,
                state: ledger
                    .map(|outcome| outcome.outcome_state.clone())
                    .unwrap_or_else(|| {
                        if approval.enabled {
                            "required"
                        } else {
                            "not_required"
                        }
                        .to_string()
                    }),
                certified_at_ms: ledger.and_then(|outcome| outcome.certified_at_ms),
                failure_reason: ledger
                    .and_then(|outcome| outcome.failure_reason.clone())
                    .or(failure_reason),
            });
        }
        let rotation_event_count = self.store.with_conn(|conn| {
            conn.query_row("SELECT COUNT(*) FROM profile_rotation_events", [], |row| {
                row.get::<_, i64>(0)
            })
        })? as u64;
        let re_certification_state = if approval_certification_outcomes.is_empty()
            || approval_certification_outcomes
                .iter()
                .all(|outcome| !outcome.enabled)
        {
            "not_required"
        } else if approval_certification_outcomes
            .iter()
            .any(|outcome| outcome.state == "failed")
        {
            "failed"
        } else if approval_certification_outcomes
            .iter()
            .any(|outcome| outcome.state == "in_progress")
        {
            "in_progress"
        } else if approval_certification_outcomes
            .iter()
            .all(|outcome| !outcome.enabled || outcome.admission == "serving")
        {
            "passed"
        } else {
            "required"
        };
        Ok(StorageHealthInputs {
            previous_revisioned_machine_profile_hash: state
                .previous_revisioned_machine_profile_hash,
            current_revisioned_machine_profile_hash: state.revisioned_machine_profile_hash,
            profile_activation_epoch: state.profile_activation_epoch,
            certification_stale_since_ms: state.certification_stale_since_ms,
            last_rotation_at_ms: state.last_rotation_at_ms,
            last_rotation_reason: state.last_rotation_reason,
            rotation_event_count,
            re_certification_state: re_certification_state.to_string(),
            evidence_requirements_divergence,
            approval_certification_outcomes,
        })
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

    pub(crate) fn knob_assignment(
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
                 WHERE module_generation < ?4 AND state IN (?5, ?6, ?7) AND kind <> ?8",
                params![
                    JOB_STATE_FAILED_TRANSIENT,
                    now_ms as i64,
                    error_bytes,
                    current_generation as i64,
                    JOB_STATE_QUEUED,
                    JOB_STATE_RUNNING,
                    JOB_STATE_PAUSED_NEEDS_REAUTH,
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
    "ane",
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
                             logical_handle, paused_at_ms, resume_deadline_ms, params_json,
                             result_json, error_json, page_count
                         ) VALUES (
                             ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7, NULL, ?8, NULL, NULL,
                             NULL, NULL, NULL, ?9, NULL, NULL, 0
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

    /// Decide at startup what to do with each persisted `models` row, which
    /// the runtime restores without `model.load` validation.
    ///
    /// An embed or rerank row whose engine is outside
    /// [`OWNED_EMBED_RERANK_ENGINES`] is deleted, and the `cache.pin` on each
    /// blob it named is dropped unless a remaining row also names that blob,
    /// so `cache.gc` can reclaim it. A row whose `model_id` is catalog-reserved
    /// (`is_reserved`) stays in the table but is not returned as registrable.
    /// Every other row is registrable.
    pub fn reconcile_persisted_registrations(
        &self,
        is_reserved: &dyn Fn(&str) -> bool,
        cache: &dyn CatalogBlobCache,
    ) -> Result<RegistrationReconciliation, SynapseStoreError> {
        let reconciliation = self.store.with_conn_fenced(|tx| {
            tx.execute(
                "DELETE FROM catalog_file_verifications WHERE NOT EXISTS (
                SELECT 1 FROM catalog_install_members m WHERE
                m.catalog_id=catalog_file_verifications.catalog_id AND
                m.manifest_digest=catalog_file_verifications.manifest_digest AND
                m.backend=catalog_file_verifications.backend AND
                m.path=catalog_file_verifications.path AND
                m.digest=catalog_file_verifications.expected_sha256)",
                [],
            )?;
            let mut stmt = tx.prepare(
                "SELECT model_id, engine, task, config_json FROM models
                 ORDER BY created_ms ASC, model_id ASC",
            )?;
            let rows = stmt
                .query_map([], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, Vec<u8>>(3)?,
                    ))
                })?
                .collect::<Result<Vec<_>, _>>()?;
            drop(stmt);

            let mut reconciliation = RegistrationReconciliation::default();
            let mut released = BTreeSet::new();
            for (model_id, engine, task, config_json) in rows {
                let embed_or_rerank = task == "embed" || task == "rerank";
                if embed_or_rerank && !OWNED_EMBED_RERANK_ENGINES.contains(&engine.as_str()) {
                    tx.execute("DELETE FROM models WHERE model_id = ?1", params![model_id])?;
                    released.extend(registration_blob_digests(&config_json));
                    reconciliation.deleted.push((model_id, engine));
                } else if is_reserved(&model_id) {
                    reconciliation.skipped_reserved.push(model_id);
                } else {
                    reconciliation
                        .registrable
                        .push(decode_model_config(config_json)?);
                }
            }
            for digest in released {
                if free_form_references_digest_tx(tx, &digest)? {
                    continue;
                }
                cache
                    .remove_pins(&digest)
                    .map_err(|error| cache_error("remove pins of", &digest, error))?;
                reconciliation.unpinned_digests.push(digest);
            }
            Ok(reconciliation)
        })?;
        Ok(reconciliation)
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
    // Download jobs have no execution TTL: they are admitted with a NULL
    // execution_expires_ms, and the kind filter keeps them exempt even if a
    // deadline were ever written. They end only through their own commit,
    // cancel or failure transactions, which also clean up their blobs.
    let execution_expired = tx.execute(
        "UPDATE jobs SET state = ?1, updated_ms = ?2, terminal_at_ms = ?2,
             execution_expires_ms = NULL, active_attempt_id = NULL, error_json = ?3
         WHERE state IN (?4, ?5) AND execution_expires_ms <= ?2 AND kind <> ?6",
        params![
            JOB_STATE_FAILED_TRANSIENT,
            now_ms,
            execution_error,
            JOB_STATE_QUEUED,
            JOB_STATE_RUNNING,
            DOWNLOAD_JOB_KIND,
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
         WHERE job_id = ?4 AND state IN (?5, ?6) AND execution_expires_ms <= ?2
           AND kind <> ?7",
        params![
            JOB_STATE_FAILED_TRANSIENT,
            now_ms,
            execution_error,
            job_id,
            JOB_STATE_QUEUED,
            JOB_STATE_RUNNING,
            DOWNLOAD_JOB_KIND,
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
        paused_at_ms, resume_deadline_ms, page_count, result_json, error_json, params_json FROM jobs";

fn unix_now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

const CERT_SELECT_SQL: &str = "SELECT assurance_class, status, machine_profile_hash,
        remote_profile_hash, identity_revision, numeric_profile_id, fingerprint,
        certified_at_ms, os_build, module_generation, evidence_json FROM cert_rows";

// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
const OWNED_CERT_SELECT_SQL: &str = "SELECT status, revisioned_machine_profile_hash,
        profile_activation_epoch, model_id, decode_fingerprint, numeric_profile_id,
        fingerprint, certified_at_ms, os_build, module_generation,
        evidence_schema_revision, processing_fingerprint, runtime_config_digest,
        constraint_runtime_identities_json, worker_path_evidence_json,
        g_dec_manifest_revision, evidence_json FROM cert_rows";

struct RawCertificationRow {
    assurance_class: String,
    status: String,
    machine_profile_hash: Option<String>,
    remote_profile_hash: Option<String>,
    identity_revision: Option<String>,
    numeric_profile_id: Option<String>,
    fingerprint: String,
    certified_at_ms: u64,
    os_build: String,
    module_generation: u64,
    evidence_json: String,
}

// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
struct RawClassScopedCertificationRow {
    certification_class: String,
    assurance_class: String,
    status: String,
    key_hash: String,
    machine_profile_hash: Option<String>,
    remote_profile_hash: Option<String>,
    identity_revision: Option<String>,
    numeric_profile_id: Option<String>,
    fingerprint: String,
    certified_at_ms: u64,
    os_build: String,
    module_generation: u64,
    evidence_json: String,
}

// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
struct RawOwnedDecodeCertificationRow {
    status: String,
    revisioned_machine_profile_hash: String,
    profile_activation_epoch: u64,
    model_id: String,
    decode_fingerprint: String,
    numeric_profile_id: Option<String>,
    fingerprint: String,
    certified_at_ms: u64,
    os_build: String,
    module_generation: u64,
    evidence_schema_revision: String,
    processing_fingerprint: String,
    runtime_config_digest: String,
    constraint_runtime_identities_json: String,
    worker_path_evidence_json: String,
    g_dec_manifest_revision: String,
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

// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
fn class_scoped_cert_row_from_row(
    row: &Row<'_>,
) -> rusqlite::Result<RawClassScopedCertificationRow> {
    Ok(RawClassScopedCertificationRow {
        certification_class: row.get(0)?,
        assurance_class: row.get(1)?,
        status: row.get(2)?,
        key_hash: row.get(3)?,
        machine_profile_hash: row.get(4)?,
        remote_profile_hash: row.get(5)?,
        identity_revision: row.get(6)?,
        numeric_profile_id: row.get(7)?,
        fingerprint: row.get(8)?,
        certified_at_ms: row.get::<_, i64>(9)? as u64,
        os_build: row.get(10)?,
        module_generation: row.get::<_, i64>(11)? as u64,
        evidence_json: row.get(12)?,
    })
}

// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
fn owned_decode_cert_row_from_row(
    row: &Row<'_>,
) -> rusqlite::Result<RawOwnedDecodeCertificationRow> {
    Ok(RawOwnedDecodeCertificationRow {
        status: row.get(0)?,
        revisioned_machine_profile_hash: row.get(1)?,
        profile_activation_epoch: row.get::<_, i64>(2)? as u64,
        model_id: row.get(3)?,
        decode_fingerprint: row.get(4)?,
        numeric_profile_id: row.get(5)?,
        fingerprint: row.get(6)?,
        certified_at_ms: row.get::<_, i64>(7)? as u64,
        os_build: row.get(8)?,
        module_generation: row.get::<_, i64>(9)? as u64,
        evidence_schema_revision: row.get(10)?,
        processing_fingerprint: row.get(11)?,
        runtime_config_digest: row.get(12)?,
        constraint_runtime_identities_json: row.get(13)?,
        worker_path_evidence_json: row.get(14)?,
        g_dec_manifest_revision: row.get(15)?,
        evidence_json: row.get(16)?,
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
        numeric_profile_id: NumericProfileId(row.numeric_profile_id.unwrap_or_default()),
        fingerprint: Fingerprint(row.fingerprint),
        certified_at_ms: row.certified_at_ms,
        os_build: row.os_build,
        module_generation: row.module_generation,
        evidence: serde_json::from_str(&row.evidence_json)?,
    })
}

// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
fn constraint_runtime_identities_digest(
    identities: &[String],
) -> Result<String, SynapseStoreError> {
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(identities)?)))
}

fn digest_text(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
fn digest_optional_text(hasher: &mut Sha256, value: Option<&str>) {
    match value {
        Some(value) => digest_text(hasher, value),
        None => {
            hasher.update([0xff; 8]);
        }
    }
}

// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
fn digest_optional_u64(hasher: &mut Sha256, value: Option<u64>) {
    match value {
        Some(value) => digest_text(hasher, &value.to_string()),
        None => {
            hasher.update([0xff; 8]);
        }
    }
}

// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_hexdigit())
        && value.bytes().all(|byte| !byte.is_ascii_uppercase())
}

// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
fn has_only_keys(object: &serde_json::Map<String, Value>, allowed: &[&str]) -> bool {
    object
        .keys()
        .all(|key| allowed.iter().any(|allowed_key| key == allowed_key))
        && allowed.iter().all(|key| object.contains_key(*key))
}

// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
fn is_catalog_model_id(value: &str) -> bool {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first.is_ascii_lowercase() || first.is_ascii_digit())
        && chars.all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ".-_".contains(ch))
}

// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
fn validate_approval(row: &ApprovalRow, verify_digest: bool) -> Result<(), SynapseStoreError> {
    if row.schema_revision != APPROVAL_SCHEMA_REVISION {
        return Err(SynapseStoreError::Decode(format!(
            "unsupported approval schema revision '{}'",
            row.schema_revision
        )));
    }
    if !is_catalog_model_id(&row.model_id) {
        return Err(SynapseStoreError::Decode(
            "approval model_id is not canonical".to_string(),
        ));
    }
    if !is_digest(&row.decode_fingerprint) {
        return Err(SynapseStoreError::Decode(
            "approval decode_fingerprint must be lowercase SHA-256".to_string(),
        ));
    }
    if row.evidence_requirements_revision != APPROVAL_EVIDENCE_REQUIREMENTS_REVISION {
        return Err(SynapseStoreError::Decode(format!(
            "unsupported approval evidence requirements revision '{}'",
            row.evidence_requirements_revision
        )));
    }
    if row.enabled {
        if row
            .approved_by
            .as_deref()
            .map(str::is_empty)
            .unwrap_or(true)
            || row.approved_at_ms.is_none()
            || row.disabled_reason.is_some()
        {
            return Err(SynapseStoreError::Decode(
                "enabled approval has invalid provenance or disabled reason".to_string(),
            ));
        }
    } else if row
        .disabled_reason
        .as_deref()
        .map(str::is_empty)
        .unwrap_or(true)
    {
        return Err(SynapseStoreError::Decode(
            "disabled approval requires a non-empty reason".to_string(),
        ));
    }
    if verify_digest {
        let recomputed = row.expected_digest()?;
        if row.semantic_digest != recomputed {
            return Err(SynapseStoreError::ApprovalDigestMismatch {
                model_id: row.model_id.clone(),
                decode_fingerprint: row.decode_fingerprint.clone(),
                observed: row.semantic_digest.clone(),
                recomputed,
            });
        }
    }
    Ok(())
}

// This storage helper is temporarily unused. Remove the dead-code allowance
// when a runtime caller is added.
#[allow(dead_code)]
fn approval_from_row(row: &Row<'_>) -> rusqlite::Result<ApprovalRow> {
    let fencing_metadata: String = row.get(13)?;
    let approved_at_ms = row.get::<_, Option<i64>>(8)?.map(|value| value as u64);
    Ok(ApprovalRow {
        row_id: row.get::<_, i64>(0)? as u64,
        schema_revision: row.get(1)?,
        model_id: row.get(2)?,
        decode_fingerprint: row.get(3)?,
        enabled: row.get::<_, i64>(4)? != 0,
        grammar_enabled: row.get::<_, i64>(5)? != 0,
        disabled_reason: row.get(6)?,
        approved_by: row.get(7)?,
        approved_at_ms,
        updated_at_ms: row.get::<_, i64>(9)? as u64,
        evidence_requirements_revision: row.get(10)?,
        semantic_digest: row.get(11)?,
        generation: row.get::<_, i64>(12)? as u64,
        fencing_metadata: serde_json::from_str(&fencing_metadata).map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(
                13,
                rusqlite::types::Type::Text,
                Box::new(error),
            )
        })?,
    })
}

// This storage helper is temporarily unused. Remove the dead-code allowance
// when a runtime caller is added.
#[allow(dead_code)]
fn load_approval_tx(
    tx: &rusqlite::Transaction<'_>,
    row_id: i64,
) -> rusqlite::Result<Option<ApprovalRow>> {
    tx.query_row(
        "SELECT row_id, schema_revision, model_id, decode_fingerprint, enabled,
                grammar_enabled, disabled_reason, approved_by, approved_at_ms,
                updated_at_ms, evidence_requirements_revision, semantic_digest,
                generation, fencing_metadata
         FROM approvals WHERE row_id = ?1",
        params![row_id],
        approval_from_row,
    )
    .optional()
}

// This storage helper is temporarily unused. Remove the dead-code allowance
// when a runtime caller is added.
#[allow(dead_code)]
fn load_approval_tx_by_identity(
    tx: &rusqlite::Transaction<'_>,
    model_id: &str,
    decode_fingerprint: &str,
) -> rusqlite::Result<Option<ApprovalRow>> {
    tx.query_row(
        "SELECT row_id, schema_revision, model_id, decode_fingerprint, enabled,
                grammar_enabled, disabled_reason, approved_by, approved_at_ms,
                updated_at_ms, evidence_requirements_revision, semantic_digest,
                generation, fencing_metadata
         FROM approvals WHERE model_id = ?1 AND decode_fingerprint = ?2",
        params![model_id, decode_fingerprint],
        approval_from_row,
    )
    .optional()
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct StoredServingArtifact {
    artifact: ModelArtifactId,
    derivation: Option<Q8IngestDerivation>,
    gc_pinned: bool,
}

fn validate_artifact_ingest(request: &ArtifactIngestRequest) -> Result<(), SynapseStoreError> {
    if request.source_format == "mlx-group-quant"
        || request.source_format == "mlx_group_quant"
        || request.source_quantization.contains("mlx")
    {
        return Err(SynapseStoreError::Decode(
            "MLX group-quant source weights are not accepted for serving ingest".to_string(),
        ));
    }
    if !is_catalog_model_id(&request.model_id)
        || request.source_format != "gguf"
        || request.source_quantization != "q4_k_m"
        || !is_digest(&request.source_digest)
    {
        return Err(SynapseStoreError::Decode(
            "serving ingest requires a canonical GGUF Q4_K_M source identity".to_string(),
        ));
    }
    if let Some(derivation) = &request.q8_derivation {
        if derivation.derivation_contract != Q8_INGEST_DERIVATION_CONTRACT
            || derivation.derived_quantization != "q8_0"
            || !is_digest(&derivation.deterministic_inputs_digest)
            || !is_digest(&derivation.derived_digest)
            || !is_digest(&derivation.verified_derived_digest)
            || derivation.derived_digest != derivation.verified_derived_digest
        {
            return Err(SynapseStoreError::Decode(
                "q8-ingest-v1 derivation must have verified deterministic Q8 output".to_string(),
            ));
        }
    }
    Ok(())
}

fn validate_stored_serving_artifact(
    artifact: &ModelArtifactId,
    derivation: Option<&Q8IngestDerivation>,
) -> Result<(), SynapseStoreError> {
    let request = ArtifactIngestRequest {
        model_id: artifact.model_id.clone(),
        source_format: artifact.source_format.clone(),
        source_quantization: artifact.source_quantization.clone(),
        source_digest: artifact.source_digest.clone(),
        q8_derivation: derivation.cloned(),
    };
    validate_artifact_ingest(&request)?;
    if artifact.derived_digest != derivation.map(|derivation| derivation.derived_digest.clone())
        || artifact.artifact_id != artifact.expected_artifact_id()
    {
        return Err(SynapseStoreError::Decode(
            "stored serving artifact identity is inconsistent".to_string(),
        ));
    }
    Ok(())
}

fn validate_serving_certification(record: &CertificationRecord) -> Result<(), SynapseStoreError> {
    record.validate().map_err(|error| {
        SynapseStoreError::Decode(format!("invalid serving certification: {error}"))
    })?;
    if !is_digest(&record.unit.catalog_fingerprint) {
        return Err(SynapseStoreError::Decode(
            "serving certification catalog fingerprint must be a lowercase SHA-256 digest"
                .to_string(),
        ));
    }
    Ok(())
}

fn validate_certification_artifact_match(
    record: &CertificationRecord,
    artifact: &StoredServingArtifact,
) -> Result<(), SynapseStoreError> {
    validate_stored_serving_artifact(&artifact.artifact, artifact.derivation.as_ref())?;
    let lineage = &record.artifact_lineage;
    if lineage.artifact_id != artifact.artifact.artifact_id
        || lineage.model_id != artifact.artifact.model_id
        || lineage.quantization != "gguf-q4-k-m-compatible"
        || lineage.source_digest != artifact.artifact.source_digest
    {
        return Err(SynapseStoreError::Decode(
            "serving certification does not match the ingested GGUF source artifact".to_string(),
        ));
    }
    match (&lineage.derived, &artifact.derivation) {
        (None, None) => Ok(()),
        (Some(recorded), Some(ingested))
            if recorded.derivation_contract == ingested.derivation_contract
                && recorded.deterministic_inputs_digest == ingested.deterministic_inputs_digest
                && recorded.source_digest == artifact.artifact.source_digest
                && recorded.derived_digest == ingested.derived_digest
                && recorded.verified_derived_digest == ingested.verified_derived_digest =>
        {
            Ok(())
        }
        _ => Err(SynapseStoreError::Decode(
            "serving certification derivation lineage does not match the ingested artifact"
                .to_string(),
        )),
    }
}

fn validate_serving_approval(
    approval: &ServingApprovalRecord,
    verify_digest: bool,
) -> Result<(), SynapseStoreError> {
    if approval.schema_revision != SERVING_APPROVAL_SCHEMA_REVISION
        || !is_digest(&approval.catalog_fingerprint)
        || approval.certification_record_id.trim().is_empty()
        || !is_digest(&approval.artifact_id)
        || approval.approved_by.trim().is_empty()
    {
        return Err(SynapseStoreError::Decode(
            "serving approval has an invalid identity or provenance".to_string(),
        ));
    }
    match approval.state {
        ServingApprovalState::Enabled if approval.reason.is_none() => {}
        ServingApprovalState::Disabled | ServingApprovalState::Revoked
            if approval
                .reason
                .as_deref()
                .is_some_and(|reason| !reason.trim().is_empty()) => {}
        _ => {
            return Err(SynapseStoreError::Decode(
                "serving approval state and reason are inconsistent".to_string(),
            ))
        }
    }
    if verify_digest && approval.semantic_digest != approval.expected_digest() {
        return Err(SynapseStoreError::Decode(
            "serving approval semantic digest mismatch".to_string(),
        ));
    }
    Ok(())
}

fn serving_artifact_from_columns(
    artifact_json: String,
    derivation_contract: Option<String>,
    deterministic_inputs_digest: Option<String>,
    derived_digest: Option<String>,
    verified_derived_digest: Option<String>,
    gc_pinned: i64,
    json_index: usize,
) -> rusqlite::Result<StoredServingArtifact> {
    let artifact: ModelArtifactId = serde_json::from_str(&artifact_json).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(
            json_index,
            rusqlite::types::Type::Text,
            Box::new(error),
        )
    })?;
    let derivation = match (
        derivation_contract,
        deterministic_inputs_digest,
        derived_digest,
        verified_derived_digest,
    ) {
        (None, None, None, None) => None,
        (
            Some(derivation_contract),
            Some(deterministic_inputs_digest),
            Some(derived_digest),
            Some(verified_derived_digest),
        ) => Some(Q8IngestDerivation {
            derivation_contract,
            deterministic_inputs_digest,
            derived_quantization: "q8_0".to_string(),
            derived_digest,
            verified_derived_digest,
        }),
        _ => return Err(rusqlite::Error::InvalidQuery),
    };
    let stored = StoredServingArtifact {
        artifact,
        derivation,
        gc_pinned: match gc_pinned {
            0 => false,
            1 => true,
            _ => return Err(rusqlite::Error::InvalidQuery),
        },
    };
    validate_stored_serving_artifact(&stored.artifact, stored.derivation.as_ref())
        .map_err(to_sql_error)?;
    Ok(stored)
}

fn serving_artifact_from_row(row: &Row<'_>) -> rusqlite::Result<StoredServingArtifact> {
    serving_artifact_from_columns(
        row.get(0)?,
        row.get(1)?,
        row.get(2)?,
        row.get(3)?,
        row.get(4)?,
        row.get(5)?,
        0,
    )
}

fn load_serving_artifact_tx(
    tx: &rusqlite::Transaction<'_>,
    artifact_id: &str,
) -> rusqlite::Result<Option<StoredServingArtifact>> {
    tx.query_row(
        "SELECT artifact_json, derivation_contract, deterministic_inputs_digest,
                derived_digest, verified_derived_digest, gc_pinned
         FROM serving_artifacts WHERE artifact_id = ?1",
        params![artifact_id],
        serving_artifact_from_row,
    )
    .optional()
}

fn load_serving_artifact_conn(
    conn: &rusqlite::Connection,
    artifact_id: &str,
) -> rusqlite::Result<Option<StoredServingArtifact>> {
    conn.query_row(
        "SELECT artifact_json, derivation_contract, deterministic_inputs_digest,
                derived_digest, verified_derived_digest, gc_pinned
         FROM serving_artifacts WHERE artifact_id = ?1",
        params![artifact_id],
        serving_artifact_from_row,
    )
    .optional()
}

fn serving_certification_from_row(row: &Row<'_>) -> rusqlite::Result<StoredServingCertification> {
    let record_json: String = row.get(0)?;
    let record: CertificationRecord = serde_json::from_str(&record_json).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(error))
    })?;
    let artifact = serving_artifact_from_columns(
        row.get(2)?,
        row.get(3)?,
        row.get(4)?,
        row.get(5)?,
        row.get(6)?,
        row.get(7)?,
        2,
    )?;
    validate_serving_certification(&record).map_err(to_sql_error)?;
    validate_certification_artifact_match(&record, &artifact).map_err(to_sql_error)?;
    Ok(StoredServingCertification {
        artifact: artifact.artifact,
        record,
        recorded_at_ms: row.get::<_, i64>(1)? as u64,
    })
}

const SERVING_CERTIFICATION_SELECT_SQL: &str = "SELECT certification.record_json,
    certification.recorded_at_ms, artifact.artifact_json, artifact.derivation_contract,
    artifact.deterministic_inputs_digest, artifact.derived_digest,
    artifact.verified_derived_digest, artifact.gc_pinned
FROM serving_certification_records AS certification
JOIN serving_artifacts AS artifact ON artifact.artifact_id = certification.artifact_id";

fn load_serving_certification_tx(
    tx: &rusqlite::Transaction<'_>,
    certification_record_id: &str,
) -> rusqlite::Result<Option<StoredServingCertification>> {
    tx.query_row(
        &format!(
            "{SERVING_CERTIFICATION_SELECT_SQL}
             WHERE certification.certification_record_id = ?1"
        ),
        params![certification_record_id],
        serving_certification_from_row,
    )
    .optional()
}

fn load_serving_certification_conn(
    conn: &rusqlite::Connection,
    certification_record_id: &str,
) -> rusqlite::Result<Option<StoredServingCertification>> {
    conn.query_row(
        &format!(
            "{SERVING_CERTIFICATION_SELECT_SQL}
             WHERE certification.certification_record_id = ?1"
        ),
        params![certification_record_id],
        serving_certification_from_row,
    )
    .optional()
}

fn serving_approval_from_row(row: &Row<'_>) -> rusqlite::Result<ServingApprovalRecord> {
    let approval = ServingApprovalRecord {
        schema_revision: SERVING_APPROVAL_SCHEMA_REVISION.to_string(),
        catalog_fingerprint: row.get(0)?,
        certification_record_id: row.get(1)?,
        artifact_id: row.get(2)?,
        state: ServingApprovalState::parse(&row.get::<_, String>(3)?).map_err(to_sql_error)?,
        reason: row.get(4)?,
        approved_by: row.get(5)?,
        approved_at_ms: row.get::<_, i64>(6)? as u64,
        updated_at_ms: row.get::<_, i64>(7)? as u64,
        generation: row.get::<_, i64>(8)? as u64,
        semantic_digest: row.get(9)?,
    };
    validate_serving_approval(&approval, true).map_err(to_sql_error)?;
    Ok(approval)
}

const SERVING_APPROVAL_SELECT_SQL: &str = "SELECT catalog_fingerprint,
    certification_record_id, artifact_id, state, reason, approved_by, approved_at_ms,
    updated_at_ms, generation, semantic_digest FROM serving_approvals";

fn load_serving_approval_tx(
    tx: &rusqlite::Transaction<'_>,
    catalog_fingerprint: &str,
) -> rusqlite::Result<Option<ServingApprovalRecord>> {
    tx.query_row(
        &format!("{SERVING_APPROVAL_SELECT_SQL} WHERE catalog_fingerprint = ?1"),
        params![catalog_fingerprint],
        serving_approval_from_row,
    )
    .optional()
}

fn load_serving_approval_conn(
    conn: &rusqlite::Connection,
    catalog_fingerprint: &str,
) -> rusqlite::Result<Option<ServingApprovalRecord>> {
    conn.query_row(
        &format!("{SERVING_APPROVAL_SELECT_SQL} WHERE catalog_fingerprint = ?1"),
        params![catalog_fingerprint],
        serving_approval_from_row,
    )
    .optional()
}

fn serving_session_from_row(row: &Row<'_>) -> rusqlite::Result<ServingSessionRecord> {
    Ok(ServingSessionRecord {
        session_id: row.get(0)?,
        catalog_fingerprint: row.get(1)?,
        approval_generation: row.get::<_, i64>(2)? as u64,
        state: ServingSessionState::parse(&row.get::<_, String>(3)?).map_err(to_sql_error)?,
        committed_token_count: row.get::<_, i64>(4)? as u64,
        terminal_reason: row.get(5)?,
        created_at_ms: row.get::<_, i64>(6)? as u64,
        updated_at_ms: row.get::<_, i64>(7)? as u64,
    })
}

const SERVING_SESSION_SELECT_SQL: &str = "SELECT session_id, catalog_fingerprint,
    approval_generation, state, committed_token_count, terminal_reason, created_at_ms,
    updated_at_ms FROM serving_sessions";

fn load_serving_session_tx(
    tx: &rusqlite::Transaction<'_>,
    session_id: &str,
) -> rusqlite::Result<Option<ServingSessionRecord>> {
    tx.query_row(
        &format!("{SERVING_SESSION_SELECT_SQL} WHERE session_id = ?1"),
        params![session_id],
        serving_session_from_row,
    )
    .optional()
}

fn load_serving_session_conn(
    conn: &rusqlite::Connection,
    session_id: &str,
) -> rusqlite::Result<Option<ServingSessionRecord>> {
    conn.query_row(
        &format!("{SERVING_SESSION_SELECT_SQL} WHERE session_id = ?1"),
        params![session_id],
        serving_session_from_row,
    )
    .optional()
}

fn retained_serving_state_from_row(row: &Row<'_>) -> rusqlite::Result<RetainedServingState> {
    let valid = match row.get::<_, i64>(2)? {
        0 => false,
        1 => true,
        _ => return Err(rusqlite::Error::InvalidQuery),
    };
    let state = RetainedServingState {
        state_id: row.get(0)?,
        catalog_fingerprint: row.get(1)?,
        valid,
        invalidation_reason: row.get(3)?,
        created_at_ms: row.get::<_, i64>(4)? as u64,
        updated_at_ms: row.get::<_, i64>(5)? as u64,
    };
    if state.valid != state.invalidation_reason.is_none() {
        return Err(rusqlite::Error::InvalidQuery);
    }
    Ok(state)
}

const RETAINED_SERVING_STATE_SELECT_SQL: &str = "SELECT state_id, catalog_fingerprint,
    valid, invalidation_reason, created_at_ms, updated_at_ms FROM serving_retained_states";

fn load_retained_serving_state_tx(
    tx: &rusqlite::Transaction<'_>,
    state_id: &str,
) -> rusqlite::Result<Option<RetainedServingState>> {
    tx.query_row(
        &format!("{RETAINED_SERVING_STATE_SELECT_SQL} WHERE state_id = ?1"),
        params![state_id],
        retained_serving_state_from_row,
    )
    .optional()
}

fn load_retained_serving_state_conn(
    conn: &rusqlite::Connection,
    state_id: &str,
) -> rusqlite::Result<Option<RetainedServingState>> {
    conn.query_row(
        &format!("{RETAINED_SERVING_STATE_SELECT_SQL} WHERE state_id = ?1"),
        params![state_id],
        retained_serving_state_from_row,
    )
    .optional()
}

fn serving_approval_refusal_tx(
    tx: &rusqlite::Transaction<'_>,
    approval: &ServingApprovalRecord,
) -> rusqlite::Result<Option<ServingRefusal>> {
    match approval.state {
        ServingApprovalState::Disabled => return Ok(Some(ServingRefusal::ArtifactDisabled)),
        ServingApprovalState::Revoked => return Ok(Some(ServingRefusal::ArtifactRevoked)),
        ServingApprovalState::Enabled => {}
    }
    let Some(certification) = load_serving_certification_tx(tx, &approval.certification_record_id)?
    else {
        return Ok(Some(ServingRefusal::CertificationMismatch));
    };
    if certification.artifact.artifact_id != approval.artifact_id
        || certification.record.unit.catalog_fingerprint != approval.catalog_fingerprint
    {
        return Ok(Some(ServingRefusal::CertificationMismatch));
    }
    Ok(None)
}

fn retained_state_refusal(state: &RetainedServingState) -> ServingRefusal {
    match state.invalidation_reason.as_deref() {
        Some("artifact_disabled") => ServingRefusal::ArtifactDisabled,
        Some("artifact_revoked") => ServingRefusal::ArtifactRevoked,
        _ => ServingRefusal::RetainedStateInvalidated,
    }
}

fn active_serving_session_count_tx(
    tx: &rusqlite::Transaction<'_>,
    catalog_fingerprint: &str,
) -> rusqlite::Result<u64> {
    Ok(tx.query_row(
        "SELECT COUNT(*) FROM serving_sessions
         WHERE catalog_fingerprint = ?1 AND state IN ('active', 'termination_requested')",
        params![catalog_fingerprint],
        |row| row.get::<_, i64>(0),
    )? as u64)
}

fn serving_artifact_unload_ready_tx(
    tx: &rusqlite::Transaction<'_>,
    catalog_fingerprint: &str,
) -> rusqlite::Result<bool> {
    let approval = load_serving_approval_tx(tx, catalog_fingerprint)?
        .ok_or(rusqlite::Error::QueryReturnedNoRows)?;
    Ok(approval.state != ServingApprovalState::Enabled
        && active_serving_session_count_tx(tx, catalog_fingerprint)? == 0)
}

fn validate_session_id(value: &str) -> Result<(), SynapseStoreError> {
    if value.is_empty() || value.len() > 256 || value.bytes().any(|byte| byte.is_ascii_control()) {
        return Err(SynapseStoreError::Decode(
            "serving session and retained-state IDs must be non-empty printable strings"
                .to_string(),
        ));
    }
    Ok(())
}

fn owned_decode_match_inputs_equal(
    left: &OwnedDecodeMatchInputs,
    right: &OwnedDecodeMatchInputs,
) -> bool {
    left.revisioned_machine_profile_hash == right.revisioned_machine_profile_hash
        && left.profile_activation_epoch == right.profile_activation_epoch
        && left.model_id == right.model_id
        && left.decode_fingerprint == right.decode_fingerprint
        && left.processing_fingerprint == right.processing_fingerprint
        && left.runtime_config_digest == right.runtime_config_digest
        && left.constraint_runtime_identities == right.constraint_runtime_identities
        && left.worker_path_evidence == right.worker_path_evidence
        && left.evidence_schema_revision == right.evidence_schema_revision
        && left.g_dec_manifest_revision == right.g_dec_manifest_revision
}

// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
fn validate_owned_decode_cert_row(
    row: &OwnedDecodeCertificationRow,
) -> Result<(), SynapseStoreError> {
    if row.profile_activation_epoch == 0
        || !is_digest(&row.revisioned_machine_profile_hash)
        || !is_digest(&row.decode_fingerprint)
        || row.decode_fingerprint != row.fingerprint.0
        || !is_catalog_model_id(&row.model_id)
        || row.processing_fingerprint.trim().is_empty()
        || row.runtime_config_digest.trim().is_empty()
        || row.evidence_schema_revision != CERT_EVIDENCE_SCHEMA_REVISION
        || row.g_dec_manifest_revision != G_DEC_MANIFEST_REVISION
    {
        return Err(SynapseStoreError::Decode(
            "malformed measured owned-decode certification identity".to_string(),
        ));
    }
    let mut identities = row.constraint_runtime_identities.clone();
    identities.sort();
    identities.dedup();
    if identities != row.constraint_runtime_identities {
        return Err(SynapseStoreError::Decode(
            "constraint runtime identities must be sorted and unique".to_string(),
        ));
    }
    Ok(())
}

/// Whether certification evidence records the probe's gate as passed.
///
/// `evidence.gates` must hold exactly one entry: [`PROBE_GATE_ID`] with status
/// `passed` and the given manifest revision. Any other entry, including a
/// G-DEC id, means the evidence claims something the probe does not measure,
/// and a `g_dec` key is the v1 shape; both fail closed.
pub(crate) fn probe_gate_evidence_passed(evidence: &Value, manifest_revision: &str) -> bool {
    if evidence.get("g_dec").is_some() {
        return false;
    }
    let Some([entry]) = evidence
        .get("gates")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
    else {
        return false;
    };
    entry.get("id").and_then(Value::as_str) == Some(PROBE_GATE_ID)
        && entry.get("status").and_then(Value::as_str) == Some("passed")
        && entry.get("manifest_revision").and_then(Value::as_str) == Some(manifest_revision)
}

// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
fn decode_class_scoped_cert_row(
    row: RawClassScopedCertificationRow,
) -> Result<ClassScopedCertificationRow, SynapseStoreError> {
    Ok(ClassScopedCertificationRow {
        certification_class: CertificationClass::parse(&row.certification_class)?,
        assurance_class: AssuranceClass::parse(&row.assurance_class)?,
        status: CertificationStatus::parse(&row.status)?,
        key_hash: row.key_hash,
        machine_profile_hash: row.machine_profile_hash,
        remote_profile_hash: row.remote_profile_hash,
        identity_revision: row.identity_revision,
        numeric_profile_id: row.numeric_profile_id.map(NumericProfileId),
        fingerprint: Fingerprint(row.fingerprint),
        certified_at_ms: row.certified_at_ms,
        os_build: row.os_build,
        module_generation: row.module_generation,
        evidence: serde_json::from_str(&row.evidence_json)?,
    })
}

// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
fn decode_owned_decode_cert_row(
    row: RawOwnedDecodeCertificationRow,
) -> Result<OwnedDecodeCertificationRow, SynapseStoreError> {
    let mut identities: Vec<String> =
        serde_json::from_str(&row.constraint_runtime_identities_json)?;
    identities.sort();
    Ok(OwnedDecodeCertificationRow {
        status: CertificationStatus::parse(&row.status)?,
        revisioned_machine_profile_hash: row.revisioned_machine_profile_hash,
        profile_activation_epoch: row.profile_activation_epoch,
        model_id: row.model_id,
        decode_fingerprint: row.decode_fingerprint,
        processing_fingerprint: row.processing_fingerprint,
        runtime_config_digest: row.runtime_config_digest,
        constraint_runtime_identities: identities,
        worker_path_evidence: serde_json::from_str(&row.worker_path_evidence_json)?,
        evidence_schema_revision: row.evidence_schema_revision,
        g_dec_manifest_revision: row.g_dec_manifest_revision,
        numeric_profile_id: row.numeric_profile_id.map(NumericProfileId),
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

// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
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
    if state
        .certification_stale_since_ms
        .is_some_and(|value| value > i64::MAX as u64)
    {
        return Err(SynapseStoreError::ProfileStateCorrupt(
            "certification stale timestamp exceeds SQLite range".to_string(),
        ));
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

// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
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
    let certification_stale_since_value = row.get::<_, Option<i64>>(6)?;
    if epoch_value.is_some_and(|value| value < 0)
        || last_rotation_value.is_some_and(|value| value < 0)
        || certification_stale_since_value.is_some_and(|value| value < 0)
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
        certification_stale_since_ms: certification_stale_since_value.map(|value| value as u64),
    })
}

// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
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

// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
fn rotation_event_id(new_hash: &str, epoch: u64) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"rotation-ledger-v1\0");
    hasher.update(new_hash.as_bytes());
    hasher.update(epoch.to_be_bytes());
    hex::encode(hasher.finalize())
}

// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
fn rotation_outcome_from_row(row: &Row<'_>) -> rusqlite::Result<RotationCertificationOutcome> {
    Ok(RotationCertificationOutcome {
        event_id: row.get(0)?,
        model_id: row.get(1)?,
        decode_fingerprint: row.get(2)?,
        outcome_state: row.get(3)?,
        certified_at_ms: row.get::<_, Option<i64>>(4)?.map(|value| value as u64),
        failure_reason: row.get(5)?,
    })
}

// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
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
    let mut stmt = conn.prepare(
        "SELECT event_id, model_id, decode_fingerprint, outcome_state,
                certified_at_ms, failure_reason
         FROM profile_rotation_certification_outcomes
         WHERE event_id = ?1 ORDER BY model_id ASC, decode_fingerprint ASC",
    )?;
    let outcomes = stmt
        .query_map(params![&event_id], rotation_outcome_from_row)?
        .collect::<Result<Vec<_>, _>>()?;
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
        outcomes,
    }))
}

// Staged storage API consumed by the epic's runtime slice; remove this allow there.
#[allow(dead_code)]
fn rotation_event_tx(
    tx: &rusqlite::Transaction<'_>,
    event_id: &str,
) -> rusqlite::Result<RotationLedgerEvent> {
    rotation_event_conn(tx, event_id)?.ok_or(rusqlite::Error::QueryReturnedNoRows)
}

fn to_sql_error(error: SynapseStoreError) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(error))
}

fn upsert_owned_decode_cert_row_tx(
    tx: &rusqlite::Transaction<'_>,
    row: &OwnedDecodeCertificationRow,
) -> rusqlite::Result<usize> {
    let evidence_json = serde_json::to_string(&row.evidence)
        .map_err(SynapseStoreError::from)
        .map_err(to_sql_error)?;
    let identities_json = serde_json::to_string(&row.constraint_runtime_identities)
        .map_err(SynapseStoreError::from)
        .map_err(to_sql_error)?;
    let identities_digest =
        constraint_runtime_identities_digest(&row.constraint_runtime_identities)
            .map_err(to_sql_error)?;
    let worker_path_json = serde_json::to_string(&row.worker_path_evidence)
        .map_err(SynapseStoreError::from)
        .map_err(to_sql_error)?;
    tx.execute(
        "INSERT INTO cert_rows (
             certification_class, assurance_class, status, key_hash,
             revisioned_machine_profile_hash, profile_activation_epoch,
             model_id, decode_fingerprint, numeric_profile_id, fingerprint,
             certified_at_ms, os_build, module_generation,
             evidence_schema_revision, processing_fingerprint,
             runtime_config_digest, constraint_runtime_identities_json,
             constraint_runtime_identities_digest, worker_path_evidence_json,
             g_dec_manifest_revision, evidence_json
         ) VALUES (
             'measured_owned_decode', 'measured', ?1, ?2, ?2, ?3, ?4, ?5,
             ?6, ?5, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17
         )
         ON CONFLICT (
             revisioned_machine_profile_hash, profile_activation_epoch,
             model_id, decode_fingerprint, evidence_schema_revision,
             constraint_runtime_identities_digest
         ) WHERE certification_class = 'measured_owned_decode'
         DO UPDATE SET
             status = excluded.status,
             numeric_profile_id = excluded.numeric_profile_id,
             fingerprint = excluded.fingerprint,
             certified_at_ms = excluded.certified_at_ms,
             os_build = excluded.os_build,
             module_generation = excluded.module_generation,
             processing_fingerprint = excluded.processing_fingerprint,
             runtime_config_digest = excluded.runtime_config_digest,
             constraint_runtime_identities_json = excluded.constraint_runtime_identities_json,
             worker_path_evidence_json = excluded.worker_path_evidence_json,
             g_dec_manifest_revision = excluded.g_dec_manifest_revision,
             evidence_json = excluded.evidence_json",
        params![
            row.status.as_str(),
            &row.revisioned_machine_profile_hash,
            row.profile_activation_epoch as i64,
            &row.model_id,
            &row.decode_fingerprint,
            row.numeric_profile_id.as_ref().map(|id| &id.0),
            row.certified_at_ms as i64,
            &row.os_build,
            row.module_generation as i64,
            &row.evidence_schema_revision,
            &row.processing_fingerprint,
            &row.runtime_config_digest,
            identities_json,
            identities_digest,
            worker_path_json,
            &row.g_dec_manifest_revision,
            evidence_json,
        ],
    )
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
        params_json: decode_optional_json(row.get(18)?, 18)?,
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
pub(crate) fn configure_serving_catalog_for_test(store: &SynapseStore) -> String {
    tests::configure_serving_catalog(store).1
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::owned_decode_certification::{
        ArtifactLineage, CertificationGateResult, CertificationRecord, CertificationUnit,
        DerivedArtifactLineage, EmbedLoadResult, KvMatrixCandidate, KvMatrixResult,
        KvSelectionEvidence, M5MeasurementEvidence, MachineScopedEvidence, MachineTuple,
        MtpSpeedRepetition, MtpSpeedResult, PlatformEnvelopeResult, ProbeEvidence,
        RuntimeConfiguration, SerialOracleFidelityResult, SpeculativeSerialFidelityResult,
        SpeculativeTelemetry, TimingArmEvidence, TokenFidelityEvidence, TokenTapResult,
        AGENTIC_BATTERY_ID, EMBED_LOAD_ID, LLAMA_CPP_ORACLE_REVISION, PLATFORM_ENVELOPE_ID,
        WAVE_1_CONTEXT_CEILING_TOKENS,
    };
    use cortexkit_store_types::{Isolation, StorageBackend};
    use synapse_core::MachineProfile;

    /// Evidence of the shape a passed generate probe writes under the current
    /// schema, spelled out literally rather than taken from the probe's own
    /// builder so a change to either side shows up as a disagreement.
    fn passed_probe_gate_evidence() -> serde_json::Value {
        serde_json::json!({
            "gates": [{
                "id": "structural_band",
                "status": "passed",
                "manifest_revision": "owned-decode-probe-gates-v1",
            }],
        })
    }

    fn owned_seed_model_config(model_id: &str) -> StoredModelConfig {
        let (family, quant, artifact_digest, derived_digest) = match model_id {
            "qwen3-0.6b-decode-f16" => (
                "qwen3-0.6b",
                "f16",
                "sha256:0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
                None,
            ),
            "lfm2-1.2b-decode-f16" => (
                "lfm2-1.2b",
                "f16",
                "sha256:60fef6ef4481c533ce7427793bed50200b55b3c68d0d00c52bc56f207a9acecd",
                None,
            ),
            "qwen3-0.6b-decode-q8_0" => (
                "qwen3-0.6b",
                "q8_0",
                "sha256:0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
                Some("17d2fbfeff90269190287f324ed93bab3bb1b4fa4aad98c3fbba1868c01cb0f2"),
            ),
            "lfm2-1.2b-decode-q8_0" => (
                "lfm2-1.2b",
                "q8_0",
                "sha256:60fef6ef4481c533ce7427793bed50200b55b3c68d0d00c52bc56f207a9acecd",
                Some("5874faabdce2567dcc0e7339e9547d79421ba312c71e3442c9cc3c4ed3cb47d0"),
            ),
            other => panic!("unknown migration seed model {other}"),
        };
        let mut build_flags = BTreeMap::new();
        if let Some(derived_digest) = derived_digest {
            build_flags.insert("quantizer_revision".to_string(), "q8-ingest-v1".to_string());
            build_flags.insert("derived_digest".to_string(), derived_digest.to_string());
        }
        StoredModelConfig {
            model_id: model_id.to_string(),
            engine: "owned-metal-decode".to_string(),
            task: "generate".to_string(),
            artifact_digest: artifact_digest.to_string(),
            artifact_format: "owned-safetensors".to_string(),
            tokenizer_sanitized_digest: "sha256:test-tokenizer".to_string(),
            model_locator: ModelAssetLocator::CacheDigest {
                digest: artifact_digest.to_string(),
            },
            tokenizer_locator: ModelAssetLocator::CacheDigest {
                digest: "sha256:test-tokenizer".to_string(),
            },
            model_source_url: "file:///model".to_string(),
            tokenizer_source_url: "file:///tokenizer".to_string(),
            pooling: "none".to_string(),
            normalize: false,
            max_tokens: 512,
            quant: quant.to_string(),
            pin: true,
            owned_family: Some(family.to_string()),
            owned_dtype: Some("f16".to_string()),
            owned_execution: Some("supervised".to_string()),
            owned_attention_units: None,
            config_locator: None,
            extra_locators: Vec::new(),
            engine_identity: EngineIdentity {
                engine: "owned-metal-decode".to_string(),
                version: "test".to_string(),
                build_flags,
            },
            numeric_profile_id: NumericProfileId(format!("numeric-{model_id}")),
            fingerprint: Fingerprint(format!("catalog-{model_id}")),
            worker_bin: None,
            worker_runtime_dir: None,
        }
    }

    fn serving_catalog_fingerprint() -> String {
        "d".repeat(64)
    }

    fn serving_artifact_request() -> ArtifactIngestRequest {
        ArtifactIngestRequest {
            model_id: "qwen3.8-27b".to_string(),
            source_format: "gguf".to_string(),
            source_quantization: "q4_k_m".to_string(),
            source_digest: "a".repeat(64),
            q8_derivation: Some(Q8IngestDerivation {
                derivation_contract: Q8_INGEST_DERIVATION_CONTRACT.to_string(),
                deterministic_inputs_digest: "b".repeat(64),
                derived_quantization: "q8_0".to_string(),
                derived_digest: "c".repeat(64),
                verified_derived_digest: "c".repeat(64),
            }),
        }
    }

    fn complete_serving_certification(
        artifact: &ModelArtifactId,
        catalog_fingerprint: &str,
    ) -> CertificationRecord {
        let machine = MachineTuple {
            machine_profile_hash: "machine-profile".to_string(),
            macos_build: "25F84".to_string(),
            unified_memory_bytes: 128 * 1024 * 1024 * 1024,
        };
        let runtime = RuntimeConfiguration {
            runtime_config_digest: "runtime-config".to_string(),
            runtime_revision: "runtime-v1".to_string(),
        };
        let unit = CertificationUnit {
            base_artifact_id: artifact.artifact_id.clone(),
            native_mtp_head_digest: "native-mtp-head".to_string(),
            depth_controller_gate_digest: "depth-controller-gate".to_string(),
            catalog_fingerprint: catalog_fingerprint.to_string(),
        };
        let timing_arm = |session: &str, tokens_per_second| TimingArmEvidence {
            loaded_session_id: session.to_string(),
            machine_profile_hash: machine.machine_profile_hash.clone(),
            macos_build: machine.macos_build.clone(),
            ac_power_connected: true,
            one_minute_load_average: 1.0,
            mean_tokens_per_second: tokens_per_second,
        };
        let candidates = [256, 512, 1024]
            .into_iter()
            .flat_map(|block_size_tokens| {
                [4096, 8192, 16384]
                    .into_iter()
                    .map(move |reused_prefix_bucket_tokens| KvMatrixCandidate {
                        block_size_tokens,
                        reused_prefix_bucket_tokens,
                        alignment_valid: true,
                        retained_memory_overhead_percent: 10.0,
                        warm_ttft_ms: if (block_size_tokens, reused_prefix_bucket_tokens)
                            == (1024, 4096)
                        {
                            1.0
                        } else {
                            2.0
                        },
                    })
            })
            .collect::<Vec<_>>();
        let selected = candidates
            .iter()
            .find(|candidate| {
                (
                    candidate.block_size_tokens,
                    candidate.reused_prefix_bucket_tokens,
                ) == (1024, 4096)
            })
            .cloned()
            .expect("fixture includes the selected KV cell");
        let manifest_digest = "agentic-manifest-digest".to_string();
        CertificationRecord {
            record_id: "serving-certification-record".to_string(),
            manifest_digest: manifest_digest.clone(),
            artifact_lineage: ArtifactLineage {
                artifact_id: artifact.artifact_id.clone(),
                model_id: artifact.model_id.clone(),
                quantization: "gguf-q4-k-m-compatible".to_string(),
                source_digest: artifact.source_digest.clone(),
                derived: Some(DerivedArtifactLineage {
                    derivation_contract: Q8_INGEST_DERIVATION_CONTRACT.to_string(),
                    deterministic_inputs_digest: "b".repeat(64),
                    source_digest: artifact.source_digest.clone(),
                    derived_digest: artifact
                        .derived_digest
                        .clone()
                        .expect("fixture is Q8 derived"),
                    verified_derived_digest: artifact
                        .derived_digest
                        .clone()
                        .expect("fixture is Q8 derived"),
                }),
            },
            unit: unit.clone(),
            machine_evidence: MachineScopedEvidence {
                machine: machine.clone(),
                probe: ProbeEvidence {
                    probe_id: "certification-probe".to_string(),
                    harness_revision: "harness-r1".to_string(),
                    observed_at_ms: 1,
                },
                runtime: runtime.clone(),
                m5_measurement: M5MeasurementEvidence {
                    measurement_id: "m5-native-head-cost".to_string(),
                    measurement_revision: "m5-r1".to_string(),
                    machine_profile_hash: machine.machine_profile_hash.clone(),
                    base_artifact_id: unit.base_artifact_id.clone(),
                    catalog_fingerprint: unit.catalog_fingerprint.clone(),
                    native_mtp_head_digest: unit.native_mtp_head_digest.clone(),
                    runtime_config_digest: runtime.runtime_config_digest.clone(),
                    head_forward_ms: 1.0,
                    backbone_step_ms: 4.0,
                    controller_constants_digest: "controller-constants".to_string(),
                    registered: true,
                    depth_zero_executes_no_head_work: true,
                    positive_depth_chains_command_buffer: true,
                },
            },
            gate_results: vec![
                CertificationGateResult::SerialOracleFidelity(SerialOracleFidelityResult {
                    manifest_digest: manifest_digest.clone(),
                    battery_id: AGENTIC_BATTERY_ID.to_string(),
                    oracle_revision: LLAMA_CPP_ORACLE_REVISION.to_string(),
                    greedy_only: true,
                    fidelity: TokenFidelityEvidence {
                        generated_token_ids_match: true,
                        stop_position_matches: true,
                        finish_reason_matches: true,
                    },
                }),
                CertificationGateResult::SpeculativeSerialFidelity(
                    SpeculativeSerialFidelityResult {
                        manifest_digest: manifest_digest.clone(),
                        battery_id: AGENTIC_BATTERY_ID.to_string(),
                        serial_certification_id: "serial-certification".to_string(),
                        leviathan_greedy_acceptance: true,
                        fidelity: TokenFidelityEvidence {
                            generated_token_ids_match: true,
                            stop_position_matches: true,
                            finish_reason_matches: true,
                        },
                    },
                ),
                CertificationGateResult::MtpSpeed(MtpSpeedResult {
                    manifest_digest: manifest_digest.clone(),
                    battery_id: AGENTIC_BATTERY_ID.to_string(),
                    generated_token_window: 256,
                    serial_warmup_last_three_tok_s: [10.0, 10.5, 10.2],
                    mtp_warmup_last_three_tok_s: [16.0, 16.5, 16.2],
                    repetitions: vec![
                        MtpSpeedRepetition {
                            serial: timing_arm("session-1", 10.0),
                            mtp: timing_arm("session-1", 16.0),
                        },
                        MtpSpeedRepetition {
                            serial: timing_arm("session-2", 10.5),
                            mtp: timing_arm("session-2", 16.5),
                        },
                        MtpSpeedRepetition {
                            serial: timing_arm("session-3", 10.2),
                            mtp: timing_arm("session-3", 16.2),
                        },
                    ],
                    telemetry: SpeculativeTelemetry {
                        proposed_depth: 4,
                        accepted_depth: 3,
                        acceptance_rate: 0.75,
                        verification_work: 42,
                        controller_decisions_digest: "controller-decisions".to_string(),
                    },
                }),
                CertificationGateResult::KvMatrix(KvMatrixResult {
                    manifest_digest: manifest_digest.clone(),
                    machine_profile_hash: machine.machine_profile_hash.clone(),
                    candidates,
                    selection: KvSelectionEvidence {
                        selected,
                        continuation_token_ids_identical: true,
                        reused_token_count: 4096,
                        reused_block_count: 4,
                        prefill_dispatches_over_reused_range: 0,
                        cold_ttft_ms: 10.0,
                        warm_ttft_ms: 1.0,
                        close_restored_allocator_accounting: true,
                    },
                }),
                CertificationGateResult::PlatformEnvelope(PlatformEnvelopeResult {
                    manifest_digest: manifest_digest.clone(),
                    envelope_id: PLATFORM_ENVELOPE_ID.to_string(),
                    machine_profile_hash: machine.machine_profile_hash.clone(),
                    macos_build: machine.macos_build.clone(),
                    unified_memory_bytes: machine.unified_memory_bytes,
                    reserved_embed_rerank_bytes: 1,
                    artifact_weight_bytes: 1,
                    kv_bytes_per_token: 1,
                    mandatory_context_ceiling_tokens: WAVE_1_CONTEXT_CEILING_TOKENS,
                    admitted_and_exercised_32k_session: true,
                    exercised_reservation_accounting: true,
                    exercised_kv_reuse: true,
                    exercised_streaming: true,
                    exercised_scheduler_interleaving: true,
                }),
                CertificationGateResult::EmbedLoad(EmbedLoadResult {
                    manifest_digest: manifest_digest.clone(),
                    workload_id: EMBED_LOAD_ID.to_string(),
                    runtime_config_digest: runtime.runtime_config_digest.clone(),
                    concurrent_clients: 8,
                    poisson_aggregate_rate_per_second: 5.0,
                    duration_seconds: 120,
                    warmup_seconds: 10,
                    completed_samples: 500,
                    failed_embeddings: 0,
                    timed_out_embeddings: 0,
                    nearest_rank_p95_ms: 150.0,
                    active_decode_context_ceiling_tokens: WAVE_1_CONTEXT_CEILING_TOKENS,
                    used_shipped_scheduler_configuration: true,
                }),
                CertificationGateResult::TokenTap(TokenTapResult {
                    manifest_digest,
                    observed_after_acceptance_before_emission: true,
                    read_only: true,
                    token_ids_identical_when_enabled: true,
                    stop_position_identical_when_enabled: true,
                    finish_reason_identical_when_enabled: true,
                    emitted_bytes_identical_when_enabled: true,
                }),
            ],
        }
    }

    pub(super) fn configure_serving_catalog(store: &SynapseStore) -> (ModelArtifactId, String) {
        let artifact = store
            .ingest_serving_artifact(&serving_artifact_request(), 1)
            .expect("valid GGUF source ingests");
        let catalog_fingerprint = serving_catalog_fingerprint();
        let record = complete_serving_certification(&artifact, &catalog_fingerprint);
        store
            .store_serving_certification(&record, 2)
            .expect("complete record matches the ingested artifact");
        store
            .approve_serving_catalog(
                &catalog_fingerprint,
                &record.record_id,
                "principal:operator",
                3,
            )
            .expect("matching record approves the catalog fingerprint");
        (artifact, catalog_fingerprint)
    }

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
    fn newest_schema_version_is_the_highest_migration_this_binary_carries() {
        // Asserts the PROPERTY the derivation must satisfy, not the derivation
        // itself. Restating `MIGRATIONS.iter().map(..).max()` here would make
        // this agree with the code by construction and pass whatever the
        // function did; the manifest-side check can only see shape, so a
        // regressed derivation that still returns a positive number is
        // invisible there. These two bounds pin it from both sides: `.min()`
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
    fn schema_trigger_set_is_exactly_the_known_guards() {
        // The migration fence above can only exercise guards that exist when it
        // runs. Rows announce themselves (fixtures start failing without them);
        // an armed guard announces nothing until something violates it, and a
        // REMOVED guard announces nothing at all - queries just stop being
        // refused. So the trigger set is pinned by name in both directions.
        // If this test fails because you ADDED a trigger: arm it in the
        // populated-store fence (write rows that exercise it mid-history),
        // then add its name here. If it fails because one is MISSING: a
        // guard the fence certifies against was dropped - that is a schema
        // change, not a test to appease.
        let (root, descriptor) = temp_descriptor("schema-trigger-set");
        let store = SynapseStore::open(&descriptor).unwrap();
        let mut triggers = store
            .store
            .with_conn(|conn| {
                let mut stmt = conn.prepare(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger' ORDER BY name",
                )?;
                let names = stmt
                    .query_map([], |row| row.get::<_, String>(0))?
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(names)
            })
            .unwrap();
        triggers.sort();
        assert_eq!(
            triggers,
            vec![
                "catalog_member_verification_cleanup".to_string(),
                "remote_checkpoint_immutable".to_string(),
                "remote_url_binding_identity_immutable".to_string(),
            ],
        );
        drop(store);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn last_migration_succeeds_for_a_populated_store() {
        // Each version is applied to the same on-disk store before its writer paths
        // populate the rows that version can represent. Historical jobs/certs cannot
        // be written by today's newer APIs until their coupled schema exists, so the
        // v1-v5 job and v1-v8 cert arms remain empty; the v6 job writer and v9 cert
        // writer populate them at the first compatible coupling. The approval writer
        // also creates the marker and rotation writer creates parent/child ledger rows.
        // Migration 6 creates both SQL triggers (remote_checkpoint_immutable and
        // remote_url_binding_identity_immutable) unconditionally, so they need no
        // separate armed-state row. Module generation/table epoch, the approval
        // marker, and profile state are conditionally meaningful guard state; their
        // real writers run in v1/v3/v9. The corruption-event table is intentionally
        // empty: its only writer requires
        // first corrupting a stored digest, which is reserved for mutation validation.
        // Mutation proof (performed during development, not shipped): append this final
        // data-moving migration: `CREATE TABLE moved_alias_rows_mutant (
        // fingerprint_a TEXT NOT NULL, fingerprint_b TEXT NOT NULL,
        // migrated_at_ms INTEGER NOT NULL); INSERT INTO moved_alias_rows_mutant
        // (fingerprint_a, fingerprint_b) SELECT fingerprint_a, fingerprint_b
        // FROM alias_rows;` It compiled and ran, then reddened this populated test on
        // the source rows while existing empty-store tests stayed green. A unique-index
        // mutant was not used because it cannot distinguish a fixture failure from a
        // migration failure as clearly.
        let model_ids = [
            "qwen3-0.6b-decode-f16",
            "lfm2-1.2b-decode-f16",
            "qwen3-0.6b-decode-q8_0",
            "lfm2-1.2b-decode-q8_0",
        ];
        let profile_before = MachineProfile {
            os_build: "23G93".to_string(),
            arch: "aarch64".to_string(),
            chip_model: "Apple M5 Max".to_string(),
            ram_class: "le_64_gib".to_string(),
            ane_subtype: Some("h17(map)".to_string()),
            engine_identities: Vec::new(),
        };
        let mut profile_after = profile_before.clone();
        profile_after.os_build = "23G94".to_string();
        let decode_fingerprint = "8bcb6dfc1bd55a38a56b5a6931132f428687e064104d106c2cc5efb898f80feb";
        let (root, descriptor) = temp_descriptor("migration-populated-step-through");
        {
            let mut legacy = open_sqlite(&descriptor).unwrap();
            let populate_version = |version: u32, store: SqliteStore| -> SqliteStore {
                match version {
                    1 => {
                        let store = SynapseStore { store };
                        store.next_module_generation().unwrap();
                        store.store
                    }
                    2 => store,
                    3 => {
                        let store = SynapseStore { store };
                        let alias_a = Fingerprint("alias-a".to_string());
                        store
                            .declare_alias_pair(
                                &alias_a,
                                &Fingerprint("alias-b1".to_string()),
                                &serde_json::json!({"source": "fixture"}),
                                1_001,
                            )
                            .unwrap();
                        store
                            .declare_alias_pair(
                                &alias_a,
                                &Fingerprint("alias-b2".to_string()),
                                &serde_json::json!({"source": "fixture"}),
                                1_002,
                            )
                            .unwrap();
                        store.store
                    }
                    4 => {
                        let store = SynapseStore { store };
                        store
                            .upsert_model(&owned_seed_model_config(model_ids[0]), 1_003)
                            .unwrap();
                        store.store
                    }
                    5 => {
                        let store = SynapseStore { store };
                        store
                            .store_perf_row(&PerfRow {
                                machine_profile_hash: "machine-profile-hash".to_string(),
                                model_id: model_ids[0].to_string(),
                                workload: "embed.batch".to_string(),
                                numeric_profile_id: NumericProfileId("numeric-qwen3".to_string()),
                                fingerprint: Fingerprint("perf-fingerprint".to_string()),
                                engine: "owned-metal-decode".to_string(),
                                measured_at_ms: 1_004,
                                os_build: "23G93".to_string(),
                                module_generation: 1,
                                throughput_tok_s: 42.5,
                                cold_load_ms: 125.0,
                                single_item_latency_p50_ms: 18.0,
                                details: serde_json::json!({"batch_size": 1}),
                            })
                            .unwrap();
                        store
                            .replace_knob_assignments(
                                "machine-profile-hash",
                                &[KnobAssignmentRow {
                                    machine_profile_hash: "machine-profile-hash".to_string(),
                                    workload: "embed.batch".to_string(),
                                    knob: PerfKnob::Balanced,
                                    model_id: model_ids[0].to_string(),
                                    numeric_profile_id: NumericProfileId(
                                        "numeric-qwen3".to_string(),
                                    ),
                                    fingerprint: Fingerprint("perf-fingerprint".to_string()),
                                    engine: "owned-metal-decode".to_string(),
                                    measured_at_ms: 1_004,
                                    os_build: "23G93".to_string(),
                                    module_generation: 1,
                                    throughput_tok_s: 42.5,
                                    single_item_latency_p50_ms: 18.0,
                                }],
                            )
                            .unwrap();
                        store.store
                    }
                    6 => {
                        let store = SynapseStore { store };
                        store
                            .bind_remote_profile_url(
                                "remote-profile-hash",
                                "https://remote.example.test/v1",
                            )
                            .unwrap();
                        let job_id = store
                            .admit_job(
                                "step-key",
                                "legacy:step-job",
                                "embed.batch",
                                1,
                                Some("cache-lease:remote-profile"),
                                &serde_json::json!({"items": ["item-1"]}),
                                1_005,
                                60_000,
                                86_400_000,
                            )
                            .unwrap()
                            .record()
                            .job_id
                            .clone();
                        assert!(job_id.starts_with("job_"));
                        assert!(store.mark_job_running(&job_id, 1, 1_006).unwrap());
                        store
                            .commit_job_page(
                                &job_id,
                                0,
                                br#"{"items":[{"id":"item-1","value":1}]}"#,
                                &[CheckpointItem {
                                    item_id: "item-1".to_string(),
                                    result: br#"{"id":"item-1","value":1}"#.to_vec(),
                                    provider_request_id: Some("provider-request-1".to_string()),
                                }],
                                1_007,
                            )
                            .unwrap();
                        store.store
                    }
                    7 | 8 => store,
                    9 => {
                        let store = SynapseStore { store };
                        let profile_hash = profile_before.revisioned_hash();
                        store
                            .store_cert_row(&CertificationRow {
                                assurance_class: AssuranceClass::Measured,
                                status: CertificationStatus::Uncertified,
                                key: CertificationKey::Measured {
                                    machine_profile_hash: profile_hash.clone(),
                                },
                                numeric_profile_id: NumericProfileId("numeric-qwen3".to_string()),
                                fingerprint: Fingerprint("new-cert-fingerprint".to_string()),
                                certified_at_ms: 1_008,
                                os_build: profile_before.os_build.clone(),
                                module_generation: 1,
                                evidence: serde_json::json!({"source": "fixture"}),
                            })
                            .unwrap();
                        for model_id in model_ids {
                            store
                                .upsert_model(&owned_seed_model_config(model_id), 1_009)
                                .unwrap();
                        }
                        let migrated = store
                            .migrate_owned_decode_approvals(
                                "owned-decode-approval-migration-v1",
                                APPROVAL_SCHEMA_REVISION,
                            )
                            .unwrap();
                        assert_eq!(migrated.outcome, "applied");
                        assert_eq!(migrated.rows, 4);
                        // This fixture populates migration v9 before the v12
                        // timestamp column exists. Write the v9 profile rows directly
                        // so the fence continues to exercise the historical schema.
                        let profile_before_hash = profile_before.revisioned_hash();
                        let profile_after_hash = profile_after.revisioned_hash();
                        let event_id = rotation_event_id(&profile_after_hash, 2);
                        let before_snapshot = serde_json::to_string(&profile_before).unwrap();
                        let after_snapshot = serde_json::to_string(&profile_after).unwrap();
                        store
                            .store
                            .with_conn_fenced(|tx| {
                                tx.execute(
                                    "UPDATE profile_state SET snapshot_json = ?1,
                                         revisioned_machine_profile_hash = ?2,
                                         profile_activation_epoch = 2,
                                         previous_revisioned_machine_profile_hash = ?3,
                                         last_rotation_reason = 'os_build',
                                         last_rotation_at_ms = 1011
                                     WHERE id = 0",
                                    params![
                                        after_snapshot,
                                        profile_after_hash,
                                        profile_before_hash
                                    ],
                                )?;
                                tx.execute(
                                    "INSERT INTO profile_rotation_events (
                                         event_id, old_revisioned_machine_profile_hash,
                                         new_revisioned_machine_profile_hash,
                                         old_profile_activation_epoch,
                                         new_profile_activation_epoch, changed_fields_json,
                                         previous_snapshot_json, current_snapshot_json,
                                         observed_at_ms, module_generation, created_at_ms
                                     ) VALUES (?1, ?2, ?3, 1, 2, ?4, ?5, ?6, 1011, 1, 1011)",
                                    params![
                                        event_id,
                                        profile_before_hash,
                                        profile_after_hash,
                                        serde_json::to_string(&["os_build"]).unwrap(),
                                        before_snapshot,
                                        after_snapshot,
                                    ],
                                )?;
                                let mut stmt = tx.prepare(
                                    "SELECT model_id, decode_fingerprint, enabled FROM approvals",
                                )?;
                                let approvals = stmt
                                    .query_map([], |row| {
                                        Ok((
                                            row.get::<_, String>(0)?,
                                            row.get::<_, String>(1)?,
                                            row.get::<_, i64>(2)? != 0,
                                        ))
                                    })?
                                    .collect::<Result<Vec<_>, _>>()?;
                                drop(stmt);
                                for (model_id, decode_fingerprint, enabled) in approvals {
                                    tx.execute(
                                        "INSERT INTO profile_rotation_certification_outcomes (
                                             event_id, model_id, decode_fingerprint, outcome_state
                                         ) VALUES (?1, ?2, ?3, ?4)",
                                        params![
                                            event_id,
                                            model_id,
                                            decode_fingerprint,
                                            if enabled { "required" } else { "not_required" },
                                        ],
                                    )?;
                                }
                                Ok(())
                            })
                            .unwrap();

                        let row = OwnedDecodeCertificationRow {
                            status: CertificationStatus::Certified,
                            revisioned_machine_profile_hash: profile_before.revisioned_hash(),
                            profile_activation_epoch: 1,
                            model_id: model_ids[0].to_string(),
                            decode_fingerprint: decode_fingerprint.to_string(),
                            processing_fingerprint: "processing-fingerprint".to_string(),
                            runtime_config_digest: "runtime-config-digest".to_string(),
                            constraint_runtime_identities: Vec::new(),
                            worker_path_evidence: serde_json::json!({}),
                            evidence_schema_revision: CERT_EVIDENCE_SCHEMA_REVISION.to_string(),
                            g_dec_manifest_revision: G_DEC_MANIFEST_REVISION.to_string(),
                            numeric_profile_id: Some(NumericProfileId("numeric-qwen3".to_string())),
                            fingerprint: Fingerprint(decode_fingerprint.to_string()),
                            certified_at_ms: 1_012,
                            os_build: profile_before.os_build.clone(),
                            module_generation: 1,
                            evidence: serde_json::json!({"source": "fixture"}),
                        };
                        let mut terminal = OwnedDecodeMatchInputs {
                            revisioned_machine_profile_hash: profile_before.revisioned_hash(),
                            profile_activation_epoch: 1,
                            model_id: model_ids[0].to_string(),
                            decode_fingerprint: decode_fingerprint.to_string(),
                            processing_fingerprint: "processing-fingerprint".to_string(),
                            runtime_config_digest: "runtime-config-digest".to_string(),
                            constraint_runtime_identities: Vec::new(),
                            worker_path_evidence: serde_json::json!({}),
                            evidence_schema_revision: CERT_EVIDENCE_SCHEMA_REVISION.to_string(),
                            g_dec_manifest_revision: G_DEC_MANIFEST_REVISION.to_string(),
                        };
                        terminal.model_id = model_ids[1].to_string();
                        let snapshot = OwnedDecodeMatchInputs {
                            model_id: model_ids[0].to_string(),
                            ..terminal.clone()
                        };
                        assert_eq!(
                            store
                                .store_owned_decode_cert_row_if_current(
                                    &snapshot, &terminal, &row, 1_013,
                                )
                                .unwrap(),
                            ProbeWriteOutcome::ProbeStale
                        );
                        store.store
                    }
                    10..=14 => store,
                    _ => panic!("no fixture rows for migration {version}"),
                }
            };

            for index in 0..MIGRATIONS.len() {
                legacy.migrate(NAMESPACE, &MIGRATIONS[..=index]).unwrap();
                legacy = populate_version(MIGRATIONS[index].version, legacy);
                // Mid-history check for the one table two arms share: models
                // is written at v4 (one row) and again at v9 (all four), so
                // the final sweep alone would stay green if the v4 arm went
                // quiet - v9's rows mask it, and migrations 5-8 would land on
                // an empty catalog untested. The claim's granularity must
                // match the writer's: assert at the version boundary where
                // masking begins.
                if MIGRATIONS[index].version == 4 {
                    let store = SynapseStore { store: legacy };
                    assert_eq!(
                        store.catalog_models().unwrap().len(),
                        1,
                        "the v4 populate arm stopped writing models; migrations \
                         5-8 would land on an empty catalog"
                    );
                    legacy = store.store;
                }
            }
        }

        let migrated = SynapseStore::open(&descriptor).unwrap();
        assert_eq!(migrated.catalog_models().unwrap().len(), 4);
        assert_eq!(migrated.alias_table().unwrap().rows.len(), 2);
        assert_eq!(migrated.approvals().unwrap().len(), 4);
        assert_eq!(migrated.rotation_events().unwrap().len(), 1);
        assert_eq!(
            migrated
                .current_perf_rows("machine-profile-hash")
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            migrated
                .knob_assignments("machine-profile-hash")
                .unwrap()
                .len(),
            1
        );
        assert_eq!(migrated.checkpoint_count("legacy:step-job").unwrap(), 1);
        let resumed_job_id = migrated
            .store
            .with_conn(|conn| {
                conn.query_row(
                    "SELECT job_id FROM jobs WHERE request_key = 'step-key'
                     ORDER BY created_ms DESC, rowid DESC LIMIT 1",
                    [],
                    |row| row.get::<_, String>(0),
                )
            })
            .unwrap();
        assert_eq!(
            migrated
                .get_job_page(&resumed_job_id, 0)
                .unwrap()
                .as_deref(),
            Some(br#"{"items":[{"id":"item-1","value":1}]}"#.as_slice())
        );
        let rotation_events = migrated.rotation_events().unwrap();
        assert_eq!(rotation_events.len(), 1);
        assert_eq!(
            migrated
                .rotation_outcomes(&rotation_events[0].event_id)
                .unwrap()
                .len(),
            4
        );
        assert!(
            migrated
                .store
                .with_conn(|conn| {
                    conn.query_row(
                        "SELECT COUNT(*) FROM approval_digest_corruption_events",
                        [],
                        |row| row.get::<_, i64>(0),
                    )
                })
                .unwrap()
                == 0
        );
        // Per-table survival sweep. The targeted assertions above check row
        // CONTENT for the interesting tables; this sweep checks that every
        // table the populate arms wrote still carries rows at all, named per
        // table so a quietly hollowed populate arm (a writer refactored into
        // a no-op, an arm emptied in a merge) fails by name instead of
        // leaving the fence green while it certifies against empty tables.
        // The unknown-table panic is the same self-maintaining shape as the
        // populate helper's version panic: a future migration that adds a
        // table must declare its expected population here or the sweep is a
        // red test, not a silently narrower fence.
        let tables = migrated
            .store
            .with_conn(|conn| {
                let mut stmt = conn.prepare(
                    "SELECT name FROM sqlite_master WHERE type = 'table' \
                     AND name NOT LIKE 'sqlite_%' ORDER BY name",
                )?;
                let names = stmt
                    .query_map([], |row| row.get::<_, String>(0))?
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(names)
            })
            .unwrap();
        for table in &tables {
            let expected_min: i64 = match table.as_str() {
                "module_meta" => 1,
                "jobs" => 1,
                "result_pages" => 1,
                "remote_checkpoints" => 1,
                "remote_url_bindings" => 1,
                "alias_rows" => 2,
                "models" => 4,
                "cert_rows" => 1,
                "perf_rows" => 1,
                "knob_assignments" => 1,
                "owned_decode_cert_rows" => 1,
                "approvals" => 4,
                "approval_migration_markers" => 1,
                "profile_state" => 1,
                "profile_rotation_events" => 1,
                "profile_rotation_certification_outcomes" => 4,
                "cert_row_rebuild_events" => 1,
                // The store framework's own bookkeeping (fence + namespace
                // version rows), written by migrate() itself - populated by
                // construction, not by an arm.
                "cortexkit_fence" => 1,
                "cortexkit_schema_version" => 1,
                // Deliberately empty: its only legitimate writer requires a
                // corrupted digest first; asserted exactly-zero above.
                "approval_digest_corruption_events" => 0,
                // These tables start empty because v11 introduces new serving
                // state without translating an older representation. Their
                // transactional writer paths are exercised by dedicated tests.
                "serving_artifacts" => 0,
                "serving_certification_records" => 0,
                "serving_approvals" => 0,
                "serving_sessions" => 0,
                "serving_retained_states" => 0,
                // v13 adds catalog install, download and self-check state with
                // no older representation to translate, so these start empty;
                // their transactions are exercised by the catalog store tests.
                // The run-sequence counter is seeded with its single row.
                "catalog_installs" => 0,
                "catalog_install_members" => 0,
                // v14 verification rows are derived only after hashing installed files.
                "catalog_file_verifications" => 0,
                "download_acquisitions" => 0,
                "download_key_bindings" => 0,
                "catalog_self_checks" => 0,
                "self_check_run_seq" => 1,
                other => panic!(
                    "table {other} has no declared population expectation; \
                     populate it in the step-through fence (or declare why it \
                     stays empty) before extending the schema"
                ),
            };
            let count = migrated
                .store
                .with_conn(|conn| {
                    conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                        row.get::<_, i64>(0)
                    })
                })
                .unwrap();
            assert!(
                count >= expected_min,
                "{table} carries {count} rows (expected at least {expected_min}); \
                 no migration was tested against data in it - the populate arm \
                 that fills it has stopped writing"
            );
        }
        drop(migrated);

        let reopened = SynapseStore::open(&descriptor).unwrap();
        assert_eq!(reopened.catalog_models().unwrap().len(), 4);
        assert_eq!(reopened.alias_table().unwrap().rows.len(), 2);
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
    fn migration_from_v5_preserves_jobs_pages_and_quarantines_legacy_measurements() {
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
        assert!(migrated
            .get_cert_row(
                CertificationClass::Embedding,
                "machine",
                &Fingerprint("fp".to_string()),
            )
            .unwrap()
            .is_none());
        let legacy_class: String = migrated
            .store
            .with_conn(|conn| {
                conn.query_row(
                    "SELECT certification_class FROM cert_rows WHERE fingerprint = 'fp'",
                    [],
                    |row| row.get(0),
                )
            })
            .unwrap();
        assert_eq!(legacy_class, "legacy_owned_decode");
        let job = migrated.get_job_at("legacy-job", 21).unwrap().unwrap();
        assert_eq!(job.request_digest, "legacy:legacy-job");
        assert_eq!(job.result_retention_ttl_ms, 100);
        assert!(migrated.get_job_page("legacy-job", 0).unwrap().is_some());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn certification_lookup_distinguishes_profiles_with_ane_subtypes() {
        let (root, descriptor) = temp_descriptor("ane-subtype-cert");
        let store = SynapseStore::open(&descriptor).unwrap();
        let fingerprint = Fingerprint("ane-fingerprint".to_string());
        let profile_without_ane = MachineProfile {
            os_build: "23G93".to_string(),
            arch: "aarch64".to_string(),
            chip_model: "Apple M5 Max".to_string(),
            ram_class: "le_64_gib".to_string(),
            ane_subtype: None,
            engine_identities: Vec::new(),
        };
        let mut profile_with_ane = profile_without_ane.clone();
        profile_with_ane.ane_subtype = Some("h17(map)".to_string());
        let without_hash = profile_without_ane.hash();
        let with_hash = profile_with_ane.hash();
        assert_ne!(without_hash, with_hash);

        store
            .store_class_scoped_cert_row(&ClassScopedCertificationRow {
                certification_class: CertificationClass::Embedding,
                assurance_class: AssuranceClass::Measured,
                status: CertificationStatus::Certified,
                key_hash: without_hash.clone(),
                machine_profile_hash: Some(without_hash.clone()),
                remote_profile_hash: None,
                identity_revision: None,
                numeric_profile_id: Some(NumericProfileId("ane-numeric".to_string())),
                fingerprint: fingerprint.clone(),
                certified_at_ms: 10,
                os_build: "23G93".to_string(),
                module_generation: 1,
                evidence: serde_json::json!({"ane_subtype": null}),
            })
            .unwrap();

        assert!(store
            .get_cert_row(CertificationClass::Embedding, &without_hash, &fingerprint)
            .unwrap()
            .is_some());
        assert!(store
            .get_cert_row(CertificationClass::Embedding, &with_hash, &fingerprint)
            .unwrap()
            .is_none());
        assert!(store
            .has_stale_cert_row(CertificationClass::Embedding, &with_hash, &fingerprint)
            .unwrap());
        let refused = crate::ensure_fingerprint_certified(
            &store,
            CertificationClass::Embedding,
            &with_hash,
            &fingerprint,
            "ane-model",
            false,
        )
        .expect_err("a changed ANE profile must require re-probing");
        assert_eq!(refused.code, "not_certified");
        assert!(refused.message.contains("stale certification rows"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn uncertified_reprobe_demotes_the_machine_fingerprint_pair() {
        let (root, descriptor) = temp_descriptor("uncertified-reprobe");
        let store = SynapseStore::open(&descriptor).unwrap();
        let fingerprint = Fingerprint("decode-fp".to_string());
        let mut row = ClassScopedCertificationRow {
            certification_class: CertificationClass::Embedding,
            assurance_class: AssuranceClass::Measured,
            status: CertificationStatus::Certified,
            key_hash: "machine-a".to_string(),
            machine_profile_hash: Some("machine-a".to_string()),
            remote_profile_hash: None,
            identity_revision: None,
            numeric_profile_id: Some(NumericProfileId("decode-np".to_string())),
            fingerprint: fingerprint.clone(),
            certified_at_ms: 10,
            os_build: "os-a".to_string(),
            module_generation: 1,
            evidence: serde_json::json!({"gate": "token_exact"}),
        };
        store.store_class_scoped_cert_row(&row).unwrap();
        assert!(store
            .get_cert_row(CertificationClass::Embedding, "machine-a", &fingerprint)
            .unwrap()
            .is_some());

        row.status = CertificationStatus::Uncertified;
        row.certified_at_ms = 20;
        row.evidence = serde_json::json!({
            "blocking_reason": "token_mismatch",
            "mismatches": [{"prompt": "corrupted fixture"}],
        });
        store.store_class_scoped_cert_row(&row).unwrap();

        assert!(store
            .get_cert_row(CertificationClass::Embedding, "machine-a", &fingerprint)
            .unwrap()
            .is_none());
        let probe = store
            .get_probe_row(CertificationClass::Embedding, "machine-a", &fingerprint)
            .unwrap()
            .unwrap();
        assert_eq!(probe.status, CertificationStatus::Uncertified);
        assert_eq!(probe.evidence, row.evidence);
        let refused = crate::ensure_fingerprint_certified(
            &store,
            CertificationClass::Embedding,
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
    fn embedding_certification_cannot_certify_rerank_with_shared_fingerprint() {
        let (root, descriptor) = temp_descriptor("cross-class-certification");
        let store = SynapseStore::open(&descriptor).unwrap();
        let fingerprint = Fingerprint("shared-fingerprint".to_string());
        store
            .store_class_scoped_cert_row(&ClassScopedCertificationRow {
                certification_class: CertificationClass::Embedding,
                assurance_class: AssuranceClass::Measured,
                status: CertificationStatus::Certified,
                key_hash: "legacy-profile-hash".to_string(),
                machine_profile_hash: Some("legacy-profile-hash".to_string()),
                remote_profile_hash: None,
                identity_revision: None,
                numeric_profile_id: Some(NumericProfileId("embedding-profile".to_string())),
                fingerprint: fingerprint.clone(),
                certified_at_ms: 1,
                os_build: "test-os".to_string(),
                module_generation: 1,
                evidence: serde_json::json!({"task": "embed"}),
            })
            .unwrap();
        assert!(store
            .get_cert_row(
                CertificationClass::Embedding,
                "legacy-profile-hash",
                &fingerprint,
            )
            .unwrap()
            .is_some());
        assert!(store
            .get_cert_row(
                CertificationClass::Rerank,
                "legacy-profile-hash",
                &fingerprint,
            )
            .unwrap()
            .is_none());
        let refused = crate::ensure_fingerprint_certified(
            &store,
            CertificationClass::Rerank,
            "legacy-profile-hash",
            &fingerprint,
            "rerank-model",
            false,
        )
        .expect_err("embedding evidence must not admit a rerank lane");
        assert_eq!(refused.code, "not_certified");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn legacy_hash_keeps_pre_upgrade_non_owned_certification_visible() {
        let (root, descriptor) = temp_descriptor("legacy-hash-upgrade");
        let store = SynapseStore::open(&descriptor).unwrap();
        let profile = MachineProfile {
            os_build: "test-os".to_string(),
            arch: "aarch64".to_string(),
            chip_model: "test-chip".to_string(),
            ram_class: "test-ram".to_string(),
            ane_subtype: None,
            engine_identities: Vec::new(),
        };
        let (legacy_hash, revisioned_hash) = crate::module_state_machine_profile_hashes(&profile);
        assert_ne!(legacy_hash, revisioned_hash);
        let fingerprint = Fingerprint("pre-upgrade-fingerprint".to_string());
        store
            .store_class_scoped_cert_row(&ClassScopedCertificationRow {
                certification_class: CertificationClass::Embedding,
                assurance_class: AssuranceClass::Measured,
                status: CertificationStatus::Certified,
                key_hash: legacy_hash.clone(),
                machine_profile_hash: Some(legacy_hash.clone()),
                remote_profile_hash: None,
                identity_revision: None,
                numeric_profile_id: Some(NumericProfileId("pre-upgrade".to_string())),
                fingerprint: fingerprint.clone(),
                certified_at_ms: 1,
                os_build: profile.os_build,
                module_generation: 1,
                evidence: serde_json::json!({"source": "pre-upgrade"}),
            })
            .unwrap();
        assert!(store
            .get_cert_row(CertificationClass::Embedding, &legacy_hash, &fingerprint)
            .unwrap()
            .is_some());
        assert!(store
            .get_cert_row(
                CertificationClass::Embedding,
                &revisioned_hash,
                &fingerprint,
            )
            .unwrap()
            .is_none());
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
            .get_cert_row(CertificationClass::Rerank, "remote-profile", &fingerprint)
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

    #[test]
    fn approval_rows_are_digest_checked_and_admin_mutations_are_fenced() {
        let (root, descriptor) = temp_descriptor("approval-row");
        let store = SynapseStore::open(&descriptor).unwrap();
        store
            .store
            .with_conn_fenced(|tx| {
                tx.execute(
                    "INSERT INTO models (
                         model_id, engine, task, fingerprint, config_json,
                         created_ms, updated_ms
                     ) VALUES ('owned-model', 'owned-metal-decode', 'generate', 'model-fp', '{}', 1, 1)",
                    [],
                )?;
                Ok(())
            })
            .unwrap();
        let fingerprint = "a".repeat(64);
        let created = store
            .create_approval("owned-model", &fingerprint, true, "operator", 10, 10)
            .unwrap();
        assert_eq!(created.semantic_digest, created.expected_digest().unwrap());
        assert!(store
            .get_approval("owned-model", &fingerprint)
            .unwrap()
            .is_some());

        store
            .store
            .with_conn_fenced(|tx| {
                tx.execute(
                    "UPDATE approvals SET grammar_enabled = 0
                     WHERE model_id = 'owned-model'",
                    [],
                )?;
                Ok(())
            })
            .unwrap();
        assert!(matches!(
            store.get_approval("owned-model", &fingerprint),
            Err(SynapseStoreError::ApprovalDigestMismatch { .. })
        ));
        let corruption_count: i64 = store
            .store
            .with_conn(|conn| {
                conn.query_row(
                    "SELECT COUNT(*) FROM approval_digest_corruption_events",
                    [],
                    |row| row.get(0),
                )
            })
            .unwrap();
        assert_eq!(corruption_count, 1);

        let mut repaired = created.clone();
        repaired.grammar_enabled = false;
        let repaired_digest = repaired.expected_digest().unwrap();
        store
            .store
            .with_conn_fenced(|tx| {
                tx.execute(
                    "UPDATE approvals SET semantic_digest = ?1
                     WHERE model_id = 'owned-model'",
                    params![repaired_digest],
                )?;
                Ok(())
            })
            .unwrap();
        let disabled = store
            .disable_approval("owned-model", &fingerprint, "operator rollback", 20)
            .unwrap();
        assert!(!disabled.enabled);
        assert_eq!(
            disabled.disabled_reason.as_deref(),
            Some("operator rollback")
        );
        let reenabled = store
            .enable_approval("owned-model", &fingerprint, "operator", 30, 30)
            .unwrap();
        assert!(reenabled.enabled);
        assert!(reenabled.disabled_reason.is_none());
        assert!(
            store
                .set_approval_grammar_enabled("owned-model", &fingerprint, false, 40)
                .unwrap()
                .semantic_digest
                .len()
                == 64
        );
        assert_eq!(store.emergency_rollback("fleet emergency", 50).unwrap(), 1);
        let rolled_back = store
            .get_approval("owned-model", &fingerprint)
            .unwrap()
            .expect("rollback leaves the row loadable");
        assert!(!rolled_back.enabled);
        assert_eq!(
            rolled_back.disabled_reason.as_deref(),
            Some("fleet emergency")
        );

        let explicitly_enabled = store
            .enable_or_create_approval("owned-model", &fingerprint, true, "principal:direct", 60)
            .unwrap();
        assert!(explicitly_enabled.enabled);
        assert!(explicitly_enabled.grammar_enabled);
        assert_eq!(
            explicitly_enabled.approved_by.as_deref(),
            Some("principal:direct")
        );
        assert_eq!(explicitly_enabled.approved_at_ms, Some(60));
        assert_eq!(explicitly_enabled.updated_at_ms, 60);
        assert_eq!(explicitly_enabled.generation, rolled_back.generation + 1);
        assert_eq!(
            explicitly_enabled.semantic_digest,
            explicitly_enabled.expected_digest().unwrap()
        );

        let second_fingerprint = "b".repeat(64);
        let explicitly_created = store
            .enable_or_create_approval(
                "owned-model",
                &second_fingerprint,
                false,
                "principal:direct",
                70,
            )
            .unwrap();
        assert!(explicitly_created.enabled);
        assert!(!explicitly_created.grammar_enabled);
        assert_eq!(explicitly_created.generation, 0);
        assert_eq!(
            explicitly_created.semantic_digest,
            explicitly_created.expected_digest().unwrap()
        );
        assert!(matches!(
            store.enable_or_create_approval(
                "owned-model",
                "not-a-digest",
                false,
                "principal:direct",
                80,
            ),
            Err(SynapseStoreError::Decode(_))
        ));
        assert_eq!(store.approvals().unwrap().len(), 2);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn approval_migration_is_one_shot_and_requires_catalog_mapping() {
        let (root, descriptor) = temp_descriptor("approval-migration");
        let store = SynapseStore::open(&descriptor).unwrap();
        let unmappable = store
            .migrate_owned_decode_approvals(
                "owned-decode-approval-migration-v1",
                APPROVAL_SCHEMA_REVISION,
            )
            .unwrap();
        assert_eq!(unmappable.outcome, "unmappable_identity");
        let model_ids = [
            "qwen3-0.6b-decode-f16",
            "lfm2-1.2b-decode-f16",
            "qwen3-0.6b-decode-q8_0",
            "lfm2-1.2b-decode-q8_0",
        ];
        store
            .store
            .with_conn_fenced(|tx| {
                for model_id in model_ids {
                    tx.execute(
                        "INSERT INTO models (
                             model_id, engine, task, fingerprint, config_json,
                             created_ms, updated_ms
                         ) VALUES (?1, 'ort', 'embed', ?1, '{}', 1, 1)",
                        params![model_id],
                    )?;
                }
                Ok(())
            })
            .unwrap();
        let wrong_shape = store
            .migrate_owned_decode_approvals(
                "owned-decode-approval-migration-v1",
                APPROVAL_SCHEMA_REVISION,
            )
            .unwrap();
        assert_eq!(wrong_shape.outcome, "unmappable_identity");
        assert!(store.approvals().unwrap().is_empty());
        for model_id in model_ids {
            store
                .upsert_model(&owned_seed_model_config(model_id), 2)
                .unwrap();
        }
        let applied = store
            .migrate_owned_decode_approvals(
                "owned-decode-approval-migration-v1",
                APPROVAL_SCHEMA_REVISION,
            )
            .unwrap();
        assert_eq!(applied.outcome, "applied");
        assert_eq!(applied.rows, 4);
        let computed_seed_digest = hex::encode(Sha256::digest(
            include_str!("../owned-decode-manifests/migration-seed-manifest-v1.json").as_bytes(),
        ));
        let marker_digest: String = store
            .store
            .with_conn(|conn| {
                conn.query_row(
                    "SELECT seed_digest FROM approval_migration_markers
                     WHERE seed_revision = 'owned-decode-approval-migration-v1'",
                    [],
                    |row| row.get(0),
                )
            })
            .unwrap();
        assert_eq!(marker_digest, computed_seed_digest);
        let already_applied = store
            .migrate_owned_decode_approvals(
                "owned-decode-approval-migration-v1",
                APPROVAL_SCHEMA_REVISION,
            )
            .unwrap();
        assert_eq!(already_applied.outcome, "already_applied");
        assert_eq!(store.approvals().unwrap().len(), 4);

        store
            .store
            .with_conn_fenced(|tx| {
                tx.execute(
                    "UPDATE approval_migration_markers SET row_count = 3
                     WHERE seed_revision = 'owned-decode-approval-migration-v1'",
                    [],
                )?;
                Ok(())
            })
            .unwrap();
        assert!(matches!(
            store.get_approval(
                "qwen3-0.6b-decode-f16",
                "8bcb6dfc1bd55a38a56b5a6931132f428687e064104d106c2cc5efb898f80feb",
            ),
            Err(SynapseStoreError::ApprovalMigrationStateCorrupt(_))
        ));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn approval_migration_rejects_tampered_seed_bytes_before_parsing() {
        let (root, descriptor) = temp_descriptor("approval-migration-seed-digest");
        let store = SynapseStore::open(&descriptor).unwrap();
        let tampered = include_str!("../owned-decode-manifests/migration-seed-manifest-v1.json")
            .replacen("\"grammar_enabled\": true", "\"grammar_enabled\": false", 1);
        let result = store
            .migrate_owned_decode_approvals_from_seed(
                &tampered,
                "owned-decode-approval-migration-v1",
                APPROVAL_SCHEMA_REVISION,
            )
            .unwrap();
        assert_eq!(result.outcome, "invalid_seed");
        assert_eq!(result.marker, "seed_digest_mismatch");
        let marker_count: i64 = store
            .store
            .with_conn(|conn| {
                conn.query_row(
                    "SELECT COUNT(*) FROM approval_migration_markers",
                    [],
                    |row| row.get(0),
                )
            })
            .unwrap();
        assert_eq!(marker_count, 0);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn owned_decode_rows_are_epoch_scoped_shape_exact_and_class_isolated() {
        let (root, descriptor) = temp_descriptor("owned-row-epochs");
        let store = SynapseStore::open(&descriptor).unwrap();
        let profile_hash = "b".repeat(64);
        let decode_fingerprint = "c".repeat(64);
        let evidence = passed_probe_gate_evidence();
        let mut row = OwnedDecodeCertificationRow {
            status: CertificationStatus::Certified,
            revisioned_machine_profile_hash: profile_hash.clone(),
            profile_activation_epoch: 1,
            model_id: "owned-model".to_string(),
            decode_fingerprint: decode_fingerprint.clone(),
            processing_fingerprint: "processing-a".to_string(),
            runtime_config_digest: "runtime-a".to_string(),
            constraint_runtime_identities: vec!["constraint-a".to_string()],
            worker_path_evidence: serde_json::json!({"worker": "a"}),
            evidence_schema_revision: CERT_EVIDENCE_SCHEMA_REVISION.to_string(),
            g_dec_manifest_revision: G_DEC_MANIFEST_REVISION.to_string(),
            numeric_profile_id: None,
            fingerprint: Fingerprint(decode_fingerprint.clone()),
            certified_at_ms: 10,
            os_build: "os-a".to_string(),
            module_generation: 1,
            evidence: evidence.clone(),
        };
        store.store_owned_decode_cert_row(&row).unwrap();
        row.profile_activation_epoch = 3;
        row.certified_at_ms = 30;
        store.store_owned_decode_cert_row(&row).unwrap();
        assert!(store
            .get_owned_decode_cert_row(
                &profile_hash,
                1,
                "owned-model",
                &decode_fingerprint,
                CERT_EVIDENCE_SCHEMA_REVISION,
                &row.constraint_runtime_identities,
            )
            .unwrap()
            .is_some());
        assert!(store
            .get_owned_decode_cert_row(
                &profile_hash,
                3,
                "owned-model",
                &decode_fingerprint,
                CERT_EVIDENCE_SCHEMA_REVISION,
                &row.constraint_runtime_identities,
            )
            .unwrap()
            .is_some());
        let match_inputs = OwnedDecodeMatchInputs {
            revisioned_machine_profile_hash: profile_hash.clone(),
            profile_activation_epoch: 3,
            model_id: "owned-model".to_string(),
            decode_fingerprint: decode_fingerprint.clone(),
            processing_fingerprint: "processing-a".to_string(),
            runtime_config_digest: "runtime-a".to_string(),
            constraint_runtime_identities: vec!["constraint-a".to_string()],
            worker_path_evidence: serde_json::json!({"worker": "a"}),
            evidence_schema_revision: CERT_EVIDENCE_SCHEMA_REVISION.to_string(),
            g_dec_manifest_revision: G_DEC_MANIFEST_REVISION.to_string(),
        };
        assert!(store
            .get_owned_decode_cert_row_matching(&match_inputs)
            .unwrap()
            .is_some());
        let mut unconstrained_inputs = match_inputs.clone();
        unconstrained_inputs.constraint_runtime_identities.clear();
        assert!(store
            .get_owned_decode_cert_row_matching(&unconstrained_inputs)
            .unwrap()
            .is_none());
        let mut unconstrained_row = row.clone();
        unconstrained_row.constraint_runtime_identities.clear();
        store
            .store_owned_decode_cert_row(&unconstrained_row)
            .unwrap();
        assert!(store
            .get_owned_decode_cert_row_matching(&unconstrained_inputs)
            .unwrap()
            .is_some());
        let mut uncertified_constraint_inputs = match_inputs.clone();
        uncertified_constraint_inputs.constraint_runtime_identities =
            vec!["constraint-b".to_string()];
        assert!(store
            .get_owned_decode_cert_row_matching(&uncertified_constraint_inputs)
            .unwrap()
            .is_none());
        assert!(store
            .get_owned_decode_cert_row(
                &profile_hash,
                2,
                "owned-model",
                &decode_fingerprint,
                CERT_EVIDENCE_SCHEMA_REVISION,
                &row.constraint_runtime_identities,
            )
            .unwrap()
            .is_none());

        for class in [
            CertificationClass::Declared,
            CertificationClass::Remote,
            CertificationClass::Embedding,
            CertificationClass::Rerank,
        ] {
            let class_row = ClassScopedCertificationRow {
                certification_class: class,
                assurance_class: AssuranceClass::Declared,
                status: CertificationStatus::Certified,
                key_hash: format!("key-{}", class.as_str()),
                machine_profile_hash: None,
                remote_profile_hash: matches!(
                    class,
                    CertificationClass::Declared | CertificationClass::Remote
                )
                .then(|| "remote-profile".to_string()),
                identity_revision: matches!(
                    class,
                    CertificationClass::Declared | CertificationClass::Remote
                )
                .then(|| "identity-v1".to_string()),
                numeric_profile_id: None,
                fingerprint: Fingerprint("shared-fp".to_string()),
                certified_at_ms: 1,
                os_build: String::new(),
                module_generation: 1,
                evidence: serde_json::json!({"class": class.as_str()}),
            };
            store.store_class_scoped_cert_row(&class_row).unwrap();
            let mut updated = class_row.clone();
            updated.certified_at_ms = 2;
            store.store_class_scoped_cert_row(&updated).unwrap();
        }
        let class_count: i64 = store
            .store
            .with_conn(|conn| {
                conn.query_row(
                    "SELECT COUNT(*) FROM cert_rows WHERE fingerprint = 'shared-fp'",
                    [],
                    |row| row.get(0),
                )
            })
            .unwrap();
        assert_eq!(class_count, 4);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn profile_activation_is_epoch_one_then_single_parent_per_rotation() {
        let (root, descriptor) = temp_descriptor("profile-rotation");
        let store = SynapseStore::open(&descriptor).unwrap();
        let fingerprint = "d".repeat(64);
        store
            .store
            .with_conn_fenced(|tx| {
                tx.execute(
                    "INSERT INTO models (
                         model_id, engine, task, fingerprint, config_json,
                         created_ms, updated_ms
                     ) VALUES ('owned-model', 'owned-metal-decode', 'generate', 'model-fp', '{}', 1, 1)",
                    [],
                )?;
                Ok(())
            })
            .unwrap();
        store
            .create_approval("owned-model", &fingerprint, true, "operator", 1, 1)
            .unwrap();
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
        assert_eq!(rotated.event.as_ref().unwrap().outcomes.len(), 1);
        let event_id = rotated.event.as_ref().unwrap().event_id.clone();
        store
            .set_rotation_certification_outcome(
                &event_id,
                "owned-model",
                &fingerprint,
                RotationOutcomeState::Passed,
                Some(25),
                None,
            )
            .unwrap();
        assert_eq!(
            store.rotation_outcomes(&event_id).unwrap()[0].outcome_state,
            "passed"
        );
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

    #[test]
    fn certification_stale_since_persists_across_store_reopen() {
        let (root, descriptor) = temp_descriptor("certification-stale-since");
        let profile = MachineProfile {
            os_build: "os-a".to_string(),
            arch: "aarch64".to_string(),
            chip_model: "chip-a".to_string(),
            ram_class: "le_32_gib".to_string(),
            ane_subtype: None,
            engine_identities: Vec::new(),
        };
        let profile_hash = profile.revisioned_hash();
        let store = SynapseStore::open(&descriptor).expect("store opens");
        store
            .observe_profile(&profile, 10, 1)
            .expect("profile activates");
        assert_eq!(
            store
                .observe_certification_stale(&profile_hash, 123)
                .expect("staleness records"),
            Some(123)
        );
        assert_eq!(
            store
                .observe_certification_stale(&profile_hash, 456)
                .expect("first observation remains fixed"),
            Some(123)
        );
        drop(store);

        let reopened = SynapseStore::open(&descriptor).expect("store reopens");
        assert_eq!(
            reopened
                .profile_state()
                .expect("profile state reads")
                .certification_stale_since_ms,
            Some(123)
        );
        drop(reopened);
        let _ = std::fs::remove_dir_all(root);
    }

    /// `latest_owned_decode_measurement_row` must find a certified row under a
    /// profile that is no longer current, because that is the only thing that
    /// separates "certified, then the machine rotated" from "never probed". The
    /// profile-scoped lookup answers `None` for both, so a staleness reading
    /// built from it alone is unsatisfiable rather than merely inaccurate.
    #[test]
    fn latest_decode_row_survives_a_rotation_the_scoped_lookup_misses() {
        let (_root, descriptor) = temp_descriptor("latest-decode-row-rotation");
        let store = SynapseStore::open(&descriptor).expect("store opens");
        let profile_a = MachineProfile {
            os_build: "os-before-rotation".to_string(),
            arch: "aarch64".to_string(),
            chip_model: "chip-a".to_string(),
            ram_class: "le_32_gib".to_string(),
            ane_subtype: None,
            engine_identities: Vec::new(),
        };
        let mut profile_b = profile_a.clone();
        profile_b.os_build = "os-after-rotation".to_string();
        store.observe_profile(&profile_a, 10, 1).expect("profile a");
        // The identity validator requires real digests, a catalog-shaped model
        // id and the current revisions; a row that fails it never reaches the
        // lookup this test is about.
        let decode_fingerprint = "d".repeat(64);

        store
            .store_owned_decode_cert_row(&OwnedDecodeCertificationRow {
                status: CertificationStatus::Certified,
                revisioned_machine_profile_hash: profile_a.revisioned_hash(),
                profile_activation_epoch: 1,
                model_id: "owned-model".to_string(),
                decode_fingerprint: decode_fingerprint.clone(),
                processing_fingerprint: "processing".to_string(),
                runtime_config_digest: "runtime".to_string(),
                constraint_runtime_identities: Vec::new(),
                worker_path_evidence: serde_json::json!({}),
                evidence_schema_revision: CERT_EVIDENCE_SCHEMA_REVISION.to_string(),
                g_dec_manifest_revision: G_DEC_MANIFEST_REVISION.to_string(),
                numeric_profile_id: None,
                fingerprint: Fingerprint(decode_fingerprint.clone()),
                certified_at_ms: 11,
                os_build: profile_a.os_build.clone(),
                module_generation: 1,
                evidence: serde_json::json!({}),
            })
            .expect("row stores under profile a");

        assert_ne!(profile_a.revisioned_hash(), profile_b.revisioned_hash());

        let scoped = store
            .get_owned_decode_measurement_row(
                &profile_b.revisioned_hash(),
                1,
                "owned-model",
                &decode_fingerprint,
                CERT_EVIDENCE_SCHEMA_REVISION,
                &[],
            )
            .expect("scoped lookup runs");
        assert!(
            scoped.is_none(),
            "the profile-scoped lookup must not see a row from the previous profile"
        );

        let latest = store
            .latest_owned_decode_measurement_row(
                "owned-model",
                &decode_fingerprint,
                CERT_EVIDENCE_SCHEMA_REVISION,
                &[],
            )
            .expect("latest lookup runs")
            .expect("a row certified under any profile must still be found");
        assert_eq!(
            latest.revisioned_machine_profile_hash,
            profile_a.revisioned_hash()
        );

        // A lane that was never certified stays None on both, so the two states
        // are distinguishable rather than merely both non-empty.
        let never = store
            .latest_owned_decode_measurement_row(
                "never-probed-model",
                &decode_fingerprint,
                CERT_EVIDENCE_SCHEMA_REVISION,
                &[],
            )
            .expect("latest lookup runs for an unknown lane");
        assert!(
            never.is_none(),
            "a never-certified lane must have no latest row"
        );
    }

    #[test]
    fn owned_decode_admission_and_health_require_full_current_evidence() {
        let (root, descriptor) = temp_descriptor("dual-guard");
        let store = SynapseStore::open(&descriptor).unwrap();
        store
            .store
            .with_conn_fenced(|tx| {
                tx.execute(
                    "INSERT INTO models (
                         model_id, engine, task, fingerprint, config_json,
                         created_ms, updated_ms
                     ) VALUES ('owned-model', 'owned-metal-decode', 'generate', 'model-fp', '{}', 1, 1)",
                    [],
                )?;
                Ok(())
            })
            .unwrap();
        let decode_fingerprint = "e".repeat(64);
        let approval = store
            .create_approval("owned-model", &decode_fingerprint, true, "operator", 1, 1)
            .unwrap();
        let profile_a = MachineProfile {
            os_build: "os-a".to_string(),
            arch: "aarch64".to_string(),
            chip_model: "chip-a".to_string(),
            ram_class: "le_32_gib".to_string(),
            ane_subtype: None,
            engine_identities: Vec::new(),
        };
        let mut profile_b = profile_a.clone();
        profile_b.os_build = "os-b".to_string();
        store.observe_profile(&profile_a, 1, 1).unwrap();
        let evidence = passed_probe_gate_evidence();
        let row = |epoch, hash: String| OwnedDecodeCertificationRow {
            status: CertificationStatus::Certified,
            revisioned_machine_profile_hash: hash,
            profile_activation_epoch: epoch,
            model_id: "owned-model".to_string(),
            decode_fingerprint: decode_fingerprint.clone(),
            processing_fingerprint: "processing".to_string(),
            runtime_config_digest: "runtime".to_string(),
            constraint_runtime_identities: vec![
                "constraint-a".to_string(),
                "constraint-b".to_string(),
            ],
            worker_path_evidence: serde_json::json!({}),
            evidence_schema_revision: CERT_EVIDENCE_SCHEMA_REVISION.to_string(),
            g_dec_manifest_revision: G_DEC_MANIFEST_REVISION.to_string(),
            numeric_profile_id: None,
            fingerprint: Fingerprint(decode_fingerprint.clone()),
            certified_at_ms: epoch,
            os_build: "os".to_string(),
            module_generation: 1,
            evidence: evidence.clone(),
        };
        let hash_a = profile_a.revisioned_hash();
        let hash_b = profile_b.revisioned_hash();
        let row_a = row(1, hash_a.clone());
        store.store_owned_decode_cert_row(&row_a).unwrap();
        let snapshot_inputs = OwnedDecodeMatchInputs {
            revisioned_machine_profile_hash: hash_a.clone(),
            profile_activation_epoch: 1,
            model_id: "owned-model".to_string(),
            decode_fingerprint: decode_fingerprint.clone(),
            processing_fingerprint: "processing".to_string(),
            runtime_config_digest: "runtime".to_string(),
            constraint_runtime_identities: vec![
                "constraint-a".to_string(),
                "constraint-b".to_string(),
            ],
            worker_path_evidence: serde_json::json!({}),
            evidence_schema_revision: CERT_EVIDENCE_SCHEMA_REVISION.to_string(),
            g_dec_manifest_revision: G_DEC_MANIFEST_REVISION.to_string(),
        };
        let mut subset_inputs = snapshot_inputs.clone();
        subset_inputs.constraint_runtime_identities = vec!["constraint-a".to_string()];
        assert!(store
            .owned_decode_admission_matching(&subset_inputs)
            .unwrap()
            .is_none());
        let mut no_constraint_inputs = snapshot_inputs.clone();
        no_constraint_inputs.constraint_runtime_identities.clear();
        assert!(store
            .owned_decode_admission_matching(&no_constraint_inputs)
            .unwrap()
            .is_none());
        let mut unconstrained_probe_row = row_a.clone();
        unconstrained_probe_row
            .constraint_runtime_identities
            .clear();
        assert_eq!(
            store
                .store_owned_decode_cert_rows_if_current(
                    &snapshot_inputs,
                    &snapshot_inputs,
                    &[row_a.clone(), unconstrained_probe_row],
                    2,
                )
                .unwrap(),
            ProbeWriteOutcome::Certified
        );
        let mut unconstrained_probe_inputs = snapshot_inputs.clone();
        unconstrained_probe_inputs
            .constraint_runtime_identities
            .clear();
        assert!(store
            .get_owned_decode_cert_row_matching(&unconstrained_probe_inputs)
            .unwrap()
            .is_some());
        let mut changed_terminal = snapshot_inputs.clone();
        changed_terminal.processing_fingerprint = "processing-changed".to_string();
        let stale = store
            .store_owned_decode_cert_row_if_current(&snapshot_inputs, &changed_terminal, &row_a, 2)
            .unwrap();
        assert_eq!(stale, ProbeWriteOutcome::ProbeStale);
        assert_eq!(
            store
                .get_owned_decode_cert_row(
                    &hash_a,
                    1,
                    "owned-model",
                    &decode_fingerprint,
                    CERT_EVIDENCE_SCHEMA_REVISION,
                    &row_a.constraint_runtime_identities,
                )
                .unwrap()
                .unwrap()
                .processing_fingerprint,
            "processing"
        );
        let stale_events: i64 = store
            .store
            .with_conn(|conn| {
                conn.query_row(
                    "SELECT COUNT(*) FROM cert_row_rebuild_events WHERE outcome = 'stale_probe'",
                    [],
                    |row| row.get(0),
                )
            })
            .unwrap();
        assert_eq!(stale_events, 1);
        assert!(store
            .owned_decode_admission("owned-model", &decode_fingerprint, &hash_a, 1)
            .unwrap()
            .is_some());
        let mut unconstrained_inputs = snapshot_inputs.clone();
        unconstrained_inputs.constraint_runtime_identities.clear();
        assert!(store
            .owned_decode_admission("owned-model", &decode_fingerprint, &hash_a, 1)
            .unwrap()
            .is_some());
        assert!(store
            .owned_decode_admission_matching(&unconstrained_inputs)
            .unwrap()
            .is_some());
        assert!(store
            .owned_decode_admission_matching(&snapshot_inputs)
            .unwrap()
            .is_some());
        let mut uncertified_constraint_inputs = snapshot_inputs.clone();
        uncertified_constraint_inputs.constraint_runtime_identities =
            vec!["constraint-b".to_string()];
        assert!(store
            .owned_decode_admission_matching(&uncertified_constraint_inputs)
            .unwrap()
            .is_none());
        assert!(store
            .owned_decode_admission("owned-model", &decode_fingerprint, &hash_a, 2)
            .unwrap()
            .is_none());
        store.observe_profile(&profile_b, 2, 1).unwrap();
        assert!(store
            .owned_decode_admission("owned-model", &decode_fingerprint, &hash_a, 1)
            .unwrap()
            .is_none());
        let row_b = row(2, hash_b.clone());
        store.store_owned_decode_cert_row(&row_b).unwrap();
        let mut unconstrained_row_b = row_b.clone();
        unconstrained_row_b.constraint_runtime_identities.clear();
        store
            .store_owned_decode_cert_row(&unconstrained_row_b)
            .unwrap();
        assert!(store
            .owned_decode_admission("owned-model", &decode_fingerprint, &hash_b, 2)
            .unwrap()
            .is_some());
        assert_eq!(
            store
                .get_approval("owned-model", &decode_fingerprint)
                .unwrap(),
            Some(approval),
            "certification writes and serving reads must not mutate approvals"
        );
        let mut incomplete_constrained = row_b;
        incomplete_constrained.evidence = serde_json::json!({"gates": []});
        store
            .store_owned_decode_cert_row(&incomplete_constrained)
            .unwrap();
        let mut incomplete = unconstrained_row_b;
        incomplete.evidence = serde_json::json!({"gates": []});
        store.store_owned_decode_cert_row(&incomplete).unwrap();
        let incomplete_inputs = OwnedDecodeMatchInputs {
            revisioned_machine_profile_hash: incomplete.revisioned_machine_profile_hash.clone(),
            profile_activation_epoch: incomplete.profile_activation_epoch,
            model_id: incomplete.model_id.clone(),
            decode_fingerprint: incomplete.decode_fingerprint.clone(),
            processing_fingerprint: incomplete.processing_fingerprint.clone(),
            runtime_config_digest: incomplete.runtime_config_digest.clone(),
            constraint_runtime_identities: Vec::new(),
            worker_path_evidence: incomplete.worker_path_evidence.clone(),
            evidence_schema_revision: incomplete.evidence_schema_revision.clone(),
            g_dec_manifest_revision: incomplete.g_dec_manifest_revision.clone(),
        };
        assert!(store
            .owned_decode_admission_matching(&incomplete_inputs)
            .unwrap()
            .is_none());
        let health = store.storage_health_inputs().unwrap();
        let approval_health = health
            .approval_certification_outcomes
            .iter()
            .find(|outcome| outcome.model_id == "owned-model")
            .unwrap();
        assert_eq!(approval_health.admission, "no_current_evidence");
        let _ = std::fs::remove_dir_all(root);
    }

    /// A row written before the evidence schema moved to v2 claimed twelve
    /// G-DEC gates from one measurement. It must stop certifying the lane the
    /// moment the module upgrades, while a v2 row recording the probe's own
    /// gate admits the same lane under the same approval.
    #[test]
    fn pre_v2_certification_rows_do_not_admit_and_v2_rows_do() {
        let (root, descriptor) = temp_descriptor("cert-evidence-v2");
        let store = SynapseStore::open(&descriptor).unwrap();
        store
            .store
            .with_conn_fenced(|tx| {
                tx.execute(
                    "INSERT INTO models (
                         model_id, engine, task, fingerprint, config_json,
                         created_ms, updated_ms
                     ) VALUES ('owned-model', 'owned-metal-decode', 'generate', 'model-fp', '{}', 1, 1)",
                    [],
                )?;
                Ok(())
            })
            .unwrap();
        let decode_fingerprint = "e".repeat(64);
        let approval = store
            .create_approval("owned-model", &decode_fingerprint, true, "operator", 1, 1)
            .unwrap();
        let profile = MachineProfile {
            os_build: "os-a".to_string(),
            arch: "aarch64".to_string(),
            chip_model: "chip-a".to_string(),
            ram_class: "le_32_gib".to_string(),
            ane_subtype: None,
            engine_identities: Vec::new(),
        };
        store.observe_profile(&profile, 1, 1).unwrap();
        let inputs = OwnedDecodeMatchInputs {
            revisioned_machine_profile_hash: profile.revisioned_hash(),
            profile_activation_epoch: 1,
            model_id: "owned-model".to_string(),
            decode_fingerprint: decode_fingerprint.clone(),
            processing_fingerprint: "processing".to_string(),
            runtime_config_digest: "runtime".to_string(),
            constraint_runtime_identities: Vec::new(),
            worker_path_evidence: serde_json::json!({}),
            evidence_schema_revision: CERT_EVIDENCE_SCHEMA_REVISION.to_string(),
            g_dec_manifest_revision: G_DEC_MANIFEST_REVISION.to_string(),
        };
        // Exactly what a v1 module persisted for a passed probe. The current
        // writer refuses these revisions, so it goes in through the raw upsert
        // the way an older module left it in the store.
        let v1_row = OwnedDecodeCertificationRow {
            status: CertificationStatus::Certified,
            revisioned_machine_profile_hash: inputs.revisioned_machine_profile_hash.clone(),
            profile_activation_epoch: 1,
            model_id: inputs.model_id.clone(),
            decode_fingerprint: decode_fingerprint.clone(),
            processing_fingerprint: inputs.processing_fingerprint.clone(),
            runtime_config_digest: inputs.runtime_config_digest.clone(),
            constraint_runtime_identities: Vec::new(),
            worker_path_evidence: inputs.worker_path_evidence.clone(),
            evidence_schema_revision: "owned-decode-cert-evidence-v1".to_string(),
            g_dec_manifest_revision: "decode-fixture-registry-v1".to_string(),
            numeric_profile_id: None,
            fingerprint: Fingerprint(decode_fingerprint.clone()),
            certified_at_ms: 1,
            os_build: "os-a".to_string(),
            module_generation: 1,
            evidence: serde_json::json!({
                "g_dec": (1..=12)
                    .map(|number| serde_json::json!({
                        "id": format!("G-DEC-{number:02}"),
                        "status": "passed",
                        "manifest_revision": "decode-fixture-registry-v1",
                    }))
                    .collect::<Vec<_>>(),
            }),
        };
        assert!(
            store.store_owned_decode_cert_row(&v1_row).is_err(),
            "the current writer must refuse v1 revisions"
        );
        store
            .store
            .with_conn_fenced(|tx| upsert_owned_decode_cert_row_tx(tx, &v1_row))
            .unwrap();

        let evaluation = store.owned_decode_admission_evaluation(&inputs).unwrap();
        assert!(evaluation.admission().is_none(), "a v1 row must not admit");
        assert_eq!(
            evaluation.refusal(),
            Some(&OwnedDecodeAdmissionRefusal::NotCertified)
        );
        assert!(store
            .owned_decode_admission(
                "owned-model",
                &decode_fingerprint,
                &inputs.revisioned_machine_profile_hash,
                1
            )
            .unwrap()
            .is_none());
        let mut v1_inputs = inputs.clone();
        v1_inputs.evidence_schema_revision = v1_row.evidence_schema_revision.clone();
        v1_inputs.g_dec_manifest_revision = v1_row.g_dec_manifest_revision.clone();
        assert!(
            store
                .owned_decode_admission_evaluation(&v1_inputs)
                .unwrap()
                .admission()
                .is_none(),
            "asking for the v1 revisions by name must not admit either"
        );
        let divergence = store
            .storage_health_inputs()
            .unwrap()
            .evidence_requirements_divergence;
        assert!(divergence.iter().any(|entry| entry.model_id == "owned-model"
            && entry.result == "evidence_schema_incompatible"));

        let mut v2_row = v1_row.clone();
        v2_row.evidence_schema_revision = CERT_EVIDENCE_SCHEMA_REVISION.to_string();
        v2_row.g_dec_manifest_revision = G_DEC_MANIFEST_REVISION.to_string();
        v2_row.evidence = passed_probe_gate_evidence();
        v2_row.certified_at_ms = 2;
        store.store_owned_decode_cert_row(&v2_row).unwrap();
        let admission = store
            .owned_decode_admission_evaluation(&inputs)
            .unwrap()
            .admission()
            .cloned()
            .expect("a v2 row recording the probe gate as passed must admit");
        assert_eq!(admission.certification.evidence, passed_probe_gate_evidence());
        assert_eq!(
            store
                .get_approval("owned-model", &decode_fingerprint)
                .unwrap(),
            Some(approval),
            "the schema change must leave the approval untouched"
        );

        // The same v2 row with the gate recorded as failed, or with a G-DEC
        // claim beside it, certifies nothing.
        for evidence in [
            serde_json::json!({"gates": [{
                "id": "structural_band",
                "status": "failed",
                "manifest_revision": "owned-decode-probe-gates-v1",
            }]}),
            serde_json::json!({"gates": [
                {
                    "id": "structural_band",
                    "status": "passed",
                    "manifest_revision": "owned-decode-probe-gates-v1",
                },
                {
                    "id": "G-DEC-01",
                    "status": "passed",
                    "manifest_revision": "owned-decode-probe-gates-v1",
                },
            ]}),
        ] {
            let mut row = v2_row.clone();
            row.evidence = evidence.clone();
            row.certified_at_ms = 3;
            store.store_owned_decode_cert_row(&row).unwrap();
            assert!(
                store
                    .owned_decode_admission_evaluation(&inputs)
                    .unwrap()
                    .admission()
                    .is_none(),
                "{evidence} must not admit"
            );
        }
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn owned_decode_refusal_mapping_tracks_each_fenced_predicate_arm() {
        use crate::owned_decode_routing::lane::{
            serving_predicate, ServingPredicateInputs, ServingRefusal,
        };

        #[derive(Clone, Copy)]
        enum FixtureArm {
            ApprovalAbsent,
            ApprovalDisabled,
            CertificationMissing,
            BoundaryMismatch,
        }

        let cases = [
            (
                "approval absent",
                FixtureArm::ApprovalAbsent,
                "cutover_disabled",
            ),
            (
                "approval disabled",
                FixtureArm::ApprovalDisabled,
                "cutover_disabled",
            ),
            (
                "certification missing",
                FixtureArm::CertificationMissing,
                "owned_decode_not_certified",
            ),
            (
                "boundary mismatch",
                FixtureArm::BoundaryMismatch,
                "owned_decode_not_certified",
            ),
        ];
        for (name, arm, expected_wire) in cases {
            let (root, descriptor) = temp_descriptor(&format!("refusal-{name}"));
            let store = SynapseStore::open(&descriptor).unwrap();
            let model_id = "qwen3-0.6b-decode-f16";
            let decode_fingerprint =
                "8bcb6dfc1bd55a38a56b5a6931132f428687e064104d106c2cc5efb898f80feb";
            store
                .upsert_model(&owned_seed_model_config(model_id), 1)
                .unwrap();
            let profile = MachineProfile {
                os_build: "test-os".to_string(),
                arch: "aarch64".to_string(),
                chip_model: "test-chip".to_string(),
                ram_class: "test-ram".to_string(),
                ane_subtype: None,
                engine_identities: Vec::new(),
            };
            store.observe_profile(&profile, 1, 1).unwrap();
            let inputs = OwnedDecodeMatchInputs {
                revisioned_machine_profile_hash: profile.revisioned_hash(),
                profile_activation_epoch: 1,
                model_id: model_id.to_string(),
                decode_fingerprint: decode_fingerprint.to_string(),
                processing_fingerprint: "processing".to_string(),
                runtime_config_digest: "runtime".to_string(),
                constraint_runtime_identities: Vec::new(),
                worker_path_evidence: serde_json::json!({}),
                evidence_schema_revision: CERT_EVIDENCE_SCHEMA_REVISION.to_string(),
                g_dec_manifest_revision: G_DEC_MANIFEST_REVISION.to_string(),
            };
            if !matches!(arm, FixtureArm::ApprovalAbsent) {
                store
                    .create_approval(model_id, decode_fingerprint, true, "operator", 1, 1)
                    .unwrap();
            }
            if matches!(arm, FixtureArm::ApprovalDisabled) {
                store
                    .disable_approval(
                        model_id,
                        decode_fingerprint,
                        "approval disabled pending certification",
                        2,
                    )
                    .unwrap();
            }
            if matches!(arm, FixtureArm::BoundaryMismatch) {
                store
                    .store_owned_decode_cert_row(&OwnedDecodeCertificationRow {
                        status: CertificationStatus::Certified,
                        revisioned_machine_profile_hash: inputs
                            .revisioned_machine_profile_hash
                            .clone(),
                        profile_activation_epoch: inputs.profile_activation_epoch,
                        model_id: inputs.model_id.clone(),
                        decode_fingerprint: inputs.decode_fingerprint.clone(),
                        processing_fingerprint: inputs.processing_fingerprint.clone(),
                        runtime_config_digest: inputs.runtime_config_digest.clone(),
                        constraint_runtime_identities: Vec::new(),
                        worker_path_evidence: inputs.worker_path_evidence.clone(),
                        evidence_schema_revision: inputs.evidence_schema_revision.clone(),
                        g_dec_manifest_revision: inputs.g_dec_manifest_revision.clone(),
                        numeric_profile_id: None,
                        fingerprint: Fingerprint(inputs.decode_fingerprint.clone()),
                        certified_at_ms: 1,
                        os_build: "test-os".to_string(),
                        module_generation: 1,
                        evidence: passed_probe_gate_evidence(),
                    })
                    .unwrap();
            }

            let evaluation = store.owned_decode_admission_evaluation(&inputs).unwrap();
            let refusal = if matches!(arm, FixtureArm::BoundaryMismatch) {
                let admission = evaluation
                    .admission()
                    .expect("fixture evidence must admit before the boundary changes");
                store
                    .disable_approval(model_id, decode_fingerprint, "boundary mutation", 2)
                    .unwrap();
                assert!(
                    !store
                        .owned_decode_dispatch_admission_matches(
                            admission.profile_activation_epoch,
                            &admission.approval.model_id,
                            &admission.approval.decode_fingerprint,
                            &admission.approval.semantic_digest,
                            admission.approval.generation,
                        )
                        .unwrap(),
                    "{name} fixture must fail the dispatch boundary recheck"
                );
                ServingRefusal::NotCertified
            } else {
                if matches!(arm, FixtureArm::ApprovalDisabled) {
                    assert_eq!(
                        evaluation.refusal(),
                        Some(&OwnedDecodeAdmissionRefusal::ApprovalDisabled {
                            disabled_reason: "approval disabled pending certification".to_string(),
                        })
                    );
                }
                let approval = evaluation.approval();
                let admitted = evaluation.admission().is_some();
                serving_predicate(&ServingPredicateInputs {
                    approval_present: approval.is_some(),
                    approval_enabled: approval.is_some_and(|row| row.enabled),
                    approval_identity_matches: approval.is_some_and(|row| {
                        row.model_id == inputs.model_id
                            && row.decode_fingerprint == inputs.decode_fingerprint
                    }),
                    current_profile_matches: admitted,
                    current_epoch_valid: true,
                    certification_matches: admitted,
                    evidence_revisions_compatible: admitted,
                    gates_complete: admitted,
                    processing_fingerprint_matches: admitted,
                    runtime_config_digest_matches: admitted,
                    worker_path_matches: admitted,
                    constrained_identities_match: admitted,
                    artifacts_trusted: true,
                    identities_installed: true,
                    quarantined: false,
                    wire_bindings_literal: true,
                    scheduler_evidence_committed: true,
                })
                .expect_err("fixture must refuse")
            };
            assert_eq!(refusal.wire_id(), expected_wire, "{name}");
            let _ = std::fs::remove_dir_all(root);
        }
    }

    #[test]
    fn serving_ingest_requires_gguf_q4km_and_verified_q8_lineage() {
        let (root, descriptor) = temp_descriptor("serving-ingest");
        let store = SynapseStore::open(&descriptor).unwrap();

        let mut mlx = serving_artifact_request();
        mlx.source_format = "mlx-group-quant".to_string();
        assert!(matches!(
            store.ingest_serving_artifact(&mlx, 1),
            Err(SynapseStoreError::Decode(message)) if message.contains("MLX group-quant")
        ));

        let mut unverified = serving_artifact_request();
        unverified
            .q8_derivation
            .as_mut()
            .unwrap()
            .verified_derived_digest = "e".repeat(64);
        assert!(matches!(
            store.ingest_serving_artifact(&unverified, 1),
            Err(SynapseStoreError::Decode(message)) if message.contains("verified deterministic Q8")
        ));

        let artifact = store
            .ingest_serving_artifact(&serving_artifact_request(), 2)
            .unwrap();
        assert_eq!(artifact.source_format, "gguf");
        assert_eq!(artifact.source_quantization, "q4_k_m");
        assert_eq!(
            artifact.derived_digest.as_deref(),
            Some("c".repeat(64).as_str())
        );
        assert_eq!(
            artifact.artifact_id,
            store
                .ingest_serving_artifact(&serving_artifact_request(), 3)
                .unwrap()
                .artifact_id,
            "ingest identity is deterministic for identical verified lineage"
        );

        let catalog_fingerprint = serving_catalog_fingerprint();
        let record = complete_serving_certification(&artifact, &catalog_fingerprint);
        store.store_serving_certification(&record, 4).unwrap();
        let mismatched_approval = store.approve_serving_catalog(
            &"f".repeat(64),
            &record.record_id,
            "principal:operator",
            5,
        );
        assert!(mismatched_approval
            .expect_err("a catalog fingerprint cannot borrow another record")
            .to_string()
            .contains("does not match certification"));
        let approved = store
            .approve_serving_catalog(
                &catalog_fingerprint,
                &record.record_id,
                "principal:operator",
                6,
            )
            .unwrap();
        assert_eq!(approved.artifact_id, artifact.artifact_id);
        assert_eq!(approved.certification_record_id, record.record_id);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn ordinary_disable_rejects_new_work_but_unloads_only_after_active_completion() {
        let (root, descriptor) = temp_descriptor("serving-disable");
        let store = SynapseStore::open(&descriptor).unwrap();
        let (artifact, catalog_fingerprint) = configure_serving_catalog(&store);
        assert_eq!(
            store
                .admit_serving_session("active-session", &catalog_fingerprint, 4)
                .unwrap(),
            ServingSessionAdmission::Admitted {
                session_id: "active-session".to_string(),
                catalog_fingerprint: catalog_fingerprint.clone(),
                approval_generation: 0,
            }
        );
        assert!(matches!(
            store
                .retain_serving_state("retained-state", &catalog_fingerprint, 5)
                .unwrap(),
            ServingContinuationAdmission::Admitted { .. }
        ));
        store
            .set_serving_artifact_gc_pin(&artifact.artifact_id, true)
            .unwrap();

        let disabled = store
            .disable_serving_catalog(&catalog_fingerprint, "maintenance", 6)
            .unwrap();
        assert_eq!(disabled.approval.state, ServingApprovalState::Disabled);
        assert_eq!(disabled.invalidated_retained_states, 1);
        assert_eq!(disabled.active_sessions, 1);
        assert_eq!(disabled.termination_requested_sessions, 0);
        assert!(!disabled.unload_artifact);
        assert_eq!(
            store
                .serving_session("active-session")
                .unwrap()
                .unwrap()
                .state,
            ServingSessionState::Active,
            "ordinary disable must not terminate active execution"
        );
        assert_eq!(
            store.admit_serving_continuation("retained-state").unwrap(),
            ServingContinuationAdmission::Refused {
                reason: ServingRefusal::ArtifactDisabled,
            }
        );
        assert_eq!(
            store
                .admit_serving_session("new-session", &catalog_fingerprint, 7)
                .unwrap(),
            ServingSessionAdmission::Refused {
                reason: ServingRefusal::ArtifactDisabled,
            }
        );
        let completed = store.complete_serving_session("active-session", 8).unwrap();
        assert_eq!(
            completed.session.terminal_reason.as_deref(),
            Some("completed")
        );
        assert!(completed.unload_artifact);
        assert_eq!(
            store
                .serving_artifact_gc_disposition(&artifact.artifact_id)
                .unwrap(),
            ArtifactGcDisposition::Pinned,
            "a GC pin persists but did not block serving disablement or unload readiness"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn emergency_revoke_terminates_at_the_next_committed_boundary() {
        let (root, descriptor) = temp_descriptor("serving-revoke");
        let store = SynapseStore::open(&descriptor).unwrap();
        let (artifact, catalog_fingerprint) = configure_serving_catalog(&store);
        assert!(matches!(
            store
                .admit_serving_session("active-session", &catalog_fingerprint, 4)
                .unwrap(),
            ServingSessionAdmission::Admitted { .. }
        ));
        assert!(matches!(
            store
                .retain_serving_state("retained-state", &catalog_fingerprint, 5)
                .unwrap(),
            ServingContinuationAdmission::Admitted { .. }
        ));
        store
            .set_serving_artifact_gc_pin(&artifact.artifact_id, true)
            .unwrap();

        let revoked = store
            .revoke_serving_catalog(&catalog_fingerprint, "critical compromise", 6)
            .unwrap();
        assert_eq!(revoked.approval.state, ServingApprovalState::Revoked);
        assert_eq!(revoked.invalidated_retained_states, 1);
        assert_eq!(revoked.termination_requested_sessions, 1);
        assert_eq!(revoked.active_sessions, 1);
        assert!(!revoked.unload_artifact);
        assert_eq!(
            store
                .serving_session("active-session")
                .unwrap()
                .unwrap()
                .state,
            ServingSessionState::TerminationRequested
        );
        assert_eq!(
            store
                .admit_serving_session("new-session", &catalog_fingerprint, 7)
                .unwrap(),
            ServingSessionAdmission::Refused {
                reason: ServingRefusal::ArtifactRevoked,
            }
        );
        assert_eq!(
            store.admit_serving_continuation("retained-state").unwrap(),
            ServingContinuationAdmission::Refused {
                reason: ServingRefusal::ArtifactRevoked,
            }
        );
        assert_eq!(
            store
                .commit_serving_session_boundary("active-session", 17, 8)
                .unwrap(),
            ServingBoundaryOutcome::Terminated {
                terminal_reason: "artifact_revoked".to_string(),
                tokens_emitted: 17,
                unload_artifact: true,
            }
        );
        let terminal = store.serving_session("active-session").unwrap().unwrap();
        assert_eq!(terminal.state, ServingSessionState::Terminated);
        // 17, not the pre-revoke count: the transaction that OBSERVES the
        // revocation also writes the new committed count before returning
        // Terminated. So the prefix a caller publishes on a terminal boundary
        // was committed by the revoke-observing transaction, not before it --
        // which is what makes publishing that progress correct rather than a
        // leak past the fence.
        assert_eq!(terminal.committed_token_count, 17);
        assert_eq!(
            terminal.terminal_reason.as_deref(),
            Some("artifact_revoked")
        );
        assert_eq!(
            store
                .serving_artifact_gc_disposition(&artifact.artifact_id)
                .unwrap(),
            ArtifactGcDisposition::Pinned,
            "blob pins affect GC only and never delay emergency revocation"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    fn populate_every_restore_table(store: &SynapseStore, seed: &str) {
        let catalog_fingerprint = hex::encode(Sha256::digest(seed.as_bytes()));
        let artifact_id = hex::encode(Sha256::digest(format!("artifact-{seed}").as_bytes()));
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
                    INSERT INTO result_pages (
                        request_digest, page_no, page_json, committed_at
                    ) VALUES ('digest-{seed}', 0, X'7B7D', 2);
                    INSERT INTO remote_checkpoints (
                        request_digest, item_id, result, page_no, committed_at
                    ) VALUES ('digest-{seed}', 'item-{seed}', X'7B7D', 0, 2);
                    INSERT INTO alias_rows (
                        fingerprint_a, fingerprint_b, valid_from_ms, evidence_json
                    ) VALUES ('alias-a-{seed}', 'alias-b-{seed}', 1, '{{}}');
                    INSERT INTO models (
                        model_id, engine, task, fingerprint, config_json, created_ms, updated_ms
                    ) VALUES ('model-{seed}', 'engine', 'embed', 'fingerprint-{seed}', X'7B7D', 1, 2);
                    INSERT INTO cert_rows (
                        certification_class, assurance_class, status, key_hash,
                        machine_profile_hash, fingerprint, certified_at_ms, os_build,
                        module_generation, evidence_json
                    ) VALUES ('embedding', 'measured', 'certified', 'key-hash-{seed}',
                              'machine-{seed}', 'cert-{seed}', 1, 'os', 1, '{{}}');
                    INSERT INTO perf_rows (
                        machine_profile_hash, model_id, workload, numeric_profile_id,
                        fingerprint, engine, measured_at_ms, os_build, module_generation,
                        throughput_tok_s, cold_load_ms, single_item_latency_p50_ms, details_json
                    ) VALUES ('machine-{seed}', 'model-{seed}', 'embed.batch', 'numeric-{seed}',
                              'perf-{seed}', 'engine', 1, 'os', 1, 3.0, 4.0, 5.0, '{{}}');
                    INSERT INTO knob_assignments (
                        machine_profile_hash, workload, knob, model_id, numeric_profile_id,
                        fingerprint, engine, measured_at_ms, os_build, module_generation,
                        throughput_tok_s, single_item_latency_p50_ms
                    ) VALUES ('machine-{seed}', 'embed.batch', 'balanced', 'model-{seed}',
                              'numeric-{seed}', 'knob-{seed}', 'engine', 1, 'os', 1, 3.0, 5.0);
                    INSERT INTO remote_url_bindings (remote_profile_hash, last_base_url)
                    VALUES ('remote-{seed}', 'https://{seed}.example.test');
                    INSERT INTO approvals (
                        schema_revision, model_id, decode_fingerprint, enabled,
                        grammar_enabled, updated_at_ms, evidence_requirements_revision,
                        semantic_digest, generation, fencing_metadata
                    ) VALUES ('schema-{seed}', 'approved-model-{seed}', 'decode-{seed}',
                              1, 1, 1, 'evidence-{seed}', 'digest-{seed}', 3, '{{}}');
                    INSERT INTO approval_digest_corruption_events (
                        model_id, decode_fingerprint, observed_digest, recomputed_digest,
                        observed_at_ms
                    ) VALUES ('approved-model-{seed}', 'decode-{seed}', 'bad', 'good', 1);
                    INSERT INTO approval_migration_markers (
                        seed_revision, schema_revision, seed_digest, applied_at_ms, row_count
                    ) VALUES ('seed-{seed}', 'schema-{seed}', 'seed-digest-{seed}', 1, 1);
                    UPDATE profile_state
                    SET snapshot_json = '{{}}', revisioned_machine_profile_hash = 'profile-{seed}',
                        profile_activation_epoch = 1
                    WHERE id = 0;
                    INSERT INTO profile_rotation_events (
                        event_id, new_revisioned_machine_profile_hash,
                        new_profile_activation_epoch, changed_fields_json,
                        current_snapshot_json, observed_at_ms, module_generation, created_at_ms
                    ) VALUES ('rotation-{seed}', 'profile-{seed}', 1, '[]', '{{}}', 1, 1, 1);
                    INSERT INTO profile_rotation_certification_outcomes (
                        event_id, model_id, decode_fingerprint, outcome_state
                    ) VALUES ('rotation-{seed}', 'approved-model-{seed}', 'decode-{seed}', 'passed');
                    INSERT INTO cert_row_rebuild_events (outcome, detail, occurred_at_ms)
                    VALUES ('rebuilt-{seed}', 'detail', 1);
                    INSERT INTO serving_artifacts (
                        artifact_id, model_id, source_format, source_quantization,
                        source_digest, artifact_json, ingested_at_ms
                    ) VALUES ('{artifact_id}', 'model-{seed}', 'gguf', 'q4_k_m',
                              'source-{seed}', '{{}}', 1);
                    INSERT INTO serving_certification_records (
                        certification_record_id, catalog_fingerprint, artifact_id,
                        record_json, recorded_at_ms
                    ) VALUES ('serving-cert-{seed}', '{catalog_fingerprint}', '{artifact_id}', '{{}}', 1);
                    INSERT INTO serving_approvals (
                        catalog_fingerprint, certification_record_id, artifact_id, state,
                        approved_by, approved_at_ms, updated_at_ms, generation, semantic_digest
                    ) VALUES ('{catalog_fingerprint}', 'serving-cert-{seed}', '{artifact_id}',
                              'enabled', 'operator-{seed}', 1, 1, 2, 'pending');
                    INSERT INTO serving_sessions (
                        session_id, catalog_fingerprint, approval_generation, state,
                        created_at_ms, updated_at_ms
                    ) VALUES ('session-{seed}', '{catalog_fingerprint}', 2, 'active', 1, 1);
                    INSERT INTO serving_retained_states (
                        state_id, catalog_fingerprint, valid, created_at_ms, updated_at_ms
                    ) VALUES ('state-{seed}', '{catalog_fingerprint}', 1, 1, 1);
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
                let mut approval = ServingApprovalRecord {
                    schema_revision: SERVING_APPROVAL_SCHEMA_REVISION.to_string(),
                    catalog_fingerprint: catalog_fingerprint.clone(),
                    certification_record_id: format!("serving-cert-{seed}"),
                    artifact_id: artifact_id.clone(),
                    state: ServingApprovalState::Enabled,
                    reason: None,
                    approved_by: format!("operator-{seed}"),
                    approved_at_ms: 1,
                    updated_at_ms: 1,
                    generation: 2,
                    semantic_digest: String::new(),
                };
                approval.semantic_digest = approval.expected_digest();
                connection.execute(
                    "UPDATE serving_approvals SET semantic_digest = ?1
                     WHERE catalog_fingerprint = ?2",
                    params![approval.semantic_digest, catalog_fingerprint],
                )?;
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

    #[cfg(unix)]
    fn verification_fixture() -> (
        std::path::PathBuf,
        SynapseStore,
        CatalogInstallRecord,
        std::path::PathBuf,
        String,
    ) {
        let (root, descriptor) = temp_descriptor("catalog-verification");
        let store = SynapseStore::open(&descriptor).unwrap();
        let path = root.join("weights");
        std::fs::write(&path, b"original").unwrap();
        let digest = crate::sha256_file(&path).unwrap();
        let install = CatalogInstallRecord {
            catalog_id: "model".into(),
            manifest_digest: "manifest".into(),
            backend: "metal".into(),
            members: vec![CatalogInstallMember {
                path: "weights".into(),
                digest: digest.clone(),
            }],
        };
        store.store.with_conn_fenced(|tx| {
            tx.execute("INSERT INTO catalog_installs VALUES ('model','manifest','metal')", [])?;
            tx.execute("INSERT INTO catalog_install_members VALUES ('model','manifest','metal','weights',?1)", params![digest])?;
            Ok(())
        }).unwrap();
        (root, store, install, path, digest)
    }

    #[cfg(unix)]
    #[test]
    fn catalog_verification_unchanged_cold_load_skips_hash_and_removal_clears_row() {
        let (root, store, install, path, digest) = verification_fixture();
        let calls = std::cell::Cell::new(0);
        for _ in 0..2 {
            let result = crate::verified_catalog_file_digest(
                &store,
                &install,
                "weights",
                &digest,
                &path,
                |p| {
                    calls.set(calls.get() + 1);
                    crate::sha256_file(p).ok()
                },
            )
            .unwrap();
            assert_eq!(result.as_deref(), Some(digest.as_str()));
        }
        assert_eq!(calls.get(), 1);
        assert_eq!(table_count(&store, "catalog_file_verifications"), 1);
        store
            .store
            .with_conn_fenced(|tx| {
                tx.execute("DELETE FROM catalog_install_members", [])?;
                Ok(())
            })
            .unwrap();
        assert_eq!(table_count(&store, "catalog_file_verifications"), 0);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn catalog_verification_touch_forces_rehash() {
        let (root, store, install, path, digest) = verification_fixture();
        crate::verified_catalog_file_digest(&store, &install, "weights", &digest, &path, |p| {
            crate::sha256_file(p).ok()
        })
        .unwrap();
        let old = std::fs::metadata(&path).unwrap().modified().unwrap();
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(old + std::time::Duration::from_secs(1))
            .unwrap();
        let calls = std::cell::Cell::new(0);
        let result =
            crate::verified_catalog_file_digest(&store, &install, "weights", &digest, &path, |p| {
                calls.set(calls.get() + 1);
                crate::sha256_file(p).ok()
            })
            .unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!(result.as_deref(), Some(digest.as_str()));
        std::fs::remove_dir_all(root).unwrap();
    }

    /// A file that changes while it is being hashed can't vouch for the bytes
    /// that are loaded afterwards. The hash closure here returns the expected
    /// digest (as it would if it read the file before the write) but bumps the
    /// file's mtime mid-hash. The result must not count as verified, and no
    /// verification row may be recorded, so the next load hashes in full.
    #[cfg(unix)]
    #[test]
    fn catalog_verification_file_changed_during_hash_is_not_recorded() {
        let (root, store, install, path, digest) = verification_fixture();
        let result =
            crate::verified_catalog_file_digest(&store, &install, "weights", &digest, &path, |p| {
                let old = std::fs::metadata(p).unwrap().modified().unwrap();
                std::fs::File::options()
                    .write(true)
                    .open(p)
                    .unwrap()
                    .set_modified(old + std::time::Duration::from_secs(1))
                    .unwrap();
                Some(digest.clone())
            })
            .unwrap();
        assert_eq!(result, None, "a file changed mid-hash must not verify");
        let calls = std::cell::Cell::new(0);
        crate::verified_catalog_file_digest(&store, &install, "weights", &digest, &path, |p| {
            calls.set(calls.get() + 1);
            crate::sha256_file(p).ok()
        })
        .unwrap();
        assert_eq!(calls.get(), 1, "no verification row may have been recorded");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn catalog_verification_content_change_with_restored_mtime_is_refused() {
        let (root, store, install, path, digest) = verification_fixture();
        crate::verified_catalog_file_digest(&store, &install, "weights", &digest, &path, |p| {
            crate::sha256_file(p).ok()
        })
        .unwrap();
        let old = std::fs::metadata(&path).unwrap().modified().unwrap();
        std::fs::write(&path, b"tampered").unwrap();
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(old)
            .unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().modified().unwrap(), old);
        let calls = std::cell::Cell::new(0);
        let result =
            crate::verified_catalog_file_digest(&store, &install, "weights", &digest, &path, |p| {
                calls.set(calls.get() + 1);
                crate::sha256_file(p).ok()
            })
            .unwrap();
        assert_ne!(
            result.as_deref(),
            Some(digest.as_str()),
            "corrupted artifact must be refused"
        );
        assert_eq!(calls.get(), 1);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn catalog_verification_startup_discards_orphaned_rows() {
        let (root, store, install, path, digest) = verification_fixture();
        crate::verified_catalog_file_digest(&store, &install, "weights", &digest, &path, |p| {
            crate::sha256_file(p).ok()
        })
        .unwrap();
        store.store.with_conn_fenced(|tx| {
            tx.execute("INSERT INTO catalog_file_verifications VALUES ('orphan','manifest','metal','weights',?1,'1','2','3','4','5')", params![digest])?;
            Ok(())
        }).unwrap();
        assert_eq!(table_count(&store, "catalog_file_verifications"), 2);
        store
            .reconcile_persisted_registrations(&|_| false, &FakeBlobCache::default())
            .unwrap();
        assert_eq!(table_count(&store, "catalog_file_verifications"), 1);
        assert!(store
            .catalog_file_verified(
                &install,
                "weights",
                &digest,
                &crate::catalog_file_stamp(&path).unwrap()
            )
            .unwrap());
        std::fs::remove_dir_all(root).unwrap();
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
    fn restore_import_drops_resurrected_certification_and_all_untrusted_state() {
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
            if *table == "approvals" {
                assert_eq!(
                    report.disabled_on_import.get(*table).copied(),
                    Some(keep_rows[*table].len() as u64)
                );
                let restored_rows = table_rows(&restored, table);
                let captured_rows = &keep_rows[*table];
                assert_eq!(restored_rows.len(), captured_rows.len());
                for (restored_row, captured_row) in restored_rows.iter().zip(captured_rows.iter()) {
                    // All non-state identity/provenance columns match captured row
                    assert_eq!(restored_row[0], captured_row[0]);
                    assert_eq!(restored_row[1], captured_row[1]);
                    assert_eq!(restored_row[2], captured_row[2]);
                    assert_eq!(restored_row[3], captured_row[3]);
                    // enabled and grammar_enabled must be forced to 0
                    assert_eq!(restored_row[4], rusqlite::types::Value::Integer(0));
                    assert_eq!(restored_row[5], rusqlite::types::Value::Integer(0));
                    // disabled_reason must be set
                    assert_eq!(
                        restored_row[6],
                        rusqlite::types::Value::Text(RESTORE_DISABLED_REASON.to_string())
                    );
                    assert_eq!(restored_row[7], captured_row[7]);
                    assert_eq!(restored_row[8], captured_row[8]);
                    assert_eq!(restored_row[10], captured_row[10]);
                    let captured_gen = match captured_row[12] {
                        rusqlite::types::Value::Integer(gen) => gen,
                        _ => panic!("generation is integer"),
                    };
                    assert_eq!(
                        restored_row[12],
                        rusqlite::types::Value::Integer(captured_gen + 1)
                    );
                    assert_eq!(restored_row[13], captured_row[13]);
                }
            } else if *table == "serving_approvals" {
                assert_eq!(
                    report.disabled_on_import.get(*table).copied(),
                    Some(keep_rows[*table].len() as u64)
                );
                let restored_rows = table_rows(&restored, table);
                let captured_rows = &keep_rows[*table];
                assert_eq!(restored_rows.len(), captured_rows.len());
                for (restored_row, captured_row) in restored_rows.iter().zip(captured_rows.iter()) {
                    assert_eq!(restored_row[0], captured_row[0]);
                    assert_eq!(restored_row[1], captured_row[1]);
                    assert_eq!(restored_row[2], captured_row[2]);
                    // state must be forced to disabled
                    assert_eq!(
                        restored_row[3],
                        rusqlite::types::Value::Text("disabled".to_string())
                    );
                    assert_eq!(
                        restored_row[4],
                        rusqlite::types::Value::Text(RESTORE_DISABLED_REASON.to_string())
                    );
                    assert_eq!(restored_row[5], captured_row[5]);
                    assert_eq!(restored_row[6], captured_row[6]);
                    let captured_gen = match captured_row[8] {
                        rusqlite::types::Value::Integer(gen) => gen,
                        _ => panic!("generation is integer"),
                    };
                    assert_eq!(
                        restored_row[8],
                        rusqlite::types::Value::Integer(captured_gen + 1)
                    );
                }
            } else {
                assert_eq!(
                    table_rows(&restored, table),
                    keep_rows[*table],
                    "trust table {table} was not restored row-for-row"
                );
            }
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
        let restored_catalog = hex::encode(Sha256::digest(b"capture"));
        assert_eq!(
            restored
                .admit_serving_session("post-restore-session", &restored_catalog, 3)
                .unwrap(),
            ServingSessionAdmission::Refused {
                reason: ServingRefusal::ArtifactDisabled,
            },
            "a restored approval refuses admission because it is forced disabled"
        );
        drop(restored);
        let _ = std::fs::remove_dir_all(capture_root);
        let _ = std::fs::remove_dir_all(live_root);
    }

    #[test]
    fn restore_import_forces_imported_approvals_disabled_preventing_resurrection() {
        let (capture_root, capture_descriptor) = temp_descriptor("restore-capture-disabled-hazard");
        let capture = SynapseStore::open(&capture_descriptor).unwrap();

        // 1. Seed capture with an ENABLED serving approval.
        let (artifact, catalog_fingerprint) = configure_serving_catalog(&capture);
        let serving_before = capture
            .serving_approval(&catalog_fingerprint)
            .unwrap()
            .expect("serving approval must exist in capture");
        assert_eq!(
            serving_before.state,
            ServingApprovalState::Enabled,
            "capture must seed an enabled serving approval"
        );
        assert!(matches!(
            capture
                .admit_serving_session("capture-session", &catalog_fingerprint, 10)
                .unwrap(),
            ServingSessionAdmission::Admitted { .. }
        ));

        // 2. Seed capture with an ENABLED decode approval in `approvals` table.
        let model_id = "qwen3-0.6b-decode-f16";
        let decode_fingerprint = hex::encode(Sha256::digest(b"decode-fingerprint-hazard"));
        capture
            .store
            .with_conn(|conn| {
                let mut row = ApprovalRow {
                    schema_revision: APPROVAL_SCHEMA_REVISION.to_string(),
                    model_id: model_id.to_string(),
                    decode_fingerprint: decode_fingerprint.clone(),
                    enabled: true,
                    grammar_enabled: true,
                    disabled_reason: None,
                    approved_by: Some("principal:operator".to_string()),
                    approved_at_ms: Some(10),
                    updated_at_ms: 10,
                    evidence_requirements_revision:
                        APPROVAL_EVIDENCE_REQUIREMENTS_REVISION.to_string(),
                    semantic_digest: String::new(),
                    row_id: 1,
                    generation: 0,
                    fencing_metadata: serde_json::json!({}),
                };
                row.semantic_digest = row.expected_digest().map_err(to_sql_error)?;
                conn.execute(
                    "INSERT INTO approvals (
                        schema_revision, model_id, decode_fingerprint, enabled, grammar_enabled,
                        disabled_reason, approved_by, approved_at_ms, updated_at_ms,
                        evidence_requirements_revision, semantic_digest, generation, fencing_metadata
                    ) VALUES (?1, ?2, ?3, 1, 1, NULL, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                    params![
                        &row.schema_revision,
                        &row.model_id,
                        &row.decode_fingerprint,
                        &row.approved_by,
                        row.approved_at_ms.map(|v| v as i64),
                        row.updated_at_ms as i64,
                        &row.evidence_requirements_revision,
                        &row.semantic_digest,
                        row.generation as i64,
                        "{}",
                    ],
                )?;
                Ok(())
            })
            .unwrap();

        let capture_path = sqlite_path(&capture_descriptor).unwrap();
        drop(capture);

        // 3. Restore capture into a clean live store.
        let (live_root, live_descriptor) = temp_descriptor("restore-live-disabled-hazard");
        let report = SynapseStore::restore_from_capture(&live_descriptor, &capture_path).unwrap();
        let restored = SynapseStore::open(&live_descriptor).unwrap();

        // 4. Assert approvals table row is forced disabled on restore.
        let restored_approval = restored
            .get_approval(model_id, &decode_fingerprint)
            .unwrap()
            .expect("approvals row must exist in restored store");
        assert!(
            !restored_approval.enabled,
            "approvals table row must be forced disabled on restore"
        );
        assert!(
            !restored_approval.grammar_enabled,
            "approvals table grammar_enabled must be 0 on restore"
        );
        assert_eq!(
            restored_approval.disabled_reason.as_deref(),
            Some(RESTORE_DISABLED_REASON)
        );

        assert_eq!(
            report.disabled_on_import.get("approvals").copied(),
            Some(1),
            "approvals table rows forced disabled mismatch"
        );
        assert_eq!(
            report.disabled_on_import.get("serving_approvals").copied(),
            Some(1),
            "serving_approvals table rows forced disabled mismatch"
        );

        // Serving predicate check for approvals row refusal.
        let predicate_refusal = crate::owned_decode_routing::lane::serving_predicate(
            &crate::owned_decode_routing::lane::ServingPredicateInputs {
                approval_present: true,
                approval_enabled: restored_approval.enabled,
                ..Default::default()
            },
        );
        assert_eq!(
            predicate_refusal,
            Err(crate::owned_decode_routing::lane::ServingRefusal::CutoverDisabled),
            "approvals row must refuse serving predicate with CutoverDisabled"
        );

        // 5. Assert serving_approvals table row is forced disabled on restore.
        let restored_serving = restored
            .serving_approval(&catalog_fingerprint)
            .unwrap()
            .expect("serving_approvals row must exist in restored store");
        assert_eq!(
            restored_serving.state,
            ServingApprovalState::Disabled,
            "serving_approvals table row must be forced disabled on restore"
        );
        assert_eq!(
            restored_serving.reason.as_deref(),
            Some(RESTORE_DISABLED_REASON)
        );

        // 6. Re-derive certification records in the restored store.
        // If the approval came back enabled, this re-derivation would resurrect
        // serving admission without operator review.
        restored
            .ingest_serving_artifact(&serving_artifact_request(), 100)
            .unwrap();
        let record = complete_serving_certification(&artifact, &catalog_fingerprint);
        restored.store_serving_certification(&record, 200).unwrap();

        // 7. Verify serving is STILL refused with ArtifactDisabled.
        let session_outcome = restored
            .admit_serving_session("post-restore-probe", &catalog_fingerprint, 300)
            .unwrap();
        assert_eq!(
            session_outcome,
            ServingSessionAdmission::Refused {
                reason: ServingRefusal::ArtifactDisabled,
            },
            "serving_approvals table must refuse admission even after certification re-derivation"
        );

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

        let error = live
            .import_migrated_capture(&capture_path, Some("approvals"))
            .unwrap_err();
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
        let before = table_rows(&live, "approvals");
        let error =
            SynapseStore::restore_from_capture(&live_descriptor, &capture_path).unwrap_err();
        let rendered = error.to_string();
        assert!(rendered.contains("target store lease"), "{rendered}");
        assert!(rendered.contains("store.db"), "{rendered}");
        assert_eq!(table_rows(&live, "approvals"), before);
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
        capture_store.migrate(NAMESPACE, &MIGRATIONS[..3]).unwrap();
        capture_store
            .with_conn(|connection| {
                connection.execute(
                    "INSERT INTO alias_rows (
                         fingerprint_a, fingerprint_b, valid_from_ms, evidence_json
                     ) VALUES ('behind-a', 'behind-b', 1, '{}')",
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
        assert_eq!(table_count(&restored, "alias_rows"), 1);
        assert_eq!(table_count(&restored, "cert_rows"), 0);
        drop(restored);
        let migrated_capture = SynapseStore::open(&capture_descriptor).unwrap();
        assert_eq!(table_count(&migrated_capture, "alias_rows"), 1);
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

        let vacuum_count_before = maintenance_vacuum_count();
        let reopened = SynapseStore::open(&descriptor).unwrap();
        let freelist_after = reopened.freelist_count().unwrap();
        let file_size_after = std::fs::metadata(&db_path).unwrap().len();
        let vacuum_count_after = maintenance_vacuum_count();

        assert_eq!(freelist_after, 0);
        assert!(
            file_size_after < file_size_before / 10,
            "expected file to shrink: before={file_size_before}, after={file_size_after}"
        );
        assert_eq!(vacuum_count_after, vacuum_count_before + 1);

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

        let vacuum_count_before = maintenance_vacuum_count();
        let reopened = SynapseStore::open(&descriptor).unwrap();
        let freelist_after = reopened.freelist_count().unwrap();
        let vacuum_count_after = maintenance_vacuum_count();

        // Freelist count must remain unchanged because no VACUUM ran.
        assert_eq!(freelist_after, freelist_before);
        assert_eq!(vacuum_count_after, vacuum_count_before);

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
                None,
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
        let mut free_form = owned_seed_model_config("qwen3-0.6b-decode-f16");
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
        let mut free_form = owned_seed_model_config("qwen3-0.6b-decode-f16");
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
            let mut config = owned_seed_model_config("qwen3-0.6b-decode-f16");
            config.model_id = model_id.to_string();
            config.engine = engine.to_string();
            config.task = task.to_string();
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
