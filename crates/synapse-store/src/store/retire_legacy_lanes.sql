CREATE TEMP TABLE retired_registrations AS
SELECT model_id FROM models
WHERE engine IN ('ane', 'ane-coreml-worker', 'coreml', 'neural_engine',
                 'owned-metal-decode', 'owned-decode', 'owned_metal_decode')
   OR (task IN ('embed', 'rerank') AND CASE
       WHEN json_valid(CAST(config_json AS TEXT))
       THEN json_type(CAST(config_json AS TEXT), '$.engine_identity.build_flags.profile') IS NULL
       ELSE 1 END);

DELETE FROM knob_assignments WHERE model_id IN (SELECT model_id FROM retired_registrations);
DELETE FROM models WHERE model_id IN (SELECT model_id FROM retired_registrations);
DROP TABLE retired_registrations;

DROP TABLE alias_rows;
DROP TABLE approval_digest_corruption_events;
DROP TABLE approval_migration_markers;
DROP TABLE approvals;
DROP TABLE cert_row_rebuild_events;
DROP TABLE cert_rows;
DROP TABLE perf_rows;
DROP TABLE profile_rotation_certification_outcomes;
DROP TABLE serving_sessions;
DROP TABLE serving_retained_states;
DROP TABLE serving_approvals;
DROP TABLE serving_certification_records;
DROP TABLE serving_artifacts;
ALTER TABLE profile_state DROP COLUMN certification_stale_since_ms;
