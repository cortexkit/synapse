INSERT INTO alias_rows (fingerprint_a, fingerprint_b, valid_from_ms, evidence_json)
VALUES ('a', 'b', 1, '{}');
INSERT INTO approvals (schema_revision, model_id, decode_fingerprint, enabled,
    grammar_enabled, updated_at_ms, evidence_requirements_revision, semantic_digest)
VALUES ('schema', 'qwen-decode', 'decode', 1, 0, 1, 'evidence', 'digest');
INSERT INTO approval_digest_corruption_events
    (model_id, decode_fingerprint, observed_digest, recomputed_digest, observed_at_ms)
VALUES ('qwen-decode', 'decode', 'bad', 'good', 1);
INSERT INTO approval_migration_markers
    (seed_revision, schema_revision, seed_digest, applied_at_ms, row_count)
VALUES ('seed', 'schema', 'digest', 1, 1);
INSERT INTO cert_rows (certification_class, assurance_class, status, key_hash,
    machine_profile_hash, fingerprint, certified_at_ms, os_build, module_generation, evidence_json)
VALUES ('embedding', 'measured', 'certified', 'machine', 'machine', 'embed-fp', 1, 'os', 1, '{}'),
       ('rerank', 'measured', 'certified', 'machine', 'machine', 'rerank-fp', 1, 'os', 1, '{}'),
       ('legacy_owned_decode', 'measured', 'uncertified', 'machine', 'machine', 'legacy-decode-fp', 1, 'os', 1, '{}');
INSERT INTO cert_rows (certification_class, assurance_class, status, key_hash,
    revisioned_machine_profile_hash, profile_activation_epoch, model_id, decode_fingerprint,
    evidence_schema_revision, fingerprint, certified_at_ms, os_build, module_generation, evidence_json,
    constraint_runtime_identities_digest)
VALUES ('measured_owned_decode', 'measured', 'certified', 'profile', 'profile', 1,
    'qwen-decode', 'decode', 'schema', 'decode-fp', 1, 'os', 1, '{}', 'constraints');
INSERT INTO cert_row_rebuild_events (outcome, detail, occurred_at_ms)
VALUES ('rebuilt', 'detail', 1);
INSERT INTO perf_rows VALUES ('machine', 'qwen-decode', 'generate', 'numeric',
    'decode', 'owned-metal-decode', 1, 'os', 1, 3.0, 4.0, 5.0, '{}');
INSERT INTO profile_rotation_events (event_id, new_revisioned_machine_profile_hash,
    new_profile_activation_epoch, changed_fields_json, current_snapshot_json,
    observed_at_ms, module_generation, created_at_ms)
VALUES ('rotation', 'profile', 1, '[]', '{}', 1, 1, 1);
INSERT INTO profile_rotation_certification_outcomes (event_id, model_id, decode_fingerprint, outcome_state)
VALUES ('rotation', 'qwen-decode', 'decode', 'passed');
UPDATE profile_state SET certification_stale_since_ms = 1 WHERE id = 0;
INSERT INTO serving_artifacts (artifact_id, model_id, source_format, source_quantization,
    source_digest, artifact_json, ingested_at_ms)
VALUES ('artifact', 'qwen-decode', 'gguf', 'q4_k_m', 'source', '{}', 1);
INSERT INTO serving_certification_records
    (certification_record_id, catalog_fingerprint, artifact_id, record_json, recorded_at_ms)
VALUES ('cert', 'catalog-fp', 'artifact', '{}', 1);
INSERT INTO serving_approvals (catalog_fingerprint, certification_record_id, artifact_id,
    state, approved_by, approved_at_ms, updated_at_ms, generation, semantic_digest)
VALUES ('catalog-fp', 'cert', 'artifact', 'enabled', 'operator', 1, 1, 1, 'digest');
INSERT INTO serving_sessions (session_id, catalog_fingerprint, approval_generation,
    state, created_at_ms, updated_at_ms)
VALUES ('session', 'catalog-fp', 1, 'active', 1, 1);
INSERT INTO serving_retained_states (state_id, catalog_fingerprint, valid, created_at_ms, updated_at_ms)
VALUES ('state', 'catalog-fp', 1, 1, 1);
INSERT INTO knob_assignments VALUES
    ('machine', 'embed', 'balanced', 'kept-metal', 'numeric', 'keep-fp', 'owned-metal', 1, 'os', 1, 3.0, 5.0),
    ('machine', 'generate', 'balanced', 'qwen-decode', 'numeric', 'decode', 'owned-metal-decode', 1, 'os', 1, 3.0, 5.0);
INSERT INTO catalog_self_checks (check_id, catalog_id, manifest_digest, backend, fingerprint,
    engine_identity_json, os_build, fixture_revision, state, generation)
VALUES ('check', 'kept-metal', 'manifest', 'metal', 'keep-fp', '{}', 'os', '1', 'passed', 1);
INSERT INTO jobs (job_id, request_key, request_digest, kind, module_generation, state,
    created_ms, updated_ms, result_retention_ttl_ms, terminal_at_ms, params_json)
VALUES ('kept-job', 'key', 'digest', 'embed.batch', 1, 'done', 1, 1, 100, 1, X'7B7D');
