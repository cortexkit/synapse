use super::*;
use synapse_owned_decode::owned_decode_certification::{
    ArtifactLineage, CertificationGateResult, CertificationRecord, CertificationUnit,
    DerivedArtifactLineage, EmbedLoadResult, KvMatrixCandidate, KvMatrixResult,
    KvSelectionEvidence, M5MeasurementEvidence, MachineScopedEvidence, MachineTuple,
    MtpSpeedRepetition, MtpSpeedResult, PlatformEnvelopeResult, ProbeEvidence,
    RuntimeConfiguration, SerialOracleFidelityResult, SpeculativeSerialFidelityResult,
    SpeculativeTelemetry, TimingArmEvidence, TokenFidelityEvidence, TokenTapResult,
    AGENTIC_BATTERY_ID, EMBED_LOAD_ID, LLAMA_CPP_ORACLE_REVISION, PLATFORM_ENVELOPE_ID,
    WAVE_1_CONTEXT_CEILING_TOKENS,
};

pub(super) fn serving_catalog_fingerprint() -> String {
    "d".repeat(64)
}

pub(super) fn serving_artifact_request() -> ArtifactIngestRequest {
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

pub(super) fn complete_serving_certification(
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
            CertificationGateResult::SpeculativeSerialFidelity(SpeculativeSerialFidelityResult {
                manifest_digest: manifest_digest.clone(),
                battery_id: AGENTIC_BATTERY_ID.to_string(),
                serial_certification_id: "serial-certification".to_string(),
                leviathan_greedy_acceptance: true,
                fidelity: TokenFidelityEvidence {
                    generated_token_ids_match: true,
                    stop_position_matches: true,
                    finish_reason_matches: true,
                },
            }),
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
