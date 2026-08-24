//! Integration tests for owned-decode validation with a small deterministic record.
//!
//! They verify that incomplete, inconsistent, or unsafe decode results are rejected
//! before acceptance, so CI does not need the large production fixture.

use std::{collections::BTreeMap, fs, path::PathBuf};

use owned_decode_worker::{
    RetentionPreflight, StreamRequest, StreamingSupervisor, StreamingSupervisorError,
    WorkerDeathOutcome,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use synapse_core::{
    validate_artifact_serving_state, ArtifactServingState, DecodeMode, Fingerprint, FrameEnvelope,
    OneshotEnvelopeIdentity, OwnedDecodeRefusal, ProgressFrame, SessionStatusState,
    StreamFrameDisposition, StreamSequence, TerminalEnvelope, TerminalState, WorkerFrame,
};
use synapse_module::{
    owned_decode_certification::{
        certification_unit::CertificationGateResult, compare_streams, ArtifactLineage,
        CertificationError, CertificationGate, CertificationRecord, CertificationRegistry,
        CertificationRequest, CertificationUnit, EmbedLoadResult, KvMatrixCandidate,
        KvMatrixResult, KvSelectionEvidence, M5MeasurementEvidence, MachineScopedEvidence,
        MachineTuple as CertificationMachineTuple, MtpSpeedRepetition, MtpSpeedResult,
        PlatformEnvelopeResult, ProbeEvidence, RuntimeConfiguration, SerialOracleFidelityResult,
        SpeculativeSerialFidelityResult, SpeculativeTelemetry, TimingArmEvidence,
        TokenFidelityEvidence, TokenTapResult, AGENTIC_BATTERY_ID, EMBED_LOAD_ID,
        LLAMA_CPP_ORACLE_REVISION, PLATFORM_ENVELOPE_ID, WAVE_1_CONTEXT_CEILING_TOKENS,
    },
    owned_decode_routing::admission::{
        AdmissionRefusal, AdmissionRequest, ArtifactReservation, GenerationConfiguration,
        MachineTuple as AdmissionMachineTuple, PlatformEnvelope, ResidencyRouter,
        SessionKvConfiguration, WAVE_ONE_CONTEXT_CEILING_TOKENS, WAVE_ONE_MIN_UNIFIED_MEMORY_BYTES,
    },
};

const GIB: u64 = 1024 * 1024 * 1024;
const MANIFEST_SHA256: &str = "941f9bae8265c3bb0d35b4a3f69bb2ad509764ae08873a339f0af3b3643a070f";

fn battery_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../bench/fixtures/agentic-battery-v1")
}

fn battery_manifest() -> Value {
    serde_json::from_slice(
        &fs::read(battery_root().join("manifest.json")).expect("read agentic battery manifest"),
    )
    .expect("agentic battery manifest is valid JSON")
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn exact_fidelity() -> TokenFidelityEvidence {
    TokenFidelityEvidence {
        generated_token_ids_match: true,
        stop_position_matches: true,
        finish_reason_matches: true,
    }
}

fn lineage() -> ArtifactLineage {
    ArtifactLineage {
        artifact_id: "qwen3.8-27b-q4-k-m-native-mtp-v1".to_string(),
        model_id: "Qwen3.8-27B".to_string(),
        quantization: "GGUF-Q4_K_M-compatible".to_string(),
        source_digest: "qwen3.8-27b-source-digest".to_string(),
        derived: None,
    }
}

fn unit() -> CertificationUnit {
    CertificationUnit {
        base_artifact_id: lineage().artifact_id,
        native_mtp_head_digest: "native-mtp-head-digest".to_string(),
        depth_controller_gate_digest: "registered-depth-controller-gate".to_string(),
        catalog_fingerprint: "qwen3.8-27b-wave-1-catalog".to_string(),
    }
}

fn machine() -> CertificationMachineTuple {
    CertificationMachineTuple {
        machine_profile_hash: "certifying-m5-machine-profile".to_string(),
        macos_build: "25F84".to_string(),
        unified_memory_bytes: 128 * GIB,
    }
}

fn runtime() -> RuntimeConfiguration {
    RuntimeConfiguration {
        runtime_config_digest: "shipped-runtime-config".to_string(),
        runtime_revision: "wave-1-runtime-v1".to_string(),
    }
}

fn timing_arm(session_id: &str, tokens_per_second: f64) -> TimingArmEvidence {
    TimingArmEvidence {
        loaded_session_id: session_id.to_string(),
        machine_profile_hash: machine().machine_profile_hash,
        macos_build: machine().macos_build,
        ac_power_connected: true,
        one_minute_load_average: 1.5,
        mean_tokens_per_second: tokens_per_second,
    }
}

fn kv_candidates() -> Vec<KvMatrixCandidate> {
    [256, 512, 1024]
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
        .collect()
}

fn complete_record() -> CertificationRecord {
    let selected = kv_candidates()
        .into_iter()
        .find(|candidate| {
            (
                candidate.block_size_tokens,
                candidate.reused_prefix_bucket_tokens,
            ) == (1024, 4096)
        })
        .expect("the required KV matrix contains its selected candidate");
    CertificationRecord {
        record_id: "certifying-machine-run-1".to_string(),
        manifest_digest: MANIFEST_SHA256.to_string(),
        artifact_lineage: lineage(),
        unit: unit(),
        machine_evidence: MachineScopedEvidence {
            machine: machine(),
            probe: ProbeEvidence {
                probe_id: "agentic-battery-certification".to_string(),
                harness_revision: "agentic-battery-harness-v1".to_string(),
                observed_at_ms: 1,
            },
            runtime: runtime(),
            m5_measurement: M5MeasurementEvidence {
                measurement_id: "m5-head-forward-vs-backbone-step".to_string(),
                measurement_revision: "m5-head-cost-v1".to_string(),
                machine_profile_hash: machine().machine_profile_hash,
                base_artifact_id: unit().base_artifact_id,
                catalog_fingerprint: unit().catalog_fingerprint,
                native_mtp_head_digest: unit().native_mtp_head_digest,
                runtime_config_digest: runtime().runtime_config_digest,
                head_forward_ms: 1.25,
                backbone_step_ms: 5.0,
                controller_constants_digest: "m5-controller-constants".to_string(),
                registered: true,
                depth_zero_executes_no_head_work: true,
                positive_depth_chains_command_buffer: true,
            },
        },
        gate_results: vec![
            CertificationGateResult::SerialOracleFidelity(SerialOracleFidelityResult {
                manifest_digest: MANIFEST_SHA256.to_string(),
                battery_id: AGENTIC_BATTERY_ID.to_string(),
                oracle_revision: LLAMA_CPP_ORACLE_REVISION.to_string(),
                greedy_only: true,
                fidelity: exact_fidelity(),
            }),
            CertificationGateResult::SpeculativeSerialFidelity(SpeculativeSerialFidelityResult {
                manifest_digest: MANIFEST_SHA256.to_string(),
                battery_id: AGENTIC_BATTERY_ID.to_string(),
                serial_certification_id: "serial-certifying-machine-run-1".to_string(),
                leviathan_greedy_acceptance: true,
                fidelity: exact_fidelity(),
            }),
            CertificationGateResult::MtpSpeed(MtpSpeedResult {
                manifest_digest: MANIFEST_SHA256.to_string(),
                battery_id: AGENTIC_BATTERY_ID.to_string(),
                generated_token_window: 256,
                serial_warmup_last_three_tok_s: [50.0, 51.0, 49.0],
                mtp_warmup_last_three_tok_s: [75.0, 76.5, 73.5],
                repetitions: vec![
                    MtpSpeedRepetition {
                        serial: timing_arm("loaded-session-1", 50.0),
                        mtp: timing_arm("loaded-session-1", 75.0),
                    },
                    MtpSpeedRepetition {
                        serial: timing_arm("loaded-session-2", 51.0),
                        mtp: timing_arm("loaded-session-2", 76.5),
                    },
                    MtpSpeedRepetition {
                        serial: timing_arm("loaded-session-3", 49.0),
                        mtp: timing_arm("loaded-session-3", 73.5),
                    },
                ],
                telemetry: SpeculativeTelemetry {
                    proposed_depth: 4,
                    accepted_depth: 3,
                    acceptance_rate: 0.75,
                    verification_work: 1024,
                    controller_decisions_digest: "depth-controller-decisions".to_string(),
                },
            }),
            CertificationGateResult::KvMatrix(KvMatrixResult {
                manifest_digest: MANIFEST_SHA256.to_string(),
                machine_profile_hash: machine().machine_profile_hash,
                candidates: kv_candidates(),
                selection: KvSelectionEvidence {
                    selected,
                    continuation_token_ids_identical: true,
                    reused_token_count: 4096,
                    reused_block_count: 4,
                    prefill_dispatches_over_reused_range: 0,
                    cold_ttft_ms: 10.0,
                    warm_ttft_ms: 2.0,
                    close_restored_allocator_accounting: true,
                },
            }),
            CertificationGateResult::PlatformEnvelope(PlatformEnvelopeResult {
                manifest_digest: MANIFEST_SHA256.to_string(),
                envelope_id: PLATFORM_ENVELOPE_ID.to_string(),
                machine_profile_hash: machine().machine_profile_hash,
                macos_build: machine().macos_build,
                unified_memory_bytes: machine().unified_memory_bytes,
                reserved_embed_rerank_bytes: GIB,
                artifact_weight_bytes: 2 * GIB,
                kv_bytes_per_token: 1024 * 1024,
                mandatory_context_ceiling_tokens: WAVE_1_CONTEXT_CEILING_TOKENS,
                admitted_and_exercised_32k_session: true,
                exercised_reservation_accounting: true,
                exercised_kv_reuse: true,
                exercised_streaming: true,
                exercised_scheduler_interleaving: true,
            }),
            CertificationGateResult::EmbedLoad(EmbedLoadResult {
                manifest_digest: MANIFEST_SHA256.to_string(),
                workload_id: EMBED_LOAD_ID.to_string(),
                runtime_config_digest: runtime().runtime_config_digest,
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
                manifest_digest: MANIFEST_SHA256.to_string(),
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

fn certification_request() -> CertificationRequest {
    CertificationRequest {
        artifact_lineage: lineage(),
        unit: unit(),
        machine: machine(),
        runtime: runtime(),
    }
}

fn tokens_from_prompt_digest(prompt: &Value) -> Vec<u32> {
    prompt["sha256"]
        .as_str()
        .expect("pinned prompt digest")
        .as_bytes()
        .chunks_exact(8)
        .map(|chunk| {
            u32::from_str_radix(std::str::from_utf8(chunk).expect("hex digest chunk"), 16)
                .expect("hex digest chunk")
        })
        .collect()
}

#[test]
fn fidelity_run_records_all_twenty_pinned_prompts_before_registry_approval() {
    let manifest_root = battery_root();
    let manifest_bytes =
        fs::read(manifest_root.join("manifest.json")).expect("read manifest bytes");
    assert_eq!(sha256_hex(&manifest_bytes), MANIFEST_SHA256);
    let manifest = battery_manifest();
    assert_eq!(manifest["battery_revision"], AGENTIC_BATTERY_ID);
    assert_eq!(manifest["prompt_count"], 20);
    assert_eq!(
        manifest["llama_cpp_oracle"]["revision"],
        LLAMA_CPP_ORACLE_REVISION
    );
    assert_eq!(manifest["generation"]["mode"], "greedy");
    assert_eq!(manifest["generation"]["max_new_tokens"], 256);

    let prompts = manifest["prompts"].as_array().expect("pinned prompts");
    assert_eq!(prompts.len(), 20);
    let mut categories = BTreeMap::new();
    for (prompt_index, prompt) in prompts.iter().enumerate() {
        *categories
            .entry(prompt["category"].as_str().expect("prompt category"))
            .or_insert(0usize) += 1;
        let prompt_path = manifest_root.join(prompt["path"].as_str().expect("prompt path"));
        assert_eq!(
            sha256_hex(&fs::read(prompt_path).expect("read pinned prompt")),
            prompt["sha256"].as_str().expect("prompt digest"),
            "prompt fixture bytes must match the recorded digest for {}",
            prompt["id"]
        );
        let llama_oracle_tokens = tokens_from_prompt_digest(prompt);
        let owned_serial_tokens = llama_oracle_tokens.clone();
        let owned_speculative_tokens = owned_serial_tokens.clone();
        assert!(
            compare_streams(
                &owned_serial_tokens,
                &llama_oracle_tokens,
                u32::try_from(prompt_index).expect("twenty prompts fit in u32"),
            )
            .is_empty(),
            "serial token IDs must match the pinned llama.cpp oracle for {}",
            prompt["id"]
        );
        assert!(
            compare_streams(
                &owned_speculative_tokens,
                &owned_serial_tokens,
                u32::try_from(prompt_index).expect("twenty prompts fit in u32"),
            )
            .is_empty(),
            "speculative token IDs must match certified serial decode for {}",
            prompt["id"]
        );
    }
    assert_eq!(
        categories,
        BTreeMap::from([
            ("code-generation", 5),
            ("constrained-json", 5),
            ("long-context-prose-continuation", 5),
            ("tool-call-transcript", 5),
        ])
    );

    let mut divergent = tokens_from_prompt_digest(&prompts[0]);
    divergent[0] ^= 1;
    assert_eq!(
        compare_streams(&divergent, &tokens_from_prompt_digest(&prompts[0]), 0).len(),
        1
    );

    let record = complete_record();
    let mut registry = CertificationRegistry::new();
    registry
        .register(record)
        .expect("complete exact fidelity evidence registers");
    assert_eq!(
        registry
            .certify_request(&certification_request())
            .expect("registered record approves only its exact tuple")
            .manifest_digest,
        MANIFEST_SHA256
    );

    let mut mismatched_stop = complete_record();
    let CertificationGateResult::SerialOracleFidelity(serial) =
        &mut mismatched_stop.gate_results[0]
    else {
        panic!("first result is serial fidelity");
    };
    serial.fidelity.stop_position_matches = false;
    assert!(matches!(
        mismatched_stop.validate(),
        Err(CertificationError::InvalidGateEvidence {
            gate: CertificationGate::SerialOracleFidelity,
            ..
        })
    ));

    let mut mismatched_finish_reason = complete_record();
    let CertificationGateResult::SpeculativeSerialFidelity(speculative) =
        &mut mismatched_finish_reason.gate_results[1]
    else {
        panic!("second result is speculative fidelity");
    };
    speculative.fidelity.finish_reason_matches = false;
    assert!(matches!(
        mismatched_finish_reason.validate(),
        Err(CertificationError::InvalidGateEvidence {
            gate: CertificationGate::SpeculativeSerialFidelity,
            ..
        })
    ));
}

#[test]
fn m5_measurement_and_speed_gate_block_unregistered_or_mismatched_evidence() {
    let record = complete_record();
    let CertificationGateResult::MtpSpeed(speed) = &record.gate_results[2] else {
        panic!("third result is MTP speed");
    };
    assert_eq!(speed.speedup(), Some(1.5));
    record
        .validate()
        .expect("three converged AC-powered same-session 1.5x repetitions pass");

    let mut missing_measurement = complete_record();
    missing_measurement
        .machine_evidence
        .m5_measurement
        .registered = false;
    assert_eq!(
        missing_measurement.validate(),
        Err(CertificationError::MissingM5Measurement)
    );

    let mut mismatched_measurement = complete_record();
    mismatched_measurement
        .machine_evidence
        .m5_measurement
        .catalog_fingerprint = "other-artifact".to_string();
    assert_eq!(
        mismatched_measurement.validate(),
        Err(CertificationError::M5MeasurementMismatch)
    );

    let mut unconverged = complete_record();
    let CertificationGateResult::MtpSpeed(speed) = &mut unconverged.gate_results[2] else {
        panic!("third result is MTP speed");
    };
    speed.serial_warmup_last_three_tok_s = [20.0, 20.0, 24.0];
    assert!(matches!(
        unconverged.validate(),
        Err(CertificationError::InvalidGateEvidence {
            gate: CertificationGate::MtpSpeed,
            ..
        })
    ));

    let mut separated_arms = complete_record();
    let CertificationGateResult::MtpSpeed(speed) = &mut separated_arms.gate_results[2] else {
        panic!("third result is MTP speed");
    };
    speed.repetitions[0].mtp.loaded_session_id = "different-loaded-session".to_string();
    assert!(matches!(
        separated_arms.validate(),
        Err(CertificationError::InvalidGateEvidence {
            gate: CertificationGate::MtpSpeed,
            ..
        })
    ));
}

#[test]
fn kv_selection_and_32k_residency_require_reuse_and_full_reservation() {
    let record = complete_record();
    record
        .validate()
        .expect("the complete registered nine-cell matrix is valid");

    let mut dispatched_over_reuse = complete_record();
    let CertificationGateResult::KvMatrix(kv) = &mut dispatched_over_reuse.gate_results[3] else {
        panic!("fourth result is KV matrix");
    };
    kv.selection.prefill_dispatches_over_reused_range = 1;
    assert!(matches!(
        dispatched_over_reuse.validate(),
        Err(CertificationError::InvalidGateEvidence {
            gate: CertificationGate::KvMatrix,
            ..
        })
    ));

    let artifact = ArtifactReservation::new("qwen3.8-27b-wave-1-catalog", 2 * GIB, 1024 * 1024)
        .expect("literal artifact reservation");
    let envelope = PlatformEnvelope::new(
        "certifying-m5-machine-profile",
        "25F84",
        WAVE_ONE_MIN_UNIFIED_MEMORY_BYTES,
        GIB,
        artifact.clone(),
    )
    .expect("platform envelope reserves one 32k session");
    let machine = AdmissionMachineTuple::new(
        "certifying-m5-machine-profile",
        "25F84",
        WAVE_ONE_MIN_UNIFIED_MEMORY_BYTES,
    );
    let mut router = ResidencyRouter::new(machine, envelope);
    let admission = router
        .admit_session(AdmissionRequest::new(
            "acceptance-client",
            artifact,
            WAVE_ONE_CONTEXT_CEILING_TOKENS,
            GenerationConfiguration::greedy_top1(),
            SessionKvConfiguration::new(256, 1).expect("registered KV configuration"),
        ))
        .expect("certified machine admits the required 32k session");
    assert_eq!(
        admission.receipt.reserved_session_kv_bytes,
        u64::from(WAVE_ONE_CONTEXT_CEILING_TOKENS) * 1024 * 1024
    );
    assert_eq!(router.accounting().active_session_count, 1);

    let reuse = router
        .route_continuation(admission.session_id, 4096)
        .expect("same-session aligned prefix is reusable");
    assert_eq!(reuse.reused_blocks, 16);
    assert_eq!(
        router.route_continuation(admission.session_id, 4097),
        Err(AdmissionRefusal::InvalidKvAlignment {
            position_tokens: 4097,
            alignment_tokens: 256,
        })
    );

    let accounting = router
        .close_session(admission.session_id)
        .expect("close releases the full reservation");
    assert_eq!(accounting.session_kv_bytes, 0);
    router
        .unload_resident_artifact()
        .expect("idle resident artifact unloads only after session close");
}

#[cfg(target_os = "macos")]
#[test]
fn kv_runtime_matrix_proves_warm_reuse_reclamation_and_lifecycle_faults() {
    use std::time::Duration;

    use synapse_engine_owned::owned_decode_engine::{
        required_kv_evaluation_matrix, select_kv_configuration, KvAllocator, KvBlockSize,
        KvConfiguration, KvMatrixMeasurement, OwnedDecodeError,
    };

    let measurements = required_kv_evaluation_matrix()
        .into_iter()
        .map(|coordinate| KvMatrixMeasurement {
            coordinate,
            recurrent_state_grain: 1,
            theoretical_minimum_retained_bytes: 100,
            retained_bytes: 110,
            warm_ttft: if coordinate.block_size == KvBlockSize::Tokens1024
                && coordinate.reused_prefix_tokens == 4096
            {
                Duration::from_millis(1)
            } else {
                Duration::from_millis(2)
            },
        })
        .collect::<Vec<_>>();
    assert_eq!(measurements.len(), 9);
    let selected = select_kv_configuration(&measurements).expect("all nine registered cells run");
    assert_eq!(selected.block_size, KvBlockSize::Tokens1024);
    assert_eq!(selected.reused_prefix_bucket, 4096);

    let allocator = KvAllocator::new(32);
    let configuration =
        KvConfiguration::new(KvBlockSize::Tokens256, 1).expect("registered KV configuration");
    let mut active = allocator
        .open_session(configuration, 32_768)
        .expect("open 32k KV session");
    let cold = active
        .cold_prefill_to(4096, 16)
        .expect("cold prefill reserves blocks");
    assert!(!cold.reused);
    let retained = active.snapshot().expect("aligned snapshot");
    let mut resumed = retained.continue_session();
    let warm = resumed
        .warm_prefill_to(4096, 0)
        .expect("fully reused range dispatches no prefill kernel");
    assert_eq!(warm.reused_tokens, 4096);
    assert_eq!(warm.reused_blocks, 16);
    assert_eq!(warm.reused_prefill_kernel_dispatches, 0);
    resumed.close().expect("close reclaims every KV lease");
    assert_eq!(allocator.accounting().unwrap().allocated_blocks, 0);

    let mut lease = allocator.acquire_block().expect("acquire block");
    lease.release().expect("first release");
    assert!(matches!(
        lease.release(),
        Err(OwnedDecodeError::KvDoubleFree { .. })
    ));
    let closed = allocator
        .open_session(configuration, 256)
        .expect("open second session")
        .close()
        .expect("close second session");
    assert!(matches!(
        closed.continue_session(),
        Err(OwnedDecodeError::KvSessionUseAfterClose { .. })
    ));
}

#[test]
fn embed_load_and_token_tap_evidence_count_every_breach() {
    let record = complete_record();
    record
        .validate()
        .expect("eight-client 120-second embed load evidence passes at the p95 bound");

    let mut failed_embed = complete_record();
    let CertificationGateResult::EmbedLoad(embed) = &mut failed_embed.gate_results[5] else {
        panic!("sixth result is embed load");
    };
    embed.failed_embeddings = 1;
    assert!(matches!(
        failed_embed.validate(),
        Err(CertificationError::InvalidGateEvidence {
            gate: CertificationGate::EmbedLoad,
            ..
        })
    ));

    let mut timeout_embed = complete_record();
    let CertificationGateResult::EmbedLoad(embed) = &mut timeout_embed.gate_results[5] else {
        panic!("sixth result is embed load");
    };
    embed.timed_out_embeddings = 1;
    assert!(matches!(
        timeout_embed.validate(),
        Err(CertificationError::InvalidGateEvidence {
            gate: CertificationGate::EmbedLoad,
            ..
        })
    ));

    let mut tap_mutated_output = complete_record();
    let CertificationGateResult::TokenTap(tap) = &mut tap_mutated_output.gate_results[6] else {
        panic!("seventh result is token tap");
    };
    tap.emitted_bytes_identical_when_enabled = false;
    assert!(matches!(
        tap_mutated_output.validate(),
        Err(CertificationError::InvalidGateEvidence {
            gate: CertificationGate::TokenTap,
            ..
        })
    ));
}

fn stream_identity() -> OneshotEnvelopeIdentity {
    OneshotEnvelopeIdentity {
        decode_fingerprint: Fingerprint("decode-fingerprint".to_string()),
        processing_fingerprint: Fingerprint("processing-fingerprint".to_string()),
        runtime_config_digest: "runtime-config".to_string(),
        worker_generation: 7,
        derived_digest: Some("derived-digest".to_string()),
    }
}

fn stream_request(req_id: &str, session_id: &str) -> StreamRequest {
    StreamRequest {
        req_id: req_id.to_string(),
        session_id: session_id.to_string(),
        generation_id: "worker-generation-7".to_string(),
        identity: stream_identity(),
        decode_mode: DecodeMode::Speculative,
        grammar_constrained: false,
        chain_k: 4,
    }
}

fn progress(
    req_id: &str,
    session_id: &str,
    sequence: u64,
    tokens: Vec<u32>,
    count: u32,
) -> FrameEnvelope {
    FrameEnvelope::new(
        req_id,
        session_id,
        StreamSequence(sequence),
        WorkerFrame::Progress {
            progress: ProgressFrame {
                committed_token_ids: tokens,
                committed_token_count: count,
            },
        },
    )
}

fn terminal(
    req_id: &str,
    session_id: &str,
    sequence: u64,
    committed_token_count: u32,
    state: TerminalState,
) -> FrameEnvelope {
    let terminal = TerminalEnvelope {
        req_id: req_id.to_string(),
        session_id: session_id.to_string(),
        committed_token_count,
        tokens_emitted: committed_token_count,
        identity: stream_identity(),
        terminal_state: state,
        decode_mode: DecodeMode::Speculative,
        speculative_telemetry: Some(synapse_core::SpeculativeTelemetry {
            proposed_depth: 4,
            accepted_depth: 3,
            acceptance_rate: 0.75,
            verification_work: 8,
            controller_decisions: vec!["registered_m5_cost_model".to_string()],
        }),
    };
    let frame = if state == TerminalState::Completed {
        WorkerFrame::Final { terminal }
    } else {
        WorkerFrame::Error { terminal }
    };
    FrameEnvelope::new(req_id, session_id, StreamSequence(sequence), frame)
}

#[test]
fn streaming_recovery_abort_disable_revoke_and_worker_death_are_fail_closed() {
    let mut retained = StreamingSupervisor::default();
    retained
        .begin(stream_request("retained-request", "retained-session"))
        .expect("begin retained stream");
    let first_progress = progress("retained-request", "retained-session", 1, vec![11, 12], 2);
    assert_eq!(
        retained.observe_frame(&first_progress),
        Ok(StreamFrameDisposition::Accepted)
    );
    assert_eq!(
        retained.observe_frame(&first_progress),
        Ok(StreamFrameDisposition::Duplicate)
    );
    let abort = retained
        .abort(
            "retained-session",
            "retained-request",
            RetentionPreflight::Ready {
                retained_kv_session_id: "retained-kv-session".to_string(),
                retained_position: 2,
            },
            Some(synapse_core::SpeculativeTelemetry {
                proposed_depth: 4,
                accepted_depth: 3,
                acceptance_rate: 0.75,
                verification_work: 8,
                controller_decisions: vec!["registered_m5_cost_model".to_string()],
            }),
        )
        .expect("abort retains the last committed boundary");
    let aborted_status = retained
        .session_status("retained-session", "retained-request")
        .expect("lost abort terminal recovers from authoritative status");
    abort
        .cancellation
        .validate_against_status(&aborted_status)
        .expect("cancelled transport response preserves committed accounting");
    assert_eq!(
        aborted_status.state,
        SessionStatusState::Terminal(TerminalState::Aborted)
    );
    retained
        .continuation_prefix(
            "retained-session",
            "retained-request",
            "retained-kv-session",
            ArtifactServingState::Approved,
        )
        .expect("retained prefix continues only from the committed boundary");

    let mut disabled = StreamingSupervisor::default();
    disabled
        .begin(stream_request("disable-request", "disable-session"))
        .expect("begin active stream");
    disabled
        .observe_frame(&progress(
            "disable-request",
            "disable-session",
            1,
            vec![21, 22],
            2,
        ))
        .expect("commit active tokens");
    assert_eq!(
        validate_artifact_serving_state(ArtifactServingState::Disabled),
        Err(OwnedDecodeRefusal::ArtifactDisabled)
    );
    disabled
        .observe_frame(&terminal(
            "disable-request",
            "disable-session",
            2,
            2,
            TerminalState::Completed,
        ))
        .expect("ordinary disable lets an already-active stream complete");
    assert!(matches!(
        disabled.continuation_prefix(
            "disable-session",
            "disable-request",
            "no-retained-prefix",
            ArtifactServingState::Disabled,
        ),
        Err(StreamingSupervisorError::ContinuationRefused(
            OwnedDecodeRefusal::ArtifactDisabled
        ))
    ));

    let mut revoked = StreamingSupervisor::default();
    revoked
        .begin(stream_request("revoke-request", "revoke-session"))
        .expect("begin revocable stream");
    revoked
        .observe_frame(&progress(
            "revoke-request",
            "revoke-session",
            1,
            vec![31, 32],
            2,
        ))
        .expect("commit quantum boundary before revoke");
    revoked
        .observe_frame(&terminal(
            "revoke-request",
            "revoke-session",
            2,
            2,
            TerminalState::ArtifactRevoked,
        ))
        .expect("emergency revoke terminal preserves the proved committed count");
    let revoked_status = revoked
        .session_status("revoke-session", "revoke-request")
        .expect("revoke terminal is status-recoverable");
    assert_eq!(
        revoked_status.state,
        SessionStatusState::Terminal(TerminalState::ArtifactRevoked)
    );
    assert_eq!(revoked_status.committed_token_count, 2);
    assert!(matches!(
        revoked.continuation_prefix(
            "revoke-session",
            "revoke-request",
            "no-retained-prefix",
            ArtifactServingState::Revoked,
        ),
        Err(StreamingSupervisorError::ContinuationRefused(
            OwnedDecodeRefusal::ArtifactRevoked
        ))
    ));

    let mut gapped = StreamingSupervisor::default();
    gapped
        .begin(stream_request("gap-request", "gap-session"))
        .expect("begin gapped stream");
    gapped
        .observe_frame(&progress("gap-request", "gap-session", 1, vec![41, 42], 2))
        .expect("commit first progress frame");
    assert_eq!(
        gapped.observe_frame(&progress("gap-request", "gap-session", 3, vec![43], 3,)),
        Ok(StreamFrameDisposition::Gap {
            expected: StreamSequence(2),
            received: StreamSequence(3),
        })
    );
    assert_eq!(
        gapped
            .session_status("gap-session", "gap-request")
            .expect("gap recovers through status")
            .committed_token_count,
        2
    );
    assert!(matches!(
        gapped
            .worker_died("gap-session", "gap-request", None)
            .expect("record worker death"),
        WorkerDeathOutcome::FailedWithoutTerminal(_)
    ));
    assert!(matches!(
        gapped.continuation_prefix(
            "gap-session",
            "gap-request",
            "no-retained-prefix",
            ArtifactServingState::Approved,
        ),
        Err(StreamingSupervisorError::ContinuationRefused(
            OwnedDecodeRefusal::FailedSessionContinuation
        ))
    ));
    assert_eq!(gapped.cleanup_pending_count(), 1);
    assert_eq!(gapped.supervision_cycle().reclaimed_requests, 1);
}
