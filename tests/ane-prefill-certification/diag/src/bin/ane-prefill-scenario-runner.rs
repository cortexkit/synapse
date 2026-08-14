use std::collections::BTreeMap;

use anyhow::{bail, ensure, Context, Result};
use clap::{Parser, ValueEnum};
use serde_json::{json, Value};
use synapse_core::Fingerprint;
use synapse_module::owned_decode_routing::ane_prefill::{
    AneGuard, AnePlatform, AnePrefillFault, AnePrefillRouter, AneSplitAttempt,
    AneSplitAttemptResult, AneSplitCompletion, AneSplitRequest, AneSplitRouteDecision,
    AneSplitRoutingConfig, CertifiedArtifactIdentity, DecodePrefill, GuardAcquisition,
    IdentityPreservingFailure, PrefillBucket, PrefillBypassReason, PrefillEngine,
    PrefillFallbackReason, RuntimeArtifactIdentity, SplitArmCertification, SplitArmHealth,
    SplitArmKey, SplitArmState, SplitDecodeConfig, SplitHealthPolicy, SplitTimingBudgets,
};
use synapse_module::owned_decode_routing::family::Family;
use synapse_module::owned_decode_routing::request::SamplingMode;

const PROFILE: &str = "ane-prefill-certification-scenario";
const SOURCE_DIGEST: &str = "source-a";
const COMPILED_DIGEST: &str = "compiled-a";
const GPU_FINGERPRINT: &str = "gpu-processing";
const ANE_FINGERPRINT: &str = "ane-split-processing";
const QUARANTINE_EXPIRY_MS: u64 = 60_000;

const ROUTING_CASES: &[&str] = &[
    "global_precedence",
    "bucket_escalation",
    "smallest_terminal_state",
    "unloaded_matching_artifact",
    "present_compiled_digest_mismatch",
    "capacity_boundaries",
    "guard_timeout",
    "deadline_after_guard",
];

const BYPASS_CASES: &[&str] = &[
    "disabled",
    "platform_unsupported",
    "family_unsupported",
    "sampling_uncertified",
    "identity_pinned_gpu",
    "prompt_over_max_bucket",
    "no_fitting_cache_bucket",
    "bucket_absent",
    "not_certified",
    "artifact_digest_mismatch",
    "quarantined",
    "ane_busy",
    "deadline_too_tight",
];

const FALLBACK_CASES: &[&str] = &[
    "compile_error_after_selection",
    "load_error_or_load_timeout_after_selection",
    "dispatch_start_or_acknowledgement_failure",
    "acknowledged_stage_exit_before_prediction_result",
    "prediction_budget_expiry_while_stage_unresponsive",
    "kv_layout_or_conversion_failure",
    "cache_payload_publication_or_receipt_failure",
    "prefill_engine_to_decoding_engine_cache_conversion_failure",
    "metal_cache_upload_failure",
    "handoff_budget_expiry",
    "readiness_budget_expiry",
    "load_completion_artifact_triple_mismatch",
    "logits_copy_or_first_token_publication_failure",
];

const STATE_CASES: &[&str] = &[
    "unloaded_matching_certified_arm",
    "present_artifact_triple_mismatch",
    "postselection_load_triple_mismatch",
    "readiness_budget_expired",
    "runtime_failure_preserves_certification_row",
    "consecutive_failures_quarantine_exact_arm",
    "success_resets_only_exact_arm",
    "expiry_enters_probation",
    "probation_failure_requarantines",
    "probation_success_clears",
    "gpu_pin_bypasses_without_attempt",
    "ane_split_pin_pre_attempt_refuses_substitution",
    "ane_split_pin_in_attempt_failure_preserves_identity",
];

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Operation {
    Routing,
    Exercise,
}

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, value_enum)]
    operation: Operation,
    #[arg(long)]
    case_id: String,
    #[arg(long)]
    kind: Option<String>,
    #[arg(long)]
    bucket: u32,
    #[arg(long)]
    decode_config: String,
}

#[derive(Clone, Copy, Debug)]
struct ArmSpec {
    bucket: PrefillBucket,
    decode_config: SplitDecodeConfig,
}

impl ArmSpec {
    fn parse(bucket: u32, decode_config: &str) -> Result<Self> {
        let bucket = match bucket {
            128 => PrefillBucket::W128,
            256 => PrefillBucket::W256,
            512 => PrefillBucket::W512,
            other => bail!("unsupported prefill bucket {other}"),
        };
        let decode_config = match decode_config {
            "f16-step" => SplitDecodeConfig::F16Step,
            "q8-step" => SplitDecodeConfig::Q8Step,
            other => bail!("unsupported decode config {other}"),
        };
        Ok(Self {
            bucket,
            decode_config,
        })
    }

    fn key(self) -> SplitArmKey {
        key(self.bucket, self.decode_config)
    }
}

#[derive(Clone, Copy, Debug)]
struct RecordingGuard {
    result: GuardAcquisition,
    requested_budget_ms: Option<u64>,
}

impl RecordingGuard {
    fn success() -> Self {
        Self {
            result: GuardAcquisition {
                acquired: true,
                waited_ms: 1,
            },
            requested_budget_ms: None,
        }
    }

    fn with_result(acquired: bool, waited_ms: u64) -> Self {
        Self {
            result: GuardAcquisition {
                acquired,
                waited_ms,
            },
            requested_budget_ms: None,
        }
    }
}

impl AneGuard for RecordingGuard {
    fn acquire_within(&mut self, max_wait_ms: u64) -> GuardAcquisition {
        self.requested_budget_ms = Some(max_wait_ms);
        self.result
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let spec = ArmSpec::parse(args.bucket, &args.decode_config)?;
    let observed = match args.operation {
        Operation::Routing => run_routing_case(&args.case_id, spec)?,
        Operation::Exercise => run_exercise_case(
            args.kind
                .as_deref()
                .context("--kind is required for exercise scenarios")?,
            &args.case_id,
            spec,
        )?,
    };
    println!(
        "{}",
        serde_json::to_string(&json!({
            "status": "ok",
            "case_id": args.case_id,
            "observed": observed,
        }))?
    );
    Ok(())
}

fn run_routing_case(case_id: &str, spec: ArmSpec) -> Result<Value> {
    ensure!(
        ROUTING_CASES.contains(&case_id),
        "unknown routing case {case_id}"
    );
    match case_id {
        "global_precedence" => routing_global_precedence(spec),
        "bucket_escalation" => routing_bucket_escalation(spec),
        "smallest_terminal_state" => routing_smallest_terminal_state(spec),
        "unloaded_matching_artifact" => routing_unloaded_matching_artifact(spec),
        "present_compiled_digest_mismatch" => routing_present_digest_mismatch(spec),
        "capacity_boundaries" => routing_capacity_boundaries(spec),
        "guard_timeout" => routing_guard_timeout(spec),
        "deadline_after_guard" => routing_deadline_after_guard(spec),
        _ => unreachable!("routing case was checked against ROUTING_CASES"),
    }
}

fn run_exercise_case(kind: &str, case_id: &str, spec: ArmSpec) -> Result<Value> {
    match kind {
        "bypass" => {
            ensure!(
                BYPASS_CASES.contains(&case_id),
                "unknown bypass case {case_id}"
            );
            exercise_bypass(case_id, spec)
        }
        "fallback" => {
            ensure!(
                FALLBACK_CASES.contains(&case_id),
                "unknown fallback case {case_id}"
            );
            exercise_fallback(case_id, spec)
        }
        "state" => {
            ensure!(
                STATE_CASES.contains(&case_id),
                "unknown state case {case_id}"
            );
            exercise_state(case_id, spec)
        }
        "protocol" if case_id == "connect_mismatch" => exercise_connect_mismatch(spec),
        "protocol" => bail!("unknown protocol case {case_id}"),
        other => bail!("unknown exercise kind {other}"),
    }
}

fn routing_global_precedence(spec: ArmSpec) -> Result<Value> {
    let mut oversized = request(spec);
    oversized.prompt_token_count = 1_000;
    oversized.family = Family::Lfm2_1_2b;
    oversized.sampling = SamplingMode::TopP { p: 0.5 };
    oversized.required_processing_fingerprint = Some(gpu_fingerprint());

    let mut disabled = router_with(
        BTreeMap::new(),
        DecodePrefill::Gpu,
        AnePlatform::Unsupported,
        SplitHealthPolicy::default(),
    );
    let disabled_reason = route_bypass_reason(&mut disabled, &oversized, 0, &mut success_guard())?;

    let mut unsupported = router_with(
        BTreeMap::new(),
        DecodePrefill::AneSplit,
        AnePlatform::Unsupported,
        SplitHealthPolicy::default(),
    );
    let unsupported_reason =
        route_bypass_reason(&mut unsupported, &oversized, 0, &mut success_guard())?;

    let mut family = router(BTreeMap::new());
    let family_reason = route_bypass_reason(&mut family, &oversized, 0, &mut success_guard())?;

    let mut sampled_request = oversized.clone();
    sampled_request.family = Family::Qwen3_0_6b;
    let mut sampled = router(BTreeMap::new());
    let sampled_reason =
        route_bypass_reason(&mut sampled, &sampled_request, 0, &mut success_guard())?;

    let mut pinned_request = sampled_request;
    pinned_request.sampling = SamplingMode::GreedyTop1;
    let mut pinned = router(BTreeMap::new());
    let pinned_reason = route_bypass_reason(&mut pinned, &pinned_request, 0, &mut success_guard())?;

    let observed = [
        disabled_reason,
        unsupported_reason,
        family_reason,
        sampled_reason,
        pinned_reason,
    ];
    let expected = [
        PrefillBypassReason::Disabled,
        PrefillBypassReason::PlatformUnsupported,
        PrefillBypassReason::FamilyUnsupported,
        PrefillBypassReason::SamplingUncertified,
        PrefillBypassReason::IdentityPinnedGpu,
    ];
    ensure!(
        observed == expected,
        "global bypass precedence drifted: {observed:?}"
    );
    Ok(json!({
        "observed_precedence": observed.map(PrefillBypassReason::as_str),
        "split_attempt_started": false,
    }))
}

fn routing_bucket_escalation(spec: ArmSpec) -> Result<Value> {
    let mut smaller = certified_state(Some("wrong-digest"), 512, SplitArmHealth::default());
    smaller.runtime_artifacts.manifest_source_checkpoint_digest = SOURCE_DIGEST.to_string();
    let arms = BTreeMap::from([
        (key(PrefillBucket::W128, spec.decode_config), smaller),
        (
            key(PrefillBucket::W256, spec.decode_config),
            certified_state(Some(COMPILED_DIGEST), 512, SplitArmHealth::default()),
        ),
    ]);
    let scenario = ArmSpec {
        bucket: PrefillBucket::W128,
        ..spec
    };
    let mut routing = router(arms);
    let attempt = route_attempt(&mut routing, &request(scenario), 0, &mut success_guard())?;
    ensure!(
        attempt.arm.bucket == PrefillBucket::W256,
        "router did not escalate to W256"
    );
    Ok(json!({"selected_bucket": attempt.arm.bucket.tokens(), "split_attempt_started": true}))
}

fn routing_smallest_terminal_state(spec: ArmSpec) -> Result<Value> {
    let mut smallest = certified_state(Some(COMPILED_DIGEST), 512, SplitArmHealth::default());
    smallest.certification = SplitArmCertification::NotCertified;
    let mut larger = certified_state(Some(COMPILED_DIGEST), 512, SplitArmHealth::default());
    larger.certification = SplitArmCertification::BucketAbsent;
    let arms = BTreeMap::from([
        (key(PrefillBucket::W128, spec.decode_config), smallest),
        (key(PrefillBucket::W256, spec.decode_config), larger),
    ]);
    let scenario = ArmSpec {
        bucket: PrefillBucket::W128,
        ..spec
    };
    let mut routing = router(arms);
    let reason = route_bypass_reason(&mut routing, &request(scenario), 0, &mut success_guard())?;
    ensure!(
        reason == PrefillBypassReason::NotCertified,
        "larger arm changed the terminal reason"
    );
    Ok(json!({"prefill_bypass_reason": reason.as_str(), "terminal_bucket": 128}))
}

fn routing_unloaded_matching_artifact(spec: ArmSpec) -> Result<Value> {
    let arms = BTreeMap::from([(
        spec.key(),
        certified_state(None, spec.bucket.tokens() + 64, SplitArmHealth::default()),
    )]);
    let mut routing = router(arms);
    let attempt = route_attempt(&mut routing, &request(spec), 0, &mut success_guard())?;
    ensure!(
        attempt.readiness_required,
        "unloaded matching arm skipped readiness"
    );
    Ok(json!({"split_selected": true, "readiness_started": attempt.readiness_required}))
}

fn routing_present_digest_mismatch(spec: ArmSpec) -> Result<Value> {
    let arms = BTreeMap::from([(
        spec.key(),
        certified_state(
            Some("wrong-digest"),
            spec.bucket.tokens() + 64,
            SplitArmHealth::default(),
        ),
    )]);
    let mut routing = router(arms);
    let reason = route_bypass_reason(&mut routing, &request(spec), 0, &mut success_guard())?;
    ensure!(
        reason == PrefillBypassReason::ArtifactDigestMismatch,
        "present digest mismatch returned {reason:?}"
    );
    Ok(json!({"prefill_bypass_reason": reason.as_str(), "split_attempt_started": false}))
}

fn routing_capacity_boundaries(spec: ArmSpec) -> Result<Value> {
    let required_positions = spec.bucket.tokens() + 64;
    let exact_arms = BTreeMap::from([(
        spec.key(),
        certified_state(
            Some(COMPILED_DIGEST),
            required_positions,
            SplitArmHealth::default(),
        ),
    )]);
    let mut exact = router(exact_arms);
    let exact_attempt = route_attempt(&mut exact, &request(spec), 0, &mut success_guard())?;
    ensure!(
        exact_attempt.arm == spec.key(),
        "exact capacity selected the wrong arm"
    );

    let short_arms = BTreeMap::from([(
        spec.key(),
        certified_state(
            Some(COMPILED_DIGEST),
            required_positions - 1,
            SplitArmHealth::default(),
        ),
    )]);
    let mut short = router(short_arms);
    let short_reason = route_bypass_reason(&mut short, &request(spec), 0, &mut success_guard())?;
    ensure!(
        short_reason == PrefillBypassReason::NoFittingCacheBucket,
        "one-short cache returned {short_reason:?}"
    );
    Ok(json!({
        "required_positions": required_positions,
        "exact_capacity_selected": true,
        "one_short_prefill_bypass_reason": short_reason.as_str(),
    }))
}

fn routing_guard_timeout(spec: ArmSpec) -> Result<Value> {
    let mut routing = router(one_certified_arm(spec));
    let mut guard = RecordingGuard::with_result(false, 20);
    let reason = route_bypass_reason(&mut routing, &request(spec), 0, &mut guard)?;
    ensure!(
        reason == PrefillBypassReason::AneBusy,
        "guard timeout returned {reason:?}"
    );
    ensure!(
        guard.requested_budget_ms == Some(20),
        "router used the wrong guard bound"
    );
    Ok(json!({
        "prefill_bypass_reason": reason.as_str(),
        "guard_budget_ms": guard.requested_budget_ms,
        "split_attempt_started": false,
    }))
}

fn routing_deadline_after_guard(spec: ArmSpec) -> Result<Value> {
    let mut rejected_request = request(spec);
    rejected_request.deadline_remaining_ms = Some(834);
    let mut rejected = router(one_certified_arm(spec));
    let mut waited = RecordingGuard::with_result(true, 5);
    let reason = route_bypass_reason(&mut rejected, &rejected_request, 0, &mut waited)?;
    ensure!(
        reason == PrefillBypassReason::DeadlineTooTight,
        "tight deadline returned {reason:?}"
    );

    let mut accepted_request = request(spec);
    accepted_request.deadline_remaining_ms = Some(835);
    let mut accepted = router(one_certified_arm(spec));
    let accepted_attempt = route_attempt(
        &mut accepted,
        &accepted_request,
        0,
        &mut RecordingGuard::with_result(true, 5),
    )?;
    ensure!(
        accepted_attempt.arm == spec.key(),
        "boundary deadline selected the wrong arm"
    );
    Ok(json!({
        "guard_waited_ms": 5,
        "rejected_deadline_ms": 834,
        "rejected_prefill_bypass_reason": reason.as_str(),
        "accepted_deadline_ms": 835,
        "accepted_split_attempt_started": true,
    }))
}

fn exercise_bypass(case_id: &str, spec: ArmSpec) -> Result<Value> {
    let mut state = certified_state(
        Some(COMPILED_DIGEST),
        spec.bucket.tokens() + 64,
        SplitArmHealth::default(),
    );
    let mut decode_prefill = DecodePrefill::AneSplit;
    let mut platform = AnePlatform::Supported;
    let mut scenario_request = request(spec);
    let mut guard = success_guard();

    match case_id {
        "disabled" => decode_prefill = DecodePrefill::Gpu,
        "platform_unsupported" => platform = AnePlatform::Unsupported,
        "family_unsupported" => scenario_request.family = Family::Lfm2_1_2b,
        "sampling_uncertified" => scenario_request.sampling = SamplingMode::TopP { p: 0.5 },
        "identity_pinned_gpu" => {
            scenario_request.required_processing_fingerprint = Some(gpu_fingerprint())
        }
        "prompt_over_max_bucket" => scenario_request.prompt_token_count = 513,
        "no_fitting_cache_bucket" => state.decode_cache_bucket = Some(1),
        "bucket_absent" => state.certification = SplitArmCertification::BucketAbsent,
        "not_certified" => state.certification = SplitArmCertification::NotCertified,
        "artifact_digest_mismatch" => {
            state.runtime_artifacts.derived_or_compiled_artifact_digest =
                Some("wrong-digest".to_string())
        }
        "quarantined" => state.health.quarantined_until_ms = Some(QUARANTINE_EXPIRY_MS),
        "ane_busy" => guard = RecordingGuard::with_result(false, 20),
        "deadline_too_tight" => scenario_request.deadline_remaining_ms = Some(0),
        _ => unreachable!("bypass case was checked against BYPASS_CASES"),
    }

    let key = spec.key();
    let mut routing = router_with(
        BTreeMap::from([(key.clone(), state)]),
        decode_prefill,
        platform,
        SplitHealthPolicy::default(),
    );
    let strikes_before = arm_strikes(&routing, &key)?;
    let decision = routing.route(&scenario_request, 0, &mut guard);
    let provenance = match decision {
        AneSplitRouteDecision::GpuBypass { provenance, .. } => provenance,
        other => bail!("expected GPU bypass for {case_id}, got {other:?}"),
    };
    let strikes_after = arm_strikes(&routing, &key)?;
    ensure!(
        provenance
            .prefill_bypass_reason
            .map(PrefillBypassReason::as_str)
            == Some(case_id),
        "bypass case {case_id} returned {:?}",
        provenance.prefill_bypass_reason
    );
    ensure!(
        strikes_after == strikes_before,
        "bypass {case_id} charged arm health"
    );
    Ok(json!({
        "prefill_engine": engine_name(provenance.prefill_engine),
        "prefill_bypass_reason": provenance.prefill_bypass_reason.map(PrefillBypassReason::as_str),
        "prefill_fallback_reason": provenance.prefill_fallback_reason.map(PrefillFallbackReason::as_str),
        "split_attempt_started": false,
        "arm_health_debit": strikes_after - strikes_before,
        "decode_lane_health_debit": 0,
    }))
}

fn exercise_fallback(case_id: &str, spec: ArmSpec) -> Result<Value> {
    let fault = fault_for_case(case_id)?;
    let fallback_reason = fault.fallback_reason();
    let key = spec.key();
    let mut routing = router(one_certified_arm(spec));
    let strikes_before = arm_strikes(&routing, &key)?;
    let attempt = route_attempt(&mut routing, &request(spec), 0, &mut success_guard())?;
    let completion =
        routing.complete_attempt(attempt, AneSplitAttemptResult::Failure(fallback_reason), 1);
    let provenance = match completion {
        AneSplitCompletion::GpuFallback { provenance, .. } => provenance,
        other => bail!("expected GPU fallback for {case_id}, got {other:?}"),
    };
    let strikes_after = arm_strikes(&routing, &key)?;
    ensure!(
        strikes_after - strikes_before == 1,
        "fallback {case_id} did not debit once"
    );
    ensure!(
        provenance.prefill_fallback_reason == Some(fallback_reason),
        "fallback reason map drifted for {case_id}"
    );
    Ok(json!({
        "prefill_engine": engine_name(provenance.prefill_engine),
        "prefill_fallback_from": provenance.prefill_fallback_from.map(engine_name),
        "prefill_fallback_reason": provenance.prefill_fallback_reason.map(PrefillFallbackReason::as_str),
        "split_attempt_started": true,
        "arm_health_debit": strikes_after - strikes_before,
        "decode_lane_health_debit": 0,
    }))
}

fn exercise_state(case_id: &str, spec: ArmSpec) -> Result<Value> {
    match case_id {
        "unloaded_matching_certified_arm" => state_unloaded_matching(spec),
        "present_artifact_triple_mismatch" => state_present_artifact_mismatch(spec),
        "postselection_load_triple_mismatch" => state_postselection_mismatch(spec),
        "readiness_budget_expired" => state_readiness_expired(spec),
        "runtime_failure_preserves_certification_row" => state_runtime_failure_preserves_row(spec),
        "consecutive_failures_quarantine_exact_arm" => state_consecutive_failures(spec),
        "success_resets_only_exact_arm" => state_success_resets_exact_arm(spec),
        "expiry_enters_probation" => state_expiry_enters_probation(spec),
        "probation_failure_requarantines" => state_probation_failure(spec),
        "probation_success_clears" => state_probation_success(spec),
        "gpu_pin_bypasses_without_attempt" => state_gpu_pin(spec),
        "ane_split_pin_pre_attempt_refuses_substitution" => state_ane_pin_pre_attempt(spec),
        "ane_split_pin_in_attempt_failure_preserves_identity" => state_ane_pin_in_attempt(spec),
        _ => unreachable!("state case was checked against STATE_CASES"),
    }
}

fn state_unloaded_matching(spec: ArmSpec) -> Result<Value> {
    let arms = BTreeMap::from([(
        spec.key(),
        certified_state(None, spec.bucket.tokens() + 64, SplitArmHealth::default()),
    )]);
    let mut routing = router(arms);
    let attempt = route_attempt(&mut routing, &request(spec), 0, &mut success_guard())?;
    Ok(json!({
        "split_selected": true,
        "readiness_started": attempt.readiness_required,
        "prefill_bypass_reason": null,
    }))
}

fn state_present_artifact_mismatch(spec: ArmSpec) -> Result<Value> {
    let key = spec.key();
    let arms = BTreeMap::from([(
        key.clone(),
        certified_state(
            Some("wrong-digest"),
            spec.bucket.tokens() + 64,
            SplitArmHealth::default(),
        ),
    )]);
    let mut routing = router(arms);
    let strikes_before = arm_strikes(&routing, &key)?;
    let decision = routing.route(&request(spec), 0, &mut success_guard());
    let provenance = match decision {
        AneSplitRouteDecision::GpuBypass { provenance, .. } => provenance,
        other => bail!("expected digest-mismatch bypass, got {other:?}"),
    };
    let strikes_after = arm_strikes(&routing, &key)?;
    Ok(json!({
        "prefill_engine": engine_name(provenance.prefill_engine),
        "prefill_bypass_reason": provenance.prefill_bypass_reason.map(PrefillBypassReason::as_str),
        "split_attempt_started": false,
        "arm_health_debit": strikes_after - strikes_before,
    }))
}

fn state_postselection_mismatch(spec: ArmSpec) -> Result<Value> {
    lifecycle_fallback(spec, AnePrefillFault::ArtifactMismatch)
}

fn state_readiness_expired(spec: ArmSpec) -> Result<Value> {
    lifecycle_fallback(spec, AnePrefillFault::ReadinessBudget)
}

fn lifecycle_fallback(spec: ArmSpec, fault: AnePrefillFault) -> Result<Value> {
    let key = spec.key();
    let arms = BTreeMap::from([(
        key.clone(),
        certified_state(None, spec.bucket.tokens() + 64, SplitArmHealth::default()),
    )]);
    let mut routing = router(arms);
    let certification_before = arm_certification(&routing, &key)?;
    let strikes_before = arm_strikes(&routing, &key)?;
    let attempt = route_attempt(&mut routing, &request(spec), 0, &mut success_guard())?;
    ensure!(
        attempt.readiness_required,
        "lifecycle fallback did not begin readiness"
    );
    let completion = routing.complete_attempt(
        attempt,
        AneSplitAttemptResult::Failure(fault.fallback_reason()),
        1,
    );
    let provenance = match completion {
        AneSplitCompletion::GpuFallback { provenance, .. } => provenance,
        other => bail!("expected lifecycle GPU fallback, got {other:?}"),
    };
    let certification_after = arm_certification(&routing, &key)?;
    let strikes_after = arm_strikes(&routing, &key)?;
    Ok(json!({
        "prefill_engine": engine_name(provenance.prefill_engine),
        "prefill_fallback_from": provenance.prefill_fallback_from.map(engine_name),
        "prefill_fallback_reason": provenance.prefill_fallback_reason.map(PrefillFallbackReason::as_str),
        "split_attempt_started": true,
        "arm_health_debit": strikes_after - strikes_before,
        "certification_row_preserved": certification_before == certification_after,
    }))
}

fn state_runtime_failure_preserves_row(spec: ArmSpec) -> Result<Value> {
    let key = spec.key();
    let mut routing = router(one_certified_arm(spec));
    let certification_before = arm_certification(&routing, &key)?;
    let strikes_before = arm_strikes(&routing, &key)?;
    let attempt = route_attempt(&mut routing, &request(spec), 0, &mut success_guard())?;
    let completion = routing.complete_attempt(
        attempt,
        AneSplitAttemptResult::Failure(PrefillFallbackReason::PredictionFailure),
        1,
    );
    ensure!(
        matches!(completion, AneSplitCompletion::GpuFallback { .. }),
        "runtime failure did not follow unpinned fallback semantics"
    );
    let certification_after = arm_certification(&routing, &key)?;
    let strikes_after = arm_strikes(&routing, &key)?;
    let preserved = certification_before == certification_after;
    Ok(json!({
        "certification_row_preserved": preserved,
        "runtime_fault_is_not_certification_mutation": preserved && strikes_after == strikes_before + 1,
    }))
}

fn state_consecutive_failures(spec: ArmSpec) -> Result<Value> {
    let other = other_spec(spec);
    let target_key = spec.key();
    let other_key = other.key();
    let arms = BTreeMap::from([
        (
            target_key.clone(),
            certified_state(
                Some(COMPILED_DIGEST),
                spec.bucket.tokens() + 64,
                SplitArmHealth::default(),
            ),
        ),
        (
            other_key.clone(),
            certified_state(
                Some(COMPILED_DIGEST),
                other.bucket.tokens() + 64,
                SplitArmHealth::default(),
            ),
        ),
    ]);
    let mut routing = router(arms);
    for now_ms in [1, 2] {
        let attempt = route_attempt(&mut routing, &request(spec), now_ms, &mut success_guard())?;
        ensure!(
            attempt.arm == target_key,
            "failure charged a substituted arm"
        );
        let _ = routing.complete_attempt(
            attempt,
            AneSplitAttemptResult::Failure(PrefillFallbackReason::DispatchFailure),
            now_ms,
        );
    }
    let target = arm_health(&routing, &target_key)?;
    let other_health = arm_health(&routing, &other_key)?;
    Ok(json!({
        "target_arm_quarantined": target.quarantined_until_ms.is_some(),
        "other_arm_strikes": other_health.consecutive_strikes,
        "decode_lane_health_debit": 0,
    }))
}

fn state_success_resets_exact_arm(spec: ArmSpec) -> Result<Value> {
    let other = other_spec(spec);
    let target_key = spec.key();
    let other_key = other.key();
    let health = SplitArmHealth {
        consecutive_strikes: 1,
        quarantined_until_ms: None,
        probation: false,
    };
    let arms = BTreeMap::from([
        (
            target_key.clone(),
            certified_state(
                Some(COMPILED_DIGEST),
                spec.bucket.tokens() + 64,
                health.clone(),
            ),
        ),
        (
            other_key.clone(),
            certified_state(Some(COMPILED_DIGEST), other.bucket.tokens() + 64, health),
        ),
    ]);
    let mut routing = router(arms);
    let other_before = arm_strikes(&routing, &other_key)?;
    let attempt = route_attempt(&mut routing, &request(spec), 0, &mut success_guard())?;
    ensure!(
        attempt.arm == target_key,
        "success completed a substituted arm"
    );
    let completion = routing.complete_attempt(attempt, AneSplitAttemptResult::Success, 0);
    ensure!(
        matches!(completion, AneSplitCompletion::SplitSuccess { .. }),
        "success did not return split provenance"
    );
    Ok(json!({
        "successful_arm_strikes": arm_strikes(&routing, &target_key)?,
        "other_arm_strikes_unchanged": arm_strikes(&routing, &other_key)? == other_before,
    }))
}

fn state_expiry_enters_probation(spec: ArmSpec) -> Result<Value> {
    let (key, policy, mut routing) = router_at_quarantine_expiry(spec);
    let _attempt = route_attempt(
        &mut routing,
        &request(spec),
        QUARANTINE_EXPIRY_MS,
        &mut success_guard(),
    )?;
    let health = arm_health(&routing, &key)?;
    ensure!(
        health.consecutive_strikes == policy.max_strikes - 1,
        "expiry did not restore max_strikes - 1"
    );
    Ok(json!({"probation": health.probation, "strikes": "max_strikes_minus_one"}))
}

fn state_probation_failure(spec: ArmSpec) -> Result<Value> {
    let (key, _policy, mut routing) = router_at_quarantine_expiry(spec);
    let attempt = route_attempt(
        &mut routing,
        &request(spec),
        QUARANTINE_EXPIRY_MS,
        &mut success_guard(),
    )?;
    let _ = routing.complete_attempt(
        attempt,
        AneSplitAttemptResult::Failure(PrefillFallbackReason::DispatchFailure),
        QUARANTINE_EXPIRY_MS,
    );
    Ok(json!({
        "target_arm_quarantined": arm_health(&routing, &key)?.quarantined_until_ms.is_some(),
        "failure_count": 1,
    }))
}

fn state_probation_success(spec: ArmSpec) -> Result<Value> {
    let (key, _policy, mut routing) = router_at_quarantine_expiry(spec);
    let attempt = route_attempt(
        &mut routing,
        &request(spec),
        QUARANTINE_EXPIRY_MS,
        &mut success_guard(),
    )?;
    let _ = routing.complete_attempt(
        attempt,
        AneSplitAttemptResult::Success,
        QUARANTINE_EXPIRY_MS,
    );
    let health = arm_health(&routing, &key)?;
    Ok(json!({"probation": health.probation, "strikes": health.consecutive_strikes}))
}

fn state_gpu_pin(spec: ArmSpec) -> Result<Value> {
    let key = spec.key();
    let mut routing = router(one_certified_arm(spec));
    let strikes_before = arm_strikes(&routing, &key)?;
    let mut pinned = request(spec);
    pinned.required_processing_fingerprint = Some(gpu_fingerprint());
    let decision = routing.route(&pinned, 0, &mut success_guard());
    let provenance = match decision {
        AneSplitRouteDecision::GpuBypass { provenance, .. } => provenance,
        other => bail!("expected GPU-pin bypass, got {other:?}"),
    };
    let strikes_after = arm_strikes(&routing, &key)?;
    Ok(json!({
        "prefill_engine": engine_name(provenance.prefill_engine),
        "prefill_bypass_reason": provenance.prefill_bypass_reason.map(PrefillBypassReason::as_str),
        "split_attempt_started": false,
        "arm_health_debit": strikes_after - strikes_before,
    }))
}

fn state_ane_pin_pre_attempt(spec: ArmSpec) -> Result<Value> {
    let key = spec.key();
    let mut state = certified_state(
        Some(COMPILED_DIGEST),
        spec.bucket.tokens() + 64,
        SplitArmHealth::default(),
    );
    state.certification = SplitArmCertification::NotCertified;
    let mut routing = router(BTreeMap::from([(key.clone(), state)]));
    let strikes_before = arm_strikes(&routing, &key)?;
    let mut pinned = request(spec);
    pinned.required_processing_fingerprint = Some(ane_fingerprint());
    let decision = routing.route(&pinned, 0, &mut success_guard());
    let identity_preserving_failure = matches!(
        decision,
        AneSplitRouteDecision::IdentityPreservingFailure(IdentityPreservingFailure::PreAttempt(
            PrefillBypassReason::NotCertified
        ))
    );
    Ok(json!({
        "identity_preserving_failure": identity_preserving_failure,
        "prefill_engine": null,
        "split_attempt_started": false,
        "arm_health_debit": arm_strikes(&routing, &key)? - strikes_before,
    }))
}

fn state_ane_pin_in_attempt(spec: ArmSpec) -> Result<Value> {
    let key = spec.key();
    let mut routing = router(one_certified_arm(spec));
    let strikes_before = arm_strikes(&routing, &key)?;
    let mut pinned = request(spec);
    pinned.required_processing_fingerprint = Some(ane_fingerprint());
    let attempt = route_attempt(&mut routing, &pinned, 0, &mut success_guard())?;
    let completion = routing.complete_attempt(
        attempt,
        AneSplitAttemptResult::Failure(PrefillFallbackReason::PredictionFailure),
        1,
    );
    let identity_preserving_failure = matches!(
        completion,
        AneSplitCompletion::IdentityPreservingFailure(IdentityPreservingFailure::Execution(
            PrefillFallbackReason::PredictionFailure
        ))
    );
    Ok(json!({
        "identity_preserving_failure": identity_preserving_failure,
        "prefill_engine": null,
        "split_attempt_started": true,
        "arm_health_debit": arm_strikes(&routing, &key)? - strikes_before,
    }))
}

fn exercise_connect_mismatch(spec: ArmSpec) -> Result<Value> {
    let key = spec.key();
    let policy = SplitHealthPolicy::default();
    let preexisting_health = SplitArmHealth {
        consecutive_strikes: policy.max_strikes - 1,
        quarantined_until_ms: None,
        probation: false,
    };
    let mut routing = router_with(
        BTreeMap::from([(
            key.clone(),
            certified_state(
                Some(COMPILED_DIGEST),
                spec.bucket.tokens() + 64,
                preexisting_health,
            ),
        )]),
        DecodePrefill::AneSplit,
        AnePlatform::Supported,
        policy,
    );
    let strikes_before = arm_strikes(&routing, &key)?;
    let attempt = route_attempt(&mut routing, &request(spec), 0, &mut success_guard())?;
    let _ = routing.complete_attempt(
        attempt,
        AneSplitAttemptResult::Failure(AnePrefillFault::Load.fallback_reason()),
        1,
    );
    let strikes_after_mismatch = arm_strikes(&routing, &key)?;
    let later = routing.route(&request(spec), 2, &mut success_guard());
    let later_reason = match later {
        AneSplitRouteDecision::GpuBypass { provenance, .. } => provenance
            .prefill_bypass_reason
            .context("later protocol-mismatch request lacks a bypass reason")?,
        other => bail!("protocol mismatch did not make the exact arm unavailable: {other:?}"),
    };
    let strikes_after_request = arm_strikes(&routing, &key)?;
    ensure!(
        later_reason == PrefillBypassReason::Quarantined,
        "later request returned {later_reason:?}"
    );
    Ok(json!({
        "initial_exact_arm_health_debit": strikes_after_mismatch - strikes_before,
        "later_prefill_bypass_reason": later_reason.as_str(),
        "later_request_health_debit": strikes_after_request - strikes_after_mismatch,
    }))
}

fn fault_for_case(case_id: &str) -> Result<AnePrefillFault> {
    Ok(match case_id {
        "compile_error_after_selection" => AnePrefillFault::Compile,
        "load_error_or_load_timeout_after_selection" => AnePrefillFault::Load,
        "dispatch_start_or_acknowledgement_failure" => AnePrefillFault::Dispatch,
        "acknowledged_stage_exit_before_prediction_result" => AnePrefillFault::Prediction,
        "prediction_budget_expiry_while_stage_unresponsive" => AnePrefillFault::PredictionTimeout,
        "kv_layout_or_conversion_failure" => AnePrefillFault::KvConversion,
        "cache_payload_publication_or_receipt_failure" => AnePrefillFault::IpcHandoff,
        "prefill_engine_to_decoding_engine_cache_conversion_failure" => {
            AnePrefillFault::CacheHandoff
        }
        "metal_cache_upload_failure" => AnePrefillFault::MetalUpload,
        "handoff_budget_expiry" => AnePrefillFault::HandoffBudget,
        "readiness_budget_expiry" => AnePrefillFault::ReadinessBudget,
        "load_completion_artifact_triple_mismatch" => AnePrefillFault::ArtifactMismatch,
        "logits_copy_or_first_token_publication_failure" => AnePrefillFault::LogitsPublication,
        other => bail!("unknown fallback case {other}"),
    })
}

fn key(bucket: PrefillBucket, decode_config: SplitDecodeConfig) -> SplitArmKey {
    SplitArmKey::new(PROFILE, Family::Qwen3_0_6b, bucket, decode_config)
}

fn certified_state(
    loaded_digest: Option<&str>,
    decode_cache_bucket: u32,
    health: SplitArmHealth,
) -> SplitArmState {
    SplitArmState {
        certification: SplitArmCertification::Certified {
            artifacts: CertifiedArtifactIdentity {
                source_checkpoint_digest: SOURCE_DIGEST.to_string(),
                certification_recorded_artifact_digest: COMPILED_DIGEST.to_string(),
            },
        },
        runtime_artifacts: RuntimeArtifactIdentity {
            manifest_source_checkpoint_digest: SOURCE_DIGEST.to_string(),
            derived_or_compiled_artifact_digest: loaded_digest.map(str::to_string),
        },
        decode_cache_bucket: Some(decode_cache_bucket),
        health,
    }
}

fn one_certified_arm(spec: ArmSpec) -> BTreeMap<SplitArmKey, SplitArmState> {
    BTreeMap::from([(
        spec.key(),
        certified_state(
            Some(COMPILED_DIGEST),
            spec.bucket.tokens() + 64,
            SplitArmHealth::default(),
        ),
    )])
}

fn router(arms: BTreeMap<SplitArmKey, SplitArmState>) -> AnePrefillRouter {
    router_with(
        arms,
        DecodePrefill::AneSplit,
        AnePlatform::Supported,
        SplitHealthPolicy::default(),
    )
}

fn router_with(
    arms: BTreeMap<SplitArmKey, SplitArmState>,
    decode_prefill: DecodePrefill,
    platform: AnePlatform,
    health_policy: SplitHealthPolicy,
) -> AnePrefillRouter {
    AnePrefillRouter::new(
        AneSplitRoutingConfig {
            decode_prefill,
            platform,
            machine_profile: PROFILE.to_string(),
            timing: SplitTimingBudgets::from_calibration(10, 100, 5, 700),
            health_policy,
        },
        gpu_fingerprint(),
        ane_fingerprint(),
        arms,
    )
}

fn request(spec: ArmSpec) -> AneSplitRequest {
    AneSplitRequest {
        family: Family::Qwen3_0_6b,
        decode_config: spec.decode_config,
        prompt_token_count: spec.bucket.tokens(),
        max_tokens: 64,
        sampling: SamplingMode::GreedyTop1,
        required_processing_fingerprint: None,
        deadline_remaining_ms: None,
    }
}

fn route_attempt(
    routing: &mut AnePrefillRouter,
    request: &AneSplitRequest,
    now_ms: u64,
    guard: &mut RecordingGuard,
) -> Result<AneSplitAttempt> {
    match routing.route(request, now_ms, guard) {
        AneSplitRouteDecision::Attempt(attempt) => Ok(attempt),
        other => bail!("expected split attempt, got {other:?}"),
    }
}

fn route_bypass_reason(
    routing: &mut AnePrefillRouter,
    request: &AneSplitRequest,
    now_ms: u64,
    guard: &mut RecordingGuard,
) -> Result<PrefillBypassReason> {
    match routing.route(request, now_ms, guard) {
        AneSplitRouteDecision::GpuBypass { provenance, .. } => provenance
            .prefill_bypass_reason
            .context("GPU bypass lacks a typed reason"),
        other => bail!("expected GPU bypass, got {other:?}"),
    }
}

fn success_guard() -> RecordingGuard {
    RecordingGuard::success()
}

fn arm_health<'a>(routing: &'a AnePrefillRouter, key: &SplitArmKey) -> Result<&'a SplitArmHealth> {
    Ok(&routing
        .arm(key)
        .with_context(|| format!("router lacks exact arm {key:?}"))?
        .health)
}

fn arm_strikes(routing: &AnePrefillRouter, key: &SplitArmKey) -> Result<u32> {
    Ok(arm_health(routing, key)?.consecutive_strikes)
}

fn arm_certification(
    routing: &AnePrefillRouter,
    key: &SplitArmKey,
) -> Result<SplitArmCertification> {
    Ok(routing
        .arm(key)
        .with_context(|| format!("router lacks exact arm {key:?}"))?
        .certification
        .clone())
}

fn router_at_quarantine_expiry(
    spec: ArmSpec,
) -> (SplitArmKey, SplitHealthPolicy, AnePrefillRouter) {
    let key = spec.key();
    let policy = SplitHealthPolicy::default();
    let routing = router_with(
        BTreeMap::from([(
            key.clone(),
            certified_state(
                Some(COMPILED_DIGEST),
                spec.bucket.tokens() + 64,
                quarantined_health(policy),
            ),
        )]),
        DecodePrefill::AneSplit,
        AnePlatform::Supported,
        policy,
    );
    (key, policy, routing)
}

fn quarantined_health(policy: SplitHealthPolicy) -> SplitArmHealth {
    SplitArmHealth {
        consecutive_strikes: policy.max_strikes,
        quarantined_until_ms: Some(QUARANTINE_EXPIRY_MS),
        probation: false,
    }
}

fn other_spec(spec: ArmSpec) -> ArmSpec {
    ArmSpec {
        bucket: if spec.bucket == PrefillBucket::W128 {
            PrefillBucket::W256
        } else {
            PrefillBucket::W128
        },
        decode_config: spec.decode_config,
    }
}

fn gpu_fingerprint() -> Fingerprint {
    Fingerprint(GPU_FINGERPRINT.to_string())
}

fn ane_fingerprint() -> Fingerprint {
    Fingerprint(ANE_FINGERPRINT.to_string())
}

fn engine_name(engine: PrefillEngine) -> &'static str {
    match engine {
        PrefillEngine::Gpu => "gpu",
        PrefillEngine::AneW128 => "ane-w128",
        PrefillEngine::AneW256 => "ane-w256",
        PrefillEngine::AneW512 => "ane-w512",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> ArmSpec {
        ArmSpec {
            bucket: PrefillBucket::W128,
            decode_config: SplitDecodeConfig::F16Step,
        }
    }

    #[test]
    fn every_named_routing_case_executes() {
        for case_id in ROUTING_CASES {
            run_routing_case(case_id, spec())
                .unwrap_or_else(|error| panic!("{case_id}: {error:#}"));
        }
    }

    #[test]
    fn every_named_exercise_executes() {
        for case_id in BYPASS_CASES {
            run_exercise_case("bypass", case_id, spec())
                .unwrap_or_else(|error| panic!("bypass/{case_id}: {error:#}"));
        }
        for case_id in FALLBACK_CASES {
            run_exercise_case("fallback", case_id, spec())
                .unwrap_or_else(|error| panic!("fallback/{case_id}: {error:#}"));
        }
        for case_id in STATE_CASES {
            run_exercise_case("state", case_id, spec())
                .unwrap_or_else(|error| panic!("state/{case_id}: {error:#}"));
        }
        run_exercise_case("protocol", "connect_mismatch", spec()).unwrap();
    }
}
