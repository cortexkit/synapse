//! Scheduler evidence ingestion for the D-009 cutover predicate.
//!
//! `decode-sched-manifest-v1` separates the five runtime-effective scheduler
//! fields (production N, yield-policy revision, aging window, DECODE weight,
//! progress-protocol revision) from the workload and evidence records. G-DEC-11
//! and the scheduler-dependent portion of G-DEC-12 cannot pass until the
//! evidence record carries committed measured values; until then the scheduler
//! evidence status is `Blocked` and production cutover stays disabled.

use crate::owned_decode_contracts::SchedulerManifest;

/// The candidate production N values; exactly one is committed. N=1 is
/// prohibited.
pub const CANDIDATE_N_VALUES: [u32; 3] = [8, 16, 32];

/// Outcome of ingesting the scheduler manifest's evidence record.
#[derive(Clone, Debug, PartialEq)]
pub enum SchedulerEvidenceStatus {
    /// The numeric scheduler commitment is present, internally consistent, and
    /// backed by executed evidence.
    Committed { production_n: u32 },
    /// The numeric commitment is missing or incomplete; every missing item is
    /// named so the blocker is actionable.
    Blocked { reasons: Vec<String> },
}

/// Whether the numeric scheduler manifest is committed and executed. This is
/// the `scheduler_evidence_committed` input of the D-009 cutover predicate.
pub fn scheduler_evidence_committed(status: &SchedulerEvidenceStatus) -> bool {
    matches!(status, SchedulerEvidenceStatus::Committed { .. })
}

/// Ingest and validate the scheduler manifest's runtime and evidence records.
///
/// A committed status requires: a runtime N in `{8,16,32}`; an evidence
/// `committed_n` equal to the runtime N; observed N, continuation count, and
/// maximum uninterruptible GPU time recorded; and non-empty sequence traces,
/// permit events, queue-depth samples, per-operation waiting samples, and
/// cancellation-latency observations. Workload observations must back the
/// cancellation-latency bound, and the embed p95 SLO must be positive.
pub fn ingest_scheduler_evidence(manifest: &SchedulerManifest) -> SchedulerEvidenceStatus {
    let mut reasons = Vec::new();
    let runtime = &manifest.runtime;
    let evidence = &manifest.evidence;
    let workload = &manifest.workload;

    if !CANDIDATE_N_VALUES.contains(&runtime.production_n) {
        reasons.push(format!(
            "runtime production_n {} is not one of the candidate values {CANDIDATE_N_VALUES:?}",
            runtime.production_n
        ));
    }

    match evidence.committed_n {
        None => reasons.push(
            "evidence committed_n is not committed (OQ-DEC-SCHED-01 measurement outstanding)"
                .to_string(),
        ),
        Some(committed) if committed != runtime.production_n => reasons.push(format!(
            "evidence committed_n {committed} does not match runtime production_n {}",
            runtime.production_n
        )),
        Some(_) => {}
    }

    if evidence.observed_n.is_none() {
        reasons.push("evidence observed_n is not recorded".to_string());
    } else if evidence.observed_n != evidence.committed_n {
        reasons.push("evidence observed_n does not match committed_n".to_string());
    }
    if evidence.continuation_count.is_none() {
        reasons.push("evidence continuation_count is not recorded".to_string());
    }
    if evidence.max_uninterruptible_gpu_time_ms.is_none() {
        reasons.push("evidence max_uninterruptible_gpu_time_ms is not recorded".to_string());
    }
    if evidence.sequence_traces.is_empty() {
        reasons.push("evidence sequence_traces are empty".to_string());
    }
    if evidence.permit_events.is_empty() {
        reasons.push("evidence permit_events are empty".to_string());
    }
    if evidence.queue_depth.is_empty() {
        reasons.push("evidence queue_depth samples are empty".to_string());
    }
    if evidence.per_operation_waiting_ms.is_empty() {
        reasons.push("evidence per_operation_waiting_ms samples are empty".to_string());
    }
    if evidence.cancellation_latency_ms.is_empty() {
        reasons.push("evidence cancellation_latency_ms observations are empty".to_string());
    }
    if workload.cancellation_observations.is_empty() {
        reasons.push("workload cancellation_observations are empty".to_string());
    }
    let slo = workload.embed_query_p95_slo_ms;
    if slo.is_nan() || slo <= 0.0 {
        reasons.push("workload embed_query_p95_slo_ms must be positive".to_string());
    }

    if reasons.is_empty() {
        SchedulerEvidenceStatus::Committed {
            production_n: runtime.production_n,
        }
    } else {
        SchedulerEvidenceStatus::Blocked { reasons }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::owned_decode_contracts::{
        SchedulerEvidenceRecord, SchedulerRuntimeRecord, SchedulerWorkloadRecord,
    };

    fn runtime() -> SchedulerRuntimeRecord {
        SchedulerRuntimeRecord {
            production_n: 16,
            yield_policy_revision: "yield-on-contention-v1".to_string(),
            decode_weight: 4,
            decode_aging_window_ms: 250,
            progress_protocol_revision: "generate-progress-v1".to_string(),
        }
    }

    fn workload() -> SchedulerWorkloadRecord {
        SchedulerWorkloadRecord {
            family: "qwen3-0.6b".to_string(),
            format: "f16".to_string(),
            context_bucket: 2048,
            prompt_length: 128,
            output_length: 64,
            embedding_concurrency: 4,
            arrival_process: "closed-loop".to_string(),
            warmup: 20,
            duration_ms: 60000,
            embed_query_p95_slo_ms: 45.0,
            baseline_calculation: "same-session embed-only p95".to_string(),
            regression_calculation: "percent delta".to_string(),
            cancellation_latency_bound_ms: 400,
            cancellation_observations: vec![125.0],
            deadline_observations: vec![90.0],
            timing_boundaries: "per-quantum boundary".to_string(),
            percentile_method: "nearest-rank".to_string(),
        }
    }

    fn committed_evidence() -> SchedulerEvidenceRecord {
        SchedulerEvidenceRecord {
            committed_n: Some(16),
            max_uninterruptible_gpu_time_ms: Some(12.5),
            observed_n: Some(16),
            continuation_count: Some(7),
            sequence_traces: vec!["trace-1".to_string()],
            permit_events: vec!["acquired".to_string()],
            queue_depth: vec![1, 2],
            per_operation_waiting_ms: vec![3.0],
            cancellation_latency_ms: vec![5.0],
            deadline_latency_ms: vec![6.0],
            ..Default::default()
        }
    }

    fn manifest(evidence: SchedulerEvidenceRecord) -> SchedulerManifest {
        SchedulerManifest {
            manifest_revision: "decode-sched-manifest-v1".to_string(),
            schema_revision: "owned-decode-contracts-v1".to_string(),
            runtime: runtime(),
            workload: workload(),
            evidence,
        }
    }

    #[test]
    fn fully_recorded_evidence_commits() {
        let status = ingest_scheduler_evidence(&manifest(committed_evidence()));
        assert_eq!(
            status,
            SchedulerEvidenceStatus::Committed { production_n: 16 }
        );
        assert!(scheduler_evidence_committed(&status));
    }

    #[test]
    fn checked_in_manifest_is_blocked_until_measurement_commits() {
        // The checked-in manifest carries the complete OQ-DEC-SCHED-01
        // factual record (per-candidate table, SLO, loadavg records,
        // machine profile, date, protocol id), but no candidate met the
        // committed embed.query p95 SLO on the M5 validation machine, so
        // committed_n stays null pending review: ingestion must report
        // blocked and cutover must stay disabled.
        let manifest_dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("owned-decode-manifests");
        let manifests = crate::owned_decode_contracts::load_manifest_dir(&manifest_dir)
            .expect("checked-in manifests load");
        let status = ingest_scheduler_evidence(&manifests.scheduler);
        let SchedulerEvidenceStatus::Blocked { reasons } = &status else {
            panic!("checked-in scheduler evidence must be blocked, got {status:?}")
        };
        assert!(
            reasons
                .iter()
                .any(|reason| reason.contains("committed_n is not committed")),
            "reasons should name the missing commitment: {reasons:?}"
        );
        assert!(!scheduler_evidence_committed(&status));
    }

    #[test]
    fn committed_n_must_match_runtime_n() {
        let mut evidence = committed_evidence();
        evidence.committed_n = Some(8);
        let SchedulerEvidenceStatus::Blocked { reasons } =
            ingest_scheduler_evidence(&manifest(evidence))
        else {
            panic!("mismatched committed_n must block")
        };
        assert!(reasons
            .iter()
            .any(|r| r.contains("does not match runtime production_n")));
    }

    #[test]
    fn non_candidate_n_is_rejected() {
        let mut manifest = manifest(committed_evidence());
        manifest.runtime.production_n = 4;
        manifest.evidence.committed_n = Some(4);
        manifest.evidence.observed_n = Some(4);
        let SchedulerEvidenceStatus::Blocked { reasons } = ingest_scheduler_evidence(&manifest)
        else {
            panic!("N=4 must block")
        };
        assert!(reasons
            .iter()
            .any(|r| r.contains("not one of the candidate values")));
    }

    #[test]
    fn each_missing_evidence_field_is_named() {
        let status = ingest_scheduler_evidence(&manifest(SchedulerEvidenceRecord::default()));
        let SchedulerEvidenceStatus::Blocked { reasons } = status else {
            panic!("empty evidence must block")
        };
        for expected in [
            "committed_n",
            "observed_n",
            "continuation_count",
            "max_uninterruptible_gpu_time_ms",
            "sequence_traces",
            "permit_events",
            "queue_depth",
            "per_operation_waiting_ms",
            "cancellation_latency_ms",
        ] {
            assert!(
                reasons.iter().any(|r| r.contains(expected)),
                "missing {expected} should be named in {reasons:?}"
            );
        }
    }

    #[test]
    fn workload_observations_and_slo_are_required() {
        let mut manifest = manifest(committed_evidence());
        manifest.workload.cancellation_observations = Vec::new();
        manifest.workload.embed_query_p95_slo_ms = 0.0;
        let SchedulerEvidenceStatus::Blocked { reasons } = ingest_scheduler_evidence(&manifest)
        else {
            panic!("missing workload observations must block")
        };
        assert!(reasons
            .iter()
            .any(|r| r.contains("cancellation_observations")));
        assert!(reasons.iter().any(|r| r.contains("embed_query_p95_slo_ms")));
    }
}
