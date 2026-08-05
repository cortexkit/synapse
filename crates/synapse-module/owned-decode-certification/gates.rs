//! Acceptance-gate runner (G-DEC-01 through G-DEC-12) and release evidence.
//!
//! The runner executes every gate that is executable in the current build and
//! records one status per gate. No applicable gate is ever skipped: gates with
//! outstanding measurement commitments (G-DEC-11, and the scheduler-dependent
//! portion of G-DEC-12) are reported `Blocked` with the blocker named, never
//! `Skipped`. [`release_ready`] is false while any gate is blocked or failed,
//! which keeps production cutover disabled until the committed scheduler
//! manifest and all required evidence pass.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use synapse_core::Fingerprint;

use crate::owned_decode_certification::cutover::{
    cutover_inputs_from_evidence, CutoverEvidenceInputs,
};
use crate::owned_decode_certification::fixture_groups::{
    GroupOutcome, CONSTRAINED_NEGATIVE_GROUP, CONSTRAINED_POSITIVE_GROUP, REQUEST_PROCESSING_GROUP,
    SCHEDULER_CONTINUITY_GROUP,
};
use crate::owned_decode_certification::fixtures::{
    parity_battery, OracleStore, ParityFixture, PARITY_GROUP,
};
use crate::owned_decode_certification::probe::{compare_streams, CertificationProbe, DecodeProbe};
use crate::owned_decode_certification::scheduler_evidence::{
    ingest_scheduler_evidence, scheduler_evidence_committed, SchedulerEvidenceStatus,
};
use crate::owned_decode_contracts::ManifestDir;
use crate::owned_decode_grammar_scheduler::QueueClass;
use crate::owned_decode_grammar_scheduler::{
    compile_grammar, load_automaton, CompileContext, GrammarSubsetManifest,
};
use crate::owned_decode_routing::certification::{CertificationStore, StructuralBandChecker};
use crate::owned_decode_routing::error::OwnedDecodeError;
use crate::owned_decode_routing::family::{Family, FamilyRegistry};
use crate::owned_decode_routing::identity::{
    ActivationDType, ConstraintRuntimeIdentity, DecodeIdentityInputs, ProcessingIdentityInputs,
    Q8Identity, RuntimeConfigManifest, WeightQuant,
};
use crate::owned_decode_routing::lane::{cutover_enabled, CutoverInputs, CutoverRecord};
use crate::owned_decode_routing::q8ingest::{Q8IngestRegistry, TrustState};
use crate::owned_decode_routing::{
    CatalogEntry, DecodeDispatch, DispatchedCommand, ExecutionSuccess, OwnedDecodeRouter,
    RoutingEnvironment, CATALOG_ENGINE, CATALOG_LANE, CATALOG_RISK_CLASS, CATALOG_TASK,
    CATALOG_WORKER,
};

/// The twelve acceptance gates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GateId {
    #[serde(rename = "G-DEC-01")]
    GDec01,
    #[serde(rename = "G-DEC-02")]
    GDec02,
    #[serde(rename = "G-DEC-03")]
    GDec03,
    #[serde(rename = "G-DEC-04")]
    GDec04,
    #[serde(rename = "G-DEC-05")]
    GDec05,
    #[serde(rename = "G-DEC-06")]
    GDec06,
    #[serde(rename = "G-DEC-07")]
    GDec07,
    #[serde(rename = "G-DEC-08")]
    GDec08,
    #[serde(rename = "G-DEC-09")]
    GDec09,
    #[serde(rename = "G-DEC-10")]
    GDec10,
    #[serde(rename = "G-DEC-11")]
    GDec11,
    #[serde(rename = "G-DEC-12")]
    GDec12,
}

/// Every gate, in order.
pub const ALL_GATES: [GateId; 12] = [
    GateId::GDec01,
    GateId::GDec02,
    GateId::GDec03,
    GateId::GDec04,
    GateId::GDec05,
    GateId::GDec06,
    GateId::GDec07,
    GateId::GDec08,
    GateId::GDec09,
    GateId::GDec10,
    GateId::GDec11,
    GateId::GDec12,
];

impl GateId {
    /// The canonical gate label (e.g. `G-DEC-04`).
    pub fn as_str(self) -> &'static str {
        match self {
            GateId::GDec01 => "G-DEC-01",
            GateId::GDec02 => "G-DEC-02",
            GateId::GDec03 => "G-DEC-03",
            GateId::GDec04 => "G-DEC-04",
            GateId::GDec05 => "G-DEC-05",
            GateId::GDec06 => "G-DEC-06",
            GateId::GDec07 => "G-DEC-07",
            GateId::GDec08 => "G-DEC-08",
            GateId::GDec09 => "G-DEC-09",
            GateId::GDec10 => "G-DEC-10",
            GateId::GDec11 => "G-DEC-11",
            GateId::GDec12 => "G-DEC-12",
        }
    }
}

/// One gate's outcome. `Skipped` exists only to be prohibited: no applicable
/// gate may ever carry it.
#[derive(Clone, Debug, PartialEq)]
pub enum GateStatus {
    /// The gate executed and every check passed.
    Passed { evidence: Vec<String> },
    /// The gate executed and a check failed.
    Failed { reason: String },
    /// The gate cannot pass until a named external commitment lands (e.g. the
    /// numeric scheduler manifest). Blocked is not skipped: the gate machinery
    /// ran and recorded exactly what is outstanding.
    Blocked { reason: String },
    /// Prohibited for applicable gates; retained so a violation is expressible
    /// and detectable by [`applicable_skips`].
    Skipped { reason: String },
}

/// One same-session throughput comparison for a single lane (G-DEC-10).
#[derive(Clone, Debug, PartialEq)]
pub struct ThroughputEvidence {
    pub family: Family,
    pub weight_quant: WeightQuant,
    pub spike_tokens_per_sec: f64,
    pub production_tokens_per_sec: f64,
    /// Baseline chain-K is one.
    pub chain_k: u32,
    /// Baseline batched verification is disabled.
    pub batched_verification: bool,
    /// Spike and production ran consecutively in one session.
    pub same_session: bool,
    /// Startup, first load, and first Q8 ingest are reported separately from
    /// steady-state throughput.
    pub startup_reported_separately: bool,
}

/// Steady-state production throughput must reach at least this fraction of the
/// same-session spike result.
pub const THROUGHPUT_RATIO_BOUND: f64 = 0.90;

impl ThroughputEvidence {
    pub fn ratio(&self) -> f64 {
        self.production_tokens_per_sec / self.spike_tokens_per_sec
    }
}

/// One grammar-cost measurement result (G-DEC-09).
#[derive(Clone, Debug, PartialEq)]
pub struct GrammarCostEvidence {
    /// Constrained masking time per token at p95.
    pub masking_p95_ms_per_token: f64,
    /// Constrained throughput divided by the corresponding unconstrained
    /// owned-worker throughput.
    pub constrained_throughput_ratio: f64,
}

/// Constrained masking p95 ship bound, in ms/token.
pub const GRAMMAR_MASKING_P95_BOUND_MS: f64 = 0.50;
/// Constrained throughput must reach at least this fraction of unconstrained.
pub const GRAMMAR_THROUGHPUT_RATIO_BOUND: f64 = 0.90;

/// The assembled release evidence set.
#[derive(Clone, Debug, PartialEq)]
pub struct ReleaseEvidence {
    pub fixture_registry_revision: String,
    /// Fixture group -> every executed fixture ID.
    pub executed_fixtures: BTreeMap<String, Vec<String>>,
    pub certification_evidence_ids: Vec<String>,
    pub gate_statuses: BTreeMap<GateId, GateStatus>,
    pub scheduler_status: SchedulerEvidenceStatus,
    pub ci_lane_revision: String,
    pub wire_binding_revision: String,
}

/// Whether the release evidence set permits shipping: every gate passed.
pub fn release_ready(evidence: &ReleaseEvidence) -> bool {
    evidence
        .gate_statuses
        .values()
        .all(|status| matches!(status, GateStatus::Passed { .. }))
}

/// Gates carrying a `Skipped` status. Must be empty for any real run: every
/// applicable gate executes.
pub fn applicable_skips(evidence: &ReleaseEvidence) -> Vec<GateId> {
    evidence
        .gate_statuses
        .iter()
        .filter(|(_, status)| matches!(status, GateStatus::Skipped { .. }))
        .map(|(gate, _)| *gate)
        .collect()
}

/// The gate runner, built from the checked-in manifest directory.
pub struct GateRunner {
    manifests: ManifestDir,
}

impl GateRunner {
    pub fn new(manifests: ManifestDir) -> Self {
        Self { manifests }
    }

    /// Execute every gate and assemble the release evidence. `oracle` and
    /// `probe` supply the parity battery comparison (G-DEC-04/05); `throughput`
    /// and `grammar_cost` supply the measurement evidence a machine run
    /// produces (G-DEC-09/10).
    pub fn run_all(
        &self,
        oracle: &OracleStore,
        probe: &mut dyn DecodeProbe,
        throughput: &[ThroughputEvidence],
        grammar_cost: Option<&GrammarCostEvidence>,
    ) -> ReleaseEvidence {
        let scheduler_status = ingest_scheduler_evidence(&self.manifests.scheduler);
        let mut gate_statuses = BTreeMap::new();
        let mut executed_fixtures: BTreeMap<String, Vec<String>> = BTreeMap::new();

        // Execute the non-parity fixture-registry groups from the checked-in
        // registry JSON; every executed entry ID is recorded per group.
        let request_processing =
            crate::owned_decode_certification::fixture_groups::run_request_processing(
                &self.manifests,
            );
        let constrained_positive =
            crate::owned_decode_certification::fixture_groups::run_constrained_positive(
                &self.manifests,
            );
        let constrained_negative =
            crate::owned_decode_certification::fixture_groups::run_constrained_negative(
                &self.manifests,
            );
        let scheduler_continuity =
            crate::owned_decode_certification::fixture_groups::run_scheduler_continuity(
                &self.manifests,
            );
        executed_fixtures.insert(
            REQUEST_PROCESSING_GROUP.to_string(),
            request_processing.executed_ids.clone(),
        );
        executed_fixtures.insert(
            CONSTRAINED_POSITIVE_GROUP.to_string(),
            constrained_positive.executed_ids.clone(),
        );
        executed_fixtures.insert(
            CONSTRAINED_NEGATIVE_GROUP.to_string(),
            constrained_negative.executed_ids.clone(),
        );
        executed_fixtures.insert(
            SCHEDULER_CONTINUITY_GROUP.to_string(),
            scheduler_continuity.executed_ids.clone(),
        );

        gate_statuses.insert(GateId::GDec01, self.gate_01_catalog(request_processing));
        gate_statuses.insert(GateId::GDec02, self.gate_02_q8_ingest());
        gate_statuses.insert(GateId::GDec03, self.gate_03_ownership());

        let (parity_status, parity_ids) = self.gate_04_parity(oracle, probe);
        gate_statuses.insert(GateId::GDec04, parity_status);
        executed_fixtures.insert(PARITY_GROUP.to_string(), parity_ids);

        let (cert_status, cert_ids) = self.gate_05_certification(oracle, probe, &scheduler_status);
        gate_statuses.insert(GateId::GDec05, cert_status);

        gate_statuses.insert(GateId::GDec06, self.gate_06_identity());
        gate_statuses.insert(GateId::GDec07, self.gate_07_oneshot());
        gate_statuses.insert(
            GateId::GDec08,
            self.gate_08_grammar(constrained_positive, constrained_negative),
        );
        gate_statuses.insert(GateId::GDec09, self.gate_09_grammar_cost(grammar_cost));
        gate_statuses.insert(GateId::GDec10, self.gate_10_throughput(throughput));
        gate_statuses.insert(GateId::GDec11, Self::gate_11_scheduler(&scheduler_status));
        gate_statuses.insert(
            GateId::GDec12,
            self.gate_12_ci(&gate_statuses, scheduler_continuity),
        );

        ReleaseEvidence {
            fixture_registry_revision: self.manifests.fixture_registry.manifest_revision.clone(),
            executed_fixtures,
            certification_evidence_ids: cert_ids,
            gate_statuses,
            scheduler_status,
            ci_lane_revision: self.manifests.ci_lane.manifest_revision.clone(),
            wire_binding_revision: self.manifests.wire_bindings.manifest_revision.clone(),
        }
    }

    /// G-DEC-01: catalog, processing identity, and load. Both families under
    /// canonical identity, shippable context buckets from the checked-in
    /// manifest, per-family processing-asset registration, and the registry's
    /// request-processing fixture group executed.
    pub fn gate_01_catalog(&self, request_processing: GroupOutcome) -> GateStatus {
        let mut evidence = Vec::new();

        let canonical_ok = CATALOG_ENGINE == "owned-metal-decode"
            && CATALOG_TASK == "generate"
            && CATALOG_LANE == "decode"
            && CATALOG_WORKER == "supervised"
            && CATALOG_RISK_CLASS == "abort_capable";
        if !canonical_ok {
            return GateStatus::Failed {
                reason: "catalog canonical identity constants are wrong".to_string(),
            };
        }
        evidence.push("canonical identity engine/task/lane/worker/risk_class".to_string());

        let mut families_seen = BTreeSet::new();
        for family in &self.manifests.context_buckets.families {
            families_seen.insert(family.family.clone());
            if family.verified_buckets.is_empty() {
                return GateStatus::Failed {
                    reason: format!("family {} has no verified bucket", family.family),
                };
            }
            let initial_set: BTreeSet<u32> = [512, 1024, 2048].into_iter().collect();
            if !family
                .verified_buckets
                .iter()
                .all(|b| initial_set.contains(b))
            {
                return GateStatus::Failed {
                    reason: format!(
                        "family {} lists buckets outside the initial {{512,1024,2048}} set",
                        family.family
                    ),
                };
            }
            evidence.push(format!(
                "context buckets {}: {:?}",
                family.family, family.verified_buckets
            ));
        }
        for required in ["qwen3-0.6b", "lfm2-1.2b"] {
            if !families_seen.contains(required) {
                return GateStatus::Failed {
                    reason: format!("context manifest missing family {required}"),
                };
            }
        }

        let registry = FamilyRegistry::production();
        for family in Family::all() {
            if !registry.contains(family) {
                return GateStatus::Failed {
                    reason: format!(
                        "family {} lacks processing-asset registration",
                        family.as_str()
                    ),
                };
            }
        }
        evidence.push("both families register tokenizer/template/special/stop/detok".to_string());

        // The registry's request-processing group executes against the
        // module's family registrations.
        if let Err(reason) = request_processing.result {
            return GateStatus::Failed {
                reason: format!("request-processing fixture group: {reason}"),
            };
        }
        evidence.push(format!(
            "request-processing fixture group executed: {}",
            request_processing.executed_ids.join(", ")
        ));

        GateStatus::Passed { evidence }
    }

    /// G-DEC-02: derived Q8 identity. Atomic ingest keyed by source digest and
    /// quantizer revision; registered digests verify before trust; unregistered
    /// objects stay untrusted; mismatch poisons; reuse avoids requantization;
    /// rederivation reproduces; rotation creates a distinct key.
    pub fn gate_02_q8_ingest(&self) -> GateStatus {
        let mut evidence = Vec::new();
        let mut registry = Q8IngestRegistry::new();
        let source = "g02-source-digest";
        let quantizer = "quantizer-v1";
        let derived = "g02-derived-digest";

        registry.register_expected_digest(source, quantizer, derived);
        let first = match registry.load_or_ingest(source, quantizer, "q8_0", b"source", |_| {
            derived.to_string()
        }) {
            Ok(outcome) => outcome,
            Err(error) => {
                return GateStatus::Failed {
                    reason: format!("registered digest ingest failed: {error:?}"),
                }
            }
        };
        if first.reused || first.entry.trust_state != TrustState::Trusted {
            return GateStatus::Failed {
                reason: "first ingest with a registered digest must publish trusted".to_string(),
            };
        }
        evidence.push("registered expected digest verifies before trust".to_string());

        let again = registry
            .load_or_ingest(source, quantizer, "q8_0", b"source", |_| {
                panic!("reuse must not requantize")
            })
            .expect("trusted entry loads");
        if !again.reused {
            return GateStatus::Failed {
                reason: "second load must reuse without requantization".to_string(),
            };
        }
        evidence.push("reuse avoids requantization".to_string());

        // No registered digest: publishes untrusted and fails closed.
        let untrusted =
            registry.load_or_ingest("g02-source-2", quantizer, "q8_0", b"other", |_| {
                "g02-derived-2".to_string()
            });
        let untrusted_entry = registry
            .entry("g02-source-2", quantizer)
            .expect("untrusted object is published");
        if untrusted != Err(OwnedDecodeError::NotCertified)
            || untrusted_entry.trust_state != TrustState::Untrusted
        {
            return GateStatus::Failed {
                reason: "unregistered digest must publish untrusted and refuse".to_string(),
            };
        }
        evidence.push("unregistered digest stays untrusted and fails closed".to_string());

        // Mismatch poisons.
        registry.register_expected_digest("g02-source-3", quantizer, "expected-digest");
        let poisoned = registry.load_or_ingest("g02-source-3", quantizer, "q8_0", b"third", |_| {
            "actual-digest".to_string()
        });
        if poisoned != Err(OwnedDecodeError::ArtifactPoisoned) {
            return GateStatus::Failed {
                reason: "digest mismatch must poison the artifact".to_string(),
            };
        }
        evidence.push("digest mismatch marks artifact_poisoned".to_string());

        // Rederivation reproduces the registered digest.
        registry.evict(source, quantizer);
        match registry.rederive(source, quantizer, b"source", |_| derived.to_string()) {
            Ok(entry) if entry.trust_state == TrustState::Trusted => {
                evidence.push("rederivation reproduces the registered digest".to_string())
            }
            other => {
                return GateStatus::Failed {
                    reason: format!("rederivation must reproduce trusted: {other:?}"),
                }
            }
        }

        // Quantizer-revision rotation creates a distinct key.
        if registry.entry(source, "quantizer-v2").is_some() {
            return GateStatus::Failed {
                reason: "rotated quantizer revision must be a distinct key".to_string(),
            };
        }
        evidence.push("quantizer-revision rotation creates a distinct artifact key".to_string());

        GateStatus::Passed { evidence }
    }

    /// G-DEC-03: isolation, ownership, and retry. The ownership manifest covers
    /// every required fault-site group with an ASan test name and run record,
    /// and the wire error bindings carry literal IDs.
    pub fn gate_03_ownership(&self) -> GateStatus {
        let ownership = &self.manifests.ownership;
        let required_groups: BTreeSet<&str> = [
            "allocation",
            "ownership_transfer",
            "partial_initialization",
            "generation",
            "cancellation",
            "timeout",
            "unload",
            "shutdown",
            "lfm2_conv_cache",
        ]
        .into_iter()
        .collect();
        let present_groups: BTreeSet<&str> = ownership
            .fault_sites
            .iter()
            .map(|s| s.group.as_str())
            .collect();
        let missing: Vec<&str> = required_groups
            .difference(&present_groups)
            .copied()
            .collect();
        if !missing.is_empty() {
            return GateStatus::Failed {
                reason: format!("ownership manifest missing fault-site groups: {missing:?}"),
            };
        }
        for site in &ownership.fault_sites {
            if site.asan_test_name.is_empty() || site.run_record.is_empty() {
                return GateStatus::Failed {
                    reason: format!(
                        "fault site {} lacks an ASan test name or run record",
                        site.id
                    ),
                };
            }
        }
        let mut evidence = vec![format!(
            "ownership manifest covers {} fault sites with ASan test names and run records",
            ownership.fault_sites.len()
        )];

        let bindings = &self.manifests.wire_bindings;
        if bindings.deadline_error_id.is_empty()
            || bindings.cancellation_error_id.is_empty()
            || bindings.deadline_error_id.contains("existing_")
            || bindings.cancellation_error_id.contains("existing_")
        {
            return GateStatus::Failed {
                reason: "wire error bindings must carry literal IDs, not placeholders".to_string(),
            };
        }
        evidence.push(format!(
            "literal wire error bindings: {} / {}",
            bindings.deadline_error_id, bindings.cancellation_error_id
        ));

        GateStatus::Passed { evidence }
    }

    /// G-DEC-04: port parity. Every lane of the battery (both families, both
    /// formats) matches the independent oracle within its structural band: zero
    /// forks for Q8, at most two top-2 swaps for f16.
    pub fn gate_04_parity(
        &self,
        oracle: &OracleStore,
        probe: &mut dyn DecodeProbe,
    ) -> (GateStatus, Vec<String>) {
        let battery = parity_battery();
        let checker = StructuralBandChecker::from_manifest(&self.manifests.structural_band);
        let mut evidence = Vec::new();
        let mut executed = Vec::new();

        for fixture in &battery {
            match self.run_parity_fixture(oracle, probe, fixture, &checker) {
                Ok(line) => {
                    evidence.push(line);
                    executed.push(fixture.id.clone());
                }
                Err(reason) => {
                    return (
                        GateStatus::Failed {
                            reason: format!("{}: {reason}", fixture.id),
                        },
                        executed,
                    )
                }
            }
        }
        (GateStatus::Passed { evidence }, executed)
    }

    fn run_parity_fixture(
        &self,
        oracle: &OracleStore,
        probe: &mut dyn DecodeProbe,
        fixture: &ParityFixture,
        checker: &StructuralBandChecker,
    ) -> Result<String, String> {
        let mut divergences = Vec::new();
        for prompt_index in 0..fixture.prompt_count {
            let oracle_stream = oracle
                .stream(&fixture.id, prompt_index)
                .ok_or_else(|| format!("oracle missing prompt {prompt_index}"))?;
            let produced = probe.generate(fixture, prompt_index);
            divergences.extend(compare_streams(&produced, oracle_stream, prompt_index));
        }
        let swaps = divergences.len() as u32;
        let max = checker
            .max_top2_swaps(fixture.family.as_str(), fixture.weight_quant.as_str())
            .ok_or_else(|| "no structural-band rule for lane".to_string())?;
        if swaps > max {
            return Err(format!(
                "{swaps} fork(s) exceed the structural band ceiling of {max}"
            ));
        }
        Ok(format!(
            "{}: {} prompts compared, {swaps} fork(s) within band (ceiling {max})",
            fixture.id, fixture.prompt_count
        ))
    }

    /// G-DEC-05: certification and D-009. Certification rows record for both
    /// families through the probe; the cutover predicate stays false while the
    /// scheduler evidence is blocked and becomes true when every input holds.
    pub fn gate_05_certification(
        &self,
        oracle: &OracleStore,
        probe: &mut dyn DecodeProbe,
        scheduler_status: &SchedulerEvidenceStatus,
    ) -> (GateStatus, Vec<String>) {
        let battery = parity_battery();
        let checker = StructuralBandChecker::from_manifest(&self.manifests.structural_band);
        let cert_probe = CertificationProbe::new(
            "gate-runner-profile",
            self.manifests.fixture_registry.manifest_revision.clone(),
            oracle,
            checker,
        );
        let mut store = CertificationStore::new();
        let mut cert_ids = Vec::new();
        let mut evidence = Vec::new();

        // Certify the f16 lane of each family (byte-identical probes record
        // zero-fork signatures).
        let mut certified_fingerprints = Vec::new();
        for fixture in battery
            .iter()
            .filter(|f| f.weight_quant == WeightQuant::F16)
        {
            let fp = decode_fingerprint_for_fixture(fixture);
            match cert_probe.certify_unconstrained_lane(probe, fixture, fp.clone(), &mut store) {
                Ok(ev) => {
                    cert_ids.push(ev.evidence_id());
                    evidence.push(format!(
                        "certified {} (top2 swaps: {})",
                        fixture.id, ev.top2_swaps
                    ));
                    certified_fingerprints.push(fp);
                }
                Err(error) => {
                    return (
                        GateStatus::Failed {
                            reason: format!("certification failed for {}: {error:?}", fixture.id),
                        },
                        cert_ids,
                    )
                }
            }
        }
        if certified_fingerprints.len() != 2 {
            return (
                GateStatus::Failed {
                    reason: "expected one f16 certification per family".to_string(),
                },
                cert_ids,
            );
        }

        // The cutover predicate must stay false while the scheduler evidence is
        // blocked, even with certification and every other input satisfied.
        let bindings = &self.manifests.wire_bindings;
        let record = CutoverRecord {
            machine_profile_hash: "gate-runner-profile".to_string(),
            enabled_catalog_entry_ids: Vec::new(),
            decode_fingerprints: certified_fingerprints.clone(),
            processing_fingerprints: Vec::new(),
            constrained_runtime_identities: Vec::new(),
            runtime_config_digest: "gate-runner-runtime".to_string(),
            fixture_registry_revision: self.manifests.fixture_registry.manifest_revision.clone(),
            context_bucket_manifest_revision: self
                .manifests
                .context_buckets
                .manifest_revision
                .clone(),
            scheduler_manifest_revision: self.manifests.scheduler.manifest_revision.clone(),
            certification_evidence_ids: cert_ids.clone(),
            wire_error_binding_revision: bindings.manifest_revision.clone(),
            acceptance_gate_evidence: Vec::new(),
            grammar_enabled: false,
        };
        let inputs = cutover_inputs_from_evidence(&CutoverEvidenceInputs {
            cert_store: &store,
            machine_profile_hash: "gate-runner-profile",
            decode_fingerprint: &certified_fingerprints[0],
            constraint_runtime_identity: None,
            artifacts_trusted: true,
            identities_installed: true,
            quarantined: false,
            wire_bindings: bindings,
            gates_passed: true,
            scheduler_status,
        });
        if scheduler_evidence_committed(scheduler_status) {
            if !cutover_enabled(&record, &inputs) {
                return (
                    GateStatus::Failed {
                        reason: "cutover predicate must hold when every input is satisfied"
                            .to_string(),
                    },
                    cert_ids,
                );
            }
            evidence.push("cutover predicate holds with committed scheduler evidence".to_string());
        } else if cutover_enabled(&record, &inputs) {
            return (
                GateStatus::Failed {
                    reason: "cutover predicate must stay false while scheduler evidence is blocked"
                        .to_string(),
                },
                cert_ids,
            );
        } else {
            evidence.push(
                "cutover predicate stays false while scheduler evidence is blocked".to_string(),
            );
        }

        (GateStatus::Passed { evidence }, cert_ids)
    }

    /// G-DEC-06: identity behavior. Decode fingerprint rotates on artifact,
    /// arithmetic, family, and quantization changes; processing fingerprint
    /// rotates only on processing assets; constraint identity is field
    /// sensitive; scheduler settings rotate the runtime digest and never the
    /// decode fingerprint.
    pub fn gate_06_identity(&self) -> GateStatus {
        let mut evidence = Vec::new();
        let base = DecodeIdentityInputs {
            family: Family::Qwen3_0_6b,
            activation_dtype: ActivationDType::F16,
            weight_quant: WeightQuant::F16,
            artifact_source_digest: "artifact-v1".to_string(),
            arithmetic_identity_revision: "arithmetic-v1".to_string(),
            q8: None,
        };
        let base_fp = base.decode_fingerprint().expect("valid base inputs");

        let mut rotated = base.clone();
        rotated.artifact_source_digest = "artifact-v2".to_string();
        if rotated.decode_fingerprint().expect("valid") == base_fp {
            return GateStatus::Failed {
                reason: "artifact bytes must rotate decode_fingerprint".to_string(),
            };
        }
        let mut rotated = base.clone();
        rotated.arithmetic_identity_revision = "arithmetic-v2".to_string();
        if rotated.decode_fingerprint().expect("valid") == base_fp {
            return GateStatus::Failed {
                reason: "arithmetic identity revision must rotate decode_fingerprint".to_string(),
            };
        }
        let mut rotated = base.clone();
        rotated.family = Family::Lfm2_1_2b;
        if rotated.decode_fingerprint().expect("valid") == base_fp {
            return GateStatus::Failed {
                reason: "engine family must rotate decode_fingerprint".to_string(),
            };
        }
        let rotated = DecodeIdentityInputs {
            family: Family::Qwen3_0_6b,
            activation_dtype: ActivationDType::F16,
            weight_quant: WeightQuant::Q8_0,
            artifact_source_digest: "artifact-v1".to_string(),
            arithmetic_identity_revision: "arithmetic-v1".to_string(),
            q8: Some(Q8Identity {
                quantizer_revision: "quantizer-v1".to_string(),
                derived_digest: "derived-v1".to_string(),
            }),
        };
        if rotated.decode_fingerprint().expect("valid") == base_fp {
            return GateStatus::Failed {
                reason: "weight quantization must rotate decode_fingerprint".to_string(),
            };
        }
        evidence.push(
            "decode_fingerprint rotates on artifact, arithmetic, family, and quant changes"
                .to_string(),
        );

        // Scheduler settings never rotate the decode fingerprint: the decode
        // identity inputs carry no scheduler field, while the runtime digest
        // rotates on exactly the runtime-effective scheduler fields.
        let runtime_a = runtime_manifest(16, 4, 250);
        let runtime_b = runtime_manifest(32, 8, 500);
        if runtime_a.digest() == runtime_b.digest() {
            return GateStatus::Failed {
                reason: "scheduler fields must rotate runtime_config_digest".to_string(),
            };
        }
        if base.decode_fingerprint().expect("valid") != base_fp {
            return GateStatus::Failed {
                reason: "scheduler changes must not rotate decode_fingerprint".to_string(),
            };
        }
        evidence.push("scheduler fields rotate runtime_config_digest only".to_string());

        // Processing fingerprint rotates on processing assets only.
        let processing = ProcessingIdentityInputs {
            decode_fingerprint: base_fp.clone(),
            tokenizer_sanitized_digest: "tok-v1".to_string(),
            prompt_template_revision: "template-v1".to_string(),
            special_token_policy_revision: "special-v1".to_string(),
            stop_token_policy_revision: "stop-v1".to_string(),
            detokenizer_revision: "detok-v1".to_string(),
        };
        let processing_fp = processing.processing_fingerprint();
        let mut rotated = processing.clone();
        rotated.tokenizer_sanitized_digest = "tok-v2".to_string();
        if rotated.processing_fingerprint() == processing_fp {
            return GateStatus::Failed {
                reason: "processing assets must rotate processing_fingerprint".to_string(),
            };
        }
        if processing.processing_fingerprint() != processing_fp {
            return GateStatus::Failed {
                reason: "processing_fingerprint must be stable for identical inputs".to_string(),
            };
        }
        evidence.push("processing_fingerprint rotates on processing assets only".to_string());

        // Constraint runtime identity is field sensitive across all seven fields.
        let cri = ConstraintRuntimeIdentity {
            base_decode_fingerprint: base_fp.clone(),
            representation_revision: "token-id-json-constraint-v1".to_string(),
            grammar_subset_revision: "synapse-json-schema-v1".to_string(),
            grammar_compiler_revision: "grammar-compiler-v1".to_string(),
            tokenizer_vocabulary_digest: "vocab-v1".to_string(),
            limits_manifest_id: "grammar-limits-v1".to_string(),
            worker_constraint_runtime_revision: "worker-constraint-v1".to_string(),
        };
        let cri_digest = cri.digest();
        let rotations = [
            ConstraintRuntimeIdentity {
                base_decode_fingerprint: Fingerprint("other".to_string()),
                ..cri.clone()
            },
            ConstraintRuntimeIdentity {
                representation_revision: "v2".to_string(),
                ..cri.clone()
            },
            ConstraintRuntimeIdentity {
                grammar_subset_revision: "v2".to_string(),
                ..cri.clone()
            },
            ConstraintRuntimeIdentity {
                grammar_compiler_revision: "v2".to_string(),
                ..cri.clone()
            },
            ConstraintRuntimeIdentity {
                tokenizer_vocabulary_digest: "vocab-v2".to_string(),
                ..cri.clone()
            },
            ConstraintRuntimeIdentity {
                limits_manifest_id: "v2".to_string(),
                ..cri.clone()
            },
            ConstraintRuntimeIdentity {
                worker_constraint_runtime_revision: "v2".to_string(),
                ..cri.clone()
            },
        ];
        for rotated in &rotations {
            if rotated.digest() == cri_digest {
                return GateStatus::Failed {
                    reason: "every constraint runtime identity field must be digest-sensitive"
                        .to_string(),
                };
            }
        }
        evidence.push("constraint runtime identity is sensitive to all seven fields".to_string());

        GateStatus::Passed { evidence }
    }

    /// G-DEC-07: end-to-end oneshot. Lane selection, admission, dispatch, and
    /// provenance over the routing seam for a certified entry; owned execution
    /// selects the owned lane while the DECODE queue class serializes as
    /// `decode`.
    pub fn gate_07_oneshot(&self) -> GateStatus {
        // The DECODE scheduler class exists and serializes as `decode`.
        let serialized = serde_json::to_value(QueueClass::Decode).expect("serializes");
        if serialized != serde_json::json!("decode") {
            return GateStatus::Failed {
                reason: "QueueClass::Decode must serialize as \"decode\"".to_string(),
            };
        }

        let profile = "gate-runner-profile";
        let entry = gate_catalog_entry();
        let fp = decode_fingerprint_for_entry(&entry);
        let mut store = CertificationStore::new();
        store.certify_unconstrained(profile, fp.clone());
        let router = OwnedDecodeRouter::new(
            FamilyRegistry::production(),
            self.manifests.context_buckets.clone(),
            Q8IngestRegistry::new(),
            Box::new(store),
        );
        // The cutover flag is not set directly: it is evaluated from a D-009
        // record and predicate inputs whose conditions all hold, modeling a
        // profile legitimately enabled for the owned lane.
        let record = CutoverRecord {
            machine_profile_hash: profile.to_string(),
            enabled_catalog_entry_ids: vec![entry.entry_id.clone()],
            decode_fingerprints: vec![fp.clone()],
            processing_fingerprints: Vec::new(),
            constrained_runtime_identities: Vec::new(),
            runtime_config_digest: "gate-runner-runtime".to_string(),
            fixture_registry_revision: self.manifests.fixture_registry.manifest_revision.clone(),
            context_bucket_manifest_revision: self
                .manifests
                .context_buckets
                .manifest_revision
                .clone(),
            scheduler_manifest_revision: self.manifests.scheduler.manifest_revision.clone(),
            certification_evidence_ids: Vec::new(),
            wire_error_binding_revision: self.manifests.wire_bindings.manifest_revision.clone(),
            acceptance_gate_evidence: vec!["G-DEC-07 cutover evaluation".to_string()],
            grammar_enabled: false,
        };
        let inputs = CutoverInputs {
            artifacts_trusted: true,
            identities_installed: true,
            unconstrained_certified: true,
            constrained_certified: true,
            quarantined: false,
            wire_bindings_literal: true,
            gates_passed: true,
            scheduler_evidence_committed: true,
        };
        let env = RoutingEnvironment::with_cutover_evaluated(
            profile,
            false,
            false,
            None,
            BTreeSet::new(),
            None,
            &record,
            &inputs,
        );
        let request: crate::owned_decode_routing::request::OneshotRequest =
            serde_json::from_value(serde_json::json!({
                "family": "qwen3-0.6b",
                "weight_quant": "f16",
                "prompt_token_count": 100,
                "max_tokens": 64
            }))
            .expect("request parses");
        let mut dispatch = GateDispatch;
        let response = match router.route_oneshot(&env, &entry, &request, "gate-gen", &mut dispatch)
        {
            Ok(response) => response,
            Err(failure) => {
                return GateStatus::Failed {
                    reason: format!("oneshot routing failed: {failure:?}"),
                }
            }
        };
        if response.lane != crate::owned_decode_routing::lane::LaneKind::OwnedDecode {
            return GateStatus::Failed {
                reason: "certified cutover-enabled request must select the owned lane".to_string(),
            };
        }
        let provenance = &response.provenance;
        if provenance.engine != CATALOG_ENGINE
            || provenance.lane != CATALOG_LANE
            || provenance.worker != CATALOG_WORKER
            || provenance.risk_class != CATALOG_RISK_CLASS
            || provenance.decode_fingerprint != fp
            || provenance.worker_generation == 0
        {
            return GateStatus::Failed {
                reason: "owned provenance must carry the canonical identity fields".to_string(),
            };
        }
        if response.generated_token_ids.is_empty() {
            return GateStatus::Failed {
                reason: "oneshot must return generated token IDs".to_string(),
            };
        }

        GateStatus::Passed {
            evidence: vec![
                "QueueClass::Decode serializes as \"decode\"".to_string(),
                "certified oneshot selects owned-metal-decode end to end".to_string(),
                "provenance carries canonical identity, fingerprints, and worker generation"
                    .to_string(),
            ],
        }
    }

    /// G-DEC-08: grammar. Accepted forms compile into the versioned wire
    /// constraint; rejected keywords, open objects, and untyped enums fail
    /// with the typed grammar errors; the compiled automaton loads; and the
    /// registry's constrained-positive and constrained-negative fixture
    /// groups execute data-driven from the checked-in registry JSON.
    pub fn gate_08_grammar(
        &self,
        constrained_positive: GroupOutcome,
        constrained_negative: GroupOutcome,
    ) -> GateStatus {
        let manifest = GrammarSubsetManifest::default();
        let context = CompileContext {
            base_decode_fingerprint: Fingerprint("gate-base-fp".to_string()),
            tokenizer_vocabulary_digest: "gate-vocab-digest".to_string(),
        };

        let positive = r#"{
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": { "name": { "type": "string" } },
            "required": ["name"],
            "additionalProperties": false
        }"#;
        let compiled = match compile_grammar(positive, &context, &manifest) {
            Ok(compiled) => compiled,
            Err(error) => {
                return GateStatus::Failed {
                    reason: format!("accepted schema failed to compile: {error:?}"),
                }
            }
        };
        if compiled.constraint.automaton_bytes.is_empty()
            || compiled.constraint.constraint_fingerprint.0.is_empty()
        {
            return GateStatus::Failed {
                reason: "compiled constraint must carry automaton bytes and a fingerprint"
                    .to_string(),
            };
        }
        if load_automaton(&compiled.constraint, &manifest).is_err() {
            return GateStatus::Failed {
                reason: "compiled automaton must load through the worker-side path".to_string(),
            };
        }

        let negatives = [
            (
                "combinator keyword",
                r#"{ "$schema": "https://json-schema.org/draft/2020-12/schema",
                     "type": "object", "allOf": [] }"#,
            ),
            (
                "open object",
                r#"{ "$schema": "https://json-schema.org/draft/2020-12/schema",
                     "type": "object", "additionalProperties": true }"#,
            ),
            (
                "untyped enum",
                r#"{ "$schema": "https://json-schema.org/draft/2020-12/schema",
                     "enum": [1, 2] }"#,
            ),
        ];
        for (name, schema) in negatives {
            match compile_grammar(schema, &context, &manifest) {
                Err(error) if error.kind == OwnedDecodeError::GrammarFeatureUnsupported => {}
                other => {
                    return GateStatus::Failed {
                        reason: format!(
                            "{name} must fail with grammar_feature_unsupported, got {other:?}"
                        ),
                    }
                }
            }
        }
        match compile_grammar("{ not json", &context, &manifest) {
            Err(error) if error.kind == OwnedDecodeError::GrammarParseFailed => {}
            other => {
                return GateStatus::Failed {
                    reason: format!(
                        "malformed JSON must fail with grammar_parse_failed, got {other:?}"
                    ),
                }
            }
        }

        // The registry's constrained groups execute data-driven: positive
        // entries compile and accept their valid documents; negative entries
        // (plus the grammar-audit rejection probes) reject with the typed
        // errors the registry names.
        let mut evidence = vec![
            "accepted object schema compiles to token-id-json-constraint-v1".to_string(),
            "compiled automaton loads through the worker-side path".to_string(),
            "combinator keywords, open objects, and untyped enums are rejected typed".to_string(),
            "malformed JSON returns grammar_parse_failed".to_string(),
        ];
        if let Err(reason) = constrained_positive.result {
            return GateStatus::Failed {
                reason: format!("constrained-positive fixture group: {reason}"),
            };
        }
        evidence.push(format!(
            "constrained-positive fixture group executed: {}",
            constrained_positive.executed_ids.join(", ")
        ));
        if let Err(reason) = constrained_negative.result {
            return GateStatus::Failed {
                reason: format!("constrained-negative fixture group: {reason}"),
            };
        }
        evidence.push(format!(
            "constrained-negative fixture group executed: {}",
            constrained_negative.executed_ids.join(", ")
        ));

        GateStatus::Passed { evidence }
    }

    /// G-DEC-09: grammar cost. The corpus manifest fixes the measurement
    /// protocol; the recorded evidence must meet the 0.50 ms/token p95 masking
    /// bound and the 90% constrained-throughput ratio.
    pub fn gate_09_grammar_cost(
        &self,
        evidence_record: Option<&GrammarCostEvidence>,
    ) -> GateStatus {
        let corpus = &self.manifests.grammar_cost;
        if corpus.fixtures.is_empty()
            || corpus.warmup == 0
            || corpus.repetitions == 0
            || corpus.sampling != "greedy_top1"
            || corpus.percentile_calculation.is_empty()
        {
            return GateStatus::Failed {
                reason: "grammar-cost corpus must fix fixtures, warmup, repetitions, sampling, and percentile calculation".to_string(),
            };
        }
        let Some(recorded) = evidence_record else {
            return GateStatus::Blocked {
                reason: "no grammar-cost measurement recorded for this machine".to_string(),
            };
        };
        if recorded.masking_p95_ms_per_token > GRAMMAR_MASKING_P95_BOUND_MS {
            return GateStatus::Failed {
                reason: format!(
                    "constrained masking p95 {} ms/token exceeds the {GRAMMAR_MASKING_P95_BOUND_MS} bound",
                    recorded.masking_p95_ms_per_token
                ),
            };
        }
        if recorded.constrained_throughput_ratio < GRAMMAR_THROUGHPUT_RATIO_BOUND {
            return GateStatus::Failed {
                reason: format!(
                    "constrained throughput ratio {} is below {GRAMMAR_THROUGHPUT_RATIO_BOUND}",
                    recorded.constrained_throughput_ratio
                ),
            };
        }
        GateStatus::Passed {
            evidence: vec![
                format!(
                    "corpus {} fixes the measurement protocol",
                    corpus.manifest_revision
                ),
                format!(
                    "masking p95 {} ms/token within bound; throughput ratio {} >= {}",
                    recorded.masking_p95_ms_per_token,
                    recorded.constrained_throughput_ratio,
                    GRAMMAR_THROUGHPUT_RATIO_BOUND
                ),
            ],
        }
    }

    /// G-DEC-10: throughput. Every family and weight format has a same-session
    /// spike/production comparison at chain-K 1 with startup reported
    /// separately, and steady-state production reaches at least 90% of spike.
    pub fn gate_10_throughput(&self, records: &[ThroughputEvidence]) -> GateStatus {
        if records.is_empty() {
            return GateStatus::Blocked {
                reason: "no throughput measurements recorded for this machine".to_string(),
            };
        }
        let mut covered = BTreeSet::new();
        let mut evidence = Vec::new();
        for record in records {
            if record.chain_k != 1 || record.batched_verification {
                return GateStatus::Failed {
                    reason: "throughput baseline requires chain-K 1 and no batched verification"
                        .to_string(),
                };
            }
            if !record.same_session {
                return GateStatus::Failed {
                    reason: "spike and production must run consecutively in one session"
                        .to_string(),
                };
            }
            if !record.startup_reported_separately {
                return GateStatus::Failed {
                    reason: "startup, first load, and first Q8 ingest must be reported separately"
                        .to_string(),
                };
            }
            if record.ratio() < THROUGHPUT_RATIO_BOUND {
                return GateStatus::Failed {
                    reason: format!(
                        "{}/{}: production throughput ratio {:.3} is below {THROUGHPUT_RATIO_BOUND}",
                        record.family.as_str(),
                        record.weight_quant.as_str(),
                        record.ratio()
                    ),
                };
            }
            covered.insert((record.family.as_str(), record.weight_quant.as_str()));
            evidence.push(format!(
                "{}/{}: {:.1} tok/s vs spike {:.1} tok/s (ratio {:.3})",
                record.family.as_str(),
                record.weight_quant.as_str(),
                record.production_tokens_per_sec,
                record.spike_tokens_per_sec,
                record.ratio()
            ));
        }
        let required: BTreeSet<(&str, &str)> = [
            ("qwen3-0.6b", "f16"),
            ("qwen3-0.6b", "q8_0"),
            ("lfm2-1.2b", "f16"),
            ("lfm2-1.2b", "q8_0"),
        ]
        .into_iter()
        .collect();
        let missing: Vec<_> = required.difference(&covered).collect();
        if !missing.is_empty() {
            return GateStatus::Failed {
                reason: format!("throughput evidence missing lanes: {missing:?}"),
            };
        }
        GateStatus::Passed { evidence }
    }

    /// G-DEC-11: scheduler isolation. Release-gated by OQ-DEC-SCHED-01: the
    /// mechanism tests live with the scheduler module; this gate passes only
    /// once the numeric manifest is committed and executed.
    pub fn gate_11_scheduler(status: &SchedulerEvidenceStatus) -> GateStatus {
        match status {
            SchedulerEvidenceStatus::Committed { production_n } => GateStatus::Passed {
                evidence: vec![format!(
                    "decode-sched-manifest-v1 committed production N={production_n} with executed evidence"
                )],
            },
            SchedulerEvidenceStatus::Blocked { reasons } => GateStatus::Blocked {
                reason: format!(
                    "numeric scheduler commitment outstanding (OQ-DEC-SCHED-01): {}",
                    reasons.join("; ")
                ),
            },
        }
    }

    /// G-DEC-12: regression and CI. The lane manifest names the mandatory
    /// `macos-metal` target and all twelve gates; the gate set carries zero
    /// skips; the registry's scheduler-continuity fixture group executes; and
    /// the scheduler-dependent release portion stays blocked while G-DEC-11
    /// is blocked.
    pub fn gate_12_ci(
        &self,
        statuses: &BTreeMap<GateId, GateStatus>,
        scheduler_continuity: GroupOutcome,
    ) -> GateStatus {
        let ci = &self.manifests.ci_lane;
        if !ci.mandatory_targets.iter().any(|t| t == "macos-metal") {
            return GateStatus::Failed {
                reason: "ci-lane-manifest-v1 must name macos-metal as mandatory".to_string(),
            };
        }
        let expected_gates: BTreeSet<String> = ALL_GATES
            .iter()
            .map(|gate| gate.as_str().to_string())
            .collect();
        let manifest_gates: BTreeSet<String> = ci.mandatory_lane_gates.iter().cloned().collect();
        if manifest_gates != expected_gates {
            return GateStatus::Failed {
                reason: "ci-lane-manifest-v1 mandatory gates must be exactly G-DEC-01..12"
                    .to_string(),
            };
        }
        if ci.normal_targets.is_empty() {
            return GateStatus::Failed {
                reason: "ci-lane-manifest-v1 must name normal targets".to_string(),
            };
        }

        let skips: Vec<GateId> = statuses
            .iter()
            .filter(|(_, status)| matches!(status, GateStatus::Skipped { .. }))
            .map(|(gate, _)| *gate)
            .collect();
        if !skips.is_empty() {
            return GateStatus::Failed {
                reason: format!("applicable gates were skipped: {skips:?}"),
            };
        }

        // The registry's scheduler-continuity group executes everywhere
        // (hardware-independent); a failure is a CI regression and takes
        // precedence over the blocked state below.
        if let Err(reason) = scheduler_continuity.result {
            return GateStatus::Failed {
                reason: format!("scheduler-continuity fixture group: {reason}"),
            };
        }
        let continuity_evidence = format!(
            "scheduler-continuity fixture group executed: {}",
            scheduler_continuity.executed_ids.join(", ")
        );

        match statuses.get(&GateId::GDec11) {
            Some(GateStatus::Blocked { .. }) | None => GateStatus::Blocked {
                reason: "scheduler-dependent release gate: G-DEC-11 is blocked, so D-009 and release claims remain blocked".to_string(),
            },
            Some(GateStatus::Failed { reason }) => GateStatus::Failed {
                reason: format!("G-DEC-11 failed: {reason}"),
            },
            Some(GateStatus::Skipped { .. }) => GateStatus::Failed {
                reason: "G-DEC-11 may not be skipped".to_string(),
            },
            Some(GateStatus::Passed { .. }) => GateStatus::Passed {
                evidence: vec![
                    "ci-lane-manifest-v1 names macos-metal mandatory and all twelve gates"
                        .to_string(),
                    "gate set executed with zero applicable skips".to_string(),
                    continuity_evidence,
                ],
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Gate helpers
// ---------------------------------------------------------------------------

fn decode_fingerprint_for_fixture(fixture: &ParityFixture) -> Fingerprint {
    DecodeIdentityInputs {
        family: fixture.family,
        activation_dtype: fixture.activation_dtype,
        weight_quant: fixture.weight_quant,
        artifact_source_digest: fixture.source_digest.clone(),
        arithmetic_identity_revision: fixture.arithmetic_identity_revision.clone(),
        q8: match (&fixture.q8_quantizer_revision, &fixture.q8_derived_digest) {
            (Some(revision), Some(digest)) => Some(Q8Identity {
                quantizer_revision: revision.clone(),
                derived_digest: digest.clone(),
            }),
            _ => None,
        },
    }
    .decode_fingerprint()
    .expect("fixture identity inputs are valid")
}

fn gate_catalog_entry() -> CatalogEntry {
    CatalogEntry {
        entry_id: "qwen3-0.6b-f16-2048".to_string(),
        engine: CATALOG_ENGINE.to_string(),
        task: CATALOG_TASK.to_string(),
        lane: CATALOG_LANE.to_string(),
        worker: CATALOG_WORKER.to_string(),
        risk_class: CATALOG_RISK_CLASS.to_string(),
        family: Family::Qwen3_0_6b,
        activation_dtype: ActivationDType::F16,
        weight_quant: WeightQuant::F16,
        arithmetic_identity_revision: "qwen3-arithmetic-v1".to_string(),
        metallib_revision: "qwen3-metallib-v1".to_string(),
        max_context_tokens: 2048,
        artifact_source_digest: "qwen3-f16-source-v1".to_string(),
        q8: None,
        owned_family: None,
        owned_dtype: None,
        quant: None,
    }
}

fn decode_fingerprint_for_entry(entry: &CatalogEntry) -> Fingerprint {
    DecodeIdentityInputs {
        family: entry.family,
        activation_dtype: entry.activation_dtype,
        weight_quant: entry.weight_quant,
        artifact_source_digest: entry.artifact_source_digest.clone(),
        arithmetic_identity_revision: entry.arithmetic_identity_revision.clone(),
        q8: None,
    }
    .decode_fingerprint()
    .expect("entry identity inputs are valid")
}

fn runtime_manifest(
    production_n: u32,
    decode_weight: u32,
    aging_window_ms: u64,
) -> RuntimeConfigManifest {
    RuntimeConfigManifest {
        worker_revision: "owned-metal-decode-worker-v1".to_string(),
        protocol_revision: "owned-metal-decode-worker-v1".to_string(),
        metallib_revision: "metallib-v1".to_string(),
        chain_k: 1,
        batched_verification: false,
        resident_limit: 1,
        attention_kv_reservation_units: 2048,
        lfm2_conv_cache_reservation_bytes: 0,
        context_manifest_revision: "decode-context-buckets-v1".to_string(),
        crash_policy_revision: "crash-policy-v1".to_string(),
        quarantine_duration_ms: 60_000,
        scheduler: crate::owned_decode_contracts::SchedulerRuntimeRecord {
            production_n,
            yield_policy_revision: "yield-on-contention-v1".to_string(),
            decode_weight,
            decode_aging_window_ms: aging_window_ms,
            progress_protocol_revision: "generate-progress-v1".to_string(),
        },
    }
}

struct GateDispatch;

impl DecodeDispatch for GateDispatch {
    fn dispatch(
        &mut self,
        command: &DispatchedCommand,
    ) -> Result<ExecutionSuccess, OwnedDecodeError> {
        let _ = command;
        Ok(ExecutionSuccess {
            generated_token_ids: vec![42, 43, 44],
            finish_reason: crate::owned_decode_routing::provenance::FinishReason::StopToken,
            lane_finish_reason: None,
            worker_generation: 1,
            last_completed_quantum_sequence: 1,
            crash_retry_count: 0,
            failure_classifications: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::owned_decode_certification::probe::OracleReproducingProbe;

    fn manifest_dir() -> ManifestDir {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("owned-decode-manifests");
        crate::owned_decode_contracts::load_manifest_dir(&path).expect("manifests load")
    }

    fn four_lane_throughput() -> Vec<ThroughputEvidence> {
        let lanes = [
            (Family::Qwen3_0_6b, WeightQuant::F16),
            (Family::Qwen3_0_6b, WeightQuant::Q8_0),
            (Family::Lfm2_1_2b, WeightQuant::F16),
            (Family::Lfm2_1_2b, WeightQuant::Q8_0),
        ];
        lanes
            .into_iter()
            .map(|(family, weight_quant)| ThroughputEvidence {
                family,
                weight_quant,
                spike_tokens_per_sec: 100.0,
                production_tokens_per_sec: 95.0,
                chain_k: 1,
                batched_verification: false,
                same_session: true,
                startup_reported_separately: true,
            })
            .collect()
    }

    fn passing_grammar_cost() -> GrammarCostEvidence {
        GrammarCostEvidence {
            masking_p95_ms_per_token: 0.25,
            constrained_throughput_ratio: 0.94,
        }
    }

    #[test]
    fn full_run_executes_every_gate_with_zero_skips() {
        let runner = GateRunner::new(manifest_dir());
        let battery = parity_battery();
        let mut oracle = OracleStore::new();
        oracle.register_synthetic_battery(&battery);
        let mut probe = OracleReproducingProbe::new(&oracle);
        let throughput = four_lane_throughput();
        let grammar_cost = passing_grammar_cost();

        let evidence = runner.run_all(&oracle, &mut probe, &throughput, Some(&grammar_cost));

        // Every gate executed: all twelve are present and none is skipped.
        assert_eq!(evidence.gate_statuses.len(), 12);
        assert!(
            applicable_skips(&evidence).is_empty(),
            "zero applicable skips"
        );

        // G-DEC-01..10 pass with the synthetic evidence; G-DEC-11 and
        // G-DEC-12 pass from the checked-in scheduler manifest: the
        // OQ-DEC-SCHED-01 protocol v2 measurement committed N=16 (largest
        // candidate meeting the embed.query p95 SLO), so the release gate
        // set is fully green with zero skips.
        for gate in &ALL_GATES {
            let status = &evidence.gate_statuses[gate];
            assert!(
                matches!(status, GateStatus::Passed { .. }),
                "{gate:?} must pass, got {status:?}"
            );
        }

        // Release is ready and the scheduler evidence is committed.
        assert!(release_ready(&evidence));
        assert!(scheduler_evidence_committed(&evidence.scheduler_status));

        // Evidence records the registry revision, executed parity fixtures for
        // both families and both formats, and certification evidence IDs.
        assert_eq!(
            evidence.fixture_registry_revision,
            "decode-fixture-registry-v1"
        );
        let parity_ids = &evidence.executed_fixtures[PARITY_GROUP];
        assert_eq!(parity_ids.len(), 4);
        assert_eq!(evidence.certification_evidence_ids.len(), 2);

        // Every non-parity registry group executed and recorded its entry IDs:
        // request-processing x2, constrained-positive x3, constrained-negative
        // x5, scheduler-continuity x7 (the checked-in registry counts).
        let request_processing = &evidence.executed_fixtures[REQUEST_PROCESSING_GROUP];
        assert_eq!(request_processing.len(), 2, "{request_processing:?}");
        let constrained_positive = &evidence.executed_fixtures[CONSTRAINED_POSITIVE_GROUP];
        assert_eq!(constrained_positive.len(), 3, "{constrained_positive:?}");
        let constrained_negative = &evidence.executed_fixtures[CONSTRAINED_NEGATIVE_GROUP];
        assert_eq!(constrained_negative.len(), 5, "{constrained_negative:?}");
        let scheduler_continuity = &evidence.executed_fixtures[SCHEDULER_CONTINUITY_GROUP];
        assert_eq!(scheduler_continuity.len(), 7, "{scheduler_continuity:?}");
        // Every non-parity registry entry ID appears in the executed evidence.
        let manifests = manifest_dir();
        for entry in &manifests.fixture_registry.entries {
            if entry.group == PARITY_GROUP {
                continue;
            }
            let group_ids = &evidence.executed_fixtures[&entry.group];
            assert!(
                group_ids.iter().any(|id| id == &entry.id),
                "registry entry {} missing from executed evidence",
                entry.id
            );
        }
    }

    #[test]
    fn q8_parity_fork_fails_the_parity_gate() {
        // A probe that forks the Q8 lanes violates the zero-fork band.
        struct Q8ForkProbe<'a> {
            oracle: &'a OracleStore,
        }
        impl DecodeProbe for Q8ForkProbe<'_> {
            fn generate(&mut self, fixture: &ParityFixture, prompt_index: u32) -> Vec<u32> {
                let mut tokens = self
                    .oracle
                    .stream(&fixture.id, prompt_index)
                    .expect("oracle registered")
                    .to_vec();
                if fixture.weight_quant == WeightQuant::Q8_0 && prompt_index == 0 {
                    tokens[0] = tokens[0].wrapping_add(1);
                }
                tokens
            }
        }

        let runner = GateRunner::new(manifest_dir());
        let battery = parity_battery();
        let mut oracle = OracleStore::new();
        oracle.register_synthetic_battery(&battery);
        let mut probe = Q8ForkProbe { oracle: &oracle };

        let (status, _) = runner.gate_04_parity(&oracle, &mut probe);
        match status {
            GateStatus::Failed { reason } => {
                assert!(
                    reason.contains("q8_0"),
                    "reason should name the Q8 lane: {reason}"
                );
                assert!(reason.contains("exceed the structural band"));
            }
            other => panic!("Q8 fork must fail the parity gate, got {other:?}"),
        }
    }

    #[test]
    fn throughput_gate_requires_all_four_lanes_and_the_ratio_bound() {
        let runner = GateRunner::new(manifest_dir());

        assert!(matches!(
            runner.gate_10_throughput(&[]),
            GateStatus::Blocked { .. }
        ));

        let mut records = four_lane_throughput();
        records.pop();
        match runner.gate_10_throughput(&records) {
            GateStatus::Failed { reason } => assert!(reason.contains("missing lanes")),
            other => panic!("missing lane must fail, got {other:?}"),
        }

        let mut records = four_lane_throughput();
        records[0].production_tokens_per_sec = 80.0; // ratio 0.80 < 0.90
        match runner.gate_10_throughput(&records) {
            GateStatus::Failed { reason } => assert!(reason.contains("below")),
            other => panic!("low ratio must fail, got {other:?}"),
        }

        let mut records = four_lane_throughput();
        records[0].chain_k = 2;
        assert!(matches!(
            runner.gate_10_throughput(&records),
            GateStatus::Failed { .. }
        ));
    }

    #[test]
    fn grammar_cost_gate_enforces_bounds_and_blocks_without_evidence() {
        let runner = GateRunner::new(manifest_dir());
        assert!(matches!(
            runner.gate_09_grammar_cost(None),
            GateStatus::Blocked { .. }
        ));
        assert!(matches!(
            runner.gate_09_grammar_cost(Some(&passing_grammar_cost())),
            GateStatus::Passed { .. }
        ));
        let too_slow = GrammarCostEvidence {
            masking_p95_ms_per_token: 0.75,
            constrained_throughput_ratio: 0.95,
        };
        assert!(matches!(
            runner.gate_09_grammar_cost(Some(&too_slow)),
            GateStatus::Failed { .. }
        ));
        let low_throughput = GrammarCostEvidence {
            masking_p95_ms_per_token: 0.25,
            constrained_throughput_ratio: 0.80,
        };
        assert!(matches!(
            runner.gate_09_grammar_cost(Some(&low_throughput)),
            GateStatus::Failed { .. }
        ));
    }

    #[test]
    fn scheduler_gate_tracks_the_evidence_status() {
        let blocked = SchedulerEvidenceStatus::Blocked {
            reasons: vec!["committed_n is not committed".to_string()],
        };
        assert!(matches!(
            GateRunner::gate_11_scheduler(&blocked),
            GateStatus::Blocked { .. }
        ));
        let committed = SchedulerEvidenceStatus::Committed { production_n: 16 };
        assert!(matches!(
            GateRunner::gate_11_scheduler(&committed),
            GateStatus::Passed { .. }
        ));
    }

    #[test]
    fn release_ready_requires_every_gate_passed() {
        let runner = GateRunner::new(manifest_dir());
        let battery = parity_battery();
        let mut oracle = OracleStore::new();
        oracle.register_synthetic_battery(&battery);
        let mut probe = OracleReproducingProbe::new(&oracle);
        let throughput = four_lane_throughput();
        let grammar_cost = passing_grammar_cost();
        let mut evidence = runner.run_all(&oracle, &mut probe, &throughput, Some(&grammar_cost));
        // The checked-in scheduler manifest is committed (protocol v2), so
        // the full synthetic evidence set is shippable as-is.
        assert!(release_ready(&evidence));

        // Demoting any single gate defeats release: the predicate requires
        // EVERY gate to pass, never a subset.
        evidence.gate_statuses.insert(
            GateId::GDec11,
            GateStatus::Blocked {
                reason: "numeric scheduler commitment outstanding".to_string(),
            },
        );
        assert!(!release_ready(&evidence));
        evidence.gate_statuses.insert(
            GateId::GDec11,
            GateStatus::Passed {
                evidence: vec!["committed".to_string()],
            },
        );
        assert!(release_ready(&evidence));
    }
}
