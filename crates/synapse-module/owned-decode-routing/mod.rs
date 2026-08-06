//! Module-side request processing and lane routing for the production
//! owned-metal-decode lane.
//!
//! This module is the routing layer that sits above the supervised worker: it
//! owns catalog validation, family processing-asset registration, request-domain
//! validation, Q8 ingest orchestration, decode/processing/runtime identity
//! computation, certification-row access, lane selection and fallback, the D-009
//! cutover predicate, provenance construction, and the end-to-end
//! `microllm.oneshot` orchestration.
//!
//! The worker execution itself lives behind the [`DecodeDispatch`] seam so the
//! routing decisions can be exercised without Metal hardware. The seam is the
//! boundary at which "fallback eligibility ends": pre-dispatch refusals are
//! decided here and may select llama, but once [`DecodeDispatch::dispatch`] is
//! invoked, every execution-phase outcome returns directly and never re-enters
//! lane selection.
//!
//! Submodules:
//! - [`error`]: stable owned-decode and grammar wire IDs and their classifications.
//! - [`family`]: per-family tokenizer/template/special/stop/detokenizer registration.
//! - [`identity`]: decode, processing, runtime-config, and constraint identities.
//! - [`request`]: request model and boundary validation.
//! - [`q8ingest`]: Q8 first-load derivation orchestration.
//! - [`certification`]: certification-row access and structural-band checks.
//! - [`lane`]: lane selection, fallback, and the cutover predicate.
//! - [`provenance`]: selected-lane response provenance.

pub mod certification;
pub mod error;
pub mod family;
pub mod identity;
pub mod lane;
pub mod provenance;
pub mod q8artifact;
pub mod q8ingest;
pub mod request;

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use synapse_core::Fingerprint;

use crate::owned_decode_contracts::ContextBucketsManifest;
use crate::owned_decode_routing::certification::{
    CertificationAccess, ConstrainedCertKey, UnconstrainedCertKey,
};
use crate::owned_decode_routing::error::OwnedDecodeError;
use crate::owned_decode_routing::family::{Family, FamilyRegistry};
use crate::owned_decode_routing::identity::{
    ActivationDType, DecodeIdentityInputs, ProcessingIdentityInputs, Q8Identity, WeightQuant,
};
use crate::owned_decode_routing::lane::{
    select_lane, FallbackReason, LaneKind, LaneOutcome, LaneSelectionContext, LlamaLane,
    OwnedEvaluation, RoutingRefusal,
};
use crate::owned_decode_routing::provenance::{
    FinishReason, LaneProvenance, OwnedProvenanceInputs,
};
use crate::owned_decode_routing::q8ingest::{Q8IngestRegistry, TrustState};
use crate::owned_decode_routing::request::{OneshotRequest, RequestValidationError};

// ---------------------------------------------------------------------------
// Production catalog integration
// ---------------------------------------------------------------------------

/// Canonical identity values every production catalog entry must carry.
pub const CATALOG_ENGINE: &str = "owned-metal-decode";
pub const CATALOG_TASK: &str = "generate";
pub const CATALOG_LANE: &str = "decode";
pub const CATALOG_WORKER: &str = "supervised";
pub const CATALOG_RISK_CLASS: &str = "abort_capable";

/// A production owned-decode catalog entry. Canonical dedicated fields are
/// authoritative; the mirrored `owned_*`/`quant` aliases are readable migration
/// aliases that must agree with the canonical values.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogEntry {
    pub entry_id: String,
    pub engine: String,
    pub task: String,
    pub lane: String,
    pub worker: String,
    pub risk_class: String,
    pub family: Family,
    pub activation_dtype: ActivationDType,
    pub weight_quant: WeightQuant,
    pub arithmetic_identity_revision: String,
    pub metallib_revision: String,
    /// Must be a verified bucket from `decode-context-buckets-v1` for the family.
    pub max_context_tokens: u32,
    pub artifact_source_digest: String,
    /// Present exactly when `weight_quant` is `q8_0`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub q8: Option<Q8Identity>,
    /// Readable migration aliases; must match the canonical values when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owned_family: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owned_dtype: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant: Option<String>,
}

/// Whether `bucket` is a verified, shippable bucket for `family` in the manifest.
fn bucket_is_verified(manifest: &ContextBucketsManifest, family: &str, bucket: u32) -> bool {
    manifest
        .families
        .iter()
        .find(|entry| entry.family == family)
        .map(|entry| entry.verified_buckets.contains(&bucket))
        .unwrap_or(false)
}

impl CatalogEntry {
    /// Validate a production catalog entry against the canonical identity and
    /// the shippable context manifest. An entry is invalid unless its canonical
    /// identity is exact, its format is supported, its context bucket is listed,
    /// and all mirrored aliases agree. Invalid entries fail closed with
    /// `owned_decode_unsupported`.
    pub fn validate(
        &self,
        context_buckets: &ContextBucketsManifest,
    ) -> Result<(), OwnedDecodeError> {
        if self.engine != CATALOG_ENGINE
            || self.task != CATALOG_TASK
            || self.lane != CATALOG_LANE
            || self.worker != CATALOG_WORKER
            || self.risk_class != CATALOG_RISK_CLASS
        {
            return Err(OwnedDecodeError::Unsupported);
        }

        // Family must be recognized (enum-guaranteed) and have a registration.
        // Format enums guarantee activation dtype f16 and weight quant f16|q8_0.

        // Q8 block must be present exactly for q8_0.
        match self.weight_quant {
            WeightQuant::Q8_0 if self.q8.is_none() => return Err(OwnedDecodeError::Unsupported),
            WeightQuant::F16 if self.q8.is_some() => return Err(OwnedDecodeError::Unsupported),
            _ => {}
        }

        // Context bucket must be a verified, shippable bucket for the family.
        if !bucket_is_verified(
            context_buckets,
            self.family.as_str(),
            self.max_context_tokens,
        ) {
            return Err(OwnedDecodeError::Unsupported);
        }

        // Mirrored aliases must agree with canonical values when present.
        if let Some(alias) = &self.owned_family {
            if alias != self.family.as_str() {
                return Err(OwnedDecodeError::Unsupported);
            }
        }
        if let Some(alias) = &self.owned_dtype {
            if alias != self.activation_dtype.as_str() {
                return Err(OwnedDecodeError::Unsupported);
            }
        }
        if let Some(alias) = &self.quant {
            if alias != self.weight_quant.as_str() {
                return Err(OwnedDecodeError::Unsupported);
            }
        }

        Ok(())
    }

    /// Build the decode-identity inputs from this catalog entry.
    pub fn decode_identity_inputs(&self) -> DecodeIdentityInputs {
        DecodeIdentityInputs {
            family: self.family,
            activation_dtype: self.activation_dtype,
            weight_quant: self.weight_quant,
            artifact_source_digest: self.artifact_source_digest.clone(),
            arithmetic_identity_revision: self.arithmetic_identity_revision.clone(),
            q8: self.q8.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// End-to-end oneshot orchestration
// ---------------------------------------------------------------------------

/// A command handed to the dispatch seam. Once this is built and dispatched,
/// fallback eligibility has ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DispatchedCommand {
    pub lane: LaneKind,
    pub decode_fingerprint: Fingerprint,
    pub processing_fingerprint: Fingerprint,
    pub prompt_token_count: u32,
    pub max_tokens: u32,
    pub generation_id: String,
    pub constrained: bool,
}

/// A successful execution outcome returned by the dispatch seam. Contains
/// generated token IDs and accounting but no authoritative text (the module
/// detokenizes).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionSuccess {
    pub generated_token_ids: Vec<u32>,
    pub finish_reason: FinishReason,
    pub lane_finish_reason: Option<String>,
    pub worker_generation: u64,
    pub last_completed_quantum_sequence: u32,
    pub crash_retry_count: u32,
    pub failure_classifications: Vec<String>,
}

/// The worker-execution seam. Routing decides the lane; the seam executes it.
/// Execution-phase failures returned here never re-enter lane selection.
pub trait DecodeDispatch {
    fn dispatch(
        &mut self,
        command: &DispatchedCommand,
    ) -> Result<ExecutionSuccess, OwnedDecodeError>;
}

/// Per-request routing environment: the machine-profile-local state that is not
/// derivable from the catalog entry alone.
///
/// The `cutover_enabled` flag is deliberately not settable by struct literal:
/// the only production construction path is [`RoutingEnvironment::with_cutover_evaluated`],
/// which derives the flag by evaluating the D-009 predicate
/// ([`crate::owned_decode_routing::lane::cutover_enabled`]) over the checked-in
/// record and evidence-derived inputs. Tests may use the clearly named
/// [`RoutingEnvironment::with_cutover_flag_for_test`].
#[derive(Clone, Debug)]
pub struct RoutingEnvironment {
    pub machine_profile_hash: String,
    /// Whether grammar (constrained decoding) is enabled at all. When false, a
    /// constrained request returns `grammar_disabled` before lane selection.
    pub grammar_enabled: bool,
    /// Whether the D-009 cutover is enabled for this profile (the preferred-lane
    /// predicate; see `lane::cutover_enabled`). Private: only the constructors
    /// below can set it.
    cutover_enabled: bool,
    /// Whether the quarantine key is currently quarantined.
    pub quarantined: bool,
    /// The configured llama fallback lane, if any.
    pub llama: Option<LlamaLane>,
    /// Fingerprints authorized as equivalent aliases of a `required_fingerprint`.
    pub equivalent_fingerprints: BTreeSet<Fingerprint>,
    /// Constraint runtime identity digest for a constrained request, if applicable.
    pub constraint_runtime_identity: Option<String>,
    /// A typed refusal discovered while resolving an owned catalog identity.
    /// This takes precedence over cutover because it describes whether the
    /// selected platform can execute the owned lane at all.
    resolution_owned_refusal: Option<OwnedDecodeError>,
    /// A typed refusal discovered while preparing the owned worker before its
    /// dispatch seam runs. It is considered only after the normal cutover,
    /// quarantine, artifact, and certification gates have selected the owned lane.
    pre_dispatch_owned_refusal: Option<OwnedDecodeError>,
}

impl RoutingEnvironment {
    /// Construct a routing environment whose cutover flag is mechanically
    /// derived from the checked-in D-009 record and evidence-derived inputs:
    /// the flag is exactly [`crate::owned_decode_routing::lane::cutover_enabled`]
    /// evaluated over them. This is the only production construction path, so
    /// a true flag cannot exist without a record-and-evidence evaluation.
    // Every environment field plus the record-and-evidence pair must be
    // explicit at the single construction site; splitting them would only
    // hide the coupling this constructor exists to enforce.
    #[allow(clippy::too_many_arguments)]
    pub fn with_cutover_evaluated(
        machine_profile_hash: impl Into<String>,
        grammar_enabled: bool,
        quarantined: bool,
        llama: Option<LlamaLane>,
        equivalent_fingerprints: BTreeSet<Fingerprint>,
        constraint_runtime_identity: Option<String>,
        record: &crate::owned_decode_routing::lane::CutoverRecord,
        inputs: &crate::owned_decode_routing::lane::CutoverInputs,
    ) -> Self {
        Self {
            machine_profile_hash: machine_profile_hash.into(),
            grammar_enabled,
            cutover_enabled: crate::owned_decode_routing::lane::cutover_enabled(record, inputs),
            quarantined,
            llama,
            equivalent_fingerprints,
            constraint_runtime_identity,
            resolution_owned_refusal: None,
            pre_dispatch_owned_refusal: None,
        }
    }

    /// Construct the fail-closed production environment when no version-controlled
    /// record verifies that this machine's owned-decode configuration passed
    /// certification and is approved for production cutover. In that case,
    /// owned decode is not preferred; fallback and constrained-request errors
    /// still use this lane-selection authority.
    pub fn without_cutover_record(
        machine_profile_hash: impl Into<String>,
        grammar_enabled: bool,
        quarantined: bool,
        llama: Option<LlamaLane>,
        equivalent_fingerprints: BTreeSet<Fingerprint>,
        constraint_runtime_identity: Option<String>,
    ) -> Self {
        Self {
            machine_profile_hash: machine_profile_hash.into(),
            grammar_enabled,
            cutover_enabled: false,
            quarantined,
            llama,
            equivalent_fingerprints,
            constraint_runtime_identity,
            resolution_owned_refusal: None,
            pre_dispatch_owned_refusal: None,
        }
    }

    /// Test-only constructor: sets the cutover flag directly without a
    /// record-and-evidence evaluation. Named so it is never mistaken for a
    /// production path; production code must use
    /// [`RoutingEnvironment::with_cutover_evaluated`].
    pub fn with_cutover_flag_for_test(
        machine_profile_hash: impl Into<String>,
        grammar_enabled: bool,
        cutover_enabled: bool,
        quarantined: bool,
        llama: Option<LlamaLane>,
        equivalent_fingerprints: BTreeSet<Fingerprint>,
        constraint_runtime_identity: Option<String>,
    ) -> Self {
        Self {
            machine_profile_hash: machine_profile_hash.into(),
            grammar_enabled,
            cutover_enabled,
            quarantined,
            llama,
            equivalent_fingerprints,
            constraint_runtime_identity,
            resolution_owned_refusal: None,
            pre_dispatch_owned_refusal: None,
        }
    }

    /// Record a typed refusal found while resolving an owned catalog identity.
    /// Lane selection receives the refusal before the cutover gate, so a
    /// substitutable request may choose llama without requiring Metal execution.
    #[must_use]
    pub fn with_resolution_owned_refusal(mut self, refusal: OwnedDecodeError) -> Self {
        debug_assert!(refusal.is_predispatch_fallback_eligible());
        self.resolution_owned_refusal = Some(refusal);
        self
    }

    /// Record a pre-dispatch owned-lane refusal discovered by the module's worker
    /// setup. Lane selection consumes this instead of allowing setup to become an
    /// execution-phase failure, so eligible refusals can select llama.
    #[must_use]
    pub fn with_pre_dispatch_owned_refusal(mut self, refusal: OwnedDecodeError) -> Self {
        debug_assert!(refusal.is_predispatch_fallback_eligible());
        self.pre_dispatch_owned_refusal = Some(refusal);
        self
    }

    /// Whether the D-009 cutover is enabled for this profile.
    #[must_use]
    pub fn cutover_enabled(&self) -> bool {
        self.cutover_enabled
    }
}

/// A terminal routing failure with its wire ID and any mapped-refusal underlying.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoutingFailure {
    pub error: RoutingRefusal,
    pub underlying_owned_decode_refusal_id: Option<OwnedDecodeError>,
}

impl RoutingFailure {
    pub fn wire_id(&self) -> &'static str {
        self.error.wire_id()
    }

    fn owned(error: OwnedDecodeError) -> Self {
        Self {
            error: RoutingRefusal::Owned(error),
            underlying_owned_decode_refusal_id: None,
        }
    }
}

/// A successfully routed response with its selected-lane provenance.
#[derive(Clone, Debug, PartialEq)]
pub struct RoutedResponse {
    pub lane: LaneKind,
    pub generated_token_ids: Vec<u32>,
    pub finish_reason: FinishReason,
    pub provenance: LaneProvenance,
}

/// The routing layer. Holds the module-owned registries used to process and
/// route a oneshot request.
pub struct OwnedDecodeRouter {
    pub families: FamilyRegistry,
    pub context_buckets: ContextBucketsManifest,
    pub q8: Q8IngestRegistry,
    pub certification: Box<dyn CertificationAccess>,
}

impl OwnedDecodeRouter {
    pub fn new(
        families: FamilyRegistry,
        context_buckets: ContextBucketsManifest,
        q8: Q8IngestRegistry,
        certification: Box<dyn CertificationAccess>,
    ) -> Self {
        Self {
            families,
            context_buckets,
            q8,
            certification,
        }
    }

    /// Evaluate the owned lane before dispatch, failing closed on the first
    /// applicable pre-dispatch refusal.
    ///
    /// Resolution refusal takes precedence because it describes a platform that
    /// cannot execute the owned lane. Normal certification and artifact gates then
    /// take precedence over worker setup: an unselected owned lane remains
    /// `NotPreferred` or `NotCertified`, while a selected lane turns a setup
    /// refusal into `OwnedEvaluation::Refused` for lane selection to handle.
    fn evaluate_owned(
        &self,
        env: &RoutingEnvironment,
        entry: &CatalogEntry,
        decode_fingerprint: &Fingerprint,
    ) -> OwnedEvaluation {
        if let Some(refusal) = env.resolution_owned_refusal {
            return OwnedEvaluation::Refused(refusal);
        }
        if !env.cutover_enabled {
            return OwnedEvaluation::NotPreferred;
        }
        if env.quarantined {
            return OwnedEvaluation::Refused(OwnedDecodeError::Quarantined);
        }
        // Q8 trust gate: a poisoned artifact refuses; an untrusted one fails
        // closed until verified.
        if entry.weight_quant.is_q8() {
            if let Some(q8) = &entry.q8 {
                if let Some(artifact) = self
                    .q8
                    .entry(&entry.artifact_source_digest, &q8.quantizer_revision)
                {
                    match artifact.trust_state {
                        TrustState::Poisoned => {
                            return OwnedEvaluation::Refused(OwnedDecodeError::ArtifactPoisoned)
                        }
                        TrustState::Untrusted => {
                            return OwnedEvaluation::Refused(OwnedDecodeError::NotCertified)
                        }
                        TrustState::Trusted => {}
                    }
                } else {
                    // No artifact ingested yet: nothing trusted to serve.
                    return OwnedEvaluation::Refused(OwnedDecodeError::NotCertified);
                }
            }
        }
        // Certification gate.
        let certified = match &env.constraint_runtime_identity {
            Some(cri) => self
                .certification
                .is_constrained_certified(&ConstrainedCertKey {
                    machine_profile_hash: env.machine_profile_hash.clone(),
                    decode_fingerprint: decode_fingerprint.clone(),
                    constraint_runtime_identity: cri.clone(),
                }),
            None => self
                .certification
                .is_unconstrained_certified(&UnconstrainedCertKey {
                    machine_profile_hash: env.machine_profile_hash.clone(),
                    decode_fingerprint: decode_fingerprint.clone(),
                }),
        };
        if !certified {
            return OwnedEvaluation::Refused(OwnedDecodeError::NotCertified);
        }
        if let Some(refusal) = env.pre_dispatch_owned_refusal {
            return OwnedEvaluation::Refused(refusal);
        }
        OwnedEvaluation::Selectable
    }

    /// Route and execute a `microllm.oneshot` request end to end.
    ///
    /// The flow is: grammar gate, family/catalog resolution, identity
    /// computation, request-boundary validation, pre-dispatch lane selection
    /// (with fallback), then exactly one dispatch to the selected lane.
    /// Execution-phase failures from the dispatch seam return directly and never
    /// re-enter lane selection.
    pub fn route_oneshot<D: DecodeDispatch>(
        &self,
        env: &RoutingEnvironment,
        entry: &CatalogEntry,
        request: &OneshotRequest,
        generation_id: &str,
        dispatch: &mut D,
    ) -> Result<RoutedResponse, RoutingFailure> {
        // 1. Grammar gate: disabled grammar returns before lane selection.
        if request.is_constrained() && !env.grammar_enabled {
            return Err(RoutingFailure::owned(OwnedDecodeError::GrammarDisabled));
        }

        // 2. Family registration (module-owned processing assets).
        let registration = self
            .families
            .get(request.family)
            .map_err(RoutingFailure::owned)?;

        // 3. Catalog validation and request/entry agreement.
        entry
            .validate(&self.context_buckets)
            .map_err(RoutingFailure::owned)?;
        if entry.family != request.family || entry.weight_quant != request.weight_quant {
            return Err(RoutingFailure::owned(OwnedDecodeError::Unsupported));
        }

        // 4. Identity computation.
        let decode_fingerprint = entry
            .decode_identity_inputs()
            .decode_fingerprint()
            .map_err(RoutingFailure::owned)?;
        let processing_fingerprint = ProcessingIdentityInputs {
            decode_fingerprint: decode_fingerprint.clone(),
            tokenizer_sanitized_digest: registration.tokenizer_sanitized_digest.clone(),
            prompt_template_revision: registration.prompt_template_revision.clone(),
            special_token_policy_revision: registration.special_token_policy_revision.clone(),
            stop_token_policy_revision: registration.stop_token_policy_revision.clone(),
            detokenizer_revision: registration.detokenizer_revision.clone(),
        }
        .processing_fingerprint();

        // 5. Request-domain validation. Invalid and context boundaries return
        //    directly and are never fallback-eligible.
        request
            .validate(entry.max_context_tokens)
            .map_err(|error| match error {
                RequestValidationError::InvalidRequest(_) => RoutingFailure {
                    error: RoutingRefusal::InvalidRequest,
                    underlying_owned_decode_refusal_id: None,
                },
                RequestValidationError::SamplingUnsupported => {
                    RoutingFailure::owned(OwnedDecodeError::SamplingUnsupported)
                }
                RequestValidationError::ContextCapacityExceeded { .. } => {
                    RoutingFailure::owned(OwnedDecodeError::ContextCapacityExceeded)
                }
            })?;

        // 6. Pre-dispatch lane selection.
        let owned = self.evaluate_owned(env, entry, &decode_fingerprint);
        let context = LaneSelectionContext {
            request,
            owned_decode_fingerprint: decode_fingerprint.clone(),
            owned_processing_fingerprint: processing_fingerprint.clone(),
            owned,
            llama: env.llama.clone(),
            equivalent_fingerprints: env.equivalent_fingerprints.clone(),
        };
        let outcome = select_lane(&context);

        // 7. Act on the decision.
        match outcome {
            LaneOutcome::Refused {
                error,
                underlying_owned_decode_refusal_id,
            } => Err(RoutingFailure {
                error,
                underlying_owned_decode_refusal_id,
            }),
            LaneOutcome::Llama { fallback_reason } => {
                let llama = env
                    .llama
                    .clone()
                    .expect("llama selected implies a configured lane");
                let command = DispatchedCommand {
                    lane: LaneKind::Llama,
                    decode_fingerprint: llama.decode_fingerprint.clone(),
                    processing_fingerprint: llama.processing_fingerprint.clone(),
                    prompt_token_count: request.prompt_token_count,
                    max_tokens: request.max_tokens,
                    generation_id: generation_id.to_string(),
                    constrained: false,
                };
                // Execution-phase failure returns directly; no re-selection.
                let success = dispatch.dispatch(&command).map_err(RoutingFailure::owned)?;
                let provenance =
                    LaneProvenance::llama(llama.decode_fingerprint, llama.processing_fingerprint)
                        .with_fallback_reason(fallback_reason_wire(fallback_reason));
                Ok(build_response(LaneKind::Llama, success, provenance))
            }
            LaneOutcome::Owned => {
                let command = DispatchedCommand {
                    lane: LaneKind::OwnedDecode,
                    decode_fingerprint: decode_fingerprint.clone(),
                    processing_fingerprint: processing_fingerprint.clone(),
                    prompt_token_count: request.prompt_token_count,
                    max_tokens: request.max_tokens,
                    generation_id: generation_id.to_string(),
                    constrained: request.is_constrained(),
                };
                // Execution-phase failure returns directly; never falls back.
                let success = dispatch.dispatch(&command).map_err(RoutingFailure::owned)?;
                let mut provenance = LaneProvenance::owned(
                    registration,
                    OwnedProvenanceInputs {
                        decode_fingerprint,
                        processing_fingerprint,
                        arithmetic_identity_revision: entry.arithmetic_identity_revision.clone(),
                        metallib_revision: entry.metallib_revision.clone(),
                        worker_generation: success.worker_generation,
                        last_completed_quantum_sequence: success.last_completed_quantum_sequence,
                    },
                );
                if success.crash_retry_count > 0 {
                    provenance = provenance.with_crash_retry(
                        success.crash_retry_count,
                        success.failure_classifications.clone(),
                    );
                }
                Ok(build_response(LaneKind::OwnedDecode, success, provenance))
            }
        }
    }
}

fn fallback_reason_wire(reason: FallbackReason) -> &'static str {
    match reason {
        FallbackReason::OwnedRefusal(error) => error.as_str(),
        FallbackReason::CutoverDisabled => "cutover_disabled",
    }
}

fn build_response(
    lane: LaneKind,
    success: ExecutionSuccess,
    mut provenance: LaneProvenance,
) -> RoutedResponse {
    if let Some(lane_reason) = success.lane_finish_reason {
        provenance.lane_finish_reason = Some(lane_reason);
    }
    RoutedResponse {
        lane,
        generated_token_ids: success.generated_token_ids,
        finish_reason: success.finish_reason,
        provenance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::owned_decode_routing::certification::CertificationStore;
    use crate::owned_decode_routing::request::SamplingMode;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn fp(s: &str) -> Fingerprint {
        Fingerprint(s.to_string())
    }

    fn context_buckets() -> ContextBucketsManifest {
        ContextBucketsManifest {
            manifest_revision: "decode-context-buckets-v1".to_string(),
            schema_revision: "owned-decode-contracts-v1".to_string(),
            families: vec![
                crate::owned_decode_contracts::ContextBucketFamily {
                    family: "qwen3-0.6b".to_string(),
                    verified_buckets: vec![512, 1024, 2048],
                },
                crate::owned_decode_contracts::ContextBucketFamily {
                    family: "lfm2-1.2b".to_string(),
                    verified_buckets: vec![512, 1024, 2048],
                },
            ],
            removed_buckets: vec![],
        }
    }

    fn catalog_entry(family: Family, weight_quant: WeightQuant) -> CatalogEntry {
        let q8 = if weight_quant.is_q8() {
            Some(Q8Identity {
                quantizer_revision: "quant-v1".to_string(),
                derived_digest: "sha256:q8-derived".to_string(),
            })
        } else {
            None
        };
        CatalogEntry {
            entry_id: format!("{}-{}", family.as_str(), weight_quant.as_str()),
            engine: CATALOG_ENGINE.to_string(),
            task: CATALOG_TASK.to_string(),
            lane: CATALOG_LANE.to_string(),
            worker: CATALOG_WORKER.to_string(),
            risk_class: CATALOG_RISK_CLASS.to_string(),
            family,
            activation_dtype: ActivationDType::F16,
            weight_quant,
            arithmetic_identity_revision: "arith-v1".to_string(),
            metallib_revision: "metallib-v1".to_string(),
            max_context_tokens: 2048,
            artifact_source_digest: "sha256:source".to_string(),
            q8,
            owned_family: Some(family.as_str().to_string()),
            owned_dtype: Some("f16".to_string()),
            quant: Some(weight_quant.as_str().to_string()),
        }
    }

    fn request(family: Family, weight_quant: WeightQuant) -> OneshotRequest {
        OneshotRequest {
            family,
            weight_quant,
            prompt_token_count: 100,
            max_tokens: 64,
            sampling: SamplingMode::GreedyTop1,
            grammar: None,
            required_fingerprint: None,
            allow_equivalent: false,
            target_fingerprint: None,
            required_processing_fingerprint: None,
            owned_only: false,
        }
    }

    fn env(cutover_enabled: bool, llama: Option<LlamaLane>) -> RoutingEnvironment {
        RoutingEnvironment::with_cutover_flag_for_test(
            "profile-a",
            true,
            cutover_enabled,
            false,
            llama,
            BTreeSet::new(),
            None,
        )
    }

    fn llama_lane() -> LlamaLane {
        LlamaLane {
            decode_fingerprint: fp("llama-decode"),
            processing_fingerprint: fp("llama-proc"),
        }
    }

    /// A dispatch seam that records every lane it was asked to dispatch and
    /// returns a configurable outcome per lane.
    struct RecordingDispatch {
        dispatched: Rc<RefCell<Vec<LaneKind>>>,
        owned_outcome: Result<ExecutionSuccess, OwnedDecodeError>,
        llama_outcome: Result<ExecutionSuccess, OwnedDecodeError>,
    }

    impl RecordingDispatch {
        fn success(lane: LaneKind) -> ExecutionSuccess {
            ExecutionSuccess {
                generated_token_ids: vec![10, 20, 30],
                finish_reason: FinishReason::StopToken,
                lane_finish_reason: None,
                worker_generation: match lane {
                    LaneKind::OwnedDecode => 1,
                    LaneKind::Llama => 0,
                },
                last_completed_quantum_sequence: 2,
                crash_retry_count: 0,
                failure_classifications: vec![],
            }
        }

        fn new_succeeding(dispatched: Rc<RefCell<Vec<LaneKind>>>) -> Self {
            Self {
                dispatched,
                owned_outcome: Ok(Self::success(LaneKind::OwnedDecode)),
                llama_outcome: Ok(Self::success(LaneKind::Llama)),
            }
        }
    }

    impl DecodeDispatch for RecordingDispatch {
        fn dispatch(
            &mut self,
            command: &DispatchedCommand,
        ) -> Result<ExecutionSuccess, OwnedDecodeError> {
            self.dispatched.borrow_mut().push(command.lane);
            match command.lane {
                LaneKind::OwnedDecode => self.owned_outcome.clone(),
                LaneKind::Llama => self.llama_outcome.clone(),
            }
        }
    }

    fn router_with_certified(
        certification: CertificationStore,
        q8: Q8IngestRegistry,
    ) -> OwnedDecodeRouter {
        OwnedDecodeRouter::new(
            FamilyRegistry::production(),
            context_buckets(),
            q8,
            Box::new(certification),
        )
    }

    fn certified_store(entry: &CatalogEntry) -> CertificationStore {
        let mut store = CertificationStore::new();
        let decode_fp = entry.decode_identity_inputs().decode_fingerprint().unwrap();
        store.certify_unconstrained("profile-a", decode_fp);
        store
    }

    #[test]
    fn catalog_validation_rejects_bad_identity_and_unlisted_bucket() {
        let buckets = context_buckets();

        let mut bad_engine = catalog_entry(Family::Qwen3_0_6b, WeightQuant::F16);
        bad_engine.engine = "llama.cpp".to_string();
        assert_eq!(
            bad_engine.validate(&buckets),
            Err(OwnedDecodeError::Unsupported)
        );

        let mut bad_bucket = catalog_entry(Family::Qwen3_0_6b, WeightQuant::F16);
        bad_bucket.max_context_tokens = 4096; // not in {512,1024,2048}
        assert_eq!(
            bad_bucket.validate(&buckets),
            Err(OwnedDecodeError::Unsupported)
        );

        let mut bad_alias = catalog_entry(Family::Qwen3_0_6b, WeightQuant::F16);
        bad_alias.owned_family = Some("lfm2-1.2b".to_string());
        assert_eq!(
            bad_alias.validate(&buckets),
            Err(OwnedDecodeError::Unsupported)
        );

        let mut q8_missing_block = catalog_entry(Family::Qwen3_0_6b, WeightQuant::Q8_0);
        q8_missing_block.q8 = None;
        assert_eq!(
            q8_missing_block.validate(&buckets),
            Err(OwnedDecodeError::Unsupported)
        );

        assert_eq!(
            catalog_entry(Family::Lfm2_1_2b, WeightQuant::F16).validate(&buckets),
            Ok(())
        );
        assert_eq!(
            catalog_entry(Family::Lfm2_1_2b, WeightQuant::Q8_0).validate(&buckets),
            Ok(())
        );
    }

    #[test]
    fn end_to_end_owned_oneshot_both_families_and_formats() {
        for family in Family::all() {
            for weight_quant in [WeightQuant::F16, WeightQuant::Q8_0] {
                let entry = catalog_entry(family, weight_quant);
                let mut q8 = Q8IngestRegistry::new();
                if weight_quant.is_q8() {
                    let q = entry.q8.clone().unwrap();
                    q8.register_expected_digest(
                        &entry.artifact_source_digest,
                        &q.quantizer_revision,
                        &q.derived_digest,
                    );
                    q8.load_or_ingest(
                        &entry.artifact_source_digest,
                        &q.quantizer_revision,
                        "q8_0",
                        b"w",
                        |_| q.derived_digest.clone(),
                    )
                    .unwrap();
                }
                let store = certified_store(&entry);
                let router = router_with_certified(store, q8);
                let request = request(family, weight_quant);
                let environment = env(true, Some(llama_lane()));
                let dispatched = Rc::new(RefCell::new(Vec::new()));
                let mut dispatch = RecordingDispatch::new_succeeding(dispatched.clone());

                let response = router
                    .route_oneshot(&environment, &entry, &request, "gen-1", &mut dispatch)
                    .expect("routed");

                assert_eq!(response.lane, LaneKind::OwnedDecode);
                assert_eq!(response.finish_reason, FinishReason::StopToken);
                assert_eq!(response.provenance.engine, "owned-metal-decode");
                assert_eq!(response.provenance.lane, "decode");
                assert_eq!(*dispatched.borrow(), vec![LaneKind::OwnedDecode]);
            }
        }
    }

    #[test]
    fn predispatch_not_certified_falls_back_to_llama() {
        let entry = catalog_entry(Family::Qwen3_0_6b, WeightQuant::F16);
        // Empty certification store => NotCertified pre-dispatch refusal.
        let router = router_with_certified(CertificationStore::new(), Q8IngestRegistry::new());
        let request = request(Family::Qwen3_0_6b, WeightQuant::F16);
        let environment = env(true, Some(llama_lane()));
        let dispatched = Rc::new(RefCell::new(Vec::new()));
        let mut dispatch = RecordingDispatch::new_succeeding(dispatched.clone());

        let response = router
            .route_oneshot(&environment, &entry, &request, "gen-1", &mut dispatch)
            .expect("falls back");
        assert_eq!(response.lane, LaneKind::Llama);
        assert_eq!(response.provenance.decode_fingerprint, fp("llama-decode"));
        assert_eq!(
            response.provenance.fallback_reason.as_deref(),
            Some("owned_decode_not_certified")
        );
        assert_eq!(*dispatched.borrow(), vec![LaneKind::Llama]);
    }

    #[test]
    fn resolution_unsupported_falls_back_before_cutover_evaluation() {
        let entry = catalog_entry(Family::Qwen3_0_6b, WeightQuant::F16);
        let router = router_with_certified(CertificationStore::new(), Q8IngestRegistry::new());
        let request = request(Family::Qwen3_0_6b, WeightQuant::F16);
        let environment = env(false, Some(llama_lane()))
            .with_resolution_owned_refusal(OwnedDecodeError::Unsupported);
        let dispatched = Rc::new(RefCell::new(Vec::new()));
        let mut dispatch = RecordingDispatch::new_succeeding(dispatched.clone());

        let response = router
            .route_oneshot(&environment, &entry, &request, "gen-1", &mut dispatch)
            .expect("resolution refusal falls back");

        assert_eq!(response.lane, LaneKind::Llama);
        assert_eq!(
            response.provenance.fallback_reason.as_deref(),
            Some("owned_decode_unsupported")
        );
        assert_eq!(*dispatched.borrow(), vec![LaneKind::Llama]);
    }

    #[test]
    fn resolution_refusal_keeps_owned_only_request_typed() {
        let entry = catalog_entry(Family::Qwen3_0_6b, WeightQuant::F16);
        let router = router_with_certified(CertificationStore::new(), Q8IngestRegistry::new());
        let mut request = request(Family::Qwen3_0_6b, WeightQuant::F16);
        request.owned_only = true;
        let environment = env(false, Some(llama_lane()))
            .with_resolution_owned_refusal(OwnedDecodeError::Unsupported);
        let dispatched = Rc::new(RefCell::new(Vec::new()));
        let mut dispatch = RecordingDispatch::new_succeeding(dispatched.clone());

        let error = router
            .route_oneshot(&environment, &entry, &request, "gen-1", &mut dispatch)
            .expect_err("owned-only request must retain the resolution refusal");

        assert_eq!(error.wire_id(), "owned_decode_unsupported");
        assert!(dispatched.borrow().is_empty(), "nothing dispatched");
    }

    #[test]
    fn worker_setup_unavailable_falls_back_before_owned_dispatch() {
        let entry = catalog_entry(Family::Qwen3_0_6b, WeightQuant::F16);
        let router = router_with_certified(certified_store(&entry), Q8IngestRegistry::new());
        let request = request(Family::Qwen3_0_6b, WeightQuant::F16);
        let environment = env(true, Some(llama_lane()))
            .with_pre_dispatch_owned_refusal(OwnedDecodeError::Unavailable);
        let dispatched = Rc::new(RefCell::new(Vec::new()));
        let mut dispatch = RecordingDispatch::new_succeeding(dispatched.clone());

        let response = router
            .route_oneshot(&environment, &entry, &request, "gen-1", &mut dispatch)
            .expect("falls back");

        assert_eq!(response.lane, LaneKind::Llama);
        assert_eq!(
            response.provenance.fallback_reason.as_deref(),
            Some("owned_decode_unavailable")
        );
        assert_eq!(*dispatched.borrow(), vec![LaneKind::Llama]);
    }

    #[test]
    fn predispatch_refusal_without_llama_returns_original() {
        let entry = catalog_entry(Family::Qwen3_0_6b, WeightQuant::F16);
        let router = router_with_certified(CertificationStore::new(), Q8IngestRegistry::new());
        let request = request(Family::Qwen3_0_6b, WeightQuant::F16);
        let environment = env(true, None); // no llama configured
        let dispatched = Rc::new(RefCell::new(Vec::new()));
        let mut dispatch = RecordingDispatch::new_succeeding(dispatched.clone());

        let err = router
            .route_oneshot(&environment, &entry, &request, "gen-1", &mut dispatch)
            .expect_err("refused");
        assert_eq!(err.wire_id(), "owned_decode_not_certified");
        assert!(dispatched.borrow().is_empty(), "nothing dispatched");
    }

    #[test]
    fn quarantined_predispatch_refusal_falls_back() {
        let entry = catalog_entry(Family::Qwen3_0_6b, WeightQuant::F16);
        let store = certified_store(&entry);
        let router = router_with_certified(store, Q8IngestRegistry::new());
        let request = request(Family::Qwen3_0_6b, WeightQuant::F16);
        let mut environment = env(true, Some(llama_lane()));
        environment.quarantined = true;
        let dispatched = Rc::new(RefCell::new(Vec::new()));
        let mut dispatch = RecordingDispatch::new_succeeding(dispatched.clone());

        let response = router
            .route_oneshot(&environment, &entry, &request, "gen-1", &mut dispatch)
            .expect("falls back");
        assert_eq!(response.lane, LaneKind::Llama);
        assert_eq!(
            response.provenance.fallback_reason.as_deref(),
            Some("owned_decode_quarantined")
        );
    }

    #[test]
    fn q8_poisoned_artifact_refuses_predispatch() {
        let entry = catalog_entry(Family::Qwen3_0_6b, WeightQuant::Q8_0);
        let mut q8 = Q8IngestRegistry::new();
        let q = entry.q8.clone().unwrap();
        // Register a mismatched expected digest so ingest poisons the artifact.
        q8.register_expected_digest(
            &entry.artifact_source_digest,
            &q.quantizer_revision,
            "sha256:other",
        );
        let _ = q8.load_or_ingest(
            &entry.artifact_source_digest,
            &q.quantizer_revision,
            "q8_0",
            b"w",
            |_| q.derived_digest.clone(),
        );

        let store = certified_store(&entry);
        let router = router_with_certified(store, q8);
        let request = request(Family::Qwen3_0_6b, WeightQuant::Q8_0);
        let environment = env(true, None);
        let dispatched = Rc::new(RefCell::new(Vec::new()));
        let mut dispatch = RecordingDispatch::new_succeeding(dispatched.clone());

        let err = router
            .route_oneshot(&environment, &entry, &request, "gen-1", &mut dispatch)
            .expect_err("poisoned");
        assert_eq!(err.wire_id(), "artifact_poisoned");
        assert!(dispatched.borrow().is_empty());
    }

    #[test]
    fn context_overflow_returns_directly_without_fallback() {
        let entry = catalog_entry(Family::Qwen3_0_6b, WeightQuant::F16);
        let store = certified_store(&entry);
        let router = router_with_certified(store, Q8IngestRegistry::new());
        let mut request = request(Family::Qwen3_0_6b, WeightQuant::F16);
        request.prompt_token_count = 2000;
        request.max_tokens = 100; // 2100 > 2048
        let environment = env(true, Some(llama_lane()));
        let dispatched = Rc::new(RefCell::new(Vec::new()));
        let mut dispatch = RecordingDispatch::new_succeeding(dispatched.clone());

        let err = router
            .route_oneshot(&environment, &entry, &request, "gen-1", &mut dispatch)
            .expect_err("overflow");
        assert_eq!(err.wire_id(), "context_capacity_exceeded");
        assert!(
            dispatched.borrow().is_empty(),
            "context refusal dispatches nothing"
        );
    }

    #[test]
    fn invalid_request_boundary_returns_directly() {
        let entry = catalog_entry(Family::Qwen3_0_6b, WeightQuant::F16);
        let store = certified_store(&entry);
        let router = router_with_certified(store, Q8IngestRegistry::new());
        let mut request = request(Family::Qwen3_0_6b, WeightQuant::F16);
        request.max_tokens = 0;
        let environment = env(true, Some(llama_lane()));
        let dispatched = Rc::new(RefCell::new(Vec::new()));
        let mut dispatch = RecordingDispatch::new_succeeding(dispatched.clone());

        let err = router
            .route_oneshot(&environment, &entry, &request, "gen-1", &mut dispatch)
            .expect_err("invalid");
        assert_eq!(err.wire_id(), "invalid_request");
        assert!(dispatched.borrow().is_empty());
        // Invalid request is a caller error, not a fallback-eligible refusal.
    }

    #[test]
    fn constrained_predispatch_refusal_maps_to_grammar_disabled_with_underlying() {
        let entry = catalog_entry(Family::Qwen3_0_6b, WeightQuant::F16);
        // Uncertified => NotCertified, which maps to grammar_disabled for constrained.
        let router = router_with_certified(CertificationStore::new(), Q8IngestRegistry::new());
        let mut request = request(Family::Qwen3_0_6b, WeightQuant::F16);
        request.grammar = Some(serde_json::json!({"type": "object"}));
        let environment = env(true, Some(llama_lane()));
        let dispatched = Rc::new(RefCell::new(Vec::new()));
        let mut dispatch = RecordingDispatch::new_succeeding(dispatched.clone());

        let err = router
            .route_oneshot(&environment, &entry, &request, "gen-1", &mut dispatch)
            .expect_err("grammar_disabled");
        assert_eq!(err.wire_id(), "grammar_disabled");
        assert_eq!(
            err.underlying_owned_decode_refusal_id,
            Some(OwnedDecodeError::NotCertified)
        );
        assert!(
            dispatched.borrow().is_empty(),
            "constrained never uses llama"
        );
    }

    #[test]
    fn grammar_disabled_before_lane_selection_when_grammar_off() {
        let entry = catalog_entry(Family::Qwen3_0_6b, WeightQuant::F16);
        let store = certified_store(&entry);
        let router = router_with_certified(store, Q8IngestRegistry::new());
        let mut request = request(Family::Qwen3_0_6b, WeightQuant::F16);
        request.grammar = Some(serde_json::json!({"type": "object"}));
        let mut environment = env(true, Some(llama_lane()));
        environment.grammar_enabled = false;
        let dispatched = Rc::new(RefCell::new(Vec::new()));
        let mut dispatch = RecordingDispatch::new_succeeding(dispatched.clone());

        let err = router
            .route_oneshot(&environment, &entry, &request, "gen-1", &mut dispatch)
            .expect_err("grammar disabled");
        assert_eq!(err.wire_id(), "grammar_disabled");
        assert!(dispatched.borrow().is_empty());
    }

    #[test]
    fn execution_phase_failure_never_falls_back() {
        let entry = catalog_entry(Family::Qwen3_0_6b, WeightQuant::F16);
        let store = certified_store(&entry);
        let router = router_with_certified(store, Q8IngestRegistry::new());
        let request = request(Family::Qwen3_0_6b, WeightQuant::F16);
        let environment = env(true, Some(llama_lane()));
        let dispatched = Rc::new(RefCell::new(Vec::new()));

        // Owned dispatch fails with an execution-phase error after dispatch.
        let mut dispatch = RecordingDispatch {
            dispatched: dispatched.clone(),
            owned_outcome: Err(OwnedDecodeError::ProtocolMismatch),
            llama_outcome: Ok(RecordingDispatch::success(LaneKind::Llama)),
        };

        let err = router
            .route_oneshot(&environment, &entry, &request, "gen-1", &mut dispatch)
            .expect_err("execution failure");
        assert_eq!(err.wire_id(), "owned_decode_protocol_mismatch");
        // Only the owned lane was dispatched; llama was never invoked even though
        // the returned ID could appear in the pre-dispatch eligibility list.
        assert_eq!(*dispatched.borrow(), vec![LaneKind::OwnedDecode]);
    }

    #[test]
    fn execution_phase_eligible_id_still_never_falls_back() {
        let entry = catalog_entry(Family::Qwen3_0_6b, WeightQuant::F16);
        let store = certified_store(&entry);
        let router = router_with_certified(store, Q8IngestRegistry::new());
        let request = request(Family::Qwen3_0_6b, WeightQuant::F16);
        let environment = env(true, Some(llama_lane()));
        let dispatched = Rc::new(RefCell::new(Vec::new()));

        // Unavailable is fallback-eligible pre-dispatch, but arising AFTER dispatch
        // it must return directly without a llama dispatch.
        let mut dispatch = RecordingDispatch {
            dispatched: dispatched.clone(),
            owned_outcome: Err(OwnedDecodeError::Unavailable),
            llama_outcome: Ok(RecordingDispatch::success(LaneKind::Llama)),
        };

        let err = router
            .route_oneshot(&environment, &entry, &request, "gen-1", &mut dispatch)
            .expect_err("execution failure");
        assert_eq!(err.wire_id(), "owned_decode_unavailable");
        assert_eq!(*dispatched.borrow(), vec![LaneKind::OwnedDecode]);
    }

    #[test]
    fn cutover_disabled_routes_substitutable_to_llama() {
        let entry = catalog_entry(Family::Qwen3_0_6b, WeightQuant::F16);
        let store = certified_store(&entry);
        let router = router_with_certified(store, Q8IngestRegistry::new());
        let request = request(Family::Qwen3_0_6b, WeightQuant::F16);
        let environment = env(false, Some(llama_lane())); // cutover disabled
        let dispatched = Rc::new(RefCell::new(Vec::new()));
        let mut dispatch = RecordingDispatch::new_succeeding(dispatched.clone());

        let response = router
            .route_oneshot(&environment, &entry, &request, "gen-1", &mut dispatch)
            .expect("llama before cutover");
        assert_eq!(response.lane, LaneKind::Llama);
        assert_eq!(
            response.provenance.fallback_reason.as_deref(),
            Some("cutover_disabled")
        );
    }

    #[test]
    fn exact_fingerprint_constraint_is_enforced_end_to_end() {
        let entry = catalog_entry(Family::Qwen3_0_6b, WeightQuant::F16);
        let store = certified_store(&entry);
        let router = router_with_certified(store, Q8IngestRegistry::new());
        let mut request = request(Family::Qwen3_0_6b, WeightQuant::F16);
        request.required_fingerprint = Some(fp("not-the-owned-fp"));
        let environment = env(true, Some(llama_lane()));
        let dispatched = Rc::new(RefCell::new(Vec::new()));
        let mut dispatch = RecordingDispatch::new_succeeding(dispatched.clone());

        let err = router
            .route_oneshot(&environment, &entry, &request, "gen-1", &mut dispatch)
            .expect_err("fingerprint mismatch");
        assert_eq!(err.wire_id(), "substitution_rejected");
        assert!(
            dispatched.borrow().is_empty(),
            "exact pin never silently falls back"
        );
    }

    #[test]
    fn catalog_entry_rejects_unknown_field() {
        // fail-closed posture: an unknown field in a caller-supplied catalog
        // entry is rejected at parse time rather than silently dropped.
        let json = serde_json::json!({
            "entry_id": "qwen3-0.6b-f16",
            "engine": CATALOG_ENGINE,
            "task": CATALOG_TASK,
            "lane": CATALOG_LANE,
            "worker": CATALOG_WORKER,
            "risk_class": CATALOG_RISK_CLASS,
            "family": "qwen3-0.6b",
            "activation_dtype": "f16",
            "weight_quant": "f16",
            "arithmetic_identity_revision": "arith-v1",
            "metallib_revision": "metallib-v1",
            "max_context_tokens": 2048,
            "artifact_source_digest": "sha256:source",
            "unknown_field": "should be rejected",
        });
        assert!(serde_json::from_value::<CatalogEntry>(json).is_err());
    }
}
