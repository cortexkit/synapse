use crate::owned_decode_contracts::ContextBucketsManifest;
use crate::owned_decode_routing::certification::CertificationStore;
use crate::owned_decode_routing::request::SamplingMode;
use crate::owned_decode_routing::*;
use crate::owned_decode_routing::{
    error::OwnedDecodeError,
    family::{Family, FamilyRegistry},
    identity::{ActivationDType, Q8Identity, WeightQuant},
    lane::{self, LaneKind, LlamaLane, ServingRefusal},
    provenance::FinishReason,
    q8ingest::Q8IngestRegistry,
    request::OneshotRequest,
};
use crate::store::{ModelAssetLocator, StoredModelConfig, SynapseStore};
use cortexkit_store_types::{Isolation, StorageBackend, StorageDescriptor};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Barrier;
use std::{collections::BTreeSet, sync::Arc};
use synapse_core::Fingerprint;
use synapse_core::{EngineIdentity, MachineProfile, NumericProfileId};

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

fn env(serving_enabled: bool, llama: Option<LlamaLane>) -> RoutingEnvironment {
    RoutingEnvironment::with_serving_evaluated(
        "profile-a",
        true,
        serving_enabled,
        serving_enabled
            .then_some(())
            .ok_or(ServingRefusal::CutoverDisabled),
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

static ROUTING_STORE_COUNTER: AtomicU64 = AtomicU64::new(0);

struct BarrierBoundaryReader {
    store: Arc<SynapseStore>,
    boundary_reached: Arc<Barrier>,
    mutation_committed: Arc<Barrier>,
}

impl lane::AdmissionBoundaryReader for BarrierBoundaryReader {
    fn admission_boundary_matches(
        &self,
        snapshot: &lane::AdmissionBoundarySnapshot,
    ) -> Result<bool, String> {
        self.boundary_reached.wait();
        self.mutation_committed.wait();
        self.store
            .owned_decode_dispatch_admission_matches(
                snapshot.profile_activation_epoch,
                &snapshot.model_id,
                &snapshot.decode_fingerprint,
                &snapshot.approval_semantic_digest,
                snapshot.approval_generation,
            )
            .map_err(|error| error.to_string())
    }
}

fn dispatch_boundary_store(
    entry: &CatalogEntry,
    decode_fingerprint: &Fingerprint,
) -> (
    std::path::PathBuf,
    Arc<SynapseStore>,
    crate::store::ApprovalRow,
) {
    let root = std::env::temp_dir().join(format!(
        "synapse-routing-boundary-{}-{}",
        std::process::id(),
        ROUTING_STORE_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let descriptor = StorageDescriptor {
        module_id: "synapse-routing-test".to_string(),
        storage_namespace: "default".to_string(),
        isolation: Isolation::Module,
        backend: StorageBackend::Sqlite {
            path: root.join("store.db").to_string_lossy().to_string(),
        },
    };
    let store = Arc::new(SynapseStore::open(&descriptor).unwrap());
    store
        .upsert_model(
            &StoredModelConfig {
                model_id: entry.entry_id.clone(),
                engine: CATALOG_ENGINE.to_string(),
                task: CATALOG_TASK.to_string(),
                artifact_digest: "sha256:source".to_string(),
                artifact_format: "owned-safetensors".to_string(),
                tokenizer_sanitized_digest: "sha256:tokenizer".to_string(),
                model_locator: ModelAssetLocator::CacheDigest {
                    digest: "sha256:source".to_string(),
                },
                tokenizer_locator: ModelAssetLocator::CacheDigest {
                    digest: "sha256:tokenizer".to_string(),
                },
                model_source_url: "file:///model".to_string(),
                tokenizer_source_url: "file:///tokenizer".to_string(),
                pooling: "none".to_string(),
                normalize: false,
                max_tokens: 2048,
                quant: entry.weight_quant.as_str().to_string(),
                pin: true,
                owned_family: Some(entry.family.as_str().to_string()),
                owned_dtype: Some("f16".to_string()),
                owned_execution: Some("supervised".to_string()),
                owned_attention_units: None,
                config_locator: None,
                extra_locators: Vec::new(),
                engine_identity: EngineIdentity {
                    engine: CATALOG_ENGINE.to_string(),
                    version: "test".to_string(),
                    build_flags: BTreeMap::new(),
                },
                numeric_profile_id: NumericProfileId("test".to_string()),
                fingerprint: decode_fingerprint.clone(),
                worker_bin: None,
                worker_runtime_dir: None,
            },
            1,
        )
        .unwrap();
    store
        .observe_profile(
            &MachineProfile {
                os_build: "test-os".to_string(),
                arch: "aarch64".to_string(),
                chip_model: "test-chip".to_string(),
                ram_class: "test".to_string(),
                ane_subtype: None,
                engine_identities: Vec::new(),
            },
            1,
            1,
        )
        .unwrap();
    let approval = store
        .create_approval(
            &entry.entry_id,
            &decode_fingerprint.0,
            true,
            "test-operator",
            1,
            1,
        )
        .unwrap();
    (root, store, approval)
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
enum BoundaryMutation {
    Disable,
    EmergencyRollback,
}

fn assert_dispatch_refuses_committed_approval_mutation(mutation: BoundaryMutation) {
    let entry = catalog_entry(Family::Qwen3_0_6b, WeightQuant::F16);
    let decode_fingerprint = entry.decode_identity_inputs().decode_fingerprint().unwrap();
    let (root, store, approval) = dispatch_boundary_store(&entry, &decode_fingerprint);
    let boundary_reached = Arc::new(Barrier::new(2));
    let mutation_committed = Arc::new(Barrier::new(2));
    let reader: Arc<dyn lane::AdmissionBoundaryReader> = Arc::new(BarrierBoundaryReader {
        store: Arc::clone(&store),
        boundary_reached: Arc::clone(&boundary_reached),
        mutation_committed: Arc::clone(&mutation_committed),
    });
    let environment = env(true, Some(llama_lane())).with_admission_boundary(
        lane::AdmissionBoundarySnapshot {
            profile_activation_epoch: 1,
            model_id: approval.model_id.clone(),
            decode_fingerprint: approval.decode_fingerprint.clone(),
            approval_semantic_digest: approval.semantic_digest.clone(),
            approval_generation: approval.generation,
        },
        reader,
    );
    let mutation_store = Arc::clone(&store);
    let mutation_model_id = approval.model_id.clone();
    let mutation_decode_fingerprint = approval.decode_fingerprint.clone();
    let mutation_thread = std::thread::spawn(move || {
        boundary_reached.wait();
        match mutation {
            BoundaryMutation::Disable => {
                mutation_store
                    .disable_approval(
                        &mutation_model_id,
                        &mutation_decode_fingerprint,
                        "operator disable",
                        2,
                    )
                    .unwrap();
            }
            BoundaryMutation::EmergencyRollback => {
                assert_eq!(
                    mutation_store
                        .emergency_rollback("emergency rollback", 2)
                        .unwrap(),
                    1
                );
            }
        }
        mutation_committed.wait();
    });
    let router = router_with_certified(certified_store(&entry), Q8IngestRegistry::new());
    let dispatched = Rc::new(RefCell::new(Vec::new()));
    let mut dispatch = RecordingDispatch::new_succeeding(dispatched.clone());
    let failure = router
        .route_oneshot(
            &environment,
            &entry,
            &request(Family::Qwen3_0_6b, WeightQuant::F16),
            "gen-approval-mutated",
            &mut dispatch,
        )
        .expect_err("approval mutation committed before dispatch must refuse");
    mutation_thread.join().unwrap();
    assert_eq!(failure.wire_id(), "owned_decode_not_certified");
    assert!(failure
        .message
        .as_deref()
        .is_some_and(|message| message.contains("stale_probe")));
    assert!(dispatched.borrow().is_empty());
    drop(environment);
    drop(store);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn owned_dispatch_refuses_disable_committed_after_admission_resolution() {
    assert_dispatch_refuses_committed_approval_mutation(BoundaryMutation::Disable);
}

#[test]
fn owned_dispatch_refuses_rollback_committed_after_admission_resolution() {
    assert_dispatch_refuses_committed_approval_mutation(BoundaryMutation::EmergencyRollback);
}
