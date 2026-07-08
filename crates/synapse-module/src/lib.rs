#![forbid(unsafe_code)]

use std::{
    env,
    sync::{Arc, OnceLock},
};

mod store;
#[cfg(unix)]
pub mod worker_host;

use cortexkit_store_types::{sqlite_store_path, Isolation, StorageBackend, StorageDescriptor};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use store::{CatalogSnapshot, SynapseStore, SynapseStoreError};
use subc_client_rs::{
    async_trait, BindDecision, HandlerOutcome, HealthReport, ModuleHandler, RequestCtx,
    RouteBindRequest, SubcModuleError,
};
use subc_protocol::{
    manifest::{
        Bindings, IdentityBinding, IdentityScope, ManagementOperation, ManagementOperationKind,
        ModuleManifest, ProviderRole, StorageBinding, StorageKind, StorageScope, TrustTier,
    },
    ModuleHelloAckBody, PROTOCOL_VERSION, SUBC_MODULE_ID_ENV,
};
use synapse_core::{ErrorClass, StableError};
use thiserror::Error;

pub const DEFAULT_MODULE_ID: &str = "synapse";

pub async fn run_from_env() -> Result<(), ModuleError> {
    let module_id = env::var(SUBC_MODULE_ID_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_MODULE_ID.to_string());
    let handler = SynapseHandler::new(module_id.clone());
    subc_client_rs::serve(manifest(&module_id), handler)
        .await
        .map_err(ModuleError::Serve)
}

#[derive(Debug, Error)]
pub enum ModuleError {
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("storage: {0}")]
    Store(#[from] SynapseStoreError),
    #[error("subc serve: {0}")]
    Serve(#[from] SubcModuleError),
}

#[derive(Clone)]
struct SynapseHandler {
    inner: Arc<SynapseHandlerInner>,
}

struct SynapseHandlerInner {
    module_id: String,
    state: OnceLock<Arc<ModuleState>>,
}

struct ModuleState {
    store: Arc<SynapseStore>,
    module_generation: u64,
    health: ModuleHealth,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct ModuleHealth {
    status: String,
    module_generation: u64,
}

#[derive(Debug, Deserialize)]
struct MethodEnvelope {
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct WireOperationError {
    code: String,
    class: ErrorClass,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    retry_after_ms: Option<u64>,
    safe_to_retry_same_request: bool,
    message: String,
}

impl WireOperationError {
    fn from_stable(error: StableError, message: impl Into<String>) -> Self {
        Self {
            code: serde_json::to_value(error.code)
                .expect("stable error code serializes")
                .as_str()
                .expect("stable error code is a string")
                .to_string(),
            class: error.class,
            retry_after_ms: error.retry_after_ms,
            safe_to_retry_same_request: error.safe_to_retry_same_request,
            message: message.into(),
        }
    }

    fn not_implemented(message: impl Into<String>) -> Self {
        Self {
            code: "not_implemented".to_string(),
            class: ErrorClass::Permanent,
            retry_after_ms: None,
            safe_to_retry_same_request: false,
            message: message.into(),
        }
    }
}

impl SynapseHandler {
    fn new(module_id: String) -> Self {
        Self {
            inner: Arc::new(SynapseHandlerInner {
                module_id,
                state: OnceLock::new(),
            }),
        }
    }

    fn state(&self) -> Option<Arc<ModuleState>> {
        self.inner.state.get().cloned()
    }

    fn initialize(&self, ack: &ModuleHelloAckBody) -> Result<Arc<ModuleState>, ModuleError> {
        let descriptor = resolve_storage_descriptor(&ack.storage, &self.inner.module_id)?;
        let store = Arc::new(SynapseStore::open(&descriptor)?);
        let module_generation = store.next_module_generation()?;
        let health = ModuleHealth {
            status: "ok".to_string(),
            module_generation,
        };
        Ok(Arc::new(ModuleState {
            store,
            module_generation,
            health,
        }))
    }
}

#[async_trait]
impl ModuleHandler for SynapseHandler {
    async fn on_hello_ack(&self, ack: &ModuleHelloAckBody) {
        if self.state().is_some() {
            return;
        }
        let state = self
            .initialize(ack)
            .unwrap_or_else(|error| panic!("synapse boot failed after HELLO_ACK: {error}"));
        let _ = self.inner.state.set(state);
    }

    async fn on_bind(&self, _req: &RouteBindRequest) -> BindDecision {
        if self.state().is_some() {
            BindDecision::accept()
        } else {
            BindDecision::reject(
                "module_not_initialized",
                "synapse has not completed HELLO_ACK initialization",
            )
        }
    }

    async fn health(&self) -> HealthReport {
        let Some(state) = self.state() else {
            return HealthReport::ok();
        };
        HealthReport {
            status: subc_client_rs::HealthStatus::Ok,
            detail: Some("ok".to_string()),
            metrics: Some(
                serde_json::to_value(&state.health).expect("module health should serialize"),
            ),
        }
    }

    async fn handle(&self, _ctx: RequestCtx, body: Vec<u8>) -> HandlerOutcome {
        let Some(state) = self.state() else {
            return channel_error(
                "module_not_initialized",
                "synapse has not completed HELLO_ACK initialization",
            );
        };

        let envelope: MethodEnvelope = match serde_json::from_slice(&body) {
            Ok(envelope) => envelope,
            Err(error) => {
                return channel_error(
                    "invalid_request",
                    format!("route request body is not decodable: {error}"),
                )
            }
        };

        dispatch_request(state, envelope)
    }
}

fn dispatch_request(state: Arc<ModuleState>, request: MethodEnvelope) -> HandlerOutcome {
    let _ = &request.params;
    match request.method.as_str() {
        "models.list" => match state.store.catalog_snapshot() {
            Ok(snapshot) => result_outcome(models_list_payload(&state, snapshot)),
            Err(error) => channel_error("store_failure", error.to_string()),
        },
        "embed.query" | "embed.batch" | "rerank.score" | "microllm.oneshot" => {
            result_outcome(error_payload(
                &state,
                WireOperationError::from_stable(
                    StableError::probe_required(),
                    format!(
                        "{} requires a completed probe before this scaffold can serve requests",
                        request.method
                    ),
                ),
            ))
        }
        "model.load" | "probe.start" | "cache.pin" | "cache.gc" => result_outcome(
            error_payload(
                &state,
                WireOperationError::not_implemented(format!(
                    "{} is reserved in the scaffold but not implemented yet",
                    request.method
                )),
            ),
        ),
        "model.status" | "probe.status" | "aliases.check_index" | "admission.status" => {
            result_outcome(error_payload(
                &state,
                WireOperationError::not_implemented(format!(
                    "{} is not implemented in the wire-alive scaffold",
                    request.method
                )),
            ))
        }
        other => channel_error(
            "unknown_method",
            format!("unknown method '{other}' for synapse management surface"),
        ),
    }
}

fn models_list_payload(state: &ModuleState, snapshot: CatalogSnapshot) -> Value {
    json!({
        "module_generation": state.module_generation,
        "table_epoch": snapshot.table_epoch,
        "models": snapshot.models,
        "alias_rows": snapshot.alias_rows,
    })
}

fn error_payload(state: &ModuleState, error: WireOperationError) -> Value {
    json!({
        "module_generation": state.module_generation,
        "error": error,
    })
}

fn result_outcome(result: Value) -> HandlerOutcome {
    match serde_json::to_vec(&json!({ "result": result })) {
        Ok(body) => HandlerOutcome::Response(body),
        Err(error) => channel_error("encode_failed", error.to_string()),
    }
}

fn channel_error(code: impl Into<String>, message: impl Into<String>) -> HandlerOutcome {
    HandlerOutcome::Error {
        code: code.into(),
        message: message.into(),
    }
}

fn resolve_storage_descriptor(
    ack_storage: &Option<Value>,
    module_id: &str,
) -> Result<StorageDescriptor, ModuleError> {
    if let Some(value) = ack_storage {
        return serde_json::from_value(value.clone()).map_err(ModuleError::Json);
    }

    let path = sqlite_store_path(&std::env::temp_dir().to_string_lossy(), module_id);
    Ok(StorageDescriptor {
        module_id: module_id.to_string(),
        storage_namespace: "default".to_string(),
        isolation: Isolation::Module,
        backend: StorageBackend::Sqlite { path },
    })
}

fn management_operations() -> Vec<ManagementOperation> {
    use ManagementOperationKind::{Mutate, Query};

    let op = |name: &str, kind| ManagementOperation {
        name: name.to_string(),
        kind,
    };

    vec![
        op("embed.query", Query),
        op("embed.batch", Query),
        op("rerank.score", Query),
        op("microllm.oneshot", Query),
        op("model.load", Mutate),
        op("model.status", Query),
        op("models.list", Query),
        op("probe.start", Mutate),
        op("probe.status", Query),
        op("aliases.check_index", Query),
        op("cache.pin", Mutate),
        op("cache.gc", Mutate),
        op("admission.status", Query),
    ]
}

fn manifest(module_id: &str) -> ModuleManifest {
    ModuleManifest {
        module_id: module_id.to_string(),
        module_version: env!("CARGO_PKG_VERSION").to_string(),
        protocol_ver: PROTOCOL_VERSION,
        trust_tier: TrustTier::FirstParty,
        provides: vec![ProviderRole::ManagementSurface {
            operations: management_operations(),
            config_schema: json!({ "type": "object" }),
            observability: Vec::new(),
            identity_scope: vec![IdentityScope::Project, IdentityScope::Session],
        }],
        consumes: Vec::new(),
        scheduled_tasks: Vec::new(),
        bindings: Bindings {
            storage: StorageBinding {
                kind: StorageKind::Sqlite,
                scope: StorageScope::Project,
                owns_schema: true,
            },
            vault_grants: Vec::new(),
            identity: IdentityBinding {
                requires: vec![IdentityScope::Project],
                optional: vec![IdentityScope::Session],
            },
        },
    }
}
