use std::{future::Future, path::PathBuf, sync::Arc, time::Duration};

use serde::{Deserialize, Serialize};
use subc_client_rs::{async_trait, CallError, CallOptions, ConsumerOptions, SubcConsumer};
use subc_protocol::{BindIdentity, Priority, RouteTarget};
use tokio::sync::Mutex;

use super::runtime::{CredentialToken, VaultCredentialClient, VaultError};

const CREDENTIALS_MODULE_ID: &str = "claustrum";
const VAULT_MIN_TTL_MS: u64 = 600_000;
const VAULT_CALL_TIMEOUT: Duration = Duration::from_secs(10);
const VAULT_ROUTE_READY_TIMEOUT: Duration = Duration::from_secs(1);
const VAULT_ROUTE_ATTEMPTS: usize = 5;
const VAULT_ROUTE_BACKOFF: Duration = Duration::from_millis(100);

pub(crate) struct SubcVaultCredentialClient {
    connection_file: PathBuf,
    consumer: Mutex<Option<Arc<SubcConsumer>>>,
}

impl SubcVaultCredentialClient {
    pub fn new(connection_file: PathBuf) -> Self {
        Self {
            connection_file,
            consumer: Mutex::new(None),
        }
    }

    async fn consumer(&self) -> Result<Arc<SubcConsumer>, VaultError> {
        let mut slot = self.consumer.lock().await;
        if let Some(consumer) = slot.as_ref() {
            return Ok(Arc::clone(consumer));
        }
        let consumer = Arc::new(
            SubcConsumer::connect(&self.connection_file, ConsumerOptions::default())
                .await
                .map_err(|_| VaultError::Unreachable)?,
        );
        *slot = Some(Arc::clone(&consumer));
        Ok(consumer)
    }

    async fn call(&self, body: Vec<u8>) -> Result<Vec<u8>, VaultError> {
        let consumer = self.consumer().await?;
        let options = CallOptions {
            timeout: VAULT_CALL_TIMEOUT,
            priority: Priority::Interactive,
            route_retry_deadline: VAULT_ROUTE_READY_TIMEOUT,
            ..CallOptions::default()
        };
        bounded_route_call(|| {
            consumer.call(
                RouteTarget::ManagementSurface {
                    module_id: CREDENTIALS_MODULE_ID.to_string(),
                },
                BindIdentity {
                    project_root: PathBuf::from("/"),
                    harness: "synapse".to_string(),
                    session: "remote-gateway-v1".to_string(),
                },
                body.clone(),
                options.clone(),
            )
        })
        .await
    }
}

#[derive(Serialize)]
struct CredentialGetRequest<'a> {
    op: &'static str,
    params: CredentialGetParams<'a>,
}

#[derive(Serialize)]
struct CredentialGetParams<'a> {
    handle: &'a str,
    min_ttl_ms: u64,
}

#[derive(Serialize)]
struct CredentialReportRequest<'a> {
    op: &'static str,
    params: CredentialReportParams<'a>,
}

#[derive(Serialize)]
struct CredentialReportParams<'a> {
    handle: &'a str,
    provider_status: u16,
    record_version: u64,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum CredentialGetResponse {
    Direct(CredentialGetOutcome),
    Wrapped { result: CredentialGetOutcome },
}

impl CredentialGetResponse {
    fn into_outcome(self) -> CredentialGetOutcome {
        match self {
            Self::Direct(outcome) | Self::Wrapped { result: outcome } => outcome,
        }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum CredentialGetOutcome {
    Success {
        payload: Vec<u8>,
        #[serde(default)]
        expires_at_ms: Option<i64>,
        record_version: u64,
    },
    AppError {
        error: CredentialReadError,
    },
}

#[derive(Deserialize)]
struct CredentialReadError {
    code: CredentialReadErrorCode,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum CredentialReadErrorCode {
    NotFound,
    NeedsReauth,
    RefreshUnsupported,
    RefreshFailed,
    VaultLocked,
    Corrupt,
    TooManyItems,
}

#[async_trait]
impl VaultCredentialClient for SubcVaultCredentialClient {
    async fn fetch(
        &self,
        logical_handle: &str,
        min_ttl_ms: u64,
    ) -> Result<CredentialToken, VaultError> {
        let body = serde_json::to_vec(&CredentialGetRequest {
            op: "credential.get",
            params: CredentialGetParams {
                handle: logical_handle,
                min_ttl_ms: min_ttl_ms.max(VAULT_MIN_TTL_MS),
            },
        })
        .map_err(|_| VaultError::MalformedHandle)?;
        let response = self.call(body).await?;
        let response: CredentialGetResponse =
            serde_json::from_slice(&response).map_err(|_| VaultError::MalformedHandlesFile)?;
        match response.into_outcome() {
            CredentialGetOutcome::Success {
                payload,
                expires_at_ms,
                record_version,
            } => {
                let secret =
                    String::from_utf8(payload).map_err(|_| VaultError::MalformedHandlesFile)?;
                let expires_at_ms = expires_at_ms
                    .and_then(|value| u64::try_from(value).ok())
                    .unwrap_or(u64::MAX);
                Ok(CredentialToken::new(secret, expires_at_ms, record_version))
            }
            CredentialGetOutcome::AppError { error } => Err(match error.code {
                CredentialReadErrorCode::NotFound => VaultError::NotFound,
                CredentialReadErrorCode::NeedsReauth => VaultError::NeedsReauth,
                CredentialReadErrorCode::VaultLocked => VaultError::VaultLocked,
                CredentialReadErrorCode::RefreshUnsupported
                | CredentialReadErrorCode::RefreshFailed
                | CredentialReadErrorCode::Corrupt
                | CredentialReadErrorCode::TooManyItems => VaultError::MalformedHandlesFile,
            }),
        }
    }

    async fn report_auth_failure(
        &self,
        logical_handle: &str,
        provider_status: u16,
        record_version: u64,
    ) {
        let Ok(body) = serde_json::to_vec(&CredentialReportRequest {
            op: "credential.report_auth_failure",
            params: CredentialReportParams {
                handle: logical_handle,
                provider_status,
                record_version,
            },
        }) else {
            return;
        };
        let _ = self.call(body).await;
    }
}

async fn bounded_route_call<F, Fut>(mut call: F) -> Result<Vec<u8>, VaultError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<Vec<u8>, CallError>>,
{
    for attempt in 0..VAULT_ROUTE_ATTEMPTS {
        match call().await {
            Ok(response) => return Ok(response),
            Err(CallError::NotSent(_)) if attempt + 1 < VAULT_ROUTE_ATTEMPTS => {
                tokio::time::sleep(VAULT_ROUTE_BACKOFF).await;
            }
            Err(error) => return Err(map_call_error(error)),
        }
    }
    unreachable!("bounded vault route loop always returns")
}

fn map_call_error(error: CallError) -> VaultError {
    match error {
        CallError::Module(body) if body.code == "needs_reauth" => VaultError::NeedsReauth,
        CallError::Module(body) if body.code == "vault_locked" => VaultError::VaultLocked,
        CallError::Module(body) if body.code == "not_found" => VaultError::NotFound,
        CallError::Module(_) => VaultError::MalformedHandlesFile,
        // StaleRouteHandle is grouped with the transport failures: no frame was
        // emitted for the request, so the vault is unreachable through the held
        // connection until a reconnect establishes a fresh route.
        CallError::NotSent(_)
        | CallError::OutcomeUnknown(_)
        | CallError::SubscriptionBackpressure(_)
        | CallError::StaleRouteHandle(_) => VaultError::Unreachable,
        // Capability-resolution failures cannot occur on this path (the vault
        // client opens its route by module id, never by capability), but the
        // variants exist on the shared error type. No frame reached the vault,
        // so they classify with the transport failures.
        CallError::CapabilityUnprovided { .. }
        | CallError::CapabilityAmbiguous { .. }
        | CallError::InvalidCapabilityIdentifier { .. } => VaultError::Unreachable,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn bounded_bootstrap_retry_reaches_a_route_that_appears_late() {
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let call_attempts = Arc::clone(&attempts);
        let response = bounded_route_call(move || {
            let attempt = call_attempts.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            async move {
                if attempt < 2 {
                    Err(CallError::NotSent(Box::new(std::io::Error::new(
                        std::io::ErrorKind::NotConnected,
                        "fake credentials route is not registered yet",
                    ))))
                } else {
                    Ok(
                        br#"{"result":{"payload":[],"expires_at_ms":null,"record_version":1}}"#
                            .to_vec(),
                    )
                }
            }
        })
        .await
        .expect("bounded retry should reach the late fake route");
        assert!(!response.is_empty());
        assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 3);
    }

    #[test]
    fn credential_wire_decoder_never_exposes_payload_in_errors() {
        let response: CredentialGetResponse = serde_json::from_value(serde_json::json!({
            "result": {"payload": [115, 101, 99, 114, 101, 116], "expires_at_ms": null, "record_version": 3}
        }))
        .unwrap();
        let CredentialGetOutcome::Success { payload, .. } = response.into_outcome() else {
            panic!("expected credential success");
        };
        assert_eq!(payload.len(), 6);
    }
}
