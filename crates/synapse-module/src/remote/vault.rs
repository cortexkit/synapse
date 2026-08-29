use std::{future::Future, path::PathBuf, sync::Arc, time::Duration};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use subc_client_rs::{async_trait, CallError, CallOptions, ConsumerOptions, SubcConsumer};
use subc_protocol::{BindIdentity, Priority, RouteTarget};
use tokio::sync::Mutex;

use super::runtime::{CredentialToken, VaultCredentialClient, VaultError};

const CREDENTIALS_MODULE_ID: &str = "claustrum";
/// Envelope key claustrum wraps every `credential.get` reply in.
const CREDENTIAL_RESULT_KEY: &str = "result";
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
    class: String,
    code: String,
}

/// Selects the credential envelope explicitly instead of by `#[serde(untagged)]`
/// variant order.
///
/// Untagged enums are tried in declaration order, so a frame carrying both a
/// wrapped `result.error` and top-level success fields resolved as a SUCCESS and
/// discarded the error — returning a token built from the stray bytes. Claustrum
/// cannot emit that frame today (their outcome is an enum whose variants cannot
/// coexist), but the envelope choice was resting on declaration order rather than
/// on a decision, and it failed toward serving a credential.
///
/// `result` therefore wins whenever it is present. The unwrapped form stays
/// accepted, but as a documented fallback rather than a coincidence of ordering.
fn decode_credential_get_response(response: &[u8]) -> Result<CredentialGetOutcome, VaultError> {
    let envelope: Value =
        serde_json::from_slice(response).map_err(|_| VaultError::MalformedHandlesFile)?;
    let outcome = match envelope.get(CREDENTIAL_RESULT_KEY) {
        Some(result) => result.clone(),
        None => envelope,
    };
    serde_json::from_value(outcome).map_err(|_| VaultError::MalformedHandlesFile)
}

fn map_credential_read_error(error: CredentialReadError) -> VaultError {
    VaultError::CredentialFailure {
        class: error.class,
        code: error.code,
    }
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
        match decode_credential_get_response(&response)? {
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
            CredentialGetOutcome::AppError { error } => Err(map_credential_read_error(error)),
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
        // Generic module call errors are top-level frames, not claustrum credential
        // result errors. They have no credential error class, so treat them as a
        // recoverable route failure instead of mislabeling a valid frame as malformed.
        CallError::Module(_) => VaultError::Unreachable,
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
        let response = decode_credential_get_response(
            br#"{"result":{"payload":[115,101,99,114,101,116],"expires_at_ms":null,"record_version":3}}"#,
        )
        .expect("a well-formed credential success must deserialize");
        let CredentialGetOutcome::Success { payload, .. } = response else {
            panic!("expected credential success");
        };
        assert_eq!(payload.len(), 6);
    }

    #[test]
    fn wrapped_error_wins_over_stray_top_level_success_fields() {
        // Envelope selection must not depend on untagged variant order: the
        // wrapped error is the reply, and the stray top-level fields are not a
        // credential. Decoding this as a success would hand a provider call a
        // token built from bytes that accompanied a failure.
        let response = decode_credential_get_response(
            br#"{"result":{"error":{"class":"transient","code":"x"}},"payload":[1],"record_version":9}"#,
        )
        .expect("the wrapped error is a well-formed frame");
        let CredentialGetOutcome::AppError { error } = response else {
            panic!("stray top-level success fields must not outrank a wrapped error");
        };
        assert_eq!(error.class, "transient");
    }

    #[test]
    fn unwrapped_outcome_is_still_accepted() {
        // The fallback is deliberate, not incidental — pin it so removing it is a
        // decision rather than a side effect.
        let response = decode_credential_get_response(
            br#"{"error":{"class":"permanent","code":"not_found"}}"#,
        )
        .expect("an unwrapped outcome remains decodable");
        let CredentialGetOutcome::AppError { error } = response else {
            panic!("expected credential error outcome");
        };
        assert_eq!(error.code, "not_found");
    }

    #[test]
    fn missing_error_code_is_a_malformed_credential_frame() {
        // `code` is non-Option, so this is currently rejected by the type rather
        // than by intent. Pinning it turns an accident into a guarantee: adding
        // #[serde(default)] later goes red here instead of silently widening
        // what synapse accepts from the vault.
        let Err(error) =
            decode_credential_get_response(br#"{"result":{"error":{"class":"permanent"}}}"#)
        else {
            panic!("credential errors without the required code are malformed");
        };

        assert_eq!(error, VaultError::MalformedHandlesFile);
    }

    #[test]
    fn unknown_code_in_known_class_deserializes_without_becoming_malformed() {
        let response = decode_credential_get_response(
            br#"{"result":{"error":{"class":"transient","code":"future_refresh_path"}}}"#,
        )
        .expect("a well-formed credential error must deserialize");
        let CredentialGetOutcome::AppError { error } = response else {
            panic!("expected credential error outcome");
        };

        assert_eq!(error.class, "transient");
        assert_eq!(error.code, "future_refresh_path");
        assert_eq!(
            map_credential_read_error(error),
            VaultError::CredentialFailure {
                class: "transient".to_string(),
                code: "future_refresh_path".to_string(),
            }
        );
    }

    #[test]
    fn missing_error_class_is_a_malformed_credential_frame() {
        let Err(error) =
            decode_credential_get_response(br#"{"result":{"error":{"code":"not_found"}}}"#)
        else {
            panic!("credential errors without the required class are malformed");
        };

        assert_eq!(error, VaultError::MalformedHandlesFile);
    }
}
