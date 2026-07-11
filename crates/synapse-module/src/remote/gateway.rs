use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{Duration, Instant, SystemTime},
};

use reqwest::{header, Method, StatusCode};
use serde::Serialize;
use synapse_core::{RemoteProvenance, ResponseProvenance, StableError};

use super::{
    classify::{classify, preset, FailureClass, ProviderFailure},
    client::{EndpointSecurity, GatewayClientError, GatewayHttpClient},
    config::{
        ConfiguredAuth, ConfiguredProvider, ConfiguredRemoteProfile, RemoteTask, ADAPTER_VERSION,
    },
    openai_compat::{parse_embedding_response, EmbeddingRequest, EMBEDDINGS_PATH},
    runtime::{
        BreakerConfig, BreakerKey, CredentialDisposition, ProviderRuntime, RemoteClass,
        RuntimeError, SentinelContinuityCheck, SentinelEmbedder, SentinelProfile,
        VaultCredentialClient,
    },
    validator::validate_embedding_response,
};
use crate::store::SynapseStore;

const MAX_ATTEMPTS: usize = 3;

#[derive(Clone)]
struct ProviderConnection {
    config: ConfiguredProvider,
    client: GatewayHttpClient,
    runtime: Arc<ProviderRuntime>,
}

#[derive(Clone, Debug)]
pub(crate) struct RemoteEmbeddingResult {
    pub vectors: Vec<Vec<f32>>,
    pub submitted_texts: Vec<String>,
    pub token_counts: Vec<u32>,
    pub provider_request_id: Option<String>,
}

#[derive(Debug)]
pub(crate) struct RemoteGatewayError {
    pub stable: StableError,
    pub message: String,
    pub provider_request_id: Option<String>,
}

impl From<RuntimeError> for RemoteGatewayError {
    fn from(error: RuntimeError) -> Self {
        Self {
            stable: error.stable,
            message: error.message,
            provider_request_id: None,
        }
    }
}

pub(crate) struct RemoteGateway {
    providers: Arc<RwLock<HashMap<String, Arc<ProviderConnection>>>>,
    profiles: HashMap<String, Arc<ConfiguredRemoteProfile>>,
    machine_profile_hash: String,
    pub continuity: Arc<SentinelContinuityCheck>,
}

impl RemoteGateway {
    pub fn new(
        store: Arc<SynapseStore>,
        providers: Vec<ConfiguredProvider>,
        credential_client: Arc<dyn VaultCredentialClient>,
        machine_profile_hash: String,
    ) -> Result<Self, RemoteGatewayError> {
        let connections = Arc::new(RwLock::new(HashMap::new()));
        let profiles = providers
            .iter()
            .flat_map(|provider| provider.models.iter())
            .map(|profile| (profile.synapse_model_id.clone(), Arc::new(profile.clone())))
            .collect::<HashMap<_, _>>();
        let embedder = Arc::new(GatewaySentinelEmbedder {
            providers: Arc::clone(&connections),
            profiles: profiles.clone(),
            machine_profile_hash: machine_profile_hash.clone(),
        });
        let continuity = Arc::new(SentinelContinuityCheck::new(store, embedder));

        for provider in providers {
            let client = GatewayHttpClient::new(
                provider.connect_timeout,
                provider.read_timeout,
                provider.response_body_limit_bytes,
            )
            .map_err(|error| RemoteGatewayError {
                stable: StableError::invalid_request(),
                message: format!("build remote provider '{}' client: {error}", provider.name),
                provider_request_id: None,
            })?;
            let runtime = Arc::new(ProviderRuntime::new(
                BreakerConfig {
                    failure_threshold: provider.breaker_failure_threshold,
                    cooldown_ms: provider.breaker_cooldown_ms,
                },
                Arc::clone(&credential_client),
                Arc::clone(&continuity),
            )?);
            runtime.register_provider(&provider.name, provider.max_concurrency)?;
            connections
                .write()
                .expect("remote provider map lock poisoned")
                .insert(
                    provider.name.clone(),
                    Arc::new(ProviderConnection {
                        config: provider,
                        client,
                        runtime,
                    }),
                );
        }

        for profile in profiles.values() {
            if profile.task == RemoteTask::Embed {
                continuity.register_profile(sentinel_profile(profile, &machine_profile_hash));
            }
        }
        Ok(Self {
            providers: connections,
            profiles,
            machine_profile_hash,
            continuity,
        })
    }

    pub fn profile(&self, model_id: &str) -> Option<Arc<ConfiguredRemoteProfile>> {
        self.profiles.get(model_id).cloned()
    }

    pub fn is_remote(&self, model_id: &str) -> bool {
        self.profiles.contains_key(model_id)
    }

    pub fn profiles(&self) -> Vec<Arc<ConfiguredRemoteProfile>> {
        self.profiles.values().cloned().collect()
    }

    pub fn catalog_entries(&self) -> Vec<serde_json::Value> {
        let mut entries = self
            .profiles
            .values()
            .map(|profile| {
                serde_json::json!({
                    "model_id": profile.synapse_model_id,
                    "state": "declared",
                    "fingerprints": [profile.fingerprint],
                    "assurance": "declared",
                    "identity_revision": profile.identity_revision,
                    "task": task_name(profile.task),
                })
            })
            .collect::<Vec<_>>();
        entries.sort_by(|left, right| left["model_id"].as_str().cmp(&right["model_id"].as_str()));
        entries
    }

    pub fn logical_handle(&self, profile: &ConfiguredRemoteProfile) -> Option<String> {
        self.providers
            .read()
            .expect("remote provider map lock poisoned")
            .get(&profile.provider)
            .and_then(|connection| connection.config.auth.logical_handle())
            .map(str::to_string)
    }

    pub fn provenance(&self, profile: &ConfiguredRemoteProfile) -> ResponseProvenance {
        ResponseProvenance {
            engine: synapse_core::EngineIdentity {
                engine: "remote_openai_compatible".to_string(),
                version: ADAPTER_VERSION.to_string(),
                build_flags: Default::default(),
            },
            remote: Some(RemoteProvenance {
                provider: profile.provider.clone(),
                deployment: profile.deployment.clone(),
                assurance: "declared".to_string(),
            }),
        }
    }

    pub fn predicted_finish_ms(
        &self,
        profile: &ConfiguredRemoteProfile,
        total_tokens: u64,
        now_ms: u64,
    ) -> Result<u64, RemoteGatewayError> {
        let connection = self.connection(&profile.provider)?;
        let lease = connection
            .runtime
            .breaker
            .admit(&breaker_key(&connection), now_ms)?;
        drop(lease);
        Ok(connection.runtime.estimator.estimate_ms(
            &profile.provider,
            "embed",
            total_tokens,
            connection.config.cold_embed_estimate_ms,
            now_ms,
        ))
    }

    pub async fn embed(
        &self,
        profile: &ConfiguredRemoteProfile,
        texts: &[String],
        class: RemoteClass,
        remaining_deadline_ms: u64,
    ) -> Result<RemoteEmbeddingResult, RemoteGatewayError> {
        let submitted = texts
            .iter()
            .map(|text| truncate_text(text, profile.max_input_tokens))
            .collect::<Vec<_>>();
        self.embed_submitted(profile, submitted, class, remaining_deadline_ms)
            .await
    }

    pub async fn ensure_certified(
        &self,
        profile: &ConfiguredRemoteProfile,
    ) -> Result<(), RemoteGatewayError> {
        self.continuity
            .check_profile(&sentinel_profile(profile, &self.machine_profile_hash), None)
            .await
            .map_err(RemoteGatewayError::from)
    }

    pub async fn calibrate(
        &self,
        profile: &ConfiguredRemoteProfile,
        module_generation: u64,
        certified_at_ms: u64,
    ) -> Result<(), RemoteGatewayError> {
        if profile.task != RemoteTask::Embed {
            return Err(RemoteGatewayError {
                stable: StableError::op_not_supported_for_remote(),
                message: format!(
                    "probe calibration for remote task '{}' is not supported in gateway v1",
                    task_name(profile.task)
                ),
                provider_request_id: None,
            });
        }
        let connection = self.connection(&profile.provider)?;
        verify_over_limit_rejection(&connection, profile).await?;
        self.continuity
            .calibrate_and_store(
                &sentinel_profile(profile, &self.machine_profile_hash),
                connection.config.auth.logical_handle(),
                certified_at_ms,
                module_generation,
            )
            .await
            .map_err(|error| {
                let stable = if error.stable == StableError::provider_protocol_violation()
                    && error.message.contains("self-noise")
                {
                    StableError::sentinel_calibration_refused()
                } else {
                    error.stable
                };
                RemoteGatewayError {
                    stable,
                    message: error.message,
                    provider_request_id: None,
                }
            })?;
        Ok(())
    }

    async fn embed_submitted(
        &self,
        profile: &ConfiguredRemoteProfile,
        submitted_texts: Vec<String>,
        class: RemoteClass,
        remaining_deadline_ms: u64,
    ) -> Result<RemoteEmbeddingResult, RemoteGatewayError> {
        let connection = self.connection(&profile.provider)?;
        execute_embedding_request(
            &connection,
            profile,
            submitted_texts,
            class,
            remaining_deadline_ms,
        )
        .await
    }

    fn connection(&self, provider: &str) -> Result<Arc<ProviderConnection>, RemoteGatewayError> {
        self.providers
            .read()
            .expect("remote provider map lock poisoned")
            .get(provider)
            .cloned()
            .ok_or_else(|| RemoteGatewayError {
                stable: StableError::provider_unavailable(Some(1_000)),
                message: format!("remote provider '{provider}' is not configured"),
                provider_request_id: None,
            })
    }
}

struct GatewaySentinelEmbedder {
    providers: Arc<RwLock<HashMap<String, Arc<ProviderConnection>>>>,
    profiles: HashMap<String, Arc<ConfiguredRemoteProfile>>,
    machine_profile_hash: String,
}

#[subc_client_rs::async_trait]
impl SentinelEmbedder for GatewaySentinelEmbedder {
    async fn embed_sentinels(
        &self,
        profile: &SentinelProfile,
        _logical_handle: Option<&str>,
    ) -> Result<Vec<Vec<f32>>, RuntimeError> {
        let configured = self
            .profiles
            .get(&profile.synapse_model_id)
            .ok_or_else(|| RuntimeError::unavailable(1_000, &profile.synapse_model_id))?;
        let connection = self
            .providers
            .read()
            .expect("remote provider map lock poisoned")
            .get(&configured.provider)
            .cloned()
            .ok_or_else(|| RuntimeError::unavailable(1_000, &configured.provider))?;
        execute_embedding_request(
            &connection,
            configured,
            profile.sentinel_texts.clone(),
            RemoteClass::Interactive,
            connection
                .config
                .read_timeout
                .as_millis()
                .min(u128::from(u64::MAX)) as u64,
        )
        .await
        .map(|result| result.vectors)
        .map_err(|error| RuntimeError::from_parts(error.stable, error.message))
    }
}

async fn verify_over_limit_rejection(
    connection: &ProviderConnection,
    profile: &ConfiguredRemoteProfile,
) -> Result<(), RemoteGatewayError> {
    let token = match &connection.config.auth {
        ConfiguredAuth::None => None,
        ConfiguredAuth::Vault { handle } => Some(
            connection
                .runtime
                .credentials
                .acquire(
                    handle,
                    connection
                        .config
                        .read_timeout
                        .as_millis()
                        .min(u128::from(u64::MAX)) as u64,
                    connection
                        .config
                        .read_timeout
                        .as_millis()
                        .min(u128::from(u64::MAX)) as u64,
                )
                .await
                .map_err(|disposition| match disposition {
                    CredentialDisposition::PauseJob => RemoteGatewayError {
                        stable: StableError::needs_reauth(),
                        message: format!(
                            "credential for remote provider '{}' needs reauthentication",
                            profile.provider
                        ),
                        provider_request_id: None,
                    },
                    CredentialDisposition::Reject(error) => error.into(),
                })?,
        ),
    };
    let oversized = std::iter::repeat_n("x", profile.max_input_tokens.saturating_add(1))
        .collect::<Vec<_>>()
        .join(" ");
    let endpoint = embedding_endpoint(&connection.config.base_url)?;
    let mut request = connection
        .client
        .request(Method::POST, endpoint)
        .header(header::CONTENT_TYPE, "application/json")
        .json(&EmbeddingRequest {
            model: profile.provider_model_id.clone(),
            input: vec![oversized],
            dimensions: Some(profile.dims),
        });
    if let Some(token) = token.as_ref() {
        request = request.bearer_auth(token.expose_secret());
    }
    let request = request.build().map_err(|error| RemoteGatewayError {
        stable: StableError::invalid_request(),
        message: format!("build remote eligibility request: {error}"),
        provider_request_id: None,
    })?;
    let security = match connection.config.auth {
        ConfiguredAuth::None => EndpointSecurity::LoopbackAuthNone,
        ConfiguredAuth::Vault { .. } => EndpointSecurity::ProviderManagedAuth,
    };
    let response = connection
        .client
        .execute(request, security)
        .await
        .map_err(|error| RemoteGatewayError {
            stable: StableError::provider_unavailable(Some(connection.config.breaker_cooldown_ms)),
            message: format!("remote eligibility probe transport failed: {error}"),
            provider_request_id: None,
        })?;
    let request_id = provider_request_id(&response.headers);
    if response.status.is_client_error()
        && response.status != StatusCode::TOO_MANY_REQUESTS
        && !matches!(
            response.status,
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN
        )
    {
        return Ok(());
    }
    if matches!(
        response.status,
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN
    ) {
        return Err(RemoteGatewayError {
            stable: StableError::needs_reauth(),
            message: format!(
                "remote provider '{}' rejected eligibility probe authentication",
                profile.provider
            ),
            provider_request_id: request_id,
        });
    }
    Err(RemoteGatewayError {
        stable: StableError::provider_protocol_violation(),
        message: if response.status.is_success() {
            "provider silently accepted an over-limit eligibility probe".to_string()
        } else {
            format!(
                "provider eligibility probe returned HTTP {} instead of an over-limit 4xx",
                response.status.as_u16()
            )
        },
        provider_request_id: request_id,
    })
}

async fn execute_embedding_request(
    connection: &ProviderConnection,
    profile: &ConfiguredRemoteProfile,
    submitted_texts: Vec<String>,
    class: RemoteClass,
    remaining_deadline_ms: u64,
) -> Result<RemoteEmbeddingResult, RemoteGatewayError> {
    let token_counts = submitted_texts
        .iter()
        .map(|text| whitespace_tokens(text).min(u32::MAX as usize) as u32)
        .collect::<Vec<_>>();
    let total_tokens = token_counts
        .iter()
        .map(|count| u64::from(*count))
        .sum::<u64>();
    let started = Instant::now();
    let mut final_request_id = None;
    let mut final_auth = None;

    for attempt in 0..MAX_ATTEMPTS {
        let now = crate::now_ms();
        let lease = connection
            .runtime
            .breaker
            .admit(&breaker_key(connection), now)
            .map_err(RemoteGatewayError::from)?;
        let pool =
            connection
                .runtime
                .pool(&profile.provider)
                .ok_or_else(|| RemoteGatewayError {
                    stable: StableError::provider_unavailable(Some(1_000)),
                    message: format!("remote provider '{}' has no runtime pool", profile.provider),
                    provider_request_id: None,
                })?;
        let _permit = pool.acquire(class).await;

        let token = match &connection.config.auth {
            ConfiguredAuth::None => None,
            ConfiguredAuth::Vault { handle } => match connection
                .runtime
                .credentials
                .acquire(
                    handle,
                    connection
                        .config
                        .read_timeout
                        .as_millis()
                        .min(u128::from(u64::MAX)) as u64,
                    remaining_deadline_ms,
                )
                .await
            {
                Ok(token) => {
                    final_auth = Some((handle.as_str(), token.record_version));
                    Some(token)
                }
                Err(CredentialDisposition::PauseJob) => {
                    return Err(RemoteGatewayError {
                        stable: StableError::needs_reauth(),
                        message: format!(
                            "credential for remote provider '{}' needs reauthentication",
                            profile.provider
                        ),
                        provider_request_id: None,
                    })
                }
                Err(CredentialDisposition::Reject(error)) => return Err(error.into()),
            },
        };

        let endpoint = embedding_endpoint(&connection.config.base_url)?;
        let mut request = connection
            .client
            .request(Method::POST, endpoint)
            .header(header::CONTENT_TYPE, "application/json")
            .json(&EmbeddingRequest {
                model: profile.provider_model_id.clone(),
                input: submitted_texts.clone(),
                dimensions: Some(profile.dims),
            });
        if let Some(token) = token.as_ref() {
            request = request.bearer_auth(token.expose_secret());
        }
        let request = request.build().map_err(|error| RemoteGatewayError {
            stable: StableError::invalid_request(),
            message: format!("build remote embedding request: {error}"),
            provider_request_id: None,
        })?;
        let security = match connection.config.auth {
            ConfiguredAuth::None => EndpointSecurity::LoopbackAuthNone,
            ConfiguredAuth::Vault { .. } => EndpointSecurity::ProviderManagedAuth,
        };
        let response = connection.client.execute(request, security).await;
        match response {
            Ok(response) if response.status.is_success() => {
                final_request_id = provider_request_id(&response.headers);
                let parsed = parse_embedding_response(&response.body).map_err(|error| {
                    RemoteGatewayError {
                        stable: StableError::provider_protocol_violation(),
                        message: format!("remote embedding response is invalid JSON: {error}"),
                        provider_request_id: final_request_id.clone(),
                    }
                })?;
                if parsed.model != profile.provider_model_id {
                    return Err(RemoteGatewayError {
                        stable: StableError::provider_protocol_violation(),
                        message: format!(
                            "provider returned model '{}' for declared model '{}'",
                            parsed.model, profile.provider_model_id
                        ),
                        provider_request_id: final_request_id,
                    });
                }
                let vectors =
                    validate_embedding_response(&parsed, submitted_texts.len(), profile.dims)
                        .map_err(|error| RemoteGatewayError {
                            stable: StableError::provider_protocol_violation(),
                            message: error.to_string(),
                            provider_request_id: final_request_id.clone(),
                        })?
                        .into_iter()
                        .map(|vector| {
                            vector
                                .into_iter()
                                .map(|value| {
                                    let value = value as f32;
                                    if value.is_finite() {
                                        Ok(value)
                                    } else {
                                        Err(())
                                    }
                                })
                                .collect::<Result<Vec<_>, _>>()
                        })
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|_| RemoteGatewayError {
                            stable: StableError::provider_protocol_violation(),
                            message: "provider vector cannot be represented as finite f32 values"
                                .to_string(),
                            provider_request_id: final_request_id.clone(),
                        })?;
                connection.runtime.breaker.record_success(&lease);
                connection.runtime.estimator.observe_latency(
                    &profile.provider,
                    "embed",
                    total_tokens,
                    started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
                    crate::now_ms(),
                );
                return Ok(RemoteEmbeddingResult {
                    vectors,
                    submitted_texts,
                    token_counts,
                    provider_request_id: final_request_id,
                });
            }
            Ok(response) => {
                final_request_id = provider_request_id(&response.headers);
                let retry_after = response
                    .headers
                    .get(header::RETRY_AFTER)
                    .and_then(|value| value.to_str().ok());
                let classification = classify(
                    preset(&connection.config.classifier_preset)
                        .expect("remote classifier was validated at config parse"),
                    ProviderFailure::Http {
                        status: response.status,
                        retry_after,
                    },
                    SystemTime::now(),
                );
                if matches!(
                    response.status,
                    StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN
                ) && (classification == FailureClass::Permanent || attempt + 1 == MAX_ATTEMPTS)
                {
                    if let Some((handle, version)) = final_auth {
                        connection
                            .runtime
                            .credentials
                            .report_terminal_auth_failure(handle, response.status.as_u16(), version)
                            .await;
                    }
                }
                match classification {
                    FailureClass::Pacing { retry_after } if attempt + 1 < MAX_ATTEMPTS => {
                        tokio::time::sleep(retry_after.unwrap_or(Duration::from_millis(250))).await;
                        continue;
                    }
                    FailureClass::Transient if attempt + 1 < MAX_ATTEMPTS => {
                        connection.runtime.breaker.record_failure(
                            &lease,
                            FailureClass::Transient,
                            crate::now_ms(),
                        );
                        continue;
                    }
                    FailureClass::Transient => {
                        connection.runtime.breaker.record_failure(
                            &lease,
                            FailureClass::Transient,
                            crate::now_ms(),
                        );
                    }
                    FailureClass::Pacing { .. } | FailureClass::Permanent => {}
                }
                let retry_after_ms = match classification {
                    FailureClass::Pacing { retry_after } => retry_after
                        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
                        .or(Some(250)),
                    FailureClass::Transient => Some(connection.config.breaker_cooldown_ms),
                    FailureClass::Permanent => None,
                };
                return Err(RemoteGatewayError {
                    stable: if matches!(
                        response.status,
                        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN
                    ) {
                        StableError::needs_reauth()
                    } else {
                        StableError::provider_unavailable(retry_after_ms)
                    },
                    message: format!(
                        "remote provider '{}' returned HTTP {}",
                        profile.provider,
                        response.status.as_u16()
                    ),
                    provider_request_id: final_request_id,
                });
            }
            Err(error) => {
                let classification = classify_client_error(&error);
                if classification == FailureClass::Permanent {
                    return Err(RemoteGatewayError {
                        stable: StableError::provider_protocol_violation(),
                        message: format!("remote provider response rejected: {error}"),
                        provider_request_id: final_request_id,
                    });
                }
                connection.runtime.breaker.record_failure(
                    &lease,
                    FailureClass::Transient,
                    crate::now_ms(),
                );
                if attempt + 1 < MAX_ATTEMPTS {
                    continue;
                }
                if is_timeout(&error) {
                    connection.runtime.estimator.observe_censored_timeout(
                        &profile.provider,
                        "embed",
                        total_tokens,
                        started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
                        crate::now_ms(),
                    );
                }
                return Err(RemoteGatewayError {
                    stable: StableError::provider_unavailable(Some(
                        connection.config.breaker_cooldown_ms,
                    )),
                    message: format!("remote provider '{}' transport failed", profile.provider),
                    provider_request_id: final_request_id,
                });
            }
        }
    }
    unreachable!("remote retry loop always returns")
}

fn sentinel_profile(
    profile: &ConfiguredRemoteProfile,
    machine_profile_hash: &str,
) -> SentinelProfile {
    SentinelProfile {
        synapse_model_id: profile.synapse_model_id.clone(),
        machine_profile_hash: machine_profile_hash.to_string(),
        remote_profile_hash: profile.remote_profile_hash.clone(),
        identity_revision: profile.identity_revision.clone(),
        fingerprint: profile.fingerprint.clone(),
        numeric_profile_id: profile.numeric_profile_id.clone(),
        sentinel_texts: profile.sentinel_texts.clone(),
        drift_gate_min: profile.drift_gate_min,
    }
}

fn breaker_key(connection: &ProviderConnection) -> BreakerKey {
    BreakerKey {
        provider: connection.config.name.clone(),
        deployment: connection.config.deployment.clone(),
    }
}

fn embedding_endpoint(base_url: &reqwest::Url) -> Result<reqwest::Url, RemoteGatewayError> {
    let mut url = base_url.clone();
    let base = url.path().trim_end_matches('/');
    url.set_path(&format!("{base}{EMBEDDINGS_PATH}"));
    Ok(url)
}

fn provider_request_id(headers: &header::HeaderMap) -> Option<String> {
    ["x-request-id", "request-id", "x-provider-request-id"]
        .iter()
        .find_map(|name| headers.get(*name))
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty() && value.len() <= 128)
        .map(str::to_string)
}

fn classify_client_error(error: &GatewayClientError) -> FailureClass {
    match error {
        GatewayClientError::BodyTooLarge { .. }
        | GatewayClientError::InvalidLoopbackHost { .. }
        | GatewayClientError::MissingHost
        | GatewayClientError::MissingPort
        | GatewayClientError::PeerNotLoopback { .. }
        | GatewayClientError::Build(_) => FailureClass::Permanent,
        GatewayClientError::Request(source) if source.is_timeout() => FailureClass::Transient,
        GatewayClientError::Request(_)
        | GatewayClientError::PeerConnectTimeout { .. }
        | GatewayClientError::PeerConnect { .. }
        | GatewayClientError::PeerAddress(_) => FailureClass::Transient,
    }
}

fn is_timeout(error: &GatewayClientError) -> bool {
    matches!(error, GatewayClientError::PeerConnectTimeout { .. })
        || matches!(error, GatewayClientError::Request(source) if source.is_timeout())
}

fn truncate_text(text: &str, max_tokens: usize) -> String {
    if whitespace_tokens(text) <= max_tokens {
        return text.to_string();
    }
    text.split_whitespace()
        .take(max_tokens)
        .collect::<Vec<_>>()
        .join(" ")
}

fn whitespace_tokens(text: &str) -> usize {
    text.split_whitespace().count().max(1)
}

fn task_name(task: RemoteTask) -> &'static str {
    match task {
        RemoteTask::Embed => "embed",
        RemoteTask::Rerank => "rerank",
        RemoteTask::Generate => "generate",
    }
}

#[derive(Serialize)]
pub(crate) struct RemoteEmbedVector {
    pub id: String,
    pub vector: Vec<f32>,
    pub content_sha256: String,
}
