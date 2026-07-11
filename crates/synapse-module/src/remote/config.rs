use std::{collections::BTreeSet, time::Duration};

use reqwest::Url;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use synapse_core::{Fingerprint, NumericProfileId};

use super::{classify, client::parse_loopback_host};

pub(super) const ADAPTER_VERSION: &str = "1.0.0";
pub(super) const DEFAULT_RESPONSE_BODY_LIMIT: usize = 32 * 1024 * 1024;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RemoteProviderConfig {
    pub name: String,
    pub base_url: String,
    pub adapter: AdapterConfig,
    pub auth: RemoteAuthConfig,
    pub models: Vec<RemoteModelConfig>,
    #[serde(default = "default_classifier")]
    pub classifier_preset: String,
    #[serde(default = "default_max_concurrency")]
    pub max_concurrency: usize,
    #[serde(default = "default_connect_timeout_ms")]
    pub connect_timeout_ms: u64,
    #[serde(default = "default_read_timeout_ms")]
    pub read_timeout_ms: u64,
    #[serde(default = "default_response_body_limit")]
    pub response_body_limit_bytes: usize,
    #[serde(default = "default_target_subbatch_ms")]
    pub target_subbatch_ms: u64,
    #[serde(default)]
    pub cold_estimate_ms: ColdEstimateConfig,
    #[serde(default)]
    pub breaker: BreakerOverrides,
    #[serde(default = "default_drift_gate_min")]
    pub drift_gate_min: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AdapterConfig {
    pub kind: AdapterKind,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AdapterKind {
    OpenaiCompatible,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum RemoteAuthConfig {
    #[serde(alias = "vault_handle")]
    Vault {
        handle: String,
    },
    None,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RemoteModelConfig {
    pub synapse_model_id: String,
    pub task: RemoteTask,
    #[serde(alias = "provider_model_id")]
    pub model: String,
    pub identity_revision: String,
    pub dims: usize,
    pub input_profile_id: String,
    #[serde(default = "default_max_input_tokens")]
    pub max_input_tokens: usize,
    #[serde(default = "default_sentinel_texts")]
    pub sentinel_texts: Vec<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RemoteTask {
    Embed,
    Rerank,
    Generate,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ColdEstimateConfig {
    #[serde(default = "default_embed_estimate_ms")]
    pub embed: u64,
    #[serde(default = "default_rerank_estimate_ms")]
    pub rerank: u64,
    #[serde(default = "default_generate_estimate_ms")]
    pub generate: u64,
}

impl Default for ColdEstimateConfig {
    fn default() -> Self {
        Self {
            embed: default_embed_estimate_ms(),
            rerank: default_rerank_estimate_ms(),
            generate: default_generate_estimate_ms(),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BreakerOverrides {
    #[serde(default = "default_breaker_failure_threshold")]
    pub failure_threshold: u32,
    #[serde(default = "default_breaker_cooldown_ms")]
    pub cooldown_ms: u64,
}

impl Default for BreakerOverrides {
    fn default() -> Self {
        Self {
            failure_threshold: default_breaker_failure_threshold(),
            cooldown_ms: default_breaker_cooldown_ms(),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ConfiguredProvider {
    pub name: String,
    pub base_url: Url,
    pub deployment: String,
    pub auth: ConfiguredAuth,
    pub classifier_preset: String,
    pub max_concurrency: usize,
    pub connect_timeout: Duration,
    pub read_timeout: Duration,
    pub response_body_limit_bytes: usize,
    pub target_subbatch_ms: u64,
    pub cold_embed_estimate_ms: u64,
    pub breaker_failure_threshold: u32,
    pub breaker_cooldown_ms: u64,
    pub drift_gate_min: f64,
    pub models: Vec<ConfiguredRemoteProfile>,
}

#[derive(Clone, Debug)]
pub(crate) enum ConfiguredAuth {
    Vault { handle: String },
    None,
}

impl ConfiguredAuth {
    pub fn logical_handle(&self) -> Option<&str> {
        match self {
            Self::Vault { handle } => Some(handle),
            Self::None => None,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ConfiguredRemoteProfile {
    pub provider: String,
    pub deployment: String,
    pub synapse_model_id: String,
    pub provider_model_id: String,
    pub identity_revision: String,
    pub dims: usize,
    pub input_profile_id: String,
    pub max_input_tokens: usize,
    pub sentinel_texts: Vec<String>,
    pub remote_profile_hash: String,
    pub numeric_profile_id: NumericProfileId,
    pub fingerprint: Fingerprint,
    pub drift_gate_min: f64,
    pub task: RemoteTask,
}

pub(crate) fn validate_remote_providers(
    configs: &[RemoteProviderConfig],
) -> Result<Vec<ConfiguredProvider>, String> {
    let mut provider_names = BTreeSet::new();
    let mut model_ids = BTreeSet::new();
    let mut configured = Vec::with_capacity(configs.len());
    for provider in configs {
        let name = nonempty("remote provider name", &provider.name)?;
        if !provider_names.insert(name.to_string()) {
            return Err(format!("duplicate remote provider name '{name}'"));
        }
        if provider.max_concurrency == 0
            || provider.response_body_limit_bytes == 0
            || provider.connect_timeout_ms == 0
            || provider.read_timeout_ms == 0
            || provider.target_subbatch_ms == 0
            || provider.breaker.failure_threshold == 0
            || provider.breaker.cooldown_ms == 0
        {
            return Err(format!("remote provider '{name}' has a zero pool, timeout, body limit, target, or breaker setting"));
        }
        if !provider.drift_gate_min.is_finite()
            || !(-1.0..=0.9999).contains(&provider.drift_gate_min)
        {
            return Err(format!(
                "remote provider '{name}' drift_gate_min must be finite and within [-1, 0.9999]"
            ));
        }
        if classify::preset(&provider.classifier_preset).is_none() {
            return Err(format!(
                "remote provider '{name}' has unknown classifier_preset '{}'",
                provider.classifier_preset
            ));
        }
        let mut base_url = Url::parse(&provider.base_url)
            .map_err(|error| format!("remote provider '{name}' base_url is invalid: {error}"))?;
        if !base_url.username().is_empty() || base_url.password().is_some() {
            return Err(format!(
                "remote provider '{name}' base_url must not contain userinfo"
            ));
        }
        if base_url.query().is_some() || base_url.fragment().is_some() {
            return Err(format!(
                "remote provider '{name}' base_url must not contain a query or fragment"
            ));
        }
        let host = base_url
            .host_str()
            .ok_or_else(|| format!("remote provider '{name}' base_url must contain a host"))?;
        let deployment = host.to_string();
        let auth = match &provider.auth {
            RemoteAuthConfig::None => {
                if base_url.scheme() != "http" && base_url.scheme() != "https" {
                    return Err(format!(
                        "remote provider '{name}' auth none requires an http(s) base_url"
                    ));
                }
                parse_loopback_host(host).map_err(|error| {
                    format!("remote provider '{name}' auth none requires a loopback-literal base_url: {error}")
                })?;
                ConfiguredAuth::None
            }
            RemoteAuthConfig::Vault { handle } => {
                if base_url.scheme() != "https" {
                    return Err(format!(
                        "remote provider '{name}' vault auth requires an https base_url"
                    ));
                }
                ConfiguredAuth::Vault {
                    handle: nonempty("vault handle", handle)?.to_string(),
                }
            }
        };
        base_url.set_fragment(None);
        base_url.set_query(None);
        let normalized_path = base_url.path().trim_end_matches('/').to_string();
        base_url.set_path(if normalized_path.is_empty() {
            "/"
        } else {
            &normalized_path
        });

        if provider.models.is_empty() {
            return Err(format!(
                "remote provider '{name}' must declare at least one model"
            ));
        }
        let mut models = Vec::with_capacity(provider.models.len());
        for model in &provider.models {
            let model_id = nonempty("synapse_model_id", &model.synapse_model_id)?;
            if !model_ids.insert(model_id.to_string()) {
                return Err(format!("duplicate remote synapse_model_id '{model_id}'"));
            }
            if model.dims == 0 || model.max_input_tokens == 0 {
                return Err(format!(
                    "remote model '{model_id}' dims and max_input_tokens must be non-zero"
                ));
            }
            if model.max_input_tokens > 1_000_000 {
                return Err(format!(
                    "remote model '{model_id}' max_input_tokens exceeds the 1000000-token config limit"
                ));
            }
            let provider_model_id = nonempty("provider model id", &model.model)?;
            let identity_revision = nonempty("identity_revision", &model.identity_revision)?;
            let input_profile_id = nonempty("input_profile_id", &model.input_profile_id)?;
            if model.sentinel_texts.is_empty()
                || model.sentinel_texts.iter().any(|text| text.is_empty())
            {
                return Err(format!(
                    "remote model '{model_id}' must have non-empty sentinel_texts"
                ));
            }
            let identity = json!({
                "adapter_kind": "openai_compatible",
                "adapter_semver_major": 1,
                "provider_deployment_id": deployment,
                "provider_model_id": provider_model_id,
                "identity_revision": identity_revision,
                "task": remote_task_name(model.task),
                "dims": model.dims,
                "input_profile_id": input_profile_id,
            });
            let canonical = canonical_json(&identity);
            let mut bytes = b"remote-profile-v1".to_vec();
            bytes.extend_from_slice(&canonical);
            let remote_profile_hash = hex::encode(Sha256::digest(&bytes));
            models.push(ConfiguredRemoteProfile {
                provider: name.to_string(),
                deployment: deployment.clone(),
                synapse_model_id: model_id.to_string(),
                provider_model_id: provider_model_id.to_string(),
                identity_revision: identity_revision.to_string(),
                dims: model.dims,
                input_profile_id: input_profile_id.to_string(),
                max_input_tokens: model.max_input_tokens,
                sentinel_texts: model.sentinel_texts.clone(),
                numeric_profile_id: NumericProfileId(remote_profile_hash.clone()),
                fingerprint: Fingerprint(remote_profile_hash.clone()),
                remote_profile_hash,
                drift_gate_min: provider.drift_gate_min,
                task: model.task,
            });
        }
        configured.push(ConfiguredProvider {
            name: name.to_string(),
            base_url,
            deployment,
            auth,
            classifier_preset: provider.classifier_preset.clone(),
            max_concurrency: provider.max_concurrency,
            connect_timeout: Duration::from_millis(provider.connect_timeout_ms),
            read_timeout: Duration::from_millis(provider.read_timeout_ms),
            response_body_limit_bytes: provider.response_body_limit_bytes,
            target_subbatch_ms: provider.target_subbatch_ms,
            cold_embed_estimate_ms: provider.cold_estimate_ms.embed,
            breaker_failure_threshold: provider.breaker.failure_threshold,
            breaker_cooldown_ms: provider.breaker.cooldown_ms,
            drift_gate_min: provider.drift_gate_min,
            models,
        });
    }
    Ok(configured)
}

fn canonical_json(value: &Value) -> Vec<u8> {
    match value {
        Value::Object(map) => {
            let sorted = map
                .iter()
                .map(|(key, value)| {
                    (
                        key.clone(),
                        serde_json::from_slice::<Value>(&canonical_json(value))
                            .expect("canonical child json"),
                    )
                })
                .collect::<std::collections::BTreeMap<_, _>>();
            serde_json::to_vec(&sorted).expect("remote identity serializes")
        }
        _ => serde_json::to_vec(value).expect("remote identity value serializes"),
    }
}

fn nonempty<'a>(label: &str, value: &'a str) -> Result<&'a str, String> {
    let value = value.trim();
    if value.is_empty() {
        Err(format!("{label} must not be empty"))
    } else {
        Ok(value)
    }
}

fn remote_task_name(task: RemoteTask) -> &'static str {
    match task {
        RemoteTask::Embed => "embed",
        RemoteTask::Rerank => "rerank",
        RemoteTask::Generate => "generate",
    }
}

fn default_classifier() -> String {
    "generic".to_string()
}
fn default_max_concurrency() -> usize {
    2
}
fn default_connect_timeout_ms() -> u64 {
    5_000
}
fn default_read_timeout_ms() -> u64 {
    30_000
}
fn default_response_body_limit() -> usize {
    DEFAULT_RESPONSE_BODY_LIMIT
}
fn default_target_subbatch_ms() -> u64 {
    10_000
}
fn default_embed_estimate_ms() -> u64 {
    15_000
}
fn default_rerank_estimate_ms() -> u64 {
    15_000
}
fn default_generate_estimate_ms() -> u64 {
    30_000
}
fn default_breaker_failure_threshold() -> u32 {
    3
}
fn default_breaker_cooldown_ms() -> u64 {
    30_000
}
fn default_drift_gate_min() -> f64 {
    0.95
}
fn default_max_input_tokens() -> usize {
    8_192
}
fn default_sentinel_texts() -> Vec<String> {
    vec![
        "Synapse remote identity sentinel alpha.".to_string(),
        "Synapse remote identity sentinel beta.".to_string(),
        "Synapse remote identity sentinel gamma.".to_string(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider(base_url: &str, auth: RemoteAuthConfig) -> RemoteProviderConfig {
        RemoteProviderConfig {
            name: "mock".to_string(),
            base_url: base_url.to_string(),
            adapter: AdapterConfig {
                kind: AdapterKind::OpenaiCompatible,
            },
            auth,
            models: vec![RemoteModelConfig {
                synapse_model_id: "remote-embed".to_string(),
                task: RemoteTask::Embed,
                model: "provider-model".to_string(),
                identity_revision: "r1".to_string(),
                dims: 3,
                input_profile_id: "whitespace-v1".to_string(),
                max_input_tokens: 128,
                sentinel_texts: default_sentinel_texts(),
            }],
            classifier_preset: default_classifier(),
            max_concurrency: 2,
            connect_timeout_ms: 100,
            read_timeout_ms: 100,
            response_body_limit_bytes: 1024,
            target_subbatch_ms: 100,
            cold_estimate_ms: ColdEstimateConfig::default(),
            breaker: BreakerOverrides::default(),
            drift_gate_min: 0.95,
        }
    }

    #[test]
    fn auth_none_is_rejected_for_non_loopback_at_config_parse_time() {
        let error =
            validate_remote_providers(&[provider("http://example.com/v1", RemoteAuthConfig::None)])
                .unwrap_err();
        assert!(error.contains("loopback-literal"));
    }

    #[test]
    fn remote_profile_hash_changes_with_identity_revision_but_not_provider_name() {
        let first = validate_remote_providers(&[provider(
            "http://127.0.0.1:1234/v1",
            RemoteAuthConfig::None,
        )])
        .unwrap();
        let mut second_config = provider("http://127.0.0.1:1234/v1", RemoteAuthConfig::None);
        second_config.name = "renamed".to_string();
        let second = validate_remote_providers(&[second_config]).unwrap();
        assert_eq!(
            first[0].models[0].remote_profile_hash,
            second[0].models[0].remote_profile_hash
        );
    }
}
