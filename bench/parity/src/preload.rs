//! Preload regression inputs, `preload/<model_id>.json`.
//!
//! Each file records the inputs that `build_stored_model_config` in
//! `crates/synapse-module` turns into the stored config of a lane already
//! serving consumers through preload config, and the full fingerprint that
//! lane has in production. `fingerprint` here recomputes the lane's numeric
//! profile and fingerprint independently from the same inputs, so a wrong
//! committed input fails here without depending on any test in the module.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::canonical::{is_sha256_hex, sha256_hex};
use crate::{perr, Result};

/// The two preload-configured lanes the regression covers.
pub const PRELOAD_MODEL_IDS: [&str; 2] = [
    "gte-modernbert-base-f16",
    "gte-reranker-modernbert-base-f32",
];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreloadInputs {
    pub model_id: String,
    pub engine: String,
    pub task: String,
    pub artifact_digest: String,
    pub sanitized_tokenizer_digest: String,
    pub quant: String,
    pub pooling: String,
    pub normalize: bool,
    pub max_tokens: u32,
    pub owned_family: String,
    pub owned_dtype: String,
    pub inline: BTreeMap<String, u64>,
    pub jobs: BTreeMap<String, u64>,
    /// Values read back from the production module's stored config when the
    /// fingerprint was captured; the engine identity is derived by the module
    /// from `owned_family` and `owned_dtype`.
    pub captured: Captured,
    /// Optional so a missing value is reported as a failure by name, rather
    /// than as a parse error.
    #[serde(default)]
    pub expected_fingerprint: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Captured {
    pub source: String,
    pub engine_identity: EngineIdentity,
    pub numeric_profile_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineIdentity {
    pub engine: String,
    pub version: String,
    pub build_flags: BTreeMap<String, String>,
}

/// Field-for-field mirror of `synapse_core::fingerprint::NumericProfile` for
/// a preload-configured owned-Metal lane. Field order matters: the module
/// hashes the struct's serde JSON, which follows declaration order.
#[derive(Serialize)]
struct NumericProfileMirror<'a> {
    model_digest: &'a str,
    quant: &'a str,
    engine: &'a EngineIdentity,
    sanitized_tokenizer_digest: &'a str,
    pooling: &'a str,
    normalization: &'a str,
    dtype: &'a str,
    flash_attention: &'a str,
    certified_shape: CertifiedShape,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_template: Option<&'a str>,
    thread_policy: &'a str,
}

#[derive(Serialize)]
struct CertifiedShape {
    max_context_tokens: u64,
    max_batch_tokens: u64,
    max_micro_batch_tokens: u64,
    max_sequences: u64,
}

fn inline_value(inputs: &PreloadInputs, section: &BTreeMap<String, u64>, key: &str) -> Result<u64> {
    section
        .get(key)
        .copied()
        .ok_or_else(|| perr!("{}: inputs lack `{key}`", inputs.model_id))
}

/// Numeric profile id and fingerprint from the committed inputs.
pub fn fingerprint(inputs: &PreloadInputs) -> Result<(String, String)> {
    let profile = NumericProfileMirror {
        model_digest: &inputs.artifact_digest,
        quant: &inputs.quant,
        engine: &inputs.captured.engine_identity,
        sanitized_tokenizer_digest: &inputs.sanitized_tokenizer_digest,
        pooling: &inputs.pooling,
        normalization: if inputs.normalize { "l2" } else { "none" },
        dtype: &inputs.owned_dtype,
        flash_attention: "disabled",
        certified_shape: CertifiedShape {
            max_context_tokens: u64::from(inputs.max_tokens),
            max_batch_tokens: inline_value(inputs, &inputs.inline, "max_tokens")?,
            max_micro_batch_tokens: inline_value(inputs, &inputs.jobs, "bulk_quantum_tokens")?,
            max_sequences: inline_value(inputs, &inputs.inline, "max_items")?,
        },
        prompt_template: match inputs.task.as_str() {
            "embed" => None,
            "rerank" => Some("synapse-rerank-bos-query-sep-doc-eos-v1"),
            other => {
                return Err(perr!(
                    "{}: task `{other}` is not embed or rerank",
                    inputs.model_id
                ))
            }
        },
        thread_policy: "balanced",
    };
    let profile_id = sha256_hex(&serde_json::to_vec(&profile).expect("profile serializes"));
    let payload = serde_json::to_vec(&serde_json::json!([
        inputs.artifact_digest,
        inputs.quant,
        profile_id
    ]))
    .expect("payload serializes");
    Ok((profile_id, sha256_hex(&payload)))
}

/// Fails, by name, when `expected_fingerprint` is absent or malformed, or
/// when the recomputed fingerprint or profile id differs from what was
/// captured.
pub fn check_preload(inputs: &PreloadInputs) -> Result<()> {
    let expected = inputs.expected_fingerprint.as_deref().ok_or_else(|| {
        perr!(
            "{}: expected_fingerprint is absent; the regression cannot pass without it",
            inputs.model_id
        )
    })?;
    if !is_sha256_hex(expected) {
        return Err(perr!(
            "{}: expected_fingerprint is not a full SHA-256",
            inputs.model_id
        ));
    }
    let (profile_id, fingerprint) = fingerprint(inputs)?;
    if profile_id != inputs.captured.numeric_profile_id {
        return Err(perr!(
            "{}: numeric profile id {profile_id} differs from the captured {}",
            inputs.model_id,
            inputs.captured.numeric_profile_id
        ));
    }
    if fingerprint != expected {
        return Err(perr!(
            "{}: fingerprint {fingerprint} differs from expected {expected}",
            inputs.model_id
        ));
    }
    Ok(())
}
