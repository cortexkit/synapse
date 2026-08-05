use serde::{Deserialize, Serialize};

use crate::{EngineIdentity, Fingerprint};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TruncationDisclosure {
    pub submitted_tokens: u32,
    pub effective_tokens: u32,
    pub truncated: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteProvenance {
    pub provider: String,
    pub deployment: String,
    pub assurance: String,
}

/// Additive owned-decode provenance. Flattening this record keeps every legacy
/// llama envelope byte-for-byte compatible when all fields are absent, while an
/// owned selection or fallback can publish the lane identities and retry trail.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OwnedDecodeResponseProvenance {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lane: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub risk_class: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_fingerprint: Option<Fingerprint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub processing_fingerprint: Option<Fingerprint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lane_finish_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker_generation: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_completed_quantum_sequence: Option<u32>,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub crash_retry_count: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub failure_classifications: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constraint_runtime_identity: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constraint_fingerprint: Option<Fingerprint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grammar_compiler_revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub underlying_owned_decode_refusal_id: Option<String>,
}

fn is_zero(value: &u32) -> bool {
    *value == 0
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseProvenance {
    pub engine: EngineIdentity,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remote: Option<RemoteProvenance>,
    #[serde(flatten)]
    pub owned_decode: OwnedDecodeResponseProvenance,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponseEnvelope<T> {
    pub fingerprint: Fingerprint,
    pub table_epoch: u64,
    pub dims: u32,
    pub provenance: ResponseProvenance,
    pub module_generation: u64,
    #[serde(default)]
    pub equivalent_to: Vec<Fingerprint>,
    #[serde(flatten)]
    pub payload: T,
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::json;

    use super::*;

    #[test]
    fn owned_decode_provenance_is_additive_and_legacy_local_shape_stays_unchanged() {
        let provenance = ResponseProvenance {
            engine: EngineIdentity {
                engine: "llama.cpp-worker".to_string(),
                version: "1".to_string(),
                build_flags: BTreeMap::new(),
            },
            remote: None,
            owned_decode: OwnedDecodeResponseProvenance {
                lane: Some("llama".to_string()),
                worker: Some("supervised".to_string()),
                decode_fingerprint: Some(Fingerprint("llama-decode".to_string())),
                processing_fingerprint: Some(Fingerprint("llama-processing".to_string())),
                fallback_reason: Some("owned_decode_not_certified".to_string()),
                ..OwnedDecodeResponseProvenance::default()
            },
        };
        let value = serde_json::to_value(provenance).unwrap();
        assert_eq!(value["lane"], "llama");
        assert_eq!(value["fallback_reason"], "owned_decode_not_certified");
        assert_eq!(value["decode_fingerprint"], "llama-decode");
        assert!(value.get("crash_retry_count").is_none());
        assert!(
            serde_json::from_value::<OwnedDecodeResponseProvenance>(json!({
                "lane": "llama",
                "unknown": true
            }))
            .is_err()
        );
    }

    #[test]
    fn remote_provenance_is_an_optional_engine_sibling_with_object_build_flags() {
        let local = ResponseProvenance {
            engine: EngineIdentity {
                engine: "ort".to_string(),
                version: "1".to_string(),
                build_flags: BTreeMap::new(),
            },
            remote: None,
            owned_decode: OwnedDecodeResponseProvenance::default(),
        };
        let local_json = serde_json::to_value(&local).unwrap();
        assert!(local_json.get("remote").is_none());
        assert!(local_json["engine"].get("build_flags").is_none());

        let remote = ResponseProvenance {
            engine: EngineIdentity {
                engine: "remote_openai_compatible".to_string(),
                version: "1.1.0".to_string(),
                build_flags: BTreeMap::from([("transport".to_string(), "rustls".to_string())]),
            },
            remote: Some(RemoteProvenance {
                provider: "openai".to_string(),
                deployment: "api.openai.com".to_string(),
                assurance: "declared".to_string(),
            }),
            owned_decode: OwnedDecodeResponseProvenance::default(),
        };
        assert_eq!(
            serde_json::to_value(remote).unwrap(),
            json!({
                "engine": {
                    "engine": "remote_openai_compatible",
                    "version": "1.1.0",
                    "build_flags": { "transport": "rustls" },
                },
                "remote": {
                    "provider": "openai",
                    "deployment": "api.openai.com",
                    "assurance": "declared",
                }
            })
        );
    }
}
