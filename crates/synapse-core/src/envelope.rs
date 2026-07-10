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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseProvenance {
    pub engine: EngineIdentity,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remote: Option<RemoteProvenance>,
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
    fn remote_provenance_is_an_optional_engine_sibling_with_object_build_flags() {
        let local = ResponseProvenance {
            engine: EngineIdentity {
                engine: "ort".to_string(),
                version: "1".to_string(),
                build_flags: BTreeMap::new(),
            },
            remote: None,
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
