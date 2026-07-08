use serde::{Deserialize, Serialize};

use crate::{EngineIdentity, Fingerprint};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TruncationDisclosure {
    pub submitted_tokens: u32,
    pub effective_tokens: u32,
    pub truncated: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseProvenance {
    pub engine: EngineIdentity,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponseEnvelope<T> {
    pub fingerprint: Fingerprint,
    pub table_epoch: u64,
    pub dims: u32,
    pub provenance: ResponseProvenance,
    pub module_generation: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub equivalent_to: Vec<Fingerprint>,
    pub result: T,
}
