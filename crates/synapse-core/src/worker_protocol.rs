use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::EngineIdentity;

pub const WORKER_PROTOCOL_VERSION: u8 = 1;
pub const DEFAULT_MAX_FRAME_BYTES: u32 = 64 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerHello {
    pub v: u8,
    pub nonce: String,
    pub engine: EngineIdentity,
    pub pid: u32,
    pub max_frame: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerHelloAck {
    pub v: u8,
    pub accept: bool,
    pub max_frame: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerPooling {
    Mean,
    Cls,
    Last,
}

impl WorkerPooling {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "mean" => Some(Self::Mean),
            "cls" => Some(Self::Cls),
            "last" => Some(Self::Last),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Cls => "cls",
            Self::Last => "last",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerTokenItem {
    pub id: String,
    pub n_tokens: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerCandidate {
    pub n_tokens: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum WorkerRequest {
    Load {
        req_id: String,
        artifact_path: String,
        artifact_digest: String,
        format: String,
        #[serde(default)]
        runtime_config: BTreeMap<String, String>,
    },
    EmbedBatch {
        req_id: String,
        model_ref: String,
        pooling: WorkerPooling,
        normalize: bool,
        items: Vec<WorkerTokenItem>,
    },
    Rerank {
        req_id: String,
        model_ref: String,
        query_n_tokens: usize,
        candidates: Vec<WorkerCandidate>,
    },
    Generate {
        req_id: String,
        model_ref: String,
        max_tokens: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        grammar: Option<String>,
    },
    Unload {
        req_id: String,
        model_ref: String,
    },
    Ping {
        req_id: String,
    },
    Shutdown {},
}

impl WorkerRequest {
    pub fn req_id(&self) -> Option<&str> {
        match self {
            Self::Load { req_id, .. }
            | Self::EmbedBatch { req_id, .. }
            | Self::Rerank { req_id, .. }
            | Self::Generate { req_id, .. }
            | Self::Unload { req_id, .. }
            | Self::Ping { req_id } => Some(req_id),
            Self::Shutdown {} => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum WorkerResponse {
    Loaded {
        req_id: String,
        model_ref: String,
        dims: usize,
        cold_load_ms: u64,
    },
    Vectors {
        req_id: String,
        dims: usize,
        n: usize,
    },
    Scores {
        req_id: String,
    },
    Text {
        req_id: String,
        text: String,
        n_prompt: usize,
        n_gen: usize,
        finish_reason: String,
    },
    Unloaded {
        req_id: String,
    },
    Pong {
        req_id: String,
        rss_mb: u64,
        models_loaded: usize,
    },
    Err {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        req_id: Option<String>,
        code: String,
        msg: String,
    },
}

impl WorkerResponse {
    pub fn req_id(&self) -> Option<&str> {
        match self {
            Self::Loaded { req_id, .. }
            | Self::Vectors { req_id, .. }
            | Self::Scores { req_id }
            | Self::Text { req_id, .. }
            | Self::Unloaded { req_id }
            | Self::Pong { req_id, .. } => Some(req_id),
            Self::Err { req_id, .. } => req_id.as_deref(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawFrameError {
    message: String,
}

impl RawFrameError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for RawFrameError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(formatter)
    }
}

impl std::error::Error for RawFrameError {}

pub fn encode_i32_frame(values: &[i32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

pub fn decode_i32_frame(bytes: &[u8]) -> Result<Vec<i32>, RawFrameError> {
    let chunks = bytes.chunks_exact(std::mem::size_of::<i32>());
    if !chunks.remainder().is_empty() {
        return Err(RawFrameError::new("i32 frame length is not divisible by 4"));
    }
    Ok(chunks
        .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("chunk has four bytes")))
        .collect())
}

pub fn encode_f32_frame(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

pub fn decode_f32_frame(bytes: &[u8]) -> Result<Vec<f32>, RawFrameError> {
    let chunks = bytes.chunks_exact(std::mem::size_of::<f32>());
    if !chunks.remainder().is_empty() {
        return Err(RawFrameError::new("f32 frame length is not divisible by 4"));
    }
    Ok(chunks
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk has four bytes")))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_i32_frames_round_trip() {
        let values = [-1, 0, 42, i32::MAX];
        assert_eq!(
            decode_i32_frame(&encode_i32_frame(&values)).unwrap(),
            values
        );
    }

    #[test]
    fn raw_f32_frames_round_trip() {
        let values = [-1.25, 0.0, 42.5, f32::INFINITY];
        assert_eq!(
            decode_f32_frame(&encode_f32_frame(&values)).unwrap(),
            values
        );
    }
}
