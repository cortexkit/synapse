//! Length-prefixed JSON protocol shared by the measurement rig and candidates.

use std::io::{self, Read, Write};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

pub const PROTOCOL_VERSION: u32 = 1;
pub const MAX_FRAME_BYTES: u32 = 256 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ShapePolicy {
    Exact,
    Bucketed,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Workload {
    Embedding,
    Rerank,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BatchShape {
    pub batch: usize,
    pub seq: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RerankPair {
    pub query: String,
    pub document: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CandidateRequest {
    PrepareShapes {
        workload: Workload,
        shapes: Vec<BatchShape>,
        max_length: usize,
        force_shapes: bool,
    },
    Embed {
        texts: Vec<String>,
        max_length: usize,
        shape_policy: ShapePolicy,
        shape: BatchShape,
    },
    Rerank {
        pairs: Vec<RerankPair>,
        max_length: usize,
        shape_policy: ShapePolicy,
        shape: BatchShape,
    },
    Shutdown,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CandidateMetadata {
    pub lane: String,
    pub model: String,
    pub provider: String,
    pub dtype: String,
    pub execution: String,
    pub notes: String,
    pub package_cache_root: Option<String>,
    pub internal_load_s: f64,
    pub eager_shape_preload: bool,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CandidateResponse {
    Ready {
        protocol_version: u32,
        metadata: CandidateMetadata,
    },
    Prepared {
        internal_wall_s: f64,
    },
    Embedding {
        vectors: Vec<Vec<f32>>,
        reported_real_tokens: u64,
        internal_infer_wall_s: f64,
    },
    Rerank {
        scores: Vec<f32>,
        reported_real_tokens: u64,
        internal_infer_wall_s: f64,
    },
    Shutdown,
    Error {
        message: String,
    },
}

pub fn read_json_frame<R: Read, T: DeserializeOwned>(reader: &mut R) -> io::Result<T> {
    let mut len_bytes = [0_u8; 4];
    reader.read_exact(&mut len_bytes)?;
    let len = u32::from_le_bytes(len_bytes);
    if len > MAX_FRAME_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("frame length {len} exceeds max {MAX_FRAME_BYTES}"),
        ));
    }
    let mut frame = vec![0_u8; len as usize];
    reader.read_exact(&mut frame)?;
    serde_json::from_slice(&frame).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("rig protocol JSON decode: {error}"),
        )
    })
}

pub fn write_json_frame<W: Write, T: Serialize>(writer: &mut W, value: &T) -> io::Result<()> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("rig protocol JSON encode: {error}"),
        )
    })?;
    let len = u32::try_from(bytes.len()).map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidData, "frame too large for u32 length")
    })?;
    if len > MAX_FRAME_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("frame length {len} exceeds max {MAX_FRAME_BYTES}"),
        ));
    }
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(&bytes)?;
    writer.flush()
}
