use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::worker_engine_names::{
    ANE_DIRECT_WORKER_ENGINE, CUDA_WORKER_ENGINE, VULKAN_WORKER_ENGINE,
};
use crate::EngineIdentity;

/// Version every worker HELLO carries in `v`; the host accepts only an exact
/// match. Version 4 adds transcription requests and observation frames.
/// synapse-module and workers deploy together: an older worker is rejected at
/// HELLO before receiving a request whose raw frame it cannot recognize.
///
/// The owned-decode envelope version (the separate top-level HELLO
/// `protocol_version`) is independent of this constant.
pub const WORKER_PROTOCOL_VERSION: u8 = 4;
pub const DEFAULT_MAX_FRAME_BYTES: u32 = 64 * 1024 * 1024;

/// HELLO engine identities of the owned embed/rerank workers (CUDA, Vulkan and
/// direct ANE). Only these are bound to a model manifest and kernel revision;
/// llama, owned decode and Core ML workers omit both HELLO fields.
pub const OWNED_WORKER_HELLO_ENGINES: [&str; 3] = [
    CUDA_WORKER_ENGINE,
    VULKAN_WORKER_ENGINE,
    ANE_DIRECT_WORKER_ENGINE,
];

pub fn is_owned_worker_hello_engine(engine: &str) -> bool {
    OWNED_WORKER_HELLO_ENGINES.contains(&engine)
}

/// Worker `ERR` codes and HELLO refusal codes the host reports by name.
pub const ERR_UNSUPPORTED_REQUEST: &str = "unsupported_request";
pub const ERR_MANIFEST_MISMATCH: &str = "manifest_mismatch";
pub const ERR_KERNEL_REVISION_MISMATCH: &str = "kernel_revision_mismatch";
pub const ERR_MODEL_UNSUPPORTED: &str = "model_unsupported";
pub const ERR_OPERATION_MISMATCH: &str = "operation_mismatch";
pub const ERR_PACKAGE_DIGEST_MISMATCH: &str = "package_digest_mismatch";
pub const ERR_HEAD_TENSOR_MISSING: &str = "head_tensor_missing";
pub const ERR_SHAPE_NOT_ADMITTED: &str = "shape_not_admitted";
pub const ERR_ANE_LANE_BUSY: &str = "ane_lane_busy";

/// `LOAD.runtime_config` key naming the manifest profile id
/// (`<slug>.<lane identity>`) an owned worker must load.
pub const LOAD_RUNTIME_PROFILE: &str = "profile";
/// `LOAD.runtime_config` key naming the operation (`embed` or `rerank`) the
/// owned worker checks against its embedded manifest before reading weights.
pub const LOAD_RUNTIME_OPERATION: &str = "operation";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerHello {
    pub v: u8,
    pub nonce: String,
    pub engine: EngineIdentity,
    pub pid: u32,
    pub max_frame: u32,
    /// Lowercase hex SHA-256 of the canonical model-manifest bytes the worker
    /// was built with. Sent by owned workers; omitted by the others.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest_digest: Option<String>,
    /// Kernel or shader revision the worker executes. Sent by owned workers;
    /// omitted by the others.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel_revision: Option<String>,
    /// Optional packed-execution capability, outside the hashed EngineIdentity.
    /// Legacy v2 peers omit it and must receive standalone requests only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packed: Option<PackedCapability>,
}

pub const PACKED_WIDTH: usize = 256;
pub const PACKED_PROGRAM_REVISION: u32 = 1;

/// Version-one fixtures compare packed embedding output with standalone output.
/// Rows end at and cross the 128-column attention tile boundary; repeated and
/// reordered rows test scattering, and a 257-token row tests standalone fallback.
pub fn packed_qualification_fixtures_v1() -> Vec<Vec<Vec<u32>>> {
    let row = |len: usize, seed: u32| {
        (0..len)
            .map(|i| 16 + (seed + i as u32 * 7) % 1000)
            .collect::<Vec<_>>()
    };
    vec![
        vec![row(127, 1), row(129, 2), row(1, 3), row(255, 4)],
        vec![
            row(128, 5),
            row(128, 6),
            row(129, 7),
            row(126, 8),
            row(1, 9),
        ],
        vec![
            row(127, 1),
            row(257, 10),
            row(127, 1),
            row(129, 2),
            row(1, 3),
            row(256, 11),
        ],
    ]
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackedCapability {
    pub width: usize,
    pub program_revision: u32,
}

impl PackedCapability {
    pub fn tight256() -> Self {
        Self {
            width: PACKED_WIDTH,
            program_revision: PACKED_PROGRAM_REVISION,
        }
    }
}

/// Indices are in the request's original row order; each inner list is one pass.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackedPlan {
    pub passes: Vec<Vec<usize>>,
    pub singles: Vec<usize>,
}

/// Portable first-fit-decreasing planning, with original index breaking ties.
/// A short row alone gains nothing from packing and stays at its natural rung.
pub fn plan_packed(lengths: &[usize]) -> Result<PackedPlan, RawFrameError> {
    if lengths.contains(&0) {
        return Err(RawFrameError::new("empty_row"));
    }
    let mut order: Vec<_> = (0..lengths.len()).collect();
    order.sort_by_key(|&i| (std::cmp::Reverse(lengths[i]), i));
    let mut plan = PackedPlan::default();
    let mut used = Vec::<usize>::new();
    for index in order {
        if lengths[index] > PACKED_WIDTH {
            plan.singles.push(index);
        } else if let Some(pass) = used
            .iter()
            .position(|&n| n + lengths[index] <= PACKED_WIDTH)
        {
            used[pass] += lengths[index];
            plan.passes[pass].push(index);
        } else {
            used.push(lengths[index]);
            plan.passes.push(vec![index]);
        }
    }
    plan.passes.retain(|pass| {
        if pass.len() == 1 && lengths[pass[0]] <= 128 {
            plan.singles.push(pass[0]);
            false
        } else {
            true
        }
    });
    plan.singles.sort_unstable();
    Ok(plan)
}

/// Validate all pass boundaries before any executable is dispatched. Execution
/// requests contain packed rows only, so every row must occur exactly once.
pub fn validate_packed_passes(
    lengths: &[usize],
    passes: &[Vec<usize>],
) -> Result<(), RawFrameError> {
    if lengths.is_empty() || lengths.contains(&0) || passes.is_empty() {
        return Err(RawFrameError::new("empty_packed_plan"));
    }
    let mut seen = vec![false; lengths.len()];
    for pass in passes {
        if pass.is_empty() {
            return Err(RawFrameError::new("empty_pass"));
        }
        let mut used = 0usize;
        for &index in pass {
            let length = lengths
                .get(index)
                .ok_or_else(|| RawFrameError::new("invalid_row_index"))?;
            if seen[index] {
                return Err(RawFrameError::new("duplicate_row"));
            }
            seen[index] = true;
            used = used
                .checked_add(*length)
                .ok_or_else(|| RawFrameError::new("packed_length_overflow"))?;
        }
        if used > PACKED_WIDTH {
            return Err(RawFrameError::new("packed_rows_too_wide"));
        }
    }
    if seen.contains(&false) {
        return Err(RawFrameError::new("missing_row"));
    }
    Ok(())
}

/// Real queries attend causally inside their segment. Padding queries attend
/// themselves to keep softmax finite. Rotary positions restart in each segment.
pub fn packed_mask_positions(
    lengths: &[usize],
    width: usize,
) -> Result<(Vec<f32>, Vec<usize>), RawFrameError> {
    let total = lengths
        .iter()
        .try_fold(0usize, |n, &v| n.checked_add(v))
        .ok_or_else(|| RawFrameError::new("packed_length_overflow"))?;
    if lengths.is_empty() || lengths.contains(&0) || total > width || width > 512 {
        return Err(RawFrameError::new("invalid_segment_lengths"));
    }
    let mut mask = vec![-10_000.0; width * width];
    let mut positions = vec![0; width];
    let mut start = 0;
    for &length in lengths {
        for q in start..start + length {
            positions[q] = q - start;
            for k in start..=q {
                mask[q * width + k] = 0.0;
            }
        }
        start += length;
    }
    for q in total..width {
        mask[q * width + q] = 0.0;
    }
    Ok((mask, positions))
}

/// The manifest digest and kernel revision a host requires from an owned
/// worker's HELLO. Hosts pass one only for lanes bound to the model manifest;
/// without one, the HELLO fields are not checked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpectedHelloBinding {
    pub manifest_digest: String,
    pub kernel_revision: String,
}

/// An owned worker's HELLO named a different (or no) manifest digest or kernel
/// revision than the host requires. `code` is [`ERR_MANIFEST_MISMATCH`] or
/// [`ERR_KERNEL_REVISION_MISMATCH`].
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{code}: worker {engine} HELLO {field} is {advertised:?}, expected {expected}")]
pub struct HelloBindingMismatch {
    pub code: &'static str,
    pub engine: String,
    pub field: &'static str,
    pub expected: String,
    pub advertised: Option<String>,
}

/// Checks an owned worker's HELLO against the binding the host requires.
///
/// With no expected binding nothing is checked, so lanes that are not bound to
/// the manifest keep accepting workers that omit both fields. With one, an
/// owned worker (see [`OWNED_WORKER_HELLO_ENGINES`]) must send exactly the
/// expected manifest digest, then exactly the expected kernel revision; a
/// missing field is refused like a different one. Other workers are never
/// bound and pass with both fields absent.
pub fn check_hello_binding(
    hello: &WorkerHello,
    expected: Option<&ExpectedHelloBinding>,
) -> Result<(), HelloBindingMismatch> {
    let Some(expected) = expected else {
        return Ok(());
    };
    if !is_owned_worker_hello_engine(&hello.engine.engine) {
        return Ok(());
    }
    let checks = [
        (
            ERR_MANIFEST_MISMATCH,
            "manifest_digest",
            &expected.manifest_digest,
            &hello.manifest_digest,
        ),
        (
            ERR_KERNEL_REVISION_MISMATCH,
            "kernel_revision",
            &expected.kernel_revision,
            &hello.kernel_revision,
        ),
    ];
    for (code, field, expected, advertised) in checks {
        if advertised.as_deref() != Some(expected.as_str()) {
            return Err(HelloBindingMismatch {
                code,
                engine: hello.engine.engine.clone(),
                field,
                expected: expected.clone(),
                advertised: advertised.clone(),
            });
        }
    }
    Ok(())
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

/// One fully composed rerank input (query, candidate and every special or
/// template token) in a `RERANK_SEQUENCES` request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerSequence {
    pub n_tokens: usize,
}

/// Where a direct-ANE worker placed a model at one shape: which compiled
/// executables run which transformer layers, and which CPU stages it uses.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnePlacementInventory {
    pub executables: Vec<AneExecutable>,
    pub cpu_stages: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AneExecutable {
    pub id: String,
    pub layers: Vec<u32>,
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
    /// One raw i32 frame in original row order, with explicit pass boundaries.
    EmbedPacked {
        req_id: String,
        model_ref: String,
        program_revision: u32,
        items: Vec<WorkerTokenItem>,
        passes: Vec<Vec<usize>>,
    },
    AneAdmitPacked {
        req_id: String,
        model_ref: String,
        program_revision: u32,
    },
    AneEvictPacked {
        req_id: String,
        model_ref: String,
        program_revision: u32,
    },
    Rerank {
        req_id: String,
        model_ref: String,
        query_n_tokens: usize,
        candidates: Vec<WorkerCandidate>,
    },
    /// Owned-worker rerank: followed by one raw i32 frame holding the composed
    /// sequences concatenated in candidate order. The worker adds no token and
    /// answers `SCORES` plus one raw f32 frame in the same order.
    RerankSequences {
        req_id: String,
        model_ref: String,
        sequences: Vec<WorkerSequence>,
    },
    /// Direct ANE: compile and make `shape` resident for `model_ref`; answered
    /// by `ADMITTED`. No raw frame follows.
    AneAdmitShape {
        req_id: String,
        model_ref: String,
        shape: usize,
    },
    /// Direct ANE: release the executables of a resident shape; answered by
    /// `EVICTED`. No raw frame follows.
    AneEvictShape {
        req_id: String,
        model_ref: String,
        shape: usize,
    },
    Generate {
        req_id: String,
        model_ref: String,
        max_tokens: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        grammar: Option<String>,
    },
    /// One raw little-endian f32 frame, row-major with 128 mel bins per frame.
    Transcribe {
        req_id: String,
        model_ref: String,
        mel_frames: usize,
        valid_mel_frames: usize,
        observe: bool,
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
    /// Wire `type` tag of this request.
    pub fn wire_type(&self) -> &'static str {
        match self {
            Self::Load { .. } => "LOAD",
            Self::EmbedBatch { .. } => "EMBED_BATCH",
            Self::EmbedPacked { .. } => "EMBED_PACKED",
            Self::AneAdmitPacked { .. } => "ANE_ADMIT_PACKED",
            Self::AneEvictPacked { .. } => "ANE_EVICT_PACKED",
            Self::Rerank { .. } => "RERANK",
            Self::RerankSequences { .. } => "RERANK_SEQUENCES",
            Self::AneAdmitShape { .. } => "ANE_ADMIT_SHAPE",
            Self::AneEvictShape { .. } => "ANE_EVICT_SHAPE",
            Self::Generate { .. } => "GENERATE",
            Self::Transcribe { .. } => "TRANSCRIBE",
            Self::Unload { .. } => "UNLOAD",
            Self::Ping { .. } => "PING",
            Self::Shutdown {} => "SHUTDOWN",
        }
    }

    pub fn req_id(&self) -> Option<&str> {
        match self {
            Self::Load { req_id, .. }
            | Self::EmbedBatch { req_id, .. }
            | Self::EmbedPacked { req_id, .. }
            | Self::AneAdmitPacked { req_id, .. }
            | Self::AneEvictPacked { req_id, .. }
            | Self::Rerank { req_id, .. }
            | Self::RerankSequences { req_id, .. }
            | Self::AneAdmitShape { req_id, .. }
            | Self::AneEvictShape { req_id, .. }
            | Self::Generate { req_id, .. }
            | Self::Transcribe { req_id, .. }
            | Self::Unload { req_id, .. }
            | Self::Ping { req_id } => Some(req_id),
            Self::Shutdown {} => None,
        }
    }

    /// Whether one raw frame follows this request's JSON frame. A worker that
    /// refuses the request must still read that frame; otherwise it would parse
    /// the raw bytes as the next request and lose the connection.
    pub fn carries_raw_frame(&self) -> bool {
        matches!(
            self,
            Self::EmbedBatch { .. }
                | Self::EmbedPacked { .. }
                | Self::Rerank { .. }
                | Self::RerankSequences { .. }
                | Self::Generate { .. }
                | Self::Transcribe { .. }
        )
    }

    /// Validate a transcription payload after the transport enforces `max_frame`.
    pub fn validate_raw_frame(&self, bytes: &[u8]) -> Result<(), RawFrameError> {
        if let Self::Transcribe { mel_frames, .. } = self {
            validate_record_frame(bytes, *mel_frames, 128 * 4)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscribeRevision {
    pub kernel_revision: String,
    pub decode_rule: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerProgressPhase {
    Setup,
    Inference,
    Control,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum WorkerResponse {
    Progress {
        req_id: String,
        completed: u64,
        phase: WorkerProgressPhase,
    },
    Loaded {
        req_id: String,
        model_ref: String,
        dims: usize,
        cold_load_ms: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        buckets: Option<Vec<usize>>,
        /// Direct ANE only: the ladder rungs whose compiled programs aned
        /// already holds (checked on each rung's first and last layer), so the
        /// module predicts a fast load instead of a compile for them. Workers
        /// that do not check omit it, and every rung is then treated as
        /// unknown.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        ane_cached_shapes: Option<Vec<usize>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        transcribe_revision: Option<TranscribeRevision>,
    },
    Vectors {
        req_id: String,
        dims: usize,
        n: usize,
    },
    Scores {
        req_id: String,
    },
    EncoderOutput {
        req_id: String,
        frames: usize,
    },
    DecodeTrace {
        req_id: String,
        steps: usize,
    },
    Transcript {
        req_id: String,
        valid_encoder_frames: usize,
        count: usize,
    },
    Text {
        req_id: String,
        text: String,
        n_prompt: usize,
        n_gen: usize,
        finish_reason: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        generated_token_ids: Vec<u32>,
    },
    Unloaded {
        req_id: String,
    },
    /// Acknowledges `ANE_ADMIT_SHAPE` once the shape is compiled and resident.
    Admitted {
        req_id: String,
        inventory: AnePlacementInventory,
        /// How many of the shape's layers were loaded from aned's compiled
        /// copy and how many were compiled. The module learns load and
        /// compile times separately from them; a worker that omits them has
        /// every admission learned as a compile.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_hit_layers: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        compiled_layers: Option<u32>,
    },
    /// Acknowledges `ANE_EVICT_SHAPE` once the shape's executables are released.
    Evicted {
        req_id: String,
    },
    Pong {
        req_id: String,
        rss_mb: u64,
        models_loaded: usize,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        placement_share: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        buckets: Option<Vec<usize>>,
        /// Direct ANE only: resident shapes per model ref. Distinct from the
        /// Core ML `buckets` ladder.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        ane_resident_shapes: Option<BTreeMap<String, Vec<usize>>>,
    },
    Err {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        req_id: Option<String>,
        code: String,
        msg: String,
    },
}

impl WorkerResponse {
    /// Build a `LOADED` response through one entry point so workers do not need
    /// to name optional fields added by later protocol extensions.
    pub fn loaded(
        req_id: String,
        model_ref: String,
        dims: usize,
        cold_load_ms: u64,
        buckets: Option<Vec<usize>>,
        ane_cached_shapes: Option<Vec<usize>>,
    ) -> Self {
        Self::Loaded {
            req_id,
            model_ref,
            dims,
            cold_load_ms,
            buckets,
            ane_cached_shapes,
            transcribe_revision: None,
        }
    }

    pub fn req_id(&self) -> Option<&str> {
        match self {
            Self::Progress { req_id, .. }
            | Self::Loaded { req_id, .. }
            | Self::Vectors { req_id, .. }
            | Self::Scores { req_id }
            | Self::EncoderOutput { req_id, .. }
            | Self::DecodeTrace { req_id, .. }
            | Self::Transcript { req_id, .. }
            | Self::Text { req_id, .. }
            | Self::Unloaded { req_id }
            | Self::Admitted { req_id, .. }
            | Self::Evicted { req_id }
            | Self::Pong { req_id, .. } => Some(req_id),
            Self::Err { req_id, .. } => req_id.as_deref(),
        }
    }
    /// Whether one raw frame follows this response's JSON frame.
    pub fn carries_raw_frame(&self) -> bool {
        matches!(
            self,
            Self::Vectors { .. }
                | Self::Scores { .. }
                | Self::EncoderOutput { .. }
                | Self::DecodeTrace { .. }
                | Self::Transcript { .. }
        )
    }

    /// Reject mismatched transcription counts as a protocol fault, not a partial result.
    pub fn validate_raw_frame(&self, bytes: &[u8]) -> Result<(), RawFrameError> {
        match self {
            Self::EncoderOutput { frames, .. } => validate_record_frame(bytes, *frames, 1024 * 4),
            Self::DecodeTrace { steps, .. } => {
                validate_record_frame(bytes, *steps, DECODE_TRACE_RECORD_BYTES)
            }
            Self::Transcript { count, .. } => {
                validate_record_frame(bytes, *count, TRANSCRIPT_RECORD_BYTES)
            }
            _ => Ok(()),
        }
    }

    /// The `ERR unsupported_request` answer to a request this worker does not
    /// serve. Callers must already have drained the request's raw frame (see
    /// [`WorkerRequest::carries_raw_frame`]).
    pub fn unsupported_request(request: &WorkerRequest) -> Self {
        Self::Err {
            req_id: request.req_id().map(str::to_owned),
            code: ERR_UNSUPPORTED_REQUEST.to_string(),
            msg: format!("{} is not supported by this worker", request.wire_type()),
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

fn validate_record_frame(
    bytes: &[u8],
    count: usize,
    record_size: usize,
) -> Result<(), RawFrameError> {
    let expected = count
        .checked_mul(record_size)
        .ok_or_else(|| RawFrameError::new("raw frame declared length overflows"))?;
    if bytes.len() != expected {
        return Err(RawFrameError::new(format!(
            "raw frame length {} does not match {count} × {record_size} = {expected}",
            bytes.len()
        )));
    }
    Ok(())
}

pub const DECODE_TRACE_RECORD_BYTES: usize = 40;
pub const TRANSCRIPT_RECORD_BYTES: usize = 12;

/// One decoder step combining encoder and predictor outputs to select a token
/// and duration. Duration ids index [0, 1, 2, 3, 4]; flags use only
/// bit 0 (emitted) and bit 1 (symbol cap hit).
#[derive(Clone, Debug, PartialEq)]
pub struct DecodeTraceRecord {
    pub frame_index: i32,
    pub token_top1: i32,
    pub token_top2: i32,
    pub duration_top1: i32,
    pub duration_top2: i32,
    pub token_logit1: f32,
    pub token_logit2: f32,
    pub duration_logit1: f32,
    pub duration_logit2: f32,
    pub flags: u32,
}

pub fn encode_decode_trace_frame(records: &[DecodeTraceRecord]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(records.len() * DECODE_TRACE_RECORD_BYTES);
    for record in records {
        bytes.extend(encode_i32_frame(&[
            record.frame_index,
            record.token_top1,
            record.token_top2,
            record.duration_top1,
            record.duration_top2,
        ]));
        bytes.extend(encode_f32_frame(&[
            record.token_logit1,
            record.token_logit2,
            record.duration_logit1,
            record.duration_logit2,
        ]));
        bytes.extend_from_slice(&record.flags.to_le_bytes());
    }
    bytes
}

pub fn decode_decode_trace_frame(
    bytes: &[u8],
    steps: usize,
) -> Result<Vec<DecodeTraceRecord>, RawFrameError> {
    validate_record_frame(bytes, steps, DECODE_TRACE_RECORD_BYTES)?;
    bytes
        .as_chunks::<DECODE_TRACE_RECORD_BYTES>()
        .0
        .iter()
        .map(|record| {
            let ints = decode_i32_frame(&record[..20])?;
            let floats = decode_f32_frame(&record[20..36])?;
            let flags = u32::from_le_bytes(record[36..40].try_into().unwrap());
            if flags & !3 != 0 {
                return Err(RawFrameError::new("decode trace has reserved flags set"));
            }
            Ok(DecodeTraceRecord {
                frame_index: ints[0],
                token_top1: ints[1],
                token_top2: ints[2],
                duration_top1: ints[3],
                duration_top2: ints[4],
                token_logit1: floats[0],
                token_logit2: floats[1],
                duration_logit1: floats[2],
                duration_logit2: floats[3],
                flags,
            })
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranscriptRecord {
    pub token_id: i32,
    pub frame_index: i32,
    pub duration: i32,
}

pub fn encode_transcript_frame(records: &[TranscriptRecord]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(records.len() * TRANSCRIPT_RECORD_BYTES);
    for record in records {
        bytes.extend(encode_i32_frame(&[
            record.token_id,
            record.frame_index,
            record.duration,
        ]));
    }
    bytes
}

pub fn decode_transcript_frame(
    bytes: &[u8],
    count: usize,
) -> Result<Vec<TranscriptRecord>, RawFrameError> {
    validate_record_frame(bytes, count, TRANSCRIPT_RECORD_BYTES)?;
    bytes
        .as_chunks::<TRANSCRIPT_RECORD_BYTES>()
        .0
        .iter()
        .map(|record| {
            let ints = decode_i32_frame(record)?;
            Ok(TranscriptRecord {
                token_id: ints[0],
                frame_index: ints[1],
                duration: ints[2],
            })
        })
        .collect()
}

pub fn encode_i32_frame(values: &[i32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

pub fn decode_i32_frame(bytes: &[u8]) -> Result<Vec<i32>, RawFrameError> {
    let (chunks, remainder) = bytes.as_chunks::<4>();
    if !remainder.is_empty() {
        return Err(RawFrameError::new("i32 frame length is not divisible by 4"));
    }
    Ok(chunks
        .iter()
        .map(|chunk| i32::from_le_bytes(*chunk))
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
    let (chunks, remainder) = bytes.as_chunks::<4>();
    if !remainder.is_empty() {
        return Err(RawFrameError::new("f32 frame length is not divisible by 4"));
    }
    Ok(chunks
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcribe_messages_round_trip_with_little_endian_raw_frames() {
        let request = WorkerRequest::Transcribe {
            req_id: "speech-1".into(),
            model_ref: "speech:0".into(),
            mel_frames: 2,
            valid_mel_frames: 1,
            observe: true,
        };
        let json = serde_json::to_vec(&request).unwrap();
        assert_eq!(json, br#"{"type":"TRANSCRIBE","req_id":"speech-1","model_ref":"speech:0","mel_frames":2,"valid_mel_frames":1,"observe":true}"#);
        assert_eq!(
            serde_json::from_slice::<WorkerRequest>(&json).unwrap(),
            request
        );
        assert_eq!(request.wire_type(), "TRANSCRIBE");
        assert_eq!(request.req_id(), Some("speech-1"));
        assert!(request.carries_raw_frame());
        let mel = encode_f32_frame(&vec![1.0; 256]);
        assert_eq!(&mel[..4], &[0, 0, 128, 63]);
        assert_eq!(decode_f32_frame(&mel).unwrap(), vec![1.0; 256]);
        request.validate_raw_frame(&mel).unwrap();
        let unsupported = WorkerResponse::unsupported_request(&request);
        assert!(
            matches!(unsupported, WorkerResponse::Err { req_id: Some(id), code, msg }
            if id == "speech-1" && code == ERR_UNSUPPORTED_REQUEST && msg.contains("TRANSCRIBE"))
        );

        let responses = [
            (
                WorkerResponse::EncoderOutput {
                    req_id: "speech-1".into(),
                    frames: 1,
                },
                r#"{"type":"ENCODER_OUTPUT","req_id":"speech-1","frames":1}"#,
                vec![0; 4096],
            ),
            (
                WorkerResponse::DecodeTrace {
                    req_id: "speech-1".into(),
                    steps: 1,
                },
                r#"{"type":"DECODE_TRACE","req_id":"speech-1","steps":1}"#,
                vec![0; 40],
            ),
            (
                WorkerResponse::Transcript {
                    req_id: "speech-1".into(),
                    valid_encoder_frames: 1,
                    count: 1,
                },
                r#"{"type":"TRANSCRIPT","req_id":"speech-1","valid_encoder_frames":1,"count":1}"#,
                vec![0; 12],
            ),
        ];
        for (response, expected, raw) in responses {
            let json = serde_json::to_vec(&response).unwrap();
            assert_eq!(json, expected.as_bytes());
            assert_eq!(
                serde_json::from_slice::<WorkerResponse>(&json).unwrap(),
                response
            );
            assert_eq!(response.req_id(), Some("speech-1"));
            assert!(response.carries_raw_frame());
            response.validate_raw_frame(&raw).unwrap();
            assert!(response.validate_raw_frame(&raw[..raw.len() - 1]).is_err());
            let mut extra = raw;
            extra.push(0);
            assert!(response.validate_raw_frame(&extra).is_err());
        }
        assert!(request.validate_raw_frame(&mel[..mel.len() - 4]).is_err());
        let mut extra = mel;
        extra.extend_from_slice(&[0; 4]);
        assert!(request.validate_raw_frame(&extra).is_err());
        let overflow = WorkerResponse::Transcript {
            req_id: "overflow".into(),
            valid_encoder_frames: 1,
            count: usize::MAX,
        };
        assert!(overflow.validate_raw_frame(&[]).is_err());
    }

    #[test]
    fn decode_trace_record_codec_pins_all_forty_bytes() {
        let record = DecodeTraceRecord {
            frame_index: -1,
            token_top1: 8192,
            token_top2: 5,
            duration_top1: 4,
            duration_top2: 0,
            token_logit1: 1.0,
            token_logit2: -2.0,
            duration_logit1: 0.5,
            duration_logit2: 4.0,
            flags: 3,
        };
        let expected = [
            255, 255, 255, 255, 0, 32, 0, 0, 5, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63, 0,
            0, 0, 192, 0, 0, 0, 63, 0, 0, 128, 64, 3, 0, 0, 0,
        ];
        assert_eq!(
            encode_decode_trace_frame(std::slice::from_ref(&record)),
            expected
        );
        assert_eq!(
            decode_decode_trace_frame(&expected, 1).unwrap(),
            vec![record.clone()]
        );
        let two = encode_decode_trace_frame(&[record.clone(), record.clone()]);
        assert_eq!(
            decode_decode_trace_frame(&two, 2).unwrap(),
            vec![record.clone(), record]
        );
        assert!(decode_decode_trace_frame(&expected[..39], 1).is_err());
        assert!(decode_decode_trace_frame(&expected, 0).is_err());
        assert!(decode_decode_trace_frame(&expected, 2).is_err());
        let mut reserved_flag = expected;
        reserved_flag[36] = 4;
        assert!(decode_decode_trace_frame(&reserved_flag, 1).is_err());
        assert!(decode_decode_trace_frame(&[], 0).unwrap().is_empty());
    }

    #[test]
    fn transcript_record_codec_pins_triple_order_and_little_endian_bytes() {
        let records = vec![TranscriptRecord {
            token_id: 258,
            frame_index: 3,
            duration: 4,
        }];
        let expected = [2, 1, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0];
        assert_eq!(encode_transcript_frame(&records), expected);
        assert_eq!(decode_transcript_frame(&expected, 1).unwrap(), records);
        assert!(decode_transcript_frame(&expected[..8], 1).is_err());
        assert!(decode_transcript_frame(&expected, 0).is_err());
        assert!(decode_transcript_frame(&[], usize::MAX).is_err());
        assert!(decode_transcript_frame(&[], 0).unwrap().is_empty());
    }

    #[test]
    fn loaded_transcribe_revision_round_trips_and_is_omitted_when_absent() {
        let mut loaded = WorkerResponse::loaded("load".into(), "speech:0".into(), 0, 1, None, None);
        let legacy = br#"{"type":"LOADED","req_id":"load","model_ref":"speech:0","dims":0,"cold_load_ms":1}"#;
        assert_eq!(serde_json::to_vec(&loaded).unwrap(), legacy);
        assert_eq!(
            serde_json::from_slice::<WorkerResponse>(legacy).unwrap(),
            loaded
        );
        let WorkerResponse::Loaded {
            transcribe_revision,
            ..
        } = &mut loaded
        else {
            unreachable!()
        };
        *transcribe_revision = Some(TranscribeRevision {
            kernel_revision: "transcribe-v1".into(),
            decode_rule: "tdt-v1".into(),
        });
        let expected = br#"{"type":"LOADED","req_id":"load","model_ref":"speech:0","dims":0,"cold_load_ms":1,"transcribe_revision":{"kernel_revision":"transcribe-v1","decode_rule":"tdt-v1"}}"#;
        assert_eq!(serde_json::to_vec(&loaded).unwrap(), expected);
        assert_eq!(
            serde_json::from_slice::<WorkerResponse>(expected).unwrap(),
            loaded
        );
    }

    #[test]
    fn packed_planner_boundaries_ties_occurrences_and_order_restoration() {
        let lengths = [1, 127, 128, 129, 255, 256, 257];
        let plan = plan_packed(&lengths).unwrap();
        assert_eq!(plan.passes, vec![vec![5], vec![4, 0], vec![3, 1]]);
        assert_eq!(plan.singles, vec![2, 6]);
        let mut occurrences = vec![0; lengths.len()];
        let mut restored = vec![0; lengths.len()];
        for &index in plan.passes.iter().flatten().chain(&plan.singles) {
            occurrences[index] += 1;
            restored[index] = lengths[index];
        }
        assert_eq!(occurrences, vec![1; lengths.len()]);
        assert_eq!(restored, lengths);
        assert_eq!(
            plan_packed(&[128, 128, 128, 128]).unwrap().passes,
            vec![vec![0, 1], vec![2, 3]]
        );
        assert_eq!(plan_packed(&[1]).unwrap().singles, vec![0]);
        assert!(plan_packed(&[0]).is_err());
    }

    #[test]
    fn packed_masks_match_independent_segment_oracle() {
        for lengths in [
            vec![1, 127, 128],
            vec![127, 129],
            vec![129, 126, 1],
            vec![255, 1],
            vec![256],
            vec![128, 1],
        ] {
            let owners: Vec<_> = lengths
                .iter()
                .enumerate()
                .flat_map(|(segment, &len)| (0..len).map(move |position| (segment, position)))
                .collect();
            let (mask, positions) = packed_mask_positions(&lengths, 256).unwrap();
            for q in 0..256 {
                assert_eq!(positions[q], owners.get(q).map_or(0, |&(_, pos)| pos));
                for k in 0..256 {
                    let allowed = match (owners.get(q), owners.get(k)) {
                        (Some(&(sq, pq)), Some(&(sk, pk))) => sq == sk && pk <= pq,
                        (None, _) => q == k,
                        _ => false,
                    };
                    assert_eq!(
                        mask[q * 256 + k],
                        if allowed { 0.0 } else { -10_000.0 },
                        "lengths={lengths:?} q={q} k={k}"
                    );
                }
            }
        }
    }

    #[test]
    fn packed_plan_validation_rejects_whole_invalid_plan() {
        assert!(validate_packed_passes(&[127, 129], &[vec![1, 0]]).is_ok());
        for passes in [
            vec![vec![0]],
            vec![vec![0, 0, 1]],
            vec![vec![0, 2]],
            vec![vec![], vec![0, 1]],
        ] {
            assert!(validate_packed_passes(&[127, 129], &passes).is_err());
        }
        assert!(validate_packed_passes(&[257], &[vec![0]]).is_err());
    }

    #[test]
    fn packed_requests_have_distinct_wire_tags_and_raw_frame() {
        let request = WorkerRequest::EmbedPacked {
            req_id: "packed".into(),
            model_ref: "model".into(),
            program_revision: 1,
            items: vec![
                WorkerTokenItem {
                    id: "a".into(),
                    n_tokens: 127,
                },
                WorkerTokenItem {
                    id: "b".into(),
                    n_tokens: 129,
                },
            ],
            passes: vec![vec![1, 0]],
        };
        let bytes = serde_json::to_vec(&request).unwrap();
        assert_eq!(
            serde_json::from_slice::<WorkerRequest>(&bytes).unwrap(),
            request
        );
        assert_eq!(request.wire_type(), "EMBED_PACKED");
        assert!(request.carries_raw_frame());
        assert_eq!(request.req_id(), Some("packed"));
        for request in [
            WorkerRequest::AneAdmitPacked {
                req_id: "admit".into(),
                model_ref: "model".into(),
                program_revision: 1,
            },
            WorkerRequest::AneEvictPacked {
                req_id: "evict".into(),
                model_ref: "model".into(),
                program_revision: 1,
            },
        ] {
            assert!(!request.carries_raw_frame());
            assert_eq!(
                serde_json::from_slice::<WorkerRequest>(&serde_json::to_vec(&request).unwrap())
                    .unwrap(),
                request
            );
        }
    }

    #[test]
    fn v3_hello_without_packed_keeps_minimal_bytes_and_binding() {
        let bytes = br#"{"v":4,"nonce":"legacy","engine":{"engine":"ane-direct-worker","version":"0.1.0"},"pid":42,"max_frame":16777216,"manifest_digest":"manifest","kernel_revision":"ane-direct-graph-v1"}"#;
        let mut hello: WorkerHello = serde_json::from_slice(bytes).unwrap();
        assert!(hello.packed.is_none());
        assert_eq!(serde_json::to_vec(&hello).unwrap(), bytes);
        let expected = ExpectedHelloBinding {
            manifest_digest: "manifest".into(),
            kernel_revision: "ane-direct-graph-v1".into(),
        };
        check_hello_binding(&hello, Some(&expected)).unwrap();
        let refusal = check_hello_binding(&hello, Some(&binding())).unwrap_err();
        hello.packed = Some(PackedCapability::tight256());
        check_hello_binding(&hello, Some(&expected)).unwrap();
        assert_eq!(
            check_hello_binding(&hello, Some(&binding())).unwrap_err(),
            refusal
        );
        let original_engine = serde_json::to_vec(&hello.engine).unwrap();
        hello.packed = None;
        assert_eq!(serde_json::to_vec(&hello.engine).unwrap(), original_engine);
    }

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

    #[test]
    fn rerank_and_generate_messages_round_trip_with_raw_frames() {
        let rerank = WorkerRequest::Rerank {
            req_id: "rerank-1".to_string(),
            model_ref: "llama:0".to_string(),
            query_n_tokens: 2,
            candidates: vec![
                WorkerCandidate { n_tokens: 3 },
                WorkerCandidate { n_tokens: 4 },
            ],
        };
        let encoded = serde_json::to_vec(&rerank).unwrap();
        assert_eq!(
            serde_json::from_slice::<WorkerRequest>(&encoded).unwrap(),
            rerank
        );
        let ids = [101, 102, 201, 202, 203, 301, 302, 303, 304];
        assert_eq!(decode_i32_frame(&encode_i32_frame(&ids)).unwrap(), ids);

        let generate = WorkerRequest::Generate {
            req_id: "generate-1".to_string(),
            model_ref: "llama:0".to_string(),
            max_tokens: 8,
            grammar: Some("root ::= \"yes\" | \"no\"".to_string()),
        };
        let encoded = serde_json::to_vec(&generate).unwrap();
        assert_eq!(
            serde_json::from_slice::<WorkerRequest>(&encoded).unwrap(),
            generate
        );
        let generated = serde_json::to_value(WorkerResponse::Text {
            req_id: "generate-1".to_string(),
            text: "yes".to_string(),
            n_prompt: 4,
            n_gen: 1,
            finish_reason: "stop".to_string(),
            generated_token_ids: vec![9693],
        })
        .unwrap();
        assert_eq!(generated["type"], "TEXT");
        assert_eq!(generated["generated_token_ids"], serde_json::json!([9693]));
    }

    fn hello(engine: &str, manifest: Option<&str>, revision: Option<&str>) -> WorkerHello {
        WorkerHello {
            v: WORKER_PROTOCOL_VERSION,
            nonce: "0123456789abcdef".to_string(),
            engine: EngineIdentity {
                engine: engine.to_string(),
                version: "test".to_string(),
                build_flags: BTreeMap::new(),
            },
            pid: 1,
            max_frame: DEFAULT_MAX_FRAME_BYTES,
            manifest_digest: manifest.map(str::to_owned),
            kernel_revision: revision.map(str::to_owned),
            packed: None,
        }
    }

    fn binding() -> ExpectedHelloBinding {
        ExpectedHelloBinding {
            manifest_digest: "a".repeat(64),
            kernel_revision: "ane-direct-graph-v1".to_string(),
        }
    }

    #[test]
    fn worker_protocol_is_version_three() {
        assert_eq!(WORKER_PROTOCOL_VERSION, 4);
    }

    #[test]
    fn hello_binding_fields_are_optional_on_the_wire() {
        let legacy = serde_json::json!({
            "v": 4,
            "nonce": "0123456789abcdef",
            "engine": {
                "engine": "llama.cpp-worker",
                "version": "0",
                "build_flags": { "backend": "metal" },
            },
            "pid": 1,
            "max_frame": DEFAULT_MAX_FRAME_BYTES,
        });
        let decoded: WorkerHello = serde_json::from_value(legacy.clone()).unwrap();
        assert_eq!(decoded.manifest_digest, None);
        assert_eq!(decoded.kernel_revision, None);
        assert_eq!(serde_json::to_value(&decoded).unwrap(), legacy);

        let bound = hello("owned-cuda", Some("abc"), Some("rev"));
        let encoded = serde_json::to_value(&bound).unwrap();
        assert_eq!(encoded["manifest_digest"], "abc");
        assert_eq!(encoded["kernel_revision"], "rev");
        assert_eq!(
            serde_json::from_value::<WorkerHello>(encoded).unwrap(),
            bound
        );
    }

    #[test]
    fn owned_hello_must_match_the_expected_binding() {
        let expected = binding();
        let digest = expected.manifest_digest.as_str();
        for engine in OWNED_WORKER_HELLO_ENGINES {
            assert_eq!(
                check_hello_binding(
                    &hello(engine, Some(digest), Some("ane-direct-graph-v1")),
                    Some(&expected)
                ),
                Ok(())
            );
            let cases = [
                (None, Some("ane-direct-graph-v1"), ERR_MANIFEST_MISMATCH),
                (
                    Some("b".repeat(64)),
                    Some("ane-direct-graph-v1"),
                    ERR_MANIFEST_MISMATCH,
                ),
                (Some(digest.to_string()), None, ERR_KERNEL_REVISION_MISMATCH),
                (
                    Some(digest.to_string()),
                    Some("ane-direct-graph-v0"),
                    ERR_KERNEL_REVISION_MISMATCH,
                ),
            ];
            for (manifest, revision, code) in cases {
                let refusal = check_hello_binding(
                    &hello(engine, manifest.as_deref(), revision),
                    Some(&expected),
                )
                .expect_err("a missing or different binding field is refused");
                assert_eq!(refusal.code, code, "{engine} {manifest:?} {revision:?}");
                assert_eq!(refusal.engine, engine);
            }
        }
    }

    #[test]
    fn unbound_lanes_and_non_owned_workers_skip_the_binding_check() {
        for engine in OWNED_WORKER_HELLO_ENGINES {
            assert_eq!(
                check_hello_binding(&hello(engine, None, None), None),
                Ok(())
            );
        }
        for engine in [
            "owned-metal-decode",
            "llama.cpp-worker",
            "ane-coreml-worker",
        ] {
            assert_eq!(
                check_hello_binding(&hello(engine, None, None), Some(&binding())),
                Ok(())
            );
        }
    }

    #[test]
    fn rerank_sequences_round_trips_with_one_i32_frame() {
        let sequences = [vec![1, 10, 11, 2], vec![1, 20, 2], vec![1, 30, 31, 32, 2]];
        let request = WorkerRequest::RerankSequences {
            req_id: "rerank-7".to_string(),
            model_ref: "owned:0".to_string(),
            sequences: sequences
                .iter()
                .map(|sequence| WorkerSequence {
                    n_tokens: sequence.len(),
                })
                .collect(),
        };
        let encoded = serde_json::to_value(&request).unwrap();
        assert_eq!(
            encoded,
            serde_json::json!({
                "type": "RERANK_SEQUENCES",
                "req_id": "rerank-7",
                "model_ref": "owned:0",
                "sequences": [{ "n_tokens": 4 }, { "n_tokens": 3 }, { "n_tokens": 5 }],
            })
        );
        assert_eq!(request.wire_type(), "RERANK_SEQUENCES");
        assert!(request.carries_raw_frame());
        assert_eq!(
            serde_json::from_value::<WorkerRequest>(encoded).unwrap(),
            request
        );
        let flat: Vec<i32> = sequences.concat();
        assert_eq!(decode_i32_frame(&encode_i32_frame(&flat)).unwrap(), flat);
    }

    #[test]
    fn ane_shape_messages_round_trip_without_raw_frames() {
        let admit = WorkerRequest::AneAdmitShape {
            req_id: "admit-1".to_string(),
            model_ref: "ane:0".to_string(),
            shape: 512,
        };
        let evict = WorkerRequest::AneEvictShape {
            req_id: "evict-1".to_string(),
            model_ref: "ane:0".to_string(),
            shape: 512,
        };
        for (request, wire) in [(admit, "ANE_ADMIT_SHAPE"), (evict, "ANE_EVICT_SHAPE")] {
            let encoded = serde_json::to_value(&request).unwrap();
            assert_eq!(encoded["type"], wire);
            assert_eq!(encoded["shape"], 512);
            assert_eq!(request.wire_type(), wire);
            assert!(!request.carries_raw_frame());
            assert_eq!(
                serde_json::from_value::<WorkerRequest>(encoded).unwrap(),
                request
            );
        }

        let admitted = WorkerResponse::Admitted {
            req_id: "admit-1".to_string(),
            inventory: AnePlacementInventory {
                executables: vec![
                    AneExecutable {
                        id: "exe-0".to_string(),
                        layers: vec![0, 1, 2],
                    },
                    AneExecutable {
                        id: "exe-1".to_string(),
                        layers: vec![3],
                    },
                ],
                cpu_stages: vec!["token_embedding".to_string(), "pooling".to_string()],
            },
            cache_hit_layers: None,
            compiled_layers: None,
        };
        let encoded = serde_json::to_value(&admitted).unwrap();
        assert_eq!(
            encoded,
            serde_json::json!({
                "type": "ADMITTED",
                "req_id": "admit-1",
                "inventory": {
                    "executables": [
                        { "id": "exe-0", "layers": [0, 1, 2] },
                        { "id": "exe-1", "layers": [3] },
                    ],
                    "cpu_stages": ["token_embedding", "pooling"],
                },
            })
        );
        assert_eq!(
            serde_json::from_value::<WorkerResponse>(encoded).unwrap(),
            admitted
        );
        let evicted = WorkerResponse::Evicted {
            req_id: "evict-1".to_string(),
        };
        let encoded = serde_json::to_value(&evicted).unwrap();
        assert_eq!(
            encoded,
            serde_json::json!({ "type": "EVICTED", "req_id": "evict-1" })
        );
        assert_eq!(
            serde_json::from_value::<WorkerResponse>(encoded).unwrap(),
            evicted
        );
    }

    /// The cache fields on `LOADED` and `ADMITTED` are optional additions to
    /// protocol v3: a worker that sends them still talks to a module that
    /// predates them, and a module still accepts a worker that omits them.
    #[test]
    fn cache_fields_are_optional_in_both_directions() {
        // The replies as a module built before the fields existed decodes
        // them: the same tags and fields, minus the new ones.
        #[derive(Debug, Deserialize)]
        #[serde(tag = "type", rename_all = "SCREAMING_SNAKE_CASE")]
        enum OlderModuleResponse {
            Loaded {
                req_id: String,
                model_ref: String,
                dims: usize,
                cold_load_ms: u64,
                #[serde(default)]
                buckets: Option<Vec<usize>>,
            },
            Admitted {
                req_id: String,
                inventory: AnePlacementInventory,
            },
        }

        let loaded = WorkerResponse::Loaded {
            req_id: "load-1".into(),
            model_ref: "ane-direct:p:sha256:1".into(),
            dims: 1024,
            cold_load_ms: 5,
            buckets: Some(vec![128, 256]),
            ane_cached_shapes: Some(vec![128]),
            transcribe_revision: None,
        };
        let admitted = WorkerResponse::Admitted {
            req_id: "admit-1".into(),
            inventory: AnePlacementInventory::default(),
            cache_hit_layers: Some(27),
            compiled_layers: Some(1),
        };
        let loaded_wire = serde_json::to_value(&loaded).unwrap();
        let admitted_wire = serde_json::to_value(&admitted).unwrap();
        assert_eq!(loaded_wire["ane_cached_shapes"], serde_json::json!([128]));
        assert_eq!(admitted_wire["cache_hit_layers"], 27);
        assert_eq!(admitted_wire["compiled_layers"], 1);
        assert_eq!(
            serde_json::from_value::<WorkerResponse>(loaded_wire.clone()).unwrap(),
            loaded
        );
        assert_eq!(
            serde_json::from_value::<WorkerResponse>(admitted_wire.clone()).unwrap(),
            admitted
        );
        match serde_json::from_value::<OlderModuleResponse>(loaded_wire).unwrap() {
            OlderModuleResponse::Loaded {
                req_id,
                model_ref,
                dims,
                cold_load_ms,
                buckets,
            } => {
                assert_eq!(
                    (req_id.as_str(), model_ref.as_str(), dims, cold_load_ms),
                    ("load-1", "ane-direct:p:sha256:1", 1024, 5)
                );
                assert_eq!(buckets, Some(vec![128, 256]));
            }
            other => panic!("decoded as {other:?}"),
        }
        match serde_json::from_value::<OlderModuleResponse>(admitted_wire).unwrap() {
            OlderModuleResponse::Admitted { req_id, inventory } => {
                assert_eq!(req_id, "admit-1");
                assert_eq!(inventory, AnePlacementInventory::default());
            }
            other => panic!("decoded as {other:?}"),
        }

        // An older worker's replies, without the fields, decode with them
        // unset and encode back without them.
        let older_loaded = serde_json::json!({
            "type": "LOADED", "req_id": "load-1", "model_ref": "m", "dims": 8,
            "cold_load_ms": 1, "buckets": [128],
        });
        let older_admitted = serde_json::json!({
            "type": "ADMITTED", "req_id": "admit-1",
            "inventory": { "executables": [], "cpu_stages": [] },
        });
        let decoded = serde_json::from_value::<WorkerResponse>(older_loaded.clone()).unwrap();
        assert!(matches!(
            decoded,
            WorkerResponse::Loaded {
                ane_cached_shapes: None,
                transcribe_revision: None,
                ..
            }
        ));
        assert_eq!(serde_json::to_value(&decoded).unwrap(), older_loaded);
        let decoded = serde_json::from_value::<WorkerResponse>(older_admitted.clone()).unwrap();
        assert!(matches!(
            decoded,
            WorkerResponse::Admitted {
                cache_hit_layers: None,
                compiled_layers: None,
                ..
            }
        ));
        assert_eq!(serde_json::to_value(&decoded).unwrap(), older_admitted);
    }

    /// The module decodes worker replies with this crate's `WorkerResponse`,
    /// so a newer worker may add optional fields to `LOADED` and `ADMITTED`
    /// only while that type ignores fields it does not know.
    #[test]
    fn worker_response_ignores_unknown_extension_fields() {
        let loaded = serde_json::json!({
            "type": "LOADED", "req_id": "load-1", "model_ref": "m", "dims": 8,
            "cold_load_ms": 1, "buckets": [128], "ane_cached_shapes": [128],
            "added_later": {"nested": true},
        });
        assert_eq!(
            serde_json::from_value::<WorkerResponse>(loaded).unwrap(),
            WorkerResponse::Loaded {
                req_id: "load-1".into(),
                model_ref: "m".into(),
                dims: 8,
                cold_load_ms: 1,
                buckets: Some(vec![128]),
                ane_cached_shapes: Some(vec![128]),
                transcribe_revision: None,
            }
        );
        let admitted = serde_json::json!({
            "type": "ADMITTED", "req_id": "admit-1",
            "inventory": { "executables": [], "cpu_stages": [] },
            "cache_hit_layers": 28, "compiled_layers": 0, "added_later": 1,
        });
        assert_eq!(
            serde_json::from_value::<WorkerResponse>(admitted).unwrap(),
            WorkerResponse::Admitted {
                req_id: "admit-1".into(),
                inventory: AnePlacementInventory::default(),
                cache_hit_layers: Some(28),
                compiled_layers: Some(0),
            }
        );
    }

    /// A worker built from this crate must keep accepting requests from a
    /// newer module that adds optional fields the same way, so neither the
    /// requests nor HELLO's acknowledgement refuse fields they do not know.
    #[test]
    fn worker_side_decoding_ignores_unknown_fields() {
        let admit = serde_json::json!({
            "type": "ANE_ADMIT_SHAPE", "req_id": "a", "model_ref": "m", "shape": 128,
            "added_later": true,
        });
        assert_eq!(
            serde_json::from_value::<WorkerRequest>(admit).unwrap(),
            WorkerRequest::AneAdmitShape {
                req_id: "a".into(),
                model_ref: "m".into(),
                shape: 128,
            }
        );
        let load = serde_json::json!({
            "type": "LOAD", "req_id": "l", "artifact_path": "/p", "artifact_digest": "d",
            "format": "safetensors-package", "runtime_config": {}, "added_later": 1,
        });
        assert!(matches!(
            serde_json::from_value::<WorkerRequest>(load).unwrap(),
            WorkerRequest::Load { .. }
        ));
        let ack = serde_json::json!({
            "v": WORKER_PROTOCOL_VERSION, "accept": true, "max_frame": 4096, "added_later": 1,
        });
        assert!(
            serde_json::from_value::<WorkerHelloAck>(ack)
                .unwrap()
                .accept
        );
    }

    #[test]
    fn pong_serializes_ane_resident_shapes_only_when_set() {
        let llama_pong = serde_json::json!({
            "type": "PONG", "req_id": "ping-1", "rss_mb": 0, "models_loaded": 1,
        });
        let coreml_pong = serde_json::json!({
            "type": "PONG", "req_id": "ping-2", "rss_mb": 12, "models_loaded": 1,
            "placement_share": 0.98, "buckets": [128, 256, 512],
        });
        for fixture in [llama_pong, coreml_pong] {
            let decoded: WorkerResponse = serde_json::from_value(fixture.clone()).unwrap();
            let WorkerResponse::Pong {
                ane_resident_shapes,
                ..
            } = &decoded
            else {
                panic!("fixture decodes as PONG");
            };
            assert_eq!(*ane_resident_shapes, None);
            assert_eq!(serde_json::to_value(&decoded).unwrap(), fixture);
        }

        let direct = WorkerResponse::Pong {
            req_id: "ping-3".to_string(),
            rss_mb: 0,
            models_loaded: 2,
            placement_share: None,
            buckets: None,
            ane_resident_shapes: Some(BTreeMap::from([
                ("ane:0".to_string(), vec![128, 256]),
                ("ane:1".to_string(), vec![2048]),
            ])),
        };
        let encoded = serde_json::to_value(&direct).unwrap();
        assert_eq!(
            encoded["ane_resident_shapes"],
            serde_json::json!({ "ane:0": [128, 256], "ane:1": [2048] })
        );
        assert_eq!(
            serde_json::from_value::<WorkerResponse>(encoded).unwrap(),
            direct
        );
    }

    #[test]
    fn loaded_constructor_preserves_wire_bytes() {
        let minimal = WorkerResponse::loaded("load-1".into(), "m".into(), 8, 5, None, None);
        assert_eq!(
            serde_json::to_string(&minimal).unwrap(),
            r#"{"type":"LOADED","req_id":"load-1","model_ref":"m","dims":8,"cold_load_ms":5}"#
        );
        let with_shapes = WorkerResponse::loaded(
            "load-1".into(),
            "m".into(),
            8,
            5,
            Some(vec![128, 256]),
            Some(vec![128]),
        );
        assert_eq!(
            serde_json::to_string(&with_shapes).unwrap(),
            r#"{"type":"LOADED","req_id":"load-1","model_ref":"m","dims":8,"cold_load_ms":5,"buckets":[128,256],"ane_cached_shapes":[128]}"#
        );
    }

    #[test]
    fn unsupported_request_answer_names_the_request() {
        let request = WorkerRequest::RerankSequences {
            req_id: "rerank-9".to_string(),
            model_ref: "llama:0".to_string(),
            sequences: vec![WorkerSequence { n_tokens: 3 }],
        };
        let WorkerResponse::Err { req_id, code, msg } =
            WorkerResponse::unsupported_request(&request)
        else {
            panic!("unsupported request answers ERR");
        };
        assert_eq!(req_id.as_deref(), Some("rerank-9"));
        assert_eq!(code, ERR_UNSUPPORTED_REQUEST);
        assert!(msg.contains("RERANK_SEQUENCES"));
    }
}
