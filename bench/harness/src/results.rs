//! Shared result schema every lane binary emits.
//!
//! Kept in one place so the tradeoff doc's tables are generated from
//! identical fields across runtimes.

use serde::{Deserialize, Serialize};

/// One lane run over one workload.
#[derive(Serialize, Deserialize, Debug)]
pub struct LaneResult {
    /// Runtime lane, e.g. "ort-cpu", "mlx", "llama-metal", "burn-wgpu".
    pub lane: String,
    /// Workload id, e.g. "embed-corpus-v1", "microllm-oneshot-v1".
    pub workload: String,
    /// Model identity as served, e.g. "Qwen3-Embedding-0.6B@bf16".
    pub model: String,
    /// Time from process start to model ready (load + warmup), seconds.
    pub cold_load_s: f64,
    /// Pure inference wall time (excludes load), seconds.
    pub infer_wall_s: f64,
    /// Total input tokens processed.
    pub input_tokens: u64,
    /// Tokens/second over infer_wall_s.
    pub tok_per_s: f64,
    /// Items processed (chunks embedded / completions generated).
    pub items: u64,
    /// Output parity vs the reference lane: mean cosine similarity of
    /// produced vectors against reference vectors (embed workload) or None
    /// for generative workloads (judged separately).
    pub parity_mean_cosine: Option<f64>,
    /// Lane-reported peak RSS if the runtime is a child server, else None
    /// (the power wrapper measures the lane process itself).
    pub self_peak_rss_bytes: Option<u64>,
    /// Free-form notes (quantization, thread caps, batch policy).
    pub notes: String,
}
