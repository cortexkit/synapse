//! Canonical names for worker HELLO handshake identities.
//!
//! These identities are validated strictly by the worker host. Every producer
//! and consumer must use the same constant so a catalog rename cannot leave a
//! worker announcing a different name than the host expects.

/// The catalog engine name used by llama model specifications.
pub const LLAMA_ENGINE: &str = "llama";
/// The HELLO identity announced by the llama.cpp worker.
pub const LLAMA_WORKER_ENGINE: &str = "llama.cpp-worker";
/// The HELLO identity announced by the MLX worker.
pub const MLX_WORKER_ENGINE: &str = "mlx-worker";
/// The HELLO identity announced by the Core ML/ANE worker.
pub const ANE_WORKER_ENGINE: &str = "ane-coreml-worker";
/// The HELLO identity announced by the owned Metal decode worker.
pub const DECODE_WORKER_ENGINE: &str = "owned-metal-decode";
/// The HELLO identity announced by the owned CUDA worker.
pub const CUDA_WORKER_ENGINE: &str = "owned-cuda";
