//! Canonical names for worker HELLO handshake identities.
//!
//! These identities are validated strictly by the worker host. Every producer
//! and consumer must use the same constant so a catalog rename cannot leave a
//! worker announcing a different name than the host expects.

/// The catalog engine name used by llama model specifications.
pub const LLAMA_ENGINE: &str = "llama";
/// The HELLO identity announced by the llama.cpp worker.
pub const LLAMA_WORKER_ENGINE: &str = "llama.cpp-worker";
/// The HELLO identity announced by the Core ML/ANE worker.
pub const ANE_WORKER_ENGINE: &str = "ane-coreml-worker";
/// The HELLO identity announced by the owned Metal decode worker.
pub const DECODE_WORKER_ENGINE: &str = "owned-metal-decode";
/// The HELLO identity announced by the owned CUDA worker.
pub const CUDA_WORKER_ENGINE: &str = "owned-cuda";

/// Worker binary file names for sibling resolution beside the module binary.
/// Release installers unpack each binary at the archive root, so a worker
/// shipped in the same install directory is a sibling of `ck-synapse`.
pub fn worker_binary_file_name(engine: &str) -> Option<&'static str> {
    match engine {
        LLAMA_ENGINE => Some("ck-synapse-worker-llama"),
        "ane" => Some("ck-synapse-worker-ane"),
        CUDA_WORKER_ENGINE => Some("ck-synapse-worker-cuda"),
        DECODE_WORKER_ENGINE => Some("ck-synapse-worker-decode"),
        _ => None,
    }
}
