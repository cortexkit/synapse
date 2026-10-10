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
/// The HELLO identity announced by the owned Vulkan worker.
pub const VULKAN_WORKER_ENGINE: &str = "owned-vulkan";
/// The HELLO identity announced by the direct Neural Engine worker, which
/// drives the ANE through the private framework rather than Core ML.
pub const ANE_DIRECT_WORKER_ENGINE: &str = "ane-direct-worker";

/// Worker binary file names for sibling resolution beside the module binary.
/// Release installers unpack each binary at the archive root, so a worker
/// shipped in the same install directory is a sibling of `ck-synapse`.
pub fn worker_binary_file_name(engine: &str) -> Option<&'static str> {
    match engine {
        LLAMA_ENGINE => Some("ck-synapse-worker-llama"),
        "ane" => Some("ck-synapse-worker-ane"),
        CUDA_WORKER_ENGINE => Some("ck-synapse-worker-cuda"),
        DECODE_WORKER_ENGINE => Some("ck-synapse-worker-decode"),
        VULKAN_WORKER_ENGINE => Some("ck-synapse-worker-vulkan"),
        ANE_DIRECT_WORKER_ENGINE => Some("ck-synapse-worker-ane-direct"),
        _ => None,
    }
}

/// CUDA kernel revision used for lane fingerprints and worker HELLO validation.
/// Keeping the string here lets the host identify kernels without linking GPU code.
pub const CUDA_KERNEL_REVISION: &str = "4d0ded67c30286fe2be37cc7413359ad745dd751";
/// SHA-256 of the Vulkan worker's embedded SPIR-V set.
pub const VULKAN_KERNEL_REVISION: &str =
    "7a5621d965123c2661ad6a795774d07b81f44a86db4d68167893ad4023e87730";
/// Revision identifying the computations compiled for the direct Neural Engine worker.
pub const ANE_DIRECT_KERNEL_REVISION: &str = "ane-direct-graph-v1";
/// Revision identifying owned Metal computations and the policy that pads inputs
/// to supported sequence-length buckets.
pub const METAL_KERNEL_REVISION: &str = "owned-metal-graph-4-bucket-2";

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn worker_names_resolve_beside_module_on_unix_and_windows() {
        for windows in [false, true] {
            let module = Path::new("install").join(if windows {
                "ck-synapse.exe"
            } else {
                "ck-synapse"
            });
            for (engine, binary) in [
                (LLAMA_ENGINE, "ck-synapse-worker-llama"),
                ("ane", "ck-synapse-worker-ane"),
                (CUDA_WORKER_ENGINE, "ck-synapse-worker-cuda"),
                (DECODE_WORKER_ENGINE, "ck-synapse-worker-decode"),
                (VULKAN_WORKER_ENGINE, "ck-synapse-worker-vulkan"),
                (ANE_DIRECT_WORKER_ENGINE, "ck-synapse-worker-ane-direct"),
            ] {
                let mut sibling = module
                    .parent()
                    .unwrap()
                    .join(worker_binary_file_name(engine).unwrap());
                if windows {
                    sibling.set_extension("exe");
                }
                let expected = if windows {
                    format!("{binary}.exe")
                } else {
                    binary.to_string()
                };
                assert_eq!(sibling, Path::new("install").join(expected));
            }
            assert_eq!(worker_binary_file_name(LLAMA_WORKER_ENGINE), None);
            assert_eq!(worker_binary_file_name(ANE_WORKER_ENGINE), None);
        }
    }
}
