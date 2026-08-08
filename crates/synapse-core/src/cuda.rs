use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{worker_engine_names::CUDA_WORKER_ENGINE, EngineIdentity};

/// The stable engine name used by the module and every owned-CUDA record.
pub const OWNED_CUDA_ENGINE: &str = CUDA_WORKER_ENGINE;
/// The backend identity describes PTX executed by the CUDA driver JIT.
pub const OWNED_CUDA_BACKEND: &str = "cuda-ptx";
pub const OWNED_CUDA_PTX_VIRTUAL_ARCH: &str = "compute_75";
pub const OWNED_CUDA_MINIMUM_DEVICE_CC: f32 = 7.5;
pub const OWNED_CUDA_MINIMUM_DRIVER_API: u32 = 12040;
pub const OWNED_CUDA_IDENTITY_REVISION: &str = "owned-cuda-identity-v1";
pub const MACHINE_PROFILE_HASH_REVISION: &str = "machine-profile-v2";

/// Hardware values observed by the capability probe before an owned worker is
/// created. Keeping the observed values separate from the package floor makes
/// below-floor refusal evidence auditable and deterministic.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CudaMachineInfo {
    pub driver_api: u32,
    pub compute_capability: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packaging_driver: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CudaUnsupportedReason {
    DriverApiBelowFloor,
    ComputeCapabilityBelowFloor,
    HardwareUnavailable,
}

impl CudaUnsupportedReason {
    pub const fn code(self) -> &'static str {
        match self {
            Self::DriverApiBelowFloor | Self::ComputeCapabilityBelowFloor => {
                "owned_cuda_unsupported"
            }
            Self::HardwareUnavailable => "backend_unavailable",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum CudaFloorDecision {
    Supported {
        observed: CudaMachineInfo,
    },
    Unsupported {
        reason: CudaUnsupportedReason,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        observed: Option<CudaMachineInfo>,
    },
}

impl CudaFloorDecision {
    #[must_use]
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Supported { .. })
    }

    #[must_use]
    pub fn refusal_code(&self) -> Option<&'static str> {
        match self {
            Self::Supported { .. } => None,
            Self::Unsupported { reason, .. } => Some(reason.code()),
        }
    }
}

/// Apply both hardware predicates. A worker must not be spawned before this
/// function returns `Supported`.
#[must_use]
pub fn evaluate_cuda_floor(
    driver_api: u32,
    compute_major: u32,
    compute_minor: u32,
    packaging_driver: Option<String>,
) -> CudaFloorDecision {
    let observed = CudaMachineInfo {
        driver_api,
        compute_capability: compute_major as f32 + (compute_minor as f32 / 10.0),
        packaging_driver,
    };
    if driver_api < OWNED_CUDA_MINIMUM_DRIVER_API {
        return CudaFloorDecision::Unsupported {
            reason: CudaUnsupportedReason::DriverApiBelowFloor,
            observed: Some(observed),
        };
    }
    if observed.compute_capability < OWNED_CUDA_MINIMUM_DEVICE_CC {
        return CudaFloorDecision::Unsupported {
            reason: CudaUnsupportedReason::ComputeCapabilityBelowFloor,
            observed: Some(observed),
        };
    }
    CudaFloorDecision::Supported { observed }
}

/// Build the immutable identity carried in manifests, fingerprints, and the
/// worker handshake. The stable model family and dtype remain separate from
/// the backend so two CUDA cells cannot be mistaken for an in-process lane.
#[must_use]
pub fn owned_cuda_engine_identity(
    family: &str,
    dtype: &str,
    kernel_revision: &str,
) -> EngineIdentity {
    let mut build_flags = BTreeMap::new();
    build_flags.insert("backend".to_string(), OWNED_CUDA_BACKEND.to_string());
    build_flags.insert("family".to_string(), family.to_string());
    build_flags.insert("dtype".to_string(), dtype.to_string());
    build_flags.insert("kernel_revision".to_string(), kernel_revision.to_string());
    build_flags.insert(
        "ptx_virtual_arch".to_string(),
        OWNED_CUDA_PTX_VIRTUAL_ARCH.to_string(),
    );
    build_flags.insert("minimum_device_cc".to_string(), "7.5".to_string());
    build_flags.insert(
        "minimum_cuda_driver_api".to_string(),
        OWNED_CUDA_MINIMUM_DRIVER_API.to_string(),
    );
    build_flags.insert("risk_class".to_string(), "abort_capable".to_string());
    build_flags.insert(
        "identity_revision".to_string(),
        OWNED_CUDA_IDENTITY_REVISION.to_string(),
    );
    EngineIdentity {
        engine: CUDA_WORKER_ENGINE.to_string(),
        version: kernel_revision.to_string(),
        build_flags,
    }
}

/// Sanitize an engine identifier for generic environment-variable construction.
/// The compatibility exception for llama is intentionally handled by the two
/// explicit helper branches below, not by a second ad-hoc sanitizer.
#[must_use]
pub fn sanitize_engine_env_component(engine: &str) -> String {
    engine
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect()
}

#[must_use]
pub fn worker_binary_env_var(engine: &str) -> String {
    if engine == "llama" {
        "SYNAPSE_LLAMA_WORKER_BIN".to_string()
    } else {
        format!(
            "SYNAPSE_{}_WORKER_BIN",
            sanitize_engine_env_component(engine)
        )
    }
}

#[must_use]
pub fn worker_runtime_dir_env_var(engine: &str) -> String {
    if engine == "llama" {
        "SYNAPSE_LLAMA_WORKER_RUNTIME_DIR".to_string()
    } else {
        format!(
            "SYNAPSE_{}_WORKER_RUNTIME_DIR",
            sanitize_engine_env_component(engine)
        )
    }
}

#[must_use]
pub fn revisioned_machine_profile_hash(stable_profile_bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(MACHINE_PROFILE_HASH_REVISION.as_bytes());
    hasher.update([0]);
    hasher.update(stable_profile_bytes);
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floor_rejects_driver_boundary_and_accepts_it_at_the_floor() {
        assert_eq!(
            evaluate_cuda_floor(12039, 7, 5, None).refusal_code(),
            Some("owned_cuda_unsupported")
        );
        assert!(evaluate_cuda_floor(12040, 7, 5, None).is_supported());
    }

    #[test]
    fn floor_rejects_compute_boundary_and_accepts_above_it() {
        assert_eq!(
            evaluate_cuda_floor(12040, 7, 4, None).refusal_code(),
            Some("owned_cuda_unsupported")
        );
        assert!(evaluate_cuda_floor(12040, 7, 5, None).is_supported());
        assert!(evaluate_cuda_floor(12040, 8, 0, None).is_supported());
    }

    #[test]
    fn identity_is_canonical_and_contains_distribution_floor() {
        let identity = owned_cuda_engine_identity("qwen3", "f16", "kernel-v1");
        assert_eq!(identity.engine, OWNED_CUDA_ENGINE);
        assert_eq!(identity.build_flags["backend"], OWNED_CUDA_BACKEND);
        assert_eq!(identity.build_flags["ptx_virtual_arch"], "compute_75");
        assert_eq!(identity.build_flags["minimum_device_cc"], "7.5");
        assert_eq!(identity.build_flags["minimum_cuda_driver_api"], "12040");
        assert_eq!(identity.build_flags["risk_class"], "abort_capable");
    }

    #[test]
    fn environment_names_sanitize_every_non_alphanumeric_character() {
        assert_eq!(
            sanitize_engine_env_component("owned-cuda.v1/ptx"),
            "OWNED_CUDA_V1_PTX"
        );
        assert_eq!(
            worker_binary_env_var("owned-cuda"),
            "SYNAPSE_OWNED_CUDA_WORKER_BIN"
        );
        assert_eq!(
            worker_runtime_dir_env_var("owned-cuda"),
            "SYNAPSE_OWNED_CUDA_WORKER_RUNTIME_DIR"
        );
        assert_eq!(worker_binary_env_var("llama"), "SYNAPSE_LLAMA_WORKER_BIN");
        assert_eq!(
            worker_runtime_dir_env_var("llama"),
            "SYNAPSE_LLAMA_WORKER_RUNTIME_DIR"
        );
    }

    #[test]
    fn revisioned_hash_changes_when_revision_changes() {
        let first = revisioned_machine_profile_hash(b"profile");
        assert_eq!(first, revisioned_machine_profile_hash(b"profile"));
        assert_ne!(first, hex::encode(Sha256::digest(b"profile")));
    }
}
