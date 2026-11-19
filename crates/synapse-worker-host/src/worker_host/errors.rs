use super::{ane_residency, WorkerHostError};
use synapse_core::{
    EngineError, EngineErrorStage, EngineRiskClass, VULKAN_LOAD_DEVICE_MEMORY_EXHAUSTED,
};

impl WorkerHostError {
    pub(super) fn load_refusal(code: String, msg: String) -> Self {
        if code == VULKAN_LOAD_DEVICE_MEMORY_EXHAUSTED {
            Self::DeviceMemoryExhausted { msg }
        } else {
            Self::WorkerErr { code, msg }
        }
    }

    pub fn to_engine_error(&self, stage: EngineErrorStage) -> EngineError {
        EngineError {
            stage: match self {
                Self::Quarantined { .. } => EngineErrorStage::WorkerCrash,
                Self::DeviceMemoryExhausted { .. } => EngineErrorStage::DeviceMemoryExhausted,
                _ => stage,
            },
            risk_class: EngineRiskClass::AbortCapable,
            message: self.to_string(),
            retry_after_ms: match self {
                Self::WorkerErr { code, .. }
                    if code == ane_residency::ERR_ANE_RESOURCES_EXHAUSTED =>
                {
                    Some(ane_residency::ANE_RESOURCES_RETRY_AFTER_MS)
                }
                Self::Quarantined { retry_after_ms, .. } => *retry_after_ms,
                Self::WorkerErr { .. } | Self::HelloRefused { .. } => None,
                _ => Some(250),
            },
            safe_to_retry_same_request: matches!(
                self,
                Self::EngineCrashed { .. }
                    | Self::DeviceMemoryExhausted { .. }
                    | Self::Quarantined {
                        retry_after_ms: Some(_),
                        ..
                    }
            ) || matches!(self, Self::WorkerErr {code, ..} if code == ane_residency::ERR_ANE_RESOURCES_EXHAUSTED),
        }
    }

    /// The typed code of a worker `ERR` or of a HELLO refusal.
    pub fn code(&self) -> Option<&str> {
        match self {
            Self::WorkerErr { code, .. } | Self::HelloRefused { code, .. } => Some(code),
            Self::DeviceMemoryExhausted { .. } => Some(VULKAN_LOAD_DEVICE_MEMORY_EXHAUSTED),
            _ => None,
        }
    }
}
