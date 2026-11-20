use super::{
    artifact_invalid_error, transcribe_host::TranscribeHostError, worker_host, WireOperationError,
};
use serde_json::json;
use synapse_core::{EngineError, EngineErrorStage, StableError};

// WorkerCrash denotes quarantine in the generic host. Ordinary transport
// failures use the operation's inference or load stage.
pub(super) fn worker_error_to_wire(
    error: EngineError,
    catalog_lane: Option<&str>,
) -> WireOperationError {
    if error.stage == EngineErrorStage::WorkerCrash {
        let retry_after_ms = error
            .retry_after_ms
            .unwrap_or(worker_host::QUARANTINE_COOLDOWN.as_millis() as u64);
        let mut wire = WireOperationError::from_stable(
            StableError::engine_crashed(Some(retry_after_ms)),
            error.message,
        );
        wire.details = Some(match catalog_lane {
            Some(lane_id) => json!({ "lane_id": lane_id, "quarantined": true }),
            None => json!({ "quarantined": true }),
        });
        return wire;
    }
    engine_error_to_wire(error)
}

pub(super) fn engine_error_to_wire(error: EngineError) -> WireOperationError {
    if error.stage == EngineErrorStage::DeviceMemoryExhausted {
        let mut wire = WireOperationError::from_stable(
            StableError::engine_crashed(error.retry_after_ms),
            error.message,
        );
        wire.details = Some(json!({"reason": "device_memory_exhausted"}));
        return wire;
    }
    if error.stage == EngineErrorStage::Load {
        return artifact_invalid_error(error.message);
    }
    WireOperationError::from_stable(
        StableError::engine_crashed(error.retry_after_ms),
        error.message,
    )
}

pub(super) fn transcribe_load_error_to_wire(error: TranscribeHostError) -> WireOperationError {
    if let TranscribeHostError::Worker(
        worker @ worker_host::WorkerHostError::DeviceMemoryExhausted { .. },
    ) = &error
    {
        return engine_error_to_wire(worker.to_engine_error(EngineErrorStage::Load));
    }
    WireOperationError::from_stable(StableError::engine_crashed(Some(1000)), error.to_string())
}
