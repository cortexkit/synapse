use std::time::Duration;

use synapse_core::{
    read_json, read_raw, write_json, write_raw, WorkerProgressPhase, WorkerRequest, WorkerResponse,
};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::time::{timeout_at, Instant};

use super::{ensure_req_id, WorkerHostError};

// INFERENCE_IDLE bounds silence between completed-work signals, not total time.
// A completed transformer layer renews it even inside a slow 8,192-token row;
// the window leaves headroom for backend work while still bounding a stalled
// stream. Only validated completed work renews it, never a heartbeat.
// Supported hardware needing a larger gap requires a measured code change.
pub(super) const INFERENCE_IDLE: Duration = Duration::from_secs(120);
// SETUP_IDLE allows 600s of silence inside native load/setup/compiler calls
// that cannot report mid-call. Each completed layer compile renews that timer.
// The larger setup window leaves headroom for opaque native calls and cold
// compilers. It bounds the gap between completed setup steps, not their total
// duration; the caller's absolute deadline is a separate limit.
pub(super) const SETUP_IDLE: Duration = Duration::from_secs(600);

pub(super) async fn exchange<S: AsyncRead + AsyncWrite + Unpin>(
    stream: &mut S,
    request: &WorkerRequest,
    raw: Option<&[u8]>,
    max_frame: u32,
    cuda: bool,
    test_idle: Option<(Duration, Duration)>,
) -> Result<(WorkerResponse, Option<Vec<u8>>), WorkerHostError> {
    let inference = matches!(
        request,
        WorkerRequest::EmbedBatch { .. }
            | WorkerRequest::EmbedPacked { .. }
            | WorkerRequest::Rerank { .. }
            | WorkerRequest::RerankSequences { .. }
            | WorkerRequest::Generate { .. }
    );
    let setup = matches!(
        request,
        WorkerRequest::Load { .. }
            | WorkerRequest::AneAdmitShape { .. }
            | WorkerRequest::AneAdmitPacked { .. }
            | WorkerRequest::AneEvictShape { .. }
            | WorkerRequest::AneEvictPacked { .. }
    );
    let mut phase = if setup || (inference && cuda) {
        WorkerProgressPhase::Setup
    } else if inference {
        WorkerProgressPhase::Inference
    } else {
        WorkerProgressPhase::Control
    };
    #[cfg(not(test))]
    let _ = test_idle;
    let window = |phase| {
        #[cfg(test)]
        if let Some((inference, setup)) = test_idle {
            return if phase == WorkerProgressPhase::Setup {
                setup
            } else {
                inference
            };
        }
        if phase == WorkerProgressPhase::Setup {
            SETUP_IDLE
        } else {
            INFERENCE_IDLE
        }
    };
    let req_id = request
        .req_id()
        .ok_or_else(|| WorkerHostError::Protocol("exchange has no request id".into()))?;
    let started = Instant::now();
    let mut last = started;
    let mut idle_until = last + window(phase);
    let mut completed = 0;
    let mut max_gap = Duration::ZERO;
    let fault = |completed, phase| {
        WorkerHostError::Protocol(format!(
        "no-progress hang: {} req_id={req_id} completed={completed} phase={phase:?} idle_window_ms={}",
        request.wire_type(), window(phase).as_millis()))
    };
    timeout_at(idle_until, async {
        write_json(stream, request, max_frame).await?;
        if let Some(raw) = raw {
            write_raw(stream, raw, max_frame).await?;
        }
        Ok::<_, WorkerHostError>(())
    })
    .await
    .map_err(|_| fault(completed, phase))??;
    loop {
        let response: WorkerResponse = timeout_at(idle_until, read_json(stream, max_frame))
            .await
            .map_err(|_| fault(completed, phase))??;
        ensure_req_id(req_id, response.req_id().unwrap_or_default())?;
        if let WorkerResponse::Progress {
            completed: next,
            phase: next_phase,
            ..
        } = response
        {
            if next <= completed
                || !(next_phase == phase
                    || (cuda
                        && inference
                        && phase == WorkerProgressPhase::Setup
                        && next_phase == WorkerProgressPhase::Inference))
            {
                return Err(WorkerHostError::Protocol(
                    "nonmonotonic progress or invalid phase".into(),
                ));
            }
            max_gap = max_gap.max(last.elapsed());
            tracing::info!(target: "worker_liveness", operation = request.wire_type(), req_id,
                phase = ?phase, gap_ms = last.elapsed().as_secs_f64() * 1000.0,
                completed = next, "completed work received");
            completed = next;
            phase = next_phase;
            last = Instant::now();
            idle_until = last + window(phase);
            continue;
        }
        let valid = matches!(
            (request, &response),
            (_, WorkerResponse::Err { .. })
                | (WorkerRequest::Load { .. }, WorkerResponse::Loaded { .. })
                | (
                    WorkerRequest::EmbedBatch { .. } | WorkerRequest::EmbedPacked { .. },
                    WorkerResponse::Vectors { .. }
                )
                | (
                    WorkerRequest::Rerank { .. } | WorkerRequest::RerankSequences { .. },
                    WorkerResponse::Scores { .. }
                )
                | (WorkerRequest::Generate { .. }, WorkerResponse::Text { .. })
                | (WorkerRequest::Ping { .. }, WorkerResponse::Pong { .. })
                | (
                    WorkerRequest::Unload { .. },
                    WorkerResponse::Unloaded { .. }
                )
                | (
                    WorkerRequest::AneAdmitShape { .. } | WorkerRequest::AneAdmitPacked { .. },
                    WorkerResponse::Admitted { .. }
                )
                | (
                    WorkerRequest::AneEvictShape { .. } | WorkerRequest::AneEvictPacked { .. },
                    WorkerResponse::Evicted { .. }
                )
        );
        if !valid {
            return Err(WorkerHostError::Protocol(
                "unexpected terminal response".into(),
            ));
        }
        let raw = if response.carries_raw_frame() {
            Some(
                timeout_at(idle_until, read_raw(stream, max_frame))
                    .await
                    .map_err(|_| fault(completed, phase))??,
            )
        } else {
            None
        };
        let expected_bytes = match (request, &response) {
            (
                WorkerRequest::EmbedBatch { items, .. } | WorkerRequest::EmbedPacked { items, .. },
                WorkerResponse::Vectors { n, dims, .. },
            ) if *n == items.len() && *dims > 0 => {
                n.checked_mul(*dims).and_then(|n| n.checked_mul(4))
            }
            (WorkerRequest::Rerank { candidates, .. }, WorkerResponse::Scores { .. }) => {
                candidates.len().checked_mul(4)
            }
            (WorkerRequest::RerankSequences { sequences, .. }, WorkerResponse::Scores { .. }) => {
                sequences.len().checked_mul(4)
            }
            (_, WorkerResponse::Vectors { .. }) => {
                return Err(WorkerHostError::Protocol(
                    "vector row count or dimensions mismatch".into(),
                ))
            }
            _ => None,
        };
        if response.carries_raw_frame()
            && (expected_bytes.is_none() || raw.as_ref().map(Vec::len) != expected_bytes)
        {
            return Err(WorkerHostError::Protocol(
                "terminal raw payload size mismatch".into(),
            ));
        }
        max_gap = max_gap.max(last.elapsed());
        tracing::info!(target: "worker_liveness", operation = request.wire_type(), req_id,
            progress_units = completed, max_gap_ms = max_gap.as_secs_f64() * 1000.0,
            phase = ?phase, gap_ms = last.elapsed().as_secs_f64() * 1000.0,
            elapsed_ms = started.elapsed().as_secs_f64() * 1000.0, "worker exchange drained");
        return Ok((response, raw));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use synapse_core::{encode_f32_frame, encode_i32_frame, WorkerPooling, WorkerTokenItem};

    fn request() -> WorkerRequest {
        WorkerRequest::EmbedBatch {
            req_id: "current".into(),
            model_ref: "model".into(),
            pooling: WorkerPooling::Mean,
            normalize: true,
            items: vec![WorkerTokenItem {
                id: "0".into(),
                n_tokens: 1,
            }],
        }
    }

    async fn frames(
        frames: Vec<WorkerResponse>,
        delay: Duration,
    ) -> Result<(WorkerResponse, Option<Vec<u8>>), WorkerHostError> {
        let (mut module, mut worker) = tokio::io::duplex(4096);
        let peer = tokio::spawn(async move {
            let _: WorkerRequest = read_json(&mut worker, 4096).await.unwrap();
            read_raw(&mut worker, 4096).await.unwrap();
            for frame in frames {
                tokio::time::sleep(delay).await;
                if write_json(&mut worker, &frame, 4096).await.is_err() {
                    return;
                }
            }
            if write_json(
                &mut worker,
                &WorkerResponse::Vectors {
                    req_id: "current".into(),
                    dims: 1,
                    n: 1,
                },
                4096,
            )
            .await
            .is_ok()
            {
                let _ = write_raw(&mut worker, &encode_f32_frame(&[9.0]), 4096).await;
            }
        });
        let result = exchange(
            &mut module,
            &request(),
            Some(&encode_i32_frame(&[9])),
            4096,
            false,
            Some((Duration::from_millis(400), Duration::from_millis(400))),
        )
        .await;
        drop(module);
        peer.await.unwrap();
        result
    }

    fn progress(id: &str, completed: u64, phase: WorkerProgressPhase) -> WorkerResponse {
        WorkerResponse::Progress {
            req_id: id.into(),
            completed,
            phase,
        }
    }

    #[tokio::test]
    async fn validated_progress_renews_silence_window_not_total_budget() {
        let work = (1..=8)
            .map(|n| progress("current", n, WorkerProgressPhase::Inference))
            .collect();
        let (_, raw) = frames(work, Duration::from_millis(80)).await.unwrap();
        assert_eq!(raw.unwrap(), encode_f32_frame(&[9.0]));
    }

    #[tokio::test]
    async fn stale_duplicate_zero_or_invalid_phase_progress_is_a_protocol_fault() {
        for work in [
            vec![progress("old", 1, WorkerProgressPhase::Inference)],
            vec![progress("current", 0, WorkerProgressPhase::Inference)],
            vec![
                progress("current", 1, WorkerProgressPhase::Inference),
                progress("current", 1, WorkerProgressPhase::Inference),
            ],
            vec![progress("current", 1, WorkerProgressPhase::Setup)],
        ] {
            assert!(matches!(
                frames(work, Duration::ZERO).await,
                Err(WorkerHostError::Protocol(_))
            ));
        }
    }
}
