use super::*;

#[cfg(unix)]
impl<S: AsyncRead + AsyncWrite + Unpin + Send + 'static> AneWorkerChannel<S> {
    pub(super) async fn load_serving(
        self: &Arc<Self>,
        supervisor: &AneResidencySupervisor,
        request: &WorkerRequest,
        pooling: synapse_core::WorkerPooling,
        normalize: bool,
        timeout: std::time::Duration,
    ) -> Result<AneServing<S>, AneResidencyError> {
        let WorkerRequest::Load { runtime_config, .. } = request else {
            return Err(AneResidencyError::Channel("expected LOAD request".into()));
        };
        // Transcription puts its pinned profile's frame limit in the request.
        // A worker reply cannot choose which metadata rules apply.
        let max_encoder_frames =
            if runtime_config.get("operation").map(String::as_str) == Some("transcribe") {
                Some(
                    runtime_config
                        .get("max_encoder_frames")
                        .and_then(|value| value.parse::<usize>().ok())
                        .filter(|&frames| frames > 0)
                        .ok_or_else(|| {
                            AneResidencyError::Channel("invalid LOAD encoder frame limit".into())
                        })?,
                )
            } else {
                None
            };

        // Timestamp the Neural Engine compile-cache check at the start of LOAD
        // so the scheduler does not overwrite observations from newer admissions.
        let check_started = tokio::time::Instant::now();
        let (response, _) = tokio::time::timeout(
            crate::worker_host::liveness::SETUP_IDLE,
            self.exchange(request, None),
        )
        .await
        .map_err(|_| AneResidencyError::Channel("LOAD deadline exceeded".into()))??;
        match response {
            WorkerResponse::Loaded {
                req_id,
                model_ref,
                dims,
                buckets: Some(buckets),
                ane_cached_shapes,
                ..
            } if request.req_id() == Some(req_id.as_str()) => {
                // If the worker did not check the cache, leave every rung
                // unknown so the scheduler still predicts compiles.
                if let Some(cached) = &ane_cached_shapes {
                    supervisor.record_compile_cache(&model_ref, &buckets, cached, check_started)?;
                }
                AneServing::new_with_frame_limit(
                    self.clone(),
                    supervisor.clone(),
                    AneServingMetadata {
                        model_ref,
                        buckets,
                        dims,
                        pooling,
                        normalize,
                        timeout,
                    },
                    max_encoder_frames,
                )
            }
            WorkerResponse::Err { code, msg, .. } => {
                Err(AneResidencyError::WorkerErr { code, msg })
            }
            other => Err(AneResidencyError::Channel(format!(
                "invalid LOAD response: {other:?}"
            ))),
        }
    }
}

impl<S: AsyncRead + AsyncWrite + Unpin + Send + 'static> AneServing<S> {
    pub fn new(
        channel: Arc<AneWorkerChannel<S>>,
        supervisor: AneResidencySupervisor,
        metadata: AneServingMetadata,
    ) -> Result<Self, AneResidencyError> {
        Self::new_with_frame_limit(channel, supervisor, metadata, None)
    }

    fn new_with_frame_limit(
        channel: Arc<AneWorkerChannel<S>>,
        supervisor: AneResidencySupervisor,
        metadata: AneServingMetadata,
        max_encoder_frames: Option<usize>,
    ) -> Result<Self, AneResidencyError> {
        let top = metadata.buckets.last().copied();
        let valid_for_operation = match max_encoder_frames {
            Some(max_frames) => top.is_some_and(|rung| rung >= max_frames),
            None => top == Some(8192) && metadata.dims > 0,
        };
        if !valid_for_operation
            || metadata.buckets.first().is_none_or(|n| *n == 0)
            || metadata.buckets.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(AneResidencyError::Channel(
                "LOAD returned invalid bucket or dimension metadata".into(),
            ));
        }
        Ok(Self {
            channel,
            supervisor,
            metadata,
            active_requests: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            preparing_packed: std::sync::atomic::AtomicBool::new(false),
            #[cfg(test)]
            recovery_gate: Mutex::new(None),
        })
    }
}
