use std::io::{Read, Write};

use anyhow::{ensure, Context, Result};
use synapse_core::worker_framing_sync::{read_frame, write_frame, write_json_frame};
use synapse_core::{
    decode_f32_frame, encode_decode_trace_frame, encode_f32_frame, encode_transcript_frame,
    TranscribeOutput, TranscribeRequest, WorkerRequest, WorkerResponse,
};

use super::{error_response, WorkerState};

#[cfg(all(test, feature = "cuda"))]
mod hardware;

#[cfg(test)]
mod grading;

#[cfg(feature = "cuda")]
pub(super) fn load_transcribe(
    state: &mut WorkerState,
    req_id: String,
    path: String,
    digest: String,
    format: String,
    config: std::collections::BTreeMap<String, String>,
    max_frame: u32,
) -> WorkerResponse {
    use synapse_core::{
        RuntimeConfig, TranscribeEngine, TranscribeRevision, ValidatedArtifact,
        CUDA_TRANSCRIBE_KERNEL_REVISION,
    };
    let result = (|| -> Result<_> {
        let profile = &synapse_engine_cuda::manifest::MANIFEST["profiles"]
            [config.get("profile").context("model_unsupported")?];
        let ms = profile["transcribe"]["max_duration_ms"]
            .as_u64()
            .context("invalid_max_encoder_frames")?;
        let maximum = config
            .get("max_encoder_frames")
            .map(|v| v.parse::<usize>())
            .transpose()?
            .unwrap_or((ms as usize * 16 / 160).div_ceil(8));
        synapse_engine_cuda::transcribe::validate_frame_envelope(maximum, max_frame)?;
        let mut runtime = RuntimeConfig { values: config };
        runtime.values.insert("model_path".into(), path);
        let mut engine = synapse_engine_cuda::transcribe::OwnedCudaTranscribeEngine::default();
        let started = std::time::Instant::now();
        let engine_model = engine
            .load(&ValidatedArtifact { digest, format }, &runtime)
            .map_err(|e| anyhow::anyhow!(e.message))?;
        let model_ref = engine_model.model_id.clone();
        state.loaded = Some(super::LoadedModel {
            model_ref: model_ref.clone(),
            engine: super::LoadedEngine::Transcribe {
                engine,
                engine_model,
            },
        });
        let mut loaded = WorkerResponse::loaded(
            req_id.clone(),
            model_ref,
            0,
            started.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
            None,
            None,
        );
        if let WorkerResponse::Loaded {
            transcribe_revision,
            ..
        } = &mut loaded
        {
            *transcribe_revision = Some(TranscribeRevision {
                kernel_revision: CUDA_TRANSCRIBE_KERNEL_REVISION.into(),
                decode_rule: synapse_speech::DECODE_RULE_REVISION.into(),
            });
        }
        Ok(loaded)
    })();
    result.unwrap_or_else(|e| {
        let message = e.to_string();
        let code = match message.as_str() {
            "transcribe_frame_limit"
            | "invalid_max_encoder_frames"
            | "package_digest_mismatch"
            | "model_unsupported" => message.as_str(),
            _ => "artifact_invalid",
        };
        error_response(Some(req_id), code, &message)
    })
}

pub(super) fn serve_transcribe<S: Read + Write>(
    stream: &mut S,
    max_frame: u32,
    state: &WorkerState,
    request: WorkerRequest,
) -> Result<()> {
    let raw = read_frame(stream, max_frame).context("read transcription mel frame")?;
    // A count mismatch is a protocol fault. Do not continue on a possibly
    // desynchronized connection or interpret a partial tensor as audio.
    request
        .validate_raw_frame(&raw)
        .context("transcription mel frame length")?;
    let WorkerRequest::Transcribe {
        req_id,
        model_ref,
        mel_frames,
        valid_mel_frames,
        observe,
    } = request
    else {
        unreachable!()
    };
    let result = (|| -> Result<TranscribeOutput> {
        let mel = decode_f32_frame(&raw)?;
        let request = TranscribeRequest {
            mel,
            mel_frames,
            valid_mel_frames,
            observe,
        };
        synapse_engine_cuda::transcribe::input_shape(&request, 8192)?;
        let model = state
            .loaded
            .as_ref()
            .filter(|m| m.model_ref == model_ref)
            .context("model_not_loaded")?;
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (model, request);
            anyhow::bail!("backend_missing");
        }
        #[cfg(feature = "cuda")]
        {
            let super::LoadedEngine::Transcribe {
                engine,
                engine_model,
            } = &model.engine
            else {
                anyhow::bail!("operation_mismatch")
            };
            let mut writer =
                synapse_core::worker_framing_sync::ProgressWriter::new(stream, &req_id, max_frame);
            engine.transcribe_with_progress(engine_model, request, &mut || {
                writer
                    .completed(synapse_core::WorkerProgressPhase::Inference)
                    .map_err(Into::into)
            })
        }
    })();
    match result {
        Ok(output) => write_transcription(stream, max_frame, &req_id, output),
        Err(error) => {
            let message = error.to_string();
            let code = if message.starts_with("invalid_request") {
                "invalid_request"
            } else if message.starts_with("duration_limit") {
                "duration_limit"
            } else {
                match message.as_str() {
                    "model_not_loaded" | "backend_missing" | "operation_mismatch" => {
                        message.as_str()
                    }
                    _ => "engine_crashed",
                }
            };
            write_json_frame(
                stream,
                &error_response(Some(req_id), code, &message),
                max_frame,
            )?;
            Ok(())
        }
    }
}

fn write_transcription<S: Write>(
    stream: &mut S,
    max_frame: u32,
    req_id: &str,
    output: TranscribeOutput,
) -> Result<()> {
    let triples = encode_transcript_frame(&output.triples);
    ensure!(
        triples.len() <= max_frame as usize,
        "transcription output exceeds max_frame"
    );
    let observation = output.observation.map(|o| {
        (
            encode_f32_frame(&o.encoder_output),
            encode_decode_trace_frame(&o.decode_trace),
            o.decode_trace.len(),
        )
    });
    if let Some((encoder, trace, _)) = &observation {
        ensure!(
            encoder.len() == output.valid_encoder_frames * 4096,
            "invalid encoder output shape"
        );
        ensure!(
            encoder.len() <= max_frame as usize && trace.len() <= max_frame as usize,
            "transcription observation exceeds max_frame"
        );
    }
    if let Some((encoder, trace, steps)) = observation {
        write_json_frame(
            stream,
            &WorkerResponse::EncoderOutput {
                req_id: req_id.into(),
                frames: output.valid_encoder_frames,
            },
            max_frame,
        )?;
        write_frame(stream, &encoder, max_frame)?;
        write_json_frame(
            stream,
            &WorkerResponse::DecodeTrace {
                req_id: req_id.into(),
                steps,
            },
            max_frame,
        )?;
        write_frame(stream, &trace, max_frame)?;
    }
    write_json_frame(
        stream,
        &WorkerResponse::Transcript {
            req_id: req_id.into(),
            valid_encoder_frames: output.valid_encoder_frames,
            count: output.triples.len(),
        },
        max_frame,
    )?;
    write_frame(stream, &triples, max_frame)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use synapse_core::worker_framing_sync::read_json_frame;
    use synapse_core::{
        DecodeTraceRecord, TranscribeObservation, TranscriptRecord, DEFAULT_MAX_FRAME_BYTES,
    };
    fn output(observe: bool) -> TranscribeOutput {
        TranscribeOutput {
            valid_encoder_frames: 1,
            triples: vec![TranscriptRecord {
                token_id: 4,
                frame_index: 0,
                duration: 2,
            }],
            observation: observe.then(|| TranscribeObservation {
                encoder_output: vec![0.25; 1024],
                decode_trace: vec![DecodeTraceRecord {
                    frame_index: 0,
                    token_top1: 4,
                    token_top2: 2,
                    duration_top1: 2,
                    duration_top2: 1,
                    token_logit1: 3.,
                    token_logit2: 2.,
                    duration_logit1: 2.,
                    duration_logit2: 1.,
                    flags: 1,
                }],
            }),
        }
    }
    #[test]
    fn transcribe_response_order_and_raw_lengths() {
        for observe in [false, true] {
            let mut bytes = Vec::new();
            write_transcription(&mut bytes, DEFAULT_MAX_FRAME_BYTES, "t", output(observe)).unwrap();
            let mut stream = Cursor::new(bytes);
            if observe {
                let r: WorkerResponse =
                    read_json_frame(&mut stream, DEFAULT_MAX_FRAME_BYTES).unwrap();
                assert!(matches!(r, WorkerResponse::EncoderOutput { frames: 1, .. }));
                let raw = read_frame(&mut stream, DEFAULT_MAX_FRAME_BYTES).unwrap();
                r.validate_raw_frame(&raw).unwrap();
                assert_eq!(raw.len(), 4096);
                let r: WorkerResponse =
                    read_json_frame(&mut stream, DEFAULT_MAX_FRAME_BYTES).unwrap();
                assert!(matches!(r, WorkerResponse::DecodeTrace { steps: 1, .. }));
                let raw = read_frame(&mut stream, DEFAULT_MAX_FRAME_BYTES).unwrap();
                r.validate_raw_frame(&raw).unwrap();
                assert_eq!(raw.len(), 40);
            }
            let r: WorkerResponse = read_json_frame(&mut stream, DEFAULT_MAX_FRAME_BYTES).unwrap();
            assert!(matches!(
                r,
                WorkerResponse::Transcript {
                    valid_encoder_frames: 1,
                    count: 1,
                    ..
                }
            ));
            let raw = read_frame(&mut stream, DEFAULT_MAX_FRAME_BYTES).unwrap();
            assert_eq!(
                synapse_core::decode_transcript_frame(&raw, 1).unwrap(),
                output(false).triples
            );
            assert_eq!(stream.position(), stream.get_ref().len() as u64);
        }
    }
    #[test]
    fn transcribe_response_refuses_oversize_before_any_observation() {
        let mut bytes = Vec::new();
        assert!(write_transcription(&mut bytes, 100, "t", output(true)).is_err());
        assert!(bytes.is_empty());
    }

    struct Duplex {
        input: Cursor<Vec<u8>>,
        output: Vec<u8>,
    }
    impl Read for Duplex {
        fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
            self.input.read(b)
        }
    }
    impl Write for Duplex {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.output.extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    #[test]
    fn transcribe_mel_length_fault_closes_before_any_response() {
        for (frames, raw, maximum) in [
            (1, vec![0; 511], DEFAULT_MAX_FRAME_BYTES),
            (usize::MAX, vec![], DEFAULT_MAX_FRAME_BYTES),
            (1, vec![0; 512], 64),
        ] {
            let mut bytes = Vec::new();
            write_frame(&mut bytes, &raw, DEFAULT_MAX_FRAME_BYTES).unwrap();
            let mut stream = Duplex {
                input: Cursor::new(bytes),
                output: Vec::new(),
            };
            let request = WorkerRequest::Transcribe {
                req_id: "bad".into(),
                model_ref: "m".into(),
                mel_frames: frames,
                valid_mel_frames: 1,
                observe: false,
            };
            assert!(
                serve_transcribe(&mut stream, maximum, &WorkerState::default(), request).is_err()
            );
            assert!(stream.output.is_empty());
        }
    }
    #[test]
    fn transcribe_nonfinite_input_is_refused_before_model_lookup() {
        let mut raw = vec![0.; 128];
        raw[3] = f32::NAN;
        let mut bytes = Vec::new();
        write_frame(&mut bytes, &encode_f32_frame(&raw), DEFAULT_MAX_FRAME_BYTES).unwrap();
        let mut stream = Duplex {
            input: Cursor::new(bytes),
            output: Vec::new(),
        };
        let request = WorkerRequest::Transcribe {
            req_id: "nan".into(),
            model_ref: "m".into(),
            mel_frames: 1,
            valid_mel_frames: 1,
            observe: false,
        };
        serve_transcribe(
            &mut stream,
            DEFAULT_MAX_FRAME_BYTES,
            &WorkerState::default(),
            request,
        )
        .unwrap();
        let response: WorkerResponse =
            read_json_frame(&mut Cursor::new(stream.output), DEFAULT_MAX_FRAME_BYTES).unwrap();
        assert!(matches!(response, WorkerResponse::Err { code, .. } if code == "invalid_request"));
    }
}
