//! In-process Parakeet TDT: GPU subsampling, full-attention encoder and encoder
//! projection, followed by the shared speech core's bounded greedy loop.

use std::collections::BTreeMap;
use synapse_core::{
    EngineError, EngineErrorStage, EngineIdentity, EngineRiskClass, LoadedModel, RuntimeConfig,
    TranscribeEngine, TranscribeOutput, TranscribeRequest, ValidatedArtifact,
    METAL_TRANSCRIBE_KERNEL_REVISION,
};

#[cfg(target_os = "macos")]
#[path = "transcribe_metal.rs"]
mod metal;

pub const ENCODER_RUNGS: [usize; 7] = [128, 256, 512, 1024, 2048, 4096, 8192];
/// Full-attention envelope verified on Metal against the SHA256-pinned Parakeet
/// duration cases in `bench/parity/fixtures/index.json`.
pub const CERTIFIED_MAX_DURATION_MS: u64 = 600_000;
pub const CERTIFIED_MAX_ENCODER_FRAMES: usize = 7500;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Progress {
    EncoderLayersCompleted(usize),
    DecodedFrames(usize),
}

pub fn engine_identity() -> EngineIdentity {
    EngineIdentity {
        engine: "owned-metal".into(),
        version: crate::ENGINE_VERSION.into(),
        build_flags: BTreeMap::from([
            ("backend".into(), "metal-mpsgraph".into()),
            ("family".into(), "parakeet-tdt".into()),
            ("dtype".into(), "f32".into()),
            (
                "kernel_revision".into(),
                METAL_TRANSCRIBE_KERNEL_REVISION.into(),
            ),
            ("risk_class".into(), "abort_safe".into()),
        ]),
    }
}

fn error(stage: EngineErrorStage, message: impl ToString) -> EngineError {
    EngineError {
        stage,
        risk_class: EngineRiskClass::AbortSafe,
        message: message.to_string(),
        retry_after_ms: None,
        safe_to_retry_same_request: false,
    }
}

fn validate_profile(cfg: &RuntimeConfig, manifest: &serde_json::Value) -> Result<(), EngineError> {
    let profile = cfg
        .values
        .get("profile")
        .map(String::as_str)
        .unwrap_or("parakeet-tdt-0.6b-v3.owned-metal");
    let entry = &manifest["profiles"][profile];
    if entry["lane"] != "owned-metal" || entry["model"] != "parakeet-tdt-0.6b-v3" {
        return Err(error(
            EngineErrorStage::Load,
            "model_unsupported: Metal transcribe profile",
        ));
    }
    // The unconverted checkpoint and all Metal graph arithmetic are f32. A
    // fingerprint must not advertise a different storage or compute precision.
    if entry["compute_dtype"] != "f32" || entry["storage_dtype"] != "f32" {
        return Err(error(
            EngineErrorStage::Load,
            "Metal transcribe requires profile compute_dtype=f32 and storage_dtype=f32",
        ));
    }
    Ok(())
}

/// Validate before touching the accelerator. An encoder rung is a preallocated
/// frame capacity. The front end's one trailing frame is zero; dropping that zero
/// at an exact capacity boundary preserves zero convolution padding. Like the
/// transformers 5.16.1 Parakeet encoder, we mask after each subsampling convolution.
#[cfg(any(target_os = "macos", test))]
fn shape(request: &TranscribeRequest, maximum: usize) -> Result<(usize, usize), EngineError> {
    let fail = |s| error(EngineErrorStage::Inference, s);
    if request.valid_mel_frames == 0
        || request.valid_mel_frames > request.mel_frames
        || request.mel_frames.checked_mul(128) != Some(request.mel.len())
        || request.mel.iter().any(|x| !x.is_finite())
        || request.mel[request.valid_mel_frames * 128..]
            .iter()
            .any(|&x| x != 0.0)
    {
        return Err(fail("invalid normalized mel shape, padding or values"));
    }
    let valid = request.valid_mel_frames.div_ceil(8);
    if valid > maximum {
        return Err(fail("valid_encoder_frames exceeds max_encoder_frames"));
    }
    let rung = ENCODER_RUNGS
        .iter()
        .copied()
        .find(|&r| r >= valid)
        .ok_or_else(|| fail("encoder length exceeds ladder"))?;
    Ok((valid, rung))
}

#[cfg(any(target_os = "macos", test))]
fn decode<J: synapse_speech::Joint>(
    joint: &mut J,
    valid: usize,
    observe: bool,
    mut encoder: Vec<f32>,
    progress: &mut impl FnMut(Progress),
) -> synapse_speech::Result<TranscribeOutput> {
    let decoded = synapse_speech::greedy(joint, valid, observe, |n| {
        progress(Progress::DecodedFrames(n))
    })?;
    let triples = decoded
        .triples
        .iter()
        .map(|t| synapse_core::TranscriptRecord {
            token_id: t[0] as i32,
            frame_index: t[1] as i32,
            duration: t[2] as i32,
        })
        .collect();
    let observation = observe.then(|| {
        encoder.truncate(valid * 1024);
        synapse_core::TranscribeObservation {
            encoder_output: encoder,
            decode_trace: decoded
                .trace
                .iter()
                .map(|t| synapse_core::DecodeTraceRecord {
                    frame_index: t.frame_index as i32,
                    token_top1: t.token_top1 as i32,
                    token_top2: t.token_top2 as i32,
                    duration_top1: t.duration_top1 as i32,
                    duration_top2: t.duration_top2 as i32,
                    token_logit1: t.token_logit1,
                    token_logit2: t.token_logit2,
                    duration_logit1: t.duration_logit1,
                    duration_logit2: t.duration_logit2,
                    flags: u32::from(t.emitted) | (u32::from(t.cap_hit) << 1),
                })
                .collect(),
        }
    });
    Ok(TranscribeOutput {
        valid_encoder_frames: valid,
        triples,
        observation,
    })
}

#[derive(Default)]
pub struct OwnedMetalTranscribeEngine {
    #[cfg(target_os = "macos")]
    models: std::collections::HashMap<String, std::sync::Mutex<metal::Model>>,
    #[cfg(target_os = "macos")]
    next_model: u64,
}

impl OwnedMetalTranscribeEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Progress is emitted synchronously for completed layer groups and decoded
    /// frame boundaries only. The module can connect this callback to its watchdog.
    pub fn transcribe_with_progress(
        &self,
        model: &LoadedModel,
        request: TranscribeRequest,
        mut progress: impl FnMut(Progress),
    ) -> Result<TranscribeOutput, EngineError> {
        #[cfg(target_os = "macos")]
        {
            let mut model = self
                .models
                .get(&model.model_id)
                .ok_or_else(|| error(EngineErrorStage::Inference, "unknown transcribe model"))?
                .lock()
                .map_err(|_| {
                    error(
                        EngineErrorStage::Inference,
                        "transcribe model mutex poisoned",
                    )
                })?;
            let (valid, rung) = shape(&request, model.maximum)?;
            model
                .run(request, valid, rung, &mut progress)
                .map_err(|e| error(EngineErrorStage::Inference, e))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (model, request, &mut progress);
            Err(error(EngineErrorStage::Inference, "Metal requires macOS"))
        }
    }
}

impl TranscribeEngine for OwnedMetalTranscribeEngine {
    fn identity(&self) -> EngineIdentity {
        engine_identity()
    }

    fn load(
        &mut self,
        artifact: &ValidatedArtifact,
        cfg: &RuntimeConfig,
    ) -> Result<LoadedModel, EngineError> {
        if artifact.format != "safetensors-package" {
            return Err(error(
                EngineErrorStage::Load,
                "Metal transcribe requires safetensors-package",
            ));
        }
        let manifest = serde_json::from_slice(include_bytes!("../../../bench/parity/models.json"))
            .map_err(|e| error(EngineErrorStage::Load, e))?;
        validate_profile(cfg, &manifest)?;
        #[cfg(target_os = "macos")]
        {
            let model = metal::Model::load(cfg).map_err(|e| error(EngineErrorStage::Load, e))?;
            let model_id = format!("owned-metal:parakeet-tdt:{}", self.next_model);
            self.next_model += 1;
            self.models
                .insert(model_id.clone(), std::sync::Mutex::new(model));
            Ok(LoadedModel { model_id })
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = cfg;
            Err(error(EngineErrorStage::Load, "Metal requires macOS"))
        }
    }

    fn transcribe(
        &self,
        model: &LoadedModel,
        request: TranscribeRequest,
    ) -> Result<TranscribeOutput, EngineError> {
        self.transcribe_with_progress(model, request, |_| {})
    }

    fn unload(&mut self, model: &LoadedModel) {
        #[cfg(target_os = "macos")]
        self.models.remove(&model.model_id);
        #[cfg(not(target_os = "macos"))]
        let _ = model;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request(valid: usize) -> TranscribeRequest {
        TranscribeRequest {
            mel: vec![0.0; (valid + 1) * 128],
            mel_frames: valid + 1,
            valid_mel_frames: valid,
            observe: false,
        }
    }
    #[test]
    fn transcribe_smallest_rung_and_exact_boundary() {
        for rung in ENCODER_RUNGS {
            assert_eq!(shape(&request(rung * 8), 8192).unwrap(), (rung, rung));
            if rung < 8192 {
                assert_eq!(
                    shape(&request(rung * 8 + 1), 8192).unwrap(),
                    (rung + 1, rung * 2)
                );
            }
        }
    }
    #[test]
    fn transcribe_rejects_bad_shapes_before_gpu_work() {
        assert!(shape(&request(1025), 128).is_err());
        let mut r = request(10);
        r.mel[1280] = 1.0;
        assert!(shape(&r, 8192).is_err());
        r.mel[1280] = 0.0;
        r.mel[0] = f32::NAN;
        assert!(shape(&r, 8192).is_err());
        r.mel.clear();
        assert!(shape(&r, 8192).is_err());
    }
    #[test]
    fn transcribe_identity_is_not_an_embed_identity() {
        let id = engine_identity();
        assert_eq!(
            id.build_flags["kernel_revision"],
            METAL_TRANSCRIBE_KERNEL_REVISION
        );
        assert!(!id.build_flags.contains_key("graph_revision"));
        assert_eq!(id.build_flags["family"], "parakeet-tdt");
    }

    #[test]
    fn transcribe_load_refuses_profile_precision_it_does_not_compute() {
        let manifest: serde_json::Value =
            serde_json::from_slice(include_bytes!("../../../bench/parity/models.json")).unwrap();
        let profile = "parakeet-tdt-0.6b-v3.owned-metal";
        let cfg = RuntimeConfig {
            values: [("profile".into(), profile.into())].into(),
        };
        validate_profile(&cfg, &manifest).unwrap();
        for field in ["compute_dtype", "storage_dtype"] {
            for invalid in [serde_json::json!("f16"), serde_json::Value::Null] {
                let mut wrong = manifest.clone();
                wrong["profiles"][profile][field] = invalid;
                let error = validate_profile(&cfg, &wrong).unwrap_err();
                assert_eq!(error.stage, EngineErrorStage::Load);
                assert!(error
                    .message
                    .contains("compute_dtype=f32 and storage_dtype=f32"));
            }
        }
    }

    struct Stub {
        calls: usize,
        at_cap_advances: bool,
        tokens: Vec<f32>,
        durations: [f32; 5],
        frames: Vec<usize>,
    }
    impl synapse_speech::Joint for Stub {
        fn reset(&mut self) -> synapse_speech::Result<()> {
            self.calls = 0;
            self.frames.clear();
            Ok(())
        }
        fn accept(&mut self, token: u32) -> synapse_speech::Result<()> {
            assert_eq!(token, 7);
            Ok(())
        }
        fn evaluate(
            &mut self,
            frame: usize,
        ) -> synapse_speech::Result<synapse_speech::JointLogits<'_>> {
            self.calls += 1;
            self.frames.push(frame);
            self.durations.fill(-10.0);
            self.durations[usize::from(self.at_cap_advances && self.calls.is_multiple_of(10))] =
                1.0;
            Ok(synapse_speech::JointLogits {
                tokens: &self.tokens,
                durations: &self.durations,
            })
        }
    }
    fn stub(at_cap_advances: bool) -> Stub {
        let mut tokens = vec![-10.0; 8193];
        tokens[7] = 1.0;
        Stub {
            calls: 0,
            at_cap_advances,
            tokens,
            durations: [0.0; 5],
            frames: vec![],
        }
    }
    #[test]
    fn transcribe_stub_zero_duration_caps_and_never_reads_padding() {
        let mut joint = stub(false);
        let mut progress = vec![];
        let out = decode(&mut joint, 251, true, vec![], &mut |p| progress.push(p)).unwrap();
        assert_eq!(joint.calls, 2510);
        assert_eq!(out.triples.len(), 2510);
        assert_eq!(
            progress,
            [Progress::DecodedFrames(125), Progress::DecodedFrames(250)]
        );
        for frame in 0..251 {
            for token in &out.triples[frame * 10..frame * 10 + 10] {
                assert_eq!(
                    (token.token_id, token.frame_index, token.duration),
                    (7, frame as i32, 0)
                );
            }
        }
        let trace = out.observation.unwrap().decode_trace;
        assert_eq!(trace.iter().filter(|t| t.flags & 2 != 0).count(), 251);
        assert!(joint.frames.iter().all(|&f| f < 251));
    }
    #[test]
    fn transcribe_stub_nonzero_duration_at_cap_does_not_skip_a_frame() {
        let mut joint = stub(true);
        let observed = decode(&mut joint, 3, true, vec![], &mut |_| {}).unwrap();
        let plain = decode(&mut joint, 3, false, vec![], &mut |_| {}).unwrap();
        assert_eq!(
            joint.frames,
            [vec![0; 10], vec![1; 10], vec![2; 10]].concat()
        );
        assert_eq!(observed.triples.len(), 30);
        assert_eq!(observed.triples, plain.triples);
        assert_eq!(observed.valid_encoder_frames, plain.valid_encoder_frames);
        let trace = observed.observation.unwrap().decode_trace;
        assert!(trace.iter().all(|t| t.flags & 2 == 0));
        for i in [9, 19, 29] {
            assert_eq!(observed.triples[i].duration, 1);
        }
    }
}
