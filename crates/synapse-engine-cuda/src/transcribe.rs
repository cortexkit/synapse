//! Parakeet's CUDA lane, independent of the token-model kernels and identities.
//! Linear algebra uses half operands with float accumulation; normalization,
//! softmax and recurrent state use float for numerical stability.
use anyhow::{ensure, Result};
use synapse_core::TranscribeRequest;

pub const RUNGS: [usize; 7] = [128, 256, 512, 1024, 2048, 4096, 8192];

pub fn input_shape(request: &TranscribeRequest, maximum: usize) -> Result<(usize, usize)> {
    ensure!(
        request.valid_mel_frames > 0,
        "invalid_request: empty mel input"
    );
    ensure!(
        request.valid_mel_frames <= request.mel_frames,
        "invalid_request: valid mel count"
    );
    ensure!(
        request.mel_frames.checked_mul(128) == Some(request.mel.len()),
        "invalid_request: mel shape"
    );
    ensure!(
        request.mel.iter().all(|v| v.is_finite()),
        "invalid_request: non-finite mel input"
    );
    let valid = request.valid_mel_frames.div_ceil(8);
    ensure!(valid <= maximum, "duration_limit: encoder frame maximum");
    let rung = RUNGS
        .into_iter()
        .find(|&r| valid <= r)
        .ok_or_else(|| anyhow::anyhow!("duration_limit"))?;
    ensure!(
        request.mel_frames <= rung * 8 + 1,
        "invalid_request: excessive mel padding"
    );
    Ok((valid, rung))
}

/// The largest rung-padded upload must fit before weights or CUDA are touched.
pub fn validate_frame_envelope(maximum: usize, max_frame: u32) -> Result<()> {
    ensure!(maximum > 0, "invalid_max_encoder_frames");
    let rung = RUNGS
        .into_iter()
        .find(|&r| maximum <= r)
        .ok_or_else(|| anyhow::anyhow!("invalid_max_encoder_frames"))?;
    ensure!(
        (rung * 8 + 1) * 512 <= max_frame as usize,
        "transcribe_frame_limit"
    );
    Ok(())
}

#[cfg(any(test, all(feature = "cuda", not(target_os = "macos"))))]
fn load_profile(id: &str, measurement_digest: Option<&str>) -> Result<crate::manifest::Profile> {
    let mut profile = crate::manifest::Profile::select(id, Some("transcribe"))?;
    if let Some(digest) = measurement_digest {
        ensure!(
            digest.strip_prefix("sha256:").is_some_and(|hex| {
                hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit())
            }),
            "package_digest_mismatch"
        );
        profile.package_digest = digest.into();
    }
    Ok(profile)
}

#[cfg(any(test, all(feature = "cuda", not(target_os = "macos"))))]
fn decode_weights(raw: &[u8], fp32: bool) -> Result<Vec<f32>> {
    ensure!(raw.len() % if fp32 { 4 } else { 2 } == 0, "partial weight");
    let values: Vec<f32> = if fp32 {
        raw.chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    } else {
        raw.chunks_exact(2)
            .map(|b| half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect()
    };
    ensure!(values.iter().all(|v| v.is_finite()), "non-finite tensor");
    Ok(values)
}

#[cfg(all(feature = "cuda", not(target_os = "macos")))]
mod enabled {
    use super::*;
    use anyhow::Context;
    use safetensors::{Dtype, SafeTensors};
    use std::{
        collections::BTreeMap,
        ffi::{c_char, c_void, CStr, CString},
        path::Path,
        ptr::NonNull,
        sync::Mutex,
    };
    use synapse_core::{
        DecodeTraceRecord, EngineError, EngineErrorStage, EngineIdentity, EngineRiskClass,
        LoadedModel, RuntimeConfig, TranscribeEngine, TranscribeObservation, TranscribeOutput,
        TranscriptRecord, ValidatedArtifact, CUDA_TRANSCRIBE_KERNEL_REVISION,
    };
    use synapse_speech::{Joint, JointLogits};

    unsafe extern "C" {
        fn synapse_parakeet_error() -> *const c_char;
        fn synapse_parakeet_create() -> *mut c_void;
        fn synapse_parakeet_create_f32() -> *mut c_void;
        fn synapse_parakeet_destroy(p: *mut c_void);
        fn synapse_parakeet_weight(
            p: *mut c_void,
            name: *const c_char,
            data: *const f32,
            n: usize,
        ) -> i32;
        fn synapse_parakeet_begin(
            p: *mut c_void,
            mel: *const f32,
            rows: i32,
            valid: i32,
            rung: i32,
        ) -> i32;
        fn synapse_parakeet_layer(p: *mut c_void, layer: i32) -> i32;
        fn synapse_parakeet_finish(p: *mut c_void, encoder: *mut f32) -> i32;
        fn synapse_parakeet_predict(p: *mut c_void, token: i32, reset: i32) -> i32;
        fn synapse_parakeet_joint(p: *mut c_void, frame: i32, logits: *mut f32) -> i32;
        fn synapse_parakeet_peak(p: *mut c_void) -> usize;
    }
    fn native_error() -> anyhow::Error {
        // The native error is a thread-local NUL-terminated string, valid until
        // the next call on this thread. Copy it before making another call.
        anyhow::anyhow!(unsafe { CStr::from_ptr(synapse_parakeet_error()) }
            .to_string_lossy()
            .into_owned())
    }
    fn checked(status: i32) -> Result<()> {
        if status == 0 {
            Ok(())
        } else {
            Err(native_error())
        }
    }

    struct ContextHandle(NonNull<c_void>);
    // The handle is exclusively accessed under the engine mutex. Every native
    // entry point binds the recorded CUDA device before accessing allocations.
    unsafe impl Send for ContextHandle {}
    impl Drop for ContextHandle {
        fn drop(&mut self) {
            unsafe { synapse_parakeet_destroy(self.0.as_ptr()) }
        }
    }

    #[derive(Default)]
    pub struct OwnedCudaTranscribeEngine {
        loaded: Option<(String, usize, Mutex<ContextHandle>)>,
        strict_f32: bool,
    }
    impl OwnedCudaTranscribeEngine {
        pub fn maximum(&self) -> Option<usize> {
            self.loaded.as_ref().map(|(_, m, _)| *m)
        }
        pub fn peak_device_bytes(&self) -> usize {
            self.loaded
                .as_ref()
                .and_then(|(_, _, c)| c.lock().ok())
                .map(|c| unsafe { synapse_parakeet_peak(c.0.as_ptr()) })
                .unwrap_or(0)
        }
        pub fn transcribe_with_progress(
            &self,
            model: &LoadedModel,
            request: TranscribeRequest,
            progress: &mut dyn FnMut() -> Result<()>,
        ) -> Result<TranscribeOutput> {
            let (id, maximum, context) = self.loaded.as_ref().context("model_not_loaded")?;
            ensure!(*id == model.model_id, "model_not_loaded");
            let (valid, rung) = input_shape(&request, *maximum)?;
            let context = context
                .lock()
                .map_err(|_| anyhow::anyhow!("CUDA transcribe lock poisoned"))?;
            let p = context.0.as_ptr();
            // Safe slices and checked dimensions bound every native read/write.
            // The context owns all GPU allocations and the mutex prevents overlap.
            unsafe {
                checked(synapse_parakeet_begin(
                    p,
                    request.mel.as_ptr(),
                    request.mel_frames as i32,
                    request.valid_mel_frames as i32,
                    rung as i32,
                ))?;
                for layer in 0..24 {
                    checked(synapse_parakeet_layer(p, layer))?;
                    progress()?;
                }
            }
            let mut encoder = if request.observe {
                vec![0.; valid * 1024]
            } else {
                Vec::new()
            };
            unsafe {
                checked(synapse_parakeet_finish(
                    p,
                    if request.observe {
                        encoder.as_mut_ptr()
                    } else {
                        std::ptr::null_mut()
                    },
                ))?;
            }
            let mut joint = GpuJoint {
                context: &context,
                logits: vec![0.; 8198],
            };
            let mut progress_error = None;
            let decoded = synapse_speech::greedy(&mut joint, valid, request.observe, |_| {
                if progress_error.is_none() {
                    progress_error = progress().err();
                }
            })?;
            if let Some(error) = progress_error {
                return Err(error);
            }
            ensure!(
                encoder.iter().all(|v| v.is_finite()),
                "non-finite encoder output"
            );
            Ok(TranscribeOutput {
                valid_encoder_frames: valid,
                triples: decoded
                    .triples
                    .iter()
                    .map(|t| TranscriptRecord {
                        token_id: t[0] as i32,
                        frame_index: t[1] as i32,
                        duration: t[2] as i32,
                    })
                    .collect(),
                observation: request.observe.then(|| TranscribeObservation {
                    encoder_output: encoder,
                    decode_trace: decoded
                        .trace
                        .into_iter()
                        .map(|s| DecodeTraceRecord {
                            frame_index: s.frame_index as i32,
                            token_top1: s.token_top1 as i32,
                            token_top2: s.token_top2 as i32,
                            duration_top1: s.duration_top1 as i32,
                            duration_top2: s.duration_top2 as i32,
                            token_logit1: s.token_logit1,
                            token_logit2: s.token_logit2,
                            duration_logit1: s.duration_logit1,
                            duration_logit2: s.duration_logit2,
                            flags: u32::from(s.emitted) | (u32::from(s.cap_hit) << 1),
                        })
                        .collect(),
                }),
            })
        }
    }
    struct GpuJoint<'a> {
        context: &'a ContextHandle,
        logits: Vec<f32>,
    }
    impl Joint for GpuJoint<'_> {
        fn reset(&mut self) -> synapse_speech::Result<()> {
            checked(unsafe { synapse_parakeet_predict(self.context.0.as_ptr(), 8192, 1) })
                .map_err(speech_error)
        }
        fn evaluate(&mut self, frame: usize) -> synapse_speech::Result<JointLogits<'_>> {
            checked(unsafe {
                synapse_parakeet_joint(
                    self.context.0.as_ptr(),
                    frame as i32,
                    self.logits.as_mut_ptr(),
                )
            })
            .map_err(speech_error)?;
            Ok(JointLogits {
                tokens: &self.logits[..8193],
                durations: &self.logits[8193..],
            })
        }
        fn accept(&mut self, token: u32) -> synapse_speech::Result<()> {
            checked(unsafe { synapse_parakeet_predict(self.context.0.as_ptr(), token as i32, 0) })
                .map_err(speech_error)
        }
    }
    fn speech_error(_: anyhow::Error) -> synapse_speech::Error {
        synapse_speech::Error::InvalidJoint("CUDA neural operation failed")
    }
    fn engine_error(stage: EngineErrorStage, error: anyhow::Error) -> EngineError {
        EngineError {
            stage,
            risk_class: EngineRiskClass::AbortCapable,
            message: error.to_string(),
            retry_after_ms: Some(250),
            safe_to_retry_same_request: false,
        }
    }

    impl OwnedCudaTranscribeEngine {
        /// Load an unpinned all-F32 package for an explicitly arranged measurement.
        /// Production LOAD never calls this method or accepts its digest override.
        /// The artifact digest is still checked against the complete package bytes;
        /// model metadata, tensor shapes and homogeneous F32 storage are required.
        pub fn load_f32_measurement(
            &mut self,
            artifact: &ValidatedArtifact,
            cfg: &RuntimeConfig,
        ) -> std::result::Result<LoadedModel, EngineError> {
            self.load_package(artifact, cfg, true)
        }

        fn load_package(
            &mut self,
            artifact: &ValidatedArtifact,
            cfg: &RuntimeConfig,
            strict_f32: bool,
        ) -> std::result::Result<LoadedModel, EngineError> {
            let result = (|| {
                let profile = super::load_profile(
                    cfg.values.get("profile").map(String::as_str).unwrap_or(""),
                    strict_f32.then_some(artifact.digest.as_str()),
                )?;
                ensure!(
                    profile.model["architecture"]["family"] == "parakeet_tdt",
                    "model_unsupported"
                );
                let row = &crate::manifest::MANIFEST["profiles"][&profile.id];
                let max_ms = row["transcribe"]["max_duration_ms"]
                    .as_u64()
                    .context("invalid_max_encoder_frames")?;
                let certified = (max_ms as usize * 16 / 160).div_ceil(8);
                let maximum = cfg
                    .values
                    .get("max_encoder_frames")
                    .map(|m| m.parse::<usize>())
                    .transpose()?
                    .unwrap_or(certified);
                ensure!(
                    maximum > 0 && maximum <= certified && maximum <= 8192,
                    "invalid_max_encoder_frames"
                );
                let path = cfg
                    .values
                    .get("model_path")
                    .or_else(|| cfg.values.get("artifact_path"))
                    .context("artifact_invalid")?;
                let path = profile.validate_header(Path::new(path), &artifact.digest)?;
                profile.verify_package(&path)?;
                crate::cuda::ensure_available()?;
                let file = std::fs::File::open(path)?;
                // Mapping is read-only and lives through all tensor uploads.
                let mmap = unsafe { memmap2::Mmap::map(&file) }?;
                let tensors = SafeTensors::deserialize(&mmap)?;
                let expected = expected_shapes();
                for (name, shape) in &expected {
                    let view = tensors
                        .tensor(name)
                        .with_context(|| format!("missing tensor {name}"))?;
                    ensure!(view.shape() == shape, "wrong shape for {name}");
                    let dtype = if strict_f32 { Dtype::F32 } else { Dtype::F16 };
                    ensure!(view.dtype() == dtype, "wrong storage dtype for {name}");
                }
                let context = ContextHandle(
                    NonNull::new(unsafe {
                        if strict_f32 {
                            synapse_parakeet_create_f32()
                        } else {
                            synapse_parakeet_create()
                        }
                    })
                    .ok_or_else(native_error)?,
                );
                for name in expected.keys() {
                    let view = tensors.tensor(name)?;
                    let data = super::decode_weights(view.data(), strict_f32)
                        .with_context(|| format!("invalid tensor {name}"))?;
                    let name = CString::new(name.as_str())?;
                    checked(unsafe {
                        synapse_parakeet_weight(
                            context.0.as_ptr(),
                            name.as_ptr(),
                            data.as_ptr(),
                            data.len(),
                        )
                    })?;
                }
                let model = LoadedModel {
                    model_id: if strict_f32 {
                        format!("cuda-transcribe-measurement-f32-{}", artifact.digest)
                    } else {
                        format!("cuda-transcribe-{}", artifact.digest)
                    },
                };
                self.loaded = Some((model.model_id.clone(), maximum, Mutex::new(context)));
                self.strict_f32 = strict_f32;
                Ok(model)
            })();
            result.map_err(|e| engine_error(EngineErrorStage::Load, e))
        }
    }
    impl TranscribeEngine for OwnedCudaTranscribeEngine {
        fn identity(&self) -> EngineIdentity {
            synapse_core::owned_cuda_engine_identity(
                "parakeet-tdt",
                if self.strict_f32 {
                    "f32-measurement"
                } else {
                    "f16"
                },
                CUDA_TRANSCRIBE_KERNEL_REVISION,
            )
        }
        fn load(
            &mut self,
            artifact: &ValidatedArtifact,
            cfg: &RuntimeConfig,
        ) -> std::result::Result<LoadedModel, EngineError> {
            self.load_package(artifact, cfg, false)
        }
        fn transcribe(
            &self,
            model: &LoadedModel,
            request: TranscribeRequest,
        ) -> std::result::Result<TranscribeOutput, EngineError> {
            self.transcribe_with_progress(model, request, &mut || Ok(()))
                .map_err(|e| engine_error(EngineErrorStage::Inference, e))
        }
        fn unload(&mut self, model: &LoadedModel) {
            if self
                .loaded
                .as_ref()
                .is_some_and(|(id, _, _)| *id == model.model_id)
            {
                self.loaded = None;
                self.strict_f32 = false;
            }
        }
    }

    fn expected_shapes() -> BTreeMap<String, Vec<usize>> {
        let mut shapes = BTreeMap::new();
        let mut put = |name: String, shape: Vec<usize>| {
            shapes.insert(name, shape);
        };
        put("decoder.embedding.weight".into(), vec![8193, 640]);
        for layer in 0..2 {
            for kind in ["ih", "hh"] {
                put(
                    format!("decoder.lstm.weight_{kind}_l{layer}"),
                    vec![2560, 640],
                );
                put(format!("decoder.lstm.bias_{kind}_l{layer}"), vec![2560]);
            }
        }
        for (name, out, input) in [
            ("decoder.decoder_projector", 640, 640),
            ("encoder_projector", 640, 1024),
            ("joint.head", 8198, 640),
        ] {
            put(format!("{name}.weight"), vec![out, input]);
            put(format!("{name}.bias"), vec![out]);
        }
        for (i, input, k) in [(0, 1, 3), (2, 1, 3), (3, 256, 1), (5, 1, 3), (6, 256, 1)] {
            put(
                format!("encoder.subsampling.layers.{i}.weight"),
                vec![256, input, k, k],
            );
            put(format!("encoder.subsampling.layers.{i}.bias"), vec![256]);
        }
        put("encoder.subsampling.linear.weight".into(), vec![1024, 4096]);
        put("encoder.subsampling.linear.bias".into(), vec![1024]);
        for layer in 0..24 {
            let l = format!("encoder.layers.{layer}");
            for n in [
                "norm_conv",
                "norm_feed_forward1",
                "norm_feed_forward2",
                "norm_out",
                "norm_self_att",
            ] {
                for s in ["weight", "bias"] {
                    put(format!("{l}.{n}.{s}"), vec![1024]);
                }
            }
            for ff in [1, 2] {
                put(
                    format!("{l}.feed_forward{ff}.linear1.weight"),
                    vec![4096, 1024],
                );
                put(
                    format!("{l}.feed_forward{ff}.linear2.weight"),
                    vec![1024, 4096],
                );
            }
            for p in ["q_proj", "k_proj", "v_proj", "o_proj", "relative_k_proj"] {
                put(format!("{l}.self_attn.{p}.weight"), vec![1024, 1024]);
            }
            for b in ["bias_u", "bias_v"] {
                put(format!("{l}.self_attn.{b}"), vec![8, 128]);
            }
            put(
                format!("{l}.conv.pointwise_conv1.weight"),
                vec![2048, 1024, 1],
            );
            put(
                format!("{l}.conv.pointwise_conv2.weight"),
                vec![1024, 1024, 1],
            );
            put(format!("{l}.conv.depthwise_conv.weight"), vec![1024, 1, 9]);
            for s in ["weight", "bias", "running_mean", "running_var"] {
                put(format!("{l}.conv.norm.{s}"), vec![1024]);
            }
        }
        shapes
    }
}
#[cfg(all(feature = "cuda", not(target_os = "macos")))]
pub use enabled::OwnedCudaTranscribeEngine;

#[cfg(test)]
mod tests {
    #[test]
    fn transcribe_f32_measurement_preserves_weight_bits_instead_of_rounding_to_half() {
        let values = [1.0001234f32, -0.0000001234, 65505., -0.];
        let raw: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let decoded = super::decode_weights(&raw, true).unwrap();
        assert_eq!(
            decoded.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        let raw_half: Vec<_> = values
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
            .collect();
        let decoded_half = super::decode_weights(&raw_half, false).unwrap();
        assert_ne!(decoded[0].to_bits(), decoded_half[0].to_bits());
        assert_eq!(decoded_half[0], 1.);
        assert!(super::decode_weights(&[0, 0, 0], true).is_err());
        assert!(super::decode_weights(&f32::NAN.to_le_bytes(), true).is_err());
        assert!(
            super::decode_weights(&half::f16::INFINITY.to_bits().to_le_bytes(), false).is_err()
        );
    }

    #[test]
    fn transcribe_measurement_digest_override_never_changes_production_pin() {
        let id = "parakeet-tdt-0.6b-v3.owned-cuda";
        let sealed = super::load_profile(id, None).unwrap();
        let digest = format!("sha256:{}", "1".repeat(64));
        let measured = super::load_profile(id, Some(&digest)).unwrap();
        assert_eq!(measured.package_digest, digest);
        assert_eq!(measured.id, sealed.id);
        assert_eq!(measured.model, sealed.model);
        assert_ne!(measured.package_digest, sealed.package_digest);
        assert_eq!(
            super::load_profile(id, None).unwrap().package_digest,
            sealed.package_digest
        );
        assert_eq!(
            crate::manifest::MANIFEST["profiles"][id]["converted_package_digest"],
            sealed.package_digest
        );
        assert!(super::load_profile(id, Some("unhashed")).is_err());
    }

    #[test]
    fn transcribe_f32_measurement_uses_pedantic_gemm_without_half_weight_copies() {
        let native = include_str!("port/cuda_parakeet.cu");
        assert_eq!(native.matches("CUBLAS_COMPUTE_32F_PEDANTIC").count(), 4);
        assert_eq!(
            native
                .matches("CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT")
                .count(),
            4
        );
        assert!(native.contains("if(!c.strict_f32) {\n            HalfBuffer hb(n);"));
        assert!(native.contains("HalfBuffer kh(c.strict_f32?0:t*H)"));
        assert!(!native.contains("FAST_TF32"));
        assert!(!native.contains("CUBLAS_TF32_TENSOR_OP_MATH"));
    }
    use super::*;

    #[test]
    fn transcribe_reductions_capture_scalars_before_reusing_shared_scratch() {
        // The portable guard protects the synchronization boundary; the opt-in
        // NVIDIA test separately checks repeatability of the actual kernels.
        let source = include_str!("port/cuda_parakeet.cu");
        for (kernel, capture, overwrite) in [
            (
                "norm",
                "float mean = sums[0] / H, var = 0.f;",
                "sums[tid] = var;",
            ),
            (
                "attention_softmax",
                "m=sums[0]; float sum=0.f;",
                "sums[tid]=sum;",
            ),
        ] {
            let after_capture = source
                .split_once(capture)
                .unwrap_or_else(|| panic!("{kernel}: missing reduced-scalar capture"))
                .1;
            let before_overwrite = after_capture
                .split_once(overwrite)
                .unwrap_or_else(|| panic!("{kernel}: missing scratch reuse"))
                .0;
            assert!(
                before_overwrite.contains("__syncthreads();"),
                "{kernel}: all threads must capture the reduced scalar before shared scratch is overwritten"
            );
        }
    }

    #[test]
    fn transcribe_shape_admission_is_checked_before_cuda() {
        let mut r = TranscribeRequest {
            mel: vec![0.; 129 * 128],
            mel_frames: 129,
            valid_mel_frames: 128,
            observe: false,
        };
        assert_eq!(input_shape(&r, 16).unwrap(), (16, 128));
        r.valid_mel_frames = 129;
        assert!(input_shape(&r, 16).is_err());
        r.valid_mel_frames = 130;
        assert!(input_shape(&r, 128).is_err());
        r.valid_mel_frames = 128;
        r.mel[0] = f32::NAN;
        assert!(input_shape(&r, 128).is_err());
        r.mel[0] = 0.;
        r.mel.pop();
        assert!(input_shape(&r, 128).is_err());
        assert!(validate_frame_envelope(7500, 64 * 1024 * 1024).is_ok());
        assert!(validate_frame_envelope(7500, 1024).is_err());
        assert!(validate_frame_envelope(8193, u32::MAX).is_err());
    }
}
