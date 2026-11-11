//! Parakeet's allocation contract and host-side Token Duration Transducer joint.
use anyhow::{ensure, Context, Result};
use synapse_parity::{manifest::Model, vulkan::Buffer};

pub const TILE: usize = 128;

#[cfg(any(feature = "vulkan", test))]
pub(crate) fn subsampling_stages(start: usize, rows: usize) -> [(usize, i32, usize); 3] {
    [
        (0, 4 * start as i32 - 3, 4 * rows + 3),
        (2, 2 * start as i32 - 1, 2 * rows + 1),
        (5, start as i32, rows),
    ]
}

pub struct Output {
    pub valid_encoder_frames: usize,
    pub encoder: Vec<f32>,
    pub trace: Vec<synapse_core::DecodeTraceRecord>,
    pub triples: Vec<synapse_core::TranscriptRecord>,
}

pub fn write_output(
    stream: &mut impl std::io::Write,
    req_id: &str,
    output: &Output,
    observe: bool,
    max_frame: u32,
) -> std::io::Result<()> {
    use synapse_core::{
        worker_framing_sync::{write_frame, write_json_frame},
        WorkerResponse,
    };
    if observe {
        write_json_frame(
            stream,
            &WorkerResponse::EncoderOutput {
                req_id: req_id.into(),
                frames: output.valid_encoder_frames,
            },
            max_frame,
        )?;
        write_frame(
            stream,
            &synapse_core::encode_f32_frame(&output.encoder),
            max_frame,
        )?;
        write_json_frame(
            stream,
            &WorkerResponse::DecodeTrace {
                req_id: req_id.into(),
                steps: output.trace.len(),
            },
            max_frame,
        )?;
        write_frame(
            stream,
            &synapse_core::encode_decode_trace_frame(&output.trace),
            max_frame,
        )?;
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
    write_frame(
        stream,
        &synapse_core::encode_transcript_frame(&output.triples),
        max_frame,
    )
}

pub fn maximum(profile: &synapse_parity::manifest::Profile) -> Result<usize> {
    let speech = profile
        .transcribe
        .as_ref()
        .context("missing speech profile")?;
    Ok(synapse_speech::valid_encoder_frames(speech.max_duration_ms * 16) as usize)
}

pub fn input_shape(mel: usize, valid: usize, maximum: usize) -> Result<(usize, usize)> {
    ensure!(
        valid > 0 && valid.checked_add(1) == Some(mel),
        "invalid_mel_frames"
    );
    let frames = valid.div_ceil(8);
    ensure!(frames <= maximum, "duration_limit");
    let rung = [128, 256, 512, 1024, 2048, 4096, 8192]
        .into_iter()
        .find(|r| *r >= frames)
        .context("duration_limit")?;
    // The frontend's final mel row remains available to the convolutions; only
    // invalid encoder rows beyond the rung are omitted.
    Ok((frames, rung))
}

pub fn validate_frame_envelope(maximum: usize, max_frame: u32) -> Result<()> {
    let (_, rung) = input_shape(maximum * 8 + 1, maximum * 8, maximum)?;
    ensure!(
        rung.checked_mul(8)
            .and_then(|n| n.checked_add(1))
            .and_then(|n| n.checked_mul(128 * 4))
            .is_some_and(|bytes| bytes <= max_frame as usize),
        "transcribe_frame_limit"
    );
    Ok(())
}

pub fn buffer_plan(model: &Model, maximum: usize) -> Result<Vec<Buffer>> {
    buffer_plan_for_precision(model, maximum, WeightPrecision::F16)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightPrecision {
    F16,
    F32,
}

impl WeightPrecision {
    pub fn width(self) -> u64 {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }

    #[cfg(feature = "vulkan")]
    pub(crate) fn from_header(header: &synapse_parity::safetensors::Header) -> Result<Self> {
        use synapse_parity::safetensors::StDType;
        let mut precision = None;
        for tensor in header.tensors.values().filter(|t| t.dtype != StDType::I64) {
            let next = match tensor.dtype {
                StDType::F16 => Self::F16,
                StDType::F32 => Self::F32,
                _ => anyhow::bail!("unsupported speech weight dtype"),
            };
            anyhow::ensure!(
                precision.is_none_or(|p| p == next),
                "mixed speech weight dtypes are not supported"
            );
            precision = Some(next);
        }
        precision.ok_or_else(|| anyhow::anyhow!("no speech weights"))
    }
}

pub fn buffer_plan_for_precision(
    model: &Model,
    maximum: usize,
    precision: WeightPrecision,
) -> Result<Vec<Buffer>> {
    let (_, rung) = input_shape(maximum * 8 + 1, maximum * 8, maximum)?;
    let t = rung as u64;
    let mut plan = Vec::new();
    for (name, shape) in synapse_parity::arch::expected_tensors(model)? {
        let width = if name.ends_with("num_batches_tracked") {
            8
        } else {
            precision.width()
        };
        plan.push(Buffer {
            name: format!("weight:{name}"),
            bytes: shape.iter().product::<u64>() * width,
            per_sub_batch: false,
        });
    }
    for (name, elements) in [
        ("mel", (t * 8 + 1) * 128),
        // Halo rows preserve all three 3x3 convolutions across tile boundaries.
        ("sub_a", (TILE as u64 * 4 + 3) * 64 * 256),
        ("sub_b", (TILE as u64 * 2 + 1) * 32 * 256),
        ("hidden", t * 1024),
        ("norm", t * 1024),
        ("out", t * 1024),
        ("ff", t * 4096),
        ("q", t * 1024),
        ("k", t * 1024),
        ("v", t * 1024),
        ("pos", (2 * t - 1) * 1024),
        ("relative", (2 * t - 1) * 1024),
        ("att", t * 1024),
        ("scores", TILE as u64 * 8 * t),
        ("prob", TILE as u64 * 8 * t),
        ("glu", t * 1024),
        ("conv", t * 2048),
        ("projected", t * 640),
        ("dummy", 1),
    ] {
        plan.push(Buffer {
            name: name.into(),
            bytes: elements * 4,
            per_sub_batch: false,
        });
    }
    Ok(plan)
}

pub fn floors(model: &Model, maximum: usize) -> Result<crate::admission::Required> {
    floors_for_precision(model, maximum, WeightPrecision::F16)
}

pub fn floors_for_precision(
    model: &Model,
    maximum: usize,
    precision: WeightPrecision,
) -> Result<crate::admission::Required> {
    let plan = buffer_plan_for_precision(model, maximum, precision)?;
    Ok(crate::admission::Required {
        min_device_local_bytes: synapse_parity::vulkan::arena_bytes(&plan),
        min_storage_buffer_range: plan.iter().map(|b| b.bytes).max().unwrap_or(0),
    })
}

#[cfg(feature = "vulkan")]
pub(crate) mod joint {
    use std::collections::BTreeMap;
    use synapse_speech::{Error, Joint, JointLogits};

    pub struct HostJoint {
        weights: BTreeMap<String, Vec<f32>>,
        pub projected: Vec<f32>,
        h: [Vec<f32>; 2],
        cell: [Vec<f32>; 2],
        predictor: Vec<f32>,
        logits: Vec<f32>,
    }
    impl HostJoint {
        pub fn load(header: &synapse_parity::safetensors::Header, data: &[u8]) -> Self {
            let weights = header
                .tensors
                .iter()
                .filter(|(n, _)| n.starts_with("decoder.") || n.starts_with("joint."))
                .map(|(n, t)| {
                    let raw = &data[t.data_offsets.0..t.data_offsets.1];
                    let values = if t.dtype == synapse_parity::safetensors::StDType::F32 {
                        raw.as_chunks::<4>()
                            .0
                            .iter()
                            .map(|v| f32::from_le_bytes(*v))
                            .collect()
                    } else {
                        raw.as_chunks::<2>()
                            .0
                            .iter()
                            .map(|v| half::f16::from_le_bytes(*v).to_f32())
                            .collect()
                    };
                    (n.clone(), values)
                })
                .collect();
            Self {
                weights,
                projected: Vec::new(),
                h: [vec![0.; 640], vec![0.; 640]],
                cell: [vec![0.; 640], vec![0.; 640]],
                predictor: vec![0.; 640],
                logits: vec![0.; 8198],
            }
        }
        fn linear(&self, input: &[f32], name: &str) -> Vec<f32> {
            let w = &self.weights[&format!("{name}.weight")];
            let bias = &self.weights[&format!("{name}.bias")];
            w.chunks_exact(input.len())
                .zip(bias)
                .map(|(row, b)| row.iter().zip(input).fold(0., |a, (w, x)| a + w * x) + b)
                .collect()
        }
        fn predict(&mut self, token: u32) {
            let mut input = self.weights["decoder.embedding.weight"]
                [token as usize * 640..(token as usize + 1) * 640]
                .to_vec();
            for layer in 0..2 {
                let ih = &self.weights[&format!("decoder.lstm.weight_ih_l{layer}")];
                let hh = &self.weights[&format!("decoder.lstm.weight_hh_l{layer}")];
                let bi = &self.weights[&format!("decoder.lstm.bias_ih_l{layer}")];
                let bh = &self.weights[&format!("decoder.lstm.bias_hh_l{layer}")];
                let gates: Vec<f32> = (0..2560)
                    .map(|r| {
                        let a = ih[r * 640..(r + 1) * 640]
                            .iter()
                            .zip(&input)
                            .fold(0., |a, (w, x)| a + w * x);
                        let b = hh[r * 640..(r + 1) * 640]
                            .iter()
                            .zip(&self.h[layer])
                            .fold(0., |a, (w, x)| a + w * x);
                        (a + bi[r]) + (b + bh[r])
                    })
                    .collect();
                for i in 0..640 {
                    let sigmoid = |x: f32| 1. / (1. + (-x).exp());
                    self.cell[layer][i] = sigmoid(gates[640 + i]) * self.cell[layer][i]
                        + sigmoid(gates[i]) * gates[1280 + i].tanh();
                    self.h[layer][i] = sigmoid(gates[1920 + i]) * self.cell[layer][i].tanh();
                }
                input.clone_from(&self.h[layer]);
            }
            self.predictor = self.linear(&input, "decoder.decoder_projector");
        }
    }
    impl Joint for HostJoint {
        fn reset(&mut self) -> synapse_speech::Result<()> {
            for v in self.h.iter_mut().chain(self.cell.iter_mut()) {
                v.fill(0.);
            }
            self.predict(synapse_speech::BLANK);
            Ok(())
        }
        fn evaluate(&mut self, frame: usize) -> synapse_speech::Result<JointLogits<'_>> {
            let encoder = self
                .projected
                .get(frame * 640..(frame + 1) * 640)
                .ok_or(Error::InvalidJoint("encoder frame out of range"))?;
            let input: Vec<f32> = encoder
                .iter()
                .zip(&self.predictor)
                .map(|(a, b)| (a + b).max(0.))
                .collect();
            self.logits = self.linear(&input, "joint.head");
            Ok(JointLogits {
                tokens: &self.logits[..8193],
                durations: &self.logits[8193..],
            })
        }
        fn accept(&mut self, token: u32) -> synapse_speech::Result<()> {
            if token >= synapse_speech::BLANK {
                return Err(Error::InvalidJoint("invalid emitted token"));
            }
            self.predict(token);
            Ok(())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn transcribe_host_joint_initial_blank_recurrence_and_reset() {
            let mut weights = BTreeMap::new();
            weights.insert("decoder.embedding.weight".into(), vec![0.; 8193 * 640]);
            for layer in 0..2 {
                for name in ["weight_ih", "weight_hh"] {
                    weights.insert(
                        format!("decoder.lstm.{name}_l{layer}"),
                        vec![0.; 2560 * 640],
                    );
                }
                let mut bias = vec![0.; 2560];
                bias[1280] = 1.;
                weights.insert(format!("decoder.lstm.bias_ih_l{layer}"), bias);
                weights.insert(format!("decoder.lstm.bias_hh_l{layer}"), vec![0.; 2560]);
            }
            let mut projector = vec![0.; 640 * 640];
            projector[0] = 1.;
            weights.insert("decoder.decoder_projector.weight".into(), projector);
            weights.insert("decoder.decoder_projector.bias".into(), vec![0.; 640]);
            let mut head = vec![0.; 8198 * 640];
            head[7 * 640] = 1.;
            weights.insert("joint.head.weight".into(), head);
            weights.insert("joint.head.bias".into(), vec![0.; 8198]);
            let mut projected = vec![0.; 640];
            projected[0] = 0.25;
            let mut joint = HostJoint {
                weights,
                projected,
                h: [vec![9.; 640], vec![9.; 640]],
                cell: [vec![9.; 640], vec![9.; 640]],
                predictor: vec![0.; 640],
                logits: vec![0.; 8198],
            };
            joint.reset().unwrap();
            // With zero matrices every sigmoid gate is 1/2 and the candidate
            // cell is tanh(1). The initial blank must run the LSTM once.
            let candidate = 1.0f32.tanh();
            let first = 0.25 + 0.5 * (0.5 * candidate).tanh();
            assert_eq!(joint.evaluate(0).unwrap().tokens[7], first);
            assert_eq!(joint.evaluate(0).unwrap().tokens[7], first);
            joint.accept(7).unwrap();
            assert_eq!(
                joint.evaluate(0).unwrap().tokens[7],
                0.25 + 0.5 * (0.75 * candidate).tanh()
            );
            assert!(joint.evaluate(1).is_err());
            assert!(joint.accept(synapse_speech::BLANK).is_err());
            joint.reset().unwrap();
            assert_eq!(
                joint.evaluate(0).unwrap().tokens[7].to_bits(),
                first.to_bits()
            );
        }

        #[test]
        fn transcribe_host_joint_reads_declared_f32_without_rounding_to_half() {
            use synapse_parity::safetensors::{
                parse_header_json, split_file, write_package, PackageTensor, StDType,
            };
            let value = 1.0001f32;
            for dtype in [StDType::F16, StDType::F32] {
                let bytes = if dtype == StDType::F32 {
                    value.to_le_bytes().to_vec()
                } else {
                    half::f16::from_f32(value).to_le_bytes().to_vec()
                };
                let package = write_package(
                    "measurement",
                    &BTreeMap::from([(
                        "decoder.embedding.weight".into(),
                        PackageTensor {
                            dtype,
                            shape: vec![1],
                            bytes,
                        },
                    )]),
                );
                let (header, data) = split_file(&package).unwrap();
                let joint = HostJoint::load(&parse_header_json(header).unwrap(), data);
                let expected = if dtype == StDType::F32 { value } else { 1. };
                assert_eq!(
                    joint.weights["decoder.embedding.weight"][0].to_bits(),
                    expected.to_bits()
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcribe_f32_inventory_preserves_activations_and_integer_counters() {
        let manifest = crate::manifest();
        let model = &manifest.models["parakeet-tdt-0.6b-v3"];
        let f16 = buffer_plan(model, 7500).unwrap();
        let f32 = buffer_plan_for_precision(model, 7500, WeightPrecision::F32).unwrap();
        for (a, b) in f16.iter().zip(&f32) {
            assert_eq!(a.name, b.name);
            assert_eq!(
                b.bytes,
                if a.name.starts_with("weight:") && !a.name.ends_with("num_batches_tracked") {
                    a.bytes * 2
                } else {
                    a.bytes
                }
            );
        }
        assert_eq!(
            floors(model, 7500).unwrap().min_device_local_bytes,
            2_021_900_544
        );
        let required = floors_for_precision(model, 7500, WeightPrecision::F32).unwrap();
        assert_eq!(required.min_device_local_bytes, 3_276_015_104);
        assert_eq!(required.min_storage_buffer_range, 134_217_728);
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn transcribe_dtype_requires_uniform_declared_float_storage() {
        use synapse_parity::safetensors::{
            parse_header_json, split_file, write_package, PackageTensor, StDType,
        };
        let mut tensors = std::collections::BTreeMap::new();
        tensors.insert(
            "counter".into(),
            PackageTensor {
                dtype: StDType::I64,
                shape: vec![],
                bytes: vec![0; 8],
            },
        );
        tensors.insert(
            "weight".into(),
            PackageTensor {
                dtype: StDType::F32,
                shape: vec![1],
                bytes: vec![0; 4],
            },
        );
        let parse = |t: &std::collections::BTreeMap<String, PackageTensor>| {
            let bytes = write_package("measurement", t);
            parse_header_json(split_file(&bytes).unwrap().0).unwrap()
        };
        assert_eq!(
            WeightPrecision::from_header(&parse(&tensors)).unwrap(),
            WeightPrecision::F32
        );
        tensors.insert(
            "other".into(),
            PackageTensor {
                dtype: StDType::F16,
                shape: vec![1],
                bytes: vec![0; 2],
            },
        );
        assert!(WeightPrecision::from_header(&parse(&tensors)).is_err());
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn transcribe_f16_shader_matches_the_v1_byte_identity() {
        use sha2::{Digest, Sha256};
        let bytes = crate::SHADERS
            .iter()
            .find(|(name, _)| *name == "transcribe")
            .unwrap()
            .1;
        // SDK 1.4.357.0 compiling the unchanged v1 source (SHA-256
        // 16db12050106cbe059a857e34cc6c90e0b45d5cdc435363750621be66028601f).
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            "6f84be1c14096b066e7bbf1b6b47aa7e97b9e243e8672a54bff102408421d1ad"
        );
        assert_eq!(
            synapse_core::worker_engine_names::VULKAN_TRANSCRIBE_KERNEL_REVISION,
            "owned-vulkan-transcribe-v1"
        );
    }

    #[cfg(feature = "transcribe-f32-measurement")]
    #[test]
    fn transcribe_measurement_shader_has_no_half_precision_float_type() {
        let bytes = crate::SHADERS
            .iter()
            .find(|(name, _)| *name == "transcribe_f32")
            .unwrap()
            .1;
        let words: Vec<_> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect();
        let (mut offset, mut floats) = (5, Vec::new());
        while offset < words.len() {
            let instruction = words[offset];
            if instruction & 0xffff == 22 {
                floats.push(words[offset + 2]);
            }
            offset += (instruction >> 16) as usize;
        }
        assert!(floats.contains(&32));
        assert!(!floats.contains(&16));
    }
    use std::io::Cursor;
    use synapse_core::{
        worker_framing_sync::{read_frame, read_json_frame},
        WorkerResponse,
    };

    #[test]
    fn transcribe_subsampling_halos_equal_full_convolution_at_tile_edges() {
        fn convolution(
            input: &[f32],
            input_start: i32,
            start: i32,
            rows: usize,
            valid: usize,
        ) -> Vec<f32> {
            (0..rows)
                .map(|r| {
                    let global = start + r as i32;
                    if global < 0 || global >= valid as i32 {
                        return 0.;
                    }
                    [-0.75, 0.5, 1.25]
                        .iter()
                        .enumerate()
                        .fold(0.125, |sum, (k, w)| {
                            let i = 2 * global + k as i32 - 1 - input_start;
                            sum + if i >= 0 {
                                input.get(i as usize).copied().unwrap_or(0.) * w
                            } else {
                                0.
                            }
                        })
                        .max(0.)
                })
                .collect()
        }
        let valid_mel = 2051usize;
        let input: Vec<_> = (0..=valid_mel)
            .map(|i| {
                if i == valid_mel {
                    0.
                } else {
                    (i % 19) as f32 - 9.
                }
            })
            .collect();
        let mut full = input.clone();
        let mut valid = valid_mel;
        for _ in 0..3 {
            valid = valid.div_ceil(2);
            full = convolution(&full, 0, 0, full.len().div_ceil(2), valid);
        }
        let mut tiled = Vec::new();
        for start in (0..512).step_by(TILE) {
            let mut part = input.clone();
            let (mut base, mut valid) = (0, valid_mel);
            for (_, out_start, rows) in subsampling_stages(start, TILE) {
                valid = valid.div_ceil(2);
                part = convolution(&part, base, out_start, rows, valid);
                base = out_start;
            }
            tiled.extend(part);
        }
        assert_eq!(&tiled[..full.len()], full);
        assert!(tiled[full.len()..].iter().all(|x| *x == 0.));
    }

    #[test]
    fn transcribe_shapes_preserve_reference_tail_and_enforce_maximum() {
        assert_eq!(input_shape(1025, 1024, 7500).unwrap(), (128, 128));
        assert_eq!(input_shape(1026, 1025, 7500).unwrap(), (129, 256));
        assert_eq!(input_shape(60001, 60000, 7500).unwrap(), (7500, 8192));
        assert_eq!(
            input_shape(60002, 60001, 7500).unwrap_err().to_string(),
            "duration_limit"
        );
        for (mel, valid) in [(0, 0), (2, 2), (1, 2), (usize::MAX, usize::MAX)] {
            assert!(input_shape(mel, valid, 7500).is_err());
        }
        validate_frame_envelope(7500, 64 * 1024 * 1024).unwrap();
        assert!(validate_frame_envelope(7500, 1024).is_err());
    }

    #[test]
    fn transcribe_floors_cover_exact_inventory_and_enter_global_probe() {
        let manifest = crate::manifest();
        let model = &manifest.models["parakeet-tdt-0.6b-v3"];
        let profile = &manifest.profiles["parakeet-tdt-0.6b-v3.owned-vulkan"];
        let maximum = maximum(profile).unwrap();
        assert_eq!(maximum, 7500);
        let required = floors(model, maximum).unwrap();
        assert_eq!(required.min_device_local_bytes, 2_021_900_544);
        assert_eq!(required.min_storage_buffer_range, 134_217_728);
        assert_eq!(
            Some(required.min_device_local_bytes),
            profile.vulkan_min_device_local_bytes
        );
        assert_eq!(
            Some(required.min_storage_buffer_range),
            profile.vulkan_min_storage_buffer_range
        );
        let global = crate::admission::requirements(&manifest, None).unwrap();
        assert!(global.min_device_local_bytes >= required.min_device_local_bytes);
        assert!(global.min_storage_buffer_range >= required.min_storage_buffer_range);
        assert_eq!(
            crate::admission::requirements(&manifest, Some("parakeet-tdt-0.6b-v3")).unwrap(),
            required
        );
        let mut broken = manifest;
        broken
            .profiles
            .get_mut("parakeet-tdt-0.6b-v3.owned-vulkan")
            .unwrap()
            .vulkan_min_device_local_bytes = Some(required.min_device_local_bytes - 1);
        assert_eq!(
            crate::admission::requirements(&broken, None).unwrap_err(),
            "manifest_mismatch"
        );
    }

    #[test]
    fn transcribe_response_order_and_raw_bytes_are_exact() {
        let output = Output {
            valid_encoder_frames: 1,
            encoder: vec![0.; 1024],
            trace: vec![synapse_core::DecodeTraceRecord {
                frame_index: 0,
                token_top1: 7,
                token_top2: 8,
                duration_top1: 2,
                duration_top2: 1,
                token_logit1: 1.,
                token_logit2: 0.,
                duration_logit1: 2.,
                duration_logit2: 1.,
                flags: 1,
            }],
            triples: vec![synapse_core::TranscriptRecord {
                token_id: 7,
                frame_index: 0,
                duration: 2,
            }],
        };
        for observe in [false, true] {
            let mut bytes = Vec::new();
            write_output(&mut bytes, "speech", &output, observe, 65536).unwrap();
            let mut input = Cursor::new(bytes);
            if observe {
                assert!(matches!(
                    read_json_frame::<_, WorkerResponse>(&mut input, 65536).unwrap(),
                    WorkerResponse::EncoderOutput { frames: 1, .. }
                ));
                assert_eq!(read_frame(&mut input, 65536).unwrap(), vec![0; 4096]);
                assert!(matches!(
                    read_json_frame::<_, WorkerResponse>(&mut input, 65536).unwrap(),
                    WorkerResponse::DecodeTrace { steps: 1, .. }
                ));
                assert_eq!(
                    read_frame(&mut input, 65536).unwrap(),
                    [
                        0i32,
                        7,
                        8,
                        2,
                        1,
                        1.0f32.to_bits() as i32,
                        0,
                        2.0f32.to_bits() as i32,
                        1.0f32.to_bits() as i32,
                        1
                    ]
                    .into_iter()
                    .flat_map(i32::to_le_bytes)
                    .collect::<Vec<_>>()
                );
            }
            assert!(matches!(
                read_json_frame::<_, WorkerResponse>(&mut input, 65536).unwrap(),
                WorkerResponse::Transcript {
                    valid_encoder_frames: 1,
                    count: 1,
                    ..
                }
            ));
            assert_eq!(
                read_frame(&mut input, 65536).unwrap(),
                vec![7, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]
            );
            assert_eq!(input.position(), input.get_ref().len() as u64);
        }
    }
}
