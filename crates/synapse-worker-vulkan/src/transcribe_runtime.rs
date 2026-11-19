//! GPU encoder and encoder projection, then the greedy decode. The decode rule
//! runs on the host; its predictor and joint run on the GPU when the device
//! supports that decoder, and on the host otherwise.
use super::{
    speech_decode::GpuJoint, Arena, Params, Prepared, SpeechBatch, SPEECH_GEMM_BIAS,
    SPEECH_GEMM_THREE,
};
use crate::transcribe::{self, joint::HostJoint, TILE};
use anyhow::{ensure, Result};
use synapse_core::{DecodeTraceRecord, TranscriptRecord};
use synapse_parity::{manifest::Profile, safetensors::Header};

use crate::transcribe::Output;

/// Where the predictor and joint run, chosen once at load.
enum Decoder {
    Gpu(GpuJoint),
    /// The host decoder, used when the GPU decoder could not be prepared.
    Host(HostJoint),
}

impl Decoder {
    fn label(&self) -> &'static str {
        match self {
            Self::Gpu(_) => "gpu",
            Self::Host(_) => "host",
        }
    }
}

#[cfg(test)]
type LayerObserver = Box<dyn Fn(&str, &[f32]) -> Result<()>>;

#[cfg(test)]
struct LayerCapture {
    names: std::collections::BTreeSet<String>,
    observe: LayerObserver,
}

pub struct Engine {
    arena: Arena,
    batch: std::cell::RefCell<SpeechBatch>,
    maximum: usize,
    largest_storage_buffer: u64,
    decoder: Decoder,
    /// Valid encoder frames of the last request, whose projection is on the GPU.
    decoded_frames: usize,
    #[cfg(test)]
    layer_capture: Option<LayerCapture>,
    #[cfg(test)]
    debug_disjoint_bindings: bool,
}

impl Engine {
    pub fn load(
        model: &synapse_parity::manifest::Model,
        profile: &Profile,
        prepared: Prepared,
        required: crate::admission::Required,
        header: &Header,
        data: &[u8],
    ) -> Result<Self> {
        let maximum = transcribe::maximum(profile)?;
        transcribe::validate_weights(model, header)?;
        let plan = transcribe::buffer_plan(model, maximum)?;
        let largest_storage_buffer = plan.iter().map(|buffer| buffer.bytes).max().unwrap_or(0);
        ensure!(
            synapse_parity::vulkan::arena_bytes(&plan) == required.min_device_local_bytes,
            "manifest_mismatch"
        );
        let arena = Arena::new(prepared.device, &plan, required.min_device_local_bytes)?;
        for (name, tensor) in &header.tensors {
            let mut bytes = data[tensor.data_offsets.0..tensor.data_offsets.1].to_vec();
            arena.transfer(&format!("weight:{name}"), &mut bytes, true)?;
        }
        // The host decoder needs its own f32 copy of the decoder weights, so
        // it is only loaded when the GPU decoder is unavailable.
        let (decoder, host_reason) = match GpuJoint::new(&arena) {
            Ok(joint) => (Decoder::Gpu(joint), None),
            Err(error) if crate::driver_failure::is_fatal(&error) => return Err(error),
            Err(error) => (
                Decoder::Host(HostJoint::load(header, data)),
                Some(format!("{error:#}")),
            ),
        };
        eprintln!(
            "{}",
            serde_json::json!({"vulkan_transcribe_load_report": {
                "requested_device_local_bytes": arena.requested_bytes,
                "driver_memory_requirements_size": arena.driver_bytes,
                "maximum_encoder_frames": maximum,
                "min_storage_buffer_range": required.min_storage_buffer_range,
                "allocations": 1
                ,"weight_precision":"f32"
                ,"decoder": decoder.label()
                ,"host_decoder_reason": host_reason
            }})
        );
        let batch =
            std::cell::RefCell::new(SpeechBatch::new(std::sync::Arc::clone(&arena.device))?);
        Ok(Self {
            arena,
            batch,
            maximum,
            largest_storage_buffer,
            decoder,
            decoded_frames: 0,
            #[cfg(test)]
            layer_capture: None,
            #[cfg(test)]
            debug_disjoint_bindings: false,
        })
    }

    /// "gpu" or "host": where this engine runs the predictor and joint.
    pub fn decoder(&self) -> &'static str {
        self.decoder.label()
    }

    pub fn device_bytes(&self) -> u64 {
        self.arena.driver_bytes
    }

    #[cfg(test)]
    pub(crate) fn debug_device_heap_usage(&self) -> Option<u64> {
        let device = &self.arena.device;
        let extensions = unsafe {
            device
                .instance
                .raw
                .enumerate_device_extension_properties(device.physical)
        }
        .ok()?;
        if !extensions.iter().any(|e|unsafe {std::ffi::CStr::from_ptr(e.extension_name.as_ptr())}==c"VK_EXT_memory_budget") {return None;}
        let mut budget = ash::vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut properties =
            ash::vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        unsafe {
            device
                .instance
                .raw
                .get_physical_device_memory_properties2(device.physical, &mut properties)
        };
        Some(budget.heap_usage[device.heap_index as usize])
    }

    #[cfg(test)]
    pub(crate) fn debug_capture_layers(
        &mut self,
        names: std::collections::BTreeSet<String>,
        observe: impl Fn(&str, &[f32]) -> Result<()> + 'static,
    ) {
        self.layer_capture = Some(LayerCapture {
            names,
            observe: Box::new(observe),
        });
    }

    #[cfg(test)]
    pub(crate) fn debug_clear_capture(&mut self) {
        self.layer_capture = None;
    }

    #[cfg(test)]
    pub(crate) fn debug_set_encoder_probe(
        &mut self,
        dispatch_limit: Option<usize>,
        disjoint_bindings: bool,
    ) -> Result<()> {
        let capacity = super::SPEECH_BATCH_CAPACITY;
        ensure!(
            dispatch_limit.is_none_or(|limit| (1..=capacity).contains(&limit)),
            "SYNAPSE_VULKAN_SPEECH_DISPATCH_LIMIT must be in 1..={capacity}"
        );
        self.batch.get_mut().dispatch_limit_override = dispatch_limit;
        self.debug_disjoint_bindings = disjoint_bindings;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn debug_submission_stats(
        &self,
    ) -> (super::speech_batch_policy::SubmissionStats, usize) {
        let mut batch = self.batch.borrow_mut();
        (std::mem::take(&mut batch.stats), batch.dispatch_limit())
    }

    #[cfg(test)]
    pub(crate) fn debug_encoder_readback_path(&self) -> &'static str {
        if self.arena.mapped {
            "mapped"
        } else {
            "staging"
        }
    }

    #[cfg(test)]
    fn capture_layer(&self, name: &str, valid: usize) -> Result<()> {
        if let Some(capture) = self
            .layer_capture
            .as_ref()
            .filter(|capture| capture.names.contains(name))
        {
            (capture.observe)(name, &self.download("hidden", valid * 1024)?)?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn debug_projection(&self) -> Result<Vec<f32>> {
        self.download("projected", self.decoded_frames * 640)
    }

    #[cfg(test)]
    pub(crate) fn debug_project_f64(&self, encoder: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            encoder.len().is_multiple_of(1024),
            "malformed debug encoder"
        );
        let mut weights = vec![0; 640 * 1024 * 4];
        let mut bias = vec![0; 640 * 4];
        self.arena
            .transfer("weight:encoder_projector.weight", &mut weights, false)?;
        self.arena
            .transfer("weight:encoder_projector.bias", &mut bias, false)?;
        let widen = |bytes: &[u8]| {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f64::from(f32::from_le_bytes(*v)))
                .collect::<Vec<_>>()
        };
        let weights = widen(&weights);
        let bias = widen(&bias);
        Ok(encoder
            .as_chunks::<1024>()
            .0
            .iter()
            .flat_map(|input| {
                weights
                    .as_chunks::<1024>()
                    .0
                    .iter()
                    .zip(&bias)
                    .map(move |(row, b)| {
                        (row.iter()
                            .zip(input)
                            .map(|(w, x)| w * f64::from(*x))
                            .sum::<f64>()
                            + b) as f32
                    })
            })
            .collect())
    }

    #[cfg(test)]
    pub(crate) fn debug_replay_projection(
        &mut self,
        projected: Vec<f32>,
        frames: usize,
    ) -> Result<synapse_speech::DecodeOutput> {
        match &mut self.decoder {
            Decoder::Host(joint) => {
                let saved = std::mem::replace(&mut joint.projected, projected);
                let result = synapse_speech::greedy(joint, frames, true, |_| {});
                joint.projected = saved;
                Ok(result?)
            }
            Decoder::Gpu(joint) => {
                // Replace the device projection for the replay, then restore it.
                let saved = read_f32(&self.arena, "projected", self.decoded_frames * 640)?;
                let mut bytes = synapse_core::encode_f32_frame(&projected);
                self.arena.transfer("projected", &mut bytes, true)?;
                joint.frames = frames;
                let result = synapse_speech::greedy(joint, frames, true, |_| {});
                let mut bytes = synapse_core::encode_f32_frame(&saved);
                self.arena.transfer("projected", &mut bytes, true)?;
                joint.frames = self.decoded_frames;
                Ok(result?)
            }
        }
    }

    fn run(&self, p: Params, buffers: [&str; 7], count: usize) -> Result<()> {
        self.batch
            .borrow_mut()
            .push(&self.arena, p, buffers, u32::try_from(count)?)
    }
    fn flush(&self) -> Result<()> {
        self.batch.borrow_mut().flush()
    }
    fn simple(&self, op: u32, input: &str, output: &str, rows: usize, cols: usize) -> Result<()> {
        self.run(
            Params {
                op,
                rows: rows as u32,
                cols: cols as u32,
                ..Default::default()
            },
            [input, "dummy", output, "dummy", "dummy", "dummy", "dummy"],
            rows * cols,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn linear(
        &self,
        input: &str,
        output: &str,
        name: &str,
        rows: usize,
        inner: usize,
        cols: usize,
        bias: bool,
    ) -> Result<()> {
        let bias_name = format!("weight:{name}.bias");
        self.run(
            Params {
                op: 100,
                rows: rows as u32,
                cols: cols as u32,
                inner: inner as u32,
                flags: if bias { SPEECH_GEMM_BIAS } else { 0 },
                ..Default::default()
            },
            [
                input,
                &format!("weight:{name}.weight"),
                output,
                "dummy",
                if bias { &bias_name } else { "dummy" },
                "dummy",
                "dummy",
            ],
            rows * cols,
        )
    }
    /// The query, key and value projections of `norm`, in one dispatch. Each
    /// output is computed exactly as by three separate bias-free `linear` calls.
    fn query_key_value(&self, attention: &str, rows: usize) -> Result<()> {
        let weight = |name: &str| format!("weight:{attention}.{name}.weight");
        self.run(
            Params {
                op: 100,
                rows: rows as u32,
                cols: 1024,
                inner: 1024,
                flags: SPEECH_GEMM_THREE,
                ..Default::default()
            },
            [
                "norm",
                &weight("q_proj"),
                "q",
                &weight("k_proj"),
                &weight("v_proj"),
                "k",
                "v",
            ],
            rows * 3 * 1024,
        )
    }
    fn norm(&self, input: &str, output: &str, name: &str, rows: usize) -> Result<()> {
        self.run(
            Params {
                op: 101,
                rows: rows as u32,
                cols: 1024,
                ..Default::default()
            },
            [
                input,
                &format!("weight:{name}.weight"),
                output,
                "dummy",
                &format!("weight:{name}.bias"),
                "dummy",
                "dummy",
            ],
            rows,
        )
    }
    fn residual(
        &self,
        hidden: &'static str,
        other: &str,
        rows: usize,
        scale: f32,
    ) -> Result<&'static str> {
        let output = hidden;
        // The subsampling workspace is no longer live. Ping-pong through it to
        // separate the bindings without introducing a copy dispatch.
        #[cfg(test)]
        let output = if self.debug_disjoint_bindings {
            if hidden == "hidden" {
                "sub_a"
            } else {
                "hidden"
            }
        } else {
            output
        };
        self.run(
            Params {
                op: 103,
                rows: rows as u32,
                cols: 1024,
                scale,
                ..Default::default()
            },
            [hidden, "dummy", output, other, "dummy", "dummy", "dummy"],
            rows * 1024,
        )?;
        Ok(output)
    }
    fn ff(
        &self,
        hidden: &'static str,
        layer: &str,
        index: usize,
        rows: usize,
    ) -> Result<&'static str> {
        self.norm(
            hidden,
            "norm",
            &format!("{layer}.norm_feed_forward{index}"),
            rows,
        )?;
        self.linear(
            "norm",
            "ff",
            &format!("{layer}.feed_forward{index}.linear1"),
            rows,
            1024,
            4096,
            false,
        )?;
        let activated = "ff";
        // Neither feed-forward block overlaps the lifetime of attention scores.
        #[cfg(test)]
        let activated = if self.debug_disjoint_bindings {
            "scores"
        } else {
            activated
        };
        self.simple(102, "ff", activated, rows, 4096)?;
        self.linear(
            activated,
            "out",
            &format!("{layer}.feed_forward{index}.linear2"),
            rows,
            4096,
            1024,
            false,
        )?;
        self.residual(hidden, "out", rows, 0.5)
    }

    #[allow(clippy::too_many_arguments)]
    fn encoder(
        &self,
        mel: &[f32],
        mel_rows: usize,
        valid: usize,
        rows: usize,
        valid_encoder: usize,
        progress: &mut impl FnMut() -> Result<()>,
    ) -> Result<()> {
        // A failed request may have left unsubmitted work. Never replay it in
        // the next request, whose scratch buffers are about to be overwritten.
        self.batch.borrow_mut().pending.clear();
        #[cfg(test)]
        if self.debug_disjoint_bindings {
            for (name, cols) in [("sub_a", 1024), ("scores", 4096)] {
                ensure!(
                    self.arena.descriptor(name)?.range >= (rows * cols * 4) as u64,
                    "encoder probe workspace {name} is too small for {rows} rows"
                );
            }
        }
        let mut input = synapse_core::encode_f32_frame(mel);
        self.arena.transfer("mel", &mut input, true)?;
        for start in (0..rows).step_by(TILE) {
            let n = TILE.min(rows - start);
            let mut source = "mel";
            let mut input_rows = mel_rows;
            let mut input_start = 0i32;
            let mut freq = 128;
            let mut valid_time = valid;
            for (index, offset, outtime) in transcribe::subsampling_stages(start, n) {
                let first = index == 0;
                let outfreq = freq / 2;
                valid_time = valid_time.div_ceil(2);
                let destination = if first { "sub_a" } else { "sub_b" };
                let base = format!("weight:encoder.subsampling.layers.{index}");
                self.run(
                    Params {
                        op: 113,
                        rows: outtime as u32,
                        cols: outfreq as u32,
                        inner: if first { 1 } else { 256 },
                        seq: input_rows as u32,
                        heads: freq as u32,
                        flags: u32::from(first),
                        window: valid_time as u32,
                        offset: offset as u32,
                        stride: input_start as u32,
                        ..Default::default()
                    },
                    [
                        source,
                        &format!("{base}.weight"),
                        destination,
                        "dummy",
                        &format!("{base}.bias"),
                        "dummy",
                        "dummy",
                    ],
                    outtime * outfreq * 256,
                )?;
                if !first {
                    let point = format!("weight:encoder.subsampling.layers.{}", index + 1);
                    self.run(
                        Params {
                            op: 104,
                            rows: outtime as u32,
                            cols: outfreq as u32,
                            window: valid_time as u32,
                            offset: offset as u32,
                            ..Default::default()
                        },
                        [
                            "sub_b",
                            &format!("{point}.weight"),
                            "sub_a",
                            "dummy",
                            &format!("{point}.bias"),
                            "dummy",
                            "dummy",
                        ],
                        outtime * outfreq * 256,
                    )?;
                }
                source = "sub_a";
                (input_rows, input_start, freq) = (outtime, offset, outfreq);
            }
            self.run(
                Params {
                    op: 100,
                    rows: n as u32,
                    cols: 1024,
                    inner: 4096,
                    flags: SPEECH_GEMM_BIAS,
                    offset: start as u32,
                    ..Default::default()
                },
                [
                    "sub_a",
                    "weight:encoder.subsampling.linear.weight",
                    "hidden",
                    "dummy",
                    "weight:encoder.subsampling.linear.bias",
                    "dummy",
                    "dummy",
                ],
                n * 1024,
            )?;
            self.flush()?;
        }
        #[cfg(test)]
        self.capture_layer("subsampling", valid_encoder)?;
        self.run(
            Params {
                op: 105,
                rows: rows as u32,
                ..Default::default()
            },
            ["dummy", "dummy", "pos", "dummy", "dummy", "dummy", "dummy"],
            (2 * rows - 1) * 1024,
        )?;
        for i in 0..24 {
            let layer = format!("encoder.layers.{i}");
            let hidden = self.ff("hidden", &layer, 1, rows)?;
            self.norm(hidden, "norm", &format!("{layer}.norm_self_att"), rows)?;
            let att = format!("{layer}.self_attn");
            self.query_key_value(&att, rows)?;
            self.linear(
                "pos",
                "relative",
                &format!("{att}.relative_k_proj"),
                2 * rows - 1,
                1024,
                1024,
                false,
            )?;
            for start in (0..rows).step_by(TILE) {
                let n = TILE.min(rows - start);
                let params = Params {
                    rows: n as u32,
                    seq: rows as u32,
                    offset: start as u32,
                    window: valid_encoder as u32,
                    ..Default::default()
                };
                self.run(
                    Params { op: 106, ..params },
                    [
                        "q",
                        "dummy",
                        "scores",
                        "k",
                        &format!("weight:{att}.bias_u"),
                        &format!("weight:{att}.bias_v"),
                        "relative",
                    ],
                    n * 8 * rows,
                )?;
                self.run(
                    Params { op: 107, ..params },
                    [
                        "scores", "dummy", "prob", "dummy", "dummy", "dummy", "dummy",
                    ],
                    n * 8,
                )?;
                self.run(
                    Params { op: 108, ..params },
                    ["prob", "dummy", "att", "dummy", "dummy", "dummy", "v"],
                    n * 1024,
                )?;
            }
            self.linear(
                "att",
                "out",
                &format!("{att}.o_proj"),
                rows,
                1024,
                1024,
                false,
            )?;
            let hidden = self.residual(hidden, "out", rows, 1.)?;
            self.norm(hidden, "norm", &format!("{layer}.norm_conv"), rows)?;
            let conv = format!("{layer}.conv");
            self.linear(
                "norm",
                "conv",
                &format!("{conv}.pointwise_conv1"),
                rows,
                1024,
                2048,
                false,
            )?;
            self.run(
                Params {
                    op: 109,
                    rows: rows as u32,
                    window: valid_encoder as u32,
                    ..Default::default()
                },
                ["conv", "dummy", "glu", "dummy", "dummy", "dummy", "dummy"],
                rows * 1024,
            )?;
            self.run(
                Params {
                    op: 110,
                    rows: rows as u32,
                    ..Default::default()
                },
                [
                    "glu",
                    &format!("weight:{conv}.depthwise_conv.weight"),
                    "norm",
                    "dummy",
                    "dummy",
                    "dummy",
                    "dummy",
                ],
                rows * 1024,
            )?;
            self.run(
                Params {
                    op: 111,
                    rows: rows as u32,
                    ..Default::default()
                },
                [
                    "norm",
                    &format!("weight:{conv}.norm.weight"),
                    "out",
                    &format!("weight:{conv}.norm.running_var"),
                    &format!("weight:{conv}.norm.bias"),
                    &format!("weight:{conv}.norm.running_mean"),
                    "dummy",
                ],
                rows * 1024,
            )?;
            self.linear(
                "out",
                "norm",
                &format!("{conv}.pointwise_conv2"),
                rows,
                1024,
                1024,
                false,
            )?;
            let hidden = self.residual(hidden, "norm", rows, 1.)?;
            let hidden = self.ff(hidden, &layer, 2, rows)?;
            self.norm(hidden, "norm", &format!("{layer}.norm_out"), rows)?;
            self.simple(112, "norm", "hidden", rows * 1024, 1)?;
            self.flush()?;
            #[cfg(test)]
            self.capture_layer(&format!("layer-{i:02}"), valid_encoder)?;
            progress()?;
        }
        self.linear(
            "hidden",
            "projected",
            "encoder_projector",
            valid_encoder,
            1024,
            640,
            true,
        )?;
        self.flush()
    }

    fn download(&self, name: &str, count: usize) -> Result<Vec<f32>> {
        read_f32(&self.arena, name, count)
    }

    pub fn transcribe(
        &mut self,
        mel: &[f32],
        mel_rows: usize,
        valid: usize,
        observe: bool,
        progress: &mut impl FnMut() -> Result<()>,
    ) -> Result<Output> {
        let valid_encoder_frames = transcribe::with_validated_input(
            mel,
            mel_rows,
            valid,
            self.maximum,
            |valid_encoder_frames, rows| {
                let mut batch = self.batch.borrow_mut();
                batch.policy.select_rung(rows);
                #[cfg(test)]
                {
                    batch.stats = Default::default();
                }
                drop(batch);
                self.encoder(mel, mel_rows, valid, rows, valid_encoder_frames, progress)?;
                Ok(valid_encoder_frames)
            },
        )?;
        // Observation only copies the finished encoder. The projection and joint
        // evaluations are identical in both modes.
        let encoder = if observe {
            self.download("hidden", valid_encoder_frames * 1024)?
        } else {
            Vec::new()
        };
        self.decoded_frames = valid_encoder_frames;
        if let Decoder::Host(joint) = &mut self.decoder {
            joint.projected = read_f32(&self.arena, "projected", valid_encoder_frames * 640)?;
        }
        let mut progress_error = None;
        let mut report = |_: usize| {
            if progress_error.is_none() {
                progress_error = progress().err();
            }
        };
        let decoded = match &mut self.decoder {
            Decoder::Host(joint) => {
                synapse_speech::greedy(joint, valid_encoder_frames, observe, &mut report)?
            }
            Decoder::Gpu(joint) => {
                joint.frames = valid_encoder_frames;
                // A failed submission surfaces as its own Vulkan error rather
                // than as the decode rule's generic joint error.
                synapse_speech::greedy(joint, valid_encoder_frames, observe, &mut report)
                    .map_err(|error| joint.take_device_error().unwrap_or_else(|| error.into()))?
            }
        };
        if let Some(error) = progress_error {
            return Err(error);
        }
        Ok(Output {
            valid_encoder_frames,
            peak_device_memory_bytes: Some(self.device_bytes()),
            peak_storage_buffer_bytes: Some(self.largest_storage_buffer),
            encoder,
            triples: decoded
                .triples
                .into_iter()
                .map(|t| TranscriptRecord {
                    token_id: t[0] as i32,
                    frame_index: t[1] as i32,
                    duration: t[2] as i32,
                })
                .collect(),
            trace: decoded
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
        })
    }
}

fn read_f32(arena: &Arena, name: &str, count: usize) -> Result<Vec<f32>> {
    let mut bytes = vec![0; count * 4];
    arena.transfer(name, &mut bytes, false)?;
    Ok(synapse_core::decode_f32_frame(&bytes)?)
}
