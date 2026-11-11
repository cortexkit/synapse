//! GPU encoder and encoder projection; the sequential joint stays in this worker.
use super::{Arena, Params, Prepared};
use crate::transcribe::{self, joint::HostJoint, TILE};
use anyhow::{ensure, Result};
use synapse_core::{DecodeTraceRecord, TranscriptRecord};
use synapse_parity::{manifest::Profile, safetensors::Header};

use crate::transcribe::Output;

#[cfg(test)]
type LayerObserver = Box<dyn Fn(&str, &[f32]) -> Result<()>>;

#[cfg(test)]
struct LayerCapture {
    names: std::collections::BTreeSet<String>,
    observe: LayerObserver,
}

pub struct Engine {
    arena: Arena,
    maximum: usize,
    joint: HostJoint,
    #[cfg(test)]
    layer_capture: Option<LayerCapture>,
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
        let plan = transcribe::buffer_plan(model, maximum)?;
        ensure!(
            synapse_parity::vulkan::arena_bytes(&plan) == required.min_device_local_bytes,
            "manifest_mismatch"
        );
        let arena = Arena::new(prepared.device, &plan, required.min_device_local_bytes)?;
        for (name, tensor) in &header.tensors {
            let mut bytes = data[tensor.data_offsets.0..tensor.data_offsets.1].to_vec();
            arena.transfer(&format!("weight:{name}"), &mut bytes, true)?;
        }
        eprintln!(
            "{}",
            serde_json::json!({"vulkan_transcribe_load_report": {
                "requested_device_local_bytes": arena.requested_bytes,
                "driver_memory_requirements_size": arena.driver_bytes,
                "maximum_encoder_frames": maximum,
                "min_storage_buffer_range": required.min_storage_buffer_range,
                "allocations": 1
            }})
        );
        Ok(Self {
            arena,
            maximum,
            joint: HostJoint::load(header, data),
            #[cfg(test)]
            layer_capture: None,
        })
    }

    pub fn device_bytes(&self) -> u64 {
        self.arena.driver_bytes
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
    pub(crate) fn debug_projection(&self) -> &[f32] {
        &self.joint.projected
    }

    #[cfg(test)]
    pub(crate) fn debug_project_f64(&self, encoder: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            encoder.len().is_multiple_of(1024),
            "malformed debug encoder"
        );
        let mut weights = vec![0; 640 * 1024 * 2];
        let mut bias = vec![0; 640 * 2];
        self.arena
            .transfer("weight:encoder_projector.weight", &mut weights, false)?;
        self.arena
            .transfer("weight:encoder_projector.bias", &mut bias, false)?;
        let widen = |bytes: &[u8]| {
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| f64::from(half::f16::from_le_bytes(*v).to_f32()))
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
        let saved = std::mem::replace(&mut self.joint.projected, projected);
        let result = synapse_speech::greedy(&mut self.joint, frames, true, |_| {});
        self.joint.projected = saved;
        Ok(result?)
    }

    fn run(&self, p: Params, buffers: [&str; 7], count: usize) -> Result<()> {
        self.arena.dispatch(p, buffers, u32::try_from(count)?)
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
                flags: u32::from(bias),
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
    fn residual(&self, other: &str, rows: usize, scale: f32) -> Result<()> {
        self.run(
            Params {
                op: 103,
                rows: rows as u32,
                cols: 1024,
                scale,
                ..Default::default()
            },
            [
                "hidden", "dummy", "hidden", other, "dummy", "dummy", "dummy",
            ],
            rows * 1024,
        )
    }
    fn ff(&self, layer: &str, index: usize, rows: usize) -> Result<()> {
        self.norm(
            "hidden",
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
        self.simple(102, "ff", "ff", rows, 4096)?;
        self.linear(
            "ff",
            "out",
            &format!("{layer}.feed_forward{index}.linear2"),
            rows,
            4096,
            1024,
            false,
        )?;
        self.residual("out", rows, 0.5)
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
                    flags: 1,
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
            self.ff(&layer, 1, rows)?;
            self.norm("hidden", "norm", &format!("{layer}.norm_self_att"), rows)?;
            let att = format!("{layer}.self_attn");
            for (buffer, name) in [("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj")] {
                self.linear(
                    "norm",
                    buffer,
                    &format!("{att}.{name}"),
                    rows,
                    1024,
                    1024,
                    false,
                )?;
            }
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
            self.residual("out", rows, 1.)?;
            self.norm("hidden", "norm", &format!("{layer}.norm_conv"), rows)?;
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
            self.residual("norm", rows, 1.)?;
            self.ff(&layer, 2, rows)?;
            self.norm("hidden", "norm", &format!("{layer}.norm_out"), rows)?;
            self.simple(112, "norm", "hidden", rows * 1024, 1)?;
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
        )
    }

    fn download(&self, name: &str, count: usize) -> Result<Vec<f32>> {
        let mut bytes = vec![0; count * 4];
        self.arena.transfer(name, &mut bytes, false)?;
        Ok(synapse_core::decode_f32_frame(&bytes)?)
    }

    pub fn transcribe(
        &mut self,
        mel: &[f32],
        mel_rows: usize,
        valid: usize,
        observe: bool,
        progress: &mut impl FnMut() -> Result<()>,
    ) -> Result<Output> {
        let (valid_encoder_frames, rows) = transcribe::input_shape(mel_rows, valid, self.maximum)?;
        ensure!(
            mel.len() == mel_rows * 128 && mel.iter().all(|x| x.is_finite()),
            "invalid_mel_frames"
        );
        self.encoder(mel, mel_rows, valid, rows, valid_encoder_frames, progress)?;
        // Observation only copies the finished encoder. The projection and joint
        // evaluations are identical in both modes.
        let encoder = if observe {
            self.download("hidden", valid_encoder_frames * 1024)?
        } else {
            Vec::new()
        };
        self.joint.projected = self.download("projected", valid_encoder_frames * 640)?;
        let mut progress_error = None;
        let decoded =
            synapse_speech::greedy(&mut self.joint, valid_encoder_frames, observe, |_| {
                if progress_error.is_none() {
                    progress_error = progress().err();
                }
            })?;
        if let Some(error) = progress_error {
            return Err(error);
        }
        Ok(Output {
            valid_encoder_frames,
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
