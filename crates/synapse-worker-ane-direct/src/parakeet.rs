//! FastConformer lowering and the host-resident TDT predictor/joint. Subsampling,
//! full attention and encoder projection belong to the 24 admitted ANE programs.
use crate::backend::{Model, Profile};
use crate::modernbert::shape;
#[cfg(test)]
use ane::PadFillMode;
use ane::{Convolution2dDescriptor, Graph, PadMode, Shape, Tensor};
use anyhow::{ensure, Result};
use synapse_speech::{Joint, JointLogits};

#[cfg(test)]
#[path = "parakeet_attention_oracle.rs"]
mod attention_oracle;
#[cfg(test)]
#[path = "parakeet_hardware.rs"]
mod hardware;

pub const HIDDEN: usize = 1024;
pub const DECODER: usize = 640;
const HEADS: usize = 8;
const HEAD: usize = HIDDEN / HEADS;

pub struct Output {
    pub frames: usize,
    pub encoder: Vec<f32>,
    pub decoded: synapse_speech::DecodeOutput,
}

pub fn max_encoder_frames(profile: &Profile) -> Result<usize> {
    let ms = profile.numeric["transcribe"]["max_duration_ms"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("transcribe_envelope_missing"))?;
    let mel = ms
        .checked_mul(16)
        .ok_or_else(|| anyhow::anyhow!("transcribe_envelope_invalid"))?
        / 160;
    let frames = usize::try_from(mel.div_ceil(8))?;
    crate::backend::rung(frames)?;
    Ok(frames)
}

pub fn validate_load(profile: &Profile, max: u32) -> Result<()> {
    let rung = crate::backend::rung(max_encoder_frames(profile)?)?;
    ensure!(
        (rung * 8 + 1) * 512 <= max as usize,
        "transcribe_max_frame_exceeded"
    );
    Ok(())
}

pub fn validate_input(
    profile: &Profile,
    mel: &[f32],
    frames: usize,
    valid: usize,
) -> Result<usize> {
    ensure!(
        valid > 0 && valid.checked_add(1) == Some(frames),
        "invalid_mel_frames"
    );
    ensure!(
        frames.checked_mul(128) == Some(mel.len()),
        "invalid_mel_frame_length"
    );
    ensure!(mel.iter().all(|x| x.is_finite()), "nonfinite_mel");
    ensure!(
        mel[valid * 128..].iter().all(|x| *x == 0.0),
        "invalid_mel_padding"
    );
    let encoder = valid.div_ceil(8);
    ensure!(
        encoder <= max_encoder_frames(profile)?,
        "transcribe_duration_limit"
    );
    Ok(encoder)
}

pub fn mel_shape(rung: usize) -> Shape {
    Shape {
        batch: 1,
        channels: 1,
        height: rung * 8,
        width: 128,
    }
}

fn scalar(g: &mut Graph, value: f32) -> Tensor {
    g.constant_with_scalar(value, Shape::channels(1))
}

fn linear(
    g: &mut Graph,
    x: Tensor,
    model: &Model,
    name: &str,
    out: usize,
    bias: bool,
) -> Result<Tensor> {
    let weight = model.tensor(&format!("{name}.weight"))?;
    ensure!(
        weight.len() == x.shape.channels * out,
        "tensor_shape_mismatch:{name}"
    );
    let y = g.inner_product(x, weight, x.shape.channels, out);
    if !bias {
        return Ok(y);
    }
    let b = model.tensor(&format!("{name}.bias"))?;
    ensure!(b.len() == out, "tensor_shape_mismatch:{name}");
    let b = g.constant(b, Shape::channels(out));
    Ok(g.addition(y, b))
}

fn norm(g: &mut Graph, x: Tensor, model: &Model, name: &str) -> Result<Tensor> {
    let mean = g.reduce_mean(x, 1);
    let centered = g.subtraction(x, mean);
    // Scale before squaring to avoid fp16 overflow. Scaling epsilon too keeps
    // ordinary centered LayerNorm.
    let abs = g.absolute(centered);
    let max = g.reduce_max(abs, 1);
    let root_eps = scalar(g, 1e-5f32.sqrt());
    let scale = g.maximum(max, root_eps);
    let scaled = g.division(centered, scale);
    let square = g.multiplication(scaled, scaled);
    let var = g.reduce_mean(square, 1);
    let eps = g.division(root_eps, scale);
    let eps = g.multiplication(eps, eps);
    let var = g.addition(var, eps);
    let exponent = scalar(g, -0.5);
    let inv = g.power(var, exponent);
    let normalized = g.multiplication(scaled, inv);
    let w = g.constant(
        model.tensor(&format!("{name}.weight"))?,
        Shape::channels(HIDDEN),
    );
    let b = g.constant(
        model.tensor(&format!("{name}.bias"))?,
        Shape::channels(HIDDEN),
    );
    let y = g.multiplication(normalized, w);
    Ok(g.addition(y, b))
}

fn silu(g: &mut Graph, x: Tensor) -> Tensor {
    let sigmoid = g.sigmoid(x);
    g.multiplication(x, sigmoid)
}

fn feed_forward(
    g: &mut Graph,
    x: Tensor,
    model: &Model,
    base: &str,
    index: usize,
) -> Result<Tensor> {
    let y = norm(g, x, model, &format!("{base}.norm_feed_forward{index}"))?;
    let y = linear(
        g,
        y,
        model,
        &format!("{base}.feed_forward{index}.linear1"),
        4096,
        false,
    )?;
    let y = silu(g, y);
    let y = linear(
        g,
        y,
        model,
        &format!("{base}.feed_forward{index}.linear2"),
        HIDDEN,
        false,
    )?;
    let half = scalar(g, 0.5);
    let y = g.multiplication(y, half);
    Ok(g.addition(x, y))
}

/// Descending relative positions, interleaved sine and cosine, in NCHW order.
fn positions(length: usize) -> Vec<f32> {
    relative_positions(length, HIDDEN)
}

fn relative_positions(length: usize, hidden: usize) -> Vec<f32> {
    let width = length * 2 - 1;
    let mut values = vec![0.0; hidden * width];
    for pair in 0..hidden / 2 {
        let freq = 1.0 / 10000f32.powf((2 * pair) as f32 / hidden as f32);
        for i in 0..width {
            let (sin, cos) = ((length as f32 - 1.0 - i as f32) * freq).sin_cos();
            values[(pair * 2) * width + i] = sin;
            values[(pair * 2 + 1) * width + i] = cos;
        }
    }
    values
}

fn head_view(g: &mut Graph, x: Tensor, length: usize) -> Tensor {
    let x = g.reshape(
        x,
        Shape {
            batch: 1,
            channels: HEADS,
            height: HEAD,
            width: length,
        },
    );
    g.transpose(x, [0, 1, 3, 2])
}

fn relative_row_offset(queries: usize, row: usize) -> usize {
    queries - 1 - row
}

fn relative_gather(g: &mut Graph, scores: Tensor, length: usize, queries: usize) -> Tensor {
    // The relative-position table is shared across queries. Gather each score
    // row using compile-time slice bounds instead of padding and reshaping an
    // odd-width score tensor, which the ANE compiler cannot lower. Only O(T)
    // slice metadata is added; no dense T-by-T selection constants are stored.
    let mut rows: Vec<_> = (0..queries)
        .map(|row| {
            g.slice(
                scores,
                [0, 0, row, relative_row_offset(queries, row)],
                [1, HEADS, 1, length],
            )
        })
        .collect();
    // Bounded fan-in keeps concatenations small even for a full query tile.
    while rows.len() > 1 {
        rows = rows
            .chunks(8)
            .map(|chunk| {
                if chunk.len() == 1 {
                    chunk[0]
                } else {
                    g.concat(chunk, 2)
                }
            })
            .collect();
    }
    rows[0]
}

fn attention(
    g: &mut Graph,
    x: Tensor,
    valid: Tensor,
    model: &Model,
    base: &str,
    length: usize,
) -> Result<Tensor> {
    let normalized = norm(g, x, model, &format!("{base}.norm_self_att"))?;
    let base = format!("{base}.self_attn");
    let q = linear(
        g,
        normalized,
        model,
        &format!("{base}.q_proj"),
        HIDDEN,
        false,
    )?;
    let k = linear(
        g,
        normalized,
        model,
        &format!("{base}.k_proj"),
        HIDDEN,
        false,
    )?;
    let v = linear(
        g,
        normalized,
        model,
        &format!("{base}.v_proj"),
        HIDDEN,
        false,
    )?;
    let q = head_view(g, q, length);
    let k = head_view(g, k, length);
    let v = head_view(g, v, length);
    let bias_shape = Shape {
        batch: 1,
        channels: HEADS,
        height: 1,
        width: HEAD,
    };
    let u = g.constant(model.tensor(&format!("{base}.bias_u"))?, bias_shape);
    let b = g.constant(model.tensor(&format!("{base}.bias_v"))?, bias_shape);
    let qu = g.addition(q, u);
    let qv = g.addition(q, b);
    let pos = g.constant(&positions(length), shape(length * 2 - 1, HIDDEN));
    let pos = linear(
        g,
        pos,
        model,
        &format!("{base}.relative_k_proj"),
        HIDDEN,
        false,
    )?;
    let pos = head_view(g, pos, length * 2 - 1);
    let scale = scalar(g, 1.0 / (HEAD as f32).sqrt());
    let one = scalar(g, 1.0);
    let sentinel = scalar(g, -10000.0);
    let mut outputs = Vec::new();
    // Tile queries, never keys: every query attends to the full sequence. A tile
    // needs only length + queries - 1 relative keys before the relative shift.
    for start in (0..length).step_by(128) {
        let queries = (length - start).min(128);
        let qu = g.slice(qu, [0, 0, start, 0], [1, HEADS, queries, HEAD]);
        let qv = g.slice(qv, [0, 0, start, 0], [1, HEADS, queries, HEAD]);
        let relative_length = length + queries - 1;
        let p = g.slice(
            pos,
            [0, 0, length - start - queries, 0],
            [1, HEADS, relative_length, HEAD],
        );
        let bd = g.matrix_multiplication(qv, p, false, true);
        let bd = relative_gather(g, bd, length, queries);
        let ac = g.matrix_multiplication(qu, k, false, true);
        let scores = g.addition(ac, bd);
        let scores = g.multiplication(scores, scale);
        let query_valid = g.slice(valid, [0, 0, 0, start], [1, 1, 1, queries]);
        let query_valid = g.transpose(query_valid, [0, 1, 3, 2]);
        let pair_valid = g.multiplication(valid, query_valid);
        let scores = g.multiplication(scores, pair_valid);
        let invalid = g.subtraction(one, pair_valid);
        let masked = g.multiplication(invalid, sentinel);
        let scores = g.addition(scores, masked);
        let scores = g.transpose(scores, [0, 3, 1, 2]);
        let probabilities = g.soft_max(scores, 1);
        let probabilities = g.transpose(probabilities, [0, 2, 3, 1]);
        // Match synapse-parakeet-mask-fix-v1: zero masked probabilities after
        // softmax, including completely padded query rows.
        let probabilities = g.multiplication(probabilities, pair_valid);
        outputs.push(g.matrix_multiplication(probabilities, v, false, false));
    }
    let y = g.concat(&outputs, 2);
    let y = g.transpose(y, [0, 1, 3, 2]);
    let y = g.reshape(y, shape(length, HIDDEN));
    let y = linear(g, y, model, &format!("{base}.o_proj"), HIDDEN, false)?;
    Ok(g.addition(x, y))
}

fn convolution(
    g: &mut Graph,
    x: Tensor,
    valid: Tensor,
    model: &Model,
    base: &str,
) -> Result<Tensor> {
    let y = norm(g, x, model, &format!("{base}.norm_conv"))?;
    let base = format!("{base}.conv");
    let y = linear(
        g,
        y,
        model,
        &format!("{base}.pointwise_conv1"),
        HIDDEN * 2,
        false,
    )?;
    let a = g.slice(y, [0, 0, 0, 0], [1, HIDDEN, 1, x.shape.width]);
    let b = g.slice(y, [0, HIDDEN, 0, 0], [1, HIDDEN, 1, x.shape.width]);
    let b = g.sigmoid(b);
    let y = g.multiplication(a, b);
    let y = g.multiplication(y, valid);
    let w = g.constant(
        model.tensor(&format!("{base}.depthwise_conv.weight"))?,
        Shape {
            batch: 1,
            channels: HIDDEN,
            height: 1,
            width: 9,
        },
    );
    let y = g.convolution_2d(
        y,
        w,
        None,
        &Convolution2dDescriptor {
            groups: HIDDEN,
            pad_mode: PadMode::Same,
        },
    );
    let weight = model.tensor(&format!("{base}.norm.weight"))?;
    let bias = model.tensor(&format!("{base}.norm.bias"))?;
    let mean = model.tensor(&format!("{base}.norm.running_mean"))?;
    let var = model.tensor(&format!("{base}.norm.running_var"))?;
    let scale: Vec<_> = weight
        .iter()
        .zip(var)
        .map(|(w, v)| w / (v + 1e-5).sqrt())
        .collect();
    let offset: Vec<_> = bias
        .iter()
        .zip(mean)
        .zip(&scale)
        .map(|((b, m), s)| b - m * s)
        .collect();
    let scale = g.constant(&scale, Shape::channels(HIDDEN));
    let offset = g.constant(&offset, Shape::channels(HIDDEN));
    let y = g.multiplication(y, scale);
    let y = g.addition(y, offset);
    let y = silu(g, y);
    let y = linear(
        g,
        y,
        model,
        &format!("{base}.pointwise_conv2"),
        HIDDEN,
        false,
    )?;
    Ok(g.addition(x, y))
}

const SUBSAMPLING: [(usize, usize, usize, usize); 5] = [
    (0, 3, 2, 1),
    (2, 3, 2, 256),
    (3, 1, 1, 1),
    (5, 3, 2, 256),
    (6, 1, 1, 1),
];

fn subsampling_stage(
    g: &mut Graph,
    x: Tensor,
    mut valid: Tensor,
    model: &Model,
    stage: (usize, usize, usize, usize),
) -> Result<(Tensor, Tensor)> {
    let (index, kernel, stride, groups) = stage;
    let base = format!("encoder.subsampling.layers.{index}");
    let w = g.constant(
        model.tensor(&format!("{base}.weight"))?,
        Shape {
            batch: x.shape.channels / groups,
            channels: 256,
            height: kernel,
            width: kernel,
        },
    );
    let b = g.constant(model.tensor(&format!("{base}.bias"))?, Shape::channels(256));
    let mut x = g.convolution_2d_strided(
        x,
        w,
        Some(b),
        &Convolution2dDescriptor {
            groups,
            pad_mode: if kernel == 1 {
                PadMode::Valid
            } else {
                PadMode::Same
            },
        },
        [stride, stride],
    );
    if stride == 2 {
        valid = g.max_pool(valid, 1, 1, 1, 2, PadMode::Valid);
    }
    let time_mask = g.transpose(valid, [0, 1, 3, 2]);
    x = g.multiplication(x, time_mask);
    if [0, 3, 6].contains(&index) {
        x = g.relu(x);
    }
    Ok((x, valid))
}

fn subsampling(
    g: &mut Graph,
    input: Tensor,
    mut valid: Tensor,
    model: &Model,
    length: usize,
) -> Result<Tensor> {
    let mut x = input;
    for stage in SUBSAMPLING {
        (x, valid) = subsampling_stage(g, x, valid, model, stage)?;
    }
    let x = g.transpose(x, [0, 1, 3, 2]);
    let x = g.reshape(x, shape(length, 256 * 16));
    linear(g, x, model, "encoder.subsampling.linear", HIDDEN, true)
}

#[cfg(test)]
fn probe_names() -> Vec<String> {
    let mut names = Vec::new();
    for (index, _, _, _) in SUBSAMPLING {
        names.push(format!("subsampling-stage-{index}"));
        if index >= 3 {
            names.push(format!("subsampling-stage-{index}-padded-width"));
        }
    }
    names.extend(
        [
            "subsampling-complete",
            "normalization",
            "feed-forward",
            "attention",
            "attention-qkv",
            "attention-content",
            "attention-position-projection",
            "attention-position-projection-constant",
            "attention-position-scores",
            "attention-relative-shift",
            "attention-static-gather",
            "attention-mask",
            "convolution",
            "conformer-block",
            "encoder-projection",
            "full-layer-0",
        ]
        .map(str::to_owned),
    );
    names
}

#[cfg(test)]
fn probe_graph(model: &Model, length: usize, name: &str) -> Result<Graph> {
    if name.starts_with("attention-") {
        return attention_probe(model, length, name);
    }
    if name == "full-layer-0" {
        return layer_graph(model, length, 0);
    }
    if name == "conformer-block" {
        return layer_graph(model, length, 1);
    }
    let mut g = Graph::new();
    if name == "encoder-projection" {
        let x = g.placeholder(shape(length, HIDDEN));
        linear(&mut g, x, model, "encoder_projector", DECODER, true)?;
    } else if name == "subsampling-complete" {
        let x = g.placeholder(mel_shape(length));
        let mask = g.placeholder(shape(length * 8, 1));
        subsampling(&mut g, x, mask, model, length)?;
    } else if ["normalization", "feed-forward", "attention", "convolution"].contains(&name) {
        let x = g.placeholder(shape(length, HIDDEN));
        let base = "encoder.layers.0";
        match name {
            "normalization" => {
                norm(&mut g, x, model, &format!("{base}.norm_self_att"))?;
            }
            "feed-forward" => {
                feed_forward(&mut g, x, model, base, 1)?;
            }
            _ => {
                let valid = g.placeholder(shape(length, 1));
                if name == "attention" {
                    attention(&mut g, x, valid, model, base, length)?;
                } else {
                    convolution(&mut g, x, valid, model, base)?;
                }
            }
        }
    } else {
        let mut input_shape = mel_shape(length);
        let mut found = false;
        for stage in SUBSAMPLING {
            if name.strip_suffix("-padded-width").unwrap_or(name)
                == format!("subsampling-stage-{}", stage.0)
            {
                if name.ends_with("-padded-width") {
                    input_shape.width = input_shape.width.max(64);
                }
                let x = g.placeholder(input_shape);
                let valid = g.placeholder(shape(input_shape.height, 1));
                subsampling_stage(&mut g, x, valid, model, stage)?;
                found = true;
                break;
            }
            input_shape.channels = 256;
            input_shape.height = input_shape.height.div_ceil(stage.2);
            input_shape.width = input_shape.width.div_ceil(stage.2);
        }
        ensure!(found, "unknown probe candidate: {name}");
    }
    Ok(g)
}

#[cfg(test)]
fn attention_probe(model: &Model, length: usize, name: &str) -> Result<Graph> {
    let mut g = Graph::new();
    let heads = |height, width| Shape {
        batch: 1,
        channels: HEADS,
        height,
        width,
    };
    let base = "encoder.layers.0.self_attn";
    match name {
        "attention-position-projection" | "attention-position-projection-constant" => {
            let constant = name.ends_with("-constant");
            let pos = if constant {
                g.constant(&positions(length), shape(length * 2 - 1, HIDDEN))
            } else {
                g.placeholder(shape(length * 2 - 1, HIDDEN))
            };
            let projected = linear(
                &mut g,
                pos,
                model,
                &format!("{base}.relative_k_proj"),
                HIDDEN,
                false,
            )?;
            let projected = head_view(&mut g, projected, length * 2 - 1);
            if constant {
                // Keep a dynamic input at the output, so the probe tests the
                // real constant projection branch inside an otherwise dynamic
                // program rather than submitting an input-free ANE program.
                let anchor = g.placeholder(projected.shape);
                g.addition(projected, anchor);
            }
        }
        "attention-position-scores" => {
            let qv = g.placeholder(heads(length, HEAD));
            let pos = g.placeholder(heads(length * 2 - 1, HEAD));
            g.matrix_multiplication(qv, pos, false, true);
        }
        "attention-relative-shift" | "attention-static-gather" => {
            let relative_length = length * 2 - 1;
            let bd = g.placeholder(heads(length, relative_length));
            if name == "attention-static-gather" {
                relative_gather(&mut g, bd, length, length);
                return Ok(g);
            }
            let bd = g.pad(bd, 0, 0, 1, 0, PadFillMode::Constant, 0.0);
            let bd = g.reshape(bd, heads(relative_length + 1, length));
            let bd = g.slice(bd, [0, 0, 1, 0], [1, HEADS, relative_length, length]);
            let bd = g.reshape(bd, heads(length, relative_length));
            g.slice(bd, [0, 0, 0, 0], [1, HEADS, length, length]);
        }
        "attention-mask" => {
            let scores = g.placeholder(heads(length, length));
            let valid = g.placeholder(shape(length, 1));
            let query = g.transpose(valid, [0, 1, 3, 2]);
            let pair_valid = g.multiplication(valid, query);
            let scores = g.multiplication(scores, pair_valid);
            let one = scalar(&mut g, 1.0);
            let invalid = g.subtraction(one, pair_valid);
            let sentinel = scalar(&mut g, -10000.0);
            let masked = g.multiplication(invalid, sentinel);
            g.addition(scores, masked);
        }
        "attention-qkv" | "attention-content" => {
            let x = g.placeholder(shape(length, HIDDEN));
            let q = linear(&mut g, x, model, &format!("{base}.q_proj"), HIDDEN, false)?;
            let k = linear(&mut g, x, model, &format!("{base}.k_proj"), HIDDEN, false)?;
            let v = linear(&mut g, x, model, &format!("{base}.v_proj"), HIDDEN, false)?;
            let q = head_view(&mut g, q, length);
            let k = head_view(&mut g, k, length);
            let v = head_view(&mut g, v, length);
            if name == "attention-content" {
                let u = g.constant(model.tensor(&format!("{base}.bias_u"))?, heads(1, HEAD));
                let q = g.addition(q, u);
                let scores = g.matrix_multiplication(q, k, false, true);
                let scale = scalar(&mut g, 1.0 / (HEAD as f32).sqrt());
                let scores = g.multiplication(scores, scale);
                let scores = g.transpose(scores, [0, 3, 1, 2]);
                let probabilities = g.soft_max(scores, 1);
                let probabilities = g.transpose(probabilities, [0, 2, 3, 1]);
                g.matrix_multiplication(probabilities, v, false, false);
            }
        }
        _ => anyhow::bail!("unknown attention probe: {name}"),
    }
    Ok(g)
}

pub fn layer_graph(model: &Model, length: usize, layer: usize) -> Result<Graph> {
    ensure!(layer < 24, "invalid_layer");
    let mut g = Graph::new();
    let input = g.placeholder(if layer == 0 {
        mel_shape(length)
    } else {
        shape(length, HIDDEN)
    });
    let mask = g.placeholder(shape(length * 8, 1));
    let mut x = if layer == 0 {
        subsampling(&mut g, input, mask, model, length)?
    } else {
        input
    };
    let mut valid = mask;
    for _ in 0..3 {
        valid = g.max_pool(valid, 1, 1, 1, 2, PadMode::Valid);
    }
    let base = format!("encoder.layers.{layer}");
    x = feed_forward(&mut g, x, model, &base, 1)?;
    x = attention(&mut g, x, valid, model, &base, length)?;
    x = convolution(&mut g, x, valid, model, &base)?;
    x = feed_forward(&mut g, x, model, &base, 2)?;
    x = norm(&mut g, x, model, &format!("{base}.norm_out"))?;
    if layer == 23 {
        // Always expose the unprojected encoder as the first leaf output.
        // Both leaf outputs are computed even when observe=false, so copying an
        // observation never changes the graph or its arithmetic.
        let one = scalar(&mut g, 1.0);
        let _encoder_output = g.multiplication(x, one);
        let _projected = linear(&mut g, x, model, "encoder_projector", DECODER, true)?;
    }
    Ok(g)
}

/// CPU neural state stays inside the worker; the shared core owns all greedy
/// policy, including blanks, durations, ties and the per-frame symbol cap.
pub struct TdtJoint<'a> {
    model: &'a Model,
    encoder: &'a [f32],
    frames: usize,
    hidden: [Vec<f32>; 2],
    cell: [Vec<f32>; 2],
    prediction: Vec<f32>,
    logits: Vec<f32>,
}

impl<'a> TdtJoint<'a> {
    pub fn new(model: &'a Model, encoder: &'a [f32], frames: usize) -> Self {
        Self {
            model,
            encoder,
            frames,
            hidden: std::array::from_fn(|_| vec![0.0; DECODER]),
            cell: std::array::from_fn(|_| vec![0.0; DECODER]),
            prediction: vec![0.0; DECODER],
            logits: vec![],
        }
    }

    fn affine(&self, x: &[f32], name: &str, output: usize) -> Result<Vec<f32>> {
        let weight = self.model.tensor(&format!("{name}.weight"))?;
        let bias = self.model.tensor(&format!("{name}.bias"))?;
        ensure!(
            weight.len() == x.len() * output && bias.len() == output,
            "tensor_shape_mismatch:{name}"
        );
        Ok(weight
            .chunks_exact(x.len())
            .zip(bias)
            .map(|(row, bias)| row.iter().zip(x).fold(*bias, |sum, (w, x)| sum + w * x))
            .collect())
    }

    fn predict(&mut self, token: u32) -> Result<()> {
        let embedding = self.model.tensor("decoder.embedding.weight")?;
        let mut x = embedding
            .get(token as usize * DECODER..(token as usize + 1) * DECODER)
            .ok_or_else(|| anyhow::anyhow!("invalid_token"))?
            .to_vec();
        for layer in 0..2 {
            let wi = self
                .model
                .tensor(&format!("decoder.lstm.weight_ih_l{layer}"))?;
            let wh = self
                .model
                .tensor(&format!("decoder.lstm.weight_hh_l{layer}"))?;
            let bi = self
                .model
                .tensor(&format!("decoder.lstm.bias_ih_l{layer}"))?;
            let bh = self
                .model
                .tensor(&format!("decoder.lstm.bias_hh_l{layer}"))?;
            ensure!(
                wi.len() == 4 * DECODER * DECODER
                    && wh.len() == wi.len()
                    && bi.len() == 4 * DECODER
                    && bh.len() == bi.len(),
                "tensor_shape_mismatch:lstm"
            );
            let mut gates = vec![0.0; 4 * DECODER];
            for row in 0..gates.len() {
                let input = wi[row * DECODER..(row + 1) * DECODER]
                    .iter()
                    .zip(&x)
                    .fold(bi[row], |sum, (w, x)| sum + w * x);
                let recurrent = wh[row * DECODER..(row + 1) * DECODER]
                    .iter()
                    .zip(&self.hidden[layer])
                    .fold(bh[row], |sum, (w, h)| sum + w * h);
                gates[row] = input + recurrent;
            }
            for i in 0..DECODER {
                let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
                self.cell[layer][i] = sigmoid(gates[DECODER + i]) * self.cell[layer][i]
                    + sigmoid(gates[i]) * gates[2 * DECODER + i].tanh();
                self.hidden[layer][i] =
                    sigmoid(gates[3 * DECODER + i]) * self.cell[layer][i].tanh();
            }
            x.clone_from(&self.hidden[layer]);
        }
        self.prediction = self.affine(&x, "decoder.decoder_projector", DECODER)?;
        Ok(())
    }
}

fn speech_error(_: anyhow::Error) -> synapse_speech::Error {
    synapse_speech::Error::InvalidJoint("invalid predictor or joint tensor")
}

impl Joint for TdtJoint<'_> {
    fn reset(&mut self) -> synapse_speech::Result<()> {
        for state in self.hidden.iter_mut().chain(&mut self.cell) {
            state.fill(0.0);
        }
        self.predict(synapse_speech::BLANK).map_err(speech_error)
    }
    fn accept(&mut self, token: u32) -> synapse_speech::Result<()> {
        self.predict(token).map_err(speech_error)
    }
    fn evaluate(&mut self, frame_index: usize) -> synapse_speech::Result<JointLogits<'_>> {
        if frame_index >= self.frames || self.encoder.len() != self.frames * DECODER {
            return Err(synapse_speech::Error::InvalidJoint("invalid encoder frame"));
        }
        let joint: Vec<_> = self.encoder[frame_index * DECODER..(frame_index + 1) * DECODER]
            .iter()
            .zip(&self.prediction)
            .map(|(e, p)| (e + p).max(0.0))
            .collect();
        self.logits = self
            .affine(&joint, "joint.head", synapse_speech::VOCAB_SIZE + 5)
            .map_err(speech_error)?;
        Ok(JointLogits {
            tokens: &self.logits[..synapse_speech::VOCAB_SIZE],
            durations: &self.logits[synapse_speech::VOCAB_SIZE..],
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn profile() -> Profile {
        Profile::select("parakeet-tdt-0.6b-v3.ane-direct-worker", "transcribe").unwrap()
    }

    pub(crate) fn synthetic_model(graphs: bool) -> Model {
        let index: serde_json::Value = serde_json::from_str(include_str!(
            "../../../bench/parity/checkpoints/parakeet-tdt-0.6b-v3/tensor-index.json"
        ))
        .unwrap();
        let mut tensors = BTreeMap::new();
        for (name, entry) in index.as_object().unwrap() {
            if name == "__metadata__" || entry["dtype"] == "I64" {
                continue;
            }
            if name.starts_with("encoder.")
                && (!graphs
                    || (name.starts_with("encoder.layers.")
                        && !name.starts_with("encoder.layers.0.")
                        && !name.starts_with("encoder.layers.1.")
                        && !name.starts_with("encoder.layers.23.")))
            {
                continue;
            }
            let count = entry["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n.as_u64().unwrap() as usize)
                .product();
            let value = if name.ends_with("running_var")
                || (name.contains("norm") && name.ends_with("weight"))
            {
                1.0
            } else {
                0.0
            };
            tensors.insert(name.clone(), vec![value; count]);
        }
        Model {
            profile: profile(),
            tensors,
            resident: BTreeMap::new(),
        }
    }

    #[test]
    fn encoder_rungs_inventory_and_cache_identity_are_speech_specific() {
        let profile = profile();
        let inventory = profile.inventory(128, "test-os").unwrap();
        assert_eq!(inventory.executables.len(), 24);
        assert_eq!(
            inventory
                .executables
                .iter()
                .map(|e| e.layers.clone())
                .collect::<Vec<_>>(),
            (0..24).map(|n| vec![n]).collect::<Vec<_>>()
        );
        assert_eq!(inventory.cpu_stages, ["tdt_predictor", "tdt_joint"]);
        assert_eq!(crate::backend::rung(129).unwrap(), 256);
        assert_eq!(profile.key(128, 0, "os"), profile.key(128, 0, "os"));
        assert_ne!(profile.key(128, 0, "os"), profile.key(256, 0, "os"));
        assert_ne!(profile.key(128, 0, "os"), profile.key(128, 23, "os"));
    }

    #[test]
    fn frame_validation_preserves_reference_padding_and_envelope() {
        let p = profile();
        assert_eq!(max_encoder_frames(&p).unwrap(), 7500);
        assert!(validate_load(&p, 64 * 1024 * 1024).is_ok());
        assert!(validate_load(&p, 1024).is_err());
        let mut mel = vec![0.0; 1025 * 128];
        assert_eq!(validate_input(&p, &mel, 1025, 1024).unwrap(), 128);
        assert!(validate_input(&p, &mel, 1025, 1025).is_err());
        mel[1024 * 128] = 1.0;
        assert!(validate_input(&p, &mel, 1025, 1024).is_err());
        mel[0] = f32::NAN;
        assert!(validate_input(&p, &mel, 1025, 1024).is_err());
    }

    #[test]
    fn neural_predictor_uses_ifgo_gates_zero_state_and_shared_greedy() {
        let mut model = synthetic_model(false);
        for token in [synapse_speech::BLANK, 3] {
            model.tensors.get_mut("decoder.embedding.weight").unwrap()[token as usize * DECODER] =
                1.0;
        }
        for layer in 0..2 {
            model
                .tensors
                .get_mut(&format!("decoder.lstm.weight_ih_l{layer}"))
                .unwrap()[2 * DECODER * DECODER] = 1.0;
        }
        model
            .tensors
            .get_mut("decoder.decoder_projector.weight")
            .unwrap()[0] = 1.0;
        model.tensors.get_mut("joint.head.weight").unwrap()[3 * DECODER] = 1.0;
        model.tensors.get_mut("joint.head.bias").unwrap()[3] = 10.0;
        model.tensors.get_mut("joint.head.bias").unwrap()[synapse_speech::VOCAB_SIZE + 4] = 10.0;
        let projection = vec![0.0; 5 * DECODER];
        let mut joint = TdtJoint::new(&model, &projection, 5);
        joint.reset().unwrap();
        let cell0 = 0.5 * 1.0f32.tanh();
        let h0 = 0.5 * cell0.tanh();
        let cell1 = 0.5 * h0.tanh();
        let expected = 10.0 + 0.5 * cell1.tanh();
        assert!((joint.evaluate(0).unwrap().tokens[3] - expected).abs() < 1e-6);
        joint.accept(3).unwrap();
        assert!(joint.evaluate(0).unwrap().tokens[3] > expected);
        let mut completions = vec![];
        let on = synapse_speech::greedy(&mut joint, 5, true, |n| completions.push(n)).unwrap();
        let off = synapse_speech::greedy(&mut joint, 5, false, |_| {}).unwrap();
        assert_eq!(on.triples, [[3, 0, 4], [3, 4, 4]]);
        assert_eq!(on.triples, off.triples);
        assert_eq!(on.trace.len(), 2);
        assert!(off.trace.is_empty());
        assert!(completions.is_empty());
        assert!(joint.evaluate(5).is_err());
    }

    #[test]
    fn static_relative_gather_matches_pad_shift_and_avoids_unsupported_kernel() {
        let mut random = 7u32;
        for (length, queries) in [(128, 128), (256, 128), (512, 128), (8192, 128), (137, 9)] {
            let relative = length + queries - 1;
            for _head in 0..3 {
                let scores: Vec<_> = (0..queries * relative)
                    .map(|_| {
                        random = random.wrapping_mul(1664525).wrapping_add(1013904223);
                        (random >> 8) as f32 / (1u32 << 24) as f32 - 0.5
                    })
                    .collect();
                // Independent replay of the reference's leading pad, reshape,
                // skipped first row, reshape back, and final key-width slice.
                let padded: Vec<_> = scores
                    .chunks_exact(relative)
                    .flat_map(|row| std::iter::once(0.0).chain(row.iter().copied()))
                    .collect();
                let shifted = &padded[queries..];
                for row in 0..queries {
                    let reference = &shifted[row * relative..row * relative + length];
                    let offset = row * relative + relative_row_offset(queries, row);
                    let gathered = &scores[offset..offset + length];
                    assert!(reference
                        .iter()
                        .map(|x| x.to_bits())
                        .eq(gathered.iter().map(|x| x.to_bits())));
                }
            }
            let mut g = Graph::new();
            let scores = g.placeholder(Shape {
                batch: 1,
                channels: HEADS,
                height: queries,
                width: relative,
            });
            let output = relative_gather(&mut g, scores, length, queries);
            assert_eq!(
                output.shape,
                Shape {
                    batch: 1,
                    channels: HEADS,
                    height: queries,
                    width: length
                }
            );
            let (mil, weights) = g.source_payload();
            assert!(
                weights.is_empty(),
                "static gathers need no dense selection constants"
            );
            assert!(!mil.contains(" = pad("));
        }
        let model = synthetic_model(true);
        let (attention, _) = probe_graph(&model, 128, "attention")
            .unwrap()
            .source_payload();
        assert!(
            !attention.contains(" = pad("),
            "production attention still uses the unsupported padding/reshape shift"
        );
    }

    #[test]
    fn compiler_probe_metadata_is_available_without_private_api() {
        let model = synthetic_model(true);
        for name in probe_names() {
            let graph = probe_graph(&model, 128, &name).unwrap();
            let (count, inputs, outputs) = graph.diagnostic_shapes();
            println!(
                "PROBE_SOURCE {name}: operations={count} inputs={inputs:?} outputs={outputs:?}"
            );
            assert!(count > 0 && !inputs.is_empty() && !outputs.is_empty());
        }
    }

    #[test]
    fn first_and_last_programs_are_stable_and_keep_accelerated_edges() {
        let model = synthetic_model(true);
        let (first, weights) = layer_graph(&model, 128, 0).unwrap().source_payload();
        assert!(first.contains("val = tensor<int32, [2]>([2, 2])"));
        assert!(first.contains("tensor<fp16, [1, 1, 1024, 128]>"));
        let (again, other_weights) = layer_graph(&model, 128, 0).unwrap().source_payload();
        assert_eq!(first, again);
        assert_eq!(weights, other_weights);
        let (last, _) = layer_graph(&model, 128, 23).unwrap().source_payload();
        assert!(last.contains("[1, 640, 1, 128]"));
        assert!(!last.contains("val = tensor<int32, [2]>([2, 2])"));
        let outputs = last
            .lines()
            .find(|line| line.trim_start().starts_with("}"))
            .unwrap();
        assert!(
            outputs.contains(','),
            "last layer must return encoder and projection: {outputs}"
        );
    }

    #[test]
    fn tiled_relative_shift_matches_global_query_key_offsets() {
        // Independent integer labels expose any off-by-one in the reshape,
        // leading pad, skipped row, or the relative-key tile's starting index.
        for length in [128, 256, 512] {
            for start in (0..length).step_by(128) {
                let queries = (length - start).min(128);
                let relative = length + queries - 1;
                let mut padded = vec![];
                for _ in 0..queries {
                    padded.push(usize::MAX);
                    padded.extend(length - start - queries..2 * length - start - 1);
                }
                let shifted = &padded[queries..];
                for q in 0..queries {
                    for k in 0..length {
                        assert_eq!(shifted[q * relative + k], length - 1 - (start + q) + k);
                    }
                }
            }
        }
    }
}
