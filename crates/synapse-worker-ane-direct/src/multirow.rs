//! Experimental multi-row Neural Engine passes for the Qwen3 embedding lane.
//!
//! The single-row path in `backend.rs` runs one row per pass, padded to a
//! ladder rung of at least 128 tokens. This module builds layer programs that
//! carry several rows per pass. Nothing in serving calls it: a caller opts in
//! by naming a [`MultiRowShape`], compiling a [`MultiRowProgram`] with
//! [`Model::compile_multirow`] and running rows through it. The program owns
//! its executables, so the shared executable budget stays the caller's
//! responsibility (28 extra executables per Qwen3 shape).
//!
//! Measured on an Apple M4 (`docs/evidence/ane-multirow-dispatch`), a pass
//! costs roughly what its total width costs, with 256 columns the cheapest per
//! column. Rows gain most when they are much shorter than their rung and are
//! packed into narrow slots; [`RowLayout::WidthFolded`] was the fastest layout
//! and its outputs were bit-identical to the single-row path.
//!
//! Every layout keeps each row's attention inside that row. Qwen3 attention is
//! causal and rows are right-padded, so a real token never sees a padding
//! token; the multi-row graphs therefore need no padding-mask input. Padding
//! positions and empty slots produce values that are never read back.
use crate::backend::{rms_cpu, Model, LADDER};
use crate::qwen::rms;
use ane::{Executable, Graph, NSQualityOfService, Shape, Tensor, TensorData, MIN_SPATIAL_WIDTH};
use anyhow::{ensure, Context, Result};

/// The largest number of rows one pass may carry. Larger passes multiply the
/// activation surfaces and attention work without a measured benefit.
pub const MAX_ROWS: usize = 16;

/// The narrowest slot a row may be padded to.
pub const MIN_SLOT_WIDTH: usize = 16;

/// The widest slot, and the widest packed tensor: the top single-row rung.
pub const MAX_WIDTH: usize = LADDER[LADDER.len() - 1];

/// The additive bias that removes a key from a query's softmax. It matches the
/// value the single-row graph uses for causal and padding masks.
const MASKED: f32 = -10_000.0;

/// Attention processes queries in tiles of this many positions, as the
/// single-row graph does, so per-tile programs stay small.
const QUERY_TILE: usize = 128;

/// Where a pass places its rows in the NCHW activation tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RowLayout {
    /// Rows on the batch axis: `[rows, hidden, 1, width]`. Attention is one
    /// batched matrix multiplication per query tile. Measured unsafe: with
    /// three or more rows of width 128 it compiles and runs but returns wrong
    /// vectors for rows after the second, so [`Model::compile_multirow`]
    /// refuses it. Only the hardware experiment compiles it.
    BatchAxis,
    /// Rows side by side on width: `[1, hidden, 1, rows * width]`. Attention
    /// slices each row's own segment and runs it separately.
    WidthSegmented,
    /// Rows side by side on width for the projections; attention reshapes the
    /// rows onto the channel axis next to the heads (`heads * rows` channels),
    /// so one batched multiplication serves every row.
    WidthFolded,
    /// Rows side by side on width; attention spans the whole packed width and a
    /// block-diagonal causal mask hides other rows. This computes scores for
    /// every pair of rows and discards most of them.
    WidthBlockMask,
}

impl RowLayout {
    pub const ALL: [RowLayout; 4] = [
        RowLayout::BatchAxis,
        RowLayout::WidthSegmented,
        RowLayout::WidthFolded,
        RowLayout::WidthBlockMask,
    ];

    pub fn name(self) -> &'static str {
        match self {
            RowLayout::BatchAxis => "batch-axis",
            RowLayout::WidthSegmented => "width-segmented",
            RowLayout::WidthFolded => "width-folded",
            RowLayout::WidthBlockMask => "width-block-mask",
        }
    }

    pub fn parse(name: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|layout| layout.name() == name)
            .with_context(|| format!("unknown row layout {name}"))
    }
}

/// A compiled multi-row program's identity: the layout, how many row slots one
/// pass carries, and the padded width of each slot. Slots may be narrower than
/// the smallest single-row rung: a pass's cost follows its total width, so
/// short rows gain most from narrow slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MultiRowShape {
    pub layout: RowLayout,
    pub rows: usize,
    pub width: usize,
}

impl MultiRowShape {
    pub fn new(layout: RowLayout, rows: usize, width: usize) -> Result<Self> {
        ensure!((2..=MAX_ROWS).contains(&rows), "invalid_row_count");
        ensure!(
            width.is_power_of_two() && (MIN_SLOT_WIDTH..=MAX_WIDTH).contains(&width),
            "invalid_shape"
        );
        let tensor_width = match layout {
            RowLayout::BatchAxis => width,
            _ => rows * width,
        };
        // The Neural Engine cannot run tensors narrower than this, and the
        // packed width must stay within the longest single-row rung.
        ensure!(
            (MIN_SPATIAL_WIDTH..=MAX_WIDTH).contains(&tensor_width),
            "invalid_shape"
        );
        Ok(Self {
            layout,
            rows,
            width,
        })
    }

    /// `layout:rows:width`, for example `width-folded:4:128`.
    pub fn parse(spec: &str) -> Result<Self> {
        let parts: Vec<&str> = spec.split(':').collect();
        ensure!(parts.len() == 3, "expected layout:rows:width, got {spec}");
        Self::new(
            RowLayout::parse(parts[0])?,
            parts[1].parse()?,
            parts[2].parse()?,
        )
    }

    pub fn label(&self) -> String {
        format!("{}:{}:{}", self.layout.name(), self.rows, self.width)
    }

    /// Batch size and sequence length of the activation tensor.
    fn batch_and_sequence(&self) -> (usize, usize) {
        match self.layout {
            RowLayout::BatchAxis => (self.rows, self.width),
            _ => (1, self.rows * self.width),
        }
    }

    /// The activation tensor shape for `channels` channels.
    pub fn tensor_shape(&self, channels: usize) -> Shape {
        let (batch, sequence) = self.batch_and_sequence();
        Shape {
            batch,
            channels,
            height: 1,
            width: sequence,
        }
    }

    /// Flat offset of one row's channel value at one position in a dense
    /// row-major activation buffer of `channels` channels.
    pub fn index(&self, channels: usize, row: usize, channel: usize, position: usize) -> usize {
        debug_assert!(row < self.rows && position < self.width && channel < channels);
        match self.layout {
            RowLayout::BatchAxis => (row * channels + channel) * self.width + position,
            _ => channel * self.rows * self.width + row * self.width + position,
        }
    }

    /// The token position each sequence column holds inside its own row.
    /// Rotary position embeddings restart at zero for every row.
    pub fn local_positions(&self) -> Vec<usize> {
        let (_, sequence) = self.batch_and_sequence();
        (0..sequence).map(|column| column % self.width).collect()
    }
}

/// One dispatch of a planned call: a multi-row pass of program `program` (an
/// index into the shapes given to [`plan_passes`]) over the listed input rows,
/// or one row that the single-row path must run on its own.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Pass {
    MultiRow { program: usize, rows: Vec<usize> },
    Single(usize),
}

/// Split a call's rows into passes. Each row goes to the narrowest program
/// whose slot fits it (ties go to the earlier program); rows are taken in
/// input order, `rows` at a time per program, and a final partial pass leaves
/// its remaining slots empty. Rows no slot fits run on the single-row path.
/// Callers restore input order from the indices each pass carries.
pub fn plan_passes(lengths: &[usize], shapes: &[MultiRowShape]) -> Vec<Pass> {
    let mut passes = Vec::new();
    let mut pending: Vec<Vec<usize>> = vec![Vec::new(); shapes.len()];
    for (index, &length) in lengths.iter().enumerate() {
        let program = shapes
            .iter()
            .enumerate()
            .filter(|(_, shape)| length > 0 && length <= shape.width)
            .min_by_key(|(program, shape)| (shape.width, *program))
            .map(|(program, _)| program);
        let Some(program) = program else {
            passes.push(Pass::Single(index));
            continue;
        };
        pending[program].push(index);
        if pending[program].len() == shapes[program].rows {
            passes.push(Pass::MultiRow {
                program,
                rows: std::mem::take(&mut pending[program]),
            });
        }
    }
    for (program, rows) in pending.into_iter().enumerate() {
        if !rows.is_empty() {
            passes.push(Pass::MultiRow { program, rows });
        }
    }
    passes
}

/// Rotary tables for every sequence column, restarting at position zero for
/// each row: `[head_dim, sequence]`, row-major, matching
/// [`crate::modernbert::rope_tables`] for a single row.
pub fn packed_rope_tables(
    shape: &MultiRowShape,
    head_dim: usize,
    theta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let (single_cosine, single_sine) = crate::modernbert::rope_tables(shape.width, head_dim, theta);
    let positions = shape.local_positions();
    let sequence = positions.len();
    let mut cosine = vec![0.0; head_dim * sequence];
    let mut sine = vec![0.0; head_dim * sequence];
    for target in 0..head_dim {
        for (column, &position) in positions.iter().enumerate() {
            cosine[target * sequence + column] = single_cosine[target * shape.width + position];
            sine[target * sequence + column] = single_sine[target * shape.width + position];
        }
    }
    (cosine, sine)
}

/// Whether query column `query` may attend to key column `key` in a sequence
/// whose rows each span `row_width` columns: same row, and not in the future.
pub fn may_attend(query: usize, key: usize, row_width: usize) -> bool {
    key <= query && key / row_width == query / row_width
}

/// The additive attention bias of one query tile: `[count, key_count]`,
/// row-major, zero where [`may_attend`] allows the pair and masked elsewhere.
pub fn tile_bias(start: usize, count: usize, key_count: usize, row_width: usize) -> Vec<f32> {
    let mut bias = vec![MASKED; count * key_count];
    for query in 0..count {
        for key in 0..key_count {
            if may_attend(start + query, key, row_width) {
                bias[query * key_count + key] = 0.0;
            }
        }
    }
    bias
}

fn apply_rope(
    graph: &mut Graph,
    input: Tensor,
    cosine: &[f32],
    sine: &[f32],
    head_dim: usize,
) -> Tensor {
    let Shape {
        batch,
        channels,
        width,
        ..
    } = input.shape;
    let half = head_dim / 2;
    let first = graph.slice(input, [0, 0, 0, 0], [batch, channels, half, width]);
    let second = graph.slice(input, [0, 0, half, 0], [batch, channels, half, width]);
    let negative = graph.constant_with_scalar(-1.0, Shape::channels(1));
    let negative_second = graph.multiplication(negative, second);
    let rotated = graph.concat(&[negative_second, first], 2);
    let table_shape = Shape {
        batch: 1,
        channels: 1,
        height: head_dim,
        width,
    };
    let cosine = graph.constant(cosine, table_shape);
    let sine = graph.constant(sine, table_shape);
    let direct = graph.multiplication(input, cosine);
    let turned = graph.multiplication(rotated, sine);
    graph.addition(direct, turned)
}

/// Causal attention over `[batch, channels, sequence, head_dim]` query, key
/// and value tensors, tiled by query position. `row_width` limits attention to
/// keys of the query's own row (pass the sequence length for one row).
fn attention(
    graph: &mut Graph,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: Tensor,
    row_width: usize,
) -> Tensor {
    let Shape {
        batch,
        channels,
        height: sequence,
        width: head_dim,
    } = query.shape;
    let mut tiles = Vec::new();
    for start in (0..sequence).step_by(QUERY_TILE) {
        let count = QUERY_TILE.min(sequence - start);
        // Keys after the tile's last query are always masked; keys of earlier
        // rows are masked by the bias but still multiplied.
        let key_count = start + count;
        let q = graph.slice(query, [0, 0, start, 0], [batch, channels, count, head_dim]);
        let k = graph.slice(key, [0, 0, 0, 0], [batch, channels, key_count, head_dim]);
        let v = graph.slice(value, [0, 0, 0, 0], [batch, channels, key_count, head_dim]);
        let scores = graph.matrix_multiplication(q, k, false, true);
        let scores = graph.multiplication(scores, scale);
        let bias = graph.constant(
            &tile_bias(start, count, key_count, row_width),
            Shape {
                batch: 1,
                channels: 1,
                height: count,
                width: key_count,
            },
        );
        let scores = graph.addition(scores, bias);
        let logits = graph.transpose(scores, [0, 3, 1, 2]);
        let probabilities = graph.soft_max(logits, 1);
        let probabilities = graph.transpose(probabilities, [0, 2, 3, 1]);
        tiles.push(graph.matrix_multiplication(probabilities, v, false, false));
    }
    if tiles.len() == 1 {
        tiles[0]
    } else {
        graph.concat(&tiles, 2)
    }
}

/// One Qwen3 decoder layer over a multi-row activation tensor. The arithmetic
/// per row matches [`crate::qwen::layer_graph`]; only the row placement and
/// the attention partitioning differ.
pub fn layer_graph(
    graph: &mut Graph,
    input: Tensor,
    model: &Model,
    layer: usize,
    shape: &MultiRowShape,
) -> Result<Tensor> {
    let p = &model.profile;
    let hidden = p.n("hidden_size");
    let heads = p.n("num_attention_heads");
    let kv_heads = p.n("num_key_value_heads");
    let dim = p.n("head_dim");
    let intermediate = p.n("intermediate_size");
    let eps = p.f("norm_eps");
    let (batch, sequence) = shape.batch_and_sequence();
    let base = format!("{}layers.{layer}", p.prefix());
    let tensor = |name: &str| model.tensor(&format!("{base}.{name}.weight"));
    let normalized = rms(graph, input, tensor("input_layernorm")?, 1, eps);
    let mut qkv = Vec::new();
    for (name, count) in [
        ("q_proj", heads),
        ("k_proj", kv_heads),
        ("v_proj", kv_heads),
    ] {
        let projected = graph.inner_product(
            normalized,
            tensor(&format!("self_attn.{name}"))?,
            hidden,
            count * dim,
        );
        qkv.push(graph.reshape(
            projected,
            Shape {
                batch,
                channels: count,
                height: dim,
                width: sequence,
            },
        ));
    }
    let query = rms(graph, qkv[0], tensor("self_attn.q_norm")?, 2, eps);
    let key = rms(graph, qkv[1], tensor("self_attn.k_norm")?, 2, eps);
    let (cos, sin) = packed_rope_tables(shape, dim, p.f("rope_theta"));
    let query = apply_rope(graph, query, &cos, &sin, dim);
    let key = apply_rope(graph, key, &cos, &sin, dim);
    let repeat = |graph: &mut Graph, input: Tensor| {
        let mut slices = Vec::new();
        for h in 0..heads {
            slices.push(graph.slice(
                input,
                [0, h / (heads / kv_heads), 0, 0],
                [batch, 1, dim, sequence],
            ));
        }
        graph.concat(&slices, 1)
    };
    let key = repeat(graph, key);
    let value = repeat(graph, qkv[2]);
    let query = graph.transpose(query, [0, 1, 3, 2]);
    let key = graph.transpose(key, [0, 1, 3, 2]);
    let value = graph.transpose(value, [0, 1, 3, 2]);
    let scale = graph.constant_with_scalar((dim as f32).sqrt().recip(), Shape::channels(1));
    let context = match shape.layout {
        RowLayout::BatchAxis => attention(graph, query, key, value, scale, shape.width),
        RowLayout::WidthBlockMask => attention(graph, query, key, value, scale, shape.width),
        RowLayout::WidthSegmented => {
            let mut rows = Vec::new();
            for row in 0..shape.rows {
                let begin = [0, 0, row * shape.width, 0];
                let size = [1, heads, shape.width, dim];
                let q = graph.slice(query, begin, size);
                let k = graph.slice(key, begin, size);
                let v = graph.slice(value, begin, size);
                rows.push(attention(graph, q, k, v, scale, shape.width));
            }
            graph.concat(&rows, 2)
        }
        RowLayout::WidthFolded => {
            // [1, heads, rows * width, dim] and [1, heads * rows, width, dim]
            // share one row-major order: channel `h * rows + r` is head `h`
            // of row `r`, so each folded channel holds exactly one row.
            let folded = Shape {
                batch: 1,
                channels: heads * shape.rows,
                height: shape.width,
                width: dim,
            };
            let q = graph.reshape(query, folded);
            let k = graph.reshape(key, folded);
            let v = graph.reshape(value, folded);
            let context = attention(graph, q, k, v, scale, shape.width);
            graph.reshape(
                context,
                Shape {
                    batch: 1,
                    channels: heads,
                    height: sequence,
                    width: dim,
                },
            )
        }
    };
    let context = graph.transpose(context, [0, 1, 3, 2]);
    let context = graph.reshape(
        context,
        Shape {
            batch,
            channels: heads * dim,
            height: 1,
            width: sequence,
        },
    );
    let projected = graph.inner_product(context, tensor("self_attn.o_proj")?, heads * dim, hidden);
    let attended = graph.addition(input, projected);
    let normalized = rms(graph, attended, tensor("post_attention_layernorm")?, 1, eps);
    let gate = graph.inner_product(normalized, tensor("mlp.gate_proj")?, hidden, intermediate);
    let up = graph.inner_product(normalized, tensor("mlp.up_proj")?, hidden, intermediate);
    let sigmoid = graph.sigmoid(gate);
    let activated = graph.multiplication(gate, sigmoid);
    let gated = graph.multiplication(activated, up);
    let output = graph.inner_product(gated, tensor("mlp.down_proj")?, intermediate, hidden);
    Ok(graph.addition(attended, output))
}

/// The layer programs of one [`MultiRowShape`], with their activation surfaces.
pub struct MultiRowProgram {
    shape: MultiRowShape,
    executables: Vec<Executable>,
    a: TensorData,
    b: TensorData,
}

impl Drop for MultiRowProgram {
    fn drop(&mut self) {
        // Executables must be released inside a pool, as the single-row
        // residents are, or the Neural Engine keeps their capacity.
        ane::autoreleasepool(|_| self.executables.clear());
    }
}

fn supported(model: &Model) -> Result<()> {
    ensure!(
        !model.profile.modern() && model.profile.operation() == "embed",
        "model_unsupported: multi-row passes support only Qwen3 embedding profiles"
    );
    Ok(())
}

impl Model {
    /// Opt-in Qwen3 fallback on an explicitly resident ladder width. Ordinary
    /// run still selects the smallest fitting rung. Padding mask and last-token
    /// pooling use the real tokens, not the requested padded width.
    pub fn run_at_rung(&self, tokens: &[u32], width: usize) -> Result<Vec<f32>> {
        supported(self)?;
        ensure!(
            LADDER.contains(&width) && !tokens.is_empty() && tokens.len() <= width,
            "invalid_fallback_rung"
        );
        ane::autoreleasepool(|_| self.run_stages_at_rung(tokens, false, width))
    }

    /// Compile one program per layer for `shape`. Refuses
    /// [`RowLayout::BatchAxis`], which returned wrong rows on hardware.
    pub fn compile_multirow(&self, shape: MultiRowShape) -> Result<MultiRowProgram> {
        ensure!(
            shape.layout != RowLayout::BatchAxis,
            "layout_unsafe: batch-axis passes returned wrong rows on the Neural Engine"
        );
        self.compile_multirow_unchecked(shape)
    }

    /// Like [`Model::compile_multirow`], but also compiles
    /// [`RowLayout::BatchAxis`]. That layout compiles and runs but returns
    /// wrong vectors for some rows, which [`Model::compile_multirow`] refuses;
    /// only the hardware measurement test calls this, to record that defect.
    /// Returns the first compile error; executables compiled before it are
    /// released.
    pub(crate) fn compile_multirow_unchecked(
        &self,
        shape: MultiRowShape,
    ) -> Result<MultiRowProgram> {
        supported(self)?;
        let hidden = self.profile.n("hidden_size");
        ane::autoreleasepool(|_| {
            let mut executables = Vec::new();
            for layer in 0..self.profile.n("num_hidden_layers") {
                let mut graph = Graph::new();
                let input = graph.placeholder(shape.tensor_shape(hidden));
                layer_graph(&mut graph, input, self, layer, &shape)?;
                let executable = graph
                    .compile(NSQualityOfService::UserInteractive)
                    .map_err(|error| anyhow::anyhow!("layer {layer}: {error}"))?;
                executables.push(executable);
            }
            Ok(MultiRowProgram {
                shape,
                executables,
                a: TensorData::new(shape.tensor_shape(hidden)),
                b: TensorData::new(shape.tensor_shape(hidden)),
            })
        })
    }
}

impl MultiRowProgram {
    pub fn shape(&self) -> MultiRowShape {
        self.shape
    }

    pub fn executable_count(&self) -> usize {
        self.executables.len()
    }

    /// Embed up to `shape.rows` token rows in one pass. Each row must fit a
    /// slot. Returns one unit-length vector per input row, in input order.
    pub fn run(&self, model: &Model, rows: &[&[u32]]) -> Result<Vec<Vec<f32>>> {
        ane::autoreleasepool(|_| self.run_unpooled(model, rows))
    }

    fn run_unpooled(&self, model: &Model, rows: &[&[u32]]) -> Result<Vec<Vec<f32>>> {
        supported(model)?;
        let shape = self.shape;
        ensure!(
            !rows.is_empty() && rows.len() <= shape.rows,
            "invalid_request: {} rows for {} slots",
            rows.len(),
            shape.rows
        );
        ensure!(
            rows.iter()
                .all(|row| !row.is_empty() && row.len() <= shape.width),
            "invalid_request: every row needs 1..={} tokens",
            shape.width
        );
        let profile = &model.profile;
        let hidden = profile.n("hidden_size");
        let pad = profile.n("pad_token_id") as u32;
        let embeddings = model.tensor(&format!("{}embed_tokens.weight", profile.prefix()))?;
        let input = pack_rows(&shape, hidden, pad, rows, embeddings)?;
        self.a.copy_from_f32(&input);
        for (layer, executable) in self.executables.iter().enumerate() {
            let (src, dst) = if layer % 2 == 0 {
                (&self.a, &self.b)
            } else {
                (&self.b, &self.a)
            };
            executable.run_cached(&[src], &[dst])?;
        }
        let surface = if self.executables.len().is_multiple_of(2) {
            &self.a
        } else {
            &self.b
        };
        let raw = surface.read_f32();
        let norm = model.tensor(&format!("{}norm.weight", profile.prefix()))?;
        Ok(rows
            .iter()
            .enumerate()
            .map(|(row, tokens)| {
                let position = tokens.len() - 1;
                let hidden_state: Vec<f32> = (0..hidden)
                    .map(|channel| raw[shape.index(hidden, row, channel, position)])
                    .collect();
                embedding_tail(hidden_state, norm, profile.f("norm_eps"))
            })
            .collect())
    }
}

/// Gather token embeddings into the packed activation layout. Slots without an
/// input row, and positions after a row's last token, hold the pad token.
pub fn pack_rows(
    shape: &MultiRowShape,
    hidden: usize,
    pad: u32,
    rows: &[&[u32]],
    embeddings: &[f32],
) -> Result<Vec<f32>> {
    let mut input = vec![0.0; shape.rows * hidden * shape.width];
    for slot in 0..shape.rows {
        let tokens = rows.get(slot).copied().unwrap_or(&[]);
        for position in 0..shape.width {
            let token = tokens.get(position).copied().unwrap_or(pad) as usize;
            let embedding = embeddings
                .get(token * hidden..(token + 1) * hidden)
                .context("invalid_request")?;
            for (channel, value) in embedding.iter().enumerate() {
                input[shape.index(hidden, slot, channel, position)] = *value;
            }
        }
    }
    Ok(input)
}

/// The single-row path's Qwen3 embedding tail: final RMS norm of the last
/// token's hidden state, then L2 normalisation, in the same order and
/// precision as `Model::run`.
pub fn embedding_tail(mut row: Vec<f32>, norm_weight: &[f32], eps: f32) -> Vec<f32> {
    rms_cpu(&mut row, Some(norm_weight), eps);
    let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
    for value in &mut row {
        *value /= norm;
    }
    row
}

#[cfg(test)]
pub(crate) mod synthetic {
    use crate::backend::{Model, Profile};
    use std::collections::BTreeMap;

    /// A Qwen3 embedding model with deterministic made-up weights for the
    /// given layers and a small embedding table. Graph construction needs real
    /// tensor sizes but no real values, and no Neural Engine.
    pub fn qwen_model(layers: &[usize], vocabulary: usize) -> Model {
        let profile = Profile::select("qwen3-embedding-0.6b.ane-direct-worker", "embed").unwrap();
        let hidden = profile.n("hidden_size");
        let heads = profile.n("num_attention_heads");
        let kv_heads = profile.n("num_key_value_heads");
        let dim = profile.n("head_dim");
        let intermediate = profile.n("intermediate_size");
        let values = |seed: usize, count: usize, centre: f32| -> Vec<f32> {
            (0..count)
                .map(|i| {
                    let mixed = (i.wrapping_mul(2_654_435_761) ^ seed.wrapping_mul(40_503)) % 1999;
                    centre + (mixed as f32 / 1999.0 - 0.5) * 0.04
                })
                .collect()
        };
        let mut tensors = BTreeMap::new();
        for &layer in layers {
            let shapes = [
                ("input_layernorm", hidden, 1.0),
                ("post_attention_layernorm", hidden, 1.0),
                ("self_attn.q_norm", dim, 1.0),
                ("self_attn.k_norm", dim, 1.0),
                ("self_attn.q_proj", heads * dim * hidden, 0.0),
                ("self_attn.k_proj", kv_heads * dim * hidden, 0.0),
                ("self_attn.v_proj", kv_heads * dim * hidden, 0.0),
                ("self_attn.o_proj", hidden * heads * dim, 0.0),
                ("mlp.gate_proj", intermediate * hidden, 0.0),
                ("mlp.up_proj", intermediate * hidden, 0.0),
                ("mlp.down_proj", hidden * intermediate, 0.0),
            ];
            for (seed, (name, count, centre)) in shapes.into_iter().enumerate() {
                tensors.insert(
                    format!("layers.{layer}.{name}.weight"),
                    values(seed + 100 * layer, count, centre),
                );
            }
        }
        tensors.insert(
            "embed_tokens.weight".into(),
            values(7, vocabulary * hidden, 0.0),
        );
        tensors.insert("norm.weight".into(), values(8, hidden, 1.0));
        Model {
            profile,
            tensors,
            resident: BTreeMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn shapes() -> Vec<MultiRowShape> {
        let mut all = Vec::new();
        for layout in RowLayout::ALL {
            for (rows, width) in [(2, 128), (4, 128), (8, 128), (2, 256), (3, 512), (2, 64)] {
                all.push(MultiRowShape::new(layout, rows, width).unwrap());
            }
            if layout != RowLayout::BatchAxis {
                all.push(MultiRowShape::new(layout, 4, 32).unwrap());
            }
        }
        all
    }

    #[test]
    fn shape_refuses_single_rows_oversized_passes_and_unrunnable_widths() {
        assert!(MultiRowShape::new(RowLayout::WidthFolded, 1, 128).is_err());
        assert!(MultiRowShape::new(RowLayout::WidthFolded, MAX_ROWS + 1, 128).is_err());
        assert!(MultiRowShape::new(RowLayout::BatchAxis, 2, 100).is_err());
        // Slots narrower than 128 are allowed as long as the whole packed
        // width (rows times slot width) stays at least 64 columns, the
        // binding's minimum spatial width.
        assert!(MultiRowShape::new(RowLayout::WidthFolded, 2, 16).is_err());
        assert!(MultiRowShape::new(RowLayout::WidthFolded, 2, 32).is_ok());
        assert!(MultiRowShape::new(RowLayout::WidthFolded, 2, 64).is_ok());
        assert!(MultiRowShape::new(RowLayout::WidthFolded, 4, 16).is_ok());
        assert!(MultiRowShape::new(RowLayout::WidthFolded, 4, 8).is_err());
        assert!(MultiRowShape::new(RowLayout::BatchAxis, 4, 32).is_err());
        assert!(MultiRowShape::new(RowLayout::WidthFolded, 2, 8192).is_err());
        for shape in shapes() {
            assert_eq!(MultiRowShape::parse(&shape.label()).unwrap(), shape);
        }
        assert!(MultiRowShape::parse("width-folded:4").is_err());
        assert!(MultiRowShape::parse("height:4:128").is_err());
    }

    #[test]
    fn packed_index_is_a_bijection_with_the_documented_layouts() {
        let channels = 3;
        for shape in shapes() {
            let tensor = shape.tensor_shape(channels);
            let mut seen = vec![false; tensor.total_elements()];
            for row in 0..shape.rows {
                for channel in 0..channels {
                    for position in 0..shape.width {
                        let index = shape.index(channels, row, channel, position);
                        let expected = match shape.layout {
                            RowLayout::BatchAxis => {
                                assert_eq!(tensor.batch, shape.rows);
                                ((row * channels) + channel) * shape.width + position
                            }
                            _ => {
                                assert_eq!(tensor.width, shape.rows * shape.width);
                                channel * tensor.width + row * shape.width + position
                            }
                        };
                        assert_eq!(index, expected);
                        assert!(!seen[index], "{} reuses offset {index}", shape.label());
                        seen[index] = true;
                    }
                }
            }
            assert!(seen.into_iter().all(|used| used));
        }
    }

    #[test]
    fn rope_positions_restart_for_every_row() {
        for shape in shapes() {
            let (single_cos, single_sin) = crate::modernbert::rope_tables(shape.width, 8, 1e6);
            let (cos, sin) = packed_rope_tables(&shape, 8, 1e6);
            let sequence = shape.tensor_shape(1).width;
            for target in 0..8 {
                for column in 0..sequence {
                    let position = column % shape.width;
                    assert_eq!(
                        cos[target * sequence + column],
                        single_cos[target * shape.width + position]
                    );
                    assert_eq!(
                        sin[target * sequence + column],
                        single_sin[target * shape.width + position]
                    );
                }
            }
        }
    }

    #[test]
    fn attention_bias_is_causal_and_never_crosses_rows() {
        // One row: the bias must equal the causal mask the single-row graph
        // builds (each query sees itself and earlier keys only).
        for (start, count, key_count) in [(0, 128, 128), (128, 128, 256), (384, 128, 512)] {
            let mut single = vec![0.0; count * key_count];
            for q in 0..count {
                for k in start + q + 1..key_count {
                    single[q * key_count + k] = MASKED;
                }
            }
            assert_eq!(tile_bias(start, count, key_count, 512), single);
        }
        // Packed rows of 128: a query sees keys of its own row, up to itself.
        let bias = tile_bias(128, 128, 256, 128);
        for q in 0..128 {
            for k in 0..256 {
                let allowed = (128..=128 + q).contains(&k);
                assert_eq!(bias[q * 256 + k] == 0.0, allowed, "query {q} key {k}");
            }
        }
        assert!(!may_attend(128, 127, 128));
        assert!(may_attend(255, 128, 128));
        assert!(!may_attend(5, 6, 128));
    }

    #[test]
    fn passes_keep_every_row_once_in_order_and_send_long_rows_alone() {
        let shape = MultiRowShape::new(RowLayout::WidthFolded, 4, 128).unwrap();
        let lengths = [10, 128, 129, 50, 1, 300, 7, 9, 11, 12, 0];
        let passes = plan_passes(&lengths, &[shape]);
        let multi = |rows: Vec<usize>| Pass::MultiRow { program: 0, rows };
        assert_eq!(
            passes,
            vec![
                Pass::Single(2),
                multi(vec![0, 1, 3, 4]),
                Pass::Single(5),
                multi(vec![6, 7, 8, 9]),
                Pass::Single(10),
            ]
        );
        assert_eq!(
            plan_passes(&[3, 4, 5], &[shape]),
            vec![multi(vec![0, 1, 2])]
        );
        assert_eq!(
            plan_passes(&[3, 4], &[]),
            vec![Pass::Single(0), Pass::Single(1)]
        );
    }

    #[test]
    fn each_row_takes_the_narrowest_slot_that_fits() {
        let wide = MultiRowShape::new(RowLayout::WidthFolded, 4, 64).unwrap();
        let narrow = MultiRowShape::new(RowLayout::WidthFolded, 2, 32).unwrap();
        let lengths = [40, 20, 64, 32, 33, 65, 5, 1];
        let passes = plan_passes(&lengths, &[wide, narrow]);
        assert_eq!(
            passes,
            vec![
                Pass::MultiRow {
                    program: 1,
                    rows: vec![1, 3]
                },
                Pass::Single(5),
                Pass::MultiRow {
                    program: 1,
                    rows: vec![6, 7]
                },
                Pass::MultiRow {
                    program: 0,
                    rows: vec![0, 2, 4]
                },
            ]
        );
        let mut covered: Vec<usize> = passes
            .iter()
            .flat_map(|pass| match pass {
                Pass::MultiRow { program, rows } => {
                    let shape = [wide, narrow][*program];
                    assert!(rows.len() <= shape.rows);
                    assert!(rows.iter().all(|&row| lengths[row] <= shape.width));
                    rows.clone()
                }
                Pass::Single(row) => vec![*row],
            })
            .collect();
        covered.sort();
        assert_eq!(covered, (0..lengths.len()).collect::<Vec<_>>());
    }

    #[test]
    fn packing_places_rows_in_slots_and_pads_the_rest() {
        let hidden = 2;
        // Token t embeds as [t, -t].
        let embeddings: Vec<f32> = (0..10).flat_map(|t| [t as f32, -(t as f32)]).collect();
        for layout in RowLayout::ALL {
            let shape = MultiRowShape::new(layout, 3, 128).unwrap();
            let rows: [&[u32]; 2] = [&[1, 2, 3], &[4]];
            let input = pack_rows(&shape, hidden, 9, &rows, &embeddings).unwrap();
            let at = |row, channel, position| input[shape.index(hidden, row, channel, position)];
            assert_eq!([at(0, 0, 0), at(0, 0, 1), at(0, 0, 2)], [1.0, 2.0, 3.0]);
            assert_eq!(at(0, 1, 2), -3.0);
            assert_eq!(at(0, 0, 3), 9.0, "padding after a row");
            assert_eq!(at(1, 0, 0), 4.0);
            assert_eq!(at(1, 1, 127), -9.0);
            assert!((0..128).all(|p| at(2, 0, p) == 9.0), "empty slot is pad");
            assert!(pack_rows(&shape, hidden, 10, &rows, &embeddings).is_err());
        }
    }

    /// The submitted program as an order-independent digest. The binding emits
    /// constants in hash-map order, so the raw MIL text and weight blob change
    /// between runs. Each weight reference is replaced by the SHA-256 of the
    /// bytes it points at, and the lines are sorted; names tie each operation
    /// to its inputs, so the digest still pins every operation and value.
    fn canonical_payload_digest(graph: &Graph) -> String {
        let (mil, weights) = graph.source_payload();
        let marker = "offset = uint64(";
        let mut lines: Vec<String> = mil
            .lines()
            .map(|line| {
                let Some(start) = line.find(marker) else {
                    return line.to_owned();
                };
                let digits = &line[start + marker.len()..];
                let end = digits.find(')').unwrap();
                let chunk: usize = digits[..end].parse().unwrap();
                let word = |at: usize| {
                    u32::from_le_bytes(weights[at..at + 4].try_into().unwrap()) as usize
                };
                let (size, data) = (word(chunk + 8), word(chunk + 16));
                let digest = format!("{:x}", Sha256::digest(&weights[data..data + size]));
                format!("{}sha256 = {digest}{}", &line[..start], &digits[end..])
            })
            .collect();
        lines.sort();
        format!("{:x}", Sha256::digest(lines.join("\n").as_bytes()))
    }

    /// Canonical digest of the single-row Qwen3 layer program for synthetic
    /// layer 0 at width 128, recorded from the graph builder before multi-row
    /// support existed. A change here means the production single-row program,
    /// and so its outputs, changed.
    const SINGLE_ROW_LAYER_PROGRAM: &str =
        "ef8ab131df93fa0ef3c6dff81dcf03d857aee6d62997db8ee906a7651de05f14";

    #[test]
    fn single_row_layer_program_is_unchanged() {
        let model = synthetic::qwen_model(&[0], 4);
        let digests: Vec<String> = (0..2)
            .map(|_| {
                let mut graph = Graph::new();
                let hidden = model.profile.n("hidden_size");
                let input = graph.placeholder(crate::modernbert::shape(128, hidden));
                let mask = graph.placeholder(crate::modernbert::shape(128, 1));
                crate::qwen::layer_graph(&mut graph, input, mask, &model, 0, 128).unwrap();
                canonical_payload_digest(&graph)
            })
            .collect();
        assert_eq!(digests[0], digests[1], "canonical digest is not stable");
        assert_eq!(digests[0], SINGLE_ROW_LAYER_PROGRAM);
    }

    #[test]
    fn multirow_layer_programs_build_with_one_input_and_the_packed_output_shape() {
        let model = synthetic::qwen_model(&[0], 4);
        let hidden = model.profile.n("hidden_size");
        for layout in RowLayout::ALL {
            for (rows, width) in [(2, 128), (2, 256), (2, 64)] {
                let shape = MultiRowShape::new(layout, rows, width).unwrap();
                let mut graph = Graph::new();
                let input = graph.placeholder(shape.tensor_shape(hidden));
                let output = layer_graph(&mut graph, input, &model, 0, &shape).unwrap();
                assert_eq!(
                    output.shape,
                    shape.tensor_shape(hidden),
                    "{}",
                    shape.label()
                );
                let (mil, _) = graph.source_payload();
                let signature = mil.lines().find(|line| line.contains("func main")).unwrap();
                assert_eq!(signature.matches("tensor<fp16").count(), 1, "{signature}");
            }
        }
    }

    #[test]
    fn multirow_refuses_other_profiles_and_oversized_calls() {
        let model = synthetic::qwen_model(&[], 4);
        let shape = MultiRowShape::new(RowLayout::WidthFolded, 2, 128).unwrap();
        let program = MultiRowProgram {
            shape,
            executables: Vec::new(),
            a: TensorData::new(shape.tensor_shape(1)),
            b: TensorData::new(shape.tensor_shape(1)),
        };
        let long = vec![1; 129];
        for rows in [vec![], vec![&[1u32][..]; 3], vec![&long[..]], vec![&[][..]]] {
            let error = program.run(&model, &rows).unwrap_err().to_string();
            assert!(error.starts_with("invalid_request"), "{error}");
        }
        let gte = Model {
            profile: crate::backend::Profile::select(
                "gte-modernbert-base.ane-direct-worker",
                "embed",
            )
            .unwrap(),
            tensors: Default::default(),
            resident: Default::default(),
        };
        let error = program.run(&gte, &[&[1]]).unwrap_err().to_string();
        assert!(error.starts_with("model_unsupported"), "{error}");
        let error = gte.compile_multirow(shape).err().unwrap().to_string();
        assert!(error.starts_with("model_unsupported"), "{error}");
        // The batch axis is refused before any graph is built or compiled.
        let batch = MultiRowShape::new(RowLayout::BatchAxis, 2, 128).unwrap();
        let error = model.compile_multirow(batch).err().unwrap().to_string();
        assert!(error.starts_with("layout_unsafe"), "{error}");
    }

    #[test]
    fn explicit_fallback_refuses_narrow_nonladder_empty_and_nonresident_rungs() {
        let model = synthetic::qwen_model(&[], 4);
        for (tokens, width) in [
            (vec![1; 129], 128),
            (vec![1; 65], 64),
            (vec![], 256),
            (vec![1], 384),
        ] {
            assert!(model
                .run_at_rung(&tokens, width)
                .unwrap_err()
                .to_string()
                .contains("invalid_fallback_rung"));
        }
        assert!(model
            .run_at_rung(&[1; 65], 256)
            .unwrap_err()
            .to_string()
            .contains("shape_not_admitted"));
        assert!(model
            .run(&[1; 65])
            .unwrap_err()
            .to_string()
            .contains("shape_not_admitted"));
    }

    #[test]
    fn embedding_tail_matches_the_single_row_arithmetic() {
        let hidden_state: Vec<f32> = (0..16).map(|i| i as f32 * 0.37 - 2.0).collect();
        let weight: Vec<f32> = (0..16).map(|i| 1.0 + i as f32 * 0.01).collect();
        let mut expected = hidden_state.clone();
        rms_cpu(&mut expected, Some(&weight), 1e-6);
        let norm = expected
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt()
            .max(1e-12);
        let expected: Vec<f32> = expected.into_iter().map(|v| v / norm).collect();
        assert_eq!(embedding_tail(hidden_state, &weight, 1e-6), expected);
    }
}

/// Neural Engine measurement of one multi-row shape against the single-row
/// path, in one process. Ignored by default; the "Reproduce on the Mac mini"
/// section of `docs/evidence/ane-multirow-dispatch/README.md` has the full
/// procedure. Never run it on a machine whose production Synapse serves from
/// the Neural Engine: it compiles 56 or more executables (28 per Qwen3 shape,
/// single-row plus multi-row), and the Neural Engine holds a limited number
/// of compiled programs, so a second compiling process can leave production
/// unable to admit shapes.
///
/// Required environment:
/// - `ANE_MULTIROW_ARM`: the shape, `layout:rows:width` (for example
///   `width-folded:4:128`). Setting it is what enables the experiment.
/// - `ANE_TEST_PACKAGES`: directory holding `qwen3-embedding-0.6b.safetensors`,
///   the converted package pinned in `bench/parity/models.json`.
/// - `ANE_MULTIROW_INPUT`: an AFT `engram.jsonl` export (text is hashed, never
///   written out).
/// - `ANE_MULTIROW_TOKENIZER`: the checkpoint's `tokenizer.json`.
/// - `ANE_MULTIROW_OUT`: where to write the JSON report.
///
/// Optional: `ANE_MULTIROW_REPEATS` (timing repetitions, default 5);
/// `ANE_MULTIROW_REPLAY=1` (also replay the export's first 64-row batch);
/// `ANE_MULTIROW_REPLAY_EXTRA` (comma-separated extra shapes for the replay);
/// `ANE_MULTIROW_REPLAY_ALL=1` (also replay every batch in the export's
/// `.meta.json` plan).
#[path = "tight_packing.rs"]
pub mod tight;

#[cfg(test)]
#[path = "multirow_replay.rs"]
mod replay;

#[cfg(test)]
mod hardware {
    use super::*;
    use crate::backend::{rung, Profile};
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::time::Instant;

    pub(super) const SLUG: &str = "qwen3-embedding-0.6b";
    const SINGLE_ROW_GATE: f64 = 0.999;
    const MULTI_ROW_GATE: f64 = 0.9999;

    pub(super) fn load_average() -> f64 {
        let mut load = [0.0f64; 3];
        assert_eq!(unsafe { libc::getloadavg(load.as_mut_ptr(), 3) }, 3);
        load[0]
    }

    pub(super) fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let norm = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        dot / (norm(a) * norm(b))
    }

    pub(super) fn max_abs(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (*x as f64 - *y as f64).abs())
            .fold(0.0, f64::max)
    }

    pub(super) fn sha(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn ids_digest(ids: &[u32]) -> String {
        sha(&ids
            .iter()
            .flat_map(|id| id.to_le_bytes())
            .collect::<Vec<_>>())
    }

    pub(super) fn ms(started: Instant) -> f64 {
        started.elapsed().as_secs_f64() * 1000.0
    }

    pub(super) struct Chunk {
        pub(super) seq: u64,
        pub(super) text_sha256: String,
        pub(super) ids: Vec<u32>,
    }

    pub(super) fn tokenize_export(path: &str, tokenizer: &str, eos: u32) -> Vec<Chunk> {
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer).unwrap();
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| {
                let record: Value = serde_json::from_str(line).unwrap();
                let text = record["text"].as_str().unwrap();
                Chunk {
                    seq: record["seq"].as_u64().unwrap(),
                    text_sha256: sha(text.as_bytes()),
                    ids: compose(&tokenizer, text, eos),
                }
            })
            .collect()
    }

    /// The catalog's Qwen3 document composition: the tokenizer's own special
    /// tokens, then exactly one terminal end-of-text token.
    pub(super) fn compose(tokenizer: &tokenizers::Tokenizer, text: &str, eos: u32) -> Vec<u32> {
        let mut ids = tokenizer.encode(text, true).unwrap().get_ids().to_vec();
        if ids.last() == Some(&eos) {
            ids.pop();
        }
        ids.push(eos);
        ids
    }

    fn single(model: &Model, ids: &[u32]) -> (Vec<f32>, f64) {
        let started = Instant::now();
        let vector = model.run(ids).unwrap();
        (vector, ms(started))
    }

    #[test]
    #[ignore = "Neural Engine experiment; set ANE_MULTIROW_ARM and the inputs documented above"]
    fn multirow_experiment() {
        let Ok(arm) = std::env::var("ANE_MULTIROW_ARM") else {
            panic!("ANE_MULTIROW_ARM is required");
        };
        let shape = MultiRowShape::parse(&arm).unwrap();
        let env = |name: &str| std::env::var(name).unwrap_or_else(|_| panic!("{name} is required"));
        let out = env("ANE_MULTIROW_OUT");
        let repeats: usize = std::env::var("ANE_MULTIROW_REPEATS")
            .map(|v| v.parse().unwrap())
            .unwrap_or(5);
        let replay = std::env::var_os("ANE_MULTIROW_REPLAY").is_some();
        crate::worker::test_private_api().unwrap();
        let mut report = json!({
            "arm": shape.label(), "layout": shape.layout.name(), "rows": shape.rows,
            "width": shape.width, "repeats": repeats, "pid": std::process::id(),
            "release": !cfg!(debug_assertions),
        });
        let write = |report: &Value| {
            std::fs::write(&out, serde_json::to_vec_pretty(report).unwrap()).unwrap();
        };

        let profile = Profile::select(&format!("{SLUG}.ane-direct-worker"), "embed").unwrap();
        let digest = profile.numeric["converted_package_digest"]
            .as_str()
            .unwrap()
            .to_owned();
        let eos = profile.n("eos_token_id") as u32;
        let root = std::path::PathBuf::from(env("ANE_TEST_PACKAGES"));
        let mut model =
            Model::load(profile, &root.join(format!("{SLUG}.safetensors")), &digest).unwrap();
        let input = env("ANE_MULTIROW_INPUT");
        let tokenizer_path = env("ANE_MULTIROW_TOKENIZER");
        let chunks = tokenize_export(&input, &tokenizer_path, eos);
        report["input"] = json!({
            "export_sha256": sha(&std::fs::read(&input).unwrap()),
            "chunks": chunks.len(),
            "tokenizer_sha256": sha(&std::fs::read(&tokenizer_path).unwrap()),
            "composed_tokens_at_most": ([16, 32, 64, 128, 256, 512, 1024].map(|limit| {
                json!({"limit": limit, "chunks": chunks.iter().filter(|c| c.ids.len() <= limit).count()})
            })),
            "composed_tokens_total": chunks.iter().map(|c| c.ids.len()).sum::<usize>(),
        });

        // The parity fixture holds, per case, the text, the token ids the
        // catalog composed for it, and the fp32 reference vector. Re-tokenize
        // each text here and count exact id matches, so the export rows below
        // are known to be composed the way the catalog composes them.
        let fixture: Value = serde_json::from_slice(
            &std::fs::read(format!(
                "../../bench/parity/fixtures/{SLUG}/{SLUG}.ref-v1.transformers-5.16.1.seed-0.json"
            ))
            .unwrap(),
        )
        .unwrap();
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).unwrap();
        let mut composition_matches = 0;
        let mut composition_checked = 0;
        let mut references = Vec::new();
        for case in fixture["cases"].as_array().unwrap() {
            let ids: Vec<u32> = serde_json::from_value(case["input_ids"].clone()).unwrap();
            if let Some(text) = case["text"].as_str() {
                composition_checked += 1;
                composition_matches += usize::from(compose(&tokenizer, text, eos) == ids);
            }
            if ids.len() <= shape.width {
                let output: Vec<f32> = serde_json::from_value(case["output"].clone()).unwrap();
                references.push((case["id"].as_str().unwrap().to_owned(), ids, output));
            }
        }
        report["tokenizer_composition"] =
            json!({"fixture_cases_checked": composition_checked, "matching": composition_matches});

        // Single-row references run at each row's own rung, so every rung up
        // to the slot width must be resident.
        let mut admitted = Vec::new();
        let top_rung = rung(shape.width).unwrap();
        for &width in LADDER.iter().filter(|&&w| w <= top_rung) {
            let load = load_average();
            let started = Instant::now();
            model.admit(width, "multirow-experiment").unwrap();
            admitted.push(json!({
                "width": width, "ms": ms(started), "executables": 28,
                "load_before": load, "load_after": load_average(),
            }));
        }
        report["single_row_compile"] = admitted.pop().unwrap();
        report["other_single_row_compiles"] = json!(admitted);
        let load = load_average();
        let started = Instant::now();
        let program = model.compile_multirow_unchecked(shape);
        report["multi_row_compile"] = json!({
            "ms": ms(started), "load_before": load, "load_after": load_average(),
            "ok": program.is_ok(),
            "error": program.as_ref().err().map(|e| format!("{e:#}")),
            "executables": program.as_ref().map(|p| p.executable_count()).unwrap_or(0),
        });
        let mut graph = Graph::new();
        let hidden = model.profile.n("hidden_size");
        let placeholder = graph.placeholder(shape.tensor_shape(hidden));
        layer_graph(&mut graph, placeholder, &model, 0, &shape).unwrap();
        let (mil, weights) = graph.source_payload();
        report["multi_row_program"] = json!({
            "layer0_operations": mil.lines().filter(|l| l.contains("[name = string(")).count(),
            "layer0_mil_bytes": mil.len(), "layer0_weight_bytes": weights.len(),
        });
        let program = match program {
            Ok(program) => program,
            Err(error) => {
                write(&report);
                println!("MULTIROW arm={} compile refused: {error:#}", shape.label());
                return;
            }
        };

        // Correctness before timing: single rows against fp32, then every
        // multi-row slot against the same row run alone.
        let mut failures = Vec::new();
        let mut gate = Vec::new();
        let mut rows: Vec<(String, Vec<u32>, Option<Vec<f32>>)> = Vec::new();
        for (id, ids, expected) in &references {
            let (vector, _) = single(&model, ids);
            let value = cosine(&vector, expected);
            if value < SINGLE_ROW_GATE {
                failures.push(format!("single {id} fp32 cosine {value}"));
            }
            gate.push(json!({"case": id, "tokens": ids.len(), "cosine_vs_fp32": value}));
            rows.push((format!("fixture:{id}"), ids.clone(), Some(expected.clone())));
        }
        report["single_row_fp32_gate"] = json!(gate);
        for chunk in chunks
            .iter()
            .filter(|c| c.ids.len() <= shape.width)
            .take(4 * shape.rows)
        {
            rows.push((format!("engram:{}", chunk.seq), chunk.ids.clone(), None));
        }
        let alone: Vec<Vec<f32>> = rows
            .iter()
            .map(|(_, ids, _)| single(&model, ids).0)
            .collect();
        let mut parity = Vec::new();
        let mut min_cosine = f64::INFINITY;
        for (group, members) in rows.chunks(shape.rows).enumerate() {
            let tokens: Vec<&[u32]> = members.iter().map(|(_, ids, _)| ids.as_slice()).collect();
            let outputs = program.run(&model, &tokens).unwrap();
            for (slot, ((name, ids, expected), output)) in members.iter().zip(outputs).enumerate() {
                let reference = &alone[group * shape.rows + slot];
                let value = cosine(&output, reference);
                min_cosine = min_cosine.min(value);
                if value < MULTI_ROW_GATE || output.len() != reference.len() {
                    failures.push(format!(
                        "{name} slot {slot}: cosine {value} len {}",
                        output.len()
                    ));
                }
                parity.push(json!({
                    "row": name, "group": group, "slot": slot, "tokens": ids.len(),
                    "cosine_vs_single": value, "max_abs_vs_single": max_abs(&output, reference),
                    "identical_to_single": &output == reference, "dims": output.len(),
                    "cosine_vs_fp32": expected.as_ref().map(|e| cosine(&output, e)),
                }));
            }
        }
        report["multi_row_parity"] = json!({"min_cosine_vs_single": min_cosine, "rows": parity});

        // Padding and neighbours must not reach a real row: the same row alone
        // (empty slots), surrounded by other rows, and in the last slot.
        let target = &rows[0].1;
        let others: Vec<&[u32]> = rows[1..shape.rows.min(rows.len())]
            .iter()
            .map(|r| r.1.as_slice())
            .collect();
        let alone_in_pass = program.run(&model, &[target]).unwrap().remove(0);
        let mut first_slot = vec![target.as_slice()];
        first_slot.extend(&others);
        let with_neighbours = program.run(&model, &first_slot).unwrap().remove(0);
        let mut last_slot = others.clone();
        last_slot.push(target);
        let in_last_slot = program.run(&model, &last_slot).unwrap().pop().unwrap();
        let isolation = json!({
            "row": rows[0].0,
            "empty_slots_vs_neighbours_max_abs": max_abs(&alone_in_pass, &with_neighbours),
            "empty_slots_vs_neighbours_identical": alone_in_pass == with_neighbours,
            "first_vs_last_slot_max_abs": max_abs(&with_neighbours, &in_last_slot),
            "first_vs_last_slot_cosine": cosine(&with_neighbours, &in_last_slot),
            "alone_in_pass_vs_single_cosine": cosine(&alone_in_pass, &alone[0]),
        });
        if cosine(&alone_in_pass, &with_neighbours) < MULTI_ROW_GATE
            || cosine(&with_neighbours, &in_last_slot) < MULTI_ROW_GATE
        {
            failures.push(format!("padding or neighbour leak: {isolation}"));
        }
        report["isolation"] = isolation;
        report["correctness_failures"] = json!(failures);
        write(&report);

        // Timing: the same N rows through N single-row runs and one multi-row
        // pass, alternating which goes first. Rows fill the rung exactly so
        // the single-row path runs at the same width.
        let floor = LADDER
            .iter()
            .rev()
            .find(|&&w| w < shape.width)
            .copied()
            .unwrap_or(0);
        let timed: Vec<&Chunk> = chunks
            .iter()
            .filter(|c| c.ids.len() > floor && c.ids.len() <= shape.width)
            .take(shape.rows)
            .collect();
        assert_eq!(timed.len(), shape.rows, "not enough rows for the rung");
        let tokens: Vec<&[u32]> = timed.iter().map(|c| c.ids.as_slice()).collect();
        for ids in &tokens {
            model.run(ids).unwrap();
        }
        program.run(&model, &tokens).unwrap();
        let mut samples = Vec::new();
        for repeat in 0..repeats {
            let order = if repeat % 2 == 0 {
                ["single", "multi"]
            } else {
                ["multi", "single"]
            };
            for path in order {
                let load_before = load_average();
                let started = Instant::now();
                if path == "single" {
                    for ids in &tokens {
                        model.run(ids).unwrap();
                    }
                } else {
                    program.run(&model, &tokens).unwrap();
                }
                let wall = ms(started);
                samples.push(json!({
                    "repeat": repeat, "path": path, "wall_ms": wall,
                    "per_row_ms": wall / shape.rows as f64,
                    "load_before": load_before, "load_after": load_average(),
                }));
            }
        }
        let median = |path: &str| {
            let mut values: Vec<f64> = samples
                .iter()
                .filter(|s| s["path"] == path)
                .map(|s| s["per_row_ms"].as_f64().unwrap())
                .collect();
            values.sort_by(f64::total_cmp);
            values[values.len() / 2]
        };
        let (single_ms, multi_ms) = (median("single"), median("multi"));
        report["timing"] = json!({
            "rows": timed.iter().map(|c| json!({"seq": c.seq, "tokens": c.ids.len(),
                "text_sha256": c.text_sha256, "ids_sha256": ids_digest(&c.ids)})).collect::<Vec<_>>(),
            "samples": samples,
            "median_single_per_row_ms": single_ms,
            "median_multi_per_row_ms": multi_ms,
            "speedup": single_ms / multi_ms,
        });
        write(&report);
        println!(
            "MULTIROW arm={} single_per_row_ms={single_ms:.3} multi_per_row_ms={multi_ms:.3} speedup={:.3} min_cosine={min_cosine:.7} failures={}",
            shape.label(),
            single_ms / multi_ms,
            report["correctness_failures"].as_array().unwrap().len()
        );

        if replay {
            let extra: Vec<MultiRowShape> = std::env::var("ANE_MULTIROW_REPLAY_EXTRA")
                .map(|specs| {
                    specs
                        .split(',')
                        .map(|spec| MultiRowShape::parse(spec).unwrap())
                        .collect()
                })
                .unwrap_or_default();
            let first: Vec<&Chunk> = chunks.iter().take(64).collect();
            // The export's own batch plan, for the whole-export replay.
            let batches: Vec<Vec<&Chunk>> = if std::env::var_os("ANE_MULTIROW_REPLAY_ALL").is_some()
            {
                let meta: Value =
                    serde_json::from_slice(&std::fs::read(format!("{input}.meta.json")).unwrap())
                        .unwrap();
                meta["batches"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|batch| {
                        let first = batch["first_chunk_seq"].as_u64().unwrap() as usize;
                        let count = batch["chunk_count"].as_u64().unwrap() as usize;
                        chunks[first..first + count].iter().collect()
                    })
                    .collect()
            } else {
                Vec::new()
            };
            let mut rungs: std::collections::BTreeSet<usize> =
                model.resident.keys().copied().collect();
            rungs.extend(
                batches
                    .iter()
                    .flatten()
                    .chain(&first)
                    .map(|c| rung(c.ids.len()).unwrap()),
            );
            // Stay well under the ~115 executables one process can hold.
            let held = 28 * (rungs.len() + 1 + extra.len());
            if held > 84 {
                report["replay"] = json!({"skipped": format!("shapes {rungs:?} and {} programs would hold {held} executables", 1 + extra.len())});
            } else {
                let mut compiles = Vec::new();
                for &width in &rungs {
                    let started = Instant::now();
                    model.admit(width, "multirow-experiment").unwrap();
                    compiles.push(json!({"single_row_width": width, "ms": ms(started)}));
                }
                let mut programs = vec![program];
                for &shape in &extra {
                    let load = load_average();
                    let started = Instant::now();
                    programs.push(model.compile_multirow_unchecked(shape).unwrap());
                    compiles.push(json!({
                        "multi_row": shape.label(), "ms": ms(started), "executables": 28,
                        "load_before": load, "load_after": load_average(),
                    }));
                }
                report["replay_setup"] = json!({
                    "programs": programs.iter().map(|p| p.shape().label()).collect::<Vec<_>>(),
                    "single_row_widths": rungs, "executables_held": held, "compiles": compiles,
                });
                report["replay"] = replay_batches(&model, &programs, &[first], repeats);
                write(&report);
                if !batches.is_empty() {
                    report["replay_all"] = replay_batches(&model, &programs, &batches, 3);
                }
            }
            write(&report);
        }
        drop(model);
        let failures = report["correctness_failures"].as_array().unwrap();
        assert!(failures.is_empty(), "{failures:?}");
    }

    /// Replay real batches in their order and lengths: today's one-row-at-a-
    /// time path, then multi-row passes (each row in the narrowest program
    /// slot that fits) with the rest on the single-row path. Passes are planned
    /// within each batch, as a serving call would.
    fn replay_batches(
        model: &Model,
        programs: &[MultiRowProgram],
        batches: &[Vec<&Chunk>],
        repeats: usize,
    ) -> Value {
        let shapes: Vec<MultiRowShape> = programs.iter().map(|p| p.shape()).collect();
        let plans: Vec<Vec<Pass>> = batches
            .iter()
            .map(|batch| {
                let lengths: Vec<usize> = batch.iter().map(|c| c.ids.len()).collect();
                plan_passes(&lengths, &shapes)
            })
            .collect();
        let rows: usize = batches.iter().map(Vec::len).sum();
        let run_single = || -> Vec<Vec<f32>> {
            batches
                .iter()
                .flatten()
                .map(|c| model.run(&c.ids).unwrap())
                .collect()
        };
        let run_multi = || -> Vec<Vec<f32>> {
            let mut all = Vec::with_capacity(rows);
            for (batch, passes) in batches.iter().zip(&plans) {
                let mut outputs = vec![Vec::new(); batch.len()];
                for pass in passes {
                    match pass {
                        Pass::Single(index) => {
                            outputs[*index] = model.run(&batch[*index].ids).unwrap()
                        }
                        Pass::MultiRow { program, rows } => {
                            let tokens: Vec<&[u32]> =
                                rows.iter().map(|&i| batch[i].ids.as_slice()).collect();
                            let vectors = programs[*program].run(model, &tokens).unwrap();
                            for (&index, vector) in rows.iter().zip(vectors) {
                                outputs[index] = vector;
                            }
                        }
                    }
                }
                all.extend(outputs);
            }
            all
        };
        let reference = run_single();
        let candidate = run_multi();
        let min_cosine = reference
            .iter()
            .zip(&candidate)
            .map(|(a, b)| cosine(a, b))
            .fold(f64::INFINITY, f64::min);
        let identical = reference
            .iter()
            .zip(&candidate)
            .filter(|(a, b)| a == b)
            .count();
        let mut samples = Vec::new();
        for repeat in 0..repeats.max(3) {
            let order = if repeat % 2 == 0 {
                ["single", "multi"]
            } else {
                ["multi", "single"]
            };
            for path in order {
                let load_before = load_average();
                let started = Instant::now();
                let outputs = if path == "single" {
                    run_single()
                } else {
                    run_multi()
                };
                let wall = ms(started);
                assert_eq!(outputs.len(), rows);
                samples.push(json!({
                    "repeat": repeat, "path": path, "wall_ms": wall,
                    "rows_per_second": rows as f64 * 1000.0 / wall,
                    "load_before": load_before, "load_after": load_average(),
                }));
            }
        }
        let mut token_counts: Vec<usize> = batches.iter().flatten().map(|c| c.ids.len()).collect();
        token_counts.sort();
        let all_passes = || plans.iter().flatten();
        let pass_counts: Vec<Value> = (0..shapes.len())
            .map(|program| {
                json!({"program": shapes[program].label(), "passes": all_passes().filter(|p| matches!(p, Pass::MultiRow { program: q, .. } if *q == program)).count()})
            })
            .collect();
        let texts: Vec<&str> = batches
            .iter()
            .flatten()
            .map(|c| c.text_sha256.as_str())
            .collect();
        let all_rows: Vec<&&Chunk> = batches.iter().flatten().collect();
        json!({
            "batches": batches.len(), "rows": rows,
            "export_seqs": [all_rows[0].seq, all_rows[rows - 1].seq],
            "text_sha256_list_sha256": sha(texts.join("\n").as_bytes()),
            "token_counts_sorted": if batches.len() == 1 { json!(token_counts) } else { Value::Null },
            "composed_tokens_total": token_counts.iter().sum::<usize>(),
            "multi_row_passes": pass_counts,
            "single_row_passes": all_passes().filter(|p| matches!(p, Pass::Single(_))).count(),
            "min_cosine_multi_vs_single": min_cosine,
            "identical_rows": identical,
            "samples": samples,
        })
    }
}
