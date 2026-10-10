//! Opt-in variable-length width packing for Qwen3. Every pass supplies a
//! block-causal mask and rotary tables as runtime IOSurface inputs; the fixed
//! width is compiled, but segment boundaries are not. No serving path calls it.
use super::*;
use crate::backend::Profile;

/// Fastest tested width for Qwen3-Embedding-0.6B on a 6,341-row recorded
/// code-search export (docs/evidence/ane-tight-packing). Every packed output
/// was bit-identical to the same row embedded alone; the audit's pass bar was
/// cosine 0.9999 against that standalone output.
pub const RECOMMENDED_TIGHT_WIDTH: usize = 256;

/// A first-fit-decreasing plan. Indices refer to the caller's input order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TightPlan {
    pub passes: Vec<Vec<usize>>,
    pub singles: Vec<usize>,
}

/// Greedy first-fit-decreasing within one engine call, with stable index ties.
/// Rows too long for this width remain on the single-row path.
pub fn plan_tight(lengths: &[usize], width: usize) -> Result<TightPlan> {
    validate_width(width)?;
    ensure!(lengths.iter().all(|&n| n > 0), "empty_row");
    let mut order: Vec<usize> = (0..lengths.len()).collect();
    order.sort_by_key(|&i| (std::cmp::Reverse(lengths[i]), i));
    let mut plan = TightPlan {
        passes: Vec::new(),
        singles: Vec::new(),
    };
    let mut used = Vec::<usize>::new();
    for index in order {
        if lengths[index] > width {
            plan.singles.push(index);
            continue;
        }
        if let Some(pass) = used.iter().position(|&n| n + lengths[index] <= width) {
            used[pass] += lengths[index];
            plan.passes[pass].push(index);
        } else {
            used.push(lengths[index]);
            plan.passes.push(vec![index]);
        }
    }
    Ok(plan)
}

fn validate_width(width: usize) -> Result<()> {
    ensure!(matches!(width, 256 | 384 | 512), "unsupported_packed_width");
    Ok(())
}

/// Build a dense runtime bias and channel-major rotary tables. Padding queries
/// attend only themselves so their softmax remains finite; real queries can
/// attend only earlier tokens in their own segment, never padding or neighbours.
pub fn runtime_operands(
    lengths: &[usize],
    width: usize,
    dim: usize,
    theta: f32,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    validate_width(width)?;
    ensure!(
        !lengths.is_empty() && lengths.iter().all(|&n| n > 0),
        "empty_row"
    );
    let total = lengths
        .iter()
        .try_fold(0usize, |a, &b| a.checked_add(b))
        .context("packed_length_overflow")?;
    ensure!(total <= width, "packed_rows_too_wide");
    ensure!(dim > 0 && dim.is_multiple_of(2), "invalid_head_dim");
    let mut mask = vec![MASKED; width * width];
    let mut positions = vec![0; width];
    let mut start = 0;
    for &length in lengths {
        for q in start..start + length {
            positions[q] = q - start;
            for k in start..=q {
                mask[q * width + k] = 0.0;
            }
        }
        start += length;
    }
    for q in total..width {
        mask[q * width + q] = 0.0;
    }
    let (base_cos, base_sin) = crate::modernbert::rope_tables(width, dim, theta);
    let mut cos = vec![0.0; dim * width];
    let mut sin = cos.clone();
    for d in 0..dim {
        for p in 0..width {
            cos[d * width + p] = base_cos[d * width + positions[p]];
            sin[d * width + p] = base_sin[d * width + positions[p]];
        }
    }
    Ok((mask, cos, sin))
}

/// Match the graph-constant encoder before the runtime staging conversion.
/// Decoded fp16 values are exactly representable in fp32, so staging them
/// cannot undo the constant encoder's truncation and underflow policy.
pub(super) fn constant_compatible_coefficients(values: &mut [f32]) {
    let bytes = ane::f32_to_fp16_bytes(values);
    for (value, bytes) in values.iter_mut().zip(bytes.as_chunks::<2>().0.iter()) {
        *value = half::f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32();
    }
}

fn surface(width: usize, channels: usize) -> Shape {
    Shape {
        batch: 1,
        channels,
        height: 1,
        width,
    }
}
fn mask_shape(width: usize) -> Shape {
    Shape {
        batch: 1,
        channels: 1,
        height: width,
        width,
    }
}
fn rope_shape(width: usize, dim: usize) -> Shape {
    Shape {
        batch: 1,
        channels: 1,
        height: dim,
        width,
    }
}
fn rope(graph: &mut Graph, input: Tensor, cos: Tensor, sin: Tensor, dim: usize) -> Tensor {
    let s = input.shape;
    let first = graph.slice(input, [0, 0, 0, 0], [1, s.channels, dim / 2, s.width]);
    let second = graph.slice(input, [0, 0, dim / 2, 0], [1, s.channels, dim / 2, s.width]);
    let negative = graph.constant_with_scalar(-1.0, Shape::channels(1));
    let negative_second = graph.multiplication(negative, second);
    let rotated = graph.concat(&[negative_second, first], 2);
    let direct = graph.multiplication(input, cos);
    let turned = graph.multiplication(rotated, sin);
    graph.addition(direct, turned)
}

/// Four runtime inputs, declared in order: activation, mask, cos, sin.
/// Query tiles multiply even the masked query/key pairs. Packing saves padding
/// columns but still computes and discards cross-row attention scores.
pub fn layer_graph(
    graph: &mut Graph,
    inputs: [Tensor; 4],
    model: &Model,
    layer: usize,
    width: usize,
) -> Result<Tensor> {
    let [input, mask, cos, sin] = inputs;
    let p = &model.profile;
    let hidden = p.n("hidden_size");
    let heads = p.n("num_attention_heads");
    let kv_heads = p.n("num_key_value_heads");
    let dim = p.n("head_dim");
    let intermediate = p.n("intermediate_size");
    let eps = p.f("norm_eps");
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
                batch: 1,
                channels: count,
                height: dim,
                width,
            },
        ));
    }
    let query = rms(graph, qkv[0], tensor("self_attn.q_norm")?, 2, eps);
    let key = rms(graph, qkv[1], tensor("self_attn.k_norm")?, 2, eps);
    let query = rope(graph, query, cos, sin, dim);
    let key = rope(graph, key, cos, sin, dim);
    let repeat = |graph: &mut Graph, input: Tensor| {
        let slices: Vec<Tensor> = (0..heads)
            .map(|h| graph.slice(input, [0, h / (heads / kv_heads), 0, 0], [1, 1, dim, width]))
            .collect();
        graph.concat(&slices, 1)
    };
    let key = repeat(graph, key);
    let value = repeat(graph, qkv[2]);
    let query = graph.transpose(query, [0, 1, 3, 2]);
    let key = graph.transpose(key, [0, 1, 3, 2]);
    let value = graph.transpose(value, [0, 1, 3, 2]);
    let scale = graph.constant_with_scalar((dim as f32).sqrt().recip(), Shape::channels(1));
    let mut tiles = Vec::new();
    for start in (0..width).step_by(QUERY_TILE) {
        let count = QUERY_TILE.min(width - start);
        let key_count = start + count;
        let q = graph.slice(query, [0, 0, start, 0], [1, heads, count, dim]);
        let k = graph.slice(key, [0, 0, 0, 0], [1, heads, key_count, dim]);
        let v = graph.slice(value, [0, 0, 0, 0], [1, heads, key_count, dim]);
        let scores = graph.matrix_multiplication(q, k, false, true);
        let scores = graph.multiplication(scores, scale);
        let bias = graph.slice(mask, [0, 0, start, 0], [1, 1, count, key_count]);
        let scores = graph.addition(scores, bias);
        let logits = graph.transpose(scores, [0, 3, 1, 2]);
        let probabilities = graph.soft_max(logits, 1);
        let probabilities = graph.transpose(probabilities, [0, 2, 3, 1]);
        tiles.push(graph.matrix_multiplication(probabilities, v, false, false));
    }
    let context = graph.concat(&tiles, 2);
    let context = graph.transpose(context, [0, 1, 3, 2]);
    let context = graph.reshape(context, surface(width, heads * dim));
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

/// Measurement variants change whether mask and rotary values are constants
/// or runtime inputs, keeping row lengths and offsets identical. The public
/// API always uses all three runtime inputs.
#[derive(Clone, Debug)]
pub(super) enum OperandMode {
    Runtime,
    RuntimeConstantCompatible,
    MaskOnly(Vec<usize>),
    Constants(Vec<usize>),
}

pub struct TightProgram {
    width: usize,
    mode: OperandMode,
    executables: Vec<Executable>,
    a: TensorData,
    b: TensorData,
    mask: TensorData,
    cos: TensorData,
    sin: TensorData,
}
impl Drop for TightProgram {
    fn drop(&mut self) {
        ane::autoreleasepool(|_| self.executables.clear());
    }
}
fn build_graph(
    model: &Model,
    layer: usize,
    width: usize,
    mode: &OperandMode,
) -> Result<(Graph, Tensor)> {
    let dim = model.profile.n("head_dim");
    let constants =
        match mode {
            OperandMode::Runtime | OperandMode::RuntimeConstantCompatible => None,
            OperandMode::MaskOnly(lengths) | OperandMode::Constants(lengths) => Some(
                runtime_operands(lengths, width, dim, model.profile.f("rope_theta"))?,
            ),
        };
    let mut graph = Graph::new();
    let input = graph.placeholder(surface(width, model.profile.n("hidden_size")));
    let mask = if matches!(mode, OperandMode::Constants(_)) {
        graph.constant(&constants.as_ref().unwrap().0, mask_shape(width))
    } else {
        graph.placeholder(mask_shape(width))
    };
    let (cos, sin) = if let Some((_, cos, sin)) = &constants {
        (
            graph.constant(cos, rope_shape(width, dim)),
            graph.constant(sin, rope_shape(width, dim)),
        )
    } else {
        (
            graph.placeholder(rope_shape(width, dim)),
            graph.placeholder(rope_shape(width, dim)),
        )
    };
    let output = layer_graph(&mut graph, [input, mask, cos, sin], model, layer, width)?;
    Ok((graph, output))
}

fn public_mode(profile: &Profile, width: usize) -> Result<OperandMode> {
    let qualified = Profile::select("qwen3-embedding-0.6b.ane-direct-worker", "embed")?;
    ensure!(width == RECOMMENDED_TIGHT_WIDTH && profile.id == qualified.id && profile.model == qualified.model && profile.numeric == qualified.numeric,
        "layout_unqualified: public tight packing enables only 256 columns with the pinned Qwen3 embedding profile");
    Ok(OperandMode::RuntimeConstantCompatible)
}

impl Model {
    /// Compile a tight-packing program of `width` columns. Only width 256 for
    /// the pinned Qwen3-Embedding-0.6B profile is enabled (`public_mode`
    /// refuses anything else), because it is the only width shown to match
    /// standalone output. Runtime rotary coefficients go through the same fp16
    /// encoder as graph constants, so packed rows equal standalone rows bit
    /// for bit. The 28 decoder layers add 28 executables to the caller's
    /// budget. Calls must be serialized: the program reuses mutable input
    /// buffers.
    pub fn compile_tight_packing(&self, width: usize) -> Result<TightProgram> {
        supported(self)?;
        validate_width(width)?;
        self.compile_tight_mode(width, public_mode(&self.profile, width)?)
    }
    pub(super) fn compile_tight_mode(
        &self,
        width: usize,
        mode: OperandMode,
    ) -> Result<TightProgram> {
        supported(self)?;
        validate_width(width)?;
        let hidden = self.profile.n("hidden_size");
        let dim = self.profile.n("head_dim");
        ane::autoreleasepool(|_| {
            let mut executables = Vec::new();
            for layer in 0..self.profile.n("num_hidden_layers") {
                let (graph, _) = build_graph(self, layer, width, &mode)?;
                executables.push(
                    graph
                        .compile(NSQualityOfService::UserInteractive)
                        .with_context(|| format!("tight layer {layer}"))?,
                );
            }
            Ok(TightProgram {
                width,
                mode,
                executables,
                a: TensorData::new(surface(width, hidden)),
                b: TensorData::new(surface(width, hidden)),
                mask: TensorData::new(mask_shape(width)),
                cos: TensorData::new(rope_shape(width, dim)),
                sin: TensorData::new(rope_shape(width, dim)),
            })
        })
    }
}

impl TightProgram {
    pub fn width(&self) -> usize {
        self.width
    }
    pub fn executable_count(&self) -> usize {
        self.executables.len()
    }

    /// Return vectors in the order rows are supplied, not packer sort order.
    /// Input IOSurface objects remain allocated across calls; their contents
    /// change. The binding caches requests by those object identities.
    pub fn run(&self, model: &Model, rows: &[&[u32]]) -> Result<Vec<Vec<f32>>> {
        ane::autoreleasepool(|_| self.run_unpooled(model, rows))
    }
    fn run_unpooled(&self, model: &Model, rows: &[&[u32]]) -> Result<Vec<Vec<f32>>> {
        supported(model)?;
        let lengths: Vec<usize> = rows.iter().map(|r| r.len()).collect();
        match &self.mode {
            OperandMode::Runtime | OperandMode::RuntimeConstantCompatible => {}
            OperandMode::MaskOnly(expected) | OperandMode::Constants(expected) => {
                ensure!(lengths == *expected, "constant_segment_geometry_changed")
            }
        }
        let p = &model.profile;
        let hidden = p.n("hidden_size");
        let embeddings = model.tensor(&format!("{}embed_tokens.weight", p.prefix()))?;
        let (input, ends) = pack_tight(
            rows,
            self.width,
            hidden,
            p.n("pad_token_id") as u32,
            embeddings,
        )?;
        self.a.copy_from_f32(&input);
        if !matches!(self.mode, OperandMode::Constants(_)) {
            let (mask, mut cos, mut sin) =
                runtime_operands(&lengths, self.width, p.n("head_dim"), p.f("rope_theta"))?;
            self.mask.copy_from_f32(&mask);
            if matches!(self.mode, OperandMode::RuntimeConstantCompatible) {
                constant_compatible_coefficients(&mut cos);
                constant_compatible_coefficients(&mut sin);
            }
            if matches!(
                self.mode,
                OperandMode::Runtime | OperandMode::RuntimeConstantCompatible
            ) {
                self.cos.copy_from_f32(&cos);
                self.sin.copy_from_f32(&sin);
            }
        }
        for (layer, executable) in self.executables.iter().enumerate() {
            let (src, dst) = if layer % 2 == 0 {
                (&self.a, &self.b)
            } else {
                (&self.b, &self.a)
            };
            let inputs: &[&TensorData] = match self.mode {
                OperandMode::Runtime | OperandMode::RuntimeConstantCompatible => {
                    &[src, &self.mask, &self.cos, &self.sin]
                }
                OperandMode::MaskOnly(_) => &[src, &self.mask],
                OperandMode::Constants(_) => &[src],
            };
            executable.run_cached(inputs, &[dst])?;
        }
        let raw = if self.executables.len().is_multiple_of(2) {
            self.a.read_f32()
        } else {
            self.b.read_f32()
        };
        let norm = model.tensor(&format!("{}norm.weight", p.prefix()))?;
        Ok(ends
            .into_iter()
            .map(|end| {
                embedding_tail(
                    (0..hidden).map(|c| raw[c * self.width + end]).collect(),
                    norm,
                    p.f("norm_eps"),
                )
            })
            .collect())
    }
}

/// Gather embeddings back to back and return the last real token of each row.
/// Padding fills only the unused suffix of the fixed-width activation tensor.
pub fn pack_tight(
    rows: &[&[u32]],
    width: usize,
    hidden: usize,
    pad: u32,
    embeddings: &[f32],
) -> Result<(Vec<f32>, Vec<usize>)> {
    validate_width(width)?;
    ensure!(
        hidden > 0 && !rows.is_empty() && rows.iter().all(|r| !r.is_empty()),
        "empty_row"
    );
    let total = rows
        .iter()
        .try_fold(0usize, |n, r| n.checked_add(r.len()))
        .context("packed_length_overflow")?;
    ensure!(total <= width, "packed_rows_too_wide");
    let mut tokens: Vec<u32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    tokens.resize(width, pad);
    let mut input = vec![0.0; width * hidden];
    for (position, id) in tokens.into_iter().enumerate() {
        let start = (id as usize).checked_mul(hidden).context("invalid_token")?;
        let end = start.checked_add(hidden).context("invalid_token")?;
        let token = embeddings.get(start..end).context("invalid_token")?;
        for channel in 0..hidden {
            input[channel * width + position] = token[channel];
        }
    }
    let mut offset = 0;
    let ends = rows
        .iter()
        .map(|r| {
            offset += r.len();
            offset - 1
        })
        .collect();
    Ok((input, ends))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn runtime_rope_coefficients_match_actual_graph_constant_bits() {
        for width in [256, 384, 512] {
            let (_, cos, sin) = runtime_operands(&[3, 17, 129], width, 128, 1_000_000.0).unwrap();
            for values in [cos, sin] {
                let mut graph = Graph::new();
                let input = graph.placeholder(rope_shape(width, 128));
                let baked = graph.constant(&values, rope_shape(width, 128));
                graph.multiplication(input, baked);
                let (mil, blob) = graph.source_payload();
                let digits = mil.split("offset = uint64(").nth(1).unwrap();
                let chunk: usize = digits.split(')').next().unwrap().parse().unwrap();
                let word =
                    |at: usize| u32::from_le_bytes(blob[at..at + 4].try_into().unwrap()) as usize;
                let (size, data) = (word(chunk + 8), word(chunk + 16));
                assert_eq!(size, values.len() * 2);
                let expected: Vec<u16> = blob[data..data + size]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| u16::from_le_bytes([b[0], b[1]]))
                    .collect();
                let mut nearest = vec![0; values.len()];
                ane::neon_convert::f32_to_f16_bulk(&values, &mut nearest);
                assert!(
                    nearest.iter().zip(&expected).any(|(a, b)| a != b),
                    "test does not exercise rounding differences"
                );
                let mut compatible = values;
                constant_compatible_coefficients(&mut compatible);
                let mut runtime = vec![0; compatible.len()];
                ane::neon_convert::f32_to_f16_bulk(&compatible, &mut runtime);
                for (index, (actual, expected)) in runtime.iter().zip(expected).enumerate() {
                    assert_eq!(*actual, expected, "coefficient {index}, width {width}");
                }
            }
        }
    }

    #[test]
    fn public_mode_accepts_only_qualified_profile_and_compatible_encoding() {
        let profile = Profile::select("qwen3-embedding-0.6b.ane-direct-worker", "embed").unwrap();
        assert!(matches!(
            public_mode(&profile, 256),
            Ok(OperandMode::RuntimeConstantCompatible)
        ));
        for width in [128, 384, 512] {
            assert!(public_mode(&profile, width).is_err());
        }
        let mut changed = profile.clone();
        changed.model["architecture"]["params"]["num_hidden_layers"] = serde_json::json!(0);
        assert!(public_mode(&changed, 256).is_err());
        changed = profile;
        changed.numeric["converted_package_digest"] = serde_json::json!("sha256:unqualified");
        assert!(public_mode(&changed, 256).is_err());
    }

    #[test]
    fn unqualified_runtime_rope_is_refused_before_compilation() {
        let mut model = synthetic::qwen_model(&[], 4);
        model.profile.model["architecture"]["params"]["num_hidden_layers"] = serde_json::json!(0);
        for width in [256, 384, 512] {
            let error = model
                .compile_tight_packing(width)
                .err()
                .expect("unqualified layout accepted");
            assert!(error.to_string().contains("layout_unqualified"));
        }
    }

    #[test]
    fn first_fit_decreasing_covers_rows_once_and_preserves_ties() {
        let plan = plan_tight(&[100, 160, 100, 300, 56], 256).unwrap();
        assert_eq!(plan.passes, vec![vec![1, 4], vec![0, 2]]);
        assert_eq!(plan.singles, vec![3]);
        assert!(plan_tight(&[0], 256).is_err());
        assert!(plan_tight(&[10], 128).is_err());
        assert!(plan_tight(&[], 256).unwrap().passes.is_empty());
    }
    #[test]
    fn runtime_mask_is_causal_isolated_and_padding_queries_are_finite() {
        let (mask, _, _) = runtime_operands(&[3, 2], 256, 4, 10000.0).unwrap();
        // The expected boundaries below are written out by hand rather than
        // derived with `runtime_operands`, so the test can't share its bug.
        for q in 0..256 {
            for k in 0..256 {
                let allowed = match q {
                    0..=2 => k <= q,
                    3..=4 => (3..=q).contains(&k),
                    _ => k == q,
                };
                assert_eq!(
                    mask[q * 256 + k],
                    if allowed { 0.0 } else { -10000.0 },
                    "q={q} k={k}"
                );
            }
        }
        assert!(runtime_operands(&[257], 256, 4, 10000.0).is_err());
        assert!(runtime_operands(&[0], 256, 4, 10000.0).is_err());
    }
    #[test]
    fn runtime_rope_restarts_at_each_variable_length_segment() {
        let (_, cos, sin) = runtime_operands(&[3, 2], 256, 4, 10000.0).unwrap();
        for d in 0..4 {
            assert_eq!(cos[d * 256], 1.0);
            assert_eq!(sin[d * 256], 0.0);
            assert_eq!(cos[d * 256 + 3], 1.0);
            assert_eq!(sin[d * 256 + 3], 0.0);
            assert_eq!(cos[d * 256 + 1], cos[d * 256 + 4]);
            assert_eq!(sin[d * 256 + 1], sin[d * 256 + 4]);
        }
        assert!((cos[2] - 2.0f32.cos()).abs() < 1e-6);
        assert!((sin[2] - 2.0f32.sin()).abs() < 1e-6);
    }
    #[test]
    fn embedding_gather_and_tail_use_real_segment_ends() {
        let (input, ends) = pack_tight(
            &[&[1, 2], &[3]],
            256,
            2,
            0,
            &[0., 1., 10., 11., 20., 21., 30., 31.],
        )
        .unwrap();
        assert_eq!(ends, vec![1, 2]);
        assert_eq!(&input[..4], &[10., 20., 30., 0.]);
        assert_eq!(&input[256..260], &[11., 21., 31., 1.]);
        assert!(pack_tight(&[&[4]], 256, 2, 0, &[0.; 8]).is_err());
        assert!(pack_tight(&[&[1; 257]], 256, 2, 0, &[0.; 8]).is_err());
        assert!(pack_tight(&[&[]], 256, 2, 0, &[0.; 8]).is_err());
    }
    #[test]
    fn graph_has_four_runtime_inputs_and_no_constant_segment_geometry() {
        let model = synthetic::qwen_model(&[0], 8);
        for width in [256, 384, 512] {
            let (graph, output) = build_graph(&model, 0, width, &OperandMode::Runtime).unwrap();
            assert_eq!(output.shape, surface(width, 1024));
            let (mil, _) = graph.source_payload();
            let signature = mil.lines().find(|line| line.contains("func main")).unwrap();
            assert_eq!(signature.matches("tensor<fp16").count(), 4, "{signature}");
        }
    }
}
