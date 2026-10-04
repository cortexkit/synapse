//! ModernBERT ANE graph lowering, adapted from the direct-API probe.
use ane::{Graph, Shape, Tensor};
use serde::Deserialize;
const MASK_MIN: f32 = -10_000.0;
#[derive(Clone, Deserialize)]
pub(super) struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub global_attn_every_n_layers: usize,
    pub global_rope_theta: f32,
    pub local_attention: usize,
    pub local_rope_theta: f32,
    pub norm_eps: f32,
}

#[derive(Clone)]
pub(super) struct Linear {
    pub weight: Vec<f32>,
}

#[derive(Clone)]
pub(super) struct LayerWeights {
    pub qkv: Linear,
    pub attention_output: Linear,
    pub attention_norm: Option<Vec<f32>>,
    pub mlp_input: Linear,
    pub mlp_output: Linear,
    pub mlp_norm: Vec<f32>,
}

pub(super) fn shape(sequence_length: usize, channels: usize) -> Shape {
    Shape {
        batch: 1,
        channels,
        height: 1,
        width: sequence_length,
    }
}

fn scalar_shape() -> Shape {
    Shape::channels(1)
}

fn layer_norm_graph(graph: &mut Graph, input: Tensor, weight: &[f32], eps: f32) -> Tensor {
    let channels = input.shape.channels;
    let weight = graph.constant(weight, Shape::channels(channels));
    // The converted residual stream is centered before rotation. Centering
    // again in the rotated basis would change the model, so use RMS here.
    let centered = input;

    // Squaring ModernBERT's late residual values directly overflows fp16. Scale
    // each token by its largest centered channel, then algebraically scale the
    // epsilon as well; this computes the same LayerNorm without large squares.
    let magnitude = graph.absolute(centered);
    let magnitude = graph.reduce_max(magnitude, 1);
    let epsilon_root = graph.constant_with_scalar(eps.sqrt(), scalar_shape());
    let scale = graph.maximum(magnitude, epsilon_root);
    let scaled = graph.division(centered, scale);
    let squared = graph.multiplication(scaled, scaled);
    let variance = graph.reduce_mean(squared, 1);
    let scaled_epsilon = graph.division(epsilon_root, scale);
    let scaled_epsilon = graph.multiplication(scaled_epsilon, scaled_epsilon);
    let variance = graph.addition(variance, scaled_epsilon);
    // Apple's private ANE compiler rejects reciprocal_square_root for this
    // reduction pattern; power with exponent -0.5 computes the same value.
    let negative_half = graph.constant_with_scalar(-0.5, scalar_shape());
    let inverse_stddev = graph.power(variance, negative_half);
    let normalized = graph.multiplication(scaled, inverse_stddev);
    graph.multiplication(normalized, weight)
}

fn gelu_graph(graph: &mut Graph, input: Tensor) -> Tensor {
    // The binding has no erf operation. This is the standard tanh GELU lowering
    // used by its encoder example and differs from exact GELU by less than 5e-4.
    let half = graph.constant_with_scalar(0.5, scalar_shape());
    let one = graph.constant_with_scalar(1.0, scalar_shape());
    let cubic_scale = graph.constant_with_scalar(0.044_715, scalar_shape());
    let sqrt_two_over_pi = graph.constant_with_scalar(0.797_884_6, scalar_shape());
    let squared = graph.multiplication(input, input);
    let cubed = graph.multiplication(squared, input);
    let cubic = graph.multiplication(cubic_scale, cubed);
    let inner = graph.addition(input, cubic);
    let argument = graph.multiplication(sqrt_two_over_pi, inner);
    let tanh = graph.tanh(argument);
    let shifted = graph.addition(one, tanh);
    let half_input = graph.multiplication(half, input);
    graph.multiplication(half_input, shifted)
}

pub(super) fn rope_tables(
    sequence_length: usize,
    head_dim: usize,
    theta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut cosine = vec![0.0; head_dim * sequence_length];
    let mut sine = vec![0.0; head_dim * sequence_length];
    for dimension in 0..head_dim / 2 {
        let frequency = theta.powf(-((2 * dimension) as f32) / head_dim as f32);
        for position in 0..sequence_length {
            let (sin, cos) = (position as f32 * frequency).sin_cos();
            for target in [dimension, dimension + head_dim / 2] {
                cosine[target * sequence_length + position] = cos;
                sine[target * sequence_length + position] = sin;
            }
        }
    }
    (cosine, sine)
}

pub(super) fn apply_rope_graph(
    graph: &mut Graph,
    input: Tensor,
    cosine: &[f32],
    sine: &[f32],
    heads: usize,
    head_dim: usize,
    sequence_length: usize,
) -> Tensor {
    let half = head_dim / 2;
    let first = graph.slice(input, [0, 0, 0, 0], [1, heads, half, sequence_length]);
    let second = graph.slice(input, [0, 0, half, 0], [1, heads, half, sequence_length]);
    let negative = graph.constant_with_scalar(-1.0, scalar_shape());
    let negative_second = graph.multiplication(negative, second);
    let rotated = graph.concat(&[negative_second, first], 2);
    let table_shape = Shape {
        batch: 1,
        channels: 1,
        height: head_dim,
        width: sequence_length,
    };
    let cosine = graph.constant(cosine, table_shape);
    let sine = graph.constant(sine, table_shape);
    let direct = graph.multiplication(input, cosine);
    let turned = graph.multiplication(rotated, sine);
    graph.addition(direct, turned)
}

fn local_distance_mask(
    query_start: usize,
    query_len: usize,
    key_start: usize,
    key_len: usize,
    radius: usize,
) -> Vec<f32> {
    let mut mask = vec![0.0; query_len * key_len];
    for query in 0..query_len {
        for key in 0..key_len {
            if (query_start + query).abs_diff(key_start + key) > radius {
                mask[query * key_len + key] = MASK_MIN;
            }
        }
    }
    mask
}

#[allow(clippy::too_many_arguments)]
fn attention_graph(
    graph: &mut Graph,
    hidden: Tensor,
    residual: Tensor,
    key_mask: Tensor,
    weights: &LayerWeights,
    config: &Config,
    layer_index: usize,
    sequence_length: usize,
) -> Tensor {
    let hidden_size = config.hidden_size;
    let heads = config.num_attention_heads;
    let head_dim = hidden_size / heads;
    let normalized = if let Some(weight) = &weights.attention_norm {
        layer_norm_graph(graph, hidden, weight, config.norm_eps)
    } else {
        hidden
    };
    let qkv = graph.inner_product(
        normalized,
        &weights.qkv.weight,
        hidden_size,
        hidden_size * 3,
    );
    let head_shape = Shape {
        batch: 1,
        channels: heads,
        height: head_dim,
        width: sequence_length,
    };
    let mut parts = Vec::with_capacity(3);
    for part in 0..3 {
        let sliced = graph.slice(
            qkv,
            [0, part * hidden_size, 0, 0],
            [1, hidden_size, 1, sequence_length],
        );
        parts.push(graph.reshape(sliced, head_shape));
    }
    let theta = if layer_index.is_multiple_of(config.global_attn_every_n_layers) {
        config.global_rope_theta
    } else {
        config.local_rope_theta
    };
    let (cosine, sine) = rope_tables(sequence_length, head_dim, theta);
    let query = apply_rope_graph(
        graph,
        parts[0],
        &cosine,
        &sine,
        heads,
        head_dim,
        sequence_length,
    );
    let key = apply_rope_graph(
        graph,
        parts[1],
        &cosine,
        &sine,
        heads,
        head_dim,
        sequence_length,
    );
    let value = parts[2];
    let permutation = [0, 1, 3, 2];
    let query = graph.transpose(query, permutation);
    let key = graph.transpose(key, permutation);
    let value = graph.transpose(value, permutation);
    let scale = graph.constant_with_scalar(1.0 / (head_dim as f32).sqrt(), scalar_shape());
    let query_tile = 128usize.min(sequence_length);
    let local_radius = (!layer_index.is_multiple_of(config.global_attn_every_n_layers))
        .then_some(config.local_attention / 2);
    let mut contexts = Vec::new();
    for query_start in (0..sequence_length).step_by(query_tile) {
        let query_end = (query_start + query_tile).min(sequence_length);
        let query_len = query_end - query_start;
        let (key_start, key_end) = if let Some(radius) = local_radius {
            (
                query_start.saturating_sub(radius),
                (query_end + radius).min(sequence_length),
            )
        } else {
            (0, sequence_length)
        };
        let key_len = key_end - key_start;
        let query_slice = graph.slice(
            query,
            [0, 0, query_start, 0],
            [1, heads, query_len, head_dim],
        );
        let key_slice = graph.slice(key, [0, 0, key_start, 0], [1, heads, key_len, head_dim]);
        let value_slice = graph.slice(value, [0, 0, key_start, 0], [1, heads, key_len, head_dim]);
        let scores = graph.matrix_multiplication(query_slice, key_slice, false, true);
        let scores = graph.multiplication(scores, scale);
        let padding = graph.slice(key_mask, [0, 0, 0, key_start], [1, 1, 1, key_len]);
        let mut masked = graph.addition(scores, padding);
        if let Some(radius) = local_radius {
            let distance = local_distance_mask(query_start, query_len, key_start, key_len, radius);
            let distance = graph.constant(
                &distance,
                Shape {
                    batch: 1,
                    channels: 1,
                    height: query_len,
                    width: key_len,
                },
            );
            masked = graph.addition(masked, distance);
        }
        // Softmax over keys runs on the channel axis rather than the last
        // (width) axis: [1, heads, queries, keys] is transposed to
        // [1, keys, heads, queries], normalized on axis 1, and transposed back
        // so the value matrix multiplication sees its original layout. This
        // preserves the attention equation while using the faster channel reduction.
        let channel_logits = graph.transpose(masked, [0, 3, 1, 2]);
        let channel_probabilities = graph.soft_max(channel_logits, 1);
        let probabilities = graph.transpose(channel_probabilities, [0, 2, 3, 1]);
        contexts.push(graph.matrix_multiplication(probabilities, value_slice, false, false));
    }
    let context = if contexts.len() == 1 {
        contexts[0]
    } else {
        graph.concat(&contexts, 2)
    };
    let context = graph.transpose(context, permutation);
    let context = graph.reshape(context, shape(sequence_length, hidden_size));
    let projected = graph.inner_product(
        context,
        &weights.attention_output.weight,
        hidden_size,
        hidden_size,
    );
    graph.addition(residual, projected)
}

pub(super) fn layer_graph(
    graph: &mut Graph,
    hidden: Tensor,
    residual: Tensor,
    key_mask: Tensor,
    weights: &LayerWeights,
    config: &Config,
    layer_index: usize,
    sequence_length: usize,
) -> Tensor {
    let attended = attention_graph(
        graph,
        hidden,
        residual,
        key_mask,
        weights,
        config,
        layer_index,
        sequence_length,
    );
    let normalized = layer_norm_graph(graph, attended, &weights.mlp_norm, config.norm_eps);
    let projected = graph.inner_product(
        normalized,
        &weights.mlp_input.weight,
        config.hidden_size,
        config.intermediate_size * 2,
    );
    let activation = graph.slice(
        projected,
        [0, 0, 0, 0],
        [1, config.intermediate_size, 1, sequence_length],
    );
    let gate = graph.slice(
        projected,
        [0, config.intermediate_size, 0, 0],
        [1, config.intermediate_size, 1, sequence_length],
    );
    let activation = gelu_graph(graph, activation);
    let gated = graph.multiplication(activation, gate);
    let output = graph.inner_product(
        gated,
        &weights.mlp_output.weight,
        config.intermediate_size,
        config.hidden_size,
    );
    graph.addition(attended, output)
}
