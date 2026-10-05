//! Qwen3 grouped-query causal attention and SwiGLU, wholly inside ANE graphs.
use crate::{
    backend::Model,
    modernbert::{apply_rope_graph, rope_tables, shape},
};
use ane::{Graph, Shape, Tensor};
use anyhow::Result;

fn rms(graph: &mut Graph, input: Tensor, weight: &[f32], axis: i64, eps: f32) -> Tensor {
    let scalar = Shape::channels(1);
    let magnitude = graph.absolute(input);
    let scale = graph.reduce_max(magnitude, axis);
    let epsilon_root = graph.constant_with_scalar(eps.sqrt(), scalar);
    let scale = graph.maximum(scale, epsilon_root);
    let scaled = graph.division(input, scale);
    let squares = graph.multiplication(scaled, scaled);
    let variance = graph.reduce_mean(squares, axis);
    let epsilon = graph.division(epsilon_root, scale);
    let epsilon = graph.multiplication(epsilon, epsilon);
    let variance = graph.addition(variance, epsilon);
    let power = graph.constant_with_scalar(-0.5, scalar);
    let inverse = graph.power(variance, power);
    let normalized = graph.multiplication(scaled, inverse);
    let weight_shape = if axis == 1 {
        Shape::channels(weight.len())
    } else {
        Shape {
            batch: 1,
            channels: 1,
            height: weight.len(),
            width: 1,
        }
    };
    let weight = graph.constant(weight, weight_shape);
    graph.multiplication(normalized, weight)
}

pub fn layer_graph(
    graph: &mut Graph,
    input: Tensor,
    mask: Tensor,
    model: &Model,
    layer: usize,
    seq: usize,
) -> Result<Tensor> {
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
                width: seq,
            },
        ));
    }
    let query = rms(graph, qkv[0], tensor("self_attn.q_norm")?, 2, eps);
    let key = rms(graph, qkv[1], tensor("self_attn.k_norm")?, 2, eps);
    let (cos, sin) = rope_tables(seq, dim, p.f("rope_theta"));
    let query = apply_rope_graph(graph, query, &cos, &sin, heads, dim, seq);
    let key = apply_rope_graph(graph, key, &cos, &sin, kv_heads, dim, seq);
    let repeat = |graph: &mut Graph, input: Tensor| {
        let mut slices = Vec::new();
        for h in 0..heads {
            slices.push(graph.slice(input, [0, h / (heads / kv_heads), 0, 0], [1, 1, dim, seq]));
        }
        graph.concat(&slices, 1)
    };
    let key = repeat(graph, key);
    let value = repeat(graph, qkv[2]);
    let query = graph.transpose(query, [0, 1, 3, 2]);
    let key = graph.transpose(key, [0, 1, 3, 2]);
    let value = graph.transpose(value, [0, 1, 3, 2]);
    let scale = graph.constant_with_scalar((dim as f32).sqrt().recip(), Shape::channels(1));
    let mut tiles = Vec::new();
    for start in (0..seq).step_by(128) {
        let count = 128.min(seq - start);
        let key_count = start + count;
        let q = graph.slice(query, [0, 0, start, 0], [1, heads, count, dim]);
        let k = graph.slice(key, [0, 0, 0, 0], [1, heads, key_count, dim]);
        let v = graph.slice(value, [0, 0, 0, 0], [1, heads, key_count, dim]);
        let scores = graph.matrix_multiplication(q, k, false, true);
        let scores = graph.multiplication(scores, scale);
        let padding = graph.slice(mask, [0, 0, 0, 0], [1, 1, 1, key_count]);
        let scores = graph.addition(scores, padding);
        let mut causal = vec![0.0; count * key_count];
        for q in 0..count {
            for k in start + q + 1..key_count {
                causal[q * key_count + k] = -10_000.0;
            }
        }
        let causal = graph.constant(
            &causal,
            Shape {
                batch: 1,
                channels: 1,
                height: count,
                width: key_count,
            },
        );
        let scores = graph.addition(scores, causal);
        let logits = graph.transpose(scores, [0, 3, 1, 2]);
        let probabilities = graph.soft_max(logits, 1);
        let probabilities = graph.transpose(probabilities, [0, 2, 3, 1]);
        tiles.push(graph.matrix_multiplication(probabilities, v, false, false));
    }
    let context = if tiles.len() == 1 {
        tiles[0]
    } else {
        graph.concat(&tiles, 2)
    };
    let context = graph.transpose(context, [0, 1, 3, 2]);
    let context = graph.reshape(context, shape(seq, heads * dim));
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
