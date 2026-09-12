//! What would a real ModernBERT encoder layer cost on the direct API?
//!
//! Everything measured so far is a single projection. That establishes the cost
//! model — a fixed cost near 92 us per dispatch, no memory cliff, arithmetic
//! bound past a few hundred positions — but it does not answer whether this
//! path is worth building. The production Core ML lane embeds a 511-token row in
//! about 25.5 ms end to end. If 22 layers here land near that, the direct API
//! buys control without speed and the case rests on packaging alone; if they
//! land far below it, there is real headroom.
//!
//! Weights are random rather than loaded. Shapes decide timing and random values
//! of the right magnitude time identically to trained ones, so this answers the
//! timing question at a fraction of the work. It answers nothing about numerics,
//! and a correctness gate against a reference is required before any of this is
//! believed as a model.
//!
//! Geometry is gte-modernbert-base as configured: 768 hidden, 12 heads of 64,
//! 1152 intermediate with a gated activation, 22 layers.

use ane::{Graph, NSQualityOfService, Shape, TensorData};

const DIM: usize = 768;
const HEADS: usize = 12;
const INTERMEDIATE: usize = 1152;
const LAYERS: usize = 22;
const REPEATS: usize = 10;

fn random_weights(count: usize, scale: f32) -> Vec<f32> {
    // A cheap deterministic spread. Timing does not depend on the values, only
    // on their count and on staying in fp16 range.
    (0..count)
        .map(|i| {
            let x = ((i * 2654435761) % 1024) as f32 / 1024.0 - 0.5;
            x * scale
        })
        .collect()
}

fn shape_at(seq: usize, channels: usize) -> Shape {
    Shape {
        batch: 1,
        channels,
        height: 1,
        width: seq,
    }
}

/// One encoder layer: attention projections, per-head scores and context, the
/// output projection, and a gated feed-forward block. Layer norms are omitted:
/// they are elementwise and cheap beside the projections, and including them
/// would add ops against the compiler's depth limit without changing what this
/// measures.
fn build_layer(graph: &mut Graph, input: ane::Tensor, seq: usize) -> ane::Tensor {
    let projection_scale = 1.0 / (DIM as f32).sqrt();
    let head_dim = DIM / HEADS;

    let query = graph.inner_product(input, &random_weights(DIM * DIM, projection_scale), DIM, DIM);
    let key = graph.inner_product(input, &random_weights(DIM * DIM, projection_scale), DIM, DIM);
    let value = graph.inner_product(input, &random_weights(DIM * DIM, projection_scale), DIM, DIM);

    // Heads are scored independently. Slicing the channel axis per head keeps
    // each matmul at the head's own width, which is what the real model does.
    let mut head_contexts = Vec::with_capacity(HEADS);
    for head in 0..HEADS {
        let begin = [0, head * head_dim, 0, 0];
        let size = [1, head_dim, 1, seq];
        let query_head = graph.slice(query, begin, size);
        let key_head = graph.slice(key, begin, size);
        let value_head = graph.slice(value, begin, size);

        let scores = graph.matrix_multiplication(query_head, key_head, true, false);
        let weights = graph.soft_max(scores, -1);
        let context = graph.matrix_multiplication(value_head, weights, false, true);
        head_contexts.push(context);
    }
    let context = graph.concat(&head_contexts, 1);
    let attended = graph.inner_product(
        context,
        &random_weights(DIM * DIM, projection_scale),
        DIM,
        DIM,
    );
    let residual = graph.addition(input, attended);

    // Gated feed-forward: one projection produces both halves, one gates the
    // other, and a final projection returns to the model width.
    let gate_scale = 1.0 / (DIM as f32).sqrt();
    let gated = graph.inner_product(
        residual,
        &random_weights(DIM * INTERMEDIATE, gate_scale),
        DIM,
        INTERMEDIATE,
    );
    let activated = graph.tanh(gated);
    let up = graph.inner_product(
        residual,
        &random_weights(DIM * INTERMEDIATE, gate_scale),
        DIM,
        INTERMEDIATE,
    );
    let hidden = graph.multiplication(activated, up);
    let down_scale = 1.0 / (INTERMEDIATE as f32).sqrt();
    let projected = graph.inner_product(
        hidden,
        &random_weights(INTERMEDIATE * DIM, down_scale),
        INTERMEDIATE,
        DIM,
    );
    graph.addition(residual, projected)
}

fn main() {
    println!("gte-modernbert-base geometry, random weights, timing only");
    println!(
        "{:>6}  {:>8}  {:>14}  {:>18}  {:>16}",
        "seq", "fused", "median us", "us per layer", "22 layers ms"
    );

    for seq in [128usize, 512] {
        for fused in [1usize, 2, 3] {
            let shape = shape_at(seq, DIM);
            let mut graph = Graph::new();
            let mut tensor = graph.placeholder(shape);
            for _ in 0..fused {
                tensor = build_layer(&mut graph, tensor, seq);
            }

            let executable = match graph.compile(NSQualityOfService::UserInteractive) {
                Ok(executable) => executable,
                Err(error) => {
                    println!("{seq:>6}  {fused:>8}  compile failed: {error:?}");
                    continue;
                }
            };

            let input = TensorData::with_f32(&vec![0.02_f32; DIM * seq], shape);
            let output = TensorData::new(shape);
            if executable.run_cached(&[&input], &[&output]).is_err() {
                println!("{seq:>6}  {fused:>8}  execute failed");
                continue;
            }

            let mut samples = Vec::with_capacity(REPEATS);
            for _ in 0..REPEATS {
                let started = std::time::Instant::now();
                if executable.run_cached(&[&input], &[&output]).is_err() {
                    break;
                }
                samples.push(started.elapsed().as_nanos() as u64);
            }
            if samples.len() < REPEATS {
                println!("{seq:>6}  {fused:>8}  execution failed mid-run");
                continue;
            }

            samples.sort_unstable();
            let median_us = samples[samples.len() / 2] as f64 / 1000.0;
            let per_layer = median_us / fused as f64;
            println!(
                "{seq:>6}  {fused:>8}  {median_us:>14.1}  {per_layer:>18.1}  {:>16.2}",
                per_layer * LAYERS as f64 / 1000.0
            );
        }
    }
}
