//! What does a new sequence shape cost on the direct API?
//!
//! This is the question that sent us here. The Core ML lane carries one
//! compiled package per sequence bucket, and each one costs specialization time
//! on first load — 32 s at 1024 positions, 579 s at 8192 — and about 275 MB on
//! disk, because every package embeds its own copy of the weights. Three
//! buckets is most of a gigabyte; a five-step ladder would be worse. That, and
//! not raw speed, is what direct control would be bought for.
//!
//! So: compile the same layer at a range of sequence lengths and time each
//! compile. If a shape costs seconds here rather than minutes, a ladder stops
//! being a packaging problem — shapes can be compiled on demand instead of
//! shipped as artifacts, and the ladder's step count stops being a cost at all.
//!
//! Note what this does not settle. Weights are baked into the compiled program
//! as fp16 constants, so each compiled shape may well hold its own copy in
//! memory; whether several shapes can share one weight buffer is a separate
//! question this probe does not answer.

use ane::{Graph, NSQualityOfService, Shape};

const DIM: usize = 768;
const HEADS: usize = 12;
const INTERMEDIATE: usize = 1152;

fn weights(count: usize, scale: f32) -> Vec<f32> {
    (0..count)
        .map(|i| (((i * 2654435761) % 1024) as f32 / 1024.0 - 0.5) * scale)
        .collect()
}

fn main() {
    println!("one ModernBERT-shaped layer, compiled at each sequence length");
    println!(
        "{:>8}  {:>16}  {:>18}",
        "seq", "build + compile s", "weight MiB in graph"
    );

    let projection_scale = 1.0 / (DIM as f32).sqrt();
    let head_dim = DIM / HEADS;

    // Four square projections plus three feed-forward matrices, the weight set
    // one encoder layer carries.
    let weight_elements = 4 * DIM * DIM + 3 * DIM * INTERMEDIATE;
    let weight_mib = (weight_elements * 2) as f64 / (1024.0 * 1024.0);

    for seq in [128usize, 256, 512, 1024, 2048] {
        let started = std::time::Instant::now();

        let shape = Shape {
            batch: 1,
            channels: DIM,
            height: 1,
            width: seq,
        };
        let mut graph = Graph::new();
        let input = graph.placeholder(shape);

        let query = graph.inner_product(input, &weights(DIM * DIM, projection_scale), DIM, DIM);
        let key = graph.inner_product(input, &weights(DIM * DIM, projection_scale), DIM, DIM);
        let value = graph.inner_product(input, &weights(DIM * DIM, projection_scale), DIM, DIM);

        let heads_shape = Shape {
            batch: 1,
            channels: HEADS,
            height: head_dim,
            width: seq,
        };
        let swap_last_two = [0, 1, 3, 2];
        let query = graph.reshape(query, heads_shape);
        let key = graph.reshape(key, heads_shape);
        let value = graph.reshape(value, heads_shape);
        let query = graph.transpose(query, swap_last_two);
        let key = graph.transpose(key, swap_last_two);
        let value = graph.transpose(value, swap_last_two);

        let scores = graph.matrix_multiplication(query, key, false, true);
        let probabilities = graph.soft_max(scores, -1);
        let context = graph.matrix_multiplication(probabilities, value, false, false);
        let context = graph.transpose(context, swap_last_two);
        let context = graph.reshape(context, shape);

        let attended =
            graph.inner_product(context, &weights(DIM * DIM, projection_scale), DIM, DIM);
        let residual = graph.addition(input, attended);

        let gated = graph.inner_product(
            residual,
            &weights(DIM * INTERMEDIATE, projection_scale),
            DIM,
            INTERMEDIATE,
        );
        let activated = graph.tanh(gated);
        let up = graph.inner_product(
            residual,
            &weights(DIM * INTERMEDIATE, projection_scale),
            DIM,
            INTERMEDIATE,
        );
        let hidden = graph.multiplication(activated, up);
        let projected = graph.inner_product(
            hidden,
            &weights(INTERMEDIATE * DIM, 1.0 / (INTERMEDIATE as f32).sqrt()),
            INTERMEDIATE,
            DIM,
        );
        let _ = graph.addition(residual, projected);

        match graph.compile(NSQualityOfService::UserInteractive) {
            Ok(_) => println!(
                "{seq:>8}  {:>16.2}  {weight_mib:>18.1}",
                started.elapsed().as_secs_f64()
            ),
            Err(error) => println!("{seq:>8}  compile failed: {error:?}"),
        }
    }
}
