//! Where do 50 ms per ModernBERT layer actually go?
//!
//! A shape-accurate layer costs about 50 ms at 512 positions on the direct API,
//! against 25.5 ms for the whole 22-layer model through Core ML. Something is
//! wrong rather than merely unoptimized: the layer is roughly 5.2 GFLOP, so
//! 50 ms is about 100 GFLOP/s from hardware that should deliver far more.
//!
//! This splits the layer into pieces that are timed separately, so the cost
//! lands on a named part instead of the whole:
//!
//!   projections   four constant-weight linears, no attention
//!   feed-forward  the gated block, also constant-weight
//!   attention     per-head scores and context from runtime tensors
//!   single matmul one dynamic matmul, to price that op alone
//!
//! The distinction that matters is CONSTANT-weight (`inner_product`, weights
//! baked at compile time) against DYNAMIC (`matrix_multiplication`, both sides
//! runtime tensors). Attention needs the dynamic form and the rest does not, so
//! if the dynamic op is the problem this separates it cleanly. If every arm is
//! slow in proportion to its arithmetic, the problem is not any single op and
//! the likeliest explanation is that this is not running on the Neural Engine
//! at all.

use ane::{Graph, NSQualityOfService, Shape, TensorData};

const DIM: usize = 768;
const HEADS: usize = 12;
const INTERMEDIATE: usize = 1152;
const SEQ: usize = 512;
const REPEATS: usize = 8;

fn weights(count: usize, scale: f32) -> Vec<f32> {
    (0..count)
        .map(|i| (((i * 2654435761) % 1024) as f32 / 1024.0 - 0.5) * scale)
        .collect()
}

fn model_shape() -> Shape {
    Shape {
        batch: 1,
        channels: DIM,
        height: 1,
        width: SEQ,
    }
}

/// Median wall time for a graph built by `build`, which receives the graph and
/// its input placeholders and returns nothing; only timing is measured here.
fn time_graph(
    label: &str,
    gflop: f64,
    inputs: usize,
    build: impl FnOnce(&mut Graph, &[ane::Tensor]),
) {
    let shape = model_shape();
    let mut graph = Graph::new();
    let placeholders: Vec<ane::Tensor> = (0..inputs).map(|_| graph.placeholder(shape)).collect();
    build(&mut graph, &placeholders);

    let executable = match graph.compile(NSQualityOfService::UserInteractive) {
        Ok(executable) => executable,
        Err(error) => {
            println!("{label:>16}  compile failed: {error:?}");
            return;
        }
    };

    let held: Vec<TensorData> = (0..inputs)
        .map(|_| TensorData::with_f32(&vec![0.02_f32; DIM * SEQ], shape))
        .collect();
    let borrowed: Vec<&TensorData> = held.iter().collect();
    let output = TensorData::new(shape);

    if executable.run_cached(&borrowed, &[&output]).is_err() {
        println!("{label:>16}  execute failed");
        return;
    }

    let mut samples = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let started = std::time::Instant::now();
        if executable.run_cached(&borrowed, &[&output]).is_err() {
            println!("{label:>16}  execution failed mid-run");
            return;
        }
        samples.push(started.elapsed().as_nanos() as u64);
    }
    samples.sort_unstable();
    let median_ms = samples[samples.len() / 2] as f64 / 1_000_000.0;
    println!(
        "{label:>16}  {median_ms:>12.2}  {gflop:>10.2}  {:>14.1}",
        gflop / (median_ms / 1000.0)
    );
}

fn main() {
    println!("gte-modernbert-base, {SEQ} positions, random weights");
    println!(
        "{:>16}  {:>12}  {:>10}  {:>14}",
        "arm", "median ms", "GFLOP", "GFLOP/s"
    );

    let projection_scale = 1.0 / (DIM as f32).sqrt();
    let macs_projection = 2.0 * (DIM * DIM * SEQ) as f64 / 1e9;

    time_graph("4 projections", 4.0 * macs_projection, 1, |g, inputs| {
        for _ in 0..4 {
            let _ = g.inner_product(inputs[0], &weights(DIM * DIM, projection_scale), DIM, DIM);
        }
    });

    time_graph("1 projection", macs_projection, 1, |g, inputs| {
        let _ = g.inner_product(inputs[0], &weights(DIM * DIM, projection_scale), DIM, DIM);
    });

    let ffn_gflop = 2.0 * (3 * DIM * INTERMEDIATE * SEQ) as f64 / 1e9;
    time_graph("gated ffn", ffn_gflop, 1, |g, inputs| {
        let gate = g.inner_product(
            inputs[0],
            &weights(DIM * INTERMEDIATE, projection_scale),
            DIM,
            INTERMEDIATE,
        );
        let activated = g.tanh(gate);
        let up = g.inner_product(
            inputs[0],
            &weights(DIM * INTERMEDIATE, projection_scale),
            DIM,
            INTERMEDIATE,
        );
        let hidden = g.multiplication(activated, up);
        let _ = g.inner_product(
            hidden,
            &weights(INTERMEDIATE * DIM, 1.0 / (INTERMEDIATE as f32).sqrt()),
            INTERMEDIATE,
            DIM,
        );
    });

    // Attention two ways, because the difference is the finding. The sliced arm
    // issues one matmul pair per head; the batched arm puts heads on the channel
    // axis and lets a single matmul cover all of them, which is how the
    // reference implementation expresses it.
    let head_dim = DIM / HEADS;
    let attention_gflop = 2.0 * (2 * SEQ * SEQ * DIM) as f64 / 1e9;

    time_graph("attn per-head", attention_gflop, 3, |g, inputs| {
        let mut contexts = Vec::with_capacity(HEADS);
        for head in 0..HEADS {
            let begin = [0, head * head_dim, 0, 0];
            let size = [1, head_dim, 1, SEQ];
            let q = g.slice(inputs[0], begin, size);
            let k = g.slice(inputs[1], begin, size);
            let v = g.slice(inputs[2], begin, size);
            let scores = g.matrix_multiplication(q, k, true, false);
            let probabilities = g.soft_max(scores, -1);
            contexts.push(g.matrix_multiplication(v, probabilities, false, true));
        }
        let _ = g.concat(&contexts, 1);
    });

    time_graph("attn batched", attention_gflop, 3, |g, inputs| {
        let heads_shape = Shape {
            batch: 1,
            channels: HEADS,
            height: head_dim,
            width: SEQ,
        };
        let swap_last_two = [0, 1, 3, 2];
        let mut per_head = Vec::with_capacity(3);
        for input in inputs.iter().take(3) {
            let reshaped = g.reshape(*input, heads_shape);
            per_head.push(g.transpose(reshaped, swap_last_two));
        }
        let scores = g.matrix_multiplication(per_head[0], per_head[1], false, true);
        let probabilities = g.soft_max(scores, -1);
        let context = g.matrix_multiplication(probabilities, per_head[2], false, false);
        let restored = g.transpose(context, swap_last_two);
        let _ = g.reshape(restored, model_shape());
    });

    // ModernBERT runs a 128-token sliding window on two of every three layers and
    // global attention on the third, so a faithful port pays the global cost far
    // less often than a uniform one. Windowed attention is expressed as query
    // tiles against a bounded key halo: each tile of 128 queries attends to its
    // own span plus 64 positions either side.
    let tile = 128usize;
    let halo = tile + 128;
    let tiles = SEQ / tile;
    let local_gflop = 2.0 * (2 * tiles * tile * halo * DIM) as f64 / 1e9;
    time_graph("attn local w128", local_gflop, 3, |g, inputs| {
        let heads_shape = Shape {
            batch: 1,
            channels: HEADS,
            height: head_dim,
            width: SEQ,
        };
        let swap_last_two = [0, 1, 3, 2];
        let mut per_head = Vec::with_capacity(3);
        for input in inputs.iter().take(3) {
            let reshaped = g.reshape(*input, heads_shape);
            per_head.push(g.transpose(reshaped, swap_last_two));
        }
        let mut contexts = Vec::with_capacity(tiles);
        for index in 0..tiles {
            // Clamp the halo to the sequence so edge tiles stay in bounds; a
            // real port also masks the padded span, which costs no matmul time.
            let key_start = index * tile == 0;
            let begin_keys = if key_start { 0 } else { index * tile - 128 };
            let span = halo.min(SEQ - begin_keys);
            let queries = g.slice(
                per_head[0],
                [0, 0, index * tile, 0],
                [1, HEADS, tile, head_dim],
            );
            let keys = g.slice(per_head[1], [0, 0, begin_keys, 0], [1, HEADS, span, head_dim]);
            let values = g.slice(per_head[2], [0, 0, begin_keys, 0], [1, HEADS, span, head_dim]);
            let scores = g.matrix_multiplication(queries, keys, false, true);
            let probabilities = g.soft_max(scores, -1);
            contexts.push(g.matrix_multiplication(probabilities, values, false, false));
        }
        let joined = g.concat(&contexts, 2);
        let restored = g.transpose(joined, swap_last_two);
        let _ = g.reshape(restored, model_shape());
    });

    // One dynamic matmul at a single head's width, to price the op by itself.
    #[allow(unused)]
    let one_head_gflop = 2.0 * (SEQ * SEQ * head_dim) as f64 / 1e9;
    time_graph("1 dynamic matmul", one_head_gflop, 2, |g, inputs| {
        let begin = [0, 0, 0, 0];
        let size = [1, head_dim, 1, SEQ];
        let q = g.slice(inputs[0], begin, size);
        let k = g.slice(inputs[1], begin, size);
        let _ = g.matrix_multiplication(q, k, true, false);
    });
}
