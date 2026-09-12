//! Is a Neural Engine dispatch bound by weight movement or by arithmetic?
//!
//! The weight sweep in `sram_cliff` cannot answer this. It grows a square
//! projection, so weight bytes and multiply-accumulates both scale as the square
//! of the dimension and move together; any slope fits either explanation.
//!
//! Holding the weight matrix fixed and growing only the sequence length breaks
//! the tie. Weights stay constant, arithmetic grows linearly with sequence, so:
//!
//!   flat               weight movement dominates
//!   linear in sequence arithmetic dominates
//!
//! This matters for fusion. If dispatches are bound by weight movement, fusing
//! layers saves real time because each layer's weights move once regardless of
//! how many are in a graph. If they are bound by arithmetic, fusion only saves
//! the fixed per-dispatch cost and nothing more.

use ane::{Graph, NSQualityOfService, Shape, TensorData};

/// Wide enough that its weights are substantial (8 MiB) but small enough to
/// leave headroom for long sequences.
const DIM: usize = 2048;

const REPEATS: usize = 12;

fn main() {
    let weight_bytes = DIM * DIM * 2;
    println!(
        "fixed weights: {:.1} MiB, growing only the sequence axis",
        weight_bytes as f64 / (1024.0 * 1024.0)
    );
    println!("{:>8}  {:>12}  {:>16}", "seq", "median us", "us per position");

    let scale = 1.0 / (DIM as f32).sqrt();
    let weights: Vec<f32> = (0..DIM * DIM)
        .map(|i| if i % 7 == 0 { scale } else { -scale })
        .collect();

    for seq in [64usize, 128, 256, 512, 1024] {
        let shape = Shape {
            batch: 1,
            channels: DIM,
            height: 1,
            width: seq,
        };

        let mut graph = Graph::new();
        let input_tensor = graph.placeholder(shape);
        let _ = graph.inner_product(input_tensor, &weights, DIM, DIM);

        let executable = match graph.compile(NSQualityOfService::UserInteractive) {
            Ok(executable) => executable,
            Err(error) => {
                println!("{seq:>8}  compile failed: {error:?}");
                continue;
            }
        };

        let input = TensorData::with_f32(&vec![0.01_f32; DIM * seq], shape);
        let output = TensorData::new(shape);

        if executable.run_cached(&[&input], &[&output]).is_err() {
            println!("{seq:>8}  execute failed");
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
            println!("{seq:>8}  execution failed mid-run");
            continue;
        }

        samples.sort_unstable();
        let median_us = samples[samples.len() / 2] as f64 / 1000.0;
        println!(
            "{seq:>8}  {median_us:>12.1}  {:>16.3}",
            median_us / seq as f64
        );
    }
}
