//! Where does this chip's Neural Engine fall off its SRAM cache?
//!
//! The binding reference reports ~32 MB of SRAM, with weights under 16 MB
//! streaming at ~15,000 GB/s and larger ones dropping to ~51 GB/s from DRAM —
//! a ~300x cliff. That number was measured on other hardware, and it decides
//! how deeply transformer layers should be fused here: fusing cuts the
//! per-dispatch cost, but a fused graph whose weights cross the threshold pays
//! far more in bandwidth than it saves in dispatch.
//!
//! GTE ModernBERT carries roughly 13.6 MB of weights per layer, which sits just
//! under a 16 MB threshold and just over half of it. So the exact location of
//! the cliff on this chip is the difference between "one layer per dispatch is
//! optimal" and "fuse two".
//!
//! Method: one `inner_product` whose weight matrix grows across the suspected
//! threshold, timed by wall clock around the cached execution path.
//!
//! The hardware counter would have been preferable, but `run_cached_with_stats`
//! reports 0 ns at every size on this machine — the statistics mask appears to
//! need setting on the model before it is loaded rather than on a request after
//! compilation, so `hwExecutionTime` is never populated. Wall clock still
//! answers this question: dispatch overhead is a constant of roughly 0.095 ms,
//! and a constant cannot disguise the ~300x change a cache cliff would produce.
//! It does mean the absolute bandwidth figures carry that fixed cost and read
//! low for the fastest rows; the SHAPE of the curve is the result here, not any
//! single number in it.

use ane::{Graph, NSQualityOfService, Shape, TensorData};

/// Sequence positions on the width axis; the minimum a placeholder accepts.
/// Kept at the floor so activations stay negligible beside the weights, which
/// are what this probe is measuring.
const SEQ: usize = 64;

const REPEATS: usize = 12;

fn main() {
    println!(
        "{:>6}  {:>12}  {:>14}  {:>14}",
        "dim", "weight MiB", "median us", "effective GB/s"
    );

    for dim in [512usize, 1024, 2048, 2816, 3072, 4096, 4608, 5120, 5632, 6144] {
        let weight_bytes = dim * dim * 2;
        let weight_mib = weight_bytes as f64 / (1024.0 * 1024.0);

        // Small magnitudes so fp16 accumulation over `dim` terms cannot
        // overflow and turn a bandwidth measurement into a numerical one.
        let scale = 1.0 / (dim as f32).sqrt();
        let weights: Vec<f32> = (0..dim * dim)
            .map(|i| if i % 7 == 0 { scale } else { -scale })
            .collect();

        let shape = Shape {
            batch: 1,
            channels: dim,
            height: 1,
            width: SEQ,
        };

        let mut graph = Graph::new();
        let input_tensor = graph.placeholder(shape);
        let _ = graph.inner_product(input_tensor, &weights, dim, dim);

        let executable = match graph.compile(NSQualityOfService::UserInteractive) {
            Ok(executable) => executable,
            Err(error) => {
                println!("{dim:>6}  {weight_mib:>12.1}  compile failed: {error:?}");
                continue;
            }
        };

        let input = TensorData::with_f32(&vec![0.01_f32; dim * SEQ], shape);
        let output = TensorData::new(shape);

        // The same TensorData objects must be reused for the cached path, and
        // the first call pays one-time setup, so it is warmup rather than a
        // sample.
        if executable.run_cached(&[&input], &[&output]).is_err() {
            println!("{dim:>6}  {weight_mib:>12.1}  execute failed");
            continue;
        }

        let mut samples = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let started = std::time::Instant::now();
            if executable.run_cached(&[&input], &[&output]).is_err() {
                println!("{dim:>6}  {weight_mib:>12.1}  execute failed mid-run");
                break;
            }
            samples.push(started.elapsed().as_nanos() as u64);
        }
        if samples.len() < REPEATS {
            continue;
        }

        samples.sort_unstable();
        let median_ns = samples[samples.len() / 2] as f64;
        let gb_per_s = weight_bytes as f64 / median_ns;

        println!(
            "{dim:>6}  {weight_mib:>12.1}  {:>14.1}  {gb_per_s:>14.1}",
            median_ns / 1000.0
        );
    }
}
