//! Does a second compiled shape cost a second copy of the weights?
//!
//! Compiling a shape on this API costs about 0.2 s against Core ML's tens to
//! hundreds of seconds, which makes a sequence ladder cheap to BUILD. It does
//! not say what a ladder COSTS TO HOLD. Core ML's packages each embed their own
//! weights, so three buckets of gte-modernbert occupy about 825 MB; if compiled
//! executables here behave the same way, the direct API fixes compile time and
//! disk while leaving resident memory exactly as it was.
//!
//! The binding reference says weights passed to `inner_product` are baked into
//! the program as fp16 constants, which suggests no sharing. Espresso's
//! architecture diagram shows weights living in their own IOSurface beside the
//! program, which suggests the opposite is achievable. Measuring settles it.
//!
//! Two arms, because they answer different questions:
//!
//!   same weights, many shapes   what a sequence ladder costs
//!   many weights, one shape     the per-layer cost of a whole model
//!
//! Resident size is read from `ps`, which reports what the process holds
//! including whatever the ANE runtime allocated on its behalf.

use ane::{Executable, Graph, NSQualityOfService, Shape, TensorData};

const DIM: usize = 768;
const REPEATS: usize = 5;

fn resident_mib() -> f64 {
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .expect("ps should report this process");
    let kib: f64 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .unwrap_or(0.0);
    kib / 1024.0
}

/// Four square projections, the weight set of one attention block, which is
/// enough mass for a copy to be unmistakable.
fn weights_for(seed: usize) -> Vec<f32> {
    let scale = 1.0 / (DIM as f32).sqrt();
    (0..DIM * DIM)
        .map(|i| ((((i + seed) * 2654435761) % 1024) as f32 / 1024.0 - 0.5) * scale)
        .collect()
}

fn compile_at(seq: usize, seed: usize) -> Option<(Executable, Shape)> {
    let shape = Shape {
        batch: 1,
        channels: DIM,
        height: 1,
        width: seq,
    };
    let mut graph = Graph::new();
    let input = graph.placeholder(shape);
    // Chained rather than parallel: four independent projections would leave
    // four terminal tensors and only one output buffer is bound, which fails at
    // execution. Chaining keeps all four weight sets baked with a single
    // terminal, which is what this probe needs.
    let mut tensor = input;
    for index in 0..4 {
        let weights = weights_for(seed + index);
        tensor = graph.inner_product(tensor, &weights, DIM, DIM);
    }
    let executable = graph.compile(NSQualityOfService::UserInteractive).ok()?;
    Some((executable, shape))
}

/// Compile and then EXECUTE, because a graph that never ran may not have
/// materialized its weights. macOS charges Neural Engine residency to the
/// calling process, so execution is the moment that cost becomes visible;
/// measuring after compile alone understates what a serving ladder holds.
fn compile_and_run(seq: usize, seed: usize) -> Result<Executable, &'static str> {
    let (executable, shape) = compile_at(seq, seed).ok_or("compile failed")?;
    let input = TensorData::with_f32(&vec![0.02_f32; DIM * seq], shape);
    let output = TensorData::new(shape);
    executable.run(&[&input], &[&output]).map_err(|_| "run failed")?;
    Ok(executable)
}

fn main() {
    let weight_mib = (4 * DIM * DIM * 2) as f64 / (1024.0 * 1024.0);
    println!("one attention block's weights: {weight_mib:.1} MiB baked per compile");

    println!("\nsame weights at {REPEATS} different shapes, each executed (a sequence ladder)");
    println!("{:>8}  {:>14}  {:>16}", "shape", "resident MiB", "growth MiB");
    let baseline = resident_mib();
    println!("{:>8}  {baseline:>14.1}  {:>16}", "none", "-");
    let mut held = Vec::new();
    for (index, seq) in [128usize, 256, 512, 1024, 2048].iter().enumerate() {
        match compile_and_run(*seq, 0) {
            Ok(executable) => {
                held.push(executable);
                let now = resident_mib();
                println!("{seq:>8}  {now:>14.1}  {:>16.1}", now - baseline);
                let _ = index;
            }
            Err(reason) => println!("{seq:>8}  {reason}"),
        }
    }
    drop(held);

    println!("\ndifferent weights at one shape, each executed (a model's layers)");
    println!("{:>8}  {:>14}  {:>16}", "layer", "resident MiB", "growth MiB");
    let baseline = resident_mib();
    println!("{:>8}  {baseline:>14.1}  {:>16}", "none", "-");
    let mut held = Vec::new();
    for layer in 0..REPEATS {
        match compile_and_run(512, layer * 7919 + 1) {
            Ok(executable) => {
                held.push(executable);
                let now = resident_mib();
                println!("{layer:>8}  {now:>14.1}  {:>16.1}", now - baseline);
            }
            Err(reason) => println!("{layer:>8}  {reason}"),
        }
    }
}
