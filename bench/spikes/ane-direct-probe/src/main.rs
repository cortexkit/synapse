//! Does the private Neural Engine API work on this machine?
//!
//! The published compatibility matrix for these bindings covers M1 through M4
//! on macOS 15. This box is an M5 Max on macOS 27, so neither the chip nor the
//! OS has been tested; the bindings resolve `_ANEInMemoryModel` by `dlopen` at
//! runtime, which a build cannot verify. Everything downstream assumes this
//! works, so it is worth one cheap answer before porting a model.
//!
//! The probe computes something whose answer is known independently: a linear
//! projection by an identity matrix must return its input unchanged. That
//! separates "the API ran" from "the API ran and the hardware computed
//! correctly", which a status code alone would not.

use ane::{Graph, NSQualityOfService, Shape, TensorData};

/// Hidden width, carried on the channel axis.
const DIM: usize = 64;

/// Sequence positions, carried on the WIDTH axis. A placeholder narrower than
/// 64 is rejected at compile time with `SpatialWidthTooSmall`, which is why the
/// binding reference says to pad shorter sequences: width is the sequence axis.
const SEQ: usize = 64;

fn main() {
    let identity: Vec<f32> = (0..DIM * DIM)
        .map(|i| if i / DIM == i % DIM { 1.0 } else { 0.0 })
        .collect();

    // Distinct per (channel, position) rather than a constant: a constant would
    // still look correct if the projection collapsed outputs together or if the
    // sequence axis were transposed. Magnitudes stay small so fp16 rounding
    // cannot be mistaken for a wrong answer. Layout is NCHW with height 1, so
    // the element for channel c at position w sits at c * SEQ + w.
    let probe: Vec<f32> = (0..DIM * SEQ)
        .map(|i| {
            let channel = (i / SEQ) as f32;
            let position = (i % SEQ) as f32;
            channel * 0.03125 + position * 0.001_953_125
        })
        .collect();

    let shape = Shape {
        batch: 1,
        channels: DIM,
        height: 1,
        width: SEQ,
    };

    let mut graph = Graph::new();
    let input_tensor = graph.placeholder(shape);
    let _projected = graph.inner_product(input_tensor, &identity, DIM, DIM);

    let executable = match graph.compile(NSQualityOfService::UserInteractive) {
        Ok(executable) => executable,
        Err(error) => {
            println!("COMPILE FAILED: {error:?}");
            std::process::exit(2);
        }
    };
    println!("compiled a graph through the private API");

    let input = TensorData::with_f32(&probe, shape);
    let output = TensorData::new(shape);

    if let Err(error) = executable.run(&[&input], &[&output]) {
        println!("EXECUTE FAILED: {error:?}");
        std::process::exit(3);
    }
    println!("executed on the Neural Engine");

    let produced = output.read_f32();
    let worst = probe
        .iter()
        .zip(produced.iter())
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0_f32, f32::max);

    println!("worst absolute error against the identity projection: {worst:e}");

    // fp16 compute over inputs bounded by 12, so a correct result lands far
    // below this; the threshold exists to catch a wrong answer, not to measure
    // precision.
    if worst > 0.05 {
        println!("VERDICT: ran, but the arithmetic is wrong");
        std::process::exit(4);
    }
    println!("VERDICT: private ANE API works on this chip and OS");
}
