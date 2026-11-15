//! Scalar, row-major attention checked against a pinned upstream module output.
use super::{relative_positions, relative_row_offset};

#[test]
fn upstream_relative_attention_matches_static_lowering_on_small_padded_sequence() {
    // The expected tensor comes from the actual transformers 5.16.1 attention
    // and positional-encoding classes, with a finite masking sentinel before
    // softmax, zeroed masked probabilities, and the diagonal projections below.
    // It is not computed by reproducing the worker's relative-shift formula.
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../tests/fixtures/parakeet_relative_attention.json"
    ))
    .unwrap();
    let length = fixture["frames"].as_u64().unwrap() as usize;
    let valid = fixture["valid_frames"].as_u64().unwrap() as usize;
    let hidden = fixture["hidden"].as_u64().unwrap() as usize;
    let heads = fixture["heads"].as_u64().unwrap() as usize;
    let head = hidden / heads;
    let relative = length * 2 - 1;
    let positions = relative_positions(length, hidden);
    let input: Vec<_> = (0..length)
        .flat_map(|t| (0..hidden).map(move |c| ((t * 17 + c * 7) % 31) as f32 / 20.0 - 0.75))
        .collect();
    let q: Vec<_> = input
        .iter()
        .enumerate()
        .map(|(i, x)| x * (0.5 + (i % hidden % 3) as f32 * 0.1))
        .collect();
    let k: Vec<_> = input
        .iter()
        .enumerate()
        .map(|(i, x)| x * (-0.4 + (i % hidden % 5) as f32 * 0.15))
        .collect();
    let v: Vec<_> = input
        .iter()
        .enumerate()
        .map(|(i, x)| x * (0.3 + (i % hidden % 7) as f32 * 0.05))
        .collect();
    let mut output = vec![0.0f32; length * hidden];
    for h in 0..heads {
        for row in 0..length {
            let mut position_scores = vec![0.0; relative];
            for (p, score) in position_scores.iter_mut().enumerate() {
                for d in 0..head {
                    let c = h * head + d;
                    let bias_v = (c as i32 % 17 - 8) as f32 * 0.015;
                    let projected = positions[c * relative + p] * (0.2 + (c % 11) as f32 * 0.03);
                    *score += (q[row * hidden + c] + bias_v) * projected;
                }
            }
            let mut scores = vec![0.0; length];
            for (key, score) in scores.iter_mut().enumerate() {
                let mut content = 0.0;
                for d in 0..head {
                    let c = h * head + d;
                    let bias_u = (c as i32 % 13 - 6) as f32 * 0.02;
                    content += (q[row * hidden + c] + bias_u) * k[key * hidden + c];
                }
                *score = (content + position_scores[relative_row_offset(length, row) + key])
                    / (head as f32).sqrt();
                if row >= valid || key >= valid {
                    *score = -10000.0;
                }
            }
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut probabilities: Vec<_> = scores.iter().map(|x| (x - maximum).exp()).collect();
            let total: f32 = probabilities.iter().sum();
            for (key, p) in probabilities.iter_mut().enumerate() {
                *p = if row >= valid || key >= valid {
                    0.0
                } else {
                    *p / total
                };
            }
            for d in 0..head {
                let c = h * head + d;
                output[row * hidden + c] = probabilities
                    .iter()
                    .enumerate()
                    .map(|(key, p)| p * v[key * hidden + c])
                    .sum::<f32>()
                    * (0.9 + (c % 2) as f32 * 0.02);
            }
        }
    }
    let expected = fixture["output"].as_array().unwrap();
    assert_eq!(output.len(), expected.len());
    for (index, (actual, expected)) in output.iter().zip(expected).enumerate() {
        assert!(
            (f64::from(*actual) - expected.as_f64().unwrap()).abs() < 2e-6,
            "upstream attention output {index}: actual={actual}, expected={expected}"
        );
    }
}
