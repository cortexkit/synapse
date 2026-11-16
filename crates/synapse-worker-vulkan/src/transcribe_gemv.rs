//! Row-major f32 GEMV for the host predictor and joint. No packing or reduced precision.

type Kernel = unsafe fn(&[f32], &[f32], &mut [f32]);

pub(crate) struct Gemv(Kernel);

impl Gemv {
    pub(crate) fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx") {
            return Self(avx);
        }
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("neon") {
            return Self(neon);
        }
        Self(scalar)
    }

    pub(crate) fn apply(&self, matrix: &[f32], input: &[f32], output: &mut [f32]) {
        assert!(!input.is_empty());
        assert_eq!(matrix.len(), input.len() * output.len());
        // SAFETY: detection at model load checks the kernel's CPU features. The
        // dimensions above bound every unaligned load/store in the SIMD kernels.
        unsafe { (self.0)(matrix, input, output) }
    }
}

fn scalar(matrix: &[f32], input: &[f32], output: &mut [f32]) {
    for (row, out) in matrix.chunks_exact(input.len()).zip(output) {
        *out = row.iter().zip(input).fold(0., |a, (w, x)| a + w * x);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn avx(matrix: &[f32], input: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;
    let width = input.len();
    let end = width / 16 * 16;
    // SAFETY: Gemv::apply validates dimensions and AVX support. Each block has
    // four complete rows; k+16 <= width. Unaligned loads accept Vec alignment.
    unsafe {
        for (block, out) in matrix
            .chunks_exact(4 * width)
            .zip(output.as_chunks_mut::<4>().0.iter_mut())
        {
            // Two independent sums per row hide add latency without needing FMA.
            let mut sums = [[_mm256_setzero_ps(); 2]; 4];
            for k in (0..end).step_by(16) {
                let x0 = _mm256_loadu_ps(input.as_ptr().add(k));
                let x1 = _mm256_loadu_ps(input.as_ptr().add(k + 8));
                for (r, sum) in sums.iter_mut().enumerate() {
                    let w = block.as_ptr().add(r * width + k);
                    sum[0] = _mm256_add_ps(sum[0], _mm256_mul_ps(_mm256_loadu_ps(w), x0));
                    sum[1] = _mm256_add_ps(sum[1], _mm256_mul_ps(_mm256_loadu_ps(w.add(8)), x1));
                }
            }
            for (r, sum) in sums.into_iter().enumerate() {
                let mut lanes = [0.; 8];
                _mm256_storeu_ps(lanes.as_mut_ptr(), _mm256_add_ps(sum[0], sum[1]));
                out[r] = lanes.into_iter().sum();
                for k in end..width {
                    out[r] += block[r * width + k] * input[k];
                }
            }
        }
    }
    // At most three rows; this also handles arbitrary non-model test shapes.
    let rows = output.len() / 4 * 4;
    scalar(&matrix[rows * width..], input, &mut output[rows..]);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon(matrix: &[f32], input: &[f32], output: &mut [f32]) {
    use std::arch::aarch64::*;
    let width = input.len();
    let end = width / 8 * 8;
    // SAFETY: Gemv::apply validates dimensions and NEON support. Each block has
    // four complete rows and k+8 <= width; vld1q/vst1q allow unaligned addresses.
    unsafe {
        for (block, out) in matrix
            .chunks_exact(4 * width)
            .zip(output.as_chunks_mut::<4>().0.iter_mut())
        {
            let mut sums = [[vdupq_n_f32(0.); 2]; 4];
            for k in (0..end).step_by(8) {
                let x0 = vld1q_f32(input.as_ptr().add(k));
                let x1 = vld1q_f32(input.as_ptr().add(k + 4));
                for (r, sum) in sums.iter_mut().enumerate() {
                    let w = block.as_ptr().add(r * width + k);
                    sum[0] = vaddq_f32(sum[0], vmulq_f32(vld1q_f32(w), x0));
                    sum[1] = vaddq_f32(sum[1], vmulq_f32(vld1q_f32(w.add(4)), x1));
                }
            }
            for (r, sum) in sums.into_iter().enumerate() {
                let mut lanes = [0.; 4];
                vst1q_f32(lanes.as_mut_ptr(), vaddq_f32(sum[0], sum[1]));
                out[r] = lanes.into_iter().sum();
                for k in end..width {
                    out[r] += block[r * width + k] * input[k];
                }
            }
        }
    }
    let rows = output.len() / 4 * 4;
    scalar(&matrix[rows * width..], input, &mut output[rows..]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcribe_simd_gemv_matches_scalar_on_predictor_and_joint_shapes() {
        let mut seed = 0x5eed_1234u32;
        let mut random = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f64 / u32::MAX as f64 * 2. - 1.) as f32
        };
        let kernel = Gemv::detect();
        for (rows, cols) in [(2560, 640), (640, 640), (8198, 640), (7, 37)] {
            let matrix: Vec<_> = (0..rows * cols).map(|_| random()).collect();
            let input: Vec<_> = (0..cols).map(|_| random()).collect();
            let mut expected = vec![0.; rows];
            let mut actual = vec![0.; rows];
            scalar(&matrix, &input, &mut expected);
            kernel.apply(&matrix, &input, &mut actual);
            for (r, (a, b)) in actual.iter().zip(&expected).enumerate() {
                // Reassociation changes rounding. Bound error by the row's
                // absolute product sum, not a relative error near cancellation.
                let magnitude: f32 = matrix[r * cols..(r + 1) * cols]
                    .iter()
                    .zip(&input)
                    .map(|(w, x)| (w * x).abs())
                    .sum();
                let tolerance = 8. * f32::EPSILON * magnitude.max(1.);
                assert!(
                    (a - b).abs() <= tolerance,
                    "{rows}x{cols} row {r}: {a} vs {b}, tolerance {tolerance}"
                );
            }
        }
    }
}
