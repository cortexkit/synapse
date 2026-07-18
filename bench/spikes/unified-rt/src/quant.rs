//! GGUF-compatible block quantization used by CUDA decode weight matrices.

use anyhow::{ensure, Result};
use clap::ValueEnum;
use half::f16;
use serde::Serialize;
use sha2::{Digest, Sha256};

pub(crate) const Q8_0_BLOCK_ELEMENTS: usize = 32;
pub(crate) const Q8_0_BLOCK_BYTES: usize = 2 + Q8_0_BLOCK_ELEMENTS;

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum WeightQuantization {
    #[default]
    None,
    Q8_0,
}

impl WeightQuantization {
    pub(crate) fn is_quantized(self) -> bool {
        !matches!(self, Self::None)
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Q8_0 => "q8_0",
        }
    }
}

/// Exact GGUF `block_q8_0` bytes: little-endian f16 scale followed by 32 i8 values.
#[derive(Clone, Debug)]
pub(crate) struct Q8_0Tensor {
    bytes: Vec<u8>,
}

impl Q8_0Tensor {
    pub(crate) fn quantize(values: &[f32], row_width: usize) -> Result<Self> {
        ensure!(row_width > 0, "Q8_0 matrix row width must be positive");
        ensure!(
            row_width % Q8_0_BLOCK_ELEMENTS == 0,
            "Q8_0 matrix row width {row_width} is not divisible by {Q8_0_BLOCK_ELEMENTS}"
        );
        ensure!(
            values.len() % row_width == 0,
            "Q8_0 matrix data length does not contain complete rows"
        );
        ensure!(
            values.iter().all(|value| value.is_finite()),
            "Q8_0 cannot encode non-finite weights"
        );

        let mut bytes = Vec::with_capacity(values.len() / Q8_0_BLOCK_ELEMENTS * Q8_0_BLOCK_BYTES);
        for block in values.chunks_exact(Q8_0_BLOCK_ELEMENTS) {
            let maximum = block.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
            let scale = maximum / 127.0;
            bytes.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            let inverse = if scale == 0.0 { 0.0 } else { scale.recip() };
            bytes.extend(
                block
                    .iter()
                    .map(|value| (value * inverse).round().clamp(-127.0, 127.0) as i8 as u8),
            );
        }
        Ok(Self { bytes })
    }

    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    #[cfg(test)]
    fn dequantize(&self) -> Vec<f32> {
        let mut values =
            Vec::with_capacity(self.bytes.len() / Q8_0_BLOCK_BYTES * Q8_0_BLOCK_ELEMENTS);
        for block in self.bytes.chunks_exact(Q8_0_BLOCK_BYTES) {
            let scale = f32::from(f16::from_bits(u16::from_le_bytes([block[0], block[1]])));
            values.extend(block[2..].iter().map(|value| (*value as i8) as f32 * scale));
        }
        values
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CudaWeight {
    pub(crate) fp32: *const f32,
    pub(crate) q8_0: *const u8,
}

impl CudaWeight {
    pub(crate) const fn null() -> Self {
        Self {
            fp32: std::ptr::null(),
            q8_0: std::ptr::null(),
        }
    }

    pub(crate) fn new(fp32: &[f32], q8_0: Option<&Q8_0Tensor>) -> Self {
        Self {
            fp32: fp32.as_ptr(),
            q8_0: q8_0.map_or(std::ptr::null(), |weight| weight.as_bytes().as_ptr()),
        }
    }
}

pub(crate) fn quantized_sha256<'a>(weights: impl IntoIterator<Item = &'a Q8_0Tensor>) -> String {
    let mut digest = Sha256::new();
    for weight in weights {
        digest.update(weight.as_bytes());
    }
    format!("{:x}", digest.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_bytes_match_the_gguf_block_layout() {
        let values = (-16..16).map(|value| value as f32).collect::<Vec<_>>();
        let quantized = Q8_0Tensor::quantize(&values, 32).unwrap();
        assert_eq!(quantized.as_bytes().len(), 34);
        assert_eq!(
            &quantized.as_bytes()[..2],
            &f16::from_f32(16.0 / 127.0).to_bits().to_le_bytes()
        );
        let expected = values
            .iter()
            .map(|value| (value * (127.0 / 16.0)).round() as i8 as u8)
            .collect::<Vec<_>>();
        assert_eq!(&quantized.as_bytes()[2..], expected);
    }

    #[test]
    fn q8_0_round_trip_has_block_scale_error_bound() {
        let values = (0..64)
            .map(|index| ((index as f32 * 0.37).sin() * 4.0) - 0.25)
            .collect::<Vec<_>>();
        let quantized = Q8_0Tensor::quantize(&values, 32).unwrap();
        let decoded = quantized.dequantize();
        for (block, source) in values.chunks_exact(32).enumerate() {
            let scale = source.iter().copied().map(f32::abs).fold(0.0, f32::max) / 127.0;
            for index in 0..32 {
                assert!((source[index] - decoded[block * 32 + index]).abs() <= scale * 0.51 + 1e-3);
            }
        }
    }

    #[test]
    fn q8_0_zero_block_is_canonical() {
        let quantized = Q8_0Tensor::quantize(&[0.0; 32], 32).unwrap();
        assert_eq!(quantized.as_bytes(), &[0; 34]);
    }
}
