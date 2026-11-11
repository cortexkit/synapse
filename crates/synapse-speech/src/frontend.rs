use std::sync::Arc;

use rustfft::{num_complex::Complex32, Fft, FftPlanner};

use crate::{
    decode_wav, Error, Result, FFT_LENGTH, HOP_LENGTH, LOG_GUARD, MEL_BINS, NORMALIZATION_EPSILON,
    PREEMPHASIS, WINDOW_LENGTH,
};

pub fn valid_mel_frames(samples_16khz: u64) -> u64 {
    samples_16khz / HOP_LENGTH as u64
}

pub fn actual_mel_frames(samples_16khz: u64) -> u64 {
    valid_mel_frames(samples_16khz) + 1
}

pub fn valid_encoder_frames(samples_16khz: u64) -> u64 {
    valid_mel_frames(samples_16khz).div_ceil(8)
}

#[derive(Debug, Clone)]
pub struct Features {
    /// Row-major [mel_frames, 128], with the frontend's own invalid tail zeroed.
    /// This tensor is not padded to an engine's preallocated input shape.
    pub values: Vec<f32>,
    pub mel_frames: usize,
    pub valid_mel_frames: usize,
    pub valid_encoder_frames: usize,
    pub samples_16khz: usize,
    /// Ceiling duration of the original audio, before resampling.
    pub duration_ms: u64,
}

pub struct Frontend {
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    filters: Vec<Vec<(usize, f32)>>,
}

impl Default for Frontend {
    fn default() -> Self {
        Self::new()
    }
}

impl Frontend {
    pub fn new() -> Self {
        let fft = FftPlanner::new().plan_fft_forward(FFT_LENGTH);
        let window = (0..WINDOW_LENGTH)
            .map(|i| {
                (0.5 - 0.5
                    * (2.0 * std::f64::consts::PI * i as f64 / (WINDOW_LENGTH - 1) as f64).cos())
                    as f32
            })
            .collect();
        let hz_to_mel = |hz: f64| {
            if hz < 1000.0 {
                hz / (200.0 / 3.0)
            } else {
                15.0 + (hz / 1000.0).ln() / (6.4f64.ln() / 27.0)
            }
        };
        let mel_to_hz = |mel: f64| {
            if mel < 15.0 {
                mel * (200.0 / 3.0)
            } else {
                1000.0 * ((mel - 15.0) * (6.4f64.ln() / 27.0)).exp()
            }
        };
        let edges: Vec<f64> = (0..MEL_BINS + 2)
            .map(|i| mel_to_hz(i as f64 * hz_to_mel(8000.0) / (MEL_BINS + 1) as f64))
            .collect();
        let filters = (0..MEL_BINS)
            .map(|mel| {
                (0..=FFT_LENGTH / 2)
                    .filter_map(|bin| {
                        let hz = bin as f64 * 16000.0 / FFT_LENGTH as f64;
                        let lower = (hz - edges[mel]) / (edges[mel + 1] - edges[mel]);
                        let upper = (edges[mel + 2] - hz) / (edges[mel + 2] - edges[mel + 1]);
                        // Librosa writes triangular weights to f32 before applying its
                        // f64 Slaney area normalization in place.
                        let weight = lower.min(upper).max(0.0) as f32;
                        let weight =
                            (f64::from(weight) * 2.0 / (edges[mel + 2] - edges[mel])) as f32;
                        (weight > 0.0).then_some((bin, weight))
                    })
                    .collect()
            })
            .collect();
        Self {
            fft,
            window,
            filters,
        }
    }

    /// WAV validation finishes before resampling or feature computation starts.
    pub fn process_wav(&self, bytes: &[u8]) -> Result<Features> {
        let audio = decode_wav(bytes)?;
        let mut features = self.process_samples(&audio.samples_16khz()?)?;
        features.duration_ms = audio.duration_ms();
        Ok(features)
    }

    /// Input is finite mono 16-kHz PCM, without clipping or dither.
    pub fn process_samples(&self, samples: &[f32]) -> Result<Features> {
        if samples.is_empty() || samples.iter().any(|x| !x.is_finite()) {
            return Err(Error::InvalidAudio("empty or non-finite samples"));
        }
        let valid = valid_mel_frames(samples.len() as u64) as usize;
        let frames = valid + 1;
        let mut emphasized = Vec::with_capacity(samples.len());
        emphasized.push(samples[0]);
        for pair in samples.windows(2) {
            emphasized.push(pair[1] - PREEMPHASIS * pair[0]);
        }
        let mut values = vec![0.0; frames * MEL_BINS];
        let mut spectrum = vec![Complex32::default(); FFT_LENGTH];
        let mut scratch = vec![Complex32::default(); self.fft.get_inplace_scratch_len()];
        let mut power = vec![0.0; FFT_LENGTH / 2 + 1];
        for frame in 0..valid {
            spectrum.fill(Complex32::default());
            for (i, &window) in self.window.iter().enumerate() {
                let index = frame as i64 * HOP_LENGTH as i64 + i as i64 - WINDOW_LENGTH as i64 / 2;
                if index >= 0 && index < emphasized.len() as i64 {
                    spectrum[i + (FFT_LENGTH - WINDOW_LENGTH) / 2].re =
                        emphasized[index as usize] * window;
                }
            }
            self.fft.process_with_scratch(&mut spectrum, &mut scratch);
            for (bin, value) in power.iter_mut().enumerate() {
                let magnitude = spectrum[bin].norm();
                *value = magnitude * magnitude;
            }
            for (mel, filter) in self.filters.iter().enumerate() {
                let energy: f32 = filter
                    .iter()
                    .map(|&(bin, weight)| power[bin] * weight)
                    .sum();
                values[frame * MEL_BINS + mel] = (energy + LOG_GUARD).ln();
            }
        }
        for mel in 0..MEL_BINS {
            // Match the reference's fp32 masked, two-pass statistics, including
            // reduction rounding on constant signals. A wider mean would turn
            // reference silence into zeros and disagree with its pinned tensors.
            let mean = reference_sum(frames, |t| values[t * MEL_BINS + mel]) / valid.max(1) as f32;
            let variance = reference_sum(frames, |t| {
                if t >= valid {
                    0.0
                } else {
                    (values[t * MEL_BINS + mel] - mean).powi(2)
                }
            }) / valid.saturating_sub(1).max(1) as f32;
            let denominator = variance.sqrt() + NORMALIZATION_EPSILON;
            for t in 0..valid {
                values[t * MEL_BINS + mel] = (values[t * MEL_BINS + mel] - mean) / denominator;
            }
        }
        if values.iter().any(|x| !x.is_finite()) {
            return Err(Error::InvalidAudio(
                "sample magnitudes overflow feature computation",
            ));
        }
        Ok(Features {
            values,
            mel_frames: frames,
            valid_mel_frames: valid,
            valid_encoder_frames: valid_encoder_frames(samples.len() as u64) as usize,
            samples_16khz: samples.len(),
            duration_ms: (samples.len() as u64).div_ceil(16),
        })
    }
}

/// PyTorch 2.14 CPU SumKernel's four-level cascade, expressed per feature so its
/// ordering is portable across SIMD widths. The masked tail participates as zero.
fn reference_sum(size: usize, mut value: impl FnMut(usize) -> f32) -> f32 {
    let ceil_log2 = usize::BITS - (size - 1).leading_zeros();
    let power = 4.max(ceil_log2 as usize / 4);
    let step = 1usize << power;
    let mask = step - 1;
    let mut accumulators = [0.0f32; 4];
    let mut i = 0;
    while i + step <= size {
        for _ in 0..step {
            accumulators[0] += value(i);
            i += 1;
        }
        for level in 1..4 {
            accumulators[level] += accumulators[level - 1];
            accumulators[level - 1] = 0.0;
            if i & (mask << (level * power)) != 0 {
                break;
            }
        }
    }
    while i < size {
        accumulators[0] += value(i);
        i += 1;
    }
    for level in 1..4 {
        accumulators[0] += accumulators[level];
    }
    accumulators[0]
}
