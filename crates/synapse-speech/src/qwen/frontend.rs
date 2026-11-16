use std::sync::Arc;

use rustfft::{num_complex::Complex32, Fft, FftPlanner};

use super::{FFT_LENGTH, HOP_LENGTH, MEL_BINS};
use crate::{decode_wav, Error, Result};

/// Both actual and valid frame counts drop the last centered STFT frame.
pub fn mel_frames(samples_16khz: u64) -> u64 {
    samples_16khz / HOP_LENGTH as u64
}

/// The convolutions restart every 100 mel frames, including the final remainder.
pub fn audio_tokens(valid_mel_frames: u64) -> u64 {
    13 * (valid_mel_frames / 100) + (valid_mel_frames % 100).div_ceil(8)
}

#[derive(Debug, Clone)]
pub struct Features {
    /// Row-major [mel_frames, 128], without engine padding or an invalid tail.
    pub values: Vec<f32>,
    pub mel_frames: usize,
    pub valid_mel_frames: usize,
    pub audio_tokens: usize,
    pub samples_16khz: usize,
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
        let window = (0..FFT_LENGTH)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / FFT_LENGTH as f32).cos())
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
        let edges: Vec<_> = (0..MEL_BINS + 2)
            .map(|i| mel_to_hz(i as f64 * hz_to_mel(8000.0) / (MEL_BINS + 1) as f64))
            .collect();
        let filters = (0..MEL_BINS)
            .map(|mel| {
                (0..=FFT_LENGTH / 2)
                    .filter_map(|bin| {
                        let hz = bin as f64 * 16000.0 / FFT_LENGTH as f64;
                        let lower = (hz - edges[mel]) / (edges[mel + 1] - edges[mel]);
                        let upper = (edges[mel + 2] - hz) / (edges[mel + 2] - edges[mel + 1]);
                        let weight = (lower.min(upper).max(0.0) * 2.0
                            / (edges[mel + 2] - edges[mel]))
                            as f32;
                        (weight > 0.0).then_some((bin, weight))
                    })
                    .collect()
            })
            .collect();
        Self {
            fft: FftPlanner::new().plan_fft_forward(FFT_LENGTH),
            window,
            filters,
        }
    }

    pub fn process_wav(&self, bytes: &[u8]) -> Result<Features> {
        let audio = decode_wav(bytes)?;
        let mut features = self.process_samples(&audio.samples_16khz()?)?;
        features.duration_ms = audio.duration_ms();
        Ok(features)
    }

    /// Finite mono 16-kHz PCM; no waveform normalization, preemphasis or dither.
    /// As in the pinned torch STFT, reflection padding requires over 200 samples.
    pub fn process_samples(&self, samples: &[f32]) -> Result<Features> {
        if samples.len() <= FFT_LENGTH / 2 || samples.iter().any(|x| !x.is_finite()) {
            return Err(Error::InvalidAudio(
                "reflection STFT requires over 200 finite samples",
            ));
        }
        let frames = mel_frames(samples.len() as u64) as usize;
        let mut values = vec![0.0f32; frames * MEL_BINS];
        let mut spectrum = vec![Complex32::default(); FFT_LENGTH];
        let mut scratch = vec![Complex32::default(); self.fft.get_inplace_scratch_len()];
        let mut power = vec![0.0; FFT_LENGTH / 2 + 1];
        for frame in 0..frames {
            for (i, &window) in self.window.iter().enumerate() {
                let index =
                    frame as isize * HOP_LENGTH as isize + i as isize - FFT_LENGTH as isize / 2;
                let index = if index < 0 {
                    -index
                } else if index >= samples.len() as isize {
                    2 * samples.len() as isize - 2 - index
                } else {
                    index
                } as usize;
                spectrum[i] = Complex32::new(samples[index] * window, 0.0);
            }
            self.fft.process_with_scratch(&mut spectrum, &mut scratch);
            for (bin, value) in power.iter_mut().enumerate() {
                *value = spectrum[bin].norm_sqr();
            }
            if power.iter().any(|value| !value.is_finite()) {
                return Err(Error::InvalidAudio(
                    "sample magnitudes overflow feature computation",
                ));
            }
            for (mel, filter) in self.filters.iter().enumerate() {
                let energy: f32 = filter
                    .iter()
                    .map(|&(bin, weight)| power[bin] * weight)
                    .sum();
                values[frame * MEL_BINS + mel] = energy.max(1e-10).log10();
            }
        }
        let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for value in &mut values {
            *value = (value.max(maximum - 8.0) + 4.0) / 4.0;
        }
        if values.iter().any(|x| !x.is_finite()) || !maximum.is_finite() {
            return Err(Error::InvalidAudio(
                "sample magnitudes overflow feature computation",
            ));
        }
        Ok(Features {
            values,
            mel_frames: frames,
            valid_mel_frames: frames,
            audio_tokens: audio_tokens(frames as u64) as usize,
            samples_16khz: samples.len(),
            duration_ms: (samples.len() as u64).div_ceil(16),
        })
    }
}
