use crate::{Error, Result};

pub const SAMPLE_RATE: u32 = 16_000;
pub const MAX_AUDIO_BYTES: usize = 256 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct Audio {
    /// Mono samples at the input sample rate, before resampling.
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl Audio {
    pub fn duration_ms(&self) -> u64 {
        (self.samples.len() as u64 * 1000).div_ceil(u64::from(self.sample_rate))
    }

    pub fn samples_16khz(&self) -> Result<Vec<f32>> {
        resample(&self.samples, self.sample_rate)
    }
}

fn invalid(reason: &'static str) -> Error {
    Error::InvalidAudio(reason)
}

fn u16le(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap())
}

fn u32le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

/// Validate the entire RIFF before decoding any samples. Unknown chunks are
/// allowed, but their lengths and word padding must still fit the container.
pub fn decode_wav(bytes: &[u8]) -> Result<Audio> {
    if bytes.len() > MAX_AUDIO_BYTES || bytes.len() < 12 {
        return Err(invalid("WAV size"));
    }
    if &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(invalid("expected RIFF/WAVE"));
    }
    if u32le(bytes, 4) as usize != bytes.len() - 8 {
        return Err(invalid("RIFF length mismatch"));
    }
    let (mut format, mut data) = (None, None);
    let mut offset = 12usize;
    while offset < bytes.len() {
        if bytes.len() - offset < 8 {
            return Err(invalid("truncated chunk header"));
        }
        let length = u32le(bytes, offset + 4) as usize;
        let start = offset + 8;
        let end = start.checked_add(length).ok_or(invalid("chunk overflow"))?;
        let next = end
            .checked_add(length % 2)
            .ok_or(invalid("chunk overflow"))?;
        if next > bytes.len() {
            return Err(invalid("truncated chunk or padding"));
        }
        match &bytes[offset..offset + 4] {
            b"fmt " if format.replace(&bytes[start..end]).is_some() => {
                return Err(invalid("duplicate format chunk"));
            }
            b"data" if data.replace(&bytes[start..end]).is_some() => {
                return Err(invalid("duplicate data chunk"));
            }
            _ => {}
        }
        offset = next;
    }
    let fmt = format.ok_or(invalid("missing format chunk"))?;
    let data = data.ok_or(invalid("missing data chunk"))?;
    if fmt.len() < 16 || fmt.len() == 17 {
        return Err(invalid("short format chunk"));
    }
    if fmt.len() >= 18 && usize::from(u16le(fmt, 16)) != fmt.len() - 18 {
        return Err(invalid("format extension length mismatch"));
    }
    let mut codec = u16le(fmt, 0);
    let channels = u16le(fmt, 2);
    let rate = u32le(fmt, 4);
    let bits = u16le(fmt, 14);
    if !(1..=8).contains(&channels) || !(8_000..=192_000).contains(&rate) {
        return Err(invalid("sample rate or channels out of range"));
    }
    if codec == 0xfffe {
        if fmt.len() < 40 || u16le(fmt, 16) < 22 || u16le(fmt, 18) != bits {
            return Err(invalid("invalid extensible format"));
        }
        // KSDATAFORMAT_SUBTYPE_PCM / IEEE_FLOAT share the GUID suffix.
        if fmt[26..40] != [0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71] {
            return Err(invalid("unsupported extensible subformat"));
        }
        codec = u16le(fmt, 24);
    }
    if !matches!((codec, bits), (1, 16) | (3, 32)) {
        return Err(invalid("only PCM16 or float32 supported"));
    }
    let width = usize::from(bits / 8);
    let block = usize::from(channels) * width;
    if usize::from(u16le(fmt, 12)) != block || u32le(fmt, 8) != rate * block as u32 {
        return Err(invalid("inconsistent block alignment or byte rate"));
    }
    if data.is_empty() || data.len() % block != 0 {
        return Err(invalid("empty or partial sample frame"));
    }
    let mut samples = Vec::with_capacity(data.len() / block);
    for frame in data.chunks_exact(block) {
        let mut sum = 0.0f64;
        for sample in frame.chunks_exact(width) {
            let x = if codec == 1 {
                f32::from(i16::from_le_bytes(sample.try_into().unwrap())) / 32768.0
            } else {
                f32::from_le_bytes(sample.try_into().unwrap())
            };
            if !x.is_finite() {
                return Err(invalid("non-finite sample"));
            }
            sum += f64::from(x);
        }
        samples.push((sum / f64::from(channels)) as f32);
    }
    Ok(Audio {
        samples,
        sample_rate: rate,
        channels,
    })
}

/// The reference's fp64 Hann-windowed sinc: six zero crossings, 0.99 of
/// the lower Nyquist, zero outside the signal, and a ceiling output length.
pub fn resample(input: &[f32], source_rate: u32) -> Result<Vec<f32>> {
    if input.is_empty() || input.iter().any(|x| !x.is_finite()) {
        return Err(invalid("empty or non-finite samples"));
    }
    if !(8_000..=192_000).contains(&source_rate) {
        return Err(invalid("sample rate out of range"));
    }
    if source_rate == SAMPLE_RATE {
        return Ok(input.to_vec());
    }
    let count = (input.len() as u64 * u64::from(SAMPLE_RATE)).div_ceil(u64::from(source_rate));
    let count = usize::try_from(count).map_err(|_| invalid("resampled length overflow"))?;
    let cutoff = 0.99 * f64::from(source_rate.min(SAMPLE_RATE)) / f64::from(source_rate);
    let radius = (6.0 / cutoff) as i64 + 1;
    let mut output = Vec::with_capacity(count);
    for i in 0..count {
        let position = i as f64 * f64::from(source_rate) / f64::from(SAMPLE_RATE);
        let center = position.floor() as i64;
        let mut sum = 0.0;
        for index in (center - radius).max(0)..=(center + radius).min(input.len() as i64 - 1) {
            let distance = (index as f64 - position) * cutoff;
            if distance.abs() < 6.0 {
                let phase = std::f64::consts::PI * distance;
                let sinc = if distance == 0.0 {
                    1.0
                } else {
                    phase.sin() / phase
                };
                let window = (distance * (std::f64::consts::PI / 12.0)).cos().powi(2);
                sum += f64::from(input[index as usize]) * cutoff * sinc * window;
            }
        }
        output.push(sum as f32);
    }
    Ok(output)
}
