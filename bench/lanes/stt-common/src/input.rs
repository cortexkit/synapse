use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Deserialize)]
pub struct InputEntry {
    pub path: PathBuf,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Utterance {
    pub id: String,
    pub path: PathBuf,
    pub split: String,
    pub locale: String,
    pub duration_seconds: f64,
}

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn read_pcm(path: &Path) -> Result<Vec<f32>> {
    let bytes = fs::read(path)?;
    wav_samples(&bytes)?;
    let mut offset = 12;
    while offset + 8 <= bytes.len() {
        let length = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into()?) as usize;
        let start = offset + 8;
        if &bytes[offset..offset + 4] == b"data" {
            return Ok(bytes[start..start + length]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|sample| f32::from_le_bytes(*sample))
                .collect());
        }
        offset = start + length + length % 2;
    }
    bail!("missing WAV data")
}

pub fn verify_audio(root: &Path, entry: &InputEntry, samples: u64) -> Result<PathBuf> {
    let root = root.canonicalize()?;
    let path = root
        .join(&entry.path)
        .canonicalize()
        .with_context(|| format!("audio {}", entry.path.display()))?;
    ensure!(path.starts_with(&root), "audio escapes WAV directory");
    let bytes = fs::read(&path)?;
    ensure!(
        sha256(&bytes) == entry.sha256,
        "SHA256 mismatch: {}",
        path.display()
    );
    ensure!(
        wav_samples(&bytes)? == samples,
        "sample count mismatch: {}",
        path.display()
    );
    Ok(path)
}

// Parse chunks rather than assuming a 44-byte header: prepared WAVs may contain
// metadata or WAVE_FORMAT_EXTENSIBLE headers before their sample payload.
pub fn wav_samples(bytes: &[u8]) -> Result<u64> {
    ensure!(
        bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "not a RIFF WAV"
    );
    let u16_at = |offset| u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
    let u32_at = |offset| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    ensure!(u32_at(4) as usize + 8 == bytes.len(), "invalid RIFF length");
    let mut offset = 12;
    let mut format = false;
    let mut samples = None;
    while offset + 8 <= bytes.len() {
        let length = u32_at(offset + 4) as usize;
        let start = offset + 8;
        let end = start.checked_add(length).context("WAV chunk overflow")?;
        ensure!(end <= bytes.len(), "truncated WAV chunk");
        match &bytes[offset..offset + 4] {
            b"fmt " => {
                ensure!(!format && length >= 16, "invalid WAV format chunk");
                let tag = u16_at(start);
                let float = tag == 3
                    || (tag == 0xfffe
                        && length >= 40
                        && u16_at(start + 16) >= 22
                        && bytes[start + 24..start + 40]
                            == [3, 0, 0, 0, 0, 0, 16, 0, 128, 0, 0, 170, 0, 56, 155, 113]);
                ensure!(
                    float
                        && u16_at(start + 2) == 1
                        && u32_at(start + 4) == 16_000
                        && u32_at(start + 8) == 64_000
                        && u16_at(start + 12) == 4
                        && u16_at(start + 14) == 32,
                    "expected mono 16 kHz float32 WAV"
                );
                format = true;
            }
            b"data" => {
                ensure!(
                    samples.is_none() && length.is_multiple_of(4),
                    "invalid WAV data chunk"
                );
                ensure!(
                    bytes[start..end]
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .all(|sample| f32::from_le_bytes(*sample).is_finite()),
                    "nonfinite audio"
                );
                samples = Some((length / 4) as u64);
            }
            _ => {}
        }
        offset = end + length % 2;
    }
    ensure!(
        format && offset == bytes.len(),
        "missing format or invalid WAV padding"
    );
    match samples {
        Some(count) if count > 0 => Ok(count),
        _ => bail!("missing or empty WAV samples"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav() -> Vec<u8> {
        let mut bytes = b"RIFF".to_vec();
        bytes.extend(40_u32.to_le_bytes());
        bytes.extend(b"WAVEfmt ");
        bytes.extend(16_u32.to_le_bytes());
        bytes.extend(3_u16.to_le_bytes());
        bytes.extend(1_u16.to_le_bytes());
        bytes.extend(16000_u32.to_le_bytes());
        bytes.extend(64000_u32.to_le_bytes());
        bytes.extend(4_u16.to_le_bytes());
        bytes.extend(32_u16.to_le_bytes());
        bytes.extend(b"data");
        bytes.extend(4_u32.to_le_bytes());
        bytes.extend(0.5_f32.to_le_bytes());
        bytes
    }

    #[test]
    fn stt_input_verifies_hash_format_samples_and_containment() {
        let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../target/scratch")
            .join(format!("stt-input-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let bytes = wav();
        fs::write(directory.join("audio.wav"), &bytes).unwrap();
        assert_eq!(read_pcm(&directory.join("audio.wav")).unwrap(), vec![0.5]);
        let mut entry = InputEntry {
            path: "audio.wav".into(),
            sha256: sha256(&bytes),
        };
        assert!(verify_audio(&directory, &entry, 1).is_ok());
        assert!(verify_audio(&directory, &entry, 2).is_err());
        entry.sha256 = "0".repeat(64);
        assert!(verify_audio(&directory, &entry, 1).is_err());
        let outside = directory.with_extension("wav");
        fs::write(&outside, &bytes).unwrap();
        entry.path = outside.canonicalize().unwrap();
        entry.sha256 = sha256(&bytes);
        assert!(verify_audio(&directory, &entry, 1).is_err());
        fs::remove_file(outside).unwrap();
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn stt_input_rejects_wrong_rate_integer_and_nonfinite_wavs() {
        let bytes = wav();
        assert_eq!(wav_samples(&bytes).unwrap(), 1);
        for (offset, value) in [(20, 1), (24, 0)] {
            let mut invalid = bytes.clone();
            invalid[offset] = value;
            assert!(wav_samples(&invalid).is_err());
        }
        let mut invalid = bytes.clone();
        invalid[44..48].copy_from_slice(&f32::INFINITY.to_le_bytes());
        assert!(wav_samples(&invalid).is_err());
        assert!(wav_samples(&bytes[..30]).is_err());
    }
}
