//! Exact speech references for the first-load check.
use crate::{refuse, Result};
use serde::Deserialize;
use serde_json::Value;
use synapse_parity::canonical::sha256_hex;

pub const CASE_IDS: [&str; 3] = [
    "silence-1s",
    "self-check-speech-clean",
    "self-check-speech-de",
];
const ROOT: &str = "fixtures/parakeet-tdt-0.6b-v3/";
const SUBSET: &[u8] =
    include_bytes!("../../../../bench/parity/fixtures/parakeet-tdt-0.6b-v3/self-check.json");
const CLEAN: &[u8] = include_bytes!(
    "../../../../bench/parity/fixtures/parakeet-tdt-0.6b-v3/self-check/speech-clean.wav"
);
const DE: &[u8] = include_bytes!(
    "../../../../bench/parity/fixtures/parakeet-tdt-0.6b-v3/self-check/speech-de.wav"
);
pub const COMPILED_BYTES: usize = SUBSET.len() + CLEAN.len() + DE.len();
const _: () = assert!(COMPILED_BYTES <= 256 * 1024);

#[derive(Debug, Deserialize)]
pub struct Case {
    pub id: String,
    pub triples: Vec<[i32; 3]>,
    pub text: String,
    pub valid_encoder_frames: usize,
    pub duration_ms: u64,
    #[serde(skip)]
    pub wav: Vec<u8>,
}

/// Hash the actual compiled subset even if its seal is invalid or absent. A bad
/// reference must have a stable key so its failure can be persisted, not retried.
pub fn revision() -> String {
    sha256_hex(SUBSET)
}

pub fn load(profile: &str) -> Result<Vec<Case>> {
    let manifest = serde_json::to_value(super::manifest()?).map_err(|e| refuse(e.to_string()))?;
    load_bytes(profile, &manifest, SUBSET, [CLEAN, DE])
}

fn load_bytes(
    profile: &str,
    manifest: &Value,
    bytes: &[u8],
    wavs: [&[u8]; 2],
) -> Result<Vec<Case>> {
    let invalid = || {
        refuse(format!(
            "speech self-check reference absent or invalid for {profile}"
        ))
    };
    let lane = &manifest["profiles"][profile];
    let speech = &lane["transcribe"];
    let seal = &speech["self_check"];
    if lane["model"] != "parakeet-tdt-0.6b-v3"
        || seal["path"] != format!("{ROOT}self-check.json")
        || seal["bytes"].as_u64() != Some(bytes.len() as u64)
        || seal["sha256"].as_str() != Some(sha256_hex(bytes).as_str())
    {
        return Err(invalid());
    }
    let document: Value = serde_json::from_slice(bytes).map_err(|_| invalid())?;
    let margin = speech["tau_margin"].as_f64().ok_or_else(invalid)?;
    if !margin.is_finite()
        || margin <= 0.0
        || document["schema"] != "synapse-parakeet-self-check-v1"
        || document["tau_margin"].as_f64() != Some(margin)
        || document["sealed_bytes"]
            .as_u64()
            .is_none_or(|n| n > 256 * 1024)
        || bytes.len() + wavs.iter().map(|wav| wav.len()).sum::<usize>() > 256 * 1024
    {
        return Err(invalid());
    }
    let cases = document["cases"].as_array().ok_or_else(invalid)?;
    if cases.len() != CASE_IDS.len() {
        return Err(invalid());
    }
    let mut result = Vec::new();
    for (index, (value, id)) in cases.iter().zip(CASE_IDS).enumerate() {
        let mut case: Case = serde_json::from_value(value.clone()).map_err(|_| invalid())?;
        if case.id != id
            || value["minimum_margin"]
                .as_f64()
                .is_none_or(|n| n < 2.0 * margin)
            || case.valid_encoder_frames == 0
        {
            return Err(invalid());
        }
        case.wav = if index == 0 {
            if case.duration_ms != 1000
                || value.get("wav").is_some()
                || !case.triples.is_empty()
                || !case.text.is_empty()
            {
                return Err(invalid());
            }
            silence_wav()
        } else {
            let wav = wavs[index - 1];
            let name = if index == 1 {
                "speech-clean"
            } else {
                "speech-de"
            };
            if value["wav"]["path"] != format!("{ROOT}self-check/{name}.wav")
                || value["wav"]["bytes"].as_u64() != Some(wav.len() as u64)
                || value["wav"]["sha256"].as_str() != Some(sha256_hex(wav).as_str())
                || case.duration_ms > if index == 1 { 3000 } else { 2000 }
            {
                return Err(invalid());
            }
            wav.to_vec()
        };
        result.push(case);
    }
    Ok(result)
}

fn silence_wav() -> Vec<u8> {
    let mut wav = b"RIFF".to_vec();
    wav.extend_from_slice(&32036u32.to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt \x10\0\0\0\x01\0\x01\0");
    wav.extend_from_slice(&16000u32.to_le_bytes());
    wav.extend_from_slice(&32000u32.to_le_bytes());
    wav.extend_from_slice(b"\x02\0\x10\0data");
    wav.extend_from_slice(&32000u32.to_le_bytes());
    wav.resize(32044, 0);
    wav
}

/// Both emitted triples and detokenized text must match exactly during the
/// load-time check, even when the reference's top two logits are close together.
pub fn compare(case: &Case, triples: &[[i32; 3]], text: &str) -> Result<()> {
    if triples != case.triples || text != case.text {
        Err(refuse(format!("speech self-check mismatch: {}", case.id)))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speech_self_check_seals_fixed_subset_and_compiled_budget() {
        for lane in [
            "owned-metal",
            "owned-cuda",
            "owned-vulkan",
            "ane-direct-worker",
        ] {
            let cases = load(&format!("parakeet-tdt-0.6b-v3.{lane}")).unwrap();
            assert_eq!(
                cases.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(),
                CASE_IDS
            );
            assert_eq!(cases[0].wav.len(), 32044);
            assert!(cases[0].wav[44..].iter().all(|b| *b == 0));
            for case in cases {
                compare(&case, &case.triples, &case.text).unwrap();
                assert!(compare(&case, &case.triples, "wrong text").is_err());
                let mut triples = case.triples.clone();
                triples.push([0, 0, 0]);
                assert!(compare(&case, &triples, &case.text).is_err());
            }
        }
    }

    #[test]
    fn speech_self_check_missing_and_invalid_seals_refuse() {
        let profile = "parakeet-tdt-0.6b-v3.owned-metal";
        let manifest = serde_json::to_value(super::super::manifest().unwrap()).unwrap();
        assert!(load("absent").is_err());
        let mut corrupt = SUBSET.to_vec();
        corrupt[0] ^= 1;
        assert!(load_bytes(profile, &manifest, &corrupt, [CLEAN, DE]).is_err());
        let mut wav = CLEAN.to_vec();
        wav[44] ^= 1;
        assert!(load_bytes(profile, &manifest, SUBSET, [&wav, DE]).is_err());
        let mut missing = manifest.clone();
        missing["profiles"][profile]["transcribe"]
            .as_object_mut()
            .unwrap()
            .remove("self_check");
        assert!(load_bytes(profile, &missing, SUBSET, [CLEAN, DE]).is_err());
    }
}
