use std::{
    fs,
    path::{Path, PathBuf},
};

use serde_json::Value;
use sha2::{Digest, Sha256};
use synapse_speech::*;

fn parity() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/parity")
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn json(path: impl AsRef<Path>) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}
fn read_seal(seal: &Value, mirror: Option<&Path>) -> Vec<u8> {
    let path = if let Some(path) = seal["path"].as_str() {
        parity().join(path)
    } else {
        mirror
            .expect("external mirror required")
            .join(seal["cache_path"].as_str().unwrap())
    };
    let bytes = fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(
        bytes.len() as u64,
        seal["bytes"].as_u64().unwrap(),
        "{}",
        path.display()
    );
    assert_eq!(
        sha(&bytes),
        seal["sha256"].as_str().unwrap(),
        "{}",
        path.display()
    );
    bytes
}
fn cases() -> Vec<Value> {
    let index = json(parity().join("fixtures/index.json"));
    let entry = &index["parakeet-tdt-0.6b-v3.ref-v1.transformers-5.16.1.seed-0"];
    let reference: Value = serde_json::from_slice(&read_seal(entry, None)).unwrap();
    assert_eq!(
        reference["tokenizer_digest"],
        format!("sha256:{TOKENIZER_SHA256}")
    );
    let cases = reference["cases"].as_array().unwrap().clone();
    assert_eq!(cases.len(), 1247);
    for case in &cases {
        for key in ["trace", "encoder", "log_mel"] {
            assert_eq!(
                case[key],
                entry["artifacts"][case["id"].as_str().unwrap()][key]
            );
        }
    }
    cases
}
fn profile() -> Value {
    let manifest = json(parity().join("models.json"));
    manifest["profiles"]
        .as_object()
        .unwrap()
        .values()
        .find(|p| p["model"] == "parakeet-tdt-0.6b-v3")
        .unwrap()["transcribe"]
        .clone()
}
fn trace(case: &Value) -> Vec<TraceStep> {
    let bytes = read_seal(&case["trace"], None);
    assert_eq!(bytes.len() % 40, 0);
    bytes
        .as_chunks::<40>()
        .0
        .iter()
        .map(|row| {
            let i = |n| u32::from_le_bytes(row[n..n + 4].try_into().unwrap());
            let f = |n| f32::from_le_bytes(row[n..n + 4].try_into().unwrap());
            assert_eq!(i(36) & !3, 0);
            TraceStep {
                frame_index: i(0),
                token_top1: i(4),
                token_top2: i(8),
                duration_top1: i(12),
                duration_top2: i(16),
                token_logit1: f(20),
                token_logit2: f(24),
                duration_logit1: f(28),
                duration_logit2: f(32),
                emitted: i(36) & 1 != 0,
                cap_hit: i(36) & 2 != 0,
            }
        })
        .collect()
}
fn floats(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0);
    let values: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&x| f32::from_le_bytes(x))
        .collect();
    assert!(values.iter().all(|x| x.is_finite()));
    values
}

/// Replay joint decisions, not neural math: the engine tests cover the neural
/// predictor against the encoder tensors. This stub binds each decision to its
/// requested encoder row and lets the production loop choose all advances.
struct Replay {
    steps: Vec<TraceStep>,
    index: usize,
    tokens: Vec<f32>,
    durations: Vec<f32>,
    encoder: Option<Vec<f32>>,
    accepts: Vec<u32>,
}
impl Joint for Replay {
    fn reset(&mut self) -> Result<()> {
        self.index = 0;
        self.accepts.clear();
        Ok(())
    }
    fn evaluate(&mut self, frame: usize) -> Result<JointLogits<'_>> {
        let step = &self.steps[self.index];
        self.index += 1;
        assert_eq!(frame, step.frame_index as usize);
        if let Some(encoder) = &self.encoder {
            assert!(encoder[frame * 1024..(frame + 1) * 1024]
                .iter()
                .all(|x| x.is_finite()));
        }
        self.tokens.fill(step.token_logit2 - 100.0);
        self.tokens[step.token_top1 as usize] = step.token_logit1;
        self.tokens[step.token_top2 as usize] = step.token_logit2;
        self.durations.fill(step.duration_logit2 - 100.0);
        self.durations[step.duration_top1 as usize] = step.duration_logit1;
        self.durations[step.duration_top2 as usize] = step.duration_logit2;
        Ok(JointLogits {
            tokens: &self.tokens,
            durations: &self.durations,
        })
    }
    fn accept(&mut self, token: u32) -> Result<()> {
        assert_eq!(token, self.steps[self.index - 1].token_top1);
        assert_ne!(token, BLANK);
        self.accepts.push(token);
        Ok(())
    }
}
fn replay(case: &Value, encoder: Option<Vec<f32>>) {
    let expected = trace(case);
    let mut joint = Replay {
        steps: expected.clone(),
        index: 0,
        tokens: vec![0.0; VOCAB_SIZE],
        durations: vec![0.0; 5],
        encoder,
        accepts: vec![],
    };
    let output = greedy(
        &mut joint,
        case["valid_encoder_frames"].as_u64().unwrap() as usize,
        true,
        |_| {},
    )
    .unwrap();
    // Require exact replay rather than exempting an argmax difference when the
    // reference's top-two logits are too close to distinguish reliably.
    assert_eq!(output.trace, expected, "{}", case["id"]);
    assert_eq!(
        serde_json::to_value(&output.triples).unwrap(),
        case["triples"],
        "{}",
        case["id"]
    );
    assert_eq!(joint.index, joint.steps.len());
    assert_eq!(
        joint.accepts,
        output.triples.iter().map(|x| x[0]).collect::<Vec<_>>()
    );
}

#[test]
fn every_fixture_matches_frame_functions_and_tokenizer_words() {
    let tokenizer = Tokenizer::pinned();
    for case in cases() {
        let samples = case["sample_count"].as_u64().unwrap();
        assert_eq!(
            actual_mel_frames(samples),
            case["mel_frames"].as_u64().unwrap()
        );
        assert_eq!(
            valid_mel_frames(samples),
            case["valid_mel_frames"].as_u64().unwrap()
        );
        assert_eq!(
            valid_encoder_frames(samples),
            case["valid_encoder_frames"].as_u64().unwrap()
        );
        let triples: Vec<Triple> = serde_json::from_value(case["triples"].clone()).unwrap();
        let result = tokenizer
            .transcript(&triples, samples as f64 / 16.0)
            .unwrap();
        assert_eq!(
            result.text,
            case["text"].as_str().unwrap(),
            "{}",
            case["id"]
        );
        let expected_words: Vec<Word> = serde_json::from_value(case["words"].clone()).unwrap();
        assert_eq!(result.words, expected_words, "{}", case["id"]);
        assert_eq!(result.no_speech, case["no_speech"].as_bool().unwrap());
    }
}

#[test]
fn every_sealed_trace_replays_exactly_through_shared_loop() {
    for case in cases() {
        replay(&case, None);
    }
}

#[test]
fn constants_match_sha256_pinned_configs_and_profile() {
    let root = parity().join("checkpoints/parakeet-tdt-0.6b-v3");
    let mut configs = vec![];
    for (file, digest) in [
        (
            "processor_config.json",
            "8346a93a3b987fa1dec57a78f045cd0817d21786589a5a096b41a57a446fd1d7",
        ),
        (
            "generation_config.json",
            "b141de6ec6d7f982ece13f98f604e3fe1807ea9c0e839185d0ab7064604209d0",
        ),
        (
            "tokenizer_config.json",
            "0b2fe0037599ee335f0b972fa682bf0ece74e4ccfec755cb7daa3405d3d3e874",
        ),
    ] {
        let bytes = fs::read(root.join(file)).unwrap();
        assert_eq!(sha(&bytes), digest);
        configs.push(serde_json::from_slice::<Value>(&bytes).unwrap());
    }
    let front = &configs[0]["feature_extractor"];
    for (key, expected) in [
        ("feature_size", MEL_BINS),
        ("hop_length", HOP_LENGTH),
        ("n_fft", FFT_LENGTH),
        ("win_length", WINDOW_LENGTH),
        ("sampling_rate", SAMPLE_RATE as usize),
    ] {
        assert_eq!(front[key].as_u64().unwrap(), expected as u64);
    }
    assert_eq!(front["preemphasis"].as_f64().unwrap() as f32, PREEMPHASIS);
    assert_eq!(front["padding_value"], 0.0);
    assert_eq!(configs[1]["decoder_start_token_id"], BLANK);
    assert_eq!(configs[1]["pad_token_id"], PAD_TOKEN);
    assert_eq!(configs[2]["clean_up_tokenization_spaces"], false);
    let tokenizer = Tokenizer::pinned();
    assert_eq!(
        tokenizer.piece(BLANK).unwrap(),
        configs[0]["blank_token"].as_str().unwrap()
    );
    for (id, field) in [(0, "unk_token"), (2, "pad_token"), (3, "eos_token")] {
        assert_eq!(
            tokenizer.piece(id).unwrap(),
            configs[2][field].as_str().unwrap()
        );
    }
    assert_eq!(
        sha(&fs::read(root.join("tokenizer.json")).unwrap()),
        TOKENIZER_SHA256
    );
    let profile = profile();
    let rule = &profile["confirmations"]["decode_rule"];
    assert_eq!(rule["revision"], DECODE_RULE_REVISION);
    assert_eq!(rule["K"], MAX_SYMBOLS_PER_FRAME);
    assert_eq!(rule["durations"], serde_json::to_value(DURATIONS).unwrap());
    assert_eq!(
        SPEECH_FRONTEND_REVISION,
        "synapse-parakeet-speech-frontend-v1"
    );
    assert_eq!(LOG_GUARD, 2f32.powi(-24));
    assert_eq!(NORMALIZATION_EPSILON, 1e-5);
}

#[test]
#[ignore = "requires the sealed external ladder mirror; set SYNAPSE_SPEECH_MIRROR"]
fn every_stored_encoder_tensor_replays_exactly() {
    let mirror =
        PathBuf::from(std::env::var_os("SYNAPSE_SPEECH_MIRROR").expect("mirror root required"));
    let mut count = 0;
    for case in cases().iter().filter(|c| !c["encoder"].is_null()) {
        let encoder = floats(&read_seal(&case["encoder"], Some(&mirror)));
        assert_eq!(
            encoder.len(),
            case["valid_encoder_frames"].as_u64().unwrap() as usize * 1024
        );
        replay(case, Some(encoder));
        count += 1;
    }
    assert_eq!(count, 79);
}

#[test]
#[ignore = "requires prepared input WAVs and external ladder mirror; set SYNAPSE_SPEECH_AUDIO and SYNAPSE_SPEECH_MIRROR"]
fn every_tensor_frontend_and_resampler_meets_its_own_seal() {
    let audio = PathBuf::from(
        std::env::var_os("SYNAPSE_SPEECH_AUDIO").expect("prepared WAV root required"),
    );
    let mirror =
        PathBuf::from(std::env::var_os("SYNAPSE_SPEECH_MIRROR").expect("mirror root required"));
    let profile = profile();
    let tau_feat = profile["tau_feat"].as_f64().unwrap();
    let tau_resample = profile["tau_resample"].as_f64().unwrap();
    let frontend = Frontend::new();
    let cases = cases();
    let mut count = 0;
    let mut resampled = 0;
    let mut worst = 0.0f64;
    for case in cases.iter().filter(|c| !c["log_mel"].is_null()) {
        let id = case["id"].as_str().unwrap();
        let raw = fs::read(audio.join(format!("{id}.wav"))).unwrap();
        if let Some(digest) = case["input"]["generated_audio_sha256"].as_str() {
            assert_eq!(sha(&raw), digest);
        }
        let result = frontend.process_wav(&raw).unwrap();
        assert_eq!(
            result.samples_16khz as u64,
            case["sample_count"].as_u64().unwrap()
        );
        assert_eq!(
            result.mel_frames as u64,
            case["mel_frames"].as_u64().unwrap()
        );
        assert_eq!(
            result.valid_mel_frames as u64,
            case["valid_mel_frames"].as_u64().unwrap()
        );
        assert_eq!(
            result.valid_encoder_frames as u64,
            case["valid_encoder_frames"].as_u64().unwrap()
        );
        let reference = floats(&read_seal(&case["log_mel"], Some(&mirror)));
        assert_eq!(reference.len(), result.values.len());
        let difference = reference
            .iter()
            .zip(&result.values)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
            .fold(0.0, f64::max);
        assert!(
            difference <= tau_feat,
            "{id}: max feature difference {difference} exceeds {tau_feat}"
        );
        worst = worst.max(difference);
        if case["input"]["recipe"] == "resampled" {
            let base_id = case["input"]["base"]["id"].as_str().unwrap();
            let base = cases.iter().find(|c| c["id"] == base_id).unwrap();
            let native = floats(&read_seal(&base["log_mel"], Some(&mirror)));
            assert_eq!(native.len(), result.values.len());
            let gap = native
                .iter()
                .zip(&result.values)
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
                .fold(0.0, f64::max);
            assert!(gap <= tau_resample, "{id}: native sanity gap {gap}");
            resampled += 1;
        }
        count += 1;
    }
    assert_eq!(count, 79);
    assert_eq!(resampled, 16);
    println!("{count} tensor fixtures, {resampled} resampled fixtures: worst max-abs feature difference {worst}");
}
